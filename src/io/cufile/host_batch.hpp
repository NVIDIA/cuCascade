/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cucascade/io/cufile/config.hpp>
#include <cucascade/io/types.hpp>

#include <cufile.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <tuple>
#include <utility>

namespace cucascade::io::cufile {
struct cufile_api;
}  // namespace cucascade::io::cufile

namespace cucascade::io::cufile::detail {

/**
 * @brief One host read served through cuFile's batch API.
 *
 * Every field must be 4 KiB-aligned (@c io::IO_BLOCK_SIZE): @c dst, @c rng.offset and
 * @c rng.size (non-zero), and the range must lie inside the file's whole blocks
 * (`rng.end() <= align_down(file_size)`).  Unaligned entries take cuFile's POSIX pool, which
 * sporadically fails them in compatibility mode; the caller cuts its reads so they never are.
 */
struct batch_read {
  void* handle{};       ///< @c CUfileHandle_t of the file (cufile_io_object::cufile_handle()).
  std::uint8_t* dst{};  ///< Host destination; submitted as @c devPtr_base with a zero offset.
  range rng;            ///< File range.
  std::uint64_t token{};  ///< The owner's; returned unchanged in the result.
};

/// How a @ref batch_read ended.
struct batch_result {
  enum class status : std::uint8_t {
    ok,          ///< All of @c rng.size bytes were read.
    short_read,  ///< cuFile completed the entry with fewer bytes (@c bytes of them, from the start).
    failed,  ///< Nothing usable was read (cuFile failed, rejected or the reader was torn down).
  };
  std::uint64_t token{};
  status outcome{status::ok};
  std::size_t bytes{};  ///< Bytes delivered from the start of the range (ok: @c rng.size).
  int error{};          ///< failed: the cuFile entry status or the cuFile error code; else 0.
};

/// The submit policy lives in the public configuration (cufile/config.hpp).
using cucascade::io::cufile::batch_submit_policy;

/// Diagnostic counters of a @ref host_batch_reader (plain integers, read by its owner).
struct host_batch_counters {
  std::uint64_t submits{0};            ///< Successful @c cuFileBatchIOSubmit calls.
  std::uint64_t submitted_entries{0};  ///< Entries those calls carried.
  std::uint64_t rejected_submits{0};   ///< Submit calls cuFile rejected.
  std::uint64_t completed{0};          ///< Entries reported @c ok.
  std::uint64_t short_reads{0};        ///< Entries reported @c short_read.
  std::uint64_t failed{0};             ///< Entries reported @c failed (for any reason).
  std::uint64_t bytes{0};              ///< Bytes delivered by @c ok and @c short_read entries.
  std::uint64_t handle_resets{0};      ///< Batch handles recreated after a rejected submit.
  std::uint64_t deferred_flushes{0};   ///< Flushes that left entries staged (wave policies).
  std::uint64_t submit_ns{0};          ///< Time spent inside cuFileBatchIOSubmit.
  /// @c GetStatus calls that returned fewer events than were in flight although the policy
  /// waits for the whole wave (a wave slower than the 1 s per-call timeout); see the class
  /// comment for why they matter.
  std::uint64_t partial_status{0};
};

/**
 * @brief Keeps up to @ref capacity aligned host reads in flight through one cuFile batch
 *        handle (@c cuFileBatchIOSetUp / @c Submit / @c GetStatus / @c Cancel / @c Destroy).
 *
 * Usage, on one owner thread: @ref add entries while @ref free_entries is non-zero, @ref flush
 * them, and @ref reap completions; @ref drain before destruction.  The cookie of an entry is its
 * index (+1), so a completion is found in O(1).  Every array is allocated by the constructor.
 *
 * A helper thread owned by the reader makes the @c Submit and @c GetStatus calls: the owner hands
 * it a wave (one mutex-protected hand-off per wave, no allocation) and collects the completions
 * it publishes; the helper calls the wake callback after each publication, so the owner need not
 * poll.  The owner thread therefore never blocks in cuFile (a compatibility-mode @c Submit does
 * much of the I/O synchronously).  The helper never allocates.
 *
 * Why waves reaped whole: libcufile 1.17.1 (compatibility mode, measured on 2x RTX 6000 Ada)
 * leaves a dangling internal pointer whenever @c cuFileBatchIOGetStatus returns fewer events than
 * the handle has in flight -- a poll that finds nothing, a timeout, or a partial return with
 * @c min_nr below the outstanding count.  The next completion dereferences it, and if the calling
 * thread's allocator reused that chunk meanwhile the process crashes in @c GetStatus (SIGSEGV).
 * Under @ref batch_submit_policy::exclusive and @ref batch_submit_policy::per_reader the helper
 * submits a wave only when the previous one was fully reaped and waits with @c min_nr equal to the
 * whole wave (1 s per call; a slower wave is counted in @c partial_status), so such returns do not
 * occur; waves of different readers may overlap (measured safe with heap spraying).
 * @ref batch_submit_policy::continuous refills while entries are pending and reaps with
 * @c min_nr 1, which exposes the bug on that library.
 *
 * Every entry passes its own destination as @c devPtr_base with @c devPtr_offset 0: libcufile
 * 1.17 in compatibility mode silently ignores a non-zero @c devPtr_offset on host memory and
 * lands every entry at the base.
 *
 * Failures:
 * - A rejected @c Submit fails every entry of that call.  The helper recreates the handle
 *   (@c Cancel, @c Destroy, @c SetUp) once nothing else is in flight on it, and only then are the
 *   rejected entries reported -- so cuFile can no longer touch their destinations when the owner
 *   sees the failure.  After @c failure_limit consecutive rejected submits (a successful one resets
 *   the count), or when the recreation fails, the reader becomes unavailable for good (logged
 *   once).
 * - A @c GetStatus error breaks the handle: the helper cancels and destroys it at once and every
 *   outstanding or staged entry is reported @c failed; the reader is unavailable for good.
 *
 * Not thread-safe: every call, including destruction, on the owner's thread (the wake callback
 * runs on the helper thread).
 */
class host_batch_reader {
 public:
  /// Called on the helper thread after it published completions; must be noexcept and cheap.
  using wake_fn = void (*)(void* context) noexcept;

  /**
   * @brief Set up one batch handle of @p depth entries, clamped to cuFile's
   *        @c max_batch_io_size when that is reported, and start the helper thread.
   *
   * Opens the cuFile driver (@c cufile_api::open_driver) first.  Never throws, never blocks on
   * I/O: without the cuFile batch group, with @p depth 0, or when @c SetUp, the allocation or the
   * thread fails, @ref available is false and @ref free_entries 0.  The cuFile table current at
   * construction (@c cufile_api::get()) serves the reader for its whole life.
   *
   * @param depth          entries in flight at most.
   * @param failure_limit  consecutive rejected submits before giving up (0 is taken as 1).
   * @param policy         when submits may run; see @ref batch_submit_policy.  The readers
   *                       alive first fix the process policy: a reader constructed with
   *                       another one meanwhile logs a warning once and uses the process
   *                       policy (@ref policy reports the one in effect).
   * @param wake           called (with @p wake_context) on the helper thread after it published
   *                       completions; may be null.
   */
  host_batch_reader(unsigned depth,
                    unsigned failure_limit,
                    batch_submit_policy policy = batch_submit_policy::exclusive,
                    wake_fn wake               = nullptr,
                    void* wake_context         = nullptr) noexcept;

  /// Stops the helper (after any wave in flight completed: a destination is never abandoned) and
  /// destroys the handle.  Requires @ref in_flight() == 0 (@ref drain first; asserted in debug
  /// builds).  No allocation, no throw.
  ~host_batch_reader();

  host_batch_reader(host_batch_reader const&)            = delete;
  host_batch_reader& operator=(host_batch_reader const&) = delete;
  host_batch_reader(host_batch_reader&&)                 = delete;
  host_batch_reader& operator=(host_batch_reader&&)      = delete;

  /// Whether the reader can take entries.
  [[nodiscard]] bool available() const noexcept { return !_disabled; }

  /// Entries the handle holds: @c depth clamped to @c max_batch_io_size (0 when unavailable
  /// from the start).
  [[nodiscard]] std::size_t capacity() const noexcept { return _capacity; }

  /// Entries handed to the helper (submitted, or about to be) and not yet reported.
  [[nodiscard]] std::size_t in_flight() const noexcept { return _in_flight; }

  /// Entries added and not yet handed to the helper (including a deferred wave).
  [[nodiscard]] std::size_t staged() const noexcept { return _staged; }

  /// Failed entries not yet reported by @ref reap.
  [[nodiscard]] std::size_t pending_failures() const noexcept { return _failed_count; }

  /// Entries @ref add can take now: capacity minus in flight, staged and unreported failures;
  /// 0 while unavailable.
  [[nodiscard]] std::size_t free_entries() const noexcept { return _disabled ? 0 : _free_top; }

  /// The submit policy in effect (see the constructor).
  [[nodiscard]] batch_submit_policy policy() const noexcept { return _policy; }

  /// Whether the helper published completions @ref reap has not collected yet.  Any thread.
  [[nodiscard]] bool completions_pending() const noexcept;

  /// Whether the last @ref flush left a wave staged (it waits for this reader's wave in flight
  /// or, under @c exclusive, for another reader's); @ref reap retries it.
  [[nodiscard]] bool deferred() const noexcept { return _deferred; }

  /// Diagnostic counters (updated by @ref reap).
  [[nodiscard]] host_batch_counters const& counters() const noexcept { return _counters; }

  /**
   * @brief Stage one entry for the next @ref flush.
   *
   * Requires @ref free_entries() > 0 (asserted).  The entry is reported exactly once by a later
   * @ref reap.  An entry that is not 4 KiB-aligned (see @ref batch_read) is a caller bug:
   * asserted in debug builds; a release build does not submit it and reports it @c failed
   * (@c CU_FILE_INVALID_VALUE).
   */
  void add(batch_read const& read) noexcept;

  /**
   * @brief Hand everything staged to the helper, which submits it in one
   *        @c cuFileBatchIOSubmit.
   *
   * No-op when nothing is staged.  Under @ref batch_submit_policy::exclusive and
   * @ref batch_submit_policy::per_reader the hand-off is deferred -- the entries stay staged --
   * while this reader's previous wave is not fully reported, and under @c exclusive also while
   * another reader's wave holds the process-wide gate; @ref reap hands the deferred wave over as
   * soon as it may.  Never blocks on I/O.
   */
  void flush() noexcept;

  /**
   * @brief Report finished entries through @p done.
   *
   * Collects what the helper published; when nothing is published and entries are in flight,
   * waits up to @p wait for a publication (0: never waits).  A deferred wave is retried first,
   * sleeping at most `min(wait, 100 us)` while another reader holds the gate and nothing of this
   * reader is in flight.  Never blocks longer than about @p wait.
   *
   * Each entry's index is freed before @p done sees it, so @p done may @ref add (and
   * @ref flush) again.  @p done is called as `done(batch_result const&)`; it must not throw (a
   * throwing @p done leaves the remaining results of this call unreported: the caller's bug) and
   * must not call @ref reap or @ref drain.
   *
   * @return The number of entries reported.
   */
  template <typename Done>
  std::size_t reap(std::chrono::microseconds wait, Done&& done)
  {
    std::size_t reported     = 0;
    std::size_t const events = collect(retry_deferred(wait));
    batch_result result;
    for (std::size_t i = 0; i < events; ++i) {
      if (take_event(i, result)) {
        ++reported;
        done(std::as_const(result));
      }
    }
    while (take_failure(result)) {
      ++reported;
      done(std::as_const(result));
    }
    std::ignore = retry_deferred(std::chrono::microseconds{0});
    return reported;
  }

  /**
   * @brief Flush, then reap until nothing is staged, in flight or pending report.
   *
   * Waits with a bounded timeout per call and logs a warning every few seconds while entries
   * stay outstanding; it never returns while cuFile may still write a destination.  Entries
   * @p done adds are flushed and drained too.  The reader stays usable afterwards.
   */
  template <typename Done>
  void drain(Done&& done)
  {
    auto const start  = std::chrono::steady_clock::now();
    auto next_warning = start + drain_warning_interval;
    for (;;) {
      flush();
      if (_in_flight == 0 && _failed_count == 0 && _staged == 0) { return; }
      std::ignore = reap(drain_poll, done);
      if (auto const now = std::chrono::steady_clock::now(); now >= next_warning) {
        log_drain_stall(now - start);
        next_warning = now + drain_warning_interval;
      }
    }
  }

 private:
  static constexpr std::chrono::microseconds drain_poll{10'000};
  static constexpr std::chrono::seconds drain_warning_interval{5};

  enum class entry_state : std::uint8_t { free, staged, in_flight, failed };

  struct entry {
    batch_read read;
    entry_state state{entry_state::free};
    int error{0};
  };

  /// State shared with the helper thread (defined in the source).
  struct shared_state;

  /// Hand a deferred wave over when allowed; with @p wait > 0 and nothing in flight, sleep up
  /// to min(wait, 100 us) for the gate first.  Returns what is left of @p wait.
  std::chrono::microseconds retry_deferred(std::chrono::microseconds wait) noexcept;
  /// Whether a wave may be handed over now (takes the process-wide gate under exclusive).
  bool may_submit() noexcept;
  void stop_waiting() noexcept;
  /// Take what the helper published (waiting up to @p wait when nothing is and entries are in
  /// flight): events into @c _events (returns their count), rejected and aborted entries into
  /// the failure list, counters, and the helper's verdicts (disabled, aborted).
  std::size_t collect(std::chrono::microseconds wait) noexcept;
  /// Turn event @p i into @p result and free its entry; false when it does not name an entry in
  /// flight.
  bool take_event(std::size_t i, batch_result& result) noexcept;
  /// Pop one failure into @p result and free its entry.
  bool take_failure(batch_result& result) noexcept;
  void mark_failed(std::uint32_t index, int error) noexcept;
  void disable(char const* why, int error) noexcept;
  void log_drain_stall(std::chrono::steady_clock::duration elapsed) const noexcept;
  void helper_loop() noexcept;

  cufile_api const* _api{nullptr};
  std::size_t _capacity{0};
  unsigned _failure_limit{1};

  std::unique_ptr<CUfileIOParams_t[]> _params;  ///< [0, _staged): the next wave.
  std::unique_ptr<CUfileIOEvents_t[]> _events;  ///< Events collected by the owner.
  std::unique_ptr<entry[]> _entries;
  std::unique_ptr<std::uint32_t[]> _free;    ///< [0, _free_top): free indices.
  std::unique_ptr<std::uint32_t[]> _failed;  ///< [0, _failed_count): failed, unreported.
  std::unique_ptr<shared_state> _shared;
  std::thread _helper;

  std::size_t _free_top{0};
  std::size_t _staged{0};
  std::size_t _in_flight{0};
  std::size_t _failed_count{0};
  batch_submit_policy _policy{batch_submit_policy::exclusive};
  bool _disabled{true};
  bool _deferred{false};    ///< A flush left entries staged.
  bool _waiting{false};     ///< Counted in the gate's waiters.
  bool _registered{false};  ///< Counted among the live readers of the process policy.
  std::chrono::steady_clock::time_point _yield_until{};

  host_batch_counters _counters;
};

}  // namespace cucascade::io::cufile::detail
