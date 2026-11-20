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

#include <cucascade/io/types.hpp>

#include <cufile.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
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

/**
 * @brief When a @ref host_batch_reader may call @c cuFileBatchIOSubmit.
 *
 * libcufile 1.17.1 in compatibility mode corrupts its batch state when a @c Submit runs while
 * batch entries are in flight -- on the same handle or on another handle of the process -- and
 * later crashes inside @c cuFileBatchIOGetStatus (SIGSEGV; measured on 2x RTX 6000 Ada: 30-85%
 * of 3000-read runs at 32-64 entries in flight, also with one entry per submit).  Submitting a
 * whole wave only while nothing is in flight in the process never crashed.
 */
enum class batch_submit_policy : std::uint8_t {
  /// Submit only while no batch entry of any reader of the process is in flight: a reader takes
  /// a process-wide gate for one wave of entries and releases it once the wave has completed.
  /// Entries staged meanwhile wait (see @ref host_batch_reader::flush).  The safe default.
  exclusive,
  /// Submit whenever entries are staged, also while others are in flight (the batch API's
  /// documented semantics).  Only for a libcufile that handles it.  Ignores the gate, so do not
  /// mix it with exclusive readers in one process.
  continuous,
};

/// Diagnostic counters of a @ref host_batch_reader (plain integers, read by its owner).
struct host_batch_counters {
  std::uint64_t submits{0};           ///< Successful @c cuFileBatchIOSubmit calls.
  std::uint64_t submitted_entries{0};  ///< Entries those calls carried.
  std::uint64_t rejected_submits{0};  ///< Submit calls cuFile rejected.
  std::uint64_t completed{0};         ///< Entries reported @c ok.
  std::uint64_t short_reads{0};       ///< Entries reported @c short_read.
  std::uint64_t failed{0};            ///< Entries reported @c failed (for any reason).
  std::uint64_t bytes{0};             ///< Bytes delivered by @c ok and @c short_read entries.
  std::uint64_t handle_resets{0};     ///< Batch handles recreated after a rejected submit.
  std::uint64_t deferred_flushes{0};  ///< Flushes that left entries staged (exclusive policy).
  std::uint64_t submit_ns{0};         ///< Time spent inside cuFileBatchIOSubmit.
};

/**
 * @brief Keeps up to @ref capacity aligned host reads in flight through one cuFile batch
 *        handle (@c cuFileBatchIOSetUp / @c Submit / @c GetStatus / @c Cancel / @c Destroy).
 *
 * Usage, on one thread: @ref add entries while @ref free_entries is non-zero, @ref flush them
 * (one @c Submit), and @ref reap completions; @ref drain before destruction.  The parameter,
 * event and bookkeeping arrays are allocated once by the constructor; the cookie of an entry
 * is its index (+1), so a completion is found in O(1).  Nothing on the hot path allocates or
 * locks, and no call blocks longer than the wait it is given.
 *
 * Every entry passes its own destination as @c devPtr_base with @c devPtr_offset 0: libcufile
 * 1.17 in compatibility mode silently ignores a non-zero @c devPtr_offset on host memory and
 * lands every entry at the base.
 *
 * Failures:
 * - A rejected @c Submit fails every entry of that call.  The handle is then recreated
 *   (@c Cancel, @c Destroy, @c SetUp) as soon as nothing else is in flight on it, and only
 *   then are the rejected entries reported (by @ref reap) -- so cuFile can no longer touch their
 *   destinations when the owner sees the failure.  Meanwhile @ref free_entries is 0.  After
 *   @c failure_limit consecutive rejected submits (a successful one resets the count), or when
 *   the recreation fails, the reader becomes unavailable for good (logged once).
 * - A @c GetStatus error breaks the handle: it is cancelled and destroyed at once and every
 *   outstanding or staged entry is reported @c failed; the reader is unavailable for good.
 *
 * Not thread-safe: every call, including destruction, on the owner's thread.
 */
class host_batch_reader {
 public:
  /**
   * @brief Set up one batch handle of @p depth entries, clamped to cuFile's
   *        @c max_batch_io_size when that is reported.
   *
   * Opens the cuFile driver (@c cufile_api::open_driver) first.  Never throws, never blocks on
   * I/O: without the cuFile batch group, with @p depth 0, or when @c SetUp or the allocation
   * fails, @ref available is false and @ref free_entries 0.  The cuFile table current at
   * construction (@c cufile_api::get()) serves the reader for its whole life.
   *
   * @param depth          entries in flight at most.
   * @param failure_limit  consecutive rejected submits before giving up (0 is taken as 1).
   * @param policy         when submits may run; see @ref batch_submit_policy.
   */
  host_batch_reader(unsigned depth,
                    unsigned failure_limit,
                    batch_submit_policy policy = batch_submit_policy::exclusive) noexcept;

  /// Destroys the handle.  Requires @ref in_flight() == 0 (@ref drain first; asserted in debug
  /// builds; a release build cancels what is still in flight before destroying the handle).
  ~host_batch_reader();

  host_batch_reader(host_batch_reader const&)            = delete;
  host_batch_reader& operator=(host_batch_reader const&) = delete;
  host_batch_reader(host_batch_reader&&)                 = delete;
  host_batch_reader& operator=(host_batch_reader&&)      = delete;

  /// Whether the reader can take entries (now or once a pending handle reset is done).
  [[nodiscard]] bool available() const noexcept { return !_disabled; }

  /// Entries the handle holds: @c depth clamped to @c max_batch_io_size (0 when unavailable
  /// from the start).
  [[nodiscard]] std::size_t capacity() const noexcept { return _capacity; }

  /// Entries submitted and not yet reported.
  [[nodiscard]] std::size_t in_flight() const noexcept { return _in_flight; }

  /// Entries added and not yet submitted (including a deferred wave).
  [[nodiscard]] std::size_t staged() const noexcept { return _staged; }

  [[nodiscard]] batch_submit_policy policy() const noexcept { return _policy; }

  /// Failed entries (rejected, or aborted with the handle) not yet reported by @ref reap.
  [[nodiscard]] std::size_t pending_failures() const noexcept { return _failed_count; }

  /**
   * @brief Entries @ref add can take now: capacity minus in flight, staged and unreported
   *        failures; 0 while unavailable or while a rejected handle waits for its reset.
   */
  [[nodiscard]] std::size_t free_entries() const noexcept
  {
    return (_disabled || _needs_reset) ? 0 : _free_top;
  }

  /// Diagnostic counters.
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
   * @brief Submit everything staged in one @c cuFileBatchIOSubmit.
   *
   * On success the entries are in flight.  On rejection they are reported @c failed by a later
   * @ref reap, after the handle was reset (see the class description).  No-op when nothing is
   * staged.  Can take milliseconds in compatibility mode (cuFile does part of the work inside
   * @c Submit).
   *
   * Under @ref batch_submit_policy::exclusive the submit is deferred -- the entries stay staged
   * -- while this reader's previous wave is in flight or another reader holds the process-wide
   * gate; @ref reap submits the deferred wave as soon as it may.
   */
  void flush() noexcept;

  /**
   * @brief Report finished entries through @p done.
   *
   * With entries in flight, one @c cuFileBatchIOGetStatus: a non-blocking poll when @p wait is
   * 0, else it waits up to @p wait for the first completion (returning as soon as there is
   * one; never longer than @p wait).  Without entries in flight it returns at once, except that
   * a deferred wave (exclusive policy) is retried, sleeping at most `min(wait, 100 us)` while
   * another reader holds the gate.  Then it resets a rejected handle once nothing is in flight
   * on it, reports the failed entries that became reportable, and submits a deferred wave that
   * may now go.
   *
   * Each entry's index is freed before @p done sees it, so @p done may @ref add (and
   * @ref flush) again.  @p done is called as `done(batch_result const&)`; it must not throw (a
   * throwing @p done leaves the remaining events of this call unreported: the caller's bug) and
   * must not call @ref reap or @ref drain.
   *
   * @return The number of entries reported.
   */
  template <typename Done>
  std::size_t reap(std::chrono::microseconds wait, Done&& done)
  {
    std::size_t reported = 0;
    std::size_t const events = poll_events(retry_deferred(wait));
    batch_result result;
    for (std::size_t i = 0; i < events; ++i) {
      if (take_event(i, result)) {
        ++reported;
        done(std::as_const(result));
      }
    }
    settle();
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
   * Polls with a bounded wait per call and logs a warning every few seconds while entries stay
   * outstanding; it never returns while cuFile may still write a destination.  Entries
   * @p done adds are flushed and drained too.  The reader stays usable afterwards.
   */
  template <typename Done>
  void drain(Done&& done)
  {
    auto const start = std::chrono::steady_clock::now();
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

  /// Exclusive policy: submit a deferred wave when allowed; with @p wait > 0 and nothing in
  /// flight, sleep up to min(wait, 100 us) for the gate first.  Returns what is left of @p wait.
  std::chrono::microseconds retry_deferred(std::chrono::microseconds wait) noexcept;
  /// Whether a submit may run now (takes the process-wide gate under the exclusive policy).
  bool may_submit() noexcept;
  /// Release the gate once this reader's wave completed (exclusive policy).
  void release_gate() noexcept;
  void stop_waiting() noexcept;
  /// One GetStatus when entries are in flight (zero timeout when failures are ready to report);
  /// returns the number of events in @c _events.  On a GetStatus error, aborts the handle.
  std::size_t poll_events(std::chrono::microseconds wait) noexcept;
  /// Turn event @p i into @p result and free its entry; false when the event is not final or
  /// does not name an entry in flight.
  bool take_event(std::size_t i, batch_result& result) noexcept;
  /// Pop one reportable failure into @p result and free its entry.
  bool take_failure(batch_result& result) noexcept;
  /// Reset a rejected handle once nothing is in flight on it.
  void settle() noexcept;
  /// Cancel and destroy the handle, fail everything outstanding and disable the reader.
  void abort_handle(int error) noexcept;
  void mark_failed(std::uint32_t index, int error) noexcept;
  void disable(char const* why, int error) noexcept;
  void log_drain_stall(std::chrono::steady_clock::duration elapsed) const noexcept;

  cufile_api const* _api{nullptr};
  CUfileBatchHandle_t _handle{nullptr};
  std::size_t _capacity{0};
  unsigned _failure_limit{1};

  std::unique_ptr<CUfileIOParams_t[]> _params;  ///< [0, _staged): the next submit.
  std::unique_ptr<CUfileIOEvents_t[]> _events;
  std::unique_ptr<entry[]> _entries;
  std::unique_ptr<std::uint32_t[]> _free;    ///< [0, _free_top): free indices.
  std::unique_ptr<std::uint32_t[]> _failed;  ///< [0, _failed_count): failed, unreported.

  std::size_t _free_top{0};
  std::size_t _staged{0};
  std::size_t _in_flight{0};
  std::size_t _failed_count{0};
  unsigned _consecutive_rejections{0};
  batch_submit_policy _policy{batch_submit_policy::exclusive};
  bool _deferred{false};   ///< A flush left entries staged.
  bool _owns_gate{false};  ///< This reader holds the process-wide submit gate.
  bool _waiting{false};    ///< Counted in the gate's waiters.
  std::chrono::steady_clock::time_point _yield_until{};
  bool _needs_reset{false};  ///< A submit was rejected: reset once nothing is in flight.
  bool _disabled{true};

  host_batch_counters _counters;
};

}  // namespace cucascade::io::cufile::detail
