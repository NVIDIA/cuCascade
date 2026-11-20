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

/**
 * @file
 * @brief The fused lane set: @c cuFileReadAsync reads on per-lane registered slots and streams,
 *        with lock-free completion reporting.  See docs/io/gds-design/10-fused-lane-reads.md §6.
 *
 * Private to the cuFile backend and its tests.
 */

#include "io/cufile/staging_pool.hpp"

#include <cucascade/cuda/device_copy_batch.hpp>

#include <cucascade/io/cufile/config.hpp>
#include <cucascade/io/types.hpp>

#include <cuda_runtime_api.h>
#include <sys/types.h>

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

namespace cucascade::io::cufile {
struct cufile_api;
}  // namespace cucascade::io::cufile

namespace cucascade::io::cufile::detail {

/// One device-to-device copy out of a lane's slot: @p bytes of the file at @p offset to @p dst.
struct lane_copy {
  std::uint8_t* dst{nullptr};
  std::size_t offset{0};  ///< file offset
  std::size_t bytes{0};
};

/**
 * @brief One fused read: a 4 KiB-aligned file window read into a lane's slot, then copied out.
 *
 * A lane stream never waits on an event of another stream.  A caller whose destinations may
 * not be written yet submits the read with @ref hold: only the read is queued, the lane
 * reports a @c held result once @ref needed_bytes of the window arrived, and keeps the slot
 * until the owner calls @ref fused_lane_set::copy_out (queue the copies now) or
 * @ref fused_lane_set::release_held (drop the data).
 */
struct lane_read {
  void* handle{nullptr};          ///< CUfileHandle_t
  range window;                   ///< 4 KiB-aligned in offset and size, at most the slot
  std::vector<lane_copy> copies;  ///< ascending file offset, inside the window; empty with hold
  /// Two-phase read: read only, report @c held, wait for @c copy_out / @c release_held.
  bool hold{false};
  /// With @ref hold: bytes from the window's start that must arrive (the end of the last
  /// copy the owner will pass to @c copy_out, relative to the window); in (0, window.size].
  std::size_t needed_bytes{0};
  std::uint64_t token{0};  ///< the owner's, returned in the result
};

/// How a lane read ended.
struct lane_result {
  enum class status : std::uint8_t {
    ok,             ///< every copy holds the file's bytes
    io_failed,      ///< cuFile refused or failed the read, or it came back short
    device_failed,  ///< a CUDA enqueue failed, or the stream reported an error
  };
  std::uint64_t token{0};
  status outcome{status::ok};
  ssize_t io_result{0};  ///< last bytes_done (negative: cuFile error, -EINVAL for a bad read)
  cudaError_t cuda_status{cudaSuccess};
  unsigned resubmits{0};  ///< short-read continuations issued
  /// A @c hold read's I/O finished (@c outcome ok): the lane keeps the data in its slot until
  /// the owner calls @c copy_out or @c release_held.  A final result follows a @c copy_out.
  bool held{false};
  /// Callback completion: when the completion callback ran (steady clock, ns since its
  /// epoch); 0 in poll mode and for failures at submission.
  std::int64_t completed_ns{0};
  unsigned lane{0};       ///< the lane that served it (diagnostics, tests)
};

/// Construction parameters of a @ref fused_lane_set.
struct lane_set_options {
  using wake_fn = void (*)(void*) noexcept;

  int device{0};
  std::size_t lanes{0};              ///< wanted; at most 64
  std::size_t slot_bytes{0};         ///< a multiple of 4 KiB
  std::size_t staging_cap_bytes{0};  ///< staging_pool cap for this device (0 = none)
  std::size_t max_device_lanes{0};   ///< process-wide budget of lanes on @ref device
  bool register_streams{true};
  lane_completion completion{lane_completion::callback};
  unsigned max_resubmits{8};
  /// Called from the CUDA callback thread when a lane finishes (callback mode), and from
  /// @ref fused_lane_set::submit when a read fails at submission.  Must not block, throw or
  /// call into the set.  @ref wake_context must outlive the set.
  wake_fn wake{nullptr};
  void* wake_context{nullptr};
  /// When set, time spent inside @c cuFileReadAsync is added here (relaxed, ns).
  std::atomic<std::uint64_t>* read_async_ns{nullptr};
};

/**
 * @brief A fixed table of up to 64 lanes on one device, each a staging slot plus an owned
 *        non-blocking stream, serving one @c cuFileReadAsync at a time.
 *
 * Worker-thread only (one thread drives @ref submit, @ref reap, @ref drain and the queries);
 * the completion callback is the one exception and touches only an atomic word.  Every public
 * entry point runs with @c lane_set_options::device current and restores the caller's device.
 * Not copyable or movable: callbacks hold the lanes' addresses.
 */
class fused_lane_set {
 public:
  static constexpr std::size_t max_lanes = 64;

  /// Builds as many lanes as it can (never throws); @ref capacity may be 0.
  explicit fused_lane_set(lane_set_options const& options) noexcept;
  /// Requires @ref in_flight() == 0 (asserted; a release build waits for busy lanes first).
  /// Waits only for the lanes' own work.  The slots go back to the staging pool; the streams
  /// are retired, never deregistered: cuFileStreamDeregister synchronizes the device and
  /// stalls the process's cuFile calls meanwhile (see the source).
  ~fused_lane_set();

  fused_lane_set(fused_lane_set const&)            = delete;
  fused_lane_set& operator=(fused_lane_set const&) = delete;
  fused_lane_set(fused_lane_set&&)                 = delete;
  fused_lane_set& operator=(fused_lane_set&&)      = delete;

  /// Usable lanes.
  [[nodiscard]] std::size_t capacity() const noexcept
  {
    return static_cast<std::size_t>(std::popcount(_usable));
  }
  /// Lanes not free: a read or copies queued, or a held read the owner has not released.
  [[nodiscard]] std::size_t in_flight() const noexcept
  {
    return static_cast<std::size_t>(std::popcount(_usable & ~_free));
  }
  [[nodiscard]] std::size_t free_lanes() const noexcept
  {
    return static_cast<std::size_t>(std::popcount(_free));
  }
  /// Lanes in the held state: reported @c held, waiting for @ref copy_out / @ref release_held.
  [[nodiscard]] std::size_t held() const noexcept
  {
    return static_cast<std::size_t>(std::popcount(_held));
  }
  /// Lanes with work queued on their stream (or a failure not yet reported): in flight and
  /// not held.
  [[nodiscard]] std::size_t active() const noexcept
  {
    return static_cast<std::size_t>(std::popcount(active_mask()));
  }
  [[nodiscard]] std::size_t slot_bytes() const noexcept { return _options.slot_bytes; }
  [[nodiscard]] int device() const noexcept { return _options.device; }
  /// Usable lanes whose stream is registered with cuFile (diagnostics).
  [[nodiscard]] std::size_t registered_streams() const noexcept;

  /**
   * @brief Issue @p read on a free lane, chosen round-robin.
   *
   * Returns false, with @p read untouched, only when no lane is free.  Otherwise @p read is
   * moved into the lane (its @c copies then holds an empty buffer the caller may reuse for
   * the next read, so recycling a @c lane_read allocates nothing) and its outcome is
   * reported by a later @ref reap or @ref drain -- including failures at submission (a
   * refused @c cuFileReadAsync, a failed enqueue of a copy or the completion, or a malformed
   * read), which are reported only once the lane is known idle, and which also call @c wake.
   * Never throws.  The caller's destinations must stay valid until the final result.
   */
  [[nodiscard]] bool submit(lane_read&& read) noexcept;

  /**
   * @brief Queue the copies of the held read on @p lane, then its completion.
   *
   * The lane reports a final result through a later @ref reap (failures as for @ref submit).
   * When the held read was short, the bytes that arrived are copied first and the rest of
   * the window is read after them, on the lane.  @p copies receives the lane's previous
   * buffer, empty.  Never throws.
   *
   * @return false, with @p copies untouched, when @p lane is not held.
   */
  [[nodiscard]] bool copy_out(unsigned lane, std::vector<lane_copy>&& copies) noexcept;

  /// Drop the held read on @p lane and free the lane.  No-op when @p lane is not held.
  void release_held(unsigned lane) noexcept;

  /**
   * @brief Report every lane that finished since the last call; never blocks.
   *
   * Short reads are resubmitted on their lane (which stays busy) and not reported.  For each
   * finished read, @p done(lane_result const&) is called after its lane was freed -- or, for
   * a @c held result, after the lane entered the held state -- so @p done may @ref submit,
   * @ref copy_out or @ref release_held.  A throwing @p done is the caller's bug: the exception propagates and the
   * lanes not yet reported are reported by the next call.
   *
   * @return The number of results reported.
   */
  template <typename Done>
  std::size_t reap(Done&& done)
  {
    device_guard const guard{_options.device};
    auto mask         = take_finished();
    std::size_t count = 0;
    while (mask != 0) {
      auto const index = static_cast<unsigned>(std::countr_zero(mask));
      mask &= mask - 1;
      lane_result result;
      if (!settle(index, result)) continue;
      ++count;
      try {
        done(std::as_const(result));
      } catch (...) {
        repost(mask);
        throw;
      }
    }
    return count;
  }

  /**
   * @brief Block until no lane has work queued, reporting as @ref reap does.
   *
   * Waits on the active lanes' streams (bounded only by CUDA and the storage, never by a
   * caller's stream: lanes wait on no foreign event); a lane is never abandoned while cuFile
   * may still write its slot.  Held lanes have nothing queued and are not waited for: they
   * are the owner's to release (or to @ref copy_out, which makes them active again).  A lane
   * whose stream reports an error without its completion arriving is reported
   * @c device_failed.
   */
  template <typename Done>
  void drain(Done&& done)
  {
    device_guard const guard{_options.device};
    while (active() != 0) {
      auto const failed = wait_busy_lanes();
      auto const count  = reap(done);
      if (failed != 0) fail_unreported(failed);
      if (count == 0 && failed == 0) std::this_thread::yield();
    }
  }

  /// Whether @ref reap has something to report now (callback mode), or may have (poll mode:
  /// true while a lane is active).
  [[nodiscard]] bool completions_pending() const noexcept
  {
    if (_completed.load(std::memory_order_acquire) != 0) return true;
    return _options.completion == lane_completion::poll && active_mask() != 0;
  }

  /// How long the oldest active read has been in flight (zero when none); resubmits keep
  /// their start, held lanes do not count (they wait for their owner, not for storage).
  [[nodiscard]] std::chrono::steady_clock::duration oldest_in_flight() const noexcept;

 private:
  /// Sets a device for a scope and restores the previous one; never throws.
  class device_guard {
   public:
    explicit device_guard(int device) noexcept;
    ~device_guard();
    device_guard(device_guard const&)            = delete;
    device_guard& operator=(device_guard const&) = delete;

   private:
    int _previous{-1};
    bool _restore{false};
  };

  enum class early_failure : std::uint8_t { none, io_failed, device_failed };

  /// What the completion callback reads; stable for the set's lifetime.
  struct callback_context {
    fused_lane_set* set{nullptr};
    std::uint64_t bit{0};
    cudaError_t status{cudaSuccess};  ///< written by the callback before it publishes the bit
    std::int64_t completed_ns{0};     ///< likewise: when it ran (steady clock, ns)
  };

  struct lane {
    staging_slot slot;
    cudaStream_t stream{nullptr};
    cudaEvent_t event{nullptr};  ///< poll mode
    bool stream_registered{false};
    // The four arguments cuFile reads (and writes) at execution: stable addresses.
    std::size_t size{0};
    off_t file_offset{0};
    off_t buffer_offset{0};
    ssize_t bytes_done{0};
    callback_context callback;
    lane_read read;
    /// The read was submitted with @c hold and its copies have not been queued yet.
    bool holding{false};
    /// Held after a short read: bytes of the window in the slot, copied before the rest of
    /// the window is read (0: the whole needed range is in the slot).
    std::size_t held_progress{0};
    cudaError_t sync_error{cudaSuccess};  ///< drain: the stream reported this
    /// Poll mode: when the lane's current read was issued; epoch while it runs copies only
    /// (a held read's copy_out), which are never skipped by the poll.
    std::chrono::steady_clock::time_point read_issued{};
    cucascade::cuda::device_copy_batch batch;  ///< a window's copies as one call
    unsigned resubmits{0};
    std::chrono::steady_clock::time_point started{};
    early_failure early{early_failure::none};
    ssize_t early_io{0};
    cudaError_t early_cuda{cudaSuccess};
    bool early_recycle{false};
  };

  static void on_done(cudaStream_t stream, cudaError_t status, void* context) noexcept;

  [[nodiscard]] bool build_lane(unsigned index) noexcept;
  void destroy_lane(unsigned index) noexcept;
  [[nodiscard]] bool wants_registered_streams() const noexcept;
  /// A new non-blocking stream for @p l, registered with cuFile when asked and possible.
  [[nodiscard]] bool create_stream(lane& l) noexcept;
  /// Retire @p l's stream (idle, or failed): kept for the life of the process, never
  /// deregistered (see the source for why).
  void retire_stream(lane& l) noexcept;
  void recycle_stream(unsigned index) noexcept;
  [[nodiscard]] unsigned pick_lane() noexcept;
  [[nodiscard]] std::uint64_t active_mask() const noexcept { return _usable & ~_free & ~_held; }
  void issue(unsigned index, bool first) noexcept;
  [[nodiscard]] bool start_read(unsigned index) noexcept;
  /// Queue the copies of the lane's read, each clipped to end at or before @p limit (a
  /// file offset).
  [[nodiscard]] cudaError_t queue_copies(unsigned index, std::size_t limit) noexcept;
  [[nodiscard]] cudaError_t queue_completion(unsigned index) noexcept;
  /// A CUDA enqueue failed after the read was queued: wait for the lane and fail it.
  void fail_queued(unsigned index, cudaError_t error) noexcept;
  void advance_window(lane& l, std::size_t progress) noexcept;
  void fail_unreported(std::uint64_t mask) noexcept;
  void fail_early(unsigned index,
                  early_failure outcome,
                  ssize_t io_result,
                  cudaError_t cuda_status,
                  bool recycle) noexcept;
  [[nodiscard]] std::uint64_t take_finished() noexcept;
  [[nodiscard]] bool settle(unsigned index, lane_result& result) noexcept;
  void release_lane(unsigned index) noexcept;
  void repost(std::uint64_t mask) noexcept;
  /// Synchronize every active lane's stream; returns the lanes whose stream reported an error.
  [[nodiscard]] std::uint64_t wait_busy_lanes() noexcept;

  lane_set_options _options;
  cufile_api const* _api{nullptr};
  std::size_t _built{0};          ///< lanes [0, _built) hold resources
  std::uint64_t _usable{0};       ///< built and not broken
  std::uint64_t _free{0};         ///< usable and idle
  std::uint64_t _held{0};         ///< reported held, waiting for the owner
  std::uint64_t _event_pending{0};  ///< poll mode: event recorded, not yet seen complete
  unsigned _cursor{0};            ///< round-robin start of the next free-lane search
  std::atomic<std::uint64_t> _completed{0};
  /// Poll mode: running mean (EWMA, 1/8) of read issue -> completion seen, ns; 0 = unknown.
  std::int64_t _read_ewma_ns{0};
  std::array<lane, max_lanes> _lanes{};
};

/// Lanes all live sets hold on @p device (the process-wide budget in use).
[[nodiscard]] std::size_t lanes_in_use_for_testing(int device) noexcept;

/// Fail the @p nth completion installation from now on (@c cudaStreamAddCallback, or the
/// event record in poll mode), process-wide; 0 disarms.
void set_lane_callback_failure_for_testing(std::size_t nth) noexcept;

/// Fail the @p nth D2D copy enqueue from now on, process-wide; 0 disarms.
void set_lane_copy_failure_for_testing(std::size_t nth) noexcept;

/// Fail the @p nth lane stream creation (construction or recycling) from now on,
/// process-wide; 0 disarms.
void set_lane_stream_create_failure_for_testing(std::size_t nth) noexcept;

/// Lane streams retired (kept, never deregistered) so far, process-wide.
[[nodiscard]] std::size_t retired_lane_streams_for_testing() noexcept;

/// Copy calls (one per @c cudaMemcpyAsync, or one per batched window) lane sets issued so far.
[[nodiscard]] std::size_t lane_copy_calls_for_testing() noexcept;

}  // namespace cucascade::io::cufile::detail
