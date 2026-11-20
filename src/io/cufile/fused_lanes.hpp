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

/// One fused read: a 4 KiB-aligned file window read into a lane's slot, then copied out.
struct lane_read {
  void* handle{nullptr};          ///< CUfileHandle_t
  range window;                   ///< 4 KiB-aligned in offset and size, at most the slot
  std::vector<lane_copy> copies;  ///< ascending file offset, inside the window
  cudaEvent_t gate{nullptr};      ///< event the copies must follow, or null
  std::uint64_t token{0};         ///< the owner's, returned in the result
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
  /// Lanes holding a read not yet reported by @ref reap.
  [[nodiscard]] std::size_t in_flight() const noexcept
  {
    return static_cast<std::size_t>(std::popcount(_usable & ~_free));
  }
  [[nodiscard]] std::size_t free_lanes() const noexcept
  {
    return static_cast<std::size_t>(std::popcount(_free));
  }
  /// Reads in flight whose copies wait on a gate.
  [[nodiscard]] std::size_t gated() const noexcept
  {
    return static_cast<std::size_t>(std::popcount(_gated));
  }
  [[nodiscard]] std::size_t slot_bytes() const noexcept { return _options.slot_bytes; }
  [[nodiscard]] int device() const noexcept { return _options.device; }
  /// Usable lanes whose stream is registered with cuFile (diagnostics).
  [[nodiscard]] std::size_t registered_streams() const noexcept;

  /**
   * @brief Issue @p read on a free lane, chosen round-robin.
   *
   * Returns false, with @p read untouched, only when no lane is free.  Otherwise @p read is
   * moved into the lane and its outcome is reported by a later @ref reap or @ref drain --
   * including failures at submission (a refused @c cuFileReadAsync, a failed enqueue of the
   * gate wait, a copy or the completion, or a malformed read), which are reported only once
   * the lane is known idle, and which also call @c wake.  Never throws.  @p read.gate need
   * only stay valid for the call; the caller's destinations must stay valid until reported.
   */
  [[nodiscard]] bool submit(lane_read&& read) noexcept;

  /**
   * @brief Report every lane that finished since the last call; never blocks.
   *
   * Short reads are resubmitted on their lane (which stays busy) and not reported.  For each
   * finished read, @p done(lane_result const&) is called after its lane was freed, so @p done
   * may @ref submit.  A throwing @p done is the caller's bug: the exception propagates and the
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
   * @brief Block until nothing is in flight, reporting as @ref reap does.
   *
   * Waits on the busy lanes' streams (bounded only by CUDA and the storage); a lane is never
   * abandoned while cuFile may still write its slot.  Reads @p done submits are drained too.
   */
  template <typename Done>
  void drain(Done&& done)
  {
    device_guard const guard{_options.device};
    while (in_flight() != 0) {
      wait_busy_lanes();
      if (reap(done) == 0) std::this_thread::yield();
    }
  }

  /// Whether @ref reap has something to report now (callback mode), or may have (poll mode:
  /// true while any read is in flight).
  [[nodiscard]] bool completions_pending() const noexcept
  {
    if (_completed.load(std::memory_order_acquire) != 0) return true;
    return _options.completion == lane_completion::poll && in_flight() != 0;
  }

  /// How long the oldest read in flight has been (zero when idle); resubmits keep their start.
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
  [[nodiscard]] bool create_stream(lane& l) noexcept;
  void destroy_stream(lane& l) noexcept;
  void recycle_stream(unsigned index) noexcept;
  [[nodiscard]] unsigned pick_lane() noexcept;
  void issue(unsigned index, bool first) noexcept;
  void fail_early(unsigned index,
                  early_failure outcome,
                  ssize_t io_result,
                  cudaError_t cuda_status,
                  bool recycle) noexcept;
  [[nodiscard]] std::uint64_t take_finished() noexcept;
  [[nodiscard]] bool settle(unsigned index, lane_result& result) noexcept;
  void release_lane(unsigned index) noexcept;
  void repost(std::uint64_t mask) noexcept;
  void wait_busy_lanes() noexcept;

  lane_set_options _options;
  cufile_api const* _api{nullptr};
  std::size_t _built{0};          ///< lanes [0, _built) hold resources
  std::uint64_t _usable{0};       ///< built and not broken
  std::uint64_t _free{0};         ///< usable and idle
  std::uint64_t _gated{0};        ///< in flight with a gate
  std::uint64_t _event_pending{0};  ///< poll mode: event recorded, not yet seen complete
  unsigned _cursor{0};            ///< round-robin start of the next free-lane search
  std::atomic<std::uint64_t> _completed{0};
  std::array<lane, max_lanes> _lanes{};
};

/// Lanes all live sets hold on @p device (the process-wide budget in use).
[[nodiscard]] std::size_t lanes_in_use_for_testing(int device) noexcept;

/// Fail the @p nth completion installation from now on (@c cudaStreamAddCallback, or the
/// event record in poll mode), process-wide; 0 disarms.
void set_lane_callback_failure_for_testing(std::size_t nth) noexcept;

/// Fail the @p nth D2D copy enqueue from now on, process-wide; 0 disarms.
void set_lane_copy_failure_for_testing(std::size_t nth) noexcept;

}  // namespace cucascade::io::cufile::detail
