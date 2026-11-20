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

#include "io/cufile/fused_lanes.hpp"

#include "io/cufile/staging_pool.hpp"

#include <cucascade/error.hpp>
#include <cucascade/io/cufile/cufile_api.hpp>
#include <cucascade/io/cufile/platform.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/log/logging.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufile.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <tuple>

namespace cucascade::io::cufile::detail {

namespace {

/// Every lane read is fixed in buffer offset, file offset and size at submission and fully
/// 4 KiB-aligned (the planner guarantees it, resubmits keep it).
constexpr unsigned stream_flags =
  CU_FILE_STREAM_FIXED_BUF_OFFSET | CU_FILE_STREAM_FIXED_FILE_OFFSET |
  CU_FILE_STREAM_FIXED_FILE_SIZE | CU_FILE_STREAM_PAGE_ALIGNED_INPUTS;

/// What bytes_done holds until cuFile writes it.
constexpr ssize_t bytes_sentinel = std::numeric_limits<ssize_t>::min();

[[nodiscard]] constexpr std::uint64_t lane_bit(unsigned index) noexcept
{
  return std::uint64_t{1} << index;
}

[[nodiscard]] constexpr bool block_aligned(std::size_t value) noexcept
{
  return value % IO_BLOCK_SIZE == 0;
}

// ---------------------------------------------------------------------------
// The process-wide lane budget per device.
// ---------------------------------------------------------------------------

struct lane_budget {
  std::mutex mutex;
  std::map<int, std::size_t> used;
};

lane_budget& budget() noexcept
{
  static auto* const state = new lane_budget;  // intentionally never destroyed
  return *state;
}

[[nodiscard]] std::size_t acquire_lanes(int device, std::size_t wanted, std::size_t limit) noexcept
{
  try {
    auto& state = budget();
    std::lock_guard lock(state.mutex);
    auto& used        = state.used[device];
    auto const grant  = used < limit ? std::min(wanted, limit - used) : std::size_t{0};
    used             += grant;
    return grant;
  } catch (...) {
    return 0;
  }
}

void release_lanes(int device, std::size_t lanes) noexcept
{
  if (lanes == 0) return;
  auto& state = budget();
  std::lock_guard lock(state.mutex);
  if (auto const it = state.used.find(device); it != state.used.end()) {
    it->second -= std::min(lanes, it->second);
  }
}

// ---------------------------------------------------------------------------
// Test seams.
// ---------------------------------------------------------------------------

std::atomic<std::size_t> callback_failure_countdown{0};
std::atomic<std::size_t> copy_failure_countdown{0};
std::atomic<std::size_t> stream_create_failure_countdown{0};
std::atomic<std::size_t> copy_calls{0};

/// Copies per window from which they are issued as one batched call.
constexpr std::size_t batch_copies_from = 5;

/// Count one event against @p countdown; true when it reaches the armed one.
[[nodiscard]] bool consume(std::atomic<std::size_t>& countdown) noexcept
{
  auto current = countdown.load(std::memory_order_relaxed);
  while (current != 0) {
    if (countdown.compare_exchange_weak(current, current - 1, std::memory_order_relaxed)) {
      return current == 1;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Retired lane streams.  A lane set does not deregister or destroy its streams: what was
// measured with libcufile 1.17 (compatibility mode) rules out every alternative --
//   * cuFileStreamDeregister synchronizes the whole device and, while it waits, holds a lock
//     other cuFile calls of the process need: deregistering (inline at shutdown, or in a
//     background thread) waits for unrelated callers' GPU work and stalls the process's
//     cuFile I/O meanwhile;
//   * destroying a still-registered stream leaves cuFile with state for a handle CUDA may
//     hand out again;
//   * a stream one thread used for cuFileReadAsync fails it from another thread once the
//     first is gone, so streams cannot be pooled for later sets (each reactor has its own
//     worker thread).
// So a set's streams are retired: kept, registered and idle, for the life of the process
// (one small object per lane of every lane set built; the slots go back to the staging
// pool as before).
// ---------------------------------------------------------------------------

std::atomic<std::size_t> retired_stream_count{0};
/// Past this many retired streams the process is warned once.
constexpr std::size_t retired_stream_warning = 1025;

std::atomic<bool> warned_stream_register{false};
std::atomic<bool> warned_no_async{false};

}  // namespace

std::size_t lanes_in_use_for_testing(int device) noexcept
{
  auto& state = budget();
  std::lock_guard lock(state.mutex);
  auto const it = state.used.find(device);
  return it == state.used.end() ? 0 : it->second;
}

void set_lane_callback_failure_for_testing(std::size_t nth) noexcept
{
  callback_failure_countdown.store(nth, std::memory_order_relaxed);
}

void set_lane_copy_failure_for_testing(std::size_t nth) noexcept
{
  copy_failure_countdown.store(nth, std::memory_order_relaxed);
}

void set_lane_stream_create_failure_for_testing(std::size_t nth) noexcept
{
  stream_create_failure_countdown.store(nth, std::memory_order_relaxed);
}

std::size_t retired_lane_streams_for_testing() noexcept
{
  return retired_stream_count.load(std::memory_order_relaxed);
}

std::size_t lane_copy_calls_for_testing() noexcept
{
  return copy_calls.load(std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// device_guard
// ---------------------------------------------------------------------------

fused_lane_set::device_guard::device_guard(int device) noexcept
{
  if (cudaGetDevice(&_previous) != cudaSuccess) {
    std::ignore = cudaGetLastError();
    return;
  }
  if (_previous == device) return;
  if (cudaSetDevice(device) == cudaSuccess) {
    _restore = true;
  } else {
    std::ignore = cudaGetLastError();
  }
}

fused_lane_set::device_guard::~device_guard()
{
  if (_restore) CUCASCADE_ASSERT_CUDA_SUCCESS(cudaSetDevice(_previous));
}

// ---------------------------------------------------------------------------
// Construction and teardown
// ---------------------------------------------------------------------------

fused_lane_set::fused_lane_set(lane_set_options const& options) noexcept : _options(options)
{
  auto const wanted = std::min(_options.lanes, max_lanes);
  if (wanted == 0 || _options.slot_bytes == 0 || !block_aligned(_options.slot_bytes) ||
      _options.device < 0) {
    return;
  }
  _api = cufile_api::get();
  if (_api == nullptr || !_api->has_async_read || _api->read_async == nullptr ||
      !cufile_api::open_driver()) {
    if (!warned_no_async.exchange(true)) {
      try {
        CUCASCADE_LOG_INFO(
          "cufile: cuFileReadAsync unavailable (library missing, too old or driver closed); "
          "fused lane reads disabled");
      } catch (...) {
        // Logging is best effort.
      }
    }
    _api = nullptr;
    return;
  }

  device_guard const guard{_options.device};
  auto const granted = acquire_lanes(_options.device, wanted, _options.max_device_lanes);
  for (unsigned index = 0; index < granted; ++index) {
    if (!build_lane(index)) break;
    _built = index + 1U;
    _usable |= lane_bit(index);
  }
  _free = _usable;
  release_lanes(_options.device, granted - _built);
  if (_built < wanted) {
    try {
      CUCASCADE_LOG_INFO("cufile: fused lane set on GPU {} runs {} of {} lanes (budget granted {})",
                         _options.device,
                         _built,
                         wanted,
                         granted);
    } catch (...) {
      // Logging is best effort.
    }
  }
}

bool fused_lane_set::wants_registered_streams() const noexcept
{
  return _options.register_streams && _api->has_stream_register &&
         _api->stream_register != nullptr && _api->stream_deregister != nullptr;
}

bool fused_lane_set::create_stream(lane& l) noexcept
{
  if (consume(stream_create_failure_countdown)) {
    l.stream = nullptr;
    return false;
  }
  if (cudaStreamCreateWithFlags(&l.stream, cudaStreamNonBlocking) != cudaSuccess) {
    std::ignore = cudaGetLastError();
    l.stream    = nullptr;
    return false;
  }
  l.stream_registered = false;
  if (wants_registered_streams()) {
    auto const status   = _api->stream_register(l.stream, stream_flags);
    l.stream_registered = status.err == CU_FILE_SUCCESS;
    if (!l.stream_registered && !warned_stream_register.exchange(true)) {
      try {
        CUCASCADE_LOG_WARN(
          "cufile: cuFileStreamRegister failed ({}); lane streams run unregistered",
          static_cast<int>(status.err));
      } catch (...) {
        // Logging is best effort.
      }
    }
  }
  return true;
}

void fused_lane_set::retire_stream(lane& l) noexcept
{
  if (l.stream == nullptr) return;
  // Kept for the life of the process (see "Retired lane streams" above).
  if (retired_stream_count.fetch_add(1, std::memory_order_relaxed) + 1 == retired_stream_warning) {
    try {
      CUCASCADE_LOG_WARN(
        "cufile: this process has retired {} fused-lane streams; build one cuFile ioctx per "
        "process: lane streams of destroyed ioctxs stay registered with cuFile until exit",
        retired_stream_warning);
    } catch (...) {
      // Logging is best effort.
    }
  }
  l.stream            = nullptr;
  l.stream_registered = false;
}

bool fused_lane_set::build_lane(unsigned index) noexcept
{
  auto& l = _lanes[index];
  l.slot  = staging_pool::instance().acquire(
    _options.device, _options.slot_bytes, _options.staging_cap_bytes, gds_available());
  if (!l.slot) return false;
  if (!create_stream(l)) {
    staging_pool::instance().release(l.slot);
    l.slot = {};
    return false;
  }
  if (_options.completion == lane_completion::poll &&
      cudaEventCreateWithFlags(&l.event, cudaEventDisableTiming) != cudaSuccess) {
    std::ignore = cudaGetLastError();
    l.event     = nullptr;
    retire_stream(l);
    staging_pool::instance().release(l.slot);
    l.slot = {};
    return false;
  }
  l.callback = callback_context{this, lane_bit(index), cudaSuccess};
  try {
    l.batch.reserve(64);
  } catch (...) {
    // Batching is an optimisation: windows with more copies than reserved fall back to
    // one call per copy if the batch cannot grow.
  }
  return true;
}

void fused_lane_set::destroy_lane(unsigned index) noexcept
{
  auto& l = _lanes[index];
  if (l.stream != nullptr) {
    // Also waits for a completion callback still returning from its wake call.  Waits only
    // for the lane's own read and copies: a lane stream never waits on a foreign event.
    if (cudaStreamSynchronize(l.stream) != cudaSuccess) {
      std::ignore = cudaGetLastError();
    }
  }
  retire_stream(l);
  if (l.event != nullptr) {
    CUCASCADE_ASSERT_CUDA_SUCCESS(cudaEventDestroy(l.event));
    l.event = nullptr;
  }
  if (l.slot) {
    staging_pool::instance().release(l.slot);
    l.slot = {};
  }
}

fused_lane_set::~fused_lane_set()
{
  if (_built == 0) return;
  device_guard const guard{_options.device};
  assert(in_flight() == 0 && "fused_lane_set destroyed with reads in flight");
  if (in_flight() != 0) {
    try {
      CUCASCADE_LOG_ERROR("cufile: fused lane set on GPU {} destroyed with {} reads in flight",
                          _options.device,
                          in_flight());
    } catch (...) {
      // Logging is best effort.
    }
  }
  // destroy_lane synchronizes each stream before anything is released.
  for (unsigned index = 0; index < _built; ++index) {
    destroy_lane(index);
  }
  release_lanes(_options.device, _built);
}

// ---------------------------------------------------------------------------
// Submission
// ---------------------------------------------------------------------------

unsigned fused_lane_set::pick_lane() noexcept
{
  // The first free lane at or after the cursor, cyclically: rotate the cursor to bit 0.
  auto const rotated = std::rotr(_free, static_cast<int>(_cursor));
  constexpr auto lanes = static_cast<unsigned>(max_lanes);
  auto const index     = (static_cast<unsigned>(std::countr_zero(rotated)) + _cursor) % lanes;
  _cursor              = (index + 1U) % lanes;
  return index;
}

bool fused_lane_set::submit(lane_read&& read) noexcept
{
  if (_free == 0) return false;
  device_guard const guard{_options.device};
  auto const index = pick_lane();
  auto const bit   = lane_bit(index);
  auto& l          = _lanes[index];
  _free &= ~bit;
  // Hand the lane's previous copy buffer back through `read`, so a caller that recycles its
  // lane_read allocates nothing per read.
  auto spare  = std::move(l.read.copies);
  l.read      = std::move(read);
  read.copies = std::move(spare);
  read.copies.clear();
  l.resubmits     = 0;
  l.holding       = l.read.hold;
  l.held_progress = 0;
  l.sync_error    = cudaSuccess;
  l.started       = std::chrono::steady_clock::now();
  issue(index, /*first=*/true);
  return true;
}

bool fused_lane_set::copy_out(unsigned lane_index, std::vector<lane_copy>&& copies) noexcept
{
  if (lane_index >= max_lanes || (_held & lane_bit(lane_index)) == 0) return false;
  device_guard const guard{_options.device};
  auto& l = _lanes[lane_index];
  _held &= ~l.callback.bit;
  l.holding = false;
  std::swap(l.read.copies, copies);
  copies.clear();
  auto const& win = l.read.window;
  for (auto const& copy : l.read.copies) {
    if (copy.offset < win.offset || copy.offset > win.end() ||
        copy.bytes > win.end() - copy.offset || (copy.bytes != 0 && copy.dst == nullptr)) {
      fail_early(lane_index, early_failure::io_failed, -EINVAL, cudaSuccess, false);
      return true;
    }
  }
  if (l.held_progress != 0) {
    // Short held read: the slot holds the window's first `held_progress` bytes.  Copy them
    // out, then read the rest into the slot (offset 0) behind those copies, on the lane.
    auto const progress = l.held_progress;
    l.held_progress     = 0;
    l.callback.status   = cudaSuccess;
    if (auto const error = queue_copies(lane_index, win.offset + progress);
        error != cudaSuccess) {
      fail_queued(lane_index, error);
      return true;
    }
    advance_window(l, progress);
    ++l.resubmits;
    issue(lane_index, /*first=*/false);
    return true;
  }
  l.callback.status       = cudaSuccess;
  l.callback.completed_ns = 0;
  l.read_issued           = {};  // copies only: polled at once
  auto error              = queue_copies(lane_index, std::numeric_limits<std::size_t>::max());
  if (error == cudaSuccess) error = queue_completion(lane_index);
  if (error != cudaSuccess) fail_queued(lane_index, error);
  return true;
}

void fused_lane_set::release_held(unsigned lane_index) noexcept
{
  if (lane_index >= max_lanes || (_held & lane_bit(lane_index)) == 0) return;
  auto& l = _lanes[lane_index];
  _held &= ~l.callback.bit;
  l.holding       = false;
  l.held_progress = 0;
  release_lane(lane_index);
}

void fused_lane_set::fail_early(unsigned index,
                                early_failure outcome,
                                ssize_t io_result,
                                cudaError_t cuda_status,
                                bool recycle) noexcept
{
  auto& l         = _lanes[index];
  l.early         = outcome;
  l.early_io      = io_result;
  l.early_cuda    = cuda_status;
  l.early_recycle = recycle;
  _completed.fetch_or(l.callback.bit, std::memory_order_acq_rel);
  if (_options.wake != nullptr) _options.wake(_options.wake_context);
}

void fused_lane_set::issue(unsigned index, bool first) noexcept
{
  auto& l     = _lanes[index];
  auto& read  = l.read;
  auto& win   = read.window;
  auto const fail_invalid = [&] {
    fail_early(index, early_failure::io_failed, -EINVAL, cudaSuccess, false);
  };

  if (first) {
    // A malformed read would break PAGE_ALIGNED_INPUTS or overrun the slot.
    if (read.handle == nullptr || win.size == 0 || win.size > _options.slot_bytes ||
        !block_aligned(win.offset) || !block_aligned(win.size) ||
        win.offset > static_cast<std::size_t>(std::numeric_limits<off_t>::max()) - win.size) {
      fail_invalid();
      return;
    }
    if (l.holding && (!read.copies.empty() || read.needed_bytes == 0 ||
                      read.needed_bytes > win.size)) {
      fail_invalid();
      return;
    }
    for (auto const& copy : read.copies) {
      if (copy.offset < win.offset || copy.offset > win.end() ||
          copy.bytes > win.end() - copy.offset ||
          (copy.bytes != 0 && copy.dst == nullptr)) {
        fail_invalid();
        return;
      }
    }
  }

  if (!start_read(index)) return;
  // A held read queues only its completion: its copies come with copy_out.
  auto error = l.holding ? cudaSuccess
                         : queue_copies(index, std::numeric_limits<std::size_t>::max());
  if (error == cudaSuccess) error = queue_completion(index);
  if (error != cudaSuccess) fail_queued(index, error);
}

bool fused_lane_set::start_read(unsigned index) noexcept
{
  auto& l          = _lanes[index];
  auto const& win  = l.read.window;
  // The arguments cuFile reads at execution; buffer_offset is always 0 (libcufile ignores a
  // non-zero one on some paths and lands the data at the base).
  l.size            = win.size;
  l.file_offset     = static_cast<off_t>(win.offset);
  l.buffer_offset   = 0;
  l.bytes_done            = bytes_sentinel;
  l.callback.status       = cudaSuccess;
  l.callback.completed_ns = 0;
  auto const timed        = _options.read_async_ns != nullptr ||
                     _options.completion == lane_completion::poll;
  auto const started      = timed ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
  l.read_issued           = started;
  auto const status       = _api->read_async(static_cast<CUfileHandle_t>(l.read.handle),
                                       l.slot.base,
                                       &l.size,
                                       &l.file_offset,
                                       &l.buffer_offset,
                                       &l.bytes_done,
                                       static_cast<CUstream>(l.stream));
  if (_options.read_async_ns != nullptr) {
    auto const spent = std::chrono::steady_clock::now() - started;
    _options.read_async_ns->fetch_add(
      static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(spent).count()),
      std::memory_order_relaxed);
  }
  if (status.err == CU_FILE_SUCCESS) return true;
  // Nothing was queued.
  auto const code = static_cast<ssize_t>(status.err);
  fail_early(index, early_failure::io_failed, code > 0 ? -code : -EIO, cudaSuccess, false);
  return false;
}

cudaError_t fused_lane_set::queue_copies(unsigned index, std::size_t limit) noexcept
{
  auto& l         = _lanes[index];
  auto const& win = l.read.window;
  auto const source = [&](lane_copy const& copy) { return l.slot.base + (copy.offset - win.offset); };
  auto const bytes_of = [&](lane_copy const& copy) -> std::size_t {
    if (copy.offset >= limit) return 0;
    return std::min(copy.bytes, limit - copy.offset);
  };

  std::size_t count = 0;
  for (auto const& copy : l.read.copies) {
    if (bytes_of(copy) != 0) ++count;
  }
  if (count >= batch_copies_from && !consume(copy_failure_countdown)) {
    // One call for the window's copies (cudaMemcpyBatchAsync where the runtime has it; the
    // lane stream is never a default stream).
    try {
      l.batch.clear();
      for (auto const& copy : l.read.copies) {
        if (auto const bytes = bytes_of(copy); bytes != 0) l.batch.add(copy.dst, source(copy), bytes);
      }
      copy_calls.fetch_add(1, std::memory_order_relaxed);
      auto const error = l.batch.enqueue(::cuda::stream_ref{l.stream});
      l.batch.clear();
      return error;
    } catch (...) {
      l.batch.clear();  // could not grow: one call per copy below
    }
  } else if (count >= batch_copies_from) {
    return cudaErrorInvalidValue;  // injected
  }
  for (auto const& copy : l.read.copies) {
    auto const bytes = bytes_of(copy);
    if (bytes == 0) continue;
    copy_calls.fetch_add(1, std::memory_order_relaxed);
    auto const error = consume(copy_failure_countdown)
                         ? cudaErrorInvalidValue
                         : cudaMemcpyAsync(copy.dst, source(copy), bytes, cudaMemcpyDeviceToDevice, l.stream);
    if (error != cudaSuccess) return error;
  }
  return cudaSuccess;
}

cudaError_t fused_lane_set::queue_completion(unsigned index) noexcept
{
  auto& l             = _lanes[index];
  bool const injected = consume(callback_failure_countdown);
  if (_options.completion == lane_completion::poll) {
    auto const error = injected ? cudaErrorInvalidResourceHandle : cudaEventRecord(l.event, l.stream);
    if (error == cudaSuccess) _event_pending |= l.callback.bit;
    return error;
  }
  return injected ? cudaErrorInvalidResourceHandle
                  : cudaStreamAddCallback(l.stream, &on_done, &l.callback, 0);
}

void fused_lane_set::fail_queued(unsigned index, cudaError_t error) noexcept
{
  // The read (and maybe some copies) is queued: wait for it (rare) so the lane is idle when
  // reported, and give the lane a fresh stream.
  auto& l          = _lanes[index];
  std::ignore      = cudaGetLastError();
  auto const drain = cudaStreamSynchronize(l.stream);
  if (drain != cudaSuccess) std::ignore = cudaGetLastError();
  fail_early(index, early_failure::device_failed, l.bytes_done, error, true);
}

void fused_lane_set::on_done(cudaStream_t, cudaError_t status, void* context) noexcept
{
  auto* const callback = static_cast<callback_context*>(context);
  auto* const set      = callback->set;
  // Copy what is needed before publishing: once the bit is visible the set may be reaped and
  // destroyed (its destructor waits for this callback to return, not for it to publish).
  auto const wake         = set->_options.wake;
  auto* const wake_target = set->_options.wake_context;
  callback->status        = status;
  callback->completed_ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now().time_since_epoch())
                             .count();
  set->_completed.fetch_or(callback->bit, std::memory_order_acq_rel);
  if (wake != nullptr) wake(wake_target);
}

// ---------------------------------------------------------------------------
// Completion
// ---------------------------------------------------------------------------

std::uint64_t fused_lane_set::take_finished() noexcept
{
  auto mask = _completed.exchange(0, std::memory_order_acq_rel);
  if (_options.completion != lane_completion::poll) return mask;
  auto pending = _event_pending & ~mask;
  if (pending == 0) return mask;
  // A read is not queried before half the recent mean read time passed since it was issued
  // (it is almost never done by then); unknown mean, or copies only: queried every time.
  // Skipping a query only delays seeing a completion, never loses it.
  auto const now        = std::chrono::steady_clock::now();
  auto const not_before = std::chrono::nanoseconds{_read_ewma_ns / 2};
  while (pending != 0) {
    auto const index = static_cast<unsigned>(std::countr_zero(pending));
    pending &= pending - 1;
    auto& l          = _lanes[index];
    bool const timed = l.read_issued != std::chrono::steady_clock::time_point{};
    if (timed && _read_ewma_ns != 0 && now - l.read_issued < not_before) continue;
    auto const status = cudaEventQuery(l.event);
    if (status == cudaErrorNotReady) continue;
    if (status != cudaSuccess) std::ignore = cudaGetLastError();
    if (timed && status == cudaSuccess) {
      auto const seen =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - l.read_issued).count();
      _read_ewma_ns = _read_ewma_ns == 0 ? seen : _read_ewma_ns + (seen - _read_ewma_ns) / 8;
    }
    l.callback.status = status;
    _event_pending &= ~l.callback.bit;
    mask |= l.callback.bit;
  }
  // cuFile's bytes_done was written before the event completed.
  std::atomic_thread_fence(std::memory_order_acquire);
  return mask;
}

void fused_lane_set::release_lane(unsigned index) noexcept
{
  auto& l   = _lanes[index];
  auto const bit = l.callback.bit;
  l.early         = early_failure::none;
  l.holding       = false;
  l.held_progress = 0;
  l.read.handle   = nullptr;
  l.read.copies.clear();
  _held &= ~bit;
  _event_pending &= ~bit;
  if ((_usable & bit) != 0) _free |= bit;
}

void fused_lane_set::recycle_stream(unsigned index) noexcept
{
  auto& l = _lanes[index];
  // Its completion callback may still be returning from `wake`.
  if (l.stream != nullptr && cudaStreamSynchronize(l.stream) != cudaSuccess) {
    std::ignore = cudaGetLastError();
  }
  retire_stream(l);
  bool usable = create_stream(l);
  if (usable && l.event != nullptr) {
    CUCASCADE_ASSERT_CUDA_SUCCESS(cudaEventDestroy(l.event));
    l.event = nullptr;
    if (cudaEventCreateWithFlags(&l.event, cudaEventDisableTiming) != cudaSuccess) {
      std::ignore = cudaGetLastError();
      l.event     = nullptr;
      usable      = false;
    }
  }
  if (!usable) {
    _usable &= ~l.callback.bit;
    try {
      CUCASCADE_LOG_WARN("cufile: fused lane {} on GPU {} lost its stream; {} lanes left",
                         index,
                         _options.device,
                         capacity());
    } catch (...) {
      // Logging is best effort.
    }
  }
}

bool fused_lane_set::settle(unsigned index, lane_result& result) noexcept
{
  auto& l = _lanes[index];
  if ((_usable & ~_free & l.callback.bit) == 0) return false;  // not in flight: stale bit
  result.token        = l.read.token;
  result.resubmits    = l.resubmits;
  result.lane         = index;
  result.completed_ns = l.callback.completed_ns;

  auto const finish = [&](lane_result::status outcome) {
    result.outcome = outcome;
    release_lane(index);
    return true;
  };

  if (l.early != early_failure::none) {
    result.io_result   = l.early_io;
    result.cuda_status = l.early_cuda;
    if (l.early_recycle) recycle_stream(index);
    return finish(l.early == early_failure::io_failed ? lane_result::status::io_failed
                                                      : lane_result::status::device_failed);
  }

  result.io_result = l.bytes_done;
  if (l.callback.status != cudaSuccess) {
    result.cuda_status = l.callback.status;
    recycle_stream(index);
    return finish(lane_result::status::device_failed);
  }
  if (l.bytes_done < 0) return finish(lane_result::status::io_failed);

  auto& win          = l.read.window;
  std::size_t needed = l.holding ? l.read.needed_bytes : 0;
  for (auto const& copy : l.read.copies) {
    needed = std::max(needed, copy.offset + copy.bytes - win.offset);
  }
  auto const done = static_cast<std::size_t>(l.bytes_done);
  // Short read: the bytes below `progress` (and, without hold, the copies within them)
  // landed.
  auto const progress = done / IO_BLOCK_SIZE * IO_BLOCK_SIZE;
  if (done < needed && (progress == 0 || l.resubmits >= _options.max_resubmits)) {
    return finish(lane_result::status::io_failed);
  }
  if (l.holding) {
    // Phase one done: keep the slot for the owner.  A short read continues in copy_out,
    // after the bytes already in the slot were copied out (the continuation reads into
    // offset 0 and would overwrite them).
    l.held_progress = done >= needed ? 0 : progress;
    _held |= l.callback.bit;
    result.outcome = lane_result::status::ok;
    result.held    = true;
    return true;
  }
  if (done >= needed) return finish(lane_result::status::ok);

  advance_window(l, progress);
  ++l.resubmits;
  issue(index, /*first=*/false);
  return false;
}

void fused_lane_set::advance_window(lane& l, std::size_t progress) noexcept
{
  auto& win = l.read.window;
  win.offset += progress;
  win.size -= progress;
  auto& copies = l.read.copies;
  copies.erase(std::remove_if(copies.begin(),
                              copies.end(),
                              [&](lane_copy const& c) { return c.offset + c.bytes <= win.offset; }),
               copies.end());
  for (auto& copy : copies) {
    if (copy.offset >= win.offset) continue;
    auto const skip = win.offset - copy.offset;
    copy.dst += skip;
    copy.offset += skip;
    copy.bytes -= skip;
  }
}

void fused_lane_set::fail_unreported(std::uint64_t mask) noexcept
{
  // Their streams reported an error; a lane whose completion has not arrived by now will not
  // get one (a sticky error stops callbacks): report it as failed rather than wait forever.
  auto const pending = _completed.load(std::memory_order_acquire);
  auto lanes         = mask & active_mask() & ~pending;
  while (lanes != 0) {
    auto const index = static_cast<unsigned>(std::countr_zero(lanes));
    lanes &= lanes - 1;
    auto& l = _lanes[index];
    if (l.early != early_failure::none) continue;
    _event_pending &= ~l.callback.bit;
    fail_early(index, early_failure::device_failed, l.bytes_done, l.sync_error, true);
  }
}

void fused_lane_set::repost(std::uint64_t mask) noexcept
{
  if (mask != 0) _completed.fetch_or(mask, std::memory_order_acq_rel);
}

std::uint64_t fused_lane_set::wait_busy_lanes() noexcept
{
  std::uint64_t failed = 0;
  auto busy            = active_mask();
  while (busy != 0) {
    auto const index = static_cast<unsigned>(std::countr_zero(busy));
    busy &= busy - 1;
    auto& l = _lanes[index];
    if (l.early != early_failure::none || l.stream == nullptr) continue;
    if (auto const status = cudaStreamSynchronize(l.stream); status != cudaSuccess) {
      std::ignore  = cudaGetLastError();
      l.sync_error = status;
      failed |= l.callback.bit;
    }
  }
  return failed;
}

std::size_t fused_lane_set::registered_streams() const noexcept
{
  std::size_t count = 0;
  auto usable       = _usable;
  while (usable != 0) {
    auto const index = static_cast<unsigned>(std::countr_zero(usable));
    usable &= usable - 1;
    if (_lanes[index].stream_registered) ++count;
  }
  return count;
}

std::chrono::steady_clock::duration fused_lane_set::oldest_in_flight() const noexcept
{
  auto busy = active_mask();
  if (busy == 0) return std::chrono::steady_clock::duration::zero();
  auto oldest = std::chrono::steady_clock::time_point::max();
  while (busy != 0) {
    auto const index = static_cast<unsigned>(std::countr_zero(busy));
    busy &= busy - 1;
    oldest = std::min(oldest, _lanes[index].started);
  }
  return std::chrono::steady_clock::now() - oldest;
}

}  // namespace cucascade::io::cufile::detail

