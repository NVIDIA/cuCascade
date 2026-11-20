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
      CUCASCADE_LOG_INFO(
        "cufile: cuFileReadAsync unavailable (library missing, too old or driver closed); "
        "fused lane reads disabled");
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
    CUCASCADE_LOG_INFO("cufile: fused lane set on GPU {} runs {} of {} lanes (budget granted {})",
                       _options.device,
                       _built,
                       wanted,
                       granted);
  }
}

bool fused_lane_set::create_stream(lane& l) noexcept
{
  if (cudaStreamCreateWithFlags(&l.stream, cudaStreamNonBlocking) != cudaSuccess) {
    std::ignore = cudaGetLastError();
    l.stream    = nullptr;
    return false;
  }
  l.stream_registered = false;
  if (_options.register_streams && _api->has_stream_register && _api->stream_register != nullptr &&
      _api->stream_deregister != nullptr) {
    auto const status   = _api->stream_register(l.stream, stream_flags);
    l.stream_registered = status.err == CU_FILE_SUCCESS;
    if (!l.stream_registered && !warned_stream_register.exchange(true)) {
      CUCASCADE_LOG_WARN(
        "cufile: cuFileStreamRegister failed ({}); lane streams run unregistered",
        static_cast<int>(status.err));
    }
  }
  return true;
}

void fused_lane_set::destroy_stream(lane& l) noexcept
{
  if (l.stream == nullptr) return;
  if (l.stream_registered) std::ignore = _api->stream_deregister(l.stream);
  l.stream_registered = false;
  CUCASCADE_ASSERT_CUDA_SUCCESS(cudaStreamDestroy(l.stream));
  l.stream = nullptr;
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
    destroy_stream(l);
    staging_pool::instance().release(l.slot);
    l.slot = {};
    return false;
  }
  l.callback = callback_context{this, lane_bit(index), cudaSuccess};
  return true;
}

void fused_lane_set::destroy_lane(unsigned index) noexcept
{
  auto& l = _lanes[index];
  if (l.stream != nullptr) {
    // Also waits for a completion callback still returning from its wake call.
    if (cudaStreamSynchronize(l.stream) != cudaSuccess) std::ignore = cudaGetLastError();
  }
  destroy_stream(l);
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
    CUCASCADE_LOG_ERROR("cufile: fused lane set on GPU {} destroyed with {} reads in flight",
                        _options.device,
                        in_flight());
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
  l.read      = std::move(read);
  l.resubmits = 0;
  l.started   = std::chrono::steady_clock::now();
  if (l.read.gate != nullptr) _gated |= bit;
  issue(index, /*first=*/true);
  return true;
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
    for (auto const& copy : read.copies) {
      if (copy.offset < win.offset || copy.offset > win.end() ||
          copy.bytes > win.end() - copy.offset ||
          (copy.bytes != 0 && copy.dst == nullptr)) {
        fail_invalid();
        return;
      }
    }
  }

  // 1. The arguments cuFile reads at execution; buffer_offset is always 0 (libcufile ignores a
  //    non-zero one on some paths and lands the data at the base).
  l.size              = win.size;
  l.file_offset       = static_cast<off_t>(win.offset);
  l.buffer_offset     = 0;
  l.bytes_done        = bytes_sentinel;
  l.callback.status   = cudaSuccess;

  // 2. The read.
  auto const status = _api->read_async(static_cast<CUfileHandle_t>(read.handle),
                                       l.slot.base,
                                       &l.size,
                                       &l.file_offset,
                                       &l.buffer_offset,
                                       &l.bytes_done,
                                       static_cast<CUstream>(l.stream));
  if (status.err != CU_FILE_SUCCESS) {
    auto const code = static_cast<ssize_t>(status.err);
    fail_early(index, early_failure::io_failed, code > 0 ? -code : -EIO, cudaSuccess, false);
    return;
  }

  // 3. The gate (first issue only: on a resubmit it has passed).
  cudaError_t error = cudaSuccess;
  if (first && read.gate != nullptr) error = cudaStreamWaitEvent(l.stream, read.gate, 0);

  // 4. The copies out of the slot.
  for (auto const& copy : read.copies) {
    if (error != cudaSuccess) break;
    if (copy.bytes == 0) continue;
    error = consume(copy_failure_countdown)
              ? cudaErrorInvalidValue
              : cudaMemcpyAsync(copy.dst,
                                l.slot.base + (copy.offset - win.offset),
                                copy.bytes,
                                cudaMemcpyDeviceToDevice,
                                l.stream);
  }

  // 5. The completion.
  if (error == cudaSuccess) {
    bool const injected = consume(callback_failure_countdown);
    if (_options.completion == lane_completion::poll) {
      error = injected ? cudaErrorInvalidResourceHandle : cudaEventRecord(l.event, l.stream);
      if (error == cudaSuccess) _event_pending |= l.callback.bit;
    } else {
      error = injected ? cudaErrorInvalidResourceHandle
                       : cudaStreamAddCallback(l.stream, &on_done, &l.callback, 0);
    }
  }
  if (error == cudaSuccess) return;

  // The read is queued: wait for it (rare) so the lane is idle when reported.
  std::ignore      = cudaGetLastError();
  auto const drain = cudaStreamSynchronize(l.stream);
  if (drain != cudaSuccess) std::ignore = cudaGetLastError();
  fail_early(index, early_failure::device_failed, l.bytes_done, error, drain != cudaSuccess);
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
  while (pending != 0) {
    auto const index = static_cast<unsigned>(std::countr_zero(pending));
    pending &= pending - 1;
    auto& l           = _lanes[index];
    auto const status = cudaEventQuery(l.event);
    if (status == cudaErrorNotReady) continue;
    if (status != cudaSuccess) std::ignore = cudaGetLastError();
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
  l.early        = early_failure::none;
  l.read.gate    = nullptr;
  l.read.handle  = nullptr;
  l.read.copies.clear();
  _gated &= ~bit;
  _event_pending &= ~bit;
  if ((_usable & bit) != 0) _free |= bit;
}

void fused_lane_set::recycle_stream(unsigned index) noexcept
{
  auto& l = _lanes[index];
  destroy_stream(l);
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
    CUCASCADE_LOG_WARN("cufile: fused lane {} on GPU {} lost its stream; {} lanes left",
                       index,
                       _options.device,
                       capacity());
  }
}

bool fused_lane_set::settle(unsigned index, lane_result& result) noexcept
{
  auto& l = _lanes[index];
  if ((_usable & ~_free & l.callback.bit) == 0) return false;  // not in flight: stale bit
  result.token     = l.read.token;
  result.resubmits = l.resubmits;
  result.lane      = index;

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
  std::size_t needed = 0;
  for (auto const& copy : l.read.copies) {
    needed = std::max(needed, copy.offset + copy.bytes - win.offset);
  }
  auto const done = static_cast<std::size_t>(l.bytes_done);
  if (done >= needed) return finish(lane_result::status::ok);

  // Short read: the bytes below `progress` and the copies within them landed.
  auto const progress = done / IO_BLOCK_SIZE * IO_BLOCK_SIZE;
  if (progress == 0 || l.resubmits >= _options.max_resubmits) {
    return finish(lane_result::status::io_failed);
  }
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
  ++l.resubmits;
  l.read.gate = nullptr;
  _gated &= ~l.callback.bit;
  issue(index, /*first=*/false);
  return false;
}

void fused_lane_set::repost(std::uint64_t mask) noexcept
{
  if (mask != 0) _completed.fetch_or(mask, std::memory_order_acq_rel);
}

void fused_lane_set::wait_busy_lanes() noexcept
{
  auto busy = _usable & ~_free;
  while (busy != 0) {
    auto const index = static_cast<unsigned>(std::countr_zero(busy));
    busy &= busy - 1;
    auto& l = _lanes[index];
    if (l.early != early_failure::none || l.stream == nullptr) continue;
    if (cudaStreamSynchronize(l.stream) != cudaSuccess) std::ignore = cudaGetLastError();
  }
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
  auto busy = _usable & ~_free;
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
