/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../pinned_host_test_hooks.hpp"

#include <cucascade/error.hpp>
#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <exception>
#include <limits>
#include <list>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace cucascade {
namespace memory {
namespace {

// Every function in this file clears the calling thread's CUDA error state only right after one of
// its own CUDA calls failed, so it never clears an unrelated error the caller has not checked yet.

// Portable: DMA-visible from every CUDA context, which multi-GPU consumers need. PR #115 observed
// CUDA 13's cudaMemcpyBatchAsync rejecting a cross-device source allocated without this flag
// (cudaErrorInvalidValue) on its multi-GPU host; other hosts accept such sources. Mapped:
// cuDF 26.04+ may read hostdevice_vector memory directly from kernels (for example
// detect_malformed_pages), and this resource advertises ::cuda::mr::device_accessible.
constexpr unsigned int pinned_host_alloc_flags = cudaHostAllocPortable | cudaHostAllocMapped;

[[nodiscard]] cudaError_t query_capture_status(::cuda::stream_ref stream,
                                               cudaStreamCaptureStatus& status) noexcept
{
  return CUCASCADE_PINNED_HOST_CUDA_CALL(stream_capture_status,
                                         cudaStreamIsCapturing(stream.get(), &status),
                                         {.stream = stream.get(), .capture_status_out = &status});
}

// Event creation must use the freeing stream's device even when another device is current.
class event_creation_device_guard {
 public:
  explicit event_creation_device_guard(int device) noexcept
  {
    int current_device = -1;
    if (CUCASCADE_PINNED_HOST_CUDA_CALL(current_device,
                                        cudaGetDevice(&current_device),
                                        {.device_out = &current_device}) != cudaSuccess) {
      std::ignore = cudaGetLastError();
      return;
    }
    if (current_device != device) {
      if (CUCASCADE_PINNED_HOST_CUDA_CALL(set_device, cudaSetDevice(device), {.device = device}) !=
          cudaSuccess) {
        std::ignore = cudaGetLastError();
        return;
      }
      _previous_device = current_device;
    }
    _is_ready = true;
  }

  event_creation_device_guard(event_creation_device_guard const&)            = delete;
  event_creation_device_guard& operator=(event_creation_device_guard const&) = delete;
  event_creation_device_guard(event_creation_device_guard&&)                 = delete;
  event_creation_device_guard& operator=(event_creation_device_guard&&)      = delete;

  ~event_creation_device_guard() noexcept
  {
    if (_previous_device >= 0 &&
        CUCASCADE_PINNED_HOST_CUDA_CALL(set_device,
                                        cudaSetDevice(_previous_device),
                                        {.device = _previous_device}) != cudaSuccess) {
      std::ignore = cudaGetLastError();
      std::terminate();
    }
  }

  [[nodiscard]] bool is_ready() const noexcept { return _is_ready; }

 private:
  int _previous_device{-1};
  bool _is_ready{false};
};

int device_of_stream(::cuda::stream_ref stream) noexcept
{
  int device = -1;
  if (CUCASCADE_PINNED_HOST_CUDA_CALL(stream_device,
                                      cudaStreamGetDevice(stream.get(), &device),
                                      {.device_out = &device, .stream = stream.get()}) ==
      cudaSuccess) {
    return device;
  }
  std::ignore = cudaGetLastError();
  return -1;
}

// Blocks until the work is complete. Returns false if the wait failed; the caller must then
// withhold storage the work may still use.
[[nodiscard]] bool wait_no_throw(::cuda::stream_ref stream) noexcept
{
  if (CUCASCADE_PINNED_HOST_CUDA_CALL(stream_synchronize,
                                      cudaStreamSynchronize(stream.get()),
                                      {.stream = stream.get()}) == cudaSuccess) {
    return true;
  }
  std::ignore = cudaGetLastError();
  return false;
}

[[nodiscard]] bool wait_no_throw(cuda::cuda_event const& event) noexcept
{
  if (event.synchronize_no_throw() == cudaSuccess) { return true; }
  std::ignore = cudaGetLastError();
  return false;
}

// A failed free has an unknown effect, so the buffer is leaked.
void free_pinned_no_throw(void* ptr) noexcept
{
  if (CUCASCADE_PINNED_HOST_CUDA_CALL(host_free, cudaFreeHost(ptr), {.buffer = ptr}) !=
      cudaSuccess) {
    std::ignore = cudaGetLastError();
  }
}

[[noreturn]] void throw_pinned_out_of_memory(std::size_t bytes)
{
  throw rmm::out_of_memory{"Pinned host allocation of " + std::to_string(bytes) +
                           " bytes failed after permitted recovery"};
}

// Returns whether completion was confirmed. An unsuccessful free still withholds the buffer.
template <typename Completion>
  requires requires(Completion const& completion) {
    { wait_no_throw(completion) } noexcept -> std::same_as<bool>;
  }
bool free_pinned_after(Completion const& completion, void* ptr) noexcept
{
  if (!wait_no_throw(completion)) { return false; }
  free_pinned_no_throw(ptr);
  return true;
}

}  // namespace

small_pinned_host_memory_resource::small_pinned_host_memory_resource(
  fixed_size_host_memory_resource& upstream, std::size_t large_cache_limit_bytes)
  : _upstream(upstream), _large_cache_limit_bytes(large_cache_limit_bytes)
{
}

small_pinned_host_memory_resource::~small_pinned_host_memory_resource() noexcept
{
  for (auto const& slab : _free_lists | std::views::join) {
    if (slab.ready && !wait_no_throw(slab.ready->event)) { quarantine_slab_locked(slab.ptr); }
  }
  for (auto const& entry : _large_cache | std::views::join) {
    free_pinned_after(entry.ready.event, entry.ptr);
  }
  for (auto& backing : _owned_allocations) {
    if (backing.quarantined) {
      // Leaking this handle keeps its blocks out of the upstream free list.
      std::ignore = backing.allocation.release();
    }
  }
  // Idle pooled events guard no storage, so the members' destructors release them without waiting.
}

void* small_pinned_host_memory_resource::allocate(::cuda::stream_ref stream,
                                                  std::size_t bytes,
                                                  std::size_t alignment)
{
  if (bytes == 0) { return nullptr; }
  if (!std::has_single_bit(alignment)) {
    throw std::invalid_argument{"Pinned allocation alignment must be a nonzero power of two"};
  }
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  CUCASCADE_CUDA_TRY(query_capture_status(stream, capture_status));
  if (capture_status != cudaStreamCaptureStatusNone) {
    CUCASCADE_FAIL("Pinned host allocation is unsupported during stream capture");
  }
  if (alignment > alignof(std::max_align_t)) {
    if (bytes > MAX_SLAB_SIZE) { _large_uncacheable.fetch_add(1, std::memory_order_relaxed); }
    return allocate_exceptional(bytes, alignment);
  }
  return bytes > MAX_SLAB_SIZE ? allocate_large(stream, bytes) : allocate_slab(stream, bytes);
}

void small_pinned_host_memory_resource::deallocate(::cuda::stream_ref stream,
                                                   void* ptr,
                                                   std::size_t bytes,
                                                   std::size_t alignment) noexcept
{
  std::ignore = try_to_deallocate(stream, ptr, bytes, alignment);
}

bool small_pinned_host_memory_resource::try_to_deallocate(::cuda::stream_ref stream,
                                                          void* ptr,
                                                          std::size_t bytes,
                                                          std::size_t alignment) noexcept
{
  if (ptr == nullptr || bytes == 0) { return false; }
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  auto const status                      = query_capture_status(stream, capture_status);
  if (status != cudaSuccess || capture_status != cudaStreamCaptureStatusNone) {
    if (status != cudaSuccess) { std::ignore = cudaGetLastError(); }
    std::lock_guard lock(_mutex);
    if (alignment <= alignof(std::max_align_t) && bytes <= MAX_SLAB_SIZE) {
      quarantine_slab_locked(ptr);
    } else {
      std::ignore = try_to_take_exceptional_locked(ptr);
    }
    return false;
  }
  if (alignment > alignof(std::max_align_t)) {
    deallocate_exceptional(stream, ptr);
    return true;
  }
  if (bytes > MAX_SLAB_SIZE) {
    deallocate_large(stream, ptr, bytes);
  } else {
    deallocate_slab(stream, ptr, bytes);
  }
  return true;
}

void small_pinned_host_memory_resource::deallocate_sync(void* ptr,
                                                        std::size_t bytes,
                                                        std::size_t alignment) noexcept
{
  ::cuda::stream_ref const stream{cudaStream_t{cudaStreamDefault}};
  if (try_to_deallocate(stream, ptr, bytes, alignment)) { std::ignore = wait_no_throw(stream); }
}

void* small_pinned_host_memory_resource::allocate_slab(::cuda::stream_ref stream, std::size_t bytes)
{
  std::size_t const idx = slab_index_for(bytes);
  free_slab slab        = [this, idx] {
    std::lock_guard lock(_mutex);
    auto& free_list = _free_lists[idx];
    if (free_list.empty()) { expand_pool_locked(idx); }
    free_slab taken = std::move(free_list.back());
    free_list.pop_back();
    return taken;
  }();
  try {
    if (slab.ready) { slab.ready->event.wait(stream); }
  } catch (...) {
    std::lock_guard lock(_mutex);
    auto& free_list = _free_lists[idx];
    // Growth reserves room for every slab, including this checked-out one.
    assert(free_list.size() < free_list.capacity());
    free_list.emplace_back(std::move(slab));
    throw;
  }
  void* const ptr = slab.ptr;
  if (slab.ready) { return_event(std::move(*slab.ready)); }
  return ptr;
}

void* small_pinned_host_memory_resource::allocate_exceptional(
  std::size_t bytes,
  std::size_t alignment,
  bool exact_attempt_failed,
  std::optional<::cuda::stream_ref> retry_stream)
{
  auto const padding = alignment > alignof(std::max_align_t) ? alignment - 1 : 0;
  if (bytes > std::numeric_limits<std::size_t>::max() - padding) {
    throw rmm::bad_alloc{"Pinned allocation size and alignment padding overflow"};
  }
  auto const alloc_bytes = bytes + padding;
  auto try_cached        = [&]() -> void* {
    if (!retry_stream) { return nullptr; }
    return try_to_take_cached_large(large_bucket_size_for(bytes), *retry_stream);
  };
  if (void* cached = try_cached()) { return cached; }
  std::map<void*, void*> pending;
  CUCASCADE_PINNED_HOST_METADATA_ALLOCATION(exceptional_allocation);
  pending.emplace(nullptr, nullptr);
  auto node = pending.extract(nullptr);

  void* base = nullptr;
  if (!exact_attempt_failed) {
    auto const status =
      CUCASCADE_PINNED_HOST_CUDA_CALL(host_alloc,
                                      cudaHostAlloc(&base, alloc_bytes, pinned_host_alloc_flags),
                                      {.bytes = alloc_bytes, .buffer_out = &base});
    if (status == cudaErrorMemoryAllocation) {
      std::ignore          = cudaGetLastError();
      exact_attempt_failed = true;
    } else {
      CUCASCADE_CUDA_TRY_ALLOC(status, alloc_bytes);
    }
  }
  if (exact_attempt_failed) {
    if (void* cached = try_cached()) { return cached; }
    if (!reclaim_large_cache(alloc_bytes)) { throw_pinned_out_of_memory(alloc_bytes); }
    CUCASCADE_CUDA_TRY_ALLOC(
      CUCASCADE_PINNED_HOST_CUDA_CALL(host_alloc,
                                      cudaHostAlloc(&base, alloc_bytes, pinned_host_alloc_flags),
                                      {.bytes = alloc_bytes, .buffer_out = &base}),
      alloc_bytes);
  }
  std::unique_ptr<void, decltype(&free_pinned_no_throw)> allocation{base, free_pinned_no_throw};
  void* ptr  = base;
  auto space = alloc_bytes;
  if (std::align(alignment, bytes, ptr, space) == nullptr) {
    throw rmm::bad_alloc{"Pinned allocation does not satisfy its alignment"};
  }
  node.key()    = ptr;
  node.mapped() = base;
  {
    std::lock_guard lock(_mutex);
    auto result = _exceptional_allocations.insert(std::move(node));
    if (!result.inserted) { CUCASCADE_FAIL("Duplicate exceptional pinned allocation"); }
  }
  std::ignore = allocation.release();
  return ptr;
}

void* small_pinned_host_memory_resource::try_to_take_exceptional_locked(void* ptr) noexcept
{
  auto const found = _exceptional_allocations.find(ptr);
  if (found == _exceptional_allocations.end()) { return nullptr; }
  void* base = found->second;
  _exceptional_allocations.erase(found);
  return base;
}

void small_pinned_host_memory_resource::deallocate_exceptional(::cuda::stream_ref stream,
                                                               void* ptr) noexcept
{
  void* base = nullptr;
  {
    std::lock_guard lock(_mutex);
    base = try_to_take_exceptional_locked(ptr);
  }
  assert(base != nullptr);
  if (base != nullptr) { free_pinned_after(stream, base); }
}

void* small_pinned_host_memory_resource::allocate_large(::cuda::stream_ref stream,
                                                        std::size_t bytes)
{
  std::size_t const bucket      = large_bucket_size_for(bytes);
  bool const cacheable          = is_cacheable(bucket);
  std::size_t const alloc_bytes = large_allocation_size(bytes);
  if (cacheable) {
    if (void* cached = try_to_take_cached_large(bucket, stream, true)) { return cached; }
  } else {
    _large_uncacheable.fetch_add(1, std::memory_order_relaxed);
  }

  void* ptr = nullptr;
  if (auto const status =
        CUCASCADE_PINNED_HOST_CUDA_CALL(host_alloc,
                                        cudaHostAlloc(&ptr, alloc_bytes, pinned_host_alloc_flags),
                                        {.bytes = alloc_bytes, .buffer_out = &ptr});
      status != cudaErrorMemoryAllocation) {
    // Returns on success. Releasing cached buffers cannot fix any other failure, so it throws.
    CUCASCADE_CUDA_TRY_ALLOC(status, alloc_bytes);
    return ptr;
  }
  std::ignore = cudaGetLastError();
  if (cacheable) {
    return allocate_exceptional(bytes, alignof(std::max_align_t), alloc_bytes == bytes, stream);
  }
  if (!reclaim_large_cache(bytes)) { throw_pinned_out_of_memory(bytes); }
  CUCASCADE_CUDA_TRY_ALLOC(
    CUCASCADE_PINNED_HOST_CUDA_CALL(host_alloc,
                                    cudaHostAlloc(&ptr, alloc_bytes, pinned_host_alloc_flags),
                                    {.bytes = alloc_bytes, .buffer_out = &ptr}),
    alloc_bytes);
  return ptr;
}

void small_pinned_host_memory_resource::deallocate_slab(::cuda::stream_ref stream,
                                                        void* ptr,
                                                        std::size_t bytes) noexcept
{
  std::size_t const idx = slab_index_for(bytes);
  auto ready            = record_ready_event(stream);
  if (!ready && !wait_no_throw(stream)) {
    std::lock_guard lock(_mutex);
    quarantine_slab_locked(ptr);
    return;
  }
  std::unique_lock lock(_mutex);
  try {
    // Pool growth reserves room for every slab, so valid deallocation does not grow this vector.
    CUCASCADE_PINNED_HOST_METADATA_ALLOCATION(slab_return);
    _free_lists[idx].emplace_back(ptr, std::move(ready));
  } catch (...) {
    quarantine_slab_locked(ptr);
    lock.unlock();
    if (ready) { return_event(std::move(*ready)); }
  }
}

void small_pinned_host_memory_resource::deallocate_large(::cuda::stream_ref stream,
                                                         void* ptr,
                                                         std::size_t bytes) noexcept
{
  std::size_t const bucket = large_bucket_size_for(bytes);
  if (!is_cacheable(bucket)) {
    free_pinned_after(stream, ptr);
    return;
  }
  std::list<large_cache_entry> incoming;
  std::list<large_cache_entry> victims;
  auto ready = record_ready_event(stream);
  if (ready) {
    try {
      CUCASCADE_PINNED_HOST_METADATA_ALLOCATION(large_cache_entry);
      incoming.emplace_back(ptr, std::move(*ready), 0);
      ready.reset();
    } catch (...) {
      // Failed node allocation leaves the event owned by ready and retains all cached entries.
    }
  }
  bool published = false;
  {
    std::lock_guard lock(_mutex);
    if (void* base = try_to_take_exceptional_locked(ptr)) {
      ptr = base;
      if (!incoming.empty()) { incoming.front().ptr = base; }
    } else if (!incoming.empty() && _large_cache_bytes <= _large_cache_limit_bytes) {
      while (bucket > _large_cache_limit_bytes - _large_cache_bytes &&
             evict_oldest_large_locked(victims)) {}
      if (bucket <= _large_cache_limit_bytes - _large_cache_bytes) {
        incoming.front().sequence = _large_cache_sequence;
        auto& entries             = _large_cache[large_bucket_index(bucket)];
        entries.splice(entries.end(), incoming);
        ++_large_cache_sequence;
        _large_cache_bytes += bucket;
        _peak_cached_large_bytes = std::max(_peak_cached_large_bytes, _large_cache_bytes);
        published                = true;
      }
    }
  }
  if (!published) {
    if (!incoming.empty()) {
      release_detached(std::move(incoming));
    } else if (ready) {
      if (free_pinned_after(ready->event, ptr)) {
        return_event(std::move(*ready));
      } else {
        retire_event(std::move(*ready));
      }
    } else {
      free_pinned_after(stream, ptr);
    }
  }
  release_detached(std::move(victims));
}

bool small_pinned_host_memory_resource::operator==(
  small_pinned_host_memory_resource const& other) const noexcept
{
  return this == &other;
}

std::size_t small_pinned_host_memory_resource::slab_index_for(std::size_t bytes) noexcept
{
  for (std::size_t i = 0; i < SLAB_SIZES.size(); ++i) {
    if (bytes <= SLAB_SIZES[i]) { return i; }
  }
  return SLAB_SIZES.size() - 1;
}

std::size_t small_pinned_host_memory_resource::large_bucket_size_for(std::size_t bytes) noexcept
{
  if (bytes <= MIN_LARGE_BUCKET) { return MIN_LARGE_BUCKET; }
  constexpr std::size_t max_bucket = std::size_t{1}
                                     << (std::numeric_limits<std::size_t>::digits - 1);
  if (bytes > max_bucket) { return bytes; }
  return std::bit_ceil(bytes);
}

std::size_t small_pinned_host_memory_resource::large_allocation_size(
  std::size_t bytes) const noexcept
{
  std::size_t const bucket = large_bucket_size_for(bytes);
  return is_cacheable(bucket) ? bucket : bytes;
}

std::size_t small_pinned_host_memory_resource::large_cache_bytes() const
{
  std::lock_guard lock(_mutex);
  return _large_cache_bytes;
}

small_pinned_host_memory_resource::statistics small_pinned_host_memory_resource::get_statistics()
  const
{
  std::scoped_lock lock(_mutex, _event_mutex);
  auto const uncacheable = _large_uncacheable.load(std::memory_order_relaxed);
  return {_large_cache_bytes,
          _peak_cached_large_bytes,
          _large_cacheable_requests + uncacheable,
          _large_hits,
          _large_misses,
          uncacheable,
          _large_evictions};
}

bool small_pinned_host_memory_resource::is_cacheable(std::size_t bucket) const noexcept
{
  return std::has_single_bit(bucket) && bucket <= _large_cache_limit_bytes;
}

std::size_t small_pinned_host_memory_resource::large_bucket_index(std::size_t bucket) noexcept
{
  assert(std::has_single_bit(bucket) && bucket >= MIN_LARGE_BUCKET);
  return static_cast<std::size_t>(std::countr_zero(bucket) - std::countr_zero(MIN_LARGE_BUCKET));
}

std::optional<small_pinned_host_memory_resource::pooled_event>
small_pinned_host_memory_resource::acquire_event(int device) noexcept
{
  if (device < 0) { return std::nullopt; }
  event_pool* pool = nullptr;
  try {
    std::lock_guard lock(_event_mutex);
    pool = std::addressof(_event_pools[device]);
    if (auto event = pool->try_to_take()) { return pooled_event{std::move(*event), device}; }
    pool->reserve();
  } catch (...) {
    return std::nullopt;
  }
  try {
    event_creation_device_guard const device_guard{device};
    if (device_guard.is_ready()) {
      return pooled_event{cuda::cuda_event{cudaEventDisableTiming}, device};
    }
  } catch (...) {
    // The throwing CUDA wrapper cleared the error. The return slot still belongs to this call.
  }
  std::lock_guard lock(_event_mutex);
  pool->release_reservation();
  return std::nullopt;
}

std::optional<small_pinned_host_memory_resource::pooled_event>
small_pinned_host_memory_resource::record_ready_event(::cuda::stream_ref stream) noexcept
{
  auto ready = acquire_event(device_of_stream(stream));
  if (!ready) { return std::nullopt; }
  try {
    ready->event.record(stream);
    return ready;
  } catch (...) {
    return_event(std::move(*ready));
    return std::nullopt;
  }
}

void small_pinned_host_memory_resource::return_event(pooled_event&& event) noexcept
{
  std::lock_guard lock(_event_mutex);
  return_event_locked(std::move(event));
}

void small_pinned_host_memory_resource::retire_event(pooled_event&& event) noexcept
{
  std::ignore = event.event.reset_no_throw();
  std::lock_guard lock(_event_mutex);
  retire_event_locked(std::move(event));
}

void small_pinned_host_memory_resource::return_event_locked(pooled_event&& event) noexcept
{
  auto const found = _event_pools.find(event.device);
  assert(found != _event_pools.end());
  found->second.release(std::move(event.event));
}

void small_pinned_host_memory_resource::retire_event_locked(pooled_event&& event) noexcept
{
  assert(event.event.get() == nullptr);
  auto const found = _event_pools.find(event.device);
  assert(found != _event_pools.end());
  found->second.release_reservation();
}

std::optional<cuda::cuda_event>
small_pinned_host_memory_resource::event_pool::try_to_take() noexcept
{
  if (_idle.empty()) { return std::nullopt; }
  cuda::cuda_event recycled = std::move(_idle.back());
  _idle.pop_back();
  return recycled;
}

void small_pinned_host_memory_resource::event_pool::reserve()
{
  if (_idle.capacity() == _live_events) {
    CUCASCADE_PINNED_HOST_METADATA_ALLOCATION(event_pool_reserve);
    _idle.reserve(_live_events + 1);
  }
  ++_live_events;
}

void small_pinned_host_memory_resource::event_pool::release_reservation() noexcept
{
  assert(_live_events > _idle.size());
  --_live_events;
}

void small_pinned_host_memory_resource::event_pool::release(cuda::cuda_event&& event) noexcept
{
  assert(_idle.size() < _idle.capacity());
  _idle.push_back(std::move(event));
}

void* small_pinned_host_memory_resource::try_to_take_cached_large(std::size_t bucket,
                                                                  ::cuda::stream_ref stream,
                                                                  bool initial_request)
{
  std::list<large_cache_entry> taken;
  {
    std::lock_guard lock(_mutex);
    if (initial_request) { ++_large_cacheable_requests; }
    auto& entries = _large_cache[large_bucket_index(bucket)];
    if (entries.empty()) {
      if (initial_request) { ++_large_misses; }
      return nullptr;
    }
    taken.splice(taken.end(), entries, entries.begin());
    _large_cache_bytes -= bucket;
  }
  auto& entry = taken.front();
  try {
    entry.ready.event.wait(stream);
  } catch (...) {
    {
      std::lock_guard lock(_mutex);
      if (_large_cache_bytes <= _large_cache_limit_bytes &&
          bucket <= _large_cache_limit_bytes - _large_cache_bytes) {
        auto& entries   = _large_cache[large_bucket_index(bucket)];
        auto const next = std::ranges::find_if(
          entries, [&entry](auto const& cached) { return cached.sequence > entry.sequence; });
        entries.splice(next, taken);
        _large_cache_bytes += bucket;
        _peak_cached_large_bytes = std::max(_peak_cached_large_bytes, _large_cache_bytes);
      }
    }
    if (!taken.empty()) {
      // Concurrent refill consumed the retained capacity. Withhold storage whose wait failed.
      retire_event(std::move(entry.ready));
    }
    throw;
  }
  void* const ptr = entry.ptr;
  {
    std::lock_guard lock(_event_mutex);
    return_event_locked(std::move(entry.ready));
    ++_large_hits;
  }
  return ptr;
}

bool small_pinned_host_memory_resource::evict_oldest_large_locked(
  std::list<large_cache_entry>& victims) noexcept
{
  auto non_empty =
    _large_cache | std::views::filter([](auto const& entries) { return !entries.empty(); });
  auto const oldest = std::ranges::min_element(
    non_empty, {}, [](auto const& entries) { return entries.front().sequence; });
  if (oldest == non_empty.end()) { return false; }
  victims.splice(victims.end(), *oldest, oldest->begin());
  auto const index = static_cast<std::size_t>(oldest.base() - _large_cache.begin());
  _large_cache_bytes -= MIN_LARGE_BUCKET << index;
  ++_large_evictions;
  return true;
}

bool small_pinned_host_memory_resource::reclaim_large_cache(std::size_t bytes)
{
  std::list<large_cache_entry> victims;
  auto remaining = bytes;
  {
    std::lock_guard lock(_mutex);
    if (_large_cache_bytes < bytes) { return false; }
    while (remaining != 0) {
      auto const before = _large_cache_bytes;
      if (!evict_oldest_large_locked(victims)) { break; }
      auto const reclaimed = before - _large_cache_bytes;
      remaining            = reclaimed >= remaining ? 0 : remaining - reclaimed;
    }
  }
  release_detached(std::move(victims));
  return remaining == 0;
}

void small_pinned_host_memory_resource::release_detached(
  std::list<large_cache_entry>&& detached) noexcept
{
  if (detached.empty()) { return; }
  for (auto& entry : detached) {
    if (!free_pinned_after(entry.ready.event, entry.ptr)) {
      std::ignore = entry.ready.event.reset_no_throw();
    }
  }
  std::lock_guard lock(_event_mutex);
  for (auto& entry : detached) {
    if (entry.ready.event.get() != nullptr) {
      return_event_locked(std::move(entry.ready));
    } else {
      retire_event_locked(std::move(entry.ready));
    }
  }
}

void small_pinned_host_memory_resource::expand_pool_locked(std::size_t slab_idx)
{
  std::size_t const upstream_block_size = _upstream.get_block_size();
  std::size_t const slab_size           = SLAB_SIZES[slab_idx];
  if (upstream_block_size < slab_size) {
    throw std::invalid_argument{"upstream block is smaller than the requested slab"};
  }
  auto allocation             = _upstream.allocate_multiple_blocks(upstream_block_size);
  std::size_t const num_slabs = upstream_block_size / slab_size;
  auto const blocks           = allocation->get_blocks();
  auto& free_list             = _free_lists[slab_idx];
  // Finish every potentially throwing metadata allocation before publishing raw slab pointers.
  free_list.reserve(free_list.capacity() + blocks.size() * num_slabs);
  _owned_allocations.push_back({std::move(allocation)});
  for (std::byte* block : blocks) {
    for (std::size_t i = 0; i < num_slabs; ++i) {
      free_list.emplace_back(block + i * slab_size, std::nullopt);
    }
  }
}

void small_pinned_host_memory_resource::quarantine_slab_locked(void* ptr) noexcept
{
  auto const address    = reinterpret_cast<std::uintptr_t>(ptr);
  auto const block_size = _upstream.get_block_size();
  for (auto& backing : _owned_allocations) {
    for (auto* block : backing.allocation->get_blocks()) {
      auto const base = reinterpret_cast<std::uintptr_t>(block);
      if (address >= base && address - base < block_size) {
        backing.quarantined = true;
        return;
      }
    }
  }
  assert(false && "slab pointer has no backing allocation");
}

}  // namespace memory
}  // namespace cucascade
