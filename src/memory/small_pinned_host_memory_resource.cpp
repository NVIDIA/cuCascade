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

#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <tuple>

namespace cucascade {
namespace memory {

small_pinned_host_memory_resource::small_pinned_host_memory_resource(
  fixed_size_host_memory_resource& upstream, std::size_t large_cache_limit_bytes)
  : _upstream(upstream), _large_cache_limit_bytes(large_cache_limit_bytes)
{
}

small_pinned_host_memory_resource::~small_pinned_host_memory_resource() noexcept
{
  bool retain_backing_allocations = _has_quarantined_slab;
  int previous_device             = -1;
  bool const have_previous        = (cudaGetDevice(&previous_device) == cudaSuccess);
  auto destroy_on                 = [&retain_backing_allocations](device_event event) {
    if (event.handle == nullptr) { return; }
    if (event.device >= 0) { std::ignore = cudaSetDevice(event.device); }
    // Event destruction alone does not wait before backing slabs return upstream.
    if (cudaEventSynchronize(event.handle) != cudaSuccess) {
      std::ignore                = cudaGetLastError();
      retain_backing_allocations = true;
    }
    CUCASCADE_ASSERT_CUDA_SUCCESS(cudaEventDestroy(event.handle));
  };
  for (auto const& list : _free_lists) {
    for (auto const& slab : list) {
      destroy_on(slab.ready);
    }
  }
  for (auto const& entries : _large_cache | std::views::values) {
    for (auto const& entry : entries) {
      sync_and_free_large_victim(entry);
    }
  }
  for (auto const& [device, events] : _event_pools) {
    for (auto event : events) {
      destroy_on({event, device});
    }
  }
  if (have_previous) { std::ignore = cudaSetDevice(previous_device); }
  if (retain_backing_allocations) {
    // Unknown completion requires retaining backing blocks, including quarantined slabs.
    for (auto& allocation : _owned_allocations) {
      std::ignore = allocation.release();
    }
  }
}

int small_pinned_host_memory_resource::device_of_stream(::cuda::stream_ref stream) noexcept
{
  int device = -1;
  if (cudaStreamGetDevice(stream.get(), &device) == cudaSuccess) { return device; }
  std::ignore = cudaGetLastError();
  if (cudaGetDevice(&device) == cudaSuccess) { return device; }
  std::ignore = cudaGetLastError();
  return -1;
}

small_pinned_host_memory_resource::device_event
small_pinned_host_memory_resource::acquire_event_locked(int device) noexcept
{
  if (device < 0) { return {}; }
  auto const found = _event_pools.find(device);
  if (found != _event_pools.end() && !found->second.empty()) {
    cudaEvent_t event = found->second.back();
    found->second.pop_back();
    return {event, device};
  }

  int previous_device = -1;
  if (cudaGetDevice(&previous_device) != cudaSuccess) {
    std::ignore = cudaGetLastError();
    return {};
  }
  bool const switch_device = previous_device != device;
  if (switch_device && cudaSetDevice(device) != cudaSuccess) {
    std::ignore = cudaGetLastError();
    return {};
  }
  cudaEvent_t event = nullptr;
  if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
    std::ignore = cudaGetLastError();
    event       = nullptr;
  }
  if (switch_device) { std::ignore = cudaSetDevice(previous_device); }
  return {event, device};
}

void small_pinned_host_memory_resource::release_event_locked(device_event event) noexcept
{
  if (event.handle == nullptr) { return; }
  try {
    _event_pools[event.device].push_back(event.handle);
  } catch (...) {
    // Previously enqueued waits capture the event record, not the handle's lifetime.
    CUCASCADE_ASSERT_CUDA_SUCCESS(cudaEventDestroy(event.handle));
  }
}

bool small_pinned_host_memory_resource::synchronize_stream_no_throw(
  ::cuda::stream_ref stream) noexcept
{
  if (cudaStreamSynchronize(stream.get()) == cudaSuccess) { return true; }
  std::ignore = cudaGetLastError();
  return false;
}

void* small_pinned_host_memory_resource::allocate(::cuda::stream_ref stream,
                                                  std::size_t bytes,
                                                  [[maybe_unused]] std::size_t alignment)
{
  if (bytes == 0) { return nullptr; }
  if (bytes > MAX_SLAB_SIZE) {
    std::size_t const bucket      = large_bucket_size_for(bytes);
    bool const cacheable          = bucket <= _large_cache_limit_bytes;
    std::size_t const alloc_bytes = cacheable ? bucket : bytes;
    if (cacheable) {
      std::lock_guard lock(_mutex);
      ++_large_cacheable_requests;
      if (void* cached = try_take_cached_large_locked(bucket, stream)) { return cached; }
      ++_large_misses;
    } else {
      _large_uncacheable.fetch_add(1, std::memory_order_relaxed);
    }
    void* ptr   = nullptr;
    auto status = cudaHostAlloc(&ptr, alloc_bytes, cudaHostAllocPortable | cudaHostAllocMapped);
    if (status == cudaSuccess) { return ptr; }
    std::ignore = cudaGetLastError();

    large_cache_type purged;
    {
      std::lock_guard lock(_mutex);
      purged = purge_large_cache_locked();
    }
    for (auto const& entries : purged | std::views::values) {
      for (auto const& victim : entries) {
        sync_and_free_large_victim(victim);
      }
    }
    if (cacheable) {
      std::lock_guard lock(_mutex);
      if (void* cached = try_take_cached_large_locked(bucket, stream)) { return cached; }
    }
    CUCASCADE_CUDA_TRY_ALLOC(
      cudaHostAlloc(&ptr, alloc_bytes, cudaHostAllocPortable | cudaHostAllocMapped), alloc_bytes);
    return ptr;
  }
  // COND: bytes <= MAX_SLAB_SIZE

  std::size_t const idx = slab_index_for(bytes);
  std::lock_guard lock(_mutex);
  if (_free_lists[idx].empty()) { expand_pool_locked(idx); }
  free_slab const slab = _free_lists[idx].back();
  if (slab.ready.handle != nullptr) {
    // A failed wait leaves the entry tracked and available for a later retry or destruction.
    CUCASCADE_CUDA_TRY(cudaStreamWaitEvent(stream.get(), slab.ready.handle, 0));
  }
  _free_lists[idx].pop_back();
  release_event_locked(slab.ready);
  return slab.ptr;
}

void small_pinned_host_memory_resource::deallocate(::cuda::stream_ref stream,
                                                   void* ptr,
                                                   std::size_t bytes,
                                                   [[maybe_unused]] std::size_t alignment) noexcept
{
  if (ptr == nullptr || bytes == 0) { return; }
  if (bytes > MAX_SLAB_SIZE) {
    std::size_t const bucket = large_bucket_size_for(bytes);
    if (bucket > _large_cache_limit_bytes) {
      if (synchronize_stream_no_throw(stream)) { CUCASCADE_ASSERT_CUDA_SUCCESS(cudaFreeHost(ptr)); }
      return;
    }
    int const device = device_of_stream(stream);
    while (true) {
      large_cache_entry victim{nullptr, {}, 0};
      device_event ready;
      bool release_instead = false;
      {
        std::lock_guard lock(_mutex);
        if (bucket <= _large_cache_limit_bytes - _large_cache_bytes) {
          ready = acquire_event_locked(device);
          if (ready.handle != nullptr &&
              cudaEventRecord(ready.handle, stream.get()) != cudaSuccess) {
            std::ignore = cudaGetLastError();
            release_event_locked(ready);
            ready = {};
          }
          if (ready.handle != nullptr) {
            try {
              _large_cache.try_emplace(bucket).first->second.push_back(
                {ptr, ready, _large_cache_sequence});
              ++_large_cache_sequence;
              _large_cache_bytes += bucket;
              _peak_cached_large_bytes = std::max(_peak_cached_large_bytes, _large_cache_bytes);
              return;
            } catch (...) {
              auto const found = _large_cache.find(bucket);
              if (found != _large_cache.end() && found->second.empty()) {
                _large_cache.erase(found);
              }
              // Cache metadata is optional; drain the recorded use before releasing storage.
              release_instead = true;
            }
          } else {
            release_instead = true;
          }
        } else {
          victim = evict_oldest_large_locked();
        }
      }
      if (release_instead) {
        if (ready.handle != nullptr) {
          sync_and_free_large_victim({ptr, ready, 0});
        } else if (synchronize_stream_no_throw(stream)) {
          CUCASCADE_ASSERT_CUDA_SUCCESS(cudaFreeHost(ptr));
        }
        return;
      }
      sync_and_free_large_victim(victim);
    }
  }
  // COND: bytes <= MAX_SLAB_SIZE

  std::size_t const idx = slab_index_for(bytes);
  int const device      = device_of_stream(stream);
  std::unique_lock lock(_mutex);
  device_event ready = acquire_event_locked(device);
  if (ready.handle != nullptr && cudaEventRecord(ready.handle, stream.get()) != cudaSuccess) {
    std::ignore = cudaGetLastError();
    release_event_locked(ready);
    ready = {};
  }
  if (ready.handle == nullptr) {
    lock.unlock();
    bool const synchronized = synchronize_stream_no_throw(stream);
    lock.lock();
    if (!synchronized) {
      _has_quarantined_slab = true;
      return;
    }
  }
  try {
    _free_lists[idx].push_back({ptr, ready});
  } catch (...) {
    _has_quarantined_slab = true;
    // The slab remains backed by _owned_allocations and will never be published again.
    if (ready.handle != nullptr) { CUCASCADE_ASSERT_CUDA_SUCCESS(cudaEventDestroy(ready.handle)); }
  }
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
  return bucket <= _large_cache_limit_bytes ? bucket : bytes;
}

std::size_t small_pinned_host_memory_resource::large_cache_bytes() const
{
  std::lock_guard lock(_mutex);
  return _large_cache_bytes;
}

small_pinned_host_memory_resource::statistics small_pinned_host_memory_resource::get_statistics()
  const
{
  std::lock_guard lock(_mutex);
  auto const uncacheable = _large_uncacheable.load(std::memory_order_relaxed);
  return {_large_cache_bytes,
          _peak_cached_large_bytes,
          _large_cacheable_requests + uncacheable,
          _large_hits,
          _large_misses,
          uncacheable,
          _large_evictions};
}

void* small_pinned_host_memory_resource::try_take_cached_large_locked(std::size_t bucket,
                                                                      ::cuda::stream_ref stream)
{
  auto found = _large_cache.find(bucket);
  if (found == _large_cache.end()) { return nullptr; }
  large_cache_entry const entry = found->second.front();
  if (entry.ready.handle != nullptr) {
    CUCASCADE_CUDA_TRY(cudaStreamWaitEvent(stream.get(), entry.ready.handle, 0));
  }
  found->second.pop_front();
  if (found->second.empty()) { _large_cache.erase(found); }
  _large_cache_bytes -= bucket;
  ++_large_hits;
  release_event_locked(entry.ready);
  return entry.ptr;
}

small_pinned_host_memory_resource::large_cache_entry
small_pinned_host_memory_resource::evict_oldest_large_locked() noexcept
{
  auto oldest = _large_cache.end();
  for (auto found = _large_cache.begin(); found != _large_cache.end(); ++found) {
    if (oldest == _large_cache.end() ||
        found->second.front().sequence < oldest->second.front().sequence) {
      oldest = found;
    }
  }
  if (oldest == _large_cache.end()) { return {nullptr, {}, 0}; }
  std::size_t const bucket      = oldest->first;
  large_cache_entry const entry = oldest->second.front();
  oldest->second.pop_front();
  if (oldest->second.empty()) { _large_cache.erase(oldest); }
  _large_cache_bytes -= bucket;
  ++_large_evictions;
  return entry;
}

small_pinned_host_memory_resource::large_cache_type
small_pinned_host_memory_resource::purge_large_cache_locked() noexcept
{
  large_cache_type entries;
  for (auto const& cached : _large_cache | std::views::values) {
    _large_evictions += static_cast<std::uint64_t>(cached.size());
  }
  entries.swap(_large_cache);
  _large_cache_bytes = 0;
  return entries;
}

void small_pinned_host_memory_resource::sync_and_free_large_victim(
  large_cache_entry const& victim) noexcept
{
  if (victim.ptr == nullptr) { return; }
  if (victim.ready.handle != nullptr && cudaEventSynchronize(victim.ready.handle) != cudaSuccess) {
    std::ignore = cudaGetLastError();
    // Completion is unknown; intentionally leak the buffer to avoid unpinning active storage.
    CUCASCADE_ASSERT_CUDA_SUCCESS(cudaEventDestroy(victim.ready.handle));
    return;
  }
  CUCASCADE_ASSERT_CUDA_SUCCESS(cudaFreeHost(victim.ptr));
  std::lock_guard lock(_mutex);
  release_event_locked(victim.ready);
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
  _owned_allocations.push_back(std::move(allocation));
  for (std::byte* block : blocks) {
    for (std::size_t i = 0; i < num_slabs; ++i) {
      free_list.push_back({block + i * slab_size, {}});
    }
  }
}

}  // namespace memory
}  // namespace cucascade
