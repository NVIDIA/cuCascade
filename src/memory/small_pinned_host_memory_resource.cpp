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

#include <bit>
#include <limits>
#include <new>
#include <stdexcept>

namespace cucascade {
namespace memory {

small_pinned_host_memory_resource::small_pinned_host_memory_resource(
  fixed_size_host_memory_resource& upstream, std::size_t large_cache_limit_bytes)
  : upstream_(upstream), large_cache_limit_bytes_(large_cache_limit_bytes)
{
}

small_pinned_host_memory_resource::~small_pinned_host_memory_resource() noexcept
{
  bool retain_backing_allocations = has_quarantined_slab_;
  int previous_device             = -1;
  bool const have_previous        = (::cudaGetDevice(&previous_device) == cudaSuccess);
  auto destroy_on                 = [&retain_backing_allocations](device_event event) {
    if (event.handle == nullptr) { return; }
    if (event.device >= 0) { (void)::cudaSetDevice(event.device); }
    // Event destruction alone does not wait before backing slabs return upstream.
    if (::cudaEventSynchronize(event.handle) != cudaSuccess) {
      (void)::cudaGetLastError();
      retain_backing_allocations = true;
    }
    CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaEventDestroy(event.handle));
  };
  for (auto const& list : free_lists_) {
    for (auto const& slab : list) {
      destroy_on(slab.ready);
    }
  }
  for (auto const& [bucket, entries] : large_cache_) {
    (void)bucket;
    for (auto const& entry : entries) {
      sync_and_free_large_victim(entry);
    }
  }
  for (auto const& [device, events] : event_pools_) {
    for (auto event : events) {
      destroy_on({event, device});
    }
  }
  if (have_previous) { (void)::cudaSetDevice(previous_device); }
  if (retain_backing_allocations) {
    // Unknown completion requires retaining backing blocks, including quarantined slabs.
    for (auto& allocation : owned_allocations_) {
      (void)allocation.release();
    }
  }
}

int small_pinned_host_memory_resource::device_of_stream(::cuda::stream_ref stream) noexcept
{
  int device = -1;
  if (::cudaStreamGetDevice(stream.get(), &device) == cudaSuccess) { return device; }
  (void)::cudaGetLastError();
  if (::cudaGetDevice(&device) == cudaSuccess) { return device; }
  (void)::cudaGetLastError();
  return -1;
}

small_pinned_host_memory_resource::device_event
small_pinned_host_memory_resource::acquire_event_locked(int device) noexcept
{
  if (device < 0) { return {}; }
  auto const found = event_pools_.find(device);
  if (found != event_pools_.end() && !found->second.empty()) {
    cudaEvent_t event = found->second.back();
    found->second.pop_back();
    return {event, device};
  }

  int previous_device = -1;
  if (::cudaGetDevice(&previous_device) != cudaSuccess) {
    (void)::cudaGetLastError();
    return {};
  }
  bool const switch_device = previous_device != device;
  if (switch_device && ::cudaSetDevice(device) != cudaSuccess) {
    (void)::cudaGetLastError();
    return {};
  }
  cudaEvent_t event = nullptr;
  if (::cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
    (void)::cudaGetLastError();
    event = nullptr;
  }
  if (switch_device) { (void)::cudaSetDevice(previous_device); }
  return {event, device};
}

void small_pinned_host_memory_resource::release_event_locked(device_event event) noexcept
{
  if (event.handle == nullptr) { return; }
  try {
    event_pools_[event.device].push_back(event.handle);
  } catch (...) {
    // Previously enqueued waits capture the event record, not the handle's lifetime.
    CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaEventDestroy(event.handle));
  }
}

bool small_pinned_host_memory_resource::synchronize_stream_no_throw(
  ::cuda::stream_ref stream) noexcept
{
  if (::cudaStreamSynchronize(stream.get()) == cudaSuccess) { return true; }
  (void)::cudaGetLastError();
  return false;
}

void* small_pinned_host_memory_resource::allocate(::cuda::stream_ref stream,
                                                  std::size_t bytes,
                                                  [[maybe_unused]] std::size_t alignment)
{
  if (bytes == 0) { return nullptr; }
  if (bytes > MAX_SLAB_SIZE) {
    std::size_t const bucket      = large_bucket_size_for(bytes);
    bool const cacheable          = bucket <= large_cache_limit_bytes_;
    std::size_t const alloc_bytes = large_allocation_size(bytes);
    if (cacheable) {
      std::lock_guard lock(mutex_);
      if (void* cached = try_take_cached_large_locked(bucket, stream)) { return cached; }
    }
    void* ptr   = nullptr;
    auto status = ::cudaHostAlloc(&ptr, alloc_bytes, cudaHostAllocPortable | cudaHostAllocMapped);
    if (status == cudaSuccess) { return ptr; }
    (void)::cudaGetLastError();

    large_cache_type purged;
    {
      std::lock_guard lock(mutex_);
      purged = purge_large_cache_locked();
    }
    for (auto const& [purged_bucket, entries] : purged) {
      (void)purged_bucket;
      for (auto const& victim : entries) {
        sync_and_free_large_victim(victim);
      }
    }
    if (cacheable) {
      std::lock_guard lock(mutex_);
      if (void* cached = try_take_cached_large_locked(bucket, stream)) { return cached; }
    }
    status = ::cudaHostAlloc(&ptr, alloc_bytes, cudaHostAllocPortable | cudaHostAllocMapped);
    if (status != cudaSuccess) {
      (void)::cudaGetLastError();
      throw std::bad_alloc{};
    }
    return ptr;
  }

  std::size_t const idx = slab_index_for(bytes);
  std::lock_guard lock(mutex_);
  if (free_lists_[idx].empty()) { expand_pool_locked(idx); }
  free_slab const slab = free_lists_[idx].back();
  if (slab.ready.handle != nullptr) {
    // A failed wait leaves the entry tracked and available for a later retry or destruction.
    CUCASCADE_CUDA_TRY(::cudaStreamWaitEvent(stream.get(), slab.ready.handle, 0));
  }
  free_lists_[idx].pop_back();
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
    if (bucket > large_cache_limit_bytes_) {
      if (synchronize_stream_no_throw(stream)) {
        CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaFreeHost(ptr));
      }
      return;
    }
    while (true) {
      large_cache_entry victim{nullptr, {}, 0};
      device_event ready;
      bool release_instead = false;
      {
        std::lock_guard lock(mutex_);
        if (bucket <= large_cache_limit_bytes_ - large_cache_bytes_) {
          ready = acquire_event_locked(device_of_stream(stream));
          if (ready.handle != nullptr &&
              ::cudaEventRecord(ready.handle, stream.get()) != cudaSuccess) {
            (void)::cudaGetLastError();
            release_event_locked(ready);
            ready = {};
          }
          if (ready.handle != nullptr) {
            try {
              auto [found, inserted] = large_cache_.try_emplace(bucket);
              (void)inserted;
              found->second.push_back({ptr, ready, large_cache_sequence_});
              ++large_cache_sequence_;
              large_cache_bytes_ += bucket;
              return;
            } catch (...) {
              auto const found = large_cache_.find(bucket);
              if (found != large_cache_.end() && found->second.empty()) {
                large_cache_.erase(found);
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
          CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaFreeHost(ptr));
        }
        return;
      }
      sync_and_free_large_victim(victim);
    }
  }

  std::size_t const idx = slab_index_for(bytes);
  int const device      = device_of_stream(stream);
  std::unique_lock lock(mutex_);
  device_event ready = acquire_event_locked(device);
  if (ready.handle != nullptr && ::cudaEventRecord(ready.handle, stream.get()) != cudaSuccess) {
    (void)::cudaGetLastError();
    release_event_locked(ready);
    ready = {};
  }
  if (ready.handle == nullptr) {
    lock.unlock();
    bool const synchronized = synchronize_stream_no_throw(stream);
    lock.lock();
    if (!synchronized) {
      has_quarantined_slab_ = true;
      return;
    }
  }
  try {
    free_lists_[idx].push_back({ptr, ready});
  } catch (...) {
    has_quarantined_slab_ = true;
    // The slab remains backed by owned_allocations_ and will never be published again.
    if (ready.handle != nullptr) {
      CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaEventDestroy(ready.handle));
    }
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
  return bucket <= large_cache_limit_bytes_ ? bucket : bytes;
}

std::size_t small_pinned_host_memory_resource::large_cache_bytes() const
{
  std::lock_guard lock(mutex_);
  return large_cache_bytes_;
}

void* small_pinned_host_memory_resource::try_take_cached_large_locked(std::size_t bucket,
                                                                      ::cuda::stream_ref stream)
{
  auto found = large_cache_.find(bucket);
  if (found == large_cache_.end()) { return nullptr; }
  large_cache_entry const entry = found->second.front();
  if (entry.ready.handle != nullptr) {
    CUCASCADE_CUDA_TRY(::cudaStreamWaitEvent(stream.get(), entry.ready.handle, 0));
  }
  found->second.pop_front();
  if (found->second.empty()) { large_cache_.erase(found); }
  large_cache_bytes_ -= bucket;
  release_event_locked(entry.ready);
  return entry.ptr;
}

small_pinned_host_memory_resource::large_cache_entry
small_pinned_host_memory_resource::evict_oldest_large_locked() noexcept
{
  auto oldest = large_cache_.end();
  for (auto found = large_cache_.begin(); found != large_cache_.end(); ++found) {
    if (oldest == large_cache_.end() ||
        found->second.front().sequence < oldest->second.front().sequence) {
      oldest = found;
    }
  }
  if (oldest == large_cache_.end()) { return {nullptr, {}, 0}; }
  std::size_t const bucket      = oldest->first;
  large_cache_entry const entry = oldest->second.front();
  oldest->second.pop_front();
  if (oldest->second.empty()) { large_cache_.erase(oldest); }
  large_cache_bytes_ -= bucket;
  return entry;
}

small_pinned_host_memory_resource::large_cache_type
small_pinned_host_memory_resource::purge_large_cache_locked() noexcept
{
  large_cache_type entries;
  entries.swap(large_cache_);
  large_cache_bytes_ = 0;
  return entries;
}

void small_pinned_host_memory_resource::sync_and_free_large_victim(
  large_cache_entry const& victim) noexcept
{
  if (victim.ptr == nullptr) { return; }
  if (victim.ready.handle != nullptr &&
      ::cudaEventSynchronize(victim.ready.handle) != cudaSuccess) {
    (void)::cudaGetLastError();
    // Completion is unknown. Quarantine the buffer instead of unpinning active storage.
    CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaEventDestroy(victim.ready.handle));
    return;
  }
  CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaFreeHost(victim.ptr));
  std::lock_guard lock(mutex_);
  release_event_locked(victim.ready);
}

void small_pinned_host_memory_resource::expand_pool_locked(std::size_t slab_idx)
{
  std::size_t const upstream_block_size = upstream_.get_block_size();
  std::size_t const slab_size           = SLAB_SIZES[slab_idx];
  if (upstream_block_size < slab_size) {
    throw std::invalid_argument{"upstream block is smaller than the requested slab"};
  }
  auto allocation             = upstream_.allocate_multiple_blocks(upstream_block_size);
  std::size_t const num_slabs = upstream_block_size / slab_size;
  auto const blocks           = allocation->get_blocks();
  auto& free_list             = free_lists_[slab_idx];
  // Finish every potentially throwing metadata allocation before publishing raw slab pointers.
  free_list.reserve(free_list.capacity() + blocks.size() * num_slabs);
  owned_allocations_.push_back(std::move(allocation));
  for (std::byte* block : blocks) {
    for (std::size_t i = 0; i < num_slabs; ++i) {
      free_list.push_back({block + i * slab_size, {}});
    }
  }
}

}  // namespace memory
}  // namespace cucascade
