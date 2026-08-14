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
#include <cassert>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace cucascade {
namespace memory {

small_pinned_host_memory_resource::small_pinned_host_memory_resource(
  fixed_size_host_memory_resource& upstream, std::size_t large_cache_limit_bytes)
  : upstream_(upstream), large_cache_limit_bytes_(large_cache_limit_bytes)
{
}

small_pinned_host_memory_resource::~small_pinned_host_memory_resource()
{
  // Slabs are suballocations released by owned_allocations_; only their CUDA events need explicit
  // cleanup.
  for (auto& pool : event_pools_) {
    for (auto& event : pool.second) {
      if (event != nullptr) { CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaEventDestroy(event)); }
    }
  }
  for (auto& list : free_lists_) {
    for (auto& slab : list) {
      if (slab.ready.handle != nullptr) {
        CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaEventDestroy(slab.ready.handle));
      }
    }
  }
  // Cached large buffers own their storage, so wait for recorded work before unpinning them.
  for (auto& bucket : large_cache_) {
    for (auto& entry : bucket.second) {
      sync_and_free_large_victim(entry);
      if (entry.ready.handle != nullptr) {
        CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaEventDestroy(entry.ready.handle));
      }
    }
  }
}

small_pinned_host_memory_resource::device_event
small_pinned_host_memory_resource::acquire_event_locked()
{
  int device = -1;
  if (::cudaGetDevice(&device) != cudaSuccess) {
    // Clear CUDA's sticky error and return an empty event for the caller's no-event fallback.
    (void)::cudaGetLastError();
    return {};
  }
  auto& pool = event_pools_[device];
  if (!pool.empty()) {
    cudaEvent_t event = pool.back();
    pool.pop_back();
    return {event, device};
  }
  cudaEvent_t event = nullptr;
  // Timing is not needed; disabling it makes record/wait cheaper.
  if (::cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
    // Clear CUDA's sticky error and return an empty event for the caller's no-event fallback.
    (void)::cudaGetLastError();
    return {};
  }
  return {event, device};
}

void small_pinned_host_memory_resource::release_event_locked(device_event event) noexcept
{
  if (event.handle != nullptr) {
    assert(event.device >= 0);
    event_pools_[event.device].push_back(event.handle);
  }
}

void* small_pinned_host_memory_resource::allocate(cuda::stream_ref stream,
                                                  std::size_t bytes,
                                                  [[maybe_unused]] std::size_t alignment)
{
  if (bytes == 0) { return nullptr; }
  // Large requests bypass upstream slabs but remain portable, mapped pinned memory for
  // cross-context use.
  if (bytes > MAX_SLAB_SIZE) {
    std::size_t const bucket      = large_bucket_size_for(bytes);
    bool const cacheable          = bucket <= large_cache_limit_bytes_;
    std::size_t const alloc_bytes = large_allocation_size(bytes);
    if (cacheable) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (void* cached = try_take_cached_large_locked(bucket, stream)) { return cached; }
    }
    void* ptr = nullptr;
    auto err  = ::cudaHostAlloc(&ptr, alloc_bytes, cudaHostAllocPortable | cudaHostAllocMapped);
    if (err == cudaSuccess) { return ptr; }
    // Clear the sticky error so a successful retry does not leave cudaGetLastError consumers
    // seeing a stale allocation failure.
    (void)::cudaGetLastError();
    // Release retained pinned memory before retrying. Wait and free outside the mutex because
    // event synchronization may block.
    std::vector<large_cache_entry> purged;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      purged = purge_large_cache_locked();
    }
    for (auto const& victim : purged) {
      sync_and_free_large_victim(victim);
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto const& victim : purged) {
        release_event_locked(victim.ready);
      }
      // A racing deallocate may have cached a matching buffer while we freed; take it if so.
      if (cacheable) {
        if (void* cached = try_take_cached_large_locked(bucket, stream)) { return cached; }
      }
    }
    err = ::cudaHostAlloc(&ptr, alloc_bytes, cudaHostAllocPortable | cudaHostAllocMapped);
    if (err != cudaSuccess) {
      (void)::cudaGetLastError();
      throw std::bad_alloc{};
    }
    return ptr;
  }

  std::size_t idx = slab_index_for(bytes);
  std::lock_guard<std::mutex> lock(mutex_);
  if (free_lists_[idx].empty()) { expand_pool_locked(idx); }
  free_slab slab = free_lists_[idx].back();
  free_lists_[idx].pop_back();
  // cudaStreamWaitEvent snapshots the event's current record, so the handle can be recycled after
  // the wait is enqueued.
  if (slab.ready.handle != nullptr) {
    CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaStreamWaitEvent(stream.get(), slab.ready.handle, 0));
    release_event_locked(slab.ready);
  }
  return slab.ptr;
}

void small_pinned_host_memory_resource::deallocate(cuda::stream_ref stream,
                                                   void* ptr,
                                                   std::size_t bytes,
                                                   [[maybe_unused]] std::size_t alignment) noexcept
{
  if (ptr == nullptr || bytes == 0) { return; }
  if (bytes > MAX_SLAB_SIZE) {
    std::size_t const bucket = large_bucket_size_for(bytes);
    if (bucket > large_cache_limit_bytes_) {
      // This bucket cannot fit under the retention cap, so bypass the cache.
      CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaFreeHost(ptr));
      return;
    }
    // Remove victims under the mutex, then wait and free outside it because event synchronization
    // may block.
    while (true) {
      large_cache_entry victim{nullptr, {}, 0};
      bool free_instead_of_caching = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (large_cache_bytes_ + bucket <= large_cache_limit_bytes_) {
          device_event event = acquire_event_locked();
          if (event.handle != nullptr &&
              ::cudaEventRecord(event.handle, stream.get()) != cudaSuccess) {
            (void)::cudaGetLastError();
            release_event_locked(event);
            event = {};
          }
          if (event.handle != nullptr) {
            large_cache_[bucket].push_back(large_cache_entry{ptr, event, large_cache_sequence_++});
            large_cache_bytes_ += bucket;
            return;
          }
          // Cache only after a successful record; otherwise reuse could race pending work.
          free_instead_of_caching = true;
        } else {
          victim = evict_oldest_large_locked();
        }
      }
      if (free_instead_of_caching) {
        // Without a recorded event, drain the deallocation stream before freeing. Deallocation is
        // noexcept, so clear synchronization errors and continue as a best-effort fallback.
        if (::cudaStreamSynchronize(stream.get()) != cudaSuccess) { (void)::cudaGetLastError(); }
        CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaFreeHost(ptr));
        return;
      }
      if (victim.ptr == nullptr) {
        // Unreachable while the bookkeeping is consistent (bucket <= limit means an empty
        // cache fits); free the buffer rather than loop forever.
        CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaFreeHost(ptr));
        return;
      }
      sync_and_free_large_victim(victim);
      if (victim.ready.handle != nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        release_event_locked(victim.ready);
      }
    }
  }

  std::size_t idx = slab_index_for(bytes);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // Record the final stream use so a later allocation can insert a dependency.
    device_event event = acquire_event_locked();
    if (event.handle != nullptr) {
      if (::cudaEventRecord(event.handle, stream.get()) == cudaSuccess) {
        free_lists_[idx].push_back(free_slab{ptr, event});
        return;
      }
      (void)::cudaGetLastError();
      release_event_locked(event);
    }
  }
  // Without a recorded event, synchronize before publishing the slab. Deallocation is noexcept, so
  // clear synchronization errors and continue as a best-effort fallback.
  if (::cudaStreamSynchronize(stream.get()) != cudaSuccess) { (void)::cudaGetLastError(); }
  // The push must happen after the synchronize completes; pushing first would let a racing
  // allocate hand the slab out before the freeing stream drains.
  std::lock_guard<std::mutex> lock(mutex_);
  free_lists_[idx].push_back(free_slab{ptr, {}});
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
  // std::bit_ceil is undefined when its result is not representable; use the exact request as the
  // bucket in that case.
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
  std::lock_guard<std::mutex> lock(mutex_);
  return large_cache_bytes_;
}

void* small_pinned_host_memory_resource::try_take_cached_large_locked(std::size_t bucket,
                                                                      ::cuda::stream_ref stream)
{
  auto it = large_cache_.find(bucket);
  if (it == large_cache_.end()) { return nullptr; }
  large_cache_entry entry = it->second.front();
  it->second.pop_front();
  if (it->second.empty()) { large_cache_.erase(it); }
  large_cache_bytes_ -= bucket;
  // cudaStreamWaitEvent snapshots the event's current record, so the handle can be recycled after
  // the wait is enqueued.
  if (entry.ready.handle != nullptr) {
    CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaStreamWaitEvent(stream.get(), entry.ready.handle, 0));
    release_event_locked(entry.ready);
  }
  return entry.ptr;
}

small_pinned_host_memory_resource::large_cache_entry
small_pinned_host_memory_resource::evict_oldest_large_locked() noexcept
{
  // Buckets never hold an empty deque (removals erase drained buckets), so every front() is
  // that bucket's oldest entry and the global oldest is the minimum sequence across fronts.
  auto oldest = large_cache_.end();
  for (auto it = large_cache_.begin(); it != large_cache_.end(); ++it) {
    if (oldest == large_cache_.end() ||
        it->second.front().sequence < oldest->second.front().sequence) {
      oldest = it;
    }
  }
  if (oldest == large_cache_.end()) { return large_cache_entry{nullptr, {}, 0}; }
  std::size_t const bucket = oldest->first;
  large_cache_entry entry  = oldest->second.front();
  oldest->second.pop_front();
  if (oldest->second.empty()) { large_cache_.erase(oldest); }
  large_cache_bytes_ -= bucket;
  return entry;
}

std::vector<small_pinned_host_memory_resource::large_cache_entry>
small_pinned_host_memory_resource::purge_large_cache_locked()
{
  std::vector<large_cache_entry> entries;
  std::size_t count = 0;
  for (auto& bucket : large_cache_) {
    count += bucket.second.size();
  }
  entries.reserve(count);
  for (auto& bucket : large_cache_) {
    for (auto& entry : bucket.second) {
      entries.push_back(entry);
    }
  }
  large_cache_.clear();
  large_cache_bytes_ = 0;
  return entries;
}

void small_pinned_host_memory_resource::sync_and_free_large_victim(
  large_cache_entry const& victim) noexcept
{
  // Wait for recorded use before unpinning. Deallocation is noexcept, so clear event errors and
  // free as a best-effort fallback.
  if (victim.ready.handle != nullptr &&
      ::cudaEventSynchronize(victim.ready.handle) != cudaSuccess) {
    (void)::cudaGetLastError();
  }
  CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaFreeHost(victim.ptr));
}

void small_pinned_host_memory_resource::expand_pool_locked(std::size_t slab_idx)
{
  std::size_t upstream_block_size = upstream_.get_block_size();
  auto allocation                 = upstream_.allocate_multiple_blocks(upstream_block_size);

  std::size_t slab_size = SLAB_SIZES[slab_idx];
  std::size_t num_slabs = upstream_block_size / slab_size;
  for (std::byte* block : allocation->get_blocks()) {
    for (std::size_t i = 0; i < num_slabs; ++i) {
      // Freshly-carved slabs were never used, so they carry no pending-work event.
      free_lists_[slab_idx].push_back(free_slab{block + i * slab_size, {}});
    }
  }
  owned_allocations_.push_back(std::move(allocation));
}

}  // namespace memory
}  // namespace cucascade
