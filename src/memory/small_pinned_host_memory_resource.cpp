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

#include <cstdlib>
#include <stdexcept>

namespace cucascade {
namespace memory {

small_pinned_host_memory_resource::small_pinned_host_memory_resource(
  fixed_size_host_memory_resource& upstream)
  : upstream_(upstream)
{
}

small_pinned_host_memory_resource::~small_pinned_host_memory_resource()
{
  // owned_allocations_ destructor returns upstream blocks to the free list.
  // free_lists_ entries are raw pointers into those blocks; no individual cleanup needed.
  // Destroy every CUDA event we own — both idle (pooled) and still attached to a
  // free slab that was never re-allocated.
  for (auto& event : event_pool_) {
    if (event != nullptr) { CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaEventDestroy(event)); }
  }
  for (auto& list : free_lists_) {
    for (auto& slab : list) {
      if (slab.ready_event != nullptr) {
        CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaEventDestroy(slab.ready_event));
      }
    }
  }
}

cudaEvent_t small_pinned_host_memory_resource::acquire_event_locked()
{
  if (!event_pool_.empty()) {
    cudaEvent_t event = event_pool_.back();
    event_pool_.pop_back();
    return event;
  }
  cudaEvent_t event = nullptr;
  // Timing is not needed; disabling it makes record/wait cheaper.
  if (::cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
    // Best effort: without an event this deallocation loses stream ordering, but
    // allocation must never throw here. Clear the sticky error and carry on.
    (void)::cudaGetLastError();
    return nullptr;
  }
  return event;
}

void small_pinned_host_memory_resource::release_event_locked(cudaEvent_t event) noexcept
{
  if (event != nullptr) { event_pool_.push_back(event); }
}

void* small_pinned_host_memory_resource::allocate([[maybe_unused]] cuda::stream_ref stream,
                                                  std::size_t bytes,
                                                  [[maybe_unused]] std::size_t alignment)
{
  if (bytes == 0) { return nullptr; }
  // cuDF calls get_pinned_memory_resource() directly from some code paths (e.g. join/sort
  // staging buffers) that bypass the allocate_host_as_pinned threshold check.  Serve those
  // with cudaHostAlloc(Portable) so the memory remains pinned AND DMA-accessible from
  // every CUDA context (multi-GPU consumers need the Portable flag; cudaMallocHost /
  // cudaHostAllocDefault produce memory that is only DMA-accessible from the allocating
  // device's context, which under CUDA 13+ makes cudaMemcpyBatchAsync reject cross-device
  // sources with cudaErrorInvalidValue). cuDF 26.04+ may access hostdevice_vector memory
  // directly from GPU kernels (e.g. detect_malformed_pages), so returning pageable memory
  // here would cause cudaErrorIllegalAddress.
  if (bytes > MAX_SLAB_SIZE) {
    void* ptr = nullptr;
    // Portable + Mapped — see numa_region_pinned_host_allocator.cpp comment.
    auto err = ::cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable | cudaHostAllocMapped);
    if (err != cudaSuccess) { throw std::bad_alloc{}; }
    return ptr;
  }

  std::size_t idx = slab_index_for(bytes);
  std::lock_guard<std::mutex> lock(mutex_);
  if (free_lists_[idx].empty()) { expand_pool_locked(idx); }
  free_slab slab = free_lists_[idx].back();
  free_lists_[idx].pop_back();
  // If this slab was recently deallocated, its ready_event captures the freeing
  // stream's last use (e.g. an in-flight async H2D copy still reading the slab).
  // Make the reusing stream wait for it so we cannot overwrite the slab before
  // that copy completes. Recording the event again (on a later deallocate) does
  // not disturb this already-enqueued wait, so the event is safe to recycle.
  if (slab.ready_event != nullptr) {
    CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaStreamWaitEvent(stream.get(), slab.ready_event, 0));
    release_event_locked(slab.ready_event);
  }
  return slab.ptr;
}

void small_pinned_host_memory_resource::deallocate([[maybe_unused]] cuda::stream_ref stream,
                                                   void* ptr,
                                                   std::size_t bytes,
                                                   [[maybe_unused]] std::size_t alignment) noexcept
{
  if (ptr == nullptr || bytes == 0) { return; }
  if (bytes > MAX_SLAB_SIZE) {
    ::cudaFreeHost(ptr);
    return;
  }

  std::size_t idx = slab_index_for(bytes);
  std::lock_guard<std::mutex> lock(mutex_);
  // Record an event on the freeing stream so a future reuse of this slab can wait
  // for any still-pending work on it (the async H2D copy in cuDF's stats filter).
  // Best effort: a null event just means this slab is recycled without ordering.
  cudaEvent_t event = acquire_event_locked();
  if (event != nullptr && ::cudaEventRecord(event, stream.get()) != cudaSuccess) {
    (void)::cudaGetLastError();
    release_event_locked(event);
    event = nullptr;
  }
  free_lists_[idx].push_back(free_slab{ptr, event});
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

void small_pinned_host_memory_resource::expand_pool_locked(std::size_t slab_idx)
{
  // Acquire one upstream block and carve it into slabs.
  std::size_t upstream_block_size = upstream_.get_block_size();
  auto allocation                 = upstream_.allocate_multiple_blocks(upstream_block_size);

  std::size_t slab_size = SLAB_SIZES[slab_idx];
  std::size_t num_slabs = upstream_block_size / slab_size;
  for (std::byte* block : allocation->get_blocks()) {
    for (std::size_t i = 0; i < num_slabs; ++i) {
      // Freshly-carved slabs were never used, so they carry no pending-work event.
      free_lists_[slab_idx].push_back(free_slab{block + i * slab_size, nullptr});
    }
  }
  owned_allocations_.push_back(std::move(allocation));
}

}  // namespace memory
}  // namespace cucascade
