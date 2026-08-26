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
  bool retain_backing_allocations = has_quarantined_slab_;
  int prev_device                 = 0;
  bool const have_prev            = (::cudaGetDevice(&prev_device) == cudaSuccess);
  auto destroy_on                 = [&retain_backing_allocations](cudaEvent_t event, int device) {
    if (event == nullptr) { return; }
    // cudaEventDestroy is device-agnostic, but making the owning device current
    // keeps the call on the context the event was created in.
    if (device >= 0) { (void)::cudaSetDevice(device); }
    // Destroying an incomplete event does not wait for its captured work. Wait
    // explicitly before owned_allocations_ returns the backing slabs upstream.
    if (::cudaEventSynchronize(event) != cudaSuccess) {
      // Fail closed: cudaEventDestroy is non-blocking for incomplete events.
      // Retain every backing allocation so upstream cannot reuse its slabs.
      (void)::cudaGetLastError();
      retain_backing_allocations = true;
    }
    CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaEventDestroy(event));
  };
  for (auto& [device, events] : event_pool_) {
    for (auto& event : events) {
      destroy_on(event, device);
    }
  }
  for (auto& list : free_lists_) {
    for (auto& slab : list) {
      destroy_on(slab.ready_event, slab.event_device);
    }
  }
  if (have_prev) { (void)::cudaSetDevice(prev_device); }
  if (retain_backing_allocations) {
    // Intentionally leak the allocation handles on an unrecoverable CUDA error.
    // Returning their blocks upstream would risk use-after-free or unordered reuse.
    for (auto& allocation : owned_allocations_) {
      (void)allocation.release();
    }
  }
}

int small_pinned_host_memory_resource::device_of_stream(::cuda::stream_ref stream) noexcept
{
  int device = -1;
  // A CUDA event is bound to the device current at creation time, and
  // cudaEventRecord requires the event and stream to be on the same device.
  // Ask the driver which device this stream belongs to rather than assuming it
  // matches the calling thread's current device.
  if (::cudaStreamGetDevice(stream.get(), &device) == cudaSuccess) { return device; }
  (void)::cudaGetLastError();
  if (::cudaGetDevice(&device) == cudaSuccess) { return device; }
  (void)::cudaGetLastError();
  return -1;
}

cudaEvent_t small_pinned_host_memory_resource::acquire_event_locked(int device)
{
  auto const pool_it = event_pool_.find(device);
  if (pool_it != event_pool_.end() && !pool_it->second.empty()) {
    auto& pool        = pool_it->second;
    cudaEvent_t event = pool.back();
    pool.pop_back();
    return event;
  }

  // Create the event on the device it will be recorded against. Without this,
  // an event created while another GPU was current fails cudaEventRecord with
  // cudaErrorInvalidResourceHandle, and the slab would be recycled unordered.
  int prev_device = -1;
  bool switched   = false;
  if (device >= 0 && ::cudaGetDevice(&prev_device) == cudaSuccess && prev_device != device) {
    if (::cudaSetDevice(device) == cudaSuccess) {
      switched = true;
    } else {
      (void)::cudaGetLastError();
      return nullptr;
    }
  }

  cudaEvent_t event = nullptr;
  // Timing is not needed; disabling it makes record/wait cheaper.
  bool const created = (::cudaEventCreateWithFlags(&event, cudaEventDisableTiming) == cudaSuccess);
  if (!created) {
    // The caller synchronizes the freeing stream when no event is available, so
    // losing an event costs performance but never ordering.
    (void)::cudaGetLastError();
    event = nullptr;
  }

  if (switched) { (void)::cudaSetDevice(prev_device); }
  return event;
}

void small_pinned_host_memory_resource::release_event_locked(cudaEvent_t event, int device) noexcept
{
  if (event == nullptr) { return; }
  try {
    event_pool_[device].push_back(event);
  } catch (...) {
    // Pool growth is only a cache optimization. Avoid terminating a noexcept
    // deallocation if host allocation fails; an already-enqueued wait is not
    // affected by destroying the event handle.
    CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaEventDestroy(event));
  }
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
    // cudaStreamWaitEvent is legal across devices, so the reusing stream may
    // belong to a different GPU than the one that recorded the event.
    cudaError_t const wait_status = ::cudaStreamWaitEvent(stream.get(), slab.ready_event, 0);
    if (wait_status != cudaSuccess) {
      // The wait was not established, so returning this pointer would recreate
      // the unordered-reuse race. Keep the slab tracked and propagate the error.
      free_lists_[idx].push_back(slab);
      CUCASCADE_CUDA_TRY(wait_status);
    }
    release_event_locked(slab.ready_event, slab.event_device);
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

  std::size_t idx         = slab_index_for(bytes);
  int const stream_device = device_of_stream(stream);
  std::unique_lock<std::mutex> lock(mutex_);
  // Record an event on the freeing stream so a future reuse of this slab can wait
  // for any still-pending work on it (the async H2D copy in cuDF's stats filter).
  cudaEvent_t event = acquire_event_locked(stream_device);
  if (event != nullptr && ::cudaEventRecord(event, stream.get()) != cudaSuccess) {
    (void)::cudaGetLastError();
    release_event_locked(event, stream_device);
    event = nullptr;
  }
  if (event == nullptr) {
    // No event means no ordering, and an unordered slab hands still-in-flight
    // source memory to the next writer. Pay for a synchronize instead: dropping
    // ordering here silently corrupts whatever the pending copy was reading.
    // The slab is not published yet, so do not hold the allocator-wide mutex
    // while waiting for unrelated work on this stream to complete.
    lock.unlock();
    if (::cudaStreamSynchronize(stream.get()) != cudaSuccess) {
      // Synchronization failure leaves ordering unknown. Quarantine this slab
      // instead of making it available for unsafe reuse.
      (void)::cudaGetLastError();
      lock.lock();
      has_quarantined_slab_ = true;
      return;
    }
    lock.lock();
  }
  free_lists_[idx].push_back(free_slab{ptr, event, stream_device});
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
      free_lists_[slab_idx].push_back(free_slab{block + i * slab_size, nullptr, -1});
    }
  }
  owned_allocations_.push_back(std::move(allocation));
}

}  // namespace memory
}  // namespace cucascade
