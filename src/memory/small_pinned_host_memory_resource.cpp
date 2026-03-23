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
}

void* small_pinned_host_memory_resource::do_allocate(std::size_t bytes,
                                                     [[maybe_unused]] rmm::cuda_stream_view stream)
{
  if (bytes == 0) { return nullptr; }
  // cuDF calls get_pinned_memory_resource() directly from some code paths (e.g. join/sort
  // staging buffers) that bypass the allocate_host_as_pinned threshold check.  Serve those
  // with cudaMallocHost so the
  // memory remains pinned and device-accessible.  cuDF 26.04+ may access
  // hostdevice_vector memory directly from GPU kernels (e.g. detect_malformed_pages),
  // so returning pageable memory here would cause cudaErrorIllegalAddress.
  if (bytes > MAX_SLAB_SIZE) {
    void* ptr = nullptr;
    auto err  = ::cudaMallocHost(&ptr, bytes);
    if (err != cudaSuccess) {
      throw std::bad_alloc{};
    }
    return ptr;
  }

  std::size_t idx = slab_index_for(bytes);
  std::lock_guard<std::mutex> lock(mutex_);
  if (free_lists_[idx].empty()) { expand_pool_locked(idx); }
  void* ptr = free_lists_[idx].back();
  free_lists_[idx].pop_back();
  return ptr;
}

void small_pinned_host_memory_resource::do_deallocate(
  void* ptr, std::size_t bytes, [[maybe_unused]] rmm::cuda_stream_view stream) noexcept
{
  if (ptr == nullptr || bytes == 0) { return; }
  if (bytes > MAX_SLAB_SIZE) {
    ::cudaFreeHost(ptr);
    return;
  }

  std::size_t idx = slab_index_for(bytes);
  std::lock_guard<std::mutex> lock(mutex_);
  free_lists_[idx].push_back(ptr);
}

bool small_pinned_host_memory_resource::do_is_equal(
  const rmm::mr::device_memory_resource& other) const noexcept
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
      free_lists_[slab_idx].push_back(block + i * slab_size);
    }
  }
  owned_allocations_.push_back(std::move(allocation));
}

}  // namespace memory
}  // namespace cucascade
