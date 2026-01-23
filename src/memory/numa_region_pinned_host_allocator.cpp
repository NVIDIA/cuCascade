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

#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <rmm/detail/nvtx/ranges.hpp>

#include <numa.h>

#include <cstring>

namespace cucascade {
namespace memory {

void* numa_region_pinned_host_memory_resource::do_allocate(
  std::size_t bytes, [[maybe_unused]] rmm::cuda_stream_view stream)
{
  RMM_FUNC_RANGE();
  // don't allocate anything if the user requested zero bytes
  if (0 == bytes) { return nullptr; }

  if (_numa_node == -1) {
    void* ptr{nullptr};
    RMM_CUDA_TRY_ALLOC(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault), bytes);
    return ptr;
  } else {
    void* ptr = numa_alloc_onnode(bytes, _numa_node);
    if (ptr == nullptr) { throw rmm::bad_alloc(std::strerror(errno)); }
    RMM_CUDA_TRY_ALLOC(cudaHostRegister(ptr, bytes, cudaHostRegisterMapped), bytes);
    return ptr;
  }
}

void numa_region_pinned_host_memory_resource::do_deallocate(
  void* ptr, std::size_t bytes, [[maybe_unused]] rmm::cuda_stream_view stream) noexcept
{
  RMM_FUNC_RANGE();
  if (_numa_node == -1) {
    RMM_ASSERT_CUDA_SUCCESS(cudaFreeHost(ptr));
  } else {
    RMM_ASSERT_CUDA_SUCCESS(cudaHostUnregister(ptr));
    numa_free(ptr, bytes);
  }
}

bool numa_region_pinned_host_memory_resource::do_is_equal(
  const rmm::mr::device_memory_resource& other) const noexcept
{
  auto* mr_ptr = dynamic_cast<numa_region_pinned_host_memory_resource const*>(&other);
  return mr_ptr == this && mr_ptr->_numa_node == this->_numa_node;
}

}  // namespace memory
}  // namespace cucascade
