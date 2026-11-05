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

#pragma once

#include <cucascade/memory/config.hpp>

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdlib>
#include <new>

namespace cucascade::test {

class shared_device_resource {
 public:
  explicit shared_device_resource(rmm::device_async_resource_ref upstream) : upstream_(upstream) {}

  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t))
  {
    return upstream_.allocate(stream, bytes, alignment);
  }

  void deallocate(::cuda::stream_ref stream,
                  void* p,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    upstream_.deallocate(stream, p, bytes, alignment);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    return allocate(::cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* p,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    deallocate(::cuda::stream_ref{cudaStream_t{nullptr}}, p, bytes, alignment);
  }

  bool operator==(shared_device_resource const&) const noexcept { return false; }

  friend void get_property(shared_device_resource const&, ::cuda::mr::device_accessible) noexcept {}

 private:
  rmm::device_async_resource_ref upstream_;
};

inline ::cuda::mr::any_resource<::cuda::mr::device_accessible> make_shared_current_device_resource(
  int, size_t)
{
  return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
    shared_device_resource{rmm::mr::get_current_device_resource_ref()}};
}

/// Plain malloc upstream for a HOST memory space: neither pinned nor bound to a NUMA node, so a
/// test can configure several logical HOST arenas on a single-node machine.
class malloc_host_resource {
 public:
  void* allocate(::cuda::stream_ref, std::size_t bytes, std::size_t = alignof(std::max_align_t))
  {
    void* p = std::malloc(bytes);
    if (p == nullptr) { throw std::bad_alloc{}; }
    return p;
  }

  void deallocate(::cuda::stream_ref,
                  void* p,
                  std::size_t,
                  std::size_t = alignof(std::max_align_t)) noexcept
  {
    std::free(p);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    return allocate(::cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* p,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    deallocate(::cuda::stream_ref{cudaStream_t{nullptr}}, p, bytes, alignment);
  }

  bool operator==(malloc_host_resource const&) const noexcept { return true; }

  friend void get_property(malloc_host_resource const&, ::cuda::mr::device_accessible) noexcept {}
  friend void get_property(malloc_host_resource const&, ::cuda::mr::host_accessible) noexcept {}
};

/// HOST memory space @p numa_id of @p capacity bytes over @ref malloc_host_resource: all of it
/// reservable, in @p block_size blocks, @p pool_size blocks per pool, one pool allocated up front.
inline memory::host_memory_space_config malloc_host_space_config(int numa_id,
                                                                 std::size_t capacity,
                                                                 std::size_t block_size,
                                                                 std::size_t pool_size)
{
  memory::host_memory_space_config config;
  config.numa_id                    = numa_id;
  config.memory_capacity            = capacity;
  config.reservation_limit_fraction = 1.0;
  config.block_size                 = block_size;
  config.pool_size                  = pool_size;
  config.initial_number_pools       = 1;
  config.mr_factory_fn              = [](int, std::size_t) {
    return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{malloc_host_resource{}};
  };
  return config;
}

}  // namespace cucascade::test
