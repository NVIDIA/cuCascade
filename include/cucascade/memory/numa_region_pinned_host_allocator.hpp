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

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <cstddef>

namespace cucascade {
namespace memory {

class numa_region_pinned_host_memory_resource final {
 public:
  explicit numa_region_pinned_host_memory_resource(int numa_node) : _numa_node(numa_node) {}
  ~numa_region_pinned_host_memory_resource()                                              = default;
  numa_region_pinned_host_memory_resource(numa_region_pinned_host_memory_resource const&) = default;
  numa_region_pinned_host_memory_resource(numa_region_pinned_host_memory_resource&&)      = default;
  numa_region_pinned_host_memory_resource& operator=(
    numa_region_pinned_host_memory_resource const&) = default;
  numa_region_pinned_host_memory_resource& operator=(numa_region_pinned_host_memory_resource&&) =
    default;

  /**
   * @brief Allocates pinned host memory of size at least \p bytes bytes.
   */
  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t));

  /**
   * @brief Deallocate memory pointed to by \p ptr.
   */
  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept;

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t));

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept;

  [[nodiscard]] bool operator==(
    numa_region_pinned_host_memory_resource const& other) const noexcept;

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   */
  friend void get_property(numa_region_pinned_host_memory_resource const&,
                           cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Enables the `cuda::mr::host_accessible` property
   */
  friend void get_property(numa_region_pinned_host_memory_resource const&,
                           cuda::mr::host_accessible) noexcept
  {
  }

 private:
  int _numa_node{-1};
};

}  // namespace memory
}  // namespace cucascade
