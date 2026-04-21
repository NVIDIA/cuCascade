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

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <array>
#include <cstddef>
#include <mutex>
#include <vector>

namespace cucascade {
namespace memory {

/**
 * @brief A small slab allocator backed by fixed_size_host_memory_resource.
 *
 * Manages three pools of pinned host memory: 512 B, 1 KB, and 2 KB.
 * Each pool is populated on demand by acquiring one upstream block from the
 * provided fixed_size_host_memory_resource and carving it into slabs of the
 * appropriate size.
 *
 * Satisfies the cuda::mr::device_accessible and cuda::mr::host_accessible
 * properties, making it compatible with rmm::host_device_async_resource_ref
 * and suitable for use as cuDF's default pinned memory resource.
 *
 * Typical use:
 * @code
 *   small_pinned_host_memory_resource slab_mr(host_fixed_mr);
 *   cudf::set_pinned_memory_resource(slab_mr);
 *   cudf::set_allocate_host_as_pinned_threshold(
 *       small_pinned_host_memory_resource::MAX_SLAB_SIZE);
 * @endcode
 *
 * This eliminates the pageable H2D transfers that cuDF would otherwise issue
 * when building column_device_view metadata arrays for cudf::concatenate.
 */
class small_pinned_host_memory_resource {
 public:
  /// Maximum allocation size handled by the slab pools.
  /// Requests larger than this use pageable memory.
  static constexpr std::size_t MAX_SLAB_SIZE = 8192;

  /**
   * @brief Construct with the upstream fixed-size host memory resource.
   *
   * @param upstream Block allocator backed by pinned host memory. Must outlive
   *                 this object.
   */
  explicit small_pinned_host_memory_resource(fixed_size_host_memory_resource& upstream);

  small_pinned_host_memory_resource(const small_pinned_host_memory_resource&)            = delete;
  small_pinned_host_memory_resource& operator=(const small_pinned_host_memory_resource&) = delete;
  small_pinned_host_memory_resource(small_pinned_host_memory_resource&&)                 = delete;
  small_pinned_host_memory_resource& operator=(small_pinned_host_memory_resource&&)      = delete;

  ~small_pinned_host_memory_resource();

  /**
   * @brief Allocate pinned memory.
   *
   * For @p bytes <= MAX_SLAB_SIZE: rounds up to the next slab boundary
   * (512 / 1 KB / 2 KB / 4 KB / 8 KB) and returns a pointer from the matching
   * free list, expanding the pool from upstream if the list is empty.
   *
   * For @p bytes > MAX_SLAB_SIZE: falls back to cudaMallocHost (pinned).
   */
  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t));

  /**
   * @brief Return memory to the appropriate pool.
   *
   * Slabs (@p bytes <= MAX_SLAB_SIZE) are returned to the free list.
   * Pinned allocations (@p bytes > MAX_SLAB_SIZE) are freed via cudaFreeHost.
   * @p bytes must equal the value passed to the corresponding allocate.
   */
  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept;

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    auto* ptr = allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
    CUCASCADE_CUDA_TRY(cudaStreamSynchronize(cudaStream_t{nullptr}));
    return ptr;
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
    CUCASCADE_ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(cudaStream_t{nullptr}));
  }

  bool operator==(small_pinned_host_memory_resource const& other) const noexcept;

  /**
   * @brief Declares that memory allocated here is accessible from GPU devices.
   * Required to satisfy rmm::host_device_async_resource_ref.
   */
  friend void get_property(small_pinned_host_memory_resource const&,
                           cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Declares that memory allocated here is accessible from the host.
   * Required to satisfy rmm::host_device_async_resource_ref.
   */
  friend void get_property(small_pinned_host_memory_resource const&,
                           cuda::mr::host_accessible) noexcept
  {
  }

  /// Slab sizes in ascending order.
  static constexpr std::array<std::size_t, 5> SLAB_SIZES{512, 1024, 2048, 4096, 8192};

  /// Returns the index into SLAB_SIZES of the smallest slab >= bytes.
  static std::size_t slab_index_for(std::size_t bytes) noexcept;

  /// Populate the free list for slab @p idx by acquiring one upstream block.
  /// Must be called with mutex_ held.
  void expand_pool_locked(std::size_t slab_idx);

  fixed_size_host_memory_resource& upstream_;
  mutable std::mutex mutex_;
  std::array<std::vector<void*>, 5> free_lists_{};
  std::vector<fixed_multiple_blocks_allocation> owned_allocations_;
};

static_assert(cuda::mr::resource_with<small_pinned_host_memory_resource,
                                      cuda::mr::device_accessible,
                                      cuda::mr::host_accessible>);

}  // namespace memory
}  // namespace cucascade
