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

#include <cucascade/cuda/stream.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <vector>

namespace cucascade {
namespace memory {

/**
 * @brief Provides pooled pinned host memory for small and large allocations
 *
 * Requests up to `MAX_SLAB_SIZE` use slabs carved from `fixed_size_host_memory_resource`. Larger
 * requests whose power-of-two bucket fits the configured cache limit reuse released portable,
 * mapped pinned buffers; other requests allocate exactly the requested size. The cache limit bounds
 * retained large-buffer capacity, excluding live allocations and slab storage.
 *
 * Allocation and deallocation are thread-safe and stream-ordered. The caller must order every prior
 * GPU access before the deallocation stream. Before accessing an asynchronous allocation on the
 * CPU, synchronize its allocation stream; `allocate_sync` supplies this synchronization. All
 * allocations must be deallocated and no calls may be in flight when this resource is destroyed.
 * Destruction waits for recorded work before returning storage.
 *
 * Provides the `::cuda::mr::device_accessible` and `::cuda::mr::host_accessible` properties
 * required by `rmm::host_device_async_resource_ref`.
 */
class small_pinned_host_memory_resource {
 public:
  static constexpr std::size_t MAX_SLAB_SIZE    = 8192;  ///< Largest request served by a slab pool.
  static constexpr std::size_t MIN_LARGE_BUCKET = 16384;  ///< Smallest large-allocation bucket.
  static constexpr std::size_t DEFAULT_LARGE_CACHE_LIMIT = 256ull << 20;  ///< Default cache cap.
  static constexpr std::array<std::size_t, 5> SLAB_SIZES{
    512, 1024, 2048, 4096, 8192};  ///< Slab sizes.

  /**
   * @brief Constructs a pinned host memory resource
   *
   * @param upstream Block allocator backed by pinned host memory; must outlive this resource
   * @param large_cache_limit_bytes Maximum retained large-buffer capacity; values below
   * `MIN_LARGE_BUCKET` disable caching
   */
  explicit small_pinned_host_memory_resource(
    fixed_size_host_memory_resource& upstream,
    std::size_t large_cache_limit_bytes = DEFAULT_LARGE_CACHE_LIMIT);

  small_pinned_host_memory_resource(const small_pinned_host_memory_resource&)            = delete;
  small_pinned_host_memory_resource& operator=(const small_pinned_host_memory_resource&) = delete;
  small_pinned_host_memory_resource(small_pinned_host_memory_resource&&)                 = delete;
  small_pinned_host_memory_resource& operator=(small_pinned_host_memory_resource&&)      = delete;
  ~small_pinned_host_memory_resource() noexcept;

  /**
   * @brief Allocates pinned host memory
   *
   * On a direct pinned-allocation failure, releases the retained large buffers and retries once. A
   * CUDA synchronization failure quarantines affected storage instead of freeing or reusing it.
   *
   * @throw std::bad_alloc If a direct pinned allocation still fails after releasing retained
   * buffers
   * @param stream Stream on which reuse dependencies are inserted
   * @param bytes Requested size
   * @param alignment Requested alignment; currently not used to select storage
   * @return Pointer to at least `bytes` bytes, or `nullptr` for a zero-byte request
   */
  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t));

  /**
   * @brief Returns memory to its pool or releases a noncacheable large buffer
   *
   * @param stream Stream ordered after the final use of `ptr`
   * @param ptr Pointer returned by this resource, or `nullptr` for a no-op
   * @param bytes Original requested allocation size
   * @param alignment Original requested alignment
   */
  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept;

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    ::cuda::stream_ref stream{cudaStream_t{cudaStreamDefault}};
    auto* ptr = allocate(stream, bytes, alignment);
    try {
      stream.sync();
    } catch (...) {
      deallocate(stream, ptr, bytes, alignment);
      throw;
    }
    return ptr;
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, ptr, bytes, alignment);
    CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaStreamSynchronize(cudaStreamDefault));
  }

  /**
   * @brief Returns retained large-buffer capacity available for reuse
   *
   * Entries removed for eviction or purge are excluded while their recorded work completes.
   *
   * @return Sum of cached bucket capacities
   */
  [[nodiscard]] std::size_t large_cache_bytes() const;

  bool operator==(small_pinned_host_memory_resource const& other) const noexcept;

  friend void get_property(small_pinned_host_memory_resource const&,
                           ::cuda::mr::device_accessible) noexcept
  {
  }

  friend void get_property(small_pinned_host_memory_resource const&,
                           ::cuda::mr::host_accessible) noexcept
  {
  }

  /**
   * @brief Returns the index of the smallest slab large enough for `bytes`
   *
   * @param bytes Request size no greater than `MAX_SLAB_SIZE`
   * @return Index into `SLAB_SIZES`
   */
  static std::size_t slab_index_for(std::size_t bytes) noexcept;

  /**
   * @brief Computes the bucket for a large request
   *
   * @param bytes Request size greater than `MAX_SLAB_SIZE`
   * @return Smallest representable power of two no less than `bytes` and `MIN_LARGE_BUCKET`, or
   * `bytes` if the next power of two is not representable
   */
  static std::size_t large_bucket_size_for(std::size_t bytes) noexcept;

  /**
   * @brief Computes the storage size for a direct large allocation
   *
   * @param bytes Requested allocation size
   * @return Bucket capacity when it fits the limit, or `bytes` otherwise
   */
  [[nodiscard]] std::size_t large_allocation_size(std::size_t bytes) const noexcept;

 private:
  struct device_event {
    cudaEvent_t handle{nullptr};
    int device{-1};
  };

  struct free_slab {
    void* ptr;
    device_event ready;
  };

  struct large_cache_entry {
    void* ptr;
    device_event ready;
    std::uint64_t sequence;
  };

  using large_cache_type = std::map<std::size_t, std::deque<large_cache_entry>>;

  // Helpers ending in _locked require mutex_. Detached entries are synchronized outside it.
  void expand_pool_locked(std::size_t slab_idx);
  static int device_of_stream(::cuda::stream_ref stream) noexcept;
  device_event acquire_event_locked(int device) noexcept;
  void release_event_locked(device_event event) noexcept;
  void* try_take_cached_large_locked(std::size_t bucket, ::cuda::stream_ref stream);
  large_cache_entry evict_oldest_large_locked() noexcept;
  large_cache_type purge_large_cache_locked() noexcept;
  void sync_and_free_large_victim(large_cache_entry const& victim) noexcept;
  static bool synchronize_stream_no_throw(::cuda::stream_ref stream) noexcept;

  fixed_size_host_memory_resource& upstream_;
  mutable std::mutex mutex_;
  std::array<std::vector<free_slab>, SLAB_SIZES.size()> free_lists_{};
  std::map<int, std::vector<cudaEvent_t>> event_pools_;
  std::vector<fixed_multiple_blocks_allocation> owned_allocations_;
  bool has_quarantined_slab_{false};
  large_cache_type large_cache_;
  std::size_t large_cache_bytes_{0};
  std::size_t const large_cache_limit_bytes_;
  std::uint64_t large_cache_sequence_{0};
};

static_assert(::cuda::mr::resource_with<small_pinned_host_memory_resource,
                                        ::cuda::mr::device_accessible,
                                        ::cuda::mr::host_accessible>);

}  // namespace memory
}  // namespace cucascade
