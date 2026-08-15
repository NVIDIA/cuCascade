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
#if __has_include(<cuda/stream>)
#include <cuda/stream>
#else
#include <cuda/stream_ref>
#endif
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
 * Requests up to `MAX_SLAB_SIZE` use slabs carved from the upstream resource. Larger requests whose
 * power-of-two bucket fits the configured cache limit can reuse released `cudaHostAlloc` buffers.
 * Requests whose bucket exceeds the limit allocate exactly the requested size and bypass the cache.
 *
 * Reuse is ordered after work submitted to the stream passed to `deallocate`. The resource
 * satisfies the ::cuda::mr::device_accessible and ::cuda::mr::host_accessible properties required
 * by rmm::host_device_async_resource_ref.
 */
class small_pinned_host_memory_resource {
 public:
  /// Largest request served by a slab pool.
  static constexpr std::size_t MAX_SLAB_SIZE = 8192;

  /// Smallest power-of-two bucket used for allocations above MAX_SLAB_SIZE.
  static constexpr std::size_t MIN_LARGE_BUCKET = 16384;

  /// Default cap on the total bytes retained by the large-allocation cache.
  static constexpr std::size_t DEFAULT_LARGE_CACHE_LIMIT = 256ull << 20;

  /**
   * @brief Constructs a pinned host memory resource
   *
   * @param upstream Block allocator backed by pinned host memory. Must outlive this object.
   * @param large_cache_limit_bytes Maximum total bucket capacity retained for large allocations. A
   * value below `MIN_LARGE_BUCKET` disables large-allocation caching.
   */
  explicit small_pinned_host_memory_resource(
    fixed_size_host_memory_resource& upstream,
    std::size_t large_cache_limit_bytes = DEFAULT_LARGE_CACHE_LIMIT);

  small_pinned_host_memory_resource(const small_pinned_host_memory_resource&)            = delete;
  small_pinned_host_memory_resource& operator=(const small_pinned_host_memory_resource&) = delete;
  small_pinned_host_memory_resource(small_pinned_host_memory_resource&&)                 = delete;
  small_pinned_host_memory_resource& operator=(small_pinned_host_memory_resource&&)      = delete;

  /**
   * @brief Releases retained buffers and CUDA events
   *
   * All work using allocations returned by this resource must be complete, and no calls may be in
   * flight when it is destroyed.
   */
  ~small_pinned_host_memory_resource();

  /**
   * @brief Allocates pinned host memory
   *
   * Requests up to `MAX_SLAB_SIZE` are rounded to the next slab size. A larger request is rounded
   * to its power-of-two bucket only when that bucket fits the cache limit; otherwise the resource
   * allocates exactly @p bytes. If a direct pinned allocation fails, retained large buffers are
   * released and the allocation is retried once.
   *
   * @throw std::bad_alloc If a direct pinned allocation still fails after retained buffers are
   * released
   *
   * @param stream CUDA stream on which reuse dependencies are inserted
   * @param bytes Number of bytes requested
   * @param alignment Requested alignment; currently not used to select storage
   * @return Pointer to at least @p bytes bytes of pinned memory, or `nullptr` when @p bytes is zero
   */
  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t));

  /**
   * @brief Deallocates pinned host memory
   *
   * Cacheable buffers are retained for reuse, while large buffers whose bucket exceeds the cache
   * limit are released. The caller must order every prior access to @p ptr before or on @p stream.
   *
   * @param stream CUDA stream ordered after the final use of @p ptr
   * @param ptr Pointer returned by this resource, or `nullptr` for a no-op
   * @param bytes Original requested allocation size; must match the corresponding call to
   * `allocate`
   * @param alignment Original requested alignment
   */
  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept;

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    auto* ptr = allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
    rmm::cuda_stream_default.synchronize();
    return ptr;
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
    rmm::cuda_stream_default.synchronize_no_throw();
  }

  /**
   * @brief Returns the large-buffer capacity currently accounted to the cache
   *
   * This excludes slab storage, live allocations, and entries already removed for eviction or
   * purge.
   *
   * @return Sum of bucket sizes for cached entries available for reuse
   */
  [[nodiscard]] std::size_t large_cache_bytes() const;

  bool operator==(small_pinned_host_memory_resource const& other) const noexcept;

  /**
   * @brief Declares that memory allocated here is accessible from GPU devices.
   * Required to satisfy rmm::host_device_async_resource_ref.
   */
  friend void get_property(small_pinned_host_memory_resource const&,
                           ::cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Declares that memory allocated here is accessible from the host.
   * Required to satisfy rmm::host_device_async_resource_ref.
   */
  friend void get_property(small_pinned_host_memory_resource const&,
                           ::cuda::mr::host_accessible) noexcept
  {
  }

  /// Slab sizes in ascending order.
  static constexpr std::array<std::size_t, 5> SLAB_SIZES{512, 1024, 2048, 4096, 8192};

  /// Returns the index into SLAB_SIZES of the smallest slab >= bytes.
  static std::size_t slab_index_for(std::size_t bytes) noexcept;

  /**
   * @brief Computes the cache bucket for a large request
   *
   * @param bytes Request size greater than `MAX_SLAB_SIZE`
   * @return Smallest representable power of two no less than @p bytes and `MIN_LARGE_BUCKET`, or @p
   * bytes when the next power of two is not representable
   */
  static std::size_t large_bucket_size_for(std::size_t bytes) noexcept;

  /**
   * @brief Computes the storage size for a direct large allocation
   *
   * @param bytes Requested allocation size
   * @return Cache bucket size when it fits the limit, or @p bytes otherwise
   */
  [[nodiscard]] std::size_t large_allocation_size(std::size_t bytes) const noexcept;

  /// Populates slab pool @p slab_idx from an upstream block. Must hold @c mutex_.
  void expand_pool_locked(std::size_t slab_idx);

  /// CUDA event and the device on which it was created. Recording succeeds only on a stream
  /// associated with the same device.
  struct device_event {
    cudaEvent_t handle{nullptr};
    int device{-1};
  };

  /// Slab available for reuse and an optional event recording its previous use.
  struct free_slab {
    void* ptr;
    device_event ready;
  };

  /// Large buffer available for reuse, its required dependency, and its insertion order.
  struct large_cache_entry {
    void* ptr;
    device_event ready;
    std::uint64_t sequence;
  };

  /// Borrows a timing-disabled event created on the current device. Returns an empty event if the
  /// device query or event creation fails. Must hold @c mutex_.
  device_event acquire_event_locked();

  /// Returns an event to its device's pool in @c event_pools_ for reuse. Must hold @c mutex_.
  void release_event_locked(device_event event) noexcept;

  /// Removes the oldest entry in @p bucket and orders @p stream after its ready event. Returns
  /// `nullptr` when the bucket is empty. Must hold @c mutex_.
  void* try_take_cached_large_locked(std::size_t bucket, ::cuda::stream_ref stream);

  /// Removes and returns the oldest entry across all buckets. Returns an entry with a null pointer
  /// when the cache is empty. Must hold @c mutex_.
  large_cache_entry evict_oldest_large_locked() noexcept;

  /// Removes and returns every cached large buffer. Must hold @c mutex_.
  std::vector<large_cache_entry> purge_large_cache_locked();

  /// Waits for @p victim's ready event, when present, and frees its buffer. If the wait fails,
  /// freeing proceeds as a best-effort fallback. Must be called without @c mutex_ held.
  static void sync_and_free_large_victim(large_cache_entry const& victim) noexcept;

  fixed_size_host_memory_resource& upstream_;
  mutable std::mutex mutex_;
  std::array<std::vector<free_slab>, 5> free_lists_{};

  // Recycled events grouped by the device on which they were created.
  std::map<int, std::vector<cudaEvent_t>> event_pools_;

  std::vector<fixed_multiple_blocks_allocation> owned_allocations_;

  // Large-allocation cache and its bucket-capacity accounting.
  std::map<std::size_t, std::deque<large_cache_entry>> large_cache_;
  std::size_t large_cache_bytes_ = 0;
  std::size_t const large_cache_limit_bytes_;
  std::uint64_t large_cache_sequence_ = 0;
};

static_assert(::cuda::mr::resource_with<small_pinned_host_memory_resource,
                                        ::cuda::mr::device_accessible,
                                        ::cuda::mr::host_accessible>);

}  // namespace memory
}  // namespace cucascade
