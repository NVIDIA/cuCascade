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
#include <atomic>
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
 * Requests up to @ref MAX_SLAB_SIZE use five slab pools: 512 B, 1 KiB, 2 KiB, 4 KiB, and 8 KiB.
 * Each pool grows on demand by carving a block from @ref fixed_size_host_memory_resource into slabs
 * of the corresponding size.
 *
 * Larger requests are rounded to a power-of-two bucket only when that bucket is representable and
 * fits the configured cache limit; other requests allocate exactly the requested size. Buffers
 * larger than @ref MAX_SLAB_SIZE are portable and mapped. @p large_cache_limit_bytes bounds entries
 * retained for stream-ordered reuse, which may still have GPU work pending.
 *
 * The cache limit is not a cap on total pinned memory. It excludes live allocations, slab storage,
 * evicted buffers whose GPU work has not finished, and storage withheld after a CUDA error. To
 * retain every buffer of a steady workload, size the limit as the sum over buckets of
 * (bucket size x peak number of buffers of that bucket outstanding at once).
 *
 * Allocation and deallocation are thread-safe and stream-ordered. The caller must order every prior
 * GPU access before the deallocation stream (the stream passed to the deallocation call). Before
 * accessing an asynchronous allocation on the CPU, synchronize its allocation stream (the stream
 * passed to the allocation call); `allocate_sync` supplies this synchronization. All allocations
 * must be deallocated and no calls may be in flight when this resource is destroyed. Destruction
 * waits for recorded work before returning storage.
 *
 * Provides the `::cuda::mr::device_accessible` and `::cuda::mr::host_accessible` properties
 * required by `rmm::host_device_async_resource_ref`.
 */
class small_pinned_host_memory_resource {
 public:
  static constexpr std::size_t MAX_SLAB_SIZE    = 8192;  ///< Largest request served by a slab pool.
  static constexpr std::size_t MIN_LARGE_BUCKET = 16384;  ///< Smallest large-allocation bucket.
  static constexpr std::size_t DEFAULT_LARGE_CACHE_LIMIT = 256ULL << 20;  ///< Default cache cap.
  static constexpr std::array<std::size_t, 5> SLAB_SIZES{
    512, 1024, 2048, 4096, 8192};  ///< Slab sizes.

  /**
   * @brief Recorded activity for requests larger than `MAX_SLAB_SIZE`
   *
   * Counters include requests that later fail, and each request is counted once regardless of
   * retries. Because of these edge cases, `large_requests` need not equal
   * `large_hits + large_misses + large_uncacheable`, and one request may count as both:
   * - Miss, then hit: the first lookup misses, but a buffer appears in the cache before the retry.
   * - Neither: waiting on the cached buffer's pending GPU work fails.
   *
   * `large_evictions` still counts a buffer that cannot be freed after a CUDA error, and does not
   * count buffers freed by the destructor.
   */
  struct statistics {
    std::size_t cached_large_bytes;       ///< Retained cache entry capacity.
    std::size_t peak_cached_large_bytes;  ///< Highest retained capacity; purges do not reset it.
    std::uint64_t large_requests;         ///< Nonzero requests larger than `MAX_SLAB_SIZE`.
    std::uint64_t large_hits;         ///< Successful cache takes after establishing reuse order.
    std::uint64_t large_misses;       ///< Initial cache lookups that found no entry.
    std::uint64_t large_uncacheable;  ///< Requests whose computed bucket exceeds the limit.
    std::uint64_t large_evictions;    ///< Cached entries detached for cap eviction or purge.
  };

  /**
   * @brief Constructs a pinned host memory resource
   *
   * @param upstream Block allocator backed by pinned host memory; must outlive this resource
   * @param large_cache_limit_bytes Maximum retained cache entry capacity shared by all devices and
   *                                streams using this resource; values below `MIN_LARGE_BUCKET`
   *                                disable large-buffer caching.
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
   * @brief Two resources are equal only if they are the same object
   */
  bool operator==(small_pinned_host_memory_resource const& other) const noexcept;

  /**
   * @brief Allocates pinned host memory
   *
   * On a direct pinned-allocation failure, releases the retained large buffers and retries once. A
   * CUDA synchronization failure quarantines affected storage instead of freeing or reusing it.
   *
   * @throw rmm::out_of_memory If a direct pinned allocation still fails after releasing retained
   *        buffers, or the upstream resource cannot supply a block for a slab pool
   * @throw rmm::bad_alloc If a direct pinned allocation fails with a non-OOM CUDA error
   * @throw cucascade::cuda_error If waiting on a reused buffer's pending GPU work fails
   * @throw std::invalid_argument If the upstream block size is smaller than the needed slab size
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

  /**
   * @brief Allocates on the default stream and synchronizes it before returning
   *
   * Throws the same exceptions as `allocate`, plus any error from synchronizing the stream.
   */
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

  /**
   * @brief Deallocates on the default stream, then synchronizes it
   */
  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    deallocate(::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, ptr, bytes, alignment);
    CUCASCADE_ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(cudaStreamDefault));
  }

  /**
   * @brief Returns retained large-buffer capacity available for reuse
   *
   * Entries removed for eviction or purge are excluded while their recorded work completes.
   *
   * @return Sum of cached bucket capacities
   */
  [[nodiscard]] std::size_t large_cache_bytes() const;

  /**
   * @brief Returns a consistent snapshot of recorded large-allocation activity
   */
  [[nodiscard]] statistics get_statistics() const;

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
   *         `bytes` if the next power of two is not representable
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

  // Helpers ending in _locked require _mutex. Detached entries are synchronized outside it.
  void expand_pool_locked(std::size_t slab_idx);
  static int device_of_stream(::cuda::stream_ref stream) noexcept;
  device_event acquire_event_locked(int device) noexcept;
  void release_event_locked(device_event event) noexcept;
  void* try_take_cached_large_locked(std::size_t bucket, ::cuda::stream_ref stream);
  large_cache_entry evict_oldest_large_locked() noexcept;
  large_cache_type purge_large_cache_locked() noexcept;
  void sync_and_free_large_victim(large_cache_entry const& victim) noexcept;
  static bool synchronize_stream_no_throw(::cuda::stream_ref stream) noexcept;

  fixed_size_host_memory_resource& _upstream;
  mutable std::mutex _mutex;
  std::array<std::vector<free_slab>, SLAB_SIZES.size()> _free_lists{};
  std::map<int, std::vector<cudaEvent_t>> _event_pools;
  std::vector<fixed_multiple_blocks_allocation> _owned_allocations;
  bool _has_quarantined_slab{false};  ///< A slab was withheld; destructor keeps backing blocks.
  large_cache_type _large_cache;      ///< Reusable large buffers keyed by bucket size.
  std::size_t _large_cache_bytes{0};  ///< Sum of bucket sizes in `_large_cache`.
  std::size_t const _large_cache_limit_bytes;
  std::uint64_t _large_cache_sequence{0};  ///< Insertion counter; lowest sequence is evicted first.
  std::size_t _peak_cached_large_bytes{0};
  std::uint64_t _large_cacheable_requests{0};
  std::uint64_t _large_hits{0};
  std::uint64_t _large_misses{0};
  std::atomic<std::uint64_t> _large_uncacheable{0};
  std::uint64_t _large_evictions{0};
};

static_assert(::cuda::mr::resource_with<small_pinned_host_memory_resource,
                                        ::cuda::mr::device_accessible,
                                        ::cuda::mr::host_accessible>);

}  // namespace memory
}  // namespace cucascade
