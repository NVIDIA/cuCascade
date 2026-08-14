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
 * @brief A pinned host memory allocator combining small slab pools with a bucketed reuse cache for
 * large allocations.
 *
 * Requests up to MAX_SLAB_SIZE are served from slab pools (SLAB_SIZES) of pinned host memory. Each
 * pool is populated on demand by acquiring one upstream block from the provided
 * fixed_size_host_memory_resource and carving it into slabs of the appropriate size.
 *
 * Requests above MAX_SLAB_SIZE are rounded up to a power-of-two bucket (MIN_LARGE_BUCKET at
 * minimum) and served from per-bucket free lists of previously released `cudaHostAlloc` buffers;
 * misses allocate a fresh buffer. Released buffers are cached rather than freed, up to a
 * configurable byte limit, sparing callers such as cuDF's parquet reader the synchronous
 * `cudaHostAlloc` / `cudaFreeHost` cost on every request. Reuse in both the slab and large paths is
 * ordered by CUDA events recorded on the freeing stream, so a buffer is never handed out while
 * another stream may still be using it.
 *
 * Satisfies the ::cuda::mr::device_accessible and ::cuda::mr::host_accessible
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
  /// Maximum allocation size handled by the slab pools. Larger requests are served from the
  /// large-allocation cache.
  static constexpr std::size_t MAX_SLAB_SIZE = 8192;

  /// Smallest power-of-two bucket used for allocations above MAX_SLAB_SIZE.
  static constexpr std::size_t MIN_LARGE_BUCKET = 16384;

  /// Default cap on the total bytes retained by the large-allocation cache.
  static constexpr std::size_t DEFAULT_LARGE_CACHE_LIMIT = 256ull << 20;

  /**
   * @brief Construct with the upstream fixed-size host memory resource.
   *
   * @param upstream Block allocator backed by pinned host memory. Must outlive this object.
   * @param large_cache_limit_bytes Maximum total bytes, measured in bucket sizes, retained by the
   * large-allocation cache.
   */
  explicit small_pinned_host_memory_resource(
    fixed_size_host_memory_resource& upstream,
    std::size_t large_cache_limit_bytes = DEFAULT_LARGE_CACHE_LIMIT);

  small_pinned_host_memory_resource(const small_pinned_host_memory_resource&)            = delete;
  small_pinned_host_memory_resource& operator=(const small_pinned_host_memory_resource&) = delete;
  small_pinned_host_memory_resource(small_pinned_host_memory_resource&&)                 = delete;
  small_pinned_host_memory_resource& operator=(small_pinned_host_memory_resource&&)      = delete;

  ~small_pinned_host_memory_resource();

  /**
   * @brief Allocate pinned memory.
   *
   * For @p bytes <= MAX_SLAB_SIZE: rounds up to the next slab boundary (512 / 1 KB / 2 KB / 4 KB /
   * 8 KB) and returns a pointer from the matching free list, expanding the pool from upstream if
   * the list is empty.
   *
   * For @p bytes > MAX_SLAB_SIZE: rounds up to the power-of-two bucket and, when the bucket fits
   * within the cache limit, returns a cached buffer when the bucket's free list has one, making @p
   * stream wait on the buffer's ready event first. On a miss (or a never-cacheable size, which
   * skips the cache lookup), allocates a fresh buffer of large_allocation_size(bytes) with
   * `cudaHostAlloc(Portable | Mapped)`; if that fails, purges the entire large cache and retries
   * once before throwing std::bad_alloc.
   */
  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t));

  /**
   * @brief Return memory to the appropriate free list.
   *
   * Slabs (@p bytes <= MAX_SLAB_SIZE) are returned to the slab free list. Larger buffers are cached
   * in their bucket's free list, evicting the oldest cached entries (in insertion order across
   * buckets) when the total would exceed the cache limit; a buffer whose bucket alone exceeds the
   * limit is freed with `cudaFreeHost` instead. Both paths record an event on @p stream so reuse
   * waits for pending work on the buffer. When no event can be recorded, @p stream is synchronized
   * instead (best effort); a large buffer is then freed rather than cached and a slab is cached
   * carrying no pending work, so a cached entry never carries pending work that reuse cannot order
   * against.
   *
   * @p bytes must equal the value passed to the corresponding allocate.
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

  /// Total bytes currently held in the large-allocation cache, measured in bucket sizes.
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

  /// Returns the bucket for a request above MAX_SLAB_SIZE: the smallest power of two >= bytes, no
  /// smaller than MIN_LARGE_BUCKET. The bucket is used for cache keying and eviction accounting;
  /// the physical size is large_allocation_size(bytes), equal to the bucket only for cacheable
  /// requests. allocate and deallocate both derive the bucket from the request size, so the pairing
  /// is deterministic.
  static std::size_t large_bucket_size_for(std::size_t bytes) noexcept;

  /// Returns the physical size cudaHostAlloc is asked for on a cache miss: the bucket when it fits
  /// within the cache limit, otherwise exactly @p bytes. A never-cacheable allocation gains nothing
  /// from bucket rounding and must not overshoot pinned memory (up to 2x for sizes just past a
  /// bucket boundary). Reads only the immutable cache limit, so no lock is needed.
  [[nodiscard]] std::size_t large_allocation_size(std::size_t bytes) const noexcept;

  /// Populate the free list for slab @p idx by acquiring one upstream block.
  /// Must be called with mutex_ held.
  void expand_pool_locked(std::size_t slab_idx);

  /// A CUDA event paired with the device whose context owns it. cudaEventRecord requires the event
  /// and the stream to share a CUDA context, so pooled events are segregated by device and an event
  /// is only ever recorded on a stream of its own device. @c device is meaningful only when @c
  /// handle is non-null.
  struct device_event {
    cudaEvent_t handle{nullptr};
    int device{-1};
  };

  /// A free slab plus, when it was just deallocated, a CUDA event recorded on
  /// the freeing stream. Reusing the slab must wait on this event so an
  /// in-flight async H2D copy that still reads the slab (e.g. cuDF's parquet
  /// stats min/max buffers) completes before another stream overwrites it.
  /// A null @c ready.handle means the slab carries no pending work: it was
  /// freshly carved and never used, or the freeing stream was synchronized
  /// before the slab was cached.
  struct free_slab {
    void* ptr;
    device_event ready;
  };

  /// A released large buffer held for reuse. Like @c free_slab, @c ready captures the freeing
  /// stream's last use of the buffer; unlike a slab, a large buffer is never cached without a
  /// recorded event (deallocate frees it instead), so @c ready.handle is non-null for every cached
  /// entry. @c sequence orders entries across buckets so eviction can drop the oldest first.
  struct large_cache_entry {
    void* ptr;
    device_event ready;
    std::uint64_t sequence;
  };

  /// Borrow a timing-disabled CUDA event owned by the calling thread's current device (recycled
  /// from @c event_pools_ or newly created). Returns a null handle when the device query or event
  /// creation fails; the caller then falls back to freeing (large path) or synchronizing (slab
  /// path) instead of caching with ordering. Must hold @c mutex_.
  device_event acquire_event_locked();

  /// Return an event to its device's pool in @c event_pools_ for reuse. Must hold @c mutex_.
  void release_event_locked(device_event event) noexcept;

  /// Pop the oldest cached buffer of @p bucket and make @p stream wait on its ready event,
  /// recycling the event. Returns null when the bucket has no cached entries. Must hold @c mutex_.
  void* try_take_cached_large_locked(std::size_t bucket, ::cuda::stream_ref stream);

  /// Remove the oldest cached large buffer across all buckets from the bookkeeping and return it
  /// with its ready event still attached, for the caller to pass to sync_and_free_large_victim
  /// outside the lock and then recycle the event. Returns a null-ptr entry when the cache is empty.
  /// Must hold @c mutex_.
  large_cache_entry evict_oldest_large_locked() noexcept;

  /// Remove every cached large buffer from the bookkeeping and return them with their ready events
  /// still attached, for the caller to pass to sync_and_free_large_victim outside the lock and then
  /// recycle the events. Must hold @c mutex_.
  std::vector<large_cache_entry> purge_large_cache_locked();

  /// Wait for a victim's recorded work and free its buffer. The ready event may have been recorded
  /// on any device's stream, so the buffer is not unpinned until the event has completed
  /// (best-effort: a failed or null event skips the wait). Must be called without @c mutex_ held;
  /// the caller recycles the event afterwards.
  static void sync_and_free_large_victim(large_cache_entry const& victim) noexcept;

  fixed_size_host_memory_resource& upstream_;
  mutable std::mutex mutex_;
  std::array<std::vector<free_slab>, 5> free_lists_{};

  /// Idle recycled events keyed by the device that owns them. Events are created in the calling
  /// thread's current context, so acquire_event_locked keys by cudaGetDevice() at acquire time;
  /// this avoids cudaSetDevice churn, and in deployment the calling thread's current device is its
  /// stream's device (the NUMA dispatcher routes by cudaGetDevice()). Should a caller ever pass a
  /// stream of another device, the record fails and the safe fallback engages, so correctness never
  /// depends on the key, only cache-hit rate.
  std::map<int, std::vector<cudaEvent_t>> event_pools_;

  std::vector<fixed_multiple_blocks_allocation> owned_allocations_;

  /// Large-allocation cache: per-bucket free lists keyed by bucket size, the current total in
  /// bucket bytes, the retention cap, and a monotonic counter stamping insertion order for
  /// eviction.
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
