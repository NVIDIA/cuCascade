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

#include <cucascade/cuda/event.hpp>
#include <cucascade/cuda/stream.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <vector>

namespace cucascade {
namespace memory {

/**
 * @brief Provides pooled pinned host memory for small and large allocations
 *
 * Requests with alignment no greater than `alignof(std::max_align_t)` and size up to @ref
 * MAX_SLAB_SIZE use five slab pools: 512 B, 1 KiB, 2 KiB, 4 KiB, and 8 KiB. Each pool grows on
 * demand by carving a block from @ref fixed_size_host_memory_resource into slabs of the
 * corresponding size.
 *
 * Larger requests with ordinary alignment initially use a power-of-two bucket only when that
 * bucket is representable and fits the configured cache limit; other requests allocate exactly the
 * requested size. Stricter alignment uses dedicated allocations with alignment padding, which are
 * never cached. Direct allocations are portable and mapped. @p large_cache_limit_bytes bounds
 * entries retained for stream-ordered reuse, which may still have GPU work pending.
 *
 * The cache limit is not a cap on total pinned memory. It excludes live allocations, slab storage,
 * evicted buffers whose GPU work has not finished, and storage withheld after a CUDA error. To
 * retain every buffer of a steady workload, size the limit as the sum over buckets of
 * (bucket size x peak number of buffers of that bucket outstanding at once).
 *
 * Allocation and deallocation are thread-safe. Memory is stream-ordered; see `allocate` and
 * `deallocate` for the caller's obligations. All allocations must be deallocated and no calls may
 * be in flight when this resource is destroyed.
 *
 * CUDA host callbacks and graph capture are unsupported. For a capturing supplied stream,
 * allocation throws and deallocation withholds storage without waiting, freeing, or publishing it.
 * Failure to determine capture status follows the same storage policy. Conflicting capture on
 * another stream remains unsupported. Zero-size allocation and null/zero-size deallocation perform
 * no CUDA calls.
 *
 * Provides the `::cuda::mr::device_accessible` and `::cuda::mr::host_accessible` properties
 * required by `rmm::host_device_async_resource_ref`, so it can serve as cuDF's pinned resource:
 *
 * @code
 *   small_pinned_host_memory_resource slab_mr(host_fixed_mr);
 *   cudf::set_pinned_memory_resource(slab_mr);
 *   cudf::set_allocate_host_as_pinned_threshold(
 *       small_pinned_host_memory_resource::MAX_SLAB_SIZE);
 * @endcode
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
   * Requests rejected for invalid alignment or unsupported/unconfirmed capture state are excluded.
   * Counters include requests that later fail, and each request is counted once regardless of
   * retries. A cached request is counted before submitting its reuse wait, but its hit is counted
   * only after submission succeeds, so concurrent snapshots can include requests whose hit
   * classification is still pending. Because of pending requests and these failure/retry cases,
   * `large_requests` need not equal `large_hits + large_misses + large_uncacheable`, and one
   * request may count as both:
   * - Miss, then hit: the first lookup misses, but a buffer appears in the cache before the retry.
   * - Neither: waiting on the cached buffer's pending GPU work fails.
   *
   * `large_evictions` still counts a buffer that cannot be freed after a CUDA error, and does not
   * count buffers freed by the destructor.
   */
  struct statistics {
    std::size_t cached_large_bytes;       ///< Retained cache entry capacity.
    std::size_t peak_cached_large_bytes;  ///< Highest retained capacity; purges do not reset it.
    std::uint64_t large_requests;         ///< Valid nonzero requests larger than `MAX_SLAB_SIZE`.
    std::uint64_t large_hits;         ///< Successful cache takes after establishing reuse order.
    std::uint64_t large_misses;       ///< Initial cache lookups that found no entry.
    std::uint64_t large_uncacheable;  ///< Requests whose size or alignment bypasses caching.
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

  /**
   * @brief Waits for recorded work before returning storage
   *
   * Wait failures are handled as in `deallocate`. An unconfirmed slab retains only its owning
   * backing allocation instead of returning it upstream. Pooled events use
   * `cuda::cuda_event::reset_no_throw()` for cleanup.
   */
  ~small_pinned_host_memory_resource() noexcept;

  /**
   * @brief Two resources are equal only if they are the same object
   */
  [[nodiscard]] bool operator==(small_pinned_host_memory_resource const& other) const noexcept;

  /**
   * @brief Allocates pinned host memory
   *
   * The returned memory is ready in stream order on `stream`: GPU work enqueued on `stream` after
   * this call may use it, but GPU work from an earlier deallocation may still be using it until
   * that point in `stream`. Before the host reads or writes it, synchronize `stream` or use
   * `allocate_sync`. This applies to every request, including reused slabs and cached large
   * buffers. A freshly allocated large buffer is host-ready on return, but callers must not rely on
   * which case occurred.
   *
   * A rounded large allocation that runs out of memory first tries exact-size dedicated storage. If
   * an exact allocation fails, recovery rechecks a matching reusable bucket and reclaims oldest
   * entries only when retained capacity covers the exact backing size, including alignment padding.
   * Completion waits and releases occur outside the cache mutex, followed by at most one final
   * exact-size attempt. Smaller partial reclamation that might otherwise suffice is deliberately
   * skipped. A non-OOM error does not start reclamation.
   *
   * @throws rmm::out_of_memory If a direct pinned allocation fails with `cudaErrorMemoryAllocation`
   * after permitted recovery, or the upstream resource cannot supply a slab block
   * @throws rmm::bad_alloc If a direct pinned allocation fails with any other CUDA error, or its
   * size plus alignment padding overflows
   * @throws cucascade::cuda_error If querying capture status or ordering a reused buffer fails
   * @throws cucascade::logic_error If the supplied stream is capturing
   * @throws std::bad_alloc If allocation metadata cannot be acquired
   * @throws std::invalid_argument For a nonzero request, if alignment is not a nonzero power of
   * two, or the upstream block size is smaller than the needed slab size
   * @param stream Stream on which reuse dependencies are inserted
   * @param bytes Requested size
   * @param alignment Requested nonzero power-of-two address alignment
   * @return Pointer to at least `bytes` bytes, or `nullptr` for a zero-byte request
   */
  [[nodiscard]] void* allocate(::cuda::stream_ref stream,
                               std::size_t bytes,
                               std::size_t alignment = alignof(std::max_align_t));

  /**
   * @brief Returns memory to its pool or releases dedicated storage
   *
   * A large buffer that needs room in a full cache evicts the oldest cached entries first. This
   * call may block until GPU work completes on `stream` or on the streams that freed the evicted
   * entries, possibly on other devices. A CUDA failure inside this call is cleared from the calling
   * thread's last-error state (sticky context errors excepted). Storage whose GPU completion cannot
   * be confirmed is withheld: it is never reused or freed.
   *
   * If cache-entry metadata cannot be reserved, retained entries are preserved and the incoming
   * buffer follows the same release and withholding rules.
   *
   * Failed stream-device discovery disables event tracking; reuse or release then requires
   * confirmed stream completion. Failure to restore the calling thread's previous CUDA device
   * terminates the process.
   *
   * @param stream Every prior GPU access to `ptr` must be ordered before work subsequently enqueued
   * on `stream`; the storage may be reused once that point is reached
   * @param ptr Pointer returned by this resource, or `nullptr` for a no-op
   * @param bytes Must equal the `bytes` passed to the matching `allocate` call; any other value is
   * undefined behavior
   * @param alignment Must equal the alignment passed to the matching `allocate` call
   */
  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept;

  /**
   * @brief Allocates on the legacy default stream and synchronizes it before returning
   *
   * The stream is named explicitly as `cudaStreamLegacy` rather than by the null handle, whose
   * meaning depends on whether the calling translation unit is compiled with per-thread default
   * streams. This keeps the stream this inline function synchronizes identical to the stream on
   * which the library orders reuse.
   *
   * Throws the same exceptions as `allocate`, plus any error from synchronizing the stream.
   */
  [[nodiscard]] void* allocate_sync(std::size_t bytes,
                                    std::size_t alignment = alignof(std::max_align_t))
  {
    if (bytes == 0) { return nullptr; }
    ::cuda::stream_ref const stream{cudaStreamLegacy};
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
   * @brief Deallocates on the legacy default stream (`cudaStreamLegacy`) and synchronizes an
   * accepted nonempty call
   *
   * A failed synchronization is ignored.
   */
  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept;

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
  [[nodiscard]] static std::size_t slab_index_for(std::size_t bytes) noexcept;

  /**
   * @brief Computes the bucket for a large request
   *
   * @param bytes Request size greater than `MAX_SLAB_SIZE`
   * @return Smallest representable power of two no less than `bytes` and `MIN_LARGE_BUCKET`, or
   *         `bytes` if the next power of two is not representable
   */
  [[nodiscard]] static std::size_t large_bucket_size_for(std::size_t bytes) noexcept;

  /**
   * @brief Computes the storage size for an ordinarily aligned direct large allocation
   *
   * @param bytes Requested allocation size
   * @return Bucket capacity when the bucket is a power of two no larger than the cache limit,
   * otherwise `bytes`
   */
  [[nodiscard]] std::size_t large_allocation_size(std::size_t bytes) const noexcept;

 private:
  /**
   * @brief An event and the device whose pool in `_event_pools` owns it
   */
  struct pooled_event {
    cuda::cuda_event event;
    int device;
  };

  /**
   * @brief Recycled events of one device
   *
   * Checked-out events and pending creations keep reserved return slots, so `release` never
   * allocates. Retiring an event or failing its creation removes its live reservation.
   */
  class event_pool {
   public:
    /**
     * @brief Takes an idle event without changing its reserved return capacity
     *
     * @return An idle event, or `std::nullopt`
     */
    [[nodiscard]] std::optional<cuda::cuda_event> try_to_take() noexcept;

    /**
     * @brief Reserves a return slot before an event is created outside the pool lock
     */
    void reserve();

    /**
     * @brief Releases the live reservation of a failed creation or retired event
     */
    void release_reservation() noexcept;

    /**
     * @brief Takes back an event with a reserved return slot
     */
    void release(cuda::cuda_event&& event) noexcept;

   private:
    std::vector<cuda::cuda_event> _idle;
    std::size_t _live_events{
      0};  ///< Live events and pending creations with reserved return capacity
  };

  struct free_slab {
    void* ptr;
    std::optional<pooled_event> ready;  ///< Empty when the slab has no pending GPU use
  };

  struct slab_backing_allocation {
    fixed_multiple_blocks_allocation allocation;
    bool quarantined{false};
  };

  struct large_cache_entry {
    void* ptr;
    pooled_event ready;      ///< Always present: only event-ordered buffers are cached
    std::uint64_t sequence;  ///< Value of `_large_cache_sequence` at insertion
  };

  static constexpr std::size_t LARGE_BUCKET_COUNT = static_cast<std::size_t>(
    std::numeric_limits<std::size_t>::digits - std::countr_zero(MIN_LARGE_BUCKET));

  // Cache, slab, and dedicated-allocation bookkeeping uses _mutex. Event-pool bookkeeping and
  // confirmed hit counting use _event_mutex; their _locked helpers require that mutex. Slab-pool
  // growth may call the upstream pinned allocator while holding _mutex.

  /**
   * @brief Decides whether an ordinarily aligned request's bucket is cacheable
   *
   * Only powers of two no larger than the cache limit are cached, so every cacheable bucket has a
   * list in `_large_cache`.
   */
  [[nodiscard]] bool is_cacheable(std::size_t bucket) const noexcept;

  /**
   * @brief Returns the index of a cacheable bucket's list in `_large_cache`
   *
   * `bucket == MIN_LARGE_BUCKET << index`. Each list holds its entries in insertion order.
   */
  [[nodiscard]] static std::size_t large_bucket_index(std::size_t bucket) noexcept;

  [[nodiscard]] void* allocate_slab(::cuda::stream_ref stream, std::size_t bytes);
  [[nodiscard]] void* allocate_large(::cuda::stream_ref stream, std::size_t bytes);
  /**
   * @brief Deallocates storage when the stream's capture status permits it
   * @return Whether subsequent stream synchronization is permitted
   */
  [[nodiscard]] bool try_to_deallocate(::cuda::stream_ref stream,
                                       void* ptr,
                                       std::size_t bytes,
                                       std::size_t alignment) noexcept;
  /// Whether the caller already made, and lost to OOM, the exact-size dedicated attempt.
  enum class exact_attempt : bool { pending, already_failed };

  /// Whether a large-cache lookup is a request's first attempt, which counts toward statistics.
  enum class large_cache_lookup : bool { first_attempt, recovery_retry };

  /**
   * @brief Allocates dedicated storage or reuses a matching bucket during recovery
   *
   * Dedicated storage is an individually owned pinned allocation, tracked in
   * `_dedicated_allocations`, that never enters a slab pool or the large cache.
   *
   * `exact_attempt::already_failed` resumes recovery after the caller's exact-size OOM without
   * duplicating that attempt. `retry_stream` permits ordinary-alignment bucket reuse before
   * dedicated allocation or reclamation.
   */
  [[nodiscard]] void* allocate_dedicated(std::size_t bytes,
                                         std::size_t alignment,
                                         exact_attempt exact,
                                         std::optional<::cuda::stream_ref> retry_stream);
  /**
   * @brief Makes the final pinned allocation attempt after an out-of-memory failure
   *
   * Reclaims cached capacity first and makes the attempt only if the cache covered `bytes`.
   *
   * @throws rmm::out_of_memory If reclamation cannot cover `bytes` or the attempt runs out of
   * memory
   * @throws rmm::bad_alloc If the attempt fails with any other CUDA error
   */
  [[nodiscard]] void* allocate_pinned_after_reclaim(std::size_t bytes);
  void deallocate_dedicated(::cuda::stream_ref stream, void* ptr) noexcept;
  /**
   * @brief Removes a dedicated pointer's record and returns its original base, or nullptr
   */
  [[nodiscard]] void* try_to_take_dedicated_locked(void* ptr) noexcept;
  void deallocate_slab(::cuda::stream_ref stream, void* ptr, std::size_t bytes) noexcept;
  void deallocate_large(::cuda::stream_ref stream, void* ptr, std::size_t bytes) noexcept;

  [[nodiscard]] std::optional<pooled_event> acquire_event(int device) noexcept;
  [[nodiscard]] std::optional<pooled_event> record_ready_event(::cuda::stream_ref stream) noexcept;
  void return_event(pooled_event&& event) noexcept;
  void retire_event(pooled_event&& event) noexcept;
  void return_event_locked(pooled_event&& event) noexcept;
  void retire_event_locked(pooled_event&& event) noexcept;

  [[nodiscard]] void* try_to_take_cached_large(std::size_t bucket,
                                               ::cuda::stream_ref stream,
                                               large_cache_lookup lookup);
  [[nodiscard]] bool evict_oldest_large_locked(std::list<large_cache_entry>& victims) noexcept;
  /**
   * @brief Releases enough oldest cached capacity when the cache covers `bytes`
   *
   * @return Whether enough capacity was detached for one retry; unconfirmed completion still
   * withholds the affected storage
   */
  [[nodiscard]] bool reclaim_large_cache(std::size_t bytes);
  void release_detached(std::list<large_cache_entry>&& detached) noexcept;

  void expand_pool_locked(std::size_t slab_idx);
  void quarantine_slab_locked(void* ptr) noexcept;

  fixed_size_host_memory_resource& _upstream;
  mutable std::mutex _mutex;
  mutable std::mutex _event_mutex;
  std::array<std::vector<free_slab>, SLAB_SIZES.size()> _free_lists{};
  std::map<int, event_pool> _event_pools;
  std::map<void*, void*>
    _dedicated_allocations;  ///< Dedicated returned pointer to allocation base.
  std::vector<slab_backing_allocation> _owned_allocations;
  std::array<std::list<large_cache_entry>, LARGE_BUCKET_COUNT> _large_cache{};
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
