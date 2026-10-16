/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cucascade/memory/error.hpp>
#include <cucascade/memory/memory_reservation.hpp>    // reservation_limit_policy, reserved_arena
#include <cucascade/memory/notification_channel.hpp>  // event_notifier
#include <cucascade/memory/oom_handling_policy.hpp>

#include <rmm/aligned.hpp>  // rmm::CUDA_ALLOCATION_ALIGNMENT
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace cucascade {
namespace memory {

class reservation_aware_memory_resource_adaptor;

namespace detail {
struct reservation_accounting_core;  // defined in src/memory/reservation_aware_memory_resource.cpp
struct reservation_state;            // defined in src/memory/reservation_aware_memory_resource.cpp
}  // namespace detail

/// @brief Policy consulted when an allocation would exceed its reservation (the sketch's "overflow
/// policy").
using overflow_policy = reservation_limit_policy;

/**
 * @brief Reservation-aware device memory resource.
 *
 * Wraps an upstream device resource and accounts every byte against a single committed-bytes
 * counter ("total allocated"). Reservations commit bytes up front (bounded by @p memory_limit);
 * allocations made through a reservation are drawn from it without touching the global counter.
 * When an allocation would exceed its reservation the overflow policy decides (ignore: proceed,
 * increase: grow the reservation, fail: throw); if it proceeds, only the excess over the
 * reservation is charged to the global counter (bounded by @p capacity). Allocations without a
 * reservation are charged to the global counter.
 *
 * Thread safety: all public member functions are safe for concurrent use. A @ref reservation may be
 * used concurrently from several threads/streams through the reservation-taking overloads; its
 * accounting is serialized by a short per-reservation critical section (a few integer operations),
 * so a reservation shared by many threads serializes them - prefer one reservation per stream for
 * hot paths.
 *
 * Policy contract: the overflow policy runs inside that critical section and must be short and
 * non-blocking, must not allocate through this resource or an adaptor over it, and may act on the
 * reservation only through the reserved_arena pointer it receives (the shipped ignore/fail/increase
 * policies comply). The OOM policy runs outside any lock.
 *
 * Lifetime: the upstream resource must outlive this object. @ref reservation handles may outlive
 * this object: they remain valid for release(), grow_by() and shrink_to_fit() (and their
 * observers) against the shared accounting core, but they can no longer be used to allocate
 * (allocation needs a live resource). Instances are non-copyable and non-movable; consequently this
 * class cannot be bound to rmm::device_async_resource_ref or ::cuda::mr::any_resource (those
 * require a copyable type) - use reservation_aware_memory_resource_adaptor for that.
 *
 * Satisfies ::cuda::mr::resource_with<_, ::cuda::mr::device_accessible> through its untracked path.
 */
class reservation_aware_memory_resource {
 public:
  /// Sentinel for @c commit_cap parameters: use this resource's memory_limit.
  static constexpr std::size_t use_memory_limit = std::numeric_limits<std::size_t>::max();

  /**
   * @brief Move-only handle to committed bytes. Releases them (and posts the release notifier) on
   * destruction. A default-constructed or moved-from handle is empty (`valid() == false`).
   *
   * All observers are lock-free and may race with concurrent allocations (values are snapshots;
   * size() and the allocated bytes are read separately and may be torn across a concurrent
   * grow_by()/shrink_to_fit()).
   */
  class reservation {
   public:
    reservation() noexcept;
    reservation(reservation&& other) noexcept;
    /// @brief Releases the current reservation first, then takes over @p other's.
    reservation& operator=(reservation&& other) noexcept;
    reservation(reservation const&)            = delete;
    reservation& operator=(reservation const&) = delete;
    ~reservation();

    /// @return true if this handle refers to a live reservation.
    [[nodiscard]] bool valid() const noexcept;
    explicit operator bool() const noexcept { return valid(); }

    /// @return reserved bytes (0 for an empty handle).
    [[nodiscard]] std::size_t size() const noexcept;
    /// @return max(0, bytes currently allocated through this reservation); may exceed size() after
    /// overflow.
    [[nodiscard]] std::size_t allocated_bytes() const noexcept;
    /// @return high-water mark of allocated_bytes() since creation or the last reset.
    [[nodiscard]] std::size_t peak_allocated_bytes() const noexcept;
    /// @return max(0, size() - allocated bytes) (0 if empty).
    [[nodiscard]] std::size_t available_bytes() const noexcept;
    /// @brief Resets the peak to 0 (not to the current value), mirroring the legacy adaptor.
    void reset_peak_allocated_bytes() noexcept;

    /**
     * @brief Grows the reservation by @p additional_bytes, committing them against @p commit_cap.
     * @return true on success; false if the commit would exceed min(commit_cap, capacity). No-op on
     * empty.
     */
    bool grow_by(std::size_t additional_bytes, std::size_t commit_cap = use_memory_limit);

    /// @brief Shrinks size() down to max(0, allocated bytes) and returns the difference to the
    /// pool. No-op if the reservation is fully used or exceeded.
    void shrink_to_fit();

    /// @brief Releases the reservation now; the handle becomes empty. Safe to call on an empty
    /// handle.
    void release() noexcept;

   private:
    friend class reservation_aware_memory_resource;
    friend class reservation_aware_memory_resource_adaptor;
    explicit reservation(std::unique_ptr<detail::reservation_state> state) noexcept;

    std::unique_ptr<detail::reservation_state> _state;
  };

  /**
   * @brief Constructs the resource.
   * @param upstream Upstream device resource (must outlive this object).
   * @param memory_limit Ceiling for committed bytes reached through reservations (default commit
   *        cap).
   * @param capacity Ceiling for all committed bytes (reservations + allocations); must be >=
   *        memory_limit and <= INT64_MAX / 2 (keeps the signed accounting overflow-free).
   * @param default_oom_policy Used when the upstream reports out-of-memory (nullptr = throw).
   * @param default_overflow_policy Used by allocate(..., reservation&) and by adaptor bindings that
   *        did not supply their own (nullptr = ignore: the allocation proceeds and the excess over
   *        the reservation is charged to the global counter).
   * @param pool_handle Pool reported in cucascade_out_of_memory; nullptr = recover from upstream
   *        (RMM cuda_async_* resources) or null.
   * @throws std::invalid_argument if memory_limit > capacity or capacity > INT64_MAX / 2.
   */
  explicit reservation_aware_memory_resource(
    rmm::device_async_resource_ref upstream,
    std::size_t memory_limit,
    std::size_t capacity,
    std::unique_ptr<oom_handling_policy> default_oom_policy  = nullptr,
    std::unique_ptr<overflow_policy> default_overflow_policy = nullptr,
    cudaMemPool_t pool_handle                                = nullptr);

  /// @brief Same as above with memory_limit == capacity (the spec's (mr, capacity, oom, overflow)
  /// form).
  explicit reservation_aware_memory_resource(
    rmm::device_async_resource_ref upstream,
    std::size_t capacity,
    std::unique_ptr<oom_handling_policy> default_oom_policy  = nullptr,
    std::unique_ptr<overflow_policy> default_overflow_policy = nullptr,
    cudaMemPool_t pool_handle                                = nullptr);

  ~reservation_aware_memory_resource();
  reservation_aware_memory_resource(reservation_aware_memory_resource const&)            = delete;
  reservation_aware_memory_resource& operator=(reservation_aware_memory_resource const&) = delete;
  reservation_aware_memory_resource(reservation_aware_memory_resource&&)                 = delete;
  reservation_aware_memory_resource& operator=(reservation_aware_memory_resource&&)      = delete;

  //===----------------------------------------------------------------------===//
  // Reservations
  //===----------------------------------------------------------------------===//

  /**
   * @brief Commits exactly @p bytes.
   * @param commit_cap Ceiling for total committed bytes this call may reach; use_memory_limit
   *        (default) means memory_limit; any other value is clamped to capacity.
   * @param release_notifier Posted when the reservation is released (may be nullptr).
   * @throws cucascade_out_of_memory (LIMIT_EXCEEDED) if the commit would exceed the ceiling.
   */
  [[nodiscard]] reservation reserve(std::size_t bytes,
                                    std::size_t commit_cap = use_memory_limit,
                                    std::unique_ptr<event_notifier> release_notifier = nullptr);

  /// @brief Like reserve() but returns an empty handle instead of throwing on insufficient space.
  [[nodiscard]] reservation try_reserve(std::size_t bytes,
                                        std::size_t commit_cap = use_memory_limit,
                                        std::unique_ptr<event_notifier> release_notifier = nullptr);

  /// @brief Commits min(bytes, ceiling - committed) (possibly 0). Always returns a valid handle.
  [[nodiscard]] reservation reserve_upto(
    std::size_t bytes,
    std::size_t commit_cap                           = use_memory_limit,
    std::unique_ptr<event_notifier> release_notifier = nullptr);

  /// @return true if @p res is a live reservation created by this resource.
  [[nodiscard]] bool owns(reservation const& res) const noexcept;

  //===----------------------------------------------------------------------===//
  // Allocation through an explicit reservation
  //===----------------------------------------------------------------------===//

  /**
   * @brief Allocates @p bytes on @p stream through @p res. If the allocation does not fit the
   * reservation, the default overflow policy is consulted and, if it returns, the allocation
   * proceeds with only the excess over the reservation charged to the global counter.
   *
   * A request with @p bytes > capacity can never be satisfied and is rejected up front with
   * cucascade_out_of_memory (LIMIT_EXCEEDED): neither the overflow policy nor the OOM policy is
   * consulted (the legacy adaptor ran both first). Any other capacity shortfall is routed to the
   * OOM policy.
   * @throws cucascade::logic_error if @p res is empty or owned by another resource.
   * @throws rmm::out_of_memory from a fail/increase overflow policy.
   * @throws cucascade_out_of_memory (LIMIT_EXCEEDED / ALLOCATION_FAILED).
   */
  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment,
                 reservation& res);

  /**
   * @brief Returns @p ptr to upstream and credits @p res; the part that was above the reservation
   * is credited to the global counter.
   *
   * @p res should be a live reservation of this resource. If it is empty or owned by another
   * resource, debug builds assert; release builds fall back to the untracked free (the padded size
   * is credited to the global counter).
   */
  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment,
                  reservation& res) noexcept;

  //===----------------------------------------------------------------------===//
  // ::cuda::mr::resource interface (untracked path: charged to the global counter)
  //===----------------------------------------------------------------------===//

  /**
   * @brief Allocates @p bytes on @p stream, charging align_up(bytes, 256) to the global counter.
   *
   * A request with @p bytes > capacity is rejected up front with cucascade_out_of_memory
   * (LIMIT_EXCEEDED) without consulting the OOM policy; any other capacity shortfall or upstream
   * out-of-memory (std::bad_alloc family) is routed to the default OOM policy.
   */
  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);
  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;
  /// @brief allocate() on the legacy default stream, then synchronizes it; if the synchronization
  /// throws, the allocation is returned before the exception propagates.
  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);
  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  /// @return true iff both refer to the same accounting core (identity).
  bool operator==(reservation_aware_memory_resource const& other) const noexcept;

  friend void get_property(reservation_aware_memory_resource const&,
                           ::cuda::mr::device_accessible) noexcept
  {
  }

  //===----------------------------------------------------------------------===//
  // Introspection (lock-free snapshots)
  //===----------------------------------------------------------------------===//

  [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept;
  [[nodiscard]] cudaMemPool_t get_pool_handle() const noexcept;
  [[nodiscard]] std::size_t get_capacity() const noexcept;
  [[nodiscard]] std::size_t get_memory_limit() const noexcept;
  /// @return committed bytes: live reservations + bytes charged to the global counter.
  [[nodiscard]] std::size_t get_total_allocated_bytes() const noexcept;
  [[nodiscard]] std::size_t get_peak_total_allocated_bytes() const noexcept;
  /// @return capacity - total allocated (clamped at 0).
  [[nodiscard]] std::size_t get_available_memory() const noexcept;
  /// @return sum of live reservation sizes (statistic).
  [[nodiscard]] std::size_t get_total_reserved_bytes() const noexcept;
  [[nodiscard]] std::size_t get_active_reservation_count() const noexcept;
  [[nodiscard]] oom_handling_policy& get_default_oom_policy() const noexcept;
  [[nodiscard]] overflow_policy& get_default_overflow_policy() const noexcept;

 private:
  friend class reservation_aware_memory_resource_adaptor;

  // Internal hooks shared with the adaptor (definitions in the .cpp).
  void* do_allocate_tracked(::cuda::stream_ref stream,
                            std::size_t bytes,
                            std::size_t alignment,
                            detail::reservation_state& state,
                            oom_handling_policy& oom,
                            overflow_policy& overflow);
  void do_deallocate_tracked(::cuda::stream_ref stream,
                             void* ptr,
                             std::size_t bytes,
                             std::size_t alignment,
                             detail::reservation_state& state) noexcept;
  void* do_allocate_untracked(::cuda::stream_ref stream,
                              std::size_t bytes,
                              std::size_t alignment,
                              oom_handling_policy& oom_policy);
  void do_deallocate_untracked(::cuda::stream_ref stream,
                               void* ptr,
                               std::size_t bytes,
                               std::size_t alignment) noexcept;
  reservation make_reservation(std::size_t committed_bytes,
                               std::unique_ptr<event_notifier> release_notifier);
  void* upstream_allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment);

  std::shared_ptr<detail::reservation_accounting_core> _core;
  rmm::device_async_resource_ref _upstream;
  cudaMemPool_t _pool_handle{nullptr};
  std::unique_ptr<oom_handling_policy> _default_oom_policy;
  std::unique_ptr<overflow_policy> _default_overflow_policy;
};

static_assert(
  ::cuda::mr::resource_with<reservation_aware_memory_resource, ::cuda::mr::device_accessible>);
// NOTE: do not add a static_assert on rmm::device_async_resource_ref bindability: it fails
// (non-copyable type).

}  // namespace memory
}  // namespace cucascade
