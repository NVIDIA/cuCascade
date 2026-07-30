/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
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

#include <cucascade/error.hpp>
#include <cucascade/utils/atomics.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/error.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace cucascade {
namespace memory {
namespace experimental {

class reservation_aware_resource_adaptor;
class memory_reservation;

using any_device_resource = ::cuda::mr::any_resource<::cuda::mr::device_accessible>;

/**
 * @brief Snapshot of the adaptor's main (non-scoped) allocation accounting.
 */
struct memory_record {
  std::int64_t num_current_allocs{0};
  std::int64_t num_total_allocs{0};
  std::int64_t current{0};
  std::int64_t total{0};
  std::int64_t peak{0};
  std::int64_t max{0};
};

/**
 * @brief Policy controlling whether a reservation may exceed the adaptor's limit.
 */
enum class allow_overbooking : bool {
  NO,   ///< Fail the request rather than exceed the limit.
  YES,  ///< Grant the request even when the memory isn't available.
};

namespace detail {

template <typename To, typename From>
[[nodiscard]] constexpr To safe_cast(From value)
{
  if constexpr (std::is_same_v<From, To>) {
    return value;
  } else {
    if (!std::in_range<To>(value)) {
      throw std::overflow_error("cucascade cast: value out of range " + std::to_string(value));
    }
    return static_cast<To>(value);
  }
}

// The adaptor is a template parameter only so that the reservation can hold one by
// value: `reservation_aware_resource_adaptor` is defined in terms of the adaptor impl
// below, so it is still incomplete here, and only a dependent member type defers the
// completeness requirement to instantiation time.
template <typename Adaptor>
  requires std::same_as<Adaptor, reservation_aware_resource_adaptor>
class memory_reservation_impl;

/**
 * @brief Shared state of a reservation-aware resource adaptor.
 *
 * Owns an upstream device resource and tracks a single main memory record (no scoped
 * records). Reservations are granted against a runtime-adjustable limit.
 */
class reservation_aware_resource_adaptor_impl {
 public:
  /**
   * @brief Construct with a primary memory resource and a memory limit.
   *
   * @param upstream_mr The primary memory resource (moved in).
   * @param limit Maximum number of bytes that may be allocated and reserved.
   */
  reservation_aware_resource_adaptor_impl(any_device_resource upstream_mr, std::int64_t limit)
    : upstream_mr_{std::move(upstream_mr)}, limit_{limit}
  {
  }

  ~reservation_aware_resource_adaptor_impl() = default;

  reservation_aware_resource_adaptor_impl(reservation_aware_resource_adaptor_impl const&) = delete;
  reservation_aware_resource_adaptor_impl(reservation_aware_resource_adaptor_impl&&)      = delete;
  reservation_aware_resource_adaptor_impl& operator=(
    reservation_aware_resource_adaptor_impl const&) = delete;
  reservation_aware_resource_adaptor_impl& operator=(reservation_aware_resource_adaptor_impl&&) =
    delete;

  [[nodiscard]] bool operator==(reservation_aware_resource_adaptor_impl const& other) const noexcept
  {
    return this == std::addressof(other);
  }

  [[nodiscard]] any_device_resource const& get_upstream_resource() const noexcept
  {
    return upstream_mr_;
  }

  [[nodiscard]] std::int64_t limit() const noexcept
  {
    return limit_.load(std::memory_order_acquire);
  }

  void set_limit(std::int64_t limit) noexcept { limit_.store(limit, std::memory_order_release); }

  [[nodiscard]] std::int64_t total_reserved() const noexcept
  {
    return total_reserved_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::int64_t current_allocated() const noexcept
  {
    return current_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::int64_t available() const noexcept
  {
    return limit() - current_allocated() - total_reserved();
  }

  [[nodiscard]] memory_record get_main_record() const
  {
    return memory_record{
      .num_current_allocs = num_current_allocs_.load(std::memory_order_acquire),
      .num_total_allocs   = num_total_allocs_.load(std::memory_order_acquire),
      .current            = current_.load(std::memory_order_acquire),
      .total              = total_.load(std::memory_order_acquire),
      .peak               = peak_.peak(),
      .max                = max_.peak(),
    };
  }

  /**
   * @brief Reserve @p size bytes against the limit.
   *
   * @param size The number of bytes to reserve.
   * @param allow_overbooking Whether to grant the reservation even when the memory
   * isn't available.
   * @return A pair of the number of bytes granted (either @p size or zero) and the
   * number of bytes by which the request overbooks the limit.
   *
   * @note Rejections are best-effort under contention: concurrent requests each claim
   * before checking, so requests that would fit individually may be rejected.
   */
  [[nodiscard]] std::pair<std::size_t, std::size_t> reserve(std::size_t size,
                                                            bool allow_overbooking)
  {
    auto const want             = safe_cast<std::int64_t>(size);
    std::int64_t const capacity = limit() - current_allocated();

    // Claim the bytes up front and roll back if they didn't fit. While a claim is
    // being rolled back the reserved total reads high, which makes a concurrent
    // `available()` pessimistic, never optimistic.
    auto const reserved         = total_reserved_.add(want, std::memory_order_acq_rel) - want;
    std::int64_t const headroom = capacity - (reserved + want);
    if (headroom >= 0) { return {size, 0}; }
    auto const overbooking = safe_cast<std::size_t>(-headroom);
    if (!allow_overbooking) {
      total_reserved_.sub(want, std::memory_order_acq_rel);
      return {0, overbooking};
    }
    return {size, overbooking};
  }

  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    void* ret = upstream_mr_.allocate(stream, bytes, alignment);
    record_allocation(safe_cast<std::int64_t>(bytes));
    return ret;
  }

  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    record_deallocation(safe_cast<std::int64_t>(bytes));
    upstream_mr_.deallocate(stream, ptr, bytes, alignment);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    auto* ptr = allocate(sync_stream_, bytes, alignment);
    sync_stream_.synchronize();
    return ptr;
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate(sync_stream_, ptr, bytes, alignment);
    sync_stream_.synchronize_no_throw();
  }

  friend void get_property(reservation_aware_resource_adaptor_impl const&,
                           ::cuda::mr::device_accessible) noexcept
  {
  }

 private:
  friend class memory_reservation_impl<reservation_aware_resource_adaptor>;

  void record_allocation(std::int64_t nbytes)
  {
    num_total_allocs_.add(1, std::memory_order_acq_rel);
    num_current_allocs_.add(1, std::memory_order_acq_rel);
    auto const current = current_.add(nbytes, std::memory_order_acq_rel);
    total_.add(nbytes, std::memory_order_acq_rel);
    peak_.update_peak(current);
    max_.update_peak(nbytes);
  }

  void record_deallocation(std::int64_t nbytes) noexcept
  {
    current_.sub(nbytes, std::memory_order_acq_rel);
    num_current_allocs_.sub(1, std::memory_order_acq_rel);
  }

  any_device_resource upstream_mr_;

  utils::atomic_bounded_counter<std::int64_t> num_current_allocs_{0};
  utils::atomic_bounded_counter<std::int64_t> num_total_allocs_{0};
  utils::atomic_bounded_counter<std::int64_t> current_{0};
  utils::atomic_bounded_counter<std::int64_t> total_{0};
  utils::atomic_peak_tracker<std::int64_t> peak_;
  utils::atomic_peak_tracker<std::int64_t> max_;  ///< Largest single allocation observed.

  std::atomic<std::int64_t> limit_;
  // Reservations move bytes in and out of this counter as they allocate, free, and die.
  utils::atomic_bounded_counter<std::int64_t> total_reserved_{0};

  rmm::cuda_stream sync_stream_{rmm::cuda_stream::flags::non_blocking};
};

/**
 * @brief Shared state of a memory reservation.
 *
 * Satisfies the `cuda::mr::resource` concept so it can be a `cuda::mr::shared_resource`.
 * Allocating moves bytes from the adaptor's reserved counter to its allocated counter;
 * the unspent balance is refunded only when the last reference dies.
 *
 * @tparam Adaptor Always `reservation_aware_resource_adaptor`.
 */
template <typename Adaptor>
  requires std::same_as<Adaptor, reservation_aware_resource_adaptor>
class memory_reservation_impl {
 public:
  memory_reservation_impl(Adaptor adaptor, std::int64_t grant, std::size_t overbooking)
    : adaptor_{std::move(adaptor)}, grant_{grant}, overbooking_{overbooking}, balance_{grant}
  {
  }

  ~memory_reservation_impl()
  {
    adaptor_->total_reserved_.sub(balance(), std::memory_order_acq_rel);
  }

  memory_reservation_impl(memory_reservation_impl const&)            = delete;
  memory_reservation_impl(memory_reservation_impl&&)                 = delete;
  memory_reservation_impl& operator=(memory_reservation_impl const&) = delete;
  memory_reservation_impl& operator=(memory_reservation_impl&&)      = delete;

  [[nodiscard]] std::int64_t grant() const noexcept { return grant_; }

  [[nodiscard]] std::size_t overbooking() const noexcept { return overbooking_; }

  [[nodiscard]] std::int64_t balance() const noexcept
  {
    return balance_.load(std::memory_order_acquire);
  }

  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    auto const amount = safe_cast<std::int64_t>(bytes);
    draw_down_res(amount);
    void* ptr = nullptr;
    try {
      ptr = adaptor_->allocate(stream, bytes, alignment);
    } catch (...) {
      balance_.fetch_add(amount, std::memory_order_acq_rel);
      throw;
    }
    adaptor_->total_reserved_.sub(amount, std::memory_order_acq_rel);
    return ptr;
  }

  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    auto const amount = safe_cast<std::int64_t>(bytes);
    balance_.fetch_add(amount, std::memory_order_acq_rel);
    adaptor_->total_reserved_.add(amount, std::memory_order_acq_rel);
    adaptor_->deallocate(stream, ptr, bytes, alignment);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    auto* ptr = allocate(adaptor_->sync_stream_, bytes, alignment);
    adaptor_->sync_stream_.synchronize();
    return ptr;
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate(adaptor_->sync_stream_, ptr, bytes, alignment);
    adaptor_->sync_stream_.synchronize_no_throw();
  }

  [[nodiscard]] bool operator==(memory_reservation_impl const& other) const noexcept
  {
    return this == std::addressof(other);
  }

  friend void get_property(memory_reservation_impl const&, ::cuda::mr::device_accessible) noexcept
  {
  }

  [[nodiscard]] Adaptor const& adaptor() const noexcept { return adaptor_; }

 private:
  void draw_down_res(std::int64_t bytes)
  {
    auto balance = balance_.load(std::memory_order_relaxed);
    do {
      if (bytes > balance) {
        CUCASCADE_FAIL("allocation of " + std::to_string(bytes) +
                         " bytes exceeds reservation (grant: " + std::to_string(grant_) +
                         ", remaining: " + std::to_string(balance) + ")",
                       rmm::out_of_memory);
      }
    } while (!balance_.compare_exchange_weak(
      balance, balance - bytes, std::memory_order_acq_rel, std::memory_order_relaxed));
  }

  Adaptor adaptor_;
  std::int64_t const grant_;
  std::size_t const overbooking_;
  std::atomic<std::int64_t> balance_;
};

}  // namespace detail

/**
 * @brief A memory resource adaptor that only allocates through reservations.
 *
 * This adaptor wraps a primary device memory resource and adds a memory limit with
 * allocation tracking. Memory is obtained by calling `reserve()` and allocating through
 * the returned `memory_reservation`.
 *
 * This class is copyable and shares ownership of its internal state via
 * `cuda::mr::shared_resource`.
 *
 * @par Allocating without a reservation
 *
 * The adaptor is itself a memory resource, so it can be handed to cudf or an RMM
 * container directly. Those allocations are tracked, and therefore still consume
 * `available()`, but they draw on no reservation and are not capped. Allocate through
 * a `memory_reservation` when the budget has to be enforced.
 *
 * @par Accounting
 *
 * Three quantities describe the state of the adaptor:
 * - `current_allocated()`: bytes currently allocated, tracked on every allocation.
 * - `total_reserved()`: bytes held by live reservations but not yet allocated.
 * - `available() == limit() - current_allocated() - total_reserved()`.
 *
 * Allocating through a reservation moves bytes from the second bucket to the first,
 * leaving `available()` unchanged; that is what makes a reservation a promise.
 */
class reservation_aware_resource_adaptor
  : public ::cuda::mr::shared_resource<detail::reservation_aware_resource_adaptor_impl> {
 public:
  /// @brief The adaptor's shared implementation.
  using impl_type = detail::reservation_aware_resource_adaptor_impl;

  /// @brief The reference-counted handle on the shared implementation.
  using shared_base = ::cuda::mr::shared_resource<impl_type>;

  /// @brief Tag this resource as device-accessible for the CCCL concept.
  friend void get_property(reservation_aware_resource_adaptor const&,
                           ::cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Construct with the specified primary memory resource and limit.
   *
   * @param upstream_mr The primary memory resource.
   * @param limit Maximum number of bytes that may be allocated and reserved.
   */
  reservation_aware_resource_adaptor(any_device_resource upstream_mr, std::int64_t limit);

  /**
   * @brief Equality comparison.
   *
   * @param other The other adaptor to compare.
   * @return True if both adaptors share the same underlying state.
   */
  [[nodiscard]] bool operator==(reservation_aware_resource_adaptor const& other) const noexcept
  {
    return get() == other.get();
  }

  /**
   * @brief Reserve an amount of memory.
   *
   * Creates a new reservation of the specified size to inform about upcoming
   * allocations.
   *
   * If overbooking is allowed, a reservation of @p size is returned even when the
   * memory isn't available. In that case the caller must free (at least)
   * `memory_reservation::overbooking()` bytes before using the reservation.
   *
   * If overbooking isn't allowed, a reservation of size zero is returned on failure,
   * with `memory_reservation::overbooking()` reporting by how much the request missed.
   * A zero-sized reservation fails at allocation time: the first allocation through it
   * throws `rmm::out_of_memory`.
   *
   * @param size The number of bytes to reserve.
   * @param overbooking_policy Whether overbooking is allowed.
   * @return The reservation. On success its grant always equals @p size and on
   * failure it always equals zero (a zero-sized reservation never fails).
   */
  [[nodiscard]] memory_reservation reserve(std::size_t size, allow_overbooking overbooking_policy);

  /**
   * @brief Get the memory limit.
   *
   * @return The limit in bytes.
   */
  [[nodiscard]] std::int64_t limit() const noexcept;

  /**
   * @brief Update the memory limit at runtime.
   *
   * @param limit The new byte limit.
   */
  void set_limit(std::int64_t limit) noexcept;

  /**
   * @brief Get the total current allocated memory through this adaptor.
   *
   * @return Total number of currently allocated bytes.
   */
  [[nodiscard]] std::int64_t current_allocated() const noexcept;

  /**
   * @brief Get the memory currently held by live reservations.
   *
   * Excludes reserved bytes that have already been allocated; those are reported by
   * `current_allocated()` instead.
   *
   * @return Total number of reserved bytes.
   */
  [[nodiscard]] std::int64_t total_reserved() const noexcept;

  /**
   * @brief Get the memory available for new reservations.
   *
   * Computed as `limit() - current_allocated() - total_reserved()`. May be negative
   * when reservations have overbooked the limit.
   *
   * @return The available memory in bytes.
   */
  [[nodiscard]] std::int64_t available() const noexcept;

  /**
   * @brief Returns a snapshot of the main memory record.
   *
   * @return A copy of the current main memory record.
   */
  [[nodiscard]] memory_record get_main_record() const;

  /**
   * @brief Get a reference to the primary upstream resource.
   *
   * @return Reference to the RMM memory resource.
   */
  [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept;
};

static_assert(
  ::cuda::mr::resource_with<reservation_aware_resource_adaptor, ::cuda::mr::device_accessible>);

/**
 * @brief A memory reservation that is itself a memory resource.
 *
 * Granted by `reservation_aware_resource_adaptor::reserve()`, a reservation holds a
 * budget of bytes carved out of the adaptor's limit. It is an RMM memory resource, so
 * it can be handed to cudf (or anything else taking a `rmm::device_async_resource_ref`),
 * and every allocation made through it is charged against that budget. An allocation
 * exceeding the remaining `balance()` throws `rmm::out_of_memory`; deallocating returns
 * the bytes to the balance.
 *
 * @par Ownership
 *
 * Like `reservation_aware_resource_adaptor`, this is a `cuda::mr::shared_resource`, so
 * copies share the same reservation and are interchangeable. RMM stores such a copy
 * inside every buffer allocated from the reservation, which is what keeps the
 * reservation alive for as long as those buffers need it to service deallocations.
 *
 * The unspent balance is refunded to the adaptor when the last copy dies. Reserving
 * more than is allocated therefore keeps the surplus out of circulation for as long as
 * any derived buffer lives, so reserve what you actually use.
 *
 * @code{.cpp}
 * auto res = adaptor.reserve(1 << 30, allow_overbooking::NO);
 * auto table = cudf::groupby(..., stream, res);
 * @endcode
 */
class memory_reservation : public ::cuda::mr::shared_resource<
                             detail::memory_reservation_impl<reservation_aware_resource_adaptor>> {
  using shared_base = ::cuda::mr::shared_resource<
    detail::memory_reservation_impl<reservation_aware_resource_adaptor>>;

 public:
  /// @brief The shared state of the reservation.
  using impl_type = detail::memory_reservation_impl<reservation_aware_resource_adaptor>;

  /// @brief Tag this resource as device-accessible for the CCCL concept.
  friend void get_property(memory_reservation const&, ::cuda::mr::device_accessible) noexcept {}

  /**
   * @brief Equality comparison.
   *
   * @param other The other reservation to compare.
   * @return True if both refer to the same reservation.
   */
  [[nodiscard]] bool operator==(memory_reservation const& other) const noexcept
  {
    return get() == other.get();
  }

  /**
   * @brief The number of bytes originally granted.
   *
   * @return The granted size in bytes.
   */
  [[nodiscard]] std::size_t grant() const noexcept
  {
    return detail::safe_cast<std::size_t>(get().grant());
  }

  /**
   * @brief The remaining unallocated size of the reservation.
   *
   * @return The remaining size in bytes.
   */
  [[nodiscard]] std::size_t balance() const noexcept
  {
    return detail::safe_cast<std::size_t>(get().balance());
  }

  /**
   * @brief The number of bytes by which the grant overbooks the adaptor's limit.
   *
   * Nonzero only when the reservation was granted with `allow_overbooking::YES`. The
   * caller must free at least this much memory before using the reservation.
   *
   * @return The overbooked size in bytes.
   */
  [[nodiscard]] std::size_t overbooking() const noexcept { return get().overbooking(); }

  /**
   * @brief The adaptor that granted the reservation.
   *
   * @return The adaptor.
   */
  [[nodiscard]] reservation_aware_resource_adaptor const& adaptor() const noexcept
  {
    return get().adaptor();
  }

 private:
  friend class reservation_aware_resource_adaptor;

  /**
   * @brief Construct from an already-granted reservation.
   *
   * Private so that only `reservation_aware_resource_adaptor` can grant reservations.
   * The reservation holds a copy of the adaptor, so the adaptor stays alive for as
   * long as any buffer allocated from the reservation needs it.
   *
   * @param adaptor The adaptor that granted the reservation.
   * @param granted The number of bytes granted.
   * @param overbooking The number of bytes by which @p granted overbooks the limit.
   */
  memory_reservation(reservation_aware_resource_adaptor const& adaptor,
                     std::size_t granted,
                     std::size_t overbooking);
};

static_assert(::cuda::mr::resource_with<memory_reservation, ::cuda::mr::device_accessible>);

}  // namespace experimental
}  // namespace memory
}  // namespace cucascade
