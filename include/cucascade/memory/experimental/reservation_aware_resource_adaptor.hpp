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

template <typename Upstream>
class reservation_aware_resource_adaptor;
class memory_reservation;

using any_device_resource = ::cuda::mr::any_resource<::cuda::mr::device_accessible>;
using any_host_resource   = ::cuda::mr::any_resource<::cuda::mr::host_accessible>;
using any_host_device_resource =
  ::cuda::mr::any_resource<::cuda::mr::device_accessible, ::cuda::mr::host_accessible>;

using device_adaptor      = reservation_aware_resource_adaptor<any_device_resource>;
using host_adaptor        = reservation_aware_resource_adaptor<any_host_resource>;
using host_device_adaptor = reservation_aware_resource_adaptor<any_host_device_resource>;

/**
 * @brief The accessibility of the memory a reservation draws on.
 *
 * Determined by the upstream of the granting adaptor and fixed for the lifetime of the
 * reservation. Selects which of `memory_reservation::as_device()`, `as_host()`, and
 * `as_host_device()` are valid.
 */
enum class reservation_accessibility : std::uint8_t {
  DEVICE,       ///< Device accessible only.
  HOST,         ///< Host accessible only.
  HOST_DEVICE,  ///< Both host and device accessible.
};

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

/**
 * @brief The adaptor instantiations that may grant a reservation.
 */
template <typename T>
concept reservation_adaptor = std::same_as<T, device_adaptor> || std::same_as<T, host_adaptor> ||
                              std::same_as<T, host_device_adaptor>;

// Forward-declared so `reservation_aware_resource_adaptor_impl` can friend it.
// Defined in memory_reservation.hpp.
template <typename Adaptor>
  requires reservation_adaptor<Adaptor>
class memory_reservation_impl;

/**
 * @brief Shared state of a reservation-aware resource adaptor.
 *
 * Owns an upstream resource and tracks a single main memory record (no scoped
 * records). Reservations are granted against a runtime-adjustable limit.
 *
 * @tparam Upstream The upstream memory resource type. Its properties are forwarded to
 * the impl via `cuda::forward_property`, so any tag advertised by `Upstream`
 * (e.g. `cuda::mr::device_accessible`) is visible on the impl and, transitively, on
 * the wrapping `shared_resource`.
 */
template <typename Upstream>
  requires ::cuda::mr::resource<Upstream>
class reservation_aware_resource_adaptor_impl
  : public ::cuda::forward_property<reservation_aware_resource_adaptor_impl<Upstream>, Upstream> {
 public:
  /**
   * @brief Construct with a primary memory resource and a memory limit.
   *
   * @param upstream_mr The primary memory resource (moved in).
   * @param limit Maximum number of bytes that may be allocated and reserved.
   */
  reservation_aware_resource_adaptor_impl(Upstream upstream_mr, std::int64_t limit)
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

  [[nodiscard]] Upstream const& upstream_resource() const noexcept { return upstream_mr_; }

  [[nodiscard]] Upstream const& get_upstream_resource() const noexcept { return upstream_mr_; }

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
   * @note The decision is made against a snapshot: `limit_` and `current_` are read
   * separately from the commit, so a concurrent allocation can still push the total
   * past the limit after a request is granted.
   */
  [[nodiscard]] std::pair<std::size_t, std::size_t> reserve(std::size_t size,
                                                            bool allow_overbooking)
  {
    auto const want = safe_cast<std::int64_t>(size);
    auto reserved   = total_reserved_.load(std::memory_order_acquire);

    // Commit the claim only once it is known to fit, so a rejected request never
    // writes to the counter and never inflates what a concurrent `available()` sees.
    // A failed exchange re-reads the limit and the allocated total as well.
    while (true) {
      std::int64_t const headroom = limit() - current_allocated() - reserved - want;
      if (headroom < 0 && !allow_overbooking) { return {0, safe_cast<std::size_t>(-headroom)}; }
      if (total_reserved_->compare_exchange_weak(
            reserved, reserved + want, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return {size, headroom < 0 ? safe_cast<std::size_t>(-headroom) : 0};
      }
    }
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

 private:
  friend class memory_reservation_impl<device_adaptor>;
  friend class memory_reservation_impl<host_adaptor>;
  friend class memory_reservation_impl<host_device_adaptor>;

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

  Upstream upstream_mr_;

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

}  // namespace detail

/**
 * @brief A memory resource adaptor that only allocates through reservations.
 *
 * This adaptor wraps a primary memory resource and adds a memory limit with allocation
 * tracking. Memory is obtained by calling `reserve()` and allocating through the returned
 * `memory_reservation`.
 *
 * This class is copyable and shares ownership of its internal state via
 * `cuda::mr::shared_resource`.
 *
 * Only three instantiations exist, one per accessibility: `device_adaptor`,
 * `host_adaptor`, and `host_device_adaptor`. The upstream's accessibility is forwarded to
 * the adaptor and on to every reservation it grants, so a `device_adaptor` yields
 * reservations usable as device resources and nothing else.
 *
 * @tparam Upstream The erased upstream resource type: `any_device_resource`,
 * `any_host_resource`, or `any_host_device_resource`.
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
template <typename Upstream>
class reservation_aware_resource_adaptor
  : public ::cuda::mr::shared_resource<detail::reservation_aware_resource_adaptor_impl<Upstream>> {
 public:
  /// @brief The adaptor's shared implementation.
  using impl_type = detail::reservation_aware_resource_adaptor_impl<Upstream>;

  /// @brief The reference-counted handle on the shared implementation.
  using shared_base = ::cuda::mr::shared_resource<impl_type>;

  /// @brief The erased upstream resource type.
  using upstream_type = Upstream;

  /**
   * @brief Construct with the specified primary memory resource and limit.
   *
   * @param upstream_mr The primary memory resource.
   * @param limit Maximum number of bytes that may be allocated and reserved.
   */
  reservation_aware_resource_adaptor(Upstream upstream_mr, std::int64_t limit);

  /**
   * @brief Equality comparison.
   *
   * @param other The other adaptor to compare.
   * @return True if both adaptors share the same underlying state.
   */
  [[nodiscard]] bool operator==(reservation_aware_resource_adaptor const& other) const noexcept
  {
    return this->get() == other.get();
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
   * @return Reference to the erased upstream resource.
   */
  [[nodiscard]] Upstream const& get_upstream_resource() const noexcept;
};

}  // namespace experimental
}  // namespace memory
}  // namespace cucascade

#include <cucascade/memory/experimental/memory_reservation.hpp>
