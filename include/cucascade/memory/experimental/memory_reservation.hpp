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

#include <cucascade/memory/experimental/reservation_aware_resource_adaptor.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <variant>

namespace cucascade {
namespace memory {
namespace experimental {
namespace detail {

/**
 * @brief Shared state of a memory reservation.
 *
 * Satisfies the `cuda::mr::resource` concept so it can be a `cuda::mr::shared_resource`.
 * Allocating moves bytes from the adaptor's reserved counter to its allocated counter;
 * the unspent balance is refunded only when the last reference dies.
 *    std::shared_ptr<const over_reservation_policy> policy);

 * The reservation's claim on `reservation_aware_resource_adaptor::total_reserved()` is
 * `reserved_part(balance())`, never the raw balance, so a soft reservation that has
 * overdrawn claims nothing rather than crediting back memory it is still using. Every
 * balance transition adjusts the counter by the change in that quantity, which keeps the
 * claim exact across allocation, partial release, and full recovery, and unwinds it to
 * zero on destruction.
 *
 * @tparam Adaptor One of `device_adaptor`, `host_adaptor`, or `host_device_adaptor`. Its
 * properties are forwarded to the reservation via `cuda::forward_property`, so a
 * reservation advertises whatever the granting adaptor advertises (e.g.
 * `cuda::mr::device_accessible`). That forwarding is what lets `memory_reservation`
 * project this impl into an erased resource of the matching accessibility.
 */
template <typename Adaptor>
  requires reservation_adaptor<Adaptor>
class memory_reservation_impl
  : public reservation_control,
    public ::cuda::forward_property<memory_reservation_impl<Adaptor>, Adaptor> {
 public:
  memory_reservation_impl(Adaptor adaptor,
                          std::int64_t grant,
                          std::size_t overbooking,
                          std::shared_ptr<const over_reservation_policy> policy)
    : adaptor_{std::move(adaptor)},
      grant_{grant},
      overbooking_{overbooking},
      policy_{std::move(policy)},
      balance_{grant}
  {
    if (!policy_) { CUCASCADE_FAIL("over_reservation_policy must not be null"); }
  }

  ~memory_reservation_impl()
  {
    adaptor_->total_reserved_.sub(reserved_part(balance()), std::memory_order_acq_rel);
  }

  memory_reservation_impl(memory_reservation_impl const&)            = delete;
  memory_reservation_impl(memory_reservation_impl&&)                 = delete;
  memory_reservation_impl& operator=(memory_reservation_impl const&) = delete;
  memory_reservation_impl& operator=(memory_reservation_impl&&)      = delete;

  [[nodiscard]] std::int64_t grant() const noexcept override { return grant_; }

  [[nodiscard]] std::int64_t balance() const noexcept override
  {
    return balance_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::size_t overbooking() const noexcept override { return overbooking_; }

  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    auto const amount      = safe_cast<std::int64_t>(bytes);
    auto const before      = draw_down_res(amount);
    auto const after       = before - amount;
    auto const claim_delta = calc_delta(before, after);

    adaptor_->total_reserved_.sub(claim_delta, std::memory_order_acq_rel);

    void* ptr = nullptr;
    try {
      ptr = adaptor_->allocate(stream, bytes, alignment);
    } catch (...) {
      auto const rollback_before = balance_.fetch_add(amount, std::memory_order_acq_rel);
      auto const rollback_delta  = calc_delta(rollback_before + amount, rollback_before);
      adaptor_->total_reserved_.add(rollback_delta, std::memory_order_acq_rel);
      throw;
    }
    return ptr;
  }

  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    auto const amount = safe_cast<std::int64_t>(bytes);
    auto const before = balance_.fetch_add(amount, std::memory_order_acq_rel);
    adaptor_->total_reserved_.add(calc_delta(before + amount, before), std::memory_order_acq_rel);
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

  /// @brief The hook `cuda::forward_property` uses to forward the adaptor's properties.
  [[nodiscard]] Adaptor const& upstream_resource() const noexcept { return adaptor_; }

 private:
  /// @brief The part of a balance that is still reserved-but-unspent. An overdraft is
  /// funded from outside the grant, so it contributes nothing to the adaptor's reserve.
  static constexpr std::int64_t reserved_part(std::int64_t balance) noexcept
  {
    return balance > 0 ? balance : 0;
  }

  /// @return The difference between two reserved portions.
  static constexpr std::int64_t calc_delta(std::int64_t lhs, std::int64_t rhs) noexcept
  {
    return reserved_part(lhs) - reserved_part(rhs);
  }

  /// @return The balance before the draw-down.
  std::int64_t draw_down_res(std::int64_t bytes)
  {
    auto balance = balance_.load(std::memory_order_relaxed);
    while (true) {
      if (bytes > balance) {
        policy_->handle_over_reservation(bytes, balance, *this);
        return balance_.fetch_sub(bytes, std::memory_order_acq_rel);
      }

      if (balance_.compare_exchange_weak(
            balance, balance - bytes, std::memory_order_acq_rel, std::memory_order_relaxed)) {
        return balance;
      }
    }
  }

  Adaptor adaptor_;
  std::int64_t const grant_;
  std::size_t const overbooking_;
  std::shared_ptr<const over_reservation_policy> policy_;
  std::atomic<std::int64_t> balance_;
};

/**
 * @brief The reference-counted handle on a reservation's shared state.
 */
template <typename Adaptor>
using reservation_handle = ::cuda::mr::shared_resource<memory_reservation_impl<Adaptor>>;

}  // namespace detail

/**
 * @brief A memory reservation, independent of the accessibility of its memory.
 *
 * Granted by `reservation_aware_resource_adaptor::reserve()`, a reservation holds a
 * budget of bytes carved out of the adaptor's limit. Every allocation made through it is
 * charged against that budget: an allocation exceeding the remaining `balance()` throws
 * `rmm::out_of_memory`, and deallocating returns the bytes to the balance.
 *
 * The default soft over-reservation policy permits allocations past the grant, reporting
 * the overdraft as a negative `balance()`. Only the grant is backed by the adaptor's
 * reserve; the overdraft is spent against the limit without a claim on it, so it shows up
 * in `current_allocated()` alone.
 *
 * This is a single type regardless of where the memory lives. Internally it holds one of
 * three shared states, chosen by the granting adaptor's upstream, and its own type says
 * nothing about accessibility. Query `accessibility()` to find out.
 *
 * @par Handing a reservation to cudf or RMM
 *
 * A reservation is not itself a memory resource. Project it into an erased resource of
 * the accessibility you need, and pass that:
 *
 * @code{.cpp}
 * auto res   = adaptor.reserve(1 << 30, allow_overbooking::NO);
 * auto table = cudf::groupby(..., stream, res.as_device());
 * @endcode
 *
 * `as_device()`, `as_host()`, and `as_host_device()` each throw
 * `cucascade::logic_error` when the reservation's memory does not have the requested
 * accessibility. Use `accessibility()`, `is_device_accessible()`, or
 * `is_host_accessible()` to branch without exceptions.
 *
 * @par Ownership
 *
 * Copies of a reservation share the same shared state and are interchangeable. The
 * handles returned by the `as_*()` methods own a reference to that same state, so a
 * buffer allocated through one keeps the reservation alive for as long as it needs to
 * service its deallocation, even after every `memory_reservation` copy is gone.
 *
 * The unspent balance is refunded to the adaptor when the last reference dies. Reserving
 * more than is allocated therefore keeps the surplus out of circulation for as long as
 * any derived buffer lives, so reserve what you actually use.
 */
class memory_reservation {
 public:
  /**
   * @brief The accessibility of the memory this reservation draws on.
   *
   * @return The accessibility, fixed at grant time by the adaptor's upstream.
   */
  [[nodiscard]] reservation_accessibility accessibility() const noexcept;

  /**
   * @brief Whether `as_device()` will succeed.
   *
   * @return True when the reservation's memory is device accessible.
   */
  [[nodiscard]] bool is_device_accessible() const noexcept;

  /**
   * @brief Whether `as_host()` will succeed.
   *
   * @return True when the reservation's memory is host accessible.
   */
  [[nodiscard]] bool is_host_accessible() const noexcept;

  /**
   * @brief Project the reservation into a device-accessible memory resource.
   *
   * The returned handle owns a reference to the reservation's shared state, so it may
   * outlive this object.
   *
   * @return An erased resource advertising `cuda::mr::device_accessible`.
   * @throws cucascade::logic_error if the reservation's memory is not device accessible.
   */
  [[nodiscard]] any_device_resource as_device() const;

  /**
   * @brief Project the reservation into a host-accessible memory resource.
   *
   * @return An erased resource advertising `cuda::mr::host_accessible`.
   * @throws cucascade::logic_error if the reservation's memory is not host accessible.
   */
  [[nodiscard]] any_host_resource as_host() const;

  /**
   * @brief Project the reservation into a host- and device-accessible memory resource.
   *
   * @return An erased resource advertising both accessibility properties.
   * @throws cucascade::logic_error unless the reservation's memory is both host and
   * device accessible.
   */
  [[nodiscard]] any_host_device_resource as_host_device() const;

  /**
   * @brief Equality comparison.
   *
   * @param other The other reservation to compare.
   * @return True if both refer to the same reservation.
   */
  [[nodiscard]] bool operator==(memory_reservation const& other) const noexcept;

  /**
   * @brief The number of bytes originally granted.
   *
   * @return The granted size in bytes.
   */
  [[nodiscard]] std::size_t grant() const noexcept;

  /**
   * @brief The remaining unallocated size of the reservation.
   *
   * Negative on a soft reservation that has allocated past its grant, by the size of
   * the overdraft. Never negative on a strict one, which refuses those allocations.
   *
   * @return The remaining size in bytes.
   */
  [[nodiscard]] std::int64_t balance() const noexcept;

  /**
   * @brief The number of bytes by which the grant overbooks the adaptor's limit.
   *
   * Nonzero only when the reservation was granted with `allow_overbooking::YES`. The
   * caller must free at least this much memory before using the reservation.
   *
   * @return The overbooked size in bytes.
   */
  [[nodiscard]] std::size_t overbooking() const noexcept;

 private:
  template <typename Upstream>
    requires ::cuda::mr::resource<Upstream>
  friend class reservation_aware_resource_adaptor;

  /// @brief The shared state, one alternative per accessibility.
  using handle_variant = std::variant<detail::reservation_handle<device_adaptor>,
                                      detail::reservation_handle<host_adaptor>,
                                      detail::reservation_handle<host_device_adaptor>>;

  /**
   * @brief Construct from an already-granted shared state.
   *
   * Private so that only `reservation_aware_resource_adaptor` can grant reservations.
   *
   * @param handle The shared state of the granted reservation.
   */
  explicit memory_reservation(handle_variant handle) : handle_{std::move(handle)} {}

  handle_variant handle_;
};

}  // namespace experimental
}  // namespace memory
}  // namespace cucascade
