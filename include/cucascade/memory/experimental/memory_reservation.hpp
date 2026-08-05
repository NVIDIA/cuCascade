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
#include <string>
#include <utility>

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
 *
 * @tparam Adaptor Always `reservation_aware_resource_adaptor`.
 *
 * The adaptor is a template parameter only so that the reservation can hold one by
 * value: `reservation_aware_resource_adaptor` is defined in terms of its impl, so it is
 * still incomplete at the friend declaration site, and only a dependent member type
 * defers the completeness requirement to instantiation time.
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
