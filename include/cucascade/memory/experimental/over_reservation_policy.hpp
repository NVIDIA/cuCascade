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

#include <cstddef>
#include <cstdint>
#include <memory>

namespace cucascade {
namespace memory {
namespace experimental {

/**
 * @brief Narrow interface to reservation state and accounting operations.
 *
 * Implemented by the shared reservation state so policies can inspect the reservation
 * without depending on the adaptor's accessibility. Future thread-safe grow/shrink
 * operations belong on this interface because their implementations must coordinate
 * reservation state with the granting adaptor's accounting.
 */
class reservation_control {
 public:
  virtual ~reservation_control() = default;

  [[nodiscard]] virtual std::int64_t grant() const noexcept      = 0;
  [[nodiscard]] virtual std::int64_t balance() const noexcept    = 0;
  [[nodiscard]] virtual std::size_t overbooking() const noexcept = 0;
};

/**
 * @brief Pluggable policy invoked when an allocation would exceed the reservation balance.
 *
 * Called only after observing `requested_bytes > balance`. Returning normally authorizes
 * the pending allocation; the allocation path then atomically subtracts the request from
 * the current balance. Throwing `rmm::out_of_memory` rejects it.
 */
class over_reservation_policy {
 public:
  virtual ~over_reservation_policy();

  /**
   * @brief Handle an allocation that would exceed the reservation's remaining balance.
   *
   * @param requested_bytes Number of bytes being requested.
   * @param observed_balance Remaining balance observed before invoking the policy.
   * @param reservation The reservation whose balance would be exceeded.
   * @throws rmm::out_of_memory if the policy decides to reject the allocation.
   */
  virtual void handle_over_reservation(std::int64_t requested_bytes,
                                       std::int64_t observed_balance,
                                       reservation_control& reservation) const = 0;
};

/**
 * @brief Hard policy — throws when the grant would be exceeded.
 */
class throw_on_over_reservation : public over_reservation_policy {
 public:
  throw_on_over_reservation();

  void handle_over_reservation(std::int64_t requested_bytes,
                               std::int64_t observed_balance,
                               reservation_control& reservation) const final;
};

/**
 * @brief Soft policy — allows the allocation to proceed past the grant.
 */
class ignore_on_over_reservation : public over_reservation_policy {
 public:
  ignore_on_over_reservation();

  void handle_over_reservation(std::int64_t requested_bytes,
                               std::int64_t observed_balance,
                               reservation_control& reservation) const final;
};

/**
 * @brief Shared stateless hard policy instance for `reserve()`.
 */
[[nodiscard]] std::shared_ptr<const over_reservation_policy> const&
thow_on_over_reservation_instance();

/**
 * @brief Shared stateless soft policy instance used by default reservations.
 */
[[nodiscard]] std::shared_ptr<const over_reservation_policy> const&
ignore_on_over_reservation_instance();

}  // namespace experimental
}  // namespace memory
}  // namespace cucascade
