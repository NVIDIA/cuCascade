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

#include <cucascade/error.hpp>
#include <cucascade/memory/experimental/over_reservation_policy.hpp>

#include <rmm/error.hpp>

namespace cucascade {
namespace memory {
namespace experimental {

over_reservation_policy::~over_reservation_policy() = default;

throw_on_over_reservation::throw_on_over_reservation() = default;

void throw_on_over_reservation::handle_over_reservation(std::int64_t requested_bytes,
                                                        std::int64_t observed_balance,
                                                        reservation_control& reservation) const
{
  CUCASCADE_FAIL("allocation of " + std::to_string(requested_bytes) +
                   " bytes exceeds reservation (grant: " + std::to_string(reservation.grant()) +
                   ", remaining: " + std::to_string(observed_balance) + ")",
                 rmm::out_of_memory);
}

ignore_on_over_reservation::ignore_on_over_reservation() = default;

void ignore_on_over_reservation::handle_over_reservation(
  [[maybe_unused]] std::int64_t requested_bytes,
  [[maybe_unused]] std::int64_t observed_balance,
  [[maybe_unused]] reservation_control& reservation) const
{
  // Allow the pending allocation to proceed; the caller subtracts from balance.
}

namespace {

std::shared_ptr<const over_reservation_policy> const make_hard_policy()
{
  static auto const policy = std::make_shared<const throw_on_over_reservation>();
  return policy;
}

std::shared_ptr<const over_reservation_policy> const make_soft_policy()
{
  static auto const policy = std::make_shared<const ignore_on_over_reservation>();
  return policy;
}

}  // namespace

std::shared_ptr<const over_reservation_policy> const& thow_on_over_reservation_instance()
{
  static auto const policy = make_hard_policy();
  return policy;
}

std::shared_ptr<const over_reservation_policy> const& ignore_on_over_reservation_instance()
{
  static auto const policy = make_soft_policy();
  return policy;
}

}  // namespace experimental
}  // namespace memory
}  // namespace cucascade
