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

#include <cucascade/memory/experimental/reservation_aware_resource_adaptor.hpp>

#include <utility>

namespace cucascade {
namespace memory {
namespace experimental {

reservation_aware_resource_adaptor::reservation_aware_resource_adaptor(
  any_device_resource primary_mr, std::int64_t limit)
  : shared_base(::cuda::mr::make_shared_resource<impl_type>(std::move(primary_mr), limit))
{
}

std::int64_t reservation_aware_resource_adaptor::limit() const noexcept { return get().limit(); }

void reservation_aware_resource_adaptor::set_limit(std::int64_t limit) noexcept
{
  get().set_limit(limit);
}

std::int64_t reservation_aware_resource_adaptor::current_allocated() const noexcept
{
  return get().current_allocated();
}

std::int64_t reservation_aware_resource_adaptor::total_reserved() const noexcept
{
  return get().total_reserved();
}

std::int64_t reservation_aware_resource_adaptor::available() const noexcept
{
  return get().available();
}

memory_record reservation_aware_resource_adaptor::get_main_record() const
{
  return get().get_main_record();
}

rmm::device_async_resource_ref reservation_aware_resource_adaptor::get_upstream_resource()
  const noexcept
{
  return rmm::device_async_resource_ref{
    const_cast<any_device_resource&>(get().get_upstream_resource())};
}

memory_reservation::memory_reservation(reservation_aware_resource_adaptor const& adaptor,
                                       std::size_t granted,
                                       std::size_t overbooking)
  : shared_base{::cuda::mr::make_shared_resource<impl_type>(
      adaptor, detail::safe_cast<std::int64_t>(granted), overbooking)}
{
}

memory_reservation reservation_aware_resource_adaptor::reserve(std::size_t size,
                                                               allow_overbooking overbooking_policy)
{
  auto const [granted, overbooking] =
    get().reserve(size, overbooking_policy == allow_overbooking::YES);
  return memory_reservation{*this, granted, overbooking};
}

}  // namespace experimental
}  // namespace memory
}  // namespace cucascade
