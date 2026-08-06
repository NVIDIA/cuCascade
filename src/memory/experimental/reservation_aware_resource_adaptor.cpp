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

#include <cucascade/memory/experimental/memory_reservation.hpp>

#include <utility>

namespace cucascade {
namespace memory {
namespace experimental {

template <typename Upstream>
  requires ::cuda::mr::resource<Upstream>
reservation_aware_resource_adaptor<Upstream>::reservation_aware_resource_adaptor(
  Upstream primary_mr, std::int64_t limit)
  : shared_base(::cuda::mr::make_shared_resource<impl_type>(std::move(primary_mr), limit))
{
}

template <typename Upstream>
  requires ::cuda::mr::resource<Upstream>
std::int64_t reservation_aware_resource_adaptor<Upstream>::limit() const noexcept
{
  return this->get().limit();
}

template <typename Upstream>
  requires ::cuda::mr::resource<Upstream>
void reservation_aware_resource_adaptor<Upstream>::set_limit(std::int64_t limit) noexcept
{
  this->get().set_limit(limit);
}

template <typename Upstream>
  requires ::cuda::mr::resource<Upstream>
std::int64_t reservation_aware_resource_adaptor<Upstream>::current_allocated() const noexcept
{
  return this->get().current_allocated();
}

template <typename Upstream>
  requires ::cuda::mr::resource<Upstream>
std::int64_t reservation_aware_resource_adaptor<Upstream>::total_reserved() const noexcept
{
  return this->get().total_reserved();
}

template <typename Upstream>
  requires ::cuda::mr::resource<Upstream>
std::int64_t reservation_aware_resource_adaptor<Upstream>::available() const noexcept
{
  return this->get().available();
}

template <typename Upstream>
  requires ::cuda::mr::resource<Upstream>
memory_record reservation_aware_resource_adaptor<Upstream>::get_main_record() const
{
  return this->get().get_main_record();
}

template <typename Upstream>
  requires ::cuda::mr::resource<Upstream>
Upstream const& reservation_aware_resource_adaptor<Upstream>::get_upstream_resource() const noexcept
{
  return this->get().upstream_resource();
}

template <typename Upstream>
  requires ::cuda::mr::resource<Upstream>
memory_reservation reservation_aware_resource_adaptor<Upstream>::reserve(
  std::size_t size, allow_overbooking overbooking_policy)
{
  auto const [granted, overbooking] =
    this->get().reserve(size, overbooking_policy == allow_overbooking::YES);

  using impl_t = detail::memory_reservation_impl<reservation_aware_resource_adaptor<Upstream>>;
  return memory_reservation{
    memory_reservation::handle_variant{::cuda::mr::make_shared_resource<impl_t>(
      *this, detail::safe_cast<std::int64_t>(granted), overbooking, grant_enforcement::STRICT)}};
}

template <typename Upstream>
  requires ::cuda::mr::resource<Upstream>
memory_reservation reservation_aware_resource_adaptor<Upstream>::reserve_soft(
  std::size_t size, allow_overbooking overbooking_policy)
{
  auto const [granted, overbooking] =
    this->get().reserve(size, overbooking_policy == allow_overbooking::YES);

  using impl_t = detail::memory_reservation_impl<reservation_aware_resource_adaptor<Upstream>>;
  return memory_reservation{
    memory_reservation::handle_variant{::cuda::mr::make_shared_resource<impl_t>(
      *this, detail::safe_cast<std::int64_t>(granted), overbooking, grant_enforcement::SOFT)}};
}

template class reservation_aware_resource_adaptor<any_device_resource>;
template class reservation_aware_resource_adaptor<any_host_resource>;
template class reservation_aware_resource_adaptor<any_host_device_resource>;

// Each adaptor must forward the accessibility of the upstream it was instantiated with.
static_assert(::cuda::mr::resource_with<device_adaptor, ::cuda::mr::device_accessible>);
static_assert(::cuda::mr::resource_with<host_adaptor, ::cuda::mr::host_accessible>);
static_assert(::cuda::mr::resource_with<host_device_adaptor,
                                        ::cuda::mr::device_accessible,
                                        ::cuda::mr::host_accessible>);

}  // namespace experimental
}  // namespace memory
}  // namespace cucascade
