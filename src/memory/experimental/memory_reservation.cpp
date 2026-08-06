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

#include <type_traits>
#include <variant>

namespace cucascade {
namespace memory {
namespace experimental {

namespace {

template <typename Handle>
constexpr bool handle_is_device_accessible =
  ::cuda::has_property<Handle, ::cuda::mr::device_accessible>;

template <typename Handle>
constexpr bool handle_is_host_accessible =
  ::cuda::has_property<Handle, ::cuda::mr::host_accessible>;

template <typename Handle>
[[nodiscard]] constexpr reservation_accessibility accessibility_of() noexcept
{
  static_assert(handle_is_device_accessible<Handle> || handle_is_host_accessible<Handle>,
                "a reservation must be accessible from somewhere");
  if constexpr (handle_is_device_accessible<Handle> && handle_is_host_accessible<Handle>) {
    return reservation_accessibility::HOST_DEVICE;
  } else if constexpr (handle_is_device_accessible<Handle>) {
    return reservation_accessibility::DEVICE;
  } else {
    return reservation_accessibility::HOST;
  }
}

}  // namespace

// Each variant alternative must forward the accessibility of the adaptor it was granted from,
// so that the corresponding as_*() projection is well-formed.
static_assert(::cuda::mr::resource_with<detail::reservation_handle<device_adaptor>,
                                        ::cuda::mr::device_accessible>);
static_assert(
  ::cuda::mr::resource_with<detail::reservation_handle<host_adaptor>, ::cuda::mr::host_accessible>);
static_assert(::cuda::mr::resource_with<detail::reservation_handle<host_device_adaptor>,
                                        ::cuda::mr::device_accessible,
                                        ::cuda::mr::host_accessible>);

reservation_accessibility memory_reservation::accessibility() const noexcept
{
  return std::visit(
    [](auto const& handle) { return accessibility_of<std::remove_cvref_t<decltype(handle)>>(); },
    handle_);
}

bool memory_reservation::is_device_accessible() const noexcept
{
  auto const access = accessibility();
  return access == reservation_accessibility::DEVICE ||
         access == reservation_accessibility::HOST_DEVICE;
}

bool memory_reservation::is_host_accessible() const noexcept
{
  auto const access = accessibility();
  return access == reservation_accessibility::HOST ||
         access == reservation_accessibility::HOST_DEVICE;
}

any_device_resource memory_reservation::as_device() const
{
  return std::visit(
    [](auto const& handle) -> any_device_resource {
      if constexpr (handle_is_device_accessible<std::remove_cvref_t<decltype(handle)>>) {
        return any_device_resource{handle};
      } else {
        CUCASCADE_FAIL("reservation memory is not device accessible");
      }
    },
    handle_);
}

any_host_resource memory_reservation::as_host() const
{
  return std::visit(
    [](auto const& handle) -> any_host_resource {
      if constexpr (handle_is_host_accessible<std::remove_cvref_t<decltype(handle)>>) {
        return any_host_resource{handle};
      } else {
        CUCASCADE_FAIL("reservation memory is not host accessible");
      }
    },
    handle_);
}

any_host_device_resource memory_reservation::as_host_device() const
{
  return std::visit(
    [](auto const& handle) -> any_host_device_resource {
      using handle_t = std::remove_cvref_t<decltype(handle)>;
      if constexpr (handle_is_device_accessible<handle_t> && handle_is_host_accessible<handle_t>) {
        return any_host_device_resource{handle};
      } else {
        CUCASCADE_FAIL("reservation memory is not both host and device accessible");
      }
    },
    handle_);
}

bool memory_reservation::operator==(memory_reservation const& other) const noexcept
{
  return handle_ == other.handle_;
}

std::size_t memory_reservation::grant() const noexcept
{
  return std::visit(
    [](auto const& handle) { return detail::safe_cast<std::size_t>(handle->grant()); }, handle_);
}

std::int64_t memory_reservation::balance() const noexcept
{
  return std::visit([](auto const& handle) { return handle->balance(); }, handle_);
}

bool memory_reservation::is_soft() const noexcept
{
  return std::visit([](auto const& handle) { return handle->is_soft(); }, handle_);
}

std::size_t memory_reservation::overbooking() const noexcept
{
  return std::visit([](auto const& handle) { return handle->overbooking(); }, handle_);
}

}  // namespace experimental
}  // namespace memory
}  // namespace cucascade
