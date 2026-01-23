/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/disk_access_limiter.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/null_device_memory_resource.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <cucascade/utils/overloaded.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <variant>

namespace cucascade {
namespace memory {

//===----------------------------------------------------------------------===//
// memory_space Implementation
//===----------------------------------------------------------------------===//

memory_space::memory_space(const gpu_memory_space_config& config)
  : _id(config.tier(), config.device_id),
    _capacity(config.memory_capacity),
    _memory_limit(config.reservation_limit()),
    _start_downgrading_memory_threshold(config.downgrade_trigger_threshold()),
    _stop_downgrading_memory_threshold(config.downgrade_stop_threshold()),
    _allocator(config.mr_factory_fn
                 ? config.mr_factory_fn(config.device_id, config.memory_capacity)
                 : make_default_gpu_memory_resource(config.device_id, config.memory_capacity)),
    _stream_pool{[&]() -> std::unique_ptr<rmm::cuda_stream_pool> {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id(config.device_id)};
      return std::make_unique<rmm::cuda_stream_pool>(16);
    }()}
{
  if (!_allocator) { throw std::invalid_argument("At least one allocator must be provided"); }

  _reservation_allocator = std::make_unique<reservation_aware_resource_adaptor>(
    _id,
    *_allocator,
    _memory_limit,
    _capacity,
    nullptr,
    nullptr,
    config.per_stream_reservation
      ? reservation_aware_resource_adaptor::AllocationTrackingScope::PER_STREAM
      : reservation_aware_resource_adaptor::AllocationTrackingScope::PER_THREAD);
}

memory_space::memory_space(const host_memory_space_config& config)
  : _id(config.tier(), config.numa_id),
    _capacity(config.memory_capacity),
    _memory_limit(config.reservation_limit()),
    _start_downgrading_memory_threshold(config.downgrade_trigger_threshold()),
    _stop_downgrading_memory_threshold(config.downgrade_stop_threshold()),
    _allocator(config.mr_factory_fn
                 ? config.mr_factory_fn(config.numa_id, config.memory_capacity)
                 : make_default_host_memory_resource(config.numa_id, config.memory_capacity))
{
  if (!_allocator) { throw std::invalid_argument("At least one allocator must be provided"); }
  _reservation_allocator =
    std::make_unique<fixed_size_host_memory_resource>(_id.device_id,
                                                      *_allocator,
                                                      _memory_limit,
                                                      _capacity,
                                                      config.block_size,
                                                      config.pool_size,
                                                      config.initial_number_pools);
}

memory_space::memory_space(const disk_memory_space_config& config)
  : _id(config.tier(), config.disk_id),
    _capacity(config.memory_capacity),
    _memory_limit(config.reservation_limit()),
    _start_downgrading_memory_threshold(config.downgrade_trigger_threshold()),
    _stop_downgrading_memory_threshold(config.downgrade_stop_threshold()),
    _allocator(std::make_unique<null_device_memory_resource>())
{
  if (config.mount_paths.empty()) {
    throw std::invalid_argument("Mount path must be provided for disk memory space");
  }
  _reservation_allocator =
    std::make_unique<disk_access_limiter>(_id, _memory_limit, _capacity, config.mount_paths);
}

memory_space::~memory_space() = default;

bool memory_space::operator==(const memory_space& other) const { return _id == other.get_id(); }

bool memory_space::operator!=(const memory_space& other) const { return !(*this == other); }

memory_space_id memory_space::get_id() const noexcept { return _id; }

Tier memory_space::get_tier() const noexcept { return _id.tier; }

int memory_space::get_device_id() const noexcept { return _id.device_id; }

std::unique_ptr<reservation> memory_space::make_reservation_or_null(size_t size)
{
  std::unique_ptr<reserved_arena> arena =
    std::visit(utils::overloaded{[&](std::unique_ptr<disk_access_limiter>& mr) {
                                   return mr->reserve(size, _notification_channel->get_notifier());
                                 },
                                 [&](std::unique_ptr<reservation_aware_resource_adaptor>& mr) {
                                   return mr->reserve(size, _notification_channel->get_notifier());
                                 },
                                 [&](std::unique_ptr<fixed_size_host_memory_resource>& mr) {
                                   return mr->reserve(size, _notification_channel->get_notifier());
                                 }},
               _reservation_allocator);
  return reservation::create(*this, std::move(arena));
}

std::unique_ptr<reservation> memory_space::make_reservation_upto(size_t size)
{
  std::unique_ptr<reserved_arena> arena = std::visit(
    utils::overloaded{[&](std::unique_ptr<disk_access_limiter>& mr) {
                        return mr->reserve_upto(size, _notification_channel->get_notifier());
                      },
                      [&](std::unique_ptr<reservation_aware_resource_adaptor>& mr) {
                        return mr->reserve_upto(size, _notification_channel->get_notifier());
                      },
                      [&](std::unique_ptr<fixed_size_host_memory_resource>& mr) {
                        return mr->reserve_upto(size, _notification_channel->get_notifier());
                      }},
    _reservation_allocator);
  return reservation::create(*this, std::move(arena));
}

std::unique_ptr<reservation> memory_space::make_reservation(size_t size)
{
  std::unique_ptr<reservation> res = make_reservation_or_null(size);
  while (!res) {
    auto status = _notification_channel->wait();
    if (status == notification_channel::wait_status::SHUTDOWN) { return nullptr; }
    if (status == notification_channel::wait_status::IDLE) { return make_reservation_upto(size); }
    res = make_reservation_or_null(size);
  }
  return res;
}

rmm::cuda_stream_view memory_space::acquire_stream() const
{
  if (!_stream_pool) {
    throw std::runtime_error("Stream pool is not available for non-GPU memory spaces");
  }
  return _stream_pool->get_stream();
}

std::size_t memory_space::get_active_reservation_count() const
{
  return std::visit(
    utils::overloaded{[&](const std::unique_ptr<disk_access_limiter>& mr) {
                        return mr->get_active_reservation_count();
                      },
                      [&](const std::unique_ptr<reservation_aware_resource_adaptor>& mr) {
                        return mr->get_active_reservation_count();
                      },
                      [&](const std::unique_ptr<fixed_size_host_memory_resource>& mr) {
                        return mr->get_active_reservation_count();
                      }},
    _reservation_allocator);
}

bool memory_space::should_downgrade_memory() const
{
  return _memory_limit - get_available_memory() >= _start_downgrading_memory_threshold;
}

bool memory_space::should_stop_downgrading_memory() const
{
  return _memory_limit - get_available_memory() <= _stop_downgrading_memory_threshold;
}

size_t memory_space::get_amount_to_downgrade() const
{
  size_t consumed = _memory_limit - get_available_memory();
  if (consumed <= _stop_downgrading_memory_threshold) { return 0; }
  return consumed - _stop_downgrading_memory_threshold;
}

size_t memory_space::get_available_memory(rmm::cuda_stream_view stream) const
{
  return std::visit(
    utils::overloaded{
      [&](const std::unique_ptr<disk_access_limiter>& mr) { return mr->get_available_memory(); },
      [&](const std::unique_ptr<reservation_aware_resource_adaptor>& mr) {
        return mr->get_available_memory(stream);
      },
      [&](const std::unique_ptr<fixed_size_host_memory_resource>& mr) {
        return mr->get_available_memory();
      }},
    _reservation_allocator);
}

size_t memory_space::get_available_memory() const
{
  return std::visit(
    utils::overloaded{
      [&](const std::unique_ptr<disk_access_limiter>& mr) { return mr->get_available_memory(); },
      [](const std::unique_ptr<reservation_aware_resource_adaptor>& mr) {
        return mr->get_available_memory();
      },
      [](const std::unique_ptr<fixed_size_host_memory_resource>& mr) {
        return mr->get_available_memory();
      }},
    _reservation_allocator);
}

size_t memory_space::get_total_reserved_memory() const
{
  return std::visit(
    utils::overloaded{[&](const std::unique_ptr<disk_access_limiter>& mr) {
                        return mr->get_total_reserved_bytes();
                      },
                      [](const std::unique_ptr<reservation_aware_resource_adaptor>& mr) {
                        return mr->get_total_reserved_bytes();
                      },
                      [](const std::unique_ptr<fixed_size_host_memory_resource>& mr) {
                        return mr->get_total_reserved_bytes();
                      }},
    _reservation_allocator);
}

size_t memory_space::get_max_memory() const noexcept { return _memory_limit; }

rmm::mr::device_memory_resource* memory_space::get_default_allocator() const noexcept
{
  return std::visit(
    utils::overloaded{[this]([[maybe_unused]] const std::unique_ptr<disk_access_limiter>& other)
                        -> rmm::mr::device_memory_resource* { return _allocator.get(); },
                      [](const std::unique_ptr<reservation_aware_resource_adaptor>& mr)
                        -> rmm::mr::device_memory_resource* { return mr.get(); },
                      [](const std::unique_ptr<fixed_size_host_memory_resource>& mr)
                        -> rmm::mr::device_memory_resource* { return mr.get(); }},
    _reservation_allocator);
}

std::string memory_space::to_string() const
{
  std::ostringstream oss;
  oss << "memory_space(tier=";
  switch (_id.tier) {
    case Tier::GPU: oss << "GPU"; break;
    case Tier::HOST: oss << "HOST"; break;
    case Tier::DISK: oss << "DISK"; break;
    default: oss << "UNKNOWN"; break;
  }
  oss << ", device_id=" << _id.device_id << ", limit=" << _memory_limit << ")";
  return oss.str();
}

void memory_space::shutdown()
{
  if (_notification_channel) { _notification_channel->shutdown(); }
}

//===----------------------------------------------------------------------===//
// memory_space_hash Implementation
//===----------------------------------------------------------------------===//

size_t memory_space_hash::operator()(const memory_space& ms) const
{
  return std::hash<int>{}(static_cast<int>(ms.get_tier())) ^
         (std::hash<size_t>{}(static_cast<size_t>(ms.get_device_id())) << 1);
}

}  // namespace memory
}  // namespace cucascade
