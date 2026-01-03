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

#include "memory/memory_reservation_manager.hpp"

#include "memory/common.hpp"
#include "memory/memory_reservation.hpp"
#include "memory/memory_space.hpp"
#include "memory/numa_region_pinned_host_allocator.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <stdexcept>

namespace cucascade {
namespace memory {

//===----------------------------------------------------------------------===//
// Reservation Strategy Implementations
//===----------------------------------------------------------------------===//

std::span<memory_space*> reservation_request_strategy::get_all_memory_resource(
  memory_reservation_manager& manager)
{
  return manager.get_mutable_all_memory_spaces();
}

std::span<memory_space*> reservation_request_strategy::get_all_memory_resource(
  memory_reservation_manager& manager, Tier tier)
{
  return manager.get_mutable_memory_spaces_for_tier(tier);
}

memory_space* reservation_request_strategy::get_memory_resource(memory_reservation_manager& manager,
                                                                memory_space_id source_id)
{
  return manager.get_mutable_memory_space(source_id.tier, source_id.device_id);
}

std::vector<memory_space*> reservation_request_strategy::get_memory_resource(
  memory_reservation_manager& manager, std::span<memory_space_id> source_ids)
{
  return manager.get_mutable_memory_space(source_ids);
}

std::vector<memory_space*> any_memory_space_in_tier_with_preference::get_candidates(
  memory_reservation_manager& manager) const
{
  auto cs = this->get_all_memory_resource(manager, tier);
  std::vector<memory_space*> candidates{cs.begin(), cs.end()};
  if (preferred_device_id.has_value()) {
    std::stable_partition(candidates.begin(),
                          candidates.end(),
                          [device_id = static_cast<int>(preferred_device_id.value())](
                            const auto* ms) { return ms->get_device_id() == device_id; });
  }
  return candidates;
}

std::vector<memory_space*> any_memory_space_in_tier::get_candidates(
  memory_reservation_manager& manager) const
{
  auto cs = this->get_all_memory_resource(manager, tier);
  return {cs.begin(), cs.end()};
}

std::vector<memory_space*> any_memory_space_in_tiers::get_candidates(
  memory_reservation_manager& manager) const
{
  auto cs = this->get_all_memory_resource(manager);
  std::vector<memory_space*> candidates;
  for (Tier t : tiers) {
    for (auto candidate : this->get_all_memory_resource(manager, t)) {
      candidates.emplace_back(candidate);
    }
  }
  return candidates;
}

std::vector<memory_space*> specific_memory_space::get_candidates(
  memory_reservation_manager& manager) const
{
  auto* ms = this->get_memory_resource(manager, target_id);
  if (ms) { return {ms}; }
  return {};
}

std::vector<memory_space*> any_memory_space_to_downgrade::get_candidates(
  [[maybe_unused]] memory_reservation_manager& manager) const
{
  return {};
}

std::vector<memory_space*> any_memory_space_to_upgrade::get_candidates(
  [[maybe_unused]] memory_reservation_manager& manager) const
{
  return {};
}

//===----------------------------------------------------------------------===//
// memory_reservation_manager Implementation
//===----------------------------------------------------------------------===//

memory_reservation_manager::memory_reservation_manager(std::vector<memory_space_config> configs)
{
  if (configs.empty()) {
    throw std::invalid_argument("At least one memory_space configuration must be provided");
  }

  // Create memory_space instances
  for (auto& config : configs) {
    // Move the allocators from config to the memory_space
    auto mem_space = std::make_unique<memory_space>(
      config.tier,
      config.device_id,
      config.memory_limit,
      static_cast<std::size_t>(config.downgrade_tigger_threshold *
                               static_cast<float>(config.memory_capacity)),
      static_cast<std::size_t>(config.downgrade_stop_threshold *
                               static_cast<float>(config.memory_capacity)),
      config.memory_capacity,
      config.mr_factory_fn(config.device_id, config.memory_capacity));
    _memory_spaces.push_back(std::move(mem_space));
  }

  // Build lookup tables
  build_lookup_tables();
}

memory_reservation_manager::~memory_reservation_manager() { shutdown(); }

std::unique_ptr<reservation> memory_reservation_manager::request_reservation(
  const reservation_request_strategy& request, size_t size)
{
  // Fast path: try to make a reservation immediately
  if (auto res = select_memory_space_and_make_reservation(request, size); res.has_value()) {
    return std::move(res.value());
  }

  // If none available, block until any memory_space can satisfy the request
  std::unique_lock<std::mutex> lock(_wait_mutex);
  for (;;) {
    if (auto res = select_memory_space_and_make_reservation(request, size); res.has_value()) {
      // Release the wait lock before returning the reservation
      lock.unlock();
      return std::move(res.value());
    }
    // Wait until notified that memory may be available again
    _wait_cv.wait(lock);
  }
}

const memory_space* memory_reservation_manager::get_memory_space(Tier tier, int32_t device_id) const
{
  memory_space_id id(tier, device_id);
  auto it = _memory_space_lookup.find(id);
  return (it != _memory_space_lookup.end()) ? it->second : nullptr;
}

std::span<const memory_space*> memory_reservation_manager::get_memory_spaces_for_tier(
  Tier tier) const
{
  auto it = _const_tier_to_memory_spaces.find(tier);
  if (it != _const_tier_to_memory_spaces.end()) {
    return const_cast<std::vector<const memory_space*>&>(it->second);
  }
  return {};
}

std::span<const memory_space*> memory_reservation_manager::get_all_memory_spaces() const noexcept
{
  return const_cast<std::vector<const memory_space*>&>(_const_memory_space_views);
}

memory_space* memory_reservation_manager::get_mutable_memory_space(Tier tier, int32_t device_id)
{
  memory_space_id id(tier, device_id);
  auto it = _memory_space_lookup.find(id);
  return (it != _memory_space_lookup.end()) ? it->second : nullptr;
}

std::vector<memory_space*> memory_reservation_manager::get_mutable_memory_space(
  std::span<memory_space_id> ids)
{
  std::vector<memory_space*> spaces;
  for (const auto& id : ids) {
    auto* ms = get_mutable_memory_space(id.tier, id.device_id);
    if (ms) { spaces.push_back(ms); }
  }
  return spaces;
}

std::span<memory_space*> memory_reservation_manager::get_mutable_memory_spaces_for_tier(Tier tier)
{
  auto it = _tier_to_memory_spaces.find(tier);
  if (it != _tier_to_memory_spaces.end()) { return it->second; }
  return {};
}

std::span<memory_space*> memory_reservation_manager::get_mutable_all_memory_spaces() noexcept
{
  return _memory_space_views;
}

size_t memory_reservation_manager::get_available_memory_for_tier(Tier tier) const
{
  size_t total_available = 0;
  auto spaces            = get_memory_spaces_for_tier(tier);

  for (const auto* space : spaces) {
    total_available += space->get_available_memory();
  }

  return total_available;
}

size_t memory_reservation_manager::get_total_reserved_memory_for_tier(Tier tier) const
{
  size_t total_reserved = 0;
  auto spaces           = get_memory_spaces_for_tier(tier);

  for (const auto* space : spaces) {
    total_reserved += space->get_total_reserved_memory();
  }

  return total_reserved;
}

size_t memory_reservation_manager::get_active_reservation_count_for_tier(Tier tier) const
{
  size_t total_count = 0;
  auto spaces        = get_memory_spaces_for_tier(tier);

  for (const auto* space : spaces) {
    total_count += space->get_active_reservation_count();
  }

  return total_count;
}

size_t memory_reservation_manager::get_total_available_memory() const
{
  size_t total = 0;
  for (const auto& space : _memory_spaces) {
    total += space->get_available_memory();
  }
  return total;
}

size_t memory_reservation_manager::get_total_reserved_memory() const
{
  size_t total = 0;
  for (const auto& space : _memory_spaces) {
    total += space->get_total_reserved_memory();
  }
  return total;
}

size_t memory_reservation_manager::get_active_reservation_count() const
{
  size_t total = 0;
  for (const auto& space : _memory_spaces) {
    total += space->get_active_reservation_count();
  }
  return total;
}

std::optional<std::unique_ptr<reservation>>
memory_reservation_manager::select_memory_space_and_make_reservation(
  const reservation_request_strategy& request, size_t size)
{
  using ReserveFnPtr = std::unique_ptr<reservation> (memory_space::*)(size_t);

  auto try_candidates = [](std::span<memory_space*> candidates,
                           ReserveFnPtr res_fn,
                           size_t size) -> std::optional<std::unique_ptr<reservation>> {
    for (memory_space* space : candidates) {
      if (space) {
        if (auto res = std::invoke(res_fn, space, size)) { return res; }
      }
    }
    return std::nullopt;
  };

  bool has_strong_ordering = request.has_strong_ordering();
  auto candidates          = request.get_candidates(*this);
  if (has_strong_ordering) {
    return try_candidates(candidates, &memory_space::make_reservation, size);
  } else {
    if (auto res = try_candidates(candidates, &memory_space::make_reservation_or_null, size)) {
      return res;
    }
    return try_candidates(candidates, &memory_space::make_reservation, size);
  }
}

void memory_reservation_manager::build_lookup_tables()
{
  _memory_space_lookup.clear();
  _tier_to_memory_spaces.clear();
  _const_tier_to_memory_spaces.clear();
  _memory_space_views.clear();
  _const_memory_space_views.clear();

  for (const auto& space : _memory_spaces) {
    memory_space* space_ptr = space.get();
    _memory_space_views.push_back(space_ptr);
    _const_memory_space_views.push_back(space_ptr);

    // Build direct lookup table
    _memory_space_lookup[space_ptr->get_id()] = space_ptr;

    // Build tier-to-spaces mapping
    _tier_to_memory_spaces[space_ptr->get_tier()].push_back(space_ptr);
    _const_tier_to_memory_spaces[space_ptr->get_tier()].push_back(space_ptr);
  }
}

void memory_reservation_manager::shutdown()
{
  for (const auto& space : _memory_spaces) {
    space->shutdown();
  }
}

}  // namespace memory
}  // namespace cucascade
