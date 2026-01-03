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

#include "memory/reservation_manager_configurator.hpp"

#include "memory/common.hpp"
#include "memory/topology_discovery.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <numa.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <set>
#include <variant>
#include <vector>

namespace cucascade {
namespace memory {

using builder_reference = reservation_manager_configurator::builder_reference;

builder_reference& reservation_manager_configurator::set_number_of_gpus(std::size_t n_gpus)
{
  assert(n_gpus > 0 && "Number of GPUs must be positive");
  _n_gpus_or_gpu_ids = n_gpus;
  return *this;
}

builder_reference& reservation_manager_configurator::set_gpu_ids(std::vector<int> gpu_ids)
{
  assert(!gpu_ids.empty() && "GPU IDs list cannot be empty");
  _n_gpus_or_gpu_ids = std::move(gpu_ids);
  return *this;
}

builder_reference& reservation_manager_configurator::set_device_tier_id_to_gpu_id_map(
  const std::unordered_map<int, int>& device_to_gpu_map)
{
  assert(!device_to_gpu_map.empty() && "GPU IDs list cannot be empty");
  _n_gpus_or_gpu_ids = device_to_gpu_map;
  return *this;
}

builder_reference& reservation_manager_configurator::set_gpu_usage_limit(std::size_t bytes)
{
  _gpu_usage_limit_or_ratio = bytes;
  return *this;
}

builder_reference& reservation_manager_configurator::set_usage_limit_ratio_per_gpu(double fraction)
{
  assert(fraction > 0.0 && fraction <= 1.0 && "Usage limit ratio must be in (0.0, 1.0]");
  _gpu_usage_limit_or_ratio = fraction;
  return *this;
}

builder_reference& reservation_manager_configurator::set_reservation_limit_ratio_per_gpu(
  double fraction)
{
  assert(fraction > 0.0 && fraction <= 1.0 && "Reservation limit ratio must be in (0.0, 1.0]");
  _gpu_reservation_limit_ratio = fraction;
  return *this;
}

builder_reference& reservation_manager_configurator::set_capacity_per_numa_node(std::size_t bytes)
{
  assert(bytes > 0 && "Capacity per NUMA node must be positive");
  _capacity_per_numa_node = bytes;
  return *this;
}

builder_reference& reservation_manager_configurator::set_reservation_limit_ratio_per_numa_node(
  double fraction)
{
  assert(fraction > 0.0 && fraction <= 1.0 && "Reservation limit ratio must be in (0.0, 1.0]");
  _cpu_reservation_limit_ratio = fraction;
  return *this;
}

builder_reference& reservation_manager_configurator::set_gpu_memory_resource_factory(
  DeviceMemoryResourceFactoryFn mr_fn)
{
  assert(mr_fn && "GPU memory resource factory cannot be nullptr");
  _gpu_mr_fn = std::move(mr_fn);
  return *this;
}

builder_reference& reservation_manager_configurator::set_cpu_memory_resource_factory(
  DeviceMemoryResourceFactoryFn mr_fn)
{
  assert(mr_fn && "CPU memory resource factory cannot be nullptr");
  _cpu_mr_fn = std::move(mr_fn);
  return *this;
}

builder_reference& reservation_manager_configurator::set_numa_ids(const std::vector<int>& numa_ids)
{
  assert(!numa_ids.empty() && "NUMA IDs list cannot be empty");
  _auto_binding_or_numa_ids = std::move(numa_ids);
  return *this;
}

builder_reference& reservation_manager_configurator::set_host_ids(const std::vector<int>& host_ids)
{
  assert(!host_ids.empty() && "NUMA IDs list cannot be empty");
  std::list<int> ids(host_ids.begin(), host_ids.end());
  _auto_binding_or_numa_ids = std::move(ids);
  return *this;
}

builder_reference& reservation_manager_configurator::use_gpu_ids_as_host()
{
  _auto_binding_or_numa_ids = same_ids_tag{};
  return *this;
}

builder_reference& reservation_manager_configurator::bind_cpu_tier_to_gpus()
{
  _auto_binding_or_numa_ids = bind_cpu_to_gpu{};
  return *this;
}

builder_reference& reservation_manager_configurator::ignore_topology()
{
  _ignore_topology = true;
  return *this;
}

std::vector<memory_space_config> reservation_manager_configurator::build(
  const system_topology_info& topology) const
{
  return build(_ignore_topology ? nullptr : &topology);
}

std::vector<memory_space_config> reservation_manager_configurator::build_with_topology() const
{
  topology_discovery discovery;
  [[maybe_unused]] bool status = discovery.discover();
  assert(status);
  auto& topology = discovery.get_topology();
  return build(&topology);
}

std::vector<memory_space_config> reservation_manager_configurator::build() const
{
  return build(nullptr);
}

std::vector<memory_space_config> reservation_manager_configurator::build(
  const system_topology_info* topology) const
{
  auto [gpu_ids, tier_ids]  = extract_gpu_ids(topology);
  auto gpu_memory_threshold = extract_gpu_memory_thresholds(gpu_ids, topology);

  std::vector<memory_space_config> configs;
  for (std::size_t index = 0; index < gpu_ids.size(); ++index) {
    int tier_id = tier_ids[index];
    configs.emplace_back(Tier::GPU,
                         tier_id,
                         gpu_memory_threshold[index].first,
                         gpu_memory_threshold[index].second,
                         _gpu_mr_fn);
  };

  std::vector<int> host_numa_ids = extract_host_ids(gpu_ids, topology);
  for (int numa_id : host_numa_ids) {
    configs.emplace_back(Tier::HOST,
                         numa_id,
                         static_cast<std::size_t>(static_cast<double>(_capacity_per_numa_node) *
                                                  _cpu_reservation_limit_ratio),
                         _capacity_per_numa_node,
                         _cpu_mr_fn);
  }

  return configs;
}

std::pair<std::vector<int>, std::vector<int>> reservation_manager_configurator::extract_gpu_ids(
  [[maybe_unused]] const system_topology_info* topology) const
{
  std::vector<int> gpu_ids;
  if (std::holds_alternative<std::size_t>(_n_gpus_or_gpu_ids)) {
    std::size_t n_gpus = std::get<std::size_t>(_n_gpus_or_gpu_ids);
    assert((n_gpus <= topology->num_gpus) && "Requested number of GPUs exceeds available GPUs");
    gpu_ids.resize(n_gpus);
    for (std::size_t gpu_id = 0; gpu_id != n_gpus; ++gpu_id) {
      gpu_ids.push_back(static_cast<int>(gpu_id));
    }
    return {gpu_ids, gpu_ids};
  } else if (std::holds_alternative<std::unordered_map<int, int>>(_n_gpus_or_gpu_ids)) {
    const auto& device_to_gpu_map = std::get<std::unordered_map<int, int>>(_n_gpus_or_gpu_ids);
    std::vector<int> tier_ids;
    for (const auto& [device_tier_id, gpu_id] : device_to_gpu_map) {
      assert(gpu_id >= 0 && (gpu_id < static_cast<int>(topology->num_gpus)) &&
             "GPU ID out of range");

      tier_ids.push_back(device_tier_id);
      gpu_ids.push_back(gpu_id);
      _gpu_mr_fn = [device_to_gpu_map, current_mr_fn = _gpu_mr_fn](
                     int tier_id,
                     size_t capacity) -> std::unique_ptr<rmm::mr::device_memory_resource> {
        auto it = device_to_gpu_map.find(tier_id);
        if (it == device_to_gpu_map.end()) {
          throw std::runtime_error("GPU ID not found in device to GPU map");
        }
        int gpu_id = it->second;
        return current_mr_fn(gpu_id, capacity);
      };
    };
    return {gpu_ids, tier_ids};
  } else {
    gpu_ids = std::get<std::vector<int>>(_n_gpus_or_gpu_ids);
    for ([[maybe_unused]] int gpu_id : gpu_ids) {
      assert(gpu_id >= 0 && (gpu_id < static_cast<int>(topology->num_gpus)) &&
             "GPU ID out of range");
    }
    return {gpu_ids, gpu_ids};
  }
}

std::vector<std::pair<size_t, size_t>>
reservation_manager_configurator::extract_gpu_memory_thresholds(
  const std::vector<int>& gpu_ids, [[maybe_unused]] const system_topology_info* topology) const
{
  std::vector<std::pair<size_t, size_t>> ts;
  for (int gpu_id : gpu_ids) {
    rmm::cuda_set_device_raii set_device(rmm::cuda_device_id{gpu_id});
    auto const [free, total] = rmm::available_device_memory();
    std::size_t capacity     = free;
    if (std::holds_alternative<std::size_t>(_gpu_usage_limit_or_ratio)) {
      capacity = std::get<std::size_t>(_gpu_usage_limit_or_ratio);
      assert(capacity <= total && "GPU usage limit cannot exceed total device memory");
    } else {
      capacity = static_cast<std::size_t>(std::get<double>(_gpu_usage_limit_or_ratio) *
                                          static_cast<double>(total));
    }
    ts.emplace_back(
      static_cast<std::size_t>(static_cast<double>(capacity) * _gpu_reservation_limit_ratio),
      capacity);
  };
  return ts;
}

std::vector<int> reservation_manager_configurator::extract_host_ids(
  const std::vector<int>& gpu_ids, const system_topology_info* topology) const
{
  std::vector<int> host_numa_ids;
  if (std::holds_alternative<bind_cpu_to_gpu>(_auto_binding_or_numa_ids)) {
    assert(topology != nullptr && "Topology must be provided when auto-binding to NUMA nodes");
    std::set<int> gpu_numa_ids;
    for (int gpu_id : gpu_ids) {
      const auto& gpu_info = topology->gpus[static_cast<std::size_t>(gpu_id)];
      gpu_numa_ids.insert(gpu_info.numa_node);
    }
    host_numa_ids.insert(host_numa_ids.end(), gpu_numa_ids.begin(), gpu_numa_ids.end());
  } else if (std::holds_alternative<std::vector<int>>(_auto_binding_or_numa_ids)) {
    host_numa_ids = std::get<std::vector<int>>(_auto_binding_or_numa_ids);
  } else {
    if (std::holds_alternative<same_ids_tag>(_auto_binding_or_numa_ids)) {
      host_numa_ids = gpu_ids;
    } else if (std::holds_alternative<std::list<int>>(_auto_binding_or_numa_ids)) {
      const auto& ids = std::get<std::list<int>>(_auto_binding_or_numa_ids);
      host_numa_ids.insert(host_numa_ids.end(), ids.begin(), ids.end());
    }
    auto current_mr_fn = _cpu_mr_fn;
    _cpu_mr_fn         = [current_mr_fn]([[maybe_unused]] int host_id, size_t capacity)
      -> std::unique_ptr<rmm::mr::device_memory_resource> { return current_mr_fn(-1, capacity); };
  }

  std::sort(host_numa_ids.begin(), host_numa_ids.end());
  host_numa_ids.erase(std::unique(host_numa_ids.begin(), host_numa_ids.end()), host_numa_ids.end());

  return host_numa_ids;
}

}  // namespace memory
}  // namespace cucascade
