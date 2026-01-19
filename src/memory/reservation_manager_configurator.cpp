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
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <numa.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <set>
#include <unordered_map>
#include <variant>
#include <vector>

namespace cucascade {
namespace memory {

using builder_reference = reservation_manager_configurator::builder_reference;

builder_reference& reservation_manager_configurator::set_number_of_gpus(std::size_t n_gpus)
{
  assert(n_gpus > 0 && "Number of GPUs must be positive");
  _num_gpus = n_gpus;
  return *this;
}

builder_reference& reservation_manager_configurator::set_gpu_usage_limit(std::size_t bytes)
{
  _gpu_capacity = bytes;
  return *this;
}

builder_reference& reservation_manager_configurator::set_usage_limit_ratio_per_gpu(double fraction)
{
  assert(fraction > 0.0 && fraction <= 1.0 && "Usage limit ratio must be in (0.0, 1.0]");
  _gpu_capacity = fraction;
  return *this;
}

builder_reference& reservation_manager_configurator::set_reservation_limit_per_gpu(size_t bytes)
{
  _gpu_reservation = bytes;
  return *this;
}

builder_reference& reservation_manager_configurator::set_reservation_fraction_per_gpu(
  double fraction)
{
  assert(fraction > 0.0 && fraction <= 1.0 && "Reservation limit ratio must be in (0.0, 1.0]");
  _gpu_reservation = fraction;
  return *this;
}

builder_reference& reservation_manager_configurator::set_downgrade_fractions_per_gpu(double start,
                                                                                     double end)
{
  assert(start > 0.0 && start <= 1.0 && "Start fraction must be in (0.0, 1.0]");
  assert(end > 0.0 && end <= 1.0 && "End fraction must be in (0.0, 1.0]");
  downgrade_fractions_per_gpu_ = {start, end};
  return *this;
}

builder_reference& reservation_manager_configurator::track_reservation_per_stream(bool enable)
{
  _per_stream_gpu_reservation = enable;
  return *this;
}

// --- cpu / host settings ---

builder_reference& reservation_manager_configurator::use_host_per_gpu()
{
  _host_creation_policy = bind_host_to_gpu_id{};
  return *this;
}

builder_reference& reservation_manager_configurator::use_host_per_numa()
{
  _host_creation_policy = bind_cpu_to_gpu_numa{};
  return *this;
}

/// set capacity per host tier
/// @param bytes Memory capacity per NUMA node in bytes.
builder_reference& reservation_manager_configurator::set_total_host_capacity(std::size_t bytes)
{
  assert(bytes > 0 && "Total host capacity must be positive");
  _host_capacity         = bytes;
  _is_capacity_per_space = false;
  return *this;
}

builder_reference& reservation_manager_configurator::set_per_host_capacity(std::size_t bytes)
{
  assert(bytes > 0 && "Capacity per NUMA node must be positive");
  _host_capacity         = bytes;
  _is_capacity_per_space = true;
  return *this;
}

builder_reference& reservation_manager_configurator::set_downgrade_fractions_per_host(double start,
                                                                                      double end)
{
  assert(start > 0.0 && start <= 1.0 && "Start fraction must be in (0.0, 1.0]");
  assert(end > 0.0 && end <= 1.0 && "End fraction must be in (0.0, 1.0]");
  downgrade_fractions_per_host_ = {start, end};
  return *this;
}

builder_reference& reservation_manager_configurator::set_reservation_fraction_per_host(
  double fraction)
{
  assert(fraction > 0.0 && fraction <= 1.0 && "Reservation limit ratio must be in (0.0, 1.0]");
  _cpu_reservation = fraction;
  return *this;
}

builder_reference& reservation_manager_configurator::set_reservation_limit_per_host(size_t bytes)
{
  _cpu_reservation = bytes;
  return *this;
}

builder_reference& reservation_manager_configurator::set_host_pool_features(
  size_t chunk_size, size_t block_size, size_t initial_block_count)
{
  assert(chunk_size > 0 && "Chunk size must be positive");
  assert(block_size > 0 && "Block size must be positive");
  assert(initial_block_count > 0 && "Initial block count must be positive");
  this->chunk_size          = chunk_size;
  this->block_size          = block_size;
  this->initial_block_count = initial_block_count;
  return *this;
}

builder_reference& reservation_manager_configurator::set_gpu_memory_resource_factory(
  DeviceMemoryResourceFactoryFn mr_fn)
{
  assert(mr_fn && "GPU memory resource factory cannot be nullptr");
  _gpu_mr_fn = std::move(mr_fn);
  return *this;
}

builder_reference& reservation_manager_configurator::set_host_memory_resource_factory(
  DeviceMemoryResourceFactoryFn mr_fn)
{
  assert(mr_fn && "CPU memory resource factory cannot be nullptr");
  _cpu_mr_fn = std::move(mr_fn);
  return *this;
}

builder_reference& reservation_manager_configurator::set_disk_mounting_point(
  int uuid, std::size_t capacity, std::string mounting_point)
{
  assert(!mounting_point.empty() && "Mounting point cannot be empty");
  _disk_mounting_points.emplace_back(uuid, capacity, std::move(mounting_point));
  return *this;
}

std::vector<memory_space_config> reservation_manager_configurator::build() const
{
  topology_discovery discovery;
  [[maybe_unused]] bool status = discovery.discover();
  assert(status);
  auto& topology = discovery.get_topology();
  return build(topology);
}

std::vector<memory_space_config> reservation_manager_configurator::build(
  const system_topology_info& topology) const
{
  auto gpus_info  = extract_gpu_ids(topology);
  auto host_infos = extract_host_ids(gpus_info, topology);

  std::vector<memory_space_config> configs;
  for (auto& info : gpus_info) {
    gpu_memory_space_config config;
    config.device_id       = info.space_id;
    config.memory_capacity = _gpu_capacity.get_capacity(info.gpu_capacity);
    config.mr_factory_fn =
      (info.space_id == info.hw_id)
        ? _gpu_mr_fn
        : [current_mr_fn = _gpu_mr_fn, hw_id = info.hw_id](
            int, size_t capacity) -> std::unique_ptr<rmm::mr::device_memory_resource> {
      return current_mr_fn(hw_id, capacity);
    };
    config.reservation_limit_fraction = _gpu_reservation.get_fraction(info.gpu_capacity);
    config.downgrade_trigger_fraction = downgrade_fractions_per_gpu_.first;
    config.downgrade_stop_fraction    = downgrade_fractions_per_gpu_.second;
    config.per_stream_reservation     = _per_stream_gpu_reservation;
    configs.emplace_back(config);
  };

  size_t per_host_capacity =
    (host_infos.size() <= 1)
      ? _host_capacity
      : (_is_capacity_per_space ? _host_capacity : _host_capacity / host_infos.size());

  for (auto& info : host_infos) {
    host_memory_space_config config;
    config.numa_id         = info.space_id;
    config.memory_capacity = per_host_capacity;
    config.mr_factory_fn =
      (info.space_id == info.numa_id)
        ? _cpu_mr_fn
        : [current_mr_fn = _cpu_mr_fn, numa_id = info.numa_id](
            int, size_t capacity) -> std::unique_ptr<rmm::mr::device_memory_resource> {
      return current_mr_fn(numa_id, capacity);
    };
    config.reservation_limit_fraction = _cpu_reservation.get_fraction(per_host_capacity);
    config.downgrade_trigger_fraction = downgrade_fractions_per_host_.first;
    config.downgrade_stop_fraction    = downgrade_fractions_per_host_.second;
    config.block_size                 = chunk_size.value_or(memory::default_block_size);
    config.pool_size                  = block_size.value_or(memory::default_pool_size);
    config.initial_number_pools =
      initial_block_count.value_or(memory::default_initial_number_pools);
    configs.emplace_back(config);
  }

  for (auto dinfo : _disk_mounting_points) {
    disk_memory_space_config config;
    config.disk_id         = std::get<int>(dinfo);
    config.mount_paths     = std::get<std::string>(dinfo);
    config.memory_capacity = std::get<std::size_t>(dinfo);
    configs.emplace_back(config);
  }

  return configs;
}

std::vector<reservation_manager_configurator::gpu_info>
reservation_manager_configurator::extract_gpu_ids(const system_topology_info& topology) const
{
  if (_num_gpus > topology.gpus.size()) {
    throw std::runtime_error("Requested number of GPUs exceeds available GPUs");
  }

  size_t num_gpus = _num_gpus;
  if (num_gpus == 0) { num_gpus = topology.gpus.size(); }

  std::vector<int> gpu_ids(num_gpus);
  std::iota(gpu_ids.begin(), gpu_ids.end(), 0);

  std::unordered_map<int, std::pair<size_t, int>> available_gpu_ids;
  for (const auto& gpu : topology.gpus) {
    auto device_id = static_cast<int32_t>(gpu.id);
    rmm::cuda_set_device_raii set_device(rmm::cuda_device_id{device_id});
    auto const [free, total]     = rmm::available_device_memory();
    available_gpu_ids[device_id] = {total, gpu.numa_node};
  }

  std::vector<gpu_info> gpu_infos;
  for (const auto& gpu_id : gpu_ids) {
    auto it = available_gpu_ids.find(gpu_id);
    if (it == available_gpu_ids.end()) {
      throw std::runtime_error("Requested GPU ID " + std::to_string(gpu_id) +
                               " is not available in the system topology");
    }
    gpu_info& info    = gpu_infos.emplace_back();
    info.space_id     = it->first;
    info.hw_id        = it->first;
    info.gpu_capacity = it->second.first;
    info.numa_id      = it->second.second;
  }

  return gpu_infos;
}

std::vector<reservation_manager_configurator::host_info>
reservation_manager_configurator::extract_host_ids(
  const std::vector<gpu_info>& gpus, [[maybe_unused]] const system_topology_info& topology) const
{
  std::vector<host_info> host_infos;
  std::set<int> host_ids_set;
  for (const auto& gpu : gpus) {
    if (std::holds_alternative<bind_host_to_gpu_id>(_host_creation_policy)) {
      host_infos.emplace_back(host_info{.space_id = gpu.space_id, .numa_id = gpu.numa_id});
    } else if (std::holds_alternative<bind_cpu_to_gpu_numa>(_host_creation_policy)) {
      if (!host_ids_set.contains(gpu.numa_id)) {
        host_infos.emplace_back(host_info{.space_id = gpu.numa_id, .numa_id = gpu.numa_id});
        host_ids_set.insert(gpu.numa_id);
      }
    }
  }
  return host_infos;
}

}  // namespace memory
}  // namespace cucascade
