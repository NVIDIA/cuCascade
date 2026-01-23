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

#pragma once

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <list>
#include <span>
#include <unordered_map>
#include <variant>
#include <vector>

namespace cucascade {
namespace memory {

//===----------------------------------------------------------------------===//
// memory_reservation_manager
//===----------------------------------------------------------------------===//

/**
 * @class reservation_manager_config_builder
 * @brief Builder class for configuring memory reservation manager settings.
 *
 * This class provides a fluent interface to configure GPU and CPU memory reservation
 * parameters, including device selection, memory limits, reservation ratios, and
 * memory resource factories. It supports both explicit device IDs and automatic
 * configuration based on system topology.
 *
 * Usage example:
 * @code
 * reservation_manager_config_builder builder;
 * builder.set_number_of_gpus(2)
 *        .set_gpu_usage_limit(2UL << 30)
 *        .set_reservation_limit_ratio_per_gpu(0.8)
 *        .set_numa_ids({0, 1})
 *        .set_capacity_per_numa_node(8UL << 30)
 *        .set_gpu_memory_resource_factory(custom_gpu_factory)
 *        .set_cpu_memory_resource_factory(custom_cpu_factory);
 * auto configs = builder.build(system_topology);
 * @endcode
 *
 * @note Either GPU IDs or number of GPUs must be set. Similarly, either explicit memory
 * limits or ratios can be specified for GPUs and NUMA nodes.
 */

class reservation_manager_configurator {
 public:
  using builder_reference = reservation_manager_configurator;

  /// === --- gpu settings --- ===
  /// @brief set number of gpus
  /// @param n_gpus Number of GPUs to configure.
  builder_reference& set_number_of_gpus(std::size_t n_gpus);

  // either set space capacity or set a ratio of gpu total capacity
  /// @brief set gpu usage limit in bytes (i.e. capacity)
  /// @param bytes Memory usage limit in bytes.
  builder_reference& set_gpu_usage_limit(std::size_t bytes);

  /// @brief set gpu usage limit as a fraction of total GPU capacity
  /// @param fraction Fraction of total GPU capacity to use.
  builder_reference& set_usage_limit_ratio_per_gpu(double fraction);

  /// \brief set reservation limit ratio per GPU
  /// @param fraction Fraction of GPU memory capacity to reserve.
  builder_reference& set_reservation_limit_per_gpu(size_t bytes);

  /// \brief set reservation limit ratio per GPU
  /// @param fraction Fraction of GPU memory capacity to reserve.
  builder_reference& set_reservation_fraction_per_gpu(double fraction);

  /// \brief set reservation limit ratio per GPU
  /// @param fraction Fraction of GPU memory capacity to reserve.
  builder_reference& set_downgrade_fractions_per_gpu(double start, double end);

  builder_reference& track_reservation_per_stream(bool enable);

  /// \brief set the function that takes in the device id and create gpu memory resource
  /// @param mr_fn Function to create GPU memory resource.
  builder_reference& set_gpu_memory_resource_factory(DeviceMemoryResourceFactoryFn mr_fn);

  // --- cpu / host settings ---

  /// @brief set host ids
  /// @param host_ids Vector of host ids.
  /// @note this is meant to be used for testing purpose only, host ids will be mapped to numa ids
  builder_reference& use_host_per_gpu();

  /// @brief automatically bind cpu tiers to gpus based on topology
  builder_reference& use_host_per_numa();

  /// set capacity per host tier
  /// @param bytes Memory capacity per NUMA node in bytes.
  builder_reference& set_total_host_capacity(std::size_t bytes);

  /// set capacity per host tier
  /// @param bytes Memory capacity per NUMA node in bytes.
  builder_reference& set_per_host_capacity(std::size_t bytes);

  /// \brief set reservation limit ratio per GPU
  /// @param fraction Fraction of GPU memory capacity to reserve.
  builder_reference& set_downgrade_fractions_per_host(double start, double end);

  /// \brief set ratio of space capacity used for reservation in cpus
  /// @param fraction Fraction of NUMA node memory capacity to reserve.
  builder_reference& set_reservation_limit_per_host(size_t bytes);

  /// \brief set ratio of space capacity used for reservation in cpus
  /// @param fraction Fraction of NUMA node memory capacity to reserve.
  builder_reference& set_reservation_fraction_per_host(double fraction);

  /// \brief set the function that takes in the numa node id and create cpu memory resource
  /// @param mr_fn Function to create CPU memory resource.
  builder_reference& set_host_memory_resource_factory(DeviceMemoryResourceFactoryFn mr_fn);

  builder_reference& set_host_pool_features(std::size_t chunk_size,
                                            std::size_t block_size,
                                            std::size_t initial_block_count);

  // --- disk settings ---

  /// \brief set the function that takes in the numa node id and create cpu memory resource
  /// @param mr_fn Function to create CPU memory resource.
  builder_reference& set_disk_mounting_point(int uuid,
                                             std::size_t capacity,
                                             std::string mounting_point);

  /// \brief build the memory space configurations based on the provided system topology
  /// @param topology System topology information.
  /// @return Vector of memory space configurations.
  std::vector<memory_space_config> build(const system_topology_info& topology) const;

  /// \brief build the memory space configurations without topology information
  /// @return Vector of memory space configurations.
  std::vector<memory_space_config> build() const;

 private:
  struct gpu_info {
    int space_id{-1};
    int hw_id{-1};
    size_t gpu_capacity{0};
    int numa_id{-1};
  };

  struct host_info {
    int space_id{-1};
    int numa_id{-1};
  };

  std::vector<gpu_info> extract_gpu_ids(const system_topology_info& topology) const;

  std::vector<host_info> extract_host_ids(const std::vector<gpu_info>& gpus,
                                          const system_topology_info& topology) const;

  struct fraction_or_size {
    fraction_or_size(double fraction) : _fraction_or_size_value(fraction) {}
    fraction_or_size(std::size_t size) : _fraction_or_size_value(size) {}

    [[nodiscard]] double get_fraction(std::size_t total_size) const
    {
      if (std::holds_alternative<double>(_fraction_or_size_value)) {
        return std::get<double>(_fraction_or_size_value);
      } else {
        assert(std::get<size_t>(_fraction_or_size_value) <= total_size);
        auto size = std::get<std::size_t>(_fraction_or_size_value);
        return static_cast<double>(size) / static_cast<double>(total_size);
      }
    }

    [[nodiscard]] std::size_t get_capacity(std::size_t total_size) const
    {
      if (std::holds_alternative<double>(_fraction_or_size_value)) {
        double fraction = std::get<double>(_fraction_or_size_value);
        return static_cast<std::size_t>(static_cast<double>(total_size) * fraction);
      } else {
        assert(std::get<size_t>(_fraction_or_size_value) <= total_size && total_size > 0);
        return std::get<std::size_t>(_fraction_or_size_value);
      }
    }

    std::variant<double, std::size_t> _fraction_or_size_value;
  };

  size_t _num_gpus{1};
  bool _per_stream_gpu_reservation{true};
  fraction_or_size _gpu_capacity{static_cast<std::size_t>(1UL << 30)};  // uses 1GB of gpu memory
  fraction_or_size _gpu_reservation{0.85};                              // uses 85% of capacity
  std::pair<double, double> downgrade_fractions_per_gpu_{0.85, 0.65};
  mutable DeviceMemoryResourceFactoryFn _gpu_mr_fn = make_default_gpu_memory_resource;

  std::size_t _host_capacity{static_cast<std::size_t>(4UL << 30)};  // 4GB
  bool _is_capacity_per_space{true};
  struct bind_host_to_gpu_id {};
  struct bind_cpu_to_gpu_numa {};
  std::variant<bind_cpu_to_gpu_numa, bind_host_to_gpu_id> _host_creation_policy{};
  std::optional<std::size_t> chunk_size;
  std::optional<std::size_t> block_size;
  std::optional<std::size_t> initial_block_count;
  std::pair<double, double> downgrade_fractions_per_host_{0.85, 0.65};
  fraction_or_size _cpu_reservation{0.85};  // 75% limit per NUMA node by default
  mutable DeviceMemoryResourceFactoryFn _cpu_mr_fn = make_default_host_memory_resource;

  std::vector<std::tuple<int, std::size_t, std::string>> _disk_mounting_points{};
};

}  // namespace memory
}  // namespace cucascade
