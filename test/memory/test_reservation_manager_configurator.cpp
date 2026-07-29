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

/**
 * Tests for host (NUMA) capacity configuration in reservation_manager_configurator.
 *
 * Test Tags:
 * [configurator] - reservation_manager_configurator tests
 */

#include <cucascade/memory/config.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <catch2/catch.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

using namespace cucascade::memory;

namespace {

constexpr std::size_t synthetic_numa_capacity = 64ull << 30;  // 64 GiB

/// Collect the host space configs produced by the builder.
std::vector<host_memory_space_config> host_configs(std::vector<memory_space_config> const& configs)
{
  std::vector<host_memory_space_config> hosts;
  for (auto const& config : configs) {
    if (auto const* host = std::get_if<host_memory_space_config>(&config)) {
      hosts.push_back(*host);
    }
  }
  return hosts;
}

/**
 * @brief Build a topology with a single real GPU and a synthetic NUMA node.
 *
 * The GPU entry must come from real discovery because the configurator queries the
 * device for its memory capacity; the NUMA capacity is overridden so the expected
 * values are independent of the machine running the test.
 */
bool make_single_gpu_topology(system_topology_info& topology, std::size_t numa_capacity)
{
  topology_discovery discovery;
  if (!discovery.discover()) { return false; }

  topology = discovery.get_topology();
  if (topology.gpus.empty()) { return false; }

  topology.gpus.resize(1);
  topology.num_gpus   = 1;
  topology.numa_nodes = {
    numa_topology_info{topology.gpus.front().numa_node, numa_capacity, numa_capacity / 2}};
  topology.num_numa_nodes = 1;
  return true;
}

}  // namespace

TEST_CASE("Configurator sets host capacity as a fraction of NUMA capacity", "[configurator]")
{
  system_topology_info topology;
  if (!make_single_gpu_topology(topology, synthetic_numa_capacity)) {
    SUCCEED("Skipped: requires at least one GPU");
    return;
  }

  reservation_manager_configurator builder;
  builder.set_gpu_ids({static_cast<int>(topology.gpus.front().id)});
  builder.use_host_per_numa();
  builder.set_usage_limit_ratio_per_host(0.25);

  auto const hosts = host_configs(builder.build(topology));

  REQUIRE(hosts.size() == 1);
  REQUIRE(hosts.front().memory_capacity == synthetic_numa_capacity / 4);
}

TEST_CASE("Configurator sets host capacity in absolute bytes", "[configurator]")
{
  system_topology_info topology;
  if (!make_single_gpu_topology(topology, synthetic_numa_capacity)) {
    SUCCEED("Skipped: requires at least one GPU");
    return;
  }

  constexpr std::size_t requested = 3ull << 30;  // 3 GiB

  reservation_manager_configurator builder;
  builder.set_gpu_ids({static_cast<int>(topology.gpus.front().id)});
  builder.use_host_per_numa();
  builder.set_per_host_capacity(requested);

  auto const hosts = host_configs(builder.build(topology));

  REQUIRE(hosts.size() == 1);
  // Absolute capacities are used verbatim, independent of the NUMA node capacity.
  REQUIRE(hosts.front().memory_capacity == requested);
}

TEST_CASE("Configurator host capacity fraction overrides a previous absolute setting",
          "[configurator]")
{
  system_topology_info topology;
  if (!make_single_gpu_topology(topology, synthetic_numa_capacity)) {
    SUCCEED("Skipped: requires at least one GPU");
    return;
  }

  reservation_manager_configurator builder;
  builder.set_gpu_ids({static_cast<int>(topology.gpus.front().id)});
  builder.use_host_per_numa();
  builder.set_total_host_capacity(1ull << 30);
  builder.set_usage_limit_ratio_per_host(0.5);

  auto const hosts = host_configs(builder.build(topology));

  REQUIRE(hosts.size() == 1);
  REQUIRE(hosts.front().memory_capacity == synthetic_numa_capacity / 2);
}

TEST_CASE("Configurator reservation limit follows the fraction-derived host capacity",
          "[configurator]")
{
  system_topology_info topology;
  if (!make_single_gpu_topology(topology, synthetic_numa_capacity)) {
    SUCCEED("Skipped: requires at least one GPU");
    return;
  }

  constexpr std::size_t reservation_bytes = 4ull << 30;  // 4 GiB

  reservation_manager_configurator builder;
  builder.set_gpu_ids({static_cast<int>(topology.gpus.front().id)});
  builder.use_host_per_numa();
  builder.set_usage_limit_ratio_per_host(0.5);
  builder.set_reservation_limit_per_host(reservation_bytes);

  auto const hosts = host_configs(builder.build(topology));

  REQUIRE(hosts.size() == 1);
  REQUIRE(hosts.front().memory_capacity == synthetic_numa_capacity / 2);
  REQUIRE(hosts.front().reservation_limit() == reservation_bytes);
}

TEST_CASE("Configurator throws when NUMA capacity is unknown and a fraction is requested",
          "[configurator]")
{
  system_topology_info topology;
  if (!make_single_gpu_topology(topology, synthetic_numa_capacity)) {
    SUCCEED("Skipped: requires at least one GPU");
    return;
  }

  // Drop NUMA information: the fraction can no longer be resolved.
  topology.numa_nodes.clear();
  topology.num_numa_nodes = 0;

  reservation_manager_configurator builder;
  builder.set_gpu_ids({static_cast<int>(topology.gpus.front().id)});
  builder.use_host_per_numa();
  builder.set_usage_limit_ratio_per_host(0.5);

  REQUIRE_THROWS_AS(builder.build(topology), std::runtime_error);
}

TEST_CASE("Configurator resolves host capacity from discovered NUMA capacity", "[configurator]")
{
  topology_discovery discovery;
  REQUIRE(discovery.discover());
  auto const& topology = discovery.get_topology();

  if (topology.gpus.empty() || topology.numa_nodes.empty()) {
    SUCCEED("Skipped: requires at least one GPU and NUMA node");
    return;
  }

  auto const numa_id       = topology.gpus.front().numa_node;
  auto const numa_capacity = topology.get_numa_memory_capacity(numa_id);
  if (numa_capacity == 0) {
    SUCCEED("Skipped: NUMA capacity is not exposed on this host");
    return;
  }

  reservation_manager_configurator builder;
  builder.set_gpu_ids({static_cast<int>(topology.gpus.front().id)});
  builder.use_host_per_numa();
  builder.set_usage_limit_ratio_per_host(0.1);

  auto const hosts = host_configs(builder.build(topology));

  REQUIRE(hosts.size() == 1);
  REQUIRE(hosts.front().memory_capacity > 0);
  REQUIRE(hosts.front().memory_capacity == static_cast<std::size_t>(
                                             static_cast<double>(numa_capacity) * 0.1));
}
