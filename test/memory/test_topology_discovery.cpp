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
 * Test Tags:
 * [memory_space] - Basic memory space functionality tests
 * [threading] - Multi-threaded tests
 * [gpu] - GPU-specific tests requiring CUDA
 * [.multi-device] - Tests requiring multiple GPU devices (hidden by default)
 *
 * Running tests:
 * - Default (includes single GPU): ./test_executable
 * - Include multi-device tests: ./test_executable "[.multi-device]"
 * - Exclude multi-device tests: ./test_executable "~[.multi-device]"
 * - Run all tests: ./test_executable "[memory_space]"
 */

#include <cucascade/memory/topology_discovery.hpp>

#include <catch2/catch.hpp>

#include <cstdlib>
#include <string>
#include <vector>

using namespace cucascade::memory;

namespace {
struct ScopedEnvVar {
  explicit ScopedEnvVar(std::string name, std::string value) : name_(std::move(name))
  {
    const char* existing = std::getenv(name_.c_str());
    if (existing) {
      had_value_ = true;
      old_value_ = existing;
    }
    setenv(name_.c_str(), value.c_str(), 1);
  }

  ~ScopedEnvVar()
  {
    if (had_value_) {
      setenv(name_.c_str(), old_value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  ScopedEnvVar(ScopedEnvVar const&)            = delete;
  ScopedEnvVar& operator=(ScopedEnvVar const&) = delete;

 private:
  std::string name_;
  std::string old_value_;
  bool had_value_ = false;
};
}  // namespace

// Test topology discovery
TEST_CASE("Topology Discovery", "[hw_topology]")
{
  topology_discovery discovery;

  // Call discover() method
  bool success = discovery.discover();

  // Verify discovery was successful
  REQUIRE(success == true);
  REQUIRE(discovery.is_discovered() == true);

  // Get the topology information
  const auto& topology = discovery.get_topology();

  // Verify basic topology information
  REQUIRE(!topology.hostname.empty());
  REQUIRE(topology.num_gpus >= 0);
  REQUIRE(topology.num_numa_nodes >= 0);
  REQUIRE(topology.num_network_devices >= 0);

  // Verify GPU information consistency
  REQUIRE(topology.gpus.size() == topology.num_gpus);

  // If GPUs are present, verify their information
  if (topology.num_gpus > 0) {
    for (const auto& gpu : topology.gpus) {
      REQUIRE(!gpu.name.empty());
      REQUIRE(!gpu.pci_bus_id.empty());
      REQUIRE(!gpu.uuid.empty());
      // NUMA node may be -1 if unknown
      REQUIRE(gpu.numa_node >= -1);
    }
  }

  // Verify network device information consistency
  REQUIRE(topology.network_devices.size() == static_cast<size_t>(topology.num_network_devices));

  // If network devices are present, verify their information
  for (const auto& net_dev : topology.network_devices) {
    REQUIRE(!net_dev.name.empty());
    // NUMA node may be -1 if unknown
    REQUIRE(net_dev.numa_node >= -1);
  }
}

TEST_CASE("Topology Discovery rejects out-of-range CUDA_VISIBLE_DEVICES", "[hw_topology]")
{
  ScopedEnvVar env("CUDA_VISIBLE_DEVICES", "99999999");

  topology_discovery discovery;
  REQUIRE_THROWS_WITH(discovery.discover(), "CUDA_VISIBLE_DEVICES entry 99999999 is out of range");
}

TEST_CASE("Topology Discovery rejects overflow CUDA_VISIBLE_DEVICES", "[hw_topology]")
{
  ScopedEnvVar env("CUDA_VISIBLE_DEVICES", "999999999999999999999999999999");

  topology_discovery discovery;
  REQUIRE_THROWS_WITH(discovery.discover(),
                      "Invalid numeric CUDA_VISIBLE_DEVICES entry: 999999999999999999999999999999");
}
