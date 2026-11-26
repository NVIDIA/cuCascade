/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "utils/allocation_counter.hpp"

#include <cucascade/memory/gpu_numa_node.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <limits>
#include <string>
#include <tuple>

namespace {

using cucascade::memory::gpu_numa_node;
using cucascade::memory::detail::read_gpu_numa_node;

/**
 * @brief The NUMA node that sysfs reports for CUDA device @p device_id, or -1; read independently
 * of the code under test.
 */
int numa_node_from_sysfs(int device_id)
{
  std::array<char, 32> bus_id{};
  REQUIRE(cudaDeviceGetPCIBusId(bus_id.data(), static_cast<int>(bus_id.size()), device_id) ==
          cudaSuccess);
  std::string lower{bus_id.data()};
  std::ranges::transform(
    lower, lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  std::ifstream file{"/sys/bus/pci/devices/" + lower + "/numa_node"};
  int node = -1;
  return file >> node ? node : -1;
}

int visible_device_count()
{
  int count = 0;
  REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
  return count;
}

}  // namespace

TEST_CASE("the NUMA node lookup of a GPU does not allocate", "[gpu_numa_node]")
{
  int const devices = visible_device_count();
  REQUIRE(devices > 0);
  // The CUDA runtime may allocate while it initializes; that is not the lookup's doing.
  std::ignore = numa_node_from_sysfs(0);

  cucascade::test::scoped_allocation_counter const allocations;
  for (int device = 0; device < devices; ++device) {
    std::ignore = read_gpu_numa_node(device);
    std::ignore = gpu_numa_node(device);
  }
  // Devices that do not exist take the failure paths.
  for (int device : {-1, devices, std::numeric_limits<int>::max()}) {
    std::ignore = read_gpu_numa_node(device);
    std::ignore = gpu_numa_node(device);
  }
  CHECK(allocations.count() == 0);
}

TEST_CASE("the NUMA node lookup of a GPU reports what sysfs does", "[gpu_numa_node]")
{
  int const devices = visible_device_count();
  for (int device = 0; device < devices; ++device) {
    CAPTURE(device);
    int const expected = numa_node_from_sysfs(device);
    CHECK(read_gpu_numa_node(device) == expected);
    // Asked twice: the remembered value is the same.
    CHECK(gpu_numa_node(device) == expected);
    CHECK(gpu_numa_node(device) == expected);
  }
  for (int device : {-1, devices, std::numeric_limits<int>::max()}) {
    CAPTURE(device);
    CHECK(read_gpu_numa_node(device) == -1);
    CHECK(gpu_numa_node(device) == -1);
  }
  // No failed lookup leaves a CUDA error behind for a later call to report.
  CHECK(cudaGetLastError() == cudaSuccess);
}
