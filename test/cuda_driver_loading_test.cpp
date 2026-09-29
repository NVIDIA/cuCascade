/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cucascade/memory/topology_discovery.hpp>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace {

struct runtime_query_result {
  bool attributes_populated;
  bool hw_decomp;
};

runtime_query_result query_fake_gpu_runtime_attribute()
{
  cucascade::memory::system_topology_info topology;
  cucascade::memory::gpu_topology_info gpu;
  gpu.pci_bus_id = "0000:01:00.0";
  topology.gpus.push_back(std::move(gpu));

  cucascade::memory::topology_discovery::discover_runtime_attributes(topology);
  auto const& attributes = topology.gpus.front().runtime_attributes;
  return {attributes.has_value(), attributes.has_value() && attributes->hw_decomp};
}

struct late_runtime_query {
  bool enabled{false};

  ~late_runtime_query()
  {
    if (!enabled) { return; }
    auto const result = query_fake_gpu_runtime_attribute();
    if (!result.attributes_populated || !result.hw_decomp) { std::_Exit(2); }
  }
};

late_runtime_query query_after_function_local_statics_are_destroyed;

}  // namespace

int main(int argc, char** argv)
{
  if (argc != 2) { return 2; }

  auto const mode = std::string_view{argv[1]};
  if (mode == "available") {
    auto const result = query_fake_gpu_runtime_attribute();
    return result.attributes_populated && result.hw_decomp ? 0 : 1;
  }
  if (mode == "unavailable") {
    std::ostringstream warning;
    auto* original_stderr    = std::cerr.rdbuf(warning.rdbuf());
    auto const result        = query_fake_gpu_runtime_attribute();
    auto const repeat_result = query_fake_gpu_runtime_attribute();
    std::cerr.rdbuf(original_stderr);
    auto const message = warning.str();
    auto const expected =
      std::string{"Warning: Failed to load CUDA driver symbol "} + "cuDeviceGetAttribute:";
    return result.attributes_populated && !result.hw_decomp && repeat_result.attributes_populated &&
               !repeat_result.hw_decomp && message.find(expected) != std::string::npos &&
               message.find(expected, message.find(expected) + expected.size()) == std::string::npos
             ? 0
             : 1;
  }
  if (mode == "passive") {
    cucascade::memory::topology_discovery discovery;
    return discovery.discover() ? 0 : 1;
  }
  if (mode == "shutdown") {
    auto const result = query_fake_gpu_runtime_attribute();
    if (!result.attributes_populated || !result.hw_decomp) { return 1; }
    query_after_function_local_statics_are_destroyed.enabled = true;
    return 0;
  }
  return 2;
}
