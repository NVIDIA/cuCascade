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

/**
 * Test Tags:
 * [topology_index] - topology_index space/candidate lookup + strategy-hash cache
 * [gpu] - requires CUDA (memory spaces are backed by real device/host resources)
 */

#include "utils/test_memory_resources.hpp"

#include <cucascade/error.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <cucascade/memory/topology_discovery.hpp>
#include <cucascade/memory/topology_index.hpp>

#include <catch2/catch.hpp>

#include <memory>
#include <vector>

using namespace cucascade::memory;

namespace {

constexpr size_t gpu_capacity  = 2ull << 30;  // 2 GB
constexpr size_t host_capacity = 4ull << 30;  // 4 GB
constexpr double limit_ratio   = 0.75;

std::unique_ptr<memory_reservation_manager> make_single_device_manager()
{
  reservation_manager_configurator builder;
  builder.set_gpu_usage_limit(gpu_capacity);
  builder.set_gpu_memory_resource_factory(cucascade::test::make_shared_current_device_resource);
  builder.set_reservation_fraction_per_gpu(limit_ratio);
  builder.set_per_host_capacity(host_capacity);
  builder.use_host_per_gpu();
  builder.set_reservation_fraction_per_host(limit_ratio);
  return std::make_unique<memory_reservation_manager>(builder.build());
}

// Minimal topology describing a single GPU on NUMA node 0.
system_topology_info single_gpu_topology()
{
  system_topology_info topology{};
  topology.hostname            = "test-host";
  topology.num_gpus            = 1;
  topology.num_numa_nodes      = 1;
  topology.num_network_devices = 0;
  gpu_topology_info gpu{};
  gpu.id        = 0;
  gpu.numa_node = 0;
  topology.gpus.push_back(gpu);
  return topology;
}

}  // namespace

TEST_CASE("topology_index::build exposes per-tier spaces", "[topology_index][gpu]")
{
  auto manager = make_single_device_manager();
  auto index   = build(single_gpu_topology(), *manager);
  REQUIRE(index != nullptr);

  // NUMA mapping still works (GPU 0 -> NUMA 0, cross-checked against the HOST space).
  REQUIRE(index->numa_node_of(0) == 0);

  // Const overload of get_spaces_of.
  auto gpu_spaces = index->get_spaces_of(Tier::GPU);
  REQUIRE(gpu_spaces.size() == 1);
  REQUIRE(gpu_spaces[0]->get_tier() == Tier::GPU);
  REQUIRE(gpu_spaces[0]->get_device_id() == 0);

  auto host_spaces = index->get_spaces_of(Tier::HOST);
  REQUIRE(host_spaces.size() == 1);
  REQUIRE(host_spaces[0]->get_tier() == Tier::HOST);

  // A tier with no configured spaces yields an empty span.
  REQUIRE(index->get_spaces_of(Tier::DISK).empty());
}

TEST_CASE("topology_index mutable and const get_spaces_of agree", "[topology_index][gpu]")
{
  auto manager = make_single_device_manager();
  // Non-const index to reach the mutable overload.
  topology_index index(single_gpu_topology(), *manager);

  std::span<memory_space*> mutable_spaces     = index.get_spaces_of(Tier::GPU);
  std::span<const memory_space*> const_spaces = std::as_const(index).get_spaces_of(Tier::GPU);

  REQUIRE(mutable_spaces.size() == const_spaces.size());
  REQUIRE(mutable_spaces.size() == 1);
  REQUIRE(mutable_spaces[0] == const_spaces[0]);  // same underlying space
}

TEST_CASE("reservation_request_strategy::hash distinguishes strategies", "[topology_index]")
{
  // Type + field sensitivity.
  REQUIRE(any_memory_space_in_tier(Tier::GPU).hash() == any_memory_space_in_tier(Tier::GPU).hash());
  REQUIRE(any_memory_space_in_tier(Tier::GPU).hash() !=
          any_memory_space_in_tier(Tier::HOST).hash());

  // Different concrete types hash differently even for the same tier.
  REQUIRE(any_memory_space_in_tier(Tier::GPU).hash() !=
          any_memory_space_in_tier_with_preference(Tier::GPU).hash());

  // Field sensitivity on specific_memory_space.
  REQUIRE(specific_memory_space(Tier::GPU, 0).hash() == specific_memory_space(Tier::GPU, 0).hash());
  REQUIRE(specific_memory_space(Tier::GPU, 0).hash() != specific_memory_space(Tier::GPU, 1).hash());

  // Preference presence changes the hash.
  REQUIRE(any_memory_space_in_tier_with_preference(Tier::GPU).hash() !=
          any_memory_space_in_tier_with_preference(Tier::GPU, 0).hash());

  // Ordered tier list is order-sensitive.
  REQUIRE(any_memory_space_in_tiers({Tier::GPU, Tier::HOST}).hash() !=
          any_memory_space_in_tiers({Tier::HOST, Tier::GPU}).hash());
}

TEST_CASE("topology_index::get_candidates memoizes and matches the strategy",
          "[topology_index][gpu]")
{
  auto manager = make_single_device_manager();
  auto index   = build(single_gpu_topology(), *manager);

  any_memory_space_in_tier strategy(Tier::GPU);

  auto expected  = strategy.get_candidates(*manager);
  auto candidate = index->get_candidates(strategy);
  REQUIRE(candidate.size() == expected.size());
  REQUIRE(candidate.size() == 1);
  REQUIRE(candidate[0] == expected[0]);

  // Second call with an equal-hash strategy is a cache hit: identical backing storage.
  any_memory_space_in_tier same_strategy(Tier::GPU);
  auto candidate2 = index->get_candidates(same_strategy);
  REQUIRE(candidate2.data() == candidate.data());
  REQUIRE(candidate2.size() == candidate.size());
}

TEST_CASE("topology_index::get_candidates without a manager throws", "[topology_index]")
{
  // The device-ids constructor leaves the index without a manager back-pointer.
  topology_index index(single_gpu_topology(), std::vector<int>{0});
  any_memory_space_in_tier strategy(Tier::GPU);
  REQUIRE_THROWS_AS(index.get_candidates(strategy), cucascade::logic_error);
}

TEST_CASE("reservation manager routes candidate selection through the index",
          "[topology_index][gpu]")
{
  auto manager = make_single_device_manager();
  manager->set_topology_index(build(single_gpu_topology(), *manager));
  REQUIRE(manager->get_topology_index() != nullptr);

  // A reservation still succeeds through the memoized candidate path.
  any_memory_space_in_tier strategy(Tier::GPU);
  auto res = manager->request_reservation(strategy, 1ull << 20);  // 1 MB
  REQUIRE(res != nullptr);
}
