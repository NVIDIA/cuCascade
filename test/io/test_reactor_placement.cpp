/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cucascade/exec/thread_util.hpp>
#include <cucascade/io/reactor_placement.hpp>
#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>
#include <cucascade/memory/topology_discovery.hpp>
#include <cucascade/memory/topology_index.hpp>

#include <catch2/catch_all.hpp>

#include <cstddef>
#include <memory>
#include <thread>
#include <vector>

namespace {

using cucascade::io::make_reactor_placements;
using cucascade::io::reactor_placement;

constexpr std::size_t MiB = 1UL << 20;

/// GPU 0 on NUMA node 0 (CPUs 0-1) and GPU 1 on NUMA node 1 (CPUs 2-3).
[[nodiscard]] cucascade::memory::topology_index two_node_index()
{
  cucascade::memory::system_topology_info topology{};
  topology.num_gpus       = 2;
  topology.num_numa_nodes = 2;
  for (unsigned int gpu = 0; gpu < 2; ++gpu) {
    cucascade::memory::gpu_topology_info info;
    info.id        = gpu;
    info.numa_node = static_cast<int>(gpu);
    info.cpu_cores = {static_cast<int>(2 * gpu), static_cast<int>(2 * gpu + 1)};
    topology.gpus.push_back(info);
  }
  return cucascade::memory::topology_index{topology, std::vector<int>{0, 1}};
}

}  // namespace

TEST_CASE("without a topology every reactor is unplaced", "[io][reactor_placement]")
{
  auto const placements = make_reactor_placements(3, nullptr, nullptr, true);
  REQUIRE(placements.size() == 3);
  for (auto const& placement : placements) {
    CHECK(placement.gpu_id == -1);
    CHECK(placement.numa_node == -1);
    CHECK(placement.staging_mr == nullptr);
    CHECK(placement.cpu_ids.empty());
  }
}

TEST_CASE("reactors are placed per GPU on its NUMA node and CPUs", "[io][reactor_placement]")
{
  auto const index      = two_node_index();
  auto const placements = make_reactor_placements(2, &index, nullptr, true);
  REQUIRE(placements.size() == 4);
  for (std::size_t i = 0; i < placements.size(); ++i) {
    int const gpu = static_cast<int>(i / 2);
    CHECK(placements[i].gpu_id == gpu);
    CHECK(placements[i].numa_node == gpu);
    CHECK(placements[i].cpu_ids == std::vector<int>{2 * gpu, 2 * gpu + 1});
    // No reservation manager: every reactor keeps the pool's shared staging.
    CHECK(placements[i].staging_mr == nullptr);
  }

  SECTION("pinning off leaves the workers unpinned")
  {
    for (auto const& placement : make_reactor_placements(1, &index, nullptr, false)) {
      CHECK(placement.cpu_ids.empty());
      CHECK(placement.numa_node == placement.gpu_id);
    }
  }
}

TEST_CASE("pin_reactor_thread pins the calling thread to the placement's CPUs",
          "[io][reactor_placement]")
{
  auto const allowed = cucascade::exec::thread_util::get_current_thread_affinity();
  REQUIRE_FALSE(allowed.empty());

  std::vector<int> seen;
  std::thread worker([&] {
    reactor_placement placement;
    placement.cpu_ids = {allowed.front()};
    cucascade::io::pin_reactor_thread(placement, "test");
    seen = cucascade::exec::thread_util::get_current_thread_affinity();
  });
  worker.join();
  CHECK(seen == std::vector<int>{allowed.front()});

  std::vector<int> unpinned;
  std::thread untouched([&] {
    cucascade::io::pin_reactor_thread(reactor_placement{}, "test");
    unpinned = cucascade::exec::thread_util::get_current_thread_affinity();
  });
  untouched.join();
  CHECK(unpinned == allowed);
}

TEST_CASE("a placed uring reactor stages from its placement's resource",
          "[io][reactor_placement][uring]")
{
  cucascade::memory::numa_region_pinned_host_memory_resource upstream{0};
  cucascade::memory::fixed_size_host_memory_resource shared{
    0, upstream, 256 * MiB, 256 * MiB, MiB, 64, 0};
  cucascade::memory::fixed_size_host_memory_resource local{
    0, upstream, 256 * MiB, 256 * MiB, 2 * MiB, 32, 0};

  auto ctx = std::make_shared<cucascade::io::uring::uring_reactor::reactor_context>(
    cucascade::io::uring::config{}, &shared);
  reactor_placement placement;
  placement.gpu_id     = 0;
  placement.numa_node  = 0;
  placement.staging_mr = &local;
  cucascade::io::reactor_layout layout;
  layout.placements = {placement, reactor_placement{}};

  cucascade::io::uring::uring_ioctx context{layout, ctx};
  CHECK(context.staging_block_size() == 2 * MiB);
  context.start();
  CHECK(local.get_total_allocated_bytes() > 0);
  // The unplaced second reactor still stages from the context's resource.
  CHECK(shared.get_total_allocated_bytes() > 0);
  context.shutdown();
}
