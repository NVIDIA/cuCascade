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
#include <cucascade/io/config.hpp>
#ifdef CUCASCADE_HAS_CUFILE
#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#endif
#include <cucascade/io/datasource_factory.hpp>
#include <cucascade/io/reactor_placement.hpp>
#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>
#include <cucascade/memory/topology_discovery.hpp>
#include <cucascade/memory/topology_index.hpp>

#include <catch2/catch_all.hpp>
#include <sched.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
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

/// Affinity of this process's thread named @p name (its /proc comm), or empty when none is.
[[nodiscard]] std::vector<int> thread_affinity_named(std::string const& name)
{
  for (auto const& task : std::filesystem::directory_iterator("/proc/self/task")) {
    std::ifstream comm(task.path() / "comm");
    std::string line;
    if (!std::getline(comm, line) || line != name) { continue; }
    cpu_set_t mask;
    CPU_ZERO(&mask);
    auto const tid = static_cast<pid_t>(std::stol(task.path().filename().string()));
    if (::sched_getaffinity(tid, sizeof(mask), &mask) != 0) { return {}; }
    std::vector<int> ids;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &mask)) { ids.push_back(cpu); }
    }
    return ids;
  }
  return {};
}

/// A reservation manager whose only HOST space is NUMA node 0's (no pinned pool is allocated
/// until a reactor starts).
[[nodiscard]] std::vector<cucascade::memory::memory_space_config> numa0_host_only()
{
  cucascade::memory::host_memory_space_config host;
  host.numa_id              = 0;
  host.memory_capacity      = 256 * MiB;
  host.initial_number_pools = 0;
  return {host};
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
    0, upstream, 256 * MiB, 256 * MiB, MiB, 64, 0};

  auto ctx = std::make_shared<cucascade::io::uring::uring_reactor::reactor_context>(
    cucascade::io::uring::config{}, &shared);
  reactor_placement placement;
  placement.gpu_id     = 0;
  placement.numa_node  = 0;
  placement.staging_mr = &local;
  cucascade::io::reactor_layout layout;
  layout.placements = {placement, reactor_placement{}};

  cucascade::io::uring::uring_ioctx context{layout, ctx};
  CHECK(context.staging_block_size() == MiB);
  CHECK(local.get_total_allocated_bytes() == 0);
  context.start();
  CHECK(local.get_total_allocated_bytes() > 0);
  // The unplaced second reactor still stages from the context's resource.
  CHECK(shared.get_total_allocated_bytes() > 0);
  context.shutdown();
}

TEST_CASE("a layout whose reactors stage in different block sizes is rejected",
          "[io][reactor_placement][uring]")
{
  // The pool reports one staging block size, which a prefetching cache's chunks must match.
  cucascade::memory::numa_region_pinned_host_memory_resource upstream{0};
  cucascade::memory::fixed_size_host_memory_resource shared{
    0, upstream, 64 * MiB, 64 * MiB, MiB, 16, 0};
  cucascade::memory::fixed_size_host_memory_resource local{
    0, upstream, 64 * MiB, 64 * MiB, 2 * MiB, 8, 0};
  auto ctx = std::make_shared<cucascade::io::uring::uring_reactor::reactor_context>(
    cucascade::io::uring::config{}, &shared);
  reactor_placement placement;
  placement.gpu_id     = 0;
  placement.staging_mr = &local;
  cucascade::io::reactor_layout layout;
  layout.placements = {placement, reactor_placement{}};
  CHECK_THROWS_AS((cucascade::io::uring::uring_ioctx{layout, ctx}), std::invalid_argument);

  // Every reactor on the 2 MiB staging is fine.
  layout.placements = {placement, placement};
  cucascade::io::uring::uring_ioctx const uniform{layout, ctx};
  CHECK(uniform.staging_block_size() == 2 * MiB);
}

TEST_CASE("pin_reactor_thread never widens the inherited affinity", "[io][reactor_placement]")
{
  auto const allowed = cucascade::exec::thread_util::get_current_thread_affinity();
  if (allowed.size() < 2) { SKIP("needs two allowed CPUs"); }

  bool restricted = false;
  std::vector<int> narrowed;
  std::vector<int> outside;
  // Catch2 assertions stay on the test thread: the worker only records what it saw.
  std::thread worker([&] {
    // As under `taskset -c <a0>`: the placement lists a0 and a1, only a0 is granted.
    restricted =
      cucascade::exec::thread_util::set_current_thread_affinity(std::vector<int>{allowed[0]});
    if (!restricted) { return; }
    reactor_placement placement;
    placement.cpu_ids = {allowed[1], allowed[0]};
    cucascade::io::pin_reactor_thread(placement, "test");
    narrowed = cucascade::exec::thread_util::get_current_thread_affinity();

    // None of the placement's CPUs is granted: the thread keeps what it had.
    placement.cpu_ids = {allowed[1]};
    cucascade::io::pin_reactor_thread(placement, "test");
    outside = cucascade::exec::thread_util::get_current_thread_affinity();
  });
  worker.join();
  REQUIRE(restricted);
  CHECK(narrowed == std::vector<int>{allowed[0]});
  CHECK(outside == std::vector<int>{allowed[0]});
}

TEST_CASE("starting a placed uring reactor pins its worker within the inherited affinity",
          "[io][reactor_placement][uring]")
{
  auto const allowed = cucascade::exec::thread_util::get_current_thread_affinity();
  if (allowed.size() < 3) { SKIP("needs three allowed CPUs"); }

  cucascade::memory::numa_region_pinned_host_memory_resource upstream{0};
  cucascade::memory::fixed_size_host_memory_resource staging{
    0, upstream, 256 * MiB, 256 * MiB, MiB, 64, 0};
  auto ctx = std::make_shared<cucascade::io::uring::uring_reactor::reactor_context>(
    cucascade::io::uring::config{}, &staging);

  // Affinity of the worker of a reactor named @p name, started (and shut down) from a thread
  // restricted to {a0, a1}, as a process under `taskset -c a0,a1` would be.
  auto const worker_mask = [&](std::string const& name, std::vector<int> cpu_ids) {
    bool restricted = false;
    std::vector<int> mask;
    std::exception_ptr failure;
    std::thread starter([&] {
      try {
        restricted = cucascade::exec::thread_util::set_current_thread_affinity(
          std::vector<int>{allowed[0], allowed[1]});
        if (!restricted) { return; }
        reactor_placement placement;
        placement.gpu_id  = 0;
        placement.cpu_ids = std::move(cpu_ids);
        cucascade::io::uring::uring_reactor reactor{ctx, name, placement};
        reactor.start();
        mask = thread_affinity_named(name + "_worker");
        reactor.shutdown();
      } catch (...) {
        failure = std::current_exception();
      }
    });
    starter.join();
    if (failure) { std::rethrow_exception(failure); }
    REQUIRE(restricted);
    return mask;
  };

  // Pinned to the placement's CPUs the inherited mask allows: a1, not a2.
  CHECK(worker_mask("s4pin-a", {allowed[1], allowed[2]}) == std::vector<int>{allowed[1]});
  // No placement CPU is allowed: the worker keeps the inherited mask.
  CHECK(worker_mask("s4pin-b", {allowed[2]}) == std::vector<int>{allowed[0], allowed[1]});
}

TEST_CASE("an io_context_registry with a topology index places its reactors per GPU",
          "[io][reactor_placement][registry]")
{
  cucascade::memory::memory_reservation_manager manager{numa0_host_only()};
  auto const index = std::make_shared<cucascade::memory::topology_index const>(two_node_index());
  cucascade::io::io_config config;
  config.uring_reactors_per_gpu = 2;
  config.pin_reactor_threads    = false;

  cucascade::io::io_context_registry registry{config, manager, index};
  auto context = registry.make_ioctx(cucascade::io::io_context_type::uring);
  REQUIRE(context != nullptr);
  CHECK(dynamic_cast<cucascade::io::uring::uring_ioctx&>(*context).reactor_gauges().size() == 4);

  cucascade::io::io_context_registry unindexed{config, manager};
  auto unplaced = unindexed.make_ioctx(cucascade::io::io_context_type::uring);
  REQUIRE(unplaced != nullptr);
  CHECK(dynamic_cast<cucascade::io::uring::uring_ioctx&>(*unplaced).reactor_gauges().size() == 2);
}

TEST_CASE("an invalid spill policy is rejected", "[io][reactor_placement][registry]")
{
  cucascade::memory::memory_reservation_manager manager{numa0_host_only()};
  for (double const ratio : {0.0,
                             0.5,
                             -1.0,
                             std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity()}) {
    CAPTURE(ratio);
    cucascade::io::io_config config;
    config.reactor_spill.ratio = ratio;
    CHECK_THROWS_AS(cucascade::io::validate_spill_policy(config.reactor_spill),
                    std::invalid_argument);
    CHECK_THROWS_AS((cucascade::io::io_context_registry{config, manager}), std::invalid_argument);
  }
  cucascade::io::numa_spill_policy policy;
  policy.ratio = 1.0;
  CHECK_NOTHROW(cucascade::io::validate_spill_policy(policy));
}

TEST_CASE("placed reactors stage from the HOST space of their GPU's NUMA node",
          "[io][reactor_placement]")
{
  cucascade::memory::memory_reservation_manager manager{numa0_host_only()};
  auto* const numa0 = manager.get_memory_space(cucascade::memory::Tier::HOST, 0)
                        ->get_memory_resource_of<cucascade::memory::Tier::HOST>();
  REQUIRE(numa0 != nullptr);

  auto const index      = two_node_index();
  auto const placements = make_reactor_placements(1, &index, &manager, true);
  REQUIRE(placements.size() == 2);
  // GPU 0's node has a HOST space; GPU 1's (node 1) has none and keeps the shared staging.
  CHECK(placements[0].staging_mr == numa0);
  CHECK(placements[1].staging_mr == nullptr);
}

TEST_CASE("the uring factory runs its reactors per GPU of the topology index",
          "[io][reactor_placement][uring]")
{
  cucascade::memory::memory_reservation_manager manager{numa0_host_only()};
  auto const index = std::make_shared<cucascade::memory::topology_index const>(two_node_index());
  cucascade::io::io_config config;
  config.uring_reactors_per_gpu = 2;
  config.pin_reactor_threads    = false;

  auto placed = cucascade::io::make_uring_ioctx_factory(manager, index)(config);
  REQUIRE(placed != nullptr);
  auto& placed_uring = dynamic_cast<cucascade::io::uring::uring_ioctx&>(*placed);
  CHECK(placed_uring.reactor_gauges().size() == 4);

  // Without a topology index: one GPU's worth of unplaced reactors.
  auto unplaced = cucascade::io::make_uring_ioctx_factory(manager)(config);
  REQUIRE(unplaced != nullptr);
  CHECK(dynamic_cast<cucascade::io::uring::uring_ioctx&>(*unplaced).reactor_gauges().size() == 2);
}

#ifdef CUCASCADE_HAS_CUFILE
namespace {

/// Exposes the reactor pool so a test can name the reactors next_reactor picks.
class probed_cufile_ioctx : public cucascade::io::cufile::cufile_ioctx {
 public:
  using cufile_ioctx::cufile_ioctx;
  using templated_ioctx::_reactors;
};

}  // namespace

TEST_CASE("a vectored cuFile read fans out over the reactors of its GPU's NUMA node",
          "[io][reactor_placement][cufile]")
{
  cucascade::io::cufile::config cfg;
  cfg.compat_mode  = cucascade::io::cufile::compat::on;
  auto ctx         = std::make_shared<cucascade::io::cufile::cufile_reactor::reactor_context>(cfg);
  auto const index = two_node_index();
  cucascade::io::reactor_layout const layout{make_reactor_placements(2, &index, nullptr, false),
                                             cucascade::io::numa_spill_policy{}};
  probed_cufile_ioctx context{layout, ctx};
  auto object       = context.open_io_object("/proc/self/exe");
  auto const& typed = dynamic_cast<cucascade::io::cufile::cufile_io_object const&>(*object);

  auto const indices = [&](std::vector<cucascade::io::cufile::cufile_reactor*> const& picked) {
    std::vector<std::size_t> out;
    for (auto* reactor : picked) {
      auto const it = std::find_if(context._reactors.begin(),
                                   context._reactors.end(),
                                   [reactor](auto const& owned) { return owned.get() == reactor; });
      out.push_back(static_cast<std::size_t>(it - context._reactors.begin()));
    }
    std::ranges::sort(out);
    return out;
  };
  using op = probed_cufile_ioctx::io_op_type;

  // A device read of GPU 1 stays on GPU 1's two reactors, however many slices it has.
  CHECK(indices(context.next_reactor(typed, 8, op::host_vector_async, 1)) ==
        std::vector<std::size_t>{2, 3});
  // A host-only vectored read fans out over the whole pool.
  CHECK(indices(context.next_reactor(typed, 8, op::host_vector_async, -1)) ==
        std::vector<std::size_t>{0, 1, 2, 3});
  // A host-only vectored write still goes to one reactor.
  CHECK(context.next_reactor(typed, 8, op::write_vector_async, -1).size() == 1);
}
#endif
