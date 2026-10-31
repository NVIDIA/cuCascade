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

#include "io/rest/reactor_placement.hpp"

#include <cucascade/exec/thread_util.hpp>
#include <cucascade/memory/net_topology.hpp>

#include <catch2/catch_all.hpp>
#include <sys/resource.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

using namespace cucascade::io::rest;
using cucascade::memory::nic_queue_irq;
using cucascade::memory::nic_topology;
namespace tu = cucascade::exec::thread_util;

namespace {

std::vector<int> iota_cpus(int first, int count)
{
  std::vector<int> v(static_cast<std::size_t>(count));
  std::iota(v.begin(), v.end(), first);
  return v;
}

/// A NIC whose queue q's IRQ is delivered to @p per_queue_cpus[q], on a node of
/// @p node_cpus.
nic_topology make_nic(std::vector<std::vector<int>> const& per_queue_cpus,
                      std::vector<int> node_cpus)
{
  nic_topology nic;
  nic.interface   = "ens5";
  nic.pci_address = "0000:00:05.0";
  nic.numa_node   = 0;
  nic.mtu         = 9001;
  nic.rx_queues   = static_cast<unsigned>(per_queue_cpus.size());
  nic.tx_queues   = nic.rx_queues;
  std::vector<int> all;
  for (std::size_t q = 0; q < per_queue_cpus.size(); ++q) {
    nic.queue_irqs.push_back(nic_queue_irq{
      100 + static_cast<int>(q), static_cast<int>(q), per_queue_cpus[q], per_queue_cpus[q]});
    all.insert(all.end(), per_queue_cpus[q].begin(), per_queue_cpus[q].end());
  }
  std::sort(all.begin(), all.end());
  all.erase(std::unique(all.begin(), all.end()), all.end());
  nic.irq_cpus  = all;
  nic.node_cpus = std::move(node_cpus);
  return nic;
}

/// 8 queues, IRQ q on CPU q, node0 = 0-47.
nic_topology steered_nic()
{
  std::vector<std::vector<int>> per_queue;
  for (int q = 0; q < 8; ++q) {
    per_queue.push_back({q});
  }
  return make_nic(per_queue, iota_cpus(0, 48));
}

nic_topology spread_nic()
{
  return make_nic(std::vector<std::vector<int>>(8, iota_cpus(0, 48)), iota_cpus(0, 48));
}

std::optional<rlim_t> soft_limit(int resource)
{
  rlimit lim{};
  if (::getrlimit(resource, &lim) != 0) { return std::nullopt; }
  return lim.rlim_cur;
}

bool has_cap_sys_nice()
{
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("CapEff:", 0) == 0) {
      return ((std::stoull(line.substr(7), nullptr, 16) >> 23) & 1ULL) != 0;
    }
  }
  return false;
}

}  // namespace

TEST_CASE("auto sizing follows steered IRQs", "[rest_pool_plan]")
{
  auto const all  = iota_cpus(0, 96);
  auto const plan = plan_rest_pool(0, true, steered_nic(), all);
  CHECK(plan.n_reactors == 8);
  CHECK(plan.pin_cpus == iota_cpus(0, 8));
  CHECK(plan.from_irqs);
  CHECK_FALSE(plan.configured);
  CHECK_THAT(plan.reason, Catch::Matchers::ContainsSubstring("steered"));

  SECTION("pin_to_irq_cpus off keeps the count, drops the pins")
  {
    auto const unpinned = plan_rest_pool(0, false, steered_nic(), all);
    CHECK(unpinned.n_reactors == 8);
    CHECK(unpinned.pin_cpus.empty());
    CHECK(unpinned.from_irqs);
    CHECK_THAT(unpinned.reason, Catch::Matchers::ContainsSubstring("pin_to_irq_cpus off"));
  }
}

TEST_CASE("auto sizing without steering falls back to the default count", "[rest_pool_plan]")
{
  auto const all = iota_cpus(0, 96);
  for (auto const& nic :
       {std::optional<nic_topology>{spread_nic()}, std::optional<nic_topology>{std::nullopt}}) {
    auto const plan = plan_rest_pool(0, true, nic, all);
    CHECK(plan.n_reactors == default_rest_n_reactors);
    CHECK(plan.pin_cpus.empty());
    CHECK_FALSE(plan.from_irqs);
    CHECK_FALSE(plan.configured);
  }

  // Two-CPU IRQs are spread at the default threshold and steered at 2.
  std::vector<std::vector<int>> pairs;
  for (int q = 0; q < 4; ++q) {
    pairs.push_back({2 * q, 2 * q + 1});
  }
  auto const nic = make_nic(pairs, iota_cpus(0, 48));
  CHECK(plan_rest_pool(0, true, nic, all).n_reactors == default_rest_n_reactors);
  auto const at2 = plan_rest_pool(0, true, nic, all, 2);
  CHECK(at2.n_reactors == 8);
  CHECK(at2.pin_cpus == iota_cpus(0, 8));
}

TEST_CASE("a configured count is pinned round-robin only when steered", "[rest_pool_plan]")
{
  auto const all = iota_cpus(0, 96);

  auto const twelve = plan_rest_pool(12, true, steered_nic(), all);
  CHECK(twelve.n_reactors == 12);
  CHECK(twelve.pin_cpus == std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3});
  CHECK(twelve.configured);
  CHECK_FALSE(twelve.from_irqs);

  auto const three = plan_rest_pool(3, true, steered_nic(), all);
  CHECK(three.n_reactors == 3);
  CHECK(three.pin_cpus == std::vector<int>{0, 1, 2});

  auto const off = plan_rest_pool(4, false, steered_nic(), all);
  CHECK(off.n_reactors == 4);
  CHECK(off.pin_cpus.empty());

  auto const spread = plan_rest_pool(4, true, spread_nic(), all);
  CHECK(spread.n_reactors == 4);
  CHECK(spread.pin_cpus.empty());
  CHECK(spread.configured);

  auto const unknown = plan_rest_pool(5, true, std::nullopt, all);
  CHECK(unknown.n_reactors == 5);
  CHECK(unknown.pin_cpus.empty());
}

TEST_CASE("pins never leave the inherited affinity", "[rest_pool_plan]")
{
  std::vector<int> const narrow{2, 3, 4, 100};

  auto const autosized = plan_rest_pool(0, true, steered_nic(), narrow);
  CHECK(autosized.n_reactors == 3);
  CHECK(autosized.pin_cpus == std::vector<int>{2, 3, 4});
  CHECK(autosized.from_irqs);

  auto const configured = plan_rest_pool(5, true, steered_nic(), narrow);
  CHECK(configured.n_reactors == 5);
  CHECK(configured.pin_cpus == std::vector<int>{2, 3, 4, 2, 3});

  std::vector<int> const disjoint{40, 41};
  auto const auto_disjoint = plan_rest_pool(0, true, steered_nic(), disjoint);
  CHECK(auto_disjoint.n_reactors == 8);
  CHECK(auto_disjoint.pin_cpus.empty());
  CHECK(auto_disjoint.from_irqs);
  CHECK_THAT(auto_disjoint.reason, Catch::Matchers::ContainsSubstring("inherited affinity"));

  auto const configured_disjoint = plan_rest_pool(3, true, steered_nic(), disjoint);
  CHECK(configured_disjoint.n_reactors == 3);
  CHECK(configured_disjoint.pin_cpus.empty());

  // An unreadable affinity (empty) pins nothing.
  CHECK(plan_rest_pool(0, true, steered_nic(), {}).pin_cpus.empty());
}

TEST_CASE("the convenience overload uses the calling thread's affinity", "[rest_pool_plan]")
{
  auto const mine = tu::get_current_thread_affinity();
  REQUIRE_FALSE(mine.empty());
  auto const plan = plan_rest_pool(0, true, steered_nic());
  for (int cpu : plan.pin_cpus) {
    CHECK(std::find(mine.begin(), mine.end(), cpu) != mine.end());
  }
}

TEST_CASE("raise_reactor_priority degrades gracefully", "[rest_pool_plan]")
{
  reactor_priority got;
  std::optional<tu::scheduling_info> sched_after;
  std::optional<int> nice_before;
  std::optional<int> nice_after;
  // On a fresh thread: whatever is achieved must not leak into the test runner.
  std::thread([&] {
    nice_before = tu::get_current_thread_niceness();
    got         = raise_reactor_priority();
    sched_after = tu::get_current_thread_scheduling();
    nice_after  = tu::get_current_thread_niceness();
  }).join();

  std::cout << "[rest_pool_plan] reactor priority on this host: " << got.description << "\n";
  REQUIRE(sched_after.has_value());
  switch (got.achieved) {
    case reactor_priority::kind::realtime_fifo:
      CHECK(sched_after->policy == tu::scheduling_policy::fifo);
      CHECK(sched_after->priority == got.value);
      CHECK(got.description == "SCHED_FIFO " + std::to_string(got.value));
      break;
    case reactor_priority::kind::nice:
      CHECK(nice_after == got.value);
      CHECK(got.description == "nice " + std::to_string(got.value));
      break;
    case reactor_priority::kind::unchanged:
      CHECK(sched_after->policy == tu::scheduling_policy::other);
      CHECK(nice_after == nice_before);
      CHECK_THAT(got.description, Catch::Matchers::StartsWith("default ("));
      break;
  }

  // Without CAP_SYS_NICE and with no RT / nice allowance nothing can change.
  bool const unprivileged =
    !has_cap_sys_nice() && soft_limit(RLIMIT_RTPRIO) == rlim_t{0} && nice_before.has_value() &&
    20 - static_cast<int>(soft_limit(RLIMIT_NICE).value_or(0)) >= *nice_before;
  if (unprivileged) {
    CHECK(got.achieved == reactor_priority::kind::unchanged);
    CHECK_THAT(got.description, Catch::Matchers::ContainsSubstring("no CAP_SYS_NICE"));
  }
}

TEST_CASE("place_current_reactor_thread pins within the plan", "[rest_pool_plan]")
{
  auto const mine = tu::get_current_thread_affinity();
  REQUIRE_FALSE(mine.empty());

  rest_pool_plan plan;
  plan.n_reactors = 2;
  plan.pin_cpus   = {mine.front(), mine.back()};

  reactor_placement placed;
  std::vector<int> affinity_after;
  std::thread([&] {
    placed         = place_current_reactor_thread(plan, 1);
    affinity_after = tu::get_current_thread_affinity();
  }).join();
  CHECK(placed.pinned);
  CHECK(placed.cpu == mine.back());
  CHECK(affinity_after == std::vector<int>{mine.back()});
  CHECK_FALSE(placed.priority.description.empty());

  rest_pool_plan const unpinned{2, {}, false, false, "test"};
  std::thread([&] {
    placed         = place_current_reactor_thread(unpinned, 0);
    affinity_after = tu::get_current_thread_affinity();
  }).join();
  CHECK_FALSE(placed.pinned);
  CHECK(placed.cpu == -1);
  CHECK(placed.priority.achieved == reactor_priority::kind::unchanged);
  CHECK(affinity_after == mine);
}

TEST_CASE("place_current_reactor_thread never widens the worker's own affinity", "[rest_pool_plan]")
{
  // The plan was made on a thread allowed every CPU; the worker inherited only one.  A pin to a
  // CPU outside the worker's mask must be refused, not applied.
  auto const mine = tu::get_current_thread_affinity();
  if (mine.size() < 2) { SKIP("needs two CPUs in the affinity mask"); }
  rest_pool_plan plan;
  plan.n_reactors = 1;
  plan.pin_cpus   = {mine.back()};
  reactor_placement placed;
  std::vector<int> affinity_after;
  bool narrowed = false;
  std::thread([&] {
    std::array<int, 1> const narrow{mine.front()};
    narrowed       = tu::set_current_thread_affinity(std::span<int const>{narrow});
    placed         = place_current_reactor_thread(plan, 0);
    affinity_after = tu::get_current_thread_affinity();
  }).join();
  REQUIRE(narrowed);
  CHECK_FALSE(placed.pinned);
  CHECK(placed.cpu == -1);
  CHECK(affinity_after == std::vector<int>{mine.front()});
}

TEST_CASE("describe_rest_pool renders the startup line", "[rest_pool_plan]")
{
  auto const nic  = std::optional<nic_topology>{steered_nic()};
  auto const plan = plan_rest_pool(0, true, nic, iota_cpus(0, 48));
  reactor_priority prio;
  prio.description = "nice -20";
  auto const line  = describe_rest_pool(plan, nic, prio);
  CHECK_THAT(line,
             Catch::Matchers::StartsWith("rest: 8 reactors (from_irqs), nic ens5 numa 0 mtu 9001 "
                                         "queues 8 irqs 100-107 on cpus 0-7 pinned 0-7 prio "
                                         "nice -20; "));

  auto const fallback =
    describe_rest_pool(plan_rest_pool(0, true, std::nullopt, {}), std::nullopt, reactor_priority{});
  CHECK_THAT(fallback,
             Catch::Matchers::StartsWith("rest: 2 reactors (default), nic unknown pinned no"));
}

TEST_CASE("pool plan on this host (informational)", "[rest_pool_plan]")
{
  namespace mem    = cucascade::memory;
  auto const iface = mem::select_egress_interface("");
  auto const nic   = iface ? mem::discover_nic(*iface) : std::nullopt;
  auto const plan  = plan_rest_pool(0, true, nic);
  reactor_placement placed;
  std::thread([&] { placed = place_current_reactor_thread(plan, 0); }).join();
  std::cout << "[rest_pool_plan] " << describe_rest_pool(plan, nic, placed.priority) << "\n";
  CHECK(plan.n_reactors >= 1);
}
