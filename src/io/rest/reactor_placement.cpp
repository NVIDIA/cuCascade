/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "io/rest/reactor_placement.hpp"

#include <cucascade/exec/thread_util.hpp>
#include <cucascade/log/logging.hpp>

#include <sched.h>
#include <sys/resource.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <exception>
#include <fstream>
#include <span>
#include <sstream>
#include <string_view>

namespace cucascade::io::rest {

namespace {

namespace tu = cucascade::exec::thread_util;

constexpr int cap_sys_nice_bit = 23;  // linux/capability.h CAP_SYS_NICE

/// Whether the calling thread's effective capability set holds CAP_SYS_NICE.
bool has_cap_sys_nice() noexcept
{
  try {
    for (char const* path : {"/proc/thread-self/status", "/proc/self/status"}) {
      std::ifstream in(path);
      if (!in) { continue; }
      std::string line;
      while (std::getline(in, line)) {
        constexpr std::string_view key = "CapEff:";
        if (line.compare(0, key.size(), key) != 0) { continue; }
        auto const caps = std::stoull(line.substr(key.size()), nullptr, 16);
        return ((caps >> cap_sys_nice_bit) & 1ULL) != 0;
      }
    }
  } catch (...) {
    return false;
  }
  return false;
}

/// The soft limit of @p resource, @c nullopt when unreadable; RLIM_INFINITY is
/// returned as-is.
std::optional<rlim_t> soft_limit(int resource) noexcept
{
  rlimit lim{};
  if (::getrlimit(resource, &lim) != 0) { return std::nullopt; }
  return lim.rlim_cur;
}

std::vector<int> intersect_in_order(std::vector<int> const& cpus, std::vector<int> const& allowed)
{
  std::vector<int> out;
  for (int c : cpus) {
    if (std::find(allowed.begin(), allowed.end(), c) != allowed.end()) { out.push_back(c); }
  }
  return out;
}

void warn_once(std::atomic<bool>& flag, std::string const& msg)
{
  if (!flag.exchange(true)) { CUCASCADE_LOG_WARN("{}", msg); }
}

std::atomic<bool> g_warned_empty_intersection{false};
std::atomic<bool> g_warned_priority{false};

}  // namespace

rest_pool_plan plan_rest_pool(std::size_t configured_n_reactors,
                              bool pin_to_irq_cpus,
                              std::optional<memory::nic_topology> const& nic,
                              std::vector<int> const& allowed_cpus,
                              std::size_t steered_max_cpus_per_irq)
{
  rest_pool_plan plan;
  bool const steered = nic.has_value() && nic->irqs_steered(steered_max_cpus_per_irq);
  std::string const nic_state =
    !nic.has_value()
      ? "no NIC information"
      : (steered ? "nic " + nic->interface + " IRQs steered to CPUs " +
                     memory::format_cpu_list(nic->irq_cpus)
                 : "nic " + nic->interface + " IRQs not steered to a strict subset of numa node " +
                     std::to_string(nic->numa_node) + " CPUs");

  if (!steered) {
    plan.configured = configured_n_reactors > 0;
    plan.n_reactors = plan.configured ? configured_n_reactors : default_rest_n_reactors;
    plan.reason =
      nic_state + (plan.configured ? "; configured count" : "; default count") + ", unpinned";
    return plan;
  }

  auto const usable = intersect_in_order(nic->irq_cpus, allowed_cpus);
  bool const pin    = pin_to_irq_cpus && !usable.empty();
  std::string why_unpinned;
  if (!pin_to_irq_cpus) {
    why_unpinned = "pin_to_irq_cpus off";
  } else if (usable.empty()) {
    why_unpinned =
      "IRQ CPUs outside the inherited affinity " + memory::format_cpu_list(allowed_cpus);
    warn_once(g_warned_empty_intersection, "rest: " + why_unpinned + "; reactors left unpinned");
  }

  if (configured_n_reactors == 0) {
    plan.from_irqs  = true;
    plan.n_reactors = pin ? usable.size() : nic->irq_cpus.size();
    if (pin) { plan.pin_cpus = usable; }
    plan.reason = nic_state + "; one reactor per IRQ CPU";
    if (pin && usable.size() < nic->irq_cpus.size()) {
      plan.reason += " within the inherited affinity " + memory::format_cpu_list(allowed_cpus);
    }
  } else {
    plan.configured = true;
    plan.n_reactors = configured_n_reactors;
    if (pin) {
      plan.pin_cpus.reserve(configured_n_reactors);
      for (std::size_t i = 0; i < configured_n_reactors; ++i) {
        plan.pin_cpus.push_back(usable[i % usable.size()]);
      }
    }
    plan.reason = nic_state + "; configured count" + (pin ? ", pinned round-robin" : "");
  }
  plan.reason += pin ? "" : ", unpinned (" + why_unpinned + ")";
  return plan;
}

rest_pool_plan plan_rest_pool(std::size_t configured_n_reactors,
                              bool pin_to_irq_cpus,
                              std::optional<memory::nic_topology> const& nic)
{
  return plan_rest_pool(
    configured_n_reactors, pin_to_irq_cpus, nic, tu::get_current_thread_affinity());
}

reactor_priority raise_reactor_priority() noexcept
{
  reactor_priority result;
  try {
    bool const cap      = has_cap_sys_nice();
    auto const rt_limit = soft_limit(RLIMIT_RTPRIO).value_or(0);

    // SCHED_FIFO, at the top priority CAP_SYS_NICE or RLIMIT_RTPRIO allows.
    int const fifo_max = ::sched_get_priority_max(SCHED_FIFO);
    if (fifo_max > 0 && (cap || rt_limit > 0)) {
      int const prio =
        cap || rt_limit == RLIM_INFINITY
          ? fifo_max
          : static_cast<int>(std::min<rlim_t>(rt_limit, static_cast<rlim_t>(fifo_max)));
      if (tu::set_current_thread_scheduling(tu::scheduling_policy::fifo, prio)) {
        result.achieved    = reactor_priority::kind::realtime_fifo;
        result.value       = prio;
        result.description = "SCHED_FIFO " + std::to_string(prio);
        return result;
      }
    }

    // The lowest nice value permitted: -20 with CAP_SYS_NICE, else 20 - RLIMIT_NICE.
    auto const nice_limit = soft_limit(RLIMIT_NICE).value_or(0);
    int target            = 20;
    if (cap || nice_limit == RLIM_INFINITY || nice_limit >= 40) {
      target = -20;
    } else {
      target = 20 - static_cast<int>(nice_limit);
    }
    target            = std::clamp(target, -20, 19);
    auto const before = tu::get_current_thread_niceness();
    if (before && target < *before && tu::set_current_thread_niceness(target)) {
      result.achieved    = reactor_priority::kind::nice;
      result.value       = target;
      result.description = "nice " + std::to_string(target);
      return result;
    }

    std::ostringstream why;
    why << "default (" << (cap ? "CAP_SYS_NICE" : "no CAP_SYS_NICE") << ", RLIMIT_RTPRIO "
        << (rt_limit == RLIM_INFINITY ? std::string{"unlimited"} : std::to_string(rt_limit))
        << ", RLIMIT_NICE "
        << (nice_limit == RLIM_INFINITY ? std::string{"unlimited"} : std::to_string(nice_limit))
        << ")";
    result.description = why.str();
    warn_once(g_warned_priority, "rest: reactor priority left at " + result.description);
  } catch (...) {
    result             = reactor_priority{};
    result.description = "default";
  }
  return result;
}

reactor_placement place_current_reactor_thread(rest_pool_plan const& plan,
                                               std::size_t reactor_index) noexcept
{
  reactor_placement placed;
  if (reactor_index < plan.pin_cpus.size()) {
    std::array<int, 1> const cpu{plan.pin_cpus[reactor_index]};
    // The plan was intersected with the planning thread's affinity; this thread may have
    // inherited a narrower one (started from another thread).  Never widen it: a CPU outside
    // this thread's own mask leaves it unpinned.
    auto const inherited = tu::get_current_thread_affinity();
    bool const allowed   = std::find(inherited.begin(), inherited.end(), cpu[0]) != inherited.end();
    if (allowed && tu::set_current_thread_affinity(std::span<int const>{cpu})) {
      placed.cpu    = cpu[0];
      placed.pinned = true;
    }
  }
  if (placed.pinned) {
    placed.priority = raise_reactor_priority();
  } else {
    placed.priority.description = "default (unpinned)";
  }
  return placed;
}

std::string describe_rest_pool(rest_pool_plan const& plan,
                               std::optional<memory::nic_topology> const& nic,
                               reactor_priority const& priority)
{
  std::ostringstream out;
  out << "rest: " << plan.n_reactors << " reactors ("
      << (plan.from_irqs ? "from_irqs" : (plan.configured ? "configured" : "default")) << "), ";
  if (nic) {
    out << "nic " << nic->interface << " numa " << nic->numa_node << " mtu " << nic->mtu
        << " queues " << nic->rx_queues << " irqs ";
    std::vector<int> irqs;
    irqs.reserve(nic->queue_irqs.size());
    for (auto const& q : nic->queue_irqs) {
      irqs.push_back(q.irq);
    }
    out << (irqs.empty() ? std::string{"none"} : memory::format_cpu_list(irqs)) << " on cpus "
        << (nic->irq_cpus.empty() ? std::string{"none"} : memory::format_cpu_list(nic->irq_cpus));
  } else {
    out << "nic unknown";
  }
  out << " pinned "
      << (plan.pin_cpus.empty() ? std::string{"no"} : memory::format_cpu_list(plan.pin_cpus))
      << " prio " << (priority.description.empty() ? "default" : priority.description) << "; "
      << plan.reason;
  return out.str();
}

}  // namespace cucascade::io::rest
