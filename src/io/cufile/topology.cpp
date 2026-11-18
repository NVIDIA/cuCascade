/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cucascade/io/cufile/topology.hpp>
#include <cucascade/memory/net_topology.hpp>

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <set>

namespace cucascade::io::cufile {
namespace {
int number(std::filesystem::path const& path)
{
  int value{-1};
  std::ifstream input(path);
  if (!(input >> value)) return -1;
  std::string trailing;
  if (input >> trailing) return -1;
  return value;
}
std::vector<int> cpu_list(std::filesystem::path const& path)
{
  std::ifstream input(path);
  std::string value;
  input >> value;
  return memory::parse_cpu_list(value);
}
bool same_core(affinity_cpu const& a, affinity_cpu const& b)
{
  return a.package == b.package && a.core == b.core;
}
std::string lower(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}
}  // namespace

affinity_topology discover_affinity_topology(std::string gpu_bdf,
                                             std::string const& root,
                                             std::optional<std::vector<int>> const& permitted_cpus)
{
  affinity_topology result;
  result.gpu_bdf = lower(std::move(gpu_bdf));
  try {
    if (result.gpu_bdf.empty()) {
      result.reason = "GPU PCI BDF unavailable";
      return result;
    }
    auto base       = std::filesystem::path(root);
    result.gpu_node = number(base / "bus/pci/devices" / result.gpu_bdf / "numa_node");
    if (result.gpu_node < 0) {
      result.reason = "GPU NUMA node unavailable";
      return result;
    }
    std::vector<int> allowed;
    if (permitted_cpus)
      allowed = *permitted_cpus;
    else {
      // Dynamic masks support hosts with more than CPU_SETSIZE CPUs.
      std::size_t count = 128;
      for (; count <= (1U << 20); count *= 2) {
        allowed.reserve(count);  // No allocation may throw while a raw CPU mask is live.
        auto* mask = CPU_ALLOC(count);
        if (!mask) break;
        auto bytes = CPU_ALLOC_SIZE(count);
        CPU_ZERO_S(bytes, mask);
        int rc = sched_getaffinity(0, bytes, mask);
        if (rc == 0)
          for (std::size_t i = 0; i < count; ++i)
            if (CPU_ISSET_S(i, bytes, mask)) allowed.push_back(static_cast<int>(i));
        CPU_FREE(mask);
        if (rc == 0) break;
      }
    }
    result.permitted_cpus = allowed;
    std::set<int> permitted(allowed.begin(), allowed.end());
    auto nodes = base / "devices/system/node";
    for (auto const& node : std::filesystem::directory_iterator(nodes)) {
      auto name = node.path().filename().string();
      if (!name.starts_with("node") || name.size() <= 4 ||
          !std::all_of(
            name.begin() + 4, name.end(), [](unsigned char c) { return std::isdigit(c); }))
        continue;
      int id = std::stoi(name.substr(4));
      for (int cpu : cpu_list(node.path() / "cpulist")) {
        if (!permitted.contains(cpu)) continue;
        auto topology = base / "devices/system/cpu" / ("cpu" + std::to_string(cpu)) / "topology";
        auto package  = number(topology / "physical_package_id");
        auto core     = number(topology / "core_id");
        if (package >= 0 && core >= 0) result.cpus.push_back({cpu, id, package, core});
      }
    }
    std::sort(result.cpus.begin(), result.cpus.end(), [](auto const& a, auto const& b) {
      return a.id < b.id;
    });
    auto ib = base / "class/infiniband";
    if (std::filesystem::exists(ib))
      for (auto const& nic : std::filesystem::directory_iterator(ib)) {
        auto device = nic.path() / "device";
        std::error_code error;
        auto canonical = std::filesystem::canonical(device, error);
        if (error) continue;  // One disappeared/broken NIC must not discard healthy topology.
        auto node = number(device / "numa_node");
        if (node < 0) continue;
        result.nics.push_back(
          {nic.path().filename().string(), canonical.filename().string(), node});
      }
    std::sort(result.nics.begin(), result.nics.end(), [&](auto const& a, auto const& b) {
      if ((a.node == result.gpu_node) != (b.node == result.gpu_node))
        return a.node == result.gpu_node;
      return a.name < b.name;
    });
    result.reason =
      result.cpus.empty() ? "no permitted CPUs with physical-core topology" : "discovered";
  } catch (std::exception const&) {
    result.reason = "sysfs topology incomplete or malformed";
    result.cpus.clear();
  }
  return result;
}

affinity_plan plan_affinity(affinity_topology const& topology,
                            unsigned submitters,
                            std::optional<std::string> const& nic,
                            std::vector<int> const& cpus,
                            std::vector<affinity_cpu>* used_cores,
                            bool include_reaper)
{
  affinity_plan plan;
  plan.submitters.resize(submitters);
  std::vector<affinity_cpu> local;
  auto& used = used_cores ? *used_cores : local;
  auto pick  = [&](int node, std::vector<int> const& restrict_cpus, std::string reason) {
    affinity_decision result;
    result.node   = node;
    result.reason = std::move(reason);
    if (node < 0 || topology.cpus.empty()) {
      result.reason = topology.reason;
      return result;
    }
    std::vector<affinity_cpu const*> candidates;
    for (auto const& cpu : topology.cpus)
      if ((cpu.node == node || !restrict_cpus.empty()) &&
          (restrict_cpus.empty() ||
           std::find(restrict_cpus.begin(), restrict_cpus.end(), cpu.id) != restrict_cpus.end()))
        candidates.push_back(&cpu);
    if (candidates.empty()) {
      result.reason = "no permitted CPU on selected node / override";
      return result;
    }
    auto found = std::find_if(candidates.begin(), candidates.end(), [&](auto cpu) {
      return std::none_of(
        used.begin(), used.end(), [&](auto const& u) { return same_core(u, *cpu); });
    });
    if (found == candidates.end()) {
      // All physical cores used: prefer an unused SMT sibling before sharing a logical CPU.
      found = std::find_if(candidates.begin(), candidates.end(), [&](auto cpu) {
        return std::none_of(
          used.begin(), used.end(), [&](auto const& u) { return u.id == cpu->id; });
      });
      if (found == candidates.end()) found = candidates.begin();
      result.reason += "; physical cores exhausted (SMT/shared CPU)";
    }
    result.cpu  = (*found)->id;
    result.node = (*found)->node;
    used.push_back(**found);
    return result;
  };
  if (include_reaper) plan.reaper = pick(topology.gpu_node, {}, "GPU-local reaper");
  auto selected = topology.nics.end();
  if (nic)
    selected = std::find_if(topology.nics.begin(), topology.nics.end(), [&](auto const& v) {
      return v.name == *nic || lower(v.bdf) == lower(*nic);
    });
  else if (!topology.nics.empty())
    selected = topology.nics.begin();
  for (auto& decision : plan.submitters) {
    if (nic && selected == topology.nics.end()) {
      decision.reason = "requested NIC topology unavailable";
      continue;
    }
    int node = selected == topology.nics.end() ? topology.gpu_node : selected->node;
    decision = pick(
      node,
      cpus,
      selected == topology.nics.end() ? "no RDMA NIC; GPU-node fallback" : "NIC-local submitter");
    if (decision.cpu < 0 && cpus.empty() && selected != topology.nics.end() &&
        topology.gpu_node >= 0)
      decision = pick(topology.gpu_node, {}, "no permitted NIC-local core; GPU-node fallback");
    if (selected != topology.nics.end()) {
      decision.nic     = selected->name;
      decision.nic_bdf = selected->bdf;
    }
  }
  return plan;
}

void apply_thread_affinity(affinity_decision& decision,
                           std::vector<int> const& restore_cpus) noexcept
{
  if (decision.cpu < 0 && restore_cpus.empty()) return;
  auto max_cpu =
    decision.cpu >= 0 ? decision.cpu : *std::max_element(restore_cpus.begin(), restore_cpus.end());
  if (max_cpu < 0) return;
  auto count = static_cast<std::size_t>(max_cpu) + 1;
  auto* mask = CPU_ALLOC(count);
  if (!mask) {
    try {
      decision.reason = "CPU mask allocation failed";
    } catch (...) {
    }
    return;
  }
  auto bytes = CPU_ALLOC_SIZE(count);
  CPU_ZERO_S(bytes, mask);
  if (decision.cpu >= 0)
    CPU_SET_S(static_cast<std::size_t>(decision.cpu), bytes, mask);
  else
    for (int cpu : restore_cpus)
      if (cpu >= 0) CPU_SET_S(static_cast<std::size_t>(cpu), bytes, mask);
  auto rc = pthread_setaffinity_np(pthread_self(), bytes, mask);
  CPU_FREE(mask);
  decision.pinned = rc == 0 && decision.cpu >= 0;
  if (rc != 0) {
    try {
      decision.reason = "thread pin rejected (cpuset changed or CPU offline)";
    } catch (...) {
    }
  }
}
}  // namespace cucascade::io::cufile
