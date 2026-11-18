/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cucascade/io/cufile/config.hpp>

#include <optional>
#include <string>
#include <vector>

namespace cucascade::io::cufile {
struct affinity_cpu {
  int id{}, node{-1}, package{-1}, core{-1};
};
struct affinity_nic {
  std::string name, bdf;
  int node{-1};
};
struct affinity_topology {
  std::string gpu_bdf;
  int gpu_node{-1};
  std::vector<affinity_cpu> cpus;
  std::vector<int> permitted_cpus;
  std::vector<affinity_nic> nics;
  std::string reason;
};
struct affinity_decision {
  int cpu{-1};
  int node{-1};
  std::string nic, nic_bdf, reason;
  bool pinned{false};
};
struct affinity_plan {
  affinity_decision reaper;
  std::vector<affinity_decision> submitters;
};

/** @brief Passive sysfs discovery; injected CPUs replace sched_getaffinity for fixture tests. */
[[nodiscard]] affinity_topology discover_affinity_topology(
  std::string gpu_bdf,
  std::string const& sysfs_root                         = "/sys",
  std::optional<std::vector<int>> const& permitted_cpus = std::nullopt);
/** @brief Deterministic planner. Used cores are shared across plans/engines by the caller. */
[[nodiscard]] affinity_plan plan_affinity(affinity_topology const&,
                                          unsigned submitters,
                                          std::optional<std::string> const& nic = {},
                                          std::vector<int> const& cpus          = {},
                                          std::vector<affinity_cpu>* used_cores = nullptr,
                                          bool include_reaper                   = true);
/** @brief Changes ONLY the calling engine-owned thread; failure is diagnostic, never fatal. */
void apply_thread_affinity(affinity_decision&, std::vector<int> const& restore_cpus = {}) noexcept;
}  // namespace cucascade::io::cufile
