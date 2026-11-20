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

#pragma once

#include <functional>
#include <istream>
#include <optional>
#include <string>
#include <vector>

namespace cucascade::memory {
/// One process CPU-placement snapshot. Hardware locality remains in topology_index.
struct cpu_constraints {
  /// Main-thread affinity when sampled, or caller affinity if the main thread is unavailable.
  /// This preserves taskset/numactl launch binding provided the main mask was not changed.
  std::vector<int> process_cpus;
  /// Effective cgroup-v2 allowance; nullopt when it cannot be established.
  std::optional<std::vector<int>> cgroup_cpus;
  /// Intersection of process_cpus and cgroup_cpus when available. Never empty.
  std::vector<int> allowed_cpus;
  bool used_caller_affinity = false;
  /// Fallback diagnostic, empty when main affinity and cgroup discovery both succeeded.
  std::string diagnostic;
};

/// Sorted, distinct intersection. An empty result is returned, never broadened.
[[nodiscard]] std::vector<int> intersect_cpu_affinity(std::vector<int> first,
                                                      std::vector<int> second);

/// Resolve a supplied process boundary against a deployment allowance; throws if empty.
[[nodiscard]] cpu_constraints resolve_cpu_constraints(std::vector<int> process_cpus,
                                                      std::optional<std::vector<int>> cgroup_cpus);

/// Parse Linux cgroup membership/mount mapping and read the nearest effective cpuset.
/// The reader returns nullopt only for a missing file and throws for unreadable/malformed data.
/// No proven mapping/controller returns nullopt; a known empty cpuset remains an empty vector.
using cpu_allowance_reader = std::function<std::optional<std::vector<int>>(std::string const&)>;
[[nodiscard]] std::optional<std::vector<int>> read_cgroup_cpu_allowance(
  std::istream& membership, std::istream& mounts, cpu_allowance_reader const& read);

/// Snapshot the process main thread's CPU mask and its cgroup-v2 allowance.
/// Unlike the current thread's mask, the main mask is not narrowed by an incidental worker
/// binding. This is not historical launch-state recovery: embedders must sample before they
/// intentionally change the main thread's mask. Discovery never changes affinity or memory
/// policy. Falls back to caller affinity if the main thread is unavailable, and to the process
/// mask if cgroup discovery is unavailable; both are reported in diagnostic. Throws when no
/// usable CPU boundary can be established. No CUDA/NVML initialization is required.
[[nodiscard]] cpu_constraints discover_cpu_constraints();
}  // namespace cucascade::memory
