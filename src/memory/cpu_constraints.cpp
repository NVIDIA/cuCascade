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

#include <cucascade/memory/cpu_constraints.hpp>
#include <cucascade/memory/net_topology.hpp>

#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <new>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace cucascade::memory {
namespace {
std::vector<int> read_affinity(pid_t tid)
{
  cpu_set_t mask;
  if (::sched_getaffinity(tid, sizeof(mask), &mask) != 0) return {};
  std::vector<int> cpus;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    if (CPU_ISSET(static_cast<std::size_t>(cpu), &mask)) cpus.push_back(cpu);
  return cpus;
}
std::optional<std::vector<int>> read_allowance_file(std::string const& path)
{
  errno = 0;
  std::ifstream file(path);
  if (!file) {
    if (errno == ENOENT) return std::nullopt;
    throw std::system_error(errno, std::generic_category(), "reading " + path);
  }
  std::string text;
  std::getline(file, text);
  if (file.bad()) throw std::runtime_error("reading " + path);
  if (text.empty()) return std::vector<int>{};
  auto cpus = parse_cpu_list(text);
  if (cpus.empty()) throw std::runtime_error("malformed CPU allowance in " + path);
  return cpus;
}
}  // namespace

std::vector<int> intersect_cpu_affinity(std::vector<int> first, std::vector<int> second)
{
  std::ranges::sort(first);
  std::ranges::sort(second);
  std::vector<int> result;
  std::set_intersection(
    first.begin(), first.end(), second.begin(), second.end(), std::back_inserter(result));
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

cpu_constraints resolve_cpu_constraints(std::vector<int> process_cpus,
                                        std::optional<std::vector<int>> cgroup_cpus)
{
  std::ranges::sort(process_cpus);
  process_cpus.erase(std::unique(process_cpus.begin(), process_cpus.end()), process_cpus.end());
  for (int cpu : process_cpus)
    if (cpu < 0 || cpu >= CPU_SETSIZE) throw std::invalid_argument("unsupported process CPU id");
  auto allowed = cgroup_cpus ? intersect_cpu_affinity(process_cpus, *cgroup_cpus) : process_cpus;
  if (allowed.empty()) throw std::runtime_error("process CPU boundary has no permitted CPUs");
  return {std::move(process_cpus), std::move(cgroup_cpus), std::move(allowed), false, {}};
}

// mountinfo escapes whitespace and backslashes as three octal digits.
std::string unescape_mount_path(std::string const& value)
{
  std::string result;
  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] == '\\' && i + 3 < value.size() &&
        value.find_first_not_of("01234567", i + 1) >= i + 4) {
      result += static_cast<char>(std::stoi(value.substr(i + 1, 3), nullptr, 8));
      i += 3;
    } else {
      result += value[i];
    }
  }
  return result;
}

// Resolve membership against the visible cgroup2 mount root, including bind mounts
// and namespace roots. Missing leaf controller files inherit the nearest ancestor's
// effective cpuset. An unreadable/malformed file must not authorize a broader ancestor.
std::optional<std::vector<int>> read_cgroup_cpu_allowance(std::istream& membership,
                                                          std::istream& mounts,
                                                          cpu_allowance_reader const& read)
{
  namespace fs = std::filesystem;
  auto valid   = [](fs::path const& path) {
    return path.is_absolute() && std::none_of(path.begin(), path.end(), [](auto const& part) {
             return part == ".." || part == ".";
           });
  };
  fs::path group;
  std::string line;
  while (std::getline(membership, line)) {
    if (line.starts_with("0::/")) {
      group = line.substr(3);
      break;
    }
  }
  if (!valid(group)) return std::nullopt;
  fs::path mount, relative;
  size_t specificity = 0;
  while (std::getline(mounts, line)) {
    auto separator = line.find(" - ");
    if (separator == std::string::npos) continue;
    std::istringstream suffix(line.substr(separator + 3));
    std::string type;
    if (!(suffix >> type) || type != "cgroup2") continue;
    std::istringstream fields(line.substr(0, separator));
    std::string id, parent, device, root_text, mount_text;
    if (!(fields >> id >> parent >> device >> root_text >> mount_text)) continue;
    fs::path root = unescape_mount_path(root_text), point = unescape_mount_path(mount_text);
    if (!valid(root) || !valid(point)) continue;
    auto within = group.lexically_relative(root);
    if (within.empty() || *within.begin() == "..") continue;
    if (root.native().size() >= specificity) {
      specificity = root.native().size();
      mount       = std::move(point);
      relative    = std::move(within);
    }
  }
  if (mount.empty()) return std::nullopt;  // No proven mapping: use inherited thread affinity.
  auto current = (mount / relative).lexically_normal();
  for (;;) {
    if (auto cpus = read((current / "cpuset.cpus.effective").string())) return cpus;
    if (current == mount) return std::nullopt;
    current = current.parent_path();
  }
}

cpu_constraints discover_cpu_constraints()
{
  auto process        = read_affinity(::getpid());
  bool const fallback = process.empty();
  if (fallback) process = read_affinity(0);
  std::optional<std::vector<int>> cgroup;
  std::string diagnostic;
  try {
    std::ifstream membership("/proc/self/cgroup"), mounts("/proc/self/mountinfo");
    cgroup = read_cgroup_cpu_allowance(membership, mounts, read_allowance_file);
  } catch (std::bad_alloc const&) {
    throw;
  } catch (std::exception const& error) {
    diagnostic = error.what();
  }
  if (!cgroup) {
    if (!diagnostic.empty()) diagnostic += "; ";
    diagnostic += "cgroup CPU allowance unavailable; using process affinity";
  }
  if (fallback) {
    if (!diagnostic.empty()) diagnostic += "; ";
    diagnostic += "main-thread affinity unavailable; using caller affinity";
  }
  auto result                 = resolve_cpu_constraints(std::move(process), std::move(cgroup));
  result.used_caller_affinity = fallback;
  result.diagnostic           = std::move(diagnostic);
  return result;
}
}  // namespace cucascade::memory
