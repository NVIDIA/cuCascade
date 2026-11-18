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

#pragma once

#include <sched.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <climits>
#include <cstddef>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace cucascade {
namespace memory {
namespace detail {

/// A set of CPU or node ids in the kernel's bitmask layout: an array of unsigned longs.
class kernel_bitmask {
 public:
  /// The calling thread's CPU affinity, or nullopt if it cannot be read.
  static std::optional<kernel_bitmask> thread_affinity(
    int (*get_affinity)(pid_t, std::size_t, cpu_set_t*) = ::sched_getaffinity)
  {
    // The kernel rejects masks smaller than its CPU count with EINVAL.
    for (std::size_t bits = 1024; bits <= (std::size_t{1} << 22); bits *= 2) {
      kernel_bitmask mask;
      mask._words.resize(bits / bits_per_word);
      if (get_affinity(0, mask.bytes(), mask.cpu_set()) == 0) { return mask; }
      if (errno != EINVAL) { break; }
    }
    return std::nullopt;
  }

  /// The CPUs of NUMA node @p node, or nullopt if they cannot be read.
  static std::optional<kernel_bitmask> node_cpus(int node)
  {
    std::ifstream file("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
    std::string list;
    if (!std::getline(file, list)) { return std::nullopt; }
    return parse_list(list);
  }

  /// Parses a kernel id list such as "0-3,8,10-11".
  static std::optional<kernel_bitmask> parse_list(std::string_view list)
  {
    kernel_bitmask mask;
    while (!list.empty() && (list.back() == '\n' || list.back() == ' ')) {
      list.remove_suffix(1);
    }
    while (!list.empty()) {
      auto const comma = list.find(',');
      auto const item  = list.substr(0, comma);
      list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
      auto const dash  = item.find('-');
      auto const first = parse_id(item.substr(0, dash));
      auto const last  = dash == std::string_view::npos ? first : parse_id(item.substr(dash + 1));
      if (!first || !last || *last < *first) { return std::nullopt; }
      for (auto id = *first; id <= *last; ++id) {
        mask.set(id);
      }
    }
    return mask;
  }

  void set(std::size_t id)
  {
    if (id / bits_per_word >= _words.size()) { _words.resize(id / bits_per_word + 1); }
    _words[id / bits_per_word] |= 1UL << (id % bits_per_word);
  }

  [[nodiscard]] bool empty() const
  {
    return std::all_of(_words.begin(), _words.end(), [](unsigned long w) { return w == 0; });
  }

  /// Restricts the calling thread to these CPUs. Returns false on failure.
  [[nodiscard]] bool apply_to_thread(int (*set_affinity)(pid_t,
                                                         std::size_t,
                                                         cpu_set_t const*) = ::sched_setaffinity)
  {
    return !empty() && set_affinity(0, bytes(), cpu_set()) == 0;
  }

  static constexpr std::size_t bits_per_word = sizeof(unsigned long) * CHAR_BIT;

 private:
  // Ids beyond this are not CPUs or nodes this kernel can have.
  static constexpr std::size_t max_id = std::size_t{1} << 22;

  static std::optional<std::size_t> parse_id(std::string_view text)
  {
    std::size_t id{};
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), id);
    if (error != std::errc{} || end != text.data() + text.size() || id >= max_id) {
      return std::nullopt;
    }
    return id;
  }

  [[nodiscard]] std::size_t bytes() const { return _words.size() * sizeof(unsigned long); }
  [[nodiscard]] cpu_set_t* cpu_set() { return reinterpret_cast<cpu_set_t*>(_words.data()); }

  std::vector<unsigned long> _words;
};

/// System operations used by the affinity guard; overridable by deterministic tests.
struct affinity_operations {
  int (*get)(pid_t, std::size_t, cpu_set_t*)       = ::sched_getaffinity;
  int (*set)(pid_t, std::size_t, cpu_set_t const*) = ::sched_setaffinity;
  std::optional<kernel_bitmask> (*node_cpus)(int)  = kernel_bitmask::node_cpus;
};

/// Runs the calling thread on the CPUs of @p node for the guard's lifetime, then restores its
/// affinity. Best effort: the affinity is left alone if the thread may not run on that node.
class scoped_run_on_node {
 public:
  explicit scoped_run_on_node(int node, affinity_operations operations = {})
    : _saved(kernel_bitmask::thread_affinity(operations.get)), _set_affinity(operations.set)
  {
    auto cpus = operations.node_cpus(node);
    if (!_saved || !cpus || !cpus->apply_to_thread(_set_affinity)) { _saved.reset(); }
  }
  ~scoped_run_on_node()
  {
    if (_saved) { std::ignore = _saved->apply_to_thread(_set_affinity); }
  }
  scoped_run_on_node(scoped_run_on_node const&)            = delete;
  scoped_run_on_node& operator=(scoped_run_on_node const&) = delete;

 private:
  std::optional<kernel_bitmask> _saved;
  int (*_set_affinity)(pid_t, std::size_t, cpu_set_t const*);
};

}  // namespace detail
}  // namespace memory
}  // namespace cucascade
