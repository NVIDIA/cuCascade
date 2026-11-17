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

#include "memory/numa_affinity.hpp"

#include <cucascade/utils/cleanup_utils.hpp>

#include <catch2/catch_all.hpp>
#include <sched.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

using cucascade::memory::detail::affinity_operations;
using cucascade::memory::detail::kernel_bitmask;
using cucascade::memory::detail::scoped_run_on_node;

using cpu_words = std::array<unsigned long, 64>;

struct affinity_state {
  cpu_words current{};
  cpu_words attempted{};
  std::array<std::size_t, 16> get_sizes{};
  std::size_t get_calls{0};
  int requested_node{-1};
  int set_calls{0};
  int get_error{0};
  int fail_set_call{0};
  std::string node_list{"0-3"};
};

thread_local affinity_state* active_state = nullptr;

int get_affinity(pid_t, std::size_t bytes, cpu_set_t* mask)
{
  auto& state = *active_state;
  if (state.get_calls < state.get_sizes.size()) { state.get_sizes[state.get_calls] = bytes; }
  ++state.get_calls;
  if (state.get_error != 0 || bytes < sizeof(cpu_words)) {
    errno = state.get_error != 0 ? state.get_error : EINVAL;
    return -1;
  }
  std::memset(mask, 0, bytes);
  std::memcpy(mask, state.current.data(), sizeof(cpu_words));
  return 0;
}

int set_affinity(pid_t, std::size_t bytes, cpu_set_t const* mask)
{
  auto& state = *active_state;
  ++state.set_calls;
  state.attempted.fill(0);
  std::memcpy(state.attempted.data(), mask, std::min(bytes, sizeof(cpu_words)));
  if (state.set_calls == state.fail_set_call) {
    errno = EPERM;
    return -1;
  }
  state.current = state.attempted;
  return 0;
}

std::optional<kernel_bitmask> node_cpus(int node)
{
  active_state->requested_node = node;
  return kernel_bitmask::parse_list(active_state->node_list);
}

struct fake_affinity {
  affinity_state state;
  affinity_state* previous{active_state};

  fake_affinity()
  {
    state.current[32] = 1;  // CPU 2048, outside the initial 1024-bit mask.
    active_state      = &state;
  }
  ~fake_affinity() { active_state = previous; }

  static affinity_operations operations() { return {get_affinity, set_affinity, node_cpus}; }
};

}  // namespace

TEST_CASE("NUMA CPU lists preserve ranges and sparse high CPU IDs", "[numa_affinity]")
{
  struct parse_case {
    std::string_view list;
    std::vector<std::size_t> cpus;
  };
  for (auto const& entry : std::array<parse_case, 4>{{{"0", {0}},
                                                      {"0-3,8,10-11", {0, 1, 2, 3, 8, 10, 11}},
                                                      {"63-65", {63, 64, 65}},
                                                      {"0,64,2048\n ", {0, 64, 2048}}}}) {
    CAPTURE(entry.list);
    fake_affinity fake;
    auto mask = kernel_bitmask::parse_list(entry.list);
    REQUIRE(mask.has_value());
    REQUIRE(mask->apply_to_thread(set_affinity));
    cpu_words expected{};
    for (auto cpu : entry.cpus) {
      expected[cpu / kernel_bitmask::bits_per_word] |= 1UL << (cpu % kernel_bitmask::bits_per_word);
    }
    CHECK(fake.state.current == expected);
  }
}

TEST_CASE("NUMA CPU lists reject malformed and overflowing IDs", "[numa_affinity]")
{
  auto const* list =
    GENERATE("-1", "4-2", "0,a", "1-", ",1", "1,,2", "4194304", "18446744073709551616");
  CAPTURE(list);
  CHECK_FALSE(kernel_bitmask::parse_list(list));
}

TEST_CASE("NUMA CPU lists accept the largest supported CPU ID", "[numa_affinity]")
{
  auto mask = kernel_bitmask::parse_list("4194303");
  REQUIRE(mask.has_value());
  CHECK_FALSE(mask->empty());
}

TEST_CASE("Empty NUMA CPU lists do not change affinity", "[numa_affinity]")
{
  fake_affinity fake;
  auto mask = kernel_bitmask::parse_list("");
  REQUIRE(mask.has_value());
  CHECK(mask->empty());
  CHECK_FALSE(mask->apply_to_thread(set_affinity));
  CHECK(fake.state.set_calls == 0);
}

TEST_CASE("NUMA affinity reads grow on EINVAL and retain high CPU bits", "[numa_affinity]")
{
  fake_affinity fake;
  auto const expected = fake.state.current;
  auto saved          = kernel_bitmask::thread_affinity(get_affinity);
  REQUIRE(saved.has_value());
  REQUIRE(fake.state.get_calls >= 2);
  REQUIRE(fake.state.get_calls <= fake.state.get_sizes.size());
  auto const last = fake.state.get_calls - 1;
  CHECK(fake.state.get_sizes[last - 1] < sizeof(cpu_words));
  CHECK(fake.state.get_sizes[last] >= sizeof(cpu_words));
  for (std::size_t i = 1; i <= last; ++i) {
    CHECK(fake.state.get_sizes[i] > fake.state.get_sizes[i - 1]);
  }
  fake.state.current.fill(0);
  REQUIRE(saved->apply_to_thread(set_affinity));
  CHECK(fake.state.current == expected);
}

TEST_CASE("NUMA affinity reads stop on failure and bound mask growth", "[numa_affinity]")
{
  fake_affinity fake;
  fake.state.get_error = GENERATE(EPERM, EINVAL);
  CHECK_FALSE(kernel_bitmask::thread_affinity(get_affinity));
  if (fake.state.get_error == EPERM) {
    CHECK(fake.state.get_calls == 1);
  } else {
    REQUIRE(fake.state.get_calls <= fake.state.get_sizes.size());
    CHECK(fake.state.get_sizes[fake.state.get_calls - 1] == (std::size_t{1} << 19));
  }
}

TEST_CASE("NUMA affinity guard switches CPUs and restores them on every exit", "[numa_affinity]")
{
  fake_affinity fake;
  auto const original = fake.state.current;
  auto const throws   = GENERATE(false, true);
  try {
    scoped_run_on_node guard(7, fake_affinity::operations());
    CHECK(fake.state.requested_node == 7);
    CHECK(fake.state.set_calls == 1);
    cpu_words expected{};
    expected[0] = 0xf;
    CHECK(fake.state.current == expected);
    if (throws) { throw std::runtime_error("leave affinity scope"); }
  } catch (std::runtime_error const&) {
    REQUIRE(throws);
  }
  CHECK(fake.state.set_calls == 2);
  CHECK(fake.state.current == original);
}

TEST_CASE("NUMA affinity guard tolerates missing CPUs and denied syscalls", "[numa_affinity]")
{
  fake_affinity fake;
  auto const original = fake.state.current;
  SECTION("unreadable original affinity") { fake.state.get_error = EPERM; }
  SECTION("invalid node CPU list") { fake.state.node_list = "invalid"; }
  SECTION("CPU-less node") { fake.state.node_list.clear(); }
  SECTION("pinning denied") { fake.state.fail_set_call = 1; }
  {
    scoped_run_on_node guard(0, fake_affinity::operations());
    CHECK(fake.state.current == original);
  }
  CHECK(fake.state.current == original);
  CHECK(fake.state.set_calls == (fake.state.fail_set_call == 1 ? 1 : 0));
}

TEST_CASE("NUMA affinity guard attempts restoration even when it is denied", "[numa_affinity]")
{
  fake_affinity fake;
  auto const original      = fake.state.current;
  fake.state.fail_set_call = 2;
  {
    scoped_run_on_node guard(0, fake_affinity::operations());
  }
  CHECK(fake.state.set_calls == 2);
  CHECK(fake.state.attempted == original);
}

TEST_CASE("NUMA affinity guard performs a real temporary switch", "[numa_affinity]")
{
  cpu_set_t original;
  if (sched_getaffinity(0, sizeof(original), &original) != 0) {
    if (errno == EINVAL || errno == EPERM) {
      SKIP("thread affinity is unavailable at this mask size");
    }
    FAIL("sched_getaffinity failed");
  }
  auto restore = cucascade::utils::on_scope_exit(
    [&]() noexcept { std::ignore = sched_setaffinity(0, sizeof(original), &original); });
  auto node = kernel_bitmask::node_cpus(0);
  if (!node || !node->apply_to_thread()) { SKIP("cannot run on node 0 CPUs"); }
  cpu_set_t node_mask;
  REQUIRE(sched_getaffinity(0, sizeof(node_mask), &node_mask) == 0);
  if (CPU_COUNT(&node_mask) < 2) { SKIP("node 0 needs at least two allowed CPUs"); }
  cpu_set_t one;
  CPU_ZERO(&one);
  CPU_SET(sched_getcpu(), &one);
  REQUIRE(sched_setaffinity(0, sizeof(one), &one) == 0);
  {
    scoped_run_on_node guard(0);
    cpu_set_t during;
    REQUIRE(sched_getaffinity(0, sizeof(during), &during) == 0);
    CHECK(CPU_EQUAL(&during, &node_mask));
  }
  cpu_set_t after;
  REQUIRE(sched_getaffinity(0, sizeof(after), &after) == 0);
  CHECK(CPU_EQUAL(&after, &one));
}
