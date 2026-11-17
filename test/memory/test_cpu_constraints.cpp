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

#include "utils/child_test_runner.hpp"

#include <cucascade/memory/cpu_constraints.hpp>

#include <catch2/catch_test_macros.hpp>
#include <sched.h>

#include <cstdlib>
#include <exception>
#include <map>
#include <sstream>
#include <thread>

TEST_CASE("metadata cpuset discovery respects ancestor restrictions and mount roots",
          "[cpu_constraints][cpuset_discovery]")
{
  using namespace cucascade::memory;
  using cpus             = std::vector<int>;
  std::string membership = "0::/parent/child\n";
  std::string mounts     = "31 20 0:28 / /cgroup rw,nosuid shared:4 - cgroup2 cgroup rw\n";
  std::map<std::string, cpus> files{{"/cgroup/cpuset.cpus.effective", {0, 1, 2, 3, 4, 5, 6, 7}},
                                    {"/cgroup/parent/cpuset.cpus.effective", {0, 1, 2, 3}}};
  std::optional<cpus> expected = cpus{0, 1, 2, 3};
  bool unreadable              = false;
  SECTION("missing leaf inherits restricted parent") {}
  SECTION("multiple missing controller levels inherit the nearest enabled ancestor")
  {
    membership                                          = "0::/parent/child/grandchild\n";
    files["/cgroup/parent/child/cpuset.cpus.effective"] = {1};
    expected                                            = cpus{1};
  }
  SECTION("readable leaf takes precedence")
  {
    files["/cgroup/parent/child/cpuset.cpus.effective"] = {1, 2};
    expected                                            = cpus{1, 2};
  }
  SECTION("namespace root")
  {
    membership = "0::/\n";
    expected   = files["/cgroup/cpuset.cpus.effective"];
  }
  SECTION("bind mount with non-root filesystem root")
  {
    mounts = "31 20 0:28 /parent /restricted rw - cgroup2 cgroup rw\n";
    files["/restricted/cpuset.cpus.effective"] = {1, 3};
    expected                                   = cpus{1, 3};
  }
  SECTION("escaped nonstandard mount point")
  {
    mounts = "31 20 0:28 /parent /restricted\\040cpus rw - cgroup2 cgroup rw\n";
    files["/restricted cpus/cpuset.cpus.effective"] = {2};
    expected                                        = cpus{2};
  }
  SECTION("a more specific visible mount wins")
  {
    mounts += "32 20 0:28 /parent /restricted rw - cgroup2 cgroup rw\n";
    files["/restricted/cpuset.cpus.effective"] = {3};
    expected                                   = cpus{3};
  }
  SECTION("unmapped membership must not guess the mount root")
  {
    mounts = "31 20 0:28 /other /cgroup rw - cgroup2 cgroup rw\n";
    expected.reset();
  }
  SECTION("unreadable leaf must not broaden to an ancestor") { unreadable = true; }
  SECTION("no accessible controller files")
  {
    files.clear();
    expected.reset();
  }
  SECTION("an empty effective set must not broaden to an ancestor")
  {
    files["/cgroup/parent/child/cpuset.cpus.effective"] = {};
    expected                                            = cpus{};
  }
  SECTION("unified mount missing")
  {
    mounts.clear();
    expected.reset();
  }
  SECTION("parent traversal rejected")
  {
    membership = "0::/../parent/child\n";
    expected.reset();
  }
  std::istringstream groups(membership), mountinfo(mounts);
  std::vector<std::string> reads;
  auto read = [&](std::string const& path) -> std::optional<cpus> {
    reads.push_back(path);
    if (unreadable) throw std::runtime_error("unreadable cpuset");
    auto it = files.find(path);
    if (it == files.end()) return std::nullopt;
    return it->second;
  };
  if (unreadable) {
    CHECK_THROWS_WITH(read_cgroup_cpu_allowance(groups, mountinfo, read), "unreadable cpuset");
    CHECK(reads.size() == 1);
  } else {
    CHECK(read_cgroup_cpu_allowance(groups, mountinfo, read) == expected);
  }
  if (!unreadable && expected == std::optional<cpus>{cpus{0, 1, 2, 3}}) {
    REQUIRE(reads.size() == 2);
    CHECK(reads.back() == "/cgroup/parent/cpuset.cpus.effective");
  }
}

TEST_CASE("CPU constraints preserve rank binding within a broader deployment", "[cpu_constraints]")
{
  using namespace cucascade::memory;
  auto first  = resolve_cpu_constraints({3, 2, 2}, std::vector<int>{0, 1, 2, 3, 4, 5});
  auto second = resolve_cpu_constraints({4, 5}, std::vector<int>{0, 1, 2, 3, 4, 5});
  CHECK(first.allowed_cpus == std::vector<int>{2, 3});
  CHECK(second.allowed_cpus == std::vector<int>{4, 5});
  CHECK(resolve_cpu_constraints({1, 2, 3}, std::vector<int>{2, 3, 4}).allowed_cpus ==
        std::vector<int>{2, 3});
  CHECK(resolve_cpu_constraints({2, 3}, std::nullopt).allowed_cpus == std::vector<int>{2, 3});
  CHECK_THROWS(resolve_cpu_constraints({2, 3}, std::vector<int>{}));
  CHECK_THROWS(resolve_cpu_constraints({2, 3}, std::vector<int>{4, 5}));
  CHECK_THROWS(resolve_cpu_constraints({}, std::nullopt));
}

TEST_CASE("CPU discovery ignores an incidental caller pin", "[cpu_constraints]")
{
  using namespace cucascade::memory;
  auto process = discover_cpu_constraints();
  if (process.allowed_cpus.size() < 2) SKIP("requires two allowed CPUs");
  cpu_constraints worker;
  std::exception_ptr error;
  std::thread pinned([&] {
    try {
      cpu_set_t mask;
      CPU_ZERO(&mask);
      CPU_SET(static_cast<size_t>(process.allowed_cpus.front()), &mask);
      if (::sched_setaffinity(0, sizeof(mask), &mask) != 0)
        throw std::runtime_error("cannot restrict test worker");
      worker = discover_cpu_constraints();
    } catch (...) {
      error = std::current_exception();
    }
  });
  pinned.join();
  if (error) std::rethrow_exception(error);
  CHECK(worker.process_cpus == process.process_cpus);
  CHECK(worker.allowed_cpus == process.allowed_cpus);
  CHECK_FALSE(worker.used_caller_affinity);
}

TEST_CASE("CPU discovery preserves an externally restricted rank", "[cpu_constraints]")
{
  using namespace cucascade::memory;
  if (std::getenv("CUCASCADE_CPU_BOUNDARY_CHILD")) {
    auto before = discover_cpu_constraints();
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(static_cast<size_t>(before.allowed_cpus.front()), &mask);
    REQUIRE(::sched_setaffinity(0, sizeof(mask), &mask) == 0);
    auto after = discover_cpu_constraints();
    CHECK(after.allowed_cpus == std::vector<int>{before.allowed_cpus.front()});
    CHECK(after.process_cpus == after.allowed_cpus);
    return;
  }
  cucascade::test::child_test_runner child("CPU discovery preserves an externally restricted rank",
                                           {{"CUCASCADE_CPU_BOUNDARY_CHILD", std::string{"1"}}});
  REQUIRE(child.wait_for(std::chrono::seconds{10}));
  INFO(child.output());
  REQUIRE(WIFEXITED(child.status()));
  CHECK(WEXITSTATUS(child.status()) == 0);
}
