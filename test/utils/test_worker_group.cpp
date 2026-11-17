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

#include "utils/child_test_runner.hpp"
#include "utils/worker_group.hpp"

#include <catch2/catch_all.hpp>
#include <sys/wait.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

using cucascade::test::start_gate_broken;
using cucascade::test::worker_group;
using namespace std::chrono_literals;

using clock_type = std::chrono::steady_clock;
using gate       = worker_group<int>::start_gate;

/// Far longer than any test here takes when the group works.
constexpr auto generous_timeout = 60s;

/// What "promptly" means below: far less than generous_timeout, far more than a thread start.
constexpr auto prompt = 10s;

}  // namespace

TEST_CASE("worker_group starts its workers together and returns their results in spawn order",
          "[worker_group]")
{
  constexpr int workers = 8;
  worker_group<int> group{workers, generous_timeout};
  std::atomic<int> arrived{0};
  std::atomic<int> arrived_when_started{workers};
  for (int i = 0; i < workers; ++i) {
    group.spawn([&, i](gate& start) {
      ++arrived;
      start.arrive_and_wait();
      // No worker passes the gate before every worker has reached it.
      if (arrived != workers) { arrived_when_started = arrived.load(); }
      return i * i;
    });
  }
  group.wait_until_started();
  CHECK(arrived == workers);

  std::vector<int> const results = group.join();
  CHECK(arrived_when_started == workers);
  REQUIRE(results.size() == workers);
  for (int i = 0; i < workers; ++i) {
    CHECK(results[static_cast<std::size_t>(i)] == i * i);
  }
}

TEST_CASE("a worker that throws before the start gate fails the group promptly", "[worker_group]")
{
  auto const started = clock_type::now();
  worker_group<int> group{3, generous_timeout};
  group.spawn([](gate&) -> int { throw std::runtime_error{"setup failed"}; });
  for (int i = 0; i < 2; ++i) {
    group.spawn([](gate& start) {
      start.arrive_and_wait();
      return 1;
    });
  }

  SECTION("join() rethrows the worker's exception")
  {
    CHECK_THROWS_WITH(group.join(), "setup failed");
  }
  SECTION("wait_until_started() rethrows the worker's exception")
  {
    CHECK_THROWS_WITH(group.wait_until_started(), "setup failed");
  }
  SECTION("a group that is destroyed without join() does not wait for the gate") {}
  CHECK(clock_type::now() - started < prompt);
}

TEST_CASE("a worker that returns without passing the start gate fails the group promptly",
          "[worker_group]")
{
  auto const started = clock_type::now();
  worker_group<int> group{2, generous_timeout};
  group.spawn([](gate&) { return 1; });
  group.spawn([](gate& start) {
    start.arrive_and_wait();
    return 2;
  });
  CHECK_THROWS_AS(group.join(), start_gate_broken);
  CHECK(clock_type::now() - started < prompt);
}

TEST_CASE("workers that are not all spawned leave the start gate when the group is destroyed",
          "[worker_group]")
{
  auto const started = clock_type::now();
  std::atomic<bool> gate_broken{false};
  {
    worker_group<int> group{3, generous_timeout};
    group.spawn([&](gate& start) {
      try {
        start.arrive_and_wait();
      } catch (start_gate_broken const&) {
        gate_broken = true;
        throw;
      }
      return 1;
    });
    // The test body fails here, before it spawns the other workers.
  }
  CHECK(gate_broken);
  CHECK(clock_type::now() - started < prompt);
}

TEST_CASE("a start gate that does not open by the deadline fails the group", "[worker_group]")
{
  auto const started = clock_type::now();
  std::mutex mutex;
  std::condition_variable released;
  bool release = false;

  worker_group<int> group{2, 200ms};
  group.spawn([](gate& start) {
    start.arrive_and_wait();
    return 1;
  });
  // Stays busy past the deadline without reaching the gate, then ends.
  group.spawn([&](gate&) {
    std::unique_lock lock{mutex};
    released.wait_for(lock, generous_timeout, [&] { return release; });
    return 2;
  });

  CHECK_THROWS_AS(group.wait_until_started(), start_gate_broken);
  CHECK(clock_type::now() - started >= 200ms);
  CHECK(clock_type::now() - started < prompt);
  {
    std::scoped_lock lock{mutex};
    release = true;
  }
  released.notify_all();
}

TEST_CASE("a worker that never ends aborts the test process instead of hanging it",
          "[worker_group]")
{
  constexpr char const* marker = "CUCASCADE_TEST_WORKER_GROUP_STUCK";
  if (auto const* value = std::getenv(marker); value != nullptr && std::string_view{value} == "1") {
    // The child: one worker blocks for good. join() must give up on it and end the process.
    std::mutex mutex;
    std::condition_variable never;
    worker_group<int> group{1, 200ms};
    group.spawn([&](gate& start) {
      start.arrive_and_wait();
      std::unique_lock lock{mutex};
      never.wait(lock, [] { return false; });
      return 0;
    });
    std::ignore = group.join();
    FAIL("join() returned although a worker is still running");
    return;
  }

  cucascade::test::child_test_runner child{Catch::getResultCapture().getCurrentTestName(),
                                           {{marker, "1"}}};
  REQUIRE(child.wait_for(std::chrono::duration_cast<std::chrono::milliseconds>(generous_timeout)));
  CHECK(WIFSIGNALED(child.status()));
  CHECK(WTERMSIG(child.status()) == SIGABRT);
  CHECK_THAT(child.output(), Catch::Matchers::ContainsSubstring("worker_group"));
}
