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

#include <cucascade/exec/thread_util.hpp>

#include <catch2/catch_test_macros.hpp>
#include <pthread.h>
#include <sched.h>

#include <array>
#include <cerrno>
#include <exception>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>

TEST_CASE("startup affinity restores the saved mask after the OS clips a request",
          "[exec][thread_affinity][affinity_rollback]")
{
  cpu_set_t saved{}, clipped{}, restored{};
  int permitted = -1, unavailable = -1;
  bool threw = false;
  std::string skip, message;
  std::exception_ptr error;
  auto checked = [](int result, char const* operation) {
    if (result) throw std::system_error(result, std::generic_category(), operation);
  };
  // Change only this disposable thread's affinity, so even broken rollback
  // cannot narrow the main test thread or any subsequent test's inherited mask.
  std::thread worker([&] {
    try {
      checked(::pthread_getaffinity_np(::pthread_self(), sizeof(saved), &saved), "read saved mask");
      if (CPU_COUNT(&saved) < 2) {
        skip = "affinity rollback requires at least two CPUs in the saved mask";
        return;
      }
      for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(static_cast<std::size_t>(cpu), &saved)) {
          permitted = cpu;
          break;
        }
      }
      // Absence from the saved mask does not prove unavailability: an inherited
      // taskset mask may be narrower than the cgroup allowance. Probe singleton
      // requests and restore after any successful expansion.
      for (int cpu = CPU_SETSIZE - 1; cpu >= 0; --cpu) {
        if (CPU_ISSET(static_cast<std::size_t>(cpu), &saved)) continue;
        cpu_set_t singleton{};
        CPU_SET(static_cast<std::size_t>(cpu), &singleton);
        auto const result =
          ::pthread_setaffinity_np(::pthread_self(), sizeof(singleton), &singleton);
        if (result == EINVAL) {
          unavailable = cpu;
          break;
        }
        checked(result, "probe unavailable CPU");
        checked(::pthread_setaffinity_np(::pthread_self(), sizeof(saved), &saved),
                "restore probe mask");
      }
      if (unavailable < 0) {
        skip = "affinity rollback requires a verified unavailable CPU below CPU_SETSIZE";
        return;
      }
      cpu_set_t requested{};
      CPU_SET(static_cast<std::size_t>(permitted), &requested);
      CPU_SET(static_cast<std::size_t>(unavailable), &requested);
      // Establish that this host accepts the mixed request and clips it. A
      // range-rejection test would throw before reaching the rollback branch.
      checked(::pthread_setaffinity_np(::pthread_self(), sizeof(requested), &requested),
              "apply mixed probe");
      checked(::pthread_getaffinity_np(::pthread_self(), sizeof(clipped), &clipped),
              "read clipped mask");
      checked(::pthread_setaffinity_np(::pthread_self(), sizeof(saved), &saved),
              "restore saved mask");
      try {
        std::array<int, 2> const cpus{permitted, unavailable};
        cucascade::exec::thread_util::scoped_thread_affinity guard(cpus);
      } catch (std::runtime_error const& failure) {
        threw   = true;
        message = failure.what();
      }
      // Observe before any other cleanup can restore affinity and hide a bug.
      checked(::pthread_getaffinity_np(::pthread_self(), sizeof(restored), &restored),
              "read restored mask");
    } catch (...) {
      error = std::current_exception();
    }
  });
  worker.join();
  if (error) std::rethrow_exception(error);
  if (!skip.empty()) SKIP(skip);
  INFO("permitted CPU=" << permitted << ", unavailable CPU=" << unavailable);
  CHECK(CPU_COUNT(&clipped) == 1);
  CHECK(CPU_ISSET(static_cast<std::size_t>(permitted), &clipped));
  CHECK(threw);
  CHECK(message == "startup CPU affinity was restricted by the operating system");
  CHECK(CPU_EQUAL(&saved, &restored));
}
