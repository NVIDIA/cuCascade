/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
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

// A failure after the first worker started (std::thread throwing, an allocation
// in the worker naming) must propagate out of the constructor instead of
// destroying a joinable std::thread, which would call std::terminate. The
// allocation counter fails each allocation of the construction in turn, so the
// sweep covers failures before, between and after worker starts.

#include "utils/allocation_counter.hpp"

#include <cucascade/exec/thread_pool.hpp>

#include <catch2/catch_all.hpp>

#include <cstddef>
#include <new>
#include <string>

TEST_CASE("static_thread_pool construction failure stops and joins the started workers",
          "[exec][thread_pool]")
{
  using cucascade::exec::static_thread_pool;
  using cucascade::test::scoped_allocation_counter;

  // Longer than the small-string buffer, so naming each worker allocates once a
  // worker has already started.
  std::string const name(48, 'w');
  std::size_t failures = 0;
  for (std::size_t n = 1;; ++n) {
    bool threw = false;
    std::size_t allocations;
    {
      scoped_allocation_counter counter;
      counter.fail_at(n);
      try {
        static_thread_pool pool(4, name);
        counter.fail_at(0);  // construction succeeded; its destructor runs unarmed
      } catch (std::bad_alloc const&) {
        threw = true;
      }
      allocations = counter.count();
    }
    if (!threw) {
      // Construction made fewer than n allocations: every failure point is covered.
      CHECK(allocations < n);
      break;
    }
    ++failures;
    CHECK(allocations >= n);
  }
  // The sweep failed at least the reserve and one allocation made after a worker started.
  CHECK(failures >= 2);
}
