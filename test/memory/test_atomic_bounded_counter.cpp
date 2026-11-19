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

#include <cucascade/utils/atomics.hpp>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <atomic>
#include <thread>

using cucascade::utils::atomic_bounded_counter;

TEST_CASE("bounded counter retries use the full request", "[memory][atomics]")
{
  for (bool const addition : {true, false}) {
    atomic_bounded_counter<unsigned> counter{0};
    std::atomic<bool> running{true};
    std::thread writer([&] {
      while (running.load(std::memory_order_relaxed)) {
        counter.native_handle().store(9);
        counter.native_handle().store(0);
      }
    });

    bool violation = false;
    for (int i = 0; i < 1000000 && !violation; ++i) {
      unsigned amount = 10;
      auto const result =
        addition ? counter.add_bounded(amount, 10) : counter.sub_bounded(amount, 0);
      // A full request must reach the bound whenever there is room to move.
      violation = amount > 0 && (addition ? result != 10 : result != 0);
    }
    running.store(false, std::memory_order_relaxed);
    writer.join();
    CHECK_FALSE(violation);
  }
}
