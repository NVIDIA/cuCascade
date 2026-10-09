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

#include <cucascade/io/tiered_queue.hpp>

#include <catch2/catch_all.hpp>

#include <atomic>
#include <thread>
#include <vector>

using cucascade::io::io_priority;
using cucascade::io::tiered_blocking_queue;

TEST_CASE("tiered_blocking_queue drains the high tier before the low tier", "[io][tiered_queue]")
{
  tiered_blocking_queue<int> queue;
  CHECK(queue.enqueue(1, io_priority::low));
  CHECK(queue.enqueue(2, io_priority::low));
  CHECK(queue.enqueue(10, io_priority::high));
  CHECK(queue.enqueue(11, io_priority::automatic));  // automatic counts as high
  CHECK(queue.size_approx() == 4);
  CHECK(queue.size_approx(io_priority::low) == 2);

  std::vector<int> order;
  int item = 0;
  while (queue.try_dequeue(item)) {
    order.push_back(item);
  }
  CHECK(order == std::vector<int>{10, 11, 1, 2});
  CHECK_FALSE(queue.wait_dequeue_timed(item, 1000));
}

TEST_CASE("tiered_blocking_queue::try_dequeue_high leaves low items queued", "[io][tiered_queue]")
{
  tiered_blocking_queue<int> queue;
  int item = 0;
  CHECK(queue.enqueue(1, io_priority::low));
  CHECK_FALSE(queue.try_dequeue_high(item));
  CHECK(queue.enqueue(7, io_priority::high));
  CHECK(queue.try_dequeue_high(item));
  CHECK(item == 7);
  CHECK_FALSE(queue.try_dequeue_high(item));
  // The low item's permit survived the failed high attempts.
  CHECK(queue.wait_dequeue_timed(item, 1000));
  CHECK(item == 1);
}

TEST_CASE("tiered_blocking_queue wakes a blocked consumer and loses nothing under contention",
          "[io][tiered_queue]")
{
  constexpr int per_producer = 20000;
  tiered_blocking_queue<int> queue;
  std::atomic<long long> sum{0};
  std::atomic<int> taken{0};

  std::jthread consumer([&] {
    int item = 0;
    while (taken.load() < 2 * per_producer) {
      if (queue.wait_dequeue_timed(item, 10000)) {
        sum += item;
        ++taken;
      }
    }
  });
  std::jthread high([&] {
    for (int i = 1; i <= per_producer; ++i) {
      REQUIRE(queue.enqueue(int{i}, io_priority::high));
    }
  });
  std::jthread low([&] {
    for (int i = 1; i <= per_producer; ++i) {
      REQUIRE(queue.enqueue(int{i}, io_priority::low));
    }
  });
  high.join();
  low.join();
  consumer.join();

  constexpr long long expected = 2LL * per_producer * (per_producer + 1) / 2;
  CHECK(sum.load() == expected);
  CHECK(queue.size_approx() == 0);
}
