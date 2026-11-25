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

#include <cucascade/io/object_cache.hpp>

#include <catch2/catch_all.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using cucascade::io::io_object;
using cucascade::io::object_cache;

class fake_object : public io_object {
 public:
  explicit fake_object(std::string path) : _path(std::move(path)) {}
  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] size_t size() const noexcept override { return 1; }

 private:
  std::string _path;
};

}  // namespace

TEST_CASE("an object is opened once and found again", "[io][object_cache]")
{
  object_cache cache;
  int opens         = 0;
  auto const object = std::make_shared<fake_object>("s3://b/a");
  auto const open   = [&] {
    ++opens;
    return object;
  };
  CHECK(cache.find_or_open("s3://b/a", open) == object);
  CHECK(cache.find_or_open("s3://b/a", open) == object);
  CHECK(opens == 1);
  CHECK(cache.size() == 1);
}

TEST_CASE("a failed open is not kept", "[io][object_cache]")
{
  object_cache cache;
  CHECK_THROWS_AS(cache.find_or_open("s3://b/a",
                                     []() -> std::shared_ptr<io_object> {
                                       throw std::runtime_error("unreachable endpoint");
                                     }),
                  std::runtime_error);
  CHECK(cache.size() == 0);
  auto const object = std::make_shared<fake_object>("s3://b/a");
  CHECK(cache.find_or_open("s3://b/a", [&] { return object; }) == object);
}

TEST_CASE("concurrent first opens of a path share one open", "[io][object_cache]")
{
  object_cache cache;
  std::atomic<int> opens{0};
  std::vector<std::thread> threads;
  std::vector<std::shared_ptr<io_object>> seen(8 * 100);
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < 100; ++i) {
        auto const path                             = "s3://b/" + std::to_string(i);
        seen[static_cast<std::size_t>(t * 100 + i)] = cache.find_or_open(path, [&] {
          ++opens;
          std::this_thread::sleep_for(std::chrono::microseconds{50});
          return std::make_shared<fake_object>(path);
        });
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  CHECK(opens.load() == 100);
  CHECK(cache.size() == 100);
  for (int t = 1; t < 8; ++t) {
    for (int i = 0; i < 100; ++i) {
      CHECK(seen[static_cast<std::size_t>(t * 100 + i)] == seen[static_cast<std::size_t>(i)]);
    }
  }
  CHECK(seen[7]->object_path() == "s3://b/7");
}

TEST_CASE("callers waiting on a failed open get its error, and later opens do not hang",
          "[io][object_cache]")
{
  constexpr int n_waiters = 4;
  auto const object       = std::make_shared<fake_object>("s3://b/a");
  for (int round = 0; round < 50; ++round) {
    object_cache cache;
    std::atomic<bool> opening{false};
    std::atomic<int> waiting{0};
    std::thread opener([&] {
      CHECK_THROWS_WITH(
        cache.find_or_open("s3://b/a",
                           [&]() -> std::shared_ptr<io_object> {
                             opening = true;
                             while (waiting.load() < n_waiters) {
                               std::this_thread::yield();
                             }
                             // Vary how far the waiters are into their wait.
                             std::this_thread::sleep_for(std::chrono::microseconds{round * 20});
                             throw std::runtime_error("unreachable endpoint");
                           }),
        "unreachable endpoint");
    });
    while (!opening.load()) {
      std::this_thread::yield();
    }
    // Each waiter either waits on the failed open and gets its error, or, arriving after it was
    // dropped, opens again itself. None sees a broken promise or blocks for good.
    std::vector<std::thread> waiters;
    std::atomic<int> errors{0};
    std::atomic<int> reopened{0};
    for (int w = 0; w < n_waiters; ++w) {
      waiters.emplace_back([&] {
        ++waiting;
        try {
          if (cache.find_or_open("s3://b/a", [&] { return object; }) == object) { ++reopened; }
        } catch (std::runtime_error const& e) {
          if (std::string{e.what()} == "unreachable endpoint") { ++errors; }
        }
      });
    }
    opener.join();
    for (auto& waiter : waiters) {
      waiter.join();
    }
    CHECK(errors.load() + reopened.load() == n_waiters);
    CHECK(cache.find_or_open("s3://b/a", [&] { return object; }) == object);
  }
}
