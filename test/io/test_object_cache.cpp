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

#include <cstddef>
#include <memory>
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

TEST_CASE("an object that was kept is found again", "[io][object_cache]")
{
  object_cache cache;
  CHECK(cache.find("s3://b/a") == nullptr);
  auto const object = std::make_shared<fake_object>("s3://b/a");
  CHECK(cache.insert("s3://b/a", object) == object);
  CHECK(cache.find("s3://b/a") == object);
  CHECK(cache.size() == 1);
}

TEST_CASE("the first object kept for a path stays", "[io][object_cache]")
{
  object_cache cache;
  auto const first  = std::make_shared<fake_object>("s3://b/a");
  auto const second = std::make_shared<fake_object>("s3://b/a");
  cache.insert("s3://b/a", first);
  CHECK(cache.insert("s3://b/a", second) == first);
  CHECK(cache.find("s3://b/a") == first);
  CHECK(cache.size() == 1);
}

TEST_CASE("paths are kept separately and concurrent inserts agree", "[io][object_cache]")
{
  object_cache cache;
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&cache] {
      for (int i = 0; i < 100; ++i) {
        auto const path = "s3://b/" + std::to_string(i);
        cache.insert(path, std::make_shared<fake_object>(path));
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  CHECK(cache.size() == 100);
  for (int i = 0; i < 100; ++i) {
    auto const path = "s3://b/" + std::to_string(i);
    REQUIRE(cache.find(path) != nullptr);
    CHECK(cache.find(path)->object_path() == path);
  }
}
