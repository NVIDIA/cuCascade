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

#include <cucascade/io/cache/metadata_store.hpp>

#include <catch2/catch_all.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <utility>

using cucascade::io::io_object;
using cucascade::io::io_object_metadata;
using cucascade::io::cache::metadata_store;

namespace {

/// Minimal io_object stand-in: the store only ever reads raw_file_cache_id().
class fake_io_object final : public io_object {
 public:
  explicit fake_io_object(std::string id) : _id(std::move(id)) {}

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _id; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _id; }
  [[nodiscard]] size_t size() const noexcept override { return 0; }

 private:
  std::string _id;
};

struct tagged_metadata final : io_object_metadata {
  explicit tagged_metadata(int t) : tag(t) {}
  int tag;
};

}  // namespace

TEST_CASE("metadata_store round-trips through both getters", "[cache][metadata_store]")
{
  metadata_store store;
  fake_io_object const obj{"s3://bucket/key.parquet"};
  auto const meta = std::make_shared<tagged_metadata>(7);

  store.register_metadata(obj, meta);

  auto const by_obj = store.get_metadata(obj);
  REQUIRE(by_obj);
  CHECK(std::static_pointer_cast<tagged_metadata>(by_obj)->tag == 7);

  auto const by_key = store.get_metadata(std::string_view{"s3://bucket/key.parquet"});
  REQUIRE(by_key);
  CHECK(by_key == by_obj);
}

TEST_CASE("metadata_store looks up heterogeneously without building a key",
          "[cache][metadata_store]")
{
  metadata_store store;
  fake_io_object const obj{"/data/lineitem.parquet"};
  store.register_metadata(obj, std::make_shared<tagged_metadata>(1));

  // All three spellings must reach the same entry: a string_view, a string
  // literal (const char*), and an owning std::string.  This only compiles —
  // and only avoids a temporary — because the map has a transparent hash and
  // std::equal_to<>.
  auto const from_view    = store.get_metadata(std::string_view{"/data/lineitem.parquet"});
  auto const from_literal = store.get_metadata("/data/lineitem.parquet");
  auto const from_string  = store.get_metadata(std::string{"/data/lineitem.parquet"});

  REQUIRE(from_view);
  CHECK(from_view == from_literal);
  CHECK(from_view == from_string);
}

TEST_CASE("metadata_store looks up a key inside a larger buffer", "[cache][metadata_store]")
{
  metadata_store store;
  fake_io_object const obj{"abc"};
  store.register_metadata(obj, std::make_shared<tagged_metadata>(3));

  // A string_view that is NOT NUL-terminated and is a slice of a longer buffer:
  // heterogeneous lookup must respect the view's length, not run to a NUL.
  std::string const haystack = "abcdef";
  auto const key             = std::string_view{haystack}.substr(0, 3);
  REQUIRE(key == "abc");

  auto const found = store.get_metadata(key);
  REQUIRE(found);
  CHECK(std::static_pointer_cast<tagged_metadata>(found)->tag == 3);

  // ...and the longer spelling must miss.
  CHECK(store.get_metadata(std::string_view{haystack}) == nullptr);
}

TEST_CASE("metadata_store returns nullptr on miss", "[cache][metadata_store]")
{
  metadata_store store;

  CHECK(store.get_metadata(std::string_view{"absent"}) == nullptr);
  CHECK(store.get_metadata(fake_io_object{"absent"}) == nullptr);
}

TEST_CASE("metadata_store overwrites an existing key and ignores null metadata",
          "[cache][metadata_store]")
{
  metadata_store store;
  fake_io_object const obj{"k"};

  store.register_metadata(obj, std::make_shared<tagged_metadata>(1));
  store.register_metadata(obj, std::make_shared<tagged_metadata>(2));

  auto const after_overwrite = store.get_metadata(std::string_view{"k"});
  REQUIRE(after_overwrite);
  CHECK(std::static_pointer_cast<tagged_metadata>(after_overwrite)->tag == 2);

  // A null registration is silently ignored — it must not erase the entry.
  store.register_metadata(obj, nullptr);
  auto const still_there = store.get_metadata(std::string_view{"k"});
  REQUIRE(still_there);
  CHECK(std::static_pointer_cast<tagged_metadata>(still_there)->tag == 2);
}
