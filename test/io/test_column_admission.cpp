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

#include <cucascade/io/column_admission.hpp>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cucascade::io::column_admission;
using cucascade::io::column_admission_config;
using cucascade::io::column_stats;

constexpr std::uint64_t chunk       = 1000;
constexpr std::uint64_t object_size = 100000;

column_admission_config config(std::size_t budget_bytes)
{
  return {.budget_bytes = budget_bytes, .chunk_bytes = chunk, .rank_margin = 1.5};
}

/// One row group of file @p path in table "t": column "a" at [0, 4000) and column "b" at
/// [4000, 8000).
void register_file(column_admission& policy, std::string const& path)
{
  std::vector<std::string> const columns{"a", "b"};
  std::array<std::uint32_t, 2> const column_of_chunk{0, 1};
  std::array<std::uint64_t, 2> const offsets{0, 4000};
  std::array<std::uint64_t, 2> const sizes{4000, 4000};
  policy.register_layout(path, "t", columns, column_of_chunk, offsets, sizes);
}

column_stats stat(column_admission const& policy, std::string const& name)
{
  auto const all = policy.stats();
  auto const it  = std::ranges::find(
    all, name, [](column_stats const& s) -> std::string const& { return s.name; });
  REQUIRE(it != all.end());
  return *it;
}

}  // namespace

TEST_CASE("a read of one column is advised as whole chunks", "[cache][admission]")
{
  column_admission policy(config(1 << 20));
  register_file(policy, "f1");
  auto const advice = policy.advise("f1", 500, 3000, object_size);
  REQUIRE(advice.ranges.size() == 1);
  CHECK(advice.ranges[0].offset() == 0);
  CHECK(advice.ranges[0].size() == 4000);
  CHECK(stat(policy, "t/a").requested == 3000);
  CHECK(stat(policy, "t/b").requested == 0);
}

TEST_CASE("a read across two columns is split by column and touching ranges merge",
          "[cache][admission]")
{
  column_admission policy(config(1 << 20));
  register_file(policy, "f1");
  auto const advice = policy.advise("f1", 0, 8000, object_size);
  REQUIRE(advice.ranges.size() == 1);
  CHECK(advice.ranges[0].offset() == 0);
  CHECK(advice.ranges[0].size() == 8000);
  CHECK(stat(policy, "t/a").admitted == 4000);
  CHECK(stat(policy, "t/b").admitted == 4000);
}

TEST_CASE("advised ranges are clipped to the object", "[cache][admission]")
{
  column_admission policy(config(1 << 20));
  register_file(policy, "f1");
  auto const advice = policy.advise("f1", 4000, 4000, 7500);
  REQUIRE(advice.ranges.size() == 1);
  CHECK(advice.ranges[0].offset() == 4000);
  CHECK(advice.ranges[0].size() == 3500);
}

TEST_CASE("an unregistered file is advised whole, at priority 0", "[cache][admission]")
{
  column_admission policy(config(1 << 20));
  CHECK_FALSE(policy.has_layout("other"));
  auto const advice = policy.advise("other", 1500, 700, object_size);
  REQUIRE(advice.ranges.size() == 1);
  CHECK(advice.ranges[0].offset() == 1000);
  CHECK(advice.ranges[0].size() == 2000);
  CHECK(advice.eviction_priority == 0);
}

TEST_CASE("eviction priority grows with the rank of the column and is capped", "[cache][admission]")
{
  column_admission policy(config(1 << 20));
  register_file(policy, "f1");
  // One full read: rank 1, priority 1 + 2 x 1.
  CHECK(policy.advise("f1", 0, 4000, object_size).eviction_priority == 3);
  // Rank 2 after the second read.
  CHECK(policy.advise("f1", 0, 4000, object_size).eviction_priority == 5);
  for (int i = 0; i < 200; ++i) {
    (void)policy.advise("f1", 0, 4000, object_size);
  }
  CHECK(policy.advise("f1", 0, 4000, object_size).eviction_priority == 250);
}

TEST_CASE("a column that does not fit the budget is refused", "[cache][admission]")
{
  column_admission policy(config(4000));  // room for one column of one file
  register_file(policy, "f1");
  REQUIRE_FALSE(policy.advise("f1", 0, 4000, object_size).ranges.empty());
  auto const refused = policy.advise("f1", 4000, 4000, object_size);
  CHECK(refused.ranges.empty());
  CHECK(stat(policy, "t/b").refused == 4000);
  CHECK(stat(policy, "t/a").admitted_footprint == 4000);
}

TEST_CASE("a column that ranks clearly higher replaces the lowest-ranked one", "[cache][admission]")
{
  column_admission policy(config(4000));
  register_file(policy, "f1");
  REQUIRE_FALSE(policy.advise("f1", 0, 4000, object_size).ranges.empty());  // a admitted, rank 1
  CHECK(policy.advise("f1", 4000, 4000, object_size).ranges.empty());       // b rank 1: not 1.5 x a
  auto const replaced = policy.advise("f1", 4000, 4000, object_size);       // b rank 2 > 1.5 x 1
  REQUIRE_FALSE(replaced.ranges.empty());
  CHECK(stat(policy, "t/b").admitted_footprint == 4000);
  CHECK(stat(policy, "t/a").admitted_footprint == 0);
}

TEST_CASE("an admitted piece stays admitted", "[cache][admission]")
{
  column_admission policy(config(4000));
  register_file(policy, "f1");
  REQUIRE_FALSE(policy.advise("f1", 0, 4000, object_size).ranges.empty());
  // A narrower read of the same column in the same file is admitted without new budget.
  CHECK_FALSE(policy.advise("f1", 1000, 500, object_size).ranges.empty());
  CHECK(stat(policy, "t/a").admitted_footprint == 4000);
}

TEST_CASE("the budget counts whole cache chunks", "[cache][admission]")
{
  column_admission policy(config(1500));
  std::vector<std::string> const columns{"c"};
  std::array<std::uint32_t, 1> const column_of_chunk{0};
  std::array<std::uint64_t, 1> const offsets{950};
  std::array<std::uint64_t, 1> const sizes{100};  // straddles two chunks: a footprint of 2000
  policy.register_layout("f1", "t", columns, column_of_chunk, offsets, sizes);
  CHECK(policy.advise("f1", 950, 100, object_size).ranges.empty());
}

/// Columns "a" at [0, 2500) and "b" at [2500, 5000): chunk 2 holds the end of a and the start of b.
void register_sharing_file(column_admission& policy, std::string const& path)
{
  std::vector<std::string> const columns{"a", "b"};
  std::array<std::uint32_t, 2> const column_of_chunk{0, 1};
  std::array<std::uint64_t, 2> const offsets{0, 2500};
  std::array<std::uint64_t, 2> const sizes{2500, 2500};
  policy.register_layout(path, "t", columns, column_of_chunk, offsets, sizes);
}

TEST_CASE("a chunk shared by two admitted columns is counted once", "[cache][admission]")
{
  column_admission policy(config(5000));  // a and b alone are 3000 each, together 5 chunks
  register_sharing_file(policy, "f1");
  REQUIRE_FALSE(policy.advise("f1", 0, 2500, object_size).ranges.empty());
  CHECK_FALSE(policy.advise("f1", 2500, 2500, object_size).ranges.empty());
  CHECK(stat(policy, "t/a").admitted_footprint == 3000);
  CHECK(stat(policy, "t/b").admitted_footprint == 2000);  // chunk 2 was already paid for
}

TEST_CASE("a replaced column frees only the chunks no other column holds", "[cache][admission]")
{
  column_admission policy(config(5000));
  std::vector<std::string> const columns{"a", "b", "c"};
  std::array<std::uint32_t, 3> const column_of_chunk{0, 1, 2};
  std::array<std::uint64_t, 3> const offsets{0, 2500, 5000};
  std::array<std::uint64_t, 3> const sizes{2500, 2500, 4000};
  policy.register_layout("f1", "t", columns, column_of_chunk, offsets, sizes);
  REQUIRE_FALSE(policy.advise("f1", 0, 2500, object_size).ranges.empty());     // a: chunks 0 to 2
  REQUIRE_FALSE(policy.advise("f1", 2500, 2500, object_size).ranges.empty());  // b: chunks 3 and 4
  // c ranks above a, so a is dropped, but only chunks 0 and 1 are freed (b still holds chunk 2),
  // which is not enough room for c.
  static_cast<void>(policy.advise("f1", 5000, 4000, object_size));
  CHECK(policy.advise("f1", 5000, 4000, object_size).ranges.empty());
  CHECK(stat(policy, "t/a").admitted_footprint == 0);
  // Admitting a again needs only the two chunks it does not share with b.
  CHECK_FALSE(policy.advise("f1", 0, 2500, object_size).ranges.empty());
  CHECK(stat(policy, "t/a").admitted_footprint == 2000);
}

TEST_CASE("columns are ranked across the files of a table", "[cache][admission]")
{
  column_admission policy(config(1 << 20));
  register_file(policy, "f1");
  register_file(policy, "f2");
  (void)policy.advise("f1", 0, 4000, object_size);
  auto const a = stat(policy, "t/a");
  CHECK(a.registered == 8000);
  CHECK(a.rank == Catch::Approx(0.5));
}

TEST_CASE("registering a file twice has no effect", "[cache][admission]")
{
  column_admission policy(config(1 << 20));
  register_file(policy, "f1");
  register_file(policy, "f1");
  CHECK(policy.has_layout("f1"));
  CHECK(stat(policy, "t/a").registered == 4000);
}

TEST_CASE("a malformed layout is rejected", "[cache][admission]")
{
  column_admission policy(config(1 << 20));
  std::vector<std::string> const columns{"a"};
  std::array<std::uint32_t, 1> const column_of_chunk{0};
  std::array<std::uint32_t, 1> const bad_column{3};
  std::array<std::uint64_t, 1> const offsets{0};
  std::array<std::uint64_t, 2> const two_sizes{10, 10};
  std::array<std::uint64_t, 1> const sizes{10};
  CHECK_THROWS_AS(policy.register_layout("f1", "t", columns, column_of_chunk, offsets, two_sizes),
                  std::invalid_argument);
  CHECK_THROWS_AS(policy.register_layout("f1", "t", columns, bad_column, offsets, sizes),
                  std::invalid_argument);
  CHECK_FALSE(policy.has_layout("f1"));
}

TEST_CASE("a zero chunk size is rejected", "[cache][admission]")
{
  CHECK_THROWS_AS(column_admission({.budget_bytes = 1, .chunk_bytes = 0}), std::invalid_argument);
}
