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

#include <cucascade/io/byte_range.hpp>
#include <cucascade/io/io_utils.hpp>

#include <catch2/catch_all.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using cucascade::io::byte_range;

constexpr auto max = std::numeric_limits<std::int64_t>::max();
constexpr auto min = std::numeric_limits<std::int64_t>::min();

}  // namespace

TEST_CASE("io_utils::align_ranges drops bad ranges and keeps the input order", "[io][io_utils]")
{
  std::vector<byte_range> const input{byte_range{5000, 10},
                                      byte_range{-1, 10},
                                      byte_range{0, 0},
                                      byte_range{100, 10},
                                      byte_range{20, -5},
                                      byte_range{max - 1, 10},
                                      byte_range{max, 1}};
  auto out = io_utils::align_ranges(input, 4096);
  REQUIRE(out.size() == 3);
  CHECK(out[0].offset() == 4096);
  CHECK(out[0].size() == 4096);
  CHECK(out[1].offset() == 0);
  CHECK(out[1].size() == 4096);
  // The aligned end would pass INT64_MAX, so it is clamped there (unaligned).
  CHECK(out[2].offset() == max - 4095);
  CHECK(out[2].size() == 4095);
}

TEST_CASE("io_utils::align_and_coalesce drops negative offsets and keeps ends within INT64_MAX",
          "[io][io_utils]")
{
  SECTION("a negative range that sorts first does not swallow a later one")
  {
    std::vector<byte_range> const input{byte_range{-10, 5}, byte_range{0, 50}};
    auto out = io_utils::align_and_coalesce(input, 1);
    REQUIRE(out.size() == 1);
    CHECK(out[0].offset() == 0);
    CHECK(out[0].size() == 50);
  }
  SECTION("a negative range reaching past zero does not absorb a valid one")
  {
    std::vector<byte_range> const input{byte_range{-100, 200}, byte_range{0, 50}};
    auto out = io_utils::align_and_coalesce_sorted(input, 1);
    REQUIRE(out.size() == 1);
    CHECK(out[0].offset() == 0);
    CHECK(out[0].size() == 50);
  }
  SECTION("ranges that are entirely negative or empty are dropped")
  {
    std::vector<byte_range> const input{
      byte_range{-100, 50}, byte_range{min, max}, byte_range{10, 0}, byte_range{20, -5}};
    CHECK(io_utils::align_and_coalesce(input, 4096).empty());
  }
  SECTION("ends near INT64_MAX are clamped, not wrapped")
  {
    std::vector<byte_range> const input{
      byte_range{max - 10, 100}, byte_range{0, max}, byte_range{max - 100, 50}};
    auto out = io_utils::align_and_coalesce(input, 1);
    REQUIRE(out.size() == 1);
    CHECK(out[0].offset() == 0);
    CHECK(out[0].size() == max);
  }
  SECTION("valid ranges plan as before")
  {
    std::vector<byte_range> const input{
      byte_range{9000, 100}, byte_range{100, 100}, byte_range{3000, 100}};
    auto out = io_utils::align_and_coalesce(input, 4096);
    REQUIRE(out.size() == 2);
    CHECK(out[0].offset() == 0);
    CHECK(out[0].size() == 4096);
    CHECK(out[1].offset() == 8192);
    CHECK(out[1].size() == 4096);
  }
}

TEST_CASE("io_utils::coalesce_ranges saturates the merge gap", "[io][io_utils]")
{
  std::vector<byte_range> const input{byte_range{1000, 10}, byte_range{0, 10}, byte_range{-5, 2}};

  SECTION("an unbounded gap merges everything")
  {
    // Unsaturated, cur_end + SIZE_MAX wrapped to cur_end - 1 and merged nothing.
    auto out = io_utils::coalesce_ranges(input, std::numeric_limits<std::size_t>::max());
    REQUIRE(out.size() == 1);
    CHECK(out[0].offset() == 0);
    CHECK(out[0].size() == 1010);
  }
  SECTION("a gap smaller than the hole keeps the ranges apart")
  {
    auto out = io_utils::coalesce_ranges(input, 989);
    REQUIRE(out.size() == 2);
    CHECK(out[0].offset() == 0);
    CHECK(out[0].size() == 10);
    CHECK(out[1].offset() == 1000);
    CHECK(out[1].size() == 10);
  }
  SECTION("the sorted variant drops a negative range before merging")
  {
    std::vector<byte_range> const sorted{
      byte_range{-100, 200}, byte_range{0, 50}, byte_range{60, 10}, byte_range{max - 10, 100}};
    auto out = io_utils::coalesce_sorted_ranges(sorted, 10);
    REQUIRE(out.size() == 2);
    CHECK(out[0].offset() == 0);
    CHECK(out[0].size() == 70);
    CHECK(out[1].offset() == max - 10);
    CHECK(out[1].size() == 10);
  }
}
