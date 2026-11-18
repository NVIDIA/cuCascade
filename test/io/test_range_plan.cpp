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

#include <cucascade/io/range_plan.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <limits>
#include <vector>

namespace {

using cucascade::io::range;
namespace plan = cucascade::io::range_plan;

// Frozen pre-extraction algorithms from HEAD 7641cf5, not forwarding wrappers.
std::array<range, 3> legacy_split_write_range(range rng, std::size_t granule)
{
  auto const begin = rng.offset;
  auto const end   = rng.end();
  if (granule == 0 || rng.empty()) return {rng, range{end, 0}, range{end, 0}};
  auto const first = begin % granule == 0 ? begin : begin - begin % granule + granule;
  auto const last  = end - end % granule;
  if (first >= end || last <= first) { return {rng, range{end, 0}, range{end, 0}}; }
  return {range{begin, first - begin}, range{first, last - first}, range{last, end - last}};
}

range legacy_staged_physical_range(range logical, std::size_t file_size, bool use_odirect)
{
  if (!use_odirect) return logical;
  auto const start = io_utils::align_down(logical.offset, cucascade::io::IO_BLOCK_SIZE);
  auto const end   = std::min(io_utils::align_up(logical.end(), cucascade::io::IO_BLOCK_SIZE),
                            io_utils::align_up(file_size, cucascade::io::IO_BLOCK_SIZE));
  return end > start ? range{start, end - start} : range{start, 0};
}

void check_range(range actual, std::size_t offset, std::size_t size)
{
  CHECK(actual.offset == offset);
  CHECK(actual.size == size);
}

}  // namespace

TEST_CASE("range plan cuts unaligned starts on the absolute grid", "[io][range_plan]")
{
  auto const piece = plan::next_cut(123, 8192, 4096, 4096);
  check_range(piece.logical, 123, 4096 - 123);
  check_range(piece.physical, 0, 4096);
  CHECK_FALSE(piece.last);
}

TEST_CASE("range plan covers an op-size-multiple slice without shifted cuts", "[io][range_plan]")
{
  std::size_t cursor        = 123;
  constexpr std::size_t end = 123 + 8192;
  constexpr std::array ends{std::size_t{4096}, std::size_t{8192}, end};
  for (auto const expected_end : ends) {
    auto const piece = plan::next_cut(cursor, end, 4096, 4096);
    CHECK(piece.logical.offset == cursor);
    CHECK(piece.logical.end() == expected_end);
    CHECK(piece.last == (expected_end == end));
    cursor = expected_end;
  }
  CHECK(cursor == end);
  auto const exhausted = plan::next_cut(cursor, end, 4096, 4096);
  CHECK(exhausted.logical.empty());
  CHECK(exhausted.physical.empty());
  CHECK(exhausted.last);

  auto const aligned = plan::next_cut(0, 8192, 4096, 4096);
  check_range(aligned.logical, 0, 4096);
  CHECK_FALSE(aligned.last);
  auto const aligned_last = plan::next_cut(4096, 8192, 4096, 4096);
  check_range(aligned_last.logical, 4096, 4096);
  CHECK(aligned_last.last);
}

TEST_CASE("range plan geometry boundaries cut before the op grid", "[io][range_plan]")
{
  constexpr std::array<std::size_t, 6> boundaries{0, 100, 100, 700, 4096, 9000};
  auto first = plan::next_cut(100, 10000, 4096, 4096, boundaries);
  check_range(first.logical, 100, 600);
  check_range(first.physical, 0, 4096);
  auto next = plan::next_cut(700, 10000, 4096, 4096, boundaries);
  check_range(next.logical, 700, 4096 - 700);
  CHECK_FALSE(next.last);
  auto last = plan::next_cut(9000, 10000, 4096, 4096, boundaries);
  check_range(last.logical, 9000, 1000);
  CHECK(last.last);
}

TEST_CASE("range plan physical windows clamp to rounded EOF", "[io][range_plan]")
{
  check_range(plan::staged_physical_range({4000, 10000}, 4096, 5000), 0, 8192);
  check_range(plan::staged_physical_range({12288, 4096}, 4096, 5000), 12288, 0);
  check_range(plan::staged_physical_range({123, 5000}, 0, 1000), 123, 877);
  auto const piece = plan::next_cut(4000, 14000, 16384, 4096, {}, 5000);
  check_range(piece.physical, 0, 8192);
}

TEST_CASE("range plan supports granules larger than 4KiB", "[io][range_plan]")
{
  constexpr std::size_t granule = 64UL << 10;
  auto const piece              = plan::next_cut(65000, 200000, 2 * granule, granule, {}, 110000);
  check_range(piece.logical, 65000, 2 * granule - 65000);
  check_range(piece.physical, 0, 2 * granule);
  auto const tail = plan::next_cut(100000, 110000, 2 * granule, granule);
  check_range(tail.physical, granule, granule);
  CHECK(tail.last);
}

TEST_CASE("range plan retains the legacy head-body-tail split", "[io][range_plan]")
{
  for (auto const granule : {std::size_t{0}, std::size_t{4096}, std::size_t{65536}}) {
    for (auto const rng :
         {range{0, 0}, range{123, 10}, range{0, 4096}, range{123, 200000}, range{65536, 131072}}) {
      auto const split   = plan::split_write_range(rng, granule);
      auto const legacy  = legacy_split_write_range(rng, granule);
      std::size_t cursor = rng.offset;
      for (std::size_t i = 0; i < split.size(); ++i) {
        CHECK(split[i].offset == cursor);
        CHECK(split[i].offset == legacy[i].offset);
        CHECK(split[i].size == legacy[i].size);
        cursor = split[i].end();
      }
      CHECK(cursor == rng.end());
    }
  }
  auto const split = plan::split_write_range({123, 200000}, 65536);
  check_range(split[0], 123, 65536 - 123);
  check_range(split[1], 65536, 131072);
  check_range(split[2], 196608, 3515);
}

TEST_CASE("range plan rejects unrepresentable file cuts and preserves widening saturation",
          "[io][range_plan]")
{
  constexpr auto max = std::numeric_limits<std::size_t>::max();
  CHECK_THROWS_AS(plan::next_cut(max - 100, max, 4096, 4096), std::invalid_argument);
  constexpr auto file_max = static_cast<std::size_t>(std::numeric_limits<off_t>::max());
  CHECK_THROWS_AS(plan::next_cut(0, file_max + 1, 4096, 4096), std::invalid_argument);
  auto const piece = plan::next_cut(file_max - 100, file_max, 4096, 4096);
  check_range(piece.logical, file_max - 100, 100);
  check_range(piece.physical, io_utils::align_down(file_max, 4096), 4096);
  CHECK(piece.last);
  // The legacy widening helper is not a submission validator: align_up saturates.
  auto const physical = plan::staged_physical_range({max - 100, 100}, 4096, max);
  check_range(physical, io_utils::align_down(max - 100, 4096), 4095);
  CHECK(physical.end() == max);
  CHECK(physical.end() % 4096 != 0);
}

TEST_CASE("range plan rejects invalid cut parameters", "[io][range_plan]")
{
  CHECK_THROWS_AS(plan::next_cut(10, 9, 4096, 4096), std::invalid_argument);
  CHECK_THROWS_AS(plan::next_cut(0, 9, 0, 4096), std::invalid_argument);
  CHECK_THROWS_AS(plan::next_cut(0, 9, 4096, 0), std::invalid_argument);
  CHECK_THROWS_AS(plan::next_cut(0, 9, 4096, 65536), std::invalid_argument);
  CHECK_THROWS_AS(plan::next_cut(0, 9, 100000, 65536), std::invalid_argument);
}

TEST_CASE("range plan handles empty physical EOF windows and out-of-slice boundaries",
          "[io][range_plan]")
{
  for (auto const cursor : {8192u, 9000u, 12288u}) {
    auto const piece = plan::next_cut(cursor, cursor + 100, 4096, 4096, {}, 5000);
    CHECK_FALSE(piece.logical.empty());
    CHECK(piece.physical.empty());
    CHECK(piece.last);
  }
  constexpr std::array<std::size_t, 3> boundaries{5000, 7000, 9000};
  auto const piece = plan::next_cut(100, 1000, 4096, 4096, boundaries);
  check_range(piece.logical, 100, 900);
  check_range(piece.physical, 0, 4096);
  CHECK(piece.last);
}

TEST_CASE("range plan physical windows fit one op slot over cursor sweeps", "[io][range_plan]")
{
  constexpr std::array<std::size_t, 4> boundaries{13, 511, 4097, 200000};
  for (auto const granule : {512u, 4096u, 65536u}) {
    for (auto const multiple : {1u, 2u, 4u}) {
      std::size_t const op = granule * multiple;
      for (std::size_t cursor = 0; cursor < 3 * op; cursor += 127) {
        auto const piece = plan::next_cut(cursor, 4 * op, op, granule, boundaries);
        CHECK(piece.physical.size <= op);
        CHECK(piece.physical.offset >= io_utils::align_down(cursor, op));
        CHECK(piece.physical.end() <= io_utils::align_down(cursor, op) + op);
      }
    }
  }
}

TEST_CASE("range plan split matches frozen legacy algorithm over grids and saturation",
          "[io][range_plan]")
{
  for (auto const granule : {0u, 1u, 2u, 4u, 8u, 16u, 512u, 4096u, 65536u}) {
    std::vector<std::size_t> points;
    if (granule <= 16) {
      for (std::size_t i = 0; i <= 3 * std::max(1u, granule); ++i) {
        points.push_back(i);
      }
    } else {
      // All neighbours of boundaries on a three-granule grid at realistic sizes.
      for (std::size_t i = 0; i <= 3; ++i) {
        if (i != 0) { points.push_back(i * granule - 1); }
        points.push_back(i * granule);
        points.push_back(i * granule + 1);
      }
    }
    constexpr auto max = std::numeric_limits<std::size_t>::max();
    points.insert(points.end(), {max - 65536, max - 4096, max - 1, max});
    for (auto const offset : points) {
      for (auto const size : points) {
        range const rng{offset, size};
        auto const actual   = plan::split_write_range(rng, granule);
        auto const expected = legacy_split_write_range(rng, granule);
        for (std::size_t i = 0; i < actual.size(); ++i) {
          check_range(actual[i], expected[i].offset, expected[i].size);
        }
      }
    }
  }
}

TEST_CASE("range plan staging matches frozen legacy O_DIRECT formula", "[io][range_plan]")
{
  constexpr auto max = std::numeric_limits<std::size_t>::max();
  constexpr std::array<std::size_t, 14> points{
    0, 1, 4095, 4096, 4097, 8191, 8192, 8193, 12287, 12288, 12289, max - 4096, max - 1, max};
  for (auto const offset : points) {
    for (auto const size : points) {
      range const rng{offset, size};
      for (auto const file_size : {std::size_t{0},
                                   rng.end(),
                                   rng.end() == 0 ? 0 : rng.end() - 1,
                                   io_utils::saturating_add(rng.end(), 1),
                                   max}) {
        for (auto const direct : {false, true}) {
          auto const expected = legacy_staged_physical_range(rng, file_size, direct);
          auto const actual =
            direct ? plan::staged_physical_range(rng, cucascade::io::IO_BLOCK_SIZE, file_size)
                   : rng;
          check_range(actual, expected.offset, expected.size);
        }
      }
    }
  }
}
