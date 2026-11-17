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

#include "cudf/row_range_detail.hpp"

#include <cudf/types.hpp>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using namespace cucascade;
using cucascade::detail::fit_column;
using cucascade::detail::fit_rows;
using cucascade::detail::validate_request;

namespace {

// fit_rows and validate_request never dereference a slot pointer; they only test it for nullptr.
int dummy_storage   = 0;
void* const present = &dummy_storage;

constexpr std::int32_t int32_id  = static_cast<std::int32_t>(cudf::type_id::INT32);
constexpr std::int32_t int64_id  = static_cast<std::int32_t>(cudf::type_id::INT64);
constexpr std::int32_t string_id = static_cast<std::int32_t>(cudf::type_id::STRING);

column_destination make_fixed_dst(std::size_t data_capacity,
                                  std::size_t dst_row_offset = 0,
                                  std::int32_t type_id       = int32_id)
{
  column_destination dst;
  dst.type_id        = type_id;
  dst.data           = {present, data_capacity};
  dst.dst_row_offset = dst_row_offset;
  return dst;
}

column_destination make_string_dst(std::size_t chars_capacity,
                                   std::size_t offsets_capacity,
                                   std::size_t dst_row_offset  = 0,
                                   std::size_t dst_byte_offset = 0,
                                   std::uint8_t offset_width   = 4)
{
  column_destination dst;
  dst.type_id         = string_id;
  dst.data            = {present, chars_capacity};
  dst.offsets         = {present, offsets_capacity};
  dst.offset_width    = offset_width;
  dst.dst_row_offset  = dst_row_offset;
  dst.dst_byte_offset = dst_byte_offset;
  return dst;
}

fit_column fixed_column(const column_destination& dst, std::size_t width)
{
  fit_column col;
  col.fixed_width = width;
  col.dst         = &dst;
  return col;
}

/// Source offsets (num_rows + 1 entries) of a string column with the given row lengths.
std::vector<std::int64_t> make_offsets(const std::vector<std::int64_t>& lengths,
                                       std::int64_t first = 0)
{
  std::vector<std::int64_t> offsets(lengths.size() + 1);
  offsets[0] = first;
  std::partial_sum(
    lengths.begin(), lengths.end(), offsets.begin() + 1, [](auto a, auto b) { return a + b; });
  for (std::size_t i = 1; i < offsets.size(); ++i) {
    offsets[i] += first;
  }
  return offsets;
}

fit_column string_column(const column_destination& dst,
                         const std::vector<std::int64_t>& offsets,
                         std::size_t* calls = nullptr)
{
  fit_column col;
  col.is_string = true;
  col.offset_at = [&offsets, calls](std::size_t row) {
    if (calls != nullptr) { ++*calls; }
    return offsets.at(row);
  };
  col.dst = &dst;
  return col;
}

std::size_t fit_one(const fit_column& col,
                    std::size_t row_begin,
                    std::size_t max_rows,
                    std::size_t num_rows)
{
  return fit_rows(std::span<const fit_column>(&col, 1), row_begin, max_rows, num_rows);
}

}  // namespace

TEST_CASE("fit_rows fixed-width columns", "[row_range_fit]")
{
  SECTION("capacity exactly fits")
  {
    auto dst = make_fixed_dst(40);
    CHECK(fit_one(fixed_column(dst, 4), 0, 10, 100) == 10);
    CHECK(fit_one(fixed_column(dst, 4), 0, 50, 100) == 10);
  }
  SECTION("one byte short")
  {
    auto dst = make_fixed_dst(39);
    CHECK(fit_one(fixed_column(dst, 4), 0, 10, 100) == 9);
  }
  SECTION("dst_row_offset > 0")
  {
    auto exact = make_fixed_dst(80, 3, int64_id);
    CHECK(fit_one(fixed_column(exact, 8), 0, 100, 100) == 7);
    auto one_short = make_fixed_dst(79, 3, int64_id);
    CHECK(fit_one(fixed_column(one_short, 8), 0, 100, 100) == 6);
  }
  SECTION("append position at or past the capacity")
  {
    auto full = make_fixed_dst(40, 10);
    CHECK(fit_one(fixed_column(full, 4), 0, 10, 100) == 0);
    auto past = make_fixed_dst(40, 1000);
    CHECK(fit_one(fixed_column(past, 4), 0, 10, 100) == 0);
  }
  SECTION("no columns: only the row clamps apply")
  {
    CHECK(fit_rows({}, 2, 10, 100) == 10);
    CHECK(fit_rows({}, 95, 10, 100) == 5);
  }
}

TEST_CASE("fit_rows STRING columns", "[row_range_fit]")
{
  // rows:    0  1  2  3  4  5  6  7
  // lengths: 3  0  5  1  7  2  0  4   -> offsets 0 3 3 8 9 16 18 18 22
  const auto offsets = make_offsets({3, 0, 5, 1, 7, 2, 0, 4});

  SECTION("everything fits")
  {
    auto dst = make_string_dst(22, 9 * 4);
    CHECK(fit_one(string_column(dst, offsets), 0, 8, 8) == 8);
  }
  SECTION("chars budget limits")
  {
    auto exact = make_string_dst(16, 1024);
    CHECK(fit_one(string_column(exact, offsets), 0, 8, 8) == 5);
    auto one_short = make_string_dst(15, 1024);
    CHECK(fit_one(string_column(one_short, offsets), 0, 8, 8) == 4);
    // Empty rows after the last fitting row are free.
    auto trailing_empty = make_string_dst(18, 1024);
    CHECK(fit_one(string_column(trailing_empty, offsets), 0, 8, 8) == 7);
  }
  SECTION("chars budget is relative to row_begin")
  {
    // From row 2: cumulative lengths 5 6 13 15 15 19.
    auto dst = make_string_dst(13, 1024);
    CHECK(fit_one(string_column(dst, offsets), 2, 8, 8) == 3);
    const auto shifted = make_offsets({3, 0, 5, 1, 7, 2, 0, 4}, 1000);
    CHECK(fit_one(string_column(dst, shifted), 2, 8, 8) == 3);
  }
  SECTION("offsets budget limits")
  {
    // k + 1 offsets are written: 6 slots hold 5 rows.
    auto dst = make_string_dst(1024, 6 * 4);
    CHECK(fit_one(string_column(dst, offsets), 0, 8, 8) == 5);
    auto one_short = make_string_dst(1024, 6 * 4 - 1);
    CHECK(fit_one(string_column(one_short, offsets), 0, 8, 8) == 4);
    auto wide = make_string_dst(1024, 6 * 8, 0, 0, 8);
    CHECK(fit_one(string_column(wide, offsets), 0, 8, 8) == 5);
    auto wide_short = make_string_dst(1024, 6 * 8 - 1, 0, 0, 8);
    CHECK(fit_one(string_column(wide_short, offsets), 0, 8, 8) == 4);
    auto appended = make_string_dst(1024, 6 * 4, 2);
    CHECK(fit_one(string_column(appended, offsets), 0, 8, 8) == 3);
    // No room for even the first (k + 1 = 1) offset beyond the append position.
    auto full = make_string_dst(1024, 6 * 4, 6);
    CHECK(fit_one(string_column(full, offsets), 0, 8, 8) == 0);
    auto past = make_string_dst(1024, 6 * 4, 60);
    CHECK(fit_one(string_column(past, offsets), 0, 8, 8) == 0);
  }
  SECTION("both budgets limit")
  {
    // chars allow 5 rows, offsets allow 3.
    auto offsets_tighter = make_string_dst(16, 4 * 4);
    CHECK(fit_one(string_column(offsets_tighter, offsets), 0, 8, 8) == 3);
    // chars allow 2 rows, offsets allow 5.
    auto chars_tighter = make_string_dst(7, 6 * 4);
    CHECK(fit_one(string_column(chars_tighter, offsets), 0, 8, 8) == 2);
  }
  SECTION("dst_byte_offset > 0")
  {
    auto exact = make_string_dst(26, 1024, 0, 10);
    CHECK(fit_one(string_column(exact, offsets), 0, 8, 8) == 5);
    auto one_short = make_string_dst(25, 1024, 0, 10);
    CHECK(fit_one(string_column(one_short, offsets), 0, 8, 8) == 4);
    auto full = make_string_dst(26, 1024, 0, 26);
    CHECK(fit_one(string_column(full, offsets), 0, 8, 8) == 0);
    // Rows 1 and 6 are empty: they fit a full chars slot.
    CHECK(fit_one(string_column(full, offsets), 1, 8, 8) == 1);
    auto past = make_string_dst(26, 1024, 0, 27);
    CHECK(fit_one(string_column(past, offsets), 1, 8, 8) == 0);
  }
}

TEST_CASE("fit_rows returns 0 when a single row exceeds the chars capacity", "[row_range_fit]")
{
  const auto offsets = make_offsets({100, 1, 1});
  auto dst           = make_string_dst(99, 1024);
  CHECK(fit_one(string_column(dst, offsets), 0, 3, 3) == 0);
  auto appended = make_string_dst(150, 1024, 0, 51);
  CHECK(fit_one(string_column(appended, offsets), 0, 3, 3) == 0);
  // The following rows are small and fit.
  CHECK(fit_one(string_column(dst, offsets), 1, 3, 3) == 2);
}

TEST_CASE("fit_rows mixed columns take the minimum", "[row_range_fit]")
{
  const auto offsets = make_offsets(std::vector<std::int64_t>(100, 2));

  auto run = [&](std::size_t int_cap, std::size_t long_cap, std::size_t chars_cap) {
    auto int_dst                 = make_fixed_dst(int_cap);
    auto long_dst                = make_fixed_dst(long_cap, 0, int64_id);
    auto str_dst                 = make_string_dst(chars_cap, 4096);
    std::vector<fit_column> cols = {
      fixed_column(int_dst, 4), string_column(str_dst, offsets), fixed_column(long_dst, 8)};
    return fit_rows(cols, 0, 100, 100);
  };

  CHECK(run(4 * 20, 8 * 50, 2 * 60) == 20);  // INT32 column limits
  CHECK(run(4 * 70, 8 * 30, 2 * 60) == 30);  // INT64 column limits
  CHECK(run(4 * 70, 8 * 50, 2 * 40) == 40);  // STRING chars limit
  CHECK(run(4 * 200, 8 * 200, 2 * 200) == 100);

  SECTION("two string columns")
  {
    const auto long_offsets      = make_offsets(std::vector<std::int64_t>(100, 10));
    auto a_dst                   = make_string_dst(2 * 60, 4096);
    auto b_dst                   = make_string_dst(10 * 25, 4096);
    std::vector<fit_column> cols = {string_column(a_dst, offsets),
                                    string_column(b_dst, long_offsets)};
    CHECK(fit_rows(cols, 0, 100, 100) == 25);
    std::vector<fit_column> reversed = {string_column(b_dst, long_offsets),
                                        string_column(a_dst, offsets)};
    CHECK(fit_rows(reversed, 0, 100, 100) == 25);
  }
  SECTION("a missing destination is rejected")
  {
    fit_column col;
    col.fixed_width = 4;
    CHECK_THROWS_AS(fit_one(col, 0, 10, 10), std::invalid_argument);
  }
}

TEST_CASE("fit_rows null mask term", "[row_range_fit]")
{
  // Two 32-bit words: 64 bits.
  auto fit = [](std::size_t mask_capacity, std::size_t dst_row_offset) {
    auto dst      = make_fixed_dst(1 << 20, dst_row_offset);
    dst.null_mask = {present, mask_capacity};
    return fit_one(fixed_column(dst, 4), 0, 1000, 1000);
  };

  CHECK(fit(8, 0) == 64);
  CHECK(fit(8, 31) == 33);
  CHECK(fit(8, 32) == 32);
  CHECK(fit(8, 33) == 31);
  // One word.
  CHECK(fit(4, 31) == 1);
  CHECK(fit(4, 32) == 0);
  CHECK(fit(4, 33) == 0);
  // A partial trailing word does not count.
  CHECK(fit(7, 31) == 1);
  CHECK(fit(3, 0) == 0);

  SECTION("an absent mask slot does not limit")
  {
    auto dst = make_fixed_dst(4 * 500, 33);
    CHECK(fit_one(fixed_column(dst, 4), 0, 1000, 1000) == 467);
  }
  SECTION("mask on a STRING column")
  {
    const auto offsets = make_offsets(std::vector<std::int64_t>(100, 1));
    auto dst           = make_string_dst(1024, 4096, 33);
    dst.null_mask      = {present, 8};
    CHECK(fit_one(string_column(dst, offsets), 0, 100, 100) == 31);
  }
}

TEST_CASE("fit_rows row clamps", "[row_range_fit]")
{
  auto dst                     = make_fixed_dst(1 << 20);
  const auto offsets           = make_offsets(std::vector<std::int64_t>(10, 1));
  auto str_dst                 = make_string_dst(1024, 4096);
  std::size_t calls            = 0;
  std::vector<fit_column> cols = {fixed_column(dst, 4), string_column(str_dst, offsets, &calls)};

  SECTION("row_begin == num_rows")
  {
    CHECK(fit_rows(cols, 10, 5, 10) == 0);
    CHECK(calls == 0);
  }
  SECTION("row_begin > num_rows")
  {
    CHECK(fit_rows(cols, 11, 5, 10) == 0);
    CHECK(calls == 0);
  }
  SECTION("max_rows clamps")
  {
    CHECK(fit_rows(cols, 0, 4, 10) == 4);
    CHECK(fit_rows(cols, 3, 4, 10) == 4);
  }
  SECTION("max_rows == 0")
  {
    CHECK(fit_rows(cols, 0, 0, 10) == 0);
    CHECK(calls == 0);
  }
  SECTION("num_rows - row_begin clamps")
  {
    CHECK(fit_rows(cols, 7, 100, 10) == 3);
    CHECK(fit_rows(cols, 9, 100, 10) == 1);
  }
}

TEST_CASE("fit_rows calls offset_at a logarithmic number of times", "[row_range_fit]")
{
  const std::size_t num_rows = 1'000'000;
  const auto offsets         = make_offsets(std::vector<std::int64_t>(num_rows, 3));

  for (std::size_t k_max : {std::size_t{1}, std::size_t{2}, std::size_t{1000}, num_rows}) {
    const auto bound = static_cast<std::size_t>(2.0 * std::log2(static_cast<double>(k_max))) + 4;
    // Chars budgets that stop the search at the start, in the middle, near the end, and never.
    for (std::size_t chars_capacity : {std::size_t{0},
                                       std::size_t{2},
                                       std::size_t{3},
                                       3 * (k_max / 3) + 1,
                                       3 * k_max - 1,
                                       3 * k_max}) {
      std::size_t calls       = 0;
      std::size_t max_row     = 0;
      auto dst                = make_string_dst(chars_capacity, 8 * (num_rows + 1));
      const std::size_t begin = num_rows - k_max;
      fit_column col;
      col.is_string = true;
      col.dst       = &dst;
      col.offset_at = [&](std::size_t row) {
        ++calls;
        REQUIRE(row >= begin);
        max_row = std::max(max_row, row);
        return offsets.at(row);
      };
      const auto k = fit_one(col, begin, k_max, num_rows);
      CAPTURE(k_max, chars_capacity);
      CHECK(k == std::min(k_max, chars_capacity / 3));
      CHECK(calls <= bound);
      CHECK(max_row <= begin + k_max);
    }
  }
}

TEST_CASE("validate_request", "[row_range_fit]")
{
  const std::vector<std::size_t> columns = {0, 2};
  row_range_request request;
  request.columns   = columns;
  request.row_begin = 0;
  request.max_rows  = 10;

  std::vector<column_destination> dst = {make_fixed_dst(64), make_string_dst(64, 64)};
  std::vector<std::int32_t> types     = {int32_id, string_id};

  SECTION("valid request")
  {
    CHECK_NOTHROW(validate_request(request, dst, types, 100));
    request.row_begin = 100;
    CHECK_NOTHROW(validate_request(request, dst, types, 100));
    dst[1].offset_width = 8;
    CHECK_NOTHROW(validate_request(request, dst, types, 100));
  }
  SECTION("row_begin past the end")
  {
    request.row_begin = 101;
    CHECK_THROWS_AS(validate_request(request, dst, types, 100), std::out_of_range);
  }
  SECTION("destination count mismatch")
  {
    dst.pop_back();
    CHECK_THROWS_AS(validate_request(request, dst, types, 100), std::invalid_argument);
  }
  SECTION("source type count mismatch")
  {
    types.push_back(int32_id);
    CHECK_THROWS_AS(validate_request(request, dst, types, 100), std::invalid_argument);
  }
  SECTION("unsupported source types")
  {
    for (auto type : {cudf::type_id::LIST, cudf::type_id::STRUCT, cudf::type_id::DICTIONARY32}) {
      types[0]       = static_cast<std::int32_t>(type);
      dst[0].type_id = types[0];
      CHECK_THROWS_AS(validate_request(request, dst, types, 100), row_range_unsupported_error);
      // Unsupported wins over a destination type mismatch.
      dst[0].type_id = int32_id;
      CHECK_THROWS_AS(validate_request(request, dst, types, 100), row_range_unsupported_error);
    }
  }
  SECTION("destination type mismatch")
  {
    dst[0].type_id = int64_id;
    CHECK_THROWS_AS(validate_request(request, dst, types, 100), std::invalid_argument);
  }
  SECTION("STRING without an offsets slot")
  {
    dst[1].offsets = {};
    CHECK_THROWS_AS(validate_request(request, dst, types, 100), std::invalid_argument);
  }
  SECTION("STRING with a bad offset_width")
  {
    for (std::uint8_t width : {std::uint8_t{0}, std::uint8_t{2}, std::uint8_t{16}}) {
      dst[1].offset_width = width;
      CHECK_THROWS_AS(validate_request(request, dst, types, 100), std::invalid_argument);
    }
  }
  SECTION("fixed-width without a data slot")
  {
    dst[0].data = {};
    CHECK_THROWS_AS(validate_request(request, dst, types, 100), std::invalid_argument);
    // Nothing can be written when max_rows == 0, so the slot may be absent.
    request.max_rows = 0;
    CHECK_NOTHROW(validate_request(request, dst, types, 100));
  }
  SECTION("empty request")
  {
    request.columns = {};
    CHECK_NOTHROW(validate_request(request, {}, {}, 100));
  }
}

TEST_CASE("fit_rows matches a brute-force reference on random inputs", "[row_range_fit]")
{
  // Reference: test every k from k_max down with the rule-1 inequalities written out directly.
  struct spec {
    bool is_string;
    std::size_t width;  // fixed width, or offset_width for STRING
    column_destination dst;
    std::vector<std::int64_t> offsets;
  };
  auto fits = [](const spec& s, std::size_t row_begin, std::size_t k) {
    const auto& d = s.dst;
    if (d.null_mask.data != nullptr &&
        (d.dst_row_offset + k + 31) / 32 * 4 > d.null_mask.capacity_bytes) {
      return false;
    }
    if (!s.is_string) { return (d.dst_row_offset + k) * s.width <= d.data.capacity_bytes; }
    const auto chars = static_cast<std::size_t>(s.offsets[row_begin + k] - s.offsets[row_begin]);
    return d.dst_byte_offset + chars <= d.data.capacity_bytes &&
           (d.dst_row_offset + k + 1) * s.width <= d.offsets.capacity_bytes;
  };

  std::mt19937_64 rng(20261112);
  auto uniform = [&](std::size_t lo, std::size_t hi) {
    return std::uniform_int_distribution<std::size_t>(lo, hi)(rng);
  };

  for (int iter = 0; iter < 2000; ++iter) {
    const std::size_t num_rows  = uniform(0, 80);
    const std::size_t row_begin = uniform(0, num_rows);
    const std::size_t max_rows  = uniform(0, 100);
    std::vector<spec> specs(uniform(1, 4));
    for (auto& s : specs) {
      s.is_string = uniform(0, 1) == 1;
      s.width     = s.is_string ? (uniform(0, 1) == 1 ? 8 : 4) : std::size_t{1} << uniform(0, 3);
      if (s.is_string) {
        std::vector<std::int64_t> lengths(num_rows);
        for (auto& len : lengths) {
          len = static_cast<std::int64_t>(uniform(0, 1) == 1 ? 0 : uniform(0, 20));
        }
        s.offsets = make_offsets(lengths, static_cast<std::int64_t>(uniform(0, 50)));
        s.dst     = make_string_dst(uniform(0, 600),
                                uniform(0, 100 * s.width),
                                uniform(0, 40),
                                uniform(0, 60),
                                static_cast<std::uint8_t>(s.width));
      } else {
        s.dst = make_fixed_dst(uniform(0, 100 * s.width), uniform(0, 40));
      }
      if (uniform(0, 1) == 1) { s.dst.null_mask = {present, uniform(0, 20)}; }
    }

    std::vector<fit_column> cols;
    for (const auto& s : specs) {
      cols.push_back(s.is_string ? string_column(s.dst, s.offsets) : fixed_column(s.dst, s.width));
    }

    const std::size_t k_max = row_begin >= num_rows ? 0 : std::min(max_rows, num_rows - row_begin);
    std::size_t expected    = k_max;
    while (expected > 0 && !std::all_of(specs.begin(), specs.end(), [&](const spec& s) {
             return fits(s, row_begin, expected);
           })) {
      --expected;
    }
    CAPTURE(iter, num_rows, row_begin, max_rows);
    REQUIRE(fit_rows(cols, row_begin, max_rows, num_rows) == expected);
  }
}
