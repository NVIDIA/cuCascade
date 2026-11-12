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
#include "utils/mixed_table.hpp"
#include "utils/mock_test_utils.hpp"

#include <cudf_test/column_wrapper.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/row_range_read.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace cucascade;
using namespace cucascade::test;

namespace {

constexpr std::size_t kInt32Col  = 0;
constexpr std::size_t kInt64Col  = 1;
constexpr std::size_t kDoubleCol = 2;
constexpr std::size_t kStringCol = 3;

::cuda::stream_ref default_stream() { return ::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}; }

std::int32_t type_tag(cudf::type_id id) { return static_cast<std::int32_t>(id); }

/// Host truth of a test table: INT32, INT64, FLOAT64 and STRING columns sharing one validity.
struct source_data {
  std::vector<std::int32_t> i32;
  std::vector<std::int64_t> i64;
  std::vector<double> f64;
  std::vector<std::string> str;
  std::vector<bool> valid;

  [[nodiscard]] std::size_t rows() const { return i32.size(); }
};

source_data make_source_data(std::size_t rows)
{
  source_data data;
  for (std::size_t i = 0; i < rows; ++i) {
    data.i32.push_back(static_cast<std::int32_t>(i) * 3 - 7);
    data.i64.push_back(static_cast<std::int64_t>(i) * 1'000'003 + 11);
    data.f64.push_back(static_cast<double>(i) * 0.5 - 3.0);
    // Variable lengths, with an empty string every 11th row. e.g. row 1 -> "b1", row 9 -> "jj9".
    data.str.push_back(i % 11 == 0
                         ? std::string{}
                         : std::string(i % 7, static_cast<char>('a' + i % 26)) + std::to_string(i));
    data.valid.push_back(i % 3 != 0);
  }
  return data;
}

/// Build the cudf table of @p data. Nullable columns carry `data.valid`; null strings are empty.
std::unique_ptr<cudf::table> make_table(const source_data& data,
                                        bool nullable,
                                        bool wide_offsets = false)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::strings_column_wrapper;

  std::vector<std::unique_ptr<cudf::column>> columns;
  if (nullable) {
    columns.push_back(
      fixed_width_column_wrapper<std::int32_t>(data.i32.begin(), data.i32.end(), data.valid.begin())
        .release());
    columns.push_back(
      fixed_width_column_wrapper<std::int64_t>(data.i64.begin(), data.i64.end(), data.valid.begin())
        .release());
    columns.push_back(
      fixed_width_column_wrapper<double>(data.f64.begin(), data.f64.end(), data.valid.begin())
        .release());
    columns.push_back(
      strings_column_wrapper(data.str.begin(), data.str.end(), data.valid.begin()).release());
  } else {
    columns.push_back(
      fixed_width_column_wrapper<std::int32_t>(data.i32.begin(), data.i32.end()).release());
    columns.push_back(
      fixed_width_column_wrapper<std::int64_t>(data.i64.begin(), data.i64.end()).release());
    columns.push_back(
      fixed_width_column_wrapper<double>(data.f64.begin(), data.f64.end()).release());
    columns.push_back(strings_column_wrapper(data.str.begin(), data.str.end()).release());
  }
  if (wide_offsets) {
    columns.back() = widen_string_offsets(
      std::move(columns.back()), default_stream(), rmm::mr::get_current_device_resource_ref());
  }
  default_stream().sync();
  return std::make_unique<cudf::table>(std::move(columns));
}

/// Expected string of row @p row: a null row holds no chars.
std::string expected_string(const source_data& data, std::size_t row, bool nullable)
{
  return nullable && !data.valid[row] ? std::string{} : data.str[row];
}

/// Chars bytes of rows [begin, end) as stored in the cudf column.
std::size_t chars_bytes(const source_data& data, std::size_t begin, std::size_t end, bool nullable)
{
  std::size_t bytes = 0;
  for (std::size_t row = begin; row < end; ++row) {
    bytes += expected_string(data, row, nullable).size();
  }
  return bytes;
}

/// A device buffer of @p bytes filled with @p fill.
rmm::device_buffer filled_buffer(std::size_t bytes, std::uint8_t fill)
{
  rmm::device_buffer buffer(bytes, default_stream());
  if (bytes > 0) { CUCASCADE_CUDA_TRY(cudaMemset(buffer.data(), fill, bytes)); }
  return buffer;
}

template <typename T>
std::vector<T> to_host(const rmm::device_buffer& buffer)
{
  std::vector<T> host(buffer.size() / sizeof(T));
  if (!host.empty()) {
    CUCASCADE_CUDA_TRY(
      cudaMemcpy(host.data(), buffer.data(), host.size() * sizeof(T), cudaMemcpyDeviceToHost));
  }
  return host;
}

/// Destination buffers of one column; an empty buffer is an absent slot.
struct dest_buffers {
  rmm::device_buffer data;
  rmm::device_buffer mask;
  rmm::device_buffer offsets;

  [[nodiscard]] column_destination slot(cudf::type_id type,
                                        std::size_t dst_row_offset  = 0,
                                        std::size_t dst_byte_offset = 0,
                                        std::uint8_t offset_width   = 4)
  {
    column_destination dst;
    dst.type_id         = type_tag(type);
    dst.data            = {data.data(), data.size()};
    dst.null_mask       = {mask.data(), mask.size()};
    dst.offsets         = {offsets.data(), offsets.size()};
    dst.offset_width    = offset_width;
    dst.dst_row_offset  = dst_row_offset;
    dst.dst_byte_offset = dst_byte_offset;
    return dst;
  }
};

constexpr std::uint8_t kFill = 0xA5;

std::size_t mask_bytes(std::size_t rows) { return (rows + 31) / 32 * 4; }

bool bit_at(const std::vector<std::uint32_t>& words, std::size_t bit)
{
  return ((words[bit / 32] >> (bit % 32)) & 1U) != 0;
}

/// Check destination rows [dst_row, dst_row + k) against source rows [row_begin, row_begin + k).
template <typename T>
void expect_fixed(const rmm::device_buffer& buffer,
                  std::size_t dst_row,
                  const std::vector<T>& source,
                  const source_data& data,
                  std::size_t row_begin,
                  std::size_t k,
                  bool nullable)
{
  const auto host = to_host<T>(buffer);
  REQUIRE(host.size() >= dst_row + k);
  for (std::size_t i = 0; i < k; ++i) {
    if (nullable && !data.valid[row_begin + i]) { continue; }
    INFO("row " << i);
    REQUIRE(host[dst_row + i] == source[row_begin + i]);
  }
}

/// Read destination offsets [0, count) as 64-bit values.
std::vector<std::int64_t> host_offsets(const rmm::device_buffer& buffer, std::uint8_t width)
{
  if (width == 8) { return to_host<std::int64_t>(buffer); }
  const auto narrow = to_host<std::int32_t>(buffer);
  return {narrow.begin(), narrow.end()};
}

/// Check the rebased offsets and chars of destination rows [dst_row, dst_row + k).
void expect_strings(const dest_buffers& dst,
                    std::uint8_t offset_width,
                    std::size_t dst_row,
                    std::size_t dst_byte,
                    const source_data& data,
                    std::size_t row_begin,
                    std::size_t k,
                    bool nullable)
{
  const auto offsets = host_offsets(dst.offsets, offset_width);
  const auto chars   = to_host<char>(dst.data);
  REQUIRE(offsets.size() >= dst_row + k + 1);
  REQUIRE(offsets[dst_row] == static_cast<std::int64_t>(dst_byte));
  for (std::size_t i = 0; i < k; ++i) {
    INFO("row " << i);
    const auto begin = offsets[dst_row + i];
    const auto end   = offsets[dst_row + i + 1];
    REQUIRE(begin <= end);
    REQUIRE(static_cast<std::size_t>(end) <= chars.size());
    const std::string actual(chars.data() + begin, static_cast<std::size_t>(end - begin));
    REQUIRE(actual == expected_string(data, row_begin + i, nullable));
  }
}

/// Check validity bits [dst_row, dst_row + k); every other bit must still hold the kFill pattern.
void expect_mask(const rmm::device_buffer& buffer,
                 std::size_t dst_row,
                 const source_data& data,
                 std::size_t row_begin,
                 std::size_t k,
                 bool nullable)
{
  const auto words = to_host<std::uint32_t>(buffer);
  const std::vector<std::uint32_t> untouched(1, 0x01010101U * kFill);
  REQUIRE(words.size() * 32 >= dst_row + k);
  for (std::size_t bit = 0; bit < words.size() * 32; ++bit) {
    INFO("bit " << bit);
    if (bit >= dst_row && bit < dst_row + k) {
      const bool expected = !nullable || data.valid[row_begin + bit - dst_row];
      REQUIRE(bit_at(words, bit) == expected);
    } else {
      REQUIRE(bit_at(words, bit) == bit_at(untouched, bit % 32));
    }
  }
}

row_range_result read(const idata_representation& source,
                      std::span<const std::size_t> columns,
                      std::size_t row_begin,
                      std::size_t max_rows,
                      std::span<column_destination> dst)
{
  row_range_request request;
  request.columns   = columns;
  request.row_begin = row_begin;
  request.max_rows  = max_rows;
  auto result = detail::read_rows_gpu(source, request, dst, default_stream(), nullptr, nullptr);
  default_stream().sync();
  return result;
}

}  // namespace

TEST_CASE("read_rows_gpu reads a mixed table window by window", "[row_range_reader_gpu]")
{
  constexpr std::size_t kRows   = 1000;
  constexpr std::size_t kWindow = 100;
  const bool nullable           = GENERATE(false, true);
  CAPTURE(nullable);

  auto space      = make_mock_memory_space(memory::Tier::GPU);
  const auto data = make_source_data(kRows);
  gpu_table_representation source(make_table(data, nullable), *space, default_stream());

  const std::size_t all_chars = chars_bytes(data, 0, kRows, nullable);
  const std::size_t mask_size = nullable ? mask_bytes(kRows) + 4 : 0;
  dest_buffers i32{filled_buffer(kRows * 4, kFill), filled_buffer(mask_size, kFill), {}};
  dest_buffers i64{filled_buffer(kRows * 8, kFill), filled_buffer(mask_size, kFill), {}};
  dest_buffers f64{filled_buffer(kRows * 8, kFill), filled_buffer(mask_size, kFill), {}};
  dest_buffers str{filled_buffer(all_chars, kFill),
                   filled_buffer(mask_size, kFill),
                   filled_buffer((kRows + 1) * 4, kFill)};

  const std::vector<std::size_t> columns{kInt32Col, kInt64Col, kDoubleCol, kStringCol};
  std::size_t rows_done  = 0;
  std::size_t chars_done = 0;
  for (std::size_t window = 0; window < kRows / kWindow; ++window) {
    CAPTURE(window);
    std::vector<column_destination> dst{i32.slot(cudf::type_id::INT32, rows_done),
                                        i64.slot(cudf::type_id::INT64, rows_done),
                                        f64.slot(cudf::type_id::FLOAT64, rows_done),
                                        str.slot(cudf::type_id::STRING, rows_done, chars_done)};
    const auto result = read(source, columns, rows_done, kWindow, dst);

    const std::size_t window_chars = chars_bytes(data, rows_done, rows_done + kWindow, nullable);
    REQUIRE(result.rows_written == kWindow);
    REQUIRE(result.data_bytes_written ==
            std::vector<std::size_t>{kWindow * 4, kWindow * 8, kWindow * 8, window_chars});
    REQUIRE(result.source_exhausted == (rows_done + kWindow == kRows));
    rows_done += kWindow;
    chars_done += window_chars;
  }

  // The concatenation of all windows equals the source.
  expect_fixed(i32.data, 0, data.i32, data, 0, kRows, nullable);
  expect_fixed(i64.data, 0, data.i64, data, 0, kRows, nullable);
  expect_fixed(f64.data, 0, data.f64, data, 0, kRows, nullable);
  expect_strings(str, 4, 0, 0, data, 0, kRows, nullable);
  if (nullable) {
    expect_mask(i32.mask, 0, data, 0, kRows, nullable);
    expect_mask(i64.mask, 0, data, 0, kRows, nullable);
    expect_mask(f64.mask, 0, data, 0, kRows, nullable);
    expect_mask(str.mask, 0, data, 0, kRows, nullable);
  }
}

TEST_CASE("read_rows_gpu truncates to the destination budget", "[row_range_reader_gpu]")
{
  constexpr std::size_t kRows = 200;
  auto space                  = make_mock_memory_space(memory::Tier::GPU);
  const auto data             = make_source_data(kRows);
  gpu_table_representation source(make_table(data, false), *space, default_stream());

  SECTION("STRING chars capacity fits only part of max_rows")
  {
    constexpr std::size_t kRowBegin = 5;
    constexpr std::size_t kMaxRows  = 100;
    // Room for 37 rows and one byte less than the 38th needs (row 42 is "aa42"-like, not empty).
    const std::size_t capacity = chars_bytes(data, kRowBegin, kRowBegin + 38, false) - 1;
    REQUIRE(!data.str[kRowBegin + 37].empty());

    dest_buffers str{filled_buffer(capacity, kFill), {}, filled_buffer((kMaxRows + 1) * 4, kFill)};
    const std::vector<std::size_t> columns{kStringCol, kInt64Col};
    dest_buffers i64{filled_buffer(kMaxRows * 8, kFill), {}, {}};
    std::vector<column_destination> dst{str.slot(cudf::type_id::STRING),
                                        i64.slot(cudf::type_id::INT64)};
    const auto result = read(source, columns, kRowBegin, kMaxRows, dst);

    REQUIRE(result.rows_written == 37);
    REQUIRE(result.rows_written < kMaxRows);
    REQUIRE_FALSE(result.source_exhausted);
    REQUIRE(result.data_bytes_written ==
            std::vector<std::size_t>{chars_bytes(data, kRowBegin, kRowBegin + 37, false), 37 * 8});
    expect_strings(str, 4, 0, 0, data, kRowBegin, 37, false);
    expect_fixed(i64.data, 0, data.i64, data, kRowBegin, 37, false);

    // Nothing is written past the delivered rows.
    const auto offsets = to_host<std::uint32_t>(str.offsets);
    REQUIRE(offsets[38] == 0x01010101U * kFill);
    const auto values = to_host<std::uint64_t>(i64.data);
    REQUIRE(values[37] == 0x0101010101010101ULL * kFill);
  }

  SECTION("STRING offsets capacity limits the rows")
  {
    const std::size_t all_chars = chars_bytes(data, 0, kRows, false);
    // 11 offsets hold 10 rows.
    dest_buffers str{filled_buffer(all_chars, kFill), {}, filled_buffer(11 * 4, kFill)};
    const std::vector<std::size_t> columns{kStringCol};
    std::vector<column_destination> dst{str.slot(cudf::type_id::STRING)};
    const auto result = read(source, columns, 0, 50, dst);
    REQUIRE(result.rows_written == 10);
    expect_strings(str, 4, 0, 0, data, 0, 10, false);
  }

  SECTION("fixed-width capacity limits the rows")
  {
    dest_buffers i64{filled_buffer(10 * 8 + 4, kFill), {}, {}};
    const std::vector<std::size_t> columns{kInt64Col};
    std::vector<column_destination> dst{i64.slot(cudf::type_id::INT64)};
    const auto result = read(source, columns, 20, 50, dst);
    REQUIRE(result.rows_written == 10);
    REQUIRE_FALSE(result.source_exhausted);
    REQUIRE(result.data_bytes_written == std::vector<std::size_t>{80});
    expect_fixed(i64.data, 0, data.i64, data, 20, 10, false);
  }

  SECTION("a capacity smaller than one row delivers nothing")
  {
    // Row 1 is "b1": two bytes do not fit a one-byte slot.
    REQUIRE(data.str[1].size() == 2);
    dest_buffers str{filled_buffer(1, kFill), {}, filled_buffer(64, kFill)};
    const std::vector<std::size_t> columns{kStringCol};
    std::vector<column_destination> dst{str.slot(cudf::type_id::STRING)};
    const auto result = read(source, columns, 1, 10, dst);

    REQUIRE(result.rows_written == 0);
    REQUIRE_FALSE(result.source_exhausted);
    REQUIRE(result.data_bytes_written == std::vector<std::size_t>{0});
    // k == 0 writes nothing, not even the first offset.
    REQUIRE(to_host<std::uint32_t>(str.offsets)[0] == 0x01010101U * kFill);
    REQUIRE(to_host<std::uint8_t>(str.data)[0] == kFill);
  }
}

TEST_CASE("read_rows_gpu rebases STRING offsets", "[row_range_reader_gpu]")
{
  constexpr std::size_t kRows     = 120;
  constexpr std::size_t kRowBegin = 17;
  constexpr std::size_t kMaxRows  = 60;
  constexpr std::size_t kDstRow   = 7;
  constexpr std::size_t kDstByte  = 13;
  const bool wide_source          = GENERATE(false, true);
  const std::uint8_t offset_width = GENERATE(std::uint8_t{4}, std::uint8_t{8});
  CAPTURE(wide_source, static_cast<int>(offset_width));

  auto space      = make_mock_memory_space(memory::Tier::GPU);
  const auto data = make_source_data(kRows);
  auto table      = make_table(data, false, wide_source);
  REQUIRE(table->view().column(kStringCol).child(0).type().id() ==
          (wide_source ? cudf::type_id::INT64 : cudf::type_id::INT32));
  gpu_table_representation source(std::move(table), *space, default_stream());

  const std::size_t bytes = chars_bytes(data, kRowBegin, kRowBegin + kMaxRows, false);
  dest_buffers str{filled_buffer(kDstByte + bytes, kFill),
                   {},
                   filled_buffer((kDstRow + kMaxRows + 1) * offset_width + 16, kFill)};
  const std::vector<std::size_t> columns{kStringCol};
  std::vector<column_destination> dst{
    str.slot(cudf::type_id::STRING, kDstRow, kDstByte, offset_width)};
  const auto result = read(source, columns, kRowBegin, kMaxRows, dst);

  REQUIRE(result.rows_written == kMaxRows);
  REQUIRE(result.data_bytes_written == std::vector<std::size_t>{bytes});
  REQUIRE_FALSE(result.source_exhausted);
  expect_strings(str, offset_width, kDstRow, kDstByte, data, kRowBegin, kMaxRows, false);

  // Offsets below dst_row_offset and past the k + 1 written ones are untouched, as are the chars
  // below dst_byte_offset.
  const auto raw_offsets = to_host<std::uint8_t>(str.offsets);
  for (std::size_t i = 0; i < raw_offsets.size(); ++i) {
    const bool written = i >= kDstRow * offset_width && i < (kDstRow + kMaxRows + 1) * offset_width;
    if (!written) {
      INFO("offsets byte " << i);
      REQUIRE(raw_offsets[i] == kFill);
    }
  }
  const auto chars = to_host<std::uint8_t>(str.data);
  REQUIRE(std::all_of(chars.begin(), chars.begin() + kDstByte, [](auto c) { return c == kFill; }));
}

TEST_CASE("read_rows_gpu writes null bits at unaligned offsets", "[row_range_reader_gpu]")
{
  constexpr std::size_t kRows     = 200;
  constexpr std::size_t kRowBegin = 3;
  constexpr std::size_t kMaxRows  = 70;
  const std::size_t dst_row       = GENERATE(std::size_t{0}, std::size_t{5}, std::size_t{33});
  CAPTURE(dst_row);

  auto space      = make_mock_memory_space(memory::Tier::GPU);
  const auto data = make_source_data(kRows);
  gpu_table_representation source(make_table(data, true), *space, default_stream());

  const std::size_t all_chars = chars_bytes(data, 0, kRows, true);
  const std::size_t mask_size = mask_bytes(dst_row + kMaxRows) + 4;
  dest_buffers i32{
    filled_buffer((dst_row + kMaxRows) * 4, kFill), filled_buffer(mask_size, kFill), {}};
  dest_buffers str{filled_buffer(all_chars, kFill),
                   filled_buffer(mask_size, kFill),
                   filled_buffer((dst_row + kMaxRows + 1) * 4, kFill)};
  const std::vector<std::size_t> columns{kInt32Col, kStringCol};
  std::vector<column_destination> dst{i32.slot(cudf::type_id::INT32, dst_row),
                                      str.slot(cudf::type_id::STRING, dst_row)};
  const auto result = read(source, columns, kRowBegin, kMaxRows, dst);

  REQUIRE(result.rows_written == kMaxRows);
  expect_fixed(i32.data, dst_row, data.i32, data, kRowBegin, kMaxRows, true);
  expect_strings(str, 4, dst_row, 0, data, kRowBegin, kMaxRows, true);
  // Bits below dst_row_offset and at/after dst_row_offset + k keep their previous value.
  expect_mask(i32.mask, dst_row, data, kRowBegin, kMaxRows, true);
  expect_mask(str.mask, dst_row, data, kRowBegin, kMaxRows, true);
}

TEST_CASE("read_rows_gpu null-mask slot rules", "[row_range_reader_gpu]")
{
  constexpr std::size_t kRows = 100;
  auto space                  = make_mock_memory_space(memory::Tier::GPU);
  const auto data             = make_source_data(kRows);
  const std::vector<std::size_t> columns{kInt32Col};

  SECTION("a source with nulls and no null_mask slot is unsupported")
  {
    gpu_table_representation source(make_table(data, true), *space, default_stream());
    dest_buffers i32{filled_buffer(kRows * 4, kFill), {}, {}};
    std::vector<column_destination> dst{i32.slot(cudf::type_id::INT32)};
    REQUIRE_THROWS_AS(read(source, columns, 0, kRows, dst), row_range_unsupported_error);
    // Nothing was written.
    REQUIRE(to_host<std::uint32_t>(i32.data)[0] == 0x01010101U * kFill);
  }

  SECTION("a non-nullable source sets the bits of a present null_mask slot")
  {
    gpu_table_representation source(make_table(data, false), *space, default_stream());
    constexpr std::size_t kDstRow = 9;
    constexpr std::size_t kK      = 50;
    dest_buffers i32{
      filled_buffer((kDstRow + kK) * 4, kFill), filled_buffer(mask_bytes(kDstRow + kK) + 4, 0), {}};
    std::vector<column_destination> dst{i32.slot(cudf::type_id::INT32, kDstRow)};
    const auto result = read(source, columns, 10, kK, dst);

    REQUIRE(result.rows_written == kK);
    expect_fixed(i32.data, kDstRow, data.i32, data, 10, kK, false);
    const auto words = to_host<std::uint32_t>(i32.mask);
    for (std::size_t bit = 0; bit < words.size() * 32; ++bit) {
      INFO("bit " << bit);
      REQUIRE(bit_at(words, bit) == (bit >= kDstRow && bit < kDstRow + kK));
    }
  }

  SECTION("a non-nullable source needs no null_mask slot")
  {
    gpu_table_representation source(make_table(data, false), *space, default_stream());
    dest_buffers i32{filled_buffer(kRows * 4, kFill), {}, {}};
    std::vector<column_destination> dst{i32.slot(cudf::type_id::INT32)};
    const auto result = read(source, columns, 0, kRows, dst);
    REQUIRE(result.rows_written == kRows);
    REQUIRE(result.source_exhausted);
    expect_fixed(i32.data, 0, data.i32, data, 0, kRows, false);
  }
}

TEST_CASE("read_rows_gpu on an exhausted source", "[row_range_reader_gpu]")
{
  constexpr std::size_t kRows = 64;
  auto space                  = make_mock_memory_space(memory::Tier::GPU);
  const auto data             = make_source_data(kRows);
  gpu_table_representation source(make_table(data, true), *space, default_stream());

  dest_buffers i32{filled_buffer(64, kFill), filled_buffer(8, kFill), {}};
  dest_buffers str{filled_buffer(64, kFill), filled_buffer(8, kFill), filled_buffer(64, kFill)};
  const std::vector<std::size_t> columns{kInt32Col, kStringCol};
  std::vector<column_destination> dst{i32.slot(cudf::type_id::INT32),
                                      str.slot(cudf::type_id::STRING)};

  SECTION("row_begin == num_rows reads nothing and reports exhaustion")
  {
    const auto result = read(source, columns, kRows, 10, dst);
    REQUIRE(result.rows_written == 0);
    REQUIRE(result.source_exhausted);
    REQUIRE(result.data_bytes_written == std::vector<std::size_t>{0, 0});
    REQUIRE(to_host<std::uint32_t>(i32.data)[0] == 0x01010101U * kFill);
    REQUIRE(to_host<std::uint32_t>(str.offsets)[0] == 0x01010101U * kFill);
    REQUIRE(to_host<std::uint32_t>(i32.mask)[0] == 0x01010101U * kFill);
  }

  SECTION("the last rows report exhaustion")
  {
    const auto result = read(source, columns, kRows - 4, 10, dst);
    REQUIRE(result.rows_written == 4);
    REQUIRE(result.source_exhausted);
    expect_fixed(i32.data, 0, data.i32, data, kRows - 4, 4, true);
    expect_strings(str, 4, 0, 0, data, kRows - 4, 4, true);
    expect_mask(i32.mask, 0, data, kRows - 4, 4, true);
  }

  SECTION("row_begin > num_rows is out of range")
  {
    REQUIRE_THROWS_AS(read(source, columns, kRows + 1, 10, dst), std::out_of_range);
  }

  SECTION("a column index past the table is out of range")
  {
    const std::vector<std::size_t> bad_columns{kInt32Col, 4};
    REQUIRE_THROWS_AS(read(source, bad_columns, 0, 10, dst), std::out_of_range);
  }
}

TEST_CASE("read_rows_gpu reads a sliced source table", "[row_range_reader_gpu]")
{
  constexpr std::size_t kRows       = 200;
  constexpr std::size_t kSliceBegin = 50;
  constexpr std::size_t kSliceEnd   = 150;
  constexpr std::size_t kRowBegin   = 10;
  constexpr std::size_t kMaxRows    = 60;
  constexpr std::size_t kDstRow     = 3;
  const bool wide_source            = GENERATE(false, true);
  CAPTURE(wide_source);

  auto space                         = make_mock_memory_space(memory::Tier::GPU);
  const auto data                    = make_source_data(kRows);
  std::shared_ptr<cudf::table> owner = make_table(data, true, wide_source);
  const auto sliced                  = cudf::slice(owner->view(),
                                                   {static_cast<cudf::size_type>(kSliceBegin),
                                                    static_cast<cudf::size_type>(kSliceEnd)})
                        .front();
  REQUIRE(sliced.column(kStringCol).offset() == static_cast<cudf::size_type>(kSliceBegin));
  const auto alloc_size = owner->alloc_size();
  gpu_table_representation source(sliced, std::move(owner), alloc_size, *space, default_stream());

  // Row r of the slice is row kSliceBegin + r of the data.
  const std::size_t data_begin = kSliceBegin + kRowBegin;
  const std::size_t bytes      = chars_bytes(data, data_begin, data_begin + kMaxRows, true);
  const std::size_t mask_size  = mask_bytes(kDstRow + kMaxRows) + 4;
  dest_buffers i32{
    filled_buffer((kDstRow + kMaxRows) * 4, kFill), filled_buffer(mask_size, kFill), {}};
  dest_buffers f64{
    filled_buffer((kDstRow + kMaxRows) * 8, kFill), filled_buffer(mask_size, kFill), {}};
  dest_buffers str{filled_buffer(bytes, kFill),
                   filled_buffer(mask_size, kFill),
                   filled_buffer((kDstRow + kMaxRows + 1) * 4, kFill)};
  const std::vector<std::size_t> columns{kStringCol, kDoubleCol, kInt32Col};
  std::vector<column_destination> dst{str.slot(cudf::type_id::STRING, kDstRow),
                                      f64.slot(cudf::type_id::FLOAT64, kDstRow),
                                      i32.slot(cudf::type_id::INT32, kDstRow)};

  SECTION("a window in the middle of the slice")
  {
    const auto result = read(source, columns, kRowBegin, kMaxRows, dst);
    REQUIRE(result.rows_written == kMaxRows);
    REQUIRE_FALSE(result.source_exhausted);
    REQUIRE(result.data_bytes_written ==
            std::vector<std::size_t>{bytes, kMaxRows * 8, kMaxRows * 4});
    expect_strings(str, 4, kDstRow, 0, data, data_begin, kMaxRows, true);
    expect_fixed(f64.data, kDstRow, data.f64, data, data_begin, kMaxRows, true);
    expect_fixed(i32.data, kDstRow, data.i32, data, data_begin, kMaxRows, true);
    expect_mask(str.mask, kDstRow, data, data_begin, kMaxRows, true);
    expect_mask(f64.mask, kDstRow, data, data_begin, kMaxRows, true);
    expect_mask(i32.mask, kDstRow, data, data_begin, kMaxRows, true);
  }

  SECTION("the end of the slice, not of the underlying table, exhausts the source")
  {
    const std::size_t slice_rows = kSliceEnd - kSliceBegin;
    const auto result            = read(source, columns, slice_rows - 20, kMaxRows, dst);
    REQUIRE(result.rows_written == 20);
    REQUIRE(result.source_exhausted);
    expect_strings(str, 4, kDstRow, 0, data, kSliceEnd - 20, 20, true);
    expect_fixed(i32.data, kDstRow, data.i32, data, kSliceEnd - 20, 20, true);
    expect_mask(i32.mask, kDstRow, data, kSliceEnd - 20, 20, true);
  }
}

TEST_CASE("read_rows_gpu refuses unsupported requests", "[row_range_reader_gpu]")
{
  auto space = make_mock_memory_space(memory::Tier::GPU);

  std::vector<std::unique_ptr<cudf::column>> table_columns;
  table_columns.push_back(cudf::test::fixed_width_column_wrapper<std::int32_t>{1, 2, 3}.release());
  table_columns.push_back(
    cudf::test::lists_column_wrapper<std::int32_t>{{1, 2}, {3}, {4, 5, 6}}.release());
  default_stream().sync();
  gpu_table_representation source(
    std::make_unique<cudf::table>(std::move(table_columns)), *space, default_stream());

  dest_buffers buffers{filled_buffer(64, kFill), filled_buffer(8, kFill), filled_buffer(64, kFill)};

  SECTION("a LIST column is unsupported")
  {
    const std::vector<std::size_t> columns{1};
    std::vector<column_destination> dst{buffers.slot(cudf::type_id::LIST)};
    REQUIRE_THROWS_AS(read(source, columns, 0, 3, dst), row_range_unsupported_error);
  }

  SECTION("a LIST column next to a supported one is unsupported and writes nothing")
  {
    dest_buffers i32{filled_buffer(64, kFill), {}, {}};
    const std::vector<std::size_t> columns{0, 1};
    std::vector<column_destination> dst{i32.slot(cudf::type_id::INT32),
                                        buffers.slot(cudf::type_id::LIST)};
    REQUIRE_THROWS_AS(read(source, columns, 0, 3, dst), row_range_unsupported_error);
    REQUIRE(to_host<std::uint32_t>(i32.data)[0] == 0x01010101U * kFill);
  }

  SECTION("a destination type that differs from the source is invalid")
  {
    const std::vector<std::size_t> columns{0};
    std::vector<column_destination> dst{buffers.slot(cudf::type_id::INT64)};
    REQUIRE_THROWS_AS(read(source, columns, 0, 3, dst), std::invalid_argument);
  }

  SECTION("a destination count that differs from the request is invalid")
  {
    const std::vector<std::size_t> columns{0};
    std::vector<column_destination> dst;
    REQUIRE_THROWS_AS(read(source, columns, 0, 3, dst), std::invalid_argument);
  }

  SECTION("the supported column of the same table still reads")
  {
    dest_buffers i32{filled_buffer(12, kFill), {}, {}};
    const std::vector<std::size_t> columns{0};
    std::vector<column_destination> dst{i32.slot(cudf::type_id::INT32)};
    const auto result = read(source, columns, 0, 3, dst);
    REQUIRE(result.rows_written == 3);
    REQUIRE(result.source_exhausted);
    REQUIRE(to_host<std::int32_t>(i32.data) == std::vector<std::int32_t>{1, 2, 3});
  }
}

TEST_CASE("read_rows_gpu edge cases", "[row_range_reader_gpu]")
{
  auto space = make_mock_memory_space(memory::Tier::GPU);

  SECTION("an empty table is exhausted at row 0")
  {
    const auto data = make_source_data(0);
    gpu_table_representation source(make_table(data, false), *space, default_stream());
    dest_buffers i32{filled_buffer(16, kFill), {}, {}};
    dest_buffers str{filled_buffer(16, kFill), {}, filled_buffer(16, kFill)};
    const std::vector<std::size_t> columns{kInt32Col, kStringCol};
    std::vector<column_destination> dst{i32.slot(cudf::type_id::INT32),
                                        str.slot(cudf::type_id::STRING)};
    const auto result = read(source, columns, 0, 10, dst);
    REQUIRE(result.rows_written == 0);
    REQUIRE(result.source_exhausted);
    REQUIRE(result.data_bytes_written == std::vector<std::size_t>{0, 0});
  }

  SECTION("all-empty strings fit a zero-byte chars slot")
  {
    const std::vector<std::string> empties(40, std::string{});
    std::vector<std::unique_ptr<cudf::column>> table_columns;
    table_columns.push_back(
      cudf::test::strings_column_wrapper(empties.begin(), empties.end()).release());
    default_stream().sync();
    gpu_table_representation source(
      std::make_unique<cudf::table>(std::move(table_columns)), *space, default_stream());

    constexpr std::size_t kDstRow = 2;
    dest_buffers str{{}, {}, filled_buffer((kDstRow + 41) * 4, kFill)};
    const std::vector<std::size_t> columns{0};
    std::vector<column_destination> dst{str.slot(cudf::type_id::STRING, kDstRow, 0)};
    const auto result = read(source, columns, 5, 100, dst);
    REQUIRE(result.rows_written == 35);
    REQUIRE(result.source_exhausted);
    REQUIRE(result.data_bytes_written == std::vector<std::size_t>{0});
    const auto offsets = to_host<std::uint32_t>(str.offsets);
    for (std::size_t i = 0; i <= 35; ++i) {
      INFO("offset " << i);
      REQUIRE(offsets[kDstRow + i] == 0);
    }
    REQUIRE(offsets[kDstRow + 36] == 0x01010101U * kFill);
    REQUIRE(offsets[0] == 0x01010101U * kFill);
  }

  SECTION("an all-null STRING column without children writes flat offsets")
  {
    constexpr std::size_t kRows    = 40;
    constexpr std::size_t kDstRow  = 2;
    constexpr std::size_t kDstByte = 6;
    const std::uint8_t width       = GENERATE(std::uint8_t{4}, std::uint8_t{8});
    CAPTURE(static_cast<int>(width));
    std::vector<std::unique_ptr<cudf::column>> table_columns;
    table_columns.push_back(std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::STRING},
      static_cast<cudf::size_type>(kRows),
      rmm::device_buffer{},
      cudf::create_null_mask(static_cast<cudf::size_type>(kRows), cudf::mask_state::ALL_NULL),
      static_cast<cudf::size_type>(kRows)));
    default_stream().sync();
    REQUIRE(table_columns.front()->num_children() == 0);
    gpu_table_representation source(
      std::make_unique<cudf::table>(std::move(table_columns)), *space, default_stream());

    dest_buffers str{filled_buffer(kDstByte, kFill),
                     filled_buffer(mask_bytes(kDstRow + kRows), kFill),
                     filled_buffer((kDstRow + kRows + 2) * width, kFill)};
    const std::vector<std::size_t> columns{0};
    std::vector<column_destination> dst{str.slot(cudf::type_id::STRING, kDstRow, kDstByte, width)};
    const auto result = read(source, columns, 5, 100, dst);
    REQUIRE(result.rows_written == 35);
    REQUIRE(result.source_exhausted);
    REQUIRE(result.data_bytes_written == std::vector<std::size_t>{0});

    const auto offsets = host_offsets(str.offsets, width);
    for (std::size_t i = 0; i <= 35; ++i) {
      INFO("offset " << i);
      REQUIRE(offsets[kDstRow + i] == static_cast<std::int64_t>(kDstByte));
    }
    // Offsets outside the k + 1 written ones are untouched, and every delivered row is null.
    const auto raw = to_host<std::uint8_t>(str.offsets);
    REQUIRE(raw[kDstRow * width - 1] == kFill);
    REQUIRE(raw[(kDstRow + 36) * width] == kFill);
    const auto words = to_host<std::uint32_t>(str.mask);
    for (std::size_t bit = kDstRow; bit < kDstRow + 35; ++bit) {
      INFO("bit " << bit);
      REQUIRE_FALSE(bit_at(words, bit));
    }
  }

  SECTION("a nullable column with no nulls needs no null_mask slot")
  {
    constexpr std::size_t kRows = 50;
    const std::vector<std::int32_t> values(kRows, 42);
    const std::vector<bool> valid(kRows, true);
    std::vector<std::unique_ptr<cudf::column>> table_columns;
    table_columns.push_back(cudf::test::fixed_width_column_wrapper<std::int32_t>(
                              values.begin(), values.end(), valid.begin())
                              .release());
    default_stream().sync();
    REQUIRE(table_columns.front()->nullable());
    REQUIRE(table_columns.front()->null_count() == 0);
    gpu_table_representation source(
      std::make_unique<cudf::table>(std::move(table_columns)), *space, default_stream());

    dest_buffers i32{filled_buffer(kRows * 4, kFill), {}, {}};
    const std::vector<std::size_t> columns{0};
    std::vector<column_destination> dst{i32.slot(cudf::type_id::INT32)};
    const auto result = read(source, columns, 0, kRows, dst);
    REQUIRE(result.rows_written == kRows);
    REQUIRE(result.source_exhausted);
    REQUIRE(to_host<std::int32_t>(i32.data) == values);
  }

  SECTION("max_rows == 0 reads nothing and is not exhausted")
  {
    const auto data = make_source_data(20);
    gpu_table_representation source(make_table(data, true), *space, default_stream());
    dest_buffers str{filled_buffer(64, kFill), filled_buffer(8, kFill), filled_buffer(64, kFill)};
    const std::vector<std::size_t> columns{kStringCol};
    std::vector<column_destination> dst{str.slot(cudf::type_id::STRING)};
    const auto result = read(source, columns, 3, 0, dst);
    REQUIRE(result.rows_written == 0);
    REQUIRE_FALSE(result.source_exhausted);
    REQUIRE(to_host<std::uint32_t>(str.offsets)[0] == 0x01010101U * kFill);
    REQUIRE(to_host<std::uint32_t>(str.mask)[0] == 0x01010101U * kFill);
  }
}

TEST_CASE("read_rows_gpu reads only the offsets it needs", "[row_range_reader_gpu]")
{
  // More rows than the first offsets chunk (1024), so the chunked host read grows several times.
  constexpr std::size_t kRows = 20000;
  auto space                  = make_mock_memory_space(memory::Tier::GPU);
  const auto data             = make_source_data(kRows);
  const bool wide_source      = GENERATE(false, true);
  CAPTURE(wide_source);
  gpu_table_representation source(make_table(data, false, wide_source), *space, default_stream());
  const std::vector<std::size_t> columns{kStringCol};

  SECTION("a huge max_rows with a chars slot that binds past the first chunk")
  {
    constexpr std::size_t kRowBegin = 100;
    constexpr std::size_t kFit      = 5000;
    REQUIRE(!data.str[kRowBegin + kFit].empty());
    const std::size_t capacity = chars_bytes(data, kRowBegin, kRowBegin + kFit + 1, false) - 1;
    dest_buffers str{filled_buffer(capacity, kFill), {}, filled_buffer((kRows + 1) * 4, kFill)};
    std::vector<column_destination> dst{str.slot(cudf::type_id::STRING)};
    const auto result = read(source, columns, kRowBegin, std::size_t{1} << 40, dst);
    REQUIRE(result.rows_written == kFit);
    REQUIRE_FALSE(result.source_exhausted);
    REQUIRE(result.data_bytes_written ==
            std::vector<std::size_t>{chars_bytes(data, kRowBegin, kRowBegin + kFit, false)});
    expect_strings(str, 4, 0, 0, data, kRowBegin, kFit, false);
  }

  SECTION("a chars slot that binds inside the first chunk")
  {
    const std::size_t capacity = chars_bytes(data, 0, 10, false);
    dest_buffers str{filled_buffer(capacity, kFill), {}, filled_buffer((kRows + 1) * 4, kFill)};
    std::vector<column_destination> dst{str.slot(cudf::type_id::STRING)};
    const auto result = read(source, columns, 0, kRows, dst);
    // Row 10 is "kkk10": it does not fit after rows [0, 10).
    REQUIRE(result.rows_written == 10);
    expect_strings(str, 4, 0, 0, data, 0, 10, false);
  }

  SECTION("everything fits: all rows across several chunks")
  {
    const std::size_t capacity = chars_bytes(data, 0, kRows, false);
    dest_buffers str{filled_buffer(capacity, kFill), {}, filled_buffer((kRows + 1) * 4, kFill)};
    std::vector<column_destination> dst{str.slot(cudf::type_id::STRING)};
    const auto result = read(source, columns, 0, kRows, dst);
    REQUIRE(result.rows_written == kRows);
    REQUIRE(result.source_exhausted);
    expect_strings(str, 4, 0, 0, data, 0, kRows, false);
  }

  SECTION("two STRING requests with different chars capacities take the smaller fit")
  {
    const std::size_t small = chars_bytes(data, 0, 3000, false);
    const std::size_t large = chars_bytes(data, 0, 9000, false);
    dest_buffers a{filled_buffer(large, kFill), {}, filled_buffer((kRows + 1) * 4, kFill)};
    dest_buffers b{filled_buffer(small, kFill), {}, filled_buffer((kRows + 1) * 4, kFill)};
    const std::vector<std::size_t> both{kStringCol, kStringCol};
    std::vector<column_destination> dst{a.slot(cudf::type_id::STRING),
                                        b.slot(cudf::type_id::STRING)};
    const auto result = read(source, both, 0, kRows, dst);
    // Row 3000 is not empty, so exactly 3000 rows fit the smaller slot.
    REQUIRE(!data.str[3000].empty());
    REQUIRE(result.rows_written == 3000);
    expect_strings(a, 4, 0, 0, data, 0, 3000, false);
    expect_strings(b, 4, 0, 0, data, 0, 3000, false);
  }
}

TEST_CASE("read_rows_gpu refuses a source on another device than the stream",
          "[row_range_reader_gpu]")
{
  // The reader compares the id of the source's memory space with the device of the stream. A
  // DISK space with id 1 reports another id without touching a CUDA device, so one GPU is enough;
  // the table's memory stays on device 0. A null writer stream leaves the writer unknown, so the
  // constructor accepts the space.
  auto other_space =
    make_mock_memory_space(memory::Tier::DISK, 1, std::make_shared<mock_no_io_backend>());
  const auto data = make_source_data(16);
  gpu_table_representation source(
    make_table(data, false), *other_space, ::cuda::stream_ref{cudaStream_t{nullptr}});
  REQUIRE(source.get_device_id() == 1);

  dest_buffers i32{filled_buffer(64, kFill), {}, {}};
  const std::vector<std::size_t> columns{kInt32Col};
  std::vector<column_destination> dst{i32.slot(cudf::type_id::INT32)};
  REQUIRE_THROWS_AS(read(source, columns, 0, 16, dst), row_range_unsupported_error);
  REQUIRE(to_host<std::uint32_t>(i32.data)[0] == 0x01010101U * kFill);
}
