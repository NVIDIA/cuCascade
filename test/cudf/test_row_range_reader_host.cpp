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
#include "utils/cudf_test_utils.hpp"
#include "utils/mixed_table.hpp"

#include <cudf_test/column_wrapper.hpp>

#include <cucascade/cudf/builtin_converters.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/cudf/host_table.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/data/row_range_read.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cucascade;

namespace {

/// Small pinned blocks, so every multi-row range crosses block boundaries.
constexpr std::size_t kBlockSize = 256;
constexpr std::size_t kRows      = 1000;
/// Byte the destination buffers hold before a read; untouched bytes must still hold it.
constexpr std::uint8_t kFill = 0xA5;

// Column indices of the source table.
constexpr std::size_t kInt8Nulls      = 0;  // INT8, nullable with nulls
constexpr std::size_t kInt32          = 1;  // INT32, no mask
constexpr std::size_t kInt64Nulls     = 2;  // INT64, nullable with nulls
constexpr std::size_t kFloat64        = 3;  // FLOAT64, no mask
constexpr std::size_t kTimestamp      = 4;  // TIMESTAMP_MILLISECONDS, no mask
constexpr std::size_t kDecimal64      = 5;  // DECIMAL64 (scale -2), no mask
constexpr std::size_t kString32Nulls  = 6;  // STRING with INT32 offsets, nullable with nulls
constexpr std::size_t kString64       = 7;  // STRING with INT64 offsets, no mask
constexpr std::size_t kInt32AllValid  = 8;  // INT32 with a mask but no nulls
constexpr std::size_t kNumSourceCols  = 9;
constexpr std::size_t kMaxStringBytes = 40;

::cuda::stream_ref shared_stream()
{
  static rmm::cuda_stream s;
  return s;
}

auto& shared_gpu_space()
{
  static auto s = [] {
    memory::gpu_memory_space_config config;
    config.device_id       = 0;
    config.memory_capacity = 1024 * 1024 * 1024;
    config.mr_factory_fn   = [](int, std::size_t) {
      return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
        rmm::mr::cuda_memory_resource{}};
    };
    return std::make_shared<memory::memory_space>(config);
  }();
  return s;
}

auto& shared_host_space()
{
  static auto s = [] {
    memory::host_memory_space_config config;
    config.numa_id              = 0;
    config.memory_capacity      = 64 * 1024 * 1024;
    config.block_size           = kBlockSize;
    config.pool_size            = 4096;
    config.initial_number_pools = 0;
    config.mr_factory_fn        = [](int, std::size_t) {
      return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
        memory::numa_region_pinned_host_memory_resource{-1}};
    };
    return std::make_shared<memory::memory_space>(config);
  }();
  return s;
}

/// Host copy of one source column: what a read must deliver.
struct host_column {
  cudf::data_type type{cudf::type_id::EMPTY};
  /// Bytes per row; 0 for STRING.
  std::size_t width{0};
  /// Fixed-width values, `width` bytes per row.
  std::vector<std::uint8_t> bytes;
  /// STRING values (empty for a null row).
  std::vector<std::string> strings;
  /// Validity per row; empty when the column has no null mask.
  std::vector<bool> valid;

  [[nodiscard]] bool is_string() const { return type.id() == cudf::type_id::STRING; }
  [[nodiscard]] bool is_valid(std::size_t row) const { return valid.empty() || valid[row]; }
  [[nodiscard]] cudf::size_type null_count() const
  {
    cudf::size_type nulls = 0;
    for (const bool v : valid) {
      nulls += v ? 0 : 1;
    }
    return nulls;
  }
  /// Chars bytes of rows [begin, begin + rows).
  [[nodiscard]] std::size_t chars_bytes(std::size_t begin, std::size_t rows) const
  {
    std::size_t total = 0;
    for (std::size_t r = begin; r < begin + rows; ++r) {
      total += strings[r].size();
    }
    return total;
  }
};

template <typename T>
std::vector<std::uint8_t> to_bytes(const std::vector<T>& values)
{
  std::vector<std::uint8_t> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

struct source_fixture {
  std::vector<host_column> columns;
  std::unique_ptr<gpu_table_representation> gpu;
  std::unique_ptr<host_data_representation> host;
};

/// Builds the nine-column source once: host oracle, the GPU table, and its host representation.
const source_fixture& source()
{
  static const source_fixture fixture = [] {
    using cudf::test::fixed_point_column_wrapper;
    using cudf::test::fixed_width_column_wrapper;
    using cudf::test::strings_column_wrapper;

    std::mt19937 rng{20261112};
    auto stream = shared_stream();
    auto mr     = rmm::mr::get_current_device_resource_ref();

    auto random_validity = [&](std::size_t rows) {
      std::vector<bool> valid(rows);
      for (std::size_t r = 0; r < rows; ++r) {
        valid[r] = rng() % 4 != 0;
      }
      return valid;
    };
    auto random_values = [&]<typename T>(T /*tag*/) {
      std::vector<T> values(kRows);
      for (auto& v : values) {
        v = static_cast<T>(static_cast<std::int64_t>(rng()) - (std::int64_t{1} << 31));
      }
      return values;
    };
    auto random_strings = [&](const std::vector<bool>& valid) {
      std::vector<std::string> strings(kRows);
      for (std::size_t r = 0; r < kRows; ++r) {
        if (!valid.empty() && !valid[r]) { continue; }
        strings[r].resize(rng() % (kMaxStringBytes + 1));
        for (auto& c : strings[r]) {
          c = static_cast<char>('a' + rng() % 26);
        }
      }
      return strings;
    };

    source_fixture f;
    f.columns.resize(kNumSourceCols);
    std::vector<std::unique_ptr<cudf::column>> cudf_columns;

    {
      auto values = random_values(std::int8_t{});
      auto valid  = random_validity(kRows);
      cudf_columns.push_back(
        fixed_width_column_wrapper<std::int8_t>(values.begin(), values.end(), valid.begin())
          .release());
      f.columns[kInt8Nulls] = {
        cudf::data_type{cudf::type_id::INT8}, 1, to_bytes(values), {}, valid};
    }
    {
      auto values = random_values(std::int32_t{});
      cudf_columns.push_back(
        fixed_width_column_wrapper<std::int32_t>(values.begin(), values.end()).release());
      f.columns[kInt32] = {cudf::data_type{cudf::type_id::INT32}, 4, to_bytes(values), {}, {}};
    }
    {
      auto values = random_values(std::int64_t{});
      auto valid  = random_validity(kRows);
      cudf_columns.push_back(
        fixed_width_column_wrapper<std::int64_t>(values.begin(), values.end(), valid.begin())
          .release());
      f.columns[kInt64Nulls] = {
        cudf::data_type{cudf::type_id::INT64}, 8, to_bytes(values), {}, valid};
    }
    {
      std::vector<double> values(kRows);
      for (auto& v : values) {
        v = static_cast<double>(rng()) / 7.0;
      }
      cudf_columns.push_back(
        fixed_width_column_wrapper<double>(values.begin(), values.end()).release());
      f.columns[kFloat64] = {cudf::data_type{cudf::type_id::FLOAT64}, 8, to_bytes(values), {}, {}};
    }
    {
      auto values = random_values(std::int64_t{});
      cudf_columns.push_back(
        fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>(values.begin(),
                                                                                values.end())
          .release());
      f.columns[kTimestamp] = {
        cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS}, 8, to_bytes(values), {}, {}};
    }
    {
      auto values = random_values(std::int64_t{});
      cudf_columns.push_back(fixed_point_column_wrapper<std::int64_t>(
                               values.begin(), values.end(), numeric::scale_type{-2})
                               .release());
      f.columns[kDecimal64] = {
        cudf::data_type{cudf::type_id::DECIMAL64, -2}, 8, to_bytes(values), {}, {}};
    }
    {
      auto valid   = random_validity(kRows);
      auto strings = random_strings(valid);
      cudf_columns.push_back(
        strings_column_wrapper(strings.begin(), strings.end(), valid.begin()).release());
      f.columns[kString32Nulls] = {cudf::data_type{cudf::type_id::STRING}, 0, {}, strings, valid};
    }
    {
      auto strings = random_strings({});
      cudf_columns.push_back(test::widen_string_offsets(
        strings_column_wrapper(strings.begin(), strings.end()).release(), stream, mr));
      f.columns[kString64] = {cudf::data_type{cudf::type_id::STRING}, 0, {}, strings, {}};
    }
    {
      auto values = random_values(std::int32_t{});
      std::vector<bool> valid(kRows, true);
      cudf_columns.push_back(
        fixed_width_column_wrapper<std::int32_t>(values.begin(), values.end(), valid.begin())
          .release());
      f.columns[kInt32AllValid] = {
        cudf::data_type{cudf::type_id::INT32}, 4, to_bytes(values), {}, valid};
    }

    // The wrappers write on cudf's default stream; finish that work before `stream` reads it.
    CUCASCADE_CUDA_TRY(cudaDeviceSynchronize());

    f.gpu = std::make_unique<gpu_table_representation>(
      std::make_unique<cudf::table>(std::move(cudf_columns)), *shared_gpu_space(), stream);

    representation_converter_registry registry;
    register_builtin_converters(registry);
    f.host = registry.convert<host_data_representation>(*f.gpu, shared_host_space().get(), stream);
    stream.sync();
    return f;
  }();
  return fixture;
}

/// Device buffers of one destination column, plus the column_destination describing them.
struct dst_column {
  rmm::device_buffer data;
  rmm::device_buffer mask;
  rmm::device_buffer offsets;
  column_destination dest;
};

rmm::device_buffer filled_buffer(std::size_t bytes)
{
  rmm::device_buffer buffer(bytes, shared_stream());
  if (bytes > 0) {
    CUCASCADE_CUDA_TRY(cudaMemsetAsync(buffer.data(), kFill, bytes, shared_stream().get()));
  }
  return buffer;
}

/// Capacities of one destination column. A zero capacity leaves that slot absent.
struct dst_spec {
  std::size_t data_bytes{0};
  std::size_t mask_bytes{0};
  std::size_t offsets_bytes{0};
  std::uint8_t offset_width{4};
  std::size_t dst_row_offset{0};
  std::size_t dst_byte_offset{0};
};

dst_column make_dst(const host_column& src, const dst_spec& spec)
{
  dst_column d;
  d.data                 = filled_buffer(spec.data_bytes);
  d.mask                 = filled_buffer(spec.mask_bytes);
  d.offsets              = filled_buffer(spec.offsets_bytes);
  d.dest.type_id         = static_cast<std::int32_t>(src.type.id());
  d.dest.data            = {d.data.data(), d.data.size()};
  d.dest.null_mask       = {d.mask.data(), d.mask.size()};
  d.dest.offsets         = {d.offsets.data(), d.offsets.size()};
  d.dest.offset_width    = spec.offset_width;
  d.dest.dst_row_offset  = spec.dst_row_offset;
  d.dest.dst_byte_offset = spec.dst_byte_offset;
  return d;
}

/// A destination with room for @p rows rows appended at the spec's append position.
dst_column make_roomy_dst(const host_column& src,
                          std::size_t rows,
                          bool with_mask,
                          std::uint8_t offset_width   = 4,
                          std::size_t dst_row_offset  = 0,
                          std::size_t dst_byte_offset = 0)
{
  dst_spec spec;
  spec.offset_width    = offset_width;
  spec.dst_row_offset  = dst_row_offset;
  spec.dst_byte_offset = dst_byte_offset;
  const auto dst_rows  = dst_row_offset + rows;
  if (src.is_string()) {
    spec.data_bytes    = dst_byte_offset + rows * kMaxStringBytes + 16;
    spec.offsets_bytes = (dst_rows + 1) * offset_width + 16;
  } else {
    spec.data_bytes = dst_rows * src.width + 16;
  }
  if (with_mask) { spec.mask_bytes = (dst_rows + 31) / 32 * 4 + 8; }
  return make_dst(src, spec);
}

std::vector<std::uint8_t> to_host(const rmm::device_buffer& buffer)
{
  std::vector<std::uint8_t> host(buffer.size());
  if (!host.empty()) {
    CUCASCADE_CUDA_TRY(cudaMemcpy(host.data(), buffer.data(), host.size(), cudaMemcpyDeviceToHost));
  }
  return host;
}

bool all_fill(const std::vector<std::uint8_t>& bytes, std::size_t begin, std::size_t end)
{
  for (std::size_t i = begin; i < end; ++i) {
    if (bytes[i] != kFill) { return false; }
  }
  return true;
}

/**
 * @brief Check that @p d holds source rows [row_begin, row_begin + k) appended at
 * (@p dst_row_offset, @p dst_byte_offset), and that every byte and bit outside that range still
 * holds the fill pattern.
 */
void verify_read(const host_column& src,
                 const dst_column& d,
                 std::size_t row_begin,
                 std::size_t k,
                 std::size_t dst_row_offset,
                 std::size_t dst_byte_offset)
{
  shared_stream().sync();
  const auto data    = to_host(d.data);
  const auto mask    = to_host(d.mask);
  const auto offsets = to_host(d.offsets);

  if (src.is_string()) {
    const std::size_t width = d.dest.offset_width;
    auto offset_at          = [&](std::size_t i) {
      if (width == 4) {
        std::int32_t v{};
        std::memcpy(&v, offsets.data() + i * 4, 4);
        return static_cast<std::int64_t>(v);
      }
      std::int64_t v{};
      std::memcpy(&v, offsets.data() + i * 8, 8);
      return v;
    };
    std::size_t byte = dst_byte_offset;
    if (k > 0) {
      for (std::size_t i = 0; i <= k; ++i) {
        REQUIRE(offset_at(dst_row_offset + i) == static_cast<std::int64_t>(byte));
        if (i == k) { break; }
        const auto& expected = src.strings[row_begin + i];
        REQUIRE(byte + expected.size() <= data.size());
        REQUIRE(std::memcmp(data.data() + byte, expected.data(), expected.size()) == 0);
        byte += expected.size();
      }
      REQUIRE(all_fill(offsets, 0, dst_row_offset * width));
      REQUIRE(all_fill(offsets, (dst_row_offset + k + 1) * width, offsets.size()));
    } else {
      REQUIRE(all_fill(offsets, 0, offsets.size()));
    }
    REQUIRE(all_fill(data, 0, dst_byte_offset));
    REQUIRE(all_fill(data, byte, data.size()));
  } else {
    const std::size_t w = src.width;
    for (std::size_t i = 0; i < k; ++i) {
      if (!src.is_valid(row_begin + i)) { continue; }  // the value under a null is unspecified
      REQUIRE(std::memcmp(data.data() + (dst_row_offset + i) * w,
                          src.bytes.data() + (row_begin + i) * w,
                          w) == 0);
    }
    REQUIRE(all_fill(data, 0, dst_row_offset * w));
    REQUIRE(all_fill(data, (dst_row_offset + k) * w, data.size()));
  }

  for (std::size_t bit = 0; bit < mask.size() * 8; ++bit) {
    const bool actual   = (mask[bit / 8] >> (bit % 8)) & 1U;
    const bool expected = bit >= dst_row_offset && bit < dst_row_offset + k
                            ? src.is_valid(row_begin + (bit - dst_row_offset))
                            : ((kFill >> (bit % 8)) & 1U) != 0;
    if (actual != expected) {
      INFO("mask bit " << bit << " dst_row_offset " << dst_row_offset << " k " << k);
      REQUIRE(actual == expected);
    }
  }
}

row_range_result read(std::span<const std::size_t> columns,
                      std::size_t row_begin,
                      std::size_t max_rows,
                      std::span<column_destination> dst)
{
  row_range_request request;
  request.columns   = columns;
  request.row_begin = row_begin;
  request.max_rows  = max_rows;
  return detail::read_rows_host(*source().host, request, dst, shared_stream(), nullptr, nullptr);
}

/// Read one column into one destination.
row_range_result read_one(std::size_t column,
                          std::size_t row_begin,
                          std::size_t max_rows,
                          dst_column& d)
{
  const std::size_t columns[] = {column};
  return read(columns, row_begin, max_rows, std::span<column_destination>{&d.dest, 1});
}

}  // namespace

TEST_CASE("host row-range reader: source layout", "[row_range_reader_host]")
{
  const auto& table = source().host->get_host_table();
  REQUIRE(table->allocation->block_size() == kBlockSize);
  REQUIRE(table->columns.size() == kNumSourceCols);
  REQUIRE(table->columns[kInt8Nulls].has_null_mask);
  REQUIRE(table->columns[kInt8Nulls].null_count > 0);
  REQUIRE(table->columns[kString32Nulls].null_count > 0);
  REQUIRE(table->columns[kString32Nulls].children.at(0).type_id ==
          static_cast<std::int32_t>(cudf::type_id::INT32));
  REQUIRE(table->columns[kString64].children.at(0).type_id ==
          static_cast<std::int32_t>(cudf::type_id::INT64));
}

TEST_CASE("host row-range reader: windows concatenate to the source", "[row_range_reader_host]")
{
  const auto& src = source();
  std::vector<std::size_t> columns(kNumSourceCols);
  std::vector<dst_column> dsts;
  for (std::size_t c = 0; c < kNumSourceCols; ++c) {
    columns[c] = c;
    // Every column gets a mask slot, including the ones whose source has no mask.
    dsts.push_back(make_roomy_dst(src.columns[c], kRows, true, c == kString64 ? 8 : 4));
  }
  std::vector<column_destination> dests(kNumSourceCols);

  const std::size_t window = GENERATE(std::size_t{1000}, std::size_t{137}, std::size_t{31});
  std::size_t row_begin    = 0;
  std::size_t calls        = 0;
  while (true) {
    for (std::size_t c = 0; c < kNumSourceCols; ++c) {
      dests[c] = dsts[c].dest;
    }
    const auto result = read(columns, row_begin, window, dests);
    ++calls;
    REQUIRE(result.rows_written == std::min(window, kRows - row_begin));
    REQUIRE(result.data_bytes_written.size() == kNumSourceCols);
    for (std::size_t c = 0; c < kNumSourceCols; ++c) {
      const auto& col = src.columns[c];
      REQUIRE(result.data_bytes_written[c] == (col.is_string()
                                                 ? col.chars_bytes(row_begin, result.rows_written)
                                                 : result.rows_written * col.width));
      dsts[c].dest.dst_row_offset += result.rows_written;
      dsts[c].dest.dst_byte_offset += result.data_bytes_written[c];
    }
    row_begin += result.rows_written;
    REQUIRE(result.source_exhausted == (row_begin == kRows));
    if (result.source_exhausted) { break; }
  }
  REQUIRE(calls == (kRows + window - 1) / window);

  for (std::size_t c = 0; c < kNumSourceCols; ++c) {
    verify_read(src.columns[c], dsts[c], 0, kRows, 0, 0);
  }

  // The destination buffers, viewed as cudf columns, equal the original GPU table.
  std::vector<cudf::column_view> views;
  for (std::size_t c = 0; c < kNumSourceCols; ++c) {
    const auto& col = src.columns[c];
    const auto* mask =
      col.valid.empty() ? nullptr : static_cast<const cudf::bitmask_type*>(dsts[c].mask.data());
    std::vector<cudf::column_view> children;
    if (col.is_string()) {
      children.emplace_back(cudf::data_type{dsts[c].dest.offset_width == 8 ? cudf::type_id::INT64
                                                                           : cudf::type_id::INT32},
                            static_cast<cudf::size_type>(kRows + 1),
                            dsts[c].offsets.data(),
                            nullptr,
                            0);
    }
    views.emplace_back(col.type,
                       static_cast<cudf::size_type>(kRows),
                       dsts[c].data.data(),
                       mask,
                       col.null_count(),
                       0,
                       children);
  }
  test::expect_cudf_tables_equal_on_stream(
    src.gpu->get_table_view(), cudf::table_view{views}, shared_stream());
}

TEST_CASE("host row-range reader: ranges cross pinned-block boundaries", "[row_range_reader_host]")
{
  const auto& src = source();
  // 300 rows from row 17: more than one block of values, of chars and of offsets.
  const std::size_t row_begin = 17;
  const std::size_t rows      = 300;
  const std::size_t column =
    GENERATE(kInt8Nulls, kInt64Nulls, kDecimal64, kString32Nulls, kString64);
  const auto& col = src.columns[column];
  if (col.is_string()) {
    REQUIRE(col.chars_bytes(row_begin, rows) > 2 * kBlockSize);
  } else {
    REQUIRE(rows * col.width > kBlockSize);
  }

  auto d            = make_roomy_dst(col, rows, true);
  const auto result = read_one(column, row_begin, rows, d);
  REQUIRE(result.rows_written == rows);
  REQUIRE_FALSE(result.source_exhausted);
  verify_read(col, d, row_begin, rows, 0, 0);
}

TEST_CASE("host row-range reader: budget truncation", "[row_range_reader_host]")
{
  const auto& src = source();

  SECTION("STRING chars budget delivers a partial window")
  {
    const std::size_t column          = GENERATE(kString32Nulls, kString64);
    const auto& col                   = src.columns[column];
    const std::size_t row_begin       = 10;
    const std::size_t dst_byte_offset = 7;
    // One byte short of 21 rows; the 21st row must not be empty for that to cut the range.
    std::size_t fit = 20;
    while (col.strings[row_begin + fit].empty()) {
      ++fit;
    }
    dst_spec spec;
    spec.dst_byte_offset = dst_byte_offset;
    spec.data_bytes      = dst_byte_offset + col.chars_bytes(row_begin, fit + 1) - 1;
    spec.offsets_bytes   = 4 * 200;
    spec.mask_bytes      = 64;
    auto d               = make_dst(col, spec);

    const auto result = read_one(column, row_begin, 100, d);
    REQUIRE(result.rows_written == fit);
    REQUIRE_FALSE(result.source_exhausted);
    REQUIRE(result.data_bytes_written.at(0) == col.chars_bytes(row_begin, fit));
    verify_read(col, d, row_begin, fit, 0, dst_byte_offset);
  }

  SECTION("a single STRING row too large delivers nothing and is not exhausted")
  {
    const auto& col       = src.columns[kString64];
    std::size_t row_begin = 500;
    while (col.strings[row_begin].size() < 2) {
      ++row_begin;
    }
    dst_spec spec;
    spec.data_bytes    = col.strings[row_begin].size() - 1;
    spec.offsets_bytes = 4 * 200;
    auto d             = make_dst(col, spec);

    const auto result = read_one(kString64, row_begin, 100, d);
    REQUIRE(result.rows_written == 0);
    REQUIRE_FALSE(result.source_exhausted);
    REQUIRE(result.data_bytes_written.at(0) == 0);
    verify_read(col, d, row_begin, 0, 0, 0);
  }

  SECTION("STRING offsets budget")
  {
    const auto& col = src.columns[kString32Nulls];
    dst_spec spec;
    spec.dst_row_offset = 3;
    spec.data_bytes     = 100 * kMaxStringBytes;
    spec.offsets_bytes  = (3 + 5 + 1) * 4 + 3;  // 5 rows, and a partial sixth offset
    spec.mask_bytes     = 64;
    auto d              = make_dst(col, spec);

    const auto result = read_one(kString32Nulls, 40, 100, d);
    REQUIRE(result.rows_written == 5);
    verify_read(col, d, 40, 5, 3, 0);
  }

  SECTION("fixed-width data and null-mask budgets")
  {
    const auto& col = src.columns[kInt64Nulls];
    dst_spec spec;
    spec.dst_row_offset = 2;
    spec.data_bytes     = 12 * 8 + 7;  // 10 more rows, and a partial eleventh
    spec.mask_bytes     = 64;
    auto d              = make_dst(col, spec);
    auto result         = read_one(kInt64Nulls, 990, 100, d);
    REQUIRE(result.rows_written == 10);
    REQUIRE(result.source_exhausted);
    verify_read(col, d, 990, 10, 2, 0);

    spec.data_bytes = 100 * 8;
    spec.mask_bytes = 4;  // one word: 30 more bits
    auto d2         = make_dst(col, spec);
    result          = read_one(kInt64Nulls, 0, 100, d2);
    REQUIRE(result.rows_written == 30);
    verify_read(col, d2, 0, 30, 2, 0);
  }
}

TEST_CASE("host row-range reader: rebased offsets", "[row_range_reader_host]")
{
  const auto& src                  = source();
  const std::size_t column         = GENERATE(kString32Nulls, kString64);  // 4- and 8-byte source
  const std::uint8_t offset_width  = GENERATE(std::uint8_t{4}, std::uint8_t{8});
  const auto& col                  = src.columns[column];
  const std::size_t row_begin      = 41;
  const std::size_t rows           = 200;
  const std::size_t dst_row_offset = 9;
  const std::size_t dst_byte_off   = 123;

  auto d            = make_roomy_dst(col, rows, true, offset_width, dst_row_offset, dst_byte_off);
  const auto result = read_one(column, row_begin, rows, d);
  REQUIRE(result.rows_written == rows);
  REQUIRE(result.data_bytes_written.at(0) == col.chars_bytes(row_begin, rows));
  verify_read(col, d, row_begin, rows, dst_row_offset, dst_byte_off);
}

TEST_CASE("host row-range reader: null bits at unaligned positions", "[row_range_reader_host]")
{
  const auto& src                  = source();
  const std::size_t dst_row_offset = GENERATE(std::size_t{0}, std::size_t{5}, std::size_t{33});
  const std::size_t row_begin      = GENERATE(std::size_t{0}, std::size_t{3}, std::size_t{64});
  const std::size_t rows           = GENERATE(std::size_t{1}, std::size_t{70});
  // Sources with nulls, with an all-valid mask, and without a mask.
  const std::size_t column = GENERATE(kInt8Nulls, kString32Nulls, kInt32AllValid, kInt32);
  const auto& col          = src.columns[column];

  auto d            = make_roomy_dst(col, rows, true, 4, dst_row_offset, 0);
  const auto result = read_one(column, row_begin, rows, d);
  REQUIRE(result.rows_written == rows);
  verify_read(col, d, row_begin, rows, dst_row_offset, 0);
}

TEST_CASE("host row-range reader: missing null-mask slot", "[row_range_reader_host]")
{
  const auto& src = source();

  SECTION("source with nulls is refused")
  {
    const std::size_t column = GENERATE(kInt64Nulls, kString32Nulls);
    auto d                   = make_roomy_dst(src.columns[column], 50, false);
    REQUIRE_THROWS_AS(read_one(column, 0, 50, d), row_range_unsupported_error);
    verify_read(src.columns[column], d, 0, 0, 0, 0);
  }

  SECTION("all-valid source is read")
  {
    const std::size_t column = GENERATE(kInt32AllValid, kInt32, kString64);
    auto d                   = make_roomy_dst(src.columns[column], 50, false);
    const auto result        = read_one(column, 20, 50, d);
    REQUIRE(result.rows_written == 50);
    verify_read(src.columns[column], d, 20, 50, 0, 0);
  }
}

TEST_CASE("host row-range reader: exhausted source and bad requests", "[row_range_reader_host]")
{
  const auto& src             = source();
  const std::size_t columns[] = {kInt32, kString32Nulls};
  std::vector<dst_column> dsts;
  dsts.push_back(make_roomy_dst(src.columns[kInt32], 10, true));
  dsts.push_back(make_roomy_dst(src.columns[kString32Nulls], 10, true));
  std::vector<column_destination> dests{dsts[0].dest, dsts[1].dest};

  SECTION("row_begin == num_rows reads nothing and reports exhaustion")
  {
    const auto result = read(columns, kRows, 10, dests);
    REQUIRE(result.rows_written == 0);
    REQUIRE(result.source_exhausted);
    REQUIRE(result.data_bytes_written == std::vector<std::size_t>{0, 0});
    verify_read(src.columns[kInt32], dsts[0], kRows, 0, 0, 0);
    verify_read(src.columns[kString32Nulls], dsts[1], kRows, 0, 0, 0);
  }

  SECTION("max_rows == 0 reads nothing and is not exhausted")
  {
    const auto result = read(columns, 5, 0, dests);
    REQUIRE(result.rows_written == 0);
    REQUIRE_FALSE(result.source_exhausted);
    verify_read(src.columns[kInt32], dsts[0], 5, 0, 0, 0);
  }

  SECTION("row_begin past the end")
  {
    REQUIRE_THROWS_AS(read(columns, kRows + 1, 10, dests), std::out_of_range);
  }

  SECTION("column index past the end")
  {
    const std::size_t bad[] = {kInt32, kNumSourceCols};
    REQUIRE_THROWS_AS(read(bad, 0, 10, dests), std::out_of_range);
  }

  SECTION("destination type mismatch")
  {
    dests[0].type_id = static_cast<std::int32_t>(cudf::type_id::INT64);
    REQUIRE_THROWS_AS(read(columns, 0, 10, dests), std::invalid_argument);
  }

  SECTION("source of another representation type")
  {
    row_range_request request;
    request.columns  = columns;
    request.max_rows = 10;
    REQUIRE_THROWS_AS(
      detail::read_rows_host(*src.gpu, request, dests, shared_stream(), nullptr, nullptr),
      std::invalid_argument);
  }
}

TEST_CASE("host row-range reader: the tightest column sets the window", "[row_range_reader_host]")
{
  const auto& src             = source();
  const std::size_t columns[] = {kInt64Nulls, kString32Nulls};
  const std::size_t row_begin = 123;

  SECTION("fixed-width data budget limits the STRING column too")
  {
    dst_spec spec;
    spec.data_bytes = 40 * 8;
    spec.mask_bytes = 64;
    auto fixed      = make_dst(src.columns[kInt64Nulls], spec);
    auto str        = make_roomy_dst(src.columns[kString32Nulls], 200, true);
    std::vector<column_destination> dests{fixed.dest, str.dest};

    const auto result = read(columns, row_begin, 200, dests);
    REQUIRE(result.rows_written == 40);
    REQUIRE(
      result.data_bytes_written ==
      std::vector<std::size_t>{40 * 8, src.columns[kString32Nulls].chars_bytes(row_begin, 40)});
    verify_read(src.columns[kInt64Nulls], fixed, row_begin, 40, 0, 0);
    verify_read(src.columns[kString32Nulls], str, row_begin, 40, 0, 0);
  }

  SECTION("STRING chars budget limits the fixed-width column too")
  {
    const auto& col = src.columns[kString32Nulls];
    // Rows that fit in 300 chars bytes.
    std::size_t fit = 0;
    while (col.chars_bytes(row_begin, fit + 1) <= 300) {
      ++fit;
    }
    REQUIRE(fit > 0);
    REQUIRE(fit < 200);
    auto fixed = make_roomy_dst(src.columns[kInt64Nulls], 200, true);
    dst_spec spec;
    spec.data_bytes    = 300;
    spec.offsets_bytes = 4 * 300;
    spec.mask_bytes    = 64;
    auto str           = make_dst(col, spec);
    std::vector<column_destination> dests{fixed.dest, str.dest};

    const auto result = read(columns, row_begin, 200, dests);
    REQUIRE(result.rows_written == fit);
    REQUIRE_FALSE(result.source_exhausted);
    verify_read(src.columns[kInt64Nulls], fixed, row_begin, fit, 0, 0);
    verify_read(col, str, row_begin, fit, 0, 0);
  }
}

TEST_CASE("host row-range reader: STRING range with no chars", "[row_range_reader_host]")
{
  const auto& src = source();
  const auto& col = src.columns[kString32Nulls];
  // A null row: its string is empty, so the range carries no chars bytes.
  std::size_t row_begin = 0;
  while (!col.strings[row_begin].empty()) {
    ++row_begin;
  }

  // The chars buffer is already full at dst_byte_offset; an empty row still fits.
  dst_spec spec;
  spec.dst_row_offset  = 4;
  spec.dst_byte_offset = 5;
  spec.data_bytes      = 5;
  spec.offsets_bytes   = (4 + 1 + 1) * 4;
  spec.mask_bytes      = 4;
  auto d               = make_dst(col, spec);

  const auto result = read_one(kString32Nulls, row_begin, 1, d);
  REQUIRE(result.rows_written == 1);
  REQUIRE(result.data_bytes_written.at(0) == 0);
  verify_read(col, d, row_begin, 1, 4, 5);
}

TEST_CASE("host row-range reader: 4-byte destination offsets overflow", "[row_range_reader_host]")
{
  const auto& src = source();
  const auto& col = src.columns[kString64];
  // Rebased offsets start just below INT32_MAX; the range's chars push them past it.
  const std::size_t dst_byte_offset =
    static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) - 10;
  const std::size_t rows = 50;
  REQUIRE(col.chars_bytes(0, rows) > 10);

  rmm::device_buffer data(dst_byte_offset + rows * kMaxStringBytes, shared_stream());
  const std::uint8_t offset_width = GENERATE(std::uint8_t{4}, std::uint8_t{8});
  auto offsets                    = filled_buffer((rows + 1) * offset_width);

  column_destination dest;
  dest.type_id                = static_cast<std::int32_t>(cudf::type_id::STRING);
  dest.data                   = {data.data(), data.size()};
  dest.offsets                = {offsets.data(), offsets.size()};
  dest.offset_width           = offset_width;
  dest.dst_byte_offset        = dst_byte_offset;
  const std::size_t columns[] = {kString64};

  if (offset_width == 4) {
    REQUIRE_THROWS_AS(read(columns, 0, rows, std::span<column_destination>{&dest, 1}),
                      std::overflow_error);
    shared_stream().sync();
    const auto host_offsets = to_host(offsets);
    REQUIRE(all_fill(host_offsets, 0, host_offsets.size()));
  } else {
    const auto result = read(columns, 0, rows, std::span<column_destination>{&dest, 1});
    REQUIRE(result.rows_written == rows);
    shared_stream().sync();
    const auto host_offsets = to_host(offsets);
    std::vector<std::int64_t> values(rows + 1);
    std::memcpy(values.data(), host_offsets.data(), values.size() * sizeof(std::int64_t));
    std::size_t byte = dst_byte_offset;
    for (std::size_t i = 0; i <= rows; ++i) {
      REQUIRE(values[i] == static_cast<std::int64_t>(byte));
      if (i < rows) { byte += col.strings[i].size(); }
    }
    std::string chars(col.chars_bytes(0, rows), '\0');
    CUCASCADE_CUDA_TRY(cudaMemcpy(chars.data(),
                                  static_cast<const std::uint8_t*>(data.data()) + dst_byte_offset,
                                  chars.size(),
                                  cudaMemcpyDeviceToHost));
    std::string expected;
    for (std::size_t i = 0; i < rows; ++i) {
      expected += col.strings[i];
    }
    REQUIRE(chars == expected);
  }
}

TEST_CASE("host row-range reader: temporaries from a scratch reservation",
          "[row_range_reader_host]")
{
  const auto& src             = source();
  auto reservation            = shared_gpu_space()->make_reservation(std::size_t{1} << 20);
  const std::size_t columns[] = {kInt8Nulls, kString32Nulls, kString64};
  std::vector<dst_column> dsts;
  dsts.push_back(make_roomy_dst(src.columns[kInt8Nulls], 333, true, 4, 7));
  dsts.push_back(make_roomy_dst(src.columns[kString32Nulls], 333, true, 8, 7, 3));
  dsts.push_back(make_roomy_dst(src.columns[kString64], 333, true, 4, 7, 3));
  std::vector<column_destination> dests{dsts[0].dest, dsts[1].dest, dsts[2].dest};

  row_range_request request;
  request.columns   = columns;
  request.row_begin = 600;
  request.max_rows  = 333;
  const auto result =
    detail::read_rows_host(*src.host, request, dests, shared_stream(), reservation.get(), nullptr);
  REQUIRE(result.rows_written == 333);
  verify_read(src.columns[kInt8Nulls], dsts[0], 600, 333, 7, 0);
  verify_read(src.columns[kString32Nulls], dsts[1], 600, 333, 7, 3);
  verify_read(src.columns[kString64], dsts[2], 600, 333, 7, 3);
}

TEST_CASE("host row-range reader: empty source", "[row_range_reader_host]")
{
  auto stream = shared_stream();
  std::vector<std::unique_ptr<cudf::column>> cudf_columns;
  cudf_columns.push_back(cudf::test::fixed_width_column_wrapper<std::int32_t>{}.release());
  cudf_columns.push_back(cudf::test::strings_column_wrapper{}.release());
  CUCASCADE_CUDA_TRY(cudaDeviceSynchronize());
  gpu_table_representation gpu(
    std::make_unique<cudf::table>(std::move(cudf_columns)), *shared_gpu_space(), stream);
  representation_converter_registry registry;
  register_builtin_converters(registry);
  auto host = registry.convert<host_data_representation>(gpu, shared_host_space().get(), stream);
  stream.sync();

  const auto& src = source();
  auto fixed      = make_roomy_dst(src.columns[kInt32], 4, true);
  auto str        = make_roomy_dst(src.columns[kString32Nulls], 4, true);
  std::vector<column_destination> dests{fixed.dest, str.dest};
  const std::size_t columns[] = {0, 1};
  row_range_request request;
  request.columns   = columns;
  request.max_rows  = 4;
  const auto result = detail::read_rows_host(*host, request, dests, stream, nullptr, nullptr);
  REQUIRE(result.rows_written == 0);
  REQUIRE(result.source_exhausted);
  verify_read(src.columns[kInt32], fixed, 0, 0, 0, 0);
  verify_read(src.columns[kString32Nulls], str, 0, 0, 0, 0);
}

namespace {

/// Host callback that parks the stream until the gate's future is ready.
void CUDART_CB wait_for_gate(void* gate) { static_cast<std::shared_future<void>*>(gate)->wait(); }

/// Holds `stream` behind a blocked host callback until open() or destruction.
class stream_gate {
 public:
  explicit stream_gate(::cuda::stream_ref stream) : _stream(stream), _future(_promise.get_future())
  {
    CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(_stream.get(), wait_for_gate, &_future));
  }
  stream_gate(const stream_gate&)            = delete;
  stream_gate& operator=(const stream_gate&) = delete;
  ~stream_gate()
  {
    open();
    _stream.sync();  // the callback reads _future: let it finish before the gate goes away
  }

  void open()
  {
    if (!_open) {
      _open = true;
      _promise.set_value();
    }
  }

 private:
  ::cuda::stream_ref _stream;
  std::promise<void> _promise;
  std::shared_future<void> _future;
  bool _open{false};
};

}  // namespace

TEST_CASE("host row-range reader: source freed and its blocks reused before the copies run",
          "[row_range_reader_host]")
{
  const auto& src = source();
  auto stream     = shared_stream();

  // A source of its own, so the test can destroy it.
  representation_converter_registry registry;
  register_builtin_converters(registry);
  auto host =
    registry.convert<host_data_representation>(*src.gpu, shared_host_space().get(), stream);
  stream.sync();
  const std::weak_ptr<memory::fixed_size_host_memory_resource::multiple_blocks_allocation> blocks =
    host->get_host_table()->allocation;
  const std::size_t source_bytes = blocks.lock()->size_bytes();

  // Fixed-width, STRING chars + offsets, and a source null mask.
  const std::size_t columns[] = {kInt64Nulls, kFloat64, kString32Nulls, kString64};
  const std::size_t row_begin = 100;
  const std::size_t rows      = 800;
  std::vector<dst_column> dsts;
  std::vector<column_destination> dests;
  for (const auto c : columns) {
    dsts.push_back(make_roomy_dst(src.columns[c], rows, true, c == kString64 ? 8 : 4));
    dests.push_back(dsts.back().dest);
  }
  stream.sync();

  {
    // Nothing enqueued on the stream from here on runs until the gate opens.
    stream_gate gate{stream};

    row_range_request request;
    request.columns   = columns;
    request.row_begin = row_begin;
    request.max_rows  = rows;
    const auto result = detail::read_rows_host(*host, request, dests, stream, nullptr, nullptr);
    REQUIRE(result.rows_written == rows);

    // The reader is done and the caller drops the source, as after releasing a read lock on a
    // HOST batch (which records no reader event). The pending copies must keep the blocks.
    host.reset();
    REQUIRE_FALSE(blocks.expired());

    // Take what the pool hands out next and overwrite it. Had the source's blocks been returned,
    // these would be the same blocks and the pending copies would read this garbage.
    auto* host_mr =
      shared_host_space()->get_memory_resource_as<memory::fixed_size_host_memory_resource>();
    REQUIRE(host_mr != nullptr);
    auto garbage = host_mr->allocate_multiple_blocks(source_bytes);
    for (std::size_t b = 0; b < garbage->size(); ++b) {
      std::memset(garbage->at(b).data(), 0x5A, garbage->block_size());
    }

    gate.open();
    stream.sync();
  }

  // The blocks were released once the stream had run the copies.
  REQUIRE(blocks.expired());
  for (std::size_t i = 0; i < dsts.size(); ++i) {
    verify_read(src.columns[columns[i]], dsts[i], row_begin, rows, 0, 0);
  }
}
