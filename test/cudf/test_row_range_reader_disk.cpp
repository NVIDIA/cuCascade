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

#include <cucascade/cudf/builtin_converters.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/data/disk_io_backend.hpp>
#include <cucascade/data/io_backend_registry.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/data/row_range_read.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <stdlib.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace cucascade;

namespace {

constexpr std::uint8_t default_fill = 0xCD;

//===----------------------------------------------------------------------===//
// Shared spaces, stream and backends
//===----------------------------------------------------------------------===//

/// Temp directory the disk memory space writes to; removed at process exit.
struct temp_dir {
  std::filesystem::path path;

  temp_dir()
  {
    auto tmpl = (std::filesystem::temp_directory_path() / "cucascade_rr_disk_XXXXXX").string();
    if (::mkdtemp(tmpl.data()) == nullptr) {
      throw std::runtime_error("mkdtemp failed for " + tmpl);
    }
    path = tmpl;
  }
  temp_dir(const temp_dir&)            = delete;
  temp_dir& operator=(const temp_dir&) = delete;
  ~temp_dir()
  {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

const temp_dir& shared_temp_dir()
{
  static temp_dir dir;
  return dir;
}

/// A GPU memory space on device 0. mock_test_utils.hpp cannot be included next to
/// row_range_detail.hpp (its unqualified `cuda::mr` then resolves to cucascade::cuda).
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

/// DISK space on the temp directory, with the default (memory-space) I/O backend.
auto& shared_disk_space()
{
  static auto s = std::make_shared<memory::memory_space>(
    memory::disk_memory_space_config{0, std::size_t{1} << 30, shared_temp_dir().path.string()});
  return s;
}

::cuda::stream_ref shared_stream()
{
  static rmm::cuda_stream s;
  return s;
}

/// Backend that delegates to a real one and records every device read it is asked for.
class recording_io_backend : public idisk_io_backend {
 public:
  explicit recording_io_backend(std::shared_ptr<idisk_io_backend> inner) : _inner(std::move(inner))
  {
  }

  void write(const std::filesystem::path& path,
             const void* dev_ptr,
             std::size_t size,
             std::size_t file_offset,
             ::cuda::stream_ref stream) override
  {
    _inner->write(path, dev_ptr, size, file_offset, stream);
  }

  void read(const std::filesystem::path& path,
            void* dev_ptr,
            std::size_t size,
            std::size_t file_offset,
            ::cuda::stream_ref stream) override
  {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _device_read_offsets.push_back(file_offset);
    }
    _inner->read(path, dev_ptr, size, file_offset, stream);
  }

  void write(const std::filesystem::path& path,
             const void* host_ptr,
             std::size_t size,
             std::size_t file_offset) override
  {
    _inner->write(path, host_ptr, size, file_offset);
  }

  void read(const std::filesystem::path& path,
            void* host_ptr,
            std::size_t size,
            std::size_t file_offset) override
  {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      ++_host_reads;
    }
    _inner->read(path, host_ptr, size, file_offset);
  }

  [[nodiscard]] std::vector<std::size_t> device_read_offsets() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _device_read_offsets;
  }

  [[nodiscard]] std::size_t host_reads() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _host_reads;
  }

 private:
  std::shared_ptr<idisk_io_backend> _inner;
  mutable std::mutex _mutex;
  std::vector<std::size_t> _device_read_offsets;
  std::size_t _host_reads{0};
};

std::shared_ptr<recording_io_backend> make_recording_backend()
{
  io_backend_registry registry;
  register_builtin_io_backends(registry);
  return std::make_shared<recording_io_backend>(registry.create_backend("pipeline"));
}

//===----------------------------------------------------------------------===//
// Host <-> device helpers
//===----------------------------------------------------------------------===//

rmm::device_buffer to_device(const void* host, std::size_t bytes)
{
  rmm::device_buffer buffer(bytes, shared_stream());
  if (bytes > 0) {
    REQUIRE(cudaMemcpy(buffer.data(), host, bytes, cudaMemcpyHostToDevice) == cudaSuccess);
  }
  return buffer;
}

rmm::device_buffer device_filled(std::size_t bytes, std::uint8_t fill)
{
  const std::vector<std::uint8_t> host(bytes, fill);
  return to_device(host.data(), bytes);
}

std::vector<std::uint8_t> to_host(const rmm::device_buffer& buffer)
{
  std::vector<std::uint8_t> host(buffer.size());
  if (!host.empty()) {
    REQUIRE(cudaMemcpy(host.data(), buffer.data(), buffer.size(), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
  }
  return host;
}

//===----------------------------------------------------------------------===//
// Host model of the source table
//===----------------------------------------------------------------------===//

/// One source column kept on the host: the ground truth every read is compared against.
struct host_column {
  cudf::type_id type{cudf::type_id::EMPTY};
  std::size_t width{0};               ///< bytes per row; 0 for STRING
  std::vector<std::uint8_t> data;     ///< fixed-width values, or STRING chars
  std::vector<std::int32_t> offsets;  ///< STRING only: num_rows + 1 offsets
  std::vector<std::uint32_t> mask;    ///< validity words; empty when the column has no mask
  cudf::size_type null_count{0};

  [[nodiscard]] bool is_string() const { return type == cudf::type_id::STRING; }

  [[nodiscard]] bool valid(std::size_t row) const
  {
    return mask.empty() || ((mask[row / 32] >> (row % 32)) & 1U) != 0;
  }

  /// Chars bytes of rows [row_begin, row_begin + k) of a STRING column.
  [[nodiscard]] std::size_t chars_bytes(std::size_t row_begin, std::size_t k) const
  {
    return static_cast<std::size_t>(offsets[row_begin + k] - offsets[row_begin]);
  }

  /// Bytes a reader reports in data_bytes_written for rows [row_begin, row_begin + k).
  [[nodiscard]] std::size_t data_bytes(std::size_t row_begin, std::size_t k) const
  {
    return is_string() ? chars_bytes(row_begin, k) : k * width;
  }
};

std::uint32_t mix(std::size_t a, std::size_t b, std::uint32_t seed)
{
  auto x = static_cast<std::uint32_t>(a * 31U + b * 17U + seed);
  x *= 2654435761U;
  x ^= x >> 15;
  x *= 2246822519U;
  x ^= x >> 13;
  return x;
}

void add_mask(host_column& col, std::size_t num_rows, std::uint32_t seed)
{
  col.mask.assign((num_rows + 31) / 32, 0U);
  col.null_count = 0;
  for (std::size_t row = 0; row < num_rows; ++row) {
    // About one row in four is null, in an irregular pattern.
    if ((mix(row, 3, seed) & 3U) != 0) {
      col.mask[row / 32] |= (1U << (row % 32));
    } else {
      ++col.null_count;
    }
  }
}

host_column make_fixed_column(
  cudf::type_id type, std::size_t width, std::size_t num_rows, bool nullable, std::uint32_t seed)
{
  host_column col;
  col.type  = type;
  col.width = width;
  col.data.resize(num_rows * width);
  for (std::size_t row = 0; row < num_rows; ++row) {
    for (std::size_t b = 0; b < width; ++b) {
      auto value = static_cast<std::uint8_t>(mix(row, b, seed) >> 24);
      if (type == cudf::type_id::BOOL8) { value = static_cast<std::uint8_t>(value & 1U); }
      col.data[row * width + b] = value;
    }
  }
  if (nullable) { add_mask(col, num_rows, seed); }
  return col;
}

host_column make_string_column(std::size_t num_rows, bool nullable, std::uint32_t seed)
{
  host_column col;
  col.type = cudf::type_id::STRING;
  col.offsets.reserve(num_rows + 1);
  col.offsets.push_back(0);
  for (std::size_t row = 0; row < num_rows; ++row) {
    // Lengths 0..12, with every fifth row empty.
    const std::size_t length = (row % 5 == 0) ? 0 : (row * 7 + seed) % 13;
    for (std::size_t c = 0; c < length; ++c) {
      col.data.push_back(static_cast<std::uint8_t>('a' + (col.data.size() + seed) % 26));
    }
    col.offsets.push_back(static_cast<std::int32_t>(col.data.size()));
  }
  if (nullable) { add_mask(col, num_rows, seed); }
  return col;
}

std::unique_ptr<cudf::column> to_cudf_column(const host_column& col, std::size_t num_rows)
{
  const auto stream = shared_stream();
  const auto rows   = static_cast<cudf::size_type>(num_rows);
  auto mask = col.mask.empty() ? cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream)
                               : test::make_null_mask_from_host(rows, col.mask, stream);
  if (col.is_string()) {
    auto offsets = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      rows + 1,
      to_device(col.offsets.data(), col.offsets.size() * sizeof(std::int32_t)),
      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream),
      0);
    return cudf::make_strings_column(rows,
                                     std::move(offsets),
                                     to_device(col.data.data(), col.data.size()),
                                     col.null_count,
                                     std::move(mask));
  }
  return std::make_unique<cudf::column>(cudf::data_type{col.type},
                                        rows,
                                        to_device(col.data.data(), col.data.size()),
                                        std::move(mask),
                                        col.null_count);
}

// Column indices of the source table built by make_source().
constexpr std::size_t col_int8                       = 0;  // no mask
constexpr std::size_t col_int16                      = 1;  // nullable
constexpr std::size_t col_int32                      = 2;  // nullable
constexpr std::size_t col_int64                      = 3;  // no mask
[[maybe_unused]] constexpr std::size_t col_float64   = 4;  // nullable
constexpr std::size_t col_bool8                      = 5;  // no mask
[[maybe_unused]] constexpr std::size_t col_timestamp = 6;  // no mask
constexpr std::size_t col_string                     = 7;  // nullable
constexpr std::size_t col_string_all                 = 8;  // no mask

/// A source table spilled to disk, plus its host ground truth.
struct disk_source {
  std::size_t num_rows{0};
  std::vector<host_column> columns;
  std::unique_ptr<disk_data_representation> rep;
};

disk_source make_source(std::size_t num_rows)
{
  disk_source src;
  src.num_rows = num_rows;
  src.columns.push_back(make_fixed_column(cudf::type_id::INT8, 1, num_rows, false, 11));
  src.columns.push_back(make_fixed_column(cudf::type_id::INT16, 2, num_rows, true, 22));
  src.columns.push_back(make_fixed_column(cudf::type_id::INT32, 4, num_rows, true, 33));
  src.columns.push_back(make_fixed_column(cudf::type_id::INT64, 8, num_rows, false, 44));
  src.columns.push_back(make_fixed_column(cudf::type_id::FLOAT64, 8, num_rows, true, 55));
  src.columns.push_back(make_fixed_column(cudf::type_id::BOOL8, 1, num_rows, false, 66));
  src.columns.push_back(
    make_fixed_column(cudf::type_id::TIMESTAMP_MILLISECONDS, 8, num_rows, false, 77));
  src.columns.push_back(make_string_column(num_rows, true, 88));
  src.columns.push_back(make_string_column(num_rows, false, 99));

  std::vector<std::unique_ptr<cudf::column>> cudf_columns;
  cudf_columns.reserve(src.columns.size());
  for (const auto& col : src.columns) {
    cudf_columns.push_back(to_cudf_column(col, num_rows));
  }
  shared_stream().sync();

  representation_converter_registry registry;
  register_builtin_converters(registry);
  auto gpu_rep = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(cudf_columns)), *shared_gpu_space(), shared_stream());
  src.rep = registry.convert<disk_data_representation>(
    *gpu_rep, shared_disk_space().get(), shared_stream());
  shared_stream().sync();
  return src;
}

//===----------------------------------------------------------------------===//
// Destination slots and their verification
//===----------------------------------------------------------------------===//

/// Destination buffers of one requested column, prefilled with a sentinel byte.
struct slot {
  rmm::device_buffer data;
  rmm::device_buffer mask;
  rmm::device_buffer offsets;
  column_destination dst;
  std::uint8_t fill{default_fill};
};

struct slot_spec {
  std::size_t data_bytes{0};
  std::size_t mask_words{0};     ///< 0 => no null-mask slot
  std::size_t offsets_bytes{0};  ///< STRING only
  std::uint8_t offset_width{4};
  std::size_t dst_row_offset{0};
  std::size_t dst_byte_offset{0};
  std::uint8_t fill{default_fill};
};

slot make_slot(const host_column& col, const slot_spec& spec)
{
  slot s;
  s.fill    = spec.fill;
  s.data    = device_filled(spec.data_bytes, spec.fill);
  s.mask    = device_filled(spec.mask_words * sizeof(std::uint32_t), spec.fill);
  s.offsets = device_filled(col.is_string() ? spec.offsets_bytes : 0, spec.fill);

  s.dst.type_id         = static_cast<std::int32_t>(col.type);
  s.dst.data            = device_buffer_slot{s.data.data(), s.data.size()};
  s.dst.null_mask       = device_buffer_slot{s.mask.data(), s.mask.size()};
  s.dst.offsets         = device_buffer_slot{s.offsets.data(), s.offsets.size()};
  s.dst.offset_width    = spec.offset_width;
  s.dst.dst_row_offset  = spec.dst_row_offset;
  s.dst.dst_byte_offset = spec.dst_byte_offset;
  return s;
}

/// A slot large enough for `rows` rows starting at `row_begin`, appended at the given position.
slot make_roomy_slot(const host_column& col,
                     std::size_t row_begin,
                     std::size_t rows,
                     std::size_t dst_row_offset  = 0,
                     std::size_t dst_byte_offset = 0,
                     std::uint8_t offset_width   = 4,
                     bool with_mask              = true,
                     std::uint8_t fill           = default_fill)
{
  slot_spec spec;
  spec.offset_width    = offset_width;
  spec.dst_row_offset  = dst_row_offset;
  spec.dst_byte_offset = dst_byte_offset;
  spec.fill            = fill;
  // A few spare bytes/words after the range, so writes past it are caught.
  spec.mask_words = with_mask ? (dst_row_offset + rows + 31) / 32 + 2 : 0;
  if (col.is_string()) {
    spec.data_bytes    = dst_byte_offset + col.chars_bytes(row_begin, rows) + 16;
    spec.offsets_bytes = (dst_row_offset + rows + 1 + 3) * offset_width;
  } else {
    spec.data_bytes = (dst_row_offset + rows + 3) * col.width;
  }
  return make_slot(col, spec);
}

/**
 * @brief Check a slot holds exactly rows [row_begin, row_begin + k) of @p col at the append
 * position, and the sentinel everywhere else.
 */
void check_slot(const host_column& col,
                const slot& s,
                std::size_t row_begin,
                std::size_t k,
                std::size_t dst_row_offset,
                std::size_t dst_byte_offset)
{
  INFO("column type id " << static_cast<int>(col.type) << ", row_begin " << row_begin << ", k " << k
                         << ", dst_row_offset " << dst_row_offset << ", dst_byte_offset "
                         << dst_byte_offset);

  // ---- data (fixed-width values or chars) ----
  {
    std::vector<std::uint8_t> expected(s.data.size(), s.fill);
    const std::size_t src_begin =
      col.is_string() ? static_cast<std::size_t>(col.offsets[row_begin]) : row_begin * col.width;
    const std::size_t dst_begin = col.is_string() ? dst_byte_offset : dst_row_offset * col.width;
    const std::size_t bytes     = col.data_bytes(row_begin, k);
    REQUIRE(dst_begin + bytes <= expected.size());
    std::copy_n(col.data.begin() + static_cast<std::ptrdiff_t>(src_begin),
                bytes,
                expected.begin() + static_cast<std::ptrdiff_t>(dst_begin));
    const bool data_matches = to_host(s.data) == expected;
    CHECK(data_matches);
  }

  // ---- rebased offsets ----
  if (col.is_string()) {
    std::vector<std::uint8_t> expected(s.offsets.size(), s.fill);
    const std::size_t width = s.dst.offset_width;
    if (k > 0) {
      REQUIRE((dst_row_offset + k + 1) * width <= expected.size());
      for (std::size_t i = 0; i <= k; ++i) {
        const auto value = static_cast<std::int64_t>(dst_byte_offset) +
                           (col.offsets[row_begin + i] - col.offsets[row_begin]);
        auto* out = expected.data() + (dst_row_offset + i) * width;
        if (width == 4) {
          const auto narrow = static_cast<std::int32_t>(value);
          std::memcpy(out, &narrow, sizeof(narrow));
        } else {
          std::memcpy(out, &value, sizeof(value));
        }
      }
    }
    const bool offsets_match = to_host(s.offsets) == expected;
    CHECK(offsets_match);
  }

  // ---- null bits ----
  if (s.mask.size() > 0) {
    std::vector<std::uint32_t> expected(s.mask.size() / sizeof(std::uint32_t));
    std::memset(expected.data(), s.fill, expected.size() * sizeof(std::uint32_t));
    REQUIRE(dst_row_offset + k <= expected.size() * 32);
    for (std::size_t i = 0; i < k; ++i) {
      const std::size_t bit    = dst_row_offset + i;
      const std::uint32_t flag = 1U << (bit % 32);
      if (col.valid(row_begin + i)) {
        expected[bit / 32] |= flag;
      } else {
        expected[bit / 32] &= ~flag;
      }
    }
    const auto raw = to_host(s.mask);
    std::vector<std::uint32_t> actual(expected.size());
    std::memcpy(actual.data(), raw.data(), actual.size() * sizeof(std::uint32_t));
    const bool mask_matches = actual == expected;
    CHECK(mask_matches);
  }
}

/// Run one read of @p columns into @p slots (slot i receives column columns[i]).
row_range_result read_rows(const disk_source& src,
                           const std::vector<std::size_t>& columns,
                           std::vector<slot>& slots,
                           std::size_t row_begin,
                           std::size_t max_rows,
                           std::shared_ptr<idisk_io_backend> io_backend = nullptr)
{
  std::vector<column_destination> dst;
  dst.reserve(slots.size());
  for (const auto& s : slots) {
    dst.push_back(s.dst);
  }
  row_range_request request;
  request.columns   = std::span<const std::size_t>(columns);
  request.row_begin = row_begin;
  request.max_rows  = max_rows;
  auto result       = detail::read_rows_disk(*src.rep,
                                       request,
                                       std::span<column_destination>(dst),
                                       shared_stream(),
                                       nullptr,
                                       nullptr,
                                       std::move(io_backend));
  shared_stream().sync();
  return result;
}

/// Check result bookkeeping and slot contents of a read that delivered @p k rows.
void check_read(const disk_source& src,
                const std::vector<std::size_t>& columns,
                const std::vector<slot>& slots,
                const row_range_result& result,
                std::size_t row_begin,
                std::size_t k)
{
  REQUIRE(result.rows_written == k);
  REQUIRE(result.source_exhausted == (row_begin + k == src.num_rows));
  REQUIRE(result.data_bytes_written.size() == columns.size());
  for (std::size_t i = 0; i < columns.size(); ++i) {
    const auto& col = src.columns[columns[i]];
    CHECK(result.data_bytes_written[i] == col.data_bytes(row_begin, k));
    check_slot(
      col, slots[i], row_begin, k, slots[i].dst.dst_row_offset, slots[i].dst.dst_byte_offset);
  }
}

std::vector<std::size_t> all_columns(const disk_source& src)
{
  std::vector<std::size_t> columns(src.columns.size());
  for (std::size_t i = 0; i < columns.size(); ++i) {
    columns[i] = i;
  }
  return columns;
}

}  // namespace

//===----------------------------------------------------------------------===//
// 1. Windowed read of a multi-type table
//===----------------------------------------------------------------------===//

TEST_CASE("disk row-range reader reads a multi-type table in windows", "[row_range_reader_disk]")
{
  constexpr std::size_t num_rows = 5000;
  constexpr std::size_t window   = 777;
  const auto src                 = make_source(num_rows);
  const auto columns             = all_columns(src);

  std::vector<slot> slots;
  for (const auto& col : src.columns) {
    slots.push_back(make_roomy_slot(col, 0, num_rows));
  }

  std::size_t row_begin = 0;
  std::size_t windows   = 0;
  while (true) {
    for (std::size_t i = 0; i < slots.size(); ++i) {
      slots[i].dst.dst_row_offset = row_begin;
      if (src.columns[i].is_string()) {
        slots[i].dst.dst_byte_offset = src.columns[i].chars_bytes(0, row_begin);
      }
    }
    const auto result      = read_rows(src, columns, slots, row_begin, window);
    const std::size_t want = std::min(window, num_rows - row_begin);
    REQUIRE(result.rows_written == want);
    REQUIRE(result.source_exhausted == (row_begin + want == num_rows));
    REQUIRE(result.data_bytes_written.size() == columns.size());
    for (std::size_t i = 0; i < columns.size(); ++i) {
      CHECK(result.data_bytes_written[i] == src.columns[i].data_bytes(row_begin, want));
    }
    row_begin += want;
    ++windows;
    if (result.source_exhausted) { break; }
    REQUIRE(windows < 100);
  }
  REQUIRE(row_begin == num_rows);
  REQUIRE(windows == (num_rows + window - 1) / window);

  // The concatenation of all windows equals the source.
  for (std::size_t i = 0; i < slots.size(); ++i) {
    check_slot(src.columns[i], slots[i], 0, num_rows, 0, 0);
  }
}

//===----------------------------------------------------------------------===//
// 2. Unaligned row ranges
//===----------------------------------------------------------------------===//

TEST_CASE("disk row-range reader handles unaligned row ranges", "[row_range_reader_disk]")
{
  constexpr std::size_t num_rows = 5000;
  const auto src                 = make_source(num_rows);
  const auto backend             = make_recording_backend();
  // Requested out of table order, to check dst[i] <-> columns[i].
  const std::vector<std::size_t> columns{
    col_string, col_int32, col_int8, col_int64, col_string_all, col_int16};

  struct range {
    std::size_t row_begin;
    std::size_t max_rows;
  };
  // Odd starts and lengths; the large ones cross several 4 KiB file blocks, the small ones stay
  // inside one, and 1023/1025 straddle the first block boundary of the 4-byte column.
  const auto r = GENERATE(range{1237, 1531},
                          range{3, 5},
                          range{1023, 3},
                          range{1025, 2049},
                          range{4095, 1},
                          range{4999, 1});
  CAPTURE(r.row_begin, r.max_rows);
  std::vector<slot> slots;
  for (const auto c : columns) {
    slots.push_back(make_roomy_slot(src.columns[c], r.row_begin, r.max_rows));
  }
  const auto result = read_rows(src, columns, slots, r.row_begin, r.max_rows, backend);
  check_read(src, columns, slots, result, r.row_begin, r.max_rows);

  // Device reads are issued at 4 KiB-aligned file offsets (direct I/O / GDS requirement).
  const auto offsets = backend->device_read_offsets();
  REQUIRE_FALSE(offsets.empty());
  const bool all_aligned =
    std::all_of(offsets.begin(), offsets.end(), [](std::size_t off) { return off % 4096 == 0; });
  CHECK(all_aligned);
  // STRING offsets are fitted from a host-side read.
  CHECK(backend->host_reads() > 0);
}

//===----------------------------------------------------------------------===//
// 3. Budget truncation
//===----------------------------------------------------------------------===//

TEST_CASE("disk row-range reader truncates to the STRING byte budget", "[row_range_reader_disk]")
{
  constexpr std::size_t num_rows = 2000;
  const auto src                 = make_source(num_rows);
  const auto& str                = src.columns[col_string];
  const std::vector<std::size_t> columns{col_string};

  SECTION("chars capacity fits only some rows")
  {
    constexpr std::size_t row_begin = 11;
    constexpr std::size_t max_rows  = 200;
    // One byte short of 21 rows; row_begin + 20 is not empty, so exactly 20 rows fit.
    REQUIRE(str.chars_bytes(row_begin + 20, 1) > 0);
    const std::size_t capacity = str.chars_bytes(row_begin, 21) - 1;

    slot_spec spec;
    spec.data_bytes    = capacity;
    spec.mask_words    = 16;
    spec.offsets_bytes = (max_rows + 1) * 4;
    std::vector<slot> slots;
    slots.push_back(make_slot(str, spec));

    const auto result = read_rows(src, columns, slots, row_begin, max_rows);
    REQUIRE(result.rows_written == 20);
    REQUIRE_FALSE(result.source_exhausted);
    check_read(src, columns, slots, result, row_begin, 20);
  }

  SECTION("chars capacity with an append offset")
  {
    constexpr std::size_t row_begin       = 301;
    constexpr std::size_t max_rows        = 500;
    constexpr std::size_t dst_byte_offset = 37;
    constexpr std::size_t capacity        = 400;

    // Brute-force rule 1 for the chars slot.
    std::size_t expected = 0;
    while (expected < max_rows &&
           dst_byte_offset + str.chars_bytes(row_begin, expected + 1) <= capacity) {
      ++expected;
    }
    REQUIRE(expected > 0);
    REQUIRE(expected < max_rows);

    slot_spec spec;
    spec.data_bytes      = capacity;
    spec.mask_words      = 32;
    spec.offsets_bytes   = (max_rows + 1) * 8;
    spec.offset_width    = 8;
    spec.dst_byte_offset = dst_byte_offset;
    std::vector<slot> slots;
    slots.push_back(make_slot(str, spec));

    const auto result = read_rows(src, columns, slots, row_begin, max_rows);
    REQUIRE_FALSE(result.source_exhausted);
    check_read(src, columns, slots, result, row_begin, expected);
  }

  SECTION("a single row larger than the chars capacity delivers nothing")
  {
    constexpr std::size_t row_begin = 1;
    const std::size_t first_row     = str.chars_bytes(row_begin, 1);
    REQUIRE(first_row >= 2);

    slot_spec spec;
    spec.data_bytes    = first_row - 1;
    spec.mask_words    = 4;
    spec.offsets_bytes = 64;
    std::vector<slot> slots;
    slots.push_back(make_slot(str, spec));

    const auto result = read_rows(src, columns, slots, row_begin, 50);
    REQUIRE(result.rows_written == 0);
    REQUIRE_FALSE(result.source_exhausted);
    // Nothing is written: data, offsets and mask keep the sentinel.
    check_read(src, columns, slots, result, row_begin, 0);
  }

  SECTION("offsets capacity limits the rows")
  {
    constexpr std::size_t row_begin      = 77;
    constexpr std::size_t max_rows       = 100;
    constexpr std::size_t dst_row_offset = 5;
    const auto width                     = GENERATE(std::uint8_t{4}, std::uint8_t{8});

    slot_spec spec;
    spec.data_bytes     = str.chars_bytes(row_begin, max_rows) + 8;
    spec.mask_words     = 8;
    spec.offset_width   = width;
    spec.dst_row_offset = dst_row_offset;
    // Room for 9 rows (10 offsets) after the append position, plus a partial offset.
    spec.offsets_bytes = (dst_row_offset + 9 + 1) * width + (width - 1U);
    std::vector<slot> slots;
    slots.push_back(make_slot(str, spec));

    const auto result = read_rows(src, columns, slots, row_begin, max_rows);
    REQUIRE(result.rows_written == 9);
    REQUIRE_FALSE(result.source_exhausted);
    check_read(src, columns, slots, result, row_begin, 9);
  }

  SECTION("the tightest column decides for a mixed request")
  {
    constexpr std::size_t row_begin = 123;
    constexpr std::size_t max_rows  = 300;
    const std::vector<std::size_t> mixed{col_int64, col_string, col_int16};

    std::vector<slot> slots;
    slots.push_back(make_roomy_slot(src.columns[col_int64], row_begin, max_rows));
    slots.push_back(make_roomy_slot(src.columns[col_string], row_begin, max_rows));
    // INT16 data slot: 41 rows and one spare byte.
    slot_spec spec;
    spec.data_bytes = 41 * 2 + 1;
    spec.mask_words = 16;
    slots.push_back(make_slot(src.columns[col_int16], spec));

    const auto result = read_rows(src, mixed, slots, row_begin, max_rows);
    REQUIRE(result.rows_written == 41);
    REQUIRE_FALSE(result.source_exhausted);
    check_read(src, mixed, slots, result, row_begin, 41);
  }
}

//===----------------------------------------------------------------------===//
// 4. Rebased offsets
//===----------------------------------------------------------------------===//

TEST_CASE("disk row-range reader rebases STRING offsets at the append position",
          "[row_range_reader_disk]")
{
  constexpr std::size_t num_rows        = 3000;
  constexpr std::size_t row_begin       = 101;
  constexpr std::size_t max_rows        = 333;
  constexpr std::size_t dst_row_offset  = 17;
  constexpr std::size_t dst_byte_offset = 29;
  const auto src                        = make_source(num_rows);
  const auto width                      = GENERATE(std::uint8_t{4}, std::uint8_t{8});
  CAPTURE(static_cast<int>(width));

  const std::vector<std::size_t> columns{col_string, col_string_all, col_int32};
  std::vector<slot> slots;
  slots.push_back(make_roomy_slot(
    src.columns[col_string], row_begin, max_rows, dst_row_offset, dst_byte_offset, width));
  slots.push_back(make_roomy_slot(
    src.columns[col_string_all], row_begin, max_rows, dst_row_offset, dst_byte_offset, width));
  slots.push_back(make_roomy_slot(src.columns[col_int32], row_begin, max_rows, dst_row_offset));

  const auto result = read_rows(src, columns, slots, row_begin, max_rows);
  // check_slot verifies the k + 1 rebased offsets, the chars at dst_byte_offset, and that every
  // byte before the append positions (and after the range) still holds the sentinel.
  check_read(src, columns, slots, result, row_begin, max_rows);
}

//===----------------------------------------------------------------------===//
// 5. Null bits at unaligned offsets
//===----------------------------------------------------------------------===//

TEST_CASE("disk row-range reader writes null bits at unaligned positions",
          "[row_range_reader_disk]")
{
  constexpr std::size_t num_rows = 1000;
  const auto src                 = make_source(num_rows);

  SECTION("destination bits outside the range are preserved")
  {
    struct layout {
      std::size_t row_begin;
      std::size_t rows;
      std::size_t dst_row_offset;
    };
    const auto l = GENERATE(layout{37, 77, 13},   // spans several words on both sides
                            layout{5, 3, 9},      // inside one source and one destination word
                            layout{31, 2, 31},    // crosses a word boundary on both sides
                            layout{64, 64, 1},    // aligned source, unaligned destination
                            layout{33, 100, 0});  // unaligned source, aligned destination
    CAPTURE(l.row_begin, l.rows, l.dst_row_offset);
    // 0xA5 = 1010'0101: both 0s and 1s must survive around the written range.
    constexpr std::uint8_t fill = 0xA5;

    // Nullable fixed-width, nullable STRING, and two all-valid sources (1s are written).
    const std::vector<std::size_t> columns{col_int32, col_string, col_int8, col_string_all};
    std::vector<slot> slots;
    for (const auto c : columns) {
      slots.push_back(
        make_roomy_slot(src.columns[c], l.row_begin, l.rows, l.dst_row_offset, 0, 4, true, fill));
    }
    REQUIRE(src.columns[col_int32].null_count > 0);
    REQUIRE(src.columns[col_string].null_count > 0);

    const auto result = read_rows(src, columns, slots, l.row_begin, l.rows);
    check_read(src, columns, slots, result, l.row_begin, l.rows);
  }

  SECTION("an all-valid source needs no null-mask slot")
  {
    const std::vector<std::size_t> columns{col_int64, col_string_all};
    std::vector<slot> slots;
    for (const auto c : columns) {
      slots.push_back(make_roomy_slot(src.columns[c], 41, 99, 0, 0, 4, false));
    }
    const auto result = read_rows(src, columns, slots, 41, 99);
    check_read(src, columns, slots, result, 41, 99);
  }

  SECTION("a nullable source without a null-mask slot is refused")
  {
    const auto c = GENERATE(col_int32, col_string);
    const std::vector<std::size_t> columns{c};
    std::vector<slot> slots;
    slots.push_back(make_roomy_slot(src.columns[c], 41, 99, 0, 0, 4, false));
    REQUIRE_THROWS_AS(read_rows(src, columns, slots, 41, 99), row_range_unsupported_error);
  }
}

//===----------------------------------------------------------------------===//
// 6. Exhausted source and request validation
//===----------------------------------------------------------------------===//

TEST_CASE("disk row-range reader reports an exhausted source", "[row_range_reader_disk]")
{
  constexpr std::size_t num_rows = 600;
  const auto src                 = make_source(num_rows);
  const std::vector<std::size_t> columns{col_int32, col_string, col_bool8};

  auto make_slots = [&](std::size_t row_begin, std::size_t rows) {
    std::vector<slot> slots;
    for (const auto c : columns) {
      slots.push_back(make_roomy_slot(src.columns[c], row_begin, rows));
    }
    return slots;
  };

  SECTION("the last rows exhaust the source")
  {
    auto slots        = make_slots(num_rows - 3, 3);
    const auto result = read_rows(src, columns, slots, num_rows - 3, 100);
    REQUIRE(result.source_exhausted);
    check_read(src, columns, slots, result, num_rows - 3, 3);
  }

  SECTION("row_begin == num_rows reads nothing")
  {
    auto slots        = make_slots(num_rows, 0);
    const auto result = read_rows(src, columns, slots, num_rows, 100);
    REQUIRE(result.rows_written == 0);
    REQUIRE(result.source_exhausted);
    check_read(src, columns, slots, result, num_rows, 0);
  }

  SECTION("row_begin > num_rows throws")
  {
    auto slots = make_slots(num_rows, 0);
    REQUIRE_THROWS_AS(read_rows(src, columns, slots, num_rows + 1, 100), std::out_of_range);
  }

  SECTION("a column index past the table throws")
  {
    const std::vector<std::size_t> bad{src.columns.size()};
    std::vector<slot> slots;
    slots.push_back(make_roomy_slot(src.columns[col_int32], 0, 10));
    REQUIRE_THROWS_AS(read_rows(src, bad, slots, 0, 10), std::out_of_range);
  }

  SECTION("a destination of another type throws")
  {
    const std::vector<std::size_t> one{col_int64};
    std::vector<slot> slots;
    slots.push_back(make_roomy_slot(src.columns[col_int64], 0, 10));
    slots[0].dst.type_id = static_cast<std::int32_t>(cudf::type_id::FLOAT64);
    REQUIRE_THROWS_AS(read_rows(src, one, slots, 0, 10), std::invalid_argument);
  }
}

//===----------------------------------------------------------------------===//
// 7. Explicit I/O backend vs the memory space's backend
//===----------------------------------------------------------------------===//

TEST_CASE("disk row-range reader gives the same result with an explicit io backend",
          "[row_range_reader_disk]")
{
  constexpr std::size_t num_rows  = 3000;
  constexpr std::size_t row_begin = 411;
  constexpr std::size_t max_rows  = 1777;
  const auto src                  = make_source(num_rows);
  const auto columns              = all_columns(src);
  const auto backend              = make_recording_backend();

  auto make_slots = [&]() {
    std::vector<slot> slots;
    for (const auto& col : src.columns) {
      slots.push_back(make_roomy_slot(col, row_begin, max_rows, 7, 11));
    }
    return slots;
  };

  // nullptr => the backend of the source's memory space; the explicit one is not involved.
  auto default_slots        = make_slots();
  const auto default_result = read_rows(src, columns, default_slots, row_begin, max_rows);
  check_read(src, columns, default_slots, default_result, row_begin, max_rows);
  REQUIRE(backend->device_read_offsets().empty());
  REQUIRE(backend->host_reads() == 0);

  auto explicit_slots = make_slots();
  const auto explicit_result =
    read_rows(src, columns, explicit_slots, row_begin, max_rows, backend);
  check_read(src, columns, explicit_slots, explicit_result, row_begin, max_rows);
  REQUIRE_FALSE(backend->device_read_offsets().empty());
  REQUIRE(backend->host_reads() > 0);

  REQUIRE(explicit_result.rows_written == default_result.rows_written);
  REQUIRE(explicit_result.source_exhausted == default_result.source_exhausted);
  REQUIRE(explicit_result.data_bytes_written == default_result.data_bytes_written);
  for (std::size_t i = 0; i < columns.size(); ++i) {
    CAPTURE(i);
    const bool same_data = to_host(explicit_slots[i].data) == to_host(default_slots[i].data);
    const bool same_mask = to_host(explicit_slots[i].mask) == to_host(default_slots[i].mask);
    const bool same_offsets =
      to_host(explicit_slots[i].offsets) == to_host(default_slots[i].offsets);
    CHECK(same_data);
    CHECK(same_mask);
    CHECK(same_offsets);
  }
}

//===----------------------------------------------------------------------===//
// 8. Edge cases: null-mask budget, empty windows, wrong source
//===----------------------------------------------------------------------===//

TEST_CASE("disk row-range reader edge cases", "[row_range_reader_disk]")
{
  constexpr std::size_t num_rows = 700;
  const auto src                 = make_source(num_rows);

  SECTION("null-mask capacity limits the rows")
  {
    // One mask word with the append position at bit 5: 27 rows fit (rule 1, null-mask term).
    constexpr std::size_t row_begin      = 19;
    constexpr std::size_t dst_row_offset = 5;
    const auto c                         = GENERATE(col_int32, col_string);
    CAPTURE(c);
    const std::vector<std::size_t> columns{c};
    auto s          = make_roomy_slot(src.columns[c], row_begin, 200, dst_row_offset);
    s.mask          = device_filled(sizeof(std::uint32_t), default_fill);
    s.dst.null_mask = device_buffer_slot{s.mask.data(), s.mask.size()};
    std::vector<slot> slots;
    slots.push_back(std::move(s));

    const auto result = read_rows(src, columns, slots, row_begin, 200);
    REQUIRE(result.rows_written == 32 - dst_row_offset);
    REQUIRE_FALSE(result.source_exhausted);
    check_read(src, columns, slots, result, row_begin, 32 - dst_row_offset);
  }

  SECTION("a window of empty strings writes offsets but no chars")
  {
    // Row 5 is empty in every STRING column (every fifth row is).
    constexpr std::size_t row_begin       = 5;
    constexpr std::size_t dst_row_offset  = 3;
    constexpr std::size_t dst_byte_offset = 13;
    const std::vector<std::size_t> columns{col_string, col_string_all};
    REQUIRE(src.columns[col_string].chars_bytes(row_begin, 1) == 0);
    REQUIRE(src.columns[col_string_all].chars_bytes(row_begin, 1) == 0);
    std::vector<slot> slots;
    for (const auto c : columns) {
      slots.push_back(
        make_roomy_slot(src.columns[c], row_begin, 1, dst_row_offset, dst_byte_offset, 8));
    }
    const auto result = read_rows(src, columns, slots, row_begin, 1);
    REQUIRE(result.rows_written == 1);
    CHECK(result.data_bytes_written == std::vector<std::size_t>{0, 0});
    check_read(src, columns, slots, result, row_begin, 1);
  }

  SECTION("max_rows == 0 delivers nothing and does not exhaust the source")
  {
    const std::vector<std::size_t> columns{col_int16, col_string, col_int64};
    std::vector<slot> slots;
    for (const auto c : columns) {
      slots.push_back(make_roomy_slot(src.columns[c], 50, 10));
    }
    const auto result = read_rows(src, columns, slots, 50, 0);
    REQUIRE(result.rows_written == 0);
    REQUIRE_FALSE(result.source_exhausted);
    check_read(src, columns, slots, result, 50, 0);
  }

  SECTION("a fixed-width-only read of the whole table")
  {
    const std::vector<std::size_t> columns{col_int8, col_int16, col_int32, col_int64, col_bool8};
    std::vector<slot> slots;
    for (const auto c : columns) {
      slots.push_back(make_roomy_slot(src.columns[c], 0, num_rows));
    }
    const auto result = read_rows(src, columns, slots, 0, num_rows + 50);
    REQUIRE(result.source_exhausted);
    check_read(src, columns, slots, result, 0, num_rows);
  }

  SECTION("a source that is not a disk representation is refused")
  {
    const auto col = make_fixed_column(cudf::type_id::INT32, 4, 10, false, 5);
    std::vector<std::unique_ptr<cudf::column>> cudf_columns;
    cudf_columns.push_back(to_cudf_column(col, 10));
    gpu_table_representation gpu_rep(
      std::make_unique<cudf::table>(std::move(cudf_columns)), *shared_gpu_space(), shared_stream());

    std::vector<slot> slots;
    slots.push_back(make_roomy_slot(col, 0, 10));
    std::vector<column_destination> dst{slots[0].dst};
    const std::vector<std::size_t> columns{0};
    row_range_request request;
    request.columns   = std::span<const std::size_t>(columns);
    request.row_begin = 0;
    request.max_rows  = 10;
    REQUIRE_THROWS_AS(detail::read_rows_disk(gpu_rep,
                                             request,
                                             std::span<column_destination>(dst),
                                             shared_stream(),
                                             nullptr,
                                             nullptr,
                                             nullptr),
                      std::invalid_argument);
  }
}
