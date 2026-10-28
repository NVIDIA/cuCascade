/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "utils/cuda_test_utils.hpp"
#include "utils/cudf_test_utils.hpp"
#include "utils/disk_test_utils.hpp"
#include "utils/gpu_source_test_utils.hpp"
#include "utils/mock_test_utils.hpp"

#include <cucascade/cudf/builtin_converters.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <set>
#include <thread>
#include <vector>

using namespace cucascade;

namespace {

/// Shared memory spaces — reused across all tests to avoid cumulative CUDA context
/// degradation from repeated creation/destruction of CUDA stream pools and pinned memory.
auto& shared_gpu_space()
{
  static auto s = test::make_mock_memory_space(memory::Tier::GPU, 0);
  return s;
}
::cuda::stream_ref shared_stream()
{
  static rmm::cuda_stream s;
  return s;
}

/// Round-trip test helper: GPU -> disk -> GPU (direct 3-step), compare tables. Runs once per
/// disk io stack (the adopted default and every forced backend that can write device memory).
void gpu_disk_round_trip_test(std::unique_ptr<cudf::table> original_table)
{
  auto& gpu_space = shared_gpu_space();

  representation_converter_registry registry;
  register_builtin_converters(registry);

  // Create GPU representation from the original table
  auto gpu_rep = std::make_unique<gpu_table_representation>(
    std::move(original_table), *gpu_space, ::cuda::stream_ref{cudaStream_t{cudaStreamDefault}});

  for (auto const backend : test::disk_test_backends) {
    auto* disk_space = test::shared_disk_space_for(backend);
    if (disk_space == nullptr) {
      WARN("skipping the " << test::to_string(backend) << " disk backend: unavailable here");
      continue;
    }
    if (!test::disk_space_can_write(*disk_space, /*device=*/true)) {
      WARN("skipping the " << test::to_string(backend) << " disk backend: no device writes");
      continue;
    }
    CAPTURE(test::to_string(backend));

    // GPU -> disk (one batched device write)
    auto disk_rep =
      registry.convert<disk_data_representation>(*gpu_rep, disk_space, shared_stream());

    // disk -> GPU (one batched device read)
    auto gpu_rep2 =
      registry.convert<gpu_table_representation>(*disk_rep, gpu_space.get(), shared_stream());

    // Compare original (from gpu_rep) with round-tripped
    test::expect_cudf_tables_equal_on_stream(
      gpu_rep->get_table_view(), gpu_rep2->get_table_view(), shared_stream());
  }
}

/// Create a simple column of the given type with uninitialized data.
/// Returns a single-column cudf::table.
std::unique_ptr<cudf::table> make_typed_table(cudf::type_id type, cudf::size_type num_rows)
{
  auto const dtype = cudf::data_type{type};
  std::vector<std::unique_ptr<cudf::column>> cols;
  if (cudf::is_numeric(dtype)) {
    // cudf::is_numeric includes BOOL8
    auto col =
      cudf::make_numeric_column(dtype, num_rows, cudf::mask_state::UNALLOCATED, shared_stream());
    cols.push_back(std::move(col));
  } else if (cudf::is_timestamp(dtype)) {
    auto col =
      cudf::make_timestamp_column(dtype, num_rows, cudf::mask_state::UNALLOCATED, shared_stream());
    cols.push_back(std::move(col));
  } else if (cudf::is_duration(dtype)) {
    auto col =
      cudf::make_duration_column(dtype, num_rows, cudf::mask_state::UNALLOCATED, shared_stream());
    cols.push_back(std::move(col));
  } else if (cudf::is_fixed_point(dtype)) {
    auto const fp_dtype = cudf::data_type{type, -2};  // scale = -2
    auto col            = cudf::make_fixed_point_column(
      fp_dtype, num_rows, cudf::mask_state::UNALLOCATED, shared_stream());
    cols.push_back(std::move(col));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

/// Create a table with ALL_NULL mask.
std::unique_ptr<cudf::table> make_typed_table_with_nulls(cudf::type_id type,
                                                         cudf::size_type num_rows)
{
  auto const dtype = cudf::data_type{type};
  std::vector<std::unique_ptr<cudf::column>> cols;
  if (cudf::is_numeric(dtype)) {
    auto col =
      cudf::make_numeric_column(dtype, num_rows, cudf::mask_state::ALL_NULL, shared_stream());
    cols.push_back(std::move(col));
  } else if (cudf::is_timestamp(dtype)) {
    auto col =
      cudf::make_timestamp_column(dtype, num_rows, cudf::mask_state::ALL_NULL, shared_stream());
    cols.push_back(std::move(col));
  } else if (cudf::is_duration(dtype)) {
    auto col =
      cudf::make_duration_column(dtype, num_rows, cudf::mask_state::ALL_NULL, shared_stream());
    cols.push_back(std::move(col));
  } else if (cudf::is_fixed_point(dtype)) {
    auto const fp_dtype = cudf::data_type{type, -2};
    auto col            = cudf::make_fixed_point_column(
      fp_dtype, num_rows, cudf::mask_state::ALL_NULL, shared_stream());
    cols.push_back(std::move(col));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

}  // namespace

TEST_CASE(
  "gpu disk spill from another current GPU orders after the source's writer on the null stream",
  "[disk][gpu-converter][multi_gpu]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires two CUDA devices"); }
  auto const backend = GENERATE(disk_io_backend::uring, disk_io_backend::cufile);
  auto* disk_space   = test::shared_disk_space_for(backend);
  if (disk_space == nullptr || !test::disk_space_can_write(*disk_space, /*device=*/true)) {
    SKIP("disk backend cannot write device memory here");
  }
  CAPTURE(test::to_string(backend));
  rmm::cuda_set_device_raii source_device{rmm::cuda_device_id{1}};
  auto gpu1 = test::make_mock_memory_space(memory::Tier::GPU, 1);
  rmm::cuda_stream writer{rmm::cuda_stream::flags::non_blocking};
  constexpr cudf::size_type rows = 1 << 20;
  constexpr std::size_t bytes    = rows * sizeof(std::int32_t);
  auto column                    = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, rows, cudf::mask_state::UNALLOCATED, writer);
  auto* data = column->mutable_view().data<std::int32_t>();
  REQUIRE(cudaMemsetAsync(data, 0x11, bytes, writer.value()) == cudaSuccess);
  writer.synchronize();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  auto table = std::make_unique<cudf::table>(std::move(columns));
  REQUIRE(cudaLaunchHostFunc(
            writer.value(),
            [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds{300}); },
            nullptr) == cudaSuccess);
  REQUIRE(cudaMemsetAsync(data, 0x22, bytes, writer.value()) == cudaSuccess);
  gpu_table_representation source{std::move(table), *gpu1, writer};
  representation_converter_registry registry;
  register_builtin_converters(registry);
  {
    rmm::cuda_set_device_raii caller_device{rmm::cuda_device_id{0}};
    auto disk = registry.convert<disk_data_representation>(
      source, disk_space, ::cuda::stream_ref{cudaStream_t{nullptr}});
    int current_device = -1;
    REQUIRE(cudaGetDevice(&current_device) == cudaSuccess);
    CHECK(current_device == 0);
    auto const& allocation = disk->get_disk_table();
    REQUIRE(allocation.columns.size() == 1);
    REQUIRE(allocation.columns.front().data_size == bytes);
    std::vector<std::uint8_t> contents(bytes);
    std::ifstream in(allocation.file_path, std::ios::binary);
    in.seekg(allocation.columns.front().data_offset);
    in.read(reinterpret_cast<char*>(contents.data()), contents.size());
    REQUIRE(in);
    CHECK(std::ranges::all_of(contents, [](auto byte) { return byte == 0x22; }));
  }
  writer.synchronize();
}

TEST_CASE("gpu disk spill rejects an explicit stream from another device before creating a file",
          "[disk][gpu-converter][multi_gpu]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires two CUDA devices"); }
  auto const backend = GENERATE(disk_io_backend::uring, disk_io_backend::cufile);
  auto* disk_space   = test::shared_disk_space_for(backend);
  if (disk_space == nullptr || !test::disk_space_can_write(*disk_space, /*device=*/true)) {
    SKIP("disk backend cannot write device memory here");
  }
  CAPTURE(test::to_string(backend));
  rmm::cuda_set_device_raii source_device{rmm::cuda_device_id{1}};
  auto gpu1 = test::make_mock_memory_space(memory::Tier::GPU, 1);
  rmm::cuda_stream writer{rmm::cuda_stream::flags::non_blocking};
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 1024, cudf::mask_state::UNALLOCATED, writer));
  auto table = std::make_unique<cudf::table>(std::move(columns));
  writer.synchronize();
  gpu_table_representation source{std::move(table), *gpu1, writer};
  representation_converter_registry registry;
  register_builtin_converters(registry);
  rmm::cuda_set_device_raii caller_device{rmm::cuda_device_id{0}};
  rmm::cuda_stream foreign_stream{rmm::cuda_stream::flags::non_blocking};
  auto files = [&] {
    std::set<std::filesystem::path> paths;
    for (auto const& entry :
         std::filesystem::directory_iterator(disk_space->get_disk_mount_path())) {
      paths.insert(entry.path());
    }
    return paths;
  };
  auto const before = files();
  REQUIRE_THROWS_AS(registry.convert<disk_data_representation>(source, disk_space, foreign_stream),
                    cucascade::logic_error);
  CHECK(files() == before);
}

// =============================================================================
// Numeric type round-trip tests
// =============================================================================

TEST_CASE("gpu disk round-trip numeric types", "[disk][gpu-converter][numeric]")
{
  auto type = GENERATE(cudf::type_id::INT8,
                       cudf::type_id::INT16,
                       cudf::type_id::INT32,
                       cudf::type_id::INT64,
                       cudf::type_id::UINT8,
                       cudf::type_id::UINT16,
                       cudf::type_id::UINT32,
                       cudf::type_id::UINT64,
                       cudf::type_id::FLOAT32,
                       cudf::type_id::FLOAT64);

  SECTION("100 rows") { gpu_disk_round_trip_test(make_typed_table(type, 100)); }
}

// =============================================================================
// BOOL8 round-trip tests
// =============================================================================

TEST_CASE("gpu disk round-trip bool type", "[disk][gpu-converter][bool]")
{
  gpu_disk_round_trip_test(make_typed_table(cudf::type_id::BOOL8, 100));
}

// =============================================================================
// Timestamp round-trip tests
// =============================================================================

TEST_CASE("gpu disk round-trip timestamp types", "[disk][gpu-converter][timestamp]")
{
  auto type = GENERATE(cudf::type_id::TIMESTAMP_DAYS,
                       cudf::type_id::TIMESTAMP_SECONDS,
                       cudf::type_id::TIMESTAMP_MILLISECONDS,
                       cudf::type_id::TIMESTAMP_MICROSECONDS,
                       cudf::type_id::TIMESTAMP_NANOSECONDS);

  gpu_disk_round_trip_test(make_typed_table(type, 100));
}

// =============================================================================
// Duration round-trip tests
// =============================================================================

TEST_CASE("gpu disk round-trip duration types", "[disk][gpu-converter][duration]")
{
  auto type = GENERATE(cudf::type_id::DURATION_DAYS,
                       cudf::type_id::DURATION_SECONDS,
                       cudf::type_id::DURATION_MILLISECONDS,
                       cudf::type_id::DURATION_MICROSECONDS,
                       cudf::type_id::DURATION_NANOSECONDS);

  gpu_disk_round_trip_test(make_typed_table(type, 100));
}

// =============================================================================
// Decimal round-trip tests
// =============================================================================

TEST_CASE("gpu disk round-trip decimal types", "[disk][gpu-converter][decimal]")
{
  auto type =
    GENERATE(cudf::type_id::DECIMAL32, cudf::type_id::DECIMAL64, cudf::type_id::DECIMAL128);

  gpu_disk_round_trip_test(make_typed_table(type, 100));
}

// =============================================================================
// Null mask round-trip tests
// =============================================================================

TEST_CASE("gpu disk round-trip all-null column", "[disk][gpu-converter][null]")
{
  gpu_disk_round_trip_test(make_typed_table_with_nulls(cudf::type_id::INT32, 100));
}

TEST_CASE("gpu disk round-trip no-null column", "[disk][gpu-converter][null]")
{
  gpu_disk_round_trip_test(make_typed_table(cudf::type_id::INT32, 100));
}

TEST_CASE("gpu disk round-trip mixed-null column", "[disk][gpu-converter][null]")
{
  // Exercise masks beyond the first quarter of the allocation, including a partial word.
  auto const num_rows = GENERATE(100, 4097);

  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows, cudf::mask_state::ALL_VALID, shared_stream());

  // Set every other bit to null using cudf utilities
  std::vector<cudf::bitmask_type> host_mask(
    cudf::bitmask_allocation_size_bytes(num_rows) / sizeof(cudf::bitmask_type),
    ~cudf::bitmask_type{0});
  // Clear every other bit (set elements 0, 2, 4, ... to null)
  constexpr auto bits_per_word = sizeof(cudf::bitmask_type) * 8;
  for (int i = 0; i < num_rows; i += 2) {
    auto word_idx = static_cast<std::size_t>(i) / bits_per_word;
    auto bit_idx  = static_cast<std::size_t>(i) % bits_per_word;
    host_mask[word_idx] &= ~(cudf::bitmask_type{1} << bit_idx);
  }
  auto dev_mask = test::make_null_mask_from_host(num_rows, host_mask, shared_stream());
  col->set_null_mask(std::move(dev_mask), (num_rows + 1) / 2);

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  gpu_disk_round_trip_test(std::make_unique<cudf::table>(std::move(cols)));
}

// =============================================================================
// STRING column round-trip tests
// =============================================================================

TEST_CASE("gpu disk round-trip string column", "[disk][gpu-converter][string]")
{
  std::vector<std::string> host_strings = {"hello", "", "world", "test123", "", "a"};
  auto const num_strings                = static_cast<cudf::size_type>(host_strings.size());

  // Compute offsets
  std::vector<int32_t> host_offsets(static_cast<std::size_t>(num_strings) + 1);
  host_offsets[0] = 0;
  for (std::size_t i = 0; i < host_strings.size(); ++i) {
    host_offsets[i + 1] = host_offsets[i] + static_cast<int32_t>(host_strings[i].size());
  }
  auto const total_chars = static_cast<std::size_t>(host_offsets.back());

  // Concatenate all chars
  std::vector<char> host_chars;
  host_chars.reserve(total_chars);
  for (const auto& s : host_strings) {
    host_chars.insert(host_chars.end(), s.begin(), s.end());
  }

  // Create device buffers
  rmm::device_buffer dev_chars(host_chars.data(), host_chars.size(), shared_stream());
  auto offsets_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                               num_strings + 1,
                                               cudf::mask_state::UNALLOCATED,
                                               shared_stream());
  CUCASCADE_CUDA_TRY(cudaMemcpyAsync(offsets_col->mutable_view().data<int32_t>(),
                                     host_offsets.data(),
                                     host_offsets.size() * sizeof(int32_t),
                                     cudaMemcpyHostToDevice,
                                     shared_stream().get()));
  shared_stream().sync();

  auto str_col =
    cudf::make_strings_column(num_strings,
                              std::move(offsets_col),
                              std::move(dev_chars),
                              0,
                              cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(str_col));
  gpu_disk_round_trip_test(std::make_unique<cudf::table>(std::move(cols)));
}

// Regression: see test_disk_host_converters.cpp — same root cause, but here we exercise the
// direct disk→GPU path through reconstruct_column_from_disk.
TEST_CASE("gpu disk round-trip empty strings column", "[disk][gpu-converter][string][empty]")
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}));
  gpu_disk_round_trip_test(std::make_unique<cudf::table>(std::move(cols)));
}

TEST_CASE("gpu disk round-trip empty strings column mixed with non-empty column",
          "[disk][gpu-converter][string][empty]")
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32}));
  cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}));
  cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64}));
  gpu_disk_round_trip_test(std::make_unique<cudf::table>(std::move(cols)));
}

TEST_CASE("gpu disk round-trip string column with nulls", "[disk][gpu-converter][string][null]")
{
  std::vector<std::string> host_strings = {"alpha", "beta", "gamma", "delta"};
  auto const num_strings                = static_cast<cudf::size_type>(host_strings.size());

  std::vector<int32_t> host_offsets(static_cast<std::size_t>(num_strings) + 1);
  host_offsets[0] = 0;
  for (std::size_t i = 0; i < host_strings.size(); ++i) {
    host_offsets[i + 1] = host_offsets[i] + static_cast<int32_t>(host_strings[i].size());
  }

  std::vector<char> host_chars;
  for (const auto& s : host_strings) {
    host_chars.insert(host_chars.end(), s.begin(), s.end());
  }

  rmm::device_buffer dev_chars(host_chars.data(), host_chars.size(), shared_stream());
  auto offsets_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                               num_strings + 1,
                                               cudf::mask_state::UNALLOCATED,
                                               shared_stream());
  CUCASCADE_CUDA_TRY(cudaMemcpyAsync(offsets_col->mutable_view().data<int32_t>(),
                                     host_offsets.data(),
                                     host_offsets.size() * sizeof(int32_t),
                                     cudaMemcpyHostToDevice,
                                     shared_stream().get()));

  // Create null mask: elements 1 and 3 are null
  auto null_mask_size = cudf::bitmask_allocation_size_bytes(num_strings);
  std::vector<cudf::bitmask_type> host_mask(null_mask_size / sizeof(cudf::bitmask_type),
                                            ~cudf::bitmask_type{0});
  host_mask[0] &= ~(cudf::bitmask_type{1} << 1);  // null at index 1
  host_mask[0] &= ~(cudf::bitmask_type{1} << 3);  // null at index 3
  auto dev_mask = test::make_null_mask_from_host(num_strings, host_mask, shared_stream());
  shared_stream().sync();

  auto str_col = cudf::make_strings_column(
    num_strings, std::move(offsets_col), std::move(dev_chars), 2, std::move(dev_mask));

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(str_col));
  gpu_disk_round_trip_test(std::make_unique<cudf::table>(std::move(cols)));
}

// =============================================================================
// LIST column round-trip tests
// =============================================================================

TEST_CASE("gpu disk round-trip list column", "[disk][gpu-converter][list]")
{
  // LIST<INT32> column

  // Create offsets: [0, 2, 5, 5, 8] -> 4 lists with lengths [2, 3, 0, 3]
  std::vector<int32_t> host_offsets = {0, 2, 5, 5, 8};
  auto const num_lists              = static_cast<cudf::size_type>(host_offsets.size() - 1);

  auto offsets_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                               num_lists + 1,
                                               cudf::mask_state::UNALLOCATED,
                                               shared_stream());
  CUCASCADE_CUDA_TRY(cudaMemcpyAsync(offsets_col->mutable_view().data<int32_t>(),
                                     host_offsets.data(),
                                     host_offsets.size() * sizeof(int32_t),
                                     cudaMemcpyHostToDevice,
                                     shared_stream().get()));

  // Create values child: 8 INT32 values
  auto values_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 8, cudf::mask_state::UNALLOCATED, shared_stream());

  shared_stream().sync();

  auto list_col = cudf::make_lists_column(num_lists,
                                          std::move(offsets_col),
                                          std::move(values_col),
                                          0,
                                          cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(list_col));
  gpu_disk_round_trip_test(std::make_unique<cudf::table>(std::move(cols)));
}

TEST_CASE("gpu disk round-trip nested list column", "[disk][gpu-converter][list][nested]")
{
  // LIST<LIST<INT32>> column

  // Inner lists: offsets [0, 2, 3] -> 2 inner lists
  std::vector<int32_t> inner_offsets = {0, 2, 3};
  auto inner_offsets_col             = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 3, cudf::mask_state::UNALLOCATED, shared_stream());
  CUCASCADE_CUDA_TRY(cudaMemcpyAsync(inner_offsets_col->mutable_view().data<int32_t>(),
                                     inner_offsets.data(),
                                     inner_offsets.size() * sizeof(int32_t),
                                     cudaMemcpyHostToDevice,
                                     shared_stream().get()));

  auto inner_values = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 3, cudf::mask_state::UNALLOCATED, shared_stream());

  auto inner_list =
    cudf::make_lists_column(2,
                            std::move(inner_offsets_col),
                            std::move(inner_values),
                            0,
                            cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  // Outer lists: offsets [0, 1, 2] -> 2 outer lists, each containing 1 inner list
  std::vector<int32_t> outer_offsets = {0, 1, 2};
  auto outer_offsets_col             = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 3, cudf::mask_state::UNALLOCATED, shared_stream());
  CUCASCADE_CUDA_TRY(cudaMemcpyAsync(outer_offsets_col->mutable_view().data<int32_t>(),
                                     outer_offsets.data(),
                                     outer_offsets.size() * sizeof(int32_t),
                                     cudaMemcpyHostToDevice,
                                     shared_stream().get()));

  shared_stream().sync();

  auto outer_list =
    cudf::make_lists_column(2,
                            std::move(outer_offsets_col),
                            std::move(inner_list),
                            0,
                            cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(outer_list));
  gpu_disk_round_trip_test(std::make_unique<cudf::table>(std::move(cols)));
}

TEST_CASE("gpu disk round-trip nullable list column", "[disk][gpu-converter][list][nullable]")
{
  // LIST<INT32> with a null list. Offsets [0, 2, 5, 5, 8] -> 4 lists, lengths [2, 3, 0, 3].
  // Mark list index 2 (the empty [5,5) range) as null so the null row stays empty and the
  // original column is well-formed. Exercises the null_count > 0 reconstruction path.
  std::vector<int32_t> host_offsets = {0, 2, 5, 5, 8};
  auto const num_lists              = static_cast<cudf::size_type>(host_offsets.size() - 1);

  auto offsets_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                               num_lists + 1,
                                               cudf::mask_state::UNALLOCATED,
                                               shared_stream());
  CUCASCADE_CUDA_TRY(cudaMemcpyAsync(offsets_col->mutable_view().data<int32_t>(),
                                     host_offsets.data(),
                                     host_offsets.size() * sizeof(int32_t),
                                     cudaMemcpyHostToDevice,
                                     shared_stream().get()));

  auto values_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 8, cudf::mask_state::UNALLOCATED, shared_stream());

  // Null mask: all lists valid except index 2.
  auto null_mask = cudf::create_null_mask(num_lists, cudf::mask_state::ALL_VALID, shared_stream());
  cudf::set_null_mask(
    reinterpret_cast<cudf::bitmask_type*>(null_mask.data()), 2, 3, false, shared_stream());

  shared_stream().sync();

  auto list_col = cudf::make_lists_column(
    num_lists, std::move(offsets_col), std::move(values_col), 1, std::move(null_mask));

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(list_col));
  gpu_disk_round_trip_test(std::make_unique<cudf::table>(std::move(cols)));
}

TEST_CASE("gpu disk round-trip nullable nested list column",
          "[disk][gpu-converter][list][nested][nullable]")
{
  // LIST<LIST<INT32>> with a null outer list. Inner: offsets [0, 2, 3] -> 2 inner lists
  // (lengths 2, 1) over 3 values. Outer: offsets [0, 2, 2] -> outer list 0 holds both inner
  // lists, outer list 1 is the empty [2,2) range and is marked null. Exercises reconstruction
  // recursion through a null mask.
  std::vector<int32_t> inner_offsets = {0, 2, 3};
  auto inner_offsets_col             = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 3, cudf::mask_state::UNALLOCATED, shared_stream());
  CUCASCADE_CUDA_TRY(cudaMemcpyAsync(inner_offsets_col->mutable_view().data<int32_t>(),
                                     inner_offsets.data(),
                                     inner_offsets.size() * sizeof(int32_t),
                                     cudaMemcpyHostToDevice,
                                     shared_stream().get()));

  auto inner_values = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 3, cudf::mask_state::UNALLOCATED, shared_stream());

  auto inner_list =
    cudf::make_lists_column(2,
                            std::move(inner_offsets_col),
                            std::move(inner_values),
                            0,
                            cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  // Outer lists: offsets [0, 2, 2] -> list 0 holds both inner lists, list 1 is empty (and null).
  std::vector<int32_t> outer_offsets = {0, 2, 2};
  auto outer_offsets_col             = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 3, cudf::mask_state::UNALLOCATED, shared_stream());
  CUCASCADE_CUDA_TRY(cudaMemcpyAsync(outer_offsets_col->mutable_view().data<int32_t>(),
                                     outer_offsets.data(),
                                     outer_offsets.size() * sizeof(int32_t),
                                     cudaMemcpyHostToDevice,
                                     shared_stream().get()));

  // Outer null mask: outer list index 1 is null.
  auto outer_null_mask = cudf::create_null_mask(2, cudf::mask_state::ALL_VALID, shared_stream());
  cudf::set_null_mask(
    reinterpret_cast<cudf::bitmask_type*>(outer_null_mask.data()), 1, 2, false, shared_stream());

  shared_stream().sync();

  auto outer_list = cudf::make_lists_column(
    2, std::move(outer_offsets_col), std::move(inner_list), 1, std::move(outer_null_mask));

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(outer_list));
  gpu_disk_round_trip_test(std::make_unique<cudf::table>(std::move(cols)));
}

// =============================================================================
// STRUCT column round-trip tests
// =============================================================================

TEST_CASE("gpu disk round-trip struct column", "[disk][gpu-converter][struct]")
{
  // STRUCT<INT32, FLOAT64>

  auto const num_rows = cudf::size_type{50};

  auto int_child   = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                             num_rows,
                                             cudf::mask_state::UNALLOCATED,
                                             shared_stream());
  auto float_child = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64},
                                               num_rows,
                                               cudf::mask_state::UNALLOCATED,
                                               shared_stream());

  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(std::move(int_child));
  children.push_back(std::move(float_child));

  auto struct_col = cudf::make_structs_column(
    num_rows, std::move(children), 0, cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(struct_col));
  gpu_disk_round_trip_test(std::make_unique<cudf::table>(std::move(cols)));
}

TEST_CASE("gpu disk round-trip nested struct column", "[disk][gpu-converter][struct][nested]")
{
  // STRUCT<STRUCT<INT32>>

  auto const num_rows = cudf::size_type{30};

  auto inner_child = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                               num_rows,
                                               cudf::mask_state::UNALLOCATED,
                                               shared_stream());

  std::vector<std::unique_ptr<cudf::column>> inner_children;
  inner_children.push_back(std::move(inner_child));
  auto inner_struct =
    cudf::make_structs_column(num_rows,
                              std::move(inner_children),
                              0,
                              cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  std::vector<std::unique_ptr<cudf::column>> outer_children;
  outer_children.push_back(std::move(inner_struct));
  auto outer_struct =
    cudf::make_structs_column(num_rows,
                              std::move(outer_children),
                              0,
                              cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(outer_struct));
  gpu_disk_round_trip_test(std::make_unique<cudf::table>(std::move(cols)));
}

TEST_CASE("gpu disk round-trip struct with null mask", "[disk][gpu-converter][struct][null]")
{
  // STRUCT<INT32> with struct-level null mask

  auto const num_rows = cudf::size_type{40};

  auto child = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                         num_rows,
                                         cudf::mask_state::UNALLOCATED,
                                         shared_stream());

  // Create struct-level null mask (every 3rd element is null)
  auto null_mask_size = cudf::bitmask_allocation_size_bytes(num_rows);
  std::vector<cudf::bitmask_type> host_mask(null_mask_size / sizeof(cudf::bitmask_type),
                                            ~cudf::bitmask_type{0});
  cudf::size_type null_count   = 0;
  constexpr auto bits_per_word = sizeof(cudf::bitmask_type) * 8;
  for (int i = 0; i < num_rows; i += 3) {
    auto word_idx = static_cast<std::size_t>(i) / bits_per_word;
    auto bit_idx  = static_cast<std::size_t>(i) % bits_per_word;
    host_mask[word_idx] &= ~(cudf::bitmask_type{1} << bit_idx);
    null_count++;
  }
  auto dev_mask = test::make_null_mask_from_host(num_rows, host_mask, shared_stream());

  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(std::move(child));

  shared_stream().sync();
  auto struct_col =
    cudf::make_structs_column(num_rows, std::move(children), null_count, std::move(dev_mask));

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(struct_col));
  gpu_disk_round_trip_test(std::make_unique<cudf::table>(std::move(cols)));
}

// =============================================================================
// DICTIONARY column round-trip tests
// =============================================================================

TEST_CASE("gpu disk round-trip dictionary column", "[disk][gpu-converter][dictionary]")
{
  // DICTIONARY32 with INT32 keys

  auto const num_rows = cudf::size_type{50};
  auto const num_keys = cudf::size_type{5};

  // Keys: 5 unique INT32 values
  auto keys = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                        num_keys,
                                        cudf::mask_state::UNALLOCATED,
                                        shared_stream());

  // Indices: INT32 values in [0, num_keys)
  auto indices = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                           num_rows,
                                           cudf::mask_state::UNALLOCATED,
                                           shared_stream());

  shared_stream().sync();

  auto dict_col = cudf::make_dictionary_column(std::move(keys), std::move(indices));

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(dict_col));
  gpu_disk_round_trip_test(std::make_unique<cudf::table>(std::move(cols)));
}

// =============================================================================
// Sliced column round-trip tests
// =============================================================================

TEST_CASE("gpu disk round-trip sliced column", "[disk][gpu-converter][sliced]")
{
  // Create a column, slice it, compact it, then round-trip

  auto const total_rows  = cudf::size_type{200};
  auto const slice_start = cudf::size_type{50};
  auto const slice_end   = cudf::size_type{150};

  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       total_rows,
                                       cudf::mask_state::ALL_VALID,
                                       shared_stream());

  std::vector<std::unique_ptr<cudf::column>> table_cols;
  table_cols.push_back(std::move(col));
  auto table = std::make_unique<cudf::table>(std::move(table_cols));

  // Slice to get a view with non-zero offset
  auto sliced_views = cudf::slice(table->view(), {slice_start, slice_end});
  REQUIRE(sliced_views.size() == 1);
  REQUIRE(sliced_views[0].num_rows() == slice_end - slice_start);

  // Compact the slice into a new table (removes offset)
  auto compacted = std::make_unique<cudf::table>(sliced_views[0], shared_stream());
  REQUIRE(compacted->num_rows() == slice_end - slice_start);

  gpu_disk_round_trip_test(std::move(compacted));
}

TEST_CASE("gpu disk round-trip sliced column with nulls", "[disk][gpu-converter][sliced][null]")
{
  auto const total_rows = cudf::size_type{200};

  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       total_rows,
                                       cudf::mask_state::ALL_VALID,
                                       shared_stream());

  // Set some nulls
  auto null_mask_size = cudf::bitmask_allocation_size_bytes(total_rows);
  std::vector<cudf::bitmask_type> host_mask(null_mask_size / sizeof(cudf::bitmask_type),
                                            ~cudf::bitmask_type{0});
  cudf::size_type null_count   = 0;
  constexpr auto bits_per_word = sizeof(cudf::bitmask_type) * 8;
  for (int i = 0; i < total_rows; i += 5) {
    auto word_idx = static_cast<std::size_t>(i) / bits_per_word;
    auto bit_idx  = static_cast<std::size_t>(i) % bits_per_word;
    host_mask[word_idx] &= ~(cudf::bitmask_type{1} << bit_idx);
    null_count++;
  }
  auto dev_mask = test::make_null_mask_from_host(total_rows, host_mask, shared_stream());
  col->set_null_mask(std::move(dev_mask), null_count);

  std::vector<std::unique_ptr<cudf::column>> table_cols;
  table_cols.push_back(std::move(col));
  auto table = std::make_unique<cudf::table>(std::move(table_cols));

  // Slice middle portion
  auto sliced_views = cudf::slice(table->view(), {cudf::size_type{50}, cudf::size_type{150}});
  shared_stream().sync();
  auto compacted = std::make_unique<cudf::table>(sliced_views[0], shared_stream());

  gpu_disk_round_trip_test(std::move(compacted));
}

// Regression: the serializers copy buffers from their start, so a view-backed slice (non-zero
// offset: null mask bits, string / list offsets) used to round-trip as the leading rows of the
// table it slices, with unshifted validity. Both buffer-serializing routes (host_data and DISK)
// compact such a table first.
TEST_CASE("view-backed sliced nested nullable tables round-trip through host_data and disk",
          "[disk][gpu-converter][sliced]")
{
  auto& gpu_space = shared_gpu_space();
  auto host_space = test::make_mock_memory_space(memory::Tier::HOST, 0);
  representation_converter_registry registry;
  register_builtin_converters(registry);
  auto source = test::make_sliced_source(*gpu_space, shared_stream());

  auto host = registry.convert<host_data_representation>(
    *source.representation, host_space.get(), shared_stream());
  auto from_host =
    registry.convert<gpu_table_representation>(*host, gpu_space.get(), shared_stream());
  test::expect_cudf_tables_equal_on_stream(
    source.expected->view(), from_host->get_table_view(), shared_stream());

  for (auto const backend : test::disk_test_backends) {
    auto* disk_space = test::shared_disk_space_for(backend);
    if (disk_space == nullptr || !test::disk_space_can_write(*disk_space, /*device=*/true)) {
      continue;
    }
    CAPTURE(test::to_string(backend));
    auto disk = registry.convert<disk_data_representation>(
      *source.representation, disk_space, shared_stream());
    auto back = registry.convert<gpu_table_representation>(*disk, gpu_space.get(), shared_stream());
    test::expect_cudf_tables_equal_on_stream(
      source.expected->view(), back->get_table_view(), shared_stream());
  }
}

// Regression: a prefix slice has offset 0, but its children keep the extent of the table it was
// cut from (dictionary indices, string / list offsets, struct fields); it used to be serialized
// with those extents and restored with the wrong row count.
TEST_CASE("view-backed prefix slices round-trip through host_data and disk",
          "[disk][gpu-converter][sliced]")
{
  auto& gpu_space = shared_gpu_space();
  auto host_space = test::make_mock_memory_space(memory::Tier::HOST, 0);
  representation_converter_registry registry;
  register_builtin_converters(registry);
  auto source = test::make_prefix_sliced_source(*gpu_space, shared_stream());

  auto host = registry.convert<host_data_representation>(
    *source.representation, host_space.get(), shared_stream());
  auto from_host =
    registry.convert<gpu_table_representation>(*host, gpu_space.get(), shared_stream());
  REQUIRE(from_host->get_table_view().num_rows() == 10);
  test::expect_cudf_tables_equal_on_stream(
    source.expected->view(), from_host->get_table_view(), shared_stream());

  for (auto const backend : test::disk_test_backends) {
    auto* disk_space = test::shared_disk_space_for(backend);
    if (disk_space == nullptr || !test::disk_space_can_write(*disk_space, /*device=*/true)) {
      continue;
    }
    CAPTURE(test::to_string(backend));
    auto disk = registry.convert<disk_data_representation>(
      *source.representation, disk_space, shared_stream());
    auto back = registry.convert<gpu_table_representation>(*disk, gpu_space.get(), shared_stream());
    REQUIRE(back->get_table_view().num_rows() == 10);
    test::expect_cudf_tables_equal_on_stream(
      source.expected->view(), back->get_table_view(), shared_stream());
  }
}

// Regression: when the compaction of a sliced source failed part-way, the column copies already
// enqueued on the conversion stream could still be reading the source after the error reached the
// caller. The stream is now drained before the exception propagates.
TEST_CASE("a failed compaction drains the conversion stream before throwing",
          "[disk][gpu-converter][sliced]")
{
  auto gpu_space = test::make_refusing_gpu_space(std::size_t{3} << 20);
  representation_converter_registry registry;
  register_builtin_converters(registry);
  auto host_space  = test::make_mock_memory_space(memory::Tier::HOST, 0);
  auto* disk_space = test::shared_disk_space_for(disk_io_backend::uring);
  bool const disk  = disk_space != nullptr && test::disk_space_can_write(*disk_space, true);

  for (int route = 0; route < (disk ? 2 : 1); ++route) {
    CAPTURE(route);
    // The stream sleeps first, so the INT32 column's copy is still queued when the INT64
    // column's allocation is refused.
    auto source = test::make_compaction_failure_source(*gpu_space, std::chrono::milliseconds{300});
    ::cuda::stream_ref const stream{source.stream};
    if (route == 0) {
      CHECK_THROWS(registry.convert<host_data_representation>(
        *source.representation, host_space.get(), stream));
    } else {
      CHECK_THROWS(
        registry.convert<disk_data_representation>(*source.representation, disk_space, stream));
    }
    CHECK(cudaStreamQuery(source.stream) == cudaSuccess);
  }
}

// Regression: GPU -> DISK waited for the writer on the default stream of the CURRENT device and
// then made the source device current for the write, whose default stream had not waited.
TEST_CASE("gpu disk spill of a table on another GPU waits for its writer on the default stream",
          "[disk][gpu-converter][multi_gpu]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires two CUDA devices"); }
  auto gpu1 = test::make_mock_memory_space(memory::Tier::GPU, 1);
  representation_converter_registry registry;
  register_builtin_converters(registry);
  auto* disk_space = test::shared_disk_space_for(disk_io_backend::uring);
  if (disk_space == nullptr || !test::disk_space_can_write(*disk_space, /*device=*/true)) {
    SKIP("the uring disk backend cannot run in this process");
  }

  // Also as a slice, which is compacted first on the source device.
  auto const slice_begin = GENERATE(0, 1);
  CAPTURE(slice_begin);
  auto source =
    test::make_late_written_source(*gpu1, 1 << 20, std::chrono::milliseconds{300}, slice_begin);
  rmm::cuda_set_device_raii current{rmm::cuda_device_id{0}};
  ::cuda::stream_ref const default_stream{cudaStream_t{nullptr}};
  auto disk =
    registry.convert<disk_data_representation>(*source.representation, disk_space, default_stream);
  auto back = registry.convert<gpu_table_representation>(*disk, gpu1.get(), default_stream);
  CHECK(test::all_bytes_are_5a(back->get_table_view(), 1));
}

// Regression: GPU -> host_data made the source device current only around the compaction, so the
// writer wait, the planning (which launches work on the stream) and the D2H copies ran with the
// caller's current device: a table on GPU 1 converted from GPU 0 on one of GPU 1's streams failed.
// (This converter needs a non-default stream: cudaMemcpyBatchAsync.)
TEST_CASE("gpu host_data conversion of a table on another GPU runs on the source device",
          "[disk][gpu-converter][multi_gpu]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires two CUDA devices"); }
  auto gpu1       = test::make_mock_memory_space(memory::Tier::GPU, 1);
  auto host_space = test::make_mock_memory_space(memory::Tier::HOST, 0);
  representation_converter_registry registry;
  register_builtin_converters(registry);
  auto const prefix = GENERATE(false, true);
  CAPTURE(prefix);

  std::optional<rmm::cuda_stream> stream1;
  test::sliced_source source;
  {
    rmm::cuda_set_device_raii on_gpu1{rmm::cuda_device_id{1}};
    stream1.emplace();
    source = prefix ? test::make_prefix_sliced_source(*gpu1, ::cuda::stream_ref{stream1->value()})
                    : test::make_sliced_source(*gpu1, ::cuda::stream_ref{stream1->value()});
  }
  std::unique_ptr<gpu_table_representation> back;
  {
    rmm::cuda_set_device_raii on_gpu0{rmm::cuda_device_id{0}};
    auto host = registry.convert<host_data_representation>(
      *source.representation, host_space.get(), ::cuda::stream_ref{stream1->value()});
    back = registry.convert<gpu_table_representation>(
      *host, gpu1.get(), ::cuda::stream_ref{stream1->value()});
  }
  rmm::cuda_set_device_raii on_gpu1{rmm::cuda_device_id{1}};
  test::expect_cudf_tables_equal_on_stream(
    source.expected->view(), back->get_table_view(), ::cuda::stream_ref{stream1->value()});
  back.reset();
  source = {};
}

// Regression: GPU -> GPU copied a view-backed slice's buffers from their start, like the other
// buffer copiers did: a nested slice failed with cudaErrorInvalidValue, a prefix slice with cuDF's
// "Invalid offsets column size", and a fixed-width nullable slice silently came back as the
// table's leading rows with unshifted validity.
TEST_CASE("gpu to gpu conversion of view-backed slices copies the sliced rows",
          "[gpu-converter][sliced][multi_gpu]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires two CUDA devices"); }
  // Registers the pinned pool, so the production host-staged path runs where peer DMA is not
  // verified.
  auto host_space = test::make_mock_memory_space(memory::Tier::HOST, 0);
  std::shared_ptr<memory::memory_space> gpus[2] = {
    test::make_mock_memory_space(memory::Tier::GPU, 0),
    test::make_mock_memory_space(memory::Tier::GPU, 1)};
  representation_converter_registry registry;
  register_builtin_converters(registry);
  auto const kind = GENERATE(0, 1, 2);  // nested slice, prefix slice, fixed-width slice
  auto const from = GENERATE(0, 1);
  int const to    = 1 - from;
  CAPTURE(kind, from);

  rmm::cuda_set_device_raii on_source{rmm::cuda_device_id{from}};
  rmm::cuda_stream source_stream;
  ::cuda::stream_ref const stream{source_stream.value()};
  auto source     = kind == 0   ? test::make_sliced_source(*gpus[from], stream)
                    : kind == 1 ? test::make_prefix_sliced_source(*gpus[from], stream)
                                : test::make_fixed_width_sliced_source(*gpus[from], stream);
  auto const rows = source.expected->num_rows();
  auto there =
    registry.convert<gpu_table_representation>(*source.representation, gpus[to].get(), stream);
  REQUIRE(there->get_device_id() == to);
  REQUIRE(there->get_table_view().num_rows() == rows);
  // Back on the source device: the second leg copies an owned, compact table.
  auto back = registry.convert<gpu_table_representation>(*there, gpus[from].get(), stream);
  REQUIRE(back->get_device_id() == from);
  test::expect_cudf_tables_equal_on_stream(source.expected->view(), back->get_table_view(), stream);
}

// Regression: a failed GPU -> GPU conversion returned while a peer copy of an earlier column was
// still queued on the target stream (behind the source's writer), reading the source. Only a
// regression check where peer DMA is verified: on a host-staged pair each staged leg synchronizes,
// so it passes either way.
TEST_CASE("a failed gpu to gpu conversion waits for its copies before throwing",
          "[gpu-converter][multi_gpu]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires two CUDA devices"); }
  auto host_space = test::make_mock_memory_space(memory::Tier::HOST, 0);
  auto gpu0       = test::make_mock_memory_space(memory::Tier::GPU, 0);
  // Refuses the 4 MiB INT64 column's buffer, after the 2 MiB INT32 column's copy was enqueued.
  auto target = test::make_refusing_gpu_space(std::size_t{3} << 20, 1);
  representation_converter_registry registry;
  register_builtin_converters(registry);

  auto source = test::make_late_written_pair_source(*gpu0, std::chrono::milliseconds{300});
  rmm::cuda_set_device_raii on_gpu0{rmm::cuda_device_id{0}};
  rmm::cuda_stream idle;
  CHECK_THROWS(registry.convert<gpu_table_representation>(
    *source.representation, target.get(), ::cuda::stream_ref{idle.value()}));
  // The drained target stream had waited on the writer event, so the producer has finished.
  CHECK(cudaStreamQuery(source.producer) == cudaSuccess);
}

// A sliced GPU -> GPU source whose compaction on the source device is refused after the INT32
// column's copy was enqueued behind the writer: the compaction drains before the error leaves.
TEST_CASE("a gpu to gpu conversion whose compaction fails drains before throwing",
          "[gpu-converter][sliced][multi_gpu]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires two CUDA devices"); }
  auto host_space = test::make_mock_memory_space(memory::Tier::HOST, 0);
  auto gpu0       = test::make_refusing_gpu_space(std::size_t{3} << 20, 0);
  auto gpu1       = test::make_mock_memory_space(memory::Tier::GPU, 1);
  representation_converter_registry registry;
  register_builtin_converters(registry);

  auto source =
    test::make_late_written_pair_source(*gpu0, std::chrono::milliseconds{300}, /*slice_begin=*/1);
  rmm::cuda_set_device_raii on_gpu0{rmm::cuda_device_id{0}};
  rmm::cuda_stream idle;
  CHECK_THROWS(registry.convert<gpu_table_representation>(
    *source.representation, gpu1.get(), ::cuda::stream_ref{idle.value()}));
  CHECK(cudaStreamQuery(source.producer) == cudaSuccess);
}

// =============================================================================
// io context selection tests
// =============================================================================

TEST_CASE("gpu disk round-trip with an explicit io context of each backend",
          "[disk][gpu-converter][backend]")
{
  auto const backend = GENERATE(disk_io_backend::uring, disk_io_backend::cufile);
  CAPTURE(test::to_string(backend));
  auto gpu_space  = test::make_mock_memory_space(memory::Tier::GPU, 0);
  auto disk_space = test::make_disk_space(backend);
  if (disk_space == nullptr) {
    WARN("the " << test::to_string(backend) << " backend cannot run in this process");
    return;
  }
  auto context = disk_space->get_io_context();
  REQUIRE(context != nullptr);
  if (!context->supports_device_write()) {
    WARN("the " << test::to_string(backend) << " context cannot write device memory yet");
    return;
  }

  representation_converter_registry registry;
  register_builtin_converters(registry);

  // Test with multiple numeric types
  auto type_id = GENERATE(cudf::type_id::INT32, cudf::type_id::INT64, cudf::type_id::FLOAT64);
  auto table   = make_typed_table(type_id, 1000);
  auto gpu_rep = std::make_unique<gpu_table_representation>(
    std::move(table), *gpu_space, ::cuda::stream_ref{cudaStream_t{cudaStreamDefault}});

  auto disk_rep =
    registry.convert<disk_data_representation>(*gpu_rep, disk_space.get(), shared_stream());

  REQUIRE(disk_rep->get_size_in_bytes() > 0);
  REQUIRE(disk_rep->get_uncompressed_data_size_in_bytes() == disk_rep->get_size_in_bytes());
  // The space kept the context it was configured with.
  REQUIRE(disk_space->get_io_context() == context);

  auto gpu_rep2 =
    registry.convert<gpu_table_representation>(*disk_rep, gpu_space.get(), shared_stream());

  test::expect_cudf_tables_equal_on_stream(
    gpu_rep->get_table_view(), gpu_rep2->get_table_view(), shared_stream());
}

TEST_CASE("disk_data_representation get_uncompressed_data_size_in_bytes", "[disk][representation]")
{
  auto gpu_space  = test::make_mock_memory_space(memory::Tier::GPU, 0);
  auto disk_space = test::make_mock_memory_space(memory::Tier::DISK, 0);

  representation_converter_registry registry;
  register_builtin_converters(registry);

  auto table   = make_typed_table(cudf::type_id::INT64, 1000);
  auto gpu_rep = std::make_unique<gpu_table_representation>(
    std::move(table), *gpu_space, ::cuda::stream_ref{cudaStream_t{cudaStreamDefault}});

  auto disk_rep =
    registry.convert<disk_data_representation>(*gpu_rep, disk_space.get(), shared_stream());

  REQUIRE(disk_rep->get_size_in_bytes() > 0);
  REQUIRE(disk_rep->get_uncompressed_data_size_in_bytes() == disk_rep->get_size_in_bytes());
}
