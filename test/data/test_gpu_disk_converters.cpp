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

#include "utils/cudf_test_utils.hpp"
#include "utils/mock_test_utils.hpp"

#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch.hpp>

#include <cstdint>
#include <memory>
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
auto& shared_disk_space()
{
  static auto s = test::make_mock_memory_space(memory::Tier::DISK, 0);
  return s;
}
rmm::cuda_stream_view shared_stream()
{
  static rmm::cuda_stream s;
  return s.view();
}

/// Round-trip test helper: GPU -> disk -> GPU (direct 3-step), compare tables.
void gpu_disk_round_trip_test(std::unique_ptr<cudf::table> original_table)
{
  auto stream      = shared_stream();
  auto& gpu_space  = shared_gpu_space();
  auto& disk_space = shared_disk_space();

  representation_converter_registry registry;
  register_builtin_converters(registry);

  // Create GPU representation from the original table
  auto gpu_rep = std::make_unique<gpu_table_representation>(std::move(original_table), *gpu_space);

  // GPU -> disk (direct via write/write_batch)
  auto disk_rep =
    registry.convert<disk_data_representation>(*gpu_rep, disk_space.get(), shared_stream());

  // disk -> GPU (direct via read)
  auto gpu_rep2 =
    registry.convert<gpu_table_representation>(*disk_rep, gpu_space.get(), shared_stream());

  // Compare original (from gpu_rep) with round-tripped
  test::expect_cudf_tables_equal_on_stream(
    gpu_rep->get_table(), gpu_rep2->get_table(), shared_stream());
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
  // Create a column with alternating nulls

  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 100, cudf::mask_state::ALL_VALID, shared_stream());

  // Set every other bit to null using cudf utilities
  auto const num_rows = 100;
  std::vector<uint8_t> host_mask(cudf::bitmask_allocation_size_bytes(num_rows), 0xFF);
  // Clear every other bit (set elements 0, 2, 4, ... to null)
  for (int i = 0; i < num_rows; i += 2) {
    auto byte_idx = static_cast<std::size_t>(i / 8);
    auto bit_idx  = static_cast<unsigned>(i % 8);
    host_mask[byte_idx] &= static_cast<uint8_t>(~(1u << bit_idx));
  }
  rmm::device_buffer dev_mask(host_mask.data(), host_mask.size(), shared_stream());
  col->set_null_mask(std::move(dev_mask), 50);  // 50 nulls out of 100

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
                                     shared_stream().value()));
  shared_stream().synchronize();

  auto str_col = cudf::make_strings_column(
    num_strings, std::move(offsets_col), std::move(dev_chars), 0, rmm::device_buffer{});

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(str_col));
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
                                     shared_stream().value()));

  // Create null mask: elements 1 and 3 are null
  auto null_mask_size = cudf::bitmask_allocation_size_bytes(num_strings);
  std::vector<uint8_t> host_mask(null_mask_size, 0xFF);
  host_mask[0] &= static_cast<uint8_t>(~(1u << 1));  // null at index 1
  host_mask[0] &= static_cast<uint8_t>(~(1u << 3));  // null at index 3
  rmm::device_buffer dev_mask(host_mask.data(), host_mask.size(), shared_stream());
  shared_stream().synchronize();

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
                                     shared_stream().value()));

  // Create values child: 8 INT32 values
  auto values_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 8, cudf::mask_state::UNALLOCATED, shared_stream());

  shared_stream().synchronize();

  auto list_col = cudf::make_lists_column(
    num_lists, std::move(offsets_col), std::move(values_col), 0, rmm::device_buffer{});

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
                                     shared_stream().value()));

  auto inner_values = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 3, cudf::mask_state::UNALLOCATED, shared_stream());

  auto inner_list = cudf::make_lists_column(
    2, std::move(inner_offsets_col), std::move(inner_values), 0, rmm::device_buffer{});

  // Outer lists: offsets [0, 1, 2] -> 2 outer lists, each containing 1 inner list
  std::vector<int32_t> outer_offsets = {0, 1, 2};
  auto outer_offsets_col             = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 3, cudf::mask_state::UNALLOCATED, shared_stream());
  CUCASCADE_CUDA_TRY(cudaMemcpyAsync(outer_offsets_col->mutable_view().data<int32_t>(),
                                     outer_offsets.data(),
                                     outer_offsets.size() * sizeof(int32_t),
                                     cudaMemcpyHostToDevice,
                                     shared_stream().value()));

  shared_stream().synchronize();

  auto outer_list = cudf::make_lists_column(
    2, std::move(outer_offsets_col), std::move(inner_list), 0, rmm::device_buffer{});

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

  auto struct_col =
    cudf::make_structs_column(num_rows, std::move(children), 0, rmm::device_buffer{});

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
    cudf::make_structs_column(num_rows, std::move(inner_children), 0, rmm::device_buffer{});

  std::vector<std::unique_ptr<cudf::column>> outer_children;
  outer_children.push_back(std::move(inner_struct));
  auto outer_struct =
    cudf::make_structs_column(num_rows, std::move(outer_children), 0, rmm::device_buffer{});

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
  std::vector<uint8_t> host_mask(null_mask_size, 0xFF);
  cudf::size_type null_count = 0;
  for (int i = 0; i < num_rows; i += 3) {
    auto byte_idx = static_cast<std::size_t>(i / 8);
    auto bit_idx  = static_cast<unsigned>(i % 8);
    host_mask[byte_idx] &= static_cast<uint8_t>(~(1u << bit_idx));
    null_count++;
  }
  rmm::device_buffer dev_mask(host_mask.data(), host_mask.size(), shared_stream());

  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(std::move(child));

  shared_stream().synchronize();
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

  shared_stream().synchronize();

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
  std::vector<uint8_t> host_mask(null_mask_size, 0xFF);
  cudf::size_type null_count = 0;
  for (int i = 0; i < total_rows; i += 5) {
    host_mask[static_cast<std::size_t>(i / 8)] &= static_cast<uint8_t>(~(1u << (i % 8)));
    null_count++;
  }
  rmm::device_buffer dev_mask(host_mask.data(), host_mask.size(), shared_stream());
  col->set_null_mask(std::move(dev_mask), null_count);

  std::vector<std::unique_ptr<cudf::column>> table_cols;
  table_cols.push_back(std::move(col));
  auto table = std::make_unique<cudf::table>(std::move(table_cols));

  // Slice middle portion
  auto sliced_views = cudf::slice(table->view(), {cudf::size_type{50}, cudf::size_type{150}});
  shared_stream().synchronize();
  auto compacted = std::make_unique<cudf::table>(sliced_views[0], shared_stream());

  gpu_disk_round_trip_test(std::move(compacted));
}

// =============================================================================
// I/O backend selection tests (TEST-10)
// =============================================================================

TEST_CASE("gpu disk round-trip with explicit pipeline backend",
          "[disk][gpu-converter][backend][pipeline]")
{
  auto gpu_space  = test::make_mock_memory_space(memory::Tier::GPU, 0);
  auto disk_space = test::make_mock_memory_space(memory::Tier::DISK, 0);

  representation_converter_registry registry;
  register_builtin_converters(registry);

  // Simple INT32 column, 100 rows
  auto table   = make_typed_table(cudf::type_id::INT32, 100);
  auto gpu_rep = std::make_unique<gpu_table_representation>(std::move(table), *gpu_space);

  auto disk_rep =
    registry.convert<disk_data_representation>(*gpu_rep, disk_space.get(), shared_stream());
  auto gpu_rep2 =
    registry.convert<gpu_table_representation>(*disk_rep, gpu_space.get(), shared_stream());

  test::expect_cudf_tables_equal_on_stream(
    gpu_rep->get_table(), gpu_rep2->get_table(), shared_stream());
}

TEST_CASE("gpu disk round-trip pipeline with multiple types",
          "[disk][gpu-converter][backend][pipeline]")
{
  auto gpu_space  = test::make_mock_memory_space(memory::Tier::GPU, 0);
  auto disk_space = test::make_mock_memory_space(memory::Tier::DISK, 0);

  representation_converter_registry registry;
  register_builtin_converters(registry);

  // Test with multiple numeric types
  auto type_id = GENERATE(cudf::type_id::INT32, cudf::type_id::INT64, cudf::type_id::FLOAT64);
  auto table   = make_typed_table(type_id, 1000);
  auto gpu_rep = std::make_unique<gpu_table_representation>(std::move(table), *gpu_space);

  auto disk_rep =
    registry.convert<disk_data_representation>(*gpu_rep, disk_space.get(), shared_stream());

  REQUIRE(disk_rep->get_size_in_bytes() > 0);
  REQUIRE(disk_rep->get_uncompressed_data_size_in_bytes() == disk_rep->get_size_in_bytes());

  auto gpu_rep2 =
    registry.convert<gpu_table_representation>(*disk_rep, gpu_space.get(), shared_stream());

  test::expect_cudf_tables_equal_on_stream(
    gpu_rep->get_table(), gpu_rep2->get_table(), shared_stream());
}

TEST_CASE("disk_data_representation get_uncompressed_data_size_in_bytes", "[disk][representation]")
{
  auto gpu_space  = test::make_mock_memory_space(memory::Tier::GPU, 0);
  auto disk_space = test::make_mock_memory_space(memory::Tier::DISK, 0);

  representation_converter_registry registry;
  register_builtin_converters(registry);

  auto table   = make_typed_table(cudf::type_id::INT64, 1000);
  auto gpu_rep = std::make_unique<gpu_table_representation>(std::move(table), *gpu_space);

  auto disk_rep =
    registry.convert<disk_data_representation>(*gpu_rep, disk_space.get(), shared_stream());

  REQUIRE(disk_rep->get_size_in_bytes() > 0);
  REQUIRE(disk_rep->get_uncompressed_data_size_in_bytes() == disk_rep->get_size_in_bytes());
}
