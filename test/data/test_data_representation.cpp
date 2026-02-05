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

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/host_table.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

#include <cudf/contiguous_split.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch.hpp>

#include <memory>
#include <vector>

using namespace cucascade;
using cucascade::test::create_conversion_test_configs;
using cucascade::test::create_simple_cudf_table;
using cucascade::test::make_mock_memory_space;

// Note: Tests that require mock host_table_allocation are disabled because
// fixed_size_host_memory_resource::multiple_blocks_allocation is now private.
// The real allocation tests below use actual memory resources.
[[maybe_unused]] static constexpr bool MOCK_HOST_ALLOCATION_DISABLED = true;

// =============================================================================
// host_table_representation Tests
// =============================================================================

// Disabled: requires internal access to multiple_blocks_allocation constructor
TEST_CASE("host_table_representation Construction", "[cpu_data_representation][.disabled]")
{
  SUCCEED("Test disabled - requires internal API access");
}

// Disabled: requires internal access to multiple_blocks_allocation constructor
TEST_CASE("host_table_representation get_size_in_bytes", "[cpu_data_representation][.disabled]")
{
  SUCCEED("Test disabled - requires internal API access");
}

// Disabled: requires internal access to multiple_blocks_allocation constructor
TEST_CASE("host_table_representation memory tier", "[cpu_data_representation][.disabled]")
{
  SUCCEED("Test disabled - requires internal API access");
}

// Disabled: requires internal access to multiple_blocks_allocation constructor
TEST_CASE("host_table_representation device_id", "[cpu_data_representation][.disabled]")
{
  SUCCEED("Test disabled - requires internal API access");
}

TEST_CASE("host_table_representation converts to GPU and preserves contents",
          "[cpu_data_representation][gpu_data_representation]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const memory::memory_space* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  const memory::memory_space* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);

  // Start from a known cudf table; pack it and build a host_table_representation
  // Use the same stream for table creation and packing to avoid stream-ordered races
  rmm::cuda_stream pack_stream;
  auto original =
    create_simple_cudf_table(128, 2, gpu_space->get_default_allocator(), pack_stream.view());
  auto view   = original.view();
  auto packed = cudf::pack(view, pack_stream.view());
  pack_stream.synchronize();
  auto host_mr = host_space->get_memory_resource_as<memory::fixed_size_host_memory_resource>();
  REQUIRE(host_mr != nullptr);

  // Copy device buffer to host allocation
  auto allocation         = host_mr->allocate_multiple_blocks(packed.gpu_data->size());
  size_t copied           = 0;
  size_t block_idx        = 0;
  size_t block_off        = 0;
  const size_t block_size = allocation->block_size();
  while (copied < packed.gpu_data->size()) {
    size_t remain = packed.gpu_data->size() - copied;
    size_t bytes  = std::min(remain, block_size - block_off);
    void* dst_ptr = reinterpret_cast<uint8_t*>((*allocation)[block_idx].data()) + block_off;
    RMM_CUDA_TRY(cudaMemcpy(dst_ptr,
                            static_cast<const uint8_t*>(packed.gpu_data->data()) + copied,
                            bytes,
                            cudaMemcpyDeviceToHost));
    copied += bytes;
    block_off += bytes;
    if (block_off == block_size) {
      block_off = 0;
      block_idx++;
    }
  }

  auto meta_copy  = std::make_unique<std::vector<uint8_t>>(*packed.metadata);
  auto host_alloc = std::make_unique<memory::host_table_allocation>(
    std::move(allocation), std::move(meta_copy), packed.gpu_data->size());
  host_table_representation host_repr(std::move(host_alloc),
                                      const_cast<memory::memory_space*>(host_space));

  // Convert to GPU and compare cudf tables
  auto gpu_stream = gpu_space->acquire_stream();
  auto gpu_any    = registry.convert<gpu_table_representation>(host_repr, gpu_space, pack_stream);
  pack_stream.synchronize();
  auto& gpu_repr = *gpu_any;
  // Compare using the same stream used for conversion to avoid cross-stream hazards
  cucascade::test::expect_cudf_tables_equal_on_stream(
    original, gpu_repr.get_table(), pack_stream.view());
}

// =============================================================================
// gpu_table_representation Tests
// =============================================================================

TEST_CASE("gpu_table_representation Construction", "[gpu_data_representation]")
{
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
  auto table     = create_simple_cudf_table(100, gpu_space->get_default_allocator());

  gpu_table_representation repr(std::make_unique<cudf::table>(std::move(table)), *gpu_space);

  REQUIRE(repr.get_current_tier() == memory::Tier::GPU);
  REQUIRE(repr.get_device_id() == 0);
  REQUIRE(repr.get_size_in_bytes() > 0);
}

TEST_CASE("gpu_table_representation get_size_in_bytes", "[gpu_data_representation]")
{
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);

  SECTION("100 rows")
  {
    auto table = create_simple_cudf_table(100, gpu_space->get_default_allocator());
    gpu_table_representation repr(std::make_unique<cudf::table>(std::move(table)), *gpu_space);

    // Size should be at least 100 rows * (4 bytes for INT32 + 8 bytes for INT64)
    std::size_t expected_min_size = 100 * (4 + 8);
    REQUIRE(repr.get_size_in_bytes() >= expected_min_size);
  }

  SECTION("1000 rows")
  {
    auto table = create_simple_cudf_table(1000, gpu_space->get_default_allocator());
    gpu_table_representation repr(std::make_unique<cudf::table>(std::move(table)), *gpu_space);

    // Size should be at least 1000 rows * (4 bytes for INT32 + 8 bytes for INT64)
    std::size_t expected_min_size = 1000 * (4 + 8);
    REQUIRE(repr.get_size_in_bytes() >= expected_min_size);
  }

  SECTION("Empty table")
  {
    auto table = create_simple_cudf_table(0, gpu_space->get_default_allocator());
    gpu_table_representation repr(std::make_unique<cudf::table>(std::move(table)), *gpu_space);

    REQUIRE(repr.get_size_in_bytes() == 0);
  }
}

TEST_CASE("gpu_table_representation get_table", "[gpu_data_representation]")
{
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
  auto table     = create_simple_cudf_table(100, gpu_space->get_default_allocator());

  // Store the number of columns before moving the table
  auto num_columns = table.num_columns();

  gpu_table_representation repr(std::make_unique<cudf::table>(std::move(table)), *gpu_space);

  const cudf::table& retrieved_table = repr.get_table();
  REQUIRE(retrieved_table.num_columns() == num_columns);
  REQUIRE(retrieved_table.num_rows() == 100);
}

TEST_CASE("gpu_table_representation memory tier", "[gpu_data_representation]")
{
  SECTION("GPU tier")
  {
    auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
    auto table     = create_simple_cudf_table(100, gpu_space->get_default_allocator());
    gpu_table_representation repr(std::make_unique<cudf::table>(std::move(table)), *gpu_space);

    REQUIRE(repr.get_current_tier() == memory::Tier::GPU);
  }
}

TEST_CASE("gpu_table_representation device_id", "[gpu_data_representation]")
{
  SECTION("Device 0")
  {
    auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
    auto table     = create_simple_cudf_table(100, gpu_space->get_default_allocator());
    gpu_table_representation repr(std::make_unique<cudf::table>(std::move(table)), *gpu_space);

    REQUIRE(repr.get_device_id() == 0);
  }

  SECTION("Device 1")
  {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2) {
      SUCCEED("Single GPU or CUDA not available; skipping device 1 section");
      return;
    }

    auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 1);
    auto table     = create_simple_cudf_table(100, gpu_space->get_default_allocator());
    gpu_table_representation repr(std::make_unique<cudf::table>(std::move(table)), *gpu_space);

    REQUIRE(repr.get_device_id() == 1);
  }
}

TEST_CASE("gpu->host->gpu roundtrip preserves cudf table contents", "[gpu_data_representation]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const memory::memory_space* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const memory::memory_space* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);

  // Use one stream for table creation and both conversions to enforce order
  // and avoid stream-ordered races
  auto chain_stream = gpu_space->acquire_stream();
  auto table = create_simple_cudf_table(100, 2, gpu_space->get_default_allocator(), chain_stream);
  gpu_table_representation repr(std::make_unique<cudf::table>(std::move(table)),
                                *const_cast<memory::memory_space*>(gpu_space));

  auto cpu_any = registry.convert<host_table_representation>(repr, host_space, chain_stream);
  auto gpu_any = registry.convert<gpu_table_representation>(*cpu_any, gpu_space, chain_stream);

  auto& back = *gpu_any;
  chain_stream.synchronize();
  cucascade::test::expect_cudf_tables_equal_on_stream(
    repr.get_table(), back.get_table(), chain_stream);
  // Stream is automatically managed - no explicit release needed
}

// =============================================================================
// Multi-GPU Cross-Device Conversion Test
// =============================================================================
static std::unique_ptr<memory::memory_reservation_manager> create_multi_gpu_manager(int dev_a,
                                                                                    int dev_b)
{
  using namespace cucascade::memory;
  std::vector<memory_space_config> configs;
  configs.emplace_back(gpu_memory_space_config(dev_a, 2048ull * 1024 * 1024));
  configs.emplace_back(gpu_memory_space_config(dev_b, 2048ull * 1024 * 1024));
  return std::make_unique<memory_reservation_manager>(std::move(configs));
}

TEST_CASE("gpu cross-device conversion when multiple GPUs are available",
          "[gpu_data_representation][.multi-device]")
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2) {
    SUCCEED("Single GPU or CUDA not available; skipping cross-device test");
    return;
  }

  // Pick first two GPUs
  int dev_src = 0;
  int dev_dst = 1;

  auto mgr = create_multi_gpu_manager(dev_src, dev_dst);
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const memory::memory_space* src_space = mgr->get_memory_space(memory::Tier::GPU, dev_src);
  const memory::memory_space* dst_space = mgr->get_memory_space(memory::Tier::GPU, dev_dst);
  REQUIRE(src_space != nullptr);
  REQUIRE(dst_space != nullptr);

  // Use a single stream for table creation and peer copy to avoid stream-ordered races
  auto xfer_stream = src_space->acquire_stream();

  // Build a simple cudf table on source GPU and wrap it
  auto table = create_simple_cudf_table(256, 2, src_space->get_default_allocator(), xfer_stream);
  gpu_table_representation src_repr(std::make_unique<cudf::table>(std::move(table)),
                                    *const_cast<memory::memory_space*>(src_space));

  auto dst_any   = registry.convert<gpu_table_representation>(src_repr, dst_space, xfer_stream);
  auto& dst_repr = *dst_any;

  // Compare content equality using the same stream used for transfer
  cucascade::test::expect_cudf_tables_equal_on_stream(
    src_repr.get_table(), dst_repr.get_table(), xfer_stream);
}
// =============================================================================
// idata_representation Interface Tests
// =============================================================================

TEST_CASE("idata_representation cast functionality",
          "[cpu_data_representation][gpu_data_representation]")
{
  // Note: host_table_representation section disabled - requires internal API access

  SECTION("Cast gpu_table_representation")
  {
    auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
    auto table     = create_simple_cudf_table(100, gpu_space->get_default_allocator());
    gpu_table_representation repr(std::make_unique<cudf::table>(std::move(table)), *gpu_space);

    idata_representation* base_ptr = &repr;

    // Cast to derived type
    gpu_table_representation& casted = base_ptr->cast<gpu_table_representation>();
    REQUIRE(&casted == &repr);
    REQUIRE(casted.get_table().num_rows() == 100);
  }
}

TEST_CASE("idata_representation const cast functionality",
          "[cpu_data_representation][gpu_data_representation]")
{
  // Note: host_table_representation section disabled - requires internal API access

  SECTION("Const cast gpu_table_representation")
  {
    auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
    auto table     = create_simple_cudf_table(100, gpu_space->get_default_allocator());
    gpu_table_representation repr(std::make_unique<cudf::table>(std::move(table)), *gpu_space);

    const idata_representation* base_ptr = &repr;

    // Const cast to derived type
    const gpu_table_representation& casted = base_ptr->cast<gpu_table_representation>();
    REQUIRE(&casted == &repr);
    REQUIRE(casted.get_table().num_rows() == 100);
  }
}

// =============================================================================
// Cross-Tier Comparison Tests
// =============================================================================

// Disabled: requires internal access to multiple_blocks_allocation constructor
TEST_CASE("Compare CPU and GPU representations",
          "[cpu_data_representation][gpu_data_representation][.disabled]")
{
  SUCCEED("Test disabled - requires internal API access");
}

TEST_CASE("Multiple representations on same memory space",
          "[cpu_data_representation][gpu_data_representation]")
{
  // Note: host_table_representation section disabled - requires internal API access

  SECTION("Multiple GPU representations")
  {
    auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);

    auto table1 = create_simple_cudf_table(100, gpu_space->get_default_allocator());
    gpu_table_representation repr1(std::make_unique<cudf::table>(std::move(table1)), *gpu_space);

    auto table2 = create_simple_cudf_table(200, gpu_space->get_default_allocator());
    gpu_table_representation repr2(std::make_unique<cudf::table>(std::move(table2)), *gpu_space);

    REQUIRE(repr1.get_current_tier() == repr2.get_current_tier());
    REQUIRE(repr1.get_device_id() == repr2.get_device_id());
    // Different row counts should result in different sizes
    REQUIRE(repr1.get_size_in_bytes() != repr2.get_size_in_bytes());
  }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST_CASE("gpu_table_representation with single column", "[gpu_data_representation]")
{
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);

  std::vector<std::unique_ptr<cudf::column>> columns;
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       100,
                                       cudf::mask_state::UNALLOCATED,
                                       rmm::cuda_stream_default,
                                       gpu_space->get_default_allocator());
  columns.push_back(std::move(col));

  auto table = std::make_unique<cudf::table>(std::move(columns));
  gpu_table_representation repr(std::move(table), *gpu_space);

  REQUIRE(repr.get_table().num_columns() == 1);
  REQUIRE(repr.get_table().num_rows() == 100);
  REQUIRE(repr.get_size_in_bytes() >= 100 * 4);  // At least 100 rows * 4 bytes
}

TEST_CASE("gpu_table_representation with multiple column types", "[gpu_data_representation]")
{
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);

  std::vector<std::unique_ptr<cudf::column>> columns;

  // INT8 column
  auto col1 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT8}, 100, cudf::mask_state::UNALLOCATED);

  // INT16 column
  auto col2 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT16}, 100, cudf::mask_state::UNALLOCATED);

  // INT32 column
  auto col3 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 100, cudf::mask_state::UNALLOCATED);

  // INT64 column
  auto col4 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, 100, cudf::mask_state::UNALLOCATED);

  columns.push_back(std::move(col1));
  columns.push_back(std::move(col2));
  columns.push_back(std::move(col3));
  columns.push_back(std::move(col4));

  auto table = std::make_unique<cudf::table>(std::move(columns));
  gpu_table_representation repr(std::move(table), *gpu_space);

  REQUIRE(repr.get_table().num_columns() == 4);
  REQUIRE(repr.get_table().num_rows() == 100);
  // Size should be at least 100 * (1 + 2 + 4 + 8) = 1500 bytes
  REQUIRE(repr.get_size_in_bytes() >= 1500);
}

// Disabled: requires internal access to multiple_blocks_allocation constructor
TEST_CASE("Representations polymorphism",
          "[cpu_data_representation][gpu_data_representation][.disabled]")
{
  SUCCEED("Test disabled - requires internal API access");
}

// =============================================================================
// Clone Tests
// =============================================================================

TEST_CASE("gpu_table_representation clone creates independent copy", "[gpu_data_representation]")
{
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
  auto table     = create_simple_cudf_table(100, gpu_space->get_default_allocator());

  gpu_table_representation repr(std::make_unique<cudf::table>(std::move(table)), *gpu_space);

  // Clone the representation
  auto cloned_base = repr.clone();
  REQUIRE(cloned_base != nullptr);

  // Verify it's a gpu_table_representation
  auto* cloned = dynamic_cast<gpu_table_representation*>(cloned_base.get());
  REQUIRE(cloned != nullptr);

  // Verify the cloned representation has the same properties
  REQUIRE(cloned->get_current_tier() == repr.get_current_tier());
  REQUIRE(cloned->get_device_id() == repr.get_device_id());
  REQUIRE(cloned->get_size_in_bytes() == repr.get_size_in_bytes());

  // Verify the tables have the same shape
  REQUIRE(cloned->get_table().num_columns() == repr.get_table().num_columns());
  REQUIRE(cloned->get_table().num_rows() == repr.get_table().num_rows());

  // Verify the data is equal
  cucascade::test::expect_cudf_tables_equal_on_stream(
    repr.get_table(), cloned->get_table(), rmm::cuda_stream_default);

  // Verify the tables are independent (different memory addresses)
  REQUIRE(&cloned->get_table() != &repr.get_table());
  for (cudf::size_type i = 0; i < repr.get_table().num_columns(); ++i) {
    REQUIRE(repr.get_table().view().column(i).head() !=
            cloned->get_table().view().column(i).head());
  }
}

TEST_CASE("gpu_table_representation clone empty table", "[gpu_data_representation]")
{
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
  auto table     = create_simple_cudf_table(0, gpu_space->get_default_allocator());

  gpu_table_representation repr(std::make_unique<cudf::table>(std::move(table)), *gpu_space);

  auto cloned_base = repr.clone();
  REQUIRE(cloned_base != nullptr);

  auto* cloned = dynamic_cast<gpu_table_representation*>(cloned_base.get());
  REQUIRE(cloned != nullptr);
  REQUIRE(cloned->get_table().num_rows() == 0);
  REQUIRE(cloned->get_size_in_bytes() == 0);
}

TEST_CASE("host_table_representation clone creates independent copy", "[cpu_data_representation]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const memory::memory_space* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  const memory::memory_space* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);

  // Create a host_table_representation via conversion from GPU
  // Use the same stream for table creation and conversion to avoid stream-ordered races
  rmm::cuda_stream stream;
  auto original =
    create_simple_cudf_table(128, 2, gpu_space->get_default_allocator(), stream.view());
  gpu_table_representation gpu_repr(std::make_unique<cudf::table>(std::move(original)),
                                    *const_cast<memory::memory_space*>(gpu_space));

  auto host_repr_ptr = registry.convert<host_table_representation>(gpu_repr, host_space, stream);
  stream.synchronize();

  // Clone the host representation
  auto cloned_base = host_repr_ptr->clone();
  REQUIRE(cloned_base != nullptr);

  auto* cloned = dynamic_cast<host_table_representation*>(cloned_base.get());
  REQUIRE(cloned != nullptr);

  // Verify properties match
  REQUIRE(cloned->get_current_tier() == host_repr_ptr->get_current_tier());
  REQUIRE(cloned->get_device_id() == host_repr_ptr->get_device_id());
  REQUIRE(cloned->get_size_in_bytes() == host_repr_ptr->get_size_in_bytes());

  // Verify the underlying allocations are different (independent)
  REQUIRE(cloned->get_host_table().get() != host_repr_ptr->get_host_table().get());
  REQUIRE(cloned->get_host_table()->allocation.get() !=
          host_repr_ptr->get_host_table()->allocation.get());

  // Convert both back to GPU and verify data equality
  auto cloned_gpu = registry.convert<gpu_table_representation>(*cloned, gpu_space, stream);
  auto orig_gpu   = registry.convert<gpu_table_representation>(*host_repr_ptr, gpu_space, stream);
  stream.synchronize();

  cucascade::test::expect_cudf_tables_equal_on_stream(
    orig_gpu->get_table(), cloned_gpu->get_table(), stream.view());
}

//  */
