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
#include <cucascade/memory/host_table_packed.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch.hpp>

#include <cstring>
#include <memory>
#include <vector>

using namespace cucascade;
using cucascade::test::create_conversion_test_configs;
using cucascade::test::create_simple_cudf_table;
using cucascade::test::make_mock_memory_space;

// Note: Tests that require mock host_table_packed_allocation are disabled because
// fixed_size_host_memory_resource::multiple_blocks_allocation is now private.
// The real allocation tests below use actual memory resources.
[[maybe_unused]] static constexpr bool MOCK_HOST_ALLOCATION_DISABLED = true;

// =============================================================================
// host_data_packed_representation Tests
// =============================================================================

// Disabled: requires internal access to multiple_blocks_allocation constructor
TEST_CASE("host_data_packed_representation Construction", "[cpu_data_representation][.disabled]")
{
  SUCCEED("Test disabled - requires internal API access");
}

// Disabled: requires internal access to multiple_blocks_allocation constructor
TEST_CASE("host_data_packed_representation get_size_in_bytes",
          "[cpu_data_representation][.disabled]")
{
  SUCCEED("Test disabled - requires internal API access");
}

// Disabled: requires internal access to multiple_blocks_allocation constructor
TEST_CASE("host_data_packed_representation memory tier", "[cpu_data_representation][.disabled]")
{
  SUCCEED("Test disabled - requires internal API access");
}

// Disabled: requires internal access to multiple_blocks_allocation constructor
TEST_CASE("host_data_packed_representation device_id", "[cpu_data_representation][.disabled]")
{
  SUCCEED("Test disabled - requires internal API access");
}

TEST_CASE("host_data_packed_representation converts to GPU and preserves contents",
          "[cpu_data_representation][gpu_data_representation]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const memory::memory_space* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  const memory::memory_space* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);

  // Start from a known cudf table; pack it and build a host_data_packed_representation
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
  auto host_alloc = std::make_unique<memory::host_table_packed_allocation>(
    std::move(allocation), std::move(meta_copy), packed.gpu_data->size());
  host_data_packed_representation host_repr(std::move(host_alloc),
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

  auto cpu_any = registry.convert<host_data_packed_representation>(repr, host_space, chain_stream);
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
  // Note: host_data_packed_representation section disabled - requires internal API access

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
  // Note: host_data_packed_representation section disabled - requires internal API access

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
  // Note: host_data_packed_representation section disabled - requires internal API access

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
  auto cloned_base = repr.clone(rmm::cuda_stream_default);
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

  auto cloned_base = repr.clone(rmm::cuda_stream_default);
  REQUIRE(cloned_base != nullptr);

  auto* cloned = dynamic_cast<gpu_table_representation*>(cloned_base.get());
  REQUIRE(cloned != nullptr);
  REQUIRE(cloned->get_table().num_rows() == 0);
  REQUIRE(cloned->get_size_in_bytes() == 0);
}

TEST_CASE("host_data_packed_representation clone creates independent copy",
          "[cpu_data_representation]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const memory::memory_space* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  const memory::memory_space* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);

  // Create a host_data_packed_representation via conversion from GPU
  // Use the same stream for table creation and conversion to avoid stream-ordered races
  rmm::cuda_stream stream;
  auto original =
    create_simple_cudf_table(128, 2, gpu_space->get_default_allocator(), stream.view());
  gpu_table_representation gpu_repr(std::make_unique<cudf::table>(std::move(original)),
                                    *const_cast<memory::memory_space*>(gpu_space));

  auto host_repr_ptr =
    registry.convert<host_data_packed_representation>(gpu_repr, host_space, stream);
  stream.synchronize();

  // Clone the host representation
  auto cloned_base = host_repr_ptr->clone(stream.view());
  REQUIRE(cloned_base != nullptr);

  auto* cloned = dynamic_cast<host_data_packed_representation*>(cloned_base.get());
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

// =============================================================================
// host_data_representation Tests
// =============================================================================

/// Read `size` bytes from a multi-block allocation starting at byte offset `offset`.
static std::vector<uint8_t> read_from_alloc(const memory::fixed_multiple_blocks_allocation& alloc,
                                            std::size_t offset,
                                            std::size_t size)
{
  std::vector<uint8_t> result(size);
  const std::size_t blk_sz = alloc->block_size();
  std::size_t bi = offset / blk_sz, bo = offset % blk_sz, di = 0;
  while (di < size) {
    std::size_t n = std::min(size - di, blk_sz - bo);
    std::memcpy(result.data() + di, alloc->at(bi).data() + bo, n);
    di += n;
    bo += n;
    if (bo == blk_sz) {
      ++bi;
      bo = 0;
    }
  }
  return result;
}

/// Synchronously copy `size` bytes from a GPU pointer to a host std::vector.
static std::vector<uint8_t> gpu_bytes(const void* ptr, std::size_t size)
{
  std::vector<uint8_t> result(size);
  if (size > 0 && ptr != nullptr) {
    RMM_CUDA_TRY(cudaMemcpy(result.data(), ptr, size, cudaMemcpyDeviceToHost));
  }
  return result;
}

/// Wrap a single column into a gpu_table_representation.
static gpu_table_representation wrap_column(std::unique_ptr<cudf::column> col,
                                            memory::memory_space& gpu_space)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  return gpu_table_representation(std::make_unique<cudf::table>(std::move(cols)), gpu_space);
}

/// Convert a gpu_table_representation to host_data_representation via the registry.
static std::unique_ptr<host_data_representation> fast_convert(
  gpu_table_representation& src,
  const memory::memory_space* host_space,
  representation_converter_registry& registry,
  rmm::cuda_stream_view stream)
{
  return registry.convert<host_data_representation>(src, host_space, stream);
}

// =============================================================================
// Converter registration
// =============================================================================

TEST_CASE("register_builtin_converters registers GPU->host_data_representation",
          "[fast][registration]")
{
  representation_converter_registry registry;
  register_builtin_converters(registry);
  REQUIRE(registry.has_converter<gpu_table_representation, host_data_representation>());
}

// =============================================================================
// Fixed-width primitive types — metadata correctness
// =============================================================================

template <cudf::type_id TypeID, typename CppType>
static void check_fixed_width_metadata(memory::memory_reservation_manager& mgr,
                                       representation_converter_registry& registry)
{
  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  constexpr int N = 100;
  auto col        = cudf::make_numeric_column(cudf::data_type{TypeID},
                                       N,
                                       cudf::mask_state::UNALLOCATED,
                                       stream.view(),
                                       gpu_space->get_default_allocator());
  stream.synchronize();

  auto repr = wrap_column(std::move(col), *const_cast<memory::memory_space*>(gpu_space));
  auto host = fast_convert(repr, host_space, registry, stream.view());
  stream.synchronize();

  REQUIRE(host != nullptr);
  REQUIRE(host->get_current_tier() == memory::Tier::HOST);
  REQUIRE(host->get_size_in_bytes() > 0);

  const auto& cols = host->get_host_table()->columns;
  REQUIRE(cols.size() == 1);
  REQUIRE(cols[0].type_id == TypeID);
  REQUIRE(cols[0].num_rows == N);
  REQUIRE(cols[0].has_data == true);
  REQUIRE(cols[0].data_size == static_cast<std::size_t>(N) * sizeof(CppType));
  REQUIRE(cols[0].has_null_mask == false);
  REQUIRE(cols[0].children.empty());
}

TEST_CASE("Fast converter metadata: fixed-width primitive types", "[fast][metadata]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  // clang-format off
  SECTION("INT8")   { check_fixed_width_metadata<cudf::type_id::INT8,   int8_t  >(mgr, registry); }
  SECTION("INT16")  { check_fixed_width_metadata<cudf::type_id::INT16,  int16_t >(mgr, registry); }
  SECTION("INT32")  { check_fixed_width_metadata<cudf::type_id::INT32,  int32_t >(mgr, registry); }
  SECTION("INT64")  { check_fixed_width_metadata<cudf::type_id::INT64,  int64_t >(mgr, registry); }
  SECTION("UINT8")  { check_fixed_width_metadata<cudf::type_id::UINT8,  uint8_t >(mgr, registry); }
  SECTION("UINT16") { check_fixed_width_metadata<cudf::type_id::UINT16, uint16_t>(mgr, registry); }
  SECTION("UINT32") { check_fixed_width_metadata<cudf::type_id::UINT32, uint32_t>(mgr, registry); }
  SECTION("UINT64") { check_fixed_width_metadata<cudf::type_id::UINT64, uint64_t>(mgr, registry); }
  SECTION("FLOAT32"){ check_fixed_width_metadata<cudf::type_id::FLOAT32, float  >(mgr, registry); }
  SECTION("FLOAT64"){ check_fixed_width_metadata<cudf::type_id::FLOAT64, double >(mgr, registry); }
  SECTION("BOOL8")  { check_fixed_width_metadata<cudf::type_id::BOOL8,   bool   >(mgr, registry); }
  // clang-format on
}

// =============================================================================
// Byte-level data integrity
// =============================================================================

TEST_CASE("Fast converter copies INT32 data bytes correctly", "[fast][data_integrity]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  constexpr int N = 200;
  auto col        = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       N,
                                       cudf::mask_state::UNALLOCATED,
                                       stream.view(),
                                       gpu_space->get_default_allocator());
  RMM_CUDA_TRY(
    cudaMemsetAsync(col->mutable_view().head(), 0xAB, N * sizeof(int32_t), stream.value()));
  const void* gpu_ptr = col->view().data<int32_t>();

  auto repr = wrap_column(std::move(col), *const_cast<memory::memory_space*>(gpu_space));
  auto host = fast_convert(repr, host_space, registry, stream.view());
  stream.synchronize();

  const auto& meta = host->get_host_table()->columns[0];
  auto actual_bytes =
    read_from_alloc(host->get_host_table()->allocation, meta.data_offset, meta.data_size);
  auto expected_bytes = gpu_bytes(gpu_ptr, meta.data_size);
  REQUIRE(actual_bytes == expected_bytes);
}

TEST_CASE("Fast converter copies FLOAT64 data bytes correctly", "[fast][data_integrity]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  constexpr int N = 150;
  auto col        = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64},
                                       N,
                                       cudf::mask_state::UNALLOCATED,
                                       stream.view(),
                                       gpu_space->get_default_allocator());
  RMM_CUDA_TRY(
    cudaMemsetAsync(col->mutable_view().head(), 0xCD, N * sizeof(double), stream.value()));
  const void* gpu_ptr = col->view().data<double>();

  auto repr = wrap_column(std::move(col), *const_cast<memory::memory_space*>(gpu_space));
  auto host = fast_convert(repr, host_space, registry, stream.view());
  stream.synchronize();

  const auto& meta = host->get_host_table()->columns[0];
  auto actual_bytes =
    read_from_alloc(host->get_host_table()->allocation, meta.data_offset, meta.data_size);
  auto expected_bytes = gpu_bytes(gpu_ptr, meta.data_size);
  REQUIRE(actual_bytes == expected_bytes);
}

// =============================================================================
// Nullable columns
// =============================================================================

TEST_CASE("Fast converter: nullable INT32 — null mask metadata and bytes", "[fast][nullable]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  constexpr int N      = 100;
  auto col             = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       N,
                                       cudf::mask_state::ALL_VALID,
                                       stream.view(),
                                       gpu_space->get_default_allocator());
  const void* mask_ptr = col->view().null_mask();

  auto repr = wrap_column(std::move(col), *const_cast<memory::memory_space*>(gpu_space));
  auto host = fast_convert(repr, host_space, registry, stream.view());
  stream.synchronize();

  const auto& meta = host->get_host_table()->columns[0];
  REQUIRE(meta.has_null_mask == true);
  REQUIRE(meta.null_mask_size == cudf::bitmask_allocation_size_bytes(N));

  auto actual_mask =
    read_from_alloc(host->get_host_table()->allocation, meta.null_mask_offset, meta.null_mask_size);
  auto expected_mask = gpu_bytes(mask_ptr, meta.null_mask_size);
  REQUIRE(actual_mask == expected_mask);
}

TEST_CASE("Fast converter: nullable INT64 — both null mask and data bytes are correct",
          "[fast][nullable]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  constexpr int N = 64;
  auto col        = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                       N,
                                       cudf::mask_state::ALL_VALID,
                                       stream.view(),
                                       gpu_space->get_default_allocator());
  RMM_CUDA_TRY(
    cudaMemsetAsync(col->mutable_view().head(), 0x77, N * sizeof(int64_t), stream.value()));
  const void* data_ptr = col->view().data<int64_t>();
  const void* mask_ptr = col->view().null_mask();

  auto repr = wrap_column(std::move(col), *const_cast<memory::memory_space*>(gpu_space));
  auto host = fast_convert(repr, host_space, registry, stream.view());
  stream.synchronize();

  const auto& meta = host->get_host_table()->columns[0];

  auto actual_data =
    read_from_alloc(host->get_host_table()->allocation, meta.data_offset, meta.data_size);
  auto expected_data = gpu_bytes(data_ptr, meta.data_size);
  REQUIRE(actual_data == expected_data);

  auto actual_mask =
    read_from_alloc(host->get_host_table()->allocation, meta.null_mask_offset, meta.null_mask_size);
  auto expected_mask = gpu_bytes(mask_ptr, meta.null_mask_size);
  REQUIRE(actual_mask == expected_mask);
}

// =============================================================================
// Timestamp and Duration columns
// =============================================================================

TEST_CASE("Fast converter: timestamp columns metadata", "[fast][timestamp]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;
  constexpr int N = 50;

  SECTION("TIMESTAMP_DAYS — stored as int32_t")
  {
    auto col = cudf::make_timestamp_column(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS},
                                           N,
                                           cudf::mask_state::UNALLOCATED,
                                           stream.view(),
                                           gpu_space->get_default_allocator());
    stream.synchronize();
    auto repr = wrap_column(std::move(col), *const_cast<memory::memory_space*>(gpu_space));
    auto host = fast_convert(repr, host_space, registry, stream.view());
    stream.synchronize();
    const auto& meta = host->get_host_table()->columns[0];
    REQUIRE(meta.type_id == cudf::type_id::TIMESTAMP_DAYS);
    REQUIRE(meta.has_data == true);
    REQUIRE(meta.data_size == static_cast<std::size_t>(N) * sizeof(int32_t));
    REQUIRE(meta.children.empty());
  }

  SECTION("TIMESTAMP_SECONDS — stored as int64_t")
  {
    auto col = cudf::make_timestamp_column(cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS},
                                           N,
                                           cudf::mask_state::UNALLOCATED,
                                           stream.view(),
                                           gpu_space->get_default_allocator());
    stream.synchronize();
    auto repr = wrap_column(std::move(col), *const_cast<memory::memory_space*>(gpu_space));
    auto host = fast_convert(repr, host_space, registry, stream.view());
    stream.synchronize();
    const auto& meta = host->get_host_table()->columns[0];
    REQUIRE(meta.type_id == cudf::type_id::TIMESTAMP_SECONDS);
    REQUIRE(meta.has_data == true);
    REQUIRE(meta.data_size == static_cast<std::size_t>(N) * sizeof(int64_t));
  }

  SECTION("TIMESTAMP_MICROSECONDS — stored as int64_t")
  {
    auto col = cudf::make_timestamp_column(cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS},
                                           N,
                                           cudf::mask_state::UNALLOCATED,
                                           stream.view(),
                                           gpu_space->get_default_allocator());
    stream.synchronize();
    auto repr = wrap_column(std::move(col), *const_cast<memory::memory_space*>(gpu_space));
    auto host = fast_convert(repr, host_space, registry, stream.view());
    stream.synchronize();
    const auto& meta = host->get_host_table()->columns[0];
    REQUIRE(meta.type_id == cudf::type_id::TIMESTAMP_MICROSECONDS);
    REQUIRE(meta.data_size == static_cast<std::size_t>(N) * sizeof(int64_t));
  }
}

TEST_CASE("Fast converter: duration columns metadata", "[fast][duration]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;
  constexpr int N = 50;

  SECTION("DURATION_DAYS — stored as int32_t")
  {
    auto col = cudf::make_duration_column(cudf::data_type{cudf::type_id::DURATION_DAYS},
                                          N,
                                          cudf::mask_state::UNALLOCATED,
                                          stream.view(),
                                          gpu_space->get_default_allocator());
    stream.synchronize();
    auto repr = wrap_column(std::move(col), *const_cast<memory::memory_space*>(gpu_space));
    auto host = fast_convert(repr, host_space, registry, stream.view());
    stream.synchronize();
    const auto& meta = host->get_host_table()->columns[0];
    REQUIRE(meta.type_id == cudf::type_id::DURATION_DAYS);
    REQUIRE(meta.data_size == static_cast<std::size_t>(N) * sizeof(int32_t));
  }

  SECTION("DURATION_MILLISECONDS — stored as int64_t")
  {
    auto col = cudf::make_duration_column(cudf::data_type{cudf::type_id::DURATION_MILLISECONDS},
                                          N,
                                          cudf::mask_state::UNALLOCATED,
                                          stream.view(),
                                          gpu_space->get_default_allocator());
    stream.synchronize();
    auto repr = wrap_column(std::move(col), *const_cast<memory::memory_space*>(gpu_space));
    auto host = fast_convert(repr, host_space, registry, stream.view());
    stream.synchronize();
    const auto& meta = host->get_host_table()->columns[0];
    REQUIRE(meta.type_id == cudf::type_id::DURATION_MILLISECONDS);
    REQUIRE(meta.data_size == static_cast<std::size_t>(N) * sizeof(int64_t));
  }

  SECTION("DURATION_NANOSECONDS — stored as int64_t")
  {
    auto col = cudf::make_duration_column(cudf::data_type{cudf::type_id::DURATION_NANOSECONDS},
                                          N,
                                          cudf::mask_state::UNALLOCATED,
                                          stream.view(),
                                          gpu_space->get_default_allocator());
    stream.synchronize();
    auto repr = wrap_column(std::move(col), *const_cast<memory::memory_space*>(gpu_space));
    auto host = fast_convert(repr, host_space, registry, stream.view());
    stream.synchronize();
    const auto& meta = host->get_host_table()->columns[0];
    REQUIRE(meta.type_id == cudf::type_id::DURATION_NANOSECONDS);
    REQUIRE(meta.data_size == static_cast<std::size_t>(N) * sizeof(int64_t));
  }
}

// =============================================================================
// Decimal columns — scale stored in metadata
// =============================================================================

TEST_CASE("Fast converter: decimal columns store scale in metadata", "[fast][decimal]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;
  constexpr int N = 50;

  SECTION("DECIMAL32 scale=-3, element size=4")
  {
    auto col = cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::DECIMAL32, -3},
                                             N,
                                             cudf::mask_state::UNALLOCATED,
                                             stream.view(),
                                             gpu_space->get_default_allocator());
    stream.synchronize();
    auto repr = wrap_column(std::move(col), *const_cast<memory::memory_space*>(gpu_space));
    auto host = fast_convert(repr, host_space, registry, stream.view());
    stream.synchronize();
    const auto& meta = host->get_host_table()->columns[0];
    REQUIRE(meta.type_id == cudf::type_id::DECIMAL32);
    REQUIRE(meta.scale == -3);
    REQUIRE(meta.has_data == true);
    REQUIRE(meta.data_size == static_cast<std::size_t>(N) * sizeof(int32_t));
    REQUIRE(meta.children.empty());
  }

  SECTION("DECIMAL64 scale=-6, element size=8")
  {
    auto col = cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::DECIMAL64, -6},
                                             N,
                                             cudf::mask_state::UNALLOCATED,
                                             stream.view(),
                                             gpu_space->get_default_allocator());
    stream.synchronize();
    auto repr = wrap_column(std::move(col), *const_cast<memory::memory_space*>(gpu_space));
    auto host = fast_convert(repr, host_space, registry, stream.view());
    stream.synchronize();
    const auto& meta = host->get_host_table()->columns[0];
    REQUIRE(meta.type_id == cudf::type_id::DECIMAL64);
    REQUIRE(meta.scale == -6);
    REQUIRE(meta.data_size == static_cast<std::size_t>(N) * sizeof(int64_t));
  }

  SECTION("DECIMAL128 scale=-9, element size=16")
  {
    auto col = cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::DECIMAL128, -9},
                                             N,
                                             cudf::mask_state::UNALLOCATED,
                                             stream.view(),
                                             gpu_space->get_default_allocator());
    stream.synchronize();
    auto repr = wrap_column(std::move(col), *const_cast<memory::memory_space*>(gpu_space));
    auto host = fast_convert(repr, host_space, registry, stream.view());
    stream.synchronize();
    const auto& meta = host->get_host_table()->columns[0];
    REQUIRE(meta.type_id == cudf::type_id::DECIMAL128);
    REQUIRE(meta.scale == -9);
    REQUIRE(meta.data_size == static_cast<std::size_t>(N) * 16);  // 128-bit = 16 bytes
  }
}

// =============================================================================
// STRING column — children metadata: offsets + chars
// =============================================================================

TEST_CASE("Fast converter: STRING column metadata structure", "[fast][string]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  // Build a strings column with 4 strings and 10 total characters
  // offsets: [0, 2, 5, 7, 10]  — child[0]: INT32, 5 elements
  // chars:   10 bytes of 'x'   — child[1]: INT8, 10 elements
  constexpr int num_strings = 4;
  constexpr int total_chars = 10;

  std::vector<int32_t> host_offsets = {0, 2, 5, 7, 10};
  rmm::device_buffer offsets_buf(host_offsets.data(),
                                 host_offsets.size() * sizeof(int32_t),
                                 stream.view(),
                                 gpu_space->get_default_allocator());
  auto offsets_col =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                   static_cast<cudf::size_type>(host_offsets.size()),
                                   std::move(offsets_buf),
                                   rmm::device_buffer{},
                                   0);

  std::vector<int8_t> host_chars(total_chars, 'x');
  rmm::device_buffer chars_buf(
    host_chars.data(), host_chars.size(), stream.view(), gpu_space->get_default_allocator());

  stream.synchronize();

  auto strings_col = cudf::make_strings_column(
    num_strings, std::move(offsets_col), std::move(chars_buf), 0, rmm::device_buffer{});

  auto repr = wrap_column(std::move(strings_col), *const_cast<memory::memory_space*>(gpu_space));
  auto host = fast_convert(repr, host_space, registry, stream.view());
  stream.synchronize();

  const auto& meta = host->get_host_table()->columns[0];
  REQUIRE(meta.type_id == cudf::type_id::STRING);
  REQUIRE(meta.num_rows == num_strings);
  // In this cudf version, STRING stores chars in the column's data buffer.
  REQUIRE(meta.has_data == true);
  REQUIRE(meta.data_size == static_cast<std::size_t>(total_chars));
  REQUIRE(meta.has_null_mask == false);
  // Single child: offsets (INT32)
  REQUIRE(meta.children.size() == 1);

  const auto& offsets_meta = meta.children[0];
  REQUIRE(offsets_meta.type_id == cudf::type_id::INT32);
  REQUIRE(offsets_meta.num_rows == num_strings + 1);
  REQUIRE(offsets_meta.has_data == true);
  REQUIRE(offsets_meta.data_size == static_cast<std::size_t>(num_strings + 1) * sizeof(int32_t));
}

// =============================================================================
// LIST column — children metadata: offsets + values
// =============================================================================

TEST_CASE("Fast converter: LIST<INT32> column metadata structure", "[fast][list]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  // 3 lists with 7 total elements: [0..1], [2..4], [5..6]
  constexpr int num_lists  = 3;
  constexpr int num_values = 7;

  std::vector<int32_t> host_offsets = {0, 2, 5, 7};
  rmm::device_buffer offsets_buf(host_offsets.data(),
                                 host_offsets.size() * sizeof(int32_t),
                                 stream.view(),
                                 gpu_space->get_default_allocator());
  auto offsets_col =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                   static_cast<cudf::size_type>(host_offsets.size()),
                                   std::move(offsets_buf),
                                   rmm::device_buffer{},
                                   0);

  auto values_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                              num_values,
                                              cudf::mask_state::UNALLOCATED,
                                              stream.view(),
                                              gpu_space->get_default_allocator());
  stream.synchronize();

  auto list_col =
    cudf::make_lists_column(num_lists, std::move(offsets_col), std::move(values_col), 0, {});

  auto repr = wrap_column(std::move(list_col), *const_cast<memory::memory_space*>(gpu_space));
  auto host = fast_convert(repr, host_space, registry, stream.view());
  stream.synchronize();

  const auto& meta = host->get_host_table()->columns[0];
  REQUIRE(meta.type_id == cudf::type_id::LIST);
  REQUIRE(meta.num_rows == num_lists);
  REQUIRE(meta.has_data == false);
  REQUIRE(meta.has_null_mask == false);
  REQUIRE(meta.children.size() == 2);

  const auto& offsets_meta = meta.children[0];
  REQUIRE(offsets_meta.type_id == cudf::type_id::INT32);
  REQUIRE(offsets_meta.num_rows == num_lists + 1);
  REQUIRE(offsets_meta.has_data == true);
  REQUIRE(offsets_meta.data_size == static_cast<std::size_t>(num_lists + 1) * sizeof(int32_t));

  const auto& values_meta = meta.children[1];
  REQUIRE(values_meta.type_id == cudf::type_id::INT32);
  REQUIRE(values_meta.num_rows == num_values);
  REQUIRE(values_meta.has_data == true);
  REQUIRE(values_meta.data_size == static_cast<std::size_t>(num_values) * sizeof(int32_t));
}

TEST_CASE("Fast converter: nullable LIST<INT32> preserves parent null mask", "[fast][list]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  constexpr int num_lists  = 3;
  constexpr int num_values = 5;

  std::vector<int32_t> host_offsets = {0, 2, 3, 5};
  rmm::device_buffer offsets_buf(host_offsets.data(),
                                 host_offsets.size() * sizeof(int32_t),
                                 stream.view(),
                                 gpu_space->get_default_allocator());
  auto offsets_col =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                   static_cast<cudf::size_type>(host_offsets.size()),
                                   std::move(offsets_buf),
                                   rmm::device_buffer{},
                                   0);

  auto values_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                              num_values,
                                              cudf::mask_state::UNALLOCATED,
                                              stream.view(),
                                              gpu_space->get_default_allocator());

  // Build a null mask marking the first list as null (bit 0 = 0)
  auto null_mask = cudf::create_null_mask(num_lists, cudf::mask_state::ALL_VALID, stream.view());
  stream.synchronize();

  auto list_col = cudf::make_lists_column(
    num_lists, std::move(offsets_col), std::move(values_col), 1, std::move(null_mask));

  auto repr = wrap_column(std::move(list_col), *const_cast<memory::memory_space*>(gpu_space));
  auto host = fast_convert(repr, host_space, registry, stream.view());
  stream.synchronize();

  const auto& meta = host->get_host_table()->columns[0];
  REQUIRE(meta.type_id == cudf::type_id::LIST);
  REQUIRE(meta.has_null_mask == true);
  REQUIRE(meta.null_mask_size == cudf::bitmask_allocation_size_bytes(num_lists));
}

// =============================================================================
// STRUCT column — field children metadata
// =============================================================================

TEST_CASE("Fast converter: STRUCT<INT32, FLOAT64> column metadata structure", "[fast][struct]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  constexpr int N = 8;
  auto field0     = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          N,
                                          cudf::mask_state::UNALLOCATED,
                                          stream.view(),
                                          gpu_space->get_default_allocator());
  auto field1     = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64},
                                          N,
                                          cudf::mask_state::UNALLOCATED,
                                          stream.view(),
                                          gpu_space->get_default_allocator());
  stream.synchronize();

  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(std::move(field0));
  children.push_back(std::move(field1));
  auto struct_col = cudf::make_structs_column(N, std::move(children), 0, {});

  auto repr = wrap_column(std::move(struct_col), *const_cast<memory::memory_space*>(gpu_space));
  auto host = fast_convert(repr, host_space, registry, stream.view());
  stream.synchronize();

  const auto& meta = host->get_host_table()->columns[0];
  REQUIRE(meta.type_id == cudf::type_id::STRUCT);
  REQUIRE(meta.num_rows == N);
  REQUIRE(meta.has_data == false);
  REQUIRE(meta.has_null_mask == false);
  REQUIRE(meta.children.size() == 2);

  REQUIRE(meta.children[0].type_id == cudf::type_id::INT32);
  REQUIRE(meta.children[0].num_rows == N);
  REQUIRE(meta.children[0].has_data == true);
  REQUIRE(meta.children[0].data_size == static_cast<std::size_t>(N) * sizeof(int32_t));

  REQUIRE(meta.children[1].type_id == cudf::type_id::FLOAT64);
  REQUIRE(meta.children[1].num_rows == N);
  REQUIRE(meta.children[1].has_data == true);
  REQUIRE(meta.children[1].data_size == static_cast<std::size_t>(N) * sizeof(double));
}

// =============================================================================
// Nested types: LIST<LIST<INT32>>
// =============================================================================

TEST_CASE("Fast converter: LIST<LIST<INT32>> nested metadata", "[fast][nested]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  // Outer list: 2 lists of inner lists
  // Inner lists: 3 inner lists with [2, 3, 2] elements = 7 total values
  // outer_offsets: [0, 2, 3]    — 2 outer lists
  // inner_offsets: [0, 2, 5, 7] — 3 inner lists
  // values: 7 INT32 elements
  constexpr int num_outer  = 2;
  constexpr int num_inner  = 3;
  constexpr int num_values = 7;

  auto values = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          num_values,
                                          cudf::mask_state::UNALLOCATED,
                                          stream.view(),
                                          gpu_space->get_default_allocator());

  std::vector<int32_t> inner_offs_h = {0, 2, 5, 7};
  rmm::device_buffer inner_offs_buf(inner_offs_h.data(),
                                    inner_offs_h.size() * sizeof(int32_t),
                                    stream.view(),
                                    gpu_space->get_default_allocator());
  auto inner_offsets =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                   static_cast<cudf::size_type>(inner_offs_h.size()),
                                   std::move(inner_offs_buf),
                                   rmm::device_buffer{},
                                   0);
  stream.synchronize();

  auto inner_list =
    cudf::make_lists_column(num_inner, std::move(inner_offsets), std::move(values), 0, {});

  std::vector<int32_t> outer_offs_h = {0, 2, 3};
  rmm::device_buffer outer_offs_buf(outer_offs_h.data(),
                                    outer_offs_h.size() * sizeof(int32_t),
                                    stream.view(),
                                    gpu_space->get_default_allocator());
  auto outer_offsets =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                   static_cast<cudf::size_type>(outer_offs_h.size()),
                                   std::move(outer_offs_buf),
                                   rmm::device_buffer{},
                                   0);
  stream.synchronize();

  auto outer_list =
    cudf::make_lists_column(num_outer, std::move(outer_offsets), std::move(inner_list), 0, {});

  auto repr = wrap_column(std::move(outer_list), *const_cast<memory::memory_space*>(gpu_space));
  auto host = fast_convert(repr, host_space, registry, stream.view());
  stream.synchronize();

  // Outer LIST
  const auto& outer_meta = host->get_host_table()->columns[0];
  REQUIRE(outer_meta.type_id == cudf::type_id::LIST);
  REQUIRE(outer_meta.num_rows == num_outer);
  REQUIRE(outer_meta.has_data == false);
  REQUIRE(outer_meta.children.size() == 2);

  // outer.children[0] = outer offsets (INT32)
  REQUIRE(outer_meta.children[0].type_id == cudf::type_id::INT32);
  REQUIRE(outer_meta.children[0].num_rows == num_outer + 1);
  REQUIRE(outer_meta.children[0].has_data == true);

  // outer.children[1] = inner LIST
  const auto& inner_meta = outer_meta.children[1];
  REQUIRE(inner_meta.type_id == cudf::type_id::LIST);
  REQUIRE(inner_meta.num_rows == num_inner);
  REQUIRE(inner_meta.has_data == false);
  REQUIRE(inner_meta.children.size() == 2);

  // inner.children[0] = inner offsets (INT32)
  REQUIRE(inner_meta.children[0].type_id == cudf::type_id::INT32);
  REQUIRE(inner_meta.children[0].num_rows == num_inner + 1);

  // inner.children[1] = values (INT32)
  REQUIRE(inner_meta.children[1].type_id == cudf::type_id::INT32);
  REQUIRE(inner_meta.children[1].num_rows == num_values);
  REQUIRE(inner_meta.children[1].has_data == true);
  REQUIRE(inner_meta.children[1].data_size ==
          static_cast<std::size_t>(num_values) * sizeof(int32_t));
}

// =============================================================================
// Nested types: LIST<STRUCT<INT32, FLOAT64>>
// =============================================================================

TEST_CASE("Fast converter: LIST<STRUCT<INT32,FLOAT64>> nested metadata", "[fast][nested]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  // 3 lists with 5 total struct elements
  // offsets: [0, 2, 2, 5]  — list[0]={s0,s1}, list[1]={} (empty), list[2]={s2,s3,s4}
  constexpr int num_lists   = 3;
  constexpr int num_structs = 5;

  // Build the STRUCT<INT32, FLOAT64> child column (num_structs elements)
  auto f0 = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                      num_structs,
                                      cudf::mask_state::UNALLOCATED,
                                      stream.view(),
                                      gpu_space->get_default_allocator());
  auto f1 = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64},
                                      num_structs,
                                      cudf::mask_state::UNALLOCATED,
                                      stream.view(),
                                      gpu_space->get_default_allocator());
  stream.synchronize();

  std::vector<std::unique_ptr<cudf::column>> struct_fields;
  struct_fields.push_back(std::move(f0));
  struct_fields.push_back(std::move(f1));
  auto struct_col = cudf::make_structs_column(num_structs, std::move(struct_fields), 0, {});

  // Build the offsets column for the LIST
  std::vector<int32_t> offsets_h = {0, 2, 2, 5};
  rmm::device_buffer offsets_buf(offsets_h.data(),
                                 offsets_h.size() * sizeof(int32_t),
                                 stream.view(),
                                 gpu_space->get_default_allocator());
  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    static_cast<cudf::size_type>(offsets_h.size()),
                                                    std::move(offsets_buf),
                                                    rmm::device_buffer{},
                                                    0);
  stream.synchronize();

  auto list_col =
    cudf::make_lists_column(num_lists, std::move(offsets_col), std::move(struct_col), 0, {});

  auto repr = wrap_column(std::move(list_col), *const_cast<memory::memory_space*>(gpu_space));
  auto host = fast_convert(repr, host_space, registry, stream.view());
  stream.synchronize();

  // Outer: LIST
  const auto& list_meta = host->get_host_table()->columns[0];
  REQUIRE(list_meta.type_id == cudf::type_id::LIST);
  REQUIRE(list_meta.num_rows == num_lists);
  REQUIRE(list_meta.has_data == false);
  REQUIRE(list_meta.children.size() == 2);

  // list.children[0] = offsets (INT32)
  const auto& offs_meta = list_meta.children[0];
  REQUIRE(offs_meta.type_id == cudf::type_id::INT32);
  REQUIRE(offs_meta.num_rows == num_lists + 1);
  REQUIRE(offs_meta.has_data == true);
  REQUIRE(offs_meta.data_size == static_cast<std::size_t>(num_lists + 1) * sizeof(int32_t));

  // list.children[1] = STRUCT<INT32, FLOAT64>
  const auto& struct_meta = list_meta.children[1];
  REQUIRE(struct_meta.type_id == cudf::type_id::STRUCT);
  REQUIRE(struct_meta.num_rows == num_structs);
  REQUIRE(struct_meta.has_data == false);
  REQUIRE(struct_meta.children.size() == 2);

  REQUIRE(struct_meta.children[0].type_id == cudf::type_id::INT32);
  REQUIRE(struct_meta.children[0].num_rows == num_structs);
  REQUIRE(struct_meta.children[0].has_data == true);
  REQUIRE(struct_meta.children[0].data_size ==
          static_cast<std::size_t>(num_structs) * sizeof(int32_t));

  REQUIRE(struct_meta.children[1].type_id == cudf::type_id::FLOAT64);
  REQUIRE(struct_meta.children[1].num_rows == num_structs);
  REQUIRE(struct_meta.children[1].has_data == true);
  REQUIRE(struct_meta.children[1].data_size ==
          static_cast<std::size_t>(num_structs) * sizeof(double));
}

// =============================================================================
// Nested types: STRUCT<LIST<INT32>, FLOAT64>
// =============================================================================

TEST_CASE("Fast converter: STRUCT<LIST<INT32>,FLOAT64> nested metadata", "[fast][nested]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  // STRUCT with 4 rows, where:
  //   field[0] = LIST<INT32>: 4 lists with 9 total INT32 values
  //              offsets: [0, 3, 3, 7, 9]
  //   field[1] = FLOAT64: 4 scalar values
  constexpr int num_rows   = 4;
  constexpr int num_values = 9;

  // Build LIST<INT32> field
  std::vector<int32_t> offs_h = {0, 3, 3, 7, 9};
  rmm::device_buffer offs_buf(offs_h.data(),
                              offs_h.size() * sizeof(int32_t),
                              stream.view(),
                              gpu_space->get_default_allocator());
  auto offs_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                 static_cast<cudf::size_type>(offs_h.size()),
                                                 std::move(offs_buf),
                                                 rmm::device_buffer{},
                                                 0);

  auto vals_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                            num_values,
                                            cudf::mask_state::UNALLOCATED,
                                            stream.view(),
                                            gpu_space->get_default_allocator());
  stream.synchronize();

  auto list_field =
    cudf::make_lists_column(num_rows, std::move(offs_col), std::move(vals_col), 0, {});

  // Build FLOAT64 field
  auto float_field = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64},
                                               num_rows,
                                               cudf::mask_state::UNALLOCATED,
                                               stream.view(),
                                               gpu_space->get_default_allocator());
  stream.synchronize();

  // Build STRUCT<LIST<INT32>, FLOAT64>
  std::vector<std::unique_ptr<cudf::column>> fields;
  fields.push_back(std::move(list_field));
  fields.push_back(std::move(float_field));
  auto struct_col = cudf::make_structs_column(num_rows, std::move(fields), 0, {});

  auto repr = wrap_column(std::move(struct_col), *const_cast<memory::memory_space*>(gpu_space));
  auto host = fast_convert(repr, host_space, registry, stream.view());
  stream.synchronize();

  // Top-level: STRUCT
  const auto& struct_meta = host->get_host_table()->columns[0];
  REQUIRE(struct_meta.type_id == cudf::type_id::STRUCT);
  REQUIRE(struct_meta.num_rows == num_rows);
  REQUIRE(struct_meta.has_data == false);
  REQUIRE(struct_meta.children.size() == 2);

  // struct.children[0] = LIST<INT32>
  const auto& list_meta = struct_meta.children[0];
  REQUIRE(list_meta.type_id == cudf::type_id::LIST);
  REQUIRE(list_meta.num_rows == num_rows);
  REQUIRE(list_meta.has_data == false);
  REQUIRE(list_meta.children.size() == 2);

  // list offsets
  const auto& list_offs_meta = list_meta.children[0];
  REQUIRE(list_offs_meta.type_id == cudf::type_id::INT32);
  REQUIRE(list_offs_meta.num_rows == num_rows + 1);
  REQUIRE(list_offs_meta.has_data == true);
  REQUIRE(list_offs_meta.data_size == static_cast<std::size_t>(num_rows + 1) * sizeof(int32_t));

  // list values
  const auto& list_vals_meta = list_meta.children[1];
  REQUIRE(list_vals_meta.type_id == cudf::type_id::INT32);
  REQUIRE(list_vals_meta.num_rows == num_values);
  REQUIRE(list_vals_meta.has_data == true);
  REQUIRE(list_vals_meta.data_size == static_cast<std::size_t>(num_values) * sizeof(int32_t));

  // struct.children[1] = FLOAT64
  const auto& float_meta = struct_meta.children[1];
  REQUIRE(float_meta.type_id == cudf::type_id::FLOAT64);
  REQUIRE(float_meta.num_rows == num_rows);
  REQUIRE(float_meta.has_data == true);
  REQUIRE(float_meta.data_size == static_cast<std::size_t>(num_rows) * sizeof(double));
}

// =============================================================================
// Empty table (0 rows)
// =============================================================================

TEST_CASE("Fast converter: empty table (0 rows)", "[fast][empty]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  auto col1 = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                        0,
                                        cudf::mask_state::UNALLOCATED,
                                        stream.view(),
                                        gpu_space->get_default_allocator());
  auto col2 = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64},
                                        0,
                                        cudf::mask_state::UNALLOCATED,
                                        stream.view(),
                                        gpu_space->get_default_allocator());
  stream.synchronize();

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col1));
  cols.push_back(std::move(col2));
  gpu_table_representation repr(std::make_unique<cudf::table>(std::move(cols)),
                                *const_cast<memory::memory_space*>(gpu_space));

  auto host = fast_convert(repr, host_space, registry, stream.view());
  stream.synchronize();

  REQUIRE(host != nullptr);
  REQUIRE(host->get_size_in_bytes() == 0);
  REQUIRE(host->get_host_table()->columns.size() == 2);

  for (const auto& meta : host->get_host_table()->columns) {
    REQUIRE(meta.num_rows == 0);
    REQUIRE(meta.has_data == false);  // size 0 → no data to copy
    REQUIRE(meta.has_null_mask == false);
  }
}

// =============================================================================
// Multi-column table spanning all primitive types
// =============================================================================

TEST_CASE("Fast converter: multi-column table with all primitive types", "[fast][multi_column]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  constexpr int N = 16;

  using TypePair                         = std::pair<cudf::type_id, std::size_t>;
  const std::vector<TypePair> type_sizes = {
    {cudf::type_id::INT8, sizeof(int8_t)},
    {cudf::type_id::INT16, sizeof(int16_t)},
    {cudf::type_id::INT32, sizeof(int32_t)},
    {cudf::type_id::INT64, sizeof(int64_t)},
    {cudf::type_id::UINT8, sizeof(uint8_t)},
    {cudf::type_id::UINT16, sizeof(uint16_t)},
    {cudf::type_id::UINT32, sizeof(uint32_t)},
    {cudf::type_id::UINT64, sizeof(uint64_t)},
    {cudf::type_id::FLOAT32, sizeof(float)},
    {cudf::type_id::FLOAT64, sizeof(double)},
    {cudf::type_id::BOOL8, sizeof(bool)},
  };

  std::vector<std::unique_ptr<cudf::column>> cols;
  for (const auto& [tid, sz] : type_sizes) {
    cols.push_back(cudf::make_numeric_column(cudf::data_type{tid},
                                             N,
                                             cudf::mask_state::UNALLOCATED,
                                             stream.view(),
                                             gpu_space->get_default_allocator()));
  }
  stream.synchronize();

  gpu_table_representation repr(std::make_unique<cudf::table>(std::move(cols)),
                                *const_cast<memory::memory_space*>(gpu_space));

  auto host = fast_convert(repr, host_space, registry, stream.view());
  stream.synchronize();

  REQUIRE(host != nullptr);
  REQUIRE(host->get_host_table()->columns.size() == type_sizes.size());

  for (std::size_t i = 0; i < type_sizes.size(); ++i) {
    const auto& [tid, sz] = type_sizes[i];
    const auto& meta      = host->get_host_table()->columns[i];
    REQUIRE(meta.type_id == tid);
    REQUIRE(meta.num_rows == N);
    REQUIRE(meta.has_data == true);
    REQUIRE(meta.data_size == static_cast<std::size_t>(N) * sz);
    REQUIRE(meta.has_null_mask == false);
    REQUIRE(meta.children.empty());
  }
}

// =============================================================================
// clone() creates an independent copy with identical byte content
// =============================================================================

TEST_CASE("host_data_representation clone: same bytes, independent allocation", "[fast][clone]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  constexpr int N = 100;
  auto col        = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       N,
                                       cudf::mask_state::UNALLOCATED,
                                       stream.view(),
                                       gpu_space->get_default_allocator());
  RMM_CUDA_TRY(
    cudaMemsetAsync(col->mutable_view().head(), 0x55, N * sizeof(int32_t), stream.value()));

  auto repr = wrap_column(std::move(col), *const_cast<memory::memory_space*>(gpu_space));
  auto host = fast_convert(repr, host_space, registry, stream.view());
  stream.synchronize();

  auto cloned_base = host->clone(stream.view());
  REQUIRE(cloned_base != nullptr);

  auto* cloned = dynamic_cast<host_data_representation*>(cloned_base.get());
  REQUIRE(cloned != nullptr);

  // Same logical properties
  REQUIRE(cloned->get_size_in_bytes() == host->get_size_in_bytes());
  REQUIRE(cloned->get_current_tier() == host->get_current_tier());

  // Different underlying allocations (fully independent)
  REQUIRE(cloned->get_host_table().get() != host->get_host_table().get());
  REQUIRE(cloned->get_host_table()->allocation.get() != host->get_host_table()->allocation.get());

  // Metadata structure is mirrored
  REQUIRE(cloned->get_host_table()->columns.size() == host->get_host_table()->columns.size());
  REQUIRE(cloned->get_host_table()->columns[0].type_id ==
          host->get_host_table()->columns[0].type_id);
  REQUIRE(cloned->get_host_table()->columns[0].data_size ==
          host->get_host_table()->columns[0].data_size);

  // Byte content is identical
  const auto& orig_meta  = host->get_host_table()->columns[0];
  const auto& clone_meta = cloned->get_host_table()->columns[0];
  auto orig_bytes =
    read_from_alloc(host->get_host_table()->allocation, orig_meta.data_offset, orig_meta.data_size);
  auto clone_bytes = read_from_alloc(
    cloned->get_host_table()->allocation, clone_meta.data_offset, clone_meta.data_size);
  REQUIRE(orig_bytes == clone_bytes);
  REQUIRE(orig_bytes[0] == 0x55);  // Known fill pattern
}

TEST_CASE("host_data_representation clone: empty table", "[fast][clone]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       0,
                                       cudf::mask_state::UNALLOCATED,
                                       stream.view(),
                                       gpu_space->get_default_allocator());
  stream.synchronize();

  auto repr = wrap_column(std::move(col), *const_cast<memory::memory_space*>(gpu_space));
  auto host = fast_convert(repr, host_space, registry, stream.view());
  stream.synchronize();

  auto cloned_base = host->clone(stream.view());
  auto* cloned     = dynamic_cast<host_data_representation*>(cloned_base.get());
  REQUIRE(cloned != nullptr);
  REQUIRE(cloned->get_size_in_bytes() == 0);
  REQUIRE(cloned->get_host_table()->columns.size() == 1);
  REQUIRE(cloned->get_host_table()->columns[0].num_rows == 0);
}

// =============================================================================
// Round-trip: HOST FAST → GPU
// =============================================================================

/// Convert a host_data_representation back to gpu_table_representation via registry.
static std::unique_ptr<gpu_table_representation> fast_back_convert(
  host_data_representation& src,
  const memory::memory_space* gpu_space,
  representation_converter_registry& registry,
  rmm::cuda_stream_view stream)
{
  return registry.convert<gpu_table_representation>(src, gpu_space, stream);
}

TEST_CASE("register_builtin_converters registers host_data_representation->GPU",
          "[fast][registration]")
{
  representation_converter_registry registry;
  register_builtin_converters(registry);
  REQUIRE(registry.has_converter<host_data_representation, gpu_table_representation>());
}

TEST_CASE("Round-trip fast: INT32 column data preserved", "[fast][roundtrip]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  constexpr int N = 64;
  auto col        = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       N,
                                       cudf::mask_state::UNALLOCATED,
                                       stream.view(),
                                       gpu_space->get_default_allocator());
  RMM_CUDA_TRY(
    cudaMemsetAsync(col->mutable_view().head(), 0xAB, N * sizeof(int32_t), stream.view()));

  auto orig_repr = wrap_column(std::move(col), *const_cast<memory::memory_space*>(gpu_space));
  auto host      = fast_convert(orig_repr, host_space, registry, stream.view());
  stream.synchronize();

  auto back = fast_back_convert(*host, gpu_space, registry, stream.view());
  stream.synchronize();

  REQUIRE(back != nullptr);
  REQUIRE(back->get_table().num_columns() == 1);
  REQUIRE(back->get_table().num_rows() == N);

  cudf::table_view back_tv = back->get_table().view();
  auto bytes               = gpu_bytes(back_tv.column(0).data<uint8_t>(), N * sizeof(int32_t));
  REQUIRE(bytes[0] == 0xAB);
  REQUIRE(bytes[N * sizeof(int32_t) - 1] == 0xAB);
}

TEST_CASE("Round-trip fast: nullable INT64 null mask preserved", "[fast][roundtrip]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  constexpr int N = 32;
  auto col        = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                       N,
                                       cudf::mask_state::ALL_VALID,
                                       stream.view(),
                                       gpu_space->get_default_allocator());
  RMM_CUDA_TRY(
    cudaMemsetAsync(col->mutable_view().head(), 0x77, N * sizeof(int64_t), stream.view()));
  stream.synchronize();

  auto orig_repr = wrap_column(std::move(col), *const_cast<memory::memory_space*>(gpu_space));
  auto host      = fast_convert(orig_repr, host_space, registry, stream.view());
  stream.synchronize();

  auto back = fast_back_convert(*host, gpu_space, registry, stream.view());
  stream.synchronize();

  REQUIRE(back->get_table().num_columns() == 1);
  cudf::table_view back_tv = back->get_table().view();
  REQUIRE(back_tv.column(0).nullable());
  REQUIRE(back_tv.column(0).null_count() == 0);

  auto data_bytes = gpu_bytes(back_tv.column(0).data<uint8_t>(), N * sizeof(int64_t));
  REQUIRE(data_bytes[0] == 0x77);
}

TEST_CASE("Round-trip fast: FLOAT64 byte integrity", "[fast][roundtrip]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  constexpr int N = 50;
  auto col        = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64},
                                       N,
                                       cudf::mask_state::UNALLOCATED,
                                       stream.view(),
                                       gpu_space->get_default_allocator());
  RMM_CUDA_TRY(
    cudaMemsetAsync(col->mutable_view().head(), 0xCD, N * sizeof(double), stream.view()));

  auto orig_repr = wrap_column(std::move(col), *const_cast<memory::memory_space*>(gpu_space));
  auto host      = fast_convert(orig_repr, host_space, registry, stream.view());
  stream.synchronize();

  auto back = fast_back_convert(*host, gpu_space, registry, stream.view());
  stream.synchronize();

  cudf::table_view back_tv = back->get_table().view();
  auto data_bytes          = gpu_bytes(back_tv.column(0).data<uint8_t>(), N * sizeof(double));
  REQUIRE(data_bytes[0] == 0xCD);
  REQUIRE(data_bytes[N * sizeof(double) - 1] == 0xCD);
}

TEST_CASE("Round-trip fast: STRING column content preserved", "[fast][roundtrip]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  constexpr int num_strings = 4;
  constexpr int total_chars = 10;

  std::vector<int32_t> host_offsets = {0, 2, 5, 7, 10};
  rmm::device_buffer offsets_buf(host_offsets.data(),
                                 host_offsets.size() * sizeof(int32_t),
                                 stream.view(),
                                 gpu_space->get_default_allocator());
  auto offsets_col =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                   static_cast<cudf::size_type>(host_offsets.size()),
                                   std::move(offsets_buf),
                                   rmm::device_buffer{},
                                   0);

  std::vector<int8_t> host_chars(total_chars, 'x');
  rmm::device_buffer chars_buf(
    host_chars.data(), host_chars.size(), stream.view(), gpu_space->get_default_allocator());
  stream.synchronize();

  auto strings_col = cudf::make_strings_column(
    num_strings, std::move(offsets_col), std::move(chars_buf), 0, rmm::device_buffer{});

  auto orig_repr =
    wrap_column(std::move(strings_col), *const_cast<memory::memory_space*>(gpu_space));
  auto host = fast_convert(orig_repr, host_space, registry, stream.view());
  stream.synchronize();

  auto back = fast_back_convert(*host, gpu_space, registry, stream.view());
  stream.synchronize();

  REQUIRE(back->get_table().num_columns() == 1);
  REQUIRE(back->get_table().num_rows() == num_strings);
  cudf::table_view back_tv = back->get_table().view();
  REQUIRE(back_tv.column(0).type().id() == cudf::type_id::STRING);
  // Chars should be preserved
  auto char_bytes =
    gpu_bytes(back_tv.column(0).data<uint8_t>(), static_cast<std::size_t>(total_chars));
  REQUIRE(char_bytes[0] == static_cast<uint8_t>('x'));
  REQUIRE(char_bytes[total_chars - 1] == static_cast<uint8_t>('x'));
}

TEST_CASE("Round-trip fast: LIST<INT32> structure preserved", "[fast][roundtrip]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  constexpr int num_lists  = 3;
  constexpr int num_values = 7;

  std::vector<int32_t> host_offsets = {0, 2, 5, 7};
  rmm::device_buffer offsets_buf(host_offsets.data(),
                                 host_offsets.size() * sizeof(int32_t),
                                 stream.view(),
                                 gpu_space->get_default_allocator());
  auto offsets_col =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                   static_cast<cudf::size_type>(host_offsets.size()),
                                   std::move(offsets_buf),
                                   rmm::device_buffer{},
                                   0);
  auto values_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                              num_values,
                                              cudf::mask_state::UNALLOCATED,
                                              stream.view(),
                                              gpu_space->get_default_allocator());
  RMM_CUDA_TRY(cudaMemsetAsync(
    values_col->mutable_view().head(), 0x33, num_values * sizeof(int32_t), stream.view()));
  stream.synchronize();

  auto list_col =
    cudf::make_lists_column(num_lists, std::move(offsets_col), std::move(values_col), 0, {});

  auto orig_repr = wrap_column(std::move(list_col), *const_cast<memory::memory_space*>(gpu_space));
  auto host      = fast_convert(orig_repr, host_space, registry, stream.view());
  stream.synchronize();

  auto back = fast_back_convert(*host, gpu_space, registry, stream.view());
  stream.synchronize();

  REQUIRE(back->get_table().num_rows() == num_lists);
  cudf::table_view back_tv = back->get_table().view();
  REQUIRE(back_tv.column(0).type().id() == cudf::type_id::LIST);
  REQUIRE(back_tv.column(0).num_children() == 2);
  // Values child bytes preserved
  auto val_bytes =
    gpu_bytes(back_tv.column(0).child(1).data<uint8_t>(), num_values * sizeof(int32_t));
  REQUIRE(val_bytes[0] == 0x33);
}

TEST_CASE("Round-trip fast: STRUCT<INT32,FLOAT64> fields preserved", "[fast][roundtrip]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  constexpr int N = 8;
  auto f0         = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                      N,
                                      cudf::mask_state::UNALLOCATED,
                                      stream.view(),
                                      gpu_space->get_default_allocator());
  auto f1         = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64},
                                      N,
                                      cudf::mask_state::UNALLOCATED,
                                      stream.view(),
                                      gpu_space->get_default_allocator());
  RMM_CUDA_TRY(
    cudaMemsetAsync(f0->mutable_view().head(), 0x11, N * sizeof(int32_t), stream.view()));
  RMM_CUDA_TRY(cudaMemsetAsync(f1->mutable_view().head(), 0x22, N * sizeof(double), stream.view()));
  stream.synchronize();

  std::vector<std::unique_ptr<cudf::column>> fields;
  fields.push_back(std::move(f0));
  fields.push_back(std::move(f1));
  auto struct_col = cudf::make_structs_column(N, std::move(fields), 0, {});

  auto orig_repr =
    wrap_column(std::move(struct_col), *const_cast<memory::memory_space*>(gpu_space));
  auto host = fast_convert(orig_repr, host_space, registry, stream.view());
  stream.synchronize();

  auto back = fast_back_convert(*host, gpu_space, registry, stream.view());
  stream.synchronize();

  REQUIRE(back->get_table().num_rows() == N);
  cudf::table_view back_tv = back->get_table().view();
  REQUIRE(back_tv.column(0).type().id() == cudf::type_id::STRUCT);
  REQUIRE(back_tv.column(0).num_children() == 2);

  auto f0_bytes = gpu_bytes(back_tv.column(0).child(0).data<uint8_t>(), N * sizeof(int32_t));
  REQUIRE(f0_bytes[0] == 0x11);
  auto f1_bytes = gpu_bytes(back_tv.column(0).child(1).data<uint8_t>(), N * sizeof(double));
  REQUIRE(f1_bytes[0] == 0x22);
}

TEST_CASE("Round-trip fast: empty table (0 rows)", "[fast][roundtrip]")
{
  memory::memory_reservation_manager mgr(create_conversion_test_configs());
  representation_converter_registry registry;
  register_builtin_converters(registry);

  const auto* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const auto* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream stream;

  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       0,
                                       cudf::mask_state::UNALLOCATED,
                                       stream.view(),
                                       gpu_space->get_default_allocator());
  stream.synchronize();

  auto orig_repr = wrap_column(std::move(col), *const_cast<memory::memory_space*>(gpu_space));
  auto host      = fast_convert(orig_repr, host_space, registry, stream.view());
  stream.synchronize();

  auto back = fast_back_convert(*host, gpu_space, registry, stream.view());
  stream.synchronize();

  REQUIRE(back != nullptr);
  REQUIRE(back->get_table().num_columns() == 1);
  REQUIRE(back->get_table().num_rows() == 0);
  REQUIRE(back->get_table().view().column(0).type().id() == cudf::type_id::INT32);
}
