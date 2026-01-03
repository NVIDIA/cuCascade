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

#pragma once

#include "data/common.hpp"
#include "data/representation_converter.hpp"
#include "memory/common.hpp"
#include "memory/memory_reservation_manager.hpp"
#include "memory/memory_space.hpp"
#include "memory/null_device_memory_resource.hpp"
#include "memory/reservation_manager_configurator.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <memory>
#include <vector>

namespace cucascade {
namespace test {

/**
 * @brief Mock memory_space for testing - provides a simple memory_space without real allocators.
 *
 * This is a lightweight memory space that can be used in tests that don't need
 * actual memory allocation functionality.
 */
class mock_memory_space : public memory::memory_space {
 public:
  mock_memory_space(memory::Tier tier, size_t device_id = 0)
    : memory::memory_space(tier,
                           static_cast<int>(device_id),
                           1024 * 1024 * 1024,                      // memory_limit
                           (1024ULL * 1024ULL * 1024ULL) * 8 / 10,  // start_downgrading_threshold
                           (1024ULL * 1024ULL * 1024ULL) / 2,       // stop_downgrading_threshold
                           1024 * 1024 * 1024,                      // capacity
                           std::make_unique<memory::null_device_memory_resource>())
  {
  }
};

/**
 * @brief Helper base class to hold memory_space - initialized before idata_representation.
 *
 * This is used with multiple inheritance to ensure the memory_space is constructed
 * before the idata_representation base class that requires it.
 */
struct mock_memory_space_holder {
  std::shared_ptr<mock_memory_space> space;

  mock_memory_space_holder(memory::Tier tier, size_t device_id)
    : space(std::make_shared<mock_memory_space>(tier, device_id))
  {
  }
};

/**
 * @brief Mock idata_representation for testing.
 *
 * Inherits from mock_memory_space_holder first to ensure it's constructed before
 * idata_representation (which needs a reference to the memory space).
 */
class mock_data_representation : private mock_memory_space_holder, public idata_representation {
 public:
  explicit mock_data_representation(memory::Tier tier, size_t size = 1024, size_t device_id = 0)
    : mock_memory_space_holder(tier, device_id)  // Construct holder first
      ,
      idata_representation(*space)  // Pass reference to base class
      ,
      _size(size)
  {
  }

  std::size_t get_size_in_bytes() const override { return _size; }

 private:
  size_t _size;
};

/**
 * @brief Create memory manager configs for conversion tests (one GPU(0) and one HOST(0)).
 *
 * @return std::vector<memory::memory_space_config> The configs
 */
inline std::vector<memory::memory_space_config> create_conversion_test_configs()
{
  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(2048ull * 1024 * 1024)
    .use_gpu_ids_as_host()
    .set_capacity_per_numa_node(4096ull * 1024 * 1024);
  return builder.build_with_topology();
}

/**
 * @brief Create a simple cuDF table for testing.
 *
 * @param num_rows Number of rows in the table
 * @param num_columns Number of columns (1 or 2 supported)
 * @return cudf::table A simple table with numeric columns
 *
 * When num_columns == 1: Creates a single INT32 column filled with 0x42
 * When num_columns == 2: Creates INT32 (0x11) and INT64 (0x22) columns
 */
inline cudf::table create_simple_cudf_table(int num_rows = 100, int num_columns = 2)
{
  std::vector<std::unique_ptr<cudf::column>> columns;

  // First column: INT32
  auto col1 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows, cudf::mask_state::UNALLOCATED);
  if (num_rows > 0) {
    auto view  = col1->mutable_view();
    auto bytes = static_cast<size_t>(num_rows) * sizeof(int32_t);
    RMM_CUDA_TRY(
      cudaMemset(const_cast<void*>(view.head()), (num_columns == 1) ? 0x42 : 0x11, bytes));
  }
  columns.push_back(std::move(col1));

  // Second column: INT64 (only if num_columns >= 2)
  if (num_columns >= 2) {
    auto col2 = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64}, num_rows, cudf::mask_state::UNALLOCATED);
    if (num_rows > 0) {
      auto view  = col2->mutable_view();
      auto bytes = static_cast<size_t>(num_rows) * sizeof(int64_t);
      RMM_CUDA_TRY(cudaMemset(const_cast<void*>(view.head()), 0x22, bytes));
    }
    columns.push_back(std::move(col2));
  }

  return cudf::table(std::move(columns));
}

}  // namespace test
}  // namespace cucascade
