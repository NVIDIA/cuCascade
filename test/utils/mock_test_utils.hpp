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

#include "utils/test_memory_resources.hpp"

#include <cucascade/data/common.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/null_device_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

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

inline std::shared_ptr<memory::memory_space> make_mock_memory_space(memory::Tier tier,
                                                                    size_t device_id = 0)
{
  if (tier == memory::Tier::GPU) {
    memory::gpu_memory_space_config config;
    config.device_id       = static_cast<int>(device_id);
    config.memory_capacity = 1024 * 1024 * 1024;
    config.mr_factory_fn   = [](int, size_t) {
      return std::make_unique<rmm::mr::cuda_memory_resource>();
    };
    return std::make_shared<memory::memory_space>(config);
  } else if (tier == memory::Tier::HOST) {
    memory::host_memory_space_config config;
    config.numa_id              = device_id;
    config.memory_capacity      = 1024 * 1024 * 1024;
    config.initial_number_pools = 0;
    config.mr_factory_fn        = [](int, size_t) {
      return std::make_unique<cucascade::memory::numa_region_pinned_host_memory_resource>(-1);
    };
    return std::make_shared<memory::memory_space>(config);
  } else if (tier == memory::Tier::DISK) {
    return std::make_shared<memory::memory_space>(
      memory::disk_memory_space_config{static_cast<int>(device_id), 1024 * 1024 * 1024, "/tmp"});
  } else {
    throw std::invalid_argument("Unsupported tier for mock memory space");
  }
}

/**
 * @brief Helper base class to hold memory_space - initialized before idata_representation.
 *
 * This is used with multiple inheritance to ensure the memory_space is constructed
 * before the idata_representation base class that requires it.
 */
struct mock_memory_space_holder {
  std::shared_ptr<memory::memory_space> space;

  mock_memory_space_holder(memory::Tier tier, size_t device_id)
    : space{make_mock_memory_space(tier, device_id)}
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

  std::unique_ptr<idata_representation> clone() override
  {
    return std::make_unique<mock_data_representation>(get_current_tier(), _size, get_device_id());
  }

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
    // Use shared pooled allocator initialized once for all tests.
    .set_gpu_memory_resource_factory(make_shared_current_device_resource)
    .use_host_per_gpu()
    .set_per_host_capacity(4096ull * 1024 * 1024);
  return builder.build();
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
inline cudf::table create_simple_cudf_table(
  int num_rows,
  int num_columns,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default)
{
  std::vector<std::unique_ptr<cudf::column>> columns;

  // First column: INT32
  auto col1 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
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
      cudf::data_type{cudf::type_id::INT64}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
    if (num_rows > 0) {
      auto view  = col2->mutable_view();
      auto bytes = static_cast<size_t>(num_rows) * sizeof(int64_t);
      RMM_CUDA_TRY(cudaMemset(const_cast<void*>(view.head()), 0x22, bytes));
    }
    columns.push_back(std::move(col2));
  }

  return cudf::table(std::move(columns));
}

inline cudf::table create_simple_cudf_table(
  int num_rows,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default)
{
  return create_simple_cudf_table(num_rows, 2, mr, stream);
}

inline cudf::table create_simple_cudf_table(
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default)
{
  return create_simple_cudf_table(100, 2, mr, stream);
}

}  // namespace test
}  // namespace cucascade
