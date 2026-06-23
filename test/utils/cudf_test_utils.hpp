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

#include <cucascade/error.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cucascade {
namespace test {

// Stream-aware variants to enforce stream ordering with async allocations
bool cudf_tables_have_equal_contents_on_stream(const cudf::table_view& left,
                                               const cudf::table_view& right,
                                               rmm::cuda_stream_view stream_view);
void expect_cudf_tables_equal_on_stream(const cudf::table_view& left,
                                        const cudf::table_view& right,
                                        rmm::cuda_stream_view stream_view);

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
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref(),
  rmm::cuda_stream_view stream      = rmm::cuda_stream_default)
{
  std::vector<std::unique_ptr<cudf::column>> columns;

  // First column: INT32
  auto col1 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  if (num_rows > 0) {
    auto view  = col1->mutable_view();
    auto bytes = static_cast<size_t>(num_rows) * sizeof(int32_t);
    CUCASCADE_CUDA_TRY(
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
      CUCASCADE_CUDA_TRY(cudaMemset(const_cast<void*>(view.head()), 0x22, bytes));
    }
    columns.push_back(std::move(col2));
  }

  return cudf::table(std::move(columns));
}

inline cudf::table create_simple_cudf_table(
  int num_rows,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref(),
  rmm::cuda_stream_view stream      = rmm::cuda_stream_default)
{
  return create_simple_cudf_table(num_rows, 2, mr, stream);
}

inline cudf::table create_simple_cudf_table(
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref(),
  rmm::cuda_stream_view stream      = rmm::cuda_stream_default)
{
  return create_simple_cudf_table(100, 2, mr, stream);
}

}  // namespace test
}  // namespace cucascade
