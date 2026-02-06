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

#include <cucascade/data/gpu_data_representation.hpp>

#include <cudf/copying.hpp>
#include <cudf/utilities/traits.hpp>

namespace cucascade {

gpu_table_representation::gpu_table_representation(std::unique_ptr<cudf::table> table,
                                                   cucascade::memory::memory_space& memory_space)
  : idata_representation(memory_space), _table(std::move(table))
{
}

std::size_t gpu_table_representation::get_size_in_bytes() const
{
  // TODO: Implement proper size calculation
  // This should return the total size of all columns in the table
  std::size_t total_size = 0;
  for (auto const& col : _table->view()) {
    // For now, we can calculate a rough estimate based on column size
    // This will need to be refined to account for all buffers (data, validity, offsets, etc.)
    total_size += static_cast<std::size_t>(col.size()) * cudf::size_of(col.type());
  }
  return total_size;
}

const cudf::table& gpu_table_representation::get_table() const { return *_table; }

std::unique_ptr<cudf::table> gpu_table_representation::release_table() { return std::move(_table); }

std::unique_ptr<idata_representation> gpu_table_representation::clone()
{
  // Create a deep copy of the cuDF table
  auto table_copy = std::make_unique<cudf::table>(_table->view());
  return std::make_unique<gpu_table_representation>(std::move(table_copy), get_memory_space());
}

}  // namespace cucascade
