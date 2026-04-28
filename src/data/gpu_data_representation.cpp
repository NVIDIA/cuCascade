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
  if (std::holds_alternative<std::unique_ptr<cudf::table>>(_table)) {
    return std::get<std::unique_ptr<cudf::table>>(_table)->alloc_size();
  } else if (std::holds_alternative<owning_table_view>(_table)) {
    return std::get<owning_table_view>(_table).alloc_size;
  }
  return 0;
}

std::size_t gpu_table_representation::get_uncompressed_data_size_in_bytes() const
{
  return get_size_in_bytes();
}

cudf::table_view gpu_table_representation::get_table_view() const
{
  if (std::holds_alternative<std::unique_ptr<cudf::table>>(_table)) {
    return std::get<std::unique_ptr<cudf::table>>(_table)->view();
  } else {
    return std::get<owning_table_view>(_table).view;
  }
}

std::unique_ptr<cudf::table> gpu_table_representation::release_table(
  [[maybe_unused]] rmm::cuda_stream_view stream)
{
  if (std::holds_alternative<owning_table_view>(_table)) {
    _table = std::make_unique<cudf::table>(std::get<owning_table_view>(_table).view, stream);
  }
  return std::move(std::get<std::unique_ptr<cudf::table>>(_table));
}

std::unique_ptr<idata_representation> gpu_table_representation::clone(rmm::cuda_stream_view stream)
{
  // Create a deep copy of the cuDF table using the provided stream
  cudf::table_view view = get_table_view();
  return std::make_unique<gpu_table_representation>(std::make_unique<cudf::table>(view, stream),
                                                    get_memory_space());
}

}  // namespace cucascade
