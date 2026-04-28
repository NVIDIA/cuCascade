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

#include <cucascade/data/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <any>
#include <cstddef>
#include <memory>
#include <variant>

namespace cucascade {

/**
 * @brief Data representation for a table being stored in GPU memory.
 *
 * This class currently represents a table just as a cuDF table along with the allocation where the
 * cudf's table data actually resides. The primary purpose for this is that the table can be
 * directly passed to cuDF APIs for processing without any additional copying while the underlying
 * memory is still owned/tracked by our memory allocator.
 *
 * TODO: Once the GPU memory resource is implemented, replace the allocation type from
 * IAllocatedMemory to the concrete type returned by the GPU memory allocator.
 */
class gpu_table_representation : public idata_representation {
 public:
  /**
   * @brief Construct a new gpu_table_representation object
   *
   * @param table Unique pointer to the cuDF table with the data (ownership is transferred)
   * @param memory_space The memory space where the GPU table resides
   */
  gpu_table_representation(std::unique_ptr<cudf::table> table,
                           cucascade::memory::memory_space& memory_space);

  /**
   * @brief Construct a new gpu_table_representation object
   *
   * @param table Unique pointer to the cuDF table with the data (ownership is transferred)
   * @tparam Owner The type of the owner of the cuDF table (e.g., a specific operator or component)
   * @param memory_space The memory space where the GPU table resides
   */
  template <typename Owner>
  gpu_table_representation(cudf::table_view table_view,
                           Owner&& owner,
                           std::size_t alloc_size,
                           cucascade::memory::memory_space& memory_space);

  /**
   * @brief Get the size of the data representation in bytes
   *
   * @return std::size_t The number of bytes used to store this representation
   */
  std::size_t get_size_in_bytes() const override;

  /**
   * @copydoc idata_representation::get_logical_data_size_in_bytes
   */
  std::size_t get_uncompressed_data_size_in_bytes() const override;

  /**
   * @brief Create a deep copy of this GPU table representation.
   *
   * The cloned representation will have its own copy of the underlying cuDF table,
   * residing in the same memory space as the original.
   *
   * @param stream CUDA stream for memory operations
   * @return std::unique_ptr<idata_representation> A new gpu_table_representation with copied data
   */
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) override;

  /**
   * @brief Get the underlying cuDF table view
   *
   * @return cudf::table_view A view of the cuDF table
   */
  cudf::table_view get_table_view() const;

  /**
   * @brief Release ownership of the underlying cuDF table
   *
   * After calling this method, this representation no longer owns the table.
   *
   * @return std::unique_ptr<cudf::table> The cuDF table
   */
  std::unique_ptr<cudf::table> release_table(rmm::cuda_stream_view stream);

 private:
  struct owning_table_view {
    std::any owner;  ///< The owner of the cuDF table
    std::size_t alloc_size{0};
    cudf::table_view view;  ///< A view of the owned table for easy access
  };

  std::variant<std::unique_ptr<cudf::table>, owning_table_view>
    _table;  ///< cudf::table is the underlying representation of the data
};

template <typename Owner>
gpu_table_representation::gpu_table_representation(cudf::table_view table_view,
                                                   Owner&& owner,
                                                   std::size_t alloc_size,
                                                   cucascade::memory::memory_space& memory_space)
  : idata_representation(memory_space),
    _table(
      owning_table_view{std::make_any<Owner>(std::forward<Owner>(owner)), alloc_size, table_view})
{
}

}  // namespace cucascade
