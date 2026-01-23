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
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/host_table.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <memory>
#include <vector>

namespace cucascade {

/**
 * @brief Data representation for a table being stored in host memory.
 *
 * This represents a table whose data is stored across multiple blocks (not necessarily contiguous)
 * in host memory. The host_table_representation doesn't own the actual data but is instead owned by
 * the multiple_blocks_allocation.
 */
class host_table_representation : public idata_representation {
 public:
  /**
   * @brief Construct a new host_table_representation object
   *
   * @param host_table The underlying allocation owning the actual data
   * @param memory_space The memory space where the host table resides
   */
  host_table_representation(std::unique_ptr<cucascade::memory::host_table_allocation> host_table,
                            cucascade::memory::memory_space* memory_space);

  /**
   * @brief Get the size of the data representation in bytes
   *
   * @return std::size_t The number of bytes used to store this representation
   */
  std::size_t get_size_in_bytes() const override;

  /**
   * @brief Get the underlying host table allocation
   *
   * @return std::unique_ptr<cucascade::memory::table_allocation> The underlying host table
   * allocation
   */
  const std::unique_ptr<cucascade::memory::host_table_allocation>& get_host_table() const;

 private:
  std::unique_ptr<cucascade::memory::host_table_allocation>
    _host_table;  ///< The allocation where the actual data resides
};

}  // namespace cucascade
