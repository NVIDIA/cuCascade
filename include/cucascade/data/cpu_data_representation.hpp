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
#include <cucascade/memory/host_table_packed.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <memory>
#include <vector>

namespace cucascade {

/**
 * @brief Data representation for a table stored in host memory using direct buffer copies.
 *
 * Directly copies each column's GPU device buffers to pinned host memory without an intermediate
 * GPU-side contiguous allocation. The column layout is described by custom per-column metadata
 * stored in host_table_allocation, enabling reconstruction without cudf's pack format.
 *
 * This avoids the extra GPU allocation and GPU-to-GPU copy that cudf::pack performs when
 * serializing a table, at the cost of managing per-column buffer metadata ourselves.
 */
class host_data_representation : public idata_representation {
 public:
  /**
   * @brief Construct a host_data_representation.
   *
   * @param host_table Allocation owning the copied column buffers and their layout metadata
   * @param memory_space The host memory space where the data resides
   */
  host_data_representation(std::unique_ptr<memory::host_table_allocation> host_table,
                           memory::memory_space* memory_space);

  /**
   * @brief Get the total number of bytes occupied by this representation.
   *
   * @return std::size_t The number of data bytes stored in the host allocation
   */
  std::size_t get_size_in_bytes() const override;

  /**
   * @brief Create a deep copy of this representation in the same memory space.
   *
   * @param stream CUDA stream (unused for host-side copies)
   * @return std::unique_ptr<idata_representation> A new host_data_representation
   */
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) override;

  /**
   * @brief Access the underlying host table allocation.
   *
   * @return const reference to the unique_ptr owning the allocation
   */
  const std::unique_ptr<memory::host_table_allocation>& get_host_table() const;

 private:
  std::unique_ptr<memory::host_table_allocation> _host_table;
};

/**
 * @brief Data representation for a table being stored in host memory using cudf::pack.
 *
 * This represents a table whose data is stored across multiple blocks (not necessarily contiguous)
 * in host memory. The host_data_packed_representation doesn't own the actual data but is instead
 * owned by the multiple_blocks_allocation.
 */
class host_data_packed_representation : public idata_representation {
 public:
  /**
   * @brief Construct a new host_data_packed_representation object
   *
   * @param host_table The underlying allocation owning the actual data
   * @param memory_space The memory space where the host table resides
   */
  host_data_packed_representation(
    std::unique_ptr<cucascade::memory::host_table_packed_allocation> host_table,
    cucascade::memory::memory_space* memory_space);

  /**
   * @brief Get the size of the data representation in bytes
   *
   * @return std::size_t The number of bytes used to store this representation
   */
  std::size_t get_size_in_bytes() const override;

  /**
   * @brief Create a deep copy of this host table representation.
   *
   * The cloned representation will have its own copy of the underlying host table,
   * residing in the same memory space as the original.
   *
   * @param stream CUDA stream (unused for host-side copies)
   * @return std::unique_ptr<idata_representation> A new host_data_packed_representation with
   * copied data
   */
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) override;

  /**
   * @brief Get the underlying host table allocation
   *
   * @return const reference to the unique_ptr owning the allocation
   */
  const std::unique_ptr<cucascade::memory::host_table_packed_allocation>& get_host_table() const;

 private:
  std::unique_ptr<cucascade::memory::host_table_packed_allocation>
    _host_table;  ///< The allocation where the actual data resides
};

}  // namespace cucascade
