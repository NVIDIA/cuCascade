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

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <cudf/types.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace cucascade {
namespace memory {

/**
 * @brief Metadata describing a single column's buffer layout within a host_table_allocation.
 *
 * Captures everything needed to reconstruct a cudf column from raw host memory bytes, without
 * relying on the cudf::pack serialization format. Children are stored recursively for nested types
 * (LIST, STRUCT, STRING, DICTIONARY32).
 */
struct column_metadata {
  cudf::type_id type_id;       ///< Column data type identifier
  cudf::size_type num_rows;    ///< Number of logical rows (elements) in this column
  cudf::size_type null_count;  ///< Number of null elements (may be UNKNOWN_NULL_COUNT)
  int32_t scale;               ///< Scale factor for DECIMAL32/64/128 types; 0 for all others

  bool has_null_mask;            ///< Whether a null mask buffer was copied
  std::size_t null_mask_offset;  ///< Byte offset of the null mask within the allocation
  std::size_t null_mask_size;    ///< Size of the null mask buffer in bytes

  bool has_data;            ///< Whether a flat data buffer was copied (false for nested types)
  std::size_t data_offset;  ///< Byte offset of the data buffer within the allocation
  std::size_t data_size;    ///< Size of the data buffer in bytes

  std::vector<column_metadata> children;  ///< Metadata for child columns (nested types)
};

/**
 * @brief Host memory allocation containing directly-copied column buffers and custom metadata.
 *
 * Unlike host_table_packed_allocation (which relies on cudf::pack/unpack and therefore requires an
 * intermediate GPU-side contiguous copy), this struct copies each column's GPU buffers
 * directly to pinned host memory. This avoids the extra GPU memory allocation and bandwidth
 * that cudf::pack incurs.
 *
 * Reconstruction uses the per-column column_metadata descriptors rather than cudf's
 * packed-table serialization format.
 */
struct host_table_allocation {
  /// Fixed-size pinned blocks containing all raw column buffer data
  memory::fixed_multiple_blocks_allocation allocation;
  /// One metadata entry per top-level column; children are stored recursively
  std::vector<column_metadata> columns;
  /// Total number of data bytes stored across all blocks
  std::size_t data_size;

  host_table_allocation(memory::fixed_multiple_blocks_allocation alloc,
                        std::vector<column_metadata> cols,
                        std::size_t data_sz)
    : allocation(std::move(alloc)), columns(std::move(cols)), data_size(data_sz)
  {
  }
};

}  // namespace memory
}  // namespace cucascade
