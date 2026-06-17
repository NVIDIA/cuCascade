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

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cucascade {
namespace memory {

/**
 * @brief Generic columnar buffer-layout descriptor for a single (possibly nested) column.
 *
 * Captures everything needed to reconstruct a column from raw memory bytes: a domain-defined
 * type tag, logical row/null counts, an optional null mask buffer, an optional flat data buffer,
 * and child descriptors for nested types. Children are stored recursively.
 *
 * @note `type_id` is an opaque, domain-defined type tag that cuCascade never interprets. It is
 * laid out to hold the numeric value of a consumer's column type enum (e.g. the integer value of
 * a `cudf::type_id`), so consumers reconstruct the concrete type by `static_cast`-ing it back.
 * Keeping it a plain `int32_t` (same width/values as the corresponding consumer enum) makes the
 * on-disk and in-memory layouts binary-compatible across the cuCascade/consumer boundary.
 */
struct column_metadata {
  int32_t type_id;     ///< Domain-defined column type tag (numeric value of the consumer's enum)
  int32_t num_rows;    ///< Number of logical rows (elements) in this column
  int32_t null_count;  ///< Number of null elements (may be a domain-defined "unknown" sentinel)
  int32_t scale;       ///< Scale factor for fixed-point / decimal types; 0 for all others

  bool has_null_mask;            ///< Whether a null mask buffer was copied
  std::size_t null_mask_offset;  ///< Byte offset of the null mask within the allocation
  std::size_t null_mask_size;    ///< Size of the null mask buffer in bytes

  bool has_data;            ///< Whether a flat data buffer was copied (false for nested types)
  std::size_t data_offset;  ///< Byte offset of the data buffer within the allocation
  std::size_t data_size;    ///< Size of the data buffer in bytes

  std::vector<column_metadata> children;  ///< Metadata for child columns (nested types)
};

}  // namespace memory
}  // namespace cucascade
