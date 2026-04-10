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

#include <cucascade/data/disk_file_format.hpp>
#include <cucascade/error.hpp>

#include <cstdint>
#include <cstring>

namespace cucascade {

namespace {

/// @brief Number of fixed bytes per serialized column_metadata entry (excluding children).
/// 4×int32 + 2×uint8 + uint16 pad + 4×uint64 + uint32 num_children + uint32 pad = 60
static constexpr std::size_t COLUMN_METADATA_FIXED_SIZE = 60;

void serialize_one_column(const memory::column_metadata& col, std::vector<uint8_t>& buf)
{
  auto const start = buf.size();
  buf.resize(start + COLUMN_METADATA_FIXED_SIZE);
  auto* dst = buf.data() + start;

  auto type_id_val = static_cast<int32_t>(col.type_id);
  std::memcpy(dst, &type_id_val, sizeof(int32_t));
  dst += sizeof(int32_t);

  std::memcpy(dst, &col.num_rows, sizeof(int32_t));
  dst += sizeof(int32_t);

  std::memcpy(dst, &col.null_count, sizeof(int32_t));
  dst += sizeof(int32_t);

  std::memcpy(dst, &col.scale, sizeof(int32_t));
  dst += sizeof(int32_t);

  auto has_null_mask_val = static_cast<uint8_t>(col.has_null_mask ? 1 : 0);
  std::memcpy(dst, &has_null_mask_val, sizeof(uint8_t));
  dst += sizeof(uint8_t);

  auto has_data_val = static_cast<uint8_t>(col.has_data ? 1 : 0);
  std::memcpy(dst, &has_data_val, sizeof(uint8_t));
  dst += sizeof(uint8_t);

  uint16_t padding = 0;
  std::memcpy(dst, &padding, sizeof(uint16_t));
  dst += sizeof(uint16_t);

  std::memcpy(dst, &col.null_mask_offset, sizeof(uint64_t));
  dst += sizeof(uint64_t);

  std::memcpy(dst, &col.null_mask_size, sizeof(uint64_t));
  dst += sizeof(uint64_t);

  std::memcpy(dst, &col.data_offset, sizeof(uint64_t));
  dst += sizeof(uint64_t);

  std::memcpy(dst, &col.data_size, sizeof(uint64_t));
  dst += sizeof(uint64_t);

  auto num_children = static_cast<uint32_t>(col.children.size());
  std::memcpy(dst, &num_children, sizeof(uint32_t));
  dst += sizeof(uint32_t);

  uint32_t padding2 = 0;
  std::memcpy(dst, &padding2, sizeof(uint32_t));

  // Recursively serialize children
  for (const auto& child : col.children) {
    serialize_one_column(child, buf);
  }
}

memory::column_metadata deserialize_one_column(const uint8_t*& cursor, const uint8_t* end)
{
  auto remaining = static_cast<std::size_t>(end - cursor);
  if (remaining < COLUMN_METADATA_FIXED_SIZE) {
    CUCASCADE_FAIL("truncated column_metadata: need " + std::to_string(COLUMN_METADATA_FIXED_SIZE) +
                   " bytes, have " + std::to_string(remaining));
  }

  memory::column_metadata col{};

  int32_t type_id_val = 0;
  std::memcpy(&type_id_val, cursor, sizeof(int32_t));
  col.type_id = static_cast<cudf::type_id>(type_id_val);
  cursor += sizeof(int32_t);

  std::memcpy(&col.num_rows, cursor, sizeof(int32_t));
  cursor += sizeof(int32_t);

  std::memcpy(&col.null_count, cursor, sizeof(int32_t));
  cursor += sizeof(int32_t);

  std::memcpy(&col.scale, cursor, sizeof(int32_t));
  cursor += sizeof(int32_t);

  uint8_t has_null_mask_val = 0;
  std::memcpy(&has_null_mask_val, cursor, sizeof(uint8_t));
  col.has_null_mask = (has_null_mask_val != 0);
  cursor += sizeof(uint8_t);

  uint8_t has_data_val = 0;
  std::memcpy(&has_data_val, cursor, sizeof(uint8_t));
  col.has_data = (has_data_val != 0);
  cursor += sizeof(uint8_t);

  // Skip padding
  cursor += sizeof(uint16_t);

  std::memcpy(&col.null_mask_offset, cursor, sizeof(uint64_t));
  cursor += sizeof(uint64_t);

  std::memcpy(&col.null_mask_size, cursor, sizeof(uint64_t));
  cursor += sizeof(uint64_t);

  std::memcpy(&col.data_offset, cursor, sizeof(uint64_t));
  cursor += sizeof(uint64_t);

  std::memcpy(&col.data_size, cursor, sizeof(uint64_t));
  cursor += sizeof(uint64_t);

  uint32_t num_children = 0;
  std::memcpy(&num_children, cursor, sizeof(uint32_t));
  cursor += sizeof(uint32_t);

  // Skip padding2
  cursor += sizeof(uint32_t);

  // Recursively deserialize children
  col.children.reserve(num_children);
  for (uint32_t i = 0; i < num_children; ++i) {
    col.children.push_back(deserialize_one_column(cursor, end));
  }

  return col;
}

}  // namespace

std::vector<uint8_t> serialize_column_metadata(const std::vector<memory::column_metadata>& columns)
{
  std::vector<uint8_t> buf;

  // Write top-level column count
  auto num_columns = static_cast<uint32_t>(columns.size());
  buf.resize(sizeof(uint32_t));
  std::memcpy(buf.data(), &num_columns, sizeof(uint32_t));

  // Serialize each top-level column recursively
  for (const auto& col : columns) {
    serialize_one_column(col, buf);
  }

  return buf;
}

std::vector<memory::column_metadata> deserialize_column_metadata(const uint8_t* data,
                                                                 std::size_t size)
{
  if (size < sizeof(uint32_t)) {
    CUCASCADE_FAIL("truncated column_metadata buffer: need at least 4 bytes for column count");
  }

  const uint8_t* cursor = data;
  const uint8_t* end    = data + size;

  uint32_t num_columns = 0;
  std::memcpy(&num_columns, cursor, sizeof(uint32_t));
  cursor += sizeof(uint32_t);

  std::vector<memory::column_metadata> columns;
  columns.reserve(num_columns);
  for (uint32_t i = 0; i < num_columns; ++i) {
    columns.push_back(deserialize_one_column(cursor, end));
  }

  return columns;
}

}  // namespace cucascade
