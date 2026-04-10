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

#include <cucascade/memory/host_table.hpp>

#include <rmm/aligned.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cucascade {

/// @brief Magic number identifying a cuCascade disk file: ASCII "CUCS" = 0x43554353.
static constexpr uint32_t DISK_FILE_MAGIC = 0x43554353u;

/// @brief Current binary format version. Increment on any layout change.
static constexpr uint32_t DISK_FILE_FORMAT_VERSION = 1u;

/// @brief Alignment boundary for column data buffers in disk files.
/// 4KB alignment ensures direct DMA without bounce buffers.
static constexpr std::size_t DISK_FILE_ALIGNMENT = 4096u;

/**
 * @brief Fixed-size header at the start of every cuCascade disk file.
 *
 * Layout (32 bytes total):
 *   [0..3]   magic          - Must be DISK_FILE_MAGIC
 *   [4..7]   version        - Must be DISK_FILE_FORMAT_VERSION
 *   [8..11]  num_columns    - Number of top-level columns
 *   [12..15] reserved_0     - Reserved for future use (must be 0)
 *   [16..23] metadata_size  - Size in bytes of the serialized column_metadata region
 *   [24..31] data_offset    - Byte offset from file start where column data begins (4KB-aligned)
 */
struct disk_file_header {
  uint32_t magic{DISK_FILE_MAGIC};
  uint32_t version{DISK_FILE_FORMAT_VERSION};
  uint32_t num_columns{0};
  uint32_t reserved_0{0};
  uint64_t metadata_size{0};
  uint64_t data_offset{0};
};

static_assert(sizeof(disk_file_header) == 32, "disk_file_header must be exactly 32 bytes");

/**
 * @brief Serialize a vector of column_metadata into a byte buffer.
 *
 * Handles recursive nested types: STRING, LIST, STRUCT, and DICTIONARY32 children
 * are serialized depth-first. Each column_metadata entry is written as its fixed-size
 * fields followed by a uint32_t child count, then each child recursively.
 *
 * @param columns The top-level column metadata to serialize.
 * @return A byte buffer containing the serialized metadata.
 */
[[nodiscard]] std::vector<uint8_t> serialize_column_metadata(
  const std::vector<memory::column_metadata>& columns);

/**
 * @brief Deserialize column_metadata from a byte buffer produced by serialize_column_metadata.
 *
 * @param data Pointer to the serialized bytes.
 * @param size Number of bytes available.
 * @return The deserialized top-level column metadata vector.
 * @throws cucascade::logic_error if the buffer is truncated or malformed.
 */
[[nodiscard]] std::vector<memory::column_metadata> deserialize_column_metadata(const uint8_t* data,
                                                                               std::size_t size);

}  // namespace cucascade
