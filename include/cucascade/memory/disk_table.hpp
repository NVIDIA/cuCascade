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

#include <cucascade/memory/column_metadata.hpp>

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace cucascade {
namespace memory {

/**
 * @brief Disk-resident allocation containing per-column metadata and a file path.
 *
 * Describes a table serialized to a single file on disk: the file path, the per-column
 * buffer-layout metadata, and the size of the file's data region (including alignment padding).
 * The column_metadata descriptors use the same generic type as in-memory column buffers, but
 * their null_mask_offset and data_offset fields are byte offsets into the file, aligned to
 * 4 KiB (DISK_FILE_ALIGNMENT); converters recompute offsets when moving data to another tier.
 */
struct disk_table_allocation {
  std::string file_path;                 ///< Path to the batch file on disk
  std::vector<column_metadata> columns;  ///< Per-column metadata (reuses existing column_metadata)
  std::size_t data_size;                 ///< File data extent in bytes, including alignment padding

  disk_table_allocation(std::string path, std::vector<column_metadata> cols, std::size_t data_sz)
    : file_path(std::move(path)), columns(std::move(cols)), data_size(data_sz)
  {
  }
};

/**
 * @brief Create a uniquely named empty file under the given base directory.
 *
 * Uses mkstemps to atomically create "{base_path}/batch_XXXXXX.cucascade", where XXXXXX is
 * replaced by a unique suffix, so concurrent callers (including other processes sharing the
 * same directory) never receive the same path. The file is created and closed; the caller
 * owns it and is responsible for removing it.
 *
 * @param base_path The directory under which the file will be created.
 * @return The path of the created file.
 * @throws std::runtime_error if the file cannot be created.
 */
std::string generate_disk_file_path(std::string_view base_path);

}  // namespace memory
}  // namespace cucascade
