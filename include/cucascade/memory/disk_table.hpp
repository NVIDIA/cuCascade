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

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace cucascade {
namespace memory {

/**
 * @brief Disk-resident allocation containing per-column metadata and a file path.
 *
 * Mirrors the role of host_table_allocation for the disk tier. Instead of pinned
 * host memory blocks, data is stored in a single file on disk identified by
 * file_path. The column_metadata descriptors are identical to those used by
 * host_table_allocation, enabling straightforward conversion between tiers.
 */
struct disk_table_allocation {
  std::string file_path;                 ///< Absolute path to the batch file on disk
  std::vector<column_metadata> columns;  ///< Per-column metadata (reuses existing column_metadata)
  std::size_t data_size;                 ///< Total bytes of data written to the file

  disk_table_allocation(std::string path, std::vector<column_metadata> cols, std::size_t data_sz)
    : file_path(std::move(path)), columns(std::move(cols)), data_size(data_sz)
  {
  }
};

/**
 * @brief Generate a unique file path under the given base directory.
 *
 * Thread-safe via atomic counter. Returns e.g. "{base_path}/batch_42.cucascade".
 *
 * @param base_path The directory under which the file will be created.
 * @return A unique file path string.
 */
std::string generate_disk_file_path(std::string_view base_path);

}  // namespace memory
}  // namespace cucascade
