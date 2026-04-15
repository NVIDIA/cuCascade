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

#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/error.hpp>

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cucascade {

// =============================================================================
// disk_data_representation
// =============================================================================

disk_data_representation::disk_data_representation(
  std::unique_ptr<memory::disk_table_allocation> disk_table, memory::memory_space& memory_space)
  : idata_representation(memory_space), _disk_table(std::move(disk_table))
{
}

disk_data_representation::~disk_data_representation() noexcept
{
  try {
    std::filesystem::remove(_disk_table->file_path);
  } catch (...) {
    // Best-effort file deletion in destructor -- silently discard errors.
  }
}

std::size_t disk_data_representation::get_size_in_bytes() const { return _disk_table->data_size; }

std::size_t disk_data_representation::get_uncompressed_data_size_in_bytes() const
{
  return _disk_table->data_size;
}

std::unique_ptr<idata_representation> disk_data_representation::clone(
  [[maybe_unused]] rmm::cuda_stream_view stream)
{
  CUCASCADE_FAIL("disk_data_representation does not support clone");
}

const memory::disk_table_allocation& disk_data_representation::get_disk_table() const
{
  return *_disk_table;
}

// =============================================================================
// generate_disk_file_path
// =============================================================================

namespace memory {

std::string generate_disk_file_path(std::string_view base_path)
{
  // Use mkstemps to atomically create a unique file, safe across processes
  // sharing the same mount path.
  std::string pattern = std::filesystem::path(base_path) / "batch_XXXXXX.cucascade";
  int suffix_len      = 10;  // length of ".cucascade"
  int fd              = ::mkstemps(pattern.data(), suffix_len);
  if (fd < 0) { throw std::runtime_error("generate_disk_file_path: mkstemps failed"); }
  ::close(fd);
  return pattern;
}

}  // namespace memory

}  // namespace cucascade
