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

#include <cucascade/data/cpu_data_representation.hpp>

#include <cstring>
#include <stdexcept>

namespace cucascade {

host_table_representation::host_table_representation(
  std::unique_ptr<cucascade::memory::host_table_allocation> host_table,
  cucascade::memory::memory_space* memory_space)
  : idata_representation(*memory_space), _host_table(std::move(host_table))
{
}

std::size_t host_table_representation::get_size_in_bytes() const { return _host_table->data_size; }

const std::unique_ptr<cucascade::memory::host_table_allocation>&
host_table_representation::get_host_table() const
{
  return _host_table;
}

std::unique_ptr<idata_representation> host_table_representation::clone()
{
  // Get the host memory resource from the memory space
  auto* host_mr = get_memory_space().get_memory_resource_of<memory::Tier::HOST>();
  if (!host_mr) {
    throw std::runtime_error("Cannot clone host_table_representation: no host memory resource");
  }

  // Allocate new blocks for the copy
  auto allocation_copy = host_mr->allocate_multiple_blocks(_host_table->data_size);

  // Copy data block by block
  const auto& src_blocks = _host_table->allocation->get_blocks();
  auto dst_blocks        = allocation_copy->get_blocks();
  std::size_t remaining  = _host_table->data_size;
  std::size_t block_size = _host_table->allocation->block_size();

  for (std::size_t i = 0; i < src_blocks.size() && remaining > 0; ++i) {
    std::size_t copy_size = std::min(remaining, block_size);
    std::memcpy(dst_blocks[i], src_blocks[i], copy_size);
    remaining -= copy_size;
  }

  // Clone the metadata
  auto metadata_copy = std::make_unique<std::vector<uint8_t>>(*_host_table->metadata);

  // Create the new host_table_allocation
  auto host_table_copy = std::make_unique<memory::host_table_allocation>(
    std::move(allocation_copy), std::move(metadata_copy), _host_table->data_size);

  return std::make_unique<host_table_representation>(std::move(host_table_copy),
                                                     &get_memory_space());
}

}  // namespace cucascade
