/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cucascade/data/remote_data_representation.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace cucascade {

remote_data_representation::remote_data_representation(
  std::unique_ptr<memory::disk_table_allocation> object_table,
  memory::memory_space& memory_space,
  std::shared_ptr<detail::remote_object const> object)
  : idata_representation(memory_space),
    _object_table(std::move(object_table)),
    _object(std::move(object))
{
  if (_object_table == nullptr) {
    throw std::invalid_argument("remote_data_representation: the object table must not be null");
  }
  if (memory_space.get_tier() != memory::Tier::REMOTE) {
    throw std::invalid_argument(
      "remote_data_representation: the memory space must be a REMOTE-tier space");
  }
}

// The handle's deleter (installed by the cucascade-cudf layer) removes the object; it is
// noexcept by contract.
remote_data_representation::~remote_data_representation() noexcept = default;

std::size_t remote_data_representation::get_size_in_bytes() const
{
  return _object_table->data_size;
}

std::size_t remote_data_representation::get_uncompressed_data_size_in_bytes() const
{
  return _object_table->data_size;
}

std::unique_ptr<idata_representation> remote_data_representation::clone(
  [[maybe_unused]] ::cuda::stream_ref stream) const
{
  CUCASCADE_FAIL("remote_data_representation does not support clone");
}

const memory::disk_table_allocation& remote_data_representation::get_object_table() const
{
  return *_object_table;
}

const std::string& remote_data_representation::get_object_uri() const
{
  return _object_table->file_path;
}

}  // namespace cucascade
