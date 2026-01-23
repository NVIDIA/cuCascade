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

}  // namespace cucascade
