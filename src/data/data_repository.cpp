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

#include <cucascade/data/data_repository.hpp>

namespace cucascade {

// Explicit template instantiations for smart pointer types
template class idata_repository<std::shared_ptr<data_batch>>;
template class idata_repository<std::unique_ptr<data_batch>>;

// Explicit specialization of get_data_batch_by_id for shared_ptr (copies the pointer)
template <>
std::shared_ptr<data_batch> idata_repository<std::shared_ptr<data_batch>>::get_data_batch_by_id(
  uint64_t batch_id, size_t partition_idx)
{
  std::unique_lock<std::mutex> lock(_mutex);

  if (partition_idx >= _data_batches.size()) {
    throw std::out_of_range("partition_idx out of range");
  }

  for (auto it = _data_batches[partition_idx].begin(); it != _data_batches[partition_idx].end();
       ++it) {
    if ((*it)->get_batch_id() == batch_id) {
      return *it;  // Return a copy of the shared_ptr
    }
  }

  return nullptr;
}

// Explicit specialization of get_data_batch_by_id for unique_ptr (not supported)
template <>
std::unique_ptr<data_batch> idata_repository<std::unique_ptr<data_batch>>::get_data_batch_by_id(
  uint64_t /*batch_id*/, size_t /*partition_idx*/)
{
  throw std::runtime_error(
    "get_data_batch_by_id is not supported for unique_ptr repositories. "
    "Use pop_data_batch to move ownership instead.");
}

}  // namespace cucascade
