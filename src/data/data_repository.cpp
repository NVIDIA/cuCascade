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
  uint64_t batch_id, std::optional<batch_state> target_state, size_t partition_idx)
{
  std::unique_lock<std::mutex> lock(_mutex);

  // Validate partition index
  if (partition_idx >= _data_batches.size()) {
    throw std::out_of_range("partition_idx out of range");
  }

  // If no target_state specified, just find and return a copy of the batch pointer
  if (!target_state.has_value()) {
    for (auto it = _data_batches[partition_idx].begin(); it != _data_batches[partition_idx].end(); ++it) {
      if ((*it)->get_batch_id() == batch_id) {
        return *it;  // Return a copy of the shared_ptr
      }
    }
    // Batch not found
    return nullptr;
  }

  // Target state specified - attempt state transition and wait if needed
  while (true) {
    // Search for the batch with the matching batch_id
    bool batch_found = false;
    for (auto it = _data_batches[partition_idx].begin(); it != _data_batches[partition_idx].end(); ++it) {
      if ((*it)->get_batch_id() == batch_id) {
        batch_found = true;
        data_batch* batch_ptr = it->get();
        bool can_transition   = false;

        switch (*target_state) {
          case batch_state::task_created: can_transition = batch_ptr->try_to_create_task(); break;
          case batch_state::processing:
            throw std::runtime_error(
              "get_data_batch_by_id cannot transition directly to processing; "
              "use pop_data_batch with task_created and call try_to_lock_for_processing() on the batch");
          case batch_state::in_transit:
            can_transition = batch_ptr->try_to_lock_for_in_transit();
            break;
          case batch_state::idle:
            // Cannot transition to idle via get - idle is a terminal state
            can_transition = false;
            break;
        }

        if (can_transition) {
          return *it;  // Return a copy of the shared_ptr
        }

        // Batch found but cannot transition - wait for state change
        break;
      }
    }

    // If batch was not found, return nullptr
    if (!batch_found) {
      return nullptr;
    }

    // Batch exists but cannot transition - wait for state changes
    _cv.wait(lock);
  }
}

// Explicit specialization of get_data_batch_by_id for unique_ptr (not supported)
template <>
std::unique_ptr<data_batch> idata_repository<std::unique_ptr<data_batch>>::get_data_batch_by_id(
  uint64_t /*batch_id*/, std::optional<batch_state> /*target_state*/, size_t /*partition_idx*/)
{
  throw std::runtime_error(
    "get_data_batch_by_id is not supported for unique_ptr repositories. "
    "Use pop_data_batch to move ownership instead.");
}

}  // namespace cucascade
