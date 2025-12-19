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

#include "data_batch.hpp"

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace cucascade {

/**
 * @brief Abstract interface for managing collections of data_batch objects within a pipeline.
 *
 * idata_repository defines the contract for storing, retrieving, and managing data batches
 * within a specific pipeline. Different implementations can provide various storage strategies,
 * such as:
 * - FIFO (First In, First Out) repositories for streaming data
 * - LRU (Least Recently Used) repositories for caching scenarios
 * - Priority-based repositories for workload-aware scheduling
 *
 * The repository is responsible for:
 * - Managing the lifecycle of data_batch objects through smart pointers
 * - Implementing eviction policies when memory pressure occurs
 * - Providing downgrade candidates for memory tier management
 * - Thread-safe access to shared data structures
 *
 * @tparam PtrType The smart pointer type used to manage data_batch lifecycle.
 *                 Typically std::shared_ptr<data_batch> or std::unique_ptr<data_batch>.
 *
 * @note Implementations must be thread-safe as multiple threads may access
 *       the repository concurrently during query execution.
 */
template <typename PtrType>
class idata_repository {
 public:
  /**
   * @brief Virtual destructor for proper cleanup of derived classes.
   */
  virtual ~idata_repository() = default;

  /**
   * @brief Add a new data batch to this repository.
   *
   * The repository takes ownership of the data_batch (for unique_ptr) or shares
   * ownership (for shared_ptr) and will manage its lifecycle according to the
   * implementation's storage policy.
   *
   * @param batch Smart pointer to the data_batch to add (ownership transferred/shared)
   *
   * @note Thread-safe operation protected by internal mutex
   */
  virtual void add_data_batch(PtrType batch)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _data_batches.push_back(std::move(batch));
  }

  /**
   * @brief Remove and return a data batch from this repository according to eviction policy.
   *
   * The specific data batch returned depends on the implementation's eviction strategy:
   * - FIFO: Returns the oldest batch
   * - LRU: Returns the least recently used batch
   * - Priority: Returns the lowest priority batch
   *
   * This method attempts to lock the batch for processing before returning it.
   * If the batch cannot be locked (e.g., it's being downgraded), it returns an
   * empty pair.
   *
   * @return std::pair<PtrType, data_batch_processing_handle> The data batch and a processing
   *         handle that will decrement the processing count when destroyed. Returns empty
   *         pair if repository is empty or batch cannot be locked for processing.
   *
   * @note Thread-safe operation protected by internal mutex
   */
  virtual std::pair<PtrType, data_batch_processing_handle> pull_data_batch()
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_data_batches.empty()) { return {PtrType{}, data_batch_processing_handle{}}; }

    auto batch = std::move(_data_batches.front());
    _data_batches.erase(_data_batches.begin());

    // Get raw pointer for the handle (works for both shared_ptr and unique_ptr)
    data_batch* batch_ptr = batch.get();

    // Try to lock for processing
    if (batch_ptr && batch_ptr->try_to_lock_for_processing()) {
      return {std::move(batch), data_batch_processing_handle{batch_ptr}};
    }

    // Could not lock for processing - put it back and return empty
    _data_batches.insert(_data_batches.begin(), std::move(batch));
    return {PtrType{}, data_batch_processing_handle{}};
  }

 protected:
  std::mutex _mutex;                   ///< Mutex for thread-safe access to repository operations
  std::vector<PtrType> _data_batches;  ///< Container for data batch pointers
};

using shared_data_repository = idata_repository<std::shared_ptr<data_batch>>;
using unique_data_repository = idata_repository<std::unique_ptr<data_batch>>;

}  // namespace cucascade
