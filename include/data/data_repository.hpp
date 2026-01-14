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

#include <condition_variable>
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
    {
      std::lock_guard<std::mutex> lock(_mutex);
      if (batch) { batch->set_state_change_cv(&_cv); }
      _data_batches.push_back(std::move(batch));
    }
    _cv.notify_all();
  }

  /**
   * @brief Notify waiting threads that a batch state may have changed.
   *
   * Call this method when a batch's state changes outside the repository
   * (e.g., after releasing a processing handle) to wake up threads blocked
   * in pull_data_batch().
   */
  void notify_state_change() { _cv.notify_all(); }

  /**
   * @brief Remove and return a data batch that can transition to the target state.
   *
   * Searches for a batch that can successfully transition to the specified target state.
   * The specific batch returned depends on the implementation's eviction strategy:
   * - FIFO: Searches from oldest to newest
   * - LRU: Searches from least recently used
   * - Priority: Searches from lowest priority
   *
   * State transition behavior:
   * - task_created: Calls try_to_create_task() on the batch. Succeeds for idle,
   *   task_created, or processing batches. Processing batches stay in processing state.
   * - processing: Not allowed here; an exception is thrown. Callers should pull with
   *   task_created and then manually call try_to_lock_for_processing() on the returned batch.
   * - in_transit: Calls try_to_lock_for_in_transit() on the batch. Succeeds only
   *   for idle batches with no active processing.
   *
   * If no batch can transition to the target state, this method blocks until either:
   * - A new batch is added to the repository
   * - notify_state_change() is called
   *
   * @param target_state The state to transition the batch to (must not be processing)
   * @return PtrType The data batch that was successfully transitioned, or nullptr
   *         if the repository is empty after waiting.
   *
   * @note Thread-safe operation protected by internal mutex and condition variable
   * @throws std::runtime_error if target_state is batch_state::processing
   */
  virtual PtrType pull_data_batch(batch_state target_state)
  {
    std::unique_lock<std::mutex> lock(_mutex);

    while (true) {
      if (_data_batches.empty()) { return PtrType{}; }

      // Search for a batch that can transition to the target state
      for (auto it = _data_batches.begin(); it != _data_batches.end(); ++it) {
        data_batch* batch_ptr = it->get();
        bool can_transition   = false;

        switch (target_state) {
          case batch_state::task_created: can_transition = batch_ptr->try_to_create_task(); break;
          case batch_state::processing:
            throw std::runtime_error(
              "pull_data_batch cannot transition directly to processing; "
              "pull with task_created and call try_to_lock_for_processing() on the batch");
          case batch_state::in_transit:
            can_transition = batch_ptr->try_to_lock_for_in_transit();
            break;
          case batch_state::idle:
            // Cannot transition to idle via pull - idle is a terminal state
            can_transition = false;
            break;
        }

        if (can_transition) {
          auto batch = std::move(*it);
          _data_batches.erase(it);
          return batch;
        }
      }

      // No batch could transition - wait for new batches or state changes
      _cv.wait(lock);
    }
  }

  /**
   * @brief Check if there are any data batches available in the repository.
   *
   * @return std::size_t The number of data batches currently stored in the repository.
   *
   * @note Thread-safe operation protected by internal mutex
   */
  std::size_t size() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _data_batches.size();
  }

  /**
   * @brief Check if the repository is empty.
   *
   * @return true if the repository has no data batches, false otherwise.
   *
   * @note Thread-safe operation protected by internal mutex
   */
  bool empty() const { return size() == 0; }

 protected:
  mutable std::mutex _mutex;           ///< Mutex for thread-safe access to repository operations
  std::condition_variable _cv;         ///< Condition variable for blocking pull operations
  std::vector<PtrType> _data_batches;  ///< Container for data batch pointers
};

using shared_data_repository = idata_repository<std::shared_ptr<data_batch>>;
using unique_data_repository = idata_repository<std::unique_ptr<data_batch>>;

}  // namespace cucascade
