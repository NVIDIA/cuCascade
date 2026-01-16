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
#include <optional>
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
   * @brief Default constructor - initializes with one empty partition.
   */
  idata_repository() : _data_batches(1) {}

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
   * @param partition_idx Index of the partition to add the batch to (default: 0)
   *
   * @note Thread-safe operation protected by internal mutex
   */
  virtual void add_data_batch(PtrType batch, size_t partition_idx = 0)
  {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      // Ensure partition exists
      if (batch) { batch->set_state_change_cv(&_cv); }
      
      if (static_cast<std::size_t>(partition_idx) >= _data_batches.size()) {
        _data_batches.resize(partition_idx + 1);
      }
      _data_batches[partition_idx].push_back(std::move(batch));
    }
    _cv.notify_all();
  }

  /**
   * @brief Notify waiting threads that a batch state may have changed.
   *
   * Call this method when a batch's state changes outside the repository
   * (e.g., after releasing a processing handle) to wake up threads blocked
   * in pop_data_batch().
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
   * - processing: Not allowed here; an exception is thrown. Callers should pop with
   *   task_created and then manually call try_to_lock_for_processing() on the returned batch.
   * - in_transit: Calls try_to_lock_for_in_transit() on the batch. Succeeds only
   *   for idle batches with no active processing.
   *
   * If no batch can transition to the target state, this method blocks until either:
   * - A new batch is added to the repository
   * - notify_state_change() is called
   *
   * @param target_state The state to transition the batch to (must not be processing)
   * @param partition_idx Index of the partition to pop from (default: 0)
   * @return PtrType The data batch that was successfully transitioned, or nullptr
   *         if the repository is empty after waiting.
   *
   * @note Thread-safe operation protected by internal mutex and condition variable
   * @throws std::runtime_error if target_state is batch_state::processing
   * @throws std::out_of_range if partition_idx is out of range
   */
  virtual PtrType pop_data_batch(batch_state target_state, size_t partition_idx = 0)
  {
    std::unique_lock<std::mutex> lock(_mutex);

    // Validate partition index
    if (partition_idx >= _data_batches.size()) {
      throw std::out_of_range("partition_idx out of range");
    }

    while (true) {
      if (_data_batches[partition_idx].empty()) { return PtrType{}; }

      // Search for a batch that can transition to the target state
      for (auto it = _data_batches[partition_idx].begin(); it != _data_batches[partition_idx].end(); ++it) {
        data_batch* batch_ptr = it->get();
        bool can_transition   = try_transition_batch_for_pop(batch_ptr, target_state);

        if (can_transition) {
          auto batch = std::move(*it);
          _data_batches[partition_idx].erase(it);
          return batch;
        }
      }

      // No batch could transition - wait for new batches or state changes
      _cv.wait(lock);
    }
  }

  /**
   * @brief Remove and return a data batch by its batch ID, optionally with state transition.
   *
   * Searches for a batch with the specified batch_id within the given partition.
   * 
   * If target_state is std::nullopt:
   * - Returns the batch immediately if found, without state transition
   * 
   * If target_state has a value:
   * - Attempts to transition the batch to the target state
   * - If the batch exists but cannot transition, this method blocks until either:
   *   - The batch can successfully transition to the target state
   *   - notify_state_change() is called
   *   - A new batch is added to the repository
   *
   * State transition behavior follows the same rules as pop_data_batch(batch_state):
   * - task_created: Calls try_to_create_task() on the batch
   * - processing: Not allowed; throws an exception
   * - in_transit: Calls try_to_lock_for_in_transit() on the batch
   *
   * @param batch_id The unique identifier of the batch to retrieve
   * @param target_state Optional state to transition the batch to (std::nullopt to skip state check)
   * @param partition_idx Index of the partition to search (default: 0)
   * @return PtrType The data batch with the matching batch_id, or nullptr if not found
   *
   * @note Thread-safe operation protected by internal mutex and condition variable
   * @throws std::runtime_error if target_state is batch_state::processing
   * @throws std::out_of_range if partition_idx is out of range
   */
  virtual PtrType pop_data_batch_by_id(uint64_t batch_id, std::optional<batch_state> target_state, size_t partition_idx = 0)
  {
    std::unique_lock<std::mutex> lock(_mutex);

    // Validate partition index
    if (partition_idx >= _data_batches.size()) {
      throw std::out_of_range("partition_idx out of range");
    }

    // If no target_state specified, just find and return the batch
    if (!target_state.has_value()) {
      for (auto it = _data_batches[partition_idx].begin(); it != _data_batches[partition_idx].end(); ++it) {
        if (it->get()->get_batch_id() == batch_id) {
          auto batch = std::move(*it);
          _data_batches[partition_idx].erase(it);
          return batch;
        }
      }
      // Batch not found
      return PtrType{};
    }

    // Target state specified - attempt state transition and wait if needed
    while (true) {
      // Search for the batch with the matching batch_id
      bool batch_found = false;
      for (auto it = _data_batches[partition_idx].begin(); it != _data_batches[partition_idx].end(); ++it) {
        if (it->get()->get_batch_id() == batch_id) {
          batch_found = true;
          data_batch* batch_ptr = it->get();
          bool can_transition = try_transition_batch_for_pop(batch_ptr, *target_state);

          if (can_transition) {
            auto batch = std::move(*it);
            _data_batches[partition_idx].erase(it);
            return batch;
          }

          // Batch found but cannot transition - wait for state change
          break;
        }
      }

      // If batch was not found, return nullptr
      if (!batch_found) {
        return PtrType{};
      }

      // Batch exists but cannot transition - wait for state changes
      _cv.wait(lock);
    }
  }

  /**
   * @brief Get a copy of a data batch by its batch ID, optionally with state transition.
   *
   * Searches for a batch with the specified batch_id within the given partition
   * and returns a copy of the pointer (does not remove from repository).
   * 
   * If target_state is std::nullopt:
   * - Returns a copy of the batch pointer immediately if found, without state transition
   * 
   * If target_state has a value:
   * - Attempts to transition the batch to the target state
   * - If the batch exists but cannot transition, this method blocks until either:
   *   - The batch can successfully transition to the target state
   *   - notify_state_change() is called
   *   - A new batch is added to the repository
   *
   * State transition behavior follows the same rules as pop_data_batch(batch_state):
   * - task_created: Calls try_to_create_task() on the batch
   * - processing: Not allowed; throws an exception
   * - in_transit: Calls try_to_lock_for_in_transit() on the batch
   *
   * @param batch_id The unique identifier of the batch to retrieve
   * @param target_state Optional state to transition the batch to (std::nullopt to skip state check)
   * @param partition_idx Index of the partition to search (default: 0)
   * @return PtrType A copy of the data batch pointer with the matching batch_id, or nullptr if not found
   *
   * @note Thread-safe operation protected by internal mutex and condition variable
   * @note Only supported for shared_ptr repositories. Will throw for unique_ptr repositories.
   * @throws std::runtime_error if target_state is batch_state::processing or if called on unique_ptr repository
   * @throws std::out_of_range if partition_idx is out of range
   */
  virtual PtrType get_data_batch_by_id(uint64_t batch_id, std::optional<batch_state> target_state, size_t partition_idx = 0);

  /**
   * @brief Get all batch IDs from a partition.
   *
   * Retrieves the batch IDs of all batches currently stored in the specified partition.
   *
   * @param partition_idx Index of the partition to query (default: 0)
   * @return std::vector<uint64_t> Vector containing all batch IDs in the partition
   *
   * @note Thread-safe operation protected by internal mutex
   * @throws std::out_of_range if partition_idx is out of range
   */
  virtual std::vector<uint64_t> get_batch_ids(size_t partition_idx = 0) const
  {
    std::lock_guard<std::mutex> lock(_mutex);

    // Validate partition index
    if (partition_idx >= _data_batches.size()) {
      throw std::out_of_range("partition_idx out of range");
    }

    std::vector<uint64_t> batch_ids;
    batch_ids.reserve(_data_batches[partition_idx].size());

    for (const auto& batch : _data_batches[partition_idx]) {
      batch_ids.push_back(batch->get_batch_id());
    }

    return batch_ids;
  }

  /**
   * @brief Check if there are any data batches available in the repository.
   *
   * @param partition_idx Index of the partition to check (default: 0)
   * @return std::size_t The number of data batches currently stored in the specified partition.
   *
   * @note Thread-safe operation protected by internal mutex
   * @throws std::out_of_range if partition_idx is out of range
   */
  std::size_t size(size_t partition_idx = 0) const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (partition_idx >= _data_batches.size()) {
      throw std::out_of_range("partition_idx out of range");
    }
    return _data_batches[partition_idx].size();
  }

  /**
   * @brief Check if the repository partition is empty.
   *
   * @param partition_idx Index of the partition to check (default: 0)
   * @return true if the partition has no data batches, false otherwise.
   *
   * @note Thread-safe operation protected by internal mutex
   * @throws std::out_of_range if partition_idx is out of range
   */
  bool empty(size_t partition_idx = 0) const { return size(partition_idx) == 0; }

  /**
   * @brief Get the total number of data batches across all partitions.
   *
   * @return std::size_t The sum of data batches in all partitions.
   *
   * @note Thread-safe operation protected by internal mutex (locks only once)
   */
  std::size_t total_size() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    std::size_t total = 0;
    for (const auto& partition : _data_batches) {
      total += partition.size();
    }
    return total;
  }

  /**
   * @brief Check if all partitions in the repository are empty.
   *
   * @return true if all partitions have no data batches, false otherwise.
   *
   * @note Thread-safe operation protected by internal mutex (locks only once)
   */
  bool all_empty() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    for (const auto& partition : _data_batches) {
      if (!partition.empty()) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Get the number of partitions in the repository.
   *
   * @return std::size_t The number of partitions currently in the repository.
   *
   * @note Thread-safe operation protected by internal mutex
   */
  std::size_t num_partitions() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _data_batches.size();
  }

private:
  /**
   * @brief Helper function to attempt state transition on a batch when poping.
   *
   * @param batch_ptr Pointer to the batch to transition
   * @param target_state The state to transition to
   * @return bool True if the transition succeeded, false otherwise
   * @throws std::runtime_error if target_state is batch_state::processing
   */
  bool try_transition_batch_for_pop(data_batch* batch_ptr, batch_state target_state) const
  {
    switch (target_state) {
      case batch_state::task_created:
        return batch_ptr->try_to_create_task();
      case batch_state::processing:
        throw std::runtime_error(
          "Pop operation cannot transition directly to processing; "
          "Pop with task_created and call try_to_lock_for_processing() on the batch");
      case batch_state::in_transit:
        return batch_ptr->try_to_lock_for_in_transit();
      case batch_state::idle:
        // Cannot transition to idle via pop - idle is a terminal state
        return false;
    }
    return false;
  }


 protected:
  mutable std::mutex _mutex;           ///< Mutex for thread-safe access to repository operations
  std::condition_variable _cv;         ///< Condition variable for blocking pop operations
  std::vector<std::vector<PtrType>> _data_batches;  ///< Container for data batch pointers (partitioned)
};

using shared_data_repository = idata_repository<std::shared_ptr<data_batch>>;
using unique_data_repository = idata_repository<std::unique_ptr<data_batch>>;

}  // namespace cucascade
