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

#include <cucascade/data/data_batch.hpp>

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

      if (static_cast<std::size_t>(partition_idx) >= _data_batches.size()) {
        _data_batches.resize(partition_idx + 1);
      }
      _data_batches[partition_idx].push_back(std::move(batch));
    }
  }

  /**
   * @brief Remove and return the next data batch from the repository.
   *
   * Returns the first batch in the partition regardless of its state.
   * If the partition is empty, returns a null pointer.
   *
   * @param partition_idx Index of the partition to pop from (default: 0)
   * @return PtrType The next data batch, or nullptr if the partition is empty.
   *
   * @note Thread-safe operation protected by internal mutex
   * @throws std::out_of_range if partition_idx is out of range
   */
  virtual PtrType pop_next_data_batch(size_t partition_idx = 0)
  {
    std::unique_lock<std::mutex> lock(_mutex);

    if (partition_idx >= _data_batches.size()) {
      throw std::out_of_range("partition_idx out of range");
    }

    if (_data_batches[partition_idx].empty()) { return PtrType{}; }

    auto batch = std::move(_data_batches[partition_idx].front());
    _data_batches[partition_idx].erase(_data_batches[partition_idx].begin());
    return batch;
  }

  /**
   * @brief Remove and return a data batch by its batch ID.
   *
   * Searches for a batch with the specified batch_id within the given partition.
   * Returns the batch if found, nullptr otherwise.
   *
   * @param batch_id The unique identifier of the batch to retrieve
   * @param partition_idx Index of the partition to search (default: 0)
   * @return PtrType The data batch with the matching batch_id, or nullptr if not found
   *
   * @note Thread-safe operation protected by internal mutex
   * @throws std::out_of_range if partition_idx is out of range
   */
  virtual PtrType pop_data_batch_by_id(uint64_t batch_id, size_t partition_idx = 0)
  {
    std::unique_lock<std::mutex> lock(_mutex);

    if (partition_idx >= _data_batches.size()) {
      throw std::out_of_range("partition_idx out of range");
    }

    for (auto it = _data_batches[partition_idx].begin(); it != _data_batches[partition_idx].end();
         ++it) {
      if (it->get()->get_batch_id() == batch_id) {
        auto batch = std::move(*it);
        _data_batches[partition_idx].erase(it);
        return batch;
      }
    }

    return PtrType{};
  }

  /**
   * @brief Get a copy of a data batch by its batch ID (does not remove from repository).
   *
   * Searches for a batch with the specified batch_id within the given partition
   * and returns a copy of the pointer.
   *
   * @param batch_id The unique identifier of the batch to retrieve
   * @param partition_idx Index of the partition to search (default: 0)
   * @return PtrType A copy of the data batch pointer with the matching batch_id, or nullptr
   *
   * @note Thread-safe operation protected by internal mutex
   * @note Only supported for shared_ptr repositories. Will throw for unique_ptr repositories.
   * @throws std::runtime_error if called on unique_ptr repository
   * @throws std::out_of_range if partition_idx is out of range
   */
  virtual PtrType get_data_batch_by_id(uint64_t batch_id, size_t partition_idx = 0);

  /**
   * @brief Get all batch IDs from a partition.
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
   * @brief Get the number of data batches in a partition.
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
      if (!partition.empty()) { return false; }
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

 protected:
  mutable std::mutex _mutex;  ///< Mutex for thread-safe access to repository operations
  std::vector<std::vector<PtrType>>
    _data_batches;  ///< Container for data batch pointers (partitioned)
};

using shared_data_repository = idata_repository<std::shared_ptr<data_batch>>;
using unique_data_repository = idata_repository<std::unique_ptr<data_batch>>;

}  // namespace cucascade
