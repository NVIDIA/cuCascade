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

#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cucascade {

/** @brief A payload and its optional logical position within a repository partition. */
using positioned_data_batch = std::pair<std::shared_ptr<data_batch>, std::optional<size_t>>;

/** @brief Uniform batch-positioning mode, established by the first successful insertion. */
enum class data_repository_ordering { ORDERED, UNORDERED, UNKNOWN };

/** @brief Whether increasing partition indices define logical sequence. */
enum class data_repository_partition_ordering { ORDERED, UNORDERED };

/**
 * @brief Thread-safe partitioned storage of shared batches and optional logical positions.
 *
 * Batch positioning and partition ordering are independent. Positions are producer assertions
 * scoped to a partition; the repository checks structure, not sortedness or SQL comparators.
 * FIFO retrieval always follows arrival order, including for positioned batches.
 *
 * The first successful insertion establishes a uniform batch mode across all partitions.
 * Draining never resets that mode. UNKNOWN means no insertion has established a mode;
 * neither UNKNOWN nor emptiness indicates producer completion.
 *
 * @note Ordered consumption requires successful producer completion and a coordinated consumer.
 * Position enumeration is a snapshot, not an atomic claim or completion signal.
 * @note Derived classes must preserve the mode and position invariants under _mutex. Protected
 * storage contains positioned_data_batch pairs; subclasses accessing it must migrate explicitly.
 */
class data_repository {
 public:
  /** @brief Initialize one empty partition with unordered partition indices. */
  data_repository() : data_repository(data_repository_partition_ordering::UNORDERED) {}

  /**
   * @brief Initialize one empty partition with an independent partition-order declaration.
   * @param partition_ordering Whether partition indices define sequence; immutable after creation.
   */
  explicit data_repository(data_repository_partition_ordering partition_ordering)
    : _data_batches(1), _partition_ordering(partition_ordering)
  {
  }

  /**
   * @brief Virtual destructor for proper cleanup of derived classes.
   */
  virtual ~data_repository() = default;

  /**
   * @brief Add a new data batch to this repository.
   *
   * The repository shares ownership of the data_batch and manages its lifecycle
   * according to the implementation's storage policy.
   *
   * @param batch Shared pointer to the data_batch to add
   * @param partition_idx Index of the partition to add the batch to (default: 0)
   *
   * @note Thread-safe operation protected by internal mutex
   * @throws std::logic_error if batch mode is ORDERED
   * @throws std::out_of_range if partition_idx cannot be represented as a partition count
   */
  virtual void add_data_batch(std::shared_ptr<data_batch> batch, size_t partition_idx = 0)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    require_unpositioned_access();
    insert_data_batch({std::move(batch), std::nullopt}, partition_idx);
    _ordering = data_repository_ordering::UNORDERED;
  }

  /**
   * @brief Add a batch at a logical position, establishing ORDERED mode on success.
   *
   * Positions may arrive in any order and may have gaps. Duplicate resident positions in one
   * partition are rejected. Reuse after consumption is also invalid, but is not tracked.
   * Failed insertion leaves contents, partition count, and batch mode unchanged.
   *
   * @param batch Shared payload to store
   * @param position Logical position within the partition
   * @param partition_idx Partition to grow or insert into (default: 0)
   * @throws std::logic_error if batch mode is UNORDERED or the position is already resident
   * @throws std::out_of_range if partition_idx cannot be represented as a partition count
   */
  virtual void add_data_batch_at_position(std::shared_ptr<data_batch> batch,
                                          size_t position,
                                          size_t partition_idx = 0)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_ordering == data_repository_ordering::UNORDERED) {
      throw std::logic_error("positioned insertion requires an UNKNOWN or ORDERED repository");
    }
    if (partition_idx < _data_batches.size()) {
      for (const auto& entry : _data_batches[partition_idx]) {
        if (entry.second == position) {
          throw std::logic_error("duplicate resident batch position in partition");
        }
      }
    }
    insert_data_batch({std::move(batch), position}, partition_idx);
    _ordering = data_repository_ordering::ORDERED;
  }

  /** @brief Get the synchronized batch-positioning mode, independent of partition ordering. */
  [[nodiscard]] data_repository_ordering get_ordering() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _ordering;
  }

  /** @brief Get the immutable partition-order declaration, independent of batch positioning. */
  [[nodiscard]] data_repository_partition_ordering get_partition_ordering() const
  {
    return _partition_ordering;
  }

  /**
   * @brief Remove a batch without retaining its logical position.
   * @return Payload, or nullptr if absent; valid in every ordering mode.
   * @note Use pop_next_data_batch_with_position() when the consumer must preserve ordering.
   * @throws std::out_of_range if partition_idx is out of range
   */
  virtual std::shared_ptr<data_batch> pop_next_data_batch(size_t partition_idx = 0)
  {
    return pop_next_data_batch_with_position(partition_idx).first;
  }

  /**
   * @brief Remove a batch without retaining its logical position.
   * @return Payload, or nullptr if absent; valid in every ordering mode.
   * @note Use pop_data_batch_by_id_with_position() when the consumer must preserve ordering.
   * @throws std::out_of_range if partition_idx is out of range
   */
  virtual std::shared_ptr<data_batch> pop_data_batch_by_id(uint64_t batch_id,
                                                           size_t partition_idx = 0)
  {
    return pop_data_batch_by_id_with_position(batch_id, partition_idx).first;
  }

  /**
   * @brief Inspect a batch without retaining its logical position.
   * @return Payload, or nullptr if absent; valid in every ordering mode.
   * @note Use get_data_batch_by_id_with_position() when the consumer must preserve ordering.
   * @throws std::out_of_range if partition_idx is out of range
   */
  virtual std::shared_ptr<data_batch> get_data_batch_by_id(uint64_t batch_id,
                                                           size_t partition_idx = 0) const
  {
    return get_data_batch_by_id_with_position(batch_id, partition_idx).first;
  }

  /**
   * @brief Remove the next batch and its position in arrival order, in any batch mode.
   * @param partition_idx Partition to pop from (default: 0)
   * @return Payload and optional position, or {nullptr, nullopt} if empty
   * @throws std::out_of_range if partition_idx is out of range
   */
  virtual positioned_data_batch pop_next_data_batch_with_position(size_t partition_idx = 0)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    require_partition(partition_idx);
    return pop_next_data_batch_unlocked(partition_idx);
  }

  /**
   * @brief Remove a batch by ID with its optional position, in any batch mode.
   * @param batch_id ID to search for
   * @param partition_idx Partition to search (default: 0)
   * @return Payload and optional position, or {nullptr, nullopt} on a miss
   * @throws std::out_of_range if partition_idx is out of range
   */
  virtual positioned_data_batch pop_data_batch_by_id_with_position(uint64_t batch_id,
                                                                   size_t partition_idx = 0)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    require_partition(partition_idx);
    return pop_data_batch_by_id_unlocked(batch_id, partition_idx);
  }

  /**
   * @brief Inspect a batch by ID with its optional position without removing it, in any mode.
   * @param batch_id ID to search for
   * @param partition_idx Partition to search (default: 0)
   * @return Payload and optional position, or {nullptr, nullopt} on a miss
   * @throws std::out_of_range if partition_idx is out of range
   */
  virtual positioned_data_batch get_data_batch_by_id_with_position(uint64_t batch_id,
                                                                   size_t partition_idx = 0) const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    require_partition(partition_idx);
    for (const auto& entry : _data_batches[partition_idx]) {
      if (entry.first->get_batch_id() == batch_id) { return entry; }
    }
    return {};
  }

  /**
   * @brief Inspect a batch at a partition-local position without removing it.
   * @param position Logical position to search for
   * @param partition_idx Partition to search (default: 0)
   * @return Payload, or nullptr if absent
   * @throws std::out_of_range if partition_idx is out of range
   * @throws std::logic_error if batch mode is UNKNOWN or UNORDERED
   */
  virtual std::shared_ptr<data_batch> get_data_batch_by_position(size_t position,
                                                                 size_t partition_idx = 0) const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    require_partition(partition_idx);
    require_positioned_access();
    for (const auto& entry : _data_batches[partition_idx]) {
      if (entry.second == position) { return entry.first; }
    }
    return nullptr;
  }

  /**
   * @brief Remove a batch at a partition-local position.
   * @param position Logical position to search for
   * @param partition_idx Partition to search (default: 0)
   * @return Payload, or nullptr if absent
   * @throws std::out_of_range if partition_idx is out of range
   * @throws std::logic_error if batch mode is UNKNOWN or UNORDERED
   */
  virtual std::shared_ptr<data_batch> pop_data_batch_by_position(size_t position,
                                                                 size_t partition_idx = 0)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    require_partition(partition_idx);
    require_positioned_access();
    auto& partition = _data_batches[partition_idx];
    for (auto it = partition.begin(); it != partition.end(); ++it) {
      if (it->second == position) {
        auto batch = std::move(it->first);
        partition.erase(it);
        return batch;
      }
    }
    return nullptr;
  }

  /**
   * @brief Get a sorted snapshot of resident positions in one partition.
   * @param partition_idx Partition to enumerate (default: 0)
   * @return Positions in increasing order; gaps and empty partitions are allowed
   * @throws std::out_of_range if partition_idx is out of range
   * @throws std::logic_error if batch mode is UNKNOWN or UNORDERED
   * @warning This snapshot is only complete after all producer operators have finished.
   * Enumeration does not claim batches; consumers must coordinate because concurrent mutation
   * may invalidate the snapshot.
   */
  virtual std::vector<size_t> get_positions(size_t partition_idx = 0) const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    require_partition(partition_idx);
    require_positioned_access();
    std::vector<size_t> positions;
    positions.reserve(_data_batches[partition_idx].size());
    for (const auto& entry : _data_batches[partition_idx]) {
      positions.push_back(*entry.second);
    }
    std::sort(positions.begin(), positions.end());
    return positions;
  }

  virtual std::vector<size_t> get_batch_positions(size_t partition_idx = 0) const
  {
    return get_positions(partition_idx);
  }

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

    for (const auto& entry : _data_batches[partition_idx]) {
      batch_ids.push_back(entry.first->get_batch_id());
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

  /**
   * @brief Grow the number of partitions in the repository.
   *
   * Growing only: the new count must be strictly greater than the current count.
   * Shrinking is not supported because it could discard batches that are already in
   * flight.
   *
   * @param new_num_partitions The desired number of partitions. Must be strictly
   *        greater than 0 and strictly greater than the current number of partitions.
   *
   * @note Thread-safe operation protected by internal mutex
   * @throws std::invalid_argument if @p new_num_partitions is 0
   * @throws std::invalid_argument if @p new_num_partitions is not strictly greater
   *         than the current number of partitions
   */
  void set_num_partitions(std::size_t new_num_partitions)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (new_num_partitions == 0) {
      throw std::invalid_argument(
        "data_repository::set_num_partitions: new number of partitions must be greater than 0");
    }
    if (new_num_partitions <= _data_batches.size()) {
      throw std::invalid_argument(
        "data_repository::set_num_partitions: new number of partitions (" +
        std::to_string(new_num_partitions) + ") must be greater than the current number (" +
        std::to_string(_data_batches.size()) + ")");
    }
    _data_batches.resize(new_num_partitions);
  }

 protected:
  mutable std::mutex _mutex;  ///< Mutex for thread-safe access to repository operations
  std::vector<std::vector<positioned_data_batch>>
    _data_batches;  ///< Partitioned arrival-order storage
  data_repository_ordering _ordering = data_repository_ordering::UNKNOWN;  ///< Guarded by _mutex

 private:
  const data_repository_partition_ordering _partition_ordering;

  // Helpers require the caller to hold _mutex.
  void require_partition(size_t partition_idx) const
  {
    if (partition_idx >= _data_batches.size()) {
      throw std::out_of_range("partition_idx out of range");
    }
  }

  void require_unpositioned_access() const
  {
    if (_ordering == data_repository_ordering::ORDERED) {
      throw std::logic_error("unpositioned access is not allowed on an ORDERED repository");
    }
  }

  void require_positioned_access() const
  {
    if (_ordering != data_repository_ordering::ORDERED) {
      throw std::logic_error("position access requires an ORDERED repository");
    }
  }

  void insert_data_batch(positioned_data_batch entry, size_t partition_idx)
  {
    if (partition_idx < _data_batches.size()) {
      _data_batches[partition_idx].push_back(std::move(entry));
      return;
    }
    if (partition_idx == std::numeric_limits<size_t>::max()) {
      throw std::out_of_range("partition_idx cannot be represented as a partition count");
    }
    // Prepare the inner allocation first. After resize succeeds, moving it cannot throw.
    std::vector<positioned_data_batch> partition;
    partition.push_back(std::move(entry));
    _data_batches.resize(partition_idx + 1);
    _data_batches[partition_idx] = std::move(partition);
  }

  positioned_data_batch pop_next_data_batch_unlocked(size_t partition_idx)
  {
    auto& partition = _data_batches[partition_idx];
    if (partition.empty()) { return {}; }
    auto entry = std::move(partition.front());
    partition.erase(partition.begin());
    return entry;
  }

  positioned_data_batch pop_data_batch_by_id_unlocked(uint64_t batch_id, size_t partition_idx)
  {
    auto& partition = _data_batches[partition_idx];
    for (auto it = partition.begin(); it != partition.end(); ++it) {
      if (it->first->get_batch_id() == batch_id) {
        auto entry = std::move(*it);
        partition.erase(it);
        return entry;
      }
    }
    return {};
  }
};

using shared_data_repository = data_repository;

}  // namespace cucascade
