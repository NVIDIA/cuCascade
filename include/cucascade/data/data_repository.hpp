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
#include <list>
#include <map>
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

/** @brief The requested operation conflicts with the established batch-positioning mode. */
class repository_ordering_error : public std::logic_error {
 public:
  using std::logic_error::logic_error;
};

/** @brief The requested position is already occupied in this partition. */
class repository_duplicate_error : public std::logic_error {
 public:
  using std::logic_error::logic_error;
};

/**
 * @brief Thread-safe partitioned storage of shared batches and optional logical positions.
 *
 * Positions are producer assertions
 * scoped to a partition; the repository checks structure, not sortedness or SQL comparators.
 * FIFO retrieval always follows arrival order, including for positioned batches.
 *
 * The first successful insertion establishes a uniform batch mode across all partitions.
 * Draining never resets that mode. UNKNOWN means no insertion has established a mode;
 * neither UNKNOWN nor emptiness indicates producer completion.
 *
 * @note Ordered consumption requires successful producer completion and a coordinated consumer.
 * Position enumeration is a snapshot, not an atomic claim or completion signal.
 * @note Public methods acquire _mutex before calling protected virtual _unlocked hooks.
 * Overrides must call base _unlocked implementations, not public methods that acquire the mutex.
 * Derived classes must preserve the mode and index invariants while holding _mutex.
 */
class data_repository {
 public:
  /** @brief Initialize one empty partition; the first successful insertion establishes its mode. */
  data_repository() : _data_batches(1) {}

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
   * @throws repository_ordering_error if batch mode is ORDERED
   * @throws std::out_of_range if partition_idx cannot be represented as a partition count
   * @throws std::invalid_argument if batch is null
   * @throws std::length_error or std::bad_alloc if storage cannot be allocated
   * @note The base implementation preserves caller ownership and repository state on failure.
   */
  void add_data_batch(const std::shared_ptr<data_batch>& batch, size_t partition_idx = 0)
  {
    if (!batch) { throw std::invalid_argument("batch must not be null"); }
    std::lock_guard<std::mutex> lock(_mutex);
    require_unpositioned_access();
    add_data_batch_unlocked(batch, partition_idx);
  }

  /**
   * @brief Add a batch at a logical position, establishing ORDERED mode on success.
   *
   * Positions may arrive in any order and may have gaps. Duplicate resident positions in one
   * partition are rejected. Removing a batch frees its position for reuse.
   * Failed insertion leaves contents, partition count, and batch mode unchanged.
   *
   * @param batch Shared payload to store
   * @param position Logical position within the partition
   * @param partition_idx Partition to grow or insert into (default: 0)
   * @throws repository_ordering_error if batch mode is UNORDERED
   * @throws repository_duplicate_error if the position is already resident
   * @throws std::out_of_range if partition_idx cannot be represented as a partition count
   * @throws std::invalid_argument if batch is null
   * @throws std::length_error or std::bad_alloc if storage cannot be allocated
   * @note The base implementation preserves caller ownership and repository state on failure.
   */
  void add_data_batch_at_position(const std::shared_ptr<data_batch>& batch,
                                  size_t position,
                                  size_t partition_idx = 0)
  {
    if (!batch) { throw std::invalid_argument("batch must not be null"); }
    std::lock_guard<std::mutex> lock(_mutex);
    if (_ordering == data_repository_ordering::UNORDERED) {
      throw repository_ordering_error(
        "positioned insertion requires an UNKNOWN or ORDERED repository");
    }
    if (partition_idx < _data_batches.size() &&
        _data_batches[partition_idx].positions.contains(position)) {
      throw repository_duplicate_error("duplicate resident batch position in partition");
    }
    add_data_batch_at_position_unlocked(batch, position, partition_idx);
  }

  /** @brief Get the synchronized batch-positioning mode. */
  [[nodiscard]] data_repository_ordering get_ordering() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _ordering;
  }

  /**
   * @brief Remove a batch without retaining its logical position.
   * @return Payload, or nullptr if absent; valid in every ordering mode.
   * @note Use pop_next_data_batch_with_position() when the consumer must preserve ordering.
   * @throws std::out_of_range if partition_idx is out of range
   */
  std::shared_ptr<data_batch> pop_next_data_batch(size_t partition_idx = 0)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    require_partition(partition_idx);
    return pop_next_data_batch_unlocked(partition_idx).first;
  }

  /**
   * @brief Remove a batch by ID, discarding its logical position.
   * @return Payload, or nullptr if absent; valid in every ordering mode.
   * @note Use pop_data_batch_by_id_with_position() when the consumer must preserve ordering.
   * @throws std::out_of_range if partition_idx is out of range
   */
  std::shared_ptr<data_batch> pop_data_batch_by_id(uint64_t batch_id, size_t partition_idx = 0)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    require_partition(partition_idx);
    return pop_data_batch_by_id_unlocked(batch_id, partition_idx).first;
  }

  /**
   * @brief Inspect a batch without retaining its logical position.
   * @return Payload, or nullptr if absent; valid in every ordering mode.
   * @note Use get_data_batch_by_id_with_position() when the consumer must preserve ordering.
   * @throws std::out_of_range if partition_idx is out of range
   */
  std::shared_ptr<data_batch> get_data_batch_by_id(uint64_t batch_id,
                                                   size_t partition_idx = 0) const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    require_partition(partition_idx);
    return get_data_batch_by_id_unlocked(batch_id, partition_idx).first;
  }

  /**
   * @brief Remove the next batch and its position in arrival order, in any batch mode.
   * @param partition_idx Partition to pop from (default: 0)
   * @return Payload and optional position, or {nullptr, nullopt} if empty
   * @throws std::out_of_range if partition_idx is out of range
   */
  positioned_data_batch pop_next_data_batch_with_position(size_t partition_idx = 0)
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
  positioned_data_batch pop_data_batch_by_id_with_position(uint64_t batch_id,
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
  positioned_data_batch get_data_batch_by_id_with_position(uint64_t batch_id,
                                                           size_t partition_idx = 0) const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    require_partition(partition_idx);
    return get_data_batch_by_id_unlocked(batch_id, partition_idx);
  }

  /**
   * @brief Inspect a batch at a partition-local position without removing it.
   * @param position Logical position to search for
   * @param partition_idx Partition to search (default: 0)
   * @return Payload, or nullptr if absent
   * @throws std::out_of_range if partition_idx is out of range
   * @throws repository_ordering_error if the requested partition is nonempty and not ORDERED
   */
  std::shared_ptr<data_batch> get_data_batch_by_position(size_t position,
                                                         size_t partition_idx = 0) const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    require_partition(partition_idx);
    if (_data_batches[partition_idx].batches.empty()) { return {}; }
    require_positioned_access();
    return get_data_batch_by_position_unlocked(position, partition_idx);
  }

  /**
   * @brief Remove a batch at a partition-local position.
   * @param position Logical position to search for
   * @param partition_idx Partition to search (default: 0)
   * @return Payload, or nullptr if absent
   * @throws std::out_of_range if partition_idx is out of range
   * @throws repository_ordering_error if the requested partition is nonempty and not ORDERED
   */
  std::shared_ptr<data_batch> pop_data_batch_by_position(size_t position, size_t partition_idx = 0)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    require_partition(partition_idx);
    if (_data_batches[partition_idx].batches.empty()) { return {}; }
    require_positioned_access();
    return pop_data_batch_by_position_unlocked(position, partition_idx);
  }

  /**
   * @brief Get a sorted snapshot of resident positions in one partition.
   * @param partition_idx Partition to enumerate (default: 0)
   * @return Positions in increasing order; gaps and empty partitions are allowed
   * @throws std::out_of_range if partition_idx is out of range
   * @throws repository_ordering_error if the requested partition is nonempty and not ORDERED
   * @warning This snapshot is only complete after all producer operators have finished.
   * Enumeration does not claim batches; consumers must coordinate because concurrent mutation
   * may invalidate the snapshot.
   */
  std::vector<size_t> get_positions(size_t partition_idx = 0) const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    require_partition(partition_idx);
    if (_data_batches[partition_idx].batches.empty()) { return {}; }
    require_positioned_access();
    return get_positions_unlocked(partition_idx);
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
  std::vector<uint64_t> get_batch_ids(size_t partition_idx = 0) const
  {
    std::lock_guard<std::mutex> lock(_mutex);

    require_partition(partition_idx);
    return get_batch_ids_unlocked(partition_idx);
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
    return partition_size_unlocked(partition_idx);
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
      total += partition.batches.size();
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
      if (!partition.batches.empty()) { return false; }
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
  struct repository_entry;
  using batch_list = std::list<repository_entry>;
  using id_index   = std::multimap<uint64_t, batch_list::iterator>;
  struct repository_entry {
    positioned_data_batch batch;
    id_index::iterator id_entry;
    repository_entry(const std::shared_ptr<data_batch>& payload, std::optional<size_t> position)
      : batch(payload, position)
    {
    }
  };
  struct partition_storage {
    batch_list batches;
    std::map<size_t, batch_list::iterator> positions;
    id_index ids;
    partition_storage()                                        = default;
    partition_storage(partition_storage&&) noexcept            = default;
    partition_storage& operator=(partition_storage&&) noexcept = default;
    partition_storage(const partition_storage&)                = delete;
    partition_storage& operator=(const partition_storage&)     = delete;
  };
  mutable std::mutex _mutex;
  std::vector<partition_storage> _data_batches;
  data_repository_ordering _ordering = data_repository_ordering::UNKNOWN;

  // All hooks run with _mutex held by a public method. Base hooks never acquire it again.
  virtual void add_data_batch_unlocked(const std::shared_ptr<data_batch>& batch,
                                       size_t partition_idx)
  {
    insert_data_batch(batch, std::nullopt, partition_idx);
    _ordering = data_repository_ordering::UNORDERED;
  }
  virtual void add_data_batch_at_position_unlocked(const std::shared_ptr<data_batch>& batch,
                                                   size_t position,
                                                   size_t partition_idx)
  {
    insert_data_batch(batch, position, partition_idx);
    _ordering = data_repository_ordering::ORDERED;
  }
  virtual positioned_data_batch pop_next_data_batch_unlocked(size_t partition_idx)
  {
    auto& partition = _data_batches[partition_idx];
    if (partition.batches.empty()) { return {}; }
    return erase_entry(partition, partition.batches.begin());
  }
  virtual positioned_data_batch pop_data_batch_by_id_unlocked(uint64_t batch_id,
                                                              size_t partition_idx)
  {
    auto& partition = _data_batches[partition_idx];
    auto it         = partition.ids.find(batch_id);
    return it == partition.ids.end() ? positioned_data_batch{} : erase_entry(partition, it->second);
  }
  virtual positioned_data_batch get_data_batch_by_id_unlocked(uint64_t batch_id,
                                                              size_t partition_idx) const
  {
    const auto& partition = _data_batches[partition_idx];
    auto it               = partition.ids.find(batch_id);
    return it == partition.ids.end() ? positioned_data_batch{} : it->second->batch;
  }
  virtual std::shared_ptr<data_batch> get_data_batch_by_position_unlocked(
    size_t position, size_t partition_idx) const
  {
    const auto& positions = _data_batches[partition_idx].positions;
    auto it               = positions.find(position);
    return it == positions.end() ? nullptr : it->second->batch.first;
  }
  virtual std::shared_ptr<data_batch> pop_data_batch_by_position_unlocked(size_t position,
                                                                          size_t partition_idx)
  {
    auto& partition = _data_batches[partition_idx];
    auto it         = partition.positions.find(position);
    return it == partition.positions.end() ? nullptr : erase_entry(partition, it->second).first;
  }
  virtual std::vector<size_t> get_positions_unlocked(size_t partition_idx) const
  {
    std::vector<size_t> positions;
    positions.reserve(_data_batches[partition_idx].positions.size());
    for (const auto& entry : _data_batches[partition_idx].positions) {
      positions.push_back(entry.first);
    }
    return positions;
  }
  virtual std::vector<uint64_t> get_batch_ids_unlocked(size_t partition_idx) const
  {
    std::vector<uint64_t> ids;
    ids.reserve(_data_batches[partition_idx].batches.size());
    for (const auto& entry : _data_batches[partition_idx].batches) {
      ids.push_back(entry.batch.first->get_batch_id());
    }
    return ids;
  }
  size_t partition_size_unlocked(size_t partition_idx) const
  {
    return _data_batches[partition_idx].batches.size();
  }

 private:
  void require_partition(size_t partition_idx) const
  {
    if (partition_idx >= _data_batches.size()) {
      throw std::out_of_range("partition_idx out of range");
    }
  }
  void require_unpositioned_access() const
  {
    if (_ordering == data_repository_ordering::ORDERED) {
      throw repository_ordering_error(
        "encountered ORDERED data_repository_ordering when expecting UNORDERED");
    }
  }
  void require_positioned_access() const
  {
    if (_ordering != data_repository_ordering::ORDERED) {
      throw repository_ordering_error("position access requires an ORDERED repository");
    }
  }
  void insert_data_batch(const std::shared_ptr<data_batch>& batch,
                         std::optional<size_t> position,
                         size_t partition_idx)
  {
    if (partition_idx == std::numeric_limits<size_t>::max()) {
      throw std::out_of_range("partition_idx cannot be represented as a partition count");
    }
    // Prepare every allocating index node before growing or mutating the destination.
    partition_storage staged;
    staged.batches.emplace_back(batch, position);
    auto entry      = staged.batches.begin();
    entry->id_entry = staged.ids.emplace(batch->get_batch_id(), entry);
    if (position) { staged.positions.emplace(*position, entry); }
    if (partition_idx >= _data_batches.size()) { _data_batches.resize(partition_idx + 1); }
    auto& partition = _data_batches[partition_idx];
    // Default allocators and integral comparators: transfer nodes without allocation.
    partition.batches.splice(partition.batches.end(), staged.batches);
    partition.ids.merge(staged.ids);
    partition.positions.merge(staged.positions);
  }
  positioned_data_batch erase_entry(partition_storage& partition, batch_list::iterator entry)
  {
    partition.ids.erase(entry->id_entry);
    if (entry->batch.second) { partition.positions.erase(*entry->batch.second); }
    auto batch = std::move(entry->batch);
    partition.batches.erase(entry);
    return batch;
  }
};

using shared_data_repository = data_repository;

}  // namespace cucascade
