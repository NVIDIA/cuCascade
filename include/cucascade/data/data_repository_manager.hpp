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
#include <cucascade/data/data_repository.hpp>

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace cucascade {

/**
 * @brief Key type for identifying a unique operator-port combination.
 *
 * Uses size_t operator_id to identify operators. The caller is responsible for mapping
 * operators to IDs.
 */
struct operator_port_key {
  size_t operator_id;
  std::string port_id;

  bool operator==(const operator_port_key& other) const
  {
    return operator_id == other.operator_id && port_id == other.port_id;
  }

  bool operator<(const operator_port_key& other) const
  {
    if (operator_id != other.operator_id) return operator_id < other.operator_id;
    return port_id < other.port_id;
  }
};

/**
 * @brief Central manager for coordinating data repositories across multiple pipelines.
 *
 * data_repository_manager serves as the top-level coordinator for data management in
 * cuCascade. It maintains a collection of idata_repository instances, each associated
 * with a specific pipeline, and provides centralized services for:
 *
 * - Repository lifecycle management (creation, access, cleanup)
 * - Cross-pipeline data batch coordination
 * - Unique batch ID generation
 * - Global eviction and memory management policies
 *
 * Architecture:
 * ```
 * data_repository_manager
 * ├── Pipeline 1 → idata_repository (FIFO/LRU/Priority)
 * ├── Pipeline 2 → idata_repository (FIFO/LRU/Priority)
 * └── Pipeline N → idata_repository (FIFO/LRU/Priority)
 * ```
 *
 * The manager abstracts the complexity of multi-pipeline data management and provides
 * a unified interface for higher-level components like the GPU executor and memory manager.
 *
 * @tparam PtrType The smart pointer type used to manage data_batch lifecycle.
 *                 Typically std::shared_ptr<data_batch> or std::unique_ptr<data_batch>.
 *
 * @note All operations are thread-safe and can be called concurrently from multiple
 *       pipeline execution threads.
 */
template <typename PtrType>
class data_repository_manager {
 public:
  using repository_type = idata_repository<PtrType>;

  /**
   * @brief Default constructor - initializes empty repository manager.
   */
  data_repository_manager() = default;

  /**
   * @brief Destructor - ensures repositories are cleared properly.
   */
  ~data_repository_manager() { _repositories.clear(); }

  /**
   * @brief Register a new data repository for the specified operator ID.
   *
   * Associates a data repository implementation with an operator ID and port. Each
   * operator-port combination can have exactly one repository, and attempting to add
   * a repository for an existing combination will replace the previous one.
   *
   * @param operator_id The unique ID of the operator associated with the repository
   * @param port_id The port identifier for this repository
   * @param repository Unique pointer to the repository implementation (ownership transferred)
   *
   * @note Thread-safe operation
   */
  void add_new_repository(size_t operator_id,
                          std::string_view port_id,
                          std::unique_ptr<repository_type> repository)
  {
    std::unique_ptr<repository_type> old_repository;
    {
      std::lock_guard<std::mutex> lock(_mutex);
      auto it = _repositories.find({operator_id, std::string(port_id)});
      if (it != _repositories.end()) { throw std::runtime_error("Repository already exists"); }
      _repositories[{operator_id, std::string(port_id)}] = std::move(repository);
    }
  }

  /**
   * @brief Add a data_batch to specified operator repositories.
   *
   * For shared_ptr: The batch is copied to each repository.
   * For unique_ptr: This method requires only one operator (single owner semantics).
   *
   * @param batch The data_batch smart pointer to add
   * @param ops The operator IDs and ports whose repositories will receive this batch
   *
   * @note Thread-safe operation
   */
  void add_data_batch(PtrType batch, std::vector<std::pair<size_t, std::string_view>> ops)
  {
    add_data_batch_impl(std::move(batch), ops);
  }

  /**
   * @brief Get direct access to a repository for advanced operations.
   *
   * Provides direct access to the underlying repository implementation, allowing
   * for repository-specific operations that aren't covered by the common interface.
   *
   * @param operator_id The unique ID of the operator whose repository is requested
   * @param port_id The port identifier for the repository
   * @return std::unique_ptr<repository_type>& Reference to the repository
   *
   * @throws std::out_of_range If no repository exists for the specified operator/port
   * @note Thread-safe for read access, but modifications should use the repository's own thread
   * safety
   */
  std::unique_ptr<repository_type>& get_repository(size_t operator_id, std::string_view port_id)
  {
    return _repositories.at({operator_id, std::string(port_id)});
  }

  /**
   * @brief Generate a globally unique data batch identifier.
   *
   * Returns a monotonically increasing ID that's unique across all pipelines
   * and repositories managed by this instance. Used to ensure data batches
   * can be uniquely identified for debugging, tracking, and cross-reference purposes.
   *
   * @return uint64_t A unique batch ID
   *
   * @note Thread-safe atomic operation with no contention
   */
  uint64_t get_next_data_batch_id() { return _next_data_batch_id++; }

  /**
   * @brief Info about leaked batches in a single repository after clear.
   */
  struct leaked_repository_info {
    size_t operator_id;
    std::string port_id;
    std::size_t count;
  };

  /**
   * @brief Clear all repositories and report any that still contained data.
   *
   * Should be called between queries to reset state. If any repository still has
   * un-consumed data batches, this is a bug — it means some operator didn't fully
   * drain its input.
   *
   * @return Per-repository info for each repository that still had un-consumed batches.
   */
  std::vector<leaked_repository_info> clear_all_repositories()
  {
    std::lock_guard<std::mutex> lock(_mutex);
    std::vector<leaked_repository_info> leaked;
    for (auto& [key, repo] : _repositories) {
      if (repo) {
        auto count = repo->total_size();
        if (count > 0) { leaked.push_back({key.operator_id, key.port_id, count}); }
      }
    }
    _repositories.clear();
    return leaked;
  }

  /**
   * @brief Get N batches from the specified repositories for downgrade.
   *
   * @param memory_space_id The memory space id to get the data batches from
   * @param amount_to_downgrade The amount of data in bytes to downgrade
   * @return std::vector<PtrType> A vector of data batches that are candidates for downgrade
   */
  std::vector<PtrType> get_data_batches_for_downgrade(
    [[maybe_unused]] cucascade::memory::memory_space_id memory_space_id,
    [[maybe_unused]] size_t amount_to_downgrade)
  {
    std::vector<PtrType> data_batches;
    // Note: Implementation would iterate through repositories and collect batches
    // This is a placeholder - actual implementation depends on how batches are tracked
    return data_batches;
  }

 private:
  // Implementation for shared_ptr - can copy to multiple repositories
  template <typename T = PtrType>
  typename std::enable_if<std::is_same<T, std::shared_ptr<data_batch>>::value>::type
  add_data_batch_impl(T batch, std::vector<std::pair<size_t, std::string_view>>& ops)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    for (auto& op : ops) {
      _repositories[{op.first, std::string(op.second)}]->add_data_batch(batch);
    }
  }

  // Implementation for unique_ptr - can only add to one repository (moves the batch)
  template <typename T = PtrType>
  typename std::enable_if<std::is_same<T, std::unique_ptr<data_batch>>::value>::type
  add_data_batch_impl(T batch, std::vector<std::pair<size_t, std::string_view>>& ops)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (ops.size() > 1) {
      throw std::runtime_error(
        "unique_ptr data_batch can only be added to one repository. "
        "Use shared_ptr for multiple destinations.");
    }
    if (!ops.empty()) {
      auto& op = ops[0];
      _repositories[{op.first, std::string(op.second)}]->add_data_batch(std::move(batch));
    } else {
      throw std::runtime_error("No operator ports provided");
    }
  }

  std::mutex _mutex;  ///< Mutex for thread-safe access
  std::atomic<uint64_t> _next_data_batch_id =
    0;  ///< Atomic counter for generating unique data batch identifiers
  std::map<operator_port_key, std::unique_ptr<repository_type>>
    _repositories;  ///< Map of operator ID to idata_repository
};

// Type aliases for common use cases
using shared_data_repository_manager = data_repository_manager<std::shared_ptr<data_batch>>;
using unique_data_repository_manager = data_repository_manager<std::unique_ptr<data_batch>>;

}  // namespace cucascade
