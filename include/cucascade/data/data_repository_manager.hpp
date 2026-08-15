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
 * cuCascade. It maintains a collection of data_repository instances, each associated
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
 * ├── Pipeline 1 → data_repository (FIFO/LRU/Priority)
 * ├── Pipeline 2 → data_repository (FIFO/LRU/Priority)
 * └── Pipeline N → data_repository (FIFO/LRU/Priority)
 * ```
 *
 * The manager abstracts the complexity of multi-pipeline data management and provides
 * a unified interface for higher-level components like the GPU executor and memory manager.
 *
 * @note All operations are thread-safe and can be called concurrently from multiple
 *       pipeline execution threads.
 */
class data_repository_manager {
 public:
  using repository_type = data_repository;

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
    // Stored as shared_ptr so accessors can hand out lifetime-safe references
    // under _mutex (see get_repository_shared); callers keep passing
    // unique_ptr because each repository still has exactly one logical owner.
    std::shared_ptr<repository_type> shared_repository{std::move(repository)};
    {
      std::lock_guard<std::mutex> lock(_mutex);
      auto it = _repositories.find({operator_id, std::string(port_id)});
      if (it != _repositories.end()) { throw std::runtime_error("Repository already exists"); }
      _repositories[{operator_id, std::string(port_id)}] = std::move(shared_repository);
    }
  }

  /**
   * @brief Add a data_batch to specified operator repositories.
   *
   * The shared batch pointer is copied to each repository.
   *
   * @param batch The data_batch smart pointer to add
   * @param ops The operator IDs and ports whose repositories will receive this batch
   *
   * @note Thread-safe operation
   */
  void add_data_batch(std::shared_ptr<data_batch> batch,
                      std::vector<std::pair<size_t, std::string_view>> ops)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    for (auto& op : ops) {
      _repositories.at({op.first, std::string(op.second)})->add_data_batch(batch);
    }
  }

  /**
   * @brief Get lifetime-safe access to a repository for advanced operations.
   *
   * Looks the repository up under _mutex and returns a shared_ptr copy, so the
   * returned repository stays valid even if a concurrent add_new_repository or
   * clear_all_repositories mutates the map after this call returns. (The old
   * variant returned a reference into the map without taking _mutex — a
   * concurrent mutation raced both the lookup and the returned reference.)
   *
   * @param operator_id The unique ID of the operator whose repository is requested
   * @param port_id The port identifier for the repository
   * @return std::shared_ptr<repository_type> Shared ownership of the repository
   *
   * @throws std::out_of_range If no repository exists for the specified operator/port
   * @note Thread-safe — the lookup holds the manager mutex; the repository's own
   *       thread safety covers subsequent operations on it
   */
  std::shared_ptr<repository_type> get_repository_shared(size_t operator_id,
                                                         std::string_view port_id)
  {
    std::lock_guard<std::mutex> lock(_mutex);
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
   * @brief Get a snapshot of all current repository pointers.
   *
   * Returns a vector of raw pointers to each non-null repository. The vector
   * is built under the manager mutex, so callers can iterate it externally
   * without holding the lock (the repositories themselves remain thread-safe).
   *
   * @return std::vector<repository_type*> Snapshot of non-null repository pointers
   * @note Thread-safe — holds the manager mutex for the duration of collection.
   */
  std::vector<repository_type*> get_repositories()
  {
    std::lock_guard<std::mutex> lock(_mutex);
    std::vector<repository_type*> result;
    result.reserve(_repositories.size());
    for (auto& [key, repo] : _repositories) {
      if (repo) { result.push_back(repo.get()); }
    }
    return result;
  }

 private:
  std::mutex _mutex;  ///< Mutex for thread-safe access
  std::atomic<uint64_t> _next_data_batch_id =
    0;  ///< Atomic counter for generating unique data batch identifiers
  /// Map of operator ID/port to data_repository. Held by shared_ptr so
  /// get_repository_shared can hand out references that survive a concurrent
  /// clear_all_repositories / add_new_repository (the manager remains the one
  /// logical owner; accessors only extend lifetime across their use).
  std::map<operator_port_key, std::shared_ptr<repository_type>> _repositories;
};

/// Compatibility alias, NOT a distinct type: kept so call sites written
/// against the pre-merge class keep compiling. Since the map moved to
/// shared_ptr storage the "shared" in the name is loosely true (accessors hand
/// out lifetime-safe shared_ptr copies), but the manager remains each
/// repository's one logical owner. Prefer `data_repository_manager` in new code.
using shared_data_repository_manager = data_repository_manager;

}  // namespace cucascade
