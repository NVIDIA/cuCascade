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

#include "data/common.hpp"

#include <cudf/table/table.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <variant>

namespace cucascade {
namespace memory {
class memory_space;
}
}  // namespace cucascade

namespace cucascade {

/**
 * @brief Represents the current state of a data_batch.
 */
enum class batch_state {
  at_rest,     ///< Batch is idle, not being processed or downgraded
  processing,  ///< Batch is currently being processed
  downgrading  ///< Batch is currently being downgraded to a lower memory tier
};

// Forward declaration
class data_batch;

/**
 * @brief RAII handle that manages processing state for a data_batch.
 *
 * When this handle goes out of scope, it decrements the processing count of the
 * associated data_batch. If the processing count drops to zero, the batch state
 * transitions from processing back to at_rest.
 *
 * @note This class is move-only to ensure proper ownership semantics.
 */
class data_batch_processing_handle {
 public:
  /**
   * @brief Default constructor creates an empty handle.
   */
  data_batch_processing_handle() : _batch(nullptr) {}

  /**
   * @brief Construct a handle for the given data_batch.
   *
   * @param batch Pointer to the data_batch to manage (non-owning)
   */
  explicit data_batch_processing_handle(data_batch* batch) : _batch(batch) {}

  /**
   * @brief Destructor decrements processing count and potentially transitions state.
   */
  ~data_batch_processing_handle();

  // Move-only semantics
  data_batch_processing_handle(const data_batch_processing_handle&)            = delete;
  data_batch_processing_handle& operator=(const data_batch_processing_handle&) = delete;

  /**
   * @brief Move constructor - transfers ownership of the handle.
   */
  data_batch_processing_handle(data_batch_processing_handle&& other) noexcept : _batch(other._batch)
  {
    other._batch = nullptr;
  }

  /**
   * @brief Move assignment operator - transfers ownership of the handle.
   */
  data_batch_processing_handle& operator=(data_batch_processing_handle&& other) noexcept
  {
    if (this != &other) {
      // Release current batch if any
      release();
      _batch       = other._batch;
      other._batch = nullptr;
    }
    return *this;
  }

  /**
   * @brief Check if this handle is valid (managing a batch).
   */
  bool valid() const { return _batch != nullptr; }

  /**
   * @brief Explicitly release the handle, decrementing the processing count.
   */
  void release();

 private:
  data_batch* _batch;  ///< Non-owning pointer to the managed data_batch
};

/**
 * @brief A data batch represents a collection of data that can be moved between different memory
 * tiers.
 *
 * data_batch is the core unit of data management in cuCascade. It wraps an
 * idata_representation and provides processing counting functionality to track when data is being
 * actively used. This enables safe memory management and efficient data movement
 * between GPU memory, host memory, and storage tiers.
 *
 * Key characteristics:
 * - Move-only semantics (no copy constructor/assignment)
 * - Processing counting for safe shared access and eviction prevention
 * - State management (at_rest, processing, downgrading) for lifecycle tracking
 * - Delegated tier management to underlying idata_representation
 * - Unique batch ID for tracking and debugging purposes
 * - Lifecycle managed by smart pointers (std::shared_ptr or std::unique_ptr)
 *
 * @note This class is not thread-safe for construction/destruction, but the state management
 *       operations are protected by an internal mutex and thread-safe.
 */
class data_batch {
 public:
  /**
   * @brief Construct a new data_batch with the given ID and data representation.
   *
   * @param batch_id Unique identifier for this batch
   * @param data Ownership of the data representation is transferred to this batch
   */
  data_batch(uint64_t batch_id, std::unique_ptr<idata_representation> data);

  /**
   * @brief Move constructor - transfers ownership of the batch and its data.
   *
   * Moves the batch_id and data from the other batch, then resets the other batch's
   * batch_id to 0 and data pointer to nullptr.
   *
   * @param other The batch to move from (will have batch_id set to 0 and data set to nullptr)
   * @throws std::runtime_error if the source batch has active processing (processing_count != 0)
   */
  data_batch(data_batch&& other);

  /**
   * @brief Move assignment operator - transfers ownership of the batch and its data.
   *
   * Performs self-assignment check, then moves the batch_id and data from the other batch.
   * Resets the other batch's batch_id to 0 and data pointer to nullptr.
   *
   * @param other The batch to move from (will have batch_id set to 0 and data set to nullptr)
   * @return data_batch& Reference to this batch
   * @throws std::runtime_error if the source batch has active processing (processing_count != 0)
   */
  data_batch& operator=(data_batch&& other);

  /**
   * @brief Get the current memory tier where this batch's data resides.
   *
   * @return Tier The memory tier (GPU, HOST, or STORAGE)
   */
  memory::Tier get_current_tier() const;

  /**
   * @brief Get the unique identifier for this data batch.
   *
   * @return uint64_t The batch ID assigned during construction
   */
  uint64_t get_batch_id() const;

  /**
   * @brief Get the current state of this batch.
   *
   * @return batch_state The current state (at_rest, processing, or downgrading)
   */
  batch_state get_state() const;

  /**
   * @brief Get the current processing count (mutex-protected).
   *
   * @return size_t The number of active processing handles
   */
  size_t get_processing_count() const;

  /**
   * @brief Get the underlying data representation.
   *
   * Returns a pointer to the idata_representation that holds the actual data.
   * This allows access to tier-specific operations and data access methods.
   *
   * @return idata_representation* Pointer to the data representation (non-owning)
   */
  idata_representation* get_data() const;

  /**
   * @brief Get the memory_space where this batch currently resides.
   *
   * Delegates to the underlying idata_representation to determine the memory space.
   *
   * @return cucascade::memory::memory_space* Pointer to the memory space
   */
  cucascade::memory::memory_space* get_memory_space() const;

  /**
   * @brief Replace the underlying data representation.
   *        Requires no active processing.
   */
  void set_data(std::unique_ptr<idata_representation> data);

  /**
   * @brief Convert the underlying representation to the target memory_space.
   *        Requires no active processing.
   */
  void convert_to_memory_space(const cucascade::memory::memory_space* target_memory_space,
                               rmm::cuda_stream_view stream);

  /**
   * @brief Attempt to lock this batch for processing operations.
   *
   * Returns true if the batch is either at_rest or already processing.
   * If successful, increments the processing count and transitions to processing state.
   *
   * @return true if the batch was successfully locked for processing
   * @return false if the batch is currently being downgraded
   */
  bool try_to_lock_for_processing();

  /**
   * @brief Attempt to lock this batch for downgrade operations.
   *
   * @return true if the batch was successfully locked (processing_count == 0 and state is at_rest)
   * @return false if the batch could not be locked
   */
  bool try_to_lock_for_downgrade();

 private:
  friend class data_batch_processing_handle;

  /**
   * @brief Decrement processing count and potentially transition state.
   *
   * Called by data_batch_processing_handle when it goes out of scope.
   * If processing count drops to zero, transitions from processing to at_rest.
   */
  void decrement_processing_count();

  mutable std::mutex _mutex;  ///< Mutex for thread-safe access to state and processing count
  uint64_t _batch_id;         ///< Unique identifier for this data batch
  std::unique_ptr<idata_representation> _data;      ///< Pointer to the actual data representation
  size_t _processing_count = 0;                     ///< Count of active processing handles
  batch_state _state       = batch_state::at_rest;  ///< Current state of the batch
};

}  // namespace cucascade
