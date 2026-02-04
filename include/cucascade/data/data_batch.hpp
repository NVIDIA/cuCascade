
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

#include <cucascade/data/common.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/common.hpp>

#include <cudf/table/table.hpp>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>

namespace cucascade {
namespace memory {
class memory_space;
}
}  // namespace cucascade

namespace cucascade {

/**
 * @brief Represents the current state of a data_batch.
 *
 * State transitions allowed:
 * - idle -> in_transit, task_created
 * - task_created -> processing, idle
 * - processing -> idle
 * - in_transit -> idle
 */
enum class batch_state {
  idle,          ///< Batch is idle, not being processed or in transit
  task_created,  ///< A task has been created for this batch, pending processing
  processing,    ///< Batch is currently being processed
  in_transit     ///< Batch is currently being moved to a different memory tier
};

// Forward declarations
class data_batch;
class data_batch_processing_handle;

/**
 * @brief RAII handle that manages processing state for a data_batch.
 *
 * When this handle goes out of scope, it decrements the processing count of the
 * associated data_batch. If the processing count drops to zero, the batch state
 * transitions from processing back to idle.
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
 * @brief Result of attempting to lock a batch for processing.
 */
enum class lock_for_processing_status {
  success,
  task_not_created,
  invalid_state,
  memory_space_mismatch,
  missing_data,
  not_attempted
};

struct lock_for_processing_result {
  bool success{false};
  data_batch_processing_handle handle{};
  lock_for_processing_status status{lock_for_processing_status::not_attempted};

  lock_for_processing_result() = default;
  lock_for_processing_result(bool success,
                             data_batch_processing_handle&& handle,
                             lock_for_processing_status status)
    : success(success), handle(std::move(handle)), status(status)
  {
  }

  // Move-only to mirror handle semantics
  lock_for_processing_result(lock_for_processing_result&&) noexcept            = default;
  lock_for_processing_result& operator=(lock_for_processing_result&&) noexcept = default;
  lock_for_processing_result(const lock_for_processing_result&)                = delete;
  lock_for_processing_result& operator=(const lock_for_processing_result&)     = delete;
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
 * - State management (idle, task_created, processing, in_transit) for lifecycle tracking
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
   * @return batch_state The current state (idle, task_created, processing, or in_transit)
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
   * @brief Set a condition variable to be notified on state changes.
   *
   * The CV is notified outside of the batch mutex.
   */
  void set_state_change_cv(std::condition_variable* cv);
  /**
   * @brief Replace the underlying data representation.
   *        Requires no active processing.
   */
  void set_data(std::unique_ptr<idata_representation> data);

  /**
   * @brief Convert the underlying representation to the target representation type.
   *        Requires no active processing.
   *
   * Uses the provided representation_converter_registry to look up the appropriate converter
   * from the current representation type to the target type.
   *
   * @tparam TargetRepresentation The target idata_representation type to convert to
   * @param registry The converter registry to use for looking up converters
   * @param target_memory_space The memory space where the new representation will be allocated
   * @param stream CUDA stream for memory operations
   * @throws std::runtime_error if no converter is registered for the type pair
   * @throws std::runtime_error if there is active processing on this batch
   */
  template <typename TargetRepresentation>
  void convert_to(representation_converter_registry& registry,
                  const cucascade::memory::memory_space* target_memory_space,
                  rmm::cuda_stream_view stream);

  /**
   * @brief Attempt to create a task for this batch.
   *
   * Transitions the batch from idle to task_created state and increments task_created_counter.
   * If the batch is already in task_created state, only increments task_created_counter.
   * If the batch is in processing state, increments task_created_counter but stays in processing.
   * This must be called before try_to_lock_for_processing().
   *
   * @return true if the batch is in idle, task_created, or processing state
   * @return false if the batch is in in_transit state
   */
  bool try_to_create_task();

  /**
   * @brief Get the current task_created counter (mutex-protected).
   *
   * This counter tracks how many task_created requests are pending for this batch.
   * It is incremented by try_to_create_task() and decremented by try_to_lock_for_processing().
   *
   * @return size_t The number of pending task_created requests
   */
  size_t get_task_created_count() const;

  /**
   * @brief Cancel a created task and return to idle state.
   *
   * Transitions the batch from task_created back to idle state.
   *
   * @return true if the batch was successfully transitioned to idle
   * @return false if the batch is not in task_created state
   */
  bool try_to_cancel_task();

  /**
   * @brief Attempt to lock this batch for processing operations.
   *
   * Returns a processing handle if the batch is in task_created state or already processing.
   * If successful, decrements task_created_counter, increments processing count,
   * and transitions to processing state (if not already processing).
   *
   * @param requested_memory_space The memory space the caller expects to process from. If the
   *        batch is not currently in this space, locking fails with
   *        lock_for_processing_status::memory_space_mismatch.
   *
   * @return lock_for_processing_result success=true with handle on success; success=false otherwise
   *         with a status describing the failure.
   * @throws std::runtime_error if task_created_count is zero (try_to_create_task() must be
   *         called before this method) while in a lockable state.
   */
  lock_for_processing_result try_to_lock_for_processing(
    memory::memory_space_id requested_memory_space);

  /**
   * @brief Attempt to lock this batch for in-transit operations (e.g., downgrade).
   *
   * Transitions the batch from idle to in_transit state.
   *
   * @return true if the batch was successfully locked (processing_count == 0 and state is idle)
   * @return false if the batch could not be locked
   */
  bool try_to_lock_for_in_transit();

  /**
   * @brief Release the in-transit lock and return to idle state.
   *
   * Transitions the batch from in_transit back to idle state.
   *
   * @return true if the batch was successfully transitioned to idle
   * @return false if the batch is not in in_transit state
   *
   * @param target_state Optional state to transition to when releasing in_transit. If not set,
   *        the batch returns to idle.
   */
  bool try_to_release_in_transit(std::optional<batch_state> target_state = std::nullopt);

  /**
   * @brief Create a deep copy of this data batch.
   *
   * Creates a new data_batch with the specified batch_id and a cloned copy of
   * the underlying data representation. The new batch will be in idle state with
   * no active processing.
   *
   * @param new_batch_id The batch ID for the cloned batch
   * @return std::shared_ptr<data_batch> A new data_batch with copied data
   * @throws std::runtime_error if there is active processing on this batch
   * @throws std::runtime_error if the underlying data is null
   */
  std::shared_ptr<data_batch> clone(uint64_t new_batch_id);

 private:
  friend class data_batch_processing_handle;

  /**
   * @brief Decrement processing count and potentially transition state.
   *
   * Called by data_batch_processing_handle when it goes out of scope.
   * If processing count drops to zero, transitions from processing to idle.
   */
  void decrement_processing_count();

  mutable std::mutex _mutex;  ///< Mutex for thread-safe access to state and processing count
  uint64_t _batch_id;         ///< Unique identifier for this data batch
  std::unique_ptr<idata_representation> _data;    ///< Pointer to the actual data representation
  size_t _processing_count                  = 0;  ///< Count of active processing handles
  size_t _task_created_count                = 0;  ///< Count of pending task_created requests
  batch_state _state                        = batch_state::idle;  ///< Current state of the batch
  std::condition_variable* _state_change_cv = nullptr;  ///< Optional CV to notify on state change
};

// Template implementation
template <typename TargetRepresentation>
void data_batch::convert_to(representation_converter_registry& registry,
                            const cucascade::memory::memory_space* target_memory_space,
                            rmm::cuda_stream_view stream)
{
  std::lock_guard<std::mutex> lock(_mutex);

  if (_processing_count != 0) {
    throw std::runtime_error("Cannot convert representation while there is active processing");
  }

  auto new_representation =
    registry.convert<TargetRepresentation>(*_data, target_memory_space, stream);
  _data = std::move(new_representation);
}

}  // namespace cucascade
