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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <utility>

namespace cucascade {
namespace memory {
class memory_space;
}
}  // namespace cucascade

namespace cucascade {

/**
 * @brief Observable state of a data_batch.
 *
 * Tracks whether the batch is idle, shared-locked (read_only), or
 * exclusively-locked (mutable_locked). Updated atomically during
 * state transitions.
 */
enum class batch_state { idle, read_only, mutable_locked };

// Forward declarations -- required before data_batch because it friends them.
class read_only_data_batch;
class mutable_data_batch;

/**
 * @brief Core data batch type representing the "idle" (unlocked) state.
 *
 * Owns the data representation, a reader-writer mutex, and subscriber bookkeeping.
 * Almost nothing is publicly accessible -- data, tier, and memory space are private
 * and can only be reached through RAII accessor types that hold the appropriate lock.
 *
 * State transitions are static methods that move ownership of the accessor,
 * making the source null at the call site. This provides compile-time enforcement:
 * once a batch is locked, the caller cannot access the idle handle.
 *
 * @note Non-copyable and non-movable. The object itself never moves; only the
 *       smart pointer to it is transferred between states.
 */
class data_batch : public std::enable_shared_from_this<data_batch> {
 public:
  // -- Construction --

  /**
   * @brief Construct a new data_batch.
   *
   * @param batch_id Unique identifier for this batch (immutable after construction).
   * @param data     Owned data representation; must not be null.
   */
  data_batch(uint64_t batch_id, std::unique_ptr<idata_representation> data);

  /** @brief Default destructor. */
  ~data_batch() = default;

  // -- Deleted move/copy (D-04/CORE-07) --
  data_batch(data_batch&&)                 = delete;
  data_batch& operator=(data_batch&&)      = delete;
  data_batch(const data_batch&)            = delete;
  data_batch& operator=(const data_batch&) = delete;

  // -- Lock-free public API --

  /**
   * @brief Get the unique batch identifier.
   *
   * Lock-free -- safe to call without acquiring an accessor.
   *
   * @return The batch ID (immutable after construction).
   */
  uint64_t get_batch_id() const;

  /**
   * @brief Increment the subscriber interest count.
   *
   * Atomic, lock-free. Returns true on the first subscription (0 -> 1).
   *
   * @return true if this was the first subscriber, false otherwise.
   */
  bool subscribe();

  /**
   * @brief Decrement the subscriber interest count.
   *
   * Atomic, lock-free.
   *
   * @throws std::runtime_error if subscriber count is already zero.
   */
  void unsubscribe();

  /**
   * @brief Get the current subscriber count.
   *
   * Atomic, lock-free.
   *
   * @return The number of active subscribers.
   */
  size_t get_subscriber_count() const;

  /**
   * @brief Get the observable lock state of this batch.
   *
   * Atomic, lock-free. Returns the current state: idle, read_only, or
   * mutable_locked. Updated during every state transition.
   *
   * @return The current batch_state.
   */
  batch_state get_state() const { return _state.load(std::memory_order_relaxed); }

  /**
   * @brief Get the number of active read_only_data_batch instances holding this batch.
   *
   * Atomic, lock-free. Counts concurrent shared-lock holders. Transitions to zero
   * when the last read_only_data_batch is destroyed (or moved-from).
   *
   * @return The current reader count.
   */
  size_t get_read_only_count() const { return _read_only_count.load(std::memory_order_acquire); }

  // -- Static transition methods (D-13/D-14/D-15/D-17) --

  /**
   * @brief Transition from read-only back to idle (release shared lock).
   *
   * @param accessor Rvalue reference to the read-only accessor (consumed).
   * @return The batch pointer, now in idle state.
   */
  [[nodiscard]] static std::shared_ptr<data_batch> to_idle(read_only_data_batch&& accessor);

  /**
   * @brief Transition from mutable back to idle (release exclusive lock).
   *
   * @param accessor Rvalue reference to the mutable accessor (consumed).
   * @return The batch pointer, now in idle state.
   */
  [[nodiscard]] static std::shared_ptr<data_batch> to_idle(mutable_data_batch&& accessor);

  // -- Non-static transitions (via shared_from_this) --
  // The caller's shared_ptr is NOT consumed. These only work when the
  // data_batch is managed by a shared_ptr (throws bad_weak_ptr otherwise).

  /**
   * @brief Transition from idle to read-only (shared lock) without consuming the caller's pointer.
   *
   * Uses shared_from_this() to obtain a new shared_ptr. Blocks until the
   * shared lock is acquired.
   *
   * @return A read_only_data_batch holding the shared lock.
   */
  [[nodiscard]] read_only_data_batch to_read_only();

  /**
   * @brief Transition from idle to mutable (exclusive lock) without consuming the caller's pointer.
   *
   * Uses shared_from_this() to obtain a new shared_ptr. Blocks until the
   * exclusive lock is acquired.
   *
   * @return A mutable_data_batch holding the exclusive lock.
   */
  [[nodiscard]] mutable_data_batch to_mutable();

  /**
   * @brief Try to transition from idle to read-only (non-blocking).
   *
   * @return An optional containing the read-only accessor on success, or
   *         std::nullopt if the lock could not be acquired immediately.
   */
  [[nodiscard]] std::optional<read_only_data_batch> try_to_read_only();

  /**
   * @brief Try to transition from idle to mutable (non-blocking).
   *
   * @return An optional containing the mutable accessor on success, or
   *         std::nullopt if the lock could not be acquired immediately.
   */
  [[nodiscard]] std::optional<mutable_data_batch> try_to_mutable();

  // -- Locked-to-locked static transitions --

  /**
   * @brief Transition from read-only to mutable (upgrade lock).
   *
   * Releases the shared lock, then acquires an exclusive lock (may block).
   * The source accessor is consumed via move.
   *
   * @param accessor Rvalue reference to the read-only accessor (consumed).
   * @return A mutable_data_batch holding the exclusive lock.
   */
  [[nodiscard]] static mutable_data_batch readonly_to_mutable(read_only_data_batch&& accessor);

  /**
   * @brief Transition from mutable to read-only (downgrade lock).
   *
   * Releases the exclusive lock, then acquires a shared lock (may block).
   * The source accessor is consumed via move.
   *
   * @param accessor Rvalue reference to the mutable accessor (consumed).
   * @return A read_only_data_batch holding the shared lock.
   */
  [[nodiscard]] static read_only_data_batch mutable_to_readonly(mutable_data_batch&& accessor);

 private:
  // -- Friend declarations (D-24/REPO-04) --
  friend class read_only_data_batch;
  friend class mutable_data_batch;

  // -- Private data accessors (D-23/CORE-02) --
  // Only friend accessor classes can call these methods.

  /**
   * @brief Get the memory tier of the held data.
   * @return The current memory tier.
   */
  memory::Tier get_current_tier() const;

  /**
   * @brief Get a raw pointer to the data representation.
   * @return Non-owning pointer to the data, or nullptr if empty.
   */
  idata_representation* get_data() const;

  /**
   * @brief Get a raw pointer to the memory space.
   * @return Non-owning pointer to the memory space, or nullptr if data is null.
   */
  memory::memory_space* get_memory_space() const;

  /**
   * @brief Replace the data representation.
   * @param data New data representation (takes ownership).
   */
  void set_data(std::unique_ptr<idata_representation> data);

  // -- Members --
  const uint64_t _batch_id;                            ///< Immutable batch identifier
  std::unique_ptr<idata_representation> _data;         ///< Owned data representation
  mutable std::shared_mutex _rw_mutex;                 ///< Reader-writer mutex
  std::atomic<size_t> _subscriber_count{0};            ///< Atomic subscriber interest count
  std::atomic<batch_state> _state{batch_state::idle};  ///< Observable lock state
  std::atomic<size_t> _read_only_count{0};  ///< Count of active read_only_data_batch instances
};

/**
 * @brief RAII read-only accessor for data_batch.
 *
 * Holds a shared lock on the parent data_batch's mutex, permitting concurrent
 * readers. Data is accessible through named methods that delegate to data_batch's
 * private interface. Clone operations are available to create independent copies
 * while the read lock is held.
 *
 * Copyable. Copying acquires a new shared lock on the same parent data_batch,
 * incrementing the reader count. The shared lock is released when this object
 * is destroyed, moved-from, or overwritten by assignment.
 */
class read_only_data_batch {
 public:
  // -- Named accessor methods (D-09/ACC-01) --

  /** @brief Get the batch identifier. */
  uint64_t get_batch_id() const { return _batch->get_batch_id(); }

  /** @brief Get the memory tier of the held data. */
  memory::Tier get_current_tier() const { return _batch->get_current_tier(); }

  /** @brief Get a raw pointer to the data representation. */
  idata_representation* get_data() const { return _batch->get_data(); }

  /** @brief Get a raw pointer to the memory space. */
  memory::memory_space* get_memory_space() const { return _batch->get_memory_space(); }

  // -- Clone operations (D-18/D-19/D-20/CLONE-01/CLONE-02) --

  /**
   * @brief Create an independent deep copy of the batch data.
   *
   * The clone has a new batch ID and its own copy of the data representation,
   * residing in the same memory space as the original.
   *
   * @param new_batch_id Batch ID for the cloned batch.
   * @param stream       CUDA stream for memory operations.
   * @return A new data_batch wrapped in shared_ptr.
   * @throws std::runtime_error if the data is null.
   */
  [[nodiscard]] std::shared_ptr<data_batch> clone(uint64_t new_batch_id,
                                                  rmm::cuda_stream_view stream) const;

  /**
   * @brief Create an independent deep copy with representation conversion.
   *
   * The clone has a new batch ID and its data is converted to TargetRepresentation
   * using the provided converter registry.
   *
   * @tparam TargetRepresentation Target representation type.
   * @param registry           Converter registry for type-keyed dispatch.
   * @param new_batch_id       Batch ID for the cloned batch.
   * @param target_memory_space Target memory space for the converted data.
   * @param stream              CUDA stream for memory operations.
   * @return A new data_batch wrapped in shared_ptr.
   */
  template <typename TargetRepresentation>
  [[nodiscard]] std::shared_ptr<data_batch> clone_to(
    representation_converter_registry& registry,
    uint64_t new_batch_id,
    const memory::memory_space* target_memory_space,
    rmm::cuda_stream_view stream) const;

  // -- Move support --
  read_only_data_batch(read_only_data_batch&& other) noexcept;
  read_only_data_batch& operator=(read_only_data_batch&& other) noexcept;
  ~read_only_data_batch();

  // -- Copy support: acquires a new shared lock, increments reader count --
  read_only_data_batch(const read_only_data_batch& other);
  read_only_data_batch& operator=(const read_only_data_batch& other);

 private:
  friend class data_batch;

  /**
   * @brief Private constructor -- only data_batch methods can create instances.
   *
   * @param parent Shared pointer to the parent data_batch (moved in).
   * @param lock   Shared lock already acquired on the parent's mutex.
   */
  read_only_data_batch(std::shared_ptr<data_batch> parent,
                       std::shared_lock<std::shared_mutex> lock);

  // INVARIANT: _batch must be declared before _lock -- destruction order is load-bearing.
  // When destroyed, _lock releases the shared lock first, then _batch drops the parent
  // reference. This prevents accessing a destroyed mutex.
  std::shared_ptr<data_batch> _batch;           ///< Parent lifetime (destroyed second)
  std::shared_lock<std::shared_mutex> _lock;    ///< Shared lock (destroyed first)
};

/**
 * @brief RAII mutable accessor for data_batch.
 *
 * Holds an exclusive lock on the parent data_batch's mutex, permitting a single
 * writer with no concurrent readers. Provides all read methods plus write methods
 * (set_data, convert_to) and clone operations (clone, clone_to).
 *
 * Move-only. The exclusive lock is released when this object is destroyed or moved-from.
 */
class mutable_data_batch {
 public:
  // -- Read methods (same as read_only, ACC-02) --

  /** @brief Get the batch identifier. */
  uint64_t get_batch_id() const { return _batch->get_batch_id(); }

  /** @brief Get the memory tier of the held data. */
  memory::Tier get_current_tier() const { return _batch->get_current_tier(); }

  /** @brief Get a raw pointer to the data representation. */
  idata_representation* get_data() const { return _batch->get_data(); }

  /** @brief Get a raw pointer to the memory space. */
  memory::memory_space* get_memory_space() const { return _batch->get_memory_space(); }

  // -- Write methods (D-10/ACC-02) --

  /**
   * @brief Replace the data representation.
   * @param data New data representation (takes ownership).
   */
  void set_data(std::unique_ptr<idata_representation> data) { _batch->set_data(std::move(data)); }

  /**
   * @brief Convert the data representation in-place.
   *
   * Replaces the held data with a new representation produced by the converter
   * registry. If the conversion involves the GPU tier, synchronizes the stream
   * before the old representation is destroyed to prevent use-after-free.
   *
   * @tparam TargetRepresentation Target representation type.
   * @param registry           Converter registry for type-keyed dispatch.
   * @param target_memory_space Target memory space for the new representation.
   * @param stream              CUDA stream for memory operations.
   */
  template <typename TargetRepresentation>
  void convert_to(representation_converter_registry& registry,
                  const memory::memory_space* target_memory_space,
                  rmm::cuda_stream_view stream);

  // -- Clone operations (CLONE-01/CLONE-02) --

  /**
   * @brief Create an independent deep copy of the batch data.
   *
   * The clone has a new batch ID and its own copy of the data representation,
   * residing in the same memory space as the original.
   *
   * @param new_batch_id Batch ID for the cloned batch.
   * @param stream       CUDA stream for memory operations.
   * @return A new data_batch wrapped in shared_ptr.
   * @throws std::runtime_error if the data is null.
   */
  [[nodiscard]] std::shared_ptr<data_batch> clone(uint64_t new_batch_id,
                                                  rmm::cuda_stream_view stream) const;

  /**
   * @brief Create an independent deep copy with representation conversion.
   *
   * The clone has a new batch ID and its data is converted to TargetRepresentation
   * using the provided converter registry.
   *
   * @tparam TargetRepresentation Target representation type.
   * @param registry           Converter registry for type-keyed dispatch.
   * @param new_batch_id       Batch ID for the cloned batch.
   * @param target_memory_space Target memory space for the converted data.
   * @param stream              CUDA stream for memory operations.
   * @return A new data_batch wrapped in shared_ptr.
   */
  template <typename TargetRepresentation>
  [[nodiscard]] std::shared_ptr<data_batch> clone_to(
    representation_converter_registry& registry,
    uint64_t new_batch_id,
    const memory::memory_space* target_memory_space,
    rmm::cuda_stream_view stream) const;

  // -- Move-only --
  mutable_data_batch(mutable_data_batch&& other) noexcept;
  mutable_data_batch& operator=(mutable_data_batch&& other) noexcept;
  ~mutable_data_batch();
  mutable_data_batch(const mutable_data_batch&)            = delete;
  mutable_data_batch& operator=(const mutable_data_batch&) = delete;

 private:
  friend class data_batch;

  /**
   * @brief Private constructor -- only data_batch methods can create instances.
   *
   * @param parent Shared pointer to the parent data_batch (moved in).
   * @param lock   Exclusive lock already acquired on the parent's mutex.
   */
  mutable_data_batch(std::shared_ptr<data_batch> parent,
                     std::unique_lock<std::shared_mutex> lock);

  // INVARIANT: _batch must be declared before _lock -- destruction order is load-bearing.
  // When destroyed, _lock releases the exclusive lock first, then _batch drops the parent
  // reference. This prevents accessing a destroyed mutex.
  std::shared_ptr<data_batch> _batch;           ///< Parent lifetime (destroyed second)
  std::unique_lock<std::shared_mutex> _lock;    ///< Exclusive lock (destroyed first)
};

// =============================================================================
// Template implementations (TargetRepresentation-templated methods only)
// =============================================================================

// -- read_only_data_batch::clone_to (deep copy + conversion, CLONE-02) --

template <typename TargetRepresentation>
std::shared_ptr<data_batch> read_only_data_batch::clone_to(
  representation_converter_registry& registry,
  uint64_t new_batch_id,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream) const
{
  auto new_representation =
    registry.convert<TargetRepresentation>(*_batch->_data, target_memory_space, stream);
  return std::make_shared<data_batch>(new_batch_id, std::move(new_representation));
}

// -- mutable_data_batch::convert_to (in-place conversion, ACC-02) --

template <typename TargetRepresentation>
void mutable_data_batch::convert_to(representation_converter_registry& registry,
                                    const memory::memory_space* target_memory_space,
                                    rmm::cuda_stream_view stream)
{
  auto new_representation =
    registry.convert<TargetRepresentation>(*_batch->_data, target_memory_space, stream);
  auto old_representation = std::move(_batch->_data);
  _batch->_data           = std::move(new_representation);

  bool needs_sync =
    old_representation != nullptr && (old_representation->get_current_tier() == memory::Tier::GPU ||
                                      _batch->_data->get_current_tier() == memory::Tier::GPU);

  if (needs_sync) {
    // Conversions involving GPU may enqueue async operations on the provided
    // stream that read from the source memory.  Synchronize before the old
    // representation is destroyed to avoid use-after-free.
    stream.synchronize();
  }
}

// -- mutable_data_batch::clone_to (deep copy + conversion, CLONE-02) --

template <typename TargetRepresentation>
std::shared_ptr<data_batch> mutable_data_batch::clone_to(
  representation_converter_registry& registry,
  uint64_t new_batch_id,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream) const
{
  auto new_representation =
    registry.convert<TargetRepresentation>(*_batch->_data, target_memory_space, stream);
  return std::make_shared<data_batch>(new_batch_id, std::move(new_representation));
}

}  // namespace cucascade
