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
#include <type_traits>
#include <utility>

namespace cucascade {
namespace memory {
class memory_space;
}
}  // namespace cucascade

namespace cucascade {

// Forward declarations -- required before data_batch because it friends them.
template <typename PtrType>
class read_only_data_batch;
template <typename PtrType>
class mutable_data_batch;

/**
 * @brief Core data batch type representing the "idle" (unlocked) state.
 *
 * Owns the data representation, a reader-writer mutex, and subscriber bookkeeping.
 * Almost nothing is publicly accessible -- data, tier, and memory space are private
 * and can only be reached through RAII accessor types that hold the appropriate lock.
 *
 * State transitions are static template methods that move ownership of the smart
 * pointer (PtrType) into an accessor, making the source pointer null at the call
 * site. This provides compile-time enforcement: once a batch is locked, the caller
 * cannot access the idle handle.
 *
 * @note Non-copyable and non-movable. The object itself never moves; only the
 *       smart pointer to it is transferred between states.
 */
class data_batch {
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
  data_batch(data_batch&&)            = delete;
  data_batch& operator=(data_batch&&) = delete;
  data_batch(const data_batch&)       = delete;
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

  // -- Static transition methods (D-13/D-14/D-15/D-17) --
  // All transitions go through idle -- no locked-to-locked shortcuts.

  /**
   * @brief Transition from idle to read-only (shared lock).
   *
   * Blocks until the shared lock is acquired. The source pointer is
   * moved-from and becomes null.
   *
   * @tparam PtrType Smart pointer type (shared_ptr or unique_ptr to data_batch).
   * @param batch Rvalue reference to the idle batch pointer (moved-from).
   * @return A read_only_data_batch holding the shared lock.
   */
  template <typename PtrType>
  [[nodiscard]] static read_only_data_batch<PtrType> to_read_only(PtrType&& batch);

  /**
   * @brief Transition from idle to mutable (exclusive lock).
   *
   * Blocks until the exclusive lock is acquired. The source pointer is
   * moved-from and becomes null.
   *
   * @tparam PtrType Smart pointer type (shared_ptr or unique_ptr to data_batch).
   * @param batch Rvalue reference to the idle batch pointer (moved-from).
   * @return A mutable_data_batch holding the exclusive lock.
   */
  template <typename PtrType>
  [[nodiscard]] static mutable_data_batch<PtrType> to_mutable(PtrType&& batch);

  /**
   * @brief Transition from read-only back to idle (release shared lock).
   *
   * @tparam PtrType Smart pointer type (shared_ptr or unique_ptr to data_batch).
   * @param accessor Rvalue reference to the read-only accessor (consumed).
   * @return The batch pointer, now in idle state.
   */
  template <typename PtrType>
  [[nodiscard]] static PtrType to_idle(read_only_data_batch<PtrType>&& accessor);

  /**
   * @brief Transition from mutable back to idle (release exclusive lock).
   *
   * @tparam PtrType Smart pointer type (shared_ptr or unique_ptr to data_batch).
   * @param accessor Rvalue reference to the mutable accessor (consumed).
   * @return The batch pointer, now in idle state.
   */
  template <typename PtrType>
  [[nodiscard]] static PtrType to_idle(mutable_data_batch<PtrType>&& accessor);

  /**
   * @brief Try to transition from idle to read-only (non-blocking).
   *
   * Attempts to acquire the shared lock without blocking. On success, the
   * source pointer is nullified. On failure, the source pointer is unchanged.
   *
   * @tparam PtrType Smart pointer type (shared_ptr or unique_ptr to data_batch).
   * @param batch Mutable lvalue reference to the idle batch pointer.
   * @return An optional containing the read-only accessor on success, or
   *         std::nullopt if the lock could not be acquired immediately.
   */
  template <typename PtrType>
  [[nodiscard]] static std::optional<read_only_data_batch<PtrType>> try_to_read_only(
    PtrType& batch);

  /**
   * @brief Try to transition from idle to mutable (non-blocking).
   *
   * Attempts to acquire the exclusive lock without blocking. On success, the
   * source pointer is nullified. On failure, the source pointer is unchanged.
   *
   * @tparam PtrType Smart pointer type (shared_ptr or unique_ptr to data_batch).
   * @param batch Mutable lvalue reference to the idle batch pointer.
   * @return An optional containing the mutable accessor on success, or
   *         std::nullopt if the lock could not be acquired immediately.
   */
  template <typename PtrType>
  [[nodiscard]] static std::optional<mutable_data_batch<PtrType>> try_to_mutable(
    PtrType& batch);

 private:
  // -- Friend declarations (D-24/REPO-04) --
  template <typename PtrType>
  friend class read_only_data_batch;
  template <typename PtrType>
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

  /**
   * @brief Convert the data representation in-place.
   *
   * @tparam TargetRepresentation Target representation type.
   * @param registry   Converter registry for type-keyed dispatch.
   * @param target_memory_space Target memory space for the new representation.
   * @param stream     CUDA stream for memory operations.
   */
  template <typename TargetRepresentation>
  void convert_to(representation_converter_registry& registry,
                  const memory::memory_space* target_memory_space,
                  rmm::cuda_stream_view stream);

  // -- Members --
  const uint64_t _batch_id;                    ///< Immutable batch identifier
  std::unique_ptr<idata_representation> _data;  ///< Owned data representation
  mutable std::shared_mutex _rw_mutex;          ///< Reader-writer mutex
  std::atomic<size_t> _subscriber_count{0};     ///< Atomic subscriber interest count
};

/**
 * @brief RAII read-only accessor for data_batch.
 *
 * Holds a shared lock on the parent data_batch's mutex, permitting concurrent
 * readers. Data is accessible through named methods that delegate to data_batch's
 * private interface. Clone operations are available to create independent copies
 * while the read lock is held.
 *
 * Move-only. The shared lock is released when this object is destroyed or moved-from.
 *
 * @tparam PtrType Smart pointer type (std::shared_ptr<data_batch> or
 *         std::unique_ptr<data_batch>).
 */
template <typename PtrType>
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
   * @return A new data_batch wrapped in PtrType.
   * @throws std::runtime_error if the data is null.
   */
  [[nodiscard]] PtrType clone(uint64_t new_batch_id, rmm::cuda_stream_view stream) const;

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
   * @return A new data_batch wrapped in PtrType.
   */
  template <typename TargetRepresentation>
  [[nodiscard]] PtrType clone_to(representation_converter_registry& registry,
                                 uint64_t new_batch_id,
                                 const memory::memory_space* target_memory_space,
                                 rmm::cuda_stream_view stream) const;

  // -- Move-only (D-12/ACC-05/ACC-06) --
  read_only_data_batch(read_only_data_batch&&) noexcept            = default;
  read_only_data_batch& operator=(read_only_data_batch&&) noexcept = default;
  read_only_data_batch(const read_only_data_batch&)                = delete;
  read_only_data_batch& operator=(const read_only_data_batch&)     = delete;

 private:
  friend class data_batch;

  /**
   * @brief Private constructor -- only data_batch static methods can create instances.
   *
   * @param parent Smart pointer to the parent data_batch (moved in).
   * @param lock   Shared lock already acquired on the parent's mutex.
   */
  read_only_data_batch(PtrType parent, std::shared_lock<std::shared_mutex> lock);

  // INVARIANT: _batch must be declared before _lock -- destruction order is load-bearing.
  // When destroyed, _lock releases the shared lock first, then _batch drops the parent
  // reference. This prevents accessing a destroyed mutex.
  PtrType _batch;                               ///< Parent lifetime (destroyed second)
  std::shared_lock<std::shared_mutex> _lock;    ///< Shared lock (destroyed first)
};

/**
 * @brief RAII mutable accessor for data_batch.
 *
 * Holds an exclusive lock on the parent data_batch's mutex, permitting a single
 * writer with no concurrent readers. Provides all read methods plus write methods
 * (set_data, convert_to) that delegate to data_batch's private interface.
 *
 * Move-only. The exclusive lock is released when this object is destroyed or moved-from.
 *
 * @tparam PtrType Smart pointer type (std::shared_ptr<data_batch> or
 *         std::unique_ptr<data_batch>).
 */
template <typename PtrType>
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
  void set_data(std::unique_ptr<idata_representation> data) {
    _batch->set_data(std::move(data));
  }

  /**
   * @brief Convert the data representation in-place.
   *
   * @tparam TargetRepresentation Target representation type.
   * @param registry           Converter registry for type-keyed dispatch.
   * @param target_memory_space Target memory space for the new representation.
   * @param stream              CUDA stream for memory operations.
   */
  template <typename TargetRepresentation>
  void convert_to(representation_converter_registry& registry,
                  const memory::memory_space* target_memory_space,
                  rmm::cuda_stream_view stream) {
    _batch->template convert_to<TargetRepresentation>(registry, target_memory_space, stream);
  }

  // -- Move-only (D-12/ACC-05/ACC-06) --
  mutable_data_batch(mutable_data_batch&&) noexcept            = default;
  mutable_data_batch& operator=(mutable_data_batch&&) noexcept = default;
  mutable_data_batch(const mutable_data_batch&)                = delete;
  mutable_data_batch& operator=(const mutable_data_batch&)     = delete;

 private:
  friend class data_batch;

  /**
   * @brief Private constructor -- only data_batch static methods can create instances.
   *
   * @param parent Smart pointer to the parent data_batch (moved in).
   * @param lock   Exclusive lock already acquired on the parent's mutex.
   */
  mutable_data_batch(PtrType parent, std::unique_lock<std::shared_mutex> lock);

  // INVARIANT: _batch must be declared before _lock -- destruction order is load-bearing.
  // When destroyed, _lock releases the exclusive lock first, then _batch drops the parent
  // reference. This prevents accessing a destroyed mutex.
  PtrType _batch;                               ///< Parent lifetime (destroyed second)
  std::unique_lock<std::shared_mutex> _lock;    ///< Exclusive lock (destroyed first)
};

// =============================================================================
// Template implementations
// =============================================================================

// -- data_batch::convert_to --

template <typename TargetRepresentation>
void data_batch::convert_to(representation_converter_registry& registry,
                            const memory::memory_space* target_memory_space,
                            rmm::cuda_stream_view stream) {
  auto new_representation =
    registry.convert<TargetRepresentation>(*_data, target_memory_space, stream);
  _data = std::move(new_representation);
}

// -- data_batch::to_read_only (idle -> shared lock, TRANS-01) --

template <typename PtrType>
read_only_data_batch<PtrType> data_batch::to_read_only(PtrType&& batch) {
  auto ptr = std::move(batch);
  std::shared_lock<std::shared_mutex> lock(ptr->_rw_mutex);
  return read_only_data_batch<PtrType>(std::move(ptr), std::move(lock));
}

// -- data_batch::to_mutable (idle -> exclusive lock, TRANS-02) --

template <typename PtrType>
mutable_data_batch<PtrType> data_batch::to_mutable(PtrType&& batch) {
  auto ptr = std::move(batch);
  std::unique_lock<std::shared_mutex> lock(ptr->_rw_mutex);
  return mutable_data_batch<PtrType>(std::move(ptr), std::move(lock));
}

// -- data_batch::to_idle (release shared lock, TRANS-03) --

template <typename PtrType>
PtrType data_batch::to_idle(read_only_data_batch<PtrType>&& accessor) {
  auto ptr = std::move(accessor._batch);
  accessor._lock.unlock();
  return ptr;
}

// -- data_batch::to_idle (release exclusive lock, TRANS-04) --

template <typename PtrType>
PtrType data_batch::to_idle(mutable_data_batch<PtrType>&& accessor) {
  auto ptr = std::move(accessor._batch);
  accessor._lock.unlock();
  return ptr;
}

// -- data_batch::try_to_read_only (non-blocking, TRANS-05) --

template <typename PtrType>
std::optional<read_only_data_batch<PtrType>> data_batch::try_to_read_only(
  PtrType& batch) {
  std::shared_lock<std::shared_mutex> lock(batch->_rw_mutex, std::try_to_lock);
  if (!lock.owns_lock()) {
    return std::nullopt;
  }
  auto ptr = std::move(batch);
  return read_only_data_batch<PtrType>(std::move(ptr), std::move(lock));
}

// -- data_batch::try_to_mutable (non-blocking, TRANS-06) --

template <typename PtrType>
std::optional<mutable_data_batch<PtrType>> data_batch::try_to_mutable(
  PtrType& batch) {
  std::unique_lock<std::shared_mutex> lock(batch->_rw_mutex, std::try_to_lock);
  if (!lock.owns_lock()) {
    return std::nullopt;
  }
  auto ptr = std::move(batch);
  return mutable_data_batch<PtrType>(std::move(ptr), std::move(lock));
}

// -- read_only_data_batch::clone (deep copy, CLONE-01) --

template <typename PtrType>
PtrType read_only_data_batch<PtrType>::clone(
  uint64_t new_batch_id, rmm::cuda_stream_view stream) const {
  if (_batch->_data == nullptr) {
    throw std::runtime_error("Cannot clone: data is null");
  }
  auto cloned_data = _batch->_data->clone(stream);
  if constexpr (std::is_same_v<PtrType, std::shared_ptr<data_batch>>) {
    return std::make_shared<data_batch>(new_batch_id, std::move(cloned_data));
  } else {
    return std::make_unique<data_batch>(new_batch_id, std::move(cloned_data));
  }
}

// -- read_only_data_batch::clone_to (deep copy + conversion, CLONE-02) --

template <typename PtrType>
template <typename TargetRepresentation>
PtrType read_only_data_batch<PtrType>::clone_to(
  representation_converter_registry& registry,
  uint64_t new_batch_id,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream) const {
  auto new_representation =
    registry.convert<TargetRepresentation>(*_batch->_data, target_memory_space, stream);
  if constexpr (std::is_same_v<PtrType, std::shared_ptr<data_batch>>) {
    return std::make_shared<data_batch>(new_batch_id, std::move(new_representation));
  } else {
    return std::make_unique<data_batch>(new_batch_id, std::move(new_representation));
  }
}

// -- Accessor private constructors --

template <typename PtrType>
read_only_data_batch<PtrType>::read_only_data_batch(
  PtrType parent, std::shared_lock<std::shared_mutex> lock)
  : _batch(std::move(parent)), _lock(std::move(lock)) {
}

template <typename PtrType>
mutable_data_batch<PtrType>::mutable_data_batch(
  PtrType parent, std::unique_lock<std::shared_mutex> lock)
  : _batch(std::move(parent)), _lock(std::move(lock)) {
}

}  // namespace cucascade
