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
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/common.hpp>

#include <atomic>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
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

// Forward declarations -- required before data_batch because it be-friends them.
class read_only_data_batch;
class mutable_data_batch;
class data_batch;

/**
 * @brief Internal data batch payload owned by data_batch.
 *
 * Owns the data representation. Data, tier, and memory-space access are exposed here,
 * but this core object is only reachable through RAII accessor types that hold the
 * appropriate lock.
 *
 * State transitions and synchronization live on data_batch. This core object only
 * exposes data/tier/memory methods to friend accessor classes.
 *
 * @note Non-copyable and non-movable. The object itself never moves.
 */
class data_batch_core {
  friend data_batch;
  friend read_only_data_batch;
  friend mutable_data_batch;

 public:
  ~data_batch_core() = default;

  // -- Deleted move/copy  --
  data_batch_core(data_batch_core&&)                 = delete;
  data_batch_core& operator=(data_batch_core&&)      = delete;
  data_batch_core(const data_batch_core&)            = delete;
  data_batch_core& operator=(const data_batch_core&) = delete;

  /**
   * @brief Get the unique batch identifier.
   *
   * Lock-free -- safe to call without acquiring an accessor.
   *
   * @return The batch ID (immutable after construction).
   */
  [[nodiscard]] uint64_t get_batch_id() const;

  // Only friend accessor classes can call these methods.

  /**
   * @brief Get the memory tier of the held data.
   * @return The current memory tier.
   */
  [[nodiscard]] memory::Tier get_current_tier() const;

  /**
   * @brief Get a raw pointer to the data representation.
   * @return Non-owning pointer to the data, or nullptr if empty.
   */
  [[nodiscard]] idata_representation* get_data() const;

  /**
   * @brief Get a raw pointer to the memory space.
   * @return Non-owning pointer to the memory space, or nullptr if data is null.
   */
  [[nodiscard]] memory::memory_space* get_memory_space() const;

  /**
   * @brief Replace the data representation.
   * @param data New data representation (takes ownership).
   */
  void set_data(std::unique_ptr<idata_representation> data) { _data = std::move(data); }

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
                  rmm::cuda_stream_view stream)
  {
    auto new_representation =
      registry.convert<TargetRepresentation>(*_data, target_memory_space, stream);
    auto old_representation = std::move(_data);
    _data                   = std::move(new_representation);

    bool needs_sync = (old_representation != nullptr &&
                       old_representation->get_current_tier() == memory::Tier::GPU) ||
                      _data->get_current_tier() == memory::Tier::GPU;

    if (needs_sync) {
      // Conversions involving GPU may enqueue async operations on the provided
      // stream that read from the source memory. Synchronize before the old
      // representation is destroyed to avoid use-after-free.
      stream.synchronize();
    }
  }

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
   * @brief Rebind the held data's device buffers to use @p stream for future deallocation.
   *
   * Forwards to gpu_table_representation::rebind_stream when the held representation is a
   * GPU table that owns its data; a no-op for every other representation (host, disk, or a
   * GPU representation backed by an externally-owned table_view).
   *
   * Rebinding before an operation that may free the data (a GPU->host downgrade, or a pipeline
   * task that consumes it) makes the buffers free on the active stream rather than the stream
   * they were produced on, keeping RMM's per-stream free lists correct and avoiding the
   * cross-stream premature-reuse hazard. Requires the exclusive (mutable) lock held by this
   * accessor.
   *
   * @note Does NOT insert cross-stream ordering -- see gpu_table_representation::rebind_stream.
   *
   * @param stream Stream used for future asynchronous deallocation of the data's buffers.
   */
  void rebind_stream(rmm::cuda_stream_view stream)
  {
    if (auto* gpu = dynamic_cast<gpu_table_representation*>(_data.get())) {
      gpu->rebind_stream(stream);
    }
  }

  /**
   * @brief Get the writer event from the underlying GPU representation, or nullptr.
   *
   * Delegates to gpu_table_representation::get_writer_event() via
   * dynamic_cast. Returns nullptr when the underlying representation is not a
   * gpu_table_representation (e.g., host or disk tier) or when no writer event has
   * been recorded yet.
   *
   * STREAM-LINEAGE: callers that cross stream / device boundaries should call
   * cudaStreamWaitEvent on the returned event (when non-null) before reading the
   * underlying memory of this batch.
   *
   * @return cudaEvent_t The writer event, or nullptr if not a GPU representation or
   *         no event recorded.
   */
  [[nodiscard]] cudaEvent_t get_writer_event() const
  {
    auto* repr = get_data();
    if (!repr) { return nullptr; }
    auto* gpu_repr = dynamic_cast<gpu_table_representation*>(repr);
    if (!gpu_repr) { return nullptr; }
    return gpu_repr->get_writer_event();
  }

 private:
  data_batch_core(uint64_t batch_id, std::unique_ptr<idata_representation> data);

  const uint64_t _batch_id;                     ///< Immutable batch identifier
  std::unique_ptr<idata_representation> _data;  ///< Owned data representation
};

/**
 * @brief Synchronized data batch type allowing thread-safe access to the core data_batch_core
 * via std::shared_mutex based RAII accessors.
 */
class data_batch : public std::enable_shared_from_this<data_batch> {
  friend read_only_data_batch;
  friend mutable_data_batch;

 public:
  /**
   * @brief Factory function to create new data_batches.
   */
  static std::shared_ptr<data_batch> make(uint64_t batch_id,
                                          std::unique_ptr<idata_representation> data);

  /**
   * @brief Get the unique batch identifier.
   *
   * Lock-free -- safe to call without acquiring an accessor
   * since batch_id is immutable after construction.
   *
   * @return The batch ID (immutable after construction).
   */
  [[nodiscard]] uint64_t get_batch_id() const { return _batch.get_batch_id(); }

  /**
   * @brief Transition from idle to read-only (shared lock) without consuming the caller's
   * pointer.
   *
   *  Blocks until the shared lock is acquired.
   *
   * @note NON-RECURSIVE: do not acquire a second accessor (read-only or mutable) for the same
   * batch on a thread that already holds one;
   * from: https://en.cppreference.com/cpp/thread/shared_mutex/lock_shared:
   * "If lock_shared is called by a thread that already owns the mutex in any mode (exclusive or
   * shared), the behavior is undefined."
   *
   * Re-acquiring AFTER the prior accessor has been released is fine.
   *
   * A held accessor must not be moved to another thread and released there;
   * from: https://en.cppreference.com/cpp/thread/shared_mutex/unlock_shared:
   * "The mutex must be locked by the current thread of execution in shared mode, otherwise, the
   * behavior is undefined."
   *
   * @return A read_only_data_batch holding the shared lock.
   */
  [[nodiscard]] read_only_data_batch get_read_only();

  /**
   * @brief Transition from idle to mutable (exclusive lock) without consuming the caller's
   * pointer.
   *
   * Uses shared_from_this() to obtain a new shared_ptr. Blocks until the
   * exclusive lock is acquired.
   *
   * @note NON-RECURSIVE: do not acquire a second accessor (read-only or mutable) for the same
   * batch on a thread that already holds one;
   * from: https://en.cppreference.com/cpp/thread/shared_mutex/lock:
   * "f lock is called by a thread that already owns the shared_mutex in any mode (exclusive or
   * shared), the behavior is undefined."
   *
   * Re-acquiring AFTER the prior accessor has been released is fine.
   *
   * A held accessor must not be moved to another thread and released there;
   * from: https://en.cppreference.com/cpp/thread/shared_mutex/unlock:
   * "The mutex must be locked by the current thread of execution, otherwise, the behavior is
   * undefined. "
   *
   * @return A mutable_data_batch holding the exclusive lock.
   */
  [[nodiscard]] mutable_data_batch get_mutable();

  /**
   * @brief Try to get a read read-only access handle (non-blocking).
   *
   * @note NON-RECURSIVE: do not call this on a thread that already holds any accessor for the
   * same batch -- recursive locking of std::shared_mutex is undefined behavior. Re-acquiring
   * AFTER the prior accessor has been released is fine. A held accessor must not be moved to
   * another thread and released there. (Debug builds detect same-thread recursion via an
   * ownership guard.)
   *
   * @return An optional containing the read-only accessor on success, or
   *         std::nullopt if the lock could not be acquired immediately.
   */
  [[nodiscard]] std::optional<read_only_data_batch> try_get_read_only();

  /**
   * @brief Try to transition from idle to mutable (non-blocking).
   *
   * @note NON-RECURSIVE: do not call this on a thread that already holds any accessor for the
   * same batch -- recursive locking of std::shared_mutex is undefined behavior. Re-acquiring
   * AFTER the prior accessor has been released is fine. A held accessor must not be moved to
   * another thread and released there. (Debug builds detect same-thread recursion via an
   * ownership guard.)
   *
   * @return An optional containing the mutable accessor on success, or
   *         std::nullopt if the lock could not be acquired immediately.
   */
  [[nodiscard]] std::optional<mutable_data_batch> try_get_mutable();

  /**
   * @brief Increment the subscriber interest count.
   */
  void subscribe();

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
  size_t get_subscriber_count() const { return _subscriber_count.load(std::memory_order_relaxed); }

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

 private:
  data_batch(uint64_t batch_id, std::unique_ptr<idata_representation> data);

  data_batch_core _batch;
  mutable std::shared_mutex _rw_mutex;

  std::atomic<size_t> _subscriber_count{0};            ///< Atomic subscriber interest count
  std::atomic<batch_state> _state{batch_state::idle};  ///< Observable lock state
  std::atomic<size_t> _read_only_count{0};  ///< Count of active read_only_data_batch instances
};

// Defined after data_batch's concrete definition to use data_batch::make
template <typename TargetRepresentation>
[[nodiscard]] std::shared_ptr<data_batch> data_batch_core::clone_to(
  representation_converter_registry& registry,
  uint64_t new_batch_id,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream) const
{
  auto new_representation =
    registry.convert<TargetRepresentation>(*_data, target_memory_space, stream);
  return data_batch::make(new_batch_id, std::move(new_representation));
}

/**
 * @brief RAII read-only accessor for data_batch.
 *
 * Holds a std::shared_lock on the owner data_batch's std::shared_mutex, permitting parallel
 * readers from multiple threads. Data is accessible through operator forwarding to data_batch_core.
 *
 * Move-only. Destruction, reset, or overwrite release the shared lock; move construction/assignment
 * transfers ownership and leaves the source invalid.
 */
class read_only_data_batch {
  friend class data_batch;

 public:
  ~read_only_data_batch();

  // -- Move-only --
  read_only_data_batch(read_only_data_batch&& other) noexcept;
  read_only_data_batch& operator=(read_only_data_batch&& other) noexcept;
  read_only_data_batch(const read_only_data_batch& other)            = delete;
  read_only_data_batch& operator=(const read_only_data_batch& other) = delete;

  // will allow only read access
  const data_batch_core* operator->() const
  {
    assert_validity();
    return &(_owner->_batch);
  }
  const data_batch_core& operator*() const
  {
    assert_validity();
    return _owner->_batch;
  }

  /**
   * @brief Whether the handle is valid or not. A handle can become invalid if the handle
   * was moved-from or reset.
   */
  [[nodiscard]] bool is_valid() const noexcept { return _owner != nullptr; }

  /**
   * @brief Get a copy of the owner of the underlying data, while retaining this read-only accessor
   * handle.
   *
   * @note The returned owner pointer must NOT be used to acquire another accessor on the thread
   * that already holds this one: accessors are NON-RECURSIVE and recursive locking of
   * std::shared_mutex is undefined behavior. Re-acquiring AFTER this accessor is released is fine.
   * A held accessor must also not be moved to another thread and released there (cross-thread
   * unlock_shared is undefined behavior).
   * Throws std::logic_error if the accessor handle was moved-from or reset.
   */
  [[nodiscard]] std::shared_ptr<data_batch> get_owner_copy() const
  {
    assert_validity();
    std::shared_ptr<data_batch> copy = this->_owner;
    return copy;
  }

  // /**
  //  * @brief Reset the passed accessor handle, releasing its read-only lock (other
  //  * read-only accessor handles continue to exist independently). Returns the underlying data
  //  owner.
  //  *
  //  * @note Calling this function invalidates this accessor handle.
  //  */
  // static std::shared_ptr<data_batch> reset_and_release_owner(read_only_data_batch&& read_only)
  // {
  //   std::shared_ptr<data_batch> copy = read_only.get_owner_copy();
  //   read_only.reset();
  //   return copy;
  // }

 private:
  /**
   * @brief Asserts whether the handle is valid or not. A handle can become invalid if the handle
   * was moved-from or reset. Throws a std::logic_error if the handle is no longer valid.
   */
  void assert_validity() const
  {
    if (not _owner) { throw std::logic_error("read_only_data_batch: invalid moved-from accessor"); }
    // INVARIANT: a valid lock is held if _owner exists.
    assert(_lock.owns_lock());
  }

  /**
   * @brief Reset the read-only accessor handle, releasing *this* read-only lock (other read-only
   * access handles continue to exist independently), and dropping the handle to the underlying
   * data owner. This handle becomes invalid after this reset.
   */
  void reset();

  /**
   * @brief Private constructor -- only data_batch methods can create instances.
   *
   * @param owner Shared pointer to the owner data_batch (moved in).
   * @param lock  Shared lock already acquired on the owner's mutex.
   */
  read_only_data_batch(std::shared_ptr<data_batch> owner, std::shared_lock<std::shared_mutex> lock);

  // Destruction order is load-bearing; but we explicitly handle destruction via reset().
  std::shared_ptr<data_batch> _owner;
  std::shared_lock<std::shared_mutex> _lock;
};

/**
 * @brief RAII mutable accessor for data_batch.
 *
 * Holds an exclusive lock on the owner data_batch's mutex, permitting a single
 * writer with no concurrent readers. Provides all read methods plus write methods
 * (set_data, convert_to) and clone operations (clone, clone_to).
 *
 * Move-only. Destruction, reset, or overwrite release the exclusive lock; move
 * construction/assignment transfers ownership and leaves the source invalid.
 */
class mutable_data_batch {
  friend class data_batch;

 public:
  ~mutable_data_batch();

  // -- Move-only --
  mutable_data_batch(mutable_data_batch&& other) noexcept;
  mutable_data_batch& operator=(mutable_data_batch&& other) noexcept;
  mutable_data_batch(const mutable_data_batch&)            = delete;
  mutable_data_batch& operator=(const mutable_data_batch&) = delete;

  data_batch_core* operator->() const
  {
    assert_validity();
    return &(_owner->_batch);
  }
  data_batch_core& operator*() const
  {
    assert_validity();
    return _owner->_batch;
  }

  /**
   * @brief Whether the handle is valid or not. A handle can become invalid if the handle
   * was moved-from.
   */
  [[nodiscard]] bool is_valid() const noexcept { return _owner != nullptr; }

  /**
   * @brief Get a copy of the owner of the underlying data, while retaining this exclusive mutable
   * accessor handle.
   *
   * @note Throws std::logic_error if the accessor handle was moved-from or reset.
   */
  [[nodiscard]] std::shared_ptr<data_batch> get_owner_copy() const
  {
    assert_validity();
    std::shared_ptr<data_batch> copy = this->_owner;
    return copy;
  }

  // /**
  //  * @brief Reset the mutable accessor handle, releasing the exclusive lock. Returns the
  //  underlying
  //  * data owner.
  //  *
  //  * @note Calling this fn invalidates this accessor handle.
  //  */
  // static std::shared_ptr<data_batch> reset_and_release_owner(mutable_data_batch&& read_write)
  // {
  //   std::shared_ptr<data_batch> copy = read_write.get_owner_copy();
  //   read_write.reset();
  //   return copy;
  // }

 private:
  /**
   * @brief Reset the mutable accessor handle, releasing the exclusive lock and dropping the
   * handle to the underlying data owner. This handle becomes invalid after this reset.
   */
  void reset();

  /**
   * @brief Asserts whether the handle is valid or not. A handle can become invalid if the handle
   * was moved-from or reset. Throws a std::logic_error if the handle is no longer valid.
   */
  void assert_validity() const
  {
    if (not _owner) { throw std::logic_error("mutable_data_batch: invalid moved-from accessor"); }
  }

  /**
   * @brief Private constructor -- only data_batch methods can create instances.
   *
   * @param owner Shared pointer to the owner data_batch (moved in).
   * @param lock   Exclusive lock already acquired on the owner's mutex.
   */
  mutable_data_batch(std::shared_ptr<data_batch> owner, std::unique_lock<std::shared_mutex> lock);

  // Destruction order is load-bearing; but we explicitly handle destruction via reset().
  std::shared_ptr<data_batch> _owner;
  std::unique_lock<std::shared_mutex> _lock;
};

template <class T>
concept data_batch_accessor =
  std::same_as<T, read_only_data_batch> || std::same_as<T, mutable_data_batch>;

/** @brief Reset the accessor handle, releasing its (shared or unique) held lock. Returns the
 * underlying data owner.
 *
 * @note Calling this function invalidates this accessor handle.
 */
template <data_batch_accessor Handle>
std::shared_ptr<data_batch> reset_and_release_owner(
  Handle handle)  // only allows std::move since the handles are move-only
{
  return handle.get_owner_copy();
  // The handle is reset via destruction at scope exit.
}

}  // namespace cucascade
