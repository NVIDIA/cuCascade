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

#include <cucascade/cuda/event.hpp>
#include <cucascade/data/common.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/common.hpp>

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

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
class idata_batch_probe;

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
  friend class read_only_data_batch;
  friend class mutable_data_batch;

 public:
  /**
   * @brief Factory function to create new data_batches.
   *
   * @param batch_id Unique identifier for this batch (immutable after construction).
   * @param data     Owned data representation; must not be null.
   * @return A shared pointer owning the new batch.
   * @throws std::runtime_error if data is null.
   */
  static std::shared_ptr<data_batch> make(
    uint64_t batch_id,
    std::unique_ptr<idata_representation> data,
    std::unique_ptr<idata_batch_probe> probe = std::make_unique<idata_batch_probe>());

  ~data_batch();

  // -- Deleted move/copy --
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
   * Atomic, lock-free.
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
   * exclusive lock is acquired and all recorded asynchronous readers have completed.
   *
   * @return A mutable_data_batch holding the exclusive lock.
   * @throws cucascade::cuda_error if a recorded reader event cannot be synchronized.
   * @throws rmm::cuda_error if a reader event's CUDA device cannot be made current.
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
   * @return An optional containing the mutable accessor on success, or std::nullopt if the lock
   *         could not be acquired immediately or a recorded asynchronous reader is still pending.
   * @throws rmm::cuda_error if a reader event's CUDA device cannot be made current for its query.
   * @throws cucascade::cuda_error if a reader event's query reports a CUDA failure.
   */
  [[nodiscard]] std::optional<mutable_data_batch> try_to_mutable();

  // -- Locked-to-locked static transitions --

  /**
   * @brief Transition from read-only to mutable (upgrade lock).
   *
   * Releases the shared lock, then acquires an exclusive lock (may block).
   * Waits for all recorded asynchronous readers before exposing mutable access.
   * The source accessor is consumed via move.
   * NOTE: The transition is not atomic.
   *
   * @param accessor Rvalue reference to the read-only accessor (consumed).
   * @return A mutable_data_batch holding the exclusive lock.
   * @throws cucascade::cuda_error if a recorded reader event cannot be synchronized.
   * @throws rmm::cuda_error if a reader event's CUDA device cannot be made current.
   */
  [[nodiscard]] static mutable_data_batch readonly_to_mutable(read_only_data_batch&& accessor);

  /**
   * @brief Transition from mutable to read-only (downgrade lock).
   *
   * Releases the exclusive lock, then acquires a shared lock (may block).
   * The source accessor is consumed via move.
   * NOTE: The transition is not atomic.
   *
   * @param accessor Rvalue reference to the mutable accessor (consumed).
   * @return A read_only_data_batch holding the shared lock.
   */
  [[nodiscard]] static read_only_data_batch mutable_to_readonly(mutable_data_batch&& accessor);

 private:
  data_batch(uint64_t batch_id,
             std::unique_ptr<idata_representation> data,
             std::unique_ptr<idata_batch_probe> probe);

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
   * @brief Record completion of asynchronous work reading the current representation.
   *
   * GPU batches retain one event for every outstanding reader stream. Non-GPU tiers are a no-op.
   * The caller must hold a shared lock while recording so the representation cannot change between
   * issuing the read and recording its completion.
   *
   * @param reader_stream Stream on which work reading the batch was enqueued.
   */
  void record_reader_event(rmm::cuda_stream_view reader_stream);

  /**
   * @brief Block until all recorded asynchronous readers have completed.
   *
   * Called while the batch's exclusive lock is held before mutable access is exposed.
   */
  void synchronize_reader_events();

  /**
   * @brief Query and recycle completed reader events without blocking.
   *
   * @return true when no recorded asynchronous reader remains in flight.
   * @throws cucascade::cuda_error if an event's query reports a CUDA failure.
   */
  [[nodiscard]] bool reader_events_complete();

  /**
   * @brief Destructor-safe form of synchronize_reader_events().
   */
  void synchronize_reader_events_no_throw() noexcept;

  struct reader_event_pool {
    // Events are stored as a pending prefix followed by reusable completed events. This bounds
    // event creation by the peak number of overlapping reads registered on this CUDA device.
    std::vector<cuda::cuda_event> events;
    std::size_t pending_event_count{0};
  };

  /**
   * @brief Move completed events from the pending prefix back into the reusable pool.
   *
   * Requires _reader_events_mutex to be held and the pool's CUDA device to be current.
   * A query that reports failure is propagated rather than counted as still-pending: a failed
   * event never completes, so absorbing it would stall every later mutable acquisition.
   *
   * @throws cucascade::cuda_error if an event's query reports a CUDA failure. @p pool is left
   *         unchanged from the failing event onward.
   */
  void recycle_completed_reader_events(reader_event_pool& pool);

  const uint64_t _batch_id;                            ///< Immutable batch identifier
  std::unique_ptr<idata_representation> _data;         ///< Owned data representation
  mutable std::shared_mutex _rw_mutex;                 ///< Reader-writer mutex
  std::atomic<size_t> _subscriber_count{0};            ///< Atomic subscriber interest count
  std::atomic<batch_state> _state{batch_state::idle};  ///< Observable lock state
  std::atomic<size_t> _read_only_count{0};  ///< Count of active read_only_data_batch instances

  // CUDA events are device-associated, so each device needs an independent reusable pool.
  std::mutex _reader_events_mutex;
  std::unordered_map<int, reader_event_pool> _reader_event_pools;

  std::unique_ptr<idata_batch_probe> _probe;
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
  // -- Named accessor methods --

  /** @brief Get the batch identifier. */
  uint64_t get_batch_id() const { return _batch->get_batch_id(); }

  /** @brief Get the memory tier of the held data. */
  memory::Tier get_current_tier() const { return _batch->get_current_tier(); }

  /** @brief Get a raw pointer to the data representation. */
  [[nodiscard]] const idata_representation* get_data() const { return _batch->get_data(); }

  /** @brief Get a raw pointer to the memory space. */
  memory::memory_space* get_memory_space() const { return _batch->get_memory_space(); }

  /**
   * @brief Get the writer event from the underlying representation, or nullptr.
   *
   * Delegates polymorphically to idata_representation::get_writer_event().
   * Returns nullptr when there is no underlying representation, when the representation's
   * tier records no writer event (the base-class default, e.g. host or disk tier), or when
   * no writer event has been recorded yet.
   *
   * STREAM-LINEAGE: callers that cross stream / device boundaries should call
   * cudaStreamWaitEvent on the returned event (when non-null) before reading the
   * underlying memory of this batch.
   *
   * @return cudaEvent_t The writer event, or nullptr if none is available.
   */
  [[nodiscard]] cudaEvent_t get_writer_event() const
  {
    auto* repr = get_data();
    return repr ? repr->get_writer_event() : nullptr;
  }

  /**
   * @brief Record completion of asynchronous work reading this batch's memory.
   *
   * Call this after enqueueing all work that reads the batch on @p reader_stream and before
   * releasing this accessor. Releasing the shared lock does not wait for the event. The next
   * mutable accessor waits for every outstanding reader event before exposing the representation,
   * preventing reuse, replacement, conversion, or downgrade while a device read is still active.
   *
   * This is a no-op for non-GPU tiers. Multiple readers and multiple calls are supported.
   * Events are pooled independently per CUDA device. If event registration fails after the read
   * has been enqueued, this call synchronizes @p reader_stream before propagating the error so an
   * untracked read can never escape the shared-lock lifetime.
   *
   * @note The shared-lock release itself never synchronizes. If this accessor owns the final
   *       shared_ptr to the batch, batch destruction synchronizes as a lifetime-safety fallback
   *       because there is no later reclaimer that can pay the wait.
   *
   * @param reader_stream Stream on which work reading the batch was enqueued.
   * @throws cucascade::cuda_error if the stream cannot be queried, synchronized during fallback,
   *         or have its event recorded, or if recycling finds an already-recorded event whose
   *         query reports a CUDA failure.
   * @throws rmm::cuda_error if the reader stream's CUDA device cannot be made current.
   * @throws std::bad_alloc if the per-device event pool cannot grow. The reader stream is
   *         synchronized before this exception propagates.
   */
  void record_reader_event(rmm::cuda_stream_view reader_stream) const
  {
    _batch->record_reader_event(reader_stream);
  }

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
  [[nodiscard]] std::shared_ptr<data_batch> clone(
    uint64_t new_batch_id,
    rmm::cuda_stream_view stream,
    std::unique_ptr<idata_batch_probe> probe = std::make_unique<idata_batch_probe>()) const;

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
   * @param probe               Optional probe for the cloned batch.
   * @return A new data_batch wrapped in shared_ptr.
   */
  template <typename TargetRepresentation>
  [[nodiscard]] std::shared_ptr<data_batch> clone_to(
    representation_converter_registry& registry,
    uint64_t new_batch_id,
    const memory::memory_space* target_memory_space,
    rmm::cuda_stream_view stream,
    std::unique_ptr<idata_batch_probe> probe = std::make_unique<idata_batch_probe>()) const;

  /**
   * @brief Deep copy + conversion, drawing the target allocation from a reservation.
   *
   * Like the memory_space overload, but the target space is derived from @p reservation and the
   * allocation draws down that reservation instead of committing fresh capacity.
   *
   * @tparam TargetRepresentation Target representation type.
   * @param registry     Converter registry for type-keyed dispatch.
   * @param new_batch_id Batch ID for the cloned batch.
   * @param reservation  Caller-owned reservation on the target memory space.
   * @param stream       CUDA stream for memory operations.
   * @param probe        Optional probe for the cloned batch.
   * @return A new data_batch wrapped in shared_ptr.
   */
  template <typename TargetRepresentation>
  [[nodiscard]] std::shared_ptr<data_batch> clone_to(
    representation_converter_registry& registry,
    uint64_t new_batch_id,
    memory::reservation& reservation,
    rmm::cuda_stream_view stream,
    std::unique_ptr<idata_batch_probe> probe = std::make_unique<idata_batch_probe>()) const;

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
  std::shared_ptr<data_batch> _batch;         ///< Parent lifetime (destroyed second)
  std::shared_lock<std::shared_mutex> _lock;  ///< Shared lock (destroyed first)
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
  // -- Read methods (same as read_only) --

  /** @brief Get the batch identifier. */
  uint64_t get_batch_id() const { return _batch->get_batch_id(); }

  /** @brief Get the memory tier of the held data. */
  memory::Tier get_current_tier() const { return _batch->get_current_tier(); }

  /** @brief Get a raw pointer to the data representation. */
  idata_representation* get_data() const { return _batch->get_data(); }

  /** @brief Get a raw pointer to the memory space. */
  memory::memory_space* get_memory_space() const { return _batch->get_memory_space(); }

  // -- Write methods --

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

  /**
   * @brief Convert the data representation in-place, drawing the target allocation from a
   * reservation.
   *
   * Like the memory_space overload, but the target space is derived from @p reservation and the
   * allocation draws down that reservation instead of committing fresh capacity (avoids
   * double-counting on the HOST tier). If the conversion involves the GPU tier, synchronizes the
   * stream before the old representation is destroyed to prevent use-after-free.
   *
   * @tparam TargetRepresentation Target representation type.
   * @param registry    Converter registry for type-keyed dispatch.
   * @param reservation Caller-owned reservation on the target memory space.
   * @param stream      CUDA stream for memory operations.
   */
  template <typename TargetRepresentation>
  void convert_to(representation_converter_registry& registry,
                  memory::reservation& reservation,
                  rmm::cuda_stream_view stream);

  /**
   * @brief Rebind the held data's device buffers to use @p stream for future deallocation.
   *
   * Delegates polymorphically to idata_representation::rebind_stream. Representations that own
   * stream-ordered device memory (e.g. a GPU table that owns its data) rebind their buffers; it
   * is a no-op for every other representation (host, disk, or a GPU representation backed by an
   * externally-owned table_view).
   *
   * Rebinding before an operation that may free the data (a GPU->host downgrade, or a pipeline
   * task that consumes it) makes the buffers free on the active stream rather than the stream
   * they were produced on, keeping RMM's per-stream free lists correct and avoiding the
   * cross-stream premature-reuse hazard. Requires the exclusive (mutable) lock held by this
   * accessor.
   *
   * @note Does NOT insert cross-stream ordering -- see idata_representation::rebind_stream.
   *
   * @param stream Stream used for future asynchronous deallocation of the data's buffers.
   */
  void rebind_stream(rmm::cuda_stream_view stream);

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
  [[nodiscard]] std::shared_ptr<data_batch> clone(
    uint64_t new_batch_id,
    rmm::cuda_stream_view stream,
    std::unique_ptr<idata_batch_probe> probe = std::make_unique<idata_batch_probe>()) const;

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
   * @param probe               Optional probe for the cloned batch.
   * @return A new data_batch wrapped in shared_ptr.
   */
  template <typename TargetRepresentation>
  [[nodiscard]] std::shared_ptr<data_batch> clone_to(
    representation_converter_registry& registry,
    uint64_t new_batch_id,
    const memory::memory_space* target_memory_space,
    rmm::cuda_stream_view stream,
    std::unique_ptr<idata_batch_probe> probe = std::make_unique<idata_batch_probe>()) const;

  /**
   * @brief Deep copy + conversion, drawing the target allocation from a reservation.
   *
   * Like the memory_space overload, but the target space is derived from @p reservation and the
   * allocation draws down that reservation instead of committing fresh capacity.
   *
   * @tparam TargetRepresentation Target representation type.
   * @param registry     Converter registry for type-keyed dispatch.
   * @param new_batch_id Batch ID for the cloned batch.
   * @param reservation  Caller-owned reservation on the target memory space.
   * @param stream       CUDA stream for memory operations.
   * @param probe        Optional probe for the cloned batch.
   * @return A new data_batch wrapped in shared_ptr.
   */
  template <typename TargetRepresentation>
  [[nodiscard]] std::shared_ptr<data_batch> clone_to(
    representation_converter_registry& registry,
    uint64_t new_batch_id,
    memory::reservation& reservation,
    rmm::cuda_stream_view stream,
    std::unique_ptr<idata_batch_probe> probe = std::make_unique<idata_batch_probe>()) const;

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
  mutable_data_batch(std::shared_ptr<data_batch> parent, std::unique_lock<std::shared_mutex> lock);

  /**
   * @brief Swap in a freshly converted representation, synchronizing @p stream first when the
   * conversion touched the GPU tier.
   *
   * Shared tail of both convert_to overloads. A GPU conversion may still have async work on
   * @p stream reading the old representation's buffers; synchronize before the old representation
   * is destroyed to avoid use-after-free.
   */
  void install_converted_representation(std::unique_ptr<idata_representation> new_representation,
                                        rmm::cuda_stream_view stream)
  {
    auto old_representation = std::move(_batch->_data);
    _batch->_data           = std::move(new_representation);

    bool needs_sync = old_representation != nullptr &&
                      (old_representation->get_current_tier() == memory::Tier::GPU ||
                       _batch->_data->get_current_tier() == memory::Tier::GPU);
    if (needs_sync) { stream.synchronize(); }
  }

  // INVARIANT: _batch must be declared before _lock -- destruction order is load-bearing.
  // When destroyed, _lock releases the exclusive lock first, then _batch drops the parent
  // reference. This prevents accessing a destroyed mutex.
  std::shared_ptr<data_batch> _batch;         ///< Parent lifetime (destroyed second)
  std::unique_lock<std::shared_mutex> _lock;  ///< Exclusive lock (destroyed first)
};

/**
 * @brief Interface for probing the data_batch class.
 *
 * Applications may implement this interface to hold additional application specific
 * data_batch metadata while probing the data_batch by overriding the provided methods
 * that expose the data_batch state when certain events occur, like state transitions.
 *
 * @note All callbacks are invoked either during batch construction (created) or while
 * the batch's exclusive lock is held via mutable_data_batch, so per batch they are
 * totally ordered and never run concurrently. It is the implementer's responsibility
 * that they return quickly, as they block other mutating changes while they execute.
 * This interface is primarily intended for bookkeeping purposes. Default impl is no-op
 */
class idata_batch_probe {
 public:
  idata_batch_probe()          = default;
  virtual ~idata_batch_probe() = default;

  virtual void created([[maybe_unused]] const uint64_t batch_id,
                       [[maybe_unused]] const idata_representation& data) noexcept
  {
  }
  virtual void conversion_started(
    [[maybe_unused]] const idata_representation& current_data,
    [[maybe_unused]] const memory::memory_space* target_memory_space) noexcept
  {
  }
  virtual void conversion_completed([[maybe_unused]] const idata_representation& data,
                                    [[maybe_unused]] const bool success) noexcept
  {
  }
  virtual void data_replaced([[maybe_unused]] const idata_representation& new_data) noexcept {}
};

// =============================================================================
// Template implementations (TargetRepresentation-templated methods only)
// =============================================================================

template <typename TargetRepresentation>
std::shared_ptr<data_batch> read_only_data_batch::clone_to(
  representation_converter_registry& registry,
  uint64_t new_batch_id,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream,
  std::unique_ptr<idata_batch_probe> probe) const
{
  auto new_representation =
    registry.convert<TargetRepresentation>(*_batch->_data, target_memory_space, stream);
  return data_batch::make(new_batch_id, std::move(new_representation), std::move(probe));
}

template <typename TargetRepresentation>
std::shared_ptr<data_batch> read_only_data_batch::clone_to(
  representation_converter_registry& registry,
  uint64_t new_batch_id,
  memory::reservation& reservation,
  rmm::cuda_stream_view stream,
  std::unique_ptr<idata_batch_probe> probe) const
{
  auto new_representation =
    registry.convert<TargetRepresentation>(*_batch->_data, reservation, stream);
  return data_batch::make(new_batch_id, std::move(new_representation), std::move(probe));
}

// -- mutable_data_batch::convert_to (in-place conversion) --

template <typename TargetRepresentation>
void mutable_data_batch::convert_to(representation_converter_registry& registry,
                                    const memory::memory_space* target_memory_space,
                                    rmm::cuda_stream_view stream)
{
  _batch->_probe->conversion_started(*(_batch->_data), target_memory_space);
  bool conversion_succeeded = false;
  // small helper that gets called when this scope is exited (including during exceptions)
  struct OnScopeExit {
    const std::unique_ptr<idata_batch_probe>& probe;
    // a ref to the unique ptr so when data is updated,
    // that new data is supplied with the notification.
    const std::unique_ptr<idata_representation>& data;
    const bool& success;

    ~OnScopeExit() { probe->conversion_completed(*data, success); }
  } on_scope_exit{
    .probe   = _batch->_probe,
    .data    = _batch->_data,
    .success = conversion_succeeded,
  };

  install_converted_representation(
    registry.convert<TargetRepresentation>(*_batch->_data, target_memory_space, stream), stream);
  conversion_succeeded = true;  // ref used by on_scope_exit helper.
}

template <typename TargetRepresentation>
void mutable_data_batch::convert_to(representation_converter_registry& registry,
                                    memory::reservation& reservation,
                                    rmm::cuda_stream_view stream)
{
  _batch->_probe->conversion_started(*(_batch->_data), &reservation.get_memory_space());
  bool conversion_succeeded = false;
  // small helper that gets called when this scope is exited (including during exceptions)
  struct OnScopeExit {
    const std::unique_ptr<idata_batch_probe>& probe;
    // a ref to the unique ptr so when data is updated,
    // that new data is supplied with the notification.
    const std::unique_ptr<idata_representation>& data;
    const bool& success;

    ~OnScopeExit() { probe->conversion_completed(*data, success); }
  } on_scope_exit{
    .probe   = _batch->_probe,
    .data    = _batch->_data,
    .success = conversion_succeeded,
  };

  install_converted_representation(
    registry.convert<TargetRepresentation>(*_batch->_data, reservation, stream), stream);
  conversion_succeeded = true;  // ref used by on_scope_exit helper.
}

template <typename TargetRepresentation>
std::shared_ptr<data_batch> mutable_data_batch::clone_to(
  representation_converter_registry& registry,
  uint64_t new_batch_id,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream,
  std::unique_ptr<idata_batch_probe> probe) const
{
  auto new_representation =
    registry.convert<TargetRepresentation>(*_batch->_data, target_memory_space, stream);
  return data_batch::make(new_batch_id, std::move(new_representation), std::move(probe));
}

template <typename TargetRepresentation>
std::shared_ptr<data_batch> mutable_data_batch::clone_to(
  representation_converter_registry& registry,
  uint64_t new_batch_id,
  memory::reservation& reservation,
  rmm::cuda_stream_view stream,
  std::unique_ptr<idata_batch_probe> probe) const
{
  auto new_representation =
    registry.convert<TargetRepresentation>(*_batch->_data, reservation, stream);
  return data_batch::make(new_batch_id, std::move(new_representation), std::move(probe));
}

}  // namespace cucascade
