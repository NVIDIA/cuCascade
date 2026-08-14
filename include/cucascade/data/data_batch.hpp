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

#include <cucascade/cuda/event_pool.hpp>
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

  ~data_batch() = default;

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

  // -- Consumer-event API --
  //
  // Mirror of the writer-event mechanism (idata_representation::record_writer_event /
  // get_writer_event). Writer events let a cross-stream READER order its reads after the
  // producing stream's writes; consumer events let a RECLAIMER order the destruction of
  // this batch's buffers after every consumer stream's reads. Without them, freeing a
  // representation whose buffers were produced on an idle stream while a consumer
  // stream's kernels/copies still read them is a use-after-free — the only prior defense
  // was a blunt host stream.synchronize() of the consumer stream at every call site.
  //
  // Contract:
  //   - A consumer calls record_consumer_event(s) AFTER enqueuing its reads of this
  //     batch's buffers on stream s (typically while still holding the shared lock via
  //     read_only_data_batch, which also delegates this call).
  //   - A reclaimer calls await_consumers(t) BEFORE destroying or replacing the batch's
  //     representation, where t is the stream that processes (or is host-synchronized
  //     ahead of) the frees. Reclaimers hold the exclusive lock (mutable_data_batch), so
  //     no new consumer can record concurrently with the await + destroy sequence — any
  //     reader that could still enqueue reads also still holds the shared lock, which
  //     blocks the reclaimer from acquiring the exclusive lock in the first place.
  //   - Consumer events complement rebind_stream (see mutable_data_batch::rebind_stream):
  //     rebinding only moves which stream frees the buffers and inserts NO cross-stream
  //     ordering; await_consumers supplies that ordering device-side.
  //
  // These methods are internally synchronized by their own mutex and are independent of
  // the reader-writer lock — like subscribe()/unsubscribe() they may be called on the
  // idle handle at any time.

  /**
   * @brief Record that the calling consumer has enqueued reads of this batch's device
   *        buffers on @p consumer_stream.
   *
   * Call AFTER the reads are enqueued (kernels launched / async copies issued) — the
   * recorded event captures the stream's contents at call time, so reads enqueued later
   * are not covered. May be called multiple times, from multiple threads, for multiple
   * streams; each call tracks one more outstanding event.
   *
   * Cheap: a pooled cudaEvent record (created lazily with cudaEventDisableTiming and
   * recycled once complete). Never host-syncs. Thread-safe.
   *
   * @param consumer_stream The stream on which the consumer's reads were enqueued.
   */
  void record_consumer_event(rmm::cuda_stream_view consumer_stream);

  /**
   * @brief Enqueue device-side waits on @p stream for all outstanding consumer events
   *        (reads previously recorded via record_consumer_event()).
   *
   * Issues cudaStreamWaitEvent(@p stream, event) per outstanding event, so work enqueued
   * on @p stream afterwards — notably stream-ordered frees of this batch's buffers — is
   * ordered after every recorded consumer read. Does NOT host-sync. Completed events are
   * recycled. Thread-safe.
   *
   * Reclaimers must call this on the stream that will process the frees (the buffers'
   * bound stream) before destroying the representation, or combine it with existing
   * host-sync semantics (a host sync of @p stream after this call transitively waits on
   * all consumers, making frees safe on any stream).
   *
   * @param stream The stream on which to enqueue the waits.
   */
  void await_consumers(rmm::cuda_stream_view stream);

  /**
   * @brief True if no recorded consumer work is still pending on the device.
   *
   * Non-blocking poll (cudaEventQuery) of all outstanding consumer events. Trivially
   * true for batches that never recorded a consumer event. Thread-safe.
   */
  [[nodiscard]] bool consumers_done() const;

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
   * NOTE: The transition is not atomic.
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

  const uint64_t _batch_id;                            ///< Immutable batch identifier
  std::unique_ptr<idata_representation> _data;         ///< Owned data representation
  mutable std::shared_mutex _rw_mutex;                 ///< Reader-writer mutex
  std::atomic<size_t> _subscriber_count{0};            ///< Atomic subscriber interest count
  std::atomic<batch_state> _state{batch_state::idle};  ///< Observable lock state
  std::atomic<size_t> _read_only_count{0};  ///< Count of active read_only_data_batch instances

  /// Outstanding consumer-read events (see the consumer-event API above). Lazily
  /// populated: batches that never record a consumer event hold no CUDA events and no
  /// heap allocations here.
  cuda::event_pool _consumer_events;

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
   * @brief Record that this consumer has enqueued reads of the batch's device buffers
   *        on @p consumer_stream.
   *
   * Delegates to data_batch::record_consumer_event(). Mirror of get_writer_event():
   * where the writer event orders this consumer's reads after the producer's writes,
   * the consumer event orders a later reclaimer's frees after this consumer's reads.
   * Call after enqueuing the reads, while still holding this shared lock.
   *
   * @param consumer_stream The stream on which the reads were enqueued.
   */
  void record_consumer_event(rmm::cuda_stream_view consumer_stream) const
  {
    _batch->record_consumer_event(consumer_stream);
  }

  /**
   * @brief Enqueue device-side waits on @p stream for all outstanding consumer events.
   *
   * Delegates to data_batch::await_consumers(). See that method for the reclaim
   * contract.
   */
  void await_consumers(rmm::cuda_stream_view stream) const { _batch->await_consumers(stream); }

  /**
   * @brief True if no recorded consumer work is still pending on the device.
   *
   * Delegates to data_batch::consumers_done().
   */
  [[nodiscard]] bool consumers_done() const { return _batch->consumers_done(); }

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
   *       To order the rebound buffers' eventual free after consumer reads still in flight
   *       on other streams, call await_consumers() on the rebound stream — the two calls
   *       together make a reclaim fully stream-ordered with no host sync.
   *
   * @param stream Stream used for future asynchronous deallocation of the data's buffers.
   */
  void rebind_stream(rmm::cuda_stream_view stream);

  /**
   * @brief Enqueue device-side waits on @p stream for all outstanding consumer events.
   *
   * Delegates to data_batch::await_consumers(). Reclaimers holding this exclusive lock
   * call this on the stream that will process the frees BEFORE destroying or replacing
   * the representation; the exclusive lock guarantees no new consumer can record
   * concurrently. Does NOT host-sync.
   */
  void await_consumers(rmm::cuda_stream_view stream) const { _batch->await_consumers(stream); }

  /**
   * @brief True if no recorded consumer work is still pending on the device.
   *
   * Delegates to data_batch::consumers_done().
   */
  [[nodiscard]] bool consumers_done() const { return _batch->consumers_done(); }

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

    // CONSUMER-EVENTS: consumers may still have reads of the old representation's
    // buffers in flight on their own streams (recorded via record_consumer_event).
    // Enqueue device-side waits on @p stream BEFORE the synchronize below so the host
    // sync — and therefore the destruction of the old representation — is also ordered
    // after every recorded consumer read, regardless of which stream frees the buffers.
    // We hold the exclusive lock, so no new consumer can record concurrently. No-op for
    // batches that never recorded a consumer event.
    _batch->await_consumers(stream);

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
