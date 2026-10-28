/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file remote_io.hpp
 * @brief Private helpers the REMOTE converters use to move one batch object through an io context.
 *
 * The counterpart of disk_io.hpp for objects: a batch is written as one object (opened
 * create_new, committed by flush) and read back through the committed object. Every helper waits
 * for its requests before it returns or unwinds, so once it is left no I/O touches the caller's
 * buffers any more.
 */

#include <cucascade/cuda/stream.hpp>
#include <cucascade/exec/admission_control.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/io/types.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace cucascade {

namespace io {
class ioctx;
}  // namespace io

namespace memory {
class memory_space;
}  // namespace memory

namespace detail {

/**
 * @brief The io context make_remote_io_context() builds: a REST context that knows the budget of
 *        its pinned staging, so the REMOTE converters can stay inside it.
 *
 * A batch is written in windows of whole parts, and device reads in windows whose staging (a
 * whole number of blocks per GET, see @ref device_read_staging) fits half the budget; every
 * conversion first reserves what its window takes from @ref staging_budget, so concurrent
 * conversions queue instead of exhausting the staging (which would fail them).
 */
class remote_rest_ioctx final : public io::rest::rest_ioctx {
 public:
  /// @param staging_bytes Total pinned staging of @p ctx (all reactors).
  /// @param staging_block Its block size.
  remote_rest_ioctx(std::size_t n_reactors,
                    std::shared_ptr<io::rest::rest_reactor::reactor_context> ctx,
                    std::size_t staging_bytes,
                    std::size_t staging_block);

  [[nodiscard]] exec::admission_control& staging_budget() noexcept { return _budget; }
  [[nodiscard]] std::size_t staging_bytes() const noexcept { return _budget.budget(); }
  [[nodiscard]] std::size_t staging_block() const noexcept { return _block; }

  /// The bytes of one S3 part of an object of @p object_size bytes (the write split, rounded
  /// down to the write granule, or larger for an object needing more than 10000 parts).
  [[nodiscard]] std::size_t part_bytes(std::size_t object_size) const noexcept;

  /// Upper bound of the pinned staging a device read of @p bytes contiguous bytes takes from an
  /// object of parts of @p part bytes: every GET the transport cuts it into (counted with
  /// @c io::rest::max_read_gets, the planner's own bound, plus the part boundaries crossed)
  /// allocates whole staging blocks.
  [[nodiscard]] std::size_t device_read_staging(std::size_t bytes, std::size_t part) const noexcept;

 private:
  exec::admission_control _budget;
  std::size_t _block;
};

/**
 * @brief An object written by the REMOTE converters (see remote_data_representation).
 *
 * Holds the context that wrote it and the committed write object (reads go through it, so they
 * carry the commit's ETag as If-Match and need no HEAD). Destroying it hands both to the
 * removal_queue, which deletes the object; it never blocks and never does I/O itself.
 */
struct remote_object {
  remote_object(std::shared_ptr<io::ioctx> context,
                std::shared_ptr<io::io_object> object,
                std::string uri,
                std::size_t size) noexcept;
  ~remote_object() noexcept;

  remote_object(remote_object const&)            = delete;
  remote_object& operator=(remote_object const&) = delete;

  std::shared_ptr<io::ioctx> context;
  std::shared_ptr<io::io_object> object;
  std::string uri;
  std::size_t size{0};
};

/**
 * @brief The process-wide owner of REMOTE object removals in flight.
 *
 * One thread, never a reactor thread, issues each removal (DeleteObject, io_priority::low) and
 * holds the context that issues it, and the written object, until the removal settled; then it
 * releases them on its own thread. So a removal is never cancelled because the space, manager or
 * representation that held the context went away first, a context's last reference is never
 * dropped on one of its own reactor threads (which would join itself), and no caller ever blocks
 * on a removal. Failures are logged (rate-limited); the bucket's lifecycle rule collects what is
 * left. Leaked on purpose (it may be used during static destruction); drained at normal exit.
 */
class removal_queue {
 public:
  /// The queue, created (with its thread) on the first call. Created eagerly by
  /// make_remote_io_context() and before every REMOTE write, so it exists before any object it
  /// may have to remove. @throws if the queue or its thread cannot be created.
  [[nodiscard]] static removal_queue& instance();

  /// The queue if it was created, else null. Never throws, never creates.
  [[nodiscard]] static removal_queue* existing() noexcept;

  /// Queue the removal of @p uri through @p context. Never blocks, never throws, and never
  /// releases @p context on the calling thread (which may be one of its reactor threads): on an
  /// allocation failure the reference is kept for good instead.
  void submit(std::shared_ptr<io::ioctx> context,
              std::shared_ptr<io::io_object> object,
              std::string uri) noexcept;

  /// Wait until every removal submitted before the call (and since) has settled and released its
  /// context. No deadline of its own (each removal is bounded by the transport's timeouts and
  /// retries). Must not be called on a reactor thread.
  void drain() noexcept;

  /// The outcome of a scoped drain (see drain(contexts, timeout)).
  struct drain_outcome {
    std::size_t abandoned{0};  ///< removals of the contexts cancelled or never issued
    bool timed_out{false};     ///< the deadline expired and the contexts were shut down
  };

  /**
   * @brief Wait until every removal submitted through one of @p contexts has settled and released
   *        its context, for at most @p timeout; at the deadline, cancel them.
   *
   * Cancelling: removals of these contexts still queued are never issued, each context is shut
   * down (which settles its in-flight, retrying and queued calls), and all their entries are
   * released on the calling thread. Never depends on the queue's thread making progress (it may
   * be blocked in a removal of another context). Null and duplicate entries are skipped.
   *
   * @pre Nothing submits further removals through @p contexts; the contexts of concurrent scoped
   *      drains are disjoint; not called on a reactor thread of one of @p contexts nor from one of
   *      their completion callbacks.
   */
  [[nodiscard]] drain_outcome drain(std::span<std::shared_ptr<io::ioctx> const> contexts,
                                    std::chrono::milliseconds timeout) noexcept;

  removal_queue(removal_queue const&)            = delete;
  removal_queue& operator=(removal_queue const&) = delete;

 private:
  removal_queue();
  ~removal_queue() = default;

  struct entry_state {
    std::exception_ptr error;  ///< written before `done`
    std::atomic<bool> done{false};
  };
  struct entry {
    std::shared_ptr<io::ioctx> context;
    std::shared_ptr<io::io_object> object;
    std::string uri;
    std::shared_ptr<entry_state> state;
    /// `context.get()`: the key of _outstanding_by_context, kept past the release of `context`.
    io::ioctx const* key{nullptr};
  };
  /// The contexts a scoped drain is cancelling; owned by that drain (on its stack) and linked
  /// into _abandoned while it cancels, so marking them allocates nothing under the lock.
  struct abandon_scope {
    std::span<io::ioctx const* const> keys;
    abandon_scope* next{nullptr};
  };

  void run(std::stop_token const& stop) noexcept;
  void pass(std::stop_token const& stop);
  void issue(entry& e) noexcept;

  /// Under _mutex: whether a scoped drain is cancelling @p key's removals.
  [[nodiscard]] bool abandoned(io::ioctx const* key) const noexcept;
  /// Under _mutex: no removal of @p keys is outstanding.
  [[nodiscard]] bool settled_for(std::span<io::ioctx const* const> keys) const noexcept;
  /// Under _mutex: account for @p released, whose contexts and objects were released.
  void forget(std::list<entry> const& released) noexcept;

  std::mutex _mutex;
  std::condition_variable_any _wake;
  std::condition_variable_any _idle;
  /// Lists, so entries move between them by splicing (no allocation once submitted). Every entry
  /// is in one of them, except while the worker or a scoped drain is releasing it.
  std::list<entry> _queued;
  std::list<entry> _inflight;
  std::size_t _outstanding{0};  ///< submitted and not yet released
  /// Outstanding removals per context (a key is erased at 0). An entry holds its context while
  /// counted, so the address is not reused meanwhile.
  std::unordered_map<io::ioctx const*, std::size_t> _outstanding_by_context;
  abandon_scope* _abandoned{nullptr};   ///< scoped drains cancelling their contexts
  io::ioctx const* _issuing{nullptr};   ///< the context the worker is issuing through
  std::exception_ptr _abandoned_error;  ///< the error of a removal never issued
  std::jthread _worker;                 ///< last: started once every other member exists
};

/**
 * @brief The io context of a REMOTE memory space for a conversion into it.
 *
 * The caller holds the returned reference for the whole conversion.
 *
 * @throws std::logic_error if the space has no io context (there is no default for REMOTE), or
 *         it cannot write (device memory, when @p device).
 */
[[nodiscard]] std::shared_ptr<io::ioctx> require_remote_io_context(
  memory::memory_space const& remote_space, bool device);

/**
 * @brief Write one batch object under @p prefix and commit it.
 *
 * Opens a fresh name (memory::generate_remote_object_uri) with write_mode::create_new, writes
 * every byte of [0, object_size): @p slices plus zeros for the gaps between them (so every part
 * completes and uploads while the batch is written), and flushes (CompleteMultipartUpload or one
 * PUT). With a context from make_remote_io_context() the write goes in windows of whole parts
 * that fit its staging, else as one request. A name that turns out to be taken (412 on the
 * conditional commit) is replaced once by a fresh one. On any failure the unflushed object is
 * destroyed, which aborts its upload, and the error is rethrown.
 *
 * @param slices Non-overlapping, inside [0, object_size); device memory when @p device_stream is
 *        set (then read in stream order on it; call with the source device current).
 * @throws std::system_error / std::runtime_error on any I/O, CUDA or short-write failure.
 */
[[nodiscard]] std::shared_ptr<remote_object const> write_remote_object(
  std::shared_ptr<io::ioctx> context,
  std::string_view prefix,
  std::span<io::write_slice const> slices,
  std::size_t object_size,
  std::optional<::cuda::stream_ref> device_stream);

/**
 * @brief Read ranges of a batch object into host buffers and wait.
 *
 * @throws std::system_error / std::runtime_error on any I/O or short-read failure.
 */
void read_remote_object_host(remote_object const& object, std::span<io::slice const> slices);

/**
 * @brief Read ranges of a batch object into device buffers and wait.
 *
 * Same contract as read_disk_file_device(): the destinations are written only after the work
 * enqueued on @p stream before the call, and are complete when this returns. Call with the
 * destination device current.
 *
 * @throws std::logic_error if the context cannot read into device memory.
 * @throws std::system_error / std::runtime_error on any I/O, CUDA or short-read failure.
 */
void read_remote_object_device(remote_object const& object,
                               std::span<io::slice const> slices,
                               ::cuda::stream_ref stream);

}  // namespace detail
}  // namespace cucascade
