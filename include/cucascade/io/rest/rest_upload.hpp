/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cucascade/io/details/runner_registry.hpp>
#include <cucascade/io/rest/authorizer.hpp>
#include <cucascade/io/rest/config.hpp>
#include <cucascade/io/rest/s3/xml_utils.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <sys/uio.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cucascade::io::rest {

/**
 * @brief Collects multipart uploads whose session was destroyed while still
 *        live (neither committed nor aborted), so they can be aborted
 *        asynchronously instead of lingering on the store.
 *
 * Owned (shared) by the @c rest_reactor and by every session it created, so
 * a session that outlives its reactor stays safe: once the reactor closed the
 * sink, @ref push refuses and the session only logs a warning.
 *
 * - @ref push (from @c ~upload_session, any thread, possibly a runner inside
 *   a completion callback): records the upload and wakes one parked runner.
 *   Never blocks beyond two short internal locks and never throws.
 * - Engines poll @ref has_pending at the top of each pass and @ref take_all
 *   to issue one asynchronous AbortMultipartUpload per entry.
 * - @c rest_reactor::abort_live_uploads (context shutdown, runners stopped)
 *   aborts what is still pending synchronously.
 * - The reactor's destructor calls @ref close; whatever is pending then is
 *   only logged.
 */
class orphan_upload_sink {
 public:
  struct entry {
    object_ref object;
    std::string upload_id;
  };

  explicit orphan_upload_sink(::cucascade::io::detail::runner_registry& registry) noexcept
    : _registry(&registry)
  {
  }

  /// Record an orphaned upload.  False when closed (or out of memory).
  bool push(object_ref const& object, std::string const& upload_id) noexcept;

  /// Whether entries are pending (relaxed hint; take_all is authoritative).
  [[nodiscard]] bool has_pending() const noexcept
  {
    return _pending.load(std::memory_order_acquire);
  }

  /// Remove and return every pending entry.
  [[nodiscard]] std::vector<entry> take_all();

  /// Refuse further pushes and drop the registry pointer; returns what was pending.
  [[nodiscard]] std::vector<entry> close() noexcept;

 private:
  std::mutex _mutex;
  ::cucascade::io::detail::runner_registry* _registry;  // guarded by _mutex; null once closed
  std::vector<entry> _entries;                          // guarded by _mutex
  std::atomic<bool> _pending{false};
};

/**
 * @brief State of one whole-object upload (an object opened for write).
 *
 * The object's byte space is cut into parts of @ref part_size bytes (part
 * @c n covers <tt>[(n-1)*part_size, n*part_size)</tt>).  Written bytes are
 * staged into per-part buffers (pinned blocks from the context's host memory
 * resource, or heap memory when there is none).  Once the upload is known to
 * be multipart (the size hint or the written high-water mark exceeds
 * @c multipart_threshold), every part that is completely staged is handed to
 * an engine for upload (UploadPart); the multipart upload itself is created
 * lazily by the first such part.  Smaller objects stay staged and are sent
 * with one PUT by the commit.  The object becomes visible only on commit
 * (PutObject, or CompleteMultipartUpload with the recorded ETags).
 *
 * Part lifecycle: @c staging (buffer allocated, bytes landing) ->
 * @c scheduled (an engine owns an upload operation for it) -> @c uploaded
 * (ETag recorded, buffer released).  An upload operation that ends without
 * success (cancelled, or failed -- which also fails the session) hands its
 * part back to @c staging.
 *
 * Several runners may stage into / upload parts of the same object
 * concurrently: every member is thread-safe (one internal mutex, never held
 * across network I/O or a copy).  Copies into a reserved range happen outside
 * the lock; the range is reserved before and landed after.
 */
class upload_session {
 public:
  /// Session lifecycle.
  enum class phase : std::uint8_t {
    open,        ///< accepting writes
    committing,  ///< commit requested; writes are refused
    committed,   ///< object visible; read-only
    failed,      ///< an upload / copy failed or the context shut down
  };

  /// Destination of one staged piece (a write range inside one part).
  struct reservation {
    std::uint32_t part{0};   ///< part number (1-based)
    std::vector<iovec> dst;  ///< staging bytes of the piece, in order
  };

  /// Outcome of @ref reserve.
  enum class reserve_status : std::uint8_t {
    reserved,  ///< @c reservation filled; copy then @ref land
    blocked,   ///< staging budget exhausted while uploads are in flight; retry later
    failed,    ///< the write cannot be staged (error set)
  };

  /// Body of one upload operation.
  struct payload {
    std::uint32_t part{0};        ///< part number (0 for a single PUT)
    range rng{};                  ///< object byte range carried
    std::vector<iovec> iov;       ///< staged bytes, in order
    std::shared_ptr<void> owner;  ///< keeps the staging alive while the upload uses it
  };

  /// Upload work produced by @ref land / @ref plan_commit.
  struct upload_work {
    std::vector<payload> parts;  ///< parts to upload (now scheduled)
    bool initiate{false};        ///< caller must send CreateMultipartUpload first
  };

  /// What a commit must do (see @ref plan_commit).
  struct commit_plan {
    enum class action : std::uint8_t {
      wait,        ///< staging / uploads still in flight; ask again later
      fail,        ///< @c error is set
      single_put,  ///< send @c put (one PUT of the whole object)
      multipart,   ///< send @c work, then CompleteMultipartUpload
    };
    action what{action::wait};
    std::exception_ptr error;
    payload put;
    upload_work work;
  };

  /**
   * @param object Target bucket / key.
   * @param cfg Upload tunables (already clamped by the reactor).
   * @param size_hint Expected final size (0: unknown).
   * @param host_mr Staging resource; null stages in heap memory.
   */
  upload_session(object_ref object,
                 rest_write_config const& cfg,
                 std::uint64_t size_hint,
                 cucascade::memory::fixed_size_host_memory_resource* host_mr);

  /**
   * @brief Abort a still-live multipart upload left behind.
   *
   * If the session dies with an upload id that was neither completed nor
   * aborted (the object was dropped without commit and before context
   * shutdown) and an @ref orphan_upload_sink is attached, the upload is
   * handed to the sink: a runner of the context aborts it asynchronously
   * (or the shutdown sweep synchronously).  Without a sink, or once the
   * reactor is gone, only a warning is logged and the upload stays on the
   * store until a bucket lifecycle rule removes it.  Never blocks.
   */
  ~upload_session();

  /// Attach the sink that receives this session's upload if it is orphaned.
  /// Call before the session is shared.
  void set_orphan_sink(std::shared_ptr<orphan_upload_sink> sink) noexcept
  {
    _orphan_sink = std::move(sink);
  }

  /**
   * @brief A failed session that only carries @p upload_id, for an engine to
   *        send AbortMultipartUpload of an orphaned upload (no sink attached:
   *        if that abort fails too, its destructor only warns).
   */
  [[nodiscard]] static std::shared_ptr<upload_session> for_abort(object_ref object,
                                                                 std::string upload_id);

  upload_session(upload_session const&)            = delete;
  upload_session& operator=(upload_session const&) = delete;

  [[nodiscard]] object_ref const& object() const noexcept { return _object; }
  [[nodiscard]] std::size_t part_size() const noexcept { return _part_size; }
  [[nodiscard]] std::size_t multipart_threshold() const noexcept { return _threshold; }

  /// High-water mark of the written range (the object size once committed).
  [[nodiscard]] std::size_t size() const noexcept { return _size.load(std::memory_order_acquire); }

  [[nodiscard]] phase current_phase() const;

  /// Whether writes are accepted (phase @c open).
  [[nodiscard]] bool writable() const { return current_phase() == phase::open; }

  /// Part number holding object byte @p offset.
  [[nodiscard]] std::uint32_t part_of(std::size_t offset) const noexcept;

  /// Object range of part @p part_number (full part size; not clamped to the size).
  [[nodiscard]] range part_range(std::uint32_t part_number) const noexcept;

  // -- staging --------------------------------------------------------------------

  /**
   * @brief Reserve @p rng (inside one part) for a copy.
   *
   * Allocates the part's staging when it has none (subject to the
   * @c max_buffered_parts back-pressure) and raises the high-water mark.
   * Fails when the session is not open (@c std::invalid_argument after a
   * commit, the stored failure otherwise), the part was already uploaded
   * (@c std::invalid_argument: an S3 object cannot be rewritten in place), or
   * staging cannot be allocated while nothing could free any.
   */
  [[nodiscard]] reserve_status reserve(range rng, reservation& out, std::exception_ptr& error);

  /**
   * @brief Complete a reserved copy into @p part_number.
   *
   * @param error Null on success; otherwise the copy failed and the session
   *        fails with it.
   * @return Parts that became ready to upload (multipart) -- the caller now
   *         owns their upload -- and whether it must initiate the upload.
   */
  [[nodiscard]] upload_work land(std::uint32_t part_number, std::exception_ptr error);

  // -- upload operations ---------------------------------------------------------

  /// The upload id (empty until CreateMultipartUpload succeeded).
  [[nodiscard]] std::string upload_id() const;

  /// True while scheduled parts must wait for an upload id that is still coming.
  [[nodiscard]] bool awaiting_upload_id() const;

  /// Claim the creation of the multipart upload (no upload id, none being
  /// created, session not failed).  True: the caller must send it.
  [[nodiscard]] bool claim_initiate();

  /// CreateMultipartUpload succeeded.
  void initiate_succeeded(std::string upload_id);

  /// A CreateMultipartUpload operation ended without an upload id (failure or
  /// cancellation); another one may be started.
  void initiate_released() noexcept;

  /// UploadPart of @p part_number succeeded: record @p etag, release staging.
  void part_uploaded(std::uint32_t part_number, std::string etag);

  /// An upload of @p part_number ended without success: back to @c staging.
  void part_released(std::uint32_t part_number) noexcept;

  /// Uploaded parts, ascending (for CompleteMultipartUpload).
  [[nodiscard]] std::vector<s3::part_record> part_records() const;

  // -- failure / abort -----------------------------------------------------------

  /// The failure that ended the session (null unless @c failed).
  [[nodiscard]] std::exception_ptr failure() const;

  /**
   * @brief Fail the session with @p error (first failure wins; no-op once
   *        committed) and release idle staging.
   *
   * @return True when the caller should send AbortMultipartUpload (claimed
   *         once: an upload id exists and no abort was claimed yet).
   */
  bool fail(std::exception_ptr error) noexcept;

  /// AbortMultipartUpload succeeded (or the upload no longer exists).
  void abort_succeeded() noexcept;

  /**
   * @brief Context shutdown: fail an uncommitted session with
   *        @c std::errc::operation_canceled and release its staging.
   *
   * @return The upload id to abort, or empty when there is nothing to abort
   *         (committed, no multipart upload, or already aborted).
   */
  [[nodiscard]] std::string cancel_for_shutdown() noexcept;

  // -- commit ------------------------------------------------------------------------

  /// Start a commit: @c open -> @c committing.  Returns the error that
  /// prevents it (already committing / committed: @c std::invalid_argument;
  /// failed: the stored failure), or null.
  [[nodiscard]] std::exception_ptr begin_commit();

  /**
   * @brief Decide the commit's network work once the session is quiescent.
   *
   * Waits (@c action::wait) while copies land, parts upload or an upload is
   * being created.  Fails when the session failed or the staged object has a
   * hole (every byte of <tt>[0, size())</tt> must have been written).  An
   * object without an upload id and at most @c multipart_threshold bytes is a
   * single PUT; otherwise every remaining part is scheduled (the last one may
   * be short) and the upload must then be completed.
   */
  [[nodiscard]] commit_plan plan_commit();

  /// The object is committed: release all staging; the session is read-only.
  void mark_committed() noexcept;

 private:
  enum class part_status : std::uint8_t { staging, scheduled, uploaded };

  struct part_state {
    std::vector<iovec> blocks;                    ///< staging blocks (block-size each)
    std::shared_ptr<void> owner;                  ///< owns the blocks
    std::map<std::size_t, std::size_t> coverage;  ///< staged [begin, end) in the part, merged
    std::size_t filled{0};                        ///< bytes covered by @c coverage
    std::size_t copies{0};                        ///< reserved copies not landed yet
    part_status status{part_status::staging};
    std::string etag;
  };

  enum class initiate_state : std::uint8_t { none, in_flight, done };

  void allocate_staging(part_state& part);
  [[nodiscard]] payload make_payload(std::uint32_t part_number,
                                     part_state const& part,
                                     std::size_t bytes) const;
  [[nodiscard]] bool multipart_decided() const noexcept;
  void release_idle_staging() noexcept;
  [[nodiscard]] std::size_t staged_parts() const noexcept;
  [[nodiscard]] std::size_t scheduled_parts() const noexcept;
  [[nodiscard]] std::size_t expected_part_bytes(std::uint32_t part_number,
                                                std::size_t size) const noexcept;

  object_ref const _object;
  std::size_t const _part_size;
  std::size_t const _threshold;
  std::size_t const _max_buffered_parts;
  std::uint64_t const _size_hint;
  cucascade::memory::fixed_size_host_memory_resource* const _host_mr;

  std::atomic<std::size_t> _size{0};

  mutable std::mutex _mutex;
  phase _phase{phase::open};                         // guarded by _mutex
  std::exception_ptr _failure;                       // guarded by _mutex
  std::string _upload_id;                            // guarded by _mutex
  initiate_state _initiate{initiate_state::none};    // guarded by _mutex
  bool _abort_claimed{false};                        // guarded by _mutex
  bool _aborted{false};                              // guarded by _mutex
  std::shared_ptr<orphan_upload_sink> _orphan_sink;  // set before sharing
  std::map<std::uint32_t, part_state> _parts;        // guarded by _mutex
};

}  // namespace cucascade::io::rest
