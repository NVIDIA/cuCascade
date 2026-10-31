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

#include "io/rest/s3/multipart.hpp"

#include <cucascade/io/io_request.hpp>
#include <cucascade/io/rest/authorizer.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

/**
 * @file upload.hpp
 * @brief The transport side of an S3 object opened for writing (docs/io/rest-transport.md
 *        "Writes"): the upload session (the shared, locked @ref s3::upload_state plus the write
 *        work parked on it), the jobs the worker's calls serve, and the abort sink.
 *
 * Writes of one object arrive on several reactors (fan-out, dispatch partitions), so the state
 * is shared.  Work parked on the session waiting for a settlement (the UploadId, the writes and
 * uploads a flush waits for) is resumed by the worker whose settlement returned it: that worker
 * takes it out of the session and runs it in the same pass, on its own connections.  Nothing
 * pins parked work to the reactor that parked it (coordinators are thread-safe, D2H copies end
 * before an item parks).  Only the abort of an upload destroyed uncommitted crosses threads,
 * through the @ref abort_sink.
 */

namespace cucascade::io::rest {
class rest_reactor;
}

namespace cucascade::io::rest::detail {

class upload_session;
struct flush_job;

/// What a parked write item resumes with.
enum class resume_kind : std::uint8_t {
  direct,       ///< a whole-part host write waiting for the UploadId
  part_upload,  ///< a write that completed its part, waiting for the UploadId
  flush,        ///< a flush waiting to plan (writes in flight, another flush) or for the UploadId
};

/// One unit of write work parked on a session.
struct parked_write {
  resume_kind what{resume_kind::flush};
  /// The reactor that parked it: a settlement on another reactor hands it back there (so parts
  /// upload on the reactor they were dispatched to); null runs it where it settles.
  rest_reactor* home{nullptr};
  std::unique_ptr<io_op_request> op;    ///< the write / flush operation (null: a planned flush)
  std::optional<s3::part_upload> part;  ///< part_upload: the claimed upload
  unsigned part_number{0};              ///< direct: the part the write covers
  std::shared_ptr<flush_job> job;       ///< flush: a planned flush waiting for the UploadId
};

/**
 * @brief Where an upload destroyed with an UploadId but never committed sends its
 *        AbortMultipartUpload (best effort): the reactors of the ioctx that opened it, round
 *        robin.  Owned by @c rest_ioctx; @ref close (before the reactors go) makes every later
 *        post a no-op, so a hook running on any thread never touches a destroyed reactor.
 */
class abort_sink {
 public:
  explicit abort_sink(std::vector<rest_reactor*> reactors) : _reactors(std::move(reactors)) {}

  /// Queue AbortMultipartUpload for @p upload_id on a running reactor.  Dropped (bucket
  /// lifecycle rules collect it) when none accepts work.
  void post(object_ref const& object, std::string const& upload_id) noexcept;
  /// Track @p session (an object opened for writing by this ioctx) for @ref close.
  void track(std::weak_ptr<upload_session> session);
  /// The ioctx goes: no more aborts, and every object it opened that is still alive gives back
  /// its part buffers (they belong to the context's host memory resource, which may go next);
  /// such an object fails every later write and flush with @c operation_canceled.
  void close() noexcept;

 private:
  std::mutex _mutex;
  std::vector<rest_reactor*> _reactors;
  std::vector<std::weak_ptr<upload_session>> _sessions;
  std::size_t _next{0};
  bool _closed{false};
};

/**
 * @brief One object opened for writing: the S3 upload state machine plus the write work parked
 *        on it.
 *
 * Shared by the @c rest_write_object and every job in flight on it; thread-safe.  Destroying it
 * fails every item still parked (@c operation_canceled): no item is ever lost.
 */
class upload_session {
 public:
  upload_session(object_ref object,
                 std::string path,
                 s3::upload_state::options const& opts,
                 std::weak_ptr<abort_sink> sink);
  ~upload_session();

  upload_session(upload_session const&)            = delete;
  upload_session& operator=(upload_session const&) = delete;
  upload_session(upload_session&&)                 = delete;
  upload_session& operator=(upload_session&&)      = delete;

  s3::upload_state state;
  object_ref const object;
  std::string const path;
  /// A commit request (PUT, CompleteMultipartUpload) of this object went out whole and its
  /// answer was lost, in any flush: it may have taken effect.  A later conditional 412 is then
  /// an unknown outcome, never EEXIST, and a NoSuchUpload answer is checked with a HEAD.
  std::atomic<bool> commit_unanswered{false};

  [[nodiscard]] s3::waiter_id next_waiter() noexcept
  {
    return _next_waiter.fetch_add(1, std::memory_order_relaxed);
  }

  /// Park @p item under @p id until the state hands @p id back.
  void park(s3::waiter_id id, parked_write item);
  /// Take a parked item back (nullopt when it was taken already or never parked).
  [[nodiscard]] std::optional<parked_write> unpark(s3::waiter_id id) noexcept;
  /// Take the parked items @p ids (the ones still parked), appending them to @p out.
  void take(std::vector<s3::waiter_id> const& ids, std::vector<parked_write>& out) noexcept;

 private:
  std::mutex _mutex;
  std::unordered_map<s3::waiter_id, parked_write> _parked;
  std::atomic<s3::waiter_id> _next_waiter{1};
};

/// A planned flush (commit) in progress.
struct flush_job {
  std::shared_ptr<upload_session> up;
  std::unique_ptr<io_op_request> op;  ///< the flush operation; settled once
  s3::flush_plan plan;
  std::string upload_id;
  std::size_t outstanding{0};  ///< part uploads of the flush not settled
  std::error_code first_ec;
  std::exception_ptr first_error;
  std::string xml;  ///< the CompleteMultipartUpload body (alive while Complete is in flight)
};

/// One part upload in flight: UploadPart of a write or of a flush.
struct part_job {
  std::shared_ptr<upload_session> up;
  s3::part_upload pu;
  std::unique_ptr<io_op_request> op;  ///< the write that completes with it (null: a flush's)
  std::shared_ptr<flush_job> flush;   ///< the flush it belongs to (null: a write's)
};

/// CreateMultipartUpload in flight, and the work that waits for its UploadId first.
struct create_job {
  std::shared_ptr<upload_session> up;
  parked_write pending;
};

/// Fail a parked item that cannot run with @p ec, settling its accounting in the state: a
/// direct write is abandoned, a claimed part upload finished with the error, a flush failed
/// (a planned flush's claimed uploads too).  Joined / waiting flushes the settlement returns
/// are appended to @p resumed (resume them; they plan again).
void fail_parked(upload_session& up,
                 parked_write& item,
                 std::error_code ec,
                 std::vector<s3::waiter_id>& resumed) noexcept;

/// @return a std::system_error exception for @p ec with @p what.
[[nodiscard]] std::exception_ptr as_exception(std::error_code ec, std::string const& what);

/// The ETag S3 gives an object completed from @p part_etags (quoted or not): the MD5 of the
/// concatenated binary part MD5s, then `-N`; lowercase, unquoted.  Empty when a part ETag is not
/// a plain 32-hex MD5 (SSE-KMS / SSE-C, other stores) or MD5 is unavailable.
[[nodiscard]] std::string multipart_etag(std::span<std::string const> part_etags);

/// The error code an exception carries for the state machine (which accounts errors as codes);
/// anything but a system_error counts as an I/O error.
[[nodiscard]] std::error_code code_of(std::exception_ptr const& error) noexcept;

}  // namespace cucascade::io::rest::detail
