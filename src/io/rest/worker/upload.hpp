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
#include "io/rest/worker/op_request.hpp"

#include <cucascade/io/rest/rest_reactor.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace cucascade::io::rest::detail {

class upload_session;
struct flush_job;

/// What a parked write operation resumes with.
enum class resume_kind : std::uint8_t {
  direct,        ///< a whole-part host write waiting for the UploadId
  part_upload,   ///< a write that completed its part, waiting for the UploadId
  flush,         ///< a flush waiting for writes in flight, another flush, or the UploadId
  abort_upload,  ///< send AbortMultipartUpload (best effort; no operation)
};

/// One unit of write work handed between threads.
struct mail {
  resume_kind what{resume_kind::flush};
  std::shared_ptr<upload_session> upload;
  std::unique_ptr<rest_io_op_request> op;
  std::optional<s3::part_upload> part;  ///< part_upload: the claimed upload
  unsigned part_number{0};              ///< direct: the part the write covers
  std::shared_ptr<flush_job> job;       ///< flush: a planned flush waiting for the UploadId
  std::error_code ec;                   ///< non-empty: fail instead of resuming
  object_ref object;                    ///< abort_upload
  std::string upload_id;                ///< abort_upload
};

/**
 * @brief Cross-thread inbox of one reactor's worker for write work (writes of one object may
 *        run on every reactor of the ioctx; a write parked on one is resumed by another).
 *
 * Any thread may @ref post; the owner drains it every pass.  After @ref close a post fails and
 * the caller settles the work itself.
 */
class mailbox {
 public:
  /// @param notify called after a successful post (wakes the owner's loop).
  explicit mailbox(std::function<void()> notify) : _notify(std::move(notify)) {}

  /// @return false when the owner is gone (closed): @p m is left with the caller.
  bool post(mail& m) noexcept;
  void drain(std::vector<mail>& out) noexcept;
  [[nodiscard]] bool empty() const noexcept { return _count.load(std::memory_order_acquire) == 0; }
  void close() noexcept;

 private:
  std::mutex _mutex;
  std::vector<mail> _items;
  std::atomic<std::size_t> _count{0};
  bool _closed{false};
  std::function<void()> _notify;
};

/**
 * @brief The transport side of one object opened for writing: the S3 upload state machine plus
 *        the write operations parked on it, and where each must be resumed.
 *
 * Shared by the @c rest_write_object and every operation in flight on it; thread-safe.
 */
class upload_session : public std::enable_shared_from_this<upload_session> {
 public:
  upload_session(object_ref object,
                 std::string path,
                 s3::upload_state::options const& opts,
                 std::weak_ptr<mailbox> abort_home);

  upload_session(upload_session const&)            = delete;
  upload_session& operator=(upload_session const&) = delete;

  s3::upload_state state;
  object_ref const object;
  std::string const path;

  [[nodiscard]] s3::waiter_id next_waiter() noexcept
  {
    return _next_waiter.fetch_add(1, std::memory_order_relaxed);
  }

  /// Park @p m under @p id until the state hands @p id back; @p home resumes it.
  void park(s3::waiter_id id, std::weak_ptr<mailbox> home, mail m);
  /// Take a parked item back (null when it was resumed already or never parked).
  [[nodiscard]] std::optional<mail> unpark(s3::waiter_id id) noexcept;
  /// Hand the parked items @p ids to their homes (with @p ec when non-empty); items whose home
  /// is gone are failed here.
  void resume(std::vector<s3::waiter_id> const& ids, std::error_code ec) noexcept;

 private:
  struct parked {
    std::weak_ptr<mailbox> home;
    mail m;
  };

  std::mutex _mutex;
  std::unordered_map<s3::waiter_id, parked> _parked;
  std::atomic<s3::waiter_id> _next_waiter{1};
};

/// Fail a write / flush operation that cannot run (its upload gone, its home closed).
void fail_mail(mail& m, std::error_code ec) noexcept;

/// Settle a planned flush that cannot continue: its claimed part uploads fail with @p ec, the
/// flush operation fails, waiters are resumed (defined with the engine's write path).
void fail_flush_job(flush_job& job, std::error_code ec) noexcept;

}  // namespace cucascade::io::rest::detail
