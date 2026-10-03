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

#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <stop_token>

namespace cucascade::io::detail {
class runner_slot;
}  // namespace cucascade::io::detail

namespace cucascade::io::rest {

class rest_reactor;

/**
 * @brief Per-runner REST transfer engine (the @c engine_type of @c rest_reactor).
 *
 * Created by @c rest_reactor::make_engine on the runner thread, driven there by
 * @ref run and destroyed there.  Owns everything the old reactor worker loop
 * kept as locals: a libcurl multi handle driven by an epoll loop
 * (@c curl_multi_socket_action) with three timerfds (libcurl timeout, retry
 * backoff, connection upkeep), a connection-sharing @c curl_share (connections
 * are thread-confined, so one cache per engine), the pool of reusable easy
 * handles (one per connection), the retry heap, CUDA events for staged
 * device reads, and the active grouped requests this runner expands.
 *
 * Work arrives through the reactor's @c request_hub: the engine pulls grouped
 * requests as its @c scheduling_policy allows (several at once: up to
 * @c max_active_groups bulk plus @c max_latency_groups latency groups with
 * GETs still to dispatch; read groups whose GETs are all on connections are
 * bounded by the connection pool instead),
 * expands one slice at a time per group into ranged GETs, and dispatches the
 * groups' operations round-robin, gated by @c scheduling_policy::may_dispatch
 * with connections as the capacity axis.  The runner's eventfd is part of the
 * epoll set, so new work, stop requests and warm-up requests wake it.
 *
 * Write requests stage their segments into the object's @c upload_session
 * (host memcpy, or a device-to-host copy on the caller's stream plus an event)
 * and upload the parts they complete (CreateMultipartUpload lazily, then
 * UploadPart with per-part retries); commit requests send the single PUT or
 * the remaining parts plus CompleteMultipartUpload (chained through the
 * coordinator finalizer).  Upload operations share the connection pool with
 * GETs: data uploads within the policy's write share, control-plane requests
 * on the latency budget.  A failed upload aborts the multipart upload.
 *
 * Leaving @ref run: while the context still admits work (user stop token or
 * deadline) untaken reads are requeued for other runners, write / commit
 * requests (whose staging state lives here) are finished in place, and
 * in-flight transfers are finished; on context shutdown everything owned is cancelled
 * with @c std::errc::operation_canceled (in-flight transfers are aborted, as
 * before the runner model).
 *
 * Not thread-safe: confined to its runner thread.
 */
class rest_engine {
 public:
  /// @throws std::runtime_error if the curl multi / epoll / timer setup fails.
  rest_engine(rest_reactor& owner, ::cucascade::io::detail::runner_slot& slot);
  ~rest_engine();

  rest_engine(rest_engine const&)            = delete;
  rest_engine& operator=(rest_engine const&) = delete;
  rest_engine(rest_engine&&)                 = delete;
  rest_engine& operator=(rest_engine&&)      = delete;

  /**
   * @brief Serve the reactor's queue until @p stop fires or @p deadline passes,
   *        then leave (requeue / drain, or cancel on shutdown).
   *
   * @return Number of grouped requests this engine retired.
   * @throws Any fatal engine error, after the work it owned was settled with it.
   */
  std::size_t run(std::stop_token stop,
                  std::optional<std::chrono::steady_clock::time_point> deadline);

 private:
  class impl;
  std::unique_ptr<impl> _impl;
};

}  // namespace cucascade::io::rest
