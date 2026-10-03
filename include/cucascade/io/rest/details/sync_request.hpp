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

#include <cucascade/io/rest/authorizer.hpp>
#include <cucascade/io/rest/config.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cucascade::io::rest::detail {

/// Knobs for one @c perform_sync call.
struct sync_request_options {
  /// HTTP statuses that complete the request successfully.
  std::vector<long> accepted_statuses{200};
  /// Treat an accepted status whose body is an S3 @c <Error> document as a
  /// retriable failure (S3 CompleteMultipartUpload may answer 200 + @c <Error>).
  bool retry_on_error_body{false};
  /// Bound the transfer by the stall detector (@c config::stall_speed_limit_bytes /
  /// @c stall_time_s) instead of the whole-request @c config::request_timeout_s —
  /// for bodies too large for a fixed deadline (e.g. a part upload).
  bool data_transfer{false};
};

/// Outcome of a successful @c perform_sync call.
struct sync_response {
  long status{0};           ///< final HTTP status (one of the accepted statuses)
  std::string body;         ///< response body (empty for HEAD)
  std::string etag;         ///< @c ETag response header, quotes included (empty if absent)
  std::size_t attempts{0};  ///< HTTP attempts made, including the successful one
  /// @c Content-Length of the final response when the server sent one (for a
  /// HEAD: the object size).
  std::optional<std::size_t> content_length;
};

/**
 * @brief Perform one object-store request synchronously on the calling thread,
 *        with the REST layer's retry policy.
 *
 * Each attempt re-authorizes @p spec via
 * @c request_authorizer::authorize_request (fresh presigned URL / signature),
 * uses a one-shot easy handle on the process-wide DNS/TLS share, and sends
 * @p body: @c PUT uploads it (@c CURLOPT_UPLOAD + read/seek callbacks),
 * @c POST posts it, @c GET / @c HEAD / @c DELETE_ send none (@p body must be
 * empty). @c Expect: 100-continue is suppressed and, for @c POST, curl's default
 * form @c Content-Type is removed unless @p spec supplies one.
 *
 * Retried (up to @c config::max_retry_attempts attempts, backoff as for GETs,
 * honoring @c Retry-After): transient curl errors, HTTP 408/429/500/502/503/504,
 * HTTP 400 with S3 code @c RequestTimeout, and — when
 * @c sync_request_options::retry_on_error_body — an accepted status carrying an
 * @c <Error> body. HTTP 403 is retried at most @c config::max_auth_retry_attempts
 * times (credential refresh / clock skew).
 *
 * Intended for control-plane calls on threads that may block: shutdown-time
 * AbortMultipartUpload, tests, and (later) the HEAD / LIST helpers.
 *
 * @param spec        Request to authorize and send.
 * @param authorizer  Signs every attempt.
 * @param cfg         Timeouts, TLS, and retry settings.
 * @param body        Request body; must stay valid for the call.
 * @param opts        Accepted statuses and retry / timeout knobs.
 * @return The final response on an accepted status.
 * @throw std::invalid_argument for a body on a method that does not carry one.
 * @throw cucascade::io::credential_error when the authorizer fails.
 * @throw std::runtime_error on a non-retriable failure or exhausted retries:
 *        @c "rest: <VERB> HTTP <status> for bucket/key[: <S3 Code>]".
 */
[[nodiscard]] sync_response perform_sync(request_spec const& spec,
                                         request_authorizer& authorizer,
                                         config const& cfg,
                                         std::string_view body            = {},
                                         sync_request_options const& opts = {});

}  // namespace cucascade::io::rest::detail
