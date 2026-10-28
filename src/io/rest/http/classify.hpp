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

/// @file classify.hpp
/// @brief Pure retry/reuse classification of a finished or failed exchange.
/// Contract: gpuio docs/arch/API.md §A.12. Table: gpuio ARCHITECTURE §7.3 (from
/// gpuio docs/research/s3_line_rate.md §2.8 and the AWS SDK 2026 retry behaviour).

#pragma once

#include "io/rest/http/message.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <system_error>

namespace cucascade::io::http {

/// Error bodies up to this size are drained into the header buffer to keep the connection warm;
/// larger ones close it (gpuio ARCHITECTURE §7.3, last paragraph).
inline constexpr std::size_t max_error_body = std::size_t{64} * 1024;

/// A `200` to a ranged GET is accepted (and sliced by the driver) only when the whole object is at
/// most this large and at most twice the requested length (gpuio ARCHITECTURE §7.3).
inline constexpr std::uint64_t max_sliced_object = std::uint64_t{1024} * 1024;

struct outcome {
  std::error_code ec;  ///< rest::errc or http_status_category
  bool retryable{false};
  enum class retry_class : std::uint8_t { none, free_stale, transient, throttle, auth } cls{};
  bool reuse_connection{true};
  bool resume_allowed{false};
};

/// Pure classification of a finished (or failed) exchange (gpuio ARCHITECTURE §7.3).
///
/// Usable both right after the head is parsed (validate before the body lands; `bytes_received`
/// 0) and after a failure. Transfer completeness is the driver's job: it reports a short body as
/// `transport_error` (`errc::http_length_mismatch` or the socket errno).
///
/// @param meta parsed head (`status == 0` when no status line arrived); `amz_error_code` filled by
///        `extract_error_code` when an error body was drained.
/// @param transport_error socket/TLS/codec failure of this exchange, or success.
/// @param m request method.
/// @param requested_first first byte requested (GET).
/// @param requested_last last byte requested, inclusive (GET; may exceed the object end).
/// @param object_size known object size; 0 means unknown (a 206 is then checked against the
///        request only).
/// @param bytes_received body bytes already delivered into the destination (resume offset).
/// @param connection_was_reused the exchange ran on a pooled keep-alive connection.
[[nodiscard]] outcome classify(response_meta const& meta,
                               std::error_code transport_error,
                               method m,
                               std::uint64_t requested_first,
                               std::uint64_t requested_last,
                               std::uint64_t object_size,
                               std::uint64_t bytes_received,
                               bool connection_was_reused) noexcept;

/// Tiny XML scan of an S3 error body: copies the text of the first `<Code>...</Code>` element
/// (whitespace-trimmed) found within the first `max_error_body` bytes into `meta.amz_error_code`.
/// @return true when a code was found.
bool extract_error_code(std::string_view body, response_meta& meta) noexcept;

}  // namespace cucascade::io::http
