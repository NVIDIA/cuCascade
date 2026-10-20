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

#include "io/rest/errc.hpp"

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

/**
 * @file http_client.hpp
 * @brief Cold-path helpers of the cloud module (ported from gpuio's `src/auth/internal.hpp`,
 *        `http_client.cpp` and `util.cpp`, the parts the IMDS link seed needs): the process
 *        environment and a minimal blocking HTTP/1.1 client. Nothing here runs on a reactor.
 */

namespace cucascade::io::rest::cloud {

/// getenv() as an optional; unset and empty are both "absent".
[[nodiscard]] std::optional<std::string> env(char const* name);

/// @return true when the variable is set to "true" (case-insensitive, surrounding blanks
///         ignored), the AWS SDK rule for `AWS_EC2_METADATA_DISABLED`.
[[nodiscard]] bool env_true(char const* name);

/// One reply of `http_exchange`.
struct http_reply {
  unsigned status{0};
  std::string body;
};

/**
 * @brief Minimal blocking HTTP/1.1 exchange over a fresh TCP connection (`Connection: close`).
 *
 * Non-blocking connect + poll() deadline, Content-Length, chunked or read-until-close framing,
 * interim 1xx heads skipped, 16 KiB head / 64 KiB body caps. Blocking: call it from a helper
 * thread, never from a reactor.
 *
 * @param host IP literal or name; @param port TCP port; @param request full request bytes.
 * @param timeout one deadline for connect + request + response.
 * @return the reply, or `dns_failed`, `connect_failed`, `connect_timeout`, `request_timeout`,
 *         `io_error`, `http_bad_message`, `http_header_too_large`, `http_length_mismatch`.
 */
[[nodiscard]] result<http_reply> http_exchange(std::string const& host,
                                               std::uint16_t port,
                                               std::string_view request,
                                               std::chrono::milliseconds timeout) noexcept;

}  // namespace cucascade::io::rest::cloud
