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
/// @brief S3 error-body helpers: the drain cap for error bodies and the `<Code>` scan.
///
/// Retry, re-sign and connection-reuse policy is not decided here: the transport and the REST
/// reactor own it.

#pragma once

#include "io/rest/http/message.hpp"

#include <cstddef>
#include <string_view>

namespace cucascade::io::http {

/// Error bodies up to this size are drained into the header buffer to keep the connection warm;
/// larger ones close it.
inline constexpr std::size_t max_error_body = std::size_t{64} * 1024;

/// Tiny XML scan of an S3 error body: copies the text of the first `<Code>...</Code>` element
/// (whitespace-trimmed) found within the first `max_error_body` bytes into `meta.amz_error_code`.
/// @return true when a code was found.
bool extract_error_code(std::string_view body, response_meta& meta) noexcept;

}  // namespace cucascade::io::http
