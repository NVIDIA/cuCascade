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

/// @file content_range.hpp
/// @brief `Content-Range` value parser (Beast has none: gpuio docs/research/beast.md §2.7).
/// Forms (RFC 9110 §14.4): `bytes first-last/complete`, `bytes first-last/*`, `bytes */complete`.
/// S3 sends `bytes a-b/total` on 206 and `bytes */total` on 416 (s3_line_rate.md §2.1,
/// dev_box_probes.md §7).

#pragma once

#include "io/rest/http/message.hpp"

#include <string_view>
#include <system_error>

namespace cucascade::io::http {

/// Parse a `Content-Range` field value (surrounding whitespace tolerated, unit case-insensitive).
/// @param v the field value.
/// @param out [out] written only on success; `complete == unknown_length` for `/*`;
///        `unsatisfied` (with `first == last == 0`) for `*/complete`.
/// @return `errc::http_bad_message` on syntax errors, overflow, `first > last` or
///         `last >= complete`; success otherwise.
[[nodiscard]] std::error_code parse_content_range(std::string_view v, content_range& out) noexcept;

}  // namespace cucascade::io::http
