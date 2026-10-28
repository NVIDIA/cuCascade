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

/// @file message.hpp
/// @brief HTTP/1.1 codec vocabulary: methods, framing, parse status, `response_meta`.
/// Contract: gpuio docs/arch/API.md §A.12. Design: gpuio docs/research/beast.md §3.2–3.3 (a
/// fixed-struct "Fields" policy instead of Beast's per-field heap nodes; no allocation anywhere in
/// the codec).

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string_view>

namespace cucascade::io::http {

enum class method : std::uint8_t { get, head, put, post, delete_ };

/// How the body of a parsed response is delimited (RFC 9112 §6.3, Beast `finish_header`).
enum class framing : std::uint8_t {
  none,      ///< no body: HEAD, 1xx, 204, 304 (Content-Length may still be reported)
  length,    ///< Content-Length bytes follow the header
  chunked,   ///< Transfer-Encoding with `chunked` as the final coding: use `chunked_decoder`
  until_eof  ///< no length information: the body runs to connection close (never keep-alive)
};

enum class parse_status : std::uint8_t { ok, need_more, error };

/// Detail of the last codec error (diagnostics and conformance tests). The `std::error_code`
/// returned to callers carries the coarse `rest::errc` (`http_bad_message`,
/// `http_header_too_large`, `http_length_mismatch`); this enum names the Beast rule that fired.
enum class parse_error : std::uint8_t {
  none,
  bad_version,
  bad_status,
  bad_reason,
  bad_line_ending,
  bad_field,
  bad_value,
  bad_content_length,
  multiple_content_length,
  bad_transfer_encoding,
  bad_chunk,
  bad_chunk_extension,
  header_limit,
  body_limit,
};

/// Sentinel for "length not known" (`Content-Range: bytes a-b/*`, chunked/EOF-delimited bodies).
inline constexpr std::uint64_t unknown_length = std::numeric_limits<std::uint64_t>::max();

/// Parsed `Content-Range` (`bytes first-last/complete` or `bytes */complete`).
/// `complete == unknown_length` for the `/*` form; `unsatisfied` for the `*/complete` (416) form.
struct content_range {
  std::uint64_t first{0}, last{0}, complete{0};
  bool unsatisfied{false};
};

/// Fixed-capacity string for the few captured header values; truncation sets `truncated`.
template <std::size_t N>
struct inline_string {  // fixed storage; truncation sets `truncated`
  static_assert(N > 0 && N <= std::numeric_limits<std::uint16_t>::max());
  std::array<char, N> data{};
  std::uint16_t len{0};
  bool truncated{false};
  [[nodiscard]] std::string_view view() const noexcept { return {data.data(), len}; }
  [[nodiscard]] bool empty() const noexcept { return len == 0; }
  void clear() noexcept
  {
    len       = 0;
    truncated = false;
  }
  /// Copy `s`, keeping at most N bytes.
  /// @return false when `s` was truncated.
  bool assign(std::string_view s) noexcept
  {
    std::size_t const n = s.size() < N ? s.size() : N;
    if (n != 0) s.copy(data.data(), n);
    len       = static_cast<std::uint16_t>(n);
    truncated = n < s.size();
    return !truncated;
  }
};

/// Everything the codec decodes from a response header (and, for `amz_error_code`, from a drained
/// XML error body via `extract_error_code`). Keep-alive and framing come from parser flags, never
/// from a field lookup.
struct response_meta {
  std::uint16_t status{0};
  std::uint8_t version{11};
  bool keep_alive{true};
  framing body{framing::none};
  std::optional<std::uint64_t> content_length{};
  std::optional<content_range> range{};
  inline_string<80> etag;  ///< quotes kept (sent back verbatim as If-Match)
  inline_string<40> request_id;
  inline_string<128> amz_id_2;
  inline_string<32> date;
  inline_string<48> amz_error_code;  ///< from a drained XML error body
  std::uint32_t retry_after_ms{0};   ///< x-amz-retry-after (ms) or Retry-After (s); capped 30 s
  /// A numeric Retry-After / x-amz-retry-after was present (`retry_after_ms` 0 then means "now",
  /// not "none": cuCascade's `honor_retry_after` replaces the backoff with it).
  bool retry_after_present{false};
  bool truncated_fields{false};  ///< some captured value did not fit its inline_string
};

/// A (name, value) pair; views into caller storage.
struct header_ref {
  std::string_view name, value;
};

/// Response parser options.
struct parse_options {
  /// Bytes allowed for the status line(s) plus fields of one exchange, including skipped 1xx
  /// heads. 16 KiB: `x-amz-meta-*` user metadata can reach ~2 KB (beast.md §2.7) and the
  /// per-connection header buffer is 16 KiB (gpuio ARCHITECTURE §3.5).
  std::uint32_t header_limit{16 * 1024};
  bool head_request{false};  ///< response to HEAD: never a body, Content-Length is the object size
};

}  // namespace cucascade::io::http
