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

/// @file request_head.hpp
/// @brief Contiguous HTTP/1.1 request head writer into caller storage (one SEND per request).
/// Contract: gpuio docs/arch/API.md §A.12. Rationale: gpuio docs/research/beast.md §2.3.5 and §3.1
/// #4 (Beast scatters the head over N+4 heap buffers; this writer writes `Name: value\r\n`
/// contiguously and records the spans so the SigV4 signer canonicalises without re-parsing).

#pragma once

#include "io/rest/http/message.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace cucascade::io::http {

/// Contiguous request head writer into caller storage (2–4 KiB per slot). Records field spans so
/// the SigV4 signer canonicalises without re-parsing. Rejects CR/LF in values (sticky overflow()).
///
/// Every builder call is a no-op once `overflow()` is set, which happens when the storage or the
/// field table is full, when a name is not a token, when a value or target contains a control
/// character (CR/LF injection), when a field is added before `start()` or after `finish()`, or
/// when `range()` gets `first > last`. Never allocates.
class request_head {
 public:
  /// Capacity of the recorded field table (`fields()`); a GET uses ~10.
  static constexpr std::size_t max_fields = 32;

  explicit request_head(std::span<char> storage) noexcept;

  /// Write the request line: `"<METHOD> <target> HTTP/1.1\r\n"`. Must be the first call.
  request_head& start(method m, std::string_view target) noexcept;  // "GET <t> HTTP/1.1\r\n"
  /// `Host: <authority>\r\n` (no whitespace allowed in the authority).
  request_head& host(std::string_view authority) noexcept;
  /// `Range: bytes=<first>-<last>\r\n` (inclusive bounds).
  request_head& range(std::uint64_t first, std::uint64_t last) noexcept;
  /// `Content-Length: <n>\r\n`.
  request_head& content_length(std::uint64_t n) noexcept;
  /// `<name>: <value>\r\n`; `name` must be a token, `value` free of control characters but HTAB.
  request_head& field(std::string_view name, std::string_view value) noexcept;
  /// `Expect: 100-continue\r\n`.
  request_head& expect_100_continue() noexcept;
  /// Append the terminating CRLF (once; later calls return the same span).
  /// @return the whole head; empty on overflow() or before start().
  [[nodiscard]] std::span<char const> finish() noexcept;  ///< appends CRLF; empty on overflow()
  /// @return every field written so far, in order (views into the storage).
  [[nodiscard]] std::span<header_ref const> fields() const noexcept;
  [[nodiscard]] bool overflow() const noexcept { return _overflow; }
  /// Start over in `storage` (which may be the same buffer).
  void reset(std::span<char> storage) noexcept;

  /// @return the method passed to start() (`get` before start()).
  [[nodiscard]] method verb() const noexcept { return _method; }
  /// @return the request target passed to start() (a view into the storage).
  [[nodiscard]] std::string_view target() const noexcept
  {
    return {_buf.data() + _target_off, _target_len};
  }
  /// @return bytes written so far.
  [[nodiscard]] std::size_t size() const noexcept { return _len; }

 private:
  bool append(std::string_view s) noexcept;
  request_head& fail() noexcept
  {
    _overflow = true;
    return *this;
  }

  std::span<char> _buf;
  std::size_t _len{0};
  std::size_t _target_off{0};
  std::size_t _target_len{0};
  std::array<header_ref, max_fields> _fields{};
  std::uint8_t _nfields{0};
  method _method{method::get};
  bool _started{false};
  bool _finished{false};
  bool _overflow{false};
};

}  // namespace cucascade::io::http
