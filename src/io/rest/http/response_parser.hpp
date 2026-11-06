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

/// @file response_parser.hpp
/// @brief Header-only HTTP/1.1 response parser: Beast `basic_parser<false>` rules with
/// `eager(false)`, concrete, allocation-free, stopping exactly at the body start.
/// Contract: gpuio docs/arch/API.md §A.12. Algorithm and framing rules: gpuio
/// docs/research/beast.md §2.2 (ported from `boost/beast/http/impl/basic_parser.ipp` and
/// `detail/basic_parser.ipp`). Use in the remote read: gpuio ARCHITECTURE §3.5 (header RECV into
/// the 16 KiB buffer, `parse_header`, leftover memcpy into the destination, then `consume_direct`
/// per body RECV).

#pragma once

#include "io/rest/http/fields.hpp"
#include "io/rest/http/message.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <system_error>
#include <type_traits>

namespace cucascade::io::http {

/// Header-only parser (Beast basic_parser rules, concrete, no allocation). parse_header consumes
/// complete lines; on `ok`, `consumed` == body offset; 1xx (except 101) are skipped internally.
///
/// Incremental protocol: `consumed` counts the bytes of *this call's* `in` that the parser took
/// (complete status lines and fields; a partial line is never consumed). On `need_more` the caller
/// keeps `in[consumed, size)`, appends newly received bytes and calls again with that span. On
/// `ok`, `in[consumed, size)` is body (or the next message for bodiless responses).
///
/// Framing (Beast `do_field` / `finish_header`, RFC 9112 §6): Content-Length lists must agree;
/// Content-Length together with Transfer-Encoding is rejected in either order; `chunked` counts
/// only as the final coding and only once; HEAD, 1xx, 204 and 304 have no body; otherwise the body
/// runs to EOF and the connection is not reusable.
class response_parser {
  /// The default `on_unknown_field` (a no-op); recognised so the parser skips the call entirely.
  using ignore_fields = decltype([](std::string_view, std::string_view) noexcept {});

 public:
  explicit response_parser(parse_options opts = {}) noexcept : _opts(opts) {}

  /// Parse response head bytes.
  /// @param in contiguous received bytes, starting where the previous call stopped consuming.
  /// @param consumed [out] bytes of `in` consumed by this call (== body offset on `ok`).
  /// @param meta [out] decoded fields; reset when each status line is parsed.
  /// @param ec [out] `errc::http_bad_message` or `errc::http_header_too_large` on `error`.
  /// @param on_unknown_field called as `(name, value)` for every field not decoded into `meta`
  ///        (including `Last-Modified` and `Content-Type`); the views are valid during the call
  ///        only. Obs-folded values arrive joined with single spaces.
  /// @return `ok` once the final (non-1xx) head is complete, `need_more`, or `error` (sticky).
  template <class OnField = ignore_fields>
  [[nodiscard]] parse_status parse_header(std::span<char const> in,
                                          std::size_t& consumed,
                                          response_meta& meta,
                                          std::error_code& ec,
                                          OnField&& on_unknown_field = {}) noexcept
  {
    using on_field_t = std::remove_reference_t<OnField>;
    if constexpr (std::is_same_v<std::remove_cv_t<on_field_t>, ignore_fields>) {
      return parse_header_impl(in, consumed, meta, ec, field_sink{nullptr, nullptr});
    } else {
      field_sink const sink{
        const_cast<void*>(static_cast<void const*>(std::addressof(on_unknown_field))),
        [](void* ctx, std::string_view name, std::string_view value) noexcept {
          (*static_cast<on_field_t*>(ctx))(name, value);
        }};
      return parse_header_impl(in, consumed, meta, ec, sink);
    }
  }

  /// @return body bytes still expected with `framing::length`; 0 when there is no body or before
  /// the head is complete; `unknown_length` for chunked and EOF-delimited bodies.
  [[nodiscard]] std::uint64_t body_remaining() const noexcept;
  /// Account `n` body bytes that bypassed the parser (leftover copy and direct receives).
  void consume_direct(std::uint64_t n) noexcept;  ///< bytes that bypassed the parser
  /// @return true when the head is complete and the body is absent or fully consumed
  /// (`framing::length`). Chunked and EOF-delimited bodies are finished by their consumer.
  [[nodiscard]] bool done() const noexcept { return _state == state::done; }
  /// Prepare for the next exchange on the connection (the parser is reusable, unlike Beast's).
  void reset(parse_options opts = {}) noexcept { *this = response_parser(opts); }

  /// @return true once the final head has been parsed.
  [[nodiscard]] bool header_done() const noexcept
  {
    return _state == state::body || _state == state::done;
  }
  /// @return number of interim (1xx other than 101) heads skipped so far (lets an
  /// `Expect: 100-continue` driver see the `100 Continue` while `parse_header` reports need_more).
  [[nodiscard]] std::uint16_t interim_responses() const noexcept { return _interim; }
  /// @return the rule that failed when `parse_header` returned `error`.
  [[nodiscard]] parse_error last_error() const noexcept { return _error; }

 private:
  struct field_sink {
    void* ctx;
    void (*fn)(void*, std::string_view, std::string_view) noexcept;
  };
  enum class state : std::uint8_t { start_line, fields, body, done, failed };

  parse_status parse_header_impl(std::span<char const> in,
                                 std::size_t& consumed,
                                 response_meta& meta,
                                 std::error_code& ec,
                                 field_sink sink) noexcept;
  bool apply_field(known_field id, std::string_view value, response_meta& meta) noexcept;
  bool finish_header(response_meta& meta) noexcept;

  parse_options _opts{};
  std::uint64_t _content_length{0};
  std::uint64_t _remaining{0};
  std::uint32_t _header_bytes{0};
  std::uint16_t _flags{0};
  std::uint16_t _interim{0};
  state _state{state::start_line};
  framing _framing{framing::none};
  parse_error _error{parse_error::none};
};

}  // namespace cucascade::io::http
