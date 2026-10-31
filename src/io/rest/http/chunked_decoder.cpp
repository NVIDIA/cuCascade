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

// Chunked body decoder. Ported from Boost.Beast `boost/beast/http/impl/basic_parser.ipp`
// (`parse_chunk_header`, `parse_chunk_body`, trailer-field handling; Boost Software License 1.0,
// Copyright (c) 2016-2019 Vinnie Falco). See gpuio docs/research/beast.md §2.2.5.

#include "io/rest/http/chunked_decoder.hpp"

#include "io/rest/errc.hpp"
#include "io/rest/http/rules.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace cucascade::io::http {

using rest::errc;
namespace {

using detail::rc;

/// Trailer section cap (the response parser's default header limit).
constexpr std::size_t max_trailer_bytes = std::size_t{16} * 1024;

std::error_code to_error_code(parse_error e) noexcept
{
  switch (e) {
    case parse_error::header_limit: return make_error_code(errc::http_header_too_large);
    case parse_error::body_limit: return make_error_code(errc::http_length_mismatch);
    default: return make_error_code(errc::http_bad_message);
  }
}

}  // namespace

parse_status chunked_decoder::feed_impl(std::span<char const> in,
                                        std::size_t& consumed,
                                        data_sink sink,
                                        std::error_code& ec) noexcept
{
  consumed = 0;
  ec.clear();
  if (_state == state::done) return parse_status::ok;
  if (_state == state::failed) {
    ec = to_error_code(_error);
    return parse_status::error;
  }

  char const* const begin = in.data();
  char const* const end   = begin + in.size();
  char const* p           = begin;
  parse_status result     = parse_status::need_more;

  auto fail = [&](parse_error e) noexcept {
    _error = e;
    result = parse_status::error;
  };

  for (bool running = true; running;) {
    switch (_state) {
      case state::size_line: {
        // chunk = chunk-size [ chunk-ext ] CRLF chunk-data CRLF; the previous chunk's trailing
        // CRLF is consumed together with the next size line (Beast `flagExpectCRLF`).
        char const* q = p;
        if (_expect_crlf) {
          if (end - q < 2) {
            running = false;
            break;
          }
          if (q[0] != '\r' || q[1] != '\n') {
            fail(parse_error::bad_chunk);
            running = false;
            break;
          }
          q += 2;
        }
        parse_error eol_err         = parse_error::none;
        char const* const scan_from = q + std::min(_skip, static_cast<std::size_t>(end - q));
        char const* const eol       = detail::find_eol(scan_from, end, eol_err);
        if (eol == nullptr) {
          if (eol_err != parse_error::none) {
            fail(eol_err);
          } else if (static_cast<std::size_t>(end - q) >= detail::max_chunk_line) {
            fail(parse_error::bad_chunk_extension);
          } else if (end > q) {
            _skip = static_cast<std::size_t>(end - q) - 1;  // rescan only the last byte (a CR)
          }
          running = false;
          break;
        }
        _skip              = 0;
        std::uint64_t size = 0;
        if (!detail::parse_hex(q, size)) {
          fail(parse_error::bad_chunk);
          running = false;
          break;
        }
        if (_limit != unknown_length && size > _limit - _total) {
          fail(parse_error::body_limit);
          running = false;
          break;
        }
        parse_error ext_err = parse_error::none;
        if (detail::parse_chunk_extensions(q, eol, ext_err) == rc::error) {
          fail(ext_err);
          running = false;
          break;
        }
        if (q != eol - 2) {
          fail(parse_error::bad_chunk_extension);
          running = false;
          break;
        }
        p            = eol;
        _expect_crlf = true;
        _remaining   = size;
        _state       = size != 0 ? state::data : state::trailers;
        break;
      }

      case state::data: {
        auto const n = static_cast<std::size_t>(
          std::min<std::uint64_t>(_remaining, static_cast<std::uint64_t>(end - p)));
        if (n != 0) {
          sink.fn(sink.ctx, std::span<char const>{p, n});
          p += n;
          _remaining -= n;
          _total += n;
        }
        if (_remaining != 0) {
          running = false;  // need more
          break;
        }
        _state = state::size_line;
        break;
      }

      case state::trailers: {
        // trailer-part = *( field-line CRLF ) CRLF. Parsed with the header grammar and ignored:
        // framing/connection fields in a trailer must not affect the message (RFC 9112 §7.1.2).
        std::size_t const limit_left = max_trailer_bytes - _trailer_bytes;
        auto const avail             = static_cast<std::size_t>(end - p);
        char const* const last       = p + std::min(avail, limit_left);
        if (last - p < 2) {
          if (avail >= limit_left) fail(parse_error::header_limit);
          running = false;
          break;
        }
        if (p[0] == '\r') {
          if (p[1] != '\n') {
            fail(parse_error::bad_line_ending);
          } else {
            p += 2;
            _state = state::done;
            result = parse_status::ok;
          }
          running = false;
          break;
        }
        detail::raw_field f;
        char const* q = p;
        rc const r    = detail::parse_field(q, last, f, _error);
        if (r == rc::error) {
          result  = parse_status::error;
          running = false;
          break;
        }
        if (r == rc::need_more) {
          if (avail >= limit_left) fail(parse_error::header_limit);
          running = false;
          break;
        }
        _trailer_bytes += static_cast<std::uint32_t>(q - p);
        p = q;
        break;
      }

      case state::done:
        result  = parse_status::ok;
        running = false;
        break;
      case state::failed: running = false; break;
    }
  }

  consumed = static_cast<std::size_t>(p - begin);
  if (result == parse_status::error) {
    _state = state::failed;
    ec     = to_error_code(_error);
  }
  return result;
}

}  // namespace cucascade::io::http
