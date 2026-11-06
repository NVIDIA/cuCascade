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

// Response head parser. Framing and connection rules are ported from Boost.Beast
// `boost/beast/http/impl/basic_parser.ipp` (`put`, `parse_start_line`, `parse_fields`,
// `finish_header` for responses, `do_field`, `keep_alive`; Boost Software License 1.0,
// Copyright (c) 2016-2019 Vinnie Falco). See gpuio docs/research/beast.md §2.2.

#include "io/rest/http/response_parser.hpp"

#include "io/rest/errc.hpp"
#include "io/rest/http/content_range.hpp"
#include "io/rest/http/fields.hpp"
#include "io/rest/http/rules.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace cucascade::io::http {

using rest::errc;
namespace {

using detail::rc;

// Parser flags (Beast `flag*`).
constexpr std::uint16_t f_http11            = 1U << 0;
constexpr std::uint16_t f_close             = 1U << 1;
constexpr std::uint16_t f_keep_alive        = 1U << 2;
constexpr std::uint16_t f_content_length    = 1U << 3;
constexpr std::uint16_t f_chunked           = 1U << 4;
constexpr std::uint16_t f_transfer_encoding = 1U << 5;
constexpr std::uint16_t f_amz_retry_after   = 1U << 6;

// Retry-After / x-amz-retry-after are honoured up to 30 s (gpuio ARCHITECTURE §7.3: "Retry-After
// (s, cap 30)"); the same cap bounds x-amz-retry-after so a bogus value cannot stall a slot.
constexpr std::uint64_t max_retry_after_ms = 30'000;

void reset_meta(response_meta& m) noexcept
{
  m.status     = 0;
  m.version    = 11;
  m.keep_alive = true;
  m.body       = framing::none;
  m.content_length.reset();
  m.range.reset();
  m.etag.clear();
  m.request_id.clear();
  m.amz_id_2.clear();
  m.date.clear();
  m.amz_error_code.clear();
  m.retry_after_ms      = 0;
  m.retry_after_present = false;
  m.truncated_fields    = false;
}

std::error_code to_error_code(parse_error e) noexcept
{
  switch (e) {
    case parse_error::header_limit: return make_error_code(errc::http_header_too_large);
    case parse_error::body_limit: return make_error_code(errc::http_length_mismatch);
    default: return make_error_code(errc::http_bad_message);
  }
}

template <std::size_t N>
void capture(inline_string<N>& dst, std::string_view v, response_meta& meta) noexcept
{
  if (!dst.assign(v)) meta.truncated_fields = true;
}

}  // namespace

parse_status response_parser::parse_header_impl(std::span<char const> in,
                                                std::size_t& consumed,
                                                response_meta& meta,
                                                std::error_code& ec,
                                                field_sink sink) noexcept
{
  consumed = 0;
  ec.clear();
  switch (_state) {
    case state::failed: ec = to_error_code(_error); return parse_status::error;
    case state::body:
    case state::done: return parse_status::ok;
    default: break;
  }

  char const* const begin = in.data();
  char const* const end   = begin + in.size();
  char const* p           = begin;
  char fold_buf[detail::max_obs_fold];  // obs-fold join space (cold; Beast keeps 4 KiB as well)
  parse_status result = parse_status::need_more;

  // The header limit window. Consuming n bytes lowers the remaining limit by n, so its end is
  // invariant within a call: compute it once (keeps the per-field critical path free of the
  // byte accounting). Beast `parse_fields`: incomplete although the whole remaining limit is
  // available => header_limit.
  std::size_t const limit_left  = _opts.header_limit - _header_bytes;
  bool const limited            = in.size() >= limit_left;
  char const* const last        = limited ? begin + limit_left : end;
  auto const need_more_or_limit = [&]() noexcept {
    if (limited) {
      _error = parse_error::header_limit;
      return parse_status::error;
    }
    return parse_status::need_more;
  };

  state st = _state;
  for (;;) {
    if (st == state::start_line) {
      char const* q    = p;
      unsigned version = 0;
      unsigned status  = 0;
      rc const r       = detail::parse_status_line(q, last, version, status, _error);
      if (r != rc::ok) {
        result = r == rc::need_more ? need_more_or_limit() : parse_status::error;
        break;
      }
      reset_meta(meta);
      meta.status     = static_cast<std::uint16_t>(status);
      meta.version    = static_cast<std::uint8_t>(version);
      _flags          = version == 11 ? f_http11 : 0;
      _content_length = 0;
      p               = q;
      st              = state::fields;
      continue;
    }

    // state::fields
    if (last - p < 2) {
      result = need_more_or_limit();
      break;
    }
    if (p[0] == '\r') {
      if (p[1] != '\n') {
        _error = parse_error::bad_line_ending;
        result = parse_status::error;
        break;
      }
      p += 2;
      _state = st;
      if (finish_header(meta)) {  // interim 1xx skipped; parse the next status line
        st = state::start_line;
        continue;
      }
      st     = _state;
      result = parse_status::ok;
      break;
    }
    detail::raw_field f;
    char const* q = p;
    rc const r    = detail::parse_field(q, last, f, _error);
    if (r != rc::ok) {
      result = r == rc::need_more ? need_more_or_limit() : parse_status::error;
      break;
    }
    std::string_view value = detail::make_view(f.first, f.last);
    if (f.folded && !detail::join_folded(f.first, f.last, fold_buf, value)) {
      _error = parse_error::header_limit;
      result = parse_status::error;
      break;
    }
    known_field const id = lookup_field(f.name);
    if (id == known_field::unknown || id == known_field::content_type ||
        id == known_field::last_modified) {
      if (sink.fn != nullptr) sink.fn(sink.ctx, f.name, value);
    } else if (!apply_field(id, value, meta)) {
      result = parse_status::error;
      break;
    }
    p = q;
  }

  _header_bytes += static_cast<std::uint32_t>(p - begin);
  _state   = st;
  consumed = static_cast<std::size_t>(p - begin);
  if (result == parse_status::error) {
    _state = state::failed;
    ec     = to_error_code(_error);
  }
  return result;
}

// Beast `do_field` (Connection, Content-Length, Transfer-Encoding) plus the S3 captures, for the
// fields `response_meta` decodes.
bool response_parser::apply_field(known_field id,
                                  std::string_view value,
                                  response_meta& meta) noexcept
{
  char const* const vend = value.data() + value.size();
  switch (id) {
    case known_field::connection:
    case known_field::proxy_connection: {
      char const* it  = value.data();
      bool need_comma = false;
      std::string_view tok;
      for (;;) {
        int const n = detail::next_opt_token(it, vend, need_comma, tok);
        if (n == 0) break;
        if (n < 0) {
          _error = parse_error::bad_value;
          return false;
        }
        if (detail::iequals_lower(tok, "close")) {
          _flags |= f_close;
        } else if (detail::iequals_lower(tok, "keep-alive")) {
          _flags |= f_keep_alive;
        }
      }
      return true;
    }

    case known_field::content_length: {
      // A Transfer-Encoding already seen: a message must not carry both (RFC 9112 §6.3), in
      // either order — the Content-Length / Transfer-Encoding request-smuggling vector.
      if ((_flags & (f_chunked | f_transfer_encoding)) != 0) {
        _error = parse_error::bad_content_length;
        return false;
      }
      {
        // Fast path: a plain number (what every server sends).
        std::uint64_t v = 0;
        if (detail::parse_dec_short(value.data(), value.data() + value.size(), v) == vend) {
          if ((_flags & f_content_length) != 0 && v != _content_length) {
            _error = parse_error::multiple_content_length;
            return false;
          }
          _content_length = v;
          _flags |= f_content_length;
          return true;
        }
      }
      // A comma list of equal values is allowed (RFC 9110 §8.6). Beast order: validate the whole
      // list, then require every element to be a number equal to any value seen before.
      std::size_t tokens = 0;
      {
        char const* it  = value.data();
        bool need_comma = false;
        std::string_view tok;
        for (int n = 0; (n = detail::next_opt_token(it, vend, need_comma, tok)) != 0; ++tokens) {
          if (n < 0) {
            _error = parse_error::bad_content_length;
            return false;
          }
        }
      }
      if (tokens == 0) {
        _error = parse_error::bad_content_length;
        return false;
      }
      bool have              = (_flags & f_content_length) != 0;
      std::uint64_t existing = _content_length;
      char const* it         = value.data();
      bool need_comma        = false;
      std::string_view tok;
      while (detail::next_opt_token(it, vend, need_comma, tok) > 0) {
        std::uint64_t v = 0;
        if (!detail::parse_dec(tok, v)) {
          _error = parse_error::bad_content_length;
          return false;
        }
        if (have && v != existing) {
          _error = parse_error::multiple_content_length;
          return false;
        }
        have     = true;
        existing = v;
      }
      // Every comma must separate two numbers ("0," and ",0" are malformed).
      if (tokens != 1 + static_cast<std::size_t>(std::count(value.begin(), value.end(), ','))) {
        _error = parse_error::bad_content_length;
        return false;
      }
      _content_length = existing;
      _flags |= f_content_length;
      return true;
    }

    case known_field::transfer_encoding: {
      if ((_flags & f_chunked) != 0 || (_flags & f_content_length) != 0) {
        // `chunked` must be the final coding, applied once; Transfer-Encoding with
        // Content-Length is smuggling.
        _error = parse_error::bad_transfer_encoding;
        return false;
      }
      _flags |= f_transfer_encoding;
      // Beast `token_list`: iteration stops at the first malformed element; the body is chunked
      // only if the first `chunked` token is also the last one.
      char const* it  = value.data();
      bool need_comma = false;
      std::string_view tok;
      bool seen_chunked = false;
      bool chunked_last = false;
      for (;;) {
        int const n = detail::next_opt_token(it, vend, need_comma, tok);
        if (n <= 0) break;
        if (seen_chunked) {
          chunked_last = false;
          break;
        }
        if (detail::iequals_lower(tok, "chunked")) {
          seen_chunked = true;
          chunked_last = true;
        }
      }
      if (chunked_last) _flags |= f_chunked;
      return true;
    }

    case known_field::content_range: {
      content_range cr;
      if (!parse_content_range(value, cr)) meta.range = cr;  // malformed: left unset (classify)
      return true;
    }
    case known_field::etag: capture(meta.etag, value, meta); return true;
    case known_field::x_amz_request_id: capture(meta.request_id, value, meta); return true;
    case known_field::x_amz_id_2: capture(meta.amz_id_2, value, meta); return true;
    case known_field::date: capture(meta.date, value, meta); return true;

    case known_field::x_amz_retry_after: {
      std::uint64_t ms = 0;
      if (detail::parse_dec(value, ms)) {
        meta.retry_after_ms      = static_cast<std::uint32_t>(std::min(ms, max_retry_after_ms));
        meta.retry_after_present = true;
        _flags |= f_amz_retry_after;
      }
      return true;
    }
    case known_field::retry_after: {
      // delta-seconds only; an HTTP-date is ignored. x-amz-retry-after (ms) takes precedence.
      std::uint64_t s = 0;
      if ((_flags & f_amz_retry_after) == 0 && detail::parse_dec(value, s)) {
        std::uint64_t const ms   = s > max_retry_after_ms / 1000 ? max_retry_after_ms : s * 1000;
        meta.retry_after_ms      = static_cast<std::uint32_t>(ms);
        meta.retry_after_present = true;
      }
      return true;
    }

    case known_field::content_type:
    case known_field::last_modified:
    case known_field::unknown: return true;  // reported through on_unknown_field by the caller
  }
  return true;
}

// Beast `finish_header` (response) plus `keep_alive()`. @return true for a skipped interim head.
bool response_parser::finish_header(response_meta& meta) noexcept
{
  unsigned const st = meta.status;
  if (st / 100 == 1 && st != 101) {
    // 1xx other than 101 is an interim response: skip it and parse the final head that follows
    // in the same buffer (Beast completes the parser here and needs a new one).
    if (_interim != UINT16_MAX) ++_interim;
    _state          = state::start_line;
    _flags          = 0;
    _content_length = 0;
    return true;
  }
  bool const http11 = (_flags & f_http11) != 0;
  if (_opts.head_request || st / 100 == 1 || st == 204 || st == 304) {
    _framing   = framing::none;  // Content-Length may be present but there is no body
    _remaining = 0;
  } else if ((_flags & f_content_length) != 0) {
    _framing   = framing::length;
    _remaining = _content_length;
  } else if ((_flags & f_chunked) != 0) {
    _framing   = framing::chunked;
    _remaining = unknown_length;
  } else {
    _framing   = framing::until_eof;  // includes Transfer-Encoding without final `chunked`
    _remaining = unknown_length;
  }
  bool keep_alive = http11 ? (_flags & f_close) == 0 : (_flags & f_keep_alive) != 0;
  if (_framing == framing::until_eof || st == 101) keep_alive = false;

  meta.keep_alive = keep_alive;
  meta.body       = _framing;
  if ((_flags & f_content_length) != 0) meta.content_length = _content_length;
  _state = (_framing == framing::none || (_framing == framing::length && _remaining == 0))
             ? state::done
             : state::body;
  return false;
}

std::uint64_t response_parser::body_remaining() const noexcept
{
  if (_state != state::body) return 0;
  return _framing == framing::length ? _remaining : unknown_length;
}

void response_parser::consume_direct(std::uint64_t n) noexcept
{
  if (_state != state::body || _framing != framing::length) return;
  _remaining -= std::min(n, _remaining);
  if (_remaining == 0) _state = state::done;
}

}  // namespace cucascade::io::http
