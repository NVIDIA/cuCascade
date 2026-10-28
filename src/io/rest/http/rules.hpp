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

/// @file rules.hpp
/// @brief RFC 9110/9112 grammar pieces shared by the response parser and the chunked decoder.
///
/// Ported from Boost.Beast (Boost Software License 1.0; Copyright (c) 2016-2019 Vinnie Falco):
/// `boost/beast/http/detail/basic_parser.ipp` (`parse_field`, `parse_token_to_eol`, `find_eol`,
/// `parse_dec`, `parse_hex`, `parse_chunk_extensions`, status-line rules) and
/// `boost/beast/http/detail/rfc7230.ipp` / `impl/rfc7230.ipp` (character tables, `token_list`,
/// `opt_token_list`). Changes for gpuio (gpuio docs/research/beast.md §3.1 #3): concrete functions
/// (no virtual callbacks), `std::error_code`-free internal results, 16-byte (SSE2) / 8-byte (SWAR)
/// control-byte and field-name scans in place of the per-byte loops, a single-scan fast path for
/// well-formed field lines that defers every other line to the exact port, and obs-fold handled
/// zero-copy (the raw folded span is joined into a stack buffer only when a folded value is used).

#pragma once

#include "io/rest/http/message.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace cucascade::io::http::detail {

enum class rc : std::uint8_t { ok, need_more, error };

/// Longest obs-folded value the parser joins (Beast `max_obs_fold`); longer ones are a header-limit
/// error, as in Beast.
inline constexpr std::size_t max_obs_fold = 4096;

/// Chunk-size lines (with extensions) longer than this are rejected (Beast `parse_chunk_header`).
inline constexpr std::size_t max_chunk_line = 8192;

// ------------------------------------------------------------------------------------------------
// Character classes (rfc7230.ipp tables)

consteval std::array<std::uint8_t, 256> make_tchar_table() noexcept
{
  std::array<std::uint8_t, 256> t{};
  for (unsigned c = '0'; c <= '9'; ++c)
    t[c] = 1;
  for (unsigned c = 'a'; c <= 'z'; ++c)
    t[c] = 1;
  for (unsigned c = 'A'; c <= 'Z'; ++c)
    t[c] = 1;
  for (char c : std::string_view{"!#$%&'*+-.^_`|~"})
    t[static_cast<unsigned char>(c)] = 1;
  return t;
}
inline constexpr auto tchar_table = make_tchar_table();

/// qdtext = HTAB / SP / "!" / %x23-5B / %x5D-7E / obs-text
consteval std::array<std::uint8_t, 256> make_qdtext_table() noexcept
{
  std::array<std::uint8_t, 256> t{};
  t['\t'] = 1;
  t[' ']  = 1;
  t['!']  = 1;
  for (unsigned c = 0x23; c <= 0x5b; ++c)
    t[c] = 1;
  for (unsigned c = 0x5d; c <= 0x7e; ++c)
    t[c] = 1;
  for (unsigned c = 0x80; c <= 0xff; ++c)
    t[c] = 1;
  return t;
}
inline constexpr auto qdtext_table = make_qdtext_table();

/// quoted-pair second octet: HTAB / SP / VCHAR / obs-text
consteval std::array<std::uint8_t, 256> make_qpchar_table() noexcept
{
  std::array<std::uint8_t, 256> t{};
  t['\t'] = 1;
  for (unsigned c = 0x20; c <= 0x7e; ++c)
    t[c] = 1;
  for (unsigned c = 0x80; c <= 0xff; ++c)
    t[c] = 1;
  return t;
}
inline constexpr auto qpchar_table = make_qpchar_table();

inline bool is_tchar(char c) noexcept { return tchar_table[static_cast<unsigned char>(c)] != 0; }
inline bool is_qdtext(char c) noexcept { return qdtext_table[static_cast<unsigned char>(c)] != 0; }
inline bool is_qpchar(char c) noexcept { return qpchar_table[static_cast<unsigned char>(c)] != 0; }
inline bool is_digit(char c) noexcept { return static_cast<unsigned char>(c - '0') < 10; }
inline bool is_ws(char c) noexcept { return c == ' ' || c == '\t'; }
inline bool is_ctl(char c) noexcept
{
  auto const u = static_cast<unsigned char>(c);
  return u < 0x20 || u == 0x7f;
}

inline int unhex(char c) noexcept
{
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

inline std::string_view make_view(char const* first, char const* last) noexcept
{
  return {first, static_cast<std::size_t>(last - first)};
}

inline char const* trim_back(char const* it, char const* first) noexcept
{
  while (it != first && is_ws(it[-1]))
    --it;
  return it;
}

inline std::string_view trim_ows(std::string_view s) noexcept
{
  while (!s.empty() && is_ws(s.front()))
    s.remove_prefix(1);
  while (!s.empty() && is_ws(s.back()))
    s.remove_suffix(1);
  return s;
}

/// ASCII case-insensitive equality against a lowercase literal.
inline bool iequals_lower(std::string_view s, std::string_view lower) noexcept
{
  if (s.size() != lower.size()) return false;
  for (std::size_t i = 0; i < s.size(); ++i) {
    auto c = static_cast<unsigned char>(s[i]);
    if (c >= 'A' && c <= 'Z') c = static_cast<unsigned char>(c | 0x20U);
    if (c != static_cast<unsigned char>(lower[i])) return false;
  }
  return true;
}

// ------------------------------------------------------------------------------------------------
// Scanning

/// First byte in [p, last) that is a control character (< 0x20, which includes HTAB/CR/LF, or
/// DEL), or `last`. Sixteen bytes per step with SSE2; otherwise eight per step on little-endian
/// targets: the classic "has byte less than n" / "has zero byte" bit tricks never miss a match and
/// their lowest flagged byte is always a true match (borrows only propagate upward from a true
/// match), so countr_zero locates it. Loads never pass `last`.
inline char const* find_ctl(char const* p, char const* last) noexcept
{
#if defined(__SSE2__)
  // 16 bytes per step (x86-64 baseline): byte <= 0x1f via unsigned min, or byte == 0x7f.
  __m128i const c1f = _mm_set1_epi8(0x1f);
  __m128i const del = _mm_set1_epi8(0x7f);
  while (last - p >= 16) {
    __m128i const x  = _mm_loadu_si128(reinterpret_cast<__m128i const*>(p));
    __m128i const lt = _mm_cmpeq_epi8(_mm_min_epu8(x, c1f), x);
    __m128i const d  = _mm_cmpeq_epi8(x, del);
    auto const m     = static_cast<unsigned>(_mm_movemask_epi8(_mm_or_si128(lt, d)));
    if (m != 0) return p + std::countr_zero(m);
    p += 16;
  }
#endif
  if constexpr (std::endian::native == std::endian::little) {
    constexpr std::uint64_t ones  = 0x0101010101010101ULL;
    constexpr std::uint64_t highs = 0x8080808080808080ULL;
    while (last - p >= 8) {
      std::uint64_t x = 0;
      std::memcpy(&x, p, sizeof x);
      std::uint64_t const below = (x - ones * 0x20U) & ~x & highs;  // byte < 0x20
      std::uint64_t const y     = x ^ (ones * 0x7fU);
      std::uint64_t const del   = (y - ones) & ~y & highs;  // byte == 0x7f
      std::uint64_t const hit   = below | del;
      if (hit != 0) return p + (std::countr_zero(hit) >> 3);
      p += 8;
    }
  }
  while (p < last && !is_ctl(*p))
    ++p;
  return p;
}

/// First byte in [p, last) outside [A-Za-z0-9-] (the characters of nearly every real field
/// name), or `last`. Sixteen bytes per step with SSE2; otherwise eight per step on little-endian
/// targets using exact per-byte range
/// tests on the low seven bits (no carries cross byte lanes, so there are no false positives);
/// bytes with the high bit set are never accepted. The caller finishes the name per byte (other
/// tchars such as '_' or '.') and checks for ':'.
inline char const* skip_common_name_chars(char const* p, char const* last) noexcept
{
#if defined(__SSE2__)
  // 16 bytes per step: (x | 0x20) in [a, z] is exactly [A-Za-z]; unsigned range tests via
  // min(x - lo, hi - lo) == x - lo.
  __m128i const bit5 = _mm_set1_epi8(0x20);
  __m128i const lo_a = _mm_set1_epi8('a');
  __m128i const n_az = _mm_set1_epi8('z' - 'a');
  __m128i const lo_0 = _mm_set1_epi8('0');
  __m128i const n_09 = _mm_set1_epi8('9' - '0');
  __m128i const dash = _mm_set1_epi8('-');
  while (last - p >= 16) {
    __m128i const x     = _mm_loadu_si128(reinterpret_cast<__m128i const*>(p));
    __m128i const ya    = _mm_sub_epi8(_mm_or_si128(x, bit5), lo_a);
    __m128i const alpha = _mm_cmpeq_epi8(_mm_min_epu8(ya, n_az), ya);
    __m128i const xd    = _mm_sub_epi8(x, lo_0);
    __m128i const digit = _mm_cmpeq_epi8(_mm_min_epu8(xd, n_09), xd);
    __m128i const ok    = _mm_or_si128(_mm_or_si128(alpha, digit), _mm_cmpeq_epi8(x, dash));
    auto const bad      = static_cast<unsigned>(~_mm_movemask_epi8(ok)) & 0xffffU;
    if (bad != 0) return p + std::countr_zero(bad);
    p += 16;
  }
#endif
  if constexpr (std::endian::native == std::endian::little) {
    constexpr std::uint64_t ones  = 0x0101010101010101ULL;
    constexpr std::uint64_t highs = 0x8080808080808080ULL;
    constexpr std::uint64_t low7  = 0x7f7f7f7f7f7f7f7fULL;
    while (last - p >= 8) {
      std::uint64_t x = 0;
      std::memcpy(&x, p, sizeof x);
      std::uint64_t const v = x & low7;
      // bytes with m < v < n (Bit Twiddling Hacks "hasbetween", lane-exact for 7-bit values)
      auto const between = [v](std::uint64_t m, std::uint64_t n) noexcept {
        return (ones * (127 + n) - v) & (v + ones * (127 - m));
      };
      std::uint64_t const dash    = x ^ (ones * 0x2dU);
      std::uint64_t const is_dash = ~(((dash & low7) + low7) | dash);  // exact zero-byte test
      std::uint64_t const ok =
        (between(0x60, 0x7b) | between(0x40, 0x5b) | between(0x2f, 0x3a) | is_dash) & ~x & highs;
      std::uint64_t const bad = ~ok & highs;
      if (bad != 0) return p + (std::countr_zero(bad) >> 3);
      p += 8;
    }
  }
  while (p < last) {
    char const c = *p;
    if ((c < 'a' || c > 'z') && (c < 'A' || c > 'Z') && (c < '0' || c > '9') && c != '-') { break; }
    ++p;
  }
  return p;
}

/// Beast `parse_token_to_eol`: scan a reason phrase or field value up to CRLF. HTAB and obs-text
/// are allowed; any other control byte is an error (`bad_char`); CR must be followed by LF.
/// @param p [in,out] start; on `ok` one past the LF.
/// @param token_last [out] on `ok`, the position of the CR.
inline rc scan_to_eol(char const*& p,
                      char const* last,
                      char const*& token_last,
                      parse_error bad_char,
                      parse_error& err) noexcept
{
  char const* q = p;
  for (;;) {
    q = find_ctl(q, last);
    if (q == last) return rc::need_more;
    if (*q == '\t') {
      ++q;
      continue;
    }
    if (*q == '\r') {
      if (q + 1 >= last) return rc::need_more;
      if (q[1] != '\n') {
        err = parse_error::bad_line_ending;
        return rc::error;
      }
      token_last = q;
      p          = q + 2;
      return rc::ok;
    }
    err = bad_char;
    return rc::error;
  }
}

/// Beast `find_eol`: one past the next CRLF in [p, last), or nullptr (need more, or
/// `bad_line_ending` in `err` when a CR is followed by something other than LF).
inline char const* find_eol(char const* p, char const* last, parse_error& err) noexcept
{
  auto const* cr =
    static_cast<char const*>(std::memchr(p, '\r', static_cast<std::size_t>(last - p)));
  if (cr == nullptr || cr + 1 >= last) return nullptr;
  if (cr[1] != '\n') {
    err = parse_error::bad_line_ending;
    return nullptr;
  }
  return cr + 2;
}

// ------------------------------------------------------------------------------------------------
// Status line (Beast `parse_version`, `parse_status`, `parse_reason`, response start line)

/// status-line = HTTP-version SP 3DIGIT SP reason-phrase CRLF; only HTTP/1.0 and HTTP/1.1.
inline rc parse_status_line(
  char const*& p, char const* last, unsigned& version, unsigned& status, parse_error& err) noexcept
{
  char const* it = p;
  if (last - it < 8) return rc::need_more;
  if (it[0] != 'H' || it[1] != 'T' || it[2] != 'T' || it[3] != 'P' || it[4] != '/' ||
      !is_digit(it[5]) || it[6] != '.' || !is_digit(it[7])) {
    err = parse_error::bad_version;
    return rc::error;
  }
  version = static_cast<unsigned>(10 * (it[5] - '0') + (it[7] - '0'));
  if (version != 10 && version != 11) {
    err = parse_error::bad_version;
    return rc::error;
  }
  it += 8;
  if (it == last) return rc::need_more;
  if (*it++ != ' ') {
    err = parse_error::bad_version;
    return rc::error;
  }
  if (last - it < 4) return rc::need_more;
  if (!is_digit(it[0]) || !is_digit(it[1]) || !is_digit(it[2]) || it[3] != ' ') {
    err = parse_error::bad_status;
    return rc::error;
  }
  status = static_cast<unsigned>(100 * (it[0] - '0') + 10 * (it[1] - '0') + (it[2] - '0'));
  it += 4;
  char const* reason_last = nullptr;
  rc const r              = scan_to_eol(it, last, reason_last, parse_error::bad_reason, err);
  if (r != rc::ok) return r;
  p = it;
  return rc::ok;
}

// ------------------------------------------------------------------------------------------------
// Fields (Beast `parse_field`)

/// One parsed `name: value` line. When `folded`, [`first`, `last`) spans several physical lines
/// (obs-fold) and must be joined with `join_folded`; otherwise it is the trimmed value.
struct raw_field {
  std::string_view name;
  char const* first{nullptr};
  char const* last{nullptr};
  bool folded{false};
};

/// header-field = field-name ":" OWS field-value OWS, with obs-fold (CRLF 1*(SP/HTAB)).
/// Exact port of Beast `parse_field` (every error case, folding, need_more points).
/// Needs one byte past the line's CRLF to rule out a continuation line (as Beast does).
/// @param p [in,out] line start; on `ok` the start of the next line.
inline rc parse_field_exact(char const*& p,
                            char const* last,
                            raw_field& out,
                            parse_error& err) noexcept
{
  char const* const first = p;
  char const* it          = skip_common_name_chars(p, last);
  for (;;) {
    if (it >= last) return rc::need_more;
    if (*it == ':') break;
    if (!is_tchar(*it)) {
      err = parse_error::bad_field;
      return rc::error;
    }
    ++it;
  }
  if (it == first) {
    err = parse_error::bad_field;
    return rc::error;
  }
  out.name   = make_view(first, it);
  out.first  = nullptr;
  out.last   = nullptr;
  out.folded = false;
  ++it;  // ':'
  for (;;) {
    while (true) {
      if (it >= last) return rc::need_more;
      if (!is_ws(*it)) break;
      ++it;
    }
    char const* const seg_first = it;
    char const* seg_last        = nullptr;
    rc const r                  = scan_to_eol(it, last, seg_last, parse_error::bad_value, err);
    if (r != rc::ok) return r;
    if (it >= last) return rc::need_more;  // look one byte past CRLF for obs-fold
    seg_last = trim_back(seg_last, seg_first);
    if (seg_last != seg_first) {
      if (out.first == nullptr) {
        out.first = seg_first;
      } else {
        out.folded = true;
      }
      out.last = seg_last;
    }
    if (!is_ws(*it)) break;
    ++it;  // continuation line
  }
  if (out.first == nullptr) out.first = out.last = it;  // empty value
  p = it;
  return rc::ok;
}

/// `parse_field` with a fast path for the common line shape: one control-byte scan finds the
/// CRLF (the only result the next line depends on), then the name and value are checked off that
/// critical path. The fast path accepts only a single-line field whose name is all tchars, whose
/// value has no control characters, and whose next line is not a continuation; anything else
/// (HTAB, obs-fold, malformed lines, too few bytes) is decided by `parse_field_exact`, so the
/// results are identical by construction.
inline rc parse_field(char const*& p, char const* last, raw_field& out, parse_error& err) noexcept
{
  char const* const eol = find_ctl(p, last);
  if (last - eol > 2 && eol[0] == '\r' && eol[1] == '\n' && !is_ws(eol[2])) {
    char const* n = skip_common_name_chars(p, eol);
    while (n != eol && *n != ':' && is_tchar(*n))
      ++n;
    if (n != p && n != eol && *n == ':') {
      char const* v = n + 1;
      while (v != eol && is_ws(*v))
        ++v;
      out.name   = make_view(p, n);
      out.first  = v;
      out.last   = trim_back(eol, v);
      out.folded = false;
      p          = eol + 2;
      return rc::ok;
    }
  }
  return parse_field_exact(p, last, out, err);
}

/// Join an obs-folded value: non-empty physical segments, each trimmed, separated by one SP
/// (Beast's `char_buffer<max_obs_fold>` result). @return false when it exceeds `max_obs_fold`.
inline bool join_folded(char const* first,
                        char const* last,
                        char (&buf)[max_obs_fold],
                        std::string_view& out) noexcept
{
  std::size_t n   = 0;
  char const* seg = first;
  while (seg < last) {
    auto const* cr =
      static_cast<char const*>(std::memchr(seg, '\r', static_cast<std::size_t>(last - seg)));
    char const* const seg_end = cr != nullptr ? cr : last;
    char const* a             = seg;
    while (a < seg_end && is_ws(*a))
      ++a;
    char const* const b = trim_back(seg_end, a);
    if (a != b) {
      auto const len = static_cast<std::size_t>(b - a);
      if (n != 0) {
        if (n == max_obs_fold) return false;
        buf[n++] = ' ';
      }
      if (len > max_obs_fold - n) return false;
      std::memcpy(buf + n, a, len);
      n += len;
    }
    seg = cr != nullptr ? cr + 2 : last;
  }
  out = {buf, n};
  return true;
}

// ------------------------------------------------------------------------------------------------
// Lists and numbers (rfc7230 `opt_token_list`, `token_list`; Beast `parse_dec`, `parse_hex`)

/// `#token` list iteration with Beast's `opt_token_list_policy` semantics (empty elements allowed,
/// tokens must be comma separated).
/// @return 1 with `tok` set, 0 at the end, -1 on a malformed list.
inline int next_opt_token(char const*& it,
                          char const* end,
                          bool& need_comma,
                          std::string_view& tok) noexcept
{
  for (;;) {
    while (it != end && is_ws(*it))
      ++it;
    if (it == end) return 0;
    if (is_tchar(*it)) {
      if (need_comma) return -1;
      char const* const t0 = it;
      while (it != end && is_tchar(*it))
        ++it;
      tok        = make_view(t0, it);
      need_comma = true;
      return 1;
    }
    if (*it != ',') return -1;
    need_comma = false;
    ++it;
  }
}

/// Beast `parse_dec`: 1*DIGIT into a uint64 with overflow detection.
inline bool parse_dec(std::string_view s, std::uint64_t& v) noexcept
{
  if (s.empty()) return false;
  std::uint64_t tmp = 0;
  for (char c : s) {
    if (!is_digit(c) || tmp > std::numeric_limits<std::uint64_t>::max() / 10) return false;
    tmp *= 10;
    auto const d = static_cast<std::uint64_t>(c - '0');
    if (std::numeric_limits<std::uint64_t>::max() - tmp < d) return false;
    tmp += d;
  }
  v = tmp;
  return true;
}

/// Fast path for the common case: 1..19 decimal digits (cannot overflow) at [p, last).
/// @return pointer past the digits, or nullptr when there are none or more than 19 (the caller
/// then falls back to the checked `parse_dec`).
inline char const* parse_dec_short(char const* p, char const* last, std::uint64_t& v) noexcept
{
  char const* it    = p;
  std::uint64_t acc = 0;
  while (it != last && is_digit(*it)) {
    acc = acc * 10 + static_cast<std::uint64_t>(*it - '0');
    ++it;
  }
  auto const n = it - p;
  if (n == 0 || n > 19) return nullptr;
  v = acc;
  return it;
}

/// Beast `parse_hex`: 1*HEXDIG with overflow detection; `it` must be followed by a non-hex byte
/// inside the buffer (the chunk line's CR guarantees one).
inline bool parse_hex(char const*& it, std::uint64_t& v) noexcept
{
  int d = unhex(*it);
  if (d < 0) return false;
  std::uint64_t tmp = 0;
  do {
    if (tmp > std::numeric_limits<std::uint64_t>::max() / 16) return false;
    tmp *= 16;
    tmp += static_cast<std::uint64_t>(d);
    d = unhex(*++it);
  } while (d >= 0);
  v = tmp;
  return true;
}

/// Beast `parse_chunk_extensions`:
///   chunk-ext      = *( BWS ";" BWS chunk-ext-name [ BWS "=" BWS chunk-ext-val ] )
///   chunk-ext-name = token;  chunk-ext-val = token / quoted-string
/// Stops at the first byte that cannot continue an extension (normally the CR). `last` must lie
/// beyond that byte (the caller passes the end of the located chunk line).
inline rc parse_chunk_extensions(char const*& it, char const* last, parse_error& err) noexcept
{
  auto bad = [&err] {
    err = parse_error::bad_chunk_extension;
    return rc::error;
  };
  for (;;) {
    if (it == last) return rc::need_more;
    if (*it != ' ' && *it != '\t' && *it != ';') return rc::ok;
    while (is_ws(*it)) {  // BWS
      if (++it == last) return rc::need_more;
    }
    if (*it != ';') return bad();
    bool more_params = true;
    while (more_params) {
      more_params = false;
      ++it;       // ';'
      for (;;) {  // BWS
        if (it == last) return rc::need_more;
        if (!is_ws(*it)) break;
        ++it;
      }
      if (!is_tchar(*it)) return bad();  // chunk-ext-name
      do {
        if (++it == last) return rc::need_more;
      } while (is_tchar(*it));
      bool bws = false;
      if (is_ws(*it)) {
        do {
          if (++it == last) return rc::need_more;
        } while (is_ws(*it));
        bws = true;
      }
      if (*it == ';') {
        more_params = true;
        continue;
      }
      if (*it != '=') {
        if (bws) return bad();
        return rc::ok;  // name without value ends the extensions (Beast returns here)
      }
      ++it;       // '='
      for (;;) {  // BWS
        if (it == last) return rc::need_more;
        if (!is_ws(*it)) break;
        ++it;
      }
      if (*it != '"') {  // token value
        if (!is_tchar(*it)) return bad();
        do {
          if (++it == last) return rc::need_more;
        } while (is_tchar(*it));
      } else {  // quoted-string value
        for (;;) {
          if (++it == last) return rc::need_more;
          if (*it == '"') break;
          if (*it == '\\') {
            if (++it == last) return rc::need_more;
            if (!is_qpchar(*it)) return bad();
          } else if (!is_qdtext(*it)) {
            return bad();
          }
        }
        ++it;
      }
    }
  }
}

}  // namespace cucascade::io::http::detail
