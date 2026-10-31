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

#include "io/rest/http/request_head.hpp"

#include "io/rest/http/rules.hpp"

#include <algorithm>
#include <cstring>

namespace cucascade::io::http {
namespace {

std::string_view method_name(method m) noexcept
{
  switch (m) {
    case method::get: return "GET";
    case method::head: return "HEAD";
    case method::put: return "PUT";
    case method::post: return "POST";
    case method::delete_: return "DELETE";
  }
  return "GET";
}

/// field-name = token
bool valid_name(std::string_view s) noexcept
{
  if (s.empty()) return false;
  return std::ranges::all_of(s, [](char c) noexcept { return detail::is_tchar(c); });
}

/// field-value: no control characters except HTAB (blocks CR/LF header injection).
bool valid_value(std::string_view s) noexcept
{
  return std::ranges::all_of(s, [](char c) noexcept { return !detail::is_ctl(c) || c == '\t'; });
}

/// request-target / authority: visible characters only (no SP, no controls).
bool valid_visible(std::string_view s) noexcept
{
  if (s.empty()) return false;
  return std::ranges::all_of(s, [](char c) noexcept {
    auto const u = static_cast<unsigned char>(c);
    return u > 0x20 && u != 0x7f;
  });
}

/// Decimal digits of `v` into `out` (room for 20); @return the digit count.
std::size_t put_dec(char* out, std::uint64_t v) noexcept
{
  char rev[20];
  std::size_t n = 0;
  do {
    rev[n++] = static_cast<char>('0' + v % 10);
    v /= 10;
  } while (v != 0);
  for (std::size_t i = 0; i < n; ++i)
    out[i] = rev[n - 1 - i];
  return n;
}

}  // namespace

request_head::request_head(std::span<char> storage) noexcept : _buf(storage) {}

void request_head::reset(std::span<char> storage) noexcept { *this = request_head(storage); }

bool request_head::append(std::string_view s) noexcept
{
  if (_overflow) return false;
  if (s.size() > _buf.size() - _len) {
    _overflow = true;
    return false;
  }
  if (!s.empty()) std::memcpy(_buf.data() + _len, s.data(), s.size());
  _len += s.size();
  return true;
}

request_head& request_head::start(method m, std::string_view target) noexcept
{
  if (_overflow) return *this;
  if (_started || !valid_visible(target)) return fail();
  _started = true;
  _method  = m;
  append(method_name(m));
  append(" ");
  _target_off = _len;
  _target_len = target.size();
  append(target);
  append(" HTTP/1.1\r\n");
  if (_overflow) _target_len = 0;
  return *this;
}

request_head& request_head::field(std::string_view name, std::string_view value) noexcept
{
  if (_overflow) return *this;
  if (!_started || _finished || _nfields == max_fields || !valid_name(name) ||
      !valid_value(value)) {
    return fail();
  }
  std::size_t const need = name.size() + 2 + value.size() + 2;
  if (need > _buf.size() - _len) return fail();
  char* const base = _buf.data() + _len;
  std::memcpy(base, name.data(), name.size());
  std::string_view(": ").copy(base + name.size(), 2);
  if (!value.empty()) std::memcpy(base + name.size() + 2, value.data(), value.size());
  std::string_view("\r\n").copy(base + name.size() + 2 + value.size(), 2);
  _fields[_nfields++] = header_ref{std::string_view{base, name.size()},
                                   std::string_view{base + name.size() + 2, value.size()}};
  _len += need;
  return *this;
}

request_head& request_head::host(std::string_view authority) noexcept
{
  if (_overflow) return *this;
  if (!valid_visible(authority)) return fail();
  return field("Host", authority);
}

request_head& request_head::range(std::uint64_t first, std::uint64_t last) noexcept
{
  if (_overflow) return *this;
  if (first > last) return fail();
  char tmp[6 + 20 + 1 + 20];  // "bytes=" + 2 x uint64 + '-'
  std::string_view("bytes=").copy(tmp, 6);
  std::size_t n = 6;
  n += put_dec(tmp + n, first);
  tmp[n++] = '-';
  n += put_dec(tmp + n, last);
  return field("Range", std::string_view{tmp, n});
}

request_head& request_head::content_length(std::uint64_t n) noexcept
{
  if (_overflow) return *this;
  char tmp[20];
  return field("Content-Length", std::string_view{tmp, put_dec(tmp, n)});
}

request_head& request_head::expect_100_continue() noexcept
{
  return field("Expect", "100-continue");
}

std::span<char const> request_head::finish() noexcept
{
  if (_overflow || !_started) return {};
  if (!_finished) {
    if (!append("\r\n")) return {};
    _finished = true;
  }
  return {_buf.data(), _len};
}

std::span<header_ref const> request_head::fields() const noexcept
{
  return {_fields.data(), _nfields};
}

}  // namespace cucascade::io::http
