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

#include "io/rest/http/iovec_sink.hpp"

#include "io/rest/errc.hpp"

#include <algorithm>
#include <cstring>

namespace cucascade::io::http {

using rest::errc;

iovec_sink::iovec_sink(std::span<iovec const> dst) noexcept : _dst(dst)
{
  for (auto const& v : dst)
    _capacity += v.iov_len;
  while (_idx < _dst.size() && _dst[_idx].iov_len == 0)
    ++_idx;
}

std::error_code iovec_sink::init(std::optional<std::uint64_t> content_length) const noexcept
{
  // Beast `span_body::reader::init`: reject a body that cannot fit before any byte arrives.
  if (content_length && *content_length > remaining()) {
    return make_error_code(errc::http_length_mismatch);
  }
  return {};
}

void iovec_sink::advance(std::uint64_t n) noexcept
{
  _written += n;
  while (n != 0) {
    std::uint64_t const room = _dst[_idx].iov_len - _off;
    if (n < room) {
      _off += n;
      return;
    }
    n -= room;
    _off = 0;
    ++_idx;
  }
  while (_idx < _dst.size() && _dst[_idx].iov_len - _off == 0) {  // skip full and empty iovecs
    _off = 0;
    ++_idx;
  }
}

void iovec_sink::copy_in(std::span<char const> leftover) noexcept
{
  std::uint64_t n           = std::min<std::uint64_t>(leftover.size(), remaining());
  char const* src           = leftover.data();
  std::size_t idx           = _idx;
  std::uint64_t off         = _off;
  std::uint64_t const total = n;
  while (n != 0) {
    iovec const& v        = _dst[idx];
    std::uint64_t const k = std::min<std::uint64_t>(v.iov_len - off, n);
    if (k != 0) std::memcpy(static_cast<char*>(v.iov_base) + off, src, k);
    src += k;
    n -= k;
    off = 0;
    ++idx;
  }
  advance(total);
}

std::span<iovec> iovec_sink::window(std::span<iovec> scratch, std::uint64_t max_bytes) noexcept
{
  std::uint64_t budget = std::min(max_bytes, remaining());
  std::size_t n        = 0;
  std::size_t idx      = _idx;
  std::uint64_t off    = _off;
  while (budget != 0 && n < scratch.size() && idx < _dst.size()) {
    iovec const& v           = _dst[idx];
    std::uint64_t const room = v.iov_len - off;
    if (room != 0) {
      std::uint64_t const k = std::min(room, budget);
      scratch[n].iov_base   = static_cast<char*>(v.iov_base) + off;
      scratch[n].iov_len    = static_cast<std::size_t>(k);
      ++n;
      budget -= k;
    }
    off = 0;
    ++idx;
  }
  return scratch.first(n);
}

bool iovec_sink::contiguous_window(std::byte*& ptr,
                                   std::uint64_t& len,
                                   std::uint64_t max_bytes) const noexcept
{
  std::uint64_t const want = std::min(max_bytes, remaining());
  if (want == 0) return false;
  iovec const& v = _dst[_idx];
  if (v.iov_len - _off < want) return false;
  ptr = static_cast<std::byte*>(v.iov_base) + _off;
  len = want;
  return true;
}

void iovec_sink::commit(std::uint64_t n) noexcept { advance(std::min(n, remaining())); }

}  // namespace cucascade::io::http
