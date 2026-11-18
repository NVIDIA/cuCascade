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

#include "io/rest/http/content_range.hpp"

#include "io/rest/errc.hpp"
#include "io/rest/http/rules.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace cucascade::io::http {

using rest::errc;
namespace {

/// 1*DIGIT at `it` (overflow-checked); advances `it`.
bool take_number(char const*& it, char const* last, std::uint64_t& v) noexcept
{
  if (char const* const e = detail::parse_dec_short(it, last, v)) {
    it = e;
    return true;
  }
  char const* e = it;
  while (e != last && detail::is_digit(*e))
    ++e;
  if (!detail::parse_dec(detail::make_view(it, e), v)) return false;
  it = e;
  return true;
}

bool take_char(char const*& it, char const* last, char c) noexcept
{
  if (it == last || *it != c) return false;
  ++it;
  return true;
}

}  // namespace

// Content-Range = range-unit SP ( range-resp / unsatisfied-range )
//   range-resp        = incl-range "/" ( complete-length / "*" )
//   incl-range        = first-pos "-" last-pos
//   unsatisfied-range = "*/" complete-length
std::error_code parse_content_range(std::string_view v, content_range& out) noexcept
{
  auto const bad           = [] { return make_error_code(errc::http_bad_message); };
  std::string_view const t = detail::trim_ows(v);
  char const* it           = t.data();
  char const* const last   = t.data() + t.size();
  if (t.size() < 6 || it[5] != ' ' ||
      (std::memcmp(it, "bytes", 5) != 0 && !detail::iequals_lower(t.substr(0, 5), "bytes"))) {
    return bad();
  }
  it += 6;
  while (it != last && *it == ' ')
    ++it;

  content_range r;
  if (take_char(it, last, '*')) {
    if (!take_char(it, last, '/') || !take_number(it, last, r.complete) || it != last) return bad();
    r.unsatisfied = true;
    out           = r;
    return {};
  }
  if (!take_number(it, last, r.first) || !take_char(it, last, '-') ||
      !take_number(it, last, r.last) || !take_char(it, last, '/')) {
    return bad();
  }
  if (take_char(it, last, '*')) {
    r.complete = unknown_length;
  } else if (!take_number(it, last, r.complete) || r.last >= r.complete) {
    return bad();
  }
  if (it != last || r.first > r.last) return bad();
  out = r;
  return {};
}

}  // namespace cucascade::io::http
