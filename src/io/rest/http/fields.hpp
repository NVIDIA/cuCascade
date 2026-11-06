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

/// @file fields.hpp
/// @brief S3 known-field table: `switch (len)` plus a masked 4/8-byte compare per candidate.
/// Replaces Beast's 129-name perfect hash (`field.ipp`) with the ~14 names the codec decodes
/// (gpuio docs/research/beast.md §2.1.4: "a `switch(len)` followed by a masked compare is
/// cheaper").

#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <type_traits>

namespace cucascade::io::http {

/// Header fields the codec recognises without a string compare chain.
enum class known_field : std::uint8_t {
  unknown,
  connection,
  content_length,
  content_range,
  content_type,
  date,
  etag,
  last_modified,
  proxy_connection,
  retry_after,
  transfer_encoding,
  x_amz_id_2,
  x_amz_request_id,
  x_amz_retry_after,
};

namespace detail {

/// A string literal usable as a template argument (C++20 class-type NTTP).
template <std::size_t N>
struct field_literal {
  char s[N];
  // NOLINTNEXTLINE(google-explicit-constructor): must be implicit so a literal deduces the NTTP
  consteval field_literal(char const (&a)[N]) noexcept
  {
    for (std::size_t i = 0; i < N; ++i)
      s[i] = a[i];
  }
  [[nodiscard]] static constexpr std::size_t size() noexcept { return N - 1; }
};

/// Byte offset of the i-th compared word for a name of length n: consecutive words, the last one
/// overlapping the previous so that every load stays inside the name.
constexpr std::size_t word_offset(std::size_t i, std::size_t n, std::size_t w) noexcept
{
  return (i + 1) * w <= n ? i * w : n - w;
}

template <class Word, std::size_t N>
constexpr Word literal_word(field_literal<N> const& l, std::size_t off) noexcept
{
  Word w = 0;
  for (std::size_t i = 0; i < sizeof(Word); ++i) {
    auto const byte = static_cast<Word>(static_cast<unsigned char>(l.s[off + i]));
    auto const shift =
      std::endian::native == std::endian::little ? 8 * i : 8 * (sizeof(Word) - 1 - i);
    w = static_cast<Word>(w | static_cast<Word>(byte << shift));
  }
  return w;
}

template <class Word>
constexpr Word lower_mask() noexcept
{
  Word m = 0;
  for (std::size_t i = 0; i < sizeof(Word); ++i)
    m = static_cast<Word>((m << 8) | 0x20U);
  return m;
}

/// The literal's words at `word_offset` positions, computed at compile time.
template <field_literal L, class Word>
inline constexpr auto literal_words = [] {
  constexpr std::size_t n = decltype(L)::size();
  constexpr std::size_t k = (n + sizeof(Word) - 1) / sizeof(Word);
  std::array<Word, k> w{};
  for (std::size_t i = 0; i < k; ++i)
    w[i] = literal_word<Word>(L, word_offset(i, n, sizeof(Word)));
  return w;
}();

template <class Word>
inline Word load_word(char const* p) noexcept
{
  Word w;
  std::memcpy(&w, p, sizeof(Word));
  return w;
}

/// Case-insensitive compare of `p[0, L.size())` against lowercase literal `L` using overlapping
/// word loads OR'ed with 0x20 per byte. Exact for the names in `known_field`: their characters are
/// lowercase letters, '-' and '2', each of which shares its 0x20-folded value only with its
/// uppercase letter or with a control character that the field-name scan already rejected.
template <field_literal L>
inline bool ieq(char const* p) noexcept
{
  constexpr std::size_t n = decltype(L)::size();
  static_assert(n >= 4);
  using word         = std::conditional_t<(n >= 8), std::uint64_t, std::uint32_t>;
  constexpr auto& lw = literal_words<L, word>;
  constexpr auto m   = lower_mask<word>();
  word diff          = 0;
  for (std::size_t i = 0; i < lw.size(); ++i) {
    diff |= static_cast<word>((load_word<word>(p + word_offset(i, n, sizeof(word))) | m) ^ lw[i]);
  }
  return diff == 0;
}

}  // namespace detail

/// Map a field name (a token, as delivered by the parser) to its `known_field` id.
/// @param name field name; case-insensitive.
/// @return `known_field::unknown` for every other name.
/// @note Exactness relies on `name` being a token (RFC 9110 §5.6.2): control characters fold onto
/// '-' and '2' under the 0x20 mask, and the parser rejects them before calling this.
#if defined(__GNUC__)
[[gnu::always_inline]]
#endif
[[nodiscard]] inline known_field
lookup_field(std::string_view name) noexcept
{
  using detail::ieq;
  char const* const p = name.data();
  switch (name.size()) {
    case 4:
      if (ieq<"etag">(p)) return known_field::etag;
      if (ieq<"date">(p)) return known_field::date;
      break;
    case 10:
      if (ieq<"x-amz-id-2">(p)) return known_field::x_amz_id_2;
      if (ieq<"connection">(p)) return known_field::connection;
      break;
    case 11:
      if (ieq<"retry-after">(p)) return known_field::retry_after;
      break;
    case 12:
      if (ieq<"content-type">(p)) return known_field::content_type;
      break;
    case 13:
      if (ieq<"content-range">(p)) return known_field::content_range;
      if (ieq<"last-modified">(p)) return known_field::last_modified;
      break;
    case 14:
      if (ieq<"content-length">(p)) return known_field::content_length;
      break;
    case 16:
      if (ieq<"x-amz-request-id">(p)) return known_field::x_amz_request_id;
      if (ieq<"proxy-connection">(p)) return known_field::proxy_connection;
      break;
    case 17:
      if (ieq<"transfer-encoding">(p)) return known_field::transfer_encoding;
      if (ieq<"x-amz-retry-after">(p)) return known_field::x_amz_retry_after;
      break;
    default: break;
  }
  return known_field::unknown;
}

}  // namespace cucascade::io::http
