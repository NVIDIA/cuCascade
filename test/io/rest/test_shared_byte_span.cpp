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

#include <cucascade/io/rest/rest_reactor.hpp>

#include <catch2/catch.hpp>

#include <cstdint>
#include <numeric>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

using cucascade::io::rest::make_shared_byte_span;
using cucascade::io::rest::shared_byte_span;

namespace {

std::vector<std::uint8_t> iota_bytes(std::size_t n)
{
  std::vector<std::uint8_t> v(n);
  std::iota(v.begin(), v.end(), std::uint8_t{0});
  return v;
}

}  // namespace

TEST_CASE("shared_byte_span exposes a span, not the underlying container",
          "[rest][shared_byte_span]")
{
  STATIC_REQUIRE(
    std::is_same_v<shared_byte_span, std::shared_ptr<const std::span<const std::uint8_t>>>);
  STATIC_REQUIRE(
    std::is_same_v<shared_byte_span::element_type, const std::span<const std::uint8_t>>);
}

TEST_CASE("make_shared_byte_span preserves contents and size", "[rest][shared_byte_span]")
{
  auto const stash = make_shared_byte_span(iota_bytes(256));

  REQUIRE(stash);
  REQUIRE(stash->size() == 256);
  for (std::size_t i = 0; i < stash->size(); ++i) {
    REQUIRE((*stash)[i] == static_cast<std::uint8_t>(i));
  }
}

TEST_CASE("make_shared_byte_span takes ownership of the buffer", "[rest][shared_byte_span]")
{
  shared_byte_span stash;
  {
    // The source vector goes out of scope here; the span must still be valid
    // because the aliasing shared_ptr's control block owns the moved-in buffer.
    auto source = iota_bytes(64);
    stash       = make_shared_byte_span(std::move(source));
  }

  REQUIRE(stash);
  REQUIRE(stash->size() == 64);
  CHECK((*stash)[0] == 0);
  CHECK((*stash)[63] == 63);
}

TEST_CASE("shared_byte_span keeps the buffer alive through the last owner",
          "[rest][shared_byte_span]")
{
  auto first              = make_shared_byte_span(iota_bytes(32));
  auto const* data_before = first->data();

  shared_byte_span second = first;
  REQUIRE(first.use_count() == 2);

  first.reset();  // drop the original owner

  // The aliasing pointer shares one control block, so the buffer survives and
  // does not move: the second handle still sees the same address and bytes.
  REQUIRE(second);
  CHECK(second->data() == data_before);
  CHECK(second->size() == 32);
  CHECK((*second)[31] == 31);
}

TEST_CASE("shared_byte_span supports the read patterns host_read uses", "[rest][shared_byte_span]")
{
  auto const stash = make_shared_byte_span(iota_bytes(128));

  // host_read's stash fast path: window arithmetic then a memcpy off .data().
  constexpr std::size_t window_lo = 1000;
  std::size_t const hi            = window_lo + stash->size();
  CHECK(hi == 1128);

  constexpr std::size_t offset = 1010;
  constexpr std::size_t size   = 16;
  REQUIRE(offset >= window_lo);
  REQUIRE(offset + size <= hi);

  auto const sub = stash->subspan(offset - window_lo, size);
  REQUIRE(sub.size() == size);
  CHECK(sub[0] == 10);
  CHECK(sub[15] == 25);
}

TEST_CASE("a default-constructed shared_byte_span is falsy", "[rest][shared_byte_span]")
{
  shared_byte_span const none;
  CHECK_FALSE(none);
}

TEST_CASE("make_shared_byte_span handles an empty buffer", "[rest][shared_byte_span]")
{
  auto const stash = make_shared_byte_span({});

  // Non-null (the probe succeeded) but empty — distinct from the null "no
  // probe" state the caller checks with operator bool.
  REQUIRE(stash);
  CHECK(stash->empty());
  CHECK(stash->size() == 0);
}
