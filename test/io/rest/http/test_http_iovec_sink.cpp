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

#include "http_s3_fixtures.hpp"
#include "http_test_support.hpp"
#include "io/rest/errc.hpp"

#include <catch2/catch_all.hpp>

#include <string>

using namespace cucascade::io::http;

TEST_CASE("http.iovec_sink.scatter_cursor", "[rest][http][sink]")
{
  std::string a(10, '.'), b(0, '.'), c(5, '.'), d(7, '.');
  iovec dst[4] = {
    {a.data(), a.size()}, {b.data(), b.size()}, {c.data(), c.size()}, {d.data(), d.size()}};
  iovec_sink s{dst};
  CHECK(s.capacity() == 22);
  CHECK(s.remaining() == 22);
  CHECK_FALSE(s.init(22));
  CHECK_FALSE(s.init(std::nullopt));
  CHECK(s.init(23) == cucascade::io::rest::errc::http_length_mismatch);

  s.copy_in(std::string_view{"0123456789AB"});  // spans a, skips empty b, into c
  CHECK(s.written() == 12);
  CHECK(a == "0123456789");
  CHECK(c == "AB...");

  iovec scratch[8];
  auto w = s.window(scratch, 100);
  REQUIRE(w.size() == 2);
  CHECK(w[0].iov_base == c.data() + 2);
  CHECK(w[0].iov_len == 3);
  CHECK(w[1].iov_base == d.data());
  CHECK(w[1].iov_len == 7);
  w = s.window(scratch, 2);  // byte cap
  REQUIRE(w.size() == 1);
  CHECK(w[0].iov_len == 2);
  w = s.window(std::span<iovec>{scratch, 1}, 100);  // iovec cap
  CHECK(w.size() == 1);

  std::byte* ptr    = nullptr;
  std::uint64_t len = 0;
  CHECK(s.contiguous_window(ptr, len, 3));  // fits in the rest of c
  CHECK(ptr == reinterpret_cast<std::byte*>(c.data() + 2));
  CHECK(len == 3);
  CHECK_FALSE(s.contiguous_window(ptr, len, 4));  // would span c and d

  std::memcpy(c.data() + 2, "CDE", 3);
  std::memcpy(d.data(), "FG", 2);
  s.commit(5);
  CHECK(s.written() == 17);
  CHECK(s.contiguous_window(ptr, len, 1000));
  CHECK(len == 5);
  CHECK(ptr == reinterpret_cast<std::byte*>(d.data() + 2));
  CHECK(s.init(6) == cucascade::io::rest::errc::http_length_mismatch);  // checks what is left
  s.commit(100);                                                        // clamped
  CHECK(s.remaining() == 0);
  CHECK(s.window(scratch, 100).empty());
  CHECK_FALSE(s.contiguous_window(ptr, len, 100));
  s.copy_in(std::string_view{"overflow"});  // dropped
  CHECK(s.written() == 22);

  iovec_sink empty;
  CHECK(empty.capacity() == 0);
  CHECK(empty.window(scratch, 10).empty());
  CHECK(empty.init(0) == std::error_code{});
}

TEST_CASE("http.iovec_sink.chunked_error_body", "[rest][http][sink]")
{
  // The chunked decoder feeding an iovec_sink scatter list (drained error body).
  std::string_view const msg = s3_fixtures::resp_403;
  response_parser p;
  response_meta meta;
  std::error_code ec;
  std::size_t used = 0;
  REQUIRE(p.parse_header(msg, used, meta, ec) == parse_status::ok);
  REQUIRE(meta.body == framing::chunked);

  std::string x(100, '\0'), y(100, '\0'), z(100, '\0');
  iovec dst[3] = {{x.data(), x.size()}, {y.data(), y.size()}, {z.data(), z.size()}};
  iovec_sink sink{dst};
  chunked_decoder dec{max_error_body};
  std::size_t body_used = 0;
  auto const st         = dec.feed(
    msg.substr(used),
    body_used,
    [&sink](std::span<char const> s) noexcept { sink.copy_in(s); },
    ec);
  REQUIRE(st == parse_status::ok);
  CHECK(used + body_used == msg.size());
  CHECK(dec.body_bytes() == sink.written());
  std::string const body = (x + y + z).substr(0, sink.written());
  CHECK(extract_error_code(body, meta));
  CHECK(meta.amz_error_code.view() == "AccessDenied");
}
