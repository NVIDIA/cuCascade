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

#include "http_test_support.hpp"

#include <catch2/catch_all.hpp>

#include <array>
#include <cstdint>
#include <string>

using namespace cucascade::io::http;

namespace {
std::string_view as_view(std::span<char const> s) { return {s.data(), s.size()}; }
}  // namespace

TEST_CASE("http.request_head.s3_get", "[rest][http][request]")
{
  std::array<char, 2048> storage{};
  request_head h{storage};
  h.start(method::get, "/data/ghcnd_all.tar.gz")
    .host("noaa-ghcn-pds.s3.amazonaws.com")
    .range(0, 1048575)
    .field("If-Match", "\"6b1f5c2b7f1a9e0d3c4b5a6978879a5c\"")
    .field("x-amz-date", "20261012T172103Z")
    .field("x-amz-content-sha256", "UNSIGNED-PAYLOAD")
    .field("Authorization",
           "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20261012/us-east-1/s3/aws4_request, "
           "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, Signature=abc123")
    .field("Connection", "keep-alive")
    .field("Accept-Encoding", "identity")
    .field("User-Agent", "cucascade/0.1");
  auto const out = h.finish();
  REQUIRE_FALSE(h.overflow());
  CHECK(as_view(out) ==
        "GET /data/ghcnd_all.tar.gz HTTP/1.1\r\n"
        "Host: noaa-ghcn-pds.s3.amazonaws.com\r\n"
        "Range: bytes=0-1048575\r\n"
        "If-Match: \"6b1f5c2b7f1a9e0d3c4b5a6978879a5c\"\r\n"
        "x-amz-date: 20261012T172103Z\r\n"
        "x-amz-content-sha256: UNSIGNED-PAYLOAD\r\n"
        "Authorization: AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20261012/us-east-1/s3/"
        "aws4_request, SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, "
        "Signature=abc123\r\n"
        "Connection: keep-alive\r\n"
        "Accept-Encoding: identity\r\n"
        "User-Agent: cucascade/0.1\r\n"
        "\r\n");
  CHECK(out.data() == storage.data());  // one contiguous buffer in caller storage
  CHECK(h.verb() == method::get);
  CHECK(h.target() == "/data/ghcnd_all.tar.gz");
  auto const f = h.fields();
  REQUIRE(f.size() == 9);
  CHECK(f[0].name == "Host");
  CHECK(f[0].value == "noaa-ghcn-pds.s3.amazonaws.com");
  CHECK(f[1].name == "Range");
  CHECK(f[1].value == "bytes=0-1048575");
  CHECK(f[8].name == "User-Agent");
  // field spans point into the head itself (signer input without re-parsing)
  CHECK(f[3].value.data() >= storage.data());
  CHECK(f[3].value.data() < storage.data() + out.size());
  // finish() is idempotent
  CHECK(h.finish().size() == out.size());
}

TEST_CASE("http.request_head.methods_and_put", "[rest][http][request]")
{
  std::array<char, 512> storage{};
  request_head h{storage};
  h.start(method::put, "/bucket/key?partNumber=2&uploadId=xyz")
    .host("s3.us-west-2.amazonaws.com:443")
    .content_length(16777216)
    .expect_100_continue();
  CHECK(as_view(h.finish()) ==
        "PUT /bucket/key?partNumber=2&uploadId=xyz HTTP/1.1\r\n"
        "Host: s3.us-west-2.amazonaws.com:443\r\n"
        "Content-Length: 16777216\r\n"
        "Expect: 100-continue\r\n"
        "\r\n");
  for (auto [m, name] : {std::pair{method::head, "HEAD /k HTTP/1.1\r\n"},
                         std::pair{method::post, "POST /k HTTP/1.1\r\n"},
                         std::pair{method::delete_, "DELETE /k HTTP/1.1\r\n"},
                         std::pair{method::get, "GET /k HTTP/1.1\r\n"}}) {
    h.reset(storage);
    h.start(m, "/k");
    CHECK(as_view(h.finish()) == std::string{name} + "\r\n");
  }
  h.reset(storage);
  h.start(method::get, "/k").range(18446744073709551614ULL, 18446744073709551615ULL);
  CHECK(h.fields()[0].value == "bytes=18446744073709551614-18446744073709551615");
  h.reset(storage);
  h.start(method::put, "/k").content_length(0);
  CHECK(h.fields()[0].value == "0");
}

TEST_CASE("http.request_head.rejects_injection_and_misuse", "[rest][http][request]")
{
  std::array<char, 1024> storage{};
  auto fresh = [&storage] {
    request_head h{storage};
    h.start(method::get, "/k");
    return h;
  };
  auto h = fresh();
  h.field("x-a", "ok\r\nInjected: yes");
  CHECK(h.overflow());
  CHECK(h.finish().empty());
  h.field("x-b", "fine");  // sticky: still failed, nothing appended
  CHECK(h.overflow());
  CHECK(h.fields().empty());

  CHECK(fresh().field("x-a", "lf\nonly").overflow());
  CHECK(fresh().field("x-a", std::string_view{"nul\0", 4}).overflow());
  CHECK(fresh().field("bad name", "v").overflow());
  CHECK(fresh().field("bad:name", "v").overflow());
  CHECK(fresh().field("", "v").overflow());
  CHECK(fresh().host("evil.com\r\nX: y").overflow());
  CHECK(fresh().host("a b").overflow());
  CHECK(fresh().range(10, 9).overflow());
  CHECK_FALSE(fresh().field("x-tab", "a\tb").overflow());  // HTAB is allowed

  request_head t{storage};
  CHECK(t.start(method::get, "/a b").overflow());
  t.reset(storage);
  CHECK(t.start(method::get, "/a\r\nHost: x").overflow());
  t.reset(storage);
  CHECK(t.start(method::get, "").overflow());

  request_head n{storage};
  CHECK(n.field("x", "y").overflow());  // before start()
  CHECK(n.finish().empty());

  auto d = fresh();
  (void)d.finish();
  CHECK(d.field("x", "y").overflow());  // after finish()

  auto e = fresh();
  CHECK(e.start(method::get, "/again").overflow());  // start() twice
}

TEST_CASE("http.request_head.capacity_limits", "[rest][http][request]")
{
  std::array<char, 40> small{};
  request_head h{small};
  h.start(method::get, "/key").host("bucket.s3.amazonaws.com");
  CHECK(h.overflow());
  CHECK(h.finish().empty());

  // Exactly full: 17-byte request line + 18-byte Host line + final CRLF = 37 bytes.
  std::array<char, 37> exact{};
  request_head x{exact};
  x.start(method::get, "/k").field("Host", "abcdefghij");
  REQUIRE_FALSE(x.overflow());
  CHECK(x.finish().size() == 37);
  std::array<char, 36> one_short{};
  request_head y{one_short};
  y.start(method::get, "/k").field("Host", "abcdefghij");
  CHECK(y.finish().empty());

  std::array<char, 4096> big{};
  request_head f{big};
  f.start(method::get, "/k");
  for (std::size_t i = 0; i < request_head::max_fields; ++i)
    f.field("x", "y");
  CHECK_FALSE(f.overflow());
  f.field("x", "y");
  CHECK(f.overflow());
}

// gpuio's `http.request_head.zero_allocations` (global operator-new counter) is not ported: the
// counter replaces the global allocator of the whole test binary.
