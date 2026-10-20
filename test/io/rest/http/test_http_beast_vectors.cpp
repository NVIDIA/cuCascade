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

// Conformance vectors ported from the Boost.Beast test suite
//   resources/beast/test/beast/http/basic_parser.cpp  (testFlatten ... testIssue2201)
//   resources/beast/test/beast/http/parser.cpp        (testParse, testTrailerHeaders, issues)
//   resources/beast/test/beast/http/message_fuzz.hpp  (chunkExtensionsTest)
// Copyright (c) 2016-2019 Vinnie Falco (vinnie dot falco at gmail dot com), distributed under the
// Boost Software License, Version 1.0 (http://www.boost.org/LICENSE_1_0.txt).
//
// Adaptation: gpuio parses responses only (gpuio docs/research/beast.md §3.2). Request-line vectors
// (testRequestLine, testRegression1) are not ported. Request-form field vectors are re-targeted
// at a response head: "HTTP/1.1 204 No Content" where the vector checks field syntax or
// connection flags (204 has no body, so keep-alive depends on the flags alone, as for a bodiless
// request), and a HEAD-mode response (Beast's `skip(true)`) for Content-Length values. Beast's
// `parsegrind` / `failgrind` become `grind` / `failgrind` (test_support.hpp): one piece, every
// split point and byte-by-byte, with identical results required. Beast error names map 1:1 onto
// `cucascade::io::http::parse_error`.

#include "http_test_support.hpp"

#include <catch2/catch_all.hpp>

#include <string>

using namespace cucascade::io::http;
using namespace http_test;
using pe = parse_error;

namespace {

std::string const r204 = "HTTP/1.1 204 No Content\r\n";

/// Beast `m(s)`: a request head with extra field lines, re-targeted at a 204 response.
std::string m(std::string const& s) { return r204 + s + "\r\n"; }

parse_options head_mode()
{
  parse_options o;
  o.head_request = true;
  return o;
}

}  // namespace

TEST_CASE("http.beast.flatten", "[rest][http][beast]")
{
  auto r = grind("HTTP/1.1 403 Not Found\r\n\r\n");
  CHECK(r.head.status == parse_status::ok);
  CHECK(r.head.meta.status == 403);
  CHECK(r.head.meta.body == framing::until_eof);

  r = grind("HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\n*****");
  CHECK(r.body == "*****");
  CHECK(r.body_complete);

  r = grind(
    "HTTP/1.1 200 OK\r\n"
    "Transfer-Encoding: chunked\r\n"
    "\r\n"
    "5;x\r\n*****\r\n"
    "0\r\nMD5: 0xff30\r\n"
    "\r\n");
  CHECK(r.head.meta.body == framing::chunked);
  CHECK(r.chunked.status == parse_status::ok);
  CHECK(r.body == "*****");

  r = grind("HTTP/1.1 200 OK\r\n\r\n*****");
  CHECK(r.head.meta.body == framing::until_eof);
  CHECK(r.body == "*****");
}

TEST_CASE("http.beast.obs_fold", "[rest][http][beast]")
{
  auto const check = [](std::string const& s, std::string_view value) {
    auto const r = grind(r204 + "f: " + s + "\r\n\r\n");
    REQUIRE(r.head.status == parse_status::ok);
    REQUIRE(r.head.fields.size() == 1);
    CHECK(r.head.fields[0].first == "f");
    CHECK(r.head.fields[0].second == value);
  };
  check("x", "x");
  check(" x", "x");
  check("\tx", "x");
  check(" \tx", "x");
  check("\t x", "x");
  check("x ", "x");
  check(" x\t", "x");
  check("\tx \t", "x");
  check(" \tx\t ", "x");
  check("\t x  \t  ", "x");
  check("\r\n x", "x");
  check(" \r\n x", "x");
  check(" \r\n\tx", "x");
  check(" \r\n\t x", "x");
  check(" \r\n \tx", "x");
  check("  \r\n \r\n \r\n x \t", "x");
  check("xy", "xy");
  check("\r\n x", "x");
  check("\r\n  x", "x");
  check("\r\n   xy", "xy");
  check("\r\n \r\n \r\n x", "x");
  check("\r\n \r\n  \r\n xy", "xy");
  check("x\r\n y", "x y");
  check("x\r\n y\r\n z ", "x y z");
}

TEST_CASE("http.beast.callbacks", "[rest][http][beast]")
{
  // Beast completes the parser on a lone 100 Continue; gpuio skips it and waits for the final
  // head in the same exchange.
  auto r = grind("HTTP/1.1 100 Continue\r\n\r\n");
  CHECK(r.head.status == parse_status::need_more);
  CHECK(r.head.interim == 1);
  CHECK(r.head.body_offset == 25);

  r = grind("HTTP/1.1 200 OK\r\nServer: test\r\nContent-Length: 1\r\n\r\n*");
  CHECK(r.head.fields.size() == 1);  // Content-Length is decoded, not reported
  CHECK(r.body == "*");

  r = grind(
    "HTTP/1.1 200 OK\r\nServer: test\r\nTransfer-Encoding: chunked\r\n\r\n1\r\n*\r\n0\r\n\r\n");
  CHECK(r.head.fields.size() == 1);
  CHECK(r.body == "*");
  r = grind(
    "HTTP/1.1 200 OK\r\nServer: test\r\nTransfer-Encoding: chunked\r\n\r\n1;x\r\n*\r\n0\r\n\r\n");
  CHECK(r.body == "*");
}

TEST_CASE("http.beast.status_line", "[rest][http][beast]")
{
  auto const status = [](std::string const& s, unsigned st) {
    auto const r = grind(s);
    REQUIRE(r.head.status == parse_status::ok);
    CHECK(r.head.meta.status == st);
  };
  auto const version = [](std::string const& s, unsigned v) {
    auto const r = grind(s);
    REQUIRE(r.head.status == parse_status::ok);
    CHECK(r.head.meta.version == v);
  };
  auto const ok = [](std::string const& s) { CHECK(grind(s).head.status == parse_status::ok); };
  status("HTTP/1.0 000 OK\r\n\r\n", 0);
  status("HTTP/1.1 012 OK\r\n\r\n", 12);
  status("HTTP/1.0 345 OK\r\n\r\n", 345);
  status("HTTP/1.0 678 OK\r\n\r\n", 678);
  status("HTTP/1.0 999 OK\r\n\r\n", 999);
  version("HTTP/1.0 200 \tX\r\n\r\n", 10);
  version("HTTP/1.1 200  X\r\n\r\n", 11);
  ok("HTTP/1.0 200 \r\n\r\n");
  ok("HTTP/1.1 200 X \r\n\r\n");
  ok("HTTP/1.1 200 X\t\r\n\r\n");
  ok("HTTP/1.1 200 \x80\x81...\xfe\xff\r\n\r\n");
  ok(
    "HTTP/1.1 200 !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~\r\n\r\n");

  failgrind("\rHTTP/1.0 200 OK\r\n\r\n", pe::bad_version);
  failgrind("\nHTTP/1.0 200 OK\r\n\r\n", pe::bad_version);
  failgrind(" HTTP/1.0 200 OK\r\n\r\n", pe::bad_version);
  failgrind("_TTP/1.0 200 OK\r\n\r\n", pe::bad_version);
  failgrind("H_TP/1.0 200 OK\r\n\r\n", pe::bad_version);
  failgrind("HT_P/1.0 200 OK\r\n\r\n", pe::bad_version);
  failgrind("HTT_/1.0 200 OK\r\n\r\n", pe::bad_version);
  failgrind("HTTP_1.0 200 OK\r\n\r\n", pe::bad_version);
  failgrind("HTTP/01.2 200 OK\r\n\r\n", pe::bad_version);
  failgrind("HTTP/3.45 200 OK\r\n\r\n", pe::bad_version);
  failgrind("HTTP/67.89 200 OK\r\n\r\n", pe::bad_version);
  failgrind("HTTP/x.0 200 OK\r\n\r\n", pe::bad_version);
  failgrind("HTTP/1.x 200 OK\r\n\r\n", pe::bad_version);
  failgrind("HTTP/1_0 200 OK\r\n\r\n", pe::bad_version);
  failgrind("HTTP/1.0  200 OK\r\n\r\n", pe::bad_status);
  failgrind("HTTP/1.0 0 OK\r\n\r\n", pe::bad_status);
  failgrind("HTTP/1.0 12 OK\r\n\r\n", pe::bad_status);
  failgrind("HTTP/1.0 3456 OK\r\n\r\n", pe::bad_status);
  failgrind("HTTP/1.0 200\r\n\r\n", pe::bad_status);
  failgrind("HTTP/1.0 200 \n\r\n\r\n", pe::bad_reason);
  failgrind("HTTP/1.0 200 \x01\r\n\r\n", pe::bad_reason);
  failgrind("HTTP/1.0 200 \x7f\r\n\r\n", pe::bad_reason);
  failgrind("HTTP/1.0 200 OK\n\r\n\r\n", pe::bad_reason);
  failgrind("HTTP/1.0 200 OK\r\r\n\r\n", pe::bad_line_ending);
  // HTTP/2+ status lines are rejected (only 1.0 and 1.1).
  failgrind("HTTP/2.0 200 OK\r\n\r\n", pe::bad_version);
}

TEST_CASE("http.beast.fields", "[rest][http][beast]")
{
  auto const ok = [](std::string const& s) { CHECK(grind(m(s)).head.status == parse_status::ok); };
  ok("f:\r\n");
  ok("f: \r\n");
  ok("f:\t\r\n");
  ok("f: \t\r\n");
  ok("f: v\r\n");
  ok("f:\tv\r\n");
  ok("f:\tv \r\n");
  ok("f:\tv\t\r\n");
  ok("f:\tv\t \r\n");
  ok("f:\r\n \r\n");
  ok("f:v\r\n");
  ok("f: v\r\n u\r\n");
  ok("!#$%&'*+-.^_`|~0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz: v\r\n");
  ok(
    "f: !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~\x80\x81...\xfe\xff\r\n");

  failgrind(m(" f: v\r\n"), pe::bad_field);
  failgrind(m("\tf: v\r\n"), pe::bad_field);
  failgrind(m("f : v\r\n"), pe::bad_field);
  failgrind(m("f\t: v\r\n"), pe::bad_field);
  failgrind(m("f: \n\r\n"), pe::bad_value);
  failgrind(m("f: v\r \r\n"), pe::bad_line_ending);
  failgrind(m("f: \r v\r\n"), pe::bad_line_ending);
  failgrind(r204 + "\r \n\r\n\r\n", pe::bad_line_ending);
  // gpuio additions: empty name, NUL and DEL in values.
  failgrind(m(": v\r\n"), pe::bad_field);
  failgrind(m(std::string{"f: a\0b\r\n", 8}), pe::bad_value);
  failgrind(m("f: a\x7f"
              "b\r\n"),
            pe::bad_value);
}

TEST_CASE("http.beast.connection_field", "[rest][http][beast]")
{
  auto const cn   = [](std::string const& s) { return r204 + "Connection: " + s + "\r\n"; };
  auto const keep = [](std::string const& msg, bool v) {
    auto const r = grind(msg);
    REQUIRE(r.head.status == parse_status::ok);
    CHECK(r.head.meta.keep_alive == v);
  };
  for (char const* s : {"close\r\n",
                        ",close\r\n",
                        " close\r\n",
                        "\tclose\r\n",
                        "close,\r\n",
                        "close\t\r\n",
                        " ,\t,,close,, ,\t,,\r\n",
                        "\r\n close\r\n",
                        "close\r\n \r\n",
                        "any,close\r\n",
                        "close,any\r\n",
                        "any\r\n ,close\r\n",
                        "close\r\n ,any\r\n",
                        "close,close\r\n"}) {
    keep(cn(s), false);
  }
  for (char const* s : {"keep-alive\r\n",
                        "keep-alive \r\n",
                        "keep-alive\t \r\n",
                        "keep-alive\t ,x\r\n",
                        "\r\n keep-alive \t\r\n",
                        "keep-alive \r\n \t \r\n",
                        "keep-alive\r\n \r\n",
                        "upgrade\r\n",
                        "upgrade \r\n",
                        "upgrade\t \r\n",
                        "upgrade\t ,x\r\n",
                        "\r\n upgrade \t\r\n",
                        "upgrade \r\n \t \r\n",
                        "upgrade\r\n \r\n",
                        "upgrade,keep-alive\r\n",
                        "upgrade,\r\n keep-alive\r\n"}) {
    keep(cn(s), true);
  }
  keep("HTTP/1.1 204 No Content\r\n\r\n", true);
  keep("HTTP/1.0 204 No Content\r\n\r\n", false);
  keep("HTTP/1.0 204 No Content\r\nConnection: keep-alive\r\n\r\n", true);
  keep("HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n", false);

  // Tokens that only resemble close/keep-alive change nothing (Beast expect_flags 0).
  for (char const* s : {"x\r\n",
                        "x,y\r\n",
                        "x ,y\r\n",
                        "x\t,y\r\n",
                        "keep\r\n",
                        ",keep\r\n",
                        " keep\r\n",
                        "\tnone\r\n",
                        "keep,\r\n",
                        "keep\t\r\n",
                        " ,\t,,keep,, ,\t,,\r\n",
                        "\r\n keep\r\n",
                        "keep\r\n \r\n",
                        "closet\r\n",
                        ",closet\r\n",
                        " closet\r\n",
                        "\tcloset\r\n",
                        "closet,\r\n",
                        "closet\t\r\n",
                        " ,\t,,closet,, ,\t,,\r\n",
                        "\r\n closet\r\n",
                        "closet\r\n \r\n",
                        "clog\r\n",
                        "key\r\n",
                        "uptown\r\n",
                        "keeper\r\n \r\n",
                        "keep-alively\r\n \r\n",
                        "up\r\n \r\n",
                        "upgrader\r\n \r\n",
                        "none\r\n",
                        "\r\n none\r\n"}) {
    keep(cn(s), true);
  }
  keep(m("ConnectioX: close\r\n"), true);
  keep(m("Condor: close\r\n"), true);
  keep(m("Connect: close\r\n"), true);
  keep(m("Connections: close\r\n"), true);

  keep(m("Proxy-Connection: close\r\n"), false);
  keep(m("Proxy-Connection: keep-alive\r\n"), true);
  keep(m("Proxy-Connection: upgrade\r\n"), true);
  keep(m("Proxy-ConnectioX: none\r\n"), true);
  keep(m("Proxy-Connections: 1\r\n"), true);
  keep(m("Proxy-Connotes: see-also\r\n"), true);

  failgrind(cn("[\r\n"), pe::bad_value);
  failgrind(cn("close[\r\n"), pe::bad_value);
  failgrind(cn("close [\r\n"), pe::bad_value);
  failgrind(cn("close, upgrade [\r\n"), pe::bad_value);
  failgrind(cn("upgrade[]\r\n"), pe::bad_value);
  failgrind(cn("keep\r\n -alive\r\n"), pe::bad_value);
  failgrind(cn("keep-alive[\r\n"), pe::bad_value);
  failgrind(cn("keep-alive []\r\n"), pe::bad_value);
  failgrind(cn("no[ne]\r\n"), pe::bad_value);
}

TEST_CASE("http.beast.content_length_field", "[rest][http][beast]")
{
  auto const c = [](std::string const& s) {
    return "HTTP/1.1 200 OK\r\nContent-Length: " + s + "\r\n";
  };
  auto const check = [&](std::string const& s, std::uint64_t v) {
    auto const r = grind(c(s), head_mode());
    REQUIRE(r.head.status == parse_status::ok);
    REQUIRE(r.head.meta.content_length.has_value());
    CHECK(*r.head.meta.content_length == v);
    CHECK(r.head.meta.body == framing::none);  // HEAD: never a body
    CHECK(r.head.done);
    CHECK(r.head.body_remaining == 0);
  };
  check("0\r\n", 0);
  check("00\r\n", 0);
  check("1\r\n", 1);
  check("01\r\n", 1);
  check("9\r\n", 9);
  check("42 \r\n", 42);
  check("42\t\r\n", 42);
  check("42 \t \r\n", 42);
  check("42\r\n \t \r\n", 42);
  // Beast testIssue1211 (parse_dec) through Content-Length.
  check("001\r\n", 1);
  check("255\r\n", 255);
  check("65535\r\n", 65535);
  check("65536\r\n", 65536);
  check("4294967295\r\n", 4294967295ULL);
  check("18446744073709551615\r\n", 18446744073709551615ULL);

  auto const no_cl = [](std::string const& s) {
    auto const r = grind(s, head_mode());
    REQUIRE(r.head.status == parse_status::ok);
    CHECK_FALSE(r.head.meta.content_length.has_value());
  };
  no_cl(m("Content-LengtX: 0\r\n"));
  no_cl(m("Content-Lengths: many\r\n"));
  no_cl(m("Content: full\r\n"));
  CHECK(grind(m("Content-Length: 0\r\nContent-Length: 0\r\n"), head_mode()).head.status ==
        parse_status::ok);

  failgrind(c("\r\n"), pe::bad_content_length, head_mode());
  failgrind(c("18446744073709551616\r\n"), pe::bad_content_length, head_mode());
  failgrind(c("0 0\r\n"), pe::bad_content_length, head_mode());
  failgrind(c("0 1\r\n"), pe::bad_content_length, head_mode());
  failgrind(c(",\r\n"), pe::bad_content_length, head_mode());
  failgrind(c("0,\r\n"), pe::bad_content_length, head_mode());
  failgrind(c("-1\r\n"), pe::bad_content_length, head_mode());
  failgrind(c("+1\r\n"), pe::bad_content_length, head_mode());
  failgrind(c("0x10\r\n"), pe::bad_content_length, head_mode());
  failgrind(
    m("Content-Length: 0\r\nContent-Length: 100\r\n"), pe::multiple_content_length, head_mode());
}

TEST_CASE("http.beast.transfer_encoding_field", "[rest][http][beast]")
{
  // Beast `ce(s)`: Transfer-Encoding followed by an empty chunked body.
  auto const ce = [](std::string const& s) {
    return "HTTP/1.1 200 OK\r\nTransfer-Encoding: " + s + "\r\n0\r\n\r\n";
  };
  auto const transfer_encoding = [](std::string const& s) {
    return "HTTP/1.1 200 OK\r\nTransfer-Encoding: " + s + "\r\n";
  };
  auto const chunked = [](std::string const& msg) {
    auto const r = grind(msg);
    REQUIRE(r.head.status == parse_status::ok);
    CHECK(r.head.meta.body == framing::chunked);
    CHECK(r.head.meta.keep_alive);
    CHECK(r.chunked.status == parse_status::ok);
  };
  auto const not_chunked = [](std::string const& msg) {
    auto const r = grind(msg);
    REQUIRE(r.head.status == parse_status::ok);
    CHECK(r.head.meta.body == framing::until_eof);  // response: delimited by EOF
    CHECK_FALSE(r.head.meta.keep_alive);
  };
  for (char const* s : {"chunked\r\n",
                        "chunked \r\n",
                        "chunked\t\r\n",
                        "chunked \t\r\n",
                        " chunked\r\n",
                        "\tchunked\r\n",
                        "chunked,\r\n",
                        "chunked ,\r\n",
                        "chunked, \r\n",
                        ",chunked\r\n",
                        ", chunked\r\n",
                        " ,chunked\r\n",
                        "chunked\r\n \r\n",
                        "\r\n chunked\r\n",
                        ",\r\n chunked\r\n",
                        "\r\n ,chunked\r\n",
                        "gzip, chunked\r\n",
                        "gzip, chunked \r\n",
                        "gzip, \r\n chunked\r\n"}) {
    chunked(ce(s));
  }
  for (char const* s : {"gzip\r\n",
                        "chunked, gzip\r\n",
                        "chunked\r\n , gzip\r\n",
                        "chunked,\r\n gzip\r\n",
                        "chunked,\r\n ,gzip\r\n",
                        "bigchunked\r\n",
                        "chunk\r\n ked\r\n",
                        "bar\r\n ley chunked\r\n",
                        "barley\r\n chunked\r\n"}) {
    not_chunked(transfer_encoding(s) + "\r\n");
  }
  for (char const* s : {"Transfer-EncodinX: none\r\n",
                        "Transfer-Encodings: 2\r\n",
                        "Transfer-Encoded: false\r\n"}) {
    auto const r = grind(m(s));
    REQUIRE(r.head.status == parse_status::ok);
    CHECK(r.head.meta.body == framing::none);  // 204, no Transfer-Encoding recognised
  }

  failgrind("HTTP/1.1 200 OK\r\nContent-Length: 1\r\nTransfer-Encoding: chunked\r\n\r\n",
            pe::bad_transfer_encoding);
  // repeated Transfer-Encoding fields concatenate
  chunked(
    "HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip\r\nTransfer-Encoding: chunked\r\n\r\n"
    "0\r\n\r\n");
  // a coding after chunked means chunked is not final
  failgrind("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nTransfer-Encoding: gzip\r\n\r\n",
            pe::bad_transfer_encoding);
  // chunked applied twice is invalid
  failgrind("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nTransfer-Encoding: chunked\r\n\r\n",
            pe::bad_transfer_encoding);
  // a response whose Transfer-Encoding does not end in chunked is delimited by EOF
  not_chunked("HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip\r\n\r\n");
  not_chunked("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked, gzip\r\n\r\n");
  // HTTP/1.0 responses with chunked framing are accepted (only requests are rejected)
  chunked(
    "HTTP/1.0 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n"
    "0\r\n\r\n");
  chunked(
    "HTTP/1.0 200 OK\r\nTransfer-Encoding: gzip, chunked\r\nConnection: keep-alive\r\n"
    "\r\n0\r\n\r\n");
  // a message must not carry both Transfer-Encoding and Content-Length, in either order
  failgrind("HTTP/1.1 200 OK\r\nContent-Length: 5\r\nTransfer-Encoding: gzip\r\n\r\n",
            pe::bad_transfer_encoding);
  failgrind("HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip\r\nContent-Length: 5\r\n\r\n",
            pe::bad_content_length);
}

TEST_CASE("http.beast.partial", "[rest][http][beast]")
{
  // No duplicate or missing fields on any split (Beast's slow-loris defense test).
  auto const r = grind(
    "HTTP/1.1 204 No Content\r\n"
    "a: 0\r\nb: 1\r\nc: 2\r\nd: 3\r\ne: 4\r\nf: 5\r\ng: 6\r\nh: 7\r\ni: 8\r\nj: 9\r\n"
    "\r\n");
  REQUIRE(r.head.fields.size() == 10);
  for (std::size_t i = 0; i < 10; ++i) {
    CHECK(r.head.fields[i].first == std::string(1, static_cast<char>('a' + i)));
    CHECK(r.head.fields[i].second == std::string(1, static_cast<char>('0' + i)));
  }
}

TEST_CASE("http.beast.limits", "[rest][http][beast]")
{
  {
    parse_options o;
    o.header_limit = 10;
    failgrind("HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n**", pe::header_limit, o);
  }
  {
    parse_options o;
    o.header_limit = 18;
    response_parser p{o};
    response_meta meta;
    std::error_code ec;
    std::size_t used = 0;
    std::string b    = "HTTP/1.1 200 OK\r\n";
    CHECK(p.parse_header(b, used, meta, ec) == parse_status::need_more);
    CHECK(used == 17);
    b.erase(0, used);
    b += "field: value\r\n";
    CHECK(p.parse_header(b, used, meta, ec) == parse_status::error);
    CHECK(ec == cucascade::io::rest::errc::http_header_too_large);
    CHECK(p.last_error() == pe::header_limit);
    CHECK_FALSE(p.done());
  }
  {
    // Body limit charged per chunk, before any of its bytes are emitted.
    auto const r = grind_chunked("2\r\n**\r\n0\r\n\r\n", 1);
    CHECK(r.status == parse_status::error);
    CHECK(r.detail == pe::body_limit);
    CHECK(r.ec == cucascade::io::rest::errc::http_length_mismatch);
    CHECK(r.body.empty());
  }
  {
    // A chunk line (with extensions) of 8 KiB is rejected.
    chunked_decoder d;
    std::string b    = "1;" + std::string(8192 - 3, 'x');
    std::size_t used = 0;
    std::error_code ec;
    auto sink = [](std::span<char const>) noexcept {};
    CHECK(d.feed(b, used, sink, ec) == parse_status::need_more);
    CHECK(used == 0);
    b += 'x';
    CHECK(d.feed(b, used, sink, ec) == parse_status::error);
    CHECK(d.last_error() == pe::bad_chunk_extension);
  }
  {
    // Trailers count against a 16 KiB cap (Beast charges them to header_limit).
    std::string body = "0\r\n";
    for (int i = 0; i < 2000; ++i)
      body += "x-trailer: 0123456789\r\n";
    body += "\r\n";
    auto const r = decode_chunked(body, {});
    CHECK(r.status == parse_status::error);
    CHECK(r.detail == pe::header_limit);
  }
}

TEST_CASE("http.beast.body", "[rest][http][beast]")
{
  auto r = grind(
    "HTTP/1.1 200 OK\r\n"
    "Transfer-Encoding: chunked\r\n"
    "Content-Type: application/octet-stream\r\n"
    "\r\n"
    "4\r\nabcd\r\n"
    "0\r\n\r\n");
  CHECK(r.body == "abcd");
  REQUIRE(r.head.fields.size() == 1);
  CHECK(r.head.fields[0].first == "Content-Type");

  r = grind(
    "HTTP/1.1 200 OK\r\n"
    "Server: test\r\n"
    "Expect: Expires, MD5-Fingerprint\r\n"
    "Transfer-Encoding: chunked\r\n"
    "\r\n"
    "5\r\n"
    "*****\r\n"
    "2;a;b=1;c=\"2\"\r\n"
    "--\r\n"
    "0;d;e=3;f=\"4\"\r\n"
    "Expires: never\r\n"
    "MD5-Fingerprint: -\r\n"
    "\r\n");
  CHECK(r.body == "*****--");

  r = grind("HTTP/1.0 200 OK\r\n\r\nhello");
  CHECK(r.body == "hello");
  CHECK(r.head.meta.body == framing::until_eof);
  CHECK_FALSE(r.head.meta.keep_alive);
  CHECK_FALSE(r.head.done);
  CHECK(r.head.body_remaining == unknown_length);

  r = grind("HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip\r\n\r\nhello");
  CHECK(r.body == "hello");
  r = grind("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked, gzip\r\n\r\nhello");
  CHECK(r.body == "hello");

  // 304 Not Modified has no body and does not need EOF.
  r = grind("HTTP/1.0 304 Not Modified\r\n\r\n");
  CHECK(r.head.done);
  CHECK(r.head.meta.body == framing::none);

  // A chunked response does not need EOF; the decoder finishes it.
  r = grind("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n");
  CHECK_FALSE(r.head.done);
  CHECK(r.chunked.status == parse_status::ok);
  CHECK(r.head.meta.keep_alive);

  // Content-Length body short of its length: the head is fine, the body is incomplete.
  r = grind("HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n12345");
  CHECK(r.head.status == parse_status::ok);
  CHECK(r.head.body_remaining == 10);
  CHECK_FALSE(r.body_complete);
}

TEST_CASE("http.beast.issues", "[rest][http][beast]")
{
  // #430: chunked with Content-Type
  auto r = grind(
    "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Type: application/octet-stream\r\n"
    "\r\n4\r\nabcd\r\n0\r\n\r\n");
  CHECK(r.body == "abcd");
  // #452: bytes after a bodiless message are not consumed
  r = grind("HTTP/1.1 204 No Content\r\n\r\ndie!");
  CHECK(r.head.status == parse_status::ok);
  CHECK(r.head.body_offset == 27);
  CHECK(r.head.done);
  // #496: leading zeros in chunk sizes
  r = grind(
    "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Type: application/octet-stream\r\n"
    "\r\n0004\r\nabcd\r\n0\r\n\r\n");
  CHECK(r.body == "abcd");
  // #692: 101 with a huge Content-Length has no body
  r = grind("HTTP/1.1 101 Switching Protocols\r\nContent-Length: 2147483648\r\n\r\n");
  CHECK(r.head.status == parse_status::ok);
  CHECK(r.head.done);
  CHECK(r.head.meta.body == framing::none);
  CHECK_FALSE(r.head.meta.keep_alive);  // the connection is no longer HTTP
  // #1187: redirect without length: head done, body to EOF
  r = grind("HTTP/1.1 301 Moved Permanently\r\nLocation: https://www.ebay.com\r\n\r\n\r\n");
  CHECK(r.head.status == parse_status::ok);
  CHECK_FALSE(r.head.done);
  CHECK(r.head.meta.body == framing::until_eof);
  // #1880: equal Content-Length values are allowed, different ones are not
  CHECK(grind("HTTP/1.1 200 OK\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n").head.status ==
        parse_status::ok);
  failgrind("HTTP/1.1 200 OK\r\nContent-Length: 0\r\nContent-Length: 1\r\n\r\n",
            pe::multiple_content_length);
  CHECK(grind("HTTP/1.1 200 OK\r\nContent-Length: 0, 0, 0\r\n\r\n").head.status ==
        parse_status::ok);
  failgrind("HTTP/1.1 200 OK\r\nContent-Length: 0, 0, 1\r\n\r\n", pe::multiple_content_length);
  failgrind("HTTP/1.1 200 OK\r\nContent-Length: 0, 0, 0\r\nContent-Length: 1\r\n\r\n",
            pe::multiple_content_length);
  // #2861: partial parse of a final chunk that is the only chunk
  {
    chunked_decoder d;
    std::error_code ec;
    std::size_t used = 0;
    auto sink        = [](std::span<char const>) noexcept {};
    std::string b    = "0\r\n";
    CHECK(d.feed(b, used, sink, ec) == parse_status::need_more);
    CHECK(used == 3);
    b = "\r";
    CHECK(d.feed(b, used, sink, ec) == parse_status::need_more);
    CHECK(used == 0);
    b = "\r\n";
    CHECK(d.feed(b, used, sink, ec) == parse_status::ok);
    CHECK(used == 2);
    CHECK(d.done());
  }
}

TEST_CASE("http.beast.parse_hex", "[rest][http][beast]")
{
  // Beast testIssue1267 through chunk-size lines.
  auto const good = [](std::string const& size_line, std::uint64_t) {
    auto const r = grind_chunked(size_line + "abc");
    CHECK(r.status == parse_status::need_more);
    CHECK(r.body == "abc");
  };
  good("f\r\n", 15);
  good("ff\r\n", 255);
  good("ffff\r\n", 65535);
  good("ffffffff\r\n", 4294967295ULL);
  good("ffffffffffffffff\r\n", 18446744073709551615ULL);
  for (char const* bad : {"\r\n", "g\r\n", "10000000000000000\r\n", "ffffffffffffffffffffff\r\n"}) {
    auto const r = grind_chunked(std::string{bad} + "data");
    CHECK(r.status == parse_status::error);
    CHECK(r.detail == pe::bad_chunk);
    CHECK(r.consumed == 0);
  }
}

TEST_CASE("http.beast.chunked_overflow", "[rest][http][beast]")
{
  {
    auto const r = grind_chunked("10000000000000000\r\ndata...");
    CHECK(r.detail == pe::bad_chunk);
    CHECK(r.consumed == 0);
  }
  {
    // Beast's default 8 MiB body limit rejects a huge chunk...
    auto const r = grind_chunked("1\r\nx\r\nFFFFFFFFFFFFFFFF\r\ndata...", 8 * 1024 * 1024);
    CHECK(r.detail == pe::body_limit);
    CHECK(r.body == "x");
  }
  {
    // ...and without a limit its first bytes are delivered.
    auto const r = grind_chunked("1\r\nx\r\nFFFFFFFFFFFFFFFF\r\ndata...");
    CHECK(r.status == parse_status::need_more);
    CHECK(r.body == "xdata...");
    CHECK(r.consumed == 31);
  }
}

TEST_CASE("http.beast.chunked_body_size", "[rest][http][beast]")
{
  std::string const body =
    "4\r\nWiki\r\n"
    "5\r\npedia\r\n"
    "E\r\n in\r\n\r\nchunks.\r\n"
    "0\r\n\r\n";
  auto r = grind_chunked(body, 23);
  CHECK(r.status == parse_status::ok);
  CHECK(r.body == "Wikipedia in\r\n\r\nchunks.");
  r = grind_chunked(body, 22);
  CHECK(r.status == parse_status::error);
  CHECK(r.detail == pe::body_limit);
}

TEST_CASE("http.beast.chunk_extensions", "[rest][http][beast]")
{
  auto const ce = [](std::string const& chunk, pe expected) {
    auto const r = grind_chunked(chunk);
    CHECK(r.detail == expected);
    if (expected == pe::none) {
      CHECK(r.status == parse_status::ok);
      CHECK(r.body == "X");
    }
  };
  ce("1;a=\"xy\"\r\nX\r\n0\r\n\r\n", pe::none);
  ce("1;a=\"\\\"\"\r\nX\r\n0\r\n\r\n", pe::none);
  ce("1;a=\"\n\"\r\nX\r\n0\r\n\r\n", pe::bad_chunk_extension);    // bare LF is not qdtext
  ce("1;a=\"\x01\"\r\nX\r\n0\r\n\r\n", pe::bad_chunk_extension);  // control char
  ce("1;a=\"\\\n\"\r\nX\r\n0\r\n\r\n", pe::bad_chunk_extension);  // quoted-pair of a control

  // message_fuzz.hpp chunkExtensionsTest, as testFuzz builds it.
  auto const msg = [](std::string const& ext) {
    return "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0" + ext + "\r\n\r\n";
  };
  for (char const* good : {"",
                           ";x",
                           ";x;y",
                           ";x=y",
                           ";x;y=z",
                           " ;x",
                           "\t;x",
                           " \t;x",
                           "\t ;x",
                           " ; x",
                           " ;\tx",
                           "\t ; \tx",
                           ";x= y",
                           " ;x= y",
                           " ; x= y",
                           R"(;x="\"")",
                           R"(;x="\\")",
                           R"(;x;y=z;z="\"";p="\\";q="1\"2\\")"}) {
    auto const r = grind(msg(good));
    CHECK(r.chunked.status == parse_status::ok);
  }
  for (char const* bad : {" ", ";", "=", " ;", "; ", " ; ", " ; x ", ";x =", ";x = ", ";x=="}) {
    auto const r = grind(msg(bad));
    CHECK(r.chunked.status == parse_status::error);
    CHECK(r.chunked.detail == pe::bad_chunk_extension);
  }
}

TEST_CASE("http.beast.parser_matrix", "[rest][http][beast]")
{
  // parser.cpp testParse (doMatrix over every two-piece split, here plus byte-by-byte).
  auto r = grind("HTTP/1.0 200 OK\r\nServer: test\r\n\r\nHello, world!");
  CHECK(r.head.meta.body == framing::until_eof);
  CHECK_FALSE(r.head.meta.content_length.has_value());
  CHECK(r.head.meta.version == 10);
  CHECK(r.head.meta.status == 200);
  REQUIRE(r.head.fields.size() == 1);
  CHECK(r.head.fields[0] == std::pair<std::string, std::string>{"Server", "test"});
  CHECK(r.body == "Hello, world!");

  r = grind(
    "HTTP/1.1 200 OK\r\n"
    "Server: test\r\n"
    "Transfer-Encoding: chunked\r\n"
    "Trailer: Content-Digest, X-Forwarded-For\r\n"
    "\r\n"
    "5\r\n"
    "*****\r\n"
    "2;a;b=1;c=\"2\"\r\n"
    "--\r\n"
    "0;d;e=3;f=\"4\"\r\n"
    "Content-Digest: 123\r\n"
    "Signature: xyz\r\n"
    "X-Forwarded-For: 203.0.113.195\r\n"
    "\r\n");
  CHECK(r.head.meta.body == framing::chunked);
  CHECK(r.head.meta.version == 11);
  CHECK(r.head.meta.keep_alive);
  CHECK(r.body == "*****--");
  CHECK(r.chunked.status == parse_status::ok);

  r = grind("HTTP/1.0 200 OK\r\nServer: test\r\nContent-Length: 5\r\n\r\n*****");
  CHECK(r.body == "*****");
  CHECK_FALSE(r.head.meta.keep_alive);  // HTTP/1.0 without keep-alive

  // skip body (HEAD)
  r = grind("HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\n*****", head_mode());
  CHECK(r.head.done);
  CHECK(r.head.meta.content_length == 5);
  CHECK(r.head.body_offset == 38);  // "*****" is left for the caller
}

TEST_CASE("http.beast.need_more_and_got_some", "[rest][http][beast]")
{
  response_parser p;
  response_meta meta;
  std::error_code ec;
  std::size_t used = 99;
  CHECK(p.parse_header(std::string_view{}, used, meta, ec) == parse_status::need_more);
  CHECK(used == 0);
  CHECK(p.parse_header(std::string_view{"H"}, used, meta, ec) == parse_status::need_more);
  CHECK(used == 0);
  std::string b = "HTTP/1.1 200 OK\r\n";
  CHECK(p.parse_header(b, used, meta, ec) == parse_status::need_more);
  CHECK(used == b.size());
  b = "Server: test\r\n\r\n";
  CHECK(p.parse_header(b, used, meta, ec) == parse_status::ok);
  CHECK(used == b.size());
  CHECK(p.header_done());
}

TEST_CASE("http.beast.trailer_headers", "[rest][http][beast]")
{
  // Framing/connection fields in a trailer must not affect the message (RFC 9112 §7.1.2).
  auto r =
    grind("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\nConnection: close\r\n\r\n");
  CHECK(r.chunked.status == parse_status::ok);
  CHECK(r.head.meta.keep_alive);
  r = grind(
    "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n"
    "Trailer: Connection, Proxy-Connection, Upgrade, Transfer-Encoding, Content-Length, Trailer, "
    "Host\r\n\r\n"
    "0\r\nConnection: close\r\nProxy-Connection: close\r\nUpgrade: websocket\r\n"
    "Transfer-Encoding: gzip\r\nContent-Length: 42\r\nTrailer: Evil\r\nHost: evil.example\r\n\r\n");
  CHECK(r.chunked.status == parse_status::ok);
  CHECK(r.head.meta.keep_alive);
  CHECK(r.head.meta.body == framing::chunked);
  CHECK_FALSE(r.head.meta.content_length.has_value());
  // Malformed trailer lines are still errors.
  failgrind("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\nbad trailer\r\n\r\n",
            pe::bad_field);
}

TEST_CASE("http.beast.header_field_limits", "[rest][http][beast]")
{
  // parser.cpp testHeaderFieldLimits: oversized names/values hit the (16 KiB) header limit.
  std::string const big(65534, 'a');
  auto r = parse_message("HTTP/1.1 200 OK\r\n" + big + ": value\r\n\r\n", {});
  CHECK(r.head.status == parse_status::error);
  CHECK(r.head.detail == pe::header_limit);
  CHECK(r.head.ec == cucascade::io::rest::errc::http_header_too_large);
  r = parse_message("HTTP/1.1 200 OK\r\nname: " + big + "\r\n\r\n", {});
  CHECK(r.head.detail == pe::header_limit);
  // An obs-folded value longer than 4 KiB is a header-limit error (Beast max_obs_fold).
  std::string const half(2100, 'v');
  r = parse_message("HTTP/1.1 204 No Content\r\nf: " + half + "\r\n " + half + "\r\n\r\n", {});
  CHECK(r.head.detail == pe::header_limit);
}
