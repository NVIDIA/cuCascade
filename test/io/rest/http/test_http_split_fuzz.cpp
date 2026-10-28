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

// Split-point fuzzing: every message is fed to the response parser and its chunked body to the
// decoder at every byte boundary, byte by byte, and under random multi-cut plans; results must be
// identical to a one-piece parse. (Every Beast vector in beast_vectors_test.cpp also goes
// through the same grind.)

#include "http_s3_fixtures.hpp"
#include "http_test_support.hpp"

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

using namespace cucascade::io::http;
using namespace http_test;

namespace {

std::vector<std::string> corpus()
{
  std::vector<std::string> c = {
    std::string{s3_fixtures::head_206} + std::string(64, 'b'),
    std::string{s3_fixtures::head_200},
    std::string{s3_fixtures::resp_403},
    std::string{s3_fixtures::resp_403_skewed},
    std::string{s3_fixtures::resp_503_slowdown},
    std::string{s3_fixtures::resp_412},
    std::string{s3_fixtures::resp_416},
    std::string{s3_fixtures::resp_404},
    "HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabc",
    // One chunked response, concatenated over three lines on purpose.
    // NOLINTNEXTLINE(bugprone-suspicious-missing-comma): deliberate concatenation
    "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
    "1a;name=\"quoted \\\" value\"\r\nabcdefghijklmnopqrstuvwxyz\r\n"
    "3\r\nXYZ\r\n0;last\r\nx-amz-checksum-crc32: AAAAAA==\r\n\r\n",
    "HTTP/1.1 200 OK\r\nX-Fold: a\r\n  b\r\n\tc\r\nContent-Length: 0\r\n\r\n",
  };
  return c;
}

}  // namespace

TEST_CASE("http.split_fuzz.every_boundary", "[rest][http][fuzz]")
{
  for (auto const& msg : corpus()) {
    auto const r = grind(msg);
    CHECK(r.head.status == parse_status::ok);
    // head-only fixtures (206/200) carry a Content-Length larger than the bytes present
    CHECK((r.body_complete || r.head.meta.body == framing::length));
  }
}

TEST_CASE("http.split_fuzz.chunked_bodies_every_boundary", "[rest][http][fuzz]")
{
  for (std::string_view body :
       {"0\r\n\r\n"sv,
        "5\r\nhello\r\n0\r\n\r\n"sv,
        "5;a=b;c\r\nhello\r\n6 ; x = \"y\"\r\n world\r\n0\r\nT: v\r\n\r\n"sv,
        "000a\r\n0123456789\r\n0\r\n\r\nNEXT-MESSAGE"sv}) {
    auto const r = grind_chunked(body);
    CHECK(r.status == parse_status::ok);
  }
  auto const r = grind_chunked("000a\r\n0123456789\r\n0\r\n\r\nNEXT-MESSAGE");
  CHECK(r.consumed == 23);  // stops exactly at the end of the chunked body
  CHECK(r.body == "0123456789");
}

TEST_CASE("http.split_fuzz.random_multi_cut", "[rest][http][fuzz]")
{
  std::mt19937_64 rng{0x5eed};
  for (auto const& msg : corpus()) {
    auto const whole = parse_message(msg, {});
    for (int round = 0; round < 200; ++round) {
      std::vector<std::size_t> cuts;
      std::uniform_int_distribution<std::size_t> pos{1, msg.size() - 1};
      std::uniform_int_distribution<int> count{1, 12};
      for (int k = count(rng); k > 0; --k)
        cuts.push_back(pos(rng));
      std::sort(cuts.begin(), cuts.end());
      auto const r = parse_message(msg, cuts);
      CHECK(r.head.status == whole.head.status);
      CHECK(r.head.body_offset == whole.head.body_offset);
      CHECK(r.head.fields == whole.head.fields);
      CHECK(r.body == whole.body);
      CHECK(r.chunked.consumed == whole.chunked.consumed);
    }
  }
}

TEST_CASE("http.split_fuzz.mutations_never_crash", "[rest][http][fuzz]")
{
  // Byte mutations of valid messages (Beast testFuzz idea): parse must terminate with a defined
  // status, consume no more than offered, and stay within the header limit.
  std::mt19937_64 rng{42};
  for (auto const& base : corpus()) {
    for (int round = 0; round < 300; ++round) {
      std::string msg = base;
      std::uniform_int_distribution<std::size_t> pos{0, msg.size() - 1};
      std::uniform_int_distribution<int> byte{0, 255};
      for (int k = 0; k < 4; ++k)
        msg[pos(rng)] = static_cast<char>(byte(rng));
      auto const r = parse_message(msg, {});
      CHECK(r.head.body_offset <= msg.size());
      if (r.head.status == parse_status::error) CHECK(r.head.ec);
    }
  }
}
