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

#include "io/rest/errc.hpp"
#include "io/rest/http/content_range.hpp"

#include <catch2/catch_all.hpp>

using namespace cucascade::io::http;

TEST_CASE("http.content_range.valid_forms", "[rest][http][range]")
{
  content_range r;
  REQUIRE_FALSE(parse_content_range("bytes 0-1048575/11441354", r));  // S3 206
  CHECK(r.first == 0);
  CHECK(r.last == 1048575);
  CHECK(r.complete == 11441354);
  CHECK_FALSE(r.unsatisfied);

  REQUIRE_FALSE(parse_content_range("bytes */11441354", r));  // S3 416
  CHECK(r.unsatisfied);
  CHECK(r.complete == 11441354);
  CHECK(r.first == 0);
  CHECK(r.last == 0);

  REQUIRE_FALSE(parse_content_range("bytes 5-9/*", r));  // unknown total
  CHECK(r.first == 5);
  CHECK(r.last == 9);
  CHECK(r.complete == unknown_length);
  CHECK_FALSE(r.unsatisfied);

  REQUIRE_FALSE(parse_content_range("  BYTES   7-7/8 \t", r));
  CHECK(r.first == 7);
  CHECK(r.complete == 8);
  REQUIRE_FALSE(parse_content_range("bytes */0", r));  // empty object
  CHECK(r.complete == 0);
  REQUIRE_FALSE(parse_content_range("bytes 0-18446744073709551614/18446744073709551615", r));
  CHECK(r.last == 18446744073709551614ULL);
}

TEST_CASE("http.content_range.invalid_forms", "[rest][http][range]")
{
  for (char const* v : {"",
                        "bytes",
                        "bytes ",
                        "bytes=0-1/2",
                        "items 0-1/2",
                        "bytes 0-1",
                        "bytes 0-1/",
                        "bytes -1/2",
                        "bytes 0-/2",
                        "bytes 1-0/2",  // first > last
                        "bytes 0-2/2",  // last >= complete
                        "bytes */*",
                        "bytes */",
                        "bytes 0-1/2x",  // trailing garbage
                        "bytes 0-1/2, 3-4/5",
                        "bytes +0-1/2",
                        "bytes 0-18446744073709551616/2",  // overflow
                        "bytes\t0-1/2"}) {
    content_range r{1, 2, 3, false};
    INFO(v);
    CHECK(parse_content_range(v, r) == cucascade::io::rest::errc::http_bad_message);
    CHECK(r.first == 1);  // untouched on failure
  }
}
