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

// S3 fixtures end to end: head parse (every split point), body framing, XML <Code> scan,
// classification (gpuio ARCHITECTURE §3.5 and §7.3).

#include "http_s3_fixtures.hpp"
#include "http_test_support.hpp"
#include "io/rest/errc.hpp"

#include <catch2/catch_all.hpp>

#include <string>

using namespace cucascade::io::http;
using namespace http_test;
using rclass = outcome::retry_class;

namespace {

constexpr std::uint64_t object_size = 11441354;  // dev_box_probes §7 object

std::string field_value(head_result const& h, std::string_view name)
{
  for (auto const& [n, v] : h.fields) {
    if (n == name) return v;
  }
  return {};
}

/// Drain an error body the way the driver does (≤ 64 KiB), then scan its <Code>.
message_result parse_error_response(std::string_view msg)
{
  message_result r = grind(msg);
  REQUIRE(r.head.status == parse_status::ok);
  REQUIRE(r.body_complete);
  REQUIRE(r.body.size() <= max_error_body);
  CHECK(extract_error_code(r.body, r.head.meta));
  return r;
}

}  // namespace

TEST_CASE("http.s3.206_partial_content", "[rest][http][s3]")
{
  std::string msg{s3_fixtures::head_206};
  msg += std::string(300, 'B');  // body bytes that arrived with the head
  auto const r = grind(msg);
  REQUIRE(r.head.status == parse_status::ok);
  auto const& m = r.head.meta;
  CHECK(m.status == 206);
  CHECK(m.version == 11);
  CHECK(m.keep_alive);  // no Connection header on HTTP/1.1
  CHECK(m.body == framing::length);
  CHECK(m.content_length == 1048576);
  REQUIRE(m.range.has_value());
  CHECK(m.range->first == 0);
  CHECK(m.range->last == 1048575);
  CHECK(m.range->complete == object_size);
  CHECK_FALSE(m.range->unsatisfied);
  CHECK(m.etag.view() == "\"6b1f5c2b7f1a9e0d3c4b5a6978879a5c\"");
  CHECK(m.request_id.view() == "5RZ8XK1S2V6QH9TB");
  CHECK(m.amz_id_2.view() ==
        "Xr9GhZqE0QnQ1wF2c5Vx3mI1yP7uN0bK4sJ8tL6dA2hR5eC9oW3vB1gT7kM0zY4pS6nD8fH2jU5qE=");
  CHECK(m.date.view() == "Mon, 12 Oct 2026 17:21:03 GMT");
  CHECK_FALSE(m.truncated_fields);
  CHECK(r.head.body_offset == s3_fixtures::head_206.size());
  CHECK(r.head.body_remaining == 1048576);
  // Unknown fields (and Last-Modified / Content-Type) reach the callback, in order.
  REQUIRE(r.head.fields.size() == 7);
  CHECK(r.head.fields[0].first == "Last-Modified");
  CHECK(field_value(r.head, "x-amz-server-side-encryption") == "AES256");
  CHECK(field_value(r.head, "Content-Type") == "binary/octet-stream");
  CHECK(field_value(r.head, "Server") == "AmazonS3");

  auto const o = classify(m, {}, method::get, 0, 1048575, object_size, 0, true);
  CHECK_FALSE(o.ec);
  CHECK(o.reuse_connection);
  CHECK_FALSE(o.retryable);
}

TEST_CASE("http.s3.206_body_into_destination", "[rest][http][s3]")
{
  // gpuio ARCHITECTURE §3.5: header RECV, parse, leftover memcpy, then direct receives +
  // consume_direct.
  std::string recv{s3_fixtures::head_206};
  std::string const leftover(1000, 'L');
  recv += leftover;
  std::string dst_a(600000, '\0');
  std::string dst_b(448576, '\0');
  iovec dst[2] = {{dst_a.data(), dst_a.size()}, {dst_b.data(), dst_b.size()}};
  iovec_sink sink{dst};

  response_parser p;
  response_meta meta;
  std::error_code ec;
  std::size_t consumed = 0;
  REQUIRE(p.parse_header(recv, consumed, meta, ec) == parse_status::ok);
  REQUIRE_FALSE(sink.init(meta.content_length));
  sink.copy_in(std::span<char const>{recv}.subspan(consumed));
  p.consume_direct(recv.size() - consumed);
  CHECK(p.body_remaining() == 1048576 - 1000);
  while (p.body_remaining() != 0) {
    iovec scratch[4];
    auto const w = sink.window(scratch, 256 * 1024);
    REQUIRE_FALSE(w.empty());
    std::size_t got = 0;
    for (auto const& v : w) {  // the "kernel" fills the window
      std::memset(v.iov_base, 'D', v.iov_len);
      got += v.iov_len;
    }
    sink.commit(got);
    p.consume_direct(got);
  }
  CHECK(p.done());
  CHECK(sink.remaining() == 0);
  CHECK(dst_a.substr(0, 1000) == leftover);
  CHECK(dst_a[1000] == 'D');
  CHECK(dst_b.back() == 'D');
}

TEST_CASE("http.s3.200_full_object_and_head", "[rest][http][s3]")
{
  auto r = grind(s3_fixtures::head_200);
  REQUIRE(r.head.status == parse_status::ok);
  CHECK(r.head.meta.status == 200);
  CHECK(r.head.meta.content_length == object_size);
  CHECK(r.head.meta.body == framing::length);
  CHECK_FALSE(r.head.meta.range.has_value());
  CHECK(r.head.meta.etag.view() == "\"6b1f5c2b7f1a9e0d3c4b5a6978879a5c\"");

  // Same head as the answer to HEAD: no body, size from Content-Length.
  parse_options o;
  o.head_request = true;
  r              = grind(s3_fixtures::head_200, o);
  CHECK(r.head.meta.body == framing::none);
  CHECK(r.head.meta.content_length == object_size);
  CHECK(r.head.done);
  CHECK(r.head.body_remaining == 0);
  CHECK(classify(r.head.meta, {}, method::head, 0, 0, 0, 0, false).ec == std::error_code{});
}

TEST_CASE("http.s3.403_access_denied_chunked_xml", "[rest][http][s3]")
{
  auto const r = parse_error_response(s3_fixtures::resp_403);
  CHECK(r.head.meta.status == 403);
  CHECK(r.head.meta.body == framing::chunked);
  CHECK(r.head.meta.request_id.view() == "0B4RH6A1QX7C2M8D");
  CHECK(r.head.meta.amz_error_code.view() == "AccessDenied");
  auto const o = classify(r.head.meta, {}, method::get, 0, 99, object_size, 0, true);
  CHECK(o.ec == cucascade::io::rest::make_http_error(403));
  CHECK_FALSE(o.retryable);
  CHECK(o.reuse_connection);  // drained chunked body, keep-alive
}

TEST_CASE("http.s3.403_request_time_too_skewed", "[rest][http][s3]")
{
  auto const r = parse_error_response(s3_fixtures::resp_403_skewed);
  CHECK(r.head.meta.amz_error_code.view() == "RequestTimeTooSkewed");
  CHECK(r.head.meta.date.view() == "Mon, 12 Oct 2026 17:21:05 GMT");  // clock offset source
  auto const o = classify(r.head.meta, {}, method::get, 0, 99, object_size, 0, true);
  CHECK(o.retryable);
  CHECK(o.cls == rclass::auth);
}

TEST_CASE("http.s3.503_slowdown_retry_after", "[rest][http][s3]")
{
  auto const r = parse_error_response(s3_fixtures::resp_503_slowdown);
  CHECK(r.head.meta.status == 503);
  CHECK(r.head.meta.retry_after_ms == 2000);
  CHECK(r.head.meta.amz_error_code.view() == "SlowDown");
  auto const o = classify(r.head.meta, {}, method::get, 0, 99, object_size, 0, true);
  CHECK(o.retryable);
  CHECK(o.cls == rclass::throttle);
  CHECK(o.ec == cucascade::io::rest::make_http_error(503));
}

TEST_CASE("http.s3.412_precondition_failed", "[rest][http][s3]")
{
  auto const r = parse_error_response(s3_fixtures::resp_412);
  CHECK(r.head.meta.amz_error_code.view() == "PreconditionFailed");
  auto const o = classify(r.head.meta, {}, method::get, 0, 99, object_size, 0, true);
  CHECK(o.ec == cucascade::io::rest::errc::object_changed);
  CHECK_FALSE(o.retryable);
}

TEST_CASE("http.s3.416_range_not_satisfiable", "[rest][http][s3]")
{
  auto const r = parse_error_response(s3_fixtures::resp_416);
  REQUIRE(r.head.meta.range.has_value());
  CHECK(r.head.meta.range->unsatisfied);
  CHECK(r.head.meta.range->complete == object_size);
  CHECK(r.head.meta.amz_error_code.view() == "InvalidRange");
  auto const o =
    classify(r.head.meta, {}, method::get, object_size, object_size + 46, object_size, 0, true);
  CHECK(o.ec == cucascade::io::rest::make_http_error(416));
  CHECK_FALSE(o.retryable);
}

TEST_CASE("http.s3.404_no_such_key", "[rest][http][s3]")
{
  auto const r = parse_error_response(s3_fixtures::resp_404);
  CHECK(r.head.meta.amz_error_code.view() == "NoSuchKey");
  CHECK(r.head.meta.body == framing::chunked);
  auto const o = classify(r.head.meta, {}, method::get, 0, 99, object_size, 0, true);
  CHECK(o.ec == cucascade::io::rest::make_http_error(404));
  CHECK_FALSE(o.retryable);
}

TEST_CASE("http.s3.interim_responses_skipped", "[rest][http][s3]")
{
  std::string msg = "HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 103 Early Hints\r\nLink: </a>\r\n\r\n";
  msg += s3_fixtures::head_206;
  auto const r = grind(msg);
  REQUIRE(r.head.status == parse_status::ok);
  CHECK(r.head.interim == 2);
  CHECK(r.head.meta.status == 206);
  CHECK(r.head.body_offset == msg.size());
  CHECK(r.head.fields.front() == std::pair<std::string, std::string>{"Link", "</a>"});

  // The interim heads count against the exchange's header limit.
  parse_options o;
  o.header_limit = static_cast<std::uint32_t>(msg.size() - 1);
  failgrind(msg, parse_error::header_limit, o);
  o.header_limit = static_cast<std::uint32_t>(msg.size());
  CHECK(grind(msg, o).head.status == parse_status::ok);
}

TEST_CASE("http.s3.connection_close_and_truncation", "[rest][http][s3]")
{
  std::string msg{s3_fixtures::head_206};
  msg.insert(msg.size() - 2, "Connection: close\r\n");
  auto const r = grind(msg);
  CHECK_FALSE(r.head.meta.keep_alive);
  auto const o = classify(r.head.meta, {}, method::get, 0, 1048575, object_size, 0, true);
  CHECK_FALSE(o.ec);
  CHECK_FALSE(o.reuse_connection);

  // Captured values longer than their inline storage are truncated and flagged.
  std::string const long_etag = "\"" + std::string(100, 'e') + "\"";
  auto const t                = grind("HTTP/1.1 204 No Content\r\nETag: " + long_etag + "\r\n\r\n");
  CHECK(t.head.meta.etag.truncated);
  CHECK(t.head.meta.truncated_fields);
  CHECK(t.head.meta.etag.view() == std::string_view{long_etag}.substr(0, 80));
}

TEST_CASE("http.s3.retry_after_forms", "[rest][http][s3]")
{
  auto ms = [](std::string const& fields) {
    return grind("HTTP/1.1 503 Slow Down\r\n" + fields + "Content-Length: 0\r\n\r\n")
      .head.meta.retry_after_ms;
  };
  CHECK(ms("Retry-After: 3\r\n") == 3000);
  CHECK(ms("Retry-After: 120\r\n") == 30000);                        // capped at 30 s
  CHECK(ms("Retry-After: Wed, 21 Oct 2026 07:28:00 GMT\r\n") == 0);  // HTTP-date ignored
  CHECK(ms("x-amz-retry-after: 750\r\n") == 750);
  CHECK(ms("Retry-After: 3\r\nx-amz-retry-after: 750\r\n") == 750);  // ms form wins
  CHECK(ms("x-amz-retry-after: 750\r\nRetry-After: 3\r\n") == 750);
  CHECK(ms("x-amz-retry-after: 99999999\r\n") == 30000);
}

TEST_CASE("http.s3.extract_error_code", "[rest][http][s3]")
{
  response_meta m;
  CHECK(extract_error_code("<Error><Code> InternalError\n</Code></Error>", m));
  CHECK(m.amz_error_code.view() == "InternalError");
  response_meta n;
  CHECK_FALSE(extract_error_code("<Error><Message>x</Message></Error>", n));
  CHECK_FALSE(extract_error_code("<Error><Code>Unterminated", n));
  CHECK_FALSE(extract_error_code("<Code></Code>", n));
  CHECK(n.amz_error_code.empty());
  // Only the first 64 KiB are scanned.
  std::string big(max_error_body, ' ');
  big += "<Code>Late</Code>";
  CHECK_FALSE(extract_error_code(big, n));
}
