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

// One test per row of the gpuio ARCHITECTURE §7.3 table (rows numbered top to bottom).
// Rows 12 (hedge on TTFB) and 14 (retry budget) are timing/budget policy outside the pure
// classifier; their inputs (retry class, Retry-After) are covered by rows 2-3.

#include "io/rest/errc.hpp"
#include "io/rest/http/classify.hpp"

#include <catch2/catch_all.hpp>

#include <cerrno>

using namespace cucascade::io::http;
using rclass = outcome::retry_class;

namespace {

constexpr std::uint64_t size = 10'000'000;

response_meta head(std::uint16_t status, std::string_view code = {})
{
  response_meta m;
  m.status         = status;
  m.body           = framing::length;
  m.content_length = 200;
  if (!code.empty()) m.amz_error_code.assign(code);
  return m;
}

response_meta ok206(std::uint64_t first, std::uint64_t last, std::uint64_t total)
{
  response_meta m;
  m.status         = 206;
  m.body           = framing::length;
  m.content_length = last - first + 1;
  m.range          = content_range{first, last, total, false};
  return m;
}

outcome get(response_meta const& m,
            std::error_code ec        = {},
            std::uint64_t first       = 0,
            std::uint64_t last        = 1023,
            std::uint64_t received    = 0,
            bool reused               = true,
            std::uint64_t object_size = size)
{
  return classify(m, ec, method::get, first, last, object_size, received, reused);
}

std::error_code sys(int e) { return {e, std::system_category()}; }

}  // namespace

TEST_CASE("http.classify.row01_reused_connection_dead_before_any_byte", "[rest][http][classify]")
{
  response_meta const none;  // no status line arrived
  for (auto ec :
       {cucascade::io::rest::make_error_code(cucascade::io::rest::errc::stale_connection),
        sys(ECONNRESET),
        sys(EPIPE),
        cucascade::io::rest::make_error_code(cucascade::io::rest::errc::http_length_mismatch)}) {
    auto const o = get(none, ec);
    CHECK(o.retryable);
    CHECK(o.cls == rclass::free_stale);
    CHECK(o.ec == cucascade::io::rest::errc::stale_connection);
    CHECK_FALSE(o.reuse_connection);
  }
  // A fresh connection failing the same way is an ordinary transient failure.
  CHECK(get(none, sys(ECONNRESET), 0, 1023, 0, false).cls == rclass::transient);
  // So is a timeout on a reused connection (not a drop).
  CHECK(get(none, cucascade::io::rest::make_error_code(cucascade::io::rest::errc::request_timeout))
          .cls == rclass::transient);
}

TEST_CASE("http.classify.row02_transient_5xx_internal_error_request_timeout",
          "[rest][http][classify]")
{
  for (int s : {500, 502, 504}) {
    auto const st = static_cast<std::uint16_t>(s);
    auto const o  = get(head(st));
    CHECK(o.retryable);
    CHECK(o.cls == rclass::transient);
    CHECK(o.ec == cucascade::io::rest::make_http_error(st));
    CHECK_FALSE(o.reuse_connection);  // retry on a new connection, different IP
  }
  CHECK(get(head(500, "InternalError")).cls == rclass::transient);
  CHECK(get(head(400, "RequestTimeout")).cls == rclass::transient);
  CHECK(get(head(400)).cls == rclass::none);  // other 400s are terminal
  CHECK_FALSE(get(head(501)).retryable);      // NotImplemented
}

TEST_CASE("http.classify.row02_transient_connect_and_tls_failures", "[rest][http][classify]")
{
  response_meta const none;
  for (auto e : {cucascade::io::rest::errc::dns_failed,
                 cucascade::io::rest::errc::connect_failed,
                 cucascade::io::rest::errc::connect_timeout,
                 cucascade::io::rest::errc::tls_handshake_failed,
                 cucascade::io::rest::errc::tls_protocol_error}) {
    auto const o = get(none, cucascade::io::rest::make_error_code(e), 0, 1023, 0, false);
    CHECK(o.retryable);
    CHECK(o.cls == rclass::transient);
    CHECK(o.ec == e);
    CHECK_FALSE(o.reuse_connection);
  }
  // Local/deterministic failures are not retried.
  for (auto e : {cucascade::io::rest::errc::tls_verify_failed,
                 cucascade::io::rest::errc::ktls_unavailable,
                 cucascade::io::rest::errc::http_bad_message,
                 cucascade::io::rest::errc::http_header_too_large,
                 cucascade::io::rest::errc::invalid_argument}) {
    CHECK_FALSE(get(none, cucascade::io::rest::make_error_code(e), 0, 1023, 0, false).retryable);
  }
  // Cancellation is never retried.
  CHECK_FALSE(
    get(ok206(0, 1023, size),
        cucascade::io::rest::make_error_code(cucascade::io::rest::errc::operation_canceled))
      .retryable);
  CHECK_FALSE(get(none, sys(ECANCELED)).retryable);
}

TEST_CASE("http.classify.row03_throttle_503_slowdown_429", "[rest][http][classify]")
{
  auto m           = head(503, "SlowDown");
  m.retry_after_ms = 1500;
  auto o           = get(m);
  CHECK(o.retryable);
  CHECK(o.cls == rclass::throttle);
  CHECK(o.ec == cucascade::io::rest::make_http_error(503));
  CHECK(o.reuse_connection);  // drained small body, keep-alive
  CHECK(get(head(429)).cls == rclass::throttle);
  CHECK(get(head(503)).cls == rclass::throttle);  // 503 without a body code (HEAD)
  // SlowDown code wins whatever the status.
  CHECK(get(head(400, "SlowDown")).cls == rclass::throttle);
  // A 503 whose error-body drain failed: still throttle, but the connection is closed.
  o = get(head(503, "SlowDown"), sys(ECONNRESET));
  CHECK(o.cls == rclass::throttle);
  CHECK_FALSE(o.reuse_connection);
}

TEST_CASE("http.classify.row04_mid_body_failure_resumes", "[rest][http][classify]")
{
  auto const m = ok206(0, 4'194'303, size);
  auto o       = get(m, sys(ECONNRESET), 0, 4'194'303, 1'000'000);
  CHECK(o.retryable);
  CHECK(o.cls == rclass::transient);
  CHECK(o.resume_allowed);
  CHECK_FALSE(o.reuse_connection);
  // EOF before Content-Length reported as a length mismatch: same treatment.
  o = get(m,
          cucascade::io::rest::make_error_code(cucascade::io::rest::errc::http_length_mismatch),
          0,
          4'194'303,
          4096);
  CHECK(o.resume_allowed);
  // Nothing received yet: retry the whole range.
  CHECK_FALSE(get(m, sys(ECONNRESET), 0, 4'194'303, 0, false).resume_allowed);
  // Not a GET: no resume.
  CHECK_FALSE(classify(head(200), sys(ECONNRESET), method::put, 0, 0, 0, 10, false).resume_allowed);
}

TEST_CASE("http.classify.row05_200_to_ranged_get", "[rest][http][classify]")
{
  response_meta m;
  m.status         = 200;
  m.body           = framing::length;
  m.content_length = 600 * 1024;
  // object ≤ 1 MiB and ≤ 2x the request: accept and slice
  auto o = get(m, {}, 100 * 1024, 400 * 1024 - 1, 0, true, 600 * 1024);
  CHECK_FALSE(o.ec);
  CHECK(o.reuse_connection);
  // object > 2x the request: fail
  o = get(m, {}, 0, 100 * 1024 - 1, 0, true, 600 * 1024);
  CHECK(o.ec == cucascade::io::rest::errc::range_ignored);
  CHECK_FALSE(o.retryable);
  CHECK_FALSE(o.reuse_connection);
  // object > 1 MiB: fail
  m.content_length = 3 * 1024 * 1024;
  CHECK(get(m, {}, 0, 2 * 1024 * 1024, 0, true, 3 * 1024 * 1024).ec ==
        cucascade::io::rest::errc::range_ignored);
  // the request covered the whole object: a 200 is exactly right
  CHECK_FALSE(get(m, {}, 0, 3 * 1024 * 1024 - 1, 0, true, 3 * 1024 * 1024).ec);
  // the object size changed underneath us
  CHECK(get(m, {}, 0, 3 * 1024 * 1024 - 1, 0, true, 5 * 1024 * 1024).ec ==
        cucascade::io::rest::errc::content_range_mismatch);
}

TEST_CASE("http.classify.row06_206_wrong_content_range", "[rest][http][classify]")
{
  CHECK_FALSE(get(ok206(1024, 2047, size), {}, 1024, 2047).ec);
  // the last range is clipped at the object end
  CHECK_FALSE(get(ok206(size - 10, size - 1, size), {}, size - 10, size + 100).ec);
  auto const bad = [](response_meta const& m, std::uint64_t f, std::uint64_t l) {
    auto const o = get(m, {}, f, l);
    CHECK(o.ec == cucascade::io::rest::errc::content_range_mismatch);
    CHECK_FALSE(o.retryable);
    CHECK_FALSE(o.reuse_connection);
  };
  bad(ok206(0, 2047, size), 1024, 2047);               // wrong start
  bad(ok206(1024, 1535, size), 1024, 2047);            // wrong end
  bad(ok206(1024, 2047, size + 1), 1024, 2047);        // total changed: object changed
  bad(ok206(1024, 2047, unknown_length), 1024, 2047);  // "/*" against a known size
  response_meta missing = ok206(1024, 2047, size);
  missing.range.reset();
  bad(missing, 1024, 2047);
  // Content-Length disagreeing with Content-Range
  auto cl           = ok206(1024, 2047, size);
  cl.content_length = 999;
  CHECK(get(cl, {}, 1024, 2047).ec == cucascade::io::rest::errc::http_length_mismatch);
  // unknown object size: checked against the request only
  CHECK_FALSE(get(ok206(0, 99, 5000), {}, 0, 1023, 0, true, 0).ec);
  CHECK(get(ok206(0, 2000, 5000), {}, 0, 1023, 0, true, 0).ec ==
        cucascade::io::rest::errc::content_range_mismatch);
}

TEST_CASE("http.classify.row07_412_object_changed", "[rest][http][classify]")
{
  auto const o = get(head(412, "PreconditionFailed"));
  CHECK(o.ec == cucascade::io::rest::errc::object_changed);
  CHECK_FALSE(o.retryable);
}

TEST_CASE("http.classify.row08_416_not_retriable", "[rest][http][classify]")
{
  auto m       = head(416, "InvalidRange");
  m.range      = content_range{0, 0, size, true};
  auto const o = get(m);
  CHECK(o.ec == cucascade::io::rest::make_http_error(416));
  CHECK_FALSE(o.retryable);
}

TEST_CASE("http.classify.row09_403_bounded_resign", "[rest][http][classify]")
{
  for (std::string_view code : {"SignatureDoesNotMatch", "RequestTimeTooSkewed", ""}) {
    auto const o = get(head(403, code));
    CHECK(o.retryable);
    CHECK(o.cls == rclass::auth);
    CHECK(o.ec == cucascade::io::rest::make_http_error(403));
  }
  CHECK(get(head(400, "ExpiredToken")).cls == rclass::auth);
}

TEST_CASE("http.classify.row10_404_nosuchkey_accessdenied_fail_fast", "[rest][http][classify]")
{
  for (auto const& m : {head(404, "NoSuchKey"), head(404), head(403, "AccessDenied")}) {
    auto const o = get(m);
    CHECK_FALSE(o.retryable);
    CHECK(o.cls == rclass::none);
    CHECK(o.ec == cucascade::io::rest::make_http_error(m.status));
  }
}

TEST_CASE("http.classify.row11_connection_close", "[rest][http][classify]")
{
  auto m       = ok206(0, 1023, size);
  m.keep_alive = false;
  auto o       = get(m);
  CHECK_FALSE(o.ec);
  CHECK_FALSE(o.reuse_connection);
  m.keep_alive = true;
  CHECK(get(m).reuse_connection);
  // Error bodies too large to drain (> 64 KiB) or EOF-delimited close the connection.
  auto big           = head(404, "NoSuchKey");
  big.content_length = max_error_body + 1;
  CHECK_FALSE(get(big).reuse_connection);
  auto eof = head(404);
  eof.body = framing::until_eof;
  CHECK_FALSE(get(eof).reuse_connection);
  // Non-GET success follows keep-alive.
  response_meta put_ok;
  put_ok.status     = 200;
  put_ok.keep_alive = false;
  CHECK_FALSE(classify(put_ok, {}, method::put, 0, 0, 0, 0, true).reuse_connection);
}

TEST_CASE("http.classify.row13_stall_timeout_resumes", "[rest][http][classify]")
{
  auto const o =
    get(ok206(0, 4'194'303, size),
        cucascade::io::rest::make_error_code(cucascade::io::rest::errc::request_timeout),
        0,
        4'194'303,
        65536);
  CHECK(o.retryable);
  CHECK(o.cls == rclass::transient);
  CHECK(o.resume_allowed);
}

TEST_CASE("http.classify.unexpected_statuses", "[rest][http][classify]")
{
  CHECK(get(head(204)).ec ==
        cucascade::io::rest::errc::http_bad_message);  // 2xx other than 200/206 to GET
  CHECK(get(head(101)).ec == cucascade::io::rest::errc::http_bad_message);
  CHECK(get(head(301)).ec ==
        cucascade::io::rest::make_http_error(301));  // region redirect: S3 layer
  CHECK_FALSE(get(head(301)).retryable);
  CHECK(classify(head(204), {}, method::delete_, 0, 0, 0, 0, true).ec == std::error_code{});
}
