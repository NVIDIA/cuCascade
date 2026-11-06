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

// Retry classification: gpuio ARCHITECTURE §7.3 (rows cited inline), sourced from
// gpuio docs/research/s3_line_rate.md §2.8 and §2.1 "503 / retry guidance".

#include "io/rest/http/classify.hpp"

#include "io/rest/errc.hpp"
#include "io/rest/http/rules.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <string_view>

namespace cucascade::io::http {

using rest::errc;
using rest::error_category;
using rest::make_http_error;
namespace {

using rclass = outcome::retry_class;

bool is_rest_errc(std::error_code ec, errc e) noexcept { return ec == make_error_code(e); }

bool is_errno(std::error_code ec, int e) noexcept
{
  return (ec.category() == std::system_category() || ec.category() == std::generic_category()) &&
         ec.value() == e;
}

bool is_os_error(std::error_code ec) noexcept
{
  return ec.category() == std::system_category() || ec.category() == std::generic_category();
}

/// Cancellation is never retried here (the request is being torn down).
bool is_cancel(std::error_code ec) noexcept
{
  return is_rest_errc(ec, errc::operation_canceled) || is_rest_errc(ec, errc::context_stopped) ||
         is_errno(ec, ECANCELED);
}

/// The peer dropped the connection (EOF/RST/EPIPE): a reused connection that dies like this before
/// any response byte is the "stale pooled connection" case (curl transfer.c:600-660).
bool is_connection_drop(std::error_code ec) noexcept
{
  return is_rest_errc(ec, errc::stale_connection) || is_rest_errc(ec, errc::http_length_mismatch) ||
         is_rest_errc(ec, errc::io_error) || is_errno(ec, ECONNRESET) || is_errno(ec, EPIPE) ||
         is_errno(ec, ECONNABORTED) || is_errno(ec, ENOTCONN) || is_errno(ec, ESHUTDOWN);
}

/// Transport failures worth a transient retry (§7.3 row 2: "connect/TLS failure, mid-body
/// reset"; row "no bytes for stall_timeout": cancel + resume). Every other rest::errc error is
/// local or deterministic (configuration, codec rejection, verification) and fails the request.
bool is_transient_transport(std::error_code ec) noexcept
{
  if (is_os_error(ec)) return true;
  if (ec.category() != error_category()) return false;
  switch (static_cast<errc>(ec.value())) {
    case errc::dns_failed:
    case errc::connect_failed:
    case errc::connect_timeout:
    case errc::tls_handshake_failed:
    case errc::tls_protocol_error:
    case errc::tls_bad_record_mac:
    case errc::stale_connection:
    case errc::http_length_mismatch:
    case errc::request_timeout:
    case errc::io_error: return true;
    default: return false;
  }
}

outcome make(std::error_code ec, rclass cls, bool reuse) noexcept
{
  outcome o;
  o.ec               = ec;
  o.cls              = cls;
  o.retryable        = cls != rclass::none;
  o.reuse_connection = reuse;
  return o;
}

/// An error body can be drained (≤ 64 KiB) to keep the connection; otherwise it is closed
/// (§7.3, "Non-2xx GET responses").
bool drainable(response_meta const& meta) noexcept
{
  switch (meta.body) {
    case framing::none: return true;
    case framing::length: return meta.content_length.value_or(0) <= max_error_body;
    case framing::chunked: return true;  // the driver's chunked_decoder enforces the cap
    case framing::until_eof: return false;
  }
  return false;
}

/// Non-2xx responses (and 2xx outside GET semantics). `alive` is false after a transport error.
outcome classify_status(response_meta const& meta, bool alive) noexcept
{
  std::uint16_t const st      = meta.status;
  std::string_view const code = meta.amz_error_code.view();
  auto const http_ec          = make_http_error(st);
  bool const keep             = alive && meta.keep_alive && drainable(meta);

  // S3 error codes override the status class (aws-c-s3 s3_util.c:795-810).
  if (code == "SlowDown") return make(http_ec, rclass::throttle, keep);
  if (code == "InternalError" || code == "RequestTimeout") {
    return make(http_ec, rclass::transient, false);
  }
  if (code == "RequestTimeTooSkewed") return make(http_ec, rclass::auth, keep);

  switch (st) {
    case 400:
      // ExpiredToken / TokenRefreshRequired: credentials rotated; re-sign with fresh credentials.
      if (code == "ExpiredToken" || code == "TokenRefreshRequired") {
        return make(http_ec, rclass::auth, keep);
      }
      return make(http_ec, rclass::none, keep);
    case 403:
      // §7.3: bounded re-sign (3) unless the answer is a definite AccessDenied.
      if (code == "AccessDenied") return make(http_ec, rclass::none, keep);
      return make(http_ec, rclass::auth, keep);
    case 404: return make(http_ec, rclass::none, keep);  // NoSuchKey / NoSuchBucket
    case 412: return make(make_error_code(errc::object_changed), rclass::none, keep);
    case 416: return make(http_ec, rclass::none, keep);  // caller may re-HEAD a suffix probe once
    case 429:
    case 503: return make(http_ec, rclass::throttle, keep);  // SlowDown (aws-c-s3 maps 503 here)
    case 500:
    case 502:
    case 504: return make(http_ec, rclass::transient, false);  // new connection, different IP
    default: break;
  }
  if (st >= 500 && st != 501 && st != 505) return make(http_ec, rclass::transient, false);
  if (st >= 300) return make(http_ec, rclass::none, keep);
  // 0 (no status) / 1xx final (101) / 2xx with an unexpected meaning.
  return make(make_error_code(errc::http_bad_message), rclass::none, false);
}

/// 200/206 to a ranged GET (§7.3 rows "200 to a ranged GET" and "206 with wrong Content-Range").
outcome classify_get_success(response_meta const& meta,
                             std::uint64_t first,
                             std::uint64_t last,
                             std::uint64_t object_size) noexcept
{
  auto const fail = [](errc e) { return make(make_error_code(e), rclass::none, false); };
  if (meta.status == 206) {
    if (!meta.range || meta.range->unsatisfied) return fail(errc::content_range_mismatch);
    content_range const& r = *meta.range;
    if (r.first != first) return fail(errc::content_range_mismatch);
    if (object_size != 0) {
      // A changed total means the object changed; the end is clipped at the object end.
      if (r.complete != object_size) return fail(errc::content_range_mismatch);
      if (r.last != std::min(last, object_size - 1)) return fail(errc::content_range_mismatch);
    } else if (r.last > last) {
      return fail(errc::content_range_mismatch);
    }
    if (meta.content_length && *meta.content_length != r.last - r.first + 1) {
      return fail(errc::http_length_mismatch);
    }
    return make({}, rclass::none, meta.keep_alive);
  }
  if (meta.status == 200) {
    // "Server ignored Range" (S3-compatible stores): the body is the whole object.
    if (!meta.content_length) return fail(errc::range_ignored);
    std::uint64_t const total = *meta.content_length;
    if (object_size != 0 && total != object_size) return fail(errc::content_range_mismatch);
    std::uint64_t const want = last - first == unknown_length ? unknown_length : last - first + 1;
    bool const whole         = first == 0 && total <= want;  // the request covered everything
    bool const sliceable =
      total <= max_sliced_object && (want >= unknown_length / 2 || total <= 2 * want);
    if (whole || sliceable) return make({}, rclass::none, meta.keep_alive);
    return fail(errc::range_ignored);
  }
  return make(make_error_code(errc::http_bad_message), rclass::none, false);
}

}  // namespace

outcome classify(response_meta const& meta,
                 std::error_code transport_error,
                 method m,
                 std::uint64_t requested_first,
                 std::uint64_t requested_last,
                 std::uint64_t object_size,
                 std::uint64_t bytes_received,
                 bool connection_was_reused) noexcept
{
  bool const is_success = meta.status >= 200 && meta.status < 300;

  if (transport_error) {
    if (is_cancel(transport_error)) return make(transport_error, rclass::none, false);
    // A complete error status explains the failure better than the broken drain that followed.
    if (meta.status >= 300) return classify_status(meta, false);
    if (!is_transient_transport(transport_error)) return make(transport_error, rclass::none, false);
    // §7.3 row 1: a reused connection that died before any response byte is retried at once on
    // a fresh connection without backoff or budget (≤ 5, counted by the caller).
    if (connection_was_reused && meta.status == 0 && bytes_received == 0 &&
        is_connection_drop(transport_error)) {
      return make(make_error_code(errc::stale_connection), rclass::free_stale, false);
    }
    // §7.3 rows 2 and 4: transient; a GET that already delivered r body bytes resumes at off + r
    // with If-Match instead of re-reading the range.
    outcome o = make(transport_error, rclass::transient, false);
    o.resume_allowed =
      m == method::get && bytes_received > 0 && (meta.status == 206 || meta.status == 200);
    return o;
  }

  if (!is_success) return classify_status(meta, true);
  if (m == method::get) {
    return classify_get_success(meta, requested_first, requested_last, object_size);
  }
  // HEAD/PUT/POST/DELETE: any 2xx completes the exchange; `Connection: close` (or HTTP/1.0
  // without keep-alive) retires the connection (§7.3 row "Connection: close").
  return make({}, rclass::none, meta.keep_alive);
}

bool extract_error_code(std::string_view body, response_meta& meta) noexcept
{
  body                             = body.substr(0, std::min(body.size(), max_error_body));
  constexpr std::string_view open  = "<Code>";
  constexpr std::string_view close = "</Code>";
  std::size_t const a              = body.find(open);
  if (a == std::string_view::npos) return false;
  std::size_t const from = a + open.size();
  std::size_t const b    = body.find(close, from);
  if (b == std::string_view::npos) return false;
  std::string_view code = body.substr(from, b - from);
  while (!code.empty() &&
         (detail::is_ws(code.front()) || code.front() == '\r' || code.front() == '\n')) {
    code.remove_prefix(1);
  }
  while (!code.empty() &&
         (detail::is_ws(code.back()) || code.back() == '\r' || code.back() == '\n')) {
    code.remove_suffix(1);
  }
  if (code.empty()) return false;
  if (!meta.amz_error_code.assign(code)) meta.truncated_fields = true;
  return true;
}

}  // namespace cucascade::io::http
