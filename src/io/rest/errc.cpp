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

// Error categories of the REST transport.  Messages are built only when formatted;
// constructing an error_code never allocates.

#include "io/rest/errc.hpp"

#include <string>

namespace cucascade::io::rest {
namespace {

class rest_error_category final : public std::error_category {
 public:
  [[nodiscard]] char const* name() const noexcept override { return "cucascade.rest"; }

  [[nodiscard]] std::string message(int value) const override
  {
    switch (static_cast<errc>(value)) {
      case errc::ok: return "success";
      case errc::operation_canceled: return "operation canceled";
      case errc::context_stopped: return "io_context stopped";
      case errc::invalid_argument: return "invalid argument";
      case errc::eof: return "range starts at or after end of object";
      case errc::short_read: return "short read";
      case errc::io_error: return "I/O error";
      case errc::unaligned: return "O_DIRECT alignment violated";
      case errc::not_registered: return "buffer not registered";
      case errc::memlock_exceeded: return "io_uring registration failed: memlock limit exceeded";
      case errc::ring_error: return "fatal io_uring error";
      case errc::staging_quarantined: return "staging buffer quarantined after CUDA failure";
      case errc::dns_failed: return "DNS resolution failed";
      case errc::connect_failed: return "connect failed";
      case errc::connect_timeout: return "connect timed out";
      case errc::tls_handshake_failed: return "TLS handshake failed";
      case errc::tls_verify_failed: return "TLS certificate verification failed";
      case errc::tls_protocol_error: return "TLS protocol error";
      case errc::tls_bad_record_mac: return "TLS record authentication failed";
      case errc::ktls_unavailable: return "kernel TLS unavailable";
      case errc::stale_connection: return "reused connection closed before response";
      case errc::http_bad_message: return "malformed HTTP message";
      case errc::http_header_too_large: return "HTTP header too large";
      case errc::http_length_mismatch: return "HTTP body length mismatch";
      case errc::content_range_mismatch: return "Content-Range does not match the request";
      case errc::range_ignored: return "server ignored Range";
      case errc::object_changed: return "object changed (precondition failed)";
      case errc::request_timeout: return "request timed out";
      case errc::retries_exhausted: return "retries exhausted";
      case errc::throttled: return "throttled by server";
      case errc::credentials_unavailable: return "no credentials available";
      case errc::not_supported: return "not supported";
      case errc::unhandled_exception: return "continuation threw an unhandled exception";
      case errc::http_unsupported_framing:
        return "chunked or close-delimited GET body (not supported; S3-compatible stores send "
               "Content-Length)";
    }
    return "unknown cucascade.rest error " + std::to_string(value);
  }
};

char const* http_reason(int status) noexcept
{
  switch (status) {
    case 100: return "Continue";
    case 200: return "OK";
    case 201: return "Created";
    case 204: return "No Content";
    case 206: return "Partial Content";
    case 301: return "Moved Permanently";
    case 304: return "Not Modified";
    case 307: return "Temporary Redirect";
    case 400: return "Bad Request";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 409: return "Conflict";
    case 411: return "Length Required";
    case 412: return "Precondition Failed";
    case 416: return "Range Not Satisfiable";
    case 429: return "Too Many Requests";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 502: return "Bad Gateway";
    case 503: return "Service Unavailable";
    case 504: return "Gateway Timeout";
    default: return "";
  }
}

class http_status_error_category final : public std::error_category {
 public:
  [[nodiscard]] char const* name() const noexcept override { return "http_status"; }

  [[nodiscard]] std::string message(int value) const override
  {
    auto msg           = "HTTP " + std::to_string(value);
    auto const* reason = http_reason(value);
    if (*reason != '\0') { (msg += ' ') += reason; }
    return msg;
  }
};

}  // namespace

std::error_category const& error_category() noexcept
{
  static rest_error_category const cat;
  return cat;
}

std::error_category const& http_status_category() noexcept
{
  static http_status_error_category const cat;
  return cat;
}

}  // namespace cucascade::io::rest
