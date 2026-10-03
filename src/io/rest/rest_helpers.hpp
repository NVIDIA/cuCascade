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

#pragma once

/**
 * @file rest_helpers.hpp
 * @brief Private (not installed) libcurl callbacks and request helpers shared
 *        by the REST reactor's caller-thread helpers (rest_reactor.cpp) and the
 *        per-runner transfer engine (rest_engine.cpp).
 */

#include <cucascade/io/rest/config.hpp>
#include <cucascade/io/rest/curl_handle.hpp>
#include <cucascade/io/rest/types.hpp>

#include <curl/curl.h>
#include <sys/uio.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cucascade::io::rest::detail {

// ---- libcurl callbacks -----------------------------------------------------

/// Write callback: copy curl's bytes into the sink's destination buffer at the
/// running cursor, and ALWAYS report the full incoming size so curl never
/// aborts the transfer with CURLE_WRITE_ERROR.  Overflow past capacity is
/// counted (total_received) but not stored, so a server that ignored the Range
/// header can be detected after the fact.
inline size_t write_to_sink(char* ptr, size_t size, size_t nmemb, void* userdata)
{
  auto* sink         = static_cast<buf_sink*>(userdata);
  size_t const bytes = size * nmemb;
  sink->total_received += bytes;
  size_t remaining = bytes;
  auto const* src  = reinterpret_cast<uint8_t const*>(ptr);
  // Scatter across the destination buffers in file order; a fused contiguous
  // GET spills from one buffer into the next as each fills.
  //
  // A null buffer inside a multi-buffer sink is a hole — the bytes bridging two
  // fused segments — so its span is stepped over rather than stored, and counted
  // in `written` because the read covered it.  A single-buffer sink is a
  // different animal: null there means a bounce-staged device read whose slot
  // submit() should already have bound (set_data), so a null that survives to
  // here is a bug.  Stop rather than silently discard the body — the short read
  // that follows fails the request instead of reporting success on bytes that
  // went nowhere.
  bool const holes_expected = sink->buffers.size() > 1;
  while (remaining > 0 && sink->active < sink->buffers.size()) {
    iovec& b = sink->buffers[sink->active];
    if (b.iov_base == nullptr && !holes_expected) { break; }
    if (sink->cursor >= b.iov_len) {
      ++sink->active;
      sink->cursor = 0;
      continue;
    }
    size_t const n = std::min(b.iov_len - sink->cursor, remaining);
    if (b.iov_base != nullptr) {
      std::memcpy(static_cast<uint8_t*>(b.iov_base) + sink->cursor, src, n);
    }
    sink->cursor += n;
    sink->written += n;
    src += n;
    remaining -= n;
  }
  return bytes;
}

/// Discard callback for HEAD requests (no body expected, but be defensive).
inline size_t write_discard(char* /*ptr*/, size_t size, size_t nmemb, void* /*userdata*/)
{
  return size * nmemb;
}

/// Accumulate the whole response body into a std::string (small control-plane
/// responses only — e.g. one ListObjectsV2 XML page).
inline size_t write_string(char* ptr, size_t size, size_t nmemb, void* userdata)
{
  auto* out = static_cast<std::string*>(userdata);
  out->append(ptr, size * nmemb);
  return size * nmemb;
}

/// Lowercase a byte.
inline char ascii_lower(char c)
{
  return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

/// Case-insensitively match @p line against "<name>:" and, on a hit, return the
/// trimmed value; otherwise return empty.
inline std::string match_header(std::string_view line, std::string_view name)
{
  if (line.size() < name.size() + 1) { return {}; }
  for (size_t i = 0; i < name.size(); ++i) {
    if (ascii_lower(line[i]) != ascii_lower(name[i])) { return {}; }
  }
  if (line[name.size()] != ':') { return {}; }
  std::string_view val = line.substr(name.size() + 1);
  // Trim surrounding whitespace and trailing CRLF.
  while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) {
    val.remove_prefix(1);
  }
  while (!val.empty() &&
         (val.back() == '\r' || val.back() == '\n' || val.back() == ' ' || val.back() == '\t')) {
    val.remove_suffix(1);
  }
  return std::string(val);
}

/// Returns true for an HTTP status line ("HTTP/1.1 200 OK").
inline bool is_http_status_line(std::string_view line) noexcept
{
  return line.size() >= 5 && ascii_lower(line[0]) == 'h' && ascii_lower(line[1]) == 't' &&
         ascii_lower(line[2]) == 't' && ascii_lower(line[3]) == 'p' && line[4] == '/';
}

/// Header callback: capture Content-Range, Retry-After and ETag.  A status
/// line starts a new response block (interim 1xx), which drops the ETag of
/// the previous block.
inline size_t capture_header(char* buffer, size_t size, size_t nitems, void* userdata)
{
  auto* hc           = static_cast<header_capture*>(userdata);
  size_t const bytes = size * nitems;
  std::string_view line(buffer, bytes);
  if (is_http_status_line(line)) { hc->etag.clear(); }
  if (auto v = match_header(line, "content-range"); !v.empty()) {
    hc->content_range = std::move(v);
  }
  if (auto v = match_header(line, "retry-after"); !v.empty()) { hc->retry_after = std::move(v); }
  if (auto v = match_header(line, "etag"); !v.empty()) { hc->etag = std::move(v); }
  return bytes;
}

/// Read callback of an upload: libcurl pulls the request body from the
/// staged @c buf_source.
inline size_t read_from_source(char* buffer, size_t size, size_t nitems, void* userdata)
{
  return static_cast<buf_source*>(userdata)->read(buffer, size * nitems);
}

/// Seek callback of an upload: libcurl rewinds the body to resend it.
inline int seek_source(void* userdata, curl_off_t offset, int origin)
{
  auto* source = static_cast<buf_source*>(userdata);
  if (origin != SEEK_SET || offset < 0 || static_cast<std::size_t>(offset) > source->size) {
    return CURL_SEEKFUNC_CANTSEEK;
  }
  source->seek(static_cast<std::size_t>(offset));
  return CURL_SEEKFUNC_OK;
}

// ---- retry classification --------------------------------------------------

/// HTTP status codes worth retrying (transient server / throttling).  Only the
/// transient 5xx are included: 500 Internal Error, 502 Bad Gateway, 503 Slow
/// Down / Service Unavailable, 504 Gateway Timeout.  Permanent 5xx (501 Not
/// Implemented, 505 HTTP Version Not Supported, ...) are NOT retried — they
/// would only burn the full retry budget on an error that cannot succeed.
inline bool is_retriable_status(long status) noexcept
{
  return status == 408 || status == 429 || status == 500 || status == 502 || status == 503 ||
         status == 504;
}

/// libcurl error codes worth retrying (transient transport failures).
inline bool is_retriable_curl(CURLcode rc) noexcept
{
  switch (rc) {
    case CURLE_COULDNT_CONNECT:
    case CURLE_COULDNT_RESOLVE_HOST:
    case CURLE_OPERATION_TIMEDOUT:
    case CURLE_GOT_NOTHING:
    case CURLE_RECV_ERROR:
    case CURLE_SEND_ERROR:
    case CURLE_PARTIAL_FILE:
    case CURLE_SSL_CONNECT_ERROR:
    case CURLE_HTTP2_STREAM: return true;
    default: return false;
  }
}

// ---- per-request helpers ---------------------------------------------------

/// Presigned-URL TTL for a request: the whole-request timeout plus clock-skew
/// headroom, with a sane floor so very short timeouts still leave a usable
/// window.
inline std::chrono::seconds presign_ttl(const config& cfg) noexcept
{
  long const base = cfg.request_timeout_s > 0 ? cfg.request_timeout_s + 60 : 300;
  return std::chrono::seconds{base};
}

/// Apply per-request TLS + timeout options on top of configure_easy_handle.
/// @p data_transfer selects the time bound: a data GET can be as large as the
/// cache block size, so it is bounded by the stall detector (a GET that keeps
/// delivering bytes is never cut off, however long it takes); everything else is
/// bounded by the whole-request timeout.
inline void apply_request_opts(CURL* h, const config& cfg, bool data_transfer = false)
{
  if (!cfg.ca_bundle_path.empty()) {
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h, CURLOPT_CAINFO, cfg.ca_bundle_path.c_str()));
  }
  if (!cfg.tls_verify) {
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h, CURLOPT_SSL_VERIFYPEER, 0L));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h, CURLOPT_SSL_VERIFYHOST, 0L));
  }
  if (data_transfer) {
    // Clears the whole-transfer default configure_easy_handle set.
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h, CURLOPT_TIMEOUT, 0L));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h, CURLOPT_LOW_SPEED_LIMIT, cfg.stall_speed_limit_bytes));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h, CURLOPT_LOW_SPEED_TIME, cfg.stall_time_s));
  } else if (cfg.request_timeout_s > 0) {
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h, CURLOPT_TIMEOUT, cfg.request_timeout_s));
  }
}

/// Build a header list from the authorizer's headers (empty in presigned mode)
/// plus an optional Range header.
inline curl_slist_ptr build_header_list(
  std::vector<std::pair<std::string, std::string>> const& headers, std::string const* range)
{
  curl_slist* list = nullptr;
  for (auto const& [k, v] : headers) {
    std::string const h = k + ": " + v;
    list                = curl_slist_append(list, h.c_str());
  }
  if (range != nullptr) { list = curl_slist_append(list, range->c_str()); }
  return curl_slist_ptr{list};
}

/// "Range: bytes=<lo>-<hi>" (inclusive end) for [offset, offset+size).
inline std::string range_header(size_t offset, size_t size)
{
  return "Range: bytes=" + std::to_string(offset) + "-" + std::to_string(offset + size - 1);
}

/// "Range: bytes=-<n>" — the last @p n bytes of an object (a suffix range).
/// Unlike range_header this needs no prior knowledge of the object's size.
inline std::string suffix_range_header(size_t n) { return "Range: bytes=-" + std::to_string(n); }

/// Parse the first-byte position out of a Content-Range value of the form
/// "bytes <first>-<last>/<total>" (the trimmed value captured by the header
/// callback).  Returns nullopt for any value that does not start with a
/// well-formed "bytes <first>-" so the caller can reject an unverifiable 206.
inline std::optional<size_t> content_range_start(std::string_view cr)
{
  constexpr std::string_view kUnit = "bytes";
  std::string_view sv{cr};
  if (sv.size() < kUnit.size()) { return std::nullopt; }
  for (size_t i = 0; i < kUnit.size(); ++i) {
    if (ascii_lower(sv[i]) != kUnit[i]) { return std::nullopt; }
  }
  sv.remove_prefix(kUnit.size());
  while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t')) {
    sv.remove_prefix(1);
  }
  if (sv.empty() || sv.front() < '0' || sv.front() > '9') { return std::nullopt; }
  size_t value = 0;
  size_t i     = 0;
  for (; i < sv.size() && sv[i] >= '0' && sv[i] <= '9'; ++i) {
    value = value * 10 + static_cast<size_t>(sv[i] - '0');
  }
  // A valid first-byte position is immediately followed by '-' (the range
  // separator); anything else ("*", end of string, ...) is not parseable.
  if (i >= sv.size() || sv[i] != '-') { return std::nullopt; }
  return value;
}

/// Backoff before the next attempt: honor a numeric Retry-After (seconds,
/// capped at 30 s) when present and enabled, else exponential base<<attempt
/// plus uniform jitter.
inline std::chrono::milliseconds compute_backoff(std::size_t attempt,
                                                 std::string const& retry_after,
                                                 const config& cfg)
{
  if (cfg.honor_retry_after && !retry_after.empty()) {
    try {
      long const secs = std::stol(retry_after);
      if (secs >= 0) {
        return std::min(std::chrono::milliseconds{secs * 1000}, std::chrono::milliseconds{30'000});
      }
    } catch (...) {
      // Non-numeric (HTTP-date) Retry-After: fall through to exponential.
    }
  }
  std::size_t const shift = std::min<std::size_t>(attempt, 16);
  auto const base         = cfg.retry_backoff_base * (std::size_t{1} << shift);
  std::chrono::milliseconds jitter{0};
  if (cfg.retry_jitter.count() > 0) {
    thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<long> dist(0, cfg.retry_jitter.count());
    jitter = std::chrono::milliseconds{dist(rng)};
  }
  return base + jitter;
}

}  // namespace cucascade::io::rest::detail
