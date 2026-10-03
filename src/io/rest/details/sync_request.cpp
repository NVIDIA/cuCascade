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

#include "../rest_helpers.hpp"

#include <cucascade/io/rest/curl_handle.hpp>
#include <cucascade/io/rest/details/sync_request.hpp>
#include <cucascade/io/rest/s3/xml_utils.hpp>
#include <cucascade/log/logging.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cucascade::io::rest::detail {

namespace {

bool ascii_iequals(std::string_view a, std::string_view b) noexcept
{
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
           return std::tolower(static_cast<unsigned char>(x)) ==
                  std::tolower(static_cast<unsigned char>(y));
         });
}

/// Response headers of one attempt; a status line starts a new response block
/// (interim 100-continue / redirects), so captured fields reset there.
struct response_headers {
  std::string etag;
  std::string retry_after;
};

size_t header_cb(char* buffer, size_t size, size_t nitems, void* userdata)
{
  auto* rh           = static_cast<response_headers*>(userdata);
  size_t const bytes = size * nitems;
  std::string_view const line(buffer, bytes);
  if (is_http_status_line(line)) {
    rh->etag.clear();
    rh->retry_after.clear();
  }
  if (auto v = match_header(line, "etag"); !v.empty()) { rh->etag = std::move(v); }
  if (auto v = match_header(line, "retry-after"); !v.empty()) { rh->retry_after = std::move(v); }
  return bytes;
}

/// Upload cursor over the caller's body for PUT.
struct body_source {
  std::string_view data;
  std::size_t cursor{0};
};

size_t read_body(char* buffer, size_t size, size_t nitems, void* userdata)
{
  auto* src    = static_cast<body_source*>(userdata);
  auto const n = std::min(size * nitems, src->data.size() - src->cursor);
  if (n > 0) { std::memcpy(buffer, src->data.data() + src->cursor, n); }
  src->cursor += n;
  return n;
}

int seek_body(void* userdata, curl_off_t offset, int origin)
{
  auto* src = static_cast<body_source*>(userdata);
  if (origin != SEEK_SET || offset < 0 || static_cast<std::size_t>(offset) > src->data.size()) {
    return CURL_SEEKFUNC_CANTSEEK;
  }
  src->cursor = static_cast<std::size_t>(offset);
  return CURL_SEEKFUNC_OK;
}

bool has_header(std::vector<std::pair<std::string, std::string>> const& headers,
                std::string_view name)
{
  return std::any_of(
    headers.begin(), headers.end(), [&](auto const& h) { return ascii_iequals(h.first, name); });
}

std::string describe(request_spec const& spec)
{
  return std::string{to_string(spec.method)} + " " + spec.object.bucket + "/" + spec.object.key +
         (spec.canonical_query.empty() ? std::string{} : "?" + spec.canonical_query);
}

}  // namespace

sync_response perform_sync(request_spec const& spec,
                           request_authorizer& authorizer,
                           config const& cfg,
                           std::string_view body,
                           sync_request_options const& opts)
{
  bool const sends_body = spec.method == request_method::PUT || spec.method == request_method::POST;
  if (!sends_body && !body.empty()) {
    throw std::invalid_argument("rest: perform_sync: a " + std::string{to_string(spec.method)} +
                                " request cannot carry a body");
  }
  auto const what        = describe(spec);
  auto const max_attempt = std::max<std::size_t>(cfg.max_retry_attempts, 1);
  std::size_t auth_failures{0};
  std::string last_error;

  for (std::size_t attempt = 0; attempt < max_attempt; ++attempt) {
    auto const authd = authorizer.authorize_request(spec, presign_ttl(cfg));

    curl_easy_ptr h{curl_easy_init()};
    if (!h) { throw std::runtime_error("rest: perform_sync: curl_easy_init failed"); }
    configure_easy_handle(h.get(), global_curl_context::instance().share_handle());
    apply_request_opts(h.get(), cfg, opts.data_transfer);

    curl_slist* list = nullptr;
    for (auto const& [k, v] : authd.headers) {
      list = curl_slist_append(list, (k + ": " + v).c_str());
    }
    if (sends_body) { list = curl_slist_append(list, "Expect:"); }
    if (spec.method == request_method::POST && !has_header(authd.headers, "content-type")) {
      list = curl_slist_append(list, "Content-Type:");
    }
    curl_slist_ptr const hdrs{list};

    response_headers rh;
    std::string response;
    body_source src{body, 0};
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_URL, authd.url.c_str()));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HTTPHEADER, hdrs.get()));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEFUNCTION, &write_string));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEDATA, &response));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERFUNCTION, &header_cb));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERDATA, &rh));
    switch (spec.method) {
      case request_method::GET:
        CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HTTPGET, 1L));
        break;
      case request_method::HEAD:
        CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_NOBODY, 1L));
        break;
      case request_method::PUT:
        CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_UPLOAD, 1L));
        CUCASCADE_CURL_CHECK(curl_easy_setopt(
          h.get(), CURLOPT_INFILESIZE_LARGE, static_cast<curl_off_t>(body.size())));
        CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_READFUNCTION, &read_body));
        CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_READDATA, &src));
        CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_SEEKFUNCTION, &seek_body));
        CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_SEEKDATA, &src));
        break;
      case request_method::POST:
        CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_POST, 1L));
        CUCASCADE_CURL_CHECK(curl_easy_setopt(
          h.get(), CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(body.size())));
        CUCASCADE_CURL_CHECK(
          curl_easy_setopt(h.get(), CURLOPT_POSTFIELDS, body.empty() ? "" : body.data()));
        break;
      case request_method::DELETE_:
        CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_CUSTOMREQUEST, "DELETE"));
        break;
    }

    CURLcode const rc = curl_easy_perform(h.get());
    long status       = 0;
    curl_easy_getinfo(h.get(), CURLINFO_RESPONSE_CODE, &status);

    bool retriable   = false;
    bool auth_failed = false;
    if (rc != CURLE_OK) {
      last_error = curl_easy_strerror(rc);
      retriable  = is_retriable_curl(rc);
    } else {
      auto const err = s3::parse_s3_error(response);
      bool const accepted =
        std::find(opts.accepted_statuses.begin(), opts.accepted_statuses.end(), status) !=
        opts.accepted_statuses.end();
      if (accepted && !(opts.retry_on_error_body && err.has_value())) {
        sync_response result{status, std::move(response), std::move(rh.etag), attempt + 1, {}};
        curl_off_t length = -1;
        if (curl_easy_getinfo(h.get(), CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &length) == CURLE_OK &&
            length >= 0) {
          result.content_length = static_cast<std::size_t>(length);
        }
        return result;
      }
      last_error = "HTTP " + std::to_string(status);
      if (err.has_value() && !err->code.empty()) { last_error += ": " + err->code; }
      retriable = accepted || is_retriable_status(status) ||
                  (status == 400 && err.has_value() && err->code == "RequestTimeout");
      if (status == 403) {
        auth_failed = true;
        retriable   = ++auth_failures < std::max<std::size_t>(cfg.max_auth_retry_attempts, 1);
      }
    }

    if (!retriable) {
      throw std::runtime_error("rest: " + std::string{to_string(spec.method)} + " " + last_error +
                               " for " + spec.object.bucket + "/" + spec.object.key);
    }
    if (attempt + 1 < max_attempt) {
      CUCASCADE_LOG_WARN("rest: perform_sync: retrying {} after {} (attempt {}/{}{})",
                         what,
                         last_error,
                         attempt + 1,
                         max_attempt,
                         auth_failed ? ", auth" : "");
      std::this_thread::sleep_for(compute_backoff(attempt, rh.retry_after, cfg));
    }
  }
  throw std::runtime_error("rest: " + std::string{to_string(spec.method)} + " exhausted retries (" +
                           last_error + ") for " + spec.object.bucket + "/" + spec.object.key);
}

}  // namespace cucascade::io::rest::detail
