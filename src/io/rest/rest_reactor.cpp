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

#include "rest_helpers.hpp"

#include <cucascade/io/rest/curl_handle.hpp>
#include <cucascade/io/rest/details/sync_request.hpp>
#include <cucascade/io/rest/rest_engine.hpp>
#include <cucascade/io/rest/rest_reactor.hpp>
#include <cucascade/io/rest/s3/sigv4.hpp>
#include <cucascade/io/uri_parser.hpp>
#include <cucascade/log/logging.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace cucascade::io::rest {

// Shared libcurl callbacks / request helpers (rest_helpers.hpp), also used by
// the per-runner engine (rest_engine.cpp).
using detail::apply_request_opts;
using detail::ascii_lower;
using detail::build_header_list;
using detail::capture_header;
using detail::compute_backoff;
using detail::content_range_start;
using detail::is_http_status_line;
using detail::is_retriable_curl;
using detail::is_retriable_status;
using detail::match_header;
using detail::presign_ttl;
using detail::suffix_range_header;
using detail::write_discard;
using detail::write_string;

namespace {

/// Headers captured for one HEAD attempt.  Separate from @c header_capture so
/// the async data-GET path parses nothing it does not consume.  A status line
/// starts a new response block within the transfer, so all fields reset there.
struct head_capture {
  std::string retry_after;
  std::string etag;
};

size_t head_header_cb(char* buffer, size_t size, size_t nitems, void* userdata)
{
  auto* hc           = static_cast<head_capture*>(userdata);
  size_t const bytes = size * nitems;
  std::string_view const line(buffer, bytes);
  if (is_http_status_line(line)) {
    hc->etag.clear();
    hc->retry_after.clear();
  }
  if (auto v = match_header(line, "etag"); !v.empty()) { hc->etag = std::move(v); }
  if (auto v = match_header(line, "retry-after"); !v.empty()) { hc->retry_after = std::move(v); }
  return bytes;
}

/// Shared sink for a suffix-range footer probe: the header callback records the
/// HTTP status (from the status line) plus Content-Range / Retry-After / ETag;
/// the body callback consults @c status to abort a non-206 response before it
/// streams a whole object into us.  @c HEADERDATA and @c WRITEDATA point at the
/// same one.
struct suffix_sink {
  std::vector<std::uint8_t> data;
  std::size_t cap{0};
  std::size_t total_received{0};  // wire bytes, incl. those dropped by cap/abort
  long status{0};
  std::string content_range;
  std::string retry_after;
  std::string etag;
};

/// Capture the status and headers used to validate or retry a suffix probe.
/// A status line starts a new response block within the transfer, so the
/// header fields reset there — only the final block's values survive.
size_t suffix_header_cb(char* buffer, size_t size, size_t nitems, void* userdata)
{
  auto* s            = static_cast<suffix_sink*>(userdata);
  size_t const bytes = size * nitems;
  std::string_view const line(buffer, bytes);
  if (is_http_status_line(line)) {
    s->etag.clear();
    s->content_range.clear();
    s->retry_after.clear();
    if (auto const sp = line.find(' '); sp != std::string_view::npos) {
      long code = 0;
      for (size_t i = sp + 1; i < line.size() && line[i] >= '0' && line[i] <= '9'; ++i) {
        code = code * 10 + (line[i] - '0');
      }
      if (code != 0) { s->status = code; }
    }
  }
  if (auto v = match_header(line, "content-range"); !v.empty()) { s->content_range = std::move(v); }
  if (auto v = match_header(line, "retry-after"); !v.empty()) { s->retry_after = std::move(v); }
  if (auto v = match_header(line, "etag"); !v.empty()) { s->etag = std::move(v); }
  return bytes;
}

/// Body callback for a suffix probe: abort a non-206 response (a deliberate
/// short write, surfacing as CURLE_WRITE_ERROR) so a server that ignores the
/// Range or answers 416/4xx never streams a whole object into us; otherwise
/// append up to @c cap bytes and report the full incoming size to curl.
size_t suffix_write_cb(char* ptr, size_t size, size_t nmemb, void* userdata)
{
  auto* s            = static_cast<suffix_sink*>(userdata);
  size_t const bytes = size * nmemb;
  s->total_received += bytes;
  if (s->status != 206) { return 0; }
  if (s->data.size() < s->cap) {
    size_t const take = std::min(s->cap - s->data.size(), bytes);
    auto const* src   = reinterpret_cast<std::uint8_t const*>(ptr);
    s->data.insert(s->data.end(), src, src + take);
  }
  return bytes;
}

}  // namespace

shared_byte_span make_shared_byte_span(std::vector<std::uint8_t> bytes)
{
  auto owner = std::make_shared<detail::byte_storage>(std::move(bytes));
  // Aliasing constructor: shares `owner`'s control block (keeping the buffer
  // alive) while the pointer itself refers to the span member inside it.
  return shared_byte_span{owner, &owner->view};
}

std::optional<size_t> content_range_total(std::string_view cr)
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
  // The range part must be a satisfied "<first>-<last>", never "*": a leading
  // digit both rejects "bytes */..." and confirms a total follows the '/'.
  if (sv.empty() || sv.front() < '0' || sv.front() > '9') { return std::nullopt; }
  auto const slash = sv.find('/');
  if (slash == std::string_view::npos) { return std::nullopt; }
  std::string_view const total = sv.substr(slash + 1);
  if (total.empty() || total.front() < '0' || total.front() > '9') { return std::nullopt; }
  size_t value = 0;
  for (char const c : total) {
    if (c < '0' || c > '9') { break; }
    value = value * 10 + static_cast<size_t>(c - '0');
  }
  return value;
}

// ---------------------------------------------------------------------------
// construction / lifecycle
// ---------------------------------------------------------------------------

rest_reactor::rest_reactor(std::shared_ptr<reactor_context> ctx, std::string_view tname)
  : _ctx(std::move(ctx)), _tname(tname)
{
  if (!_ctx) { throw std::invalid_argument("rest_reactor: reactor_context must be non-null"); }
  _config = _ctx->cfg();
  if (!_ctx->authorizer()) {
    throw std::invalid_argument("rest_reactor: context authorizer must be non-null");
  }
  if (_config.max_connections == 0) {
    throw std::invalid_argument("rest_reactor: max_connections must be > 0");
  }
  if (_config.max_retry_attempts == 0) { _config.max_retry_attempts = 1; }
  // S3 limits: parts are 5 MiB..5 GiB (only the last may be smaller); a single
  // PUT carries at most 5 GiB.
  constexpr std::size_t min_part_size = 5UL << 20;
  constexpr std::size_t max_put_size  = 5UL << 30;
  _config.write.part_size = std::clamp(_config.write.part_size, min_part_size, max_put_size);
  _config.write.multipart_threshold = std::min(_config.write.multipart_threshold, max_put_size);
  _config.write.max_buffered_parts  = std::max<std::size_t>(_config.write.max_buffered_parts, 1);

  // Touch the process-wide curl context so global init + the shared cache are
  // ready before any handle is created — including the blocking HEAD that
  // rest_ioctx::create_io_object issues before any runner exists.
  (void)global_curl_context::instance();
}

// The owning templated_ioctx shuts down (and waits for every runner) before it
// destroys the reactor, so no engine can still reference it here.
rest_reactor::~rest_reactor()
{
  // Sessions may outlive the reactor: from now on they only log.
  for (auto const& orphan : _orphans->close()) {
    CUCASCADE_LOG_WARN(
      "rest_reactor: multipart upload {} of {}/{} was dropped without commit and could not be "
      "aborted; it stays on the store until a bucket lifecycle rule removes it",
      orphan.upload_id,
      orphan.object.bucket,
      orphan.object.key);
  }
}

std::unique_ptr<rest_engine> rest_reactor::make_engine(::cucascade::io::detail::runner_slot& slot)
{
  return std::make_unique<rest_engine>(*this, slot);
}

std::size_t rest_reactor::host_read(io_object_type const& file,
                                    std::size_t offset,
                                    std::size_t size,
                                    std::uint8_t* dst)
{
  if (size == 0) return 0;
  size = std::min(size, file.size() > offset ? file.size() - offset : std::size_t{0});
  if (size == 0) return 0;
  if (dst == nullptr) throw std::invalid_argument("rest_reactor::host_read: null destination");

  if (auto const& stash = file.stash(); stash) {
    auto const lo = file.stash_window_lo();
    auto const hi = lo + stash->size();
    if (offset >= lo && offset + size <= hi) {
      std::memcpy(dst, stash->data() + (offset - lo), size);
      return size;
    }
  }

  std::shared_ptr<const io_object> owner;
  try {
    owner = file.shared_from_this();
  } catch (std::bad_weak_ptr const&) {
    owner = std::shared_ptr<const io_object>(&file, [](io_object const*) {});
  }

  auto coordinator = std::make_shared<grouped_coordinator>(size, 1);
  auto future      = coordinator->get_future();
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{offset, size}, host_buffer{dst});
  _hub.enqueue(grouped_io_request::create(std::move(owner), std::move(slices), coordinator));
  return std::move(future).get();
}

std::unique_ptr<rest_reactor::io_object_type> rest_reactor::create_io_object_for_write(
  std::string path, write_open_options opts)
{
  auto parsed = cucascade::io::parse(path);
  if (parsed.scheme != "s3") {
    throw std::invalid_argument("rest_reactor::create_io_object_for_write: unsupported scheme '" +
                                parsed.scheme + "'");
  }
  if (opts.mode != write_mode::create_or_truncate) {
    // An object store has no partial update of an existing object.
    throw std::system_error(std::make_error_code(std::errc::not_supported),
                            "rest: only write_mode::create_or_truncate is supported for " + path);
  }
  auto session = std::make_shared<upload_session>(object_ref{parsed.host, parsed.path},
                                                  _config.write,
                                                  opts.size_hint,
                                                  _ctx->host_memory_resource());
  session->set_orphan_sink(_orphans);
  {
    std::lock_guard lock{_sessions_mtx};
    std::erase_if(_sessions, [](std::weak_ptr<upload_session> const& s) { return s.expired(); });
    _sessions.push_back(session);
  }
  return std::make_unique<io_object_type>(
    std::move(path), std::move(parsed.host), std::move(parsed.path), std::move(session));
}

std::size_t rest_reactor::host_write(io_object_type const& object,
                                     std::size_t offset,
                                     std::size_t size,
                                     std::uint8_t const* source,
                                     write_options opts)
{
  if (size == 0) return 0;
  if (source == nullptr) throw std::invalid_argument("rest_reactor::host_write: null source");
  std::shared_ptr<const io_object> owner;
  try {
    owner = object.shared_from_this();
  } catch (std::bad_weak_ptr const&) {
    owner = std::shared_ptr<const io_object>(&object, [](io_object const*) {});
  }
  std::vector<write_segment> segments;
  segments.push_back(write_segment{range{offset, size}, host_source{source}});
  auto coordinator = std::make_shared<grouped_coordinator>(size, segments.size());
  auto future      = coordinator->get_future();
  _hub.enqueue(grouped_io_request::create_write(
    std::move(owner), std::move(segments), opts, std::move(coordinator)));
  return std::move(future).get();
}

void rest_reactor::abort_live_uploads() noexcept
{
  std::vector<std::shared_ptr<upload_session>> live;
  try {
    std::lock_guard lock{_sessions_mtx};
    for (auto const& weak : _sessions) {
      if (auto session = weak.lock()) live.push_back(std::move(session));
    }
    std::erase_if(_sessions, [](std::weak_ptr<upload_session> const& s) { return s.expired(); });
  } catch (...) {
    CUCASCADE_LOG_ERROR("rest_reactor: could not enumerate live uploads at shutdown");
    return;
  }

  // Orphaned uploads no engine aborted yet: abort them here like live ones.
  try {
    for (auto& orphan : _orphans->take_all()) {
      live.push_back(
        upload_session::for_abort(std::move(orphan.object), std::move(orphan.upload_id)));
    }
  } catch (...) {
    CUCASCADE_LOG_ERROR("rest_reactor: could not collect orphaned uploads at shutdown");
  }

  // Best effort, bounded: a few attempts with the control-plane timeout.
  auto cfg               = _config;
  cfg.max_retry_attempts = std::min<std::size_t>(cfg.max_retry_attempts, 3);
  for (auto const& session : live) {
    auto const upload_id = session->cancel_for_shutdown();
    if (upload_id.empty()) continue;
    try {
      request_spec spec;
      spec.method          = request_method::DELETE_;
      spec.object          = session->object();
      spec.canonical_query = "uploadId=" + s3::uri_encode(upload_id, true);
      detail::sync_request_options options;
      options.accepted_statuses = {200, 204, 404};
      static_cast<void>(detail::perform_sync(spec, *_ctx->authorizer(), cfg, {}, options));
      session->abort_succeeded();
    } catch (std::exception const& error) {
      CUCASCADE_LOG_WARN("rest_reactor: AbortMultipartUpload of {}/{} failed at shutdown: {}",
                         session->object().bucket,
                         session->object().key,
                         error.what());
    } catch (...) {
      CUCASCADE_LOG_WARN("rest_reactor: AbortMultipartUpload of {}/{} failed at shutdown",
                         session->object().bucket,
                         session->object().key);
    }
  }
}

void rest_reactor::warmup(std::string bucket)
{
  {
    std::lock_guard lk{_warm_mtx};
    _warm_bucket       = std::move(bucket);
    _warm_requested_at = std::chrono::steady_clock::now();
    _warm_generation.fetch_add(1, std::memory_order_acq_rel);
  }
  // Every registered runner primes its own (thread-confined) pool.
  _hub.registry().wake_all();
}

rest_reactor::warm_request rest_reactor::current_warm_request() const
{
  std::lock_guard lk{_warm_mtx};
  return warm_request{
    _warm_generation.load(std::memory_order_relaxed), _warm_bucket, _warm_requested_at};
}

head_object_result rest_reactor::head_object(std::string_view bucket, std::string_view key)
{
  object_ref const obj{std::string(bucket), std::string(key)};
  std::string last_error;
  for (std::size_t attempt = 0; attempt < _config.max_retry_attempts; ++attempt) {
    head_capture hc;
    auto const authd =
      _ctx->authorizer()->authorize(obj, request_method::HEAD, presign_ttl(_config));

    curl_easy_ptr h{curl_easy_init()};
    if (!h) { throw std::runtime_error("rest_reactor::head_object: curl_easy_init failed"); }
    configure_easy_handle(h.get(), global_curl_context::instance().share_handle());
    apply_request_opts(h.get(), _config);

    curl_slist_ptr hdrs = build_header_list(authd.headers, nullptr);
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_URL, authd.url.c_str()));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_NOBODY, 1L));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HTTPHEADER, hdrs.get()));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEFUNCTION, &write_discard));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERFUNCTION, &head_header_cb));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERDATA, &hc));

    CURLcode const rc = curl_easy_perform(h.get());
    long status       = 0;
    curl_easy_getinfo(h.get(), CURLINFO_RESPONSE_CODE, &status);

    if (rc == CURLE_OK && status == 200) {
      curl_off_t cl = -1;
      curl_easy_getinfo(h.get(), CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &cl);
      if (cl < 0) {
        throw std::runtime_error("rest_reactor::head_object: missing Content-Length for " +
                                 obj.bucket + "/" + obj.key);
      }
      return head_object_result{static_cast<size_t>(cl), std::move(hc.etag)};
    }

    last_error =
      rc != CURLE_OK ? std::string(curl_easy_strerror(rc)) : ("HTTP " + std::to_string(status));
    bool const retriable =
      (rc != CURLE_OK && is_retriable_curl(rc)) || (rc == CURLE_OK && is_retriable_status(status));
    if (!retriable) {
      throw std::runtime_error("rest_reactor::head_object: " + last_error + " for " + obj.bucket +
                               "/" + obj.key);
    }
    if (attempt + 1 < _config.max_retry_attempts) {
      CUCASCADE_LOG_WARN("rest_reactor::head_object: retrying {}/{} after {} (attempt {}/{})",
                         obj.bucket,
                         obj.key,
                         last_error,
                         attempt + 1,
                         _config.max_retry_attempts);
      std::this_thread::sleep_for(compute_backoff(attempt, hc.retry_after, _config));
    }
  }
  throw std::runtime_error("rest_reactor::head_object: exhausted retries (" + last_error +
                           ") for " + obj.bucket + "/" + obj.key);
}

size_t rest_reactor::head_object_size(std::string_view bucket, std::string_view key)
{
  return head_object(bucket, key).object_size;
}

std::string rest_reactor::list_page(std::string_view bucket,
                                    std::string_view prefix,
                                    std::string_view canonical_query)
{
  std::string const bucket_s{bucket};
  std::string const prefix_s{prefix};
  std::string last_error;
  for (std::size_t attempt = 0; attempt < _config.max_retry_attempts; ++attempt) {
    header_capture hc;
    auto const authd = _ctx->authorizer()->authorize_list(
      bucket_s, std::string{canonical_query}, presign_ttl(_config));

    curl_easy_ptr h{curl_easy_init()};
    if (!h) { throw std::runtime_error("rest_reactor::list_page: curl_easy_init failed"); }
    configure_easy_handle(h.get(), global_curl_context::instance().share_handle());
    apply_request_opts(h.get(), _config);

    std::string body;
    curl_slist_ptr hdrs = build_header_list(authd.headers, nullptr);
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_URL, authd.url.c_str()));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HTTPGET, 1L));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HTTPHEADER, hdrs.get()));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEFUNCTION, &write_string));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEDATA, &body));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERFUNCTION, &capture_header));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERDATA, &hc));

    CURLcode const rc = curl_easy_perform(h.get());
    long status       = 0;
    curl_easy_getinfo(h.get(), CURLINFO_RESPONSE_CODE, &status);

    if (rc == CURLE_OK && status == 200) { return body; }

    last_error =
      rc != CURLE_OK ? std::string(curl_easy_strerror(rc)) : ("HTTP " + std::to_string(status));
    bool const retriable =
      (rc != CURLE_OK && is_retriable_curl(rc)) || (rc == CURLE_OK && is_retriable_status(status));
    if (!retriable) {
      throw std::runtime_error("rest_reactor::list_page: " + last_error + " for " + bucket_s + "/" +
                               prefix_s);
    }
    if (attempt + 1 < _config.max_retry_attempts) {
      CUCASCADE_LOG_WARN("rest_reactor::list_page: retrying {}/{} after {} (attempt {}/{})",
                         bucket_s,
                         prefix_s,
                         last_error,
                         attempt + 1,
                         _config.max_retry_attempts);
      std::this_thread::sleep_for(compute_backoff(attempt, hc.retry_after, _config));
    }
  }
  throw std::runtime_error("rest_reactor::list_page: exhausted retries (" + last_error + ") for " +
                           bucket_s + "/" + prefix_s);
}

footer_probe rest_reactor::fetch_footer_suffix(std::string_view bucket,
                                               std::string_view key,
                                               std::size_t n)
{
  // Bind-time, blocking, and off the reactor's pooled connections: each call
  // opens a fresh TCP+TLS connection and every file's probe runs on one reactor.
  footer_probe probe;
  if (n == 0) { return probe; }

  object_ref const obj{std::string(bucket), std::string(key)};
  std::string last_error;
  for (std::size_t attempt = 0; attempt < _config.max_retry_attempts; ++attempt) {
    suffix_sink sink;
    sink.cap = n;

    auto const authd =
      _ctx->authorizer()->authorize(obj, request_method::GET, presign_ttl(_config));

    curl_easy_ptr h{curl_easy_init()};
    if (!h) {
      throw std::runtime_error("rest_reactor::fetch_footer_suffix: curl_easy_init failed");
    }
    configure_easy_handle(h.get(), global_curl_context::instance().share_handle());
    apply_request_opts(h.get(), _config);

    std::string const range = suffix_range_header(n);
    curl_slist_ptr hdrs     = build_header_list(authd.headers, &range);
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_URL, authd.url.c_str()));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HTTPHEADER, hdrs.get()));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEFUNCTION, &suffix_write_cb));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEDATA, &sink));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERFUNCTION, &suffix_header_cb));
    CUCASCADE_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERDATA, &sink));

    CURLcode const rc = curl_easy_perform(h.get());
    long status       = 0;
    curl_easy_getinfo(h.get(), CURLINFO_RESPONSE_CODE, &status);

    // suffix_write_cb aborts any non-206 body, so a CURLE_WRITE_ERROR here is our
    // own doing and the HTTP status is still valid; only a different curl error
    // (no HTTP status) is a genuine transport failure.
    if (rc != CURLE_OK && rc != CURLE_WRITE_ERROR) {
      last_error = std::string(curl_easy_strerror(rc));
      if (!is_retriable_curl(rc)) {
        throw std::runtime_error("rest_reactor::fetch_footer_suffix: " + last_error + " for " +
                                 obj.bucket + "/" + obj.key);
      }
    } else if (status == 206) {
      // Trust the 206 only when the window origin and total both parse and the
      // delivered byte count matches exactly; an unverifiable 206 (missing /
      // "*" Content-Range) reports an empty probe so the caller HEADs instead.
      auto const total = content_range_total(sink.content_range);
      auto const start = content_range_start(sink.content_range);
      if (total && start && *start <= *total && sink.data.size() == *total - *start) {
        probe.object_size = *total;
        probe.window_lo   = *start;
        probe.bytes       = make_shared_byte_span(std::move(sink.data));
        probe.etag        = std::move(sink.etag);
      }
      return probe;
    } else if (status == 200 || status == 416) {
      // Range ignored (full body) or unsatisfiable (416): probe unusable but the
      // object exists — report empty so the caller falls back to a HEAD.
      return probe;
    } else if (is_retriable_status(status)) {
      last_error = "HTTP " + std::to_string(status);
    } else {
      // 404 / 403 / 401 / ... — an error a HEAD would not recover from either.
      throw std::runtime_error("rest_reactor::fetch_footer_suffix: HTTP " + std::to_string(status) +
                               " for " + obj.bucket + "/" + obj.key);
    }

    if (attempt + 1 < _config.max_retry_attempts) {
      CUCASCADE_LOG_WARN(
        "rest_reactor::fetch_footer_suffix: retrying {}/{} after {} (attempt {}/{})",
        obj.bucket,
        obj.key,
        last_error,
        attempt + 1,
        _config.max_retry_attempts);
      std::this_thread::sleep_for(compute_backoff(attempt, sink.retry_after, _config));
    }
  }
  throw std::runtime_error("rest_reactor::fetch_footer_suffix: exhausted retries (" + last_error +
                           ") for " + obj.bucket + "/" + obj.key);
}

// ---------------------------------------------------------------------------
// batched footer resolve
// ---------------------------------------------------------------------------

namespace {

/// Backing storage for a footer payload delivered by resolve_footer_batch:
/// the byte buffer plus the entry's budget reservation.  Member order is the
/// release contract — `lease` is declared after `budget`, so it is destroyed
/// first and returns its bytes to a still-alive admission_control even when
/// this storage outlives the ioctx that created it.
struct leased_byte_storage {
  std::shared_ptr<exec::admission_control> budget;
  exec::admission_control::slot lease;
  std::vector<std::uint8_t> bytes;
  std::span<const std::uint8_t> view;

  leased_byte_storage(std::shared_ptr<exec::admission_control> b,
                      exec::admission_control::slot l,
                      std::vector<std::uint8_t> data)
    : budget(std::move(b)), lease(std::move(l)), bytes(std::move(data)), view(bytes)
  {
  }

  leased_byte_storage(leased_byte_storage const&)            = delete;
  leased_byte_storage& operator=(leased_byte_storage const&) = delete;
  leased_byte_storage(leased_byte_storage&&)                 = delete;
  leased_byte_storage& operator=(leased_byte_storage&&)      = delete;
};

shared_byte_span make_leased_byte_span(std::shared_ptr<exec::admission_control> budget,
                                       exec::admission_control::slot lease,
                                       std::vector<std::uint8_t> bytes)
{
  auto owner =
    std::make_shared<leased_byte_storage>(std::move(budget), std::move(lease), std::move(bytes));
  return shared_byte_span{owner, &owner->view};
}

enum class footer_entry_stage : std::uint8_t { pending, transfer, backoff, done };
enum class footer_entry_kind : std::uint8_t { probe, head };

/// Per-entry state of one batched footer resolve.  Lives in a fixed-size
/// vector for the whole batch — the curl callbacks hold pointers into it.
struct footer_entry {
  std::size_t pos{0};  // position in the batch's parallel spans
  footer_entry_stage stage{footer_entry_stage::pending};
  footer_entry_kind kind{footer_entry_kind::probe};
  std::size_t attempt{0};
  suffix_sink sink;
  head_capture head;
  exec::admission_control::slot lease;
  curl_easy_ptr easy;
  curl_slist_ptr headers;
  std::string range;
  std::string last_error;
  std::chrono::steady_clock::time_point retry_at{};
};

}  // namespace

void rest_reactor::resolve_footer_batch(std::span<std::string const> paths,
                                        std::span<object_ref const> objects,
                                        std::span<std::size_t const> indices,
                                        std::size_t max_inflight,
                                        std::shared_ptr<exec::admission_control> budget,
                                        std::function<void(footer_resolve_result)> const& on_result,
                                        std::stop_token stop)
{
  std::size_t const window = _config.footer_probe_bytes;

  curl_multi_ptr multi{curl_multi_init()};
  if (!multi) {
    throw std::runtime_error("rest_reactor::resolve_footer_batch: curl_multi_init failed");
  }
  CUCASCADE_CURLM_CHECK(curl_multi_setopt(multi.get(), CURLMOPT_PIPELINING, CURLPIPE_NOTHING));
  CUCASCADE_CURLM_CHECK(
    curl_multi_setopt(multi.get(), CURLMOPT_MAXCONNECTS, static_cast<long>(max_inflight)));
  CUCASCADE_CURLM_CHECK(
    curl_multi_setopt(multi.get(), CURLMOPT_MAX_HOST_CONNECTIONS, static_cast<long>(max_inflight)));

  // curl_multi_wakeup is the one multi function that is safe to call from
  // another thread; a wakeup with no poll in flight makes the next poll
  // return early, so the stop signal cannot be lost between the check and
  // the poll.
  std::stop_callback wake{stop, [&multi] { curl_multi_wakeup(multi.get()); }};

  std::vector<footer_entry> entries(paths.size());
  for (std::size_t i = 0; i < entries.size(); ++i) {
    entries[i].pos = i;
    // Single-probe parity: fetch_footer_suffix skips the suffix GET entirely
    // when the window is zero, so a zero-window batch entry starts at the
    // HEAD fallback directly (no GET, no lease).
    if (window == 0) { entries[i].kind = footer_entry_kind::head; }
  }

  // Unwind safety: should anything below throw while transfers are in
  // flight, every easy handle still attached to the multi must be detached
  // BEFORE `entries` (which owns the handles) is destroyed — cleaning up an
  // easy handle still added to a multi is undefined.  Normal exits detach in
  // process_completions / cancel_remaining, leaving this a no-op.
  struct multi_detach {
    CURLM* m;
    std::vector<footer_entry>* entries;
    ~multi_detach()
    {
      for (auto& e : *entries) {
        if (e.easy) { curl_multi_remove_handle(m, e.easy.get()); }
      }
    }
  } detach_guard{multi.get(), &entries};

  std::size_t undelivered   = entries.size();
  std::size_t active        = 0;
  std::size_t next_to_start = 0;
  std::exception_ptr callback_error;

  // Backoff bookkeeping: a count plus a deadline min-heap, so the event loop
  // never rescans the whole entry vector.  Heap records whose entry left the
  // backoff stage are skipped lazily on pop.
  std::size_t backoff_count = 0;
  using retry_record        = std::pair<std::chrono::steady_clock::time_point, footer_entry*>;
  std::priority_queue<retry_record, std::vector<retry_record>, std::greater<>> retry_heap;

  auto deliver = [&](footer_entry& e, footer_resolve_result&& r) {
    e.stage = footer_entry_stage::done;
    --undelivered;
    try {
      on_result(std::move(r));
    } catch (...) {
      // First exception wins; throws from the cancel sweep's own deliveries
      // are suppressed.
      if (!callback_error) { callback_error = std::current_exception(); }
    }
  };

  auto error_result = [&](footer_entry const& e, std::exception_ptr err) {
    footer_resolve_result r;
    r.index = indices[e.pos];
    r.path  = paths[e.pos];
    r.error = std::move(err);
    return r;
  };

  auto fail_entry = [&](footer_entry& e, std::string const& what) {
    // Buffer before lease: the bytes must be gone before the ledger says so.
    e.sink  = suffix_sink{};
    e.lease = {};
    deliver(e,
            error_result(e,
                         std::make_exception_ptr(std::runtime_error(
                           "rest_reactor::resolve_footer_batch: " + what + " for " +
                           objects[e.pos].bucket + "/" + objects[e.pos].key))));
  };

  auto cancel_remaining = [&] {
    for (auto& e : entries) {
      if (e.stage == footer_entry_stage::done) { continue; }
      if (e.easy) {
        curl_multi_remove_handle(multi.get(), e.easy.get());
        e.easy.reset();
        e.headers.reset();
        if (e.stage == footer_entry_stage::transfer) { --active; }
      }
      e.sink  = suffix_sink{};
      e.lease = {};
      deliver(e,
              error_result(e,
                           std::make_exception_ptr(
                             std::system_error(std::make_error_code(std::errc::operation_canceled),
                                               "rest_reactor::resolve_footer_batch: canceled"))));
    }
  };

  auto submit = [&](footer_entry& e) {
    bool const is_probe = e.kind == footer_entry_kind::probe;
    try {
      auto const authd =
        _ctx->authorizer()->authorize(objects[e.pos],
                                      is_probe ? request_method::GET : request_method::HEAD,
                                      presign_ttl(_config));

      e.easy = curl_easy_ptr{curl_easy_init()};
      if (!e.easy) {
        throw std::runtime_error("rest_reactor::resolve_footer_batch: curl_easy_init failed");
      }
      configure_easy_handle(e.easy.get(), global_curl_context::instance().share_handle());
      apply_request_opts(e.easy.get(), _config);
      if (is_probe) {
        e.sink     = suffix_sink{};
        e.sink.cap = window;
        // Reserve up front so the buffer never grows past the budgeted window
        // while bytes stream in.
        e.sink.data.reserve(window);
        e.range   = suffix_range_header(window);
        e.headers = build_header_list(authd.headers, &e.range);
        CUCASCADE_CURL_CHECK(
          curl_easy_setopt(e.easy.get(), CURLOPT_WRITEFUNCTION, &suffix_write_cb));
        CUCASCADE_CURL_CHECK(curl_easy_setopt(e.easy.get(), CURLOPT_WRITEDATA, &e.sink));
        CUCASCADE_CURL_CHECK(
          curl_easy_setopt(e.easy.get(), CURLOPT_HEADERFUNCTION, &suffix_header_cb));
        CUCASCADE_CURL_CHECK(curl_easy_setopt(e.easy.get(), CURLOPT_HEADERDATA, &e.sink));
      } else {
        e.head    = head_capture{};
        e.headers = build_header_list(authd.headers, nullptr);
        CUCASCADE_CURL_CHECK(curl_easy_setopt(e.easy.get(), CURLOPT_NOBODY, 1L));
        CUCASCADE_CURL_CHECK(curl_easy_setopt(e.easy.get(), CURLOPT_WRITEFUNCTION, &write_discard));
        CUCASCADE_CURL_CHECK(
          curl_easy_setopt(e.easy.get(), CURLOPT_HEADERFUNCTION, &head_header_cb));
        CUCASCADE_CURL_CHECK(curl_easy_setopt(e.easy.get(), CURLOPT_HEADERDATA, &e.head));
      }
      CUCASCADE_CURL_CHECK(curl_easy_setopt(e.easy.get(), CURLOPT_URL, authd.url.c_str()));
      CUCASCADE_CURL_CHECK(curl_easy_setopt(e.easy.get(), CURLOPT_HTTPHEADER, e.headers.get()));
      CUCASCADE_CURL_CHECK(curl_easy_setopt(e.easy.get(), CURLOPT_PRIVATE, &e));
      CUCASCADE_CURLM_CHECK(curl_multi_add_handle(multi.get(), e.easy.get()));
      e.stage = footer_entry_stage::transfer;
      ++active;
    } catch (...) {
      // Per-entry failure: the authorizer may throw on credential errors and
      // any curl setup check may throw; the entry gets that exception and
      // siblings continue.  The handle is not in the multi here — every check
      // above precedes curl_multi_add_handle, and a failed add does not add.
      e.easy.reset();
      e.headers.reset();
      e.sink  = suffix_sink{};
      e.lease = {};
      deliver(e, error_result(e, std::current_exception()));
    }
  };

  auto schedule_retry = [&](footer_entry& e, std::string const& retry_after) {
    if (e.attempt + 1 < _config.max_retry_attempts) {
      CUCASCADE_LOG_WARN(
        "rest_reactor::resolve_footer_batch: retrying {}/{} after {} (attempt {}/{})",
        objects[e.pos].bucket,
        objects[e.pos].key,
        e.last_error,
        e.attempt + 1,
        _config.max_retry_attempts);
      e.retry_at =
        std::chrono::steady_clock::now() + compute_backoff(e.attempt, retry_after, _config);
      e.attempt += 1;
      e.stage = footer_entry_stage::backoff;
      ++backoff_count;
      retry_heap.emplace(e.retry_at, &e);
    } else {
      fail_entry(e, "exhausted retries (" + e.last_error + ")");
    }
  };

  auto finish_probe = [&](footer_entry& e, CURLcode rc, long status) {
    if (rc != CURLE_OK && rc != CURLE_WRITE_ERROR) {
      e.last_error = std::string(curl_easy_strerror(rc));
      if (!is_retriable_curl(rc)) {
        fail_entry(e, e.last_error);
        return;
      }
      schedule_retry(e, e.sink.retry_after);
      return;
    }
    if (status == 206) {
      auto const total = content_range_total(e.sink.content_range);
      auto const start = content_range_start(e.sink.content_range);
      if (total && start && *start <= *total && e.sink.data.size() == *total - *start) {
        footer_resolve_result r;
        r.index     = indices[e.pos];
        r.path      = paths[e.pos];
        r.window_lo = *start;
        r.object    = std::make_shared<rest_io_object>(
          paths[e.pos], objects[e.pos].bucket, objects[e.pos].key, *total, std::move(e.sink.etag));
        r.footer = make_leased_byte_span(budget, std::move(e.lease), std::move(e.sink.data));
        deliver(e, std::move(r));
        return;
      }
      // Unverifiable 206: like the blocking path, fall back to a HEAD.  The
      // body bytes and the lease are both returned — a HEAD delivers no
      // payload, and a discarded buffer must not outlive its reservation.
      e.sink    = suffix_sink{};
      e.lease   = {};
      e.kind    = footer_entry_kind::head;
      e.attempt = 0;
      submit(e);
      return;
    }
    if (status == 200 || status == 416) {
      e.sink    = suffix_sink{};
      e.lease   = {};
      e.kind    = footer_entry_kind::head;
      e.attempt = 0;
      submit(e);
      return;
    }
    if (is_retriable_status(status)) {
      e.last_error = "HTTP " + std::to_string(status);
      schedule_retry(e, e.sink.retry_after);
      return;
    }
    fail_entry(e, "HTTP " + std::to_string(status));
  };

  auto finish_head = [&](footer_entry& e, CURLcode rc, long status, curl_off_t content_length) {
    if (rc == CURLE_OK && status == 200) {
      if (content_length < 0) {
        fail_entry(e, "missing Content-Length");
        return;
      }
      footer_resolve_result r;
      r.index  = indices[e.pos];
      r.path   = paths[e.pos];
      r.object = std::make_shared<rest_io_object>(paths[e.pos],
                                                  objects[e.pos].bucket,
                                                  objects[e.pos].key,
                                                  static_cast<size_t>(content_length),
                                                  std::move(e.head.etag));
      deliver(e, std::move(r));
      return;
    }
    e.last_error =
      rc != CURLE_OK ? std::string(curl_easy_strerror(rc)) : ("HTTP " + std::to_string(status));
    bool const retriable =
      (rc != CURLE_OK && is_retriable_curl(rc)) || (rc == CURLE_OK && is_retriable_status(status));
    if (!retriable) {
      fail_entry(e, e.last_error);
      return;
    }
    schedule_retry(e, e.head.retry_after);
  };

  auto process_completions = [&] {
    int msgs_left = 0;
    while (CURLMsg* msg = curl_multi_info_read(multi.get(), &msgs_left)) {
      // After the first callback throw (or a stop), nothing more may be
      // delivered as success — undrained completions stay attached and fall
      // to the cancel sweep.
      if (callback_error || stop.stop_requested()) { break; }
      if (msg->msg != CURLMSG_DONE) { continue; }
      CURL* h           = msg->easy_handle;
      CURLcode const rc = msg->data.result;
      void* priv        = nullptr;
      curl_easy_getinfo(h, CURLINFO_PRIVATE, &priv);
      auto& e     = *static_cast<footer_entry*>(priv);
      long status = 0;
      curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
      curl_off_t content_length = -1;
      if (e.kind == footer_entry_kind::head) {
        curl_easy_getinfo(h, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &content_length);
      }
      CUCASCADE_CURLM_CHECK(curl_multi_remove_handle(multi.get(), h));
      e.easy.reset();
      e.headers.reset();
      --active;
      if (e.kind == footer_entry_kind::probe) {
        finish_probe(e, rc, status);
      } else {
        finish_head(e, rc, status, content_length);
      }
    }
  };

  // Returns false when a blocking budget wait was cut short by @p stop.
  auto start_pending = [&] {
    while (!callback_error && !stop.stop_requested() && active < max_inflight &&
           next_to_start < entries.size()) {
      auto& e = entries[next_to_start];
      if (window != 0) {
        exec::admission_control::slot lease;
        if (active > 0 || backoff_count > 0) {
          // Never block on budget while a transfer or a due retry could
          // still make progress and release bytes.
          lease = budget->try_acquire(window);
          if (!lease) { return true; }
        } else {
          lease = budget->acquire(window, stop);
          if (!lease) { return false; }
        }
        e.lease = std::move(lease);
      }
      ++next_to_start;
      submit(e);
    }
    return true;
  };

  auto resubmit_due = [&] {
    auto const now = std::chrono::steady_clock::now();
    while (!callback_error && !stop.stop_requested() && !retry_heap.empty() &&
           active < max_inflight) {
      auto const [due, e] = retry_heap.top();
      if (e->stage != footer_entry_stage::backoff) {
        retry_heap.pop();
        continue;
      }
      if (due > now) { break; }
      retry_heap.pop();
      --backoff_count;
      submit(*e);
    }
  };

  auto poll_timeout_ms = [&] {
    long timeout = 100;
    while (!retry_heap.empty() && retry_heap.top().second->stage != footer_entry_stage::backoff) {
      retry_heap.pop();
    }
    if (!retry_heap.empty()) {
      auto const dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                        retry_heap.top().first - std::chrono::steady_clock::now())
                        .count();
      timeout = std::min(timeout, std::max<long>(1, static_cast<long>(dt)));
    }
    return timeout;
  };

  try {
    while (undelivered > 0) {
      if (stop.stop_requested() || callback_error) {
        cancel_remaining();
        break;
      }
      resubmit_due();
      if (!start_pending()) {
        cancel_remaining();
        break;
      }
      if (undelivered == 0 || stop.stop_requested() || callback_error) { continue; }
      int running = 0;
      CUCASCADE_CURLM_CHECK(curl_multi_perform(multi.get(), &running));
      process_completions();
      if (undelivered == 0 || stop.stop_requested() || callback_error) { continue; }
      int numfds = 0;
      CUCASCADE_CURLM_CHECK(
        curl_multi_poll(multi.get(), nullptr, 0, static_cast<int>(poll_timeout_ms()), &numfds));
    }
  } catch (...) {
    // Driver failure (a curl multi error, an allocation failure): deliver
    // the cancel sweep first so exactly-once holds even here, then surface
    // the driver's own exception, not a callback's.
    cancel_remaining();
    throw;
  }

  if (callback_error) { std::rethrow_exception(callback_error); }
}

// ---------------------------------------------------------------------------
// capabilities / factory
// ---------------------------------------------------------------------------

bool rest_reactor::supports(std::string_view path)
{
  try {
    auto const parsed = cucascade::io::parse(path);
    return parsed.scheme == "s3";
  } catch (...) {
    return false;
  }
}

std::unique_ptr<rest_reactor::io_object_type> rest_reactor::create_io_object(std::string /*path*/)
{
  // The object size requires a HEAD round-trip and the authorizer, both of
  // which live on the reactor instance — see rest_ioctx::create_io_object.
  throw std::logic_error(
    "rest_reactor::create_io_object: use rest_ioctx::create_io_object (needs HEAD + authorizer)");
}

std::vector<byte_range> rest_reactor::align_and_coalesce(std::span<const byte_range> ranges,
                                                         std::optional<size_t> alignment)
{
  // No physical block alignment for REST: honor a caller alignment >= 1 as a
  // lower bound, otherwise treat alignment as 1 (byte) — i.e. pure coalescing.
  size_t const align = std::max<size_t>(alignment.value_or(1), 1);

  std::vector<byte_range> aligned;
  aligned.reserve(ranges.size());
  for (auto const& r : ranges) {
    if (r.size() <= 0) { continue; }
    auto const offset  = static_cast<size_t>(r.offset());
    auto const end     = offset + static_cast<size_t>(r.size());
    size_t const start = (offset / align) * align;
    size_t const stop  = ((end + align - 1) / align) * align;
    aligned.emplace_back(static_cast<int64_t>(start), static_cast<int64_t>(stop - start));
  }
  if (aligned.empty()) { return aligned; }

  std::sort(aligned.begin(), aligned.end(), [](auto const& a, auto const& b) {
    return a.offset() < b.offset();
  });

  std::vector<byte_range> coalesced;
  coalesced.reserve(aligned.size());
  coalesced.push_back(aligned.front());
  for (size_t i = 1; i < aligned.size(); ++i) {
    auto& last            = coalesced.back();
    auto const last_start = static_cast<size_t>(last.offset());
    auto const last_end   = last_start + static_cast<size_t>(last.size());
    auto const cur_start  = static_cast<size_t>(aligned[i].offset());
    auto const cur_end    = cur_start + static_cast<size_t>(aligned[i].size());
    if (cur_start <= last_end) {  // overlap or adjacency
      size_t const new_end = std::max(last_end, cur_end);
      last                 = {last.offset(), static_cast<int64_t>(new_end - last_start)};
    } else {
      coalesced.push_back(aligned[i]);
    }
  }
  return coalesced;
}

}  // namespace cucascade::io::rest
