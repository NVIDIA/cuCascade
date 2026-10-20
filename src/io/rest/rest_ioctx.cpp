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

#include "io/rest/s3/multipart.hpp"
#include "io/rest/worker/upload.hpp"

#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/io/uri_parser.hpp>
#include <cucascade/log/logging.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace cucascade::io::rest {

rest_ioctx::rest_ioctx(std::size_t n_reactors, std::shared_ptr<rest_reactor::reactor_context> ctx)
  : templated_ioctx<rest_reactor>(n_reactors,
                                  [ctx = std::move(ctx), i = 0]() mutable {
                                    return std::make_unique<rest_reactor>(
                                      ctx, "rest-" + std::to_string(i++));
                                  }),
    _lister(
      [this](std::string_view bucket, std::string_view prefix, std::string_view canonical_query) {
        if (_reactors.empty()) {
          throw std::runtime_error("rest_ioctx::list_objects: no reactors");
        }
        return _reactors.front()->list_page(bucket, prefix, canonical_query);
      },
      _reactors.empty() ? s3::default_max_scanned_objects
                        : _reactors.front()->get_config().list_max_scanned,
      _reactors.empty() ? s3::default_max_list_objects
                        : _reactors.front()->get_config().list_max_matches,
      "rest_ioctx::list_objects")
{
  // Created once here and never reassigned: payload leases capture it by
  // shared_ptr (so they may outlive this ioctx) and the stash gauges read it
  // without the coordination mutex.
  if (std::size_t const inflight = footer_resolve_inflight_cap(); inflight > 0) {
    auto const& cfg   = _reactors.front()->get_config();
    std::size_t bytes = cfg.footer_resolve_stash_budget;
    if (bytes == config::footer_resolve_auto) { bytes = 2 * inflight * cfg.footer_probe_bytes; }
    _footer_budget = std::make_shared<exec::admission_control>(std::max<std::size_t>(bytes, 1));
  }
}

std::size_t rest_ioctx::footer_stash_reserved_bytes() const noexcept
{
  return _footer_budget ? _footer_budget->reserved() : 0;
}

std::size_t rest_ioctx::footer_stash_reserved_peak_bytes() const noexcept
{
  return _footer_budget ? _footer_budget->peak_reserved() : 0;
}

std::size_t rest_ioctx::footer_resolve_inflight_cap() const
{
  if (_reactors.empty()) { return 0; }
  auto const& cfg = _reactors.front()->get_config();
  if (cfg.footer_resolve_max_inflight == config::footer_resolve_auto) {
    return std::max<std::size_t>(1, _reactors.size() * cfg.max_connections);
  }
  return cfg.footer_resolve_max_inflight;
}

void rest_ioctx::resolve_footer_objects(std::span<std::string const> paths,
                                        std::function<void(footer_resolve_result)> const& on_result,
                                        std::stop_token stop)
{
  if (paths.empty()) {
    throw std::invalid_argument("rest_ioctx::resolve_footer_objects: empty batch");
  }
  if (_reactors.empty()) {
    throw std::runtime_error("rest_ioctx::resolve_footer_objects: no reactors");
  }
  std::size_t const max_inflight = footer_resolve_inflight_cap();
  if (max_inflight == 0 || !_footer_budget) {
    throw std::invalid_argument(
      "rest_ioctx::resolve_footer_objects: disabled (footer_resolve_max_inflight == 0)");
  }
  if (_footer_budget->budget() < _reactors.front()->get_config().footer_probe_bytes) {
    // A budget below one probe window cannot admit any entry without
    // over-committing past the cap, so it cannot be honored as a hard cap.
    throw std::invalid_argument(
      "rest_ioctx::resolve_footer_objects: footer_resolve_stash_budget smaller than "
      "footer_probe_bytes");
  }

  // Parse up front; a bad scheme is a per-entry error (isolation), not a
  // batch error.
  std::vector<std::string> valid_paths;
  std::vector<object_ref> valid_objects;
  std::vector<std::size_t> valid_indices;
  std::vector<std::pair<std::size_t, std::exception_ptr>> parse_errors;
  valid_paths.reserve(paths.size());
  valid_objects.reserve(paths.size());
  valid_indices.reserve(paths.size());
  for (std::size_t i = 0; i < paths.size(); ++i) {
    try {
      auto parsed = cucascade::io::parse(paths[i]);
      if (parsed.scheme != "s3") {
        throw std::invalid_argument("rest_ioctx::resolve_footer_objects: unsupported scheme '" +
                                    parsed.scheme + "'");
      }
      valid_paths.push_back(paths[i]);
      valid_objects.push_back(object_ref{std::move(parsed.host), std::move(parsed.path)});
      valid_indices.push_back(i);
    } catch (...) {
      // A malformed path joins bad-scheme paths as a per-entry error —
      // parsing must never cost the batch its exactly-once delivery.
      parse_errors.emplace_back(i, std::current_exception());
    }
  }

  std::exception_ptr callback_error;
  auto deliver_guarded = [&](footer_resolve_result&& r) {
    try {
      on_result(std::move(r));
    } catch (...) {
      // First exception wins; later throws during a cancel sweep are
      // suppressed.
      if (!callback_error) { callback_error = std::current_exception(); }
    }
  };
  auto canceled = [] {
    return std::make_exception_ptr(
      std::system_error(std::make_error_code(std::errc::operation_canceled),
                        "rest_ioctx::resolve_footer_objects: canceled"));
  };

  // FIFO admission: one active batch per ioctx; the wait is stop-aware, so a
  // queued batch whose token fires is removed without ever becoming active.
  {
    std::unique_lock lk(_footer_resolve_mutex);
    std::uint64_t const ticket = _footer_resolve_next_ticket++;
    _footer_resolve_queue.push_back(ticket);
    bool const admitted = _footer_resolve_cv.wait(lk, stop, [&] {
      return !_footer_resolve_active && !_footer_resolve_queue.empty() &&
             _footer_resolve_queue.front() == ticket;
    });
    if (!admitted) {
      std::erase(_footer_resolve_queue, ticket);
      lk.unlock();
      _footer_resolve_cv.notify_all();
      for (std::size_t i = 0; i < paths.size(); ++i) {
        footer_resolve_result r;
        r.index = i;
        r.path  = paths[i];
        r.error = canceled();
        deliver_guarded(std::move(r));
      }
      if (callback_error) { std::rethrow_exception(callback_error); }
      return;
    }
    _footer_resolve_active = true;
    _footer_resolve_queue.pop_front();
  }

  struct gate_release {
    rest_ioctx* self;
    ~gate_release()
    {
      {
        std::lock_guard lk(self->_footer_resolve_mutex);
        self->_footer_resolve_active = false;
      }
      self->_footer_resolve_cv.notify_all();
    }
  } release{this};

  // Deliver parse failures first (serial, on this thread); if a callback
  // throws, every not-yet-delivered entry is swept as canceled and the first
  // exception is rethrown — same rule as the engine.
  std::size_t parse_pos = 0;
  for (; parse_pos < parse_errors.size() && !callback_error; ++parse_pos) {
    footer_resolve_result r;
    r.index = parse_errors[parse_pos].first;
    r.path  = paths[parse_errors[parse_pos].first];
    r.error = std::move(parse_errors[parse_pos].second);
    deliver_guarded(std::move(r));
  }
  if (callback_error) {
    for (std::size_t p = parse_pos; p < parse_errors.size(); ++p) {
      footer_resolve_result r;
      r.index = parse_errors[p].first;
      r.path  = paths[parse_errors[p].first];
      r.error = canceled();
      deliver_guarded(std::move(r));
    }
    for (std::size_t v = 0; v < valid_indices.size(); ++v) {
      footer_resolve_result r;
      r.index = valid_indices[v];
      r.path  = valid_paths[v];
      r.error = canceled();
      deliver_guarded(std::move(r));
    }
    std::rethrow_exception(callback_error);
  }

  if (valid_indices.empty()) { return; }

  _reactors.front()->resolve_footer_batch(
    valid_paths, valid_objects, valid_indices, max_inflight, _footer_budget, on_result, stop);
}

void rest_ioctx::list_objects_paged(
  std::string_view bucket,
  std::string_view prefix,
  std::size_t page_size,
  std::function<bool(s3::list_objects_v2_page const&)> const& sink,
  std::optional<std::size_t> max_scanned)
{
  if (_reactors.empty()) { throw std::runtime_error("rest_ioctx::list_objects: no reactors"); }
  _lister.list_objects_paged(bucket, prefix, page_size, sink, max_scanned);
}

std::vector<s3::list_entry> rest_ioctx::list_objects(std::string_view bucket,
                                                     std::string_view prefix,
                                                     std::size_t page_size,
                                                     std::optional<std::size_t> max_keys)
{
  return _lister.list_objects(bucket, prefix, page_size, max_keys);
}

std::size_t rest_ioctx::list_max_matches() const { return _lister.list_max_matches(); }

std::shared_ptr<io_object> rest_ioctx::create_io_object(std::string path)
{
  auto parsed = cucascade::io::parse(path);
  if (parsed.scheme != "s3") {
    throw std::invalid_argument("rest_ioctx::create_io_object: unsupported scheme '" +
                                parsed.scheme + "'");
  }
  if (_reactors.empty()) { throw std::runtime_error("rest_ioctx::create_io_object: no reactors"); }

  // A blocking HEAD on the caller thread (a one-time metadata round-trip) via
  // any reactor's authorizer — head_object uses a local easy handle and
  // does not touch worker state, so any reactor is equivalent.
  auto head = _reactors.front()->head_object(parsed.host, parsed.path);
  return note_unqualified(std::make_shared<rest_io_object>(std::move(path),
                                                           std::move(parsed.host),
                                                           std::move(parsed.path),
                                                           head.object_size,
                                                           std::move(head.etag)));
}

std::shared_ptr<io_object> rest_ioctx::note_unqualified(std::shared_ptr<io_object> obj)
{
  if (rest_io_object::is_strong_tag(obj->validation_tag())) { return obj; }
  constexpr std::size_t k_max_warned_paths = 64;
  auto const& path                         = obj->object_path();
  {
    std::lock_guard lk(_unqualified_mtx);
    if (std::ranges::find(_unqualified_paths, path) != _unqualified_paths.end()) { return obj; }
    if (_unqualified_paths.size() == k_max_warned_paths) { _unqualified_paths.clear(); }
    _unqualified_paths.push_back(path);
  }
  auto const tag = obj->validation_tag();
  CUCASCADE_LOG_WARN(
    "rest_ioctx: '{}' opened with {} ETag; its bytes and metadata are cached only within this open",
    path,
    tag.empty()             ? "no"
    : tag.starts_with("W/") ? "a weak"
                            : "a malformed");
  return obj;
}

std::shared_ptr<io_object> rest_ioctx::create_io_object(std::string path, open_hint hint)
{
  if (hint == open_hint::parquet_footer_probe) {
    return create_footer_probe_object(std::move(path));
  }
  return create_io_object(std::move(path));
}

std::shared_ptr<io_object> rest_ioctx::create_io_object(std::string path, std::uint64_t known_size)
{
  auto parsed = cucascade::io::parse(path);
  if (parsed.scheme != "s3") {
    throw std::invalid_argument("rest_ioctx::create_io_object: unsupported scheme '" +
                                parsed.scheme + "'");
  }
  // The size came from a ListObjectsV2 response: build the io_object with zero
  // network — no HEAD, no probe.
  return note_unqualified(std::make_shared<rest_io_object>(std::move(path),
                                                           std::move(parsed.host),
                                                           std::move(parsed.path),
                                                           static_cast<size_t>(known_size)));
}

std::shared_ptr<io_object> rest_ioctx::create_io_object_for_write(std::string path,
                                                                  write_open_options options)
{
  auto parsed = cucascade::io::parse(path);
  if (parsed.scheme != "s3") {
    throw std::invalid_argument("rest_ioctx::create_io_object_for_write: unsupported scheme '" +
                                parsed.scheme + "'");
  }
  if (_reactors.empty()) {
    throw std::runtime_error("rest_ioctx::create_io_object_for_write: no reactors");
  }
  if (options.mode == write_mode::create_or_open || options.mode == write_mode::open_existing) {
    throw std::system_error(std::make_error_code(std::errc::not_supported),
                            "rest_ioctx: an S3 object cannot be modified in place ('" + path +
                              "' opened in a mode that keeps its content)");
  }
  // The part size is the write split (templated_ioctx cuts every write slice at its multiples,
  // so one write operation never spans two parts), or a multiple of it when size_hint needs
  // more than 10000 parts.
  auto const granule = write_granule();
  auto const split =
    std::max(granule, _config.write_split_bytes - _config.write_split_bytes % granule);
  s3::upload_state::options opts;
  opts.part_size  = s3::choose_part_size(split, options.size_hint, split);
  opts.create_new = options.mode == write_mode::create_new;
  auto upload     = std::make_shared<detail::upload_session>(
    object_ref{parsed.host, parsed.path}, path, opts, _reactors.front()->write_inbox());
  return std::make_shared<rest_write_object>(
    std::move(path), std::move(parsed.host), std::move(parsed.path), std::move(upload));
}

void rest_ioctx::flush_io(io_object const& object)
{
  auto const* typed = dynamic_cast<rest_write_object const*>(&object);
  if (typed == nullptr) {
    throw std::system_error(
      std::make_error_code(std::errc::bad_file_descriptor),
      "rest_ioctx::flush: '" + object.object_path() + "' was not opened for writing");
  }
  if (_reactors.empty()) { throw std::runtime_error("rest_ioctx::flush: no reactors"); }
  _reactors.front()->commit(*typed);
}

exec::semi_future<std::size_t> rest_ioctx::mixed_writev_async_io(
  io_object const& object, std::vector<prepared_io_slice>&& slices) noexcept
{
  // The base cuts only slices larger than the split, at its multiples; a smaller one may still
  // straddle a multiple. Part sizes are multiples of the split (create_io_object_for_write), so
  // cutting at every multiple keeps each write operation inside one part. A slice whose range or
  // source is invalid is left alone: the base's validation rejects it.
  try {
    auto const granule = write_granule();
    auto const piece =
      std::max(granule, _config.write_split_bytes - _config.write_split_bytes % granule);
    std::vector<prepared_io_slice> cut;
    cut.reserve(slices.size() + 1);
    for (auto& slice : slices) {
      auto const begin     = slice.offset();
      auto const size      = slice.size();
      auto* const host     = std::get_if<std::uint8_t*>(&slice.h_buffer.buffer);
      bool const source    = slice.has_device_request() || (host != nullptr && *host != nullptr);
      bool const straddles = slice.is_write() && source && size > 1 && size <= piece &&
                             size - 1 <= std::numeric_limits<std::size_t>::max() - begin &&
                             begin / piece != (begin + size - 1) / piece;
      if (!straddles) {
        cut.push_back(std::move(slice));
        continue;
      }
      auto const boundary = begin - begin % piece + piece;
      auto tail           = slice;
      slice.rng           = range{begin, boundary - begin};
      tail.rng            = range{boundary, begin + size - boundary};
      if (tail.has_device_request()) {
        tail.d_buffer.data += boundary - begin;
      } else {
        tail.h_buffer.buffer = *host + (boundary - begin);
      }
      cut.push_back(std::move(slice));
      cut.push_back(std::move(tail));
    }
    slices = std::move(cut);
  } catch (...) {
    return exec::make_semi_future<std::size_t>(std::current_exception());
  }
  return templated_ioctx<rest_reactor>::mixed_writev_async_io(object, std::move(slices));
}

namespace {

/// The bucket out of an @c s3:// URL, whether or not it names an object.
///
/// @c io::parse cannot serve this: it rejects a bucket-only URL with "empty
/// object key", which is exactly the shape a warm-up takes -- warming is about
/// the endpoint, and naming a data file to reach it would be beside the point.
std::optional<std::string> bucket_of(std::string_view url)
{
  constexpr std::string_view k_scheme = "s3://";
  if (url.size() <= k_scheme.size()) { return std::nullopt; }
  for (std::size_t i = 0; i < k_scheme.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(url[i])) != k_scheme[i]) { return std::nullopt; }
  }
  auto const rest   = url.substr(k_scheme.size());
  auto const bucket = rest.substr(0, rest.find('/'));
  if (bucket.empty()) { return std::nullopt; }
  return std::string{bucket};
}

}  // namespace

void rest_ioctx::warmup(std::string_view bucket_url) noexcept
{
  try {
    // Only the bucket is read: an object key, if the caller passed one, names
    // nothing the connection pool cares about.
    auto const bucket = bucket_of(bucket_url);
    if (!bucket.has_value()) { return; }

    // Connections are per-endpoint, so the bucket -- not the object -- is the
    // identity that decides whether a warm-up is redundant. A thousand-file
    // scan calling this per file collapses to one round.
    {
      std::lock_guard lk{_warm_mtx};
      auto const now = std::chrono::steady_clock::now();
      // A pooled connection idle for longer than conn_max_age is closed rather
      // than reused, so a warm-up older than that is stale; 0 (the 4 s
      // default) has no honest horizon of its own: fall back to a minute.
      auto const stale_after =
        _config.conn_max_age.count() > 0
          ? std::chrono::duration_cast<std::chrono::steady_clock::duration>(_config.conn_max_age)
          : std::chrono::duration_cast<std::chrono::steady_clock::duration>(
              std::chrono::seconds{60});
      bool const same_bucket = _warmed_bucket == *bucket;
      if (same_bucket && _warmed_at.has_value() && now - *_warmed_at < stale_after) { return; }
      _warmed_bucket = *bucket;
      _warmed_at     = now;
    }

    for (auto& reactor : _reactors) {
      if (reactor) { reactor->warmup(*bucket); }
    }
  } catch (...) {  // NOLINT(bugprone-empty-catch)
    // Warming is an optimization; a query that would have run without it still
    // runs. Nothing here is worth failing a read over.
  }
}

std::shared_ptr<io_object> rest_ioctx::create_footer_probe_object(std::string path)
{
  auto parsed = cucascade::io::parse(path);
  if (parsed.scheme != "s3") {
    throw std::invalid_argument("rest_ioctx::create_io_object: unsupported scheme '" +
                                parsed.scheme + "'");
  }
  if (_reactors.empty()) { throw std::runtime_error("rest_ioctx::create_io_object: no reactors"); }

  // One suffix-range GET resolves the size and stashes the footer; cuDF's
  // trailer/footer reads are then served from the stash by host_read.
  footer_probe probe = _reactors.front()->fetch_footer_suffix(
    parsed.host, parsed.path, _reactors.front()->get_config().footer_probe_bytes);
  if (!probe.bytes) {
    // Unusable suffix response (200 full body, 416, missing / "*" Content-Range):
    // fall back to a plain HEAD for the size, with no stash.
    auto head = _reactors.front()->head_object(parsed.host, parsed.path);
    return note_unqualified(std::make_shared<rest_io_object>(std::move(path),
                                                             std::move(parsed.host),
                                                             std::move(parsed.path),
                                                             head.object_size,
                                                             std::move(head.etag)));
  }
  return note_unqualified(std::make_shared<rest_io_object>(std::move(path),
                                                           std::move(parsed.host),
                                                           std::move(parsed.path),
                                                           probe.object_size,
                                                           probe.window_lo,
                                                           probe.bytes,
                                                           std::move(probe.etag)));
}

}  // namespace cucascade::io::rest
