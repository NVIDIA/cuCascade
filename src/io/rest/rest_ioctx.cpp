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

#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/io/uri_parser.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
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
  // shared_ptr (so they may outlive this ioctx) and perf_snapshot() reads it
  // without the coordination mutex.
  if (std::size_t const inflight = footer_resolve_inflight_cap(); inflight > 0) {
    auto const& cfg   = _reactors.front()->get_config();
    std::size_t bytes = cfg.footer_resolve_stash_budget;
    if (bytes == config::footer_resolve_auto) { bytes = 2 * inflight * cfg.footer_probe_bytes; }
    _footer_budget = std::make_shared<exec::admission_control>(std::max<std::size_t>(bytes, 1));
  }
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

rest_perf_snapshot rest_ioctx::perf_snapshot() const noexcept
{
  rest_perf_snapshot agg;
  for (auto const& r : _reactors) {
    auto const s = r->perf_snapshot();
    agg.chunk_get_ns_total += s.chunk_get_ns_total;
    agg.chunk_get_count += s.chunk_get_count;
    agg.chunk_get_ns_max = std::max(agg.chunk_get_ns_max, s.chunk_get_ns_max);
    agg.queue_wait_ns_total += s.queue_wait_ns_total;
    agg.queue_wait_count += s.queue_wait_count;
    if (s.ttfb_ns != 0 && (agg.ttfb_ns == 0 || s.ttfb_ns < agg.ttfb_ns)) {
      agg.ttfb_ns = s.ttfb_ns;  // smallest non-zero first-GET latency across the pool
    }
    agg.h2d_observed_ns_total += s.h2d_observed_ns_total;
    agg.h2d_observed_count += s.h2d_observed_count;
    agg.h2d_observed_ns_max = std::max(agg.h2d_observed_ns_max, s.h2d_observed_ns_max);
    agg.retries_total += s.retries_total;
    agg.terminal_failures_total += s.terminal_failures_total;
    agg.device_stream_sync_total += s.device_stream_sync_total;
    agg.payload_bytes_read_total += s.payload_bytes_read_total;
    agg.blocking_host_get_count += s.blocking_host_get_count;
    agg.blocking_host_get_wall_ns_total += s.blocking_host_get_wall_ns_total;
    agg.blocking_host_get_wall_ns_max =
      std::max(agg.blocking_host_get_wall_ns_max, s.blocking_host_get_wall_ns_max);
  }
  if (_footer_budget) {
    agg.footer_stash_reserved_bytes      = _footer_budget->reserved();
    agg.footer_stash_reserved_peak_bytes = _footer_budget->peak_reserved();
  }
  return agg;
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
  // any reactor's authorizer — head_object uses a local easy handle and does
  // not touch worker state, so any reactor is equivalent.
  auto head = _reactors.front()->head_object(parsed.host, parsed.path);
  return std::make_shared<rest_io_object>(std::move(path),
                                          std::move(parsed.host),
                                          std::move(parsed.path),
                                          head.object_size,
                                          std::move(head.etag));
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
  return std::make_shared<rest_io_object>(std::move(path),
                                          std::move(parsed.host),
                                          std::move(parsed.path),
                                          static_cast<size_t>(known_size));
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
    return std::make_shared<rest_io_object>(std::move(path),
                                            std::move(parsed.host),
                                            std::move(parsed.path),
                                            head.object_size,
                                            std::move(head.etag));
  }
  return std::make_shared<rest_io_object>(std::move(path),
                                          std::move(parsed.host),
                                          std::move(parsed.path),
                                          probe.object_size,
                                          probe.window_lo,
                                          probe.bytes,
                                          std::move(probe.etag));
}

}  // namespace cucascade::io::rest
