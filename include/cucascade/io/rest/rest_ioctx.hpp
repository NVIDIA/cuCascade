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

#include <cucascade/exec/admission_control.hpp>
#include <cucascade/io/rest/object_store_lister.hpp>
#include <cucascade/io/rest/rest_reactor.hpp>
#include <cucascade/io/rest/s3/list_parser.hpp>
#include <cucascade/io/templated_ioctx.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace cucascade::io::rest {

// ---------------------------------------------------------------------------
// rest_ioctx
// ---------------------------------------------------------------------------

/**
 * @brief RESTful object-store (s3://) ioctx. Specialisation of
 *        @c templated_ioctx<rest_reactor>.
 *
 * Owns a pool of @c rest_reactor workers (load-balanced by the base) that share
 * one @p authorizer.  Overrides @c create_io_object to resolve an object's size
 * via a blocking HEAD before constructing the @c rest_io_object — the static
 * reactor factory cannot do this since it needs the authorizer + a round-trip.
 */
class rest_ioctx : public templated_ioctx<rest_reactor> {
 public:
  /// Build a pool of @p n_reactors reactors, all sharing @p ctx (one context per
  /// pool: it carries the per-reactor @c config, the presigning authorizer, and
  /// the pinned bounce-staging resource — all of which must outlive this ioctx).
  /// The ioctx config is sourced from the reactors themselves — see
  /// @c templated_ioctx.
  ///
  /// @p n_reactors is a total; 0 sizes the pool from the egress NIC (one reactor per IRQ CPU,
  /// pinned to it at the highest permitted priority, when the NIC's queue IRQs are steered to a
  /// strict subset of its NUMA node's cores; the default 2 otherwise).
  /// @throws std::invalid_argument when the config is invalid (@c config::validate) or
  /// @c rest.fan_out exceeds a positive @p n_reactors (the default fan-out of 2 is clamped to a
  /// one-reactor pool instead, as the former hard-coded fan-out was).
  rest_ioctx(std::size_t n_reactors, std::shared_ptr<rest_reactor::reactor_context> ctx);
  ~rest_ioctx() override;

  rest_ioctx(rest_ioctx const&)            = delete;
  rest_ioctx& operator=(rest_ioctx const&) = delete;
  rest_ioctx(rest_ioctx&&)                 = delete;
  rest_ioctx& operator=(rest_ioctx&&)      = delete;

  /// Counters of the whole pool: the reactors' @c transport_stats summed (extremes for the MSS
  /// and in-flight peaks).  A snapshot; safe while the pool runs.
  [[nodiscard]] transport_stats stats() const noexcept;

  [[nodiscard]] io_context_type type() const noexcept override { return io_context_type::restful; }

  /// Stream a bucket's ListObjectsV2 pages under @p prefix to @p sink, one call
  /// per page (a page holds at most 1000 entries, so peak memory is one page
  /// regardless of bucket population).  @p sink returns false to stop early —
  /// no further LIST requests are issued.  @p page_size is clamped to [1,1000]
  /// (0 and >1000 mean 1000).  Throws (never truncates) on a truncated page
  /// without a continuation token, and once more than @p max_scanned entries
  /// have been scanned across pages (bounds time / request count on a prefix
  /// whose population dwarfs the caller's matches).
  void list_objects_paged(std::string_view bucket,
                          std::string_view prefix,
                          std::size_t page_size,
                          std::function<bool(s3::list_objects_v2_page const&)> const& sink,
                          std::optional<std::size_t> max_scanned = std::nullopt);

  /// Whole-listing convenience over @c list_objects_paged: every object under
  /// @p prefix, in document order, with sizes.  Throws (never truncates) when
  /// the accumulated entries would exceed @p max_keys — a partial key set would
  /// resolve a glob to a silently incomplete table.
  [[nodiscard]] std::vector<s3::list_entry> list_objects(
    std::string_view bucket,
    std::string_view prefix,
    std::size_t page_size               = 1000,
    std::optional<std::size_t> max_keys = std::nullopt);

  /// The configured matched cap (@c config.list_max_matches) — exposed so a
  /// glob layer one level up can bound its match set without a reactor handle.
  /// Falls back to the built-in default when the pool is empty (never in
  /// practice).
  [[nodiscard]] std::size_t list_max_matches() const;

  /// Resolve many objects' footers concurrently.  Per-entry semantics are
  /// IDENTICAL to @c open_io_object(path, parquet_footer_probe): one verified
  /// suffix GET; 200/416/unverifiable-206 fall back to a HEAD supplying
  /// size+tag; the same retry policy per entry, never stalling siblings.  The
  /// exchanges run on the first reactor's connections (on a private helper
  /// transport when the pool is not running), and @p on_result is invoked ON
  /// THE CALLER'S THREAD, SERIALLY, as each entry lands — completion order, no
  /// all-entries barrier.
  ///
  /// Every input occurrence is delivered exactly once (duplicates delivered
  /// per occurrence, disambiguated by index); no callback runs after the
  /// call returns.  On @p stop, in-flight transfers abort and every
  /// undelivered entry receives one std::system_error(operation_canceled) —
  /// including a batch cancelled while queued behind another batch (one
  /// active batch per ioctx; concurrent calls FIFO-serialize).  If
  /// @p on_result throws, the remaining entries are cancelled (delivered as
  /// canceled, their callback throws suppressed) and the first exception is
  /// rethrown after the sweep.  Throws directly only on submission errors:
  /// an empty batch, the API disabled via
  /// @c config::footer_resolve_max_inflight == 0, or an explicit
  /// @c footer_resolve_stash_budget smaller than @c footer_probe_bytes.  An
  /// unparsable or non-s3 path is a per-entry error, not a batch error.
  /// Should the transfer driver itself fail (its helper thread cannot be
  /// started, or it fails mid-batch), every undelivered entry is delivered as
  /// canceled before that failure is rethrown — exactly-once holds on every
  /// exit path.  This ioctx must outlive the call.
  void resolve_footer_objects(std::span<std::string const> paths,
                              std::function<void(footer_resolve_result)> const& on_result,
                              std::stop_token stop = {});

  /// Live / high-water bytes reserved from the footer-resolve stash budget.
  /// Both are 0 when the batched footer API is disabled or has never run on
  /// this ioctx.  Safe to call while the pool is running.
  [[nodiscard]] std::size_t footer_stash_reserved_bytes() const noexcept;
  [[nodiscard]] std::size_t footer_stash_reserved_peak_bytes() const noexcept;

  /// Open every reactor's connection pool against @p bucket_url's bucket, so the
  /// query's first reads find pooled connections instead of paying TCP+TLS on
  /// the hot path.  Fans the work out to the reactors and returns immediately:
  /// each reactor's connection cache is thread-confined, so only its own worker
  /// can fill it.
  ///
  /// Rate-limited rather than run once, because what goes stale is the
  /// connection, not the bucket.  @c conn_max_age is a hard cap on reusing a
  /// pooled connection (its idle cutoff), so a pool warmed before an idle
  /// gap longer than that is cold again by the next query -- a warm-once flag
  /// would serve the first query and no other.  Re-warms when the endpoint
  /// changes or the last warm-up is at least @c conn_max_age old, which is free
  /// for back-to-back queries and self-correcting for spaced-out ones.
  void warmup(std::string_view bucket_url) noexcept override;

 protected:
  /// This pool's placement (plan and NIC), computed at construction and shared with its
  /// reactors; never stored on the reactor_context, which several pools may share.
  std::shared_ptr<detail::pool_placement const> _placement;

  /// Open @p path (s3://bucket/key) for writing: a @c rest_write_object whose
  /// writes become an S3 upload, committed by @c flush (docs/io/rest-transport.md
  /// "Writes").  No request is sent here.  @c write_mode::create_or_truncate and
  /// @c create_new are supported (the latter commits with `If-None-Match: *`);
  /// the modes that keep existing content fail with @c not_supported (an S3
  /// object cannot be modified in place).
  ///
  /// Lifetime: flush or drop every write object before destroying this ioctx.  An object that
  /// outlives it gives back its part buffers when the ioctx is destroyed (they belong to the
  /// context's host memory resource), can no longer write or commit (@c operation_canceled),
  /// and its upload is not aborted (bucket lifecycle rules collect it).
  std::shared_ptr<io_object> create_io_object_for_write(std::string path,
                                                        write_open_options options) override;

  /// Commit @p object: upload what is left and complete the upload (blocking): one flush slice
  /// through @ref flush_async_io.  The path's cached bytes and metadata are invalidated once
  /// the commit settled.
  void flush_io(io_object const& object) override;

  /// As @ref flush_io, asynchronously: one flush slice on the least-backlogged reactor, the
  /// path invalidated once it settled and before the returned future resolves.
  exec::semi_future<void> flush_async_io(io_object const& object,
                                         io_priority priority) noexcept override;

  /// Cut every write slice that straddles a part boundary at it (one write operation is never
  /// spread over two S3 parts), then dispatch as the base does.
  exec::semi_future<std::size_t> mixed_writev_async_io(
    io_object const& object, std::vector<prepared_io_slice>&& slices) noexcept override;

  /// Backend hook invoked by @c ioctx::open_io_object: parse @p path
  /// (s3://bucket/key), HEAD it for the size, and build a @c rest_io_object.
  /// Throws on a non-s3 scheme or a failed HEAD.
  std::shared_ptr<io_object> create_io_object(std::string path) override;

  /// Hinted open: @c open_hint::parquet_footer_probe resolves the size AND stashes
  /// the object's trailing bytes in one suffix-range GET, stashed on the
  /// returned io_object; every other hint falls back to the plain HEAD path above.
  std::shared_ptr<io_object> create_io_object(std::string path, open_hint hint) override;

  /// Known-size open: the caller already learned the object's size (e.g. from a
  /// ListObjectsV2 response), so the io_object is built with ZERO network — no
  /// HEAD, no probe.
  std::shared_ptr<io_object> create_io_object(std::string path, std::uint64_t known_size) override;

 private:
  rest_ioctx(std::shared_ptr<detail::pool_placement const> placement,
             std::shared_ptr<rest_reactor::reactor_context> ctx);

  /// Resolve @p path with a single suffix-range GET: it discovers the size and
  /// stashes the object's trailing bytes on the returned io_object so cuDF's
  /// footer reads are served locally.  Falls back to a plain HEAD (no stash)
  /// when the response is unusable.
  std::shared_ptr<io_object> create_footer_probe_object(std::string path);

  object_store_lister _lister;
  /// Where uploads destroyed uncommitted send their AbortMultipartUpload (closed first in the
  /// destructor, before the reactors go).
  std::shared_ptr<detail::abort_sink> _abort_sink;
  std::atomic<std::size_t> _control_next{0};

  /// The effective resolve_footer_objects concurrency cap: the configured knob, or one
  /// reactor's max_connections under footer_resolve_auto (a batch runs on one reactor).  0 =
  /// the API is disabled.
  [[nodiscard]] std::size_t footer_resolve_inflight_cap() const;

  /// The reactor a control-plane call (HEAD, footer probe, LIST, footer batch) runs on: the
  /// least backlogged, ties rotated, so control traffic does not pile onto reactor 0.
  [[nodiscard]] rest_reactor& control_reactor();

  // Batched-footer-resolve coordination: one active batch per ioctx, later
  // calls FIFO-parked on the ticket queue (stop-aware — a queued batch whose
  // token fires is removed without ever becoming active).  _footer_budget is
  // created in the constructor and never reassigned, so the stash gauges may
  // read it without the mutex.
  mutable std::mutex _footer_resolve_mutex;
  std::condition_variable_any _footer_resolve_cv;
  std::deque<std::uint64_t> _footer_resolve_queue;
  std::uint64_t _footer_resolve_next_ticket{0};
  bool _footer_resolve_active{false};
  std::shared_ptr<exec::admission_control> _footer_budget;

  /// Warn once per path when @p obj has no strong ETag (none, weak or malformed).
  /// Such an object cannot be revalidated across opens, so its bytes and
  /// metadata are cached only within this open.  Returns @p obj unchanged.
  std::shared_ptr<io_object> warn_if_no_strong_etag(std::shared_ptr<io_object> obj);

  /// Guards the warm-up rate limiter.  Contended once per query at most, and
  /// never on a read path.
  std::mutex _warm_mtx;
  std::mutex _weak_etag_mtx;
  std::vector<std::string> _weak_etag_paths;  ///< paths already warned about, bounded
  /// Bucket the pools were last warmed against, and when.  An unset time means
  /// "never warmed", which no elapsed comparison can express.
  std::string _warmed_bucket;
  std::optional<std::chrono::steady_clock::time_point> _warmed_at;
};

}  // namespace cucascade::io::rest
