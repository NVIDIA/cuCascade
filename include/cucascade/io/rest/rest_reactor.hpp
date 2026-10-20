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

#include <cucascade/cuda/stream.hpp>
#include <cucascade/exec/admission_control.hpp>
#include <cucascade/io/cache/types.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/rest/authorizer.hpp>
#include <cucascade/io/rest/config.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
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

namespace detail {
class mailbox;
class shared_services;
class upload_session;
}  // namespace detail

/// Parse the total object length out of a Content-Range value of the form
/// "bytes <first>-<last>/<total>".  Returns nullopt when the unit is not
/// "bytes", the range is unsatisfied ("bytes */..."), or the total is unknown
/// ("*") — i.e. any response the footer probe cannot trust.
[[nodiscard]] std::optional<std::size_t> content_range_total(std::string_view content_range);

// ---------------------------------------------------------------------------
// shared_byte_span
// ---------------------------------------------------------------------------

namespace detail {

/// Owns a byte buffer plus a span over it.  Exists so @ref make_shared_byte_span
/// can hand out a shared_ptr to the *span* (via the aliasing constructor) while
/// the shared_ptr's control block keeps the *buffer* alive.  Never held
/// directly by callers.
struct byte_storage {
  std::vector<std::uint8_t> bytes;
  std::span<const std::uint8_t> view;

  // `bytes` is declared first, so it is already initialised when `view` binds
  // to it — the span never sees a moved-from buffer.
  explicit byte_storage(std::vector<std::uint8_t> b) : bytes(std::move(b)), view(bytes) {}

  // Non-copyable, non-movable: `view` points into `bytes`, so copying would
  // deep-copy the buffer and leave the copy's span aimed at the original's
  // allocation.  Only ever built in place by make_shared, so neither is needed.
  byte_storage(byte_storage const&)            = delete;
  byte_storage& operator=(byte_storage const&) = delete;
  byte_storage(byte_storage&&)                 = delete;
  byte_storage& operator=(byte_storage&&)      = delete;
};

}  // namespace detail

/// A shared, immutable view over a byte buffer.
///
/// Deliberately a span rather than a @c vector: consumers only ever read
/// through it (@c data / @c size / @c subspan), so exposing the container type —
/// and with it its allocator, growth policy and mutation API — would leak an
/// implementation detail into the interface.  Ownership still rides along: the
/// shared_ptr is built with the aliasing constructor, so the control block
/// retains the underlying buffer while the pointer itself refers to the span.
using shared_byte_span = std::shared_ptr<const std::span<const std::uint8_t>>;

/// Take ownership of @p bytes and return a @ref shared_byte_span over it.
/// A single allocation: the buffer and its span live in one control block.
[[nodiscard]] shared_byte_span make_shared_byte_span(std::vector<std::uint8_t> bytes);

// ---------------------------------------------------------------------------
// footer_probe
// ---------------------------------------------------------------------------

/// Result of a suffix-range footer probe: the object's total size plus the
/// trailing window [window_lo, object_size) captured in @c bytes.  @c bytes is
/// null when the probe could not be satisfied (the caller then falls back to a
/// HEAD).  Shared, not copied, with the io_object that carries it for this open.
struct footer_probe {
  std::size_t object_size{0};
  std::size_t window_lo{0};
  shared_byte_span bytes;
  /// ETag from the verified 206, quotes preserved; empty otherwise.
  std::string etag;
};

/// Result of a blocking HEAD: the object's size plus its ETag when the server
/// sent one (quotes preserved, empty otherwise).
struct head_object_result {
  std::size_t object_size{0};
  std::string etag;
};

// ---------------------------------------------------------------------------
// footer_resolve_result
// ---------------------------------------------------------------------------

/// One resolved entry of a batched footer resolve
/// (@c rest_ioctx::resolve_footer_objects).  Exactly one of {object, error}
/// is set.
///
/// @c object is stashless — identity, size and validation tag only.  The
/// suffix window bytes arrive in @c footer instead, whose buffer is the byte
/// lease: it is allocated against the ioctx-wide footer budget
/// (@c config::footer_resolve_stash_budget) and the bytes return to that
/// budget when the buffer is freed, so the intended lifetime is parse-only.
/// Reads on @c object inside the window re-GET over the network.  @c footer
/// is null on the HEAD-fallback path (probe unusable), where @c window_lo
/// stays 0.
struct footer_resolve_result {
  std::size_t index{0};               ///< position in the submitted span
  std::string path;                   ///< the submitted path, verbatim
  std::shared_ptr<io_object> object;  ///< stashless: size + validation tag
  shared_byte_span footer;            ///< suffix window bytes (the lease)
  std::size_t window_lo{0};           ///< file offset of footer->front()
  std::exception_ptr error;           ///< per-entry failure, isolated
};

// ---------------------------------------------------------------------------
// rest_io_object
// ---------------------------------------------------------------------------

/**
 * @brief Concrete @c io_object backed by a RESTful object-store key.
 *
 * Stores the object identity and metadata captured when it was opened: the
 * path, the bucket + key the reactor authorizes against, the size, and the ETag
 * the opening response carried.  The cache id is the path plus that ETag when
 * it is a strong entity-tag, so a new version of a key is a new cache identity;
 * an open without a usable strong ETag gets a unique per-open id instead.
 * Does no I/O of its own.
 */
class rest_io_object : public io_object {
 public:
  /// The cache id of an open with a strong validator.  Public so a caller that
  /// knows a path and tag can derive the key without opening the object.
  [[nodiscard]] static std::string generation_key(std::string_view path,
                                                  std::string_view strong_tag)
  {
    std::string key;
    key.reserve(path.size() + 1 + strong_tag.size());
    key.append(path).push_back('\x1f');
    key.append(strong_tag);
    return key;
  }

  /// True iff @p etag is one strong entity-tag as RFC 7232 §2.3 defines it:
  /// `"` etagc* `"`, with etagc = %x21 / %x23-7E / obs-text.  Rejects the weak
  /// form, `*`, lists, unquoted tokens and anything with control characters.
  [[nodiscard]] static bool is_strong_tag(std::string_view etag) noexcept
  {
    if (etag.size() < 2 || etag.front() != '"' || etag.back() != '"') { return false; }
    for (char const ch : etag.substr(1, etag.size() - 2)) {
      auto const c = static_cast<unsigned char>(ch);
      if (c == '"' || c < 0x21 || c == 0x7F) { return false; }
    }
    return true;
  }

  rest_io_object(
    std::string path, std::string bucket, std::string key, size_t size, std::string etag = {})
    : _path(std::move(path)),
      _bucket(std::move(bucket)),
      _key(std::move(key)),
      _file_size(size),
      _etag(std::move(etag)),
      _cache_id(make_cache_id(_path, _etag))
  {
  }

  /// As above, but carrying a suffix-range footer stash: @p stash holds the
  /// object's bytes over [window_lo, object_size), so @c rest_reactor::host_read
  /// serves any read fully inside that window from memory instead of a GET.
  rest_io_object(std::string path,
                 std::string bucket,
                 std::string key,
                 size_t object_size,
                 size_t window_lo,
                 shared_byte_span stash,
                 std::string etag = {})
    : _path(std::move(path)),
      _bucket(std::move(bucket)),
      _key(std::move(key)),
      _file_size(object_size),
      _window_lo(window_lo),
      _stash(std::move(stash)),
      _etag(std::move(etag)),
      _cache_id(make_cache_id(_path, _etag))
  {
  }

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _cache_id; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] size_t size() const noexcept override { return _file_size; }
  [[nodiscard]] std::string_view validation_tag() const noexcept override { return _etag; }

  [[nodiscard]] const std::string& bucket() const noexcept { return _bucket; }
  [[nodiscard]] const std::string& key() const noexcept { return _key; }
  [[nodiscard]] object_ref get_object_ref() const { return object_ref{_bucket, _key}; }

  /// Trailing bytes prefetched at open (a suffix-range footer probe), or null
  /// when the object was opened without one.  A read fully inside
  /// [stash_window_lo, size) is served from here by @c host_read.
  [[nodiscard]] shared_byte_span const& stash() const noexcept { return _stash; }
  [[nodiscard]] size_t stash_window_lo() const noexcept { return _window_lo; }

 private:
  [[nodiscard]] static std::string make_cache_id(std::string const& path, std::string const& etag)
  {
    if (is_strong_tag(etag)) { return generation_key(path, etag); }
    static std::atomic<std::uint64_t> next_open{0};
    return generation_key(path, "open#" + std::to_string(next_open.fetch_add(1)));
  }

  std::string _path;
  std::string _bucket;
  std::string _key;
  size_t _file_size{0};
  size_t _window_lo{0};
  shared_byte_span _stash;
  std::string _etag;
  std::string _cache_id;
};

// ---------------------------------------------------------------------------
// rest_write_object
// ---------------------------------------------------------------------------

/**
 * @brief An S3 object opened for writing: one upload in progress.
 *
 * Writes land in parts of @c config::write_split_bytes (part k+1 holds [kP, (k+1)P)); nothing
 * is visible to readers until @c ioctx::flush commits the object (CompleteMultipartUpload, or
 * one PUT for an object of at most one part).  @ref size is the end of the highest byte written;
 * @ref validation_tag is empty until the commit and the new ETag after it.  Destroying an object
 * that has an UploadId but was never committed aborts its upload (best effort, through a live
 * reactor of the ioctx that opened it).  Design: docs/io/rest-uring-design.md section 5.
 */
class rest_write_object final : public rest_io_object {
 public:
  rest_write_object(std::string path,
                    std::string bucket,
                    std::string key,
                    std::shared_ptr<detail::upload_session> upload);
  ~rest_write_object() override;

  rest_write_object(rest_write_object const&)            = delete;
  rest_write_object& operator=(rest_write_object const&) = delete;
  rest_write_object(rest_write_object&&)                 = delete;
  rest_write_object& operator=(rest_write_object&&)      = delete;

  [[nodiscard]] size_t size() const noexcept override
  {
    return _size.load(std::memory_order_acquire);
  }
  [[nodiscard]] bool writable() const noexcept override { return true; }
  void note_written(std::size_t end) const noexcept override;
  [[nodiscard]] std::string_view validation_tag() const noexcept override;

  /// The upload this object writes through (internal).
  [[nodiscard]] std::shared_ptr<detail::upload_session> const& upload() const noexcept
  {
    return _upload;
  }

  /// Record the committed object's ETag (internal; once).
  void set_committed_tag(std::string etag) const noexcept;

 private:
  mutable std::atomic<std::size_t> _size{0};
  std::shared_ptr<detail::upload_session> _upload;
  mutable std::string _committed_tag;
  mutable std::atomic<bool> _committed{false};
};

// ---------------------------------------------------------------------------
// transport_stats
// ---------------------------------------------------------------------------

/// Cumulative transport counters of one reactor, all endpoints (gpuio's "counters that matter
/// for tuning", E2).  A snapshot, refreshed by the worker every 100 ms.
struct transport_stats {
  // requests
  std::uint64_t requests{0};  ///< HTTP exchanges started
  std::uint64_t responses_2xx{0}, responses_3xx{0}, responses_4xx{0}, responses_5xx{0};
  std::uint64_t throttles{0};  ///< 503 SlowDown / 429
  std::uint64_t if_match_412{0}, range_ignored{0};
  std::uint64_t retries{0}, resumes{0}, stale_retries{0}, auth_retries{0};
  std::uint64_t timeouts{0}, cancels{0};
  std::uint64_t bytes_received{0}, bytes_landed{0}, bytes_sent{0};
  std::uint64_t segments_le_1mib{0}, segments_le_4mib{0}, segments_le_16mib{0},
    segments_gt_16mib{0};
  std::uint64_t admission_waits{0};
  std::uint64_t device_copies{0}, device_copy_failures{0};
  // connections and TLS
  std::uint64_t connects{0}, connect_failures{0}, reused_checkouts{0};
  std::uint64_t handshakes_full{0}, handshakes_resumed{0}, handshake_failures{0};
  std::uint64_t ktls_rx{0}, ktls_tx{0}, ktls_fallbacks{0}, no_pad_set{0};
  std::uint64_t control_records_key_update{0}, close_notify{0};
  std::uint64_t waitall_watchdog_cancels{0}, idle_cutoff_closes{0}, connection_close_closes{0};
  std::uint32_t server_request_limit{0};  ///< learned per-connection limit (max over endpoints)
  // controller
  unsigned conn_target{0};   ///< the endpoint controller's target (max over endpoints)
  double link_gbps_seed{0};  ///< the controller's link seed
  // reactor
  std::uint64_t enters{0}, wakes{0};
};

// ---------------------------------------------------------------------------
// rest_reactor
// ---------------------------------------------------------------------------

/**
 * @brief Single-threaded io_uring reactor for RESTful object storage (s3://...).
 *
 * One worker thread owns one io_uring ring (SINGLE_ISSUER | DEFER_TASKRUN), its HTTP/1.1
 * connection pools (kTLS first, user-space TLS otherwise), the per-endpoint estimator samples,
 * the retry back-offs and the device copies.  Models the reactor concept consumed by
 * @c templated_ioctx, writes included.  Every request is authorized through the context's
 * @c request_authorizer on every attempt.  Design: docs/io/rest-uring-design.md.
 */
class rest_reactor {
 public:
  /// Every read is an HTTP round trip, so what a request costs is dominated by
  /// the fact that it IS a request.  Given the whole range set at once the
  /// reactor can fuse adjacent ranges and keep every connection busy, which it
  /// cannot do when ranges arrive one at a time as a reader walks the file.
  static constexpr bool prefers_bulk_io = true;

  /// Device sources are staged through the part buffers (D2H on the caller's stream).
  static constexpr bool supports_device_write = true;

  /// Shared, immutable services for a pool of reactors.  One instance is built
  /// by @c rest_ioctx and shared (via shared_ptr) across every reactor in the
  /// pool: the request @c authorizer, the pinned bounce-staging resource, the
  /// primitive @c config, and the transport state every reactor shares (TLS
  /// context, DNS resolver thread, endpoint table, per-endpoint estimators and
  /// retry budgets; built on first use).
  class reactor_context {
   public:
    reactor_context(config cfg,
                    std::shared_ptr<request_authorizer> authorizer,
                    cucascade::memory::fixed_size_host_memory_resource* host_mr = nullptr);
    ~reactor_context();

    reactor_context(reactor_context const&)            = delete;
    reactor_context& operator=(reactor_context const&) = delete;

    [[nodiscard]] const config& cfg() const noexcept { return _config; }
    [[nodiscard]] const std::shared_ptr<request_authorizer>& authorizer() const noexcept
    {
      return _authorizer;
    }
    [[nodiscard]] cucascade::memory::fixed_size_host_memory_resource* host_memory_resource()
      const noexcept
    {
      return _host_mr;
    }

    /// The transport state every reactor of the pool shares (internal): endpoints, DNS, the TLS
    /// provider, per-endpoint estimators and retry budgets.  Built on first call; thread-safe.
    [[nodiscard]] detail::shared_services& services() const;

   private:
    config _config;
    std::shared_ptr<request_authorizer> _authorizer;
    cucascade::memory::fixed_size_host_memory_resource* _host_mr{nullptr};
    mutable std::mutex _services_mutex;
    mutable std::unique_ptr<detail::shared_services> _services;
  };

  using io_object_type       = rest_io_object;
  using reactor_config_type  = config;
  using reactor_context_type = reactor_context;

  explicit rest_reactor(std::shared_ptr<reactor_context> ctx,
                        std::string_view tname = "rest_reactor");
  ~rest_reactor();

  rest_reactor(rest_reactor const&)            = delete;
  rest_reactor& operator=(rest_reactor const&) = delete;

  /// The reactor's effective config (copied from its context at construction and
  /// clamped to legal values).  templated_ioctx reads its own _config from here
  /// so the config lives in one place — the context.
  [[nodiscard]] const reactor_config_type& get_config() const noexcept { return _config; }

  /// Staging block size, taken from the context's host resource (0 when the
  /// context has none).  INVARIANT: when a prefetching cache is present this
  /// MUST equal its chunk size — the worker plans a fragmented fill's extent as
  /// @c cache::fill_span(fill, chunk->offset, this value), so a larger staging
  /// block writes past the end of the pinned chunk and a smaller one marks a
  /// chunk cached while only part of it was fetched.  Checked once in
  /// @c ioctx::initialize_cache.
  [[nodiscard]] std::size_t staging_block_size() const noexcept
  {
    return _ctx == nullptr || _ctx->host_memory_resource() == nullptr
             ? std::size_t{0}
             : _ctx->host_memory_resource()->get_block_size();
  }

  // -- dispatch / lifecycle ------------------------------------------------

  /// Launch the worker thread.  Idempotent while running; shutdown is terminal
  /// and a later start is ignored.
  void start();

  void enqueue(std::unique_ptr<grouped_io_request> req) noexcept;
  void interrupt();
  void shutdown() noexcept;

  /// Bytes of queued-but-not-yet-submitted work — the reactor's backlog, and the
  /// signal @c templated_ioctx::next_reactor balances dispatch against.  Counts
  /// only what is waiting: an operation stops counting the moment a connection
  /// picks it up.  A hint, read without ordering against the queue itself.
  [[nodiscard]] std::size_t queued_bytes() const noexcept
  {
    return _queued_bytes.load(std::memory_order_relaxed);
  }

  /// Synchronous host read: served from the footer stash when it covers the
  /// range, else through the worker (blocking the caller; the reactor must run).
  size_t host_read(const io_object_type& file, size_t offset, size_t size, uint8_t* dst);

  /// Open connections to @p bucket's endpoint before any read needs them
  /// (TCP + TLS + kTLS, not outstanding work).  Returns immediately.
  void warmup(std::string bucket);

  /// Blocking HEAD: the object's size and ETag.  Runs on the worker's pooled
  /// connections while the reactor runs, else on a private ring on the calling
  /// thread.
  head_object_result head_object(std::string_view bucket, std::string_view key);

  /// Size-only convenience wrapper around @c head_object.
  size_t head_object_size(std::string_view bucket, std::string_view key);

  /// Blocking suffix-range GET of the last @p n bytes of an object, resolving
  /// the size and stashing the parquet footer in a single round-trip.  On a
  /// well-formed 206 the returned @c footer_probe carries the object size, the
  /// window origin, the trailing bytes, and the ETag; on any unusable response
  /// (200 full body, missing / unsatisfied Content-Range) @c bytes is null so
  /// the caller falls back to a HEAD.
  footer_probe fetch_footer_suffix(std::string_view bucket, std::string_view key, std::size_t n);

  /// Batched footer resolve engine: every entry gets the same per-attempt
  /// semantics as @c fetch_footer_suffix plus the HEAD fallback; all entries
  /// share one private ring and connection pool, driven on a short-lived helper
  /// thread (at most @p max_inflight transfers and connections).  Each probe reserves
  /// @c footer_probe_bytes from @p budget before its GET (non-blocking while any
  /// transfer is active; a blocking, stop-aware wait only when none is) and the
  /// delivered payload buffer carries the reservation until it is freed.
  /// @p on_result runs on the caller's thread, serially; see
  /// @c rest_ioctx::resolve_footer_objects for the delivery contract.
  void resolve_footer_batch(std::span<std::string const> paths,
                            std::span<object_ref const> objects,
                            std::span<std::size_t const> indices,
                            std::size_t max_inflight,
                            std::shared_ptr<exec::admission_control> budget,
                            std::function<void(footer_resolve_result)> const& on_result,
                            std::stop_token stop);

  /// Blocking bucket-level ListObjectsV2 GET for one page: the raw XML body on
  /// HTTP 200.  @p canonical_query is the pre-encoded, key-sorted request query
  /// (authorization is added via @c authorize_list).  @p prefix is only for
  /// error text.  Transient failures are retried and WARN-logged.
  std::string list_page(std::string_view bucket,
                        std::string_view prefix,
                        std::string_view canonical_query);

  // -- writes ---------------------------------------------------------------

  /// Concept stub: an upload needs the pool's context; @c rest_ioctx overrides
  /// @c create_io_object_for_write.  Always throws.
  static std::unique_ptr<io_object_type> create_io_object_for_write(std::string path,
                                                                    write_open_options options);

  /// Synchronous write through the worker (blocking the caller; the reactor must
  /// run).  Writes every byte or throws.
  std::size_t host_write(const io_object_type& object,
                         std::size_t offset,
                         std::size_t size,
                         const std::uint8_t* src);

  /// Concept stub: a commit needs a reactor; @c rest_ioctx overrides
  /// @c flush_io.  Always throws.
  static void flush(const io_object_type& object);

  /// Commit @p object through the worker (blocking): upload what is left and
  /// complete the upload (or PUT the whole object).
  void commit(const io_object_type& object);

  /// The worker's inbox for write work parked on other reactors (internal; empty before
  /// start).
  [[nodiscard]] std::weak_ptr<detail::mailbox> write_inbox() const;

  // -- capabilities / factory ----------------------------------------------

  /// True iff @p path is an s3:// URL this reactor can serve.
  [[nodiscard]] static bool supports(std::string_view path);

  /// Concept stub: real object creation needs a HEAD + authorizer and lives in
  /// @c rest_ioctx::create_io_object.  Always throws.
  static std::unique_ptr<io_object_type> create_io_object(std::string path);

  /// REST has no physical block alignment, so this only coalesces overlapping /
  /// adjacent ranges (honoring a caller-supplied alignment >= 1 as a lower
  /// bound) into a minimal sorted set — fewer ranges means fewer GETs.
  static std::vector<byte_range> align_and_coalesce(std::span<const byte_range> ranges,
                                                    std::optional<size_t> alignment = std::nullopt);

  // -- observability ---------------------------------------------------------

  /// Transport counters of this reactor (a snapshot, refreshed every 100 ms by
  /// the worker; zero before it ran).
  [[nodiscard]] transport_stats stats() const;

 private:
  class worker;

  std::shared_ptr<reactor_context> _ctx;
  config _config;
  std::string _tname;
  std::atomic<std::size_t> _queued_bytes{0};
  std::unique_ptr<worker> _worker;
};

}  // namespace cucascade::io::rest
