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
#include <cucascade/io/tiered_queue.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace cucascade::io::rest {

namespace detail {
class shared_services;
struct pool_placement;
}  // namespace detail

class reactor_worker;
class private_driver;

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

  /// A strong ETag is the object's version (@c object_tag); a weak, malformed
  /// or absent one scopes the identity to this open (@c opened_handle), exactly
  /// as @ref raw_file_cache_id scopes the cache id.
  [[nodiscard]] bool validation_tag_is_strong() const noexcept override
  {
    return is_strong_tag(validation_tag());
  }

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
// transport_stats
// ---------------------------------------------------------------------------

/// Cumulative counters of one REST reactor (its scheduler and its io_uring transport), a
/// snapshot published by the worker once per loop pass.  @c rest_ioctx::stats sums them over the
/// pool (@c mss_min / @c mss_max are the extremes).
struct transport_stats {
  // scheduler
  std::uint64_t requests_started{0};  ///< grouped requests taken from the queue
  std::uint64_t gets{0};              ///< data GETs completed successfully
  std::uint64_t get_bytes{0};         ///< bytes delivered by those GETs
  std::uint64_t heads{0};             ///< HEAD exchanges completed
  std::uint64_t control_requests{0};  ///< HEAD / footer probe / LIST exchanges completed
  std::uint64_t warmups{0};           ///< warm-up exchanges issued
  std::uint64_t upkeeps{0};           ///< idle upkeep exchanges issued
  std::uint64_t retries{0};           ///< transient retries scheduled (data and control)
  std::uint64_t auth_retries{0};      ///< 403 re-sign retries
  std::uint64_t stale_retries{0};     ///< free retries after a pooled connection died
  std::uint64_t failures{0};          ///< operations that failed terminally
  std::uint64_t h2d_copies{0};        ///< staged host-to-device copies
  std::uint64_t staging_waits{0};     ///< operations parked until staging blocks returned
  std::uint64_t passes{0};            ///< worker loop iterations
  std::uint64_t waits{0};             ///< blocking waits
  std::uint64_t max_inflight{0};
  std::uint64_t established{0};  ///< connected connections right now
  std::uint64_t responses_2xx{0};
  std::uint64_t responses_3xx{0};
  std::uint64_t responses_4xx{0};
  std::uint64_t responses_5xx{0};
  // transport
  std::uint64_t connects{0};
  std::uint64_t connect_failures{0};
  std::uint64_t handshakes{0};
  std::uint64_t ktls_sessions{0};           ///< connections with kTLS RX and TX
  std::uint64_t userspace_tls_sessions{0};  ///< connections on the user-space record layer
  std::uint64_t ktls_fallbacks{0};          ///< user space although kTLS was allowed
  std::uint64_t reused{0};                  ///< exchanges on a pooled connection
  std::uint64_t closed{0};
  std::uint64_t idle_closes{0};     ///< reuse refused: idle longer than conn_max_age
  std::uint64_t dead_closes{0};     ///< reuse refused: input / HUP / error on the idle socket
  std::uint64_t stale_failures{0};  ///< pooled connections that died before any response byte
  std::uint64_t sends{0};
  std::uint64_t recvs{0};
  std::uint64_t bytes_sent{0};
  std::uint64_t recv_direct_bytes{0};   ///< body bytes received straight into the destination
  std::uint64_t recv_scratch_bytes{0};  ///< bytes received through the scratch buffer
  std::uint64_t timeouts{0};
  std::uint64_t stalls{0};  ///< data GETs cancelled by the stall detector
  std::uint64_t net_errors{0};
  std::uint64_t control_records{0};  ///< kTLS control records consumed
  std::uint64_t mtu_rejects{0};
  std::uint64_t mtu_fallbacks{0};
  std::uint64_t mss_min{0};  ///< smallest TCP_MAXSEG observed (0 = none)
  std::uint64_t mss_max{0};

  /// Add @p other (sums; extremes for @c mss_min / @c mss_max / @c max_inflight).
  void merge(transport_stats const& other) noexcept;
};

// ---------------------------------------------------------------------------
// rest_reactor
// ---------------------------------------------------------------------------

/**
 * @brief Single-threaded io_uring reactor for RESTful object storage (s3://...).
 *
 * Owns one worker thread driving one io_uring HTTP/1.1 transport (up to
 * @c config::max_connections keep-alive connections; kTLS when the kernel accepts the keys,
 * the user-space TLS record layer otherwise), a two-tier MPSC request queue (demand before
 * prefetch), a control-plane queue (HEAD, footer probe, LIST, warm-up), dynamic CuCascade
 * staging for device reads, and a retry heap.  Physical GETs are sized from the backlog
 * (uring_ultra's adaptive segment planner).  Models the reactor concept consumed by
 * @c templated_ioctx.  Every attempt is authorized afresh through the context's
 * @c request_authorizer.  Blocking control-plane calls run on the worker's connections while
 * it runs, else on a private transport on a short-lived helper thread (never the caller's).
 * That helper is a whole reactor loop (a thread, an io_uring ring, new connections) per call,
 * far heavier than the libcurl easy handle it replaces: start the pool before issuing
 * control-plane calls.  A call made from a reactor thread (a completion callback) also takes
 * the helper path, so it never waits on a reactor loop that may be its own.
 */
class rest_reactor {
 public:
  /// Every read is an HTTP round trip, so what a request costs is dominated by
  /// the fact that it IS a request.  Given the whole range set at once the
  /// reactor can fuse adjacent ranges and keep every connection busy, which it
  /// cannot do when ranges arrive one at a time as a reader walks the file.
  static constexpr bool prefers_bulk_io = true;

  /// Shared, immutable services for a pool of reactors.  One instance is built
  /// by @c rest_ioctx and shared (via shared_ptr) across every reactor in the
  /// pool, so it is the natural home for things that are shared rather than
  /// per-reactor: the presigning @c authorizer, the pinned bounce-staging
  /// resource, and — in future — a shared connection pool, a registered-buffer
  /// table, etc.  It also carries the primitive @c config (separating the
  /// injected collaborators from the plain, file-settable tunables).
  class reactor_context {
   public:
    reactor_context(config cfg,
                    std::shared_ptr<request_authorizer> authorizer,
                    cucascade::memory::fixed_size_host_memory_resource* host_mr = nullptr)
      : _config(std::move(cfg)), _authorizer(std::move(authorizer)), _host_mr(host_mr)
    {
    }

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

    /// Transport state every reactor of the pool shares (internal): the DNS resolver thread and
    /// the TLS provider.  Built on first call; thread-safe.
    [[nodiscard]] detail::shared_services& services() const;

   private:
    config _config;
    std::shared_ptr<request_authorizer> _authorizer;
    cucascade::memory::fixed_size_host_memory_resource* _host_mr{nullptr};
    mutable std::mutex _services_mutex;
    mutable std::shared_ptr<detail::shared_services> _services;
  };

  using io_object_type       = rest_io_object;
  using reactor_config_type  = config;
  using reactor_context_type = reactor_context;

  /// @p pool_index and @p placement (internal: the pool's placement plan, owned by the ioctx)
  /// select this reactor's CPU pin; without a placement the worker is left where it starts.
  explicit rest_reactor(std::shared_ptr<reactor_context> ctx,
                        std::string_view tname                                  = "rest_reactor",
                        std::size_t pool_index                                  = 0,
                        std::shared_ptr<detail::pool_placement const> placement = nullptr);
  ~rest_reactor();

  rest_reactor(rest_reactor const&)            = delete;
  rest_reactor& operator=(rest_reactor const&) = delete;

  /// The reactor's effective config (copied from its context at construction and
  /// clamped to legal values).  templated_ioctx reads its own _config from here
  /// so the config lives in one place — the context — rather than being passed
  /// in separately.
  [[nodiscard]] const reactor_config_type& get_config() const noexcept { return _config; }

  /// Staging block size, taken from the context's host resource (0 when the
  /// context has none).  INVARIANT: when a prefetching cache is present this
  /// MUST equal its chunk size — the worker plans a fragmented fill's extent as
  /// @c cache::fill_span(fill, chunk->offset, this value), so a larger staging
  /// block writes past the end of the pinned chunk and a smaller one marks a
  /// chunk cached while only part of it was fetched.  Checked once in
  /// @c ioctx::initialize_cache, which is the only place both sizes are known
  /// (the reactor is built, and may already be started, before the cache
  /// exists).
  [[nodiscard]] std::size_t staging_block_size() const noexcept
  {
    return _ctx == nullptr || _ctx->host_memory_resource() == nullptr
             ? std::size_t{0}
             : _ctx->host_memory_resource()->get_block_size();
  }

  // -- dispatch / lifecycle ------------------------------------------------

  /// Launch the worker thread. Split out of the constructor so a reactor can be
  /// built cheaply and parked until it is actually needed. Idempotent while
  /// running; shutdown is terminal and a later start is ignored.
  void start();

  void enqueue(std::unique_ptr<grouped_io_request> req) noexcept;
  void interrupt();
  void shutdown() noexcept;

  /// Bytes of queued-but-not-yet-submitted work — the reactor's backlog, and the
  /// signal @c rest_ioctx::next_reactor balances dispatch against.  Counts only
  /// what is waiting: a chunk stops counting the moment a connection picks it up,
  /// because in-flight work is already bounded by @c max_connections and is
  /// therefore the same ceiling on every reactor, while the queue is where an
  /// unevenly-loaded pool actually diverges.
  ///
  /// A hint, not a synchronization point: it is read without ordering against
  /// the queue itself, so a concurrent enqueue or dequeue may not be reflected
  /// yet.  Dispatch only needs to be right on average.
  [[nodiscard]] std::size_t queued_bytes() const noexcept
  {
    return _queued_bytes.load(std::memory_order_relaxed);
  }

  /// Synchronous host read: served from the footer stash when it covers the range, else a
  /// ranged GET through the worker.  Blocks the caller.
  size_t host_read(const io_object_type& file, size_t offset, size_t size, uint8_t* dst);

  /// Ask the worker to make its connection pool hot against @p bucket before any read needs
  /// it.  Returns immediately: the worker issues, at the top of its next pass, one
  /// bucket-scoped @c ListObjectsV2 capped at zero keys per idle connection slot (up to
  /// @c max_connections), each on a fresh connection (an idle connection in the slot is closed
  /// first, like curl's FRESH_CONNECT), so TCP, TLS and HTTP are all validated.  A HEAD is not
  /// used because it is signed per object and @c sigv4_authorizer refuses an empty key.  The
  /// responses are discarded and failures are logged at debug only; a second call before the
  /// first is serviced replaces the target.  Rate limiting is @c rest_ioctx::warmup's job.
  void warmup(std::string bucket);

  /// Blocking HEAD to discover an object's size and ETag.  Used by the ioctx to
  /// build an @c rest_io_object.  @p bucket / @p key identify the object.  Bounded by
  /// @c request_timeout_s per attempt; transient failures are retried.
  head_object_result head_object(std::string_view bucket, std::string_view key);

  /// Size-only convenience wrapper around @c head_object.
  size_t head_object_size(std::string_view bucket, std::string_view key);

  /// Blocking suffix-range GET of the last @p n bytes of an object, resolving
  /// the size and stashing the parquet footer in a single round-trip.  On a
  /// well-formed 206 the returned @c footer_probe carries the object size, the
  /// window origin, the trailing bytes, and the ETag; on any unusable response
  /// (200 full body, missing / unsatisfied Content-Range) @c bytes is null so
  /// the caller falls back to a HEAD.  @p bucket / @p key identify the object.
  footer_probe fetch_footer_suffix(std::string_view bucket, std::string_view key, std::size_t n);

  /// Batched footer resolve engine: every entry gets the same per-attempt
  /// semantics as @c fetch_footer_suffix plus the HEAD fallback, run as
  /// exchanges on this reactor's connections (or, when the reactor is not
  /// running, on a private transport on a short-lived helper thread), so
  /// connections are reused across entries and at most @p max_inflight
  /// transfers are on the wire at once.  The caller's thread only
  /// coordinates (budget, retries' fallbacks, delivery); io_uring never runs
  /// on it, and when the helper cannot be started every entry is delivered
  /// as canceled before the error is rethrown.
  /// @p paths / @p objects / @p indices are parallel: @p indices carries each
  /// entry's position in the caller's original batch.  Each probe reserves
  /// @c footer_probe_bytes from @p budget before its GET is issued
  /// (non-blocking while any transfer is active; a blocking, stop-aware wait
  /// only when none is) and the delivered payload buffer carries the
  /// reservation until it is freed.  @p on_result runs on the caller's
  /// thread, serially, as entries land; see
  /// @c rest_ioctx::resolve_footer_objects for the delivery contract.
  /// Assumes non-empty input and max_inflight >= 1; concurrent-batch
  /// serialization is the ioctx's job, not this method's.
  void resolve_footer_batch(std::span<std::string const> paths,
                            std::span<object_ref const> objects,
                            std::span<std::size_t const> indices,
                            std::size_t max_inflight,
                            std::shared_ptr<exec::admission_control> budget,
                            std::function<void(footer_resolve_result)> const& on_result,
                            std::stop_token stop);

  /// Blocking bucket-level ListObjectsV2 GET for one page: returns the raw XML
  /// body on HTTP 200.  @p canonical_query is the pre-encoded, key-sorted
  /// request query (no auth params — authorization is added via
  /// @c authorize_list).  @p prefix is only for retry-log / error text.
  /// Control-plane op: transient failures are retried and WARN-logged.
  std::string list_page(std::string_view bucket,
                        std::string_view prefix,
                        std::string_view canonical_query);

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

  /// This reactor's counters (a snapshot published once per worker pass; zero before it ran).
  [[nodiscard]] transport_stats stats() const noexcept;

  /// Worker-side control-plane work (HEAD, probe, LIST, warm-up); internal.
  struct control_job;

 private:
  friend class reactor_worker;
  friend class private_driver;
  struct intake;
  struct atomic_stats;

  void worker_loop(const std::stop_token& stop_token);
  /// Hand @p jobs to the worker, or run them on a private transport on a helper thread when
  /// the worker is not running.  Blocks until every job completed.
  void run_controls(std::vector<std::shared_ptr<control_job>> const& jobs);
  /// Queue @p job on the running worker.  False when the worker does not accept work.
  bool post_control(std::shared_ptr<control_job> job);

  // Shared services + tunables for the whole reactor pool; kept alive for this
  // reactor's lifetime (the authorizer is used on every request).
  std::shared_ptr<reactor_context> _ctx;
  config _config;  // copy of _ctx->cfg() for hot-path access
  // Thread name prefix captured at construction; applied to the worker in start().
  std::string _tname;
  // Position in the pool and the pool's placement (select the worker's CPU pin).
  std::size_t _pool_index{0};
  std::shared_ptr<detail::pool_placement const> _placement;

  // The worker's queues and wake-up fd (shared with a private helper's layout).
  std::unique_ptr<intake> _intake;
  std::unique_ptr<atomic_stats> _stats;

  std::stop_source _stop_source;
  mutable std::mutex _enqueue_mutex;
  bool _running{false};
  bool _accepting{false};
  bool _stopped{false};

  // Logical bytes not yet assigned to a connection.  Retries are already claimed
  // work and therefore never get counted a second time.
  std::atomic<std::size_t> _queued_bytes{0};

  std::jthread _worker;
};

}  // namespace cucascade::io::rest
