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
#include <cucascade/io/details/request_hub.hpp>
#include <cucascade/io/details/runner_registry.hpp>
#include <cucascade/io/rest/authorizer.hpp>
#include <cucascade/io/rest/config.hpp>
#include <cucascade/io/rest/rest_engine.hpp>
#include <cucascade/io/rest/rest_upload.hpp>
#include <cucascade/io/rest/types.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <atomic>
#include <chrono>
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
 * Stores the object identity and metadata captured when it was opened.
 * Does no I/O of its own.
 *
 * An object opened for write (@c rest_ioctx::open_io_object_for_write) carries
 * an @ref upload_session: it is writable until committed, @ref size reports
 * the written high-water mark, and it becomes readable (and read-only) once
 * @c commit_async succeeded -- before that the object does not exist on the
 * store.
 */
class rest_io_object : public io_object {
 public:
  /// Writable object backed by @p session (see @c rest_reactor::create_io_object_for_write).
  rest_io_object(std::string path,
                 std::string bucket,
                 std::string key,
                 std::shared_ptr<upload_session> session)
    : _path(std::move(path)),
      _bucket(std::move(bucket)),
      _key(std::move(key)),
      _session(std::move(session))
  {
  }

  rest_io_object(
    std::string path, std::string bucket, std::string key, size_t size, std::string etag = {})
    : _path(std::move(path)),
      _bucket(std::move(bucket)),
      _key(std::move(key)),
      _file_size(size),
      _etag(std::move(etag))
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
      _etag(std::move(etag))
  {
  }

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] size_t size() const noexcept override
  {
    return _session != nullptr ? _session->size() : _file_size;
  }
  [[nodiscard]] std::string_view validation_tag() const noexcept override { return _etag; }

  [[nodiscard]] const std::string& bucket() const noexcept { return _bucket; }
  [[nodiscard]] const std::string& key() const noexcept { return _key; }
  [[nodiscard]] object_ref get_object_ref() const { return object_ref{_bucket, _key}; }

  /// Trailing bytes prefetched at open (a suffix-range footer probe), or null
  /// when the object was opened without one.  A read fully inside
  /// [stash_window_lo, size) is served from here by @c host_read.
  [[nodiscard]] shared_byte_span const& stash() const noexcept { return _stash; }
  [[nodiscard]] size_t stash_window_lo() const noexcept { return _window_lo; }

  /// Upload state of an object opened for write; null for read-only objects.
  [[nodiscard]] std::shared_ptr<upload_session> const& session() const noexcept { return _session; }

  /// Whether writes are accepted (opened for write and not committed / failed).
  [[nodiscard]] bool writable() const { return _session != nullptr && _session->writable(); }

 private:
  std::string _path;
  std::string _bucket;
  std::string _key;
  size_t _file_size{0};
  size_t _window_lo{0};
  shared_byte_span _stash;
  std::string _etag;
  std::shared_ptr<upload_session> _session;
};

// ---------------------------------------------------------------------------
// rest_reactor
// ---------------------------------------------------------------------------

/**
 * @brief Shared, thread-safe dispatcher for RESTful object storage (s3://...).
 *
 * Models the reactor concept (v2) consumed by @c templated_ioctx: it owns the
 * context-wide @c request_hub (admission, per-class queue, runner registry),
 * the shared @ref reactor_context (config, presigning authorizer, pinned
 * staging resource) and the caller-thread blocking helpers (HEAD, LIST,
 * footer probes).  The transfers themselves run on runner threads, each in
 * its own @ref rest_engine built by @ref make_engine (curl multi + epoll loop,
 * connection cache, easy-handle pool, retry heap).  Presigned GET/HEAD URLs
 * come from the context's @c request_authorizer, re-issued on every attempt.
 *
 * Thread-safety: every public member may be called from any thread.
 */
class rest_reactor {
 public:
  /// Every read is an HTTP round trip, so what a request costs is dominated by
  /// the fact that it IS a request.  Given the whole range set at once the
  /// reactor can fuse adjacent ranges and keep every connection busy, which it
  /// cannot do when ranges arrive one at a time as a reader walks the file.
  static constexpr bool prefers_bulk_io = true;

  /// Whole-object uploads (PUT / multipart), from host and device sources.
  static constexpr bool supports_write        = true;
  static constexpr bool supports_device_write = true;

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

   private:
    config _config;
    std::shared_ptr<request_authorizer> _authorizer;
    cucascade::memory::fixed_size_host_memory_resource* _host_mr{nullptr};
  };

  using io_object_type       = rest_io_object;
  using reactor_config_type  = config;
  using reactor_context_type = reactor_context;
  using engine_type          = rest_engine;

  /// A warm-up target recorded by @ref warmup (see @ref current_warm_request).
  struct warm_request {
    std::uint64_t generation{0};  ///< 0: no warm-up was ever requested
    std::string bucket;
    std::chrono::steady_clock::time_point requested_at{};
  };

  explicit rest_reactor(std::shared_ptr<reactor_context> ctx,
                        std::string_view tname = "rest_reactor");
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

  // -- runner model ---------------------------------------------------------

  /// The request hub shared by the ioctx front end and every engine.
  [[nodiscard]] ::cucascade::io::detail::request_hub& hub() noexcept { return _hub; }
  [[nodiscard]] ::cucascade::io::detail::request_hub const& hub() const noexcept { return _hub; }

  /// Build the engine of the runner owning @p slot (called on that runner's thread).
  /// @throws std::runtime_error if the engine's curl / epoll setup fails.
  [[nodiscard]] std::unique_ptr<rest_engine> make_engine(
    ::cucascade::io::detail::runner_slot& slot);

  /// Shared services (authorizer, staging resource, config) for the engines.
  [[nodiscard]] reactor_context const& context() const noexcept { return *_ctx; }

  /// Name given at construction (log context only).
  [[nodiscard]] std::string const& name() const noexcept { return _tname; }

  /// Bytes not yet taken of all queued requests (a hint; see @c request_hub::queued_bytes).
  [[nodiscard]] std::size_t queued_bytes() const noexcept { return _hub.queued_bytes(); }

  /**
   * @brief Open @p path (s3://bucket/key) for a whole-object upload.
   *
   * No network I/O: the multipart upload (if any) is created lazily by the
   * first part.  Only @c write_mode::create_or_truncate is supported -- an
   * object store has no in-place update of an existing object.
   *
   * @throws std::invalid_argument for a non-s3 path.
   * @throws std::system_error (@c std::errc::not_supported) for any other mode.
   */
  [[nodiscard]] std::unique_ptr<io_object_type> create_io_object_for_write(std::string path,
                                                                           write_open_options opts);

  /**
   * @brief Synchronous host write: publishes a write request and waits for it.
   *
   * Like @ref host_read it needs a runner (@c start() or an external
   * @c run*); without one it fails with @c std::errc::operation_canceled.
   */
  std::size_t host_write(io_object_type const& object,
                         std::size_t offset,
                         std::size_t size,
                         std::uint8_t const* source,
                         write_options opts);

  /**
   * @brief Abort every live, uncommitted upload of this reactor
   *        (AbortMultipartUpload, synchronous, best effort) and fail its
   *        session with @c std::errc::operation_canceled.
   *
   * Called by @c rest_ioctx::shutdown once every runner has stopped, so no
   * part upload races the abort.  Also aborts the orphaned uploads still
   * pending in @ref orphan_uploads.
   */
  void abort_live_uploads() noexcept;

  /// Uploads of sessions destroyed without commit, waiting for an engine to
  /// abort them (see @c orphan_upload_sink).
  [[nodiscard]] orphan_upload_sink& orphan_uploads() noexcept { return *_orphans; }

  /// Synchronous buffered host read (blocking ranged GET).  Blocks the caller
  /// until a runner served it: needs a runner (@c start() or an external
  /// @c run*); without one admission is closed and it fails fast with
  /// @c std::errc::operation_canceled.
  size_t host_read(const io_object_type& file, size_t offset, size_t size, uint8_t* dst);

  /// Ask every runner to open its connection pool against @p bucket before any
  /// read needs it.  Returns immediately: the request is recorded here and the
  /// registered runners are woken; each engine primes its own pool on its own
  /// thread at the top of its next pass, because the connection cache it fills
  /// is thread-confined (see the @c curl_share warning) and is reachable from
  /// nowhere else.  An engine created later (a new @c run* call) primes from
  /// a request younger than @c conn_max_age, so warming before @c start()
  /// still works.  Coalescing is the caller's job -- a second call before the
  /// first is serviced simply replaces the target.
  ///
  /// The request is a bucket-scoped @c ListObjectsV2 capped at zero keys, not a
  /// HEAD: a HEAD is signed per object and @c sigv4_authorizer refuses an empty
  /// key, whereas @c authorize_list already signs a bucket-only URI, so this
  /// keeps warm-up traffic off the query's data files without touching the
  /// signing path.  The response is discarded and never inspected -- the
  /// handshake is what is being bought, so even a 403 is a success.
  void warmup(std::string bucket);

  /// Generation of the latest @ref warmup request (0: none).  Lock-free; polled
  /// by the engines.
  [[nodiscard]] std::uint64_t warm_generation() const noexcept
  {
    return _warm_generation.load(std::memory_order_acquire);
  }

  /// The latest @ref warmup request (consistent snapshot).
  [[nodiscard]] warm_request current_warm_request() const;

  /// Blocking HEAD to discover an object's size and ETag.  Used by the ioctx to
  /// build an @c rest_io_object.  @p bucket / @p key identify the object.
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
  /// semantics as @c fetch_footer_suffix plus the HEAD fallback, but all
  /// entries share one curl multi driven on the caller's thread, so
  /// connections are reused across entries (at most @p max_inflight pooled)
  /// and at most @p max_inflight transfers are on the wire at once.
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

 private:
  // Shared services + tunables for the whole reactor; kept alive for this
  // reactor's lifetime (the authorizer is used on every request).
  std::shared_ptr<reactor_context> _ctx;
  config _config;  // copy of _ctx->cfg(), clamped to legal values
  // Name captured at construction (log context).
  std::string _tname;

  // Set by warmup() on a caller thread and polled by every engine at the top
  // of a pass.  The bucket is guarded because a std::string is not atomically
  // publishable; the generation is what the engines actually poll.
  mutable std::mutex _warm_mtx;
  std::string _warm_bucket;                                    // guarded by _warm_mtx
  std::chrono::steady_clock::time_point _warm_requested_at{};  // guarded by _warm_mtx
  std::atomic<std::uint64_t> _warm_generation{0};              // written under _warm_mtx

  ::cucascade::io::detail::request_hub _hub;

  // Upload sessions of the objects opened for write (pruned when expired),
  // aborted by abort_live_uploads() at context shutdown.
  std::mutex _sessions_mtx;
  std::vector<std::weak_ptr<upload_session>> _sessions;  // guarded by _sessions_mtx

  // Shared with every session created here; closed by the destructor.
  std::shared_ptr<orphan_upload_sink> _orphans{
    std::make_shared<orphan_upload_sink>(_hub.registry())};
};

}  // namespace cucascade::io::rest
