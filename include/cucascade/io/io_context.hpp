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
#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/cache/config.hpp>
#include <cucascade/io/cache/metadata_store.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/io/uri_parser.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cucascade::io {

enum class io_context_type { uring, restful, kvikio, s3rdma };

/// Hint passed to @c open_io_object so a backend can tailor how it resolves an
/// object's metadata.  @c generic resolves the size however is cheapest for the
/// scheme (a HEAD for object stores).  @c parquet_footer_probe asks the backend
/// to resolve the size *and* stash the object's trailing bytes in one
/// round-trip (a suffix-range GET), so the parquet footer reads that follow are
/// served locally instead of costing extra round-trips.
enum class open_hint { generic, parquet_footer_probe };

namespace cache {
class prefetching_cache;
class prefetching_handle;
}  // namespace cache

class datasource;

}  // namespace cucascade::io

namespace cucascade::memory {
class topology_index;
}  // namespace cucascade::memory

namespace cucascade::memory {
class memory_reservation_manager;
}  // namespace cucascade::memory

namespace cucascade::io {

// ---------------------------------------------------------------------------
// ioctx
// ---------------------------------------------------------------------------

/**
 * @brief Abstract shared context passed to every datasource.
 *
 * Holds resources that are shared across all datasources (cache, reactor
 * threads, ...). Extend this class to provide a concrete I/O backend.
 */
class ioctx : public std::enable_shared_from_this<ioctx> {
 public:
  ioctx();
  virtual ~ioctx();

  [[nodiscard]] virtual io_context_type type() const noexcept = 0;

  /// Start the backend's reactors: launch their worker threads and allocate
  /// per-reactor staging.  Deferred from construction so an ioctx can be built
  /// and parked (e.g. in a per-query map of contexts) without spending thread
  /// or pinned-memory resources until it is first used.  Must be called before
  /// the read API is exercised.  Idempotent.  Backends with no reactors
  /// (blocking) inherit the default no-op.
  virtual void start() {}

  virtual void shutdown() noexcept = 0;

  /// Open the backend-appropriate io_object for @p path (local fds / an
  /// object-store HEAD / ...).  Throws on unsupported / unreachable paths
  /// (callers that want a check-without-open should use @c supports()).
  /// The cudf-coupled datasource layer wraps the result in a
  /// @c cucascade::io::datasource; cudf-free callers drive the read APIs
  /// below directly.
  ///
  /// A leading @c file: URI scheme is stripped HERE, at the single funnel above
  /// the @c create_io_object virtual, rather than at each call site (see
  /// @c strip_file_scheme): an un-stripped URI would reach a local backend's
  /// "unsupported path" throw, which surfaces as a runtime fallback rather than
  /// a clean decline.
  [[nodiscard]] std::shared_ptr<io_object> open_io_object(std::string path)
  {
    return create_io_object(strip_file_scheme(path));
  }

  /// As above, forwarding @p hint to the backend's io_object resolution so it
  /// can, e.g., prefetch a parquet footer in the same round-trip as the size.
  [[nodiscard]] std::shared_ptr<io_object> open_io_object(std::string path, open_hint hint)
  {
    return create_io_object(strip_file_scheme(path), hint);
  }

  /// As above, with the object's size already known (e.g. from an S3
  /// ListObjectsV2 response), so a backend that can act on it skips its size
  /// discovery entirely (no HEAD for object stores).
  [[nodiscard]] std::shared_ptr<io_object> open_io_object(std::string path,
                                                          std::uint64_t known_size)
  {
    return create_io_object(strip_file_scheme(path), known_size);
  }

  /**
   * @brief Open (and, depending on @p opts, create) @p path for writing.
   *
   * Strips a leading @c file: scheme like @ref open_io_object.  The returned
   * object accepts the write API below and may also be read through this
   * ioctx.
   *
   * @param path Local path or object URL.
   * @param opts Open / create behaviour.
   * @return The writable io_object.
   * @throws std::system_error with @c std::errc::not_supported when the backend
   *         cannot write; backend-specific errors otherwise (e.g. ENOENT for
   *         @c write_mode::open_existing on a missing local file).
   */
  [[nodiscard]] std::shared_ptr<io_object> open_io_object_for_write(std::string path,
                                                                    write_open_options opts = {})
  {
    return create_io_object_for_write(strip_file_scheme(path), opts);
  }

  /// Open the backend's connections to whatever serves @p bucket_url, ahead of
  /// the first read, so a query does not pay connection setup on its hot path.
  ///
  /// @p bucket_url names a container, not an object -- @c "s3://my-bucket".  A
  /// full object URL is accepted and its key ignored, since connections are per
  /// endpoint and the object adds nothing to the identity.  Deliberately not an
  /// @c io_object: opening one is itself a round trip over the connection being
  /// warmed, which would leave the warm-up nothing left to hide, and it would
  /// tie warm-up traffic to individual data files.
  ///
  /// Best-effort by contract: it must not throw and must not block the caller,
  /// and a failed warm-up is never a reason to fail a read.  For a transport
  /// that is what warms, even a rejected request is a success -- a 403 still
  /// completed the DNS lookup, the TCP connect and the TLS handshake, which is
  /// all that was being bought.
  ///
  /// The default is a no-op, which is the right answer for every backend whose
  /// "connection" is a file descriptor it already holds.
  virtual void warmup(std::string_view /*bucket_url*/) noexcept {}

  /// Whether this backend can serve reads for @p path.  Backends should
  /// validate scheme/protocol support and any backend-specific
  /// preconditions (e.g. file existence for local-disk backends).
  [[nodiscard]] virtual bool supports(std::string_view path) const noexcept = 0;

  // -- Backend capabilities ---------------------------------------------------

  /// Whether the backend can stream data directly into device memory
  /// (e.g. via O_DIRECT + GDS).  Used by @c datasource to answer the
  /// equivalent cudf::io::datasource queries.
  [[nodiscard]] virtual bool supports_device_read() const noexcept = 0;

  [[nodiscard]] virtual bool supports_host_to_device_read() const noexcept = 0;

  /// Whether the backend can serve a batch of host reads in a single dispatch
  /// (cf. @c host_readv_async_io).  When false, the prefetching layer
  /// cannot amortise per-request overhead and must fall back to
  /// @c scan_stage::none.
  [[nodiscard]] virtual bool supports_vector_host_read() const noexcept = 0;

  /// Whether the backend can efficiently serve a batch of device reads.
  /// Backends may still process mixed slices serially when this is false.
  [[nodiscard]] virtual bool supports_device_range_read() const noexcept = 0;

  /// Whether the backend implements the write API (@ref open_io_object_for_write,
  /// @ref host_write, @ref writev_async, ...).  Conservatively false.
  [[nodiscard]] virtual bool supports_write() const noexcept { return false; }

  /// Whether the backend accepts device-memory write sources
  /// (@ref device_write_async, @c device_source segments).  Conservatively false.
  [[nodiscard]] virtual bool supports_device_write() const noexcept { return false; }

  /// Whether this backend would rather be handed one batched request covering
  /// everything a reader needs than a stream of small reads as the reader walks
  /// the file.
  ///
  /// Unlike the supports_* flags this is a preference, not a capability: a
  /// backend that says no can still serve a batch, and one that says yes can
  /// still serve small reads.  It reflects what the request itself costs.  For
  /// an object store a read is a round trip, so the shape of the request set
  /// dominates and knowing all of it up front is worth a great deal; for a local
  /// file a read is a syscall against page cache or NVMe, and batching buys
  /// little while forcing the caller to materialise ranges it may not need.
  ///
  /// Conservatively false: a backend opts in.
  [[nodiscard]] virtual bool prefers_bulk_io() const noexcept { return false; }

  /// The smallest unit this backend can address, in bytes.  A read is widened
  /// out to a multiple of it before being issued: a local file opened O_DIRECT
  /// can only transfer whole pages, while an object store addresses single
  /// bytes and pays nothing for an odd offset.
  ///
  /// Conservatively 1 -- a backend that has not opted in is never widened.
  [[nodiscard]] virtual std::size_t min_alignment_requirement() const noexcept { return 1; }

  /// The largest gap between two ranges still worth bridging into one request,
  /// in bytes.  The bridged bytes are fetched and discarded, traded against the
  /// cost of a second request: a page for a local file, a whole round trip for
  /// an object store, which is why the two want very different answers.
  ///
  /// Conservatively 0 -- only adjacent ranges are fused.
  [[nodiscard]] virtual std::size_t merge_gap_size() const noexcept { return 0; }

  /// How many scan tasks the readahead manager may keep in flight against this
  /// backend at once, as configured on its reactors.  Zero means this backend
  /// opts out of readahead scheduling entirely.
  ///
  /// The bound is a property of the backend, not of the query: it reflects the
  /// queue depth the device is worth driving at (see the per-backend defaults
  /// on each reactor config).  The base returns 0 so a backend that has not
  /// opted in is never scheduled against.
  [[nodiscard]] virtual std::size_t n_max_concurrent_scans() const noexcept { return 0; }

  /// Size of one staging block on this backend's reactors, in bytes.  A reactor
  /// that fills cache chunks computes each fragmented fill's extent with
  /// @c cache::fill_span(fill, chunk->offset, staging_block_size), so this MUST
  /// equal @c prefetching_cache::chunk_size() — @ref initialize_cache checks it.
  ///
  /// Conservatively 0 — a backend that has not opted in does not stage through
  /// cache chunks and is not checked.
  [[nodiscard]] virtual std::size_t staging_block_size() const noexcept { return 0; }

  /// Build the prefetching cache.  One-shot — calling twice is a no-op
  /// after the first successful build.  The cache holds a raw
  /// back-pointer to this ioctx and stays alive until @ref
  /// shutdown_cache is called (or this ioctx is destroyed).  The cache
  /// builds and owns its @c buffer_pool from @p reservation_manager's
  /// HOST-tier memory spaces; @p buffer_pool_slabs sizes that pool.
  ///
  /// The cache constructs itself in an "armed" or "unarmed" state
  /// depending on @c supports_vector_host_read(); the ioctx is unaware
  /// of that distinction — it simply forwards lookups through @c cache().
  void initialize_cache(
    cucascade::memory::memory_reservation_manager& reservation_manager,
    io::cache::config const& cache_config,
    std::shared_ptr<const cucascade::memory::topology_index> topology_index) noexcept;

  /// Tear down the cache (drains background workers and any in-flight
  /// IO via @c admission_control).  Idempotent.  The owner (scan
  /// manager) calls this BEFORE releasing the @c buffer_pool the cache
  /// was constructed with — otherwise workers may issue final IO
  /// against a destroyed pool.
  void shutdown_cache() noexcept;

  /// Every concrete derived class MUST call this as the very first
  /// statement in its destructor.  It drains the cache (so its workers
  /// stop issuing IO) while the derived object's reactors / handles
  /// are still alive.  Without this, the cache's defensive shutdown
  /// in @c ~ioctx would run AFTER the derived part of the
  /// object has been destroyed, and worker callbacks would reach
  /// already-destroyed reactors.
  ///
  /// Idempotent — calling @c shutdown_cache directly before this is
  /// fine.  Cheap when no cache was ever initialised.
  void pre_destroy() noexcept { shutdown_cache(); }

  [[nodiscard]] cache::prefetching_cache* cache() noexcept { return _cache.get(); }

  /// True iff @c host_read / @c device_read should consult the cache
  /// before falling through to the backend.  Computed live so it tracks
  /// @ref initialize_cache / @ref shutdown_cache transitions.
  [[nodiscard]] inline bool uses_prefetching_cache() const noexcept
  {
    return can_use_prefetching_cache() && _cache;
  }

  /// Per-file metadata cache that lives independently of the prefetching
  /// cache.  Always available — callers that have parsed file metadata
  /// (e.g. a parquet footer) park it here so a later scan of the same
  /// path can skip the parse without depending on whether the
  /// prefetching machinery has been wired up.
  [[nodiscard]] cache::metadata_store& metadata_store() noexcept { return _metadata_store; }
  [[nodiscard]] cache::metadata_store const& metadata_store() const noexcept
  {
    return _metadata_store;
  }

  // -- Physical range alignment ------------------------------------------------

  /// Align each input range's ends outward to the backend's I/O alignment and
  /// coalesce overlapping/adjacent results into a minimal set of aligned,
  /// non-overlapping ranges (sorted by offset).  @p alignment is a lower bound:
  /// when unset, or smaller than the backend's optimal alignment, the backend
  /// uses its own alignment instead.
  [[nodiscard]] virtual std::vector<byte_range> align_and_coalesce(
    std::span<const byte_range> ranges,
    std::optional<size_t> alignment = std::nullopt) const noexcept = 0;

  // -- Cache-aware reads --------------------------------------------------------
  //
  // The read entry points callers should use: when the prefetching cache is
  // armed they serve (and account) the read through it, otherwise they fall
  // through to the backend primitives (*_io below).  @p handle is the scan's
  // prefetching_handle (from a prior fadvise/insert), passed as a raw pointer
  // so the cache can consume/observe it; it may be null when the caller made
  // no prefetch reservation.  All async variants return @c exec::semi_future.
  // @p opts selects the scheduling class on the uncached path; with
  // @c request_class::automatic a read carrying a @p handle is @c background,
  // otherwise it is classified by size (see @ref resolve_request_class).

  size_t host_read(const io_object& obj,
                   size_t offset,
                   size_t size,
                   uint8_t* dst,
                   cache::prefetching_handle* handle = nullptr);

  [[nodiscard]] exec::semi_future<size_t> host_read_async(
    const io_object& obj,
    size_t offset,
    size_t size,
    uint8_t* dst,
    cache::prefetching_handle* handle = nullptr,
    io_options opts                   = {});

  [[nodiscard]] exec::semi_future<size_t> device_read_async(
    const io_object& obj,
    size_t offset,
    size_t size,
    uint8_t* dst,
    ::cuda::stream_ref stream,
    cache::prefetching_handle* handle = nullptr,
    io_options opts                   = {});

  // -- Cache-aware writes --------------------------------------------------------
  //
  // Non-virtual entry points: they validate, classify (@c automatic ->
  // @c request_class::write), invalidate any prefetching-cache chunks covering
  // the written ranges (before submission and again on completion), and forward
  // to the backend write hooks.  The value of a resolved write future is the
  // total number of bytes requested.  Source buffers must stay valid until the
  // future resolves.  Distinct requests are unordered, even on the same object;
  // concurrent writes to overlapping ranges leave undefined content.  Backends
  // without write support fail with @c std::errc::not_supported.

  /**
   * @brief Synchronously write @p size bytes from host memory @p src at @p offset.
   *
   * @return Number of bytes written (== @p size).
   * @throws std::invalid_argument if @p src is null and @p size is non-zero.
   * @throws std::system_error on I/O failure or @c not_supported.
   */
  std::size_t host_write(const io_object& obj,
                         std::size_t offset,
                         std::size_t size,
                         const std::uint8_t* src,
                         write_options opts = {});

  /**
   * @brief Asynchronously write @p size bytes from host memory @p src at @p offset.
   *
   * Errors (including invalid arguments) are reported through the future.
   */
  [[nodiscard]] exec::semi_future<std::size_t> host_write_async(const io_object& obj,
                                                                std::size_t offset,
                                                                std::size_t size,
                                                                const std::uint8_t* src,
                                                                write_options opts = {});

  /**
   * @brief Asynchronously write @p size bytes from device memory @p src at @p offset.
   *
   * The write observes all work enqueued on @p stream before this call.
   * Errors (including invalid arguments) are reported through the future.
   */
  [[nodiscard]] exec::semi_future<std::size_t> device_write_async(const io_object& obj,
                                                                  std::size_t offset,
                                                                  std::size_t size,
                                                                  const std::uint8_t* src,
                                                                  ::cuda::stream_ref stream,
                                                                  write_options opts = {});

  /**
   * @brief Vectored / mixed-source write.
   *
   * Segments may arrive in any order and may be written concurrently; they
   * must not overlap each other.  Device sources with @c device_id < 0 are
   * bound to the calling thread's current CUDA device.
   *
   * @return Future resolving with the total bytes of all segments; it fails
   *         with @c std::invalid_argument for overlapping segments or null
   *         sources.
   */
  [[nodiscard]] exec::semi_future<std::size_t> writev_async(const io_object& obj,
                                                            std::vector<write_segment> segments,
                                                            write_options opts = {});

  /**
   * @brief Make previously completed writes durable.
   *
   * Local files: fdatasync().  Object stores: resolved no-op.  Covers every
   * write whose future resolved before this call; writes still in flight are
   * not waited for.
   */
  [[nodiscard]] exec::semi_future<void> flush_async(const io_object& obj);

  /**
   * @brief Commit a writable object.
   *
   * Object stores: single PUT or CompleteMultipartUpload.  Local files:
   * optional data sync (@p durability).  Afterwards the object is read-only;
   * further writes fail with @c std::invalid_argument.
   *
   * With a prefetching cache attached, every cached chunk of the object
   * (keyed by its cache id / path) is invalidated once the commit settled,
   * before the returned future resolves: object-store writes only become
   * visible at commit, so their write completions cannot invalidate alone.
   */
  [[nodiscard]] exec::semi_future<void> commit_async(
    const io_object& obj, write_durability durability = write_durability::none);

  // -- Runner API ---------------------------------------------------------------
  //
  // Backends with runners (io_uring, REST) execute asynchronous requests on
  // threads that drive the context through run*().  @ref start spawns such
  // threads internally; callers may instead (or additionally) donate their own
  // threads.  Each run*() call builds a private backend engine on the calling
  // thread, pulls queued requests until stopped, then stops pulling, drains its
  // in-flight operations to completion, destroys the engine and returns the
  // number of grouped requests it completed.  Backends without runners throw
  // @c std::logic_error.

  /**
   * @brief Drive this context on the calling thread until @p token is stopped
   *        or @ref shutdown is called.
   * @return Number of grouped requests this runner completed.
   * @throws std::logic_error if the backend has no runners, or if the calling
   *         thread is already inside run*() of this context.
   */
  std::size_t run(std::stop_token token);

  /**
   * @brief As @ref run, also returning once @p duration has elapsed.  In-flight
   *        operations are drained before returning, so the call may overrun by
   *        up to one operation's latency.
   */
  std::size_t run_for(std::chrono::steady_clock::duration duration, std::stop_token token = {});

  /// As @ref run_for, with an absolute @p deadline.
  std::size_t run_until(std::chrono::steady_clock::time_point deadline, std::stop_token token = {});

  /// Number of threads currently inside run*() for this context.
  [[nodiscard]] virtual std::size_t active_runners() const noexcept { return 0; }

  /// Aggregate queue / runner statistics.
  [[nodiscard]] virtual queue_stats stats() const noexcept { return {}; }

  // -- Backend primitives (cache-unaware) ----------------------------------------

  virtual size_t host_read_io(const io_object& obj, size_t offset, size_t size, uint8_t* dst) = 0;

  virtual exec::semi_future<size_t> host_read_async_io(
    const io_object& obj, size_t offset, size_t size, uint8_t* dst, io_options opts = {}) noexcept;

  virtual exec::semi_future<size_t> device_read_async_io(const io_object& obj,
                                                         size_t offset,
                                                         size_t size,
                                                         uint8_t* dst,
                                                         ::cuda::stream_ref stream,
                                                         io_options opts = {}) noexcept;

  virtual exec::semi_future<size_t> host_readv_async_io(const io_object& obj,
                                                        std::span<const slice> slices,
                                                        io_options opts = {}) noexcept;

  virtual exec::semi_future<size_t> device_readv_async_io(const io_object& obj,
                                                          std::span<const slice> slices,
                                                          ::cuda::stream_ref stream,
                                                          io_options opts = {}) noexcept;

  /// The sole asynchronous backend read hook. All scalar/vector host/device
  /// APIs construct prepared slices and forward here; reactors perform physical
  /// chunking only after queue and slot pressure are known.  @p opts.cls is
  /// already resolved (never @c automatic) when reached through
  /// @ref host_device_readv_async_io.
  virtual exec::semi_future<size_t> mixed_readv_async_io(const io_object& obj,
                                                         std::vector<prepared_io_slice>&& slices,
                                                         io_options opts = {}) noexcept = 0;

  /// Funnel to @ref mixed_readv_async_io that resolves @c request_class::automatic
  /// from the slices' total byte count.
  exec::semi_future<size_t> host_device_readv_async_io(const io_object& obj,
                                                       std::vector<prepared_io_slice>&& slices,
                                                       io_options opts = {}) noexcept
  {
    if (opts.cls == request_class::automatic) {
      std::size_t bytes = 0;
      for (auto const& current : slices) {
        bytes += current.size();
      }
      opts.cls = resolve_request_class(opts.cls, io_kind::read, bytes);
    }
    return mixed_readv_async_io(obj, std::move(slices), opts);
  }

  bool can_use_prefetching_cache() const noexcept
  {
    return supports_vector_host_read() || supports_host_to_device_read();
  }

 protected:
  /// Backend hook: open native handles / resolve metadata for @p path and
  /// return a populated io_object.  Invoked by @c open_datasource; not part of
  /// the public surface (callers receive a ready @c datasource).  Throws
  /// on unsupported / unreachable paths.
  virtual std::shared_ptr<io_object> create_io_object(std::string path) = 0;

  /// Hinted variant.  The base implementation ignores @p hint and delegates to
  /// the required @c create_io_object(path); a backend that can act on the hint
  /// (e.g. rest_ioctx's suffix-range footer probe) overrides this.  Kept a
  /// distinct virtual — not a defaulted argument on the pure virtual above — so
  /// the hint dispatches on the dynamic type instead of binding statically.
  virtual std::shared_ptr<io_object> create_io_object(std::string path, open_hint hint);

  /// Known-size variant.  The base implementation ignores @p known_size and
  /// delegates to the required @c create_io_object(path); a backend whose size
  /// discovery would otherwise cost a round-trip overrides this to build the
  /// io_object without one.  Same distinct-virtual rationale as the hint
  /// variant above.
  virtual std::shared_ptr<io_object> create_io_object(std::string path, std::uint64_t known_size);

  // -- Write / runner backend hooks (defaults: not supported) --------------------

  /// Open @p path (scheme already stripped) for writing.  Default throws
  /// @c std::system_error(std::errc::not_supported).
  [[nodiscard]] virtual std::shared_ptr<io_object> create_io_object_for_write(
    std::string path, write_open_options opts);

  /// Synchronous host write on the calling thread.  @p opts.cls is resolved.
  /// Default throws @c std::system_error(std::errc::not_supported).
  virtual std::size_t host_write_io(const io_object& obj,
                                    std::size_t offset,
                                    std::size_t size,
                                    const std::uint8_t* src,
                                    write_options opts);

  /// The sole asynchronous backend write hook.  Segments are non-empty,
  /// validated disjoint, device sources carry a resolved @c device_id, and
  /// @p opts.cls is resolved.  Default resolves with @c not_supported.
  [[nodiscard]] virtual exec::semi_future<std::size_t> mixed_writev_async_io(
    const io_object& obj, std::vector<write_segment>&& segments, write_options opts) noexcept;

  /// Backend flush hook.  Default resolves with @c not_supported.
  [[nodiscard]] virtual exec::semi_future<void> flush_async_io(const io_object& obj) noexcept;

  /// Backend commit hook.  Default resolves with @c not_supported.
  [[nodiscard]] virtual exec::semi_future<void> commit_async_io(
    const io_object& obj, write_durability durability) noexcept;

  /// Runner hook behind run / run_for / run_until.  @p deadline is empty for
  /// @ref run.  Default throws @c std::logic_error("backend has no runners").
  virtual std::size_t run_impl(std::stop_token token,
                               std::optional<std::chrono::steady_clock::time_point> deadline);

  /// Owned by this ioctx.  Built by @ref initialize_cache, destroyed
  /// by @ref shutdown_cache (or the ioctx destructor as a safety net,
  /// though callers are expected to drive the lifecycle explicitly so
  /// reactors stay alive while workers drain).
  std::unique_ptr<cache::prefetching_cache> _cache;

  /// Independent of the prefetching machinery — exposed via @c metadata_store().
  cache::metadata_store _metadata_store;
};

}  // namespace cucascade::io
