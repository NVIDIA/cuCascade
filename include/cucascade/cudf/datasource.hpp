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

#include <cucascade/exec/invocable.hpp>
#include <cucascade/io/cache/fs_cache.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/types.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/version_config.hpp>

#if CUDF_VERSION_MAJOR < 26 || (CUDF_VERSION_MAJOR == 26 && CUDF_VERSION_MINOR < 12)
#include <rmm/cuda_stream_view.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <span>

namespace cucascade::io {

#if CUDF_VERSION_MAJOR > 26 || (CUDF_VERSION_MAJOR == 26 && CUDF_VERSION_MINOR >= 12)
using cudf_stream_type = ::cuda::stream_ref;
#else
using cudf_stream_type = rmm::cuda_stream_view;
#endif

// ---------------------------------------------------------------------------
// datasource
// ---------------------------------------------------------------------------

/**
 * @brief Concrete @c cudf::io::datasource backed by a cucascade::io backend.
 *
 * The cudf bridge over the cudf-free io core: every read forwards to the
 * ioctx's cache-aware read APIs (which return @c exec::semi_future) and this
 * layer converts to the @c std::future / @c datasource::buffer shapes cudf
 * expects.  Lives in the cucascade-cudf bundle — the io core has no cudf
 * dependency.
 *
 * Ownership model: one scan owns one @c datasource.  The underlying
 * @c io_object can be shared across multiple datasources (e.g. when
 * the same file is scanned in different pipelines), but the datasource
 * itself stores per-scan state (notably the @c cache_handle returned
 * by an @c fadvise call) and is therefore not safe to share.
 */
/// Why a datasource did or did not start a prefetch, so the readahead can
/// attribute a refusal instead of only counting one.  A refusal is normal --
/// the readahead offers work the read path is free to turn down -- but the
/// reason decides whether it means "we are out of memory" or "we are too late".
enum class prefetch_refusal : std::uint8_t {
  /// IO went out.
  issued,
  /// No prefetching cache for this scan, so there was never anything to issue.
  no_cache,
  /// The executor had already started reading this split: prefetching now would
  /// issue the same IO a second time and race the reader for its chunks.
  consumer_ahead,
  /// The cache could not retain staging buffers through the point where IO was
  /// issued.  Normal allocation pressure is reported earlier by prepare().
  memory_pressure,
  /// The cache refused for a reason of its own (already loading, request
  /// cancelled, backend cannot serve vectored host reads).
  other,
};

/// How preparing one datasource's prefetch request turned out.  Kept apart from
/// a plain bool because "there was nothing to prepare" and "the pool had nothing
/// to give" are opposite answers to "should the readahead be worried".
enum class prepare_result : std::uint8_t {
  /// The request owns staging buffers and its chunks can now be claimed.
  prepared,
  /// The pool could not satisfy the request. It remains queued for a retry.
  allocation_failed,
  /// The consumer reached this split before preparation completed.
  fallen_behind,
  /// No request on this datasource: no prefetching cache, or no fadvise.
  nothing_to_prepare,
};

class datasource : public cudf::io::datasource {
 public:
  explicit datasource(std::shared_ptr<ioctx> io_ctx, std::shared_ptr<io_object> io_obj);

  ~datasource() override;

  datasource(datasource const&)            = delete;
  datasource& operator=(datasource const&) = delete;

  // ---- Context accessors ---------------------------------------------------

  [[nodiscard]] std::shared_ptr<ioctx> io_ctx() const { return _io_ctx; }

  /// The underlying io_object this datasource reads through.  Exposed so
  /// callers that received the datasource from @c open_datasource can still
  /// reach the io_object (e.g. as the metadata-store cache key).
  [[nodiscard]] const io_object& get_io_object() const noexcept { return *_io_object; }

  /// Backend-parsed metadata for this datasource's io_object, looked up in the
  /// ioctx's metadata store (null when no cache or no entry). Independent of
  /// the prefetching machinery.
  [[nodiscard]] std::shared_ptr<io_object_metadata> metadata() const;

  [[nodiscard]] bool store_metadata(std::shared_ptr<io_object_metadata> metadata);

  // ---- cudf::io::datasource overrides ---------------------------------------

  [[nodiscard]] size_t size() const override;

  [[nodiscard]] bool supports_device_read() const override;

  [[nodiscard]] bool supports_vector_host_read() const;

  [[nodiscard]] bool is_device_read_preferred(size_t) const override;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t size) override;

  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst) override;

  std::future<std::unique_ptr<datasource::buffer>> host_read_async(size_t offset,
                                                                   size_t size) override;

  std::unique_ptr<datasource::buffer> device_read(size_t offset,
                                                  size_t size,
                                                  cudf_stream_type stream) override;
  size_t device_read(size_t offset, size_t size, uint8_t* dst, cudf_stream_type stream) override;

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        cudf_stream_type stream) override;

  /// \brief Vectored form of @c device_read_async: read every range into its own
  /// device destination in a single dispatch.
  ///
  /// \note Not a @c cudf::io::datasource override — cudf has no batched device
  /// read.  Callers holding many ranges (e.g. a parquet scan's column chunks)
  /// should prefer this over one @c device_read_async per range: it costs one
  /// request instead of N, and lets the backend fuse and order the whole batch.
  std::future<size_t> device_read_ranges_async(std::span<const slice> slices,
                                               ::cuda::stream_ref stream);

  std::future<size_t> host_read_ranges_async(std::span<const slice> slices);

  // ---- Advisory IO ---------------------------------------------------------

  /// \brief Return a fresh datasource that shares this one's @c ioctx and
  /// @c io_object (so it points at the same file) but carries an
  /// empty @c cache_handle.
  ///
  /// \note Used when a single file is split across multiple scans (e.g. several
  /// row_group_slices from the same parquet file).  Each split owns its
  /// own datasource via @c duplicate so it can call @c fadvise without
  /// stomping on another scan's handle — io_objects are deliberately
  /// shareable across datasources, but handles are not.
  [[nodiscard]] std::unique_ptr<datasource> duplicate() const;

  /// \brief Hint the IO layer about @p ranges that this scan will (or might) read
  /// soon.
  ///
  /// Hands @p ranges to the prefetching cache, stashes the returned
  /// @c cache_handle on this datasource (which disposes the request when
  /// it goes away) and drives it to @c scan_stage::initialized.  No-op when the
  /// cache is unavailable.  A second inserting call while an active handle is
  /// already stored is a caller bug and only logs a warning: the datasource
  /// lifecycle expects one insert per scan.
  void fadvise(std::span<const cudf::io::text::byte_range_info> ranges, std::optional<int> dev_id);

  /// Drive the stashed handle's consumer stage to @p site.
  void update(cache::scan_stage site);

  /// Allocate staging buffers for the stashed request, ahead of prefetching it.
  /// @p wait_for_eviction lets the call wait on the evictor rather than fail on
  /// a momentarily empty pool.  See @c fs_cache::prepare.
  prepare_result prepare_prefetch(bool wait_for_eviction);

  /// Issue prefetch IO for the stashed handle.  @p on_done fires exactly once
  /// with the outcome — inline when no IO is issued, otherwise from the IO
  /// completion.  Returns @c prefetch_refusal::issued when IO went out, and
  /// otherwise why it did not.
  prefetch_refusal prefetch_async(exec::invocable<void(bool) noexcept> on_done);

  [[nodiscard]] bool uses_fs_cache() const noexcept;

  /// Diagnostics: nanoseconds demand reads through this datasource spent blocked
  /// on its in-flight prefetch (see @c cache::cache_handle::demand_wait_ns).
  [[nodiscard]] std::uint64_t demand_wait_ns() const noexcept
  {
    return _prefetch_handle.demand_wait_ns();
  }

  /// Diagnostics: cache chunks named by this datasource's prefetch request (0
  /// without one).
  [[nodiscard]] std::size_t cache_chunk_count() const noexcept
  {
    auto const chunks = _prefetch_handle.chunks();
    return chunks ? chunks->size() : 0;
  }

  /// Whether the backend serving this datasource would rather be handed one
  /// batched request than a stream of small reads.  See @c ioctx::prefers_bulk_io.
  [[nodiscard]] bool prefers_bulk_io() const noexcept;

 private:
  [[nodiscard]] bool uses_fs_cache();

  std::shared_ptr<ioctx> _io_ctx;
  std::shared_ptr<io_object> _io_object;
  /// Handle of the most recent insert into the prefetching cache, or empty
  /// if none was made.  Disposing it lets the cache reclaim the request.
  cache::cache_handle _prefetch_handle;
};

/// Open a datasource for @p path on @p io_ctx: creates the backend-appropriate
/// io_object (local fds / object-store HEAD / ...) and wraps it in a
/// @c datasource bound to that ioctx.  Throws on unsupported / unreachable
/// paths (callers that want a check-without-open should use
/// @c ioctx::supports()).
[[nodiscard]] std::unique_ptr<datasource> open_datasource(std::shared_ptr<ioctx> io_ctx,
                                                          std::string path);

/// As above, forwarding @p hint to the backend's io_object resolution so it can,
/// e.g., prefetch a parquet footer in the same round-trip as the size
/// (@c open_hint::parquet_footer_probe).  Backends that cannot act on the hint
/// fall back to the plain open.
[[nodiscard]] std::unique_ptr<datasource> open_datasource(std::shared_ptr<ioctx> io_ctx,
                                                          std::string path,
                                                          open_hint hint);

/// As above, with the object's size already known (e.g. from an S3
/// ListObjectsV2 response), so a backend that can act on it skips its size
/// discovery entirely (no HEAD for object stores).
[[nodiscard]] std::unique_ptr<datasource> open_datasource(std::shared_ptr<ioctx> io_ctx,
                                                          std::string path,
                                                          std::uint64_t known_size);

}  // namespace cucascade::io
