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

#include <cucascade/io/cache/prefetching_cache.hpp>
#include <cucascade/io/io_context.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <span>

namespace cucascade::io {

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
 * itself stores per-scan state (notably the @c prefetching_handle returned
 * by an @c fadvise call) and is therefore not safe to share.
 */
class datasource : public cudf::io::datasource {
 public:
  explicit datasource(std::shared_ptr<ioctx> io_ctx, std::shared_ptr<io_object> io_object);

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
                                                  rmm::cuda_stream_view stream) override;
  size_t device_read(size_t offset,
                     size_t size,
                     uint8_t* dst,
                     rmm::cuda_stream_view stream) override;

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view stream) override;

  // ---- Advisory IO ---------------------------------------------------------

  /// \brief Return a fresh datasource that shares this one's @c ioctx and
  /// @c io_object (so it points at the same file) but carries an
  /// empty @c prefetching_handle.
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
  /// The behaviour depends on @p site and the io_ctx's
  /// @c preferred_prefetching_stage:
  ///   - @c speculative / @c immediate: only honored when @p site matches
  ///     the ioctx's preferred mode.  Hands @p ranges to the prefetching
  ///     cache and stashes the returned @c prefetching_handle on this
  ///     datasource so a later @c fadvise(disposable) can cancel.
  ///   - @c disposable: always honored.  If a handle is stored (i.e. a
  ///     prior speculative/immediate call enqueued work), cancel it so the
  ///     cache worker drops still-pending entries.
  ///   - @c none: no-op (either the call site asked for nothing, or the
  ///     backend opted out of prefetching).
  ///
  /// Calling @c fadvise with a non-disposable @p site while a handle is
  /// already stored emits a warning: the datasource lifecycle expects one
  /// speculative-or-immediate insert per scan, with a single
  /// @c disposable call at consume time.
  void fadvise(std::span<const cudf::io::text::byte_range_info> ranges, std::optional<int> dev_id);

  void prefetch(cache::prefetching_stage site);

 private:
  [[nodiscard]] bool uses_prefetching_cache();

  std::shared_ptr<ioctx> _io_ctx;
  std::shared_ptr<io_object> _io_object;
  /// Handle of the most recent speculative/immediate insert into the
  /// prefetching cache, or empty if none was made.  fadvise(disposable)
  /// uses this to cancel still-pending work.
  cache::prefetching_handle _prefetch_handle;
};

/// Open a datasource for @p path on @p io_ctx: creates the backend-appropriate
/// io_object (local fds / object-store HEAD / ...) and wraps it in a
/// @c datasource bound to that ioctx.  Throws on unsupported / unreachable
/// paths (callers that want a check-without-open should use
/// @c ioctx::supports()).
[[nodiscard]] std::unique_ptr<datasource> open_datasource(std::shared_ptr<ioctx> io_ctx,
                                                          std::string path);

}  // namespace cucascade::io
