/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cucascade/io/cache/config.hpp>
#ifdef CUCASCADE_HAS_CUFILE
#include <cucascade/io/cufile/config.hpp>
#endif
#ifdef CUCASCADE_HAS_KVIKIO
#include <cucascade/io/kvikio/config.hpp>
#endif
#include <cucascade/io/object_store_config.hpp>
#include <cucascade/io/rest/config.hpp>
#include <cucascade/io/uring/config.hpp>

#include <cstddef>

namespace cucascade::io {

/// IO backend that serves managed reads.
enum class io_backend {
  /// cuCascade's own IO stack: uring for local paths, REST for @c s3:// URLs.
  native,
  /// The kvikIO backend (drives @c kvikio::FileHandle directly).
  kvikio,
  /// The cuFile / GPUDirect Storage read reactor: local regular files are
  /// served by @c cufile_ioctx instead of uring (device destinations via
  /// @c cuFileRead, everything else via POSIX); @c s3:// URLs still go to the
  /// REST backend.  Needs a build that found @c cufile.h
  /// (@c CUCASCADE_HAS_CUFILE); without it no cuFile backend is registered and
  /// local paths resolve to nothing rather than to another backend.
  cufile,
};

/**
 * @brief Top-level configuration for the cucascade::io datasource layer.
 *
 * Consumed by @c io_context_registry and the per-backend ioctx factories.
 *
 * @c backend selects the IO stack: @ref io_backend::native routes local paths
 * to @c uring_ioctx and @c s3:// URLs to the REST backend, @ref io_backend::kvikio
 * routes both local paths and @c s3:// objects to @c kvikio_context (LIST still
 * goes to the REST backend), and @ref io_backend::cufile routes local regular
 * files to @c cufile_ioctx (@c s3:// stays on REST).
 *
 * @c cache is the read path's whole caching configuration, and its only home;
 * @ref apply_cache_mode derives @c uring.use_odirect and @c cache.dispose_on_idle
 * from it, so neither is settable on its own.
 *
 * Sub-configs:
 *  - @c uring   — uring reactor tunables (local-disk IO path).
 *  - @c rest    — REST reactor tunables (S3/object-store IO path).
 *  - @c kvikio  — kvikIO fallback tunables (local-disk catch-all path); present
 *    only when the library is built with CUCASCADE_BUILD_CUDF, which is what
 *    supplies kvikIO.
 *  - @c cufile  — cuFile / GDS read reactor tunables (local-disk GDS path);
 *    present only when the library is built with @c cufile.h
 *    (@c CUCASCADE_HAS_CUFILE).  Independent of kvikIO.
 *  - @c cache   — caching mode, eviction policy and prefetching-cache tunables.
 *  - @c object_store — object-store credentials and endpoint.
 */
struct io_config {
  /// IO backend that serves managed reads.
  io_backend backend{io_backend::native};

  /// Number of uring reactor worker threads for the local-disk IO path.
  std::size_t uring_n_reactors{4};

  /// Number of REST reactor worker threads for the S3/object-store IO path
  /// (each its own libcurl event loop + connection pool).
  std::size_t rest_n_reactors{2};

  /// Local (uring) reactor configuration.  @c use_odirect is derived from
  /// @ref cache; physical operation size is selected by the worker.
  uring::config uring{};

  /// REST (S3/object-store) reactor configuration — timeouts, TLS, logical
  /// merge hints, retry policy, and connection limits.  Physical GET sizing is
  /// worker-owned.
  rest::config rest{};

#ifdef CUCASCADE_HAS_KVIKIO
  /// kvikIO fallback configuration — thread-pool size, task/bounce sizing,
  /// O_DIRECT, compat mode.  All fields default to "unset", leaving kvikIO's
  /// own env-var-seeded defaults in place.  Note these are process-global once
  /// applied; see @ref kvikio_config.
  kvikio_config kvikio{};
#endif

#ifdef CUCASCADE_HAS_CUFILE
  /// Number of cuFile reactor worker threads for the GDS local-disk IO path.
  /// Each reactor serves one synchronous transfer at a time; a vectored read
  /// with more than two slices fans out across all of them.
  std::size_t cufile_n_reactors{4};

  /// cuFile / GDS read reactor configuration.  Only consulted when
  /// @c backend == @ref io_backend::cufile.  Not derived from @ref cache: the
  /// cuFile backend never arms the prefetching cache.
  cufile::config cufile{};
#endif

  /// The read path's caching configuration: mode, eviction policy and the
  /// prefetching cache's tunables.  One block, rather than a mode here and the
  /// tunables in a sibling one.
  cache::config cache{};

  /// Object-store credentials and endpoint consumed by the REST reactor.
  /// Empty fields disable the S3/REST backend.
  object_store_config object_store{};

  /// Refresh the knobs derived from @ref cache.
  void apply_cache_mode() noexcept
  {
    cache.apply_mode();
    uring.use_odirect = cache.use_odirect();
  }
};

}  // namespace cucascade::io
