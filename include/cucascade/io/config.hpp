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
#include <cucascade/io/kvikio/config.hpp>
#include <cucascade/io/object_store_config.hpp>
#include <cucascade/io/rest/config.hpp>
#include <cucascade/io/uring/config.hpp>

#include <cstddef>

namespace cucascade::io {

/**
 * @brief Storage caching behavior selected at the io_context level.
 *
 * A single knob for two coupled decisions — whether local reads bypass the OS
 * page cache with O_DIRECT, and whether cucascade's prefetching cache is on:
 *   - @c none     : O_DIRECT on,  prefetching cache off  (direct, uncached)
 *   - @c os       : O_DIRECT off, prefetching cache off  (rely on the OS page cache)
 *   - @c prefetch : O_DIRECT on,  prefetching cache on   (cucascade prefetching cache)
 */
enum class cache_level {
  none,
  os,
  prefetch,
};

/// True when @p level reads through O_DIRECT (i.e. everything but @c os).
[[nodiscard]] constexpr bool odirect_enabled(cache_level level) noexcept
{
  return level != cache_level::os;
}

/// True when @p level enables the prefetching cache (only @c prefetch).
[[nodiscard]] constexpr bool prefetch_enabled(cache_level level) noexcept
{
  return level == cache_level::prefetch;
}

/**
 * @brief Top-level configuration for the cucascade::io datasource layer.
 *
 * Consumed by @c io_context_registry and the per-backend ioctx factories.
 *
 * Sub-configs:
 *  - @c local   — uring reactor tunables (local-disk IO path).
 *  - @c rest    — REST reactor tunables (S3/object-store IO path).
 *  - @c kvikio  — kvikIO fallback tunables (local-disk catch-all path).
 *  - @c cache   — prefetching cache tunables.
 *  - @c object_store — object-store credentials and endpoint.
 */
struct io_config {
  /// Number of uring reactor worker threads for the local-disk IO path.
  std::size_t uring_n_reactors{1};

  /// Number of REST reactor worker threads for the S3/object-store IO path
  /// (each its own libcurl event loop + connection pool).
  std::size_t rest_n_reactors{2};

  /// Storage caching behavior (O_DIRECT + prefetching cache).  Sources
  /// @c local.use_odirect and whether the prefetching cache is enabled via
  /// @c odirect_enabled() / @c prefetch_enabled().  Defaults to @c none
  /// (O_DIRECT on, prefetching cache off).
  cache_level caching{};

  /// Local (uring) reactor configuration — bounce-slot size, O_DIRECT,
  /// ring depth, etc.  @c local.use_odirect is derived from @c cache_level
  /// when the ioctx is built through the datasource factory.
  uring::config local{};

  /// REST (S3/object-store) reactor configuration — timeouts, TLS, chunking,
  /// retry policy, etc.
  rest::config rest{};

  /// kvikIO fallback configuration — thread-pool size, task/bounce sizing,
  /// O_DIRECT, compat mode.  All fields default to "unset", leaving kvikIO's
  /// own env-var-seeded defaults in place.  Note these are process-global once
  /// applied; see @ref kvikio_config.
  kvikio_config kvikio{};

  /// Prefetching cache configuration — in-flight budget, pool sizing,
  /// dispose-after-use policy.
  cache::config cache{};

  /// Object-store credentials and endpoint consumed by the REST reactor.
  /// Empty fields disable the S3/REST backend.
  object_store_config object_store{};
};

}  // namespace cucascade::io
