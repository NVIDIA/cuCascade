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
#ifdef CUCASCADE_HAS_KVIKIO
#include <cucascade/io/kvikio/config.hpp>
#endif
#include <cucascade/io/object_store_config.hpp>
#include <cucascade/io/rest/config.hpp>
#include <cucascade/io/uring/config.hpp>

#include <cstddef>

namespace cucascade::io {

/**
 * @brief Top-level configuration for the cucascade::io datasource layer.
 *
 * Consumed by @c io_context_registry and the per-backend ioctx factories.
 *
 * Sub-configs:
 *  - @c local   — uring reactor tunables (local-disk IO path).
 *  - @c rest    — REST reactor tunables (S3/object-store IO path).
 *  - @c kvikio  — kvikIO fallback tunables (local-disk catch-all path); present
 *    only when the library is built with CUCASCADE_BUILD_CUDF, which is what
 *    supplies kvikIO.
 *  - @c cache   — prefetching cache tunables.
 *  - @c object_store — object-store credentials and endpoint.
 */
struct io_config {
  /// Number of uring reactor worker threads for the local-disk IO path.
  std::size_t uring_n_reactors{1};

  /// Number of REST reactor worker threads for the S3/object-store IO path
  /// (each its own libcurl event loop + connection pool).
  std::size_t rest_n_reactors{2};

  /// Enable the prefetching cache on the ioctx.  When false the cache is
  /// constructed but unarmed (no background IO threads).
  bool enable_prefetch_cache{false};

  /// Local (uring) reactor configuration — bounce-slot size, O_DIRECT,
  /// ring depth, etc.
  uring::config local{};

  /// REST (S3/object-store) reactor configuration — timeouts, TLS, chunking,
  /// retry policy, etc.
  rest::config rest{};

#ifdef CUCASCADE_HAS_KVIKIO
  /// kvikIO fallback configuration — thread-pool size, task/bounce sizing,
  /// O_DIRECT, compat mode.  All fields default to "unset", leaving kvikIO's
  /// own env-var-seeded defaults in place.  Note these are process-global once
  /// applied; see @ref kvikio_config.
  kvikio_config kvikio{};
#endif

  /// Prefetching cache configuration — in-flight budget, pool sizing,
  /// dispose-after-use policy.
  cache::config cache{};

  /// Object-store credentials and endpoint consumed by the REST reactor.
  /// Empty fields disable the S3/REST backend.
  object_store_config object_store{};
};

}  // namespace cucascade::io
