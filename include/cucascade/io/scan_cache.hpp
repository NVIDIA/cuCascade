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

#include <cucascade/io/advising_reader.hpp>
#include <cucascade/io/column_admission.hpp>
#include <cucascade/io/datasource_factory.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/object_cache.hpp>
#include <cucascade/io/object_store_config.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace cucascade::io {

/// Settings of a @ref scan_cache.
struct scan_cache_config {
  /// Size of the pinned host cache, in bytes. All of it is allocated when the cache is created.
  std::size_t cache_bytes{0};

  /// Cache chunk size, in bytes (see @ref default_cache_chunk_bytes).
  std::size_t block_bytes{default_cache_chunk_bytes};

  /// Share of the pool in use at which the evictor starts. Leave headroom below 1.0 for the pinned
  /// staging blocks of reads that are not cached, which come from the same pool.
  double eviction_threshold{0.97};

  /// Share of the pool that one eviction pass frees.
  double eviction_batch{0.02};

  /// Share of the cache that admitted columns may fill, counted in whole cache chunks.
  double admission_budget{0.96};

  /// Reads smaller than this are not cached. 0 caches nothing.
  std::size_t min_advise_bytes{default_cache_chunk_bytes};

  /// REST reactor threads.
  std::size_t reactors{4};

  /// Connections each REST reactor drives. Reading from S3, 4 x 256 was faster than 16 x 128,
  /// which spent time on connect and stall retries.
  std::size_t connections{256};

  /// A GET that moves under 64 KiB/s for this many seconds is aborted and retried, so a request
  /// that gets no bytes does not hold a query up for the 30 s default.
  long stall_seconds{5};

  /// Threads that pin the cache when it is created.
  std::size_t pinning_threads{8};

  /// NUMA node to place the cache on. Unset places it on the node of the current CUDA device, found
  /// from the system topology, and falls back to -1 if that is not known. -1 leaves the placement
  /// to the CUDA driver, which pins one pool at a time: about 0.3 s per GiB, against about 0.2 s
  /// per GiB on a node, and the driver path cannot use more than one thread.
  std::optional<int> numa_node;

  /// Object store endpoint, region, credentials, TLS and transport. Requests are signed in the
  /// @c Authorization header unless this says otherwise.
  object_store_config object_store = [] {
    object_store_config store;
    store.s3_signing_mode = object_store_config::signing_mode::header;
    return store;
  }();
};

/**
 * @brief A host-memory cache for the parquet reads of object-store scans, as a single object.
 *
 * Assembles what a reader needs to use the prefetching cache: the pinned host cache, the REST io
 * context, a @ref column_admission policy that decides which columns to keep, and an
 * @ref object_cache that makes reopening a file free. @ref open returns an @ref advising_reader
 * that a reader reads through; its reads are advised to the cache under the admission policy.
 *
 * The cache is pinned when the object is created, which takes about 0.2 s per GiB when it is placed
 * on a NUMA node (the default), so that the first query does not pay for it.
 *
 * Every reader from @ref open shares ownership of the cache, so a reader may outlive this object:
 * the cache is released when the last of them is destroyed.
 *
 * Thread-safe.
 */
class scan_cache {
 public:
  /// @throws std::invalid_argument If @c config.cache_bytes is 0, @c config.block_bytes is 0 or
  /// larger than one 128 MiB pool, or a fraction is not in [0, 1].
  /// @throws std::runtime_error If the REST io context or the prefetching cache cannot be created.
  explicit scan_cache(scan_cache_config const& config);
  ~scan_cache();

  scan_cache(scan_cache const&)            = delete;
  scan_cache& operator=(scan_cache const&) = delete;
  scan_cache(scan_cache&&)                 = delete;
  scan_cache& operator=(scan_cache&&)      = delete;

  /**
   * @brief Open an object-store object for reading.
   *
   * The first open of a path resolves its size, version tag and footer with one request, shared by
   * concurrent first opens; later opens of it need none. The reader advises its reads to the cache
   * under the admission policy (see @ref advising_reader).
   */
  [[nodiscard]] std::unique_ptr<advising_reader> open(std::string const& path);

  /// Register the column chunks of one file with the admission policy (see
  /// @ref column_admission::register_layout).
  void register_layout(std::string const& path,
                       std::string const& table,
                       std::vector<std::string> const& columns,
                       std::span<std::uint32_t const> column_of_chunk,
                       std::span<std::uint64_t const> offsets,
                       std::span<std::uint64_t const> sizes);

  /// Per-column admission accounting.
  [[nodiscard]] std::vector<column_stats> stats() const;

  /// Counters of the prefetching cache (reads, hits, misses, evictions) as text.
  [[nodiscard]] std::string summary() const;

 private:
  /// What the cache is made of, owned jointly by this object and every reader it opened.
  struct state;

  std::shared_ptr<state> _state;
  std::size_t _min_advise_bytes;
};

}  // namespace cucascade::io
