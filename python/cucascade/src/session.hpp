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

#include <cucascade/cudf/column_admission.hpp>

#include <cudf/io/datasource.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cucascade::python {

/// Settings of the process-wide cuCascade session behind the Python bindings.
struct session_options {
  /// Size of the pinned host cache, in bytes. All of it is allocated when the session is created.
  std::size_t cache_bytes{0};
  /// Cache chunk size, in bytes. 256 KiB measured best on S3: smaller chunks cost more in
  /// per-chunk bookkeeping than they save in padding.
  std::size_t block_bytes{256UL << 10};
  /// Share of the pool in use at which the evictor starts. Leave headroom below 1.0 for the pinned
  /// staging blocks of reads that are not cached, which come from the same pool.
  double eviction_threshold{0.97};
  /// Share of the pool one eviction pass frees.
  double eviction_batch{0.02};
  /// Share of the cache that admitted columns may fill, counted in whole cache chunks.
  double admission_budget{0.9};
  /// Reads smaller than this are not cached. 0 caches nothing.
  std::size_t min_advise_bytes{256UL << 10};
  /// REST reactor threads and the connections each one drives.
  std::size_t reactors{16};
  std::size_t connections{128};
  /// A GET that moves under 64 KiB/s for this many seconds is aborted and retried.
  long stall_seconds{5};
  /// Threads that pin the cache at startup.
  std::size_t pinning_threads{8};
  /// S3 endpoint, region and credentials.
  std::string endpoint;
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string session_token;
};

/// Create the session: pin the cache, start the REST reactors and arm the cache. Does nothing if
/// a session exists already.
void start_session(session_options const& options);

/// Open an @c s3:// object whose reads are advised to the cache. The first open of a file resolves
/// its size, version tag and footer with one request; later opens of it need none.
std::unique_ptr<cudf::io::datasource> open(std::string const& path);

/// Register the column chunks of one file with the admission policy (see
/// @ref cucascade::io::column_admission::register_layout).
void register_layout(std::string const& path,
                     std::string const& table,
                     std::vector<std::string> const& columns,
                     std::vector<std::uint32_t> const& column_of_chunk,
                     std::vector<std::uint64_t> const& offsets,
                     std::vector<std::uint64_t> const& sizes);

/// Per-column admission accounting.
std::vector<cucascade::io::column_stats> column_stats();

/// Counters of the cache (reads, hits, misses, evictions) as text; empty without a session.
std::string cache_summary();

/// Drain the cache and the reactors. Every datasource must be destroyed first.
void stop_session();

}  // namespace cucascade::python
