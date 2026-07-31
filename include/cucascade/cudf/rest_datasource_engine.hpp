/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <cstddef>
#include <memory>
#include <string>

namespace cucascade::io {

class ioctx;

/**
 * @brief Self-contained REST datasource engine for S3/HTTP object-store reads.
 *
 * Owns a NUMA-local pinned host staging pool and a pool of libcurl reactor
 * threads backed by SigV4 presigned-URL signing. Callers open individual
 * @c datasource instances via @c open(); each datasource shares the engine's
 * @c ioctx and memory pool but carries its own per-scan @c prefetching_handle.
 *
 * The engine must outlive every datasource it produces.
 *
 * @code{.cpp}
 * auto engine = cucascade::io::rest_datasource_engine::make_s3(...);
 * auto ds = engine->open("s3://my-bucket/data/lineitem.parquet");
 * ds->fadvise(byte_ranges, device_id);
 * @endcode
 */
class rest_datasource_engine {
 public:
  static constexpr std::size_t default_block_size    = 1UL << 20;
  static constexpr std::size_t default_pool_capacity = 20UL * 128UL * (1UL << 20);

  /**
   * @brief Construct an S3-backed REST engine with SigV4 presigned-URL signing.
   *
   * Credentials are static for the lifetime of the engine. For short-lived STS
   * tokens, reconstruct the engine before the token expires.
   *
   * @param access_key_id     AWS access key ID.
   * @param secret_access_key AWS secret access key.
   * @param session_token     STS session token; empty for long-lived credentials.
   * @param region            AWS region (e.g. @c "us-east-1").
   * @param endpoint          S3-compatible endpoint host (e.g. @c "s3.amazonaws.com"
   *                          or a MinIO host:port). Leave empty to derive from region.
   * @param n_reactors        Number of libcurl reactor threads.
   * @param tls_verify        Whether to verify TLS peer certificates.
   * @param pool_capacity     Total capacity of the pinned host staging pool in bytes.
   * @param block_size        Fixed block size in bytes for the staging pool.
   * @return A ready-to-use engine. The engine's @c ioctx is started before returning.
   */
  explicit rest_datasource_engine(std::string access_key_id,
                                  std::string secret_access_key,
                                  std::string session_token,
                                  std::string region,
                                  std::string endpoint,
                                  std::size_t n_reactors      = 4,
                                  bool        tls_verify      = true,
                                  std::size_t pool_capacity   = default_pool_capacity,
                                  std::size_t block_size      = default_block_size,
                                  std::size_t max_connections = 16,
                                  std::size_t chunk_size      = 8UL << 20,
                                  std::size_t max_n_chunks    = 16,
                                  bool        enable_cache    = false);

  ~rest_datasource_engine();

  rest_datasource_engine(rest_datasource_engine const&)            = delete;
  rest_datasource_engine& operator=(rest_datasource_engine const&) = delete;

  /**
   * @brief Open a datasource for the given S3 URI.
   *
   * Issues an HTTP HEAD request to resolve the object size.
   *
   * @param path S3 URI of the form @c "s3://bucket/key".
   * @return A @c datasource bound to this engine's @c ioctx. The returned
   *         datasource must not outlive this engine.
   * @throw std::runtime_error if the HEAD request fails or the URI is malformed.
   */
  [[nodiscard]] std::unique_ptr<datasource> open(std::string path) const;

 private:
  cucascade::memory::numa_region_pinned_host_memory_resource          _upstream;
  cucascade::memory::fixed_size_host_memory_resource                  _host_mr;
  std::shared_ptr<ioctx>                                              _io_ctx;
  std::unique_ptr<cucascade::memory::memory_reservation_manager>      _reservation_manager;
};

}  // namespace cucascade::io
