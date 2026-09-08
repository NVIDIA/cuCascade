/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>
#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>

#include <cstddef>
#include <memory>
#include <string>

namespace cucascade::io {

/**
 * @brief Self-contained io_uring datasource engine for local NVMe reads.
 *
 * Owns the full stack needed for O_DIRECT io_uring reads: a NUMA-local pinned
 * host memory pool, a pool of io_uring reactor threads, and an @c ioctx.
 * Callers open individual @c datasource instances via @c open(); each datasource
 * shares the engine's @c ioctx and memory pool but carries its own per-scan
 * @c prefetching_handle.
 *
 * The engine must outlive every datasource it produces.
 *
 * @code{.cpp}
 * cucascade::io::uring_datasource_engine engine;
 * auto ds = engine.open("/mnt/nvme/data/lineitem.parquet");
 * ds->fadvise(byte_ranges, device_id);
 * // ... read through ds as a cudf::io::datasource ...
 * @endcode
 */
class uring_datasource_engine {
 public:
  static constexpr std::size_t default_block_size    = 1UL << 20;   ///< 1 MiB
  static constexpr std::size_t default_pool_capacity = 20UL * 128UL * (1UL << 20);  ///< ~2.5 GiB

  /**
   * @brief Construct a uring datasource engine.
   *
   * @param n_reactors    Number of io_uring reactor threads.
   * @param pool_capacity Total capacity of the pinned host staging pool in bytes.
   * @param block_size    Size of each fixed-size block in the staging pool in bytes.
   *                      Must be a power of two and at least the alignment required
   *                      by O_DIRECT on the target filesystem.
   * @param use_odirect   Whether to open files with @c O_DIRECT (bypasses page cache).
   * @param numa_node     NUMA node from which to allocate the pinned staging pool.
   */
  explicit uring_datasource_engine(std::size_t n_reactors   = 2,
                                   std::size_t pool_capacity = default_pool_capacity,
                                   std::size_t block_size    = default_block_size,
                                   bool        use_odirect   = true,
                                   int         numa_node     = 0);

  ~uring_datasource_engine();

  uring_datasource_engine(uring_datasource_engine const&)            = delete;
  uring_datasource_engine& operator=(uring_datasource_engine const&) = delete;

  /**
   * @brief Open a datasource for the given local file path.
   *
   * @param path Absolute or relative path to a local file.
   * @return A @c datasource bound to this engine's @c ioctx. The returned
   *         datasource must not outlive this engine.
   * @throw std::runtime_error if the file cannot be opened.
   */
  [[nodiscard]] std::unique_ptr<datasource> open(std::string path) const;

 private:
  cucascade::memory::numa_region_pinned_host_memory_resource _upstream;
  cucascade::memory::fixed_size_host_memory_resource         _host_mr;
  std::shared_ptr<uring::uring_reactor::reactor_context>     _reactor_ctx;
  std::shared_ptr<ioctx>                                     _io_ctx;
};

}  // namespace cucascade::io
