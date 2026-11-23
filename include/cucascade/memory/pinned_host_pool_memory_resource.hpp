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

#include <cucascade/cuda/stream.hpp>
#include <cucascade/memory/detail/pinned_host_pool_memory_resource_impl.hpp>

#include <cuda/memory_resource>

#include <cstddef>

namespace cucascade {
namespace memory {

/**
 * @brief Settings of a `pinned_host_pool_memory_resource`, fixed at construction.
 *
 * The defaults are deliberately invalid and must be set.
 */
struct pinned_host_pool_config {
  int numa_node{-1};              ///< Requested NUMA node for the pool. Use 0 on non-NUMA systems.
  std::size_t max_pool_bytes{0};  ///< Byte limit of the pool; not 0.
};

/**
 * @brief A pinned host memory resource backed by a CUDA HostNuma memory pool.
 *
 * The resource requests `pinned_host_pool_config::numa_node` when creating the CUDA pool.
 *
 * The handle is a cheap copyable value: copies share one pool and compare equal, handles over
 * different pools compare unequal, and the pool is destroyed with the last handle. A moved-from
 * handle may only be assigned to or destroyed. Allocation, deallocation and the counters are
 * thread-safe.
 *
 * `allocate`, `deallocate`, `allocate_sync`, `deallocate_sync` and equality come from
 * `::cuda::mr::shared_resource`, which forwards them to
 * `detail::pinned_host_pool_memory_resource_impl`. Their rules:
 *
 * - `alignment` must be a power of two no larger than
 *   `detail::pinned_host_pool_memory_resource_impl::pool_alignment`; returned pointers are
 *   aligned to that constant.
 * - Allocation throws `cucascade::logic_error` for invalid alignment. Deallocation asserts
 *   on invalid alignment in debug builds. With valid alignment, allocating 0 bytes returns
 *   `nullptr` and deallocating `nullptr` or 0 bytes is a no-op.
 * - `allocate` and `deallocate` are ordered on the caller's stream. `allocate_sync` and
 *   `deallocate_sync` use a private stream on device 0 and wait for it.
 * - With valid alignment, a single request larger than `max_pool_bytes()` throws
 *   `rmm::out_of_memory`. CUDA manages total pool growth using the configured size, and
 *   `reserved_bytes()` may exceed it.
 * - The host may use memory from `allocate(stream, ...)` only after the allocation on `stream`
 *   has completed, and memory from `allocate_sync` at once. `deallocate_sync` requires every use
 *   of the memory to have finished.
 * - Callers must order each use after allocation and each free after allocation and all uses,
 *   including across streams.
 * - Calls that reach CUDA on the legacy default stream or `cudaStreamPerThread` require a
 *   current CUDA context on the calling thread.
 * - Submit a free for every allocation before destroying the last handle. Handles must not
 *   live in static storage.
 */
class pinned_host_pool_memory_resource
  : public ::cuda::mr::shared_resource<detail::pinned_host_pool_memory_resource_impl> {
  using shared_base = ::cuda::mr::shared_resource<detail::pinned_host_pool_memory_resource_impl>;

 public:
  /**
   * @brief Creates a new pool.
   *
   * @param config The pool settings.
   * @throws cucascade::logic_error if `config` breaks a rule of `pinned_host_pool_config`.
   * @throws cucascade::cuda_error if CUDA cannot be initialized or cannot create the pool or the
   *         private stream, e.g., when no GPU is visible, the node does not exist, or the driver
   *         lacks HostNuma pool support.
   */
  explicit pinned_host_pool_memory_resource(pinned_host_pool_config const& config);

  /**
   * @brief Returns the configured byte limit.
   *
   * @return `pinned_host_pool_config::max_pool_bytes` as given at construction.
   */
  [[nodiscard]] std::size_t max_pool_bytes() const noexcept { return get().max_pool_bytes(); }

  /**
   * @brief Returns the bytes the pool currently holds, in use or idle.
   *
   * Synchronize the relevant streams before reading this counter to measure the pool after
   * queued work completes.
   *
   * @return The reserved size of the pool in bytes.
   * @throws cucascade::cuda_error if the driver query fails.
   */
  [[nodiscard]] std::size_t reserved_bytes() const { return get().reserved_bytes(); }

  /**
   * @brief Returns the bytes of the pool currently allocated to callers.
   *
   * Follows the same timing rule as `reserved_bytes()`.
   *
   * @return The allocated size of the pool in bytes.
   * @throws cucascade::cuda_error if the driver query fails.
   */
  [[nodiscard]] std::size_t used_bytes() const { return get().used_bytes(); }
};

static_assert(::cuda::mr::resource_with<pinned_host_pool_memory_resource,
                                        ::cuda::mr::device_accessible,
                                        ::cuda::mr::host_accessible>);

}  // namespace memory
}  // namespace cucascade
