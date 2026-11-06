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

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cucascade::memory {
class fixed_size_host_memory_resource;
class memory_reservation_manager;
class topology_index;
}  // namespace cucascade::memory

namespace cucascade::io {

/**
 * @brief Where one reactor of a pool lives: its home GPU, that GPU's NUMA node,
 *        the pinned staging it bounces device data through, and the CPUs its
 *        worker runs on.
 *
 * A default-constructed placement is "unplaced": no home GPU, the staging of
 * the pool's shared reactor context, and an unpinned worker.  A pool built
 * without a topology index is all unplaced reactors, which is the behaviour
 * before placements existed.  An unplaced reactor in a placed pool is remote to
 * every GPU: it serves host-only requests and device requests that spill.
 *
 * Used by the uring and cuFile pools.  REST reactors are not placed per GPU:
 * see @c rest::plan_rest_pool in src/io/rest/reactor_placement.hpp.
 */
struct reactor_placement {
  /// Home GPU (CUDA device id), or -1 for none.
  int gpu_id{-1};
  /// NUMA node of the home GPU, or -1 when unknown.  Device requests prefer
  /// reactors on the NUMA node of their GPU (see @ref numa_spill_policy).
  int numa_node{-1};
  /// Pinned staging of this reactor; null uses the pool's reactor-context
  /// resource.  Must outlive the reactor.
  cucascade::memory::fixed_size_host_memory_resource* staging_mr{nullptr};
  /// CPUs the worker thread is pinned to, narrowed to the CPUs it inherited
  /// (see @ref pin_reactor_thread); empty leaves it unpinned.
  std::vector<int> cpu_ids;
};

/**
 * @brief When a device request leaves the reactors of its GPU's NUMA node.
 *
 * Any reactor can serve any request, so locality is a preference: a request
 * whose GPU has local reactors goes to them unless the least backlogged local
 * reactor holds more than @ref min_backlog_bytes queued bytes and more than
 * @ref ratio times the least backlog among the other reactors.  Then it is
 * dispatched over the whole pool, least backlogged first.  @ref ratio must be
 * finite and at least 1 (see @ref validate_spill_policy).
 */
struct numa_spill_policy {
  /// Local backlog (bytes) below which a request never leaves its NUMA node.
  std::size_t min_backlog_bytes{64UL << 20};
  /// How many times busier than the least busy remote reactor the least busy
  /// local reactor must be before a request spills.
  double ratio{2.0};
};

/**
 * @brief Check @p policy: @c ratio must be finite and at least 1 (below 1 a
 *        local node would spill while still less busy than the rest of the pool).
 * @throws std::invalid_argument otherwise.
 */
inline void validate_spill_policy(numa_spill_policy const& policy)
{
  if (!std::isfinite(policy.ratio) || policy.ratio < 1.0) {
    throw std::invalid_argument("numa_spill_policy: ratio must be finite and at least 1, got " +
                                std::to_string(policy.ratio));
  }
}

/**
 * @brief Reactor placements of a pool, one per reactor, and its spill policy.
 */
struct reactor_layout {
  std::vector<reactor_placement> placements;
  numa_spill_policy spill{};

  /// @p n_reactors unplaced reactors (see @ref reactor_placement).
  [[nodiscard]] static reactor_layout unplaced(std::size_t n_reactors)
  {
    return reactor_layout{std::vector<reactor_placement>(n_reactors), numa_spill_policy{}};
  }
};

/**
 * @brief Place @p reactors_per_gpu reactors on every GPU of @p topology.
 *
 * Each GPU of the index gets @p reactors_per_gpu placements (in the index's GPU
 * order) carrying the GPU's NUMA node, the HOST memory space of that node as
 * staging, and, when @p pin_threads is set, the GPU's local CPUs from the
 * topology.  A GPU keeps the pool's shared staging (null @c staging_mr) when its
 * NUMA node is unknown, has no HOST space, or that space's block size differs
 * from the first HOST space's (every reactor of a pool must stage in blocks of
 * the size its prefetching cache chunks have).
 *
 * @param reactors_per_gpu     reactors per GPU.
 * @param topology            GPU<->NUMA index; null (or no GPUs) yields
 *                            @p reactors_per_gpu unplaced reactors.
 * @param reservation_manager source of the per-NUMA HOST staging; null leaves
 *                            every reactor on the pool's shared staging.
 * @param pin_threads         whether to pin workers to their GPU's local CPUs.
 */
[[nodiscard]] std::vector<reactor_placement> make_reactor_placements(
  std::size_t reactors_per_gpu,
  cucascade::memory::topology_index const* topology,
  cucascade::memory::memory_reservation_manager* reservation_manager,
  bool pin_threads);

/**
 * @brief Pin the calling thread to @p placement's CPUs; a no-op when it has none.
 *
 * Reactor workers call this first, so the ring and every per-worker allocation
 * are made on the home NUMA node.  The thread is pinned to the placement's CPUs
 * that its current affinity -- inherited from the thread that spawned it, so
 * from @c taskset / @c numactl -C / @c sched_setaffinity on the process -- still
 * allows: pinning narrows a thread, never widens it past what the operator
 * granted.  When none of the placement's CPUs is allowed, or the affinity
 * cannot be read or set, the thread is left as it was and the event is logged
 * under @p name (a warning the first time, debug after).
 */
void pin_reactor_thread(reactor_placement const& placement, std::string_view name) noexcept;

}  // namespace cucascade::io
