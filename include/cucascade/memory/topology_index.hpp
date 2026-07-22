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

#include <cucascade/error.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cucascade::memory {

/// @brief Fast bidirectional lookup over a discovered hardware topology,
///        scoped to a specific set of GPU device ids.
///
/// cucascade's @c system_topology_info stores GPUs as a flat vector, each
/// carrying its NUMA node — answering "which NUMA node owns this GPU?" or
/// "which GPUs sit on this NUMA node?" means scanning that vector every time.
/// This index builds both maps once at construction so callers (NUMA-aware
/// bounce-buffer placement, per-node reactor pools, ...) can resolve either
/// direction in O(1).
///
/// The index is scoped to the @p device_ids it is built with — typically the
/// GPUs cuCascade actually reserved memory on, not necessarily every GPU the
/// topology discovered.  @c gpu_ids(), @c gpus_of() and @c numa_node_of() only
/// ever report those device ids; topology GPUs outside the set are ignored.
///
/// The index owns a copy of the topology, so it stays valid independently of
/// the @c topology_discovery that produced it.  NUMA node ids are taken
/// verbatim from the topology, including the sentinel @c -1 for "unknown" (also
/// used when a requested device id is absent from the topology).
class topology_index {
 public:
  /// @brief Build the index from explicit device ids.
  /// @param topology    the system topology to resolve NUMA nodes from.
  /// @param device_ids  GPU device ids to scope the index to.  NUMA nodes are
  ///                    taken directly from the topology; ids absent from the
  ///                    topology resolve to -1.
  topology_index(cucascade::memory::system_topology_info topology, std::vector<int> device_ids)
    : _topology(std::move(topology)), _gpu_ids(std::move(device_ids))
  {
    std::unordered_map<int, int> topology_numa;
    for (auto const& gpu : _topology.gpus) {
      topology_numa[static_cast<int>(gpu.id)] = gpu.numa_node;
    }
    for (int const gpu_id : _gpu_ids) {
      auto it              = topology_numa.find(gpu_id);
      int const numa_node  = it == topology_numa.end() ? -1 : it->second;
      _gpu_to_numa[gpu_id] = numa_node;
      _numa_to_gpus[numa_node].push_back(gpu_id);
    }
  }

  /// @brief Build the index by extracting device ids from a reservation manager.
  ///
  /// GPU ids come from GPU-tier memory spaces; host NUMA nodes (HOST-tier) are
  /// cross-checked so a GPU's topology NUMA node is only used when a matching
  /// HOST space exists — otherwise the GPU falls back to NUMA -1.
  ///
  /// @param topology  the system topology to resolve NUMA nodes from.
  /// @param manager   reservation manager whose GPU/HOST spaces define the scope.
  ///
  /// @note The index keeps a non-owning back-pointer to @p manager so it can resolve
  ///       memory spaces and candidates on demand.  The manager MUST outlive this index
  ///       (the manager-dependent accessors @c get_spaces_of and @c get_candidates are
  ///       undefined otherwise); the NUMA accessors do not touch the manager.
  topology_index(cucascade::memory::system_topology_info topology,
                 cucascade::memory::memory_reservation_manager& manager)
    : _topology(std::move(topology)), _manager(&manager)
  {
    auto extract_ids = [](cucascade::memory::Tier tier) {
      return [tier](const cucascade::memory::memory_reservation_manager& manager) {
        auto spaces = manager.get_memory_spaces_for_tier(tier);
        std::vector<int> ids;
        ids.reserve(spaces.size());
        std::transform(spaces.begin(), spaces.end(), std::back_inserter(ids), [](auto* space) {
          return space->get_device_id();
        });
        return ids;
      };
    };

    _gpu_ids                         = extract_ids(cucascade::memory::Tier::GPU)(manager);
    std::vector<int> host_numa_nodes = extract_ids(cucascade::memory::Tier::HOST)(manager);

    // Resolve each device id's NUMA node from the topology once.
    std::unordered_map<int, int> topology_numa;
    for (auto const& gpu : _topology.gpus) {
      topology_numa[static_cast<int>(gpu.id)] = gpu.numa_node;
    }
    auto default_numa = -1;
    for (int const gpu_id : _gpu_ids) {
      auto it = topology_numa.find(gpu_id);
      int const numa_node =
        it == topology_numa.end() ? default_numa
        : std::find(host_numa_nodes.begin(), host_numa_nodes.end(), it->second) !=
            host_numa_nodes.end()
          ? it->second
          : default_numa;
      _gpu_to_numa[gpu_id] = numa_node;
      _numa_to_gpus[numa_node].push_back(gpu_id);
    }

    // Snapshot the manager's memory spaces per tier so get_spaces_of() can hand out both
    // mutable and const views without re-querying the manager on every call.  Mutable
    // pointers are recovered via the manager's non-const lookup to avoid const_cast.
    for (const auto* space : manager.get_all_memory_spaces()) {
      cucascade::memory::Tier const tier = space->get_tier();
      cucascade::memory::memory_space* mutable_space =
        manager.get_memory_space(tier, space->get_device_id());
      _tier_spaces[tier].push_back(mutable_space);
      _const_tier_spaces[tier].push_back(mutable_space);
    }
  }

  /// @brief The topology this index was built from.
  [[nodiscard]] const cucascade::memory::system_topology_info& get_topology() const noexcept
  {
    return _topology;
  }

  /// @brief NUMA node hosting @p gpu.
  /// @param gpu  CUDA device id.
  /// @return the GPU's NUMA node, or @c -1 if the GPU is not in this index's
  ///         device set (the same sentinel the topology uses for an unknown
  ///         node).
  [[nodiscard]] int numa_node_of(int gpu) const
  {
    auto it = _gpu_to_numa.find(gpu);
    return it == _gpu_to_numa.end() ? -1 : it->second;
  }

  /// @brief GPUs attached to @p numa.
  /// @param numa  NUMA node id.
  /// @return a view of this index's device ids on that node (in scope order),
  ///         or an empty span if none map to it.  The span is valid for the
  ///         lifetime of this index.
  [[nodiscard]] std::span<const int> gpus_of(int numa) const
  {
    auto it = _numa_to_gpus.find(numa);
    return it == _numa_to_gpus.end() ? std::span<const int>{} : std::span<const int>{it->second};
  }

  [[nodiscard]] std::span<const int> gpu_ids() const noexcept { return _gpu_ids; }

  /// @brief Mutable memory spaces in @p tier (snapshot taken at build time).
  /// @return a view of the manager's spaces for that tier, or an empty span if none.
  ///         The span is valid for the lifetime of this index.
  [[nodiscard]] std::span<cucascade::memory::memory_space*> get_spaces_of(
    cucascade::memory::Tier tier)
  {
    auto it = _tier_spaces.find(tier);
    if (it == _tier_spaces.end()) { return {}; }
    return it->second;
  }

  /// @brief Const memory spaces in @p tier (snapshot taken at build time).
  /// @return a read-only view of the manager's spaces for that tier, or an empty span.
  [[nodiscard]] std::span<const cucascade::memory::memory_space*> get_spaces_of(
    cucascade::memory::Tier tier) const
  {
    auto it = _const_tier_spaces.find(tier);
    if (it == _const_tier_spaces.end()) { return {}; }
    // span<const memory_space*> binds to a non-const vector of const pointers; the
    // const_cast drops only the container's constness, not the pointees' (same pattern as
    // memory_reservation_manager::get_memory_spaces_for_tier).
    return const_cast<std::vector<const cucascade::memory::memory_space*>&>(it->second);
  }

  /// @brief Candidate memory spaces for a reservation @p strategy, memoized by its hash.
  ///
  /// The first call for a given @c strategy.hash() computes the candidate list via the
  /// strategy and caches it; later calls with an equal hash return the cached span
  /// directly.  Candidate *sets* depend only on the topology (live free-memory is checked
  /// later, when a reservation is actually made), so caching by strategy identity is
  /// sound.
  ///
  /// @return a view of the candidate spaces, valid for the lifetime of this index.
  /// @throws cucascade::logic_error if this index was not built from a reservation
  ///         manager (the device-ids constructor cannot resolve candidates).
  [[nodiscard]] std::span<cucascade::memory::memory_space*> get_candidates(
    const cucascade::memory::reservation_request_strategy& strategy) const
  {
    if (_manager == nullptr) {
      CUCASCADE_FAIL("topology_index::get_candidates requires an index built from a manager");
    }
    std::size_t const key = strategy.hash();

    // Fast path: concurrent readers share the lock on a cache hit.
    {
      std::shared_lock<std::shared_mutex> read_lock(_candidate_mutex);
      auto it = _candidate_cache.find(key);
      if (it != _candidate_cache.end()) { return it->second; }
    }

    // Cache miss: compute outside the lock (the manager's space set is immutable), then
    // insert under an exclusive lock.  A concurrent miss on the same key is harmless:
    // emplace keeps the first inserted value and both callers return the same node.
    std::vector<cucascade::memory::memory_space*> computed = strategy.get_candidates(*_manager);
    std::unique_lock<std::shared_mutex> write_lock(_candidate_mutex);
    // References into an unordered_map node stay valid across later inserts, and cached
    // entries are never mutated after insertion, so the returned span outlives the lock.
    return _candidate_cache.emplace(key, std::move(computed)).first->second;
  }

 private:
  cucascade::memory::system_topology_info _topology;
  std::unordered_map<int, int> _gpu_to_numa;                ///< GPU device id -> NUMA node.
  std::unordered_map<int, std::vector<int>> _numa_to_gpus;  ///< NUMA node -> GPU device ids.
  std::vector<int> _gpu_ids;  ///< Scoped GPU device ids, in caller order (for span stability).

  /// Non-owning back-pointer to the manager the index was built from (nullptr when built
  /// from explicit device ids).  The manager must outlive this index.
  cucascade::memory::memory_reservation_manager* _manager{nullptr};

  /// Per-tier memory-space snapshots (mutable + const views over the same pointers).
  std::unordered_map<cucascade::memory::Tier, std::vector<cucascade::memory::memory_space*>>
    _tier_spaces;
  std::unordered_map<cucascade::memory::Tier, std::vector<const cucascade::memory::memory_space*>>
    _const_tier_spaces;

  /// Candidate lists memoized by reservation-strategy hash.  Mutable so get_candidates can
  /// populate the cache while remaining a const query.  A shared_mutex lets concurrent
  /// cache hits proceed in parallel; only the (rare) insert takes the lock exclusively.
  mutable std::shared_mutex _candidate_mutex;
  mutable std::unordered_map<std::size_t, std::vector<cucascade::memory::memory_space*>>
    _candidate_cache;
};

/// @brief Build a topology index scoped to a reservation manager's memory spaces.
///
/// Convenience factory returning a shared, immutable index.  The index keeps a non-owning
/// back-pointer to @p manager, so @p manager must outlive the returned index.
///
/// @param topology  the system topology to resolve NUMA nodes from.
/// @param manager   reservation manager whose GPU/HOST spaces define the scope.
/// @return a shared const topology index.
[[nodiscard]] inline std::shared_ptr<const topology_index> build(
  cucascade::memory::system_topology_info topology,
  cucascade::memory::memory_reservation_manager& manager)
{
  return std::make_shared<const topology_index>(std::move(topology), manager);
}

}  // namespace cucascade::memory
