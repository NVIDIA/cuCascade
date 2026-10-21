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

#include <cucascade/exec/thread_util.hpp>
#include <cucascade/io/reactor_placement.hpp>
#include <cucascade/log/logging.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/topology_index.hpp>

#include <cstddef>
#include <string_view>
#include <vector>

namespace cucascade::io {
namespace {

using cucascade::memory::Tier;

/// The pinned staging of @p numa_node's HOST space, or null when there is no
/// such space or its block size is not @p block_size.
cucascade::memory::fixed_size_host_memory_resource* numa_staging(
  cucascade::memory::memory_reservation_manager& rm, int numa_node, std::size_t block_size)
{
  if (numa_node < 0) { return nullptr; }
  auto* space = rm.get_memory_space(Tier::HOST, numa_node);
  if (space == nullptr) { return nullptr; }
  auto* resource = space->get_memory_resource_of<Tier::HOST>();
  if (resource == nullptr) { return nullptr; }
  if (resource->get_block_size() != block_size) {
    CUCASCADE_LOG_WARN(
      "make_reactor_placements: HOST space of NUMA node {} stages in {}-byte blocks, not the "
      "pool's {}; its reactors use the shared staging",
      numa_node,
      resource->get_block_size(),
      block_size);
    return nullptr;
  }
  return resource;
}

/// Block size of the first HOST space's staging (the pool's shared staging), or
/// 0 when there is none.
std::size_t shared_staging_block_size(cucascade::memory::memory_reservation_manager& rm)
{
  auto host_spaces = rm.get_memory_spaces_for_tier(Tier::HOST);
  if (host_spaces.empty()) { return 0; }
  auto* resource = host_spaces.front()->get_memory_resource_of<Tier::HOST>();
  return resource == nullptr ? 0 : resource->get_block_size();
}

}  // namespace

std::vector<reactor_placement> make_reactor_placements(
  std::size_t reactors_per_gpu,
  cucascade::memory::topology_index const* topology,
  cucascade::memory::memory_reservation_manager* reservation_manager,
  bool pin_threads)
{
  std::vector<reactor_placement> placements;
  if (topology == nullptr || topology->gpu_ids().empty()) {
    placements.resize(reactors_per_gpu);
    return placements;
  }

  auto const block_size =
    reservation_manager == nullptr ? 0 : shared_staging_block_size(*reservation_manager);
  auto const& gpus = topology->get_topology().gpus;
  placements.reserve(reactors_per_gpu * topology->gpu_ids().size());
  for (int const gpu : topology->gpu_ids()) {
    reactor_placement placement;
    placement.gpu_id    = gpu;
    placement.numa_node = topology->numa_node_of(gpu);
    if (block_size != 0) {
      placement.staging_mr = numa_staging(*reservation_manager, placement.numa_node, block_size);
    }
    if (pin_threads) {
      for (auto const& info : gpus) {
        if (static_cast<int>(info.id) == gpu) {
          placement.cpu_ids = info.cpu_cores;
          break;
        }
      }
    }
    placements.insert(placements.end(), reactors_per_gpu, placement);
  }
  return placements;
}

void pin_reactor_thread(reactor_placement const& placement, std::string_view name) noexcept
{
  if (placement.cpu_ids.empty()) { return; }
  if (!cucascade::exec::thread_util::set_current_thread_affinity(placement.cpu_ids)) {
    CUCASCADE_LOG_WARN("{}: could not pin the worker to the {} CPUs of GPU {} (NUMA node {})",
                       name,
                       placement.cpu_ids.size(),
                       placement.gpu_id,
                       placement.numa_node);
  }
}

}  // namespace cucascade::io
