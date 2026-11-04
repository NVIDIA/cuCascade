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

#include "io/cache/controlled_reactor.hpp"
#include "utils/test_memory_resources.hpp"

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/cache/fs_cache.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <cucascade/memory/topology_discovery.hpp>
#include <cucascade/memory/topology_index.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cucascade::test {

inline std::unique_ptr<memory::memory_reservation_manager> identity_memory(
  double host_reservation_fraction = 1.0, std::size_t host_capacity = 256ULL << 20)
{
  memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(2ULL << 30)
    .set_reservation_fraction_per_gpu(0.75)
    .set_gpu_memory_resource_factory(make_shared_current_device_resource)
    .set_per_numa_region_capacity(host_capacity)
    .use_gpu_id_as_host_id()
    .set_reservation_fraction_per_numa_region(host_reservation_fraction);
  return std::make_unique<memory::memory_reservation_manager>(builder.build());
}

inline io::cache::config identity_cache_config()
{
  io::cache::config cfg;
  cfg.mode                            = io::cache::cache_mode::cucs;
  cfg.eviction                        = io::cache::eviction_policy::lru;
  cfg.min_prefetching_budget_fraction = 0.5;
  cfg.eviction_threshold_fraction     = 1.0;
  cfg.apply_mode();
  return cfg;
}

/// Hint [0, size) of @p ds and take staging buffers for it, as a scan does.
inline void advise(io::datasource& ds, std::size_t size)
{
  std::vector<cudf::io::text::byte_range_info> ranges;
  ranges.emplace_back(0, static_cast<std::int64_t>(size));
  ds.fadvise(ranges, 0);
  REQUIRE(ds.prepare_prefetch(false) == io::prepare_result::prepared);
}

inline void advise_range(io::datasource& source, std::size_t offset, std::size_t size)
{
  std::array<cudf::io::text::byte_range_info, 1> ranges{cudf::io::text::byte_range_info{
    static_cast<std::int64_t>(offset), static_cast<std::int64_t>(size)}};
  source.fadvise(ranges, 0);
}

template <typename Predicate>
inline bool wait_for_cache(Predicate predicate, std::chrono::milliseconds budget)
{
  auto const deadline = std::chrono::steady_clock::now() + budget;
  do {
    if (predicate()) { return true; }
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
  } while (std::chrono::steady_clock::now() < deadline);
  return predicate();
}

inline io::cache::config allocation_cache_config(
  io::cache::eviction_policy policy = io::cache::eviction_policy::lru, bool constrained = false)
{
  io::cache::config cfg;
  cfg.mode     = io::cache::cache_mode::cucs;
  cfg.eviction = policy;
  if (constrained) {
    cfg.min_prefetching_budget_fraction = 0.25;
    cfg.eviction_threshold_fraction     = 1.0;
  }
  cfg.apply_mode();
  return cfg;
}

struct cache_test_fixture {
  explicit cache_test_fixture(io::cache::config config                 = identity_cache_config(),
                              controlled_reactor_config backend_config = {},
                              double host_reservation_fraction         = 1.0,
                              std::size_t host_capacity                = 256ULL << 20)
    : memory(identity_memory(host_reservation_fraction, host_capacity)),
      context(std::make_shared<controlled_ioctx>(backend_config))
  {
    context->start();
    context->initialize_cache(*memory, config, nullptr);
  }

  ~cache_test_fixture()
  {
    context->shutdown();
    context->shutdown_cache();
  }

  cache_test_fixture(cache_test_fixture const&)            = delete;
  cache_test_fixture& operator=(cache_test_fixture const&) = delete;

  std::unique_ptr<io::datasource> open(std::string const& path,
                                       std::size_t bytes,
                                       std::string const& tag = "\"one\"")
  {
    return std::make_unique<io::datasource>(context, held_object(path, tag, bytes));
  }

  std::unique_ptr<cucascade::memory::memory_reservation_manager> memory;
  std::shared_ptr<controlled_ioctx> context;
};

}  // namespace cucascade::test

namespace cucascade::io::cache {

/// Reaches the private request entry points a datasource uses through
/// fadvise / prepare_prefetch, so a test can hold the handles and inspect
/// their chunks.
struct fs_cache_test_access {
  /// @p gpu_id selects the request's preferred NUMA node through the cache's topology, as
  /// @c datasource::fadvise does for the scan's device.
  static cache_handle request(fs_cache& cache,
                              io_object const& obj,
                              std::span<const fs_cache::byte_range> ranges,
                              std::optional<int> gpu_id = std::nullopt)
  {
    return cache.initiate_prefetching_request(obj, ranges, gpu_id);
  }

  static prepare_result prepare(fs_cache& cache,
                                cache_handle& handle,
                                bool wait_for_eviction = false)
  {
    return cache.prepare(handle, wait_for_eviction);
  }
};

}  // namespace cucascade::io::cache

namespace cucascade::test {

/// Two logical HOST arenas behind one armed cache, for per-arena reservation accounting and
/// NUMA-scoped reclaim through the real fs_cache and buffer_pool.
///
/// HOST spaces @ref node_a and @ref node_b hold @ref arena_chunks chunks each over a malloc
/// upstream (no NUMA binding, so a one-node host can run it), and the cache reserves
/// @ref reserved_chunks of each for prefetching.  The topology maps GPU g to NUMA node g, so a
/// request made for node g prefers arena g, as a scan on GPU g does.  The pool takes a whole
/// request from one arena and falls back to the other only when the preferred one cannot supply
/// it; @ref exhaust forces that.  @ref default_config evicts on explicit demand only, so the
/// background evictor does not move the counts a test asserts.
///
/// Chunk i of @ref object is bytes [i * chunk, (i + 1) * chunk); the reclaim sweep visits a
/// generation's chunks in that order.  Handles from @ref hold and allocations from @ref exhaust
/// stay alive until the fixture is destroyed, which drops them before shutting the cache down.
struct two_arena_cache_fixture {
  static constexpr int node_a                  = 0;
  static constexpr int node_b                  = 1;
  static constexpr std::size_t chunk           = controlled_chunk_bytes;
  static constexpr std::size_t arena_chunks    = 16;
  static constexpr std::size_t reserved_chunks = 2;
  static constexpr std::size_t object_chunks   = 32;

  /// LRU with the threshold at the whole pool, and a prefetch reservation of
  /// @ref reserved_chunks per arena.
  static io::cache::config default_config()
  {
    auto cfg = allocation_cache_config();
    cfg.min_prefetching_budget_fraction =
      static_cast<double>(reserved_chunks) / static_cast<double>(arena_chunks);
    cfg.eviction_threshold_fraction = 1.0;
    return cfg;
  }

  explicit two_arena_cache_fixture(io::cache::config config = default_config())
    : memory(two_arena_memory()), context(std::make_shared<controlled_ioctx>())
  {
    context->start();
    context->initialize_cache(*memory, config, one_gpu_per_node());
    REQUIRE(context->cache() != nullptr);
    REQUIRE(context->cache()->is_armed());
    REQUIRE(context->cache()->chunk_size() == chunk);
    source = std::make_unique<io::datasource>(
      context, held_object("controlled://two-arena", "\"one\"", object_chunks * chunk));
  }

  ~two_arena_cache_fixture()
  {
    held.clear();
    pressure.clear();
    source.reset();
    context->shutdown();
    context->shutdown_cache();
  }

  two_arena_cache_fixture(two_arena_cache_fixture const&)            = delete;
  two_arena_cache_fixture& operator=(two_arena_cache_fixture const&) = delete;

  [[nodiscard]] io::cache::fs_cache& cache() const { return *context->cache(); }

  [[nodiscard]] io::io_object const& object() const { return source->get_io_object(); }

  [[nodiscard]] cucascade::memory::memory_space& space(int node) const
  {
    auto* host = memory->get_memory_space(cucascade::memory::Tier::HOST, node);
    REQUIRE(host != nullptr);
    return *host;
  }

  [[nodiscard]] cucascade::memory::fixed_size_host_memory_resource& arena(int node) const
  {
    auto* mr = space(node).get_memory_resource_of<cucascade::memory::Tier::HOST>();
    REQUIRE(mr != nullptr);
    return *mr;
  }

  /// HOST budget arena @p node can still hand out: what a reclaim gives back shows up here.
  [[nodiscard]] std::size_t available(int node) const { return space(node).get_available_memory(); }

  /// A request for chunks [first, first + count) of @ref object made for GPU @p node, so its
  /// staging buffers prefer arena @p node.  Not prepared yet.
  [[nodiscard]] io::cache::cache_handle request(int node, std::size_t first, std::size_t count)
  {
    std::array<io::byte_range, 1> const ranges{io::byte_range{
      static_cast<std::int64_t>(first * chunk), static_cast<std::int64_t>(count * chunk)}};
    auto handle = io::cache::fs_cache_test_access::request(cache(), object(), ranges, node);
    REQUIRE(static_cast<bool>(handle));
    return handle;
  }

  io::cache::prepare_result prepare(io::cache::cache_handle& handle, bool wait_for_eviction = false)
  {
    return io::cache::fs_cache_test_access::prepare(cache(), handle, wait_for_eviction);
  }

  /// @ref request, prepared, and kept subscribed until the fixture goes away.
  io::cache::cache_handle& hold(int node, std::size_t first, std::size_t count)
  {
    auto& handle = held.emplace_back(request(node, first, count));
    REQUIRE(prepare(handle) == io::cache::prepare_result::prepared);
    return handle;
  }

  /// Take every byte arena @p node has left outside the cache, so the cache can only allocate
  /// there what the arena's prefetch reservation still covers.
  void exhaust(int node)
  {
    auto& mr = arena(node);
    if (auto const bytes = mr.get_available_memory(); bytes > 0) {
      pressure.push_back(mr.allocate_multiple_blocks(bytes));
    }
    REQUIRE(available(node) == 0);
  }

  /// NUMA node of the arena that supplied @p c's buffer.
  [[nodiscard]] static int node_of(io::cache::cached_chunk const& c)
  {
    return static_cast<int>(c.numa_node);
  }

  std::unique_ptr<cucascade::memory::memory_reservation_manager> memory;
  std::shared_ptr<controlled_ioctx> context;
  std::unique_ptr<io::datasource> source;  ///< owns @ref object
  std::deque<io::cache::cache_handle> held;
  std::vector<cucascade::memory::fixed_size_host_memory_resource::fixed_multiple_blocks_allocation>
    pressure;

 private:
  static constexpr std::size_t blocks_per_pool = 4;

  static std::unique_ptr<cucascade::memory::memory_reservation_manager> two_arena_memory()
  {
    std::vector<cucascade::memory::memory_space_config> configs;
    for (int const node : {node_a, node_b}) {
      configs.emplace_back(
        malloc_host_space_config(node, arena_chunks * chunk, chunk, blocks_per_pool));
    }
    return std::make_unique<cucascade::memory::memory_reservation_manager>(std::move(configs));
  }

  static std::shared_ptr<const cucascade::memory::topology_index> one_gpu_per_node()
  {
    cucascade::memory::system_topology_info topology{};
    for (int const node : {node_a, node_b}) {
      cucascade::memory::gpu_topology_info gpu;
      gpu.id        = static_cast<unsigned int>(node);
      gpu.numa_node = node;
      topology.gpus.push_back(std::move(gpu));
    }
    return std::make_shared<const cucascade::memory::topology_index>(
      std::move(topology), std::vector<int>{node_a, node_b});
  }
};

}  // namespace cucascade::test
