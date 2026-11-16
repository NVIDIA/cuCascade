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

#include <cucascade/io/cache/fs_cache.hpp>
#include <cucascade/io/config.hpp>
#include <cucascade/io/datasource_factory.hpp>
#include <cucascade/io/scan_cache.hpp>
#include <cucascade/log/logging.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cucascade::io {

namespace {

constexpr std::size_t pool_bytes = 128UL << 20;  // size of one pinned pool

/// The NUMA node of the current CUDA device, or -1 if it cannot be found.
int current_device_numa_node()
{
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    CUCASCADE_LOG_WARN("scan_cache: no current CUDA device; leaving cache placement to the driver");
    return -1;
  }
  auto const node = memory::gpu_numa_node(device);
  if (node < 0) {
    CUCASCADE_LOG_WARN(
      "scan_cache: NUMA node of GPU {} is unknown; leaving cache placement to the driver", device);
  }
  return node;
}

/// Throw unless @p value is a share in [0, 1]; NaN is not.
void check_fraction(double value, char const* name)
{
  if (!(value >= 0.0 && value <= 1.0)) {
    throw std::invalid_argument(std::string{"scan_cache: "} + name + " must be in [0, 1]");
  }
}

}  // namespace

struct scan_cache::state {
  std::unique_ptr<cucascade::memory::memory_reservation_manager> memory;
  std::unique_ptr<io_context_registry> registry;
  std::shared_ptr<ioctx> context;
  std::shared_ptr<column_admission> admission;
  object_cache objects;

  state()                        = default;
  state(state const&)            = delete;
  state& operator=(state const&) = delete;
  ~state()
  {
    if (context) { context->shutdown_cache(); }
  }
};

scan_cache::scan_cache(scan_cache_config const& cfg)
  : _state(std::make_shared<state>()), _min_advise_bytes(cfg.min_advise_bytes)
{
  if (cfg.cache_bytes == 0) {
    throw std::invalid_argument("scan_cache: cache_bytes must be positive");
  }
  if (cfg.block_bytes == 0 || cfg.block_bytes > pool_bytes) {
    throw std::invalid_argument("scan_cache: block_bytes must be positive and at most 128 MiB");
  }
  check_fraction(cfg.eviction_threshold, "eviction_threshold");
  check_fraction(cfg.eviction_batch, "eviction_batch");
  check_fraction(cfg.admission_budget, "admission_budget");

  cucascade::memory::host_memory_space_config host;
  host.numa_id         = cfg.numa_node ? *cfg.numa_node : current_device_numa_node();
  host.memory_capacity = cfg.cache_bytes;
  // The default 0.9 would make only 90% of the cache usable.
  host.reservation_limit_fraction = 1.0;
  host.block_size                 = cfg.block_bytes;
  host.pool_size                  = pool_bytes / cfg.block_bytes;
  // Pin the whole cache now. A pool that grows on demand makes every cached read wait for it.
  host.initial_number_pools = (cfg.cache_bytes + pool_bytes - 1) / pool_bytes;
  host.initial_pool_threads = cfg.pinning_threads;
  std::vector<cucascade::memory::memory_space_config> spaces;
  spaces.emplace_back(host);
  _state->memory =
    std::make_unique<cucascade::memory::memory_reservation_manager>(std::move(spaces));

  io_config config;
  config.backend                           = io_backend::native;
  config.rest_n_reactors                   = cfg.reactors;
  config.rest.max_connections              = cfg.connections;
  config.rest.stall_time_s                 = cfg.stall_seconds;
  config.cache.mode                        = cache::cache_mode::cucs;
  config.cache.eviction                    = cache::eviction_policy::lru;
  config.cache.eviction_threshold_fraction = cfg.eviction_threshold;
  config.cache.eviction_batch_fraction     = cfg.eviction_batch;
  config.object_store                      = cfg.object_store;
  config.apply_cache_mode();

  _state->registry = std::make_unique<io_context_registry>(config, *_state->memory);
  _state->context  = _state->registry->make_ioctx(io_context_type::restful);
  auto& context    = _state->context;
  if (!context) {
    throw std::runtime_error(
      "scan_cache: could not create the REST io context (check endpoint, region and credentials)");
  }
  context->start();
  if (config.cache.use_fs_cache()) {
    context->initialize_cache(*_state->memory, config.cache, nullptr);
  }
  if (context->cache() == nullptr) {
    throw std::runtime_error("scan_cache: could not create the prefetching cache (see the log)");
  }

  _state->admission = std::make_shared<column_admission>(column_admission_config{
    .budget_bytes =
      static_cast<std::size_t>(static_cast<double>(cfg.cache_bytes) * cfg.admission_budget),
    .chunk_bytes = context->cache()->chunk_size()});
}

scan_cache::~scan_cache() = default;

std::unique_ptr<advising_reader> scan_cache::open(std::string const& path)
{
  // The first open of a file resolves its size, version tag and footer with one request. Reopening
  // the kept handle needs none, and keeps the version tag the cache shares chunks by.
  auto object = _state->objects.find_or_open(
    path, [&] { return _state->context->open_io_object(path, open_hint::parquet_footer_probe); });
  // The reader's context shares ownership of the whole cache, so the reader may outlive this
  // object.
  return std::make_unique<advising_reader>(std::shared_ptr<ioctx>(_state, _state->context.get()),
                                           std::move(object),
                                           _state->admission,
                                           _min_advise_bytes);
}

void scan_cache::register_layout(std::string const& path,
                                 std::string const& table,
                                 std::vector<std::string> const& columns,
                                 std::span<std::uint32_t const> column_of_chunk,
                                 std::span<std::uint64_t const> offsets,
                                 std::span<std::uint64_t const> sizes)
{
  _state->admission->register_layout(path, table, columns, column_of_chunk, offsets, sizes);
}

std::vector<column_stats> scan_cache::stats() const { return _state->admission->stats(); }

std::string scan_cache::summary() const { return _state->context->cache()->summary(); }

}  // namespace cucascade::io
