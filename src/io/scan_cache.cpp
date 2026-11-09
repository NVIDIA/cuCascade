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
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>
#include <string>
#include <utility>

namespace cucascade::io {

namespace {

constexpr std::size_t pool_bytes = 128UL << 20;  // size of one pinned pool

/// "bus:device.function" of a PCI bus id, which NVML and CUDA write with different domain widths.
std::string pci_location(std::string const& bus_id)
{
  auto const colon     = bus_id.find(':');
  std::string location = colon == std::string::npos ? bus_id : bus_id.substr(colon + 1);
  std::ranges::transform(
    location, location.begin(), [](unsigned char c) { return std::tolower(c); });
  return location;
}

/// The NUMA node of the current CUDA device, or -1 if it cannot be found.
int current_device_numa_node()
{
  int device = 0;
  std::array<char, 32> bus_id{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetPCIBusId(bus_id.data(), static_cast<int>(bus_id.size()), device) !=
        cudaSuccess) {
    return -1;
  }
  cucascade::memory::topology_discovery topology;
  if (!topology.discover()) { return -1; }
  auto const wanted = pci_location(bus_id.data());
  for (auto const& gpu : topology.get_topology().gpus) {
    if (pci_location(gpu.pci_bus_id) == wanted) { return gpu.numa_node; }
  }
  return -1;
}

}  // namespace

scan_cache::scan_cache(scan_cache_options const& options)
  : _min_advise_bytes(options.min_advise_bytes)
{
  if (options.cache_bytes == 0) {
    throw std::invalid_argument("scan_cache: cache_bytes must be positive");
  }

  cucascade::memory::host_memory_space_config host;
  host.numa_id         = options.numa_node ? *options.numa_node : current_device_numa_node();
  host.memory_capacity = options.cache_bytes;
  // The default 0.9 would make only 90% of the cache usable.
  host.reservation_limit_fraction = 1.0;
  host.block_size                 = options.block_bytes;
  host.pool_size                  = pool_bytes / options.block_bytes;
  // Pin the whole cache now. A pool that grows on demand makes every cached read wait for it.
  host.initial_number_pools = (options.cache_bytes + pool_bytes - 1) / pool_bytes;
  host.initial_pool_threads = options.pinning_threads;
  std::vector<cucascade::memory::memory_space_config> spaces;
  spaces.emplace_back(host);
  _memory = std::make_unique<cucascade::memory::memory_reservation_manager>(std::move(spaces));

  io_config config;
  config.backend                           = io_backend::native;
  config.rest_n_reactors                   = options.reactors;
  config.rest.max_connections              = options.connections;
  config.rest.stall_time_s                 = options.stall_seconds;
  config.cache.mode                        = cache::cache_mode::cucs;
  config.cache.eviction                    = cache::eviction_policy::lru;
  config.cache.eviction_threshold_fraction = options.eviction_threshold;
  config.cache.eviction_batch_fraction     = options.eviction_batch;
  config.object_store.endpoint             = options.endpoint;
  config.object_store.region               = options.region;
  config.object_store.access_key           = options.access_key;
  config.object_store.secret_key           = options.secret_key;
  config.object_store.session_token        = options.session_token;
  config.object_store.s3_signing_mode      = object_store_config::signing_mode::header;
  config.apply_cache_mode();

  _registry = std::make_unique<io_context_registry>(config, *_memory);
  _context  = _registry->make_ioctx(io_context_type::restful);
  if (!_context) {
    throw std::runtime_error(
      "scan_cache: could not create the REST io context (check endpoint, region and credentials)");
  }
  _context->start();
  if (config.cache.use_fs_cache()) { _context->initialize_cache(*_memory, config.cache, nullptr); }

  _admission = std::make_shared<column_admission>(column_admission_config{
    .budget_bytes =
      static_cast<std::size_t>(static_cast<double>(options.cache_bytes) * options.admission_budget),
    .chunk_bytes = options.block_bytes});
  _objects   = std::make_shared<object_cache>();
}

scan_cache::~scan_cache() { _context->shutdown_cache(); }

std::unique_ptr<advising_reader> scan_cache::open(std::string const& path)
{
  // The first open of a file resolves its size, version tag and footer with one request. Reopening
  // the kept handle needs none, and keeps the version tag the cache shares chunks by.
  auto object = _objects->find(path);
  if (!object) {
    object =
      _objects->insert(path, _context->open_io_object(path, open_hint::parquet_footer_probe));
  }
  return std::make_unique<advising_reader>(
    _context, std::move(object), _admission, path, _min_advise_bytes);
}

void scan_cache::register_layout(std::string const& path,
                                 std::string const& table,
                                 std::vector<std::string> const& columns,
                                 std::span<std::uint32_t const> column_of_chunk,
                                 std::span<std::uint64_t const> offsets,
                                 std::span<std::uint64_t const> sizes)
{
  _admission->register_layout(path, table, columns, column_of_chunk, offsets, sizes);
}

std::vector<column_stats> scan_cache::stats() const { return _admission->stats(); }

std::string scan_cache::summary() const
{
  auto const* cache = _context->cache();
  return cache == nullptr ? std::string{} : cache->summary();
}

}  // namespace cucascade::io
