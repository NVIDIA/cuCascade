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

#include "session.hpp"

#include <cucascade/cudf/advising_datasource.hpp>
#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/cache/fs_cache.hpp>
#include <cucascade/io/config.hpp>
#include <cucascade/io/datasource_factory.hpp>
#include <cucascade/io/object_cache.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

#include <mutex>
#include <stdexcept>
#include <utility>

namespace cucascade::python {

namespace {

constexpr std::size_t pool_bytes = 128UL << 20;  // size of one pinned pool

struct session {
  std::unique_ptr<cucascade::memory::memory_reservation_manager> memory;
  std::unique_ptr<cucascade::io::io_context_registry> registry;
  std::shared_ptr<cucascade::io::ioctx> context;
  std::shared_ptr<cucascade::io::column_admission> admission;
  std::shared_ptr<cucascade::io::object_cache> objects;
  std::size_t min_advise_bytes{0};
};

std::mutex session_mutex;
std::unique_ptr<session> current;

std::shared_ptr<cucascade::io::ioctx> context()
{
  std::lock_guard lock(session_mutex);
  if (!current) { throw std::runtime_error("cucascade: no session; call start_session first"); }
  return current->context;
}

}  // namespace

void start_session(session_options const& options)
{
  std::lock_guard lock(session_mutex);
  if (current) { return; }
  if (options.cache_bytes == 0) {
    throw std::invalid_argument("cucascade: cache_bytes must be positive");
  }
  auto next = std::make_unique<session>();

  cucascade::memory::host_memory_space_config host;
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
  next->memory = std::make_unique<cucascade::memory::memory_reservation_manager>(std::move(spaces));

  cucascade::io::io_config config;
  config.backend                           = cucascade::io::io_backend::native;
  config.rest_n_reactors                   = options.reactors;
  config.rest.max_connections              = options.connections;
  config.rest.stall_time_s                 = options.stall_seconds;
  config.cache.mode                        = cucascade::io::cache::cache_mode::cucs;
  config.cache.eviction                    = cucascade::io::cache::eviction_policy::lru;
  config.cache.eviction_threshold_fraction = options.eviction_threshold;
  config.cache.eviction_batch_fraction     = options.eviction_batch;
  config.object_store.endpoint             = options.endpoint;
  config.object_store.region               = options.region;
  config.object_store.access_key           = options.access_key;
  config.object_store.secret_key           = options.secret_key;
  config.object_store.session_token        = options.session_token;
  config.object_store.s3_signing_mode = cucascade::io::object_store_config::signing_mode::header;
  config.apply_cache_mode();

  next->registry = std::make_unique<cucascade::io::io_context_registry>(config, *next->memory);
  next->context  = next->registry->make_ioctx(cucascade::io::io_context_type::restful);
  if (!next->context) {
    throw std::runtime_error(
      "cucascade: could not create the REST io context (check endpoint, region and credentials)");
  }
  next->context->start();
  if (config.cache.use_fs_cache()) {
    next->context->initialize_cache(*next->memory, config.cache, nullptr);
  }

  next->admission =
    std::make_shared<cucascade::io::column_admission>(cucascade::io::column_admission_config{
      .budget_bytes = static_cast<std::size_t>(static_cast<double>(options.cache_bytes) *
                                               options.admission_budget),
      .chunk_bytes  = options.block_bytes});
  next->objects          = std::make_shared<cucascade::io::object_cache>();
  next->min_advise_bytes = options.min_advise_bytes;
  current                = std::move(next);
}

std::unique_ptr<cudf::io::datasource> open(std::string const& path)
{
  auto ctx = context();
  std::shared_ptr<cucascade::io::column_admission> admission;
  std::shared_ptr<cucascade::io::object_cache> objects;
  std::size_t min_advise_bytes;
  {
    std::lock_guard lock(session_mutex);
    admission        = current->admission;
    objects          = current->objects;
    min_advise_bytes = current->min_advise_bytes;
  }

  // The first open of a file resolves its size, version tag and footer with one request. Reopening
  // the kept handle needs none, and keeps the version tag the cache shares chunks by.
  auto object = objects->find(path);
  if (!object) {
    object = objects->insert(
      path, ctx->open_io_object(path, cucascade::io::open_hint::parquet_footer_probe));
  }
  auto base = std::make_unique<cucascade::io::datasource>(ctx, std::move(object));
  return std::make_unique<cucascade::io::advising_datasource>(
    std::move(base), std::move(admission), path, min_advise_bytes);
}

void register_layout(std::string const& path,
                     std::string const& table,
                     std::vector<std::string> const& columns,
                     std::vector<std::uint32_t> const& column_of_chunk,
                     std::vector<std::uint64_t> const& offsets,
                     std::vector<std::uint64_t> const& sizes)
{
  std::shared_ptr<cucascade::io::column_admission> admission;
  {
    std::lock_guard lock(session_mutex);
    if (!current) { throw std::runtime_error("cucascade: no session; call start_session first"); }
    admission = current->admission;
  }
  admission->register_layout(path, table, columns, column_of_chunk, offsets, sizes);
}

std::vector<cucascade::io::column_stats> column_stats()
{
  std::shared_ptr<cucascade::io::column_admission> admission;
  {
    std::lock_guard lock(session_mutex);
    if (!current) { return {}; }
    admission = current->admission;
  }
  return admission->stats();
}

std::string cache_summary()
{
  std::lock_guard lock(session_mutex);
  if (!current || current->context->cache() == nullptr) { return {}; }
  return current->context->cache()->summary();
}

void stop_session()
{
  std::lock_guard lock(session_mutex);
  if (!current) { return; }
  current->context->shutdown_cache();
  current.reset();
}

}  // namespace cucascade::python
