/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cucascade/cudf/rest_datasource_engine.hpp>
#include <cucascade/io/cache/config.hpp>
#include <cucascade/io/cache/prefetching_cache.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/io/rest/rest_reactor.hpp>
#include <cucascade/io/rest/s3/sigv4_authorizer.hpp>
#include <cucascade/io/rest/s3/static_credentials.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <cucascade/memory/topology_discovery.hpp>
#include <cucascade/memory/topology_index.hpp>

namespace cucascade::io {

rest_datasource_engine::rest_datasource_engine(std::string access_key_id,
                                               std::string secret_access_key,
                                               std::string session_token,
                                               std::string region,
                                               std::string endpoint,
                                               std::size_t n_reactors,
                                               bool tls_verify,
                                               std::size_t pool_capacity,
                                               std::size_t block_size,
                                               std::size_t max_connections,
                                               std::size_t chunk_size,
                                               std::size_t max_n_chunks,
                                               bool enable_cache)
  : _upstream(0, true), _host_mr(0, _upstream, pool_capacity, pool_capacity, block_size, 128, 1)
{
  rest::s3::static_credentials creds{.access_key_id     = std::move(access_key_id),
                                     .secret_access_key = std::move(secret_access_key),
                                     .session_token     = std::move(session_token),
                                     .expires_at        = std::nullopt};

  auto authorizer = std::make_shared<rest::s3::sigv4_presigned_authorizer>(
    std::move(creds), std::move(region), std::move(endpoint));

  rest::config rest_cfg{};
  rest_cfg.bounce_block_size = _host_mr.get_block_size();
  rest_cfg.tls_verify        = tls_verify;
  rest_cfg.max_connections   = max_connections;
  rest_cfg.chunk_size        = chunk_size;
  rest_cfg.max_n_chunks      = max_n_chunks;

  auto rest_ctx = std::make_shared<rest::rest_reactor::reactor_context>(
    std::move(rest_cfg), std::move(authorizer), &_host_mr);

  auto io_ctx = std::make_shared<rest::rest_ioctx>(n_reactors, std::move(rest_ctx));
  io_ctx->start();
  _io_ctx = io_ctx;

  if (enable_cache) {
    memory::topology_discovery discovery;
    static_cast<void>(discovery.discover());
    auto const& topology = discovery.get_topology();

    auto configs = memory::reservation_manager_configurator{}
                     .set_number_of_gpus(1)
                     .use_numa_id_as_host_id()
                     .set_total_host_capacity(pool_capacity)
                     .build(topology);

    _reservation_manager = std::make_unique<memory::memory_reservation_manager>(std::move(configs));

    io::cache::config cache_cfg{};
    cache_cfg.dispose_after_use = true;

    auto topo_index = std::make_shared<memory::topology_index>(topology, std::vector<int>{0});

    _io_ctx->initialize_cache(*_reservation_manager, cache_cfg, std::move(topo_index));
  }
}

rest_datasource_engine::~rest_datasource_engine()
{
  _io_ctx->pre_destroy();
  _io_ctx->shutdown();
}

std::unique_ptr<datasource> rest_datasource_engine::open(std::string path) const
{
  return open_datasource(_io_ctx, std::move(path));
}

std::string rest_datasource_engine::cache_summary() const
{
  auto* cache = _io_ctx->cache();
  if (cache == nullptr) { return "prefetching cache not initialized"; }
  return cache->summary();
}

}  // namespace cucascade::io
