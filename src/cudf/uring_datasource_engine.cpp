/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cucascade/cudf/uring_datasource_engine.hpp>
#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>

namespace cucascade::io {

uring_datasource_engine::uring_datasource_engine(std::size_t n_reactors,
                                                 std::size_t pool_capacity,
                                                 std::size_t block_size,
                                                 bool use_odirect,
                                                 int numa_node)
  : _upstream(numa_node, true),
    _host_mr(0, _upstream, pool_capacity, pool_capacity, block_size, 128, 1),
    _reactor_ctx(std::make_shared<uring::uring_reactor::reactor_context>(
      uring::uring_reactor::reactor_config_type{.bounce_size = _host_mr.get_block_size(),
                                               .use_odirect = use_odirect},
      &_host_mr)),
    _io_ctx(std::make_shared<uring::uring_ioctx>(n_reactors, _reactor_ctx))
{
  _io_ctx->start();
}

uring_datasource_engine::~uring_datasource_engine() { _io_ctx->shutdown(); }

std::unique_ptr<datasource> uring_datasource_engine::open(std::string path) const
{
  return open_datasource(_io_ctx, std::move(path));
}

}  // namespace cucascade::io
