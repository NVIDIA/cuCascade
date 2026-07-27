/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cucascade/cudf/rest_datasource_engine.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/io/rest/rest_reactor.hpp>
#include <cucascade/io/rest/s3/sigv4_authorizer.hpp>
#include <cucascade/io/rest/s3/static_credentials.hpp>

namespace cucascade::io {

rest_datasource_engine::rest_datasource_engine(std::string access_key_id,
                                               std::string secret_access_key,
                                               std::string session_token,
                                               std::string region,
                                               std::string endpoint,
                                               std::size_t n_reactors,
                                               bool tls_verify,
                                               std::size_t pool_capacity,
                                               std::size_t block_size)
  : _upstream(0, true),
    _host_mr(0, _upstream, pool_capacity, pool_capacity, block_size, 128, 1)
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

  auto rest_ctx = std::make_shared<rest::rest_reactor::reactor_context>(
    std::move(rest_cfg), std::move(authorizer), &_host_mr);

  auto io_ctx = std::make_shared<rest::rest_ioctx>(n_reactors, std::move(rest_ctx));
  io_ctx->start();
  _io_ctx = std::move(io_ctx);
}

rest_datasource_engine::~rest_datasource_engine() { _io_ctx->shutdown(); }

std::unique_ptr<datasource> rest_datasource_engine::open(std::string path) const
{
  return open_datasource(_io_ctx, std::move(path));
}

}  // namespace cucascade::io
