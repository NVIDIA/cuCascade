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

#include "loopback_range_server.hpp"

#include <cucascade/io/rest/rest_ioctx.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace cucascade::test {

inline std::vector<std::uint8_t> rest_test_payload(std::size_t size)
{
  std::vector<std::uint8_t> bytes(size);
  for (std::size_t i = 0; i < size; ++i) {
    bytes[i] = static_cast<std::uint8_t>((i * 131U + 17U) & 0xffU);
  }
  return bytes;
}

inline constexpr std::size_t rest_test_footer_bytes = 512;

inline std::vector<std::uint8_t> rest_test_parquet_bytes()
{
  auto bytes = rest_test_payload(4096);
  std::array<std::uint8_t, 4> const magic{'P', 'A', 'R', '1'};
  std::copy(magic.begin(), magic.end(), bytes.begin());
  std::copy(magic.begin(), magic.end(), bytes.end() - 4);
  for (std::size_t i = 0; i < 4; ++i) {
    bytes[bytes.size() - 8 + i] =
      static_cast<std::uint8_t>((rest_test_footer_bytes >> (8U * i)) & 0xffU);
  }
  return bytes;
}

inline void require_rest_bytes_equal(std::span<std::uint8_t const> actual,
                                     std::span<std::uint8_t const> expected)
{
  REQUIRE(actual.size() == expected.size());
  CHECK(std::equal(actual.begin(), actual.end(), expected.begin(), expected.end()));
}

inline io::rest::config rest_contract_config()
{
  io::rest::config cfg{};
  cfg.request_timeout_s       = 5;
  cfg.max_connections         = 4;
  cfg.max_retry_attempts      = 2;
  cfg.max_auth_retry_attempts = 1;
  cfg.retry_backoff_base      = std::chrono::milliseconds{1};
  cfg.retry_jitter            = std::chrono::milliseconds{0};
  cfg.honor_retry_after       = false;
  cfg.tls_verify              = false;
  return cfg;
}

inline std::shared_ptr<io::rest::rest_reactor::reactor_context> make_rest_contract_context(
  loopback_range_server const& server,
  io::rest::config cfg                          = rest_contract_config(),
  memory::fixed_size_host_memory_resource* host = nullptr)
{
  return std::make_shared<io::rest::rest_reactor::reactor_context>(
    cfg, std::make_shared<list_capable_mock_authorizer>(server.endpoint()), host);
}

inline std::shared_ptr<io::rest::rest_ioctx> make_rest_contract_ioctx(
  loopback_range_server const& server,
  io::rest::config cfg                          = rest_contract_config(),
  std::size_t reactors                          = 1,
  memory::fixed_size_host_memory_resource* host = nullptr)
{
  auto context =
    std::make_shared<io::rest::rest_ioctx>(reactors, make_rest_contract_context(server, cfg, host));
  context->start();
  return context;
}

inline std::unique_ptr<io::rest::rest_reactor> make_rest_contract_reactor(
  loopback_range_server const& server, io::rest::config cfg = rest_contract_config())
{
  return std::make_unique<io::rest::rest_reactor>(make_rest_contract_context(server, cfg),
                                                  "rest-contract-test");
}

template <typename Context>
class rest_shutdown_guard {
 public:
  explicit rest_shutdown_guard(Context& context,
                               std::shared_ptr<range_response_control> control = {})
    : _context(context), _control(std::move(control))
  {
  }

  ~rest_shutdown_guard()
  {
    if (_control) { _control->release_all(); }
    _context.shutdown();
  }

  rest_shutdown_guard(rest_shutdown_guard const&)            = delete;
  rest_shutdown_guard& operator=(rest_shutdown_guard const&) = delete;

 private:
  Context& _context;
  std::shared_ptr<range_response_control> _control;
};

inline std::vector<std::uint8_t> rest_device_bytes(void const* device,
                                                   std::size_t size,
                                                   cudaStream_t stream)
{
  std::vector<std::uint8_t> bytes(size);
  REQUIRE(cudaMemcpyAsync(bytes.data(), device, size, cudaMemcpyDeviceToHost, stream) ==
          cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  return bytes;
}

}  // namespace cucascade::test
