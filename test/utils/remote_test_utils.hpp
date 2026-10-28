/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

/**
 * @file remote_test_utils.hpp
 * @brief A REMOTE memory space over the loopback S3 server (test/io/rest/loopback_range_server.hpp)
 *        and a context from make_remote_io_context().
 */

#include "io/rest/loopback_range_server.hpp"

#include <cucascade/cudf/remote_io_context.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cucascade {
namespace test {

inline constexpr std::size_t remote_part_bytes = std::size_t{5} << 20;  // S3's minimum part

/// The transport configuration of the loopback tests: 5 MiB parts, quick retries, no controller.
inline io::rest::config remote_rest_config()
{
  io::rest::config cfg{};
  cfg.request_timeout_s       = 10;
  cfg.max_connections         = 8;
  cfg.max_retry_attempts      = 3;
  cfg.max_auth_retry_attempts = 1;
  cfg.retry_backoff_base      = std::chrono::milliseconds{1};
  cfg.honor_retry_after       = false;
  cfg.enable_controller       = false;
  cfg.write_split_bytes       = remote_part_bytes;
  // No tls.ko on CI hosts: user-space TLS for the HTTPS cases.
  cfg.tls.ktls = io::tls::ktls_mode::disable;
  return cfg;
}

/// Options of a context talking to @p server (HTTP, or HTTPS verified against @p ca_path).
inline remote_io_context_options remote_options(loopback_range_server const& server,
                                                std::string ca_path = {})
{
  remote_io_context_options options;
  options.object_store.endpoint       = server.endpoint();
  options.object_store.region         = "us-east-1";
  options.object_store.access_key     = "AKIAIOSFODNN7EXAMPLE";
  options.object_store.secret_key     = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
  options.object_store.ca_bundle_path = std::move(ca_path);
  options.object_store.tls_verify     = true;
  options.rest                        = remote_rest_config();
  options.n_reactors                  = 1;
  options.staging_bytes_per_reactor   = std::size_t{32} << 20;
  return options;
}

/// A loopback object store: written keys are what GET / HEAD see; other keys are 404.
inline std::unique_ptr<loopback_range_server> make_remote_store(s3_write_policy policy    = {},
                                                                range_fault_policy faults = {},
                                                                loopback_tls tls          = {})
{
  policy.missing_key_404 = true;
  std::vector<std::uint8_t> fixture(4096, std::uint8_t{0x5a});
  return std::make_unique<loopback_range_server>(
    std::move(fixture),
    std::move(faults),
    std::vector<listed_object>{},
    std::unordered_map<std::string, key_response_script>{},
    std::move(policy),
    tls);
}

inline constexpr char const* remote_test_prefix = "s3://bkt/spill";

/// A REMOTE memory space under remote_test_prefix served by @p context (may be null).
inline std::shared_ptr<memory::memory_space> make_remote_space(
  std::shared_ptr<io::ioctx> context,
  int remote_id        = 0,
  std::size_t capacity = memory::unbounded_remote_capacity,
  std::string prefix   = remote_test_prefix)
{
  memory::remote_memory_space_config config;
  config.remote_id       = remote_id;
  config.memory_capacity = capacity;
  config.uri_prefix      = std::move(prefix);
  config.io_context      = std::move(context);
  return std::make_shared<memory::memory_space>(config);
}

/// Wait until @p done() holds or @p timeout passed. @return done().
template <class Done>
bool eventually(Done&& done, std::chrono::milliseconds timeout = std::chrono::seconds{10})
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (!done()) {
    if (std::chrono::steady_clock::now() >= deadline) { return done(); }
    std::this_thread::sleep_for(std::chrono::milliseconds{2});
  }
  return true;
}

}  // namespace test
}  // namespace cucascade
