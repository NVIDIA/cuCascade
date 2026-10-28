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

// cuCascade's public headers come first: liburing (through the net headers) pulls in
// <linux/fs.h>, whose BLOCK_SIZE macro breaks moodycamel's queue headers when they follow it.
#include "io/rest/net/config.hpp"
#include "io/rest/net/endpoint.hpp"
#include "io/rest/planner/config.hpp"
#include "io/rest/planner/planner_remote.hpp"

#include <cucascade/io/rest/config.hpp>
#include <cucascade/io/rest/rest_reactor.hpp>
#include <cucascade/io/tls/tls.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cucascade::io::rest::detail {

/// Where a request goes: the authority of an authorized URL.
struct url_parts {
  bool tls{true};
  std::string_view host;       ///< without brackets for an IPv6 literal
  std::string_view authority;  ///< "host[:port]" exactly as in the URL (the Host header)
  std::uint16_t port{443};
  std::string_view target;  ///< path + query, at least "/"
};

/// Split `scheme://authority/target` (http or https only).
/// @throws std::invalid_argument for anything else.
[[nodiscard]] url_parts split_url(std::string_view url);

/**
 * @brief Transport state shared by every reactor of a REST ioctx (gpuio `remote_client` minus
 *        credentials: cuCascade keeps its authorizer seam).
 *
 * Owns the endpoint table and the DNS resolver thread (`net::remote_context`), the TLS provider
 * of the pool's TLS policy (built on the first https endpoint), and one planner
 * `endpoint_control` per endpoint (estimator, controller, retry budget; gpuio C1-C8, X2).
 * Endpoint registration is cold and mutex-protected; reactors read endpoints lock-free.
 * Outlives every reactor and every caller-thread engine of the pool (the reactor context owns it).
 */
class shared_services {
 public:
  explicit shared_services(config cfg);
  ~shared_services();

  shared_services(shared_services const&)            = delete;
  shared_services& operator=(shared_services const&) = delete;

  [[nodiscard]] config const& cfg() const noexcept { return _cfg; }
  [[nodiscard]] net::remote_context& rc() noexcept { return _rc; }

  /// The endpoint id of (@p tls, @p host, @p port), registered on first use.  Any thread.
  /// @param max_connections connection ceiling per pool for this endpoint (0 = the config's);
  ///        a different ceiling is a different endpoint (the batched footer resolve's private
  ///        pool runs `max_inflight` connections, whatever `max_connections` says).
  /// @throws std::system_error when the endpoint cannot be registered (bad TLS policy, table
  ///         full).
  [[nodiscard]] std::uint32_t endpoint(bool tls,
                                       std::string_view host,
                                       std::uint16_t port,
                                       unsigned max_connections = 0);

  /// The planner state of endpoint @p id (created with the endpoint).  Lock-free.
  [[nodiscard]] planner::endpoint_control& control(std::uint32_t id) noexcept
  {
    return *_controls[id].load(std::memory_order_acquire);
  }

  [[nodiscard]] planner::remote_planner_config const& planner_config() const noexcept
  {
    return _planner;
  }
  [[nodiscard]] planner::retry_config const& retry_config() const noexcept { return _retry; }
  [[nodiscard]] planner::remote_scheduler_config const& scheduler_config() const noexcept
  {
    return _sched;
  }
  [[nodiscard]] net::connection_config const& connection_config() const noexcept { return _conn; }

  /// Reactors whose worker runs (part caps, connection shares, warm-up shares).
  [[nodiscard]] unsigned live_reactors() const noexcept
  {
    auto const n = _live.load(std::memory_order_relaxed);
    return n == 0 ? 1U : n;
  }
  void reactor_started() noexcept { _live.fetch_add(1, std::memory_order_relaxed); }
  void reactor_stopped() noexcept { _live.fetch_sub(1, std::memory_order_relaxed); }

  /// Apply the process-wide IMDS link seed (gpuio C4) to every endpoint once it resolved.  Cheap
  /// when nothing changed; called from the reactors' ticks.
  void poll_link_seed() noexcept;

 private:
  [[nodiscard]] std::shared_ptr<tls::provider const> tls_provider();

  config _cfg;
  planner::remote_planner_config _planner;
  planner::retry_config _retry;
  planner::remote_scheduler_config _sched;
  net::connection_config _conn;
  net::remote_context _rc;

  std::mutex _mutex;  ///< endpoint registration, provider construction
  std::shared_ptr<tls::provider const> _tls;
  std::unordered_map<std::string, std::uint32_t> _by_key;
  std::array<std::atomic<planner::endpoint_control*>, net::remote_context::max_endpoints>
    _controls{};
  std::vector<std::unique_ptr<planner::endpoint_control>> _owned_controls;
  std::atomic<unsigned> _live{0};
  std::atomic<bool> _seed_applied{false};
};

}  // namespace cucascade::io::rest::detail
