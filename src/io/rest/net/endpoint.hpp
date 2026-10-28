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

#include "io/rest/net/common.hpp"
#include "io/rest/net/config.hpp"
#include "io/rest/net/dns_resolver.hpp"
#include "io/rest/net/streams.hpp"

#include <cucascade/io/tls/tls.hpp>

#include <sys/socket.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

/**
 * @file endpoint.hpp
 * @brief Context-wide remote state shared by every reactor's pool (gpuio `remote_env.hpp` minus
 *        the `reactor_env` binding and the warm-up broadcast): the endpoint table (pool key =
 *        scheme, host, port, TLS provider; gpuio T6 identity rule), per-endpoint connection / TLS
 *        / keep-warm policy, and the DNS resolver thread. Warm-up is a per-pool call the REST
 *        worker makes on its own thread (`connection_pool::warmup`).
 *
 * The Env contract the pool is written against lives with its implementation: `loop_env`
 * (uring_loop.hpp).
 */

namespace cucascade::io::net {

/// Policy of one endpoint (pool key = scheme, host, port, TLS provider; gpuio CHANGE_REQUESTS gpuio
/// T6 identity rule + curl §9d rule 1 "match only on identical TLS config and endpoint").
struct endpoint_config {
  std::string host{};  ///< DNS name or IP literal
  std::uint16_t port{443};
  bool tls{true};  ///< false: http:// (plain_stream)
  /// Required when `tls`: makes the sessions, AEADs and KDFs of the endpoint's TLS policy. Shared
  /// by every reactor; outlives every pool that uses it.
  std::shared_ptr<tls::provider const> tls_provider{};
  connection_config conn{};      ///< sockets, DNS, receive policy
  keep_warm_config keep_warm{};  ///< default off (min_connections = 0)
  /// Connections of one reactor reserved for urgent (latency-class) checkouts. 0 = auto:
  /// max(2, ceil(latency_reserved_fraction x max_connections_per_reactor)), always leaving bulk
  /// at least one (gpuio T14 `latency_reserved_connections`, s3_line_rate §3.2).
  unsigned reserved_connections{0};
  double latency_reserved_fraction{0.05};
  /// Close a connection after this age (graceful: `last_request` asks the driver to send
  /// `Connection: close`). 0 = unlimited (curl: 24 h max lifetime).
  std::chrono::seconds max_connection_age{0};
  /// Concurrent connects (TCP + TLS) per endpoint per reactor (warm-up rate limit, gpuio §4.5).
  unsigned max_concurrent_connects{16};
  /// WAITALL watchdog (gpuio T13 finding, kernel >= 6.14): a kTLS body receive with MSG_WAITALL
  /// still in flight after this long whose socket has bytes queued but no POLLIN (kTLS masks
  /// POLLIN while a TLS 1.3 KeyUpdate is pending, so io_uring's poll-driven WAITALL retry never
  /// runs) is cancelled, keeping the bytes it placed (done_io), and continued with one one-shot
  /// receive that returns -EIO for the queued KeyUpdate. Checked by the pool tick (250 ms); gpuio
  /// T13b: cancelled only after TCP_INFO bytes_received and FIONREAD stayed unchanged for this long
  /// (two ticks) after that, so a slow-but-alive connection is never cancelled. 0 = off.
  std::chrono::milliseconds waitall_watchdog{500};
};

/// A published endpoint (immutable except the keep-warm policy pointer).
struct endpoint_desc {
  std::uint32_t id{0};
  endpoint_config cfg{};
  std::uint32_t dns{dns_resolver::invalid};  ///< resolver handle (names only)
  bool literal{false};                       ///< host is an IP literal: no DNS
  sockaddr_storage literal_addr{};
  socklen_t literal_len{0};
  stream_options stream{};
  unsigned reserved{2};
  std::uint64_t max_connection_bytes{0};  ///< tls::config::max_connection_bytes (0 = unlimited)
  std::atomic<keep_warm_config const*> keep_warm{nullptr};
};

/// @return connections reserved for urgent checkouts (same formula as gpuio T14's
/// `latency_reserved_connections`: max(2, ceil(f x conns)), bulk keeps at least one; 0 if <= 1).
[[nodiscard]] unsigned reserved_connections_for(double fraction, unsigned conns) noexcept;

/// @return true when kTLS body receives go to io-wq (`IOSQE_ASYNC`): placement `iowq`, or
/// `automatic` on EC2 (`rest::cloud::running_on_ec2`, DMI only; gpuio T8: EC2 64 connections
/// inline 1.79 GB/s vs io-wq 3.33 GB/s; the dev box pays +50% CPU/GB for ~2% with io-wq, so inline
/// elsewhere).
[[nodiscard]] bool resolve_iowq_placement(tls::recv_placement placement) noexcept;

/// @brief Context-wide remote state shared by all reactors' pools (one per ioctx). Endpoint
/// registration and keep-warm policy changes are cold, mutex-protected and publish with release
/// stores; reactors read lock-free.
class remote_context {
 public:
  static constexpr std::uint32_t max_endpoints = 256;

  explicit remote_context(dns_resolver::options dns = {}) noexcept;
  ~remote_context();
  remote_context(remote_context const&)            = delete;
  remote_context& operator=(remote_context const&) = delete;

  /// @brief Register an endpoint (idempotent for identical (tls, host, port, provider)). Any
  /// thread. @return the endpoint id, errc::invalid_argument (no host, TLS without a provider),
  /// or std::errc::too_many_files_open when the table is full.
  [[nodiscard]] result<std::uint32_t> add_endpoint(endpoint_config cfg) noexcept;
  /// @return the endpoint, or nullptr for an unknown id. Lock-free.
  [[nodiscard]] endpoint_desc const* endpoint(std::uint32_t id) const noexcept
  {
    if (id >= max_endpoints) return nullptr;
    return _eps[id].load(std::memory_order_acquire);
  }
  [[nodiscard]] std::uint32_t endpoint_count() const noexcept
  {
    return _count.load(std::memory_order_acquire);
  }
  /// @brief Replace the keep-warm policy of @p id (gpuio API.md A.6 `set_keep_warm`). Any thread.
  void set_keep_warm(std::uint32_t id, keep_warm_config cfg) noexcept;
  /// @return the current keep-warm policy (stable reference: old policies are kept alive).
  [[nodiscard]] keep_warm_config const& keep_warm(std::uint32_t id) const noexcept;

  [[nodiscard]] dns_resolver& dns() noexcept { return _dns; }

 private:
  dns_resolver _dns;
  std::array<std::atomic<endpoint_desc const*>, max_endpoints> _eps{};
  std::atomic<std::uint32_t> _count{0};
  std::mutex _m;
  std::vector<std::unique_ptr<endpoint_desc>> _owned;
  std::vector<std::unique_ptr<keep_warm_config>> _keep_warm_versions;
};

}  // namespace cucascade::io::net
