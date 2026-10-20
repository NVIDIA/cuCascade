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

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

/**
 * @file config.hpp
 * @brief Connection and keep-warm policy of the net layer (gpuio `connection_config` and
 *        `keep_warm_config`, gpuio's defaults; evidence: gpuio ARCHITECTURE §4, §11).
 *
 * All structs are aggregates usable with designated initialisers; zero means "auto" where noted.
 * The kTLS receive placement (gpuio `connection_config::tls_recv_placement`) is part of the TLS
 * policy here: `tls::config::placement` (docs/io/rest-uring-design.md section 3).
 */

namespace cucascade::io::net {

/// @brief Connections, DNS, sockets and body receives (gpuio ARCHITECTURE §4, §11).
struct connection_config {
  unsigned max_connections_per_reactor{64};  ///< "add reactors, not sockets"
  std::chrono::milliseconds connect_timeout{3000}, tls_timeout{3000};
  /// 4 s: S3 closes idle keep-alive connections at ~5 s (gpuio dev_box_probes §7).
  std::chrono::milliseconds reuse_idle_max{4000};
  bool watch_idle_fin{true};
  std::chrono::seconds dns_ttl{30};
  unsigned dns_max_addresses{32};
  std::chrono::seconds dns_address_max_age{300};
  /// Response-head buffer per connection; <= 16 KiB (clamped to the fixed slot buffer).
  std::uint32_t header_buffer_bytes{16 * KiB};
  std::uint64_t body_recv_cap{4 * MiB};
  /// Close a connection (at check_in) once it has carried requests for this long; 0 = unlimited.
  std::chrono::seconds max_connection_age{0};
  /// A WAITALL body receive still in flight after this long with bytes queued but no POLLIN
  /// (TLS 1.3 KeyUpdate decrypted mid-receive, kernel >= 6.14) is sampled, and cancelled and
  /// continued with one one-shot receive once TCP_INFO bytes_received and FIONREAD then stay
  /// unchanged for this long again (gpuio ARCHITECTURE §4.2 rule 4, gpuio T13b). Checked from the
  /// pool tick; 0 = off.
  std::chrono::milliseconds waitall_watchdog{500};
  /// MSG_WAITALL on body receives, each capped to min(remaining, body_recv_cap): one CQE per
  /// receive (~240 CQEs/GB at 4 MiB vs ~36k without; gpuio E-KTLS-1).
  bool recv_waitall{true};
  /// Unbound io-wq workers with io-wq placement; 0 = auto = 16 (never "no unbound workers", and
  /// never unlimited: uncapped io-wq halved EC2 throughput, 5.45 vs 11.56 GB/s; gpuio U15/T9).
  unsigned iowq_max_unbound{0};
  /// Only with io-wq placement; empty = the reactor's NUMA-local CPUs minus its own CPU when the
  /// reactor thread is pinned to exactly one CPU.
  std::vector<int> iowq_cpus{};
  /// A cancelled GET keeps receiving when at most this much remains (gpuio N17/X9).
  std::uint64_t cancel_abort_bytes{1 * MiB};
  std::uint64_t error_body_drain_max{64 * KiB};
  bool tcp_nodelay{true};
  unsigned keepalive_idle_s{60}, keepalive_interval_s{10}, keepalive_count{3};
  std::chrono::milliseconds tcp_user_timeout{30000};
  std::vector<std::string> bind_interfaces{};  ///< SO_BINDTODEVICE round-robin (multi-NIC)
};

/// @brief Keep-warm probing of idle endpoints (gpuio ARCHITECTURE §4.5, §11; N11).
struct keep_warm_config {
  unsigned min_connections{0};  ///< 0 = off
  std::chrono::milliseconds interval{2500};
  std::string probe_key{};  ///< HEAD target; empty => HeadBucket
};

}  // namespace cucascade::io::net
