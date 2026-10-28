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

/// @file pool_support.cpp
/// @brief Non-template parts of the remote path (gpuio src/net/pool_support.cpp): socket
/// options, errno mapping, the WAITALL-watchdog probe, the io-wq placement entry point, counter
/// merging (connection_pool.hpp), the context-wide `remote_context` and the kTLS receive
/// placement (endpoint.hpp).

#include "io/rest/cloud/imds.hpp"
#include "io/rest/net/connection_pool.hpp"
#include "io/rest/net/endpoint.hpp"

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>

namespace cucascade::io::net {

// ------------------------------------------------------------------------------------------------
// Sockets
// ------------------------------------------------------------------------------------------------

namespace {

int set_int(int fd, int level, int opt, int v) noexcept
{
  return ::setsockopt(fd, level, opt, &v, sizeof v) == 0 ? 0 : errno;
}

}  // namespace

std::error_code apply_socket_options(int fd,
                                     connection_config const& cfg,
                                     std::string_view bind_iface,
                                     bool& bind_failed) noexcept
{
  bind_failed = false;
  // gpuio ARCHITECTURE §4.4: TCP_NODELAY=1; SO_KEEPALIVE with KEEPIDLE 60 / INTVL 10 / CNT 3
  // (dead-peer and 350 s middlebox timers, s3_line_rate §3.4 item 5); TCP_USER_TIMEOUT bounds
  // dead-peer detection on sends. No SO_RCVBUF (autotuning wins), no SO_BUSY_POLL (§11 "not
  // offered").
  if (cfg.tcp_nodelay) {
    if (int e = set_int(fd, IPPROTO_TCP, TCP_NODELAY, 1)) return {e, std::system_category()};
  }
  if (cfg.keepalive_idle_s != 0) {
    if (int e = set_int(fd, SOL_SOCKET, SO_KEEPALIVE, 1)) return {e, std::system_category()};
    if (int e = set_int(fd, IPPROTO_TCP, TCP_KEEPIDLE, static_cast<int>(cfg.keepalive_idle_s)))
      return {e, std::system_category()};
    if (cfg.keepalive_interval_s != 0) {
      if (int e =
            set_int(fd, IPPROTO_TCP, TCP_KEEPINTVL, static_cast<int>(cfg.keepalive_interval_s)))
        return {e, std::system_category()};
    }
    if (cfg.keepalive_count != 0) {
      if (int e = set_int(fd, IPPROTO_TCP, TCP_KEEPCNT, static_cast<int>(cfg.keepalive_count)))
        return {e, std::system_category()};
    }
  }
  if (cfg.tcp_user_timeout.count() > 0) {
    auto const ms =
      static_cast<unsigned>(std::min<long long>(cfg.tcp_user_timeout.count(), 0x7fffffff));
    if (::setsockopt(fd, IPPROTO_TCP, TCP_USER_TIMEOUT, &ms, sizeof ms) != 0)
      return {errno, std::system_category()};
  }
  if (!bind_iface.empty()) {
    // SO_BINDTODEVICE round-robin over bind_interfaces (multi-NIC). A failure (EPERM without
    // CAP_NET_RAW on old kernels, unknown device) is counted and the socket stays unbound.
    char name[64]{};
    std::size_t const n = std::min(bind_iface.size(), sizeof name - 1);
    std::memcpy(name, bind_iface.data(), n);
    if (::setsockopt(fd, SOL_SOCKET, SO_BINDTODEVICE, name, static_cast<socklen_t>(n + 1)) != 0)
      bind_failed = true;
  }
  return {};
}

std::error_code map_socket_errno(int err) noexcept
{
  switch (err) {
    case 0: return {};
    case ECANCELED: return make_error_code(errc::operation_canceled);
    case EKEYEXPIRED:  // TLS 1.3 KeyUpdate not installed (kernel < 6.14): reconnect
    case EBADMSG:      // kTLS AEAD failure / malformed record
    case EMSGSIZE: return make_error_code(errc::tls_protocol_error);
    default: return {err, std::system_category()};
  }
}

// ------------------------------------------------------------------------------------------------
// WAITALL watchdog probe (gpuio ARCHITECTURE §4.2 rule 4, gpuio T13b)
// ------------------------------------------------------------------------------------------------

namespace {

/// The kernel's `struct tcp_info` (uapi <linux/tcp.h>) up to `tcpi_bytes_received` (Linux 4.1).
/// glibc's <netinet/tcp.h> copy stops at `tcpi_total_retrans`, and <linux/tcp.h> conflicts with
/// it, so the prefix is spelled out here; the kernel copies min(optlen, its size).
struct tcp_info_rx_prefix {
  std::uint8_t head[8];   // state .. fastopen_client_fail
  std::uint32_t u32[24];  // rto .. total_retrans
  std::uint64_t pacing_rate, max_pacing_rate, bytes_acked, bytes_received;
};
static_assert(offsetof(tcp_info_rx_prefix, bytes_received) == 128);  // uapi offset
static_assert(sizeof(tcp_info_rx_prefix) == 136);

}  // namespace

bool sample_rx_stall(int fd, rx_stall_sample& out) noexcept
{
  pollfd p{fd, POLLIN, 0};
  int inq = 0;
  if (::poll(&p, 1, 0) != 0 || ::ioctl(fd, FIONREAD, &inq) != 0 || inq <= 0) return false;
  tcp_info_rx_prefix ti{};
  socklen_t len = sizeof ti;
  if (::getsockopt(fd, IPPROTO_TCP, TCP_INFO, &ti, &len) != 0 || len < sizeof ti) return false;
  out.bytes_received = ti.bytes_received;
  out.inq            = inq;
  return true;
}

// ------------------------------------------------------------------------------------------------
// io-wq placement (gpuio ARCHITECTURE §4.3; the mechanism is in uring_loop.cpp)
// ------------------------------------------------------------------------------------------------

iowq_placement apply_iowq_placement(io_uring& ring, connection_config const& cfg) noexcept
{
  return apply_iowq_placement(
    ring, cfg.iowq_max_unbound, std::span<int const>(cfg.iowq_cpus.data(), cfg.iowq_cpus.size()));
}

// ------------------------------------------------------------------------------------------------
// Counters
// ------------------------------------------------------------------------------------------------

void merge_counters(endpoint_counters& into, endpoint_counters const& from) noexcept
{
  constexpr std::size_t n = sizeof(endpoint_counters) / sizeof(std::uint64_t);
  static_assert(sizeof(endpoint_counters) == n * sizeof(std::uint64_t),
                "endpoint_counters holds std::uint64_t fields only");
  auto a       = std::bit_cast<std::array<std::uint64_t, n>>(into);
  auto const b = std::bit_cast<std::array<std::uint64_t, n>>(from);
  for (std::size_t i = 0; i < n; ++i)
    a[i] += b[i];
  std::uint64_t const tls_version = from.tls_version != 0 ? from.tls_version : into.tls_version;
  std::uint64_t const cipher_id   = from.tls_version != 0 ? from.cipher_id : into.cipher_id;
  std::uint64_t const limit       = std::max(into.server_request_limit, from.server_request_limit);
  into                            = std::bit_cast<endpoint_counters>(a);
  into.tls_version                = tls_version;  // last negotiated, not a sum
  into.cipher_id                  = cipher_id;
  into.server_request_limit       = limit;  // max-merged
}

// ------------------------------------------------------------------------------------------------
// remote_context
// ------------------------------------------------------------------------------------------------

unsigned reserved_connections_for(double fraction, unsigned conns) noexcept
{
  if (conns <= 1) return 0;
  double const f = std::isfinite(fraction) ? std::clamp(fraction, 0.0, 1.0) : 0.05;
  auto r         = static_cast<unsigned>(std::ceil(f * static_cast<double>(conns)));
  r              = std::max(r, 2u);
  return std::min(r, conns - 1);  // bulk keeps at least one connection
}

bool resolve_iowq_placement(tls::recv_placement placement) noexcept
{
  switch (placement) {
    case tls::recv_placement::iowq: return true;
    case tls::recv_placement::inline_: return false;
    case tls::recv_placement::automatic: break;
  }
  // gpuio T8: decided from DMI (the endpoint is interned before IMDS could answer), cached.
  return rest::cloud::running_on_ec2();
}

remote_context::remote_context(dns_resolver::options dns) noexcept : _dns(dns) {}

remote_context::~remote_context() = default;

result<std::uint32_t> remote_context::add_endpoint(endpoint_config cfg) noexcept
{
  if (cfg.host.empty() || (cfg.tls && !cfg.tls_provider) || cfg.port == 0)
    return errc::invalid_argument;
  std::lock_guard g(_m);
  std::uint32_t const n = _count.load(std::memory_order_relaxed);
  for (std::uint32_t i = 0; i < n; ++i) {
    endpoint_desc const* d = _eps[i].load(std::memory_order_relaxed);
    if (d->cfg.tls == cfg.tls && d->cfg.port == cfg.port && d->cfg.host == cfg.host &&
        d->cfg.tls_provider == cfg.tls_provider)
      return i;
  }
  if (n >= max_endpoints) return make_error_code(std::errc::too_many_files_open);
  try {
    auto d            = std::make_unique<endpoint_desc>();
    d->id             = n;
    d->literal        = parse_ip_literal(cfg.host, cfg.port, d->literal_addr, d->literal_len);
    d->stream.waitall = cfg.conn.recv_waitall;
    if (cfg.tls && cfg.tls_provider) {
      tls::config const& policy = cfg.tls_provider->policy();
      d->stream.iowq            = resolve_iowq_placement(policy.placement);
      d->stream.rx_record_layer = policy.fallback_rx == tls::user_space_rx::record_layer;
      d->max_connection_bytes   = policy.max_connection_bytes;
    }
    unsigned const conns = std::max(1u, cfg.conn.max_connections_per_reactor);
    d->reserved          = cfg.reserved_connections != 0
                             ? std::min(cfg.reserved_connections, conns > 1 ? conns - 1 : 0u)
                             : reserved_connections_for(cfg.latency_reserved_fraction, conns);
    auto kw              = std::make_unique<keep_warm_config>(cfg.keep_warm);
    d->keep_warm.store(kw.get(), std::memory_order_relaxed);
    d->cfg = std::move(cfg);
    if (!d->literal) {
      d->dns = _dns.intern(d->cfg.host, d->cfg.port);
      if (d->dns == dns_resolver::invalid) return make_error_code(std::errc::too_many_files_open);
    }
    _keep_warm_versions.push_back(std::move(kw));
    _eps[n].store(d.get(), std::memory_order_release);
    _owned.push_back(std::move(d));
    _count.store(n + 1, std::memory_order_release);
    return n;
  } catch (...) {
    return make_error_code(std::errc::not_enough_memory);
  }
}

void remote_context::set_keep_warm(std::uint32_t id, keep_warm_config cfg) noexcept
{
  endpoint_desc const* d = endpoint(id);
  if (d == nullptr) return;
  std::lock_guard g(_m);
  try {
    auto kw = std::make_unique<keep_warm_config>(std::move(cfg));
    // Old versions stay alive until the context dies (readers hold plain references).
    const_cast<endpoint_desc*>(d)->keep_warm.store(kw.get(), std::memory_order_release);
    _keep_warm_versions.push_back(std::move(kw));
  } catch (...) {  // NOLINT(bugprone-empty-catch): noexcept boundary: allocation failure: the
                   // previous keep-warm config stays in force
  }
}

keep_warm_config const& remote_context::keep_warm(std::uint32_t id) const noexcept
{
  static keep_warm_config const off{};
  endpoint_desc const* d = endpoint(id);
  if (d == nullptr) return off;
  keep_warm_config const* k = d->keep_warm.load(std::memory_order_acquire);
  return k != nullptr ? *k : off;
}

}  // namespace cucascade::io::net
