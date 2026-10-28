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

#include "io/rest/worker/services.hpp"

#include "io/rest/cloud/imds.hpp"

#include <cucascade/log/logging.hpp>

#include <algorithm>
#include <charconv>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace cucascade::io::rest::detail {

namespace {

/// The retry policy of cuCascade's knobs on top of gpuio's defaults (gpuio X1/X2/X3).  The
/// throttle base keeps gpuio's 20:1 ratio to the transient base (50 ms -> 1 s by default), so a
/// test that shortens the backoff shortens both classes.
planner::retry_config make_retry_config(config const& cfg)
{
  planner::retry_config r{};
  r.max_attempts      = static_cast<unsigned>(std::max<std::size_t>(cfg.max_retry_attempts, 1));
  r.max_auth_attempts = static_cast<unsigned>(cfg.max_auth_retry_attempts);
  if (cfg.retry_backoff_base.count() > 0) {
    r.transient_base = cfg.retry_backoff_base;
    r.throttle_base  = cfg.retry_backoff_base * 20;
  }
  r.honor_retry_after = cfg.honor_retry_after;
  return r;
}

planner::remote_planner_config make_planner_config(config const& cfg)
{
  planner::remote_planner_config p{};
  p.link_gbps         = cfg.link_gbps;
  p.enable_controller = cfg.enable_controller;
  if (cfg.min_segment_bytes != 0) p.min_segment_bytes = cfg.min_segment_bytes;
  if (cfg.max_segment_bytes != 0) p.max_segment_bytes = cfg.max_segment_bytes;
  p.min_segment_bytes = std::min(p.min_segment_bytes, p.max_segment_bytes);
  return p;
}

net::connection_config make_connection_config(config const& cfg)
{
  net::connection_config c{};
  c.max_connections_per_reactor =
    static_cast<unsigned>(std::max<std::size_t>(cfg.max_connections, 1));
  if (cfg.connect_timeout.count() > 0) c.connect_timeout = cfg.connect_timeout;
  if (cfg.tls_handshake_timeout.count() > 0) c.tls_timeout = cfg.tls_handshake_timeout;
  if (cfg.conn_max_age.count() > 0) {
    c.reuse_idle_max = std::chrono::duration_cast<std::chrono::milliseconds>(cfg.conn_max_age);
  }
  c.iowq_max_unbound = cfg.iowq_max_unbound;
  return c;
}

/// The pool's TLS policy: `cfg.tls` with the libcurl-era overrides applied.
tls::config make_tls_config(config const& cfg)
{
  tls::config t = cfg.tls;
  if (!cfg.ca_bundle_path.empty()) t.ca_file = cfg.ca_bundle_path;
  if (!cfg.tls_verify) t.verify = false;
  return t;
}

}  // namespace

url_parts split_url(std::string_view url)
{
  url_parts out;
  constexpr std::string_view https = "https://";
  constexpr std::string_view http  = "http://";
  std::string_view rest;
  if (url.starts_with(https)) {
    out.tls  = true;
    out.port = 443;
    rest     = url.substr(https.size());
  } else if (url.starts_with(http)) {
    out.tls  = false;
    out.port = 80;
    rest     = url.substr(http.size());
  } else {
    throw std::invalid_argument("rest_reactor: authorizer URL is not http(s): " +
                                std::string(url.substr(0, 64)));
  }
  auto const slash = rest.find_first_of("/?");
  out.authority    = rest.substr(0, slash);
  out.target       = slash == std::string_view::npos ? std::string_view{"/"} : rest.substr(slash);
  if (out.authority.empty()) throw std::invalid_argument("rest_reactor: URL without a host");

  std::string_view port_text;
  if (out.authority.front() == '[') {
    auto const close = out.authority.find(']');
    if (close == std::string_view::npos) {
      throw std::invalid_argument("rest_reactor: malformed IPv6 authority in URL");
    }
    out.host = out.authority.substr(1, close - 1);
    if (close + 1 < out.authority.size()) {
      if (out.authority[close + 1] != ':') {
        throw std::invalid_argument("rest_reactor: malformed authority in URL");
      }
      port_text = out.authority.substr(close + 2);
    }
  } else {
    auto const colon = out.authority.rfind(':');
    out.host         = out.authority.substr(0, colon);
    if (colon != std::string_view::npos) port_text = out.authority.substr(colon + 1);
  }
  if (!port_text.empty()) {
    unsigned port   = 0;
    auto const* end = port_text.data() + port_text.size();
    auto const res  = std::from_chars(port_text.data(), end, port);
    if (res.ec != std::errc{} || res.ptr != end || port == 0 || port > 65535) {
      throw std::invalid_argument("rest_reactor: bad port in URL");
    }
    out.port = static_cast<std::uint16_t>(port);
  }
  if (out.host.empty()) throw std::invalid_argument("rest_reactor: URL without a host");
  return out;
}

shared_services::shared_services(config cfg)
  : _cfg(std::move(cfg)),
    _planner(make_planner_config(_cfg)),
    _retry(make_retry_config(_cfg)),
    _conn(make_connection_config(_cfg))
{
  // C4: the link seed resolves once per process on a helper thread; endpoints pick it up from
  // the reactors' ticks.  Only an automatic link needs it.
  if (_planner.enable_controller && _planner.link_gbps <= 0) {
    std::ignore = cloud::start_link_seed();
  }
}

shared_services::~shared_services() = default;

std::shared_ptr<tls::provider const> shared_services::tls_provider()
{
  // Caller holds _mutex.
  if (_tls == nullptr) { _tls = tls::make_openssl_provider(make_tls_config(_cfg)); }
  return _tls;
}

std::uint32_t shared_services::endpoint(bool tls,
                                        std::string_view host,
                                        std::uint16_t port,
                                        unsigned max_connections)
{
  if (max_connections == _conn.max_connections_per_reactor) max_connections = 0;
  std::string key;
  key.reserve(host.size() + 24);
  key.append(tls ? "https://" : "http://").append(host).push_back(':');
  key.append(std::to_string(port));
  if (max_connections != 0) key.append("#").append(std::to_string(max_connections));

  std::lock_guard lock(_mutex);
  if (auto it = _by_key.find(key); it != _by_key.end()) return it->second;

  net::endpoint_config ep{};
  ep.host = std::string(host);
  ep.port = port;
  ep.tls  = tls;
  if (tls) ep.tls_provider = tls_provider();
  ep.conn = _conn;
  if (max_connections != 0) ep.conn.max_connections_per_reactor = max_connections;
  ep.keep_warm.min_connections = _cfg.keep_warm_connections;
  ep.latency_reserved_fraction = _planner.latency_reserved_fraction;
  auto const id                = _rc.add_endpoint(std::move(ep));
  if (!id) { throw std::system_error(id.error(), "rest_reactor: cannot register endpoint " + key); }
  if (_controls[*id].load(std::memory_order_relaxed) == nullptr) {
    auto control = std::make_unique<planner::endpoint_control>(_planner, _retry);
    if (_seed_applied.load(std::memory_order_relaxed)) {
      if (auto seed = cloud::link_gbps_seed()) control->apply_link_seed(*seed);
    }
    _controls[*id].store(control.get(), std::memory_order_release);
    _owned_controls.push_back(std::move(control));
  }
  _by_key.emplace(std::move(key), *id);
  return *id;
}

void shared_services::poll_link_seed() noexcept
{
  if (_seed_applied.load(std::memory_order_acquire)) return;
  if (!_planner.enable_controller || _planner.link_gbps > 0) {
    _seed_applied.store(true, std::memory_order_release);
    return;
  }
  if (!cloud::link_seed_ready()) return;
  auto const seed = cloud::link_gbps_seed();
  std::lock_guard lock(_mutex);
  if (_seed_applied.load(std::memory_order_relaxed)) return;
  if (seed) {
    for (auto const& control : _owned_controls) {
      control->apply_link_seed(*seed);
    }
    CUCASCADE_LOG_INFO("rest_reactor: controller link seed {} Gbps", *seed);
  }
  _seed_applied.store(true, std::memory_order_release);
}

}  // namespace cucascade::io::rest::detail

namespace cucascade::io::rest {

rest_reactor::reactor_context::reactor_context(
  config cfg,
  std::shared_ptr<request_authorizer> authorizer,
  cucascade::memory::fixed_size_host_memory_resource* host_mr)
  : _config(std::move(cfg)), _authorizer(std::move(authorizer)), _host_mr(host_mr)
{
}

rest_reactor::reactor_context::~reactor_context() = default;

std::size_t max_read_gets(config const& cfg, std::uint64_t bytes) noexcept
{
  // The same planner config the engines plan with (make_planner_config), and the capacity bound
  // of its chunk cutter: no plan of `bytes` has more GETs.
  return planner::remote_chunk_capacity(bytes, detail::make_planner_config(cfg));
}

detail::shared_services& rest_reactor::reactor_context::services() const
{
  std::lock_guard lock(_services_mutex);
  if (_services == nullptr) { _services = std::make_unique<detail::shared_services>(_config); }
  return *_services;
}

}  // namespace cucascade::io::rest
