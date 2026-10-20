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

/**
 * @file openssl_session_cache.cpp
 * @brief Opt-in client session cache per (host, port) (T12; gpuio tls.md §2.11 "Session
 * resumption", curl.md §9a): store SSL_SESSION references, TLS 1.3 single use, TLS 1.2 reuse,
 * capacity per endpoint >= the reconnect burst (curl's 2 is too small for S3 fan-out).  Ported
 * from gpuio `src/tls/session_cache.cpp`.
 */

#include "io/tls/openssl_internal.hpp"

#include <algorithm>
#include <ctime>

namespace cucascade::io::tls::openssl {

namespace {

bool usable(SSL_SESSION const* s, std::time_t now) noexcept
{
  if (SSL_SESSION_is_resumable(s) != 1) { return false; }
  auto const issued  = SSL_SESSION_get_time_ex(s);
  auto const timeout = SSL_SESSION_get_timeout(s);
  return timeout <= 0 || issued + static_cast<std::time_t>(timeout) > now;
}

bool is_tls13(SSL_SESSION const* s) noexcept
{
  return SSL_SESSION_get_protocol_version(s) == TLS1_3_VERSION;
}

}  // namespace

session_cache::session_cache(unsigned per_endpoint)
  : _cap(std::max(1u, per_endpoint)),
    _endpoints(max_endpoints),
    _slots(static_cast<std::size_t>(max_endpoints) * std::max(1u, per_endpoint), nullptr)
{
}

session_cache::~session_cache()
{
  for (SSL_SESSION* s : _slots) {
    if (s != nullptr) { SSL_SESSION_free(s); }
  }
}

void session_cache::clear_locked(unsigned b) noexcept
{
  endpoint& e = _endpoints[b];
  for (unsigned i = 0; i < _cap; ++i) {
    if (SSL_SESSION*& s = slot(b, i); s != nullptr) {
      SSL_SESSION_free(s);
      s = nullptr;
    }
  }
  e.head  = 0;
  e.count = 0;
}

int session_cache::bucket(std::string_view host, std::uint16_t port) noexcept
{
  if (host.empty() || host.size() >= endpoint{}.host.size()) { return -1; }
  std::lock_guard lock(_mu);
  ++_clock;
  unsigned free_idx = max_endpoints;
  unsigned lru_idx  = 0;
  for (unsigned i = 0; i < max_endpoints; ++i) {
    endpoint& e = _endpoints[i];
    if (!e.used) {
      if (free_idx == max_endpoints) { free_idx = i; }
      continue;
    }
    if (e.port == port && std::string_view{e.host.data(), e.host_len} == host) {
      e.last_use = _clock;
      return static_cast<int>(i);
    }
    if (e.last_use < _endpoints[lru_idx].last_use || !_endpoints[lru_idx].used) { lru_idx = i; }
  }
  unsigned const idx = free_idx != max_endpoints ? free_idx : lru_idx;
  clear_locked(idx);
  endpoint& e = _endpoints[idx];
  std::copy(host.begin(), host.end(), e.host.begin());
  e.host_len = static_cast<std::uint16_t>(host.size());
  e.port     = port;
  e.used     = true;
  e.last_use = _clock;
  return static_cast<int>(idx);
}

SSL_SESSION* session_cache::pop_newest_locked(unsigned b) noexcept
{
  endpoint& e = _endpoints[b];
  if (e.count == 0) { return nullptr; }
  SSL_SESSION*& s  = slot(b, (e.head + e.count - 1) % _cap);
  SSL_SESSION* out = s;
  s                = nullptr;
  --e.count;
  return out;
}

SSL_SESSION* session_cache::take(int b) noexcept
{
  if (b < 0 || static_cast<unsigned>(b) >= max_endpoints) { return nullptr; }
  auto const ub         = static_cast<unsigned>(b);
  std::time_t const now = std::time(nullptr);
  std::lock_guard lock(_mu);
  endpoint& e = _endpoints[ub];
  while (e.count > 0) {
    SSL_SESSION* s = slot(ub, (e.head + e.count - 1) % _cap);
    if (!usable(s, now)) {
      SSL_SESSION_free(pop_newest_locked(ub));
      continue;
    }
    if (is_tls13(s)) { return pop_newest_locked(ub); }  // RFC 8446 C.4: tickets are single use
    // TLS 1.2: keep ours and hand out a copy, so the connection that uses it (and is later freed
    // without close_notify) cannot mark the cached object not_resumable.
    return SSL_SESSION_dup(s);
  }
  return nullptr;
}

void session_cache::put(int b, SSL_SESSION* s) noexcept
{
  if (s == nullptr) { return; }
  if (b < 0 || static_cast<unsigned>(b) >= max_endpoints) {
    SSL_SESSION_free(s);
    return;
  }
  auto const ub = static_cast<unsigned>(b);
  std::lock_guard lock(_mu);
  endpoint& e = _endpoints[ub];
  if (!is_tls13(s)) { clear_locked(ub); }  // a TLS 1.2 session replaces the endpoint's others
  if (e.count == _cap) {                   // full: evict the oldest
    SSL_SESSION*& oldest = slot(ub, e.head);
    SSL_SESSION_free(oldest);
    oldest = nullptr;
    e.head = (e.head + 1) % _cap;
    --e.count;
  }
  slot(ub, (e.head + e.count) % _cap) = s;
  ++e.count;
}

void session_cache::drop(int b) noexcept
{
  if (b < 0 || static_cast<unsigned>(b) >= max_endpoints) { return; }
  std::lock_guard lock(_mu);
  clear_locked(static_cast<unsigned>(b));
}

unsigned session_cache::count(int b) const noexcept
{
  if (b < 0 || static_cast<unsigned>(b) >= max_endpoints) { return 0; }
  std::lock_guard lock(_mu);
  return _endpoints[static_cast<unsigned>(b)].count;
}

}  // namespace cucascade::io::tls::openssl
