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

/// @file dns_resolver.cpp
/// @brief Resolver thread, accumulation, refresh-ahead, benching and lock-free publication
/// (dns_resolver.hpp; s3_line_rate §2.7 "DNS refresh").

#include "io/rest/net/dns_resolver.hpp"

#include <arpa/inet.h>
#include <netdb.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>

namespace cucascade::io::net {

std::uint64_t mono_ns() noexcept
{
  timespec ts{};
  ::clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
         static_cast<std::uint64_t>(ts.tv_nsec);
}

bool parse_ip_literal(std::string_view host,
                      std::uint16_t port,
                      sockaddr_storage& out,
                      socklen_t& len) noexcept
{
  if (host.size() >= 2 && host.front() == '[' && host.back() == ']')
    host = host.substr(1, host.size() - 2);
  if (host.empty() || host.size() >= INET6_ADDRSTRLEN) return false;
  char buf[INET6_ADDRSTRLEN]{};
  std::memcpy(buf, host.data(), host.size());
  out      = sockaddr_storage{};
  auto* v4 = reinterpret_cast<sockaddr_in*>(&out);
  if (::inet_pton(AF_INET, buf, &v4->sin_addr) == 1) {
    v4->sin_family = AF_INET;
    v4->sin_port   = htons(port);
    len            = sizeof(sockaddr_in);
    return true;
  }
  out      = sockaddr_storage{};
  auto* v6 = reinterpret_cast<sockaddr_in6*>(&out);
  if (::inet_pton(AF_INET6, buf, &v6->sin6_addr) == 1) {
    v6->sin6_family = AF_INET6;
    v6->sin6_port   = htons(port);
    len             = sizeof(sockaddr_in6);
    return true;
  }
  out = sockaddr_storage{};
  return false;
}

socklen_t sockaddr_length(sockaddr_storage const& a) noexcept
{
  switch (a.ss_family) {
    case AF_INET: return sizeof(sockaddr_in);
    case AF_INET6: return sizeof(sockaddr_in6);
    default: return 0;
  }
}

bool same_address(sockaddr_storage const& a, sockaddr_storage const& b) noexcept
{
  if (a.ss_family != b.ss_family) return false;
  if (a.ss_family == AF_INET) {
    auto const& x = reinterpret_cast<sockaddr_in const&>(a);
    auto const& y = reinterpret_cast<sockaddr_in const&>(b);
    return x.sin_port == y.sin_port && x.sin_addr.s_addr == y.sin_addr.s_addr;
  }
  if (a.ss_family == AF_INET6) {
    auto const& x = reinterpret_cast<sockaddr_in6 const&>(a);
    auto const& y = reinterpret_cast<sockaddr_in6 const&>(b);
    return x.sin6_port == y.sin6_port &&
           std::memcmp(&x.sin6_addr, &y.sin6_addr, sizeof x.sin6_addr) == 0 &&
           x.sin6_scope_id == y.sin6_scope_id;
  }
  return false;
}

char const* format_address(sockaddr_storage const& a, std::span<char> out) noexcept
{
  if (out.empty()) return "";
  out[0] = '\0';
  char ip[INET6_ADDRSTRLEN]{};
  if (a.ss_family == AF_INET) {
    auto const& x = reinterpret_cast<sockaddr_in const&>(a);
    ::inet_ntop(AF_INET, &x.sin_addr, ip, sizeof ip);
    std::snprintf(out.data(), out.size(), "%s:%u", ip, unsigned{ntohs(x.sin_port)});
  } else if (a.ss_family == AF_INET6) {
    auto const& x = reinterpret_cast<sockaddr_in6 const&>(a);
    ::inet_ntop(AF_INET6, &x.sin6_addr, ip, sizeof ip);
    std::snprintf(out.data(), out.size(), "[%s]:%u", ip, unsigned{ntohs(x.sin6_port)});
  }
  return out.data();
}

namespace {

int getaddrinfo_lookup(void*,
                       char const* host,
                       std::uint16_t port,
                       std::vector<sockaddr_storage>& out) noexcept
{
  addrinfo hints{};
  hints.ai_family   = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;
  char service[8];
  std::snprintf(service, sizeof service, "%u", unsigned{port});
  addrinfo* res = nullptr;
  int const rc  = ::getaddrinfo(host, service, &hints, &res);
  if (rc != 0) return rc;
  try {
    for (addrinfo* a = res; a != nullptr; a = a->ai_next) {
      if (a->ai_family != AF_INET && a->ai_family != AF_INET6) continue;
      if (a->ai_addrlen > sizeof(sockaddr_storage)) continue;
      sockaddr_storage s{};
      std::memcpy(&s, a->ai_addr, a->ai_addrlen);
      out.push_back(s);
    }
  } catch (...) {
    ::freeaddrinfo(res);
    return EAI_MEMORY;
  }
  ::freeaddrinfo(res);
  return out.empty() ? EAI_NODATA : 0;
}

constexpr std::uint64_t ms_to_ns(std::chrono::milliseconds d) noexcept
{
  return d.count() <= 0 ? 0 : static_cast<std::uint64_t>(d.count()) * 1'000'000ULL;
}

}  // namespace

dns_resolver::dns_resolver(options opts) noexcept : _opts(opts)
{
  if (_opts.lookup == nullptr) _opts.lookup = &getaddrinfo_lookup;
  if (_opts.max_addresses == 0) _opts.max_addresses = 1;
  _entries.reset(new (std::nothrow) entry[max_hosts]);
}

dns_resolver::~dns_resolver()
{
  {
    std::lock_guard g(_m);
    _stop = true;
  }
  _cv.notify_all();
  if (_thread.joinable()) _thread.join();
  if (_entries) {
    std::uint32_t const n = _count.load(std::memory_order_acquire);
    for (std::uint32_t i = 0; i < n; ++i) {
      _entries[i].current.store(nullptr, std::memory_order_relaxed);
    }
  }
}

std::uint64_t dns_resolver::address_key(sockaddr_storage const& a) noexcept
{
  // FNV-1a over family, port and address bytes; never 0 (0 = empty bench slot).
  std::uint64_t h = 1469598103934665603ULL;
  auto const mix  = [&h](void const* p, std::size_t n) noexcept {
    auto const* b = static_cast<unsigned char const*>(p);
    for (std::size_t i = 0; i < n; ++i) {
      h ^= b[i];
      h *= 1099511628211ULL;
    }
  };
  mix(&a.ss_family, sizeof a.ss_family);
  if (a.ss_family == AF_INET) {
    auto const& x = reinterpret_cast<sockaddr_in const&>(a);
    mix(&x.sin_port, sizeof x.sin_port);
    mix(&x.sin_addr, sizeof x.sin_addr);
  } else if (a.ss_family == AF_INET6) {
    auto const& x = reinterpret_cast<sockaddr_in6 const&>(a);
    mix(&x.sin6_port, sizeof x.sin6_port);
    mix(&x.sin6_addr, sizeof x.sin6_addr);
  }
  return h | 1;
}

std::uint32_t dns_resolver::intern(std::string_view host, std::uint16_t port) noexcept
{
  if (!_entries || host.empty()) return invalid;
  std::lock_guard g(_m);
  std::uint32_t const n = _count.load(std::memory_order_relaxed);
  for (std::uint32_t i = 0; i < n; ++i)
    if (_entries[i].port == port && _entries[i].host == host) return i;
  if (n >= max_hosts) return invalid;
  entry& e = _entries[n];
  try {
    e.host.assign(host);
    e.accum.reserve(_opts.max_addresses);
  } catch (...) {
    return invalid;
  }
  e.port = port;
  _count.store(n + 1, std::memory_order_release);
  start_thread_locked();
  _kick = true;
  _cv.notify_one();
  return n;
}

void dns_resolver::start_thread_locked() noexcept
{
  if (_started.load(std::memory_order_relaxed)) return;
  try {
    _thread = std::thread([this] { run(); });
    _started.store(true, std::memory_order_release);
  } catch (...) {  // NOLINT(bugprone-empty-catch): noexcept boundary: no thread
                   // (std::system_error): _started stays false and the next add retries the start
  }
}

dns_resolver::status dns_resolver::pick(std::uint32_t h,
                                        sockaddr_storage& out,
                                        socklen_t& len,
                                        std::uint64_t now_ns) noexcept
{
  if (h >= _count.load(std::memory_order_acquire)) return status::failed;
  entry& e = _entries[h];
  e.last_pick_ns.store(now_ns, std::memory_order_relaxed);
  auto const l = e.current.load(std::memory_order_acquire);
  if (l == nullptr || l->addrs.empty())
    return e.failed.load(std::memory_order_acquire) ? status::failed : status::pending;
  // Past its TTL (a host that was idle and is in use again): serve the old answer, re-query now.
  if (now_ns > l->resolved_at_ns && now_ns - l->resolved_at_ns > ms_to_ns(_opts.ttl))
    e.refresh.store(true, std::memory_order_relaxed);
  auto const n           = static_cast<std::uint32_t>(l->addrs.size());
  std::uint32_t const at = e.rr.fetch_add(1, std::memory_order_relaxed);
  for (std::uint32_t k = 0; k < n; ++k) {
    sockaddr_storage const& a = l->addrs[(at + k) % n];
    if (!benched(h, a, now_ns)) {
      out = a;
      len = sockaddr_length(a);
      return status::ok;
    }
  }
  // Every address is benched: use the round-robin one anyway (failing would be worse) and ask
  // for a fresh answer.
  out = l->addrs[at % n];
  len = sockaddr_length(out);
  e.refresh.store(true, std::memory_order_relaxed);
  return status::ok;
}

void dns_resolver::bench(std::uint32_t h,
                         sockaddr_storage const& addr,
                         std::chrono::milliseconds d,
                         std::uint64_t now_ns) noexcept
{
  if (h >= _count.load(std::memory_order_acquire)) return;
  entry& e                  = _entries[h];
  std::uint64_t const key   = address_key(addr);
  std::uint64_t const until = now_ns + ms_to_ns(d);
  bench_slot* victim        = e.bench.data();
  for (auto& s : e.bench) {
    std::uint64_t const k = s.key.load(std::memory_order_relaxed);
    if (k == key || k == 0) {
      victim = &s;
      break;
    }
    if (s.until_ns.load(std::memory_order_relaxed) <
        victim->until_ns.load(std::memory_order_relaxed))
      victim = &s;
  }
  victim->until_ns.store(until, std::memory_order_relaxed);
  victim->key.store(key, std::memory_order_release);
}

bool dns_resolver::benched(std::uint32_t h,
                           sockaddr_storage const& addr,
                           std::uint64_t now_ns) const noexcept
{
  if (h >= _count.load(std::memory_order_acquire)) return false;
  entry const& e          = _entries[h];
  std::uint64_t const key = address_key(addr);
  for (auto const& s : e.bench)
    if (s.key.load(std::memory_order_acquire) == key)
      return s.until_ns.load(std::memory_order_relaxed) > now_ns;
  return false;
}

void dns_resolver::refresh_ahead(std::uint32_t h) noexcept
{
  if (h >= _count.load(std::memory_order_acquire)) return;
  _entries[h].refresh.store(true, std::memory_order_relaxed);
  // No notify (lock-free for reactors): the thread polls at least once per 100 ms while a
  // refresh is requested (see run()).
}

std::shared_ptr<dns_resolver::address_list const> dns_resolver::get(std::string_view host,
                                                                    std::uint16_t port) noexcept
{
  std::uint32_t const h = intern(host, port);
  if (h == invalid) return nullptr;
  return _entries[h].current.load(std::memory_order_acquire);
}

void dns_resolver::refresh_ahead(std::string_view host, std::uint16_t port) noexcept
{
  std::uint32_t const h = intern(host, port);
  if (h == invalid) return;
  refresh_ahead(h);
  std::lock_guard g(_m);
  _kick = true;
  _cv.notify_one();
}

void dns_resolver::bench(sockaddr_storage const& addr, std::chrono::seconds d) noexcept
{
  std::uint64_t const now = mono_ns();
  std::uint32_t const n   = _count.load(std::memory_order_acquire);
  for (std::uint32_t i = 0; i < n; ++i) {
    auto const l = _entries[i].current.load(std::memory_order_acquire);
    if (l == nullptr) continue;
    for (auto const& a : l->addrs)
      if (same_address(a, addr)) {
        bench(i, addr, std::chrono::duration_cast<std::chrono::milliseconds>(d), now);
        break;
      }
  }
}

int dns_resolver::last_error(std::uint32_t h) const noexcept
{
  if (h >= _count.load(std::memory_order_acquire)) return 0;
  return _entries[h].error.load(std::memory_order_relaxed);
}

std::size_t dns_resolver::address_count(std::uint32_t h) const noexcept
{
  if (h >= _count.load(std::memory_order_acquire)) return 0;
  auto const l = _entries[h].current.load(std::memory_order_acquire);
  return l == nullptr ? 0 : l->addrs.size();
}

std::uint64_t dns_resolver::resolved_at_ns(std::uint32_t h) const noexcept
{
  if (h >= _count.load(std::memory_order_acquire)) return 0;
  return _entries[h].resolved_ns.load(std::memory_order_acquire);
}

std::uint64_t dns_resolver::lookups_of(std::uint32_t h) const noexcept
{
  if (h >= _count.load(std::memory_order_acquire)) return 0;
  return _entries[h].lookups.load(std::memory_order_relaxed);
}

std::uint64_t dns_resolver::due_ns(entry const& e, std::uint64_t now) const noexcept
{
  if (e.refresh.load(std::memory_order_relaxed)) return now;
  auto const l = e.current.load(std::memory_order_relaxed);
  if (l == nullptr) return e.failed.load(std::memory_order_relaxed) ? e.retry_at_ns : now;
  if (e.retry_at_ns != 0) return e.retry_at_ns;  // failed refresh with a previous answer
  // Refresh ahead of the TTL while the host is in use; idle hosts keep their last answer.
  std::uint64_t const last_pick = e.last_pick_ns.load(std::memory_order_relaxed);
  if (last_pick + ms_to_ns(_opts.idle_stop) < now && last_pick < l->resolved_at_ns)
    return ~std::uint64_t{0};
  auto const ahead = static_cast<std::uint64_t>(static_cast<double>(ms_to_ns(_opts.ttl)) *
                                                std::clamp(_opts.refresh_ahead, 0.05, 1.0));
  return l->resolved_at_ns + ahead;
}

void dns_resolver::resolve(std::uint32_t h) noexcept
{
  entry& e = _entries[h];
  e.refresh.store(false, std::memory_order_relaxed);
  std::vector<sockaddr_storage> found;
  int rc = 0;
  try {
    found.reserve(16);
    rc = _opts.lookup(_opts.lookup_ctx, e.host.c_str(), e.port, found);
  } catch (...) {
    rc = EAI_MEMORY;
  }
  _lookups.fetch_add(1, std::memory_order_relaxed);
  e.lookups.fetch_add(1, std::memory_order_relaxed);
  std::uint64_t const now = mono_ns();
  std::lock_guard g(_m);
  if (rc != 0 || found.empty()) {
    _failures.fetch_add(1, std::memory_order_relaxed);
    e.error.store(rc != 0 ? rc : EAI_NODATA, std::memory_order_relaxed);
    e.retry_at_ns = now + ms_to_ns(_opts.failure_retry);
    // Stale-while-error: keep the previous answer; only a host with none reports failure.
    if (e.current.load(std::memory_order_relaxed) == nullptr)
      e.failed.store(true, std::memory_order_release);
    return;
  }
  // Accumulate (s3_line_rate §2.7): refresh last_seen for re-seen addresses, add new ones, age
  // out those not seen for address_max_age, keep the max_addresses most recently seen.
  for (auto const& a : found) {
    auto it = std::find_if(
      e.accum.begin(), e.accum.end(), [&](accum_addr const& x) { return same_address(x.addr, a); });
    if (it != e.accum.end())
      it->last_seen_ns = now;
    else
      e.accum.push_back(accum_addr{a, now});
  }
  std::uint64_t const max_age = ms_to_ns(_opts.address_max_age);
  std::erase_if(e.accum, [&](accum_addr const& x) { return now - x.last_seen_ns > max_age; });
  if (e.accum.size() > _opts.max_addresses) {
    std::stable_sort(e.accum.begin(), e.accum.end(), [](accum_addr const& x, accum_addr const& y) {
      return x.last_seen_ns > y.last_seen_ns;
    });
    e.accum.resize(_opts.max_addresses);
  }
  std::shared_ptr<address_list> next;
  try {
    next = std::make_shared<address_list>();
    next->addrs.reserve(e.accum.size());
    for (auto const& x : e.accum)
      next->addrs.push_back(x.addr);
    next->resolved_at_ns = now;
  } catch (...) {
    e.retry_at_ns = now + ms_to_ns(_opts.failure_retry);
    return;
  }
  e.current.store(std::move(next), std::memory_order_release);
  e.resolved_ns.store(now, std::memory_order_release);
  e.error.store(0, std::memory_order_relaxed);
  e.failed.store(false, std::memory_order_release);
  e.retry_at_ns = 0;
}

void dns_resolver::run() noexcept
{
  std::vector<std::uint32_t> work;
  try {
    work.reserve(max_hosts);
  } catch (...) {  // NOLINT(bugprone-empty-catch): noexcept boundary: reserve is only a
                   // pre-allocation; the loop grows work on demand
  }
  std::unique_lock lk(_m);
  while (!_stop) {
    std::uint64_t const now = mono_ns();
    std::uint64_t next =
      now + std::uint64_t{1'000'000'000};  // re-evaluate at least once per second
    work.clear();
    std::uint32_t const n = _count.load(std::memory_order_relaxed);
    for (std::uint32_t i = 0; i < n; ++i) {
      std::uint64_t const due = due_ns(_entries[i], now);
      if (due <= now)
        work.push_back(i);
      else
        next = std::min(next, due);
    }
    if (!work.empty()) {
      lk.unlock();
      for (std::uint32_t h : work)
        resolve(h);  // getaddrinfo without the lock
      lk.lock();
      continue;
    }
    // Lock-free refresh requests from reactors are not notified: poll them every 100 ms.
    next                     = std::min<std::uint64_t>(next, now + 100'000'000ULL);
    std::uint64_t const wait = next > now ? next - now : 0;
    _kick                    = false;
    _cv.wait_for(lk, std::chrono::nanoseconds(wait), [this] { return _stop || _kick; });
  }
}

}  // namespace cucascade::io::net
