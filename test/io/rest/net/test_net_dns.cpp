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

/// @file test_dns.cpp
/// @brief DNS resolver thread (dns_resolver.hpp): IP literals, TTL refresh-ahead while in use,
/// stop when idle, accumulation up to 32 with max age, round-robin, benching, failures and
/// stale-while-error, race-free picks from many threads (TSAN), real getaddrinfo("localhost").

#include "io/rest/net/dns_resolver.hpp"
#include "net_test_support.hpp"

#include <arpa/inet.h>
#include <catch2/catch_all.hpp>
#include <netdb.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace cucascade::io::net;
using namespace std::chrono_literals;

namespace {

/// Injectable resolver: answer k returns addresses 10.0.k.1 .. 10.0.k.per (or a fixed set), or
/// fails while `fail` is set.
struct fake_dns {
  std::atomic<unsigned> calls{0};
  std::atomic<bool> fail{false};
  std::atomic<bool> rotate{true};
  unsigned per{8};
  static int lookup(void* ctx,
                    char const* host,
                    std::uint16_t port,
                    std::vector<sockaddr_storage>& out) noexcept
  {
    auto* f          = static_cast<fake_dns*>(ctx);
    unsigned const k = f->calls.fetch_add(1);
    if (f->fail.load()) return EAI_AGAIN;
    if (std::string(host) == "nx.invalid") return EAI_NONAME;
    unsigned const set = f->rotate.load() ? k : 0;
    for (unsigned i = 0; i < f->per; ++i) {
      sockaddr_storage s{};
      auto* a            = reinterpret_cast<sockaddr_in*>(&s);
      a->sin_family      = AF_INET;
      a->sin_port        = htons(port);
      a->sin_addr.s_addr = htonl((10u << 24) | ((set & 0xffu) << 8) | ((set >> 8) << 16) | (i + 1));
      out.push_back(s);
    }
    return 0;
  }
};

std::string ip_of(sockaddr_storage const& s)
{
  char b[64];
  return format_address(s, std::span<char>(b));
}

dns_resolver::status pick_wait(dns_resolver& r,
                               std::uint32_t h,
                               sockaddr_storage& out,
                               std::chrono::milliseconds timeout = 2000ms)
{
  socklen_t len  = 0;
  auto const end = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    auto const st = r.pick(h, out, len, mono_ns());
    if (st != dns_resolver::status::pending || std::chrono::steady_clock::now() > end) return st;
    std::this_thread::sleep_for(1ms);
  }
}

}  // namespace

TEST_CASE("net: IP literals bypass the resolver", "[rest][net][dns]")
{
  sockaddr_storage s{};
  socklen_t len = 0;
  REQUIRE(parse_ip_literal("127.0.0.1", 8444, s, len));
  CHECK(s.ss_family == AF_INET);
  CHECK(len == sizeof(sockaddr_in));
  CHECK(ntohs(reinterpret_cast<sockaddr_in&>(s).sin_port) == 8444);
  REQUIRE(parse_ip_literal("[::1]", 443, s, len));
  CHECK(s.ss_family == AF_INET6);
  REQUIRE(parse_ip_literal("fe80::1", 443, s, len));
  CHECK_FALSE(parse_ip_literal("localhost", 443, s, len));
  CHECK_FALSE(parse_ip_literal("s3.us-east-1.amazonaws.com", 443, s, len));
  CHECK_FALSE(parse_ip_literal("", 443, s, len));
  sockaddr_storage t{};
  REQUIRE(parse_ip_literal("127.0.0.1", 8444, t, len));
  REQUIRE(parse_ip_literal("127.0.0.1", 8444, s, len));
  CHECK(same_address(s, t));
  char b[64];
  CHECK(std::string(format_address(s, std::span<char>(b))) == "127.0.0.1:8444");
}

TEST_CASE("net: resolver round-robin, accumulation (cap 32, max age), TTL refresh-ahead",
          "[rest][net][dns]")
{
  fake_dns f;
  dns_resolver::options o;
  o.ttl             = 200ms;
  o.refresh_ahead   = 0.5;  // re-query every 100 ms while in use
  o.max_addresses   = 32;
  o.address_max_age = 60s;
  o.lookup          = &fake_dns::lookup;
  o.lookup_ctx      = &f;
  dns_resolver r(o);
  CHECK_FALSE(r.thread_started());  // lazy
  std::uint32_t const h = r.intern("s3.test", 443);
  REQUIRE(h != dns_resolver::invalid);
  CHECK(r.intern("s3.test", 443) == h);  // idempotent
  CHECK(r.thread_started());
  sockaddr_storage a{};
  REQUIRE(pick_wait(r, h, a) == dns_resolver::status::ok);
  CHECK(r.address_count(h) == 8);

  SECTION("round-robin visits every address")
  {
    std::set<std::string> seen;
    socklen_t len = 0;
    for (int i = 0; i < 8; ++i) {
      REQUIRE(r.pick(h, a, len, mono_ns()) == dns_resolver::status::ok);
      seen.insert(ip_of(a));
    }
    CHECK(seen.size() == 8);
  }
  SECTION("refresh-ahead while in use; answers accumulate up to 32")
  {
    // Keep picking (in use) for ~700 ms: >= 5 refreshes of 8 new addresses each.
    auto const end = std::chrono::steady_clock::now() + 700ms;
    socklen_t len  = 0;
    while (std::chrono::steady_clock::now() < end) {
      (void)r.pick(h, a, len, mono_ns());
      std::this_thread::sleep_for(5ms);
    }
    CHECK(r.lookups_of(h) >= 5);
    CHECK(r.address_count(h) == 32);  // accumulated, capped
    // Most recently seen addresses kept: the newest answer is in the list.
    std::uint64_t const age = mono_ns() - r.resolved_at_ns(h);
    CHECK(age < 250'000'000ULL);  // never older than the TTL while in use
  }
  SECTION("idle hosts stop refreshing")
  {
    dns_resolver::options o2 = o;
    o2.idle_stop             = 100ms;
    fake_dns f2;
    o2.lookup_ctx = &f2;
    dns_resolver r2(o2);
    std::uint32_t const h2 = r2.intern("idle.test", 443);
    REQUIRE(pick_wait(r2, h2, a) == dns_resolver::status::ok);
    std::this_thread::sleep_for(600ms);  // no picks
    CHECK(r2.lookups_of(h2) <= 3);       // first + at most the in-use refresh window
  }
}

TEST_CASE("net: resolver max address age drops addresses no longer returned", "[rest][net][dns]")
{
  fake_dns f;
  dns_resolver::options o;
  o.ttl             = 100ms;
  o.refresh_ahead   = 0.5;
  o.address_max_age = 300ms;
  o.lookup          = &fake_dns::lookup;
  o.lookup_ctx      = &f;
  dns_resolver r(o);
  std::uint32_t const h = r.intern("age.test", 443);
  sockaddr_storage a{};
  REQUIRE(pick_wait(r, h, a) == dns_resolver::status::ok);
  f.rotate.store(false);  // from now on the same 8 addresses
  auto const end = std::chrono::steady_clock::now() + 900ms;
  socklen_t len  = 0;
  while (std::chrono::steady_clock::now() < end) {
    (void)r.pick(h, a, len, mono_ns());
    std::this_thread::sleep_for(5ms);
  }
  CHECK(r.address_count(h) == 8);  // the rotating ones aged out (300 ms)
}

TEST_CASE("net: resolver benching skips failed addresses until the bench expires",
          "[rest][net][dns]")
{
  fake_dns f;
  f.rotate.store(false);
  f.per = 3;
  dns_resolver::options o;
  o.lookup     = &fake_dns::lookup;
  o.lookup_ctx = &f;
  dns_resolver r(o);
  std::uint32_t const h = r.intern("bench.test", 443);
  sockaddr_storage a{}, bad{};
  REQUIRE(pick_wait(r, h, bad) == dns_resolver::status::ok);
  std::uint64_t const now = mono_ns();
  r.bench(h, bad, 200ms, now);
  CHECK(r.benched(h, bad, now));
  socklen_t len = 0;
  for (int i = 0; i < 12; ++i) {
    REQUIRE(r.pick(h, a, len, now) == dns_resolver::status::ok);
    CHECK_FALSE(same_address(a, bad));
  }
  CHECK_FALSE(r.benched(h, bad, now + 300'000'000ULL));
  // Everything benched: still an address (failing would be worse), plus a refresh request.
  sockaddr_storage all[3];
  for (auto& x : all) {
    REQUIRE(r.pick(h, x, len, now) == dns_resolver::status::ok);
    r.bench(h, x, 10s, now);
  }
  CHECK(r.pick(h, a, len, now) == dns_resolver::status::ok);
}

TEST_CASE("net: resolver failures: failed without an answer, stale-while-error with one",
          "[rest][net][dns]")
{
  fake_dns f;
  dns_resolver::options o;
  o.failure_retry = 50ms;
  o.ttl           = 100ms;
  o.refresh_ahead = 0.5;
  o.lookup        = &fake_dns::lookup;
  o.lookup_ctx    = &f;
  dns_resolver r(o);
  std::uint32_t const nx = r.intern("nx.invalid", 443);
  sockaddr_storage a{};
  CHECK(pick_wait(r, nx, a) == dns_resolver::status::failed);
  CHECK(r.last_error(nx) == EAI_NONAME);
  // A host that resolved once keeps its answer while lookups fail.
  std::uint32_t const h = r.intern("flaky.test", 443);
  REQUIRE(pick_wait(r, h, a) == dns_resolver::status::ok);
  f.fail.store(true);
  std::this_thread::sleep_for(300ms);
  socklen_t len = 0;
  CHECK(r.pick(h, a, len, mono_ns()) == dns_resolver::status::ok);
  CHECK(r.lookup_failures() >= 1);
  f.fail.store(false);
  // Recovery of a failed host after failure_retry.
  fake_dns g;
  g.fail.store(true);
  o.lookup_ctx = &g;
  dns_resolver r2(o);
  std::uint32_t const h2 = r2.intern("late.test", 443);
  CHECK(pick_wait(r2, h2, a) == dns_resolver::status::failed);
  g.fail.store(false);
  CHECK(pick_wait(r2, h2, a, 1000ms) != dns_resolver::status::pending);
  auto const end = std::chrono::steady_clock::now() + 1s;
  while (r2.pick(h2, a, len, mono_ns()) != dns_resolver::status::ok &&
         std::chrono::steady_clock::now() < end)
    std::this_thread::sleep_for(5ms);
  CHECK(r2.pick(h2, a, len, mono_ns()) == dns_resolver::status::ok);
}

TEST_CASE("net: resolver picks are race-free across reactor threads while lists are replaced",
          "[rest][net][dns][stress]")
{
  fake_dns f;
  dns_resolver::options o;
  o.ttl           = 20ms;  // republish constantly while 4 threads pick
  o.refresh_ahead = 0.5;
  o.lookup        = &fake_dns::lookup;
  o.lookup_ctx    = &f;
  dns_resolver r(o);
  std::uint32_t const h = r.intern("busy.test", 443);
  sockaddr_storage a{};
  REQUIRE(pick_wait(r, h, a) == dns_resolver::status::ok);
  std::atomic<bool> stop{false};
  std::atomic<std::uint64_t> ok{0};
  std::vector<std::thread> t;
  for (int i = 0; i < 4; ++i)
    t.emplace_back([&] {
      sockaddr_storage x{};
      socklen_t len = 0;
      while (!stop.load()) {
        if (r.pick(h, x, len, mono_ns()) == dns_resolver::status::ok && x.ss_family == AF_INET)
          ok.fetch_add(1);
        r.bench(h, x, 1ms, mono_ns());
      }
    });
  std::this_thread::sleep_for(300ms);
  stop.store(true);
  for (auto& x : t)
    x.join();
  CHECK(ok.load() > 1000);
  CHECK(r.lookups_of(h) >= 5);
}

TEST_CASE("net: resolver getaddrinfo(localhost) and the B.6 get() shape", "[rest][net][dns]")
{
  dns_resolver r;
  auto const end = std::chrono::steady_clock::now() + 3s;
  std::shared_ptr<dns_resolver::address_list const> l;
  while (!(l = r.get("localhost", 8444)) && std::chrono::steady_clock::now() < end)
    std::this_thread::sleep_for(2ms);
  REQUIRE(l != nullptr);
  REQUIRE_FALSE(l->addrs.empty());
  bool loopback = false;
  for (auto const& s : l->addrs) {
    if (s.ss_family == AF_INET)
      loopback |= ntohl(reinterpret_cast<sockaddr_in const&>(s).sin_addr.s_addr) == 0x7f000001u;
    if (s.ss_family == AF_INET6) loopback = true;
  }
  CHECK(loopback);
}
