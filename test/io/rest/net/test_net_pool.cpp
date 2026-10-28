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

/// @file test_net_pool.cpp
/// @brief connection_pool over a `uring_loop` driven by the test thread, against an in-process
/// plain HTTP server with scripted behaviour (gpuio tests/unit/net/test_pool_local.cpp, the
/// plain cases): GETs byte-exact into iovecs (contiguous / scattered, one CQE per WAITALL
/// receive), warmest-first reuse, the over-read guard, FIN detection by the idle watch, the idle
/// cutoff, the stale-reuse free retry, the learned per-connection request limit with pre-opened
/// replacements, keep-warm and warm-up as background work, the urgent reserve, Content-Length
/// mismatch, connect failure and timeout, shutdown with GETs in flight. And `http1_exchange` over
/// the pool: interim 1xx heads, error-body drain and classification, Content-Range validation.

#include "get_client.hpp"
#include "io/rest/http/request_head.hpp"
#include "io/rest/net/connection_pool.hpp"
#include "io/rest/net/http1_exchange.hpp"
#include "io/rest/net/uring_loop.hpp"
#include "net_test_support.hpp"

#include <catch2/catch_all.hpp>
#include <dirent.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace cucascade::io::net;
using namespace std::chrono_literals;
using cucascade::io::rest::KiB;
using cucascade::io::rest::MiB;
using cucascade::io::rest::priority;
using client_t = net_test::get_client<loop_env>;
using job_t    = client_t::job;
namespace rest = cucascade::io::rest;
namespace http = cucascade::io::http;

namespace {

constexpr std::size_t npos = static_cast<std::size_t>(-1);

endpoint_config make_ep(std::uint16_t port)
{
  endpoint_config e;
  e.host                             = "127.0.0.1";
  e.port                             = port;
  e.tls                              = false;
  e.conn.max_connections_per_reactor = 8;
  return e;
}

job_t get_job(std::uint32_t ep, std::uint64_t size, std::vector<std::byte>& dst)
{
  job_t j;
  j.endpoint = ep;
  j.path     = "/" + std::to_string(size);
  j.dst      = {iovec{dst.data(), dst.size()}};
  return j;
}

unsigned open_fds()
{
  unsigned n = 0;
  if (DIR* d = ::opendir("/proc/self/fd")) {
    while (::readdir(d) != nullptr)
      ++n;
    ::closedir(d);
  }
  return n;
}

struct fixture {
  explicit fixture(endpoint_config cfg, dns_resolver::options dns = {})
    : rc(dns), ep(add(rc, std::move(cfg))), tl(), client(rc, tl.env())
  {
  }
  static std::uint32_t add(remote_context& rc, endpoint_config cfg)
  {
    auto r = rc.add_endpoint(std::move(cfg));
    REQUIRE(r.has_value());
    return *r;
  }
  uring_loop& loop() { return tl.loop(); }
  endpoint_counters const& ctr() { return client.pool().counters(ep); }
  bool run(job_t& j, std::chrono::milliseconds t = 30s)
  {
    return tl.run_until([&] { return j.done; }, t);
  }
  void settle(std::chrono::milliseconds t)
  {
    auto const end = std::chrono::steady_clock::now() + t;
    while (std::chrono::steady_clock::now() < end)
      tl.run_once(2'000'000ULL);
  }
  /// Shut the pool down and reap every CQE (the worker's stop protocol in miniature).
  void quiesce()
  {
    client.pool().shutdown();
    loop().cancel_all();
    REQUIRE(loop().drain([&]() noexcept { return client.pool().quiescent(); }, 10'000'000'000ULL));
    CHECK(loop().inflight_ops() == 0);
    CHECK(loop().work_ops() == 0);
    CHECK(loop().background_ops() == 0);
  }
  remote_context rc;
  std::uint32_t ep;
  net_test::test_loop tl;
  client_t client;
};

#define REQUIRE_URING()                                                                     \
  do {                                                                                      \
    if (!net_test::uring_available()) SKIP("io_uring with DEFER_TASKRUN is not available"); \
  } while (0)

}  // namespace

TEST_CASE("net: plain HTTP GET byte-exact, contiguous and scattered destinations",
          "[rest][net][pool]")
{
  REQUIRE_URING();
  net_test::test_server srv({.tls = false});
  REQUIRE(srv.ok());
  fixture harness(make_ep(srv.port()));
  std::uint64_t const size = 8 * MiB + 12345;
  SECTION("contiguous: the body lands straight in the destination")
  {
    std::vector<std::byte> dst(size);
    job_t j = get_job(harness.ep, size, dst);
    harness.client.start(j);
    REQUIRE(harness.run(j));
    REQUIRE_FALSE(j.ec);
    CHECK(j.status == 200);
    CHECK(j.body == size);
    CHECK(net_test::check_pattern(dst, srv.seed(), 0) == npos);
    CHECK(harness.ctr().connects_ok == 1);
    // WAITALL: one CQE per min(remaining, 4 MiB) receive (gpuio N1).
    CHECK(harness.ctr().body_cqes <= (size + 4 * MiB - 1) / (4 * MiB) + 1);
    CHECK(harness.ctr().body_bytes + harness.ctr().header_bytes >= size);
  }
  SECTION("scattered (RECVMSG|WAITALL into 3 ranges)")
  {
    std::vector<std::byte> buf(size + 2 * MiB);
    std::uint64_t const a = 1 * MiB + 7, b = 3 * MiB, c = size - a - b;
    job_t j;
    j.endpoint = harness.ep;
    j.path     = "/" + std::to_string(size);
    j.dst      = {iovec{buf.data(), a},
                  iovec{buf.data() + a + 4096, b},
                  iovec{buf.data() + a + b + 2 * 4096, c}};
    harness.client.start(j);
    REQUIRE(harness.run(j));
    REQUIRE_FALSE(j.ec);
    CHECK(net_test::check_pattern(std::span(buf.data(), a), srv.seed(), 0) == npos);
    CHECK(net_test::check_pattern(std::span(buf.data() + a + 4096, b), srv.seed(), a) == npos);
    CHECK(net_test::check_pattern(std::span(buf.data() + a + b + 8192, c), srv.seed(), a + b) ==
          npos);
  }
  SECTION("ranged GET: 206 into the destination")
  {
    std::vector<std::byte> dst(1 * MiB);
    job_t j = get_job(harness.ep, size, dst);
    j.range = std::pair<std::uint64_t, std::uint64_t>{4096, 4096 + dst.size() - 1};
    harness.client.start(j);
    REQUIRE(harness.run(j));
    REQUIRE_FALSE(j.ec);
    CHECK(j.status == 206);
    CHECK(net_test::check_pattern(dst, srv.seed(), 4096) == npos);
  }
  harness.quiesce();
}

TEST_CASE("net: keep-alive reuse is warmest-first (LIFO)", "[rest][net][pool]")
{
  REQUIRE_URING();
  net_test::test_server srv({.tls = false});
  REQUIRE(srv.ok());
  fixture harness(make_ep(srv.port()));
  harness.client.pool().warmup(harness.ep, 3);
  REQUIRE(
    harness.tl.run_until([&] { return harness.client.pool().idle_count(harness.ep) == 3; }, 5s));
  std::vector<std::byte> dst(64 * KiB);
  // Two GETs at once take two connections; the one checked in last is the next one used.
  job_t a = get_job(harness.ep, dst.size(), dst);
  job_t b = get_job(harness.ep, dst.size(), dst);
  harness.client.start(a);
  harness.client.start(b);
  REQUIRE(harness.tl.run_until([&] { return a.done && b.done; }, 10s));
  REQUIRE_FALSE(a.ec);
  REQUIRE_FALSE(b.ec);
  CHECK_FALSE(a.reused);  // warm connections have served no request yet
  CHECK(a.slot != b.slot);
  std::uint32_t const last = a.t_end > b.t_end ? a.slot : b.slot;
  for (int i = 0; i < 5; ++i) {
    job_t j = get_job(harness.ep, dst.size(), dst);
    harness.client.start(j);
    REQUIRE(harness.run(j));
    REQUIRE_FALSE(j.ec);
    CHECK(j.reused);
    CHECK(j.slot == last);  // the warmest connection, every time
  }
  CHECK(harness.ctr().connects_ok == 3);
  CHECK(harness.ctr().reused_checkouts == 5);
  CHECK(srv.connections() == 3);
  harness.quiesce();
}

TEST_CASE("net: over-read guard: body receives never exceed the bytes the response owes",
          "[rest][net][pool]")
{
  REQUIRE_URING();
  net_test::test_server srv({.tls = false, .idle_close = 1500ms});
  REQUIRE(srv.ok());
  fixture harness(make_ep(srv.port()));
  std::uint64_t const size = 1 * MiB + 333;
  std::vector<std::byte> dst(3 * MiB);  // destination larger than the body
  SECTION("guard on: the last receive is min(remaining, cap)")
  {
    job_t j       = get_job(harness.ep, size, dst);
    auto const t0 = net_test::now_ns();
    harness.client.start(j);
    REQUIRE(harness.run(j, 5s));
    REQUIRE_FALSE(j.ec);
    CHECK(net_test::now_ns() - t0 < 1'000'000'000ULL);
    CHECK(net_test::check_pattern(std::span(dst.data(), size), srv.seed(), 0) == npos);
    CHECK(harness.ctr().overread_guarded >= 1);
    CHECK(harness.ctr().overread_issued == 0);
    CHECK(harness.ctr().body_bytes_requested == harness.ctr().body_bytes);
  }
  SECTION("guard off (test knob): the WAITALL receive waits for the peer to close")
  {
    harness.client.pool().debug().disable_overread_guard = true;
    job_t j                                              = get_job(harness.ep, size, dst);
    harness.client.start(j);
    harness.settle(600ms);
    CHECK_FALSE(j.done);  // stuck in an over-long WAITALL receive
    CHECK(harness.ctr().overread_issued >= 1);
    // It only completes when the server closes the idle connection (1.5 s here; 75 s nginx,
    // ~5 s S3).
    REQUIRE(harness.run(j, 10s));
    CHECK(net_test::now_ns() - j.t_start >= 1'200'000'000ULL);
  }
  harness.quiesce();
}

TEST_CASE("net: idle watch closes a FIN'd connection before the next checkout",
          "[rest][net][pool][liveness]")
{
  REQUIRE_URING();
  net_test::test_server srv({.tls = false, .idle_close = 300ms});
  REQUIRE(srv.ok());
  fixture harness(make_ep(srv.port()));
  std::vector<std::byte> dst(64 * KiB);
  job_t j = get_job(harness.ep, dst.size(), dst);
  harness.client.start(j);
  REQUIRE(harness.run(j));
  REQUIRE_FALSE(j.ec);
  REQUIRE(harness.client.pool().idle_count(harness.ep) == 1);
  CHECK(harness.ctr().watch_armed >= 1);
  // The server closes the idle connection at ~300 ms; the watch CQE closes our slot at once.
  REQUIRE(harness.tl.run_until([&] { return harness.ctr().idle_fin_evictions != 0; }, 5s));
  CHECK(harness.client.pool().idle_count(harness.ep) == 0);
  // The next GET connects afresh: no stale reuse, no failed request.
  job_t j2 = get_job(harness.ep, dst.size(), dst);
  harness.client.start(j2);
  REQUIRE(harness.run(j2));
  CHECK_FALSE(j2.ec);
  CHECK_FALSE(j2.reused);
  CHECK(harness.ctr().stale_detected == 0);
  CHECK(harness.ctr().connects_ok == 2);
  harness.quiesce();
}

TEST_CASE("net: idle cutoff closes connections idle longer than reuse_idle_max",
          "[rest][net][pool][liveness]")
{
  REQUIRE_URING();
  net_test::test_server srv({.tls = false});
  REQUIRE(srv.ok());
  endpoint_config e     = make_ep(srv.port());
  e.conn.reuse_idle_max = 200ms;
  fixture harness(std::move(e));
  std::vector<std::byte> dst(4 * KiB);
  job_t j = get_job(harness.ep, dst.size(), dst);
  harness.client.start(j);
  REQUIRE(harness.run(j));
  REQUIRE(harness.client.pool().idle_count(harness.ep) == 1);
  harness.settle(600ms);  // the periodic tick (250 ms) applies the cutoff
  CHECK(harness.client.pool().idle_count(harness.ep) == 0);
  CHECK(harness.ctr().idle_cutoff_closes == 1);
  job_t j2 = get_job(harness.ep, dst.size(), dst);
  harness.client.start(j2);
  REQUIRE(harness.run(j2));
  CHECK_FALSE(j2.ec);
  CHECK_FALSE(j2.reused);
  harness.quiesce();
}

TEST_CASE("net: stale reuse (silent server close) is retried for free on a fresh connection",
          "[rest][net][pool][liveness]")
{
  REQUIRE_URING();
  bool const watch = GENERATE(false, true);
  net_test::test_server srv({.tls = false, .close_after_requests = 2});
  REQUIRE(srv.ok());
  endpoint_config e     = make_ep(srv.port());
  e.conn.watch_idle_fin = watch;
  fixture harness(std::move(e));
  std::vector<std::byte> dst(128 * KiB);
  for (int i = 0; i < 5; ++i) {
    job_t j = get_job(harness.ep, dst.size(), dst);
    harness.client.start(j);
    REQUIRE(harness.run(j));
    REQUIRE_FALSE(j.ec);
    CHECK(net_test::check_pattern(dst, srv.seed(), 0) == npos);
    harness.settle(30ms);  // the server's FIN reaches us while the connection is idle
  }
  if (watch) {
    // Detected at idle by the watch: the request never meets the dead connection.
    CHECK(harness.ctr().stale_detected == 0);
    CHECK(harness.ctr().idle_fin_evictions >= 2);
  } else {
    // Found by reuse: died before any response byte => errc::stale_connection, free retry.
    CHECK(harness.ctr().stale_detected >= 2);
    CHECK(harness.client.stale_retries >= 2);
  }
  CHECK(harness.client.failed == 0);
  harness.quiesce();
}

TEST_CASE("net: server request limit is learned and replacements are pre-opened",
          "[rest][net][pool][liveness]")
{
  REQUIRE_URING();
  // `Connection: close` on the 5th response of every connection (S3 does it on the 100th).
  net_test::test_server srv({.tls = false, .request_limit = 5});
  REQUIRE(srv.ok());
  fixture harness(make_ep(srv.port()));
  std::vector<std::byte> dst(16 * KiB);
  for (int i = 0; i < 40; ++i) {
    job_t j = get_job(harness.ep, dst.size(), dst);
    harness.client.start(j);
    REQUIRE(harness.run(j));
    REQUIRE_FALSE(j.ec);
    CHECK(net_test::check_pattern(dst, srv.seed(), 0) == npos);
  }
  auto const& c = harness.ctr();
  CHECK(harness.client.pool().server_request_limit(harness.ep) == 5);
  CHECK(c.server_request_limit == 5);
  CHECK(c.peer_closes >= 2);
  CHECK(c.peer_close_requests == 5 * c.peer_closes);
  CHECK(c.replacements >= 1);  // the reconnect happens ahead of the close
  CHECK(harness.client.failed == 0);
  CHECK(harness.client.stale_retries == 0);
  // After learning, the next request after each close finds a connection ready: far fewer
  // checkout misses than closes.
  CHECK(c.checkout_misses < c.peer_closes);
  harness.quiesce();
}

TEST_CASE("net: Content-Length mismatch (truncated body) fails the GET", "[rest][net][pool]")
{
  REQUIRE_URING();
  net_test::test_server srv({.tls = false, .short_body_by = 1000});
  REQUIRE(srv.ok());
  fixture harness(make_ep(srv.port()));
  harness.client.free_retry = false;
  std::vector<std::byte> dst(256 * KiB + 1000);
  job_t j = get_job(harness.ep, 256 * KiB, dst);
  harness.client.start(j);
  REQUIRE(harness.run(j, 5s));
  CHECK(j.ec == rest::errc::http_length_mismatch);
  CHECK(j.body == 256 * KiB);  // what arrived is in the destination
  CHECK(net_test::check_pattern(std::span(dst.data(), 256 * KiB), srv.seed(), 0) == npos);
  harness.quiesce();
}

TEST_CASE("net: a response head larger than a small header buffer fails as too large",
          "[rest][net][pool]")
{
  REQUIRE_URING();
  net_test::test_server srv({.tls = false});
  REQUIRE(srv.ok());
  // header_buffer_bytes below the server's head: the parser's limit is the buffer the
  // connection really has, so the head fails as too large (never a zero-length receive).
  auto cfg                     = make_ep(srv.port());
  cfg.conn.header_buffer_bytes = 24;
  fixture harness(std::move(cfg));
  harness.client.free_retry = false;
  std::vector<std::byte> dst(4 * KiB);
  job_t j = get_job(harness.ep, dst.size(), dst);
  harness.client.start(j);
  REQUIRE(harness.run(j, 5s));
  CHECK(j.ec == rest::errc::http_header_too_large);
  harness.quiesce();
}

TEST_CASE("net: interim 1xx responses are skipped", "[rest][net][pool]")
{
  REQUIRE_URING();
  net_test::test_server srv({.tls = false, .interim_heads = 3});
  REQUIRE(srv.ok());
  fixture harness(make_ep(srv.port()));
  std::vector<std::byte> dst(100 * KiB);
  for (int i = 0; i < 3; ++i) {
    job_t j = get_job(harness.ep, dst.size(), dst);
    harness.client.start(j);
    REQUIRE(harness.run(j));
    REQUIRE_FALSE(j.ec);
    CHECK(j.status == 200);
    CHECK(net_test::check_pattern(dst, srv.seed(), 0) == npos);
  }
  CHECK(harness.ctr().connects_ok == 1);  // the connection survives the interim heads
  harness.quiesce();
}

TEST_CASE("net: connect refused, connect timeout (DNS never answers), DNS failure",
          "[rest][net][pool]")
{
  REQUIRE_URING();
  SECTION("refused")
  {
    fixture harness(make_ep(net_test::closed_port()));
    std::vector<std::byte> dst(4 * KiB);
    job_t j = get_job(harness.ep, dst.size(), dst);
    harness.client.start(j);
    REQUIRE(harness.run(j, 5s));
    CHECK(j.ec == rest::errc::connect_failed);
    CHECK(harness.ctr().connect_failures == 1);
    CHECK(harness.client.connect_failures == 1);
    harness.quiesce();
  }
  SECTION("connect_timeout from the reactor heap while the resolver stalls")
  {
    struct slow {
      static int lookup(void*, char const*, std::uint16_t, std::vector<sockaddr_storage>&) noexcept
      {
        std::this_thread::sleep_for(1500ms);
        return EAI_AGAIN;
      }
    };
    dns_resolver::options o;
    o.lookup               = &slow::lookup;
    endpoint_config e      = make_ep(net_test::closed_port());
    e.host                 = "stalls.test";
    e.conn.connect_timeout = 200ms;
    fixture harness(std::move(e), o);
    std::vector<std::byte> dst(4 * KiB);
    job_t j = get_job(harness.ep, dst.size(), dst);
    harness.client.start(j);
    REQUIRE(harness.run(j, 5s));
    CHECK(j.ec == rest::errc::connect_timeout);
    CHECK(harness.ctr().connect_timeouts == 1);
    CHECK(net_test::now_ns() - j.t_start >= 180'000'000ULL);
    CHECK(net_test::now_ns() - j.t_start < 1'400'000'000ULL);  // the deadline, not the lookup
    harness.quiesce();
  }
  SECTION("DNS: unresolvable host => dns_failed; resolvable name => connected via resolver")
  {
    net_test::test_server srv({.tls = false});
    REQUIRE(srv.ok());
    struct table {
      static int lookup(void*,
                        char const* host,
                        std::uint16_t port,
                        std::vector<sockaddr_storage>& out) noexcept
      {
        if (std::string(host) != "fake.test") return EAI_NONAME;
        sockaddr_storage s{};
        socklen_t len = 0;
        (void)parse_ip_literal("127.0.0.1", port, s, len);
        out.push_back(s);
        return 0;
      }
    };
    dns_resolver::options o;
    o.lookup          = &table::lookup;
    endpoint_config e = make_ep(srv.port());
    e.host            = "fake.test";
    fixture harness(std::move(e), o);
    std::vector<std::byte> dst(64 * KiB);
    job_t j = get_job(harness.ep, dst.size(), dst);
    harness.client.start(j);
    REQUIRE(harness.run(j, 5s));
    CHECK_FALSE(j.ec);
    CHECK(harness.rc.dns().lookups() >= 1);
    endpoint_config bad = make_ep(srv.port());
    bad.host            = "nx.invalid";
    auto const bad_id   = harness.rc.add_endpoint(std::move(bad));
    REQUIRE(bad_id.has_value());
    job_t k = get_job(*bad_id, dst.size(), dst);
    harness.client.start(k);
    REQUIRE(harness.run(k, 5s));
    CHECK(k.ec == rest::errc::dns_failed);
    harness.quiesce();
  }
}

namespace {

class inert_kdf final : public cucascade::io::tls::kdf {
 public:
  std::error_code hkdf_expand_label(std::size_t,
                                    std::span<std::uint8_t const>,
                                    std::string_view,
                                    std::span<std::uint8_t const>,
                                    std::span<std::uint8_t>) const noexcept override
  {
    return std::make_error_code(std::errc::not_supported);
  }
  std::error_code tls12_prf(std::size_t,
                            std::span<std::uint8_t const>,
                            std::string_view,
                            std::span<std::uint8_t const>,
                            std::span<std::uint8_t const>,
                            std::span<std::uint8_t>) const noexcept override
  {
    return std::make_error_code(std::errc::not_supported);
  }
};

/// A provider that is never asked for anything: the no_tls pool fails before the handshake.
class inert_provider final : public cucascade::io::tls::provider {
 public:
  std::unique_ptr<cucascade::io::tls::session> new_session(
    int, cucascade::io::tls::server_identity const&, std::error_code&) const noexcept override
  {
    return nullptr;
  }
  std::unique_ptr<cucascade::io::tls::aead> new_aead(
    cucascade::io::tls::aead_cipher) const noexcept override
  {
    return nullptr;
  }
  cucascade::io::tls::kdf const& kdf() const noexcept override { return _kdf; }
  cucascade::io::tls::config const& policy() const noexcept override { return _cfg; }
  cucascade::io::tls::protocol_version max_version() const noexcept override
  {
    return cucascade::io::tls::protocol_version::tls1_3;
  }
  std::string_view name() const noexcept override { return "inert"; }
  std::string_view trust_source() const noexcept override { return "none"; }

 private:
  cucascade::io::tls::config _cfg{};
  inert_kdf _kdf{};
};

}  // namespace

TEST_CASE("net: https needs a provider; a pool built without TLS fails https with not_supported",
          "[rest][net][pool]")
{
  REQUIRE_URING();
  net_test::test_server srv({.tls = false});
  REQUIRE(srv.ok());
  {
    remote_context rc;
    endpoint_config e = make_ep(srv.port());
    e.tls             = true;
    CHECK(rc.add_endpoint(e).error() == rest::errc::invalid_argument);  // TLS needs a provider
  }
  endpoint_config e = make_ep(srv.port());
  e.tls             = true;
  e.tls_provider    = std::make_shared<inert_provider>();
  fixture harness(std::move(e));
  std::vector<std::byte> dst(4 * KiB);
  job_t j = get_job(harness.ep, dst.size(), dst);
  harness.client.start(j);
  REQUIRE(harness.run(j, 5s));
  CHECK(j.ec == rest::errc::not_supported);  // TCP connected, then no TLS in this build
  CHECK(harness.ctr().connect_failures == 1);
  harness.quiesce();
}

TEST_CASE("net: keep-warm probes idle connections and holds the floor (background work)",
          "[rest][net][pool][liveness]")
{
  REQUIRE_URING();
  net_test::test_server srv({.tls = false});
  REQUIRE(srv.ok());
  endpoint_config e = make_ep(srv.port());
  e.keep_warm       = keep_warm_config{.min_connections = 2, .interval = 150ms};
  fixture harness(std::move(e));
  harness.client.probe_path = "/16";
  // The floor opens 2 connections in the background (no request needed).
  harness.client.pool().tick();
  REQUIRE(
    harness.tl.run_until([&] { return harness.client.pool().idle_count(harness.ep) == 2; }, 5s));
  CHECK(harness.loop().work_ops() == 0);  // keep-warm never counts as outstanding work
  unsigned const before = srv.requests();
  harness.settle(800ms);
  CHECK(harness.ctr().keep_warm_probes >= 4);
  CHECK(harness.ctr().keep_warm_failures == 0);
  CHECK(srv.requests() >= before + 4);  // HEAD /16 probes reached the server
  CHECK(harness.client.pool().idle_count(harness.ep) == 2);
  harness.rc.set_keep_warm(harness.ep, {});  // policy off
  std::vector<std::byte> dst(4 * KiB);
  job_t j = get_job(harness.ep, dst.size(), dst);
  harness.client.start(j);
  REQUIRE(harness.run(j));
  CHECK_FALSE(j.ec);
  CHECK(j.reused);
  harness.quiesce();
}

TEST_CASE("net: warm-up connects in the background, rate-limited per endpoint",
          "[rest][net][pool][liveness]")
{
  REQUIRE_URING();
  net_test::test_server srv({.tls = false});
  REQUIRE(srv.ok());
  endpoint_config e         = make_ep(srv.port());
  e.max_concurrent_connects = 3;
  fixture harness(std::move(e));
  harness.client.pool().warmup(harness.ep, 6);
  std::uint32_t max_setup = 0;
  bool background_only    = true;
  REQUIRE(harness.tl.run_until(
    [&] {
      auto const g    = harness.client.pool().gauges(harness.ep);
      max_setup       = std::max(max_setup, g.connecting + g.handshaking + g.resolving);
      background_only = background_only && harness.loop().work_ops() == 0;
      return harness.client.pool().idle_count(harness.ep) == 6;
    },
    10s));
  CHECK(background_only);
  CHECK(max_setup <= 3);
  CHECK(harness.ctr().warmup_started == 6);
  CHECK(harness.ctr().connects_ok == 6);
  harness.quiesce();
}

TEST_CASE("net: urgent reserve: bulk cannot take the last reserved connections",
          "[rest][net][pool]")
{
  REQUIRE_URING();
  net_test::test_server srv({.tls = false});
  REQUIRE(srv.ok());
  endpoint_config e                  = make_ep(srv.port());
  e.conn.max_connections_per_reactor = 6;  // reserved = max(2, ceil(0.05 x 6)) = 2
  fixture harness(std::move(e));
  harness.client.pool().warmup(harness.ep, 6);
  REQUIRE(
    harness.tl.run_until([&] { return harness.client.pool().idle_count(harness.ep) == 6; }, 5s));
  auto& pool = harness.client.pool();
  std::vector<conn_slot*> held;
  for (int i = 0; i < 4; ++i) {
    conn_slot* c = pool.checkout(harness.ep, priority::bulk);
    REQUIRE(c != nullptr);
    held.push_back(c);
  }
  CHECK(pool.checkout(harness.ep, priority::normal) == nullptr);  // reserve
  CHECK(harness.ctr().reserve_denials == 1);
  for (int i = 0; i < 2; ++i) {
    conn_slot* c = pool.checkout(harness.ep, priority::urgent);
    REQUIRE(c != nullptr);
    held.push_back(c);
  }
  for (conn_slot* c : held)
    pool.check_in(*c);
  CHECK(pool.idle_count(harness.ep) == 6);
  harness.quiesce();
}

TEST_CASE("net: concurrent GETs over several connections of one endpoint", "[rest][net][pool]")
{
  REQUIRE_URING();
  net_test::test_server srv({.tls = false});
  REQUIRE(srv.ok());
  fixture harness(make_ep(srv.port()));
  constexpr int n = 8;
  std::vector<std::vector<std::byte>> dst(n, std::vector<std::byte>(4 * MiB + 99));
  std::vector<job_t> jobs;
  jobs.reserve(n);
  for (auto& d : dst)
    jobs.push_back(get_job(harness.ep, d.size(), d));
  for (auto& j : jobs)
    harness.client.start(j);
  REQUIRE(harness.tl.run_until(
    [&] { return std::ranges::all_of(jobs, [](auto const& j) -> bool { return j.done; }); }, 30s));
  for (int i = 0; i < n; ++i) {
    CHECK_FALSE(jobs[static_cast<std::size_t>(i)].ec);
    CHECK(net_test::check_pattern(dst[static_cast<std::size_t>(i)], srv.seed(), 0) == npos);
  }
  CHECK(harness.ctr().connects_ok == 8);
  CHECK(harness.client.pool().idle_count(harness.ep) == 8);
  harness.quiesce();
}

TEST_CASE("net: shutdown with GETs in flight reaps every CQE and leaks nothing",
          "[rest][net][pool]")
{
  REQUIRE_URING();
  unsigned const fds0 = open_fds();
  {
    net_test::test_server srv({.tls = false, .write_chunk = 4096});
    REQUIRE(srv.ok());
    fixture harness(make_ep(srv.port()));
    harness.client.pool().warmup(harness.ep, 2);  // idle connections with watches armed
    std::vector<std::vector<std::byte>> dst(4, std::vector<std::byte>(64 * MiB));
    std::vector<job_t> jobs;
    jobs.reserve(4);
    for (auto& d : dst)
      jobs.push_back(get_job(harness.ep, d.size(), d));
    for (auto& j : jobs)
      harness.client.start(j);
    REQUIRE(harness.tl.run_until([&] { return harness.ctr().body_bytes > 8 * MiB; }, 10s));
    std::uint32_t const held = harness.client.pool().ops_held();
    CHECK(held >= 8);   // busy slots: 2 tokens each, + idle watches
    harness.quiesce();  // shutdown + sync cancel ANY|ALL + reap until quiescent
    for (auto& j : jobs) {
      CHECK(j.done);
      CHECK((j.ec == rest::errc::context_stopped || !j.ec));
    }
    auto const g = harness.client.pool().gauges(harness.ep);
    CHECK(g.free == g.capacity);
    CHECK(harness.client.pool().ops_held() == 0);
  }
  CHECK(open_fds() <= fds0);  // every socket closed
}

// ------------------------------------------------------------------------------------------------
// http1_exchange over the pool
// ------------------------------------------------------------------------------------------------

namespace {

/// One `http1_exchange` per GET, driven as the REST worker does (http1_exchange.hpp "Driving").
struct exchange_driver {
  using pool_t     = connection_pool<loop_env, exchange_driver>;
  using exchange_t = http1_exchange<pool_t, exchange_driver>;

  exchange_driver(remote_context& rc, loop_env env) : pool(std::move(env), *this, rc) {}

  // ---- Owner ----
  exchange_window body_window(conn_slot&, std::uint64_t, std::uint64_t) noexcept
  {
    return exchange_window{std::span<iovec const>(&dst, 1), false};
  }
  bool keep_receiving(conn_slot&, std::uint64_t) noexcept { return true; }

  // ---- Handler ----
  void on_send(conn_slot& c, std::error_code ec, std::size_t n) noexcept
  {
    step(c, ex.on_send(c, ec, n));
  }
  void on_header(conn_slot& c, std::error_code ec, std::size_t n) noexcept
  {
    step(c, ex.on_header(c, ec, n));
  }
  void on_body(conn_slot& c, std::error_code ec, std::size_t n) noexcept
  {
    step(c, ex.on_body(c, ec, n));
  }
  void on_ready(std::uint32_t) noexcept
  {
    if (want && !started) begin();
  }

  void get(std::uint32_t endpoint, std::string_view path, exchange_spec s, iovec d) noexcept
  {
    ep      = endpoint;
    target  = std::string(path);
    spec    = s;
    dst     = d;
    done    = false;
    started = false;
    want    = true;
    begin();
  }

  void begin() noexcept
  {
    conn_slot* c = pool.checkout(ep, priority::normal);
    if (c == nullptr) {
      (void)pool.connect_ahead(ep, 1);
      return;
    }
    started = true;
    reused  = c->reused;
    http::request_head h(std::span<char>(c->tx_buf));
    h.start(http::method::get, target).host("localhost");
    if (spec.ranged) h.range(spec.first, spec.last);
    auto const head = h.finish();
    head_iov        = iovec{const_cast<char*>(head.data()), head.size()};
    spec.tx         = std::span<iovec const>(&head_iov, 1);
    spec.head_iovs  = 1;
    step(*c, ex.start(pool, *this, *c, spec));
  }

  void step(conn_slot& c, exchange_t::step s) noexcept
  {
    if (s != exchange_t::step::done) return;
    result = ex.result();
    error  = std::string(ex.error_body());
    if (result.reusable)
      pool.check_in(c);
    else
      pool.close(c);
    done = true;
    want = false;
  }

  pool_t pool;
  exchange_t ex{};
  std::uint32_t ep{0};
  std::string target;
  exchange_spec spec{};
  iovec dst{};
  iovec head_iov{};
  exchange_result result{};
  std::string error;
  bool done{false}, started{false}, want{false}, reused{false};
};

}  // namespace

TEST_CASE("net: http1_exchange: ranged GET, interim heads, error-body drain, Content-Range check",
          "[rest][net][pool][exchange]")
{
  REQUIRE_URING();
  SECTION("ranged 206 validated before the body lands; 1xx heads skipped; connection reused")
  {
    net_test::test_server srv({.tls = false, .interim_heads = 2});
    REQUIRE(srv.ok());
    remote_context rc;
    auto const ep = rc.add_endpoint(make_ep(srv.port()));
    REQUIRE(ep.has_value());
    net_test::test_loop tl;
    exchange_driver d(rc, tl.env());
    std::vector<std::byte> dst(300 * KiB);
    for (int i = 0; i < 2; ++i) {
      exchange_spec s{};
      s.ranged      = true;
      s.first       = 1000;
      s.last        = 1000 + dst.size() - 1;
      s.object_size = 1 * MiB;
      d.get(*ep, "/1048576", s, iovec{dst.data(), dst.size()});
      REQUIRE(tl.run_until([&] { return d.done; }, 10s));
      CHECK_FALSE(d.result.transport);
      CHECK_FALSE(d.result.outcome.ec);
      CHECK(d.result.status == 206);
      CHECK(d.result.interim == 2);
      CHECK(d.result.body_bytes == dst.size());
      CHECK(d.result.reusable);
      CHECK(net_test::check_pattern(dst, srv.seed(), 1000) == npos);
    }
    CHECK(d.reused);
    CHECK(d.pool.counters(*ep).connects_ok == 1);
    d.pool.shutdown();
    tl.loop().cancel_all();
    REQUIRE(tl.loop().drain([&]() noexcept { return d.pool.quiescent(); }));
  }
  SECTION("503 SlowDown: the error body is drained, the code extracted, the connection kept")
  {
    net_test::test_server srv({.tls = false, .error_status = 503, .error_code = "SlowDown"});
    REQUIRE(srv.ok());
    remote_context rc;
    auto const ep = rc.add_endpoint(make_ep(srv.port()));
    REQUIRE(ep.has_value());
    net_test::test_loop tl;
    exchange_driver d(rc, tl.env());
    std::vector<std::byte> dst(64 * KiB);
    d.get(*ep, "/65536", exchange_spec{}, iovec{dst.data(), dst.size()});
    REQUIRE(tl.run_until([&] { return d.done; }, 10s));
    CHECK_FALSE(d.result.transport);
    CHECK(d.result.status == 503);
    CHECK(d.result.complete);
    CHECK(d.result.reusable);  // drained: the connection stays warm across a throttle
    CHECK(d.result.outcome.ec == rest::make_http_error(503));
    CHECK(d.result.outcome.cls == http::outcome::retry_class::throttle);
    CHECK(d.error.find("<Code>SlowDown</Code>") != std::string::npos);
    CHECK(d.pool.idle_count(*ep) == 1);
    d.pool.shutdown();
    tl.loop().cancel_all();
    REQUIRE(tl.loop().drain([&]() noexcept { return d.pool.quiescent(); }));
  }
  SECTION("a 206 whose Content-Range disagrees with the request fails before any body byte")
  {
    net_test::test_server srv({.tls = false});
    REQUIRE(srv.ok());
    remote_context rc;
    auto const ep = rc.add_endpoint(make_ep(srv.port()));
    REQUIRE(ep.has_value());
    net_test::test_loop tl;
    exchange_driver d(rc, tl.env());
    std::vector<std::byte> dst(4096, std::byte{0x7e});
    exchange_spec s{};
    s.ranged      = true;
    s.first       = 0;
    s.last        = dst.size() - 1;
    s.object_size = 2 * MiB;  // the server's object is 1 MiB: total mismatch
    d.get(*ep, "/1048576", s, iovec{dst.data(), dst.size()});
    REQUIRE(tl.run_until([&] { return d.done; }, 10s));
    CHECK(d.result.outcome.ec == rest::errc::content_range_mismatch);
    CHECK_FALSE(d.result.reusable);
    CHECK(d.result.body_bytes == 0);
    CHECK(std::all_of(dst.begin(), dst.end(), [](std::byte b) { return b == std::byte{0x7e}; }));
    d.pool.shutdown();
    tl.loop().cancel_all();
    REQUIRE(tl.loop().drain([&]() noexcept { return d.pool.quiescent(); }));
  }
}
