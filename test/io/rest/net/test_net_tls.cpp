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

/// @file test_net_tls.cpp
/// @brief HTTPS through the pool's TLS seam (`tls_support`) against an in-process OpenSSL server
/// (gpuio tests/unit/net/test_pool_local.cpp kTLS cases and test_tls_stream.cpp, ported): kTLS
/// RX+TX installed by cuCascade from the exported keys (TLS 1.3 and 1.2, NO_PAD, tickets drained
/// by `ktls::on_control_record`, mid-body ticket + KeyUpdate with the reply on the kTLS socket),
/// the user-space shell on tier 2 (built-in record layer) and tier 1 (library opener), the
/// automatic fallback under `prefer` and the failure under `require`, close_notify, truncated
/// bodies, handshake timeout and verification failure, shutdown with TLS GETs in flight.
/// kTLS cases SKIP when the kernel TLS module is absent.

#include "get_client.hpp"
#include "io/rest/net/connection_pool.hpp"
#include "io/rest/net/tls_stream.hpp"
#include "io/rest/net/uring_loop.hpp"
#include "io/tls/ktls.hpp"
#include "net_test_support.hpp"

#include <cucascade/io/tls/tls.hpp>

#include <catch2/catch_all.hpp>
#include <dirent.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

using namespace cucascade::io::net;
using namespace std::chrono_literals;
using cucascade::io::rest::KiB;
using cucascade::io::rest::MiB;
using client_t = net_test::get_client<loop_env, tls_support>;
using job_t    = client_t::job;
namespace rest = cucascade::io::rest;
namespace tls  = cucascade::io::tls;

namespace {

constexpr std::size_t npos = static_cast<std::size_t>(-1);

std::shared_ptr<tls::provider const> provider(tls::config cfg)
{
  cfg.ca_blob = net_test::localhost_cert().cert_pem;
  return tls::make_openssl_provider(cfg);
}

endpoint_config make_ep(std::uint16_t port, tls::config cfg = {})
{
  endpoint_config e;
  e.host                             = "127.0.0.1";
  e.port                             = port;
  e.tls                              = true;
  e.tls_provider                     = provider(std::move(cfg));
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
  explicit fixture(endpoint_config cfg) : ep(add(rc, std::move(cfg))), tl(), client(rc, tl.env()) {}
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
  /// The first live slot of the endpoint (one connection in these tests).
  conn_slot const* live_slot()
  {
    for (auto const& s : client.pool().slots(ep))
      if (s.state == conn_state::idle || s.state == conn_state::busy) return &s;
    return nullptr;
  }
  void quiesce()
  {
    client.pool().shutdown();
    loop().cancel_all();
    REQUIRE(loop().drain([&]() noexcept { return client.pool().quiescent(); }, 10'000'000'000ULL));
    CHECK(loop().inflight_ops() == 0);
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

#define REQUIRE_KTLS()                                                                 \
  do {                                                                                 \
    REQUIRE_URING();                                                                   \
    if (!net_test::ktls_available()) SKIP("kernel TLS module (tls.ko) is not loaded"); \
  } while (0)

tls::config user_space(tls::user_space_rx rx)
{
  tls::config c;
  c.ktls        = tls::ktls_mode::disable;
  c.fallback_rx = rx;
  return c;
}

}  // namespace

TEST_CASE("net: kTLS GET (TLS 1.3 / 1.2): keys installed by cuCascade, tickets drained, NO_PAD",
          "[rest][net][tls][ktls]")
{
  REQUIRE_KTLS();
  bool const tls13 = GENERATE(true, false);
  net_test::test_server srv(
    {.tls = true, .max_version = tls13 ? TLS1_3_VERSION : TLS1_2_VERSION, .num_tickets = 2});
  REQUIRE(srv.ok());
  fixture harness(make_ep(srv.port()));
  std::uint64_t const size = 16 * MiB;
  std::vector<std::byte> dst(size);
  job_t j = get_job(harness.ep, size, dst);
  harness.client.start(j);
  REQUIRE(harness.run(j));
  REQUIRE_FALSE(j.ec);
  CHECK(j.body == size);
  CHECK(net_test::check_pattern(dst, srv.seed(), 0) == npos);
  auto const& c = harness.ctr();
  CHECK(c.ktls_rx == 1);
  CHECK(c.ktls_tx == 1);
  CHECK(c.ktls_fallbacks == 0);
  CHECK(c.tls_version == (tls13 ? 0x0304U : 0x0303U));
  if (tls13) {
    CHECK(c.no_pad_set == 1);
    // Both tickets consumed by ktls::on_control_record, never written into the body.
    harness.settle(50ms);
    CHECK(c.control_records_ticket == 2);
    CHECK(c.eio_drains + c.idle_readable_drains >= 1);
  }
  CHECK(c.body_cqes <= 5);  // 16 MiB / 4 MiB WAITALL receives
  conn_slot const* s = harness.live_slot();
  REQUIRE(s != nullptr);
  CHECK(s->kind == kind_ktls);
  CHECK(s->tls == nullptr);  // the kTLS path carries no user-space state
  CHECK((s->rekey != nullptr) == (tls13 && tls::ktls::probe().tls13_rekey));
  // A second GET on the same connection: still exact.
  std::fill(dst.begin(), dst.end(), std::byte{0});
  job_t j2 = get_job(harness.ep, size, dst);
  harness.client.start(j2);
  REQUIRE(harness.run(j2));
  REQUIRE_FALSE(j2.ec);
  CHECK(j2.reused);
  CHECK(net_test::check_pattern(dst, srv.seed(), 0) == npos);
  CHECK(harness.ctr().connects_ok == 1);
  harness.quiesce();
}

TEST_CASE("net: kTLS mid-body ticket and KeyUpdate: short WAITALL CQE, -EIO drain, re-key, resume",
          "[rest][net][tls][ktls]")
{
  REQUIRE_KTLS();
  if (!tls::ktls::probe().tls13_rekey) SKIP("kernel < 6.14: TLS 1.3 kTLS cannot re-key");
  std::uint64_t const size = 12 * MiB;
  bool const requested     = GENERATE(true, false);
  bool const scattered     = GENERATE(false, true);
  net_test::test_server srv({.tls                  = true,
                             .num_tickets          = 0,
                             .ticket_at            = 1 * MiB + 12345,
                             .key_update_at        = 6 * MiB + 777,
                             .key_update_requested = requested});
  REQUIRE(srv.ok());
  fixture harness(make_ep(srv.port()));
  std::vector<std::byte> buf(size + 8192);
  job_t j;
  j.endpoint = harness.ep;
  j.path     = "/" + std::to_string(size);
  if (scattered)
    j.dst = {iovec{buf.data(), 5 * MiB}, iovec{buf.data() + 5 * MiB + 8192, size - 5 * MiB}};
  else
    j.dst = {iovec{buf.data(), size}};
  harness.client.start(j);
  REQUIRE(harness.run(j));
  REQUIRE_FALSE(j.ec);
  CHECK(j.body == size);
  if (scattered) {
    CHECK(net_test::check_pattern(std::span(buf.data(), 5 * MiB), srv.seed(), 0) == npos);
    CHECK(net_test::check_pattern(
            std::span(buf.data() + 5 * MiB + 8192, size - 5 * MiB), srv.seed(), 5 * MiB) == npos);
  } else {
    CHECK(net_test::check_pattern(std::span(buf.data(), size), srv.seed(), 0) == npos);
  }
  auto const& c = harness.ctr();
  CHECK(c.control_records_ticket == 1);
  CHECK(c.control_records_key_update == 1);
  CHECK(c.waitall_short_cqes + c.waitall_watchdog_cancels >= 1);  // rule 3 (or the watchdog)
  CHECK(c.eio_drains == 2);
  // The connection stays usable after the re-key (TX re-keyed and the reply sent when
  // requested: the server decrypts the next request with its new key).
  std::vector<std::byte> small(64 * KiB);
  job_t j2 = get_job(harness.ep, small.size(), small);
  harness.client.start(j2);
  REQUIRE(harness.run(j2));
  CHECK_FALSE(j2.ec);
  CHECK(j2.reused);
  CHECK(net_test::check_pattern(small, srv.seed(), 0) == npos);
  harness.quiesce();
}

TEST_CASE("net: user-space TLS GETs byte-exact on tier 2 and tier 1, contiguous and scattered",
          "[rest][net][tls][user_space]")
{
  REQUIRE_URING();
  auto const rx    = GENERATE(tls::user_space_rx::record_layer, tls::user_space_rx::library);
  bool const tls13 = GENERATE(true, false);
  net_test::test_server srv(
    {.tls = true, .max_version = tls13 ? TLS1_3_VERSION : TLS1_2_VERSION, .num_tickets = 2});
  REQUIRE(srv.ok());
  fixture harness(make_ep(srv.port(), user_space(rx)));
  for (std::uint64_t const size :
       {std::uint64_t{1}, std::uint64_t{100}, 16 * KiB + 5, 1 * MiB + 3, 8 * MiB}) {
    std::vector<std::byte> dst(size);
    job_t j = get_job(harness.ep, size, dst);
    harness.client.start(j);
    REQUIRE(harness.run(j));
    REQUIRE_FALSE(j.ec);
    CHECK(j.body == size);
    CHECK(net_test::check_pattern(dst, srv.seed(), 0) == npos);
  }
  {
    std::uint64_t const size = 3 * MiB + 17;
    std::vector<std::byte> buf(size + 8192);
    job_t j;
    j.endpoint = harness.ep;
    j.path     = "/" + std::to_string(size);
    j.dst      = {iovec{buf.data(), 1 * MiB + 1},
                  iovec{buf.data() + 1 * MiB + 4097, size - 1 * MiB - 1}};
    harness.client.start(j);
    REQUIRE(harness.run(j));
    REQUIRE_FALSE(j.ec);
    CHECK(net_test::check_pattern(std::span(buf.data(), 1 * MiB + 1), srv.seed(), 0) == npos);
    CHECK(net_test::check_pattern(std::span(buf.data() + 1 * MiB + 4097, size - 1 * MiB - 1),
                                  srv.seed(),
                                  1 * MiB + 1) == npos);
  }
  auto const& c = harness.ctr();
  CHECK(c.connects_ok == 1);     // every GET on the one connection
  CHECK(c.ktls_fallbacks == 0);  // kTLS disabled by policy: not a fallback
  CHECK(c.ktls_rx == 0);
  CHECK(c.reuse_desync_closes == 0);
  conn_slot const* s = harness.live_slot();
  REQUIRE(s != nullptr);
  CHECK(s->kind == kind_tls);
  REQUIRE(s->tls != nullptr);
  auto const* st = static_cast<tls_stream const*>(s->tls);
  CHECK(st->record_layer() == (rx == tls::user_space_rx::record_layer));
  CHECK(st->counters().plaintext_rx >= 8 * MiB);
  if (rx == tls::user_space_rx::record_layer) CHECK(st->counters().waitall_recvs >= 1);
  harness.quiesce();
}

TEST_CASE(
  "net: user-space mid-body ticket and KeyUpdate byte-exact; the reply precedes the "
  "next request",
  "[rest][net][tls][user_space]")
{
  REQUIRE_URING();
  auto const rx        = GENERATE(tls::user_space_rx::record_layer, tls::user_space_rx::library);
  bool const requested = GENERATE(true, false);
  std::uint64_t const size = 6 * MiB;
  net_test::test_server srv({.tls                  = true,
                             .num_tickets          = 0,
                             .ticket_at            = 1 * MiB + 99,
                             .key_update_at        = 3 * MiB + 5,
                             .key_update_requested = requested});
  REQUIRE(srv.ok());
  fixture harness(make_ep(srv.port(), user_space(rx)));
  for (int round = 0; round < 3; ++round) {
    std::vector<std::byte> dst(size);
    job_t j = get_job(harness.ep, size, dst);
    harness.client.start(j);
    REQUIRE(harness.run(j));
    REQUIRE_FALSE(j.ec);
    CHECK(net_test::check_pattern(dst, srv.seed(), 0) == npos);
  }
  CHECK(harness.ctr().connects_ok == 1);  // the TX state follows the server's KeyUpdates
  conn_slot const* s = harness.live_slot();
  REQUIRE(s != nullptr);
  REQUIRE(s->tls != nullptr);
  auto const* st = static_cast<tls_stream const*>(s->tls);
  CHECK(st->counters().key_updates == (requested ? 3U : 0U));  // `key_update` stops: replies
  harness.quiesce();
}

TEST_CASE(
  "net: kTLS prefer falls back to user space against a TLS 1.2 CBC-only server; require "
  "fails",
  "[rest][net][tls][ktls]")
{
  REQUIRE_KTLS();
  net_test::test_server srv({.tls         = true,
                             .min_version = TLS1_2_VERSION,
                             .max_version = TLS1_2_VERSION,
                             .cipher_list = "ECDHE-RSA-AES128-SHA:AES128-SHA"});
  REQUIRE(srv.ok());
  auto const mode = GENERATE(tls::ktls_mode::require, tls::ktls_mode::prefer);
  tls::config tc;
  tc.max_version = tls::version_bound::tls1_2;
  tc.cipher_list = "ECDHE-RSA-AES128-SHA:AES128-SHA";  // neither kTLS nor tier 2 serve CBC
  tc.ktls        = mode;
  fixture harness(make_ep(srv.port(), tc));
  std::vector<std::byte> dst(256 * KiB);
  job_t j = get_job(harness.ep, dst.size(), dst);
  harness.client.start(j);
  REQUIRE(harness.run(j, 10s));
  if (mode == tls::ktls_mode::require) {
    CHECK(j.ec == rest::errc::ktls_unavailable);
    CHECK(harness.ctr().ktls_fallbacks == 0);
  } else {
    REQUIRE_FALSE(j.ec);
    CHECK(net_test::check_pattern(dst, srv.seed(), 0) == npos);
    CHECK(harness.ctr().ktls_fallbacks == 1);  // counted: kTLS was asked for
    conn_slot const* s = harness.live_slot();
    REQUIRE(s != nullptr);
    CHECK(s->kind == kind_tls);
    REQUIRE(s->tls != nullptr);
    CHECK_FALSE(static_cast<tls_stream const*>(s->tls)->record_layer());  // tier 1 serves CBC
  }
  harness.quiesce();
}

TEST_CASE("net: close_notify behind the response: the connection is dropped, the next GET is clean",
          "[rest][net][tls]")
{
  REQUIRE_URING();
  bool const ktls = GENERATE(false, true);
  if (ktls && !net_test::ktls_available()) SKIP("kernel TLS module (tls.ko) is not loaded");
  net_test::test_server srv({.tls = true, .num_tickets = 0, .close_notify_at_end = true});
  REQUIRE(srv.ok());
  fixture harness(
    make_ep(srv.port(), ktls ? tls::config{} : user_space(tls::user_space_rx::record_layer)));
  std::vector<std::byte> dst(512 * KiB);
  for (int i = 0; i < 2; ++i) {
    job_t j = get_job(harness.ep, dst.size(), dst);
    harness.client.start(j);
    REQUIRE(harness.run(j));
    REQUIRE_FALSE(j.ec);
    CHECK(net_test::check_pattern(dst, srv.seed(), 0) == npos);
    // The close_notify is acted on (at check-in or by the idle watch) before the next checkout.
    REQUIRE(
      harness.tl.run_until([&] { return harness.client.pool().idle_count(harness.ep) == 0; }, 5s));
  }
  CHECK(harness.ctr().connects_ok == 2);
  CHECK(harness.client.failed == 0);
  CHECK(harness.ctr().reuse_desync_closes == 0);
  CHECK(harness.ctr().idle_fin_evictions + harness.ctr().close_notify >= 2);
  harness.quiesce();
}

TEST_CASE("net: TLS truncated body (FIN without close_notify) fails the GET", "[rest][net][tls]")
{
  REQUIRE_URING();
  bool const ktls = GENERATE(false, true);
  if (ktls && !net_test::ktls_available()) SKIP("kernel TLS module (tls.ko) is not loaded");
  net_test::test_server srv({.tls = true, .num_tickets = 0, .fin_at = 300 * KiB});
  REQUIRE(srv.ok());
  fixture harness(
    make_ep(srv.port(), ktls ? tls::config{} : user_space(tls::user_space_rx::record_layer)));
  harness.client.free_retry = false;
  std::vector<std::byte> dst(1 * MiB);
  job_t j = get_job(harness.ep, dst.size(), dst);
  harness.client.start(j);
  REQUIRE(harness.run(j, 10s));
  CHECK(j.ec);
  CHECK((j.ec == rest::errc::http_length_mismatch || j.ec == rest::errc::tls_protocol_error));
  CHECK(j.body <= 300 * KiB);
  CHECK(net_test::check_pattern(std::span(dst.data(), j.body), srv.seed(), 0) == npos);
  harness.quiesce();
}

TEST_CASE("net: TLS handshake timeout and certificate verification failure", "[rest][net][tls]")
{
  REQUIRE_URING();
  SECTION("silent server: tls_timeout from the reactor heap")
  {
    net_test::test_server srv({.tls = true, .silent = true});
    REQUIRE(srv.ok());
    endpoint_config e  = make_ep(srv.port());
    e.conn.tls_timeout = 300ms;
    fixture harness(std::move(e));
    std::vector<std::byte> dst(4 * KiB);
    job_t j = get_job(harness.ep, dst.size(), dst);
    harness.client.start(j);
    REQUIRE(harness.run(j, 5s));
    CHECK(j.ec == rest::errc::connect_timeout);
    CHECK(harness.ctr().handshake_timeouts == 1);
    CHECK(net_test::now_ns() - j.t_start >= 280'000'000ULL);
    harness.quiesce();
  }
  SECTION("untrusted certificate: tls_verify_failed")
  {
    net_test::test_server srv({.tls = true, .untrusted = true});
    REQUIRE(srv.ok());
    fixture harness(make_ep(srv.port()));
    std::vector<std::byte> dst(4 * KiB);
    job_t j = get_job(harness.ep, dst.size(), dst);
    harness.client.start(j);
    REQUIRE(harness.run(j, 5s));
    CHECK(j.ec == rest::errc::tls_verify_failed);
    CHECK(harness.ctr().handshake_failures == 1);
    harness.quiesce();
  }
}

TEST_CASE("net: shutdown with TLS GETs in flight reaps every CQE and leaks nothing",
          "[rest][net][tls]")
{
  REQUIRE_URING();
  bool const ktls = GENERATE(false, true);
  if (ktls && !net_test::ktls_available()) SKIP("kernel TLS module (tls.ko) is not loaded");
  unsigned const fds0 = open_fds();
  {
    net_test::test_server srv({.tls = true, .num_tickets = 2, .write_chunk = 4096});
    REQUIRE(srv.ok());
    fixture harness(
      make_ep(srv.port(), ktls ? tls::config{} : user_space(tls::user_space_rx::record_layer)));
    harness.client.pool().warmup(harness.ep, 2);  // idle connections with watches armed
    std::vector<std::vector<std::byte>> dst(4, std::vector<std::byte>(64 * MiB));
    std::vector<job_t> jobs;
    jobs.reserve(4);
    for (auto& d : dst)
      jobs.push_back(get_job(harness.ep, d.size(), d));
    for (auto& j : jobs)
      harness.client.start(j);
    REQUIRE(harness.tl.run_until([&] { return harness.ctr().body_bytes > 8 * MiB; }, 20s));
    harness.quiesce();
    for (auto& j : jobs) {
      CHECK(j.done);
      CHECK((j.ec == rest::errc::context_stopped || !j.ec));
    }
    auto const g = harness.client.pool().gauges(harness.ep);
    CHECK(g.free == g.capacity);
    CHECK(harness.client.pool().ops_held() == 0);
  }
  CHECK(open_fds() <= fds0);  // every socket closed, every TLS object freed with it
}
