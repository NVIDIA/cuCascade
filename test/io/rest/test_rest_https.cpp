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

/// @file test_rest_https.cpp
/// @brief End-to-end HTTPS reads through @c rest_ioctx against the loopback server's TLS mode
/// (test PKI from io/tls/tls_test_pki.hpp): every TLS tier (kTLS installed by cuCascade, the
/// built-in user-space record layer, the library's record layer) on TLS 1.3 and 1.2, HEAD and
/// multi-GET reads byte-exact over reused connections; verification on with the right CA file,
/// off, and on against an unknown CA (fails without a retry storm); a server KeyUpdate in the
/// middle of every body; a body cut without close_notify retried on a new connection.  kTLS
/// cases SKIP when the kernel TLS module is absent (the kTLS decision logic itself is covered with
/// fakes in net/test_net_transport.cpp).

#include "io/tls/ktls.hpp"
#include "rest_contract_test_utils.hpp"

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/io/tls/config.hpp>
#include <cucascade/io/types.hpp>

#include <catch2/catch_all.hpp>
#include <unistd.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using cucascade::io::slice;
using cucascade::io::rest::rest_ioctx;
using cucascade::io::rest::rest_reactor;
using cucascade::io::rest::transport_stats;
using cucascade::test::key_response_script;
using cucascade::test::list_capable_mock_authorizer;
using cucascade::test::loopback_range_server;
using cucascade::test::loopback_tls;
using cucascade::test::range_fault_policy;
using cucascade::test::require_rest_bytes_equal;
using cucascade::test::response_body_policy;
using cucascade::test::rest_contract_config;
using cucascade::test::rest_shutdown_guard;
using cucascade::test::rest_test_payload;
using cucascade::test::scripted_response;
using namespace std::chrono_literals;
namespace tls = cucascade::io::tls;
namespace pki = cucascade::test::tls;

/// A @c rest_ioctx whose reactors' transport counters the test can read.
class probed_ioctx final : public rest_ioctx {
 public:
  using rest_ioctx::rest_ioctx;
  [[nodiscard]] transport_stats reactor_stats(std::size_t i = 0) const
  {
    return _reactors.at(i)->stats();
  }
};

std::shared_ptr<probed_ioctx> make_ioctx(loopback_range_server const& server,
                                         cucascade::io::rest::config cfg)
{
  auto authorizer = std::make_shared<list_capable_mock_authorizer>(server.endpoint());
  auto context    = std::make_shared<rest_reactor::reactor_context>(cfg, authorizer, nullptr);
  auto ioctx      = std::make_shared<probed_ioctx>(1, std::move(context));
  ioctx->start();
  return ioctx;
}

/// The worker publishes its counters once per loop pass: poll until @p ready holds.
template <typename Predicate>
transport_stats wait_for_stats(probed_ioctx const& ioctx,
                               Predicate ready,
                               std::chrono::milliseconds timeout = 5s)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  auto stats          = ioctx.reactor_stats();
  while (!ready(stats) && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(20ms);
    stats = ioctx.reactor_stats();
  }
  return stats;
}

std::vector<std::uint8_t> read_range(probed_ioctx& ioctx,
                                     cucascade::io::io_object const& object,
                                     std::size_t offset,
                                     std::size_t size)
{
  std::vector<std::uint8_t> out(size);
  std::vector<slice> slices{slice{offset, size, out.data()}};
  REQUIRE(std::move(ioctx.host_readv_async_io(object, slices)).get(30s) == size);
  return out;
}

bool ktls_available() { return ::access("/proc/net/tls_stat", R_OK) == 0; }

enum class tier : std::uint8_t { kernel, record_layer, library };

char const* name_of(tier t)
{
  switch (t) {
    case tier::kernel: return "kTLS";
    case tier::record_layer: return "user space (built-in record layer)";
    case tier::library: return "user space (library record layer)";
  }
  return "?";
}

/// HTTPS reads verified against @p ca_path on @p t; GETs of at most 1 MiB, so one read is many.
cucascade::io::rest::config https_config(tier t, tls::version_bound version, std::string ca_path)
{
  auto cfg              = rest_contract_config();
  cfg.tls_verify        = true;
  cfg.ca_bundle_path    = std::move(ca_path);
  cfg.max_connections   = 2;
  cfg.min_segment_bytes = 1UL << 20;
  cfg.max_segment_bytes = 1UL << 20;
  cfg.tls.min_version   = version;
  cfg.tls.max_version   = version;
  cfg.tls.ktls          = t == tier::kernel ? tls::ktls_mode::prefer : tls::ktls_mode::disable;
  cfg.tls.fallback_rx =
    t == tier::library ? tls::user_space_rx::library : tls::user_space_rx::record_layer;
  return cfg;
}

loopback_tls https() { return loopback_tls{.enabled = true}; }

}  // namespace

TEST_CASE("rest_ioctx reads byte-exact over HTTPS on every TLS tier", "[rest][e2e][tls]")
{
  auto const t       = GENERATE(tier::kernel, tier::record_layer, tier::library);
  auto const version = GENERATE(tls::version_bound::tls1_3, tls::version_bound::tls1_2);
  if (t == tier::kernel && !ktls_available()) { SKIP("kernel TLS module (tls.ko) is not loaded"); }
  INFO(name_of(t) << (version == tls::version_bound::tls1_3 ? ", TLS 1.3" : ", TLS 1.2"));
  pki::ca_file const ca;
  auto payload = rest_test_payload((6UL << 20) + 4099);
  loopback_range_server server(payload, {}, {}, {}, https());
  REQUIRE(server.https());
  auto const cfg = https_config(t, version, ca.path());
  auto ioctx     = make_ioctx(server, cfg);
  rest_shutdown_guard cleanup{*ioctx};

  // A HEAD sizes the object over TLS, then whole and odd-sized reads, many GETs each.
  auto object = ioctx->open_io_object("s3://tls-bucket/object.bin");
  REQUIRE(object->size() == payload.size());
  CHECK(server.head_count() == 1);
  auto const whole = read_range(*ioctx, *object, 0, payload.size());
  require_rest_bytes_equal(whole, payload);
  auto const odd = read_range(*ioctx, *object, 12345, 777777);
  require_rest_bytes_equal(odd, {payload.data() + 12345, 777777});
  CHECK(server.get_count() > 6);

  // Connections are reused across GETs: at most one handshake per pooled connection.
  CHECK(server.tls_handshake_count() >= 1);
  CHECK(server.tls_handshake_count() <= cfg.max_connections);
  CHECK(server.tls_handshake_failure_count() == 0);
  auto const exchanges = server.get_count() + server.head_count();
  auto const stats =
    wait_for_stats(*ioctx, [&](transport_stats const& s) { return s.responses_2xx >= exchanges; });
  CHECK(stats.responses_2xx == exchanges);
  CHECK(stats.handshakes == server.tls_handshake_count());
  CHECK(stats.reused == exchanges - stats.handshakes);
  if (t == tier::kernel) {
    CHECK(stats.ktls_sessions == stats.handshakes);
    CHECK(stats.ktls_fallbacks == 0);
  } else {
    // This host's path: the user-space record layer (or the library's) on every connection.
    CHECK(stats.ktls_sessions == 0);
    CHECK(stats.userspace_tls_sessions == stats.handshakes);
    CHECK(stats.ktls_fallbacks == 0);  // kTLS disabled by policy is not a fallback
  }
}

TEST_CASE("rest_ioctx over HTTPS with verification off reads from an untrusted endpoint",
          "[rest][e2e][tls]")
{
  // The default policy (kTLS prefer): on this host's kernel without tls.ko the connection falls
  // back to the user-space record layer and counts it.
  auto payload = rest_test_payload(1UL << 20);
  loopback_range_server server(payload, {}, {}, {}, https());
  auto cfg = rest_contract_config();
  REQUIRE_FALSE(cfg.tls_verify);
  REQUIRE(cfg.ca_bundle_path.empty());
  auto ioctx = make_ioctx(server, cfg);
  rest_shutdown_guard cleanup{*ioctx};
  auto object    = ioctx->open_io_object("s3://tls-bucket/object.bin", payload.size());
  auto const got = read_range(*ioctx, *object, 77, payload.size() - 77);
  require_rest_bytes_equal(got, {payload.data() + 77, payload.size() - 77});
  CHECK(server.tls_handshake_count() >= 1);
  auto const stats = wait_for_stats(*ioctx, [](transport_stats const& s) { return s.gets >= 1; });
  CHECK(stats.ktls_sessions + stats.userspace_tls_sessions == stats.handshakes);
  if (!ktls_available()) { CHECK(stats.ktls_fallbacks == stats.handshakes); }
}

TEST_CASE("rest_ioctx over HTTPS with kTLS required fails without tls.ko", "[rest][e2e][tls]")
{
  if (ktls_available()) { SKIP("the kernel TLS module is loaded: kTLS is installable here"); }
  auto payload = rest_test_payload(64UL << 10);
  loopback_range_server server(payload, {}, {}, {}, https());
  auto cfg               = rest_contract_config();
  cfg.tls.ktls           = tls::ktls_mode::require;
  cfg.max_retry_attempts = 4;
  auto ioctx             = make_ioctx(server, cfg);
  rest_shutdown_guard cleanup{*ioctx};
  auto object = ioctx->open_io_object("s3://tls-bucket/object.bin", payload.size());
  std::vector<std::uint8_t> out(4096);
  CHECK_THROWS_WITH(ioctx->host_read(*object, 0, out.size(), out.data()),
                    Catch::Matchers::ContainsSubstring("kernel TLS"));
  CHECK(server.get_count() == 0);
  // Final, not retried: one handshake (the server may see it fail: the client resets the
  // connection as soon as the kernel refused the keys).
  for (int i = 0;
       i < 100 && server.tls_handshake_count() + server.tls_handshake_failure_count() == 0;
       ++i) {
    std::this_thread::sleep_for(10ms);
  }
  std::this_thread::sleep_for(100ms);
  CHECK(server.tls_handshake_count() + server.tls_handshake_failure_count() == 1);
}

TEST_CASE("rest_ioctx over HTTPS fails an unknown CA once, without a retry storm",
          "[rest][e2e][tls]")
{
  auto const t = GENERATE(tier::kernel, tier::record_layer);
  if (t == tier::kernel && !ktls_available()) { SKIP("kernel TLS module (tls.ko) is not loaded"); }
  INFO(name_of(t));
  auto const other = pki::make_pki();  // a CA that did not sign the server's certificate
  pki::ca_file const ca{other};
  auto payload = rest_test_payload(256UL << 10);
  loopback_range_server server(payload, {}, {}, {}, https());
  auto cfg               = https_config(t, tls::version_bound::tls1_3, ca.path());
  cfg.max_connections    = 1;
  cfg.max_retry_attempts = 4;
  auto ioctx             = make_ioctx(server, cfg);
  rest_shutdown_guard cleanup{*ioctx};
  auto object = ioctx->open_io_object("s3://tls-bucket/object.bin", payload.size());
  std::vector<std::uint8_t> out(4096);
  auto const started = std::chrono::steady_clock::now();
  std::string message;
  try {
    (void)ioctx->host_read(*object, 0, out.size(), out.data());
    FAIL("the read succeeded against an untrusted certificate");
  } catch (std::exception const& error) {
    message = error.what();
  }
  INFO(message);
  CHECK(std::chrono::steady_clock::now() - started < 5s);
  CHECK_THAT(message, Catch::Matchers::ContainsSubstring("verif", Catch::CaseSensitive::No));
  // A verification failure is deterministic: one handshake, no retries, no request sent.
  for (int i = 0; i < 100 && server.tls_handshake_failure_count() == 0; ++i) {
    std::this_thread::sleep_for(10ms);
  }
  std::this_thread::sleep_for(100ms);
  CHECK(server.tls_handshake_count() == 0);
  CHECK(server.tls_handshake_failure_count() == 1);
  CHECK(server.get_count() == 0);
}

TEST_CASE("rest_ioctx over HTTPS survives a server KeyUpdate in the middle of every body",
          "[rest][e2e][tls][key_update]")
{
  auto const t = GENERATE(tier::kernel, tier::record_layer, tier::library);
  if (t == tier::kernel) {
    if (!ktls_available()) { SKIP("kernel TLS module (tls.ko) is not loaded"); }
    if (!cucascade::io::tls::ktls::probe().tls13_rekey) {
      SKIP("the kernel cannot re-key TLS 1.3 kTLS (Linux < 6.14)");
    }
  }
  INFO(name_of(t));
  pki::ca_file const ca;
  auto payload = rest_test_payload((4UL << 20) + 1);
  range_fault_policy fault{};
  fault.body.key_update_after_bytes = 100000;  // inside a record of every gets body
  loopback_range_server server(payload, fault, {}, {}, https());
  auto cfg            = https_config(t, tls::version_bound::tls1_3, ca.path());
  cfg.max_connections = 1;
  auto ioctx          = make_ioctx(server, cfg);
  rest_shutdown_guard cleanup{*ioctx};
  auto object = ioctx->open_io_object("s3://tls-bucket/object.bin", payload.size());

  auto const first = read_range(*ioctx, *object, 0, payload.size());
  require_rest_bytes_equal(first, payload);
  auto const second = read_range(*ioctx, *object, 333, 2UL << 20);
  require_rest_bytes_equal(second, {payload.data() + 333, 2UL << 20});

  CHECK(server.key_updates_sent() == server.get_count());
  // Re-keyed in place: the one connection carried every GET.
  CHECK(server.tls_handshake_count() == 1);
  auto const gets = server.get_count();
  auto const stats =
    wait_for_stats(*ioctx, [&](transport_stats const& s) { return s.responses_2xx >= gets; });
  CHECK(stats.retries == 0);
  CHECK(stats.reused == gets - 1);  // one connection carried every GET
  if (t == tier::kernel) {
    CHECK(stats.ktls_sessions == 1);
    CHECK(stats.control_records >= gets);
  }
}

TEST_CASE("rest_ioctx over HTTPS retries a body cut without close_notify", "[rest][e2e][tls]")
{
  auto const t = GENERATE(tier::kernel, tier::record_layer);
  if (t == tier::kernel && !ktls_available()) { SKIP("kernel TLS module (tls.ko) is not loaded"); }
  INFO(name_of(t));
  pki::ca_file const ca;
  auto payload               = rest_test_payload(2UL << 20);
  constexpr std::size_t cut  = 200000;
  constexpr std::size_t size = 1UL << 20;
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["cut.bin"].gets = {
    scripted_response{.body = response_body_policy{.drop_after_bytes = cut}}, scripted_response{}};
  loopback_range_server server(payload, {}, {}, scripts, https());
  auto cfg               = https_config(t, tls::version_bound::tls1_3, ca.path());
  cfg.max_connections    = 1;
  cfg.max_segment_bytes  = 32UL << 20;
  cfg.max_retry_attempts = 3;
  auto ioctx             = make_ioctx(server, cfg);
  rest_shutdown_guard cleanup{*ioctx};
  auto object = ioctx->open_io_object("s3://tls-bucket/cut.bin", payload.size());

  auto const got = read_range(*ioctx, *object, 1000, size);
  require_rest_bytes_equal(got, {payload.data() + 1000, size});
  CHECK(server.body_drops() == 1);
  CHECK(server.get_count("cut.bin") == 2);
  // The truncated connection is not reused: the retry handshakes anew.
  CHECK(server.tls_handshake_count() == 2);
  auto const requests = server.get_requests();
  REQUIRE(requests.size() == 2);
  REQUIRE(requests[1].ranges.size() == 1);
  // The whole range again (no resume): the cut response's bytes are not trusted.
  CHECK(requests[1].ranges.front().starts_with("bytes=1000-"));
  auto const stats =
    wait_for_stats(*ioctx, [](transport_stats const& s) { return s.retries >= 1; });
  CHECK(stats.retries == 1);
}

TEST_CASE("rest_ioctx loopback HTTPS GET throughput (user-space TLS, 1 GiB)",
          "[.][rest_throughput]")
{
  // A smoke number, not a test: 64 x 16 MiB ranged GETs over loopback HTTPS on the user-space
  // record layer (this host has no tls.ko), 2 reactors x 16 connections.  The loopback server
  // encrypts with one OpenSSL thread per connection, so the number bounds both ends.
  constexpr std::size_t object_bytes = 16UL << 20;
  constexpr std::size_t reads        = 64;
  auto payload                       = rest_test_payload(object_bytes);
  loopback_range_server server(payload, {}, {}, {}, https());
  auto cfg              = rest_contract_config();
  cfg.max_connections   = 16;
  cfg.min_segment_bytes = 16UL << 20;
  cfg.max_segment_bytes = 16UL << 20;
  cfg.tls.ktls          = tls::ktls_mode::prefer;
  auto authorizer       = std::make_shared<list_capable_mock_authorizer>(server.endpoint());
  auto context          = std::make_shared<rest_reactor::reactor_context>(cfg, authorizer, nullptr);
  auto ioctx            = std::make_shared<probed_ioctx>(2, std::move(context));
  ioctx->start();
  rest_shutdown_guard cleanup{*ioctx};
  auto object = ioctx->open_io_object("s3://bench-bucket/object.bin", payload.size());
  std::vector<std::uint8_t> out(reads * object_bytes);
  // Warm the pool (handshakes are not part of the number).
  ioctx->warmup("s3://bench-bucket");
  std::this_thread::sleep_for(500ms);
  auto const started = std::chrono::steady_clock::now();
  std::vector<cucascade::exec::semi_future<std::size_t>> futures;
  futures.reserve(reads);
  for (std::size_t i = 0; i < reads; ++i) {
    futures.push_back(
      ioctx->host_read_async_io(*object, 0, object_bytes, out.data() + i * object_bytes));
  }
  for (auto& f : futures) {
    REQUIRE(std::move(f).get(120s) == object_bytes);
  }
  auto const seconds =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  require_rest_bytes_equal({out.data() + (reads - 1) * object_bytes, object_bytes}, payload);
  auto const gib   = static_cast<double>(reads * object_bytes) / static_cast<double>(1UL << 30);
  auto const stats = ioctx->stats();
  std::printf(
    "rest loopback HTTPS: %.3f GiB in %.3f s = %.2f GiB/s (gets %llu, user-space TLS sessions "
    "%llu, kTLS sessions %llu)\n",
    gib,
    seconds,
    gib / seconds,
    static_cast<unsigned long long>(stats.gets),
    static_cast<unsigned long long>(stats.userspace_tls_sessions),
    static_cast<unsigned long long>(stats.ktls_sessions));
  CHECK(stats.gets >= reads);
}
