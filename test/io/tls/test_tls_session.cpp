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
 * @file test_tls_session.cpp
 * @brief The OpenSSL `tls::session` against a loopback TLS server: handshakes (TLS 1.2 / 1.3,
 * AES-128/256-GCM), verification (CA file and blob, wrong host, unknown CA, IP literals, verify
 * off), key export against the server's view, user-space tiers 2 (built-in opener + library
 * sealer) and 1 (library opener and sealer after `detach_from_socket`) with KeyUpdates, tickets,
 * close_notify and tampered records, the SIGPIPE guard, session resumption and SSLKEYLOGFILE.
 * Ported in part from gpuio `tests/unit/tls/test_handshake_local.cpp`.
 */

#include "io/tls/key_schedule.hpp"
#include "io/tls/openssl_internal.hpp"
#include "io/tls/record_opener.hpp"
#include "tls_test_pki.hpp"
#include "tls_test_server.hpp"

#include <cucascade/io/tls/tls.hpp>

#include <catch2/catch_all.hpp>
#include <openssl/ssl.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace cucascade::io::tls;
namespace t    = cucascade::test::tls;
namespace ossl = cucascade::io::tls::openssl;

namespace {

config trusting(ktls_mode k = ktls_mode::disable)
{
  config c;
  c.ca_blob = t::default_pki().ca.cert_pem;
  c.ktls    = k;
  return c;
}

t::server_ctx_options server_version(int v)
{
  t::server_ctx_options o;
  o.min_version = v;
  o.max_version = v;
  return o;
}

/// A client connection: a non-blocking socket and a session on it.
struct client {
  client(provider const& p,
         std::uint16_t port,
         std::string_view host = "localhost",
         bool ipv6             = false)
    : fd(t::connect_loopback(port, ipv6))
  {
    REQUIRE(fd.get() >= 0);
    std::error_code ec;
    s = p.new_session(fd.get(), {host, port}, ec);
    REQUIRE(s != nullptr);
    CHECK_FALSE(ec);
  }
  std::error_code handshake() { return t::drive_handshake(*s, fd.get()); }
  std::string error_text() const
  {
    std::array<char, 256> b{};
    s->describe_error(b);
    return b.data();
  }

  t::unique_fd fd;
  std::unique_ptr<session> s;
};

bool same_hex(std::string const& hex, traffic_keys const& k)
{
  static constexpr char digits[] = "0123456789abcdef";
  std::string h;
  for (std::size_t i = 0; i < k.secret_len; ++i) {
    h.push_back(digits[k.secret[i] >> 4]);
    h.push_back(digits[k.secret[i] & 0xF]);
  }
  return !hex.empty() && h == hex;
}

/// User-space TLS on a finished session: tier 2 (built-in opener) or tier 1 (library opener).
struct user_tls {
  user_tls(provider const& p, client& c, bool tier2)
  {
    if (tier2) {
      traffic_keys rx;
      REQUIRE_FALSE(c.s->export_keys(direction::receive, rx));
      std::error_code ec;
      own = make_record_opener(std::move(rx), p, ec);
      REQUIRE(own != nullptr);
    }
    REQUIRE_FALSE(c.s->detach_from_socket());
    REQUIRE_FALSE(c.s->detach_from_socket());  // idempotent
    sealer = c.s->sealer();
    REQUIRE(sealer != nullptr);
    rx.fd     = c.fd.get();
    rx.opener = tier2 ? own.get() : c.s->opener();
    REQUIRE(rx.opener != nullptr);
    rx.on_key_update = [this] { return !sealer->request_key_update_reply(); };
  }
  std::unique_ptr<record_opener> own;
  record_sealer* sealer{nullptr};
  t::user_rx rx;
};

}  // namespace

TEST_CASE("tls session: TLS 1.2 and 1.3 handshakes with AES-GCM, verify on", "[tls]")
{
  struct hs_case {
    char const* name;
    int version;
    char const* suites;
    std::uint16_t suite;
    aead_cipher cipher;
    bool ca_file;
  };
  for (hs_case const& hc : {hs_case{"1.3 AES-128",
                                    TLS1_3_VERSION,
                                    "TLS_AES_128_GCM_SHA256",
                                    0x1301,
                                    aead_cipher::aes_128_gcm,
                                    false},
                            hs_case{"1.3 AES-256",
                                    TLS1_3_VERSION,
                                    "TLS_AES_256_GCM_SHA384",
                                    0x1302,
                                    aead_cipher::aes_256_gcm,
                                    true},
                            hs_case{"1.2 AES-128",
                                    TLS1_2_VERSION,
                                    "ECDHE-ECDSA-AES128-GCM-SHA256",
                                    0xC02B,
                                    aead_cipher::aes_128_gcm,
                                    true},
                            hs_case{"1.2 AES-256",
                                    TLS1_2_VERSION,
                                    "ECDHE-ECDSA-AES256-GCM-SHA384",
                                    0xC02C,
                                    aead_cipher::aes_256_gcm,
                                    false}}) {
    INFO(hc.name);
    t::server_ctx_options so = server_version(hc.version);
    if (hc.version == TLS1_3_VERSION) {
      so.ciphersuites = hc.suites;
    } else {
      so.cipher_list = hc.suites;
    }
    std::atomic<bool> sni_ok{false};
    t::loopback_server srv(t::make_server_ctx(t::default_pki(), so), [&](t::server_conn& c) {
      char const* sni = SSL_get_servername(c.ssl(), TLSEXT_NAMETYPE_host_name);
      sni_ok          = sni != nullptr && std::string_view{sni} == "localhost";
      c.wait_peer_close();
    });
    std::optional<t::ca_file> file;
    config cfg = trusting();
    if (hc.ca_file) {
      file.emplace();
      cfg.ca_blob.clear();
      cfg.ca_file = file->path();
    }
    auto const p = make_openssl_provider(cfg);
    {
      client c(*p, srv.port(), "LocalHost.");  // normalised: lower-cased, trailing dot dropped
      REQUIRE_FALSE(c.handshake());
      session_info const i = c.s->info();
      CHECK(i.version ==
            (hc.version == TLS1_3_VERSION ? protocol_version::tls1_3 : protocol_version::tls1_2));
      CHECK(i.cipher_suite == hc.suite);
      CHECK(i.cipher == hc.cipher);
      CHECK(i.alpn_http11);
      CHECK_FALSE(i.resumed);
      CHECK(i.buffered_bytes == 0);
      std::error_code ec;
      CHECK(c.s->handshake(ec) == handshake_status::done);  // idempotent once done
    }
    srv.join();
    CHECK(sni_ok);
    CHECK(srv.accepted() == 1);
  }
}

TEST_CASE("tls session: verification failures, IP literals and verify off", "[tls]")
{
  std::atomic<int> sni_seen{0};
  t::loopback_server srv4(t::make_server_ctx(t::default_pki()), [&](t::server_conn& c) {
    if (SSL_get_servername(c.ssl(), TLSEXT_NAMETYPE_host_name) != nullptr) { ++sni_seen; }
    c.wait_peer_close();
  });
  auto const p = make_openssl_provider(trusting());
  SECTION("wrong host name")
  {
    client c(*p, srv4.port(), "other.example");
    CHECK(c.handshake() == make_error_code(errc::verify_failed));
    CHECK(c.error_text().find("verify") != std::string::npos);
  }
  SECTION("unknown CA")
  {
    config cfg   = trusting();
    cfg.ca_blob  = t::make_pki().ca.cert_pem;  // a different, unrelated CA
    auto const q = make_openssl_provider(cfg);
    client c(*q, srv4.port());
    CHECK(c.handshake() == make_error_code(errc::verify_failed));
    CHECK_FALSE(c.error_text().empty());
  }
  SECTION("verify off accepts an unknown CA (development endpoints)")
  {
    config cfg   = trusting();
    cfg.verify   = false;
    cfg.ca_blob  = t::make_pki().ca.cert_pem;
    auto const q = make_openssl_provider(cfg);
    client c(*q, srv4.port(), "whatever.example");
    CHECK_FALSE(c.handshake());
  }
  SECTION("IP literal: verified against the IP SAN, no SNI sent")
  {
    client c(*p, srv4.port(), "127.0.0.1");
    CHECK_FALSE(c.handshake());
    client wrong(*p, srv4.port(), "127.0.0.2");
    CHECK(wrong.handshake() == make_error_code(errc::verify_failed));
  }
  SECTION("sni_override replaces the host for SNI and identity")
  {
    config cfg       = trusting();
    cfg.sni_override = "localhost";
    auto const q     = make_openssl_provider(cfg);
    client c(*q, srv4.port(), "bogus.example");
    CHECK_FALSE(c.handshake());
  }
  SECTION("IPv6 literal with brackets")
  {
    std::unique_ptr<t::loopback_server> srv6;
    try {
      srv6 = std::make_unique<t::loopback_server>(
        t::make_server_ctx(t::default_pki()), [](t::server_conn& c) { c.wait_peer_close(); }, true);
    } catch (std::exception const&) {
      SKIP("no IPv6 loopback");
    }
    client c(*p, srv6->port(), "[::1]", true);
    CHECK_FALSE(c.handshake());
  }
  SECTION("an empty or overlong host is refused before any I/O")
  {
    t::unique_fd fd(t::connect_loopback(srv4.port()));
    std::error_code ec;
    CHECK(p->new_session(fd.get(), {"", 443}, ec) == nullptr);
    CHECK(ec == std::make_error_code(std::errc::invalid_argument));
    CHECK(p->new_session(fd.get(), {std::string(300, 'a'), 443}, ec) == nullptr);
    CHECK(p->new_session(-1, {"localhost", 443}, ec) == nullptr);
  }
}

TEST_CASE("tls session: exported keys match the server's view", "[tls]")
{
  SECTION("TLS 1.3: both traffic secrets with kTLS possible, wiped once exported")
  {
    t::loopback_server srv(t::make_server_ctx(t::default_pki(), server_version(TLS1_3_VERSION)),
                           [](t::server_conn& c) { c.wait_peer_close(); });
    auto const p = make_openssl_provider(trusting(ktls_mode::prefer));
    client c(*p, srv.port());
    REQUIRE_FALSE(c.handshake());
    traffic_keys rx;
    traffic_keys tx;
    REQUIRE_FALSE(c.s->export_keys(direction::receive, rx));
    REQUIRE_FALSE(c.s->export_keys(direction::send, tx));
    // The server logs the client secret once it processed our Finished (SSL_accept returned).
    for (int i = 0; i < 500 && srv.accepted() == 0; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    REQUIRE(srv.accepted() == 1);
    CHECK(same_hex(srv.keylog_secret("SERVER_TRAFFIC_SECRET_0"), rx));
    CHECK(same_hex(srv.keylog_secret("CLIENT_TRAFFIC_SECRET_0"), tx));
    for (traffic_keys const* k : {&rx, &tx}) {
      CHECK(k->version == protocol_version::tls1_3);
      CHECK(k->cipher_suite == 0x1301);
      CHECK(k->cipher == aead_cipher::aes_128_gcm);
      CHECK(k->key_len == 16);
      CHECK(k->iv_len == 12);
      CHECK(k->secret_len == 32);
      CHECK(k->sequence == 0);
    }
    // The key and IV are the RFC 8446 §7.3 derivation of the secret.
    std::array<std::uint8_t, 16> key{};
    std::array<std::uint8_t, 12> iv{};
    REQUIRE_FALSE(tls13_traffic_keys(p->kdf(),
                                     suite_of(protocol_version::tls1_3, 0x1301),
                                     std::span<std::uint8_t const>(rx.secret.data(), 32),
                                     key,
                                     iv));
    CHECK(std::equal(key.begin(), key.end(), rx.key.begin()));
    CHECK(iv == rx.iv);
    traffic_keys again;
    CHECK(c.s->export_keys(direction::receive, again) == make_error_code(errc::not_supported));
  }
  SECTION("TLS 1.3 user space only: the client secret is never captured (F6)")
  {
    t::loopback_server srv(t::make_server_ctx(t::default_pki(), server_version(TLS1_3_VERSION)),
                           [](t::server_conn& c) { c.wait_peer_close(); });
    auto const p = make_openssl_provider(trusting(ktls_mode::disable));
    client c(*p, srv.port());
    REQUIRE_FALSE(c.handshake());
    traffic_keys k;
    CHECK(c.s->export_keys(direction::send, k) == make_error_code(errc::not_supported));
    CHECK(k.key_len == 0);
    CHECK_FALSE(c.s->export_keys(direction::receive, k));
    config lib      = trusting(ktls_mode::disable);
    lib.fallback_rx = user_space_rx::library;  // tier 1 only: no capture at all
    auto const q    = make_openssl_provider(lib);
    client d(*q, srv.port());
    REQUIRE_FALSE(d.handshake());
    CHECK(d.s->export_keys(direction::receive, k) == make_error_code(errc::not_supported));
  }
  SECTION("TLS 1.2: the key block of the master secret, sequence 1, both directions")
  {
    t::loopback_server srv(t::make_server_ctx(t::default_pki(), server_version(TLS1_2_VERSION)),
                           [](t::server_conn& c) { c.wait_peer_close(); });
    auto const p = make_openssl_provider(trusting(ktls_mode::prefer));
    client c(*p, srv.port());
    REQUIRE_FALSE(c.handshake());
    traffic_keys rx;
    traffic_keys tx;
    REQUIRE_FALSE(c.s->export_keys(direction::receive, rx));
    REQUIRE_FALSE(c.s->export_keys(direction::send, tx));
    CHECK(rx.version == protocol_version::tls1_2);
    CHECK(rx.sequence == 1);
    CHECK(tx.sequence == 1);
    CHECK(rx.iv_len == 4);
    CHECK(rx.secret_len == 0);
    // Server view: the same block from the server's master secret and the hello randoms.
    std::vector<std::uint8_t> const master = srv.master_key();
    REQUIRE(master.size() == 48);
    auto const cr = srv.client_random();
    auto const sr = srv.server_random();
    tls12_key_block kb;
    REQUIRE_FALSE(tls12_derive_key_block(
      p->kdf(), suite_of(protocol_version::tls1_2, rx.cipher_suite), master, cr, sr, kb));
    CHECK(rx.key_len == kb.key_len);
    CHECK(std::equal(rx.key.begin(), rx.key.begin() + rx.key_len, kb.server_write_key.begin()));
    CHECK(std::equal(tx.key.begin(), tx.key.begin() + tx.key_len, kb.client_write_key.begin()));
    CHECK(std::equal(rx.iv.begin(), rx.iv.begin() + 4, kb.server_write_iv.begin()));
    CHECK(std::equal(tx.iv.begin(), tx.iv.begin() + 4, kb.client_write_iv.begin()));
    // Exporting TLS 1.2 keys again works (nothing captured); after detach it does not.
    traffic_keys again;
    REQUIRE_FALSE(c.s->export_keys(direction::receive, again));
    CHECK(again.key == rx.key);
    REQUIRE_FALSE(c.s->detach_from_socket());
    CHECK(c.s->export_keys(direction::receive, again) == make_error_code(errc::not_supported));
  }
}

TEST_CASE("tls session: user-space tiers round trip with KeyUpdate, tickets and close_notify",
          "[tls]")
{
  for (bool const tier2 : {true, false}) {
    for (int const version : {TLS1_3_VERSION, TLS1_2_VERSION}) {
      bool const tls13 = version == TLS1_3_VERSION;
      INFO((tier2 ? "tier 2 " : "tier 1 ") << (tls13 ? "TLS 1.3" : "TLS 1.2"));
      std::atomic<bool> request_ok{false};
      std::atomic<bool> second_ok{false};
      std::atomic<unsigned> key_updates_at_server{0};
      t::loopback_server srv(
        t::make_server_ctx(t::default_pki(), server_version(version)), [&](t::server_conn& c) {
          request_ok = c.expect_pattern(1000, 1);
          if (!request_ok) { return; }
          c.send_pattern(100'000, 2);
          if (tls13) {
            c.new_session_ticket();  // a ticket mid-stream
            c.key_update(true);      // the client owes a reply before its next record
          }
          c.send_pattern(100'000, 2, 100'000);
          second_ok             = c.expect_pattern(70'000, 3);
          key_updates_at_server = c.key_updates_received();
          c.send_pattern(5'000, 4);
          c.close_notify();
          c.wait_peer_close();
        });
      auto const p = make_openssl_provider(trusting(ktls_mode::disable));
      {
        client c(*p, srv.port());
        REQUIRE_FALSE(c.handshake());
        user_tls u(*p, c, tier2);
        REQUIRE(t::send_sealed(c.fd.get(), *u.sealer, t::pattern(1000, 1)));
        std::vector<std::byte> got(200'000);
        REQUIRE(u.rx.receive(got, 7'000));
        CHECK(t::check_pattern(got, 2) == std::string_view::npos);
        CHECK(u.rx.key_updates == (tls13 ? 1U : 0U));
        // The owed KeyUpdate reply goes out ahead of this request (RFC 8446 §4.6.3).
        REQUIRE(t::send_sealed(c.fd.get(), *u.sealer, t::pattern(70'000, 3), 2));
        std::vector<std::byte> tail(5'000);
        REQUIRE(u.rx.receive(tail));
        CHECK(t::check_pattern(tail, 4) == std::string_view::npos);
        CHECK(u.rx.wait_closed());  // close_notify
        CHECK(u.rx.closed);
        auto const r = u.rx.opener->open(u.rx.ring.view(), {}, 0, false);
        CHECK(r.stop == open_stop::closed);  // sticky
        CHECK(u.rx.opener->pending() == 0);
        if (tier2) {
          auto const& b = dynamic_cast<builtin_record_opener const&>(*u.rx.opener);
          CHECK(b.counters().tickets == (tls13 ? 3U : 0U));  // 2 after the handshake + 1
          CHECK(b.counters().key_updates == (tls13 ? 1U : 0U));
        }
      }
      srv.join();
      CHECK(request_ok);
      CHECK(second_ok);
      CHECK(key_updates_at_server == (tls13 ? 1U : 0U));
    }
  }
}

TEST_CASE("tls session: a tampered record fails both user-space tiers", "[tls]")
{
  for (bool const tier2 : {true, false}) {
    INFO((tier2 ? "tier 2" : "tier 1"));
    t::loopback_server srv(t::make_server_ctx(t::default_pki()), [&](t::server_conn& c) {
      c.send_pattern(3000, 5);
      c.send_tampered(t::pattern(4000, 6), 100);  // inside the encrypted body
      c.wait_peer_close();
    });
    auto const p = make_openssl_provider(trusting(ktls_mode::disable));
    {
      client c(*p, srv.port());
      REQUIRE_FALSE(c.handshake());
      user_tls u(*p, c, tier2);
      std::vector<std::byte> got(3000);
      REQUIRE(u.rx.receive(got));
      CHECK(t::check_pattern(got, 5) == std::string_view::npos);
      std::vector<std::byte> bad(4000, std::byte{0x5A});
      CHECK_FALSE(u.rx.receive(bad));
      CHECK(u.rx.failed);
      CHECK(u.rx.opener->error() == make_error_code(errc::protocol_error));
      // No forged plaintext stays in the window: tier 2 decrypted the record straight into it and
      // wiped it (F4); tier 1 never wrote unverified bytes there.
      std::byte const left_over = tier2 ? std::byte{0} : std::byte{0x5A};
      CHECK(std::all_of(bad.begin(), bad.end(), [&](std::byte b) { return b == left_over; }));
      auto const r = u.rx.opener->open(u.rx.ring.view(), {}, 0, false);
      CHECK(r.stop == open_stop::failed);  // sticky
    }
  }
}

TEST_CASE("tls session: tier 1 KeyUpdate storm drains the library's control output", "[tls]")
{
  constexpr int updates = 120;
  t::loopback_server srv(t::make_server_ctx(t::default_pki(), server_version(TLS1_3_VERSION)),
                         [&](t::server_conn& c) {
                           for (int i = 0; i < updates; ++i) {
                             c.key_update(true);
                             c.send_pattern(1, 7, static_cast<std::uint64_t>(i));
                           }
                           c.send_pattern(10, 8);
                           c.wait_peer_close();
                         });
  auto const p = make_openssl_provider(trusting(ktls_mode::disable));
  {
    client c(*p, srv.port());
    REQUIRE_FALSE(c.handshake());
    user_tls u(*p, c, false);
    unsigned flushes = 0;
    // Answer every stop by sending whatever the library owes right away.
    u.rx.on_key_update = [&] {
      ++flushes;
      return !u.sealer->request_key_update_reply() && t::send_sealed(c.fd.get(), *u.sealer, {});
    };
    std::vector<std::byte> got(updates);
    REQUIRE(u.rx.receive(got));
    CHECK(t::check_pattern(got, 7) == std::string_view::npos);
    std::vector<std::byte> tail(10);
    REQUIRE(u.rx.receive(tail));
    CHECK(t::check_pattern(tail, 8) == std::string_view::npos);
    CHECK(flushes >= 1);
    CHECK_FALSE(u.rx.failed);
  }
}

TEST_CASE("tls session: tier 1 consumes tickets and feeds the session cache (TLS 1.3 resumption)",
          "[tls]")
{
  config cfg        = trusting(ktls_mode::disable);
  cfg.fallback_rx   = user_space_rx::library;
  cfg.session_cache = true;
  auto const p      = make_openssl_provider(cfg);
  t::loopback_server srv(t::make_server_ctx(t::default_pki(), server_version(TLS1_3_VERSION)),
                         [](t::server_conn& c) {
                           c.send_pattern(100, 1);
                           c.wait_peer_close();
                         });
  {
    client c(*p, srv.port());
    REQUIRE_FALSE(c.handshake());
    CHECK_FALSE(c.s->info().resumed);
    user_tls u(*p, c, false);
    std::vector<std::byte> got(100);
    REQUIRE(u.rx.receive(got));  // the two tickets come first and reach the cache
  }
  {
    client c(*p, srv.port());
    REQUIRE_FALSE(c.handshake());
    CHECK(c.s->info().resumed);
  }
}

TEST_CASE("tls session: TLS 1.2 session resumed on the next connection (T12 opt-in)", "[tls]")
{
  config cfg        = trusting(ktls_mode::disable);
  cfg.max_version   = version_bound::tls1_2;
  cfg.session_cache = true;
  auto const p      = make_openssl_provider(cfg);
  t::loopback_server srv(t::make_server_ctx(t::default_pki(), server_version(TLS1_2_VERSION)),
                         [](t::server_conn& c) { c.wait_peer_close(); });
  {
    client first(*p, srv.port());
    REQUIRE_FALSE(first.handshake());
    CHECK_FALSE(first.s->info().resumed);
  }
  {
    client second(*p, srv.port());
    REQUIRE_FALSE(second.handshake());
    CHECK(second.s->info().resumed);
  }
  {
    client other(*p, srv.port(), "127.0.0.1");  // different endpoint key: no session
    REQUIRE_FALSE(other.handshake());
    CHECK_FALSE(other.s->info().resumed);
  }
  // Off by default: never resumed.
  auto const q = make_openssl_provider(trusting(ktls_mode::disable));
  for (int i = 0; i < 2; ++i) {
    client c(*q, srv.port());
    REQUIRE_FALSE(c.handshake());
    CHECK_FALSE(c.s->info().resumed);
  }
}

TEST_CASE("tls session: a peer reset never raises SIGPIPE in the caller (T14)", "[tls]")
{
  // OpenSSL's socket BIO writes with write(2).  In a child with the default SIGPIPE disposition,
  // a handshake towards a closed peer must fail with an error code, not kill the process.
  auto const p    = make_openssl_provider(trusting());
  pid_t const pid = ::fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {
    ::signal(SIGPIPE, SIG_DFL);
    int sv[2] = {-1, -1};
    if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0, sv) != 0) { ::_exit(2); }
    ::close(sv[1]);  // peer gone: the ClientHello write gets EPIPE (+ SIGPIPE without the guard)
    std::error_code ec;
    auto s = p->new_session(sv[0], {"localhost", 443}, ec);
    if (s == nullptr) { ::_exit(4); }
    auto const st = s->handshake(ec);
    bool const ok = st == handshake_status::failed && ec == make_error_code(errc::handshake_failed);
    ::_exit(ok ? 0 : 3);
  }
  int status = 0;
  REQUIRE(::waitpid(pid, &status, 0) == pid);
  CHECK_FALSE(WIFSIGNALED(status));
  REQUIRE(WIFEXITED(status));
  CHECK(WEXITSTATUS(status) == 0);
}

TEST_CASE("tls session: SSLKEYLOGFILE receives NSS key log lines only on opt-in", "[tls]")
{
  t::temp_dir dir("keylog");
  std::string const path = (dir.path() / "keys.log").string();
  char const* old        = std::getenv("SSLKEYLOGFILE");
  std::optional<std::string> saved;
  if (old != nullptr) { saved = old; }
  ::setenv("SSLKEYLOGFILE", path.c_str(), 1);
  t::loopback_server srv(t::make_server_ctx(t::default_pki(), server_version(TLS1_3_VERSION)),
                         [](t::server_conn& c) { c.wait_peer_close(); });
  {
    auto const p = make_openssl_provider(trusting());  // default: no key log written
    client c(*p, srv.port());
    REQUIRE_FALSE(c.handshake());
  }
  CHECK_FALSE(std::filesystem::exists(path));
  {
    config cfg     = trusting();
    cfg.keylog_env = true;
    auto const p   = make_openssl_provider(cfg);
    client c(*p, srv.port());
    REQUIRE_FALSE(c.handshake());
  }
  if (saved) {
    ::setenv("SSLKEYLOGFILE", saved->c_str(), 1);
  } else {
    ::unsetenv("SSLKEYLOGFILE");
  }
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  CHECK(ss.str().find("CLIENT_TRAFFIC_SECRET_0 ") != std::string::npos);
  CHECK(ss.str().find("SERVER_HANDSHAKE_TRAFFIC_SECRET ") != std::string::npos);
}

TEST_CASE("tls session: misuse before the handshake is refused", "[tls]")
{
  t::loopback_server srv(t::make_server_ctx(t::default_pki()),
                         [](t::server_conn& c) { c.wait_peer_close(); });
  auto const p = make_openssl_provider(trusting());
  client c(*p, srv.port());
  traffic_keys k;
  CHECK(c.s->export_keys(direction::receive, k) ==
        std::make_error_code(std::errc::invalid_argument));
  CHECK(c.s->detach_from_socket() == std::make_error_code(std::errc::invalid_argument));
  CHECK(c.s->opener() == nullptr);
  CHECK(c.s->sealer() == nullptr);
  CHECK(c.s->info().version == protocol_version::none);
  REQUIRE_FALSE(c.handshake());
  REQUIRE_FALSE(c.s->detach_from_socket());
  REQUIRE(c.s->sealer() != nullptr);
  CHECK(c.s->sealer()->record_overhead() >= 22);
}
