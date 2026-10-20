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
 * @file test_ktls.cpp
 * @brief kTLS installed by cuCascade from `session::export_keys` (src/io/tls/ktls.cpp) against
 * a loopback OpenSSL server: plaintext `send`/`recv` after `install` (TLS 1.2 and 1.3,
 * AES-128/256-GCM, NO_PAD), control records drained with `recvmsg` + `TLS_GET_RECORD_TYPE` and
 * handled by `on_control_record` (tickets skipped, KeyUpdate re-key of RX and, when requested,
 * our reply + TX re-key through `flush_reply`, close_notify), a tampered record, and the control
 * record rules in isolation.  Tests that need the kernel module SKIP when `install` reports the
 * kernel refused (tls.ko missing, TCP_ULP blocked).
 */

#include "io/tls/key_schedule.hpp"
#include "io/tls/ktls.hpp"
#include "tls_test_pki.hpp"
#include "tls_test_server.hpp"

#include <cucascade/io/tls/tls.hpp>

#include <catch2/catch_all.hpp>
#include <linux/tls.h>
#include <openssl/ssl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#ifndef SOL_TLS
#define SOL_TLS 282
#endif

using namespace cucascade::io::tls;
namespace t = cucascade::test::tls;

namespace {

config trusting(ktls_mode k = ktls_mode::prefer)
{
  config c;
  c.ca_blob = t::default_pki().ca.cert_pem;
  c.ktls    = k;
  return c;
}

t::server_ctx_options server_options(int version, std::string const& suite)
{
  t::server_ctx_options o;
  o.min_version = version;
  o.max_version = version;
  if (version == TLS1_3_VERSION) {
    o.ciphersuites = suite;
  } else {
    o.cipher_list = suite;
  }
  return o;
}

std::map<std::string, long> read_tls_stat()
{
  std::map<std::string, long> out;
  std::ifstream f("/proc/net/tls_stat");
  std::string key;
  long value = 0;
  while (f >> key >> value) {
    out[key] = value;
  }
  return out;
}

long delta(std::map<std::string, long> const& before,
           std::map<std::string, long> const& after,
           std::string const& key)
{
  auto const a = after.find(key);
  auto const b = before.find(key);
  return (a != after.end() ? a->second : 0) - (b != before.end() ? b->second : 0);
}

/// The client side of a kTLS connection, after `install`: plaintext send/recv, control records
/// drained with recvmsg + TLS_GET_RECORD_TYPE (what the transport does on -EIO).
struct ktls_client {
  int fd{-1};
  ktls::rekey_state* state{nullptr};
  unsigned controls{0};
  unsigned replies{0};
  bool closed{false};
  int last_errno{0};
  std::error_code error{};

  bool drain_control(int timeout_ms)
  {
    std::array<std::byte, 16384 + 256> buf{};
    for (;;) {
      iovec iov{buf.data(), buf.size()};
      alignas(cmsghdr) std::array<char, CMSG_SPACE(sizeof(unsigned char))> cbuf{};
      msghdr msg{};
      msg.msg_iov        = &iov;
      msg.msg_iovlen     = 1;
      msg.msg_control    = cbuf.data();
      msg.msg_controllen = cbuf.size();
      ssize_t const n    = ::recvmsg(fd, &msg, 0);
      if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        if (!t::wait_fd(fd, POLLIN, timeout_ms)) { return false; }
        continue;
      }
      if (n <= 0) {
        last_errno = n < 0 ? errno : 0;
        return false;
      }
      std::uint8_t type = ktls::record_data;
      for (cmsghdr* c = CMSG_FIRSTHDR(&msg); c != nullptr; c = CMSG_NXTHDR(&msg, c)) {
        if (c->cmsg_level == SOL_TLS && c->cmsg_type == TLS_GET_RECORD_TYPE) {
          type = *reinterpret_cast<unsigned char*>(CMSG_DATA(c));
        }
      }
      if ((msg.msg_flags & MSG_EOR) == 0) { return false; }  // larger than the buffer
      ++controls;
      auto const a =
        ktls::on_control_record(fd,
                                state,
                                type,
                                std::span<std::byte const>(buf.data(), static_cast<std::size_t>(n)),
                                error);
      switch (a) {
        case ktls::control_action::consumed: return true;
        case ktls::control_action::closed: closed = true; return true;
        case ktls::control_action::failed: return false;
        case ktls::control_action::reply_owed: {
          for (;;) {
            auto const e = ktls::flush_reply(fd, state);
            if (e == std::errc::operation_would_block) {
              if (!t::wait_fd(fd, POLLOUT, timeout_ms)) { return false; }
              continue;
            }
            if (e) {
              error = e;
              return false;
            }
            ++replies;
            return true;
          }
        }
      }
      return false;
    }
  }

  /// Receive exactly @p dst.size() plaintext bytes; an empty @p dst reads to close_notify.
  bool receive(std::span<std::byte> dst, int timeout_ms = 10000)
  {
    if (dst.empty()) {  // only control records may come before close_notify
      while (!closed) {
        if (!drain_control(timeout_ms)) { return false; }
      }
      return true;
    }
    std::size_t got = 0;
    while (got < dst.size()) {
      ssize_t const n = ::recv(fd, dst.data() + got, dst.size() - got, 0);
      if (n > 0) {
        got += static_cast<std::size_t>(n);
        continue;
      }
      if (n == 0) { return false; }
      if (errno == EIO) {  // a non-data record is queued: drain it (the transport's -EIO path)
        if (!drain_control(timeout_ms) || closed) { return false; }
        continue;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        if (!t::wait_fd(fd, POLLIN, timeout_ms)) { return false; }
        continue;
      }
      if (errno == EINTR) { continue; }
      last_errno = errno;
      return false;
    }
    return true;
  }
};

/// Handshake + export + install; SKIPs when the kernel refused kTLS.
struct ktls_connection {
  ktls_connection(provider const& p, std::uint16_t port) : fd(t::connect_loopback(port))
  {
    REQUIRE(fd.get() >= 0);
    std::error_code ec;
    s = p.new_session(fd.get(), {"localhost", port}, ec);
    REQUIRE(s != nullptr);
    REQUIRE_FALSE(t::drive_handshake(*s, fd.get()));
    REQUIRE(s->info().buffered_bytes == 0);
    REQUIRE_FALSE(s->export_keys(direction::receive, rx));
    REQUIRE_FALSE(s->export_keys(direction::send, tx));
    result = ktls::install(fd.get(), rx, tx, p.policy().expect_no_pad);
    rekey  = ktls::make_rekey_state(rx, tx, p.kdf());
    rx.wipe();
    tx.wipe();
    c.fd    = fd.get();
    c.state = rekey.get();
  }
  t::unique_fd fd;
  std::unique_ptr<session> s;
  traffic_keys rx;
  traffic_keys tx;
  ktls::install_result result;
  ktls::rekey_state_ptr rekey;
  ktls_client c;
};

#define REQUIRE_KTLS(conn)                                                         \
  do {                                                                             \
    if (!(conn).result.rx && !(conn).result.tx) {                                  \
      WARN("kernel refused kTLS: " << (conn).result.error.message());              \
      SKIP("kTLS not installable here (" << (conn).result.error.message() << ")"); \
    }                                                                              \
  } while (false)

/// TLS 1.3 traffic keys of @p suite from a made-up secret.
traffic_keys made_up_keys13(std::uint16_t suite, std::uint8_t fill)
{
  suite_info const s = suite_of(protocol_version::tls1_3, suite);
  openssl_kdf const k;
  traffic_keys out;
  out.version      = protocol_version::tls1_3;
  out.cipher       = s.cipher;
  out.cipher_suite = suite;
  out.key_len      = s.key_len;
  out.iv_len       = 12;
  out.secret_len   = s.hash_len;
  out.secret.fill(fill);
  REQUIRE_FALSE(tls13_traffic_keys(k,
                                   s,
                                   std::span<std::uint8_t const>(out.secret.data(), s.hash_len),
                                   std::span<std::uint8_t>(out.key.data(), s.key_len),
                                   std::span<std::uint8_t, 12>(out.iv)));
  return out;
}

std::span<std::byte const> as_bytes(std::vector<std::uint8_t> const& v)
{
  return {reinterpret_cast<std::byte const*>(v.data()), v.size()};
}

}  // namespace

TEST_CASE("ktls: install, then plain send/recv carry plaintext (TLS 1.2 and 1.3)", "[tls]")
{
  struct kcase {
    char const* name;
    int version;
    char const* suite;
  };
  for (kcase const& kc : {kcase{"1.3 AES-128", TLS1_3_VERSION, "TLS_AES_128_GCM_SHA256"},
                          kcase{"1.3 AES-256", TLS1_3_VERSION, "TLS_AES_256_GCM_SHA384"},
                          kcase{"1.2 AES-128", TLS1_2_VERSION, "ECDHE-ECDSA-AES128-GCM-SHA256"},
                          kcase{"1.2 AES-256", TLS1_2_VERSION, "ECDHE-ECDSA-AES256-GCM-SHA384"}}) {
    INFO(kc.name);
    bool const tls13 = kc.version == TLS1_3_VERSION;
    std::atomic<bool> request_ok{false};
    t::loopback_server srv(
      t::make_server_ctx(t::default_pki(), server_options(kc.version, kc.suite)),
      [&](t::server_conn& c) {
        request_ok = c.expect_pattern(20'000, 1);
        c.send_pattern(300'000, 2);
        c.wait_peer_close();
      });
    auto const before = read_tls_stat();
    auto const p      = make_openssl_provider(trusting());
    {
      ktls_connection k(*p, srv.port());
      REQUIRE_KTLS(k);
      CHECK(k.result.rx);
      CHECK(k.result.tx);
      CHECK_FALSE(k.result.error);
      CHECK(k.result.no_pad == tls13);  // T4: TLS 1.3 only
      CHECK((k.rekey != nullptr) == (tls13 && ktls::probe().tls13_rekey));
      REQUIRE(t::send_all(k.fd.get(), t::pattern(20'000, 1)));  // the kernel encrypts
      std::vector<std::byte> got(300'000);
      REQUIRE(k.c.receive(got));
      CHECK(t::check_pattern(got, 2) == std::string_view::npos);
      CHECK(k.c.controls == (tls13 ? 2U : 0U));  // OpenSSL's two tickets, skipped
      CHECK(k.c.replies == 0);
    }
    srv.join();
    CHECK(request_ok);
    auto const after = read_tls_stat();
    if (!before.empty()) {
      CHECK(delta(before, after, "TlsTxSw") >= 1);
      CHECK(delta(before, after, "TlsRxSw") >= 1);
    }
  }
}

TEST_CASE("ktls: KeyUpdate requested mid-stream re-keys both directions (Linux >= 6.14)", "[tls]")
{
  if (!ktls::probe().tls13_rekey) { SKIP("the kernel cannot re-key TLS 1.3 kTLS"); }
  std::atomic<bool> request_ok{false};
  std::atomic<bool> after_update_ok{false};
  std::atomic<unsigned> updates_at_server{0};
  t::loopback_server srv(
    t::make_server_ctx(t::default_pki(), server_options(TLS1_3_VERSION, "TLS_AES_128_GCM_SHA256")),
    [&](t::server_conn& c) {
      request_ok = c.expect_pattern(1000, 1);
      c.send_pattern(50'000, 2);
      c.new_session_ticket();
      c.key_update(true);  // the client re-keys RX and owes a reply + TX re-key
      c.send_pattern(50'000, 3);
      after_update_ok   = c.expect_pattern(20'000, 4);  // decrypted under the client's new key
      updates_at_server = c.key_updates_received();
      c.key_update(false);  // RX re-key only
      c.send_pattern(1000, 5);
      c.close_notify();
      c.wait_peer_close();
    });
  auto const before = read_tls_stat();
  auto const p      = make_openssl_provider(trusting());
  {
    ktls_connection k(*p, srv.port());
    REQUIRE_KTLS(k);
    REQUIRE(k.rekey != nullptr);
    REQUIRE(t::send_all(k.fd.get(), t::pattern(1000, 1)));
    std::vector<std::byte> a(50'000);
    REQUIRE(k.c.receive(a));
    CHECK(t::check_pattern(a, 2) == std::string_view::npos);
    std::vector<std::byte> b(50'000);
    REQUIRE(k.c.receive(b));
    CHECK(t::check_pattern(b, 3) == std::string_view::npos);
    CHECK(k.c.replies == 1);
    REQUIRE(t::send_all(k.fd.get(), t::pattern(20'000, 4)));
    std::vector<std::byte> c5(1000);
    REQUIRE(k.c.receive(c5));
    CHECK(t::check_pattern(c5, 5) == std::string_view::npos);
    std::vector<std::byte> none;
    CHECK(k.c.receive(none));  // close_notify
    CHECK(k.c.closed);
    CHECK(k.c.error == make_error_code(errc::closed));
    CHECK(k.c.controls == 2 + 1 + 1 + 1 + 1);  // tickets, ticket, 2 KeyUpdates, close_notify
  }
  srv.join();
  CHECK(request_ok);
  CHECK(after_update_ok);
  CHECK(updates_at_server == 1);
  auto const after = read_tls_stat();
  if (!before.empty()) {
    CHECK(delta(before, after, "TlsRxRekeyOk") >= 2);
    CHECK(delta(before, after, "TlsTxRekeyOk") >= 1);
  }
}

TEST_CASE("ktls: a tampered record fails the receive (EBADMSG)", "[tls]")
{
  t::loopback_server srv(
    t::make_server_ctx(t::default_pki(),
                       server_options(TLS1_2_VERSION, "ECDHE-ECDSA-AES128-GCM-SHA256")),
    [&](t::server_conn& c) {
      c.send_pattern(2000, 1);
      c.send_tampered(t::pattern(3000, 2), 200);
      c.wait_peer_close();
    });
  auto const p = make_openssl_provider(trusting());
  ktls_connection k(*p, srv.port());
  REQUIRE_KTLS(k);
  std::vector<std::byte> a(2000);
  REQUIRE(k.c.receive(a));
  CHECK(t::check_pattern(a, 1) == std::string_view::npos);
  std::vector<std::byte> b(3000);
  CHECK_FALSE(k.c.receive(b));
  CHECK(k.c.last_errno == EBADMSG);
}

TEST_CASE("ktls: install refuses what the kernel path does not serve", "[tls]")
{
  traffic_keys rx = made_up_keys13(0x1301, 1);
  traffic_keys tx = made_up_keys13(0x1301, 2);
  SECTION("a non-AES-GCM suite or mismatched directions: nothing touched")
  {
    traffic_keys chacha = made_up_keys13(0x1301, 3);
    chacha.cipher       = aead_cipher::other;
    chacha.cipher_suite = 0x1303;
    auto r              = ktls::install(-1, chacha, tx, true);
    CHECK_FALSE(r.rx);
    CHECK_FALSE(r.tx);
    CHECK(r.error == make_error_code(errc::not_supported));
    traffic_keys other = made_up_keys13(0x1302, 3);
    r                  = ktls::install(-1, rx, other, true);
    CHECK(r.error == make_error_code(errc::not_supported));
  }
  SECTION("TLS 1.3 on a kernel without TLS 1.3 kTLS RX (< 6.0)")
  {
    ktls::override_kernel_version_for_testing(5, 15);
    auto const r = ktls::install(-1, rx, tx, true);
    ktls::override_kernel_version_for_testing(0, 0);
    CHECK(r.error == make_error_code(errc::not_supported));
  }
  SECTION("not a TCP socket: TCP_ULP fails, the socket stays usable for user space")
  {
    int sv[2] = {-1, -1};
    REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    auto const r = ktls::install(sv[0], rx, tx, true);
    CHECK_FALSE(r.rx);
    CHECK_FALSE(r.tx);
    CHECK(r.error);
    char x = 'x';
    CHECK(::send(sv[0], &x, 1, MSG_NOSIGNAL) == 1);
    ::close(sv[0]);
    ::close(sv[1]);
  }
}

TEST_CASE("ktls: re-key state exists for TLS 1.3 on re-key-capable kernels only", "[tls]")
{
  openssl_kdf const k;
  traffic_keys rx = made_up_keys13(0x1301, 1);
  traffic_keys tx = made_up_keys13(0x1301, 2);
  {
    ktls::override_kernel_version_for_testing(6, 14);
    CHECK(ktls::make_rekey_state(rx, tx, k) != nullptr);
    ktls::override_kernel_version_for_testing(6, 13);
    CHECK(ktls::make_rekey_state(rx, tx, k) == nullptr);
    ktls::override_kernel_version_for_testing(0, 0);
  }
  traffic_keys rx12;
  rx12.version = protocol_version::tls1_2;
  CHECK(ktls::make_rekey_state(rx12, rx12, k) == nullptr);
  CHECK_FALSE(ktls::flush_reply(-1, nullptr));  // nothing owed
}

TEST_CASE("ktls: control record rules (tickets, KeyUpdate framing, alerts, data)", "[tls]")
{
  openssl_kdf const kdf;
  auto rec = [](std::vector<std::uint8_t> const& v) { return as_bytes(v); };
  std::error_code ec;
  SECTION("without re-key state (TLS 1.2, or TLS 1.3 on a kernel < 6.14)")
  {
    using A = ktls::control_action;
    CHECK(ktls::on_control_record(-1, nullptr, 22, rec({0, 0, 0, 0}), ec) == A::consumed);
    CHECK(ktls::on_control_record(-1, nullptr, 22, rec({0, 0, 0, 0, 0, 0, 0, 0}), ec) ==
          A::consumed);  // two HelloRequests
    std::vector<std::uint8_t> nst{4, 0, 0, 3, 9, 9, 9};
    CHECK(ktls::on_control_record(-1, nullptr, 22, rec(nst), ec) == A::consumed);
    CHECK(ktls::on_control_record(-1, nullptr, 22, rec({4, 0, 0, 9, 1}), ec) == A::failed);
    CHECK(ec == make_error_code(errc::protocol_error));  // a ticket spanning records
    CHECK(ktls::on_control_record(-1, nullptr, 22, rec({24, 0, 0, 1, 1}), ec) == A::failed);
    CHECK(ec == make_error_code(errc::protocol_error));  // KeyUpdate: reconnect
    CHECK(ktls::on_control_record(-1, nullptr, 22, rec({}), ec) == A::failed);
    CHECK(ktls::on_control_record(-1, nullptr, 21, rec({1, 0}), ec) == A::closed);
    CHECK(ec == make_error_code(errc::closed));
    CHECK(ktls::on_control_record(-1, nullptr, 21, rec({2, 0}), ec) == A::closed);
    CHECK(ktls::on_control_record(-1, nullptr, 21, rec({2, 40}), ec) == A::failed);
    CHECK(ktls::on_control_record(-1, nullptr, 21, rec({1, 0, 0}), ec) == A::failed);
    CHECK(ktls::on_control_record(-1, nullptr, 23, rec({1, 2, 3}), ec) == A::failed);
    CHECK(ktls::on_control_record(-1, nullptr, 20, rec({1}), ec) == A::failed);
  }
  SECTION("with re-key state: tickets across records, KeyUpdate rules")
  {
    using A = ktls::control_action;
    ktls::override_kernel_version_for_testing(6, 14);
    traffic_keys rx = made_up_keys13(0x1301, 1);
    traffic_keys tx = made_up_keys13(0x1301, 2);
    auto st         = ktls::make_rekey_state(rx, tx, kdf);
    ktls::override_kernel_version_for_testing(0, 0);
    REQUIRE(st != nullptr);
    // A 9-byte ticket split over three records, header split too.
    CHECK(ktls::on_control_record(-1, st.get(), 22, rec({4, 0}), ec) == A::consumed);
    CHECK(ktls::on_control_record(-1, st.get(), 22, rec({0, 9, 1, 2, 3}), ec) == A::consumed);
    // An alert while the ticket is incomplete is an interleaving violation.
    auto bad = ktls::make_rekey_state(rx, tx, kdf);
    if (bad) {
      CHECK(ktls::on_control_record(-1, bad.get(), 22, rec({4, 0, 0, 9, 1}), ec) == A::consumed);
      CHECK(ktls::on_control_record(-1, bad.get(), 21, rec({1, 0}), ec) == A::failed);
    }
    CHECK(ktls::on_control_record(-1, st.get(), 22, rec({4, 5, 6, 7, 8, 9}), ec) == A::consumed);
    // KeyUpdate framing errors fail before any re-key.
    CHECK(ktls::on_control_record(-1, st.get(), 22, rec({24, 0, 0, 2, 0, 0}), ec) == A::failed);
    auto st2 = ktls::make_rekey_state(rx, tx, kdf);
    if (st2) {
      CHECK(ktls::on_control_record(-1, st2.get(), 22, rec({24, 0, 0, 1, 2}), ec) == A::failed);
    }
    auto st3 = ktls::make_rekey_state(rx, tx, kdf);
    if (st3) {
      CHECK(ktls::on_control_record(-1, st3.get(), 22, rec({24, 0, 0, 1, 0, 4}), ec) == A::failed);
      // A well-formed KeyUpdate on a socket without kTLS: the TLS_RX re-key itself fails.
      auto st4 = ktls::make_rekey_state(rx, tx, kdf);
      CHECK(ktls::on_control_record(-1, st4.get(), 22, rec({24, 0, 0, 1, 1}), ec) == A::failed);
      CHECK(ec);
    }
    CHECK(ktls::on_control_record(-1, st.get(), 22, rec({2, 0, 0, 0}), ec) == A::failed);
  }
}
