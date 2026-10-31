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

/// @file test_net_transport.cpp
/// @brief Units of the REST io_uring transport (io/rest/net/uring_transport.hpp): URL parsing,
/// the effective-MTU rule, the kTLS receive interpretation and receive placement, the TLS
/// post-handshake decision (kTLS prefer / require / disable, RX-only refusal, user-space
/// fallback) over fake sessions and a fake kernel install, so the kTLS logic is covered on hosts
/// without tls.ko, and the generic exchange with a request body (fixed iovecs, a generator,
/// `Expect: 100-continue`) against a loopback echo server, plus a chunked buffered response.

#include "io/rest/errc.hpp"
#include "io/rest/net/tls_stream.hpp"
#include "io/rest/net/uring_transport.hpp"

#include <arpa/inet.h>
#include <catch2/catch_all.hpp>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

using namespace cucascade::io;
using namespace std::chrono_literals;

namespace {

// ---- fakes of the TLS layer ------------------------------------------------------------------

class fake_opener final : public tls::record_opener {
 public:
  tls::open_result open(tls::ciphertext_view,
                        std::span<iovec const>,
                        std::size_t,
                        bool) noexcept override
  {
    return {};
  }
  [[nodiscard]] bool progress_possible(tls::ciphertext_view) const noexcept override
  {
    return false;
  }
  [[nodiscard]] std::size_t pending() const noexcept override { return 0; }
  [[nodiscard]] std::error_code error() const noexcept override { return {}; }
};

class fake_sealer final : public tls::record_sealer {
 public:
  tls::seal_result seal(std::span<iovec const>,
                        std::size_t,
                        std::span<std::byte>,
                        unsigned) noexcept override
  {
    return {};
  }
  [[nodiscard]] std::size_t record_overhead() const noexcept override { return 64; }
  std::error_code request_key_update_reply() noexcept override { return {}; }
};

class fake_session final : public tls::session {
 public:
  explicit fake_session(std::size_t buffered = 0) : _buffered(buffered) {}
  tls::handshake_status handshake(std::error_code&) noexcept override
  {
    return tls::handshake_status::done;
  }
  [[nodiscard]] tls::session_info info() const noexcept override
  {
    tls::session_info i;
    i.version        = tls::protocol_version::tls1_2;
    i.cipher_suite   = 0xC02F;
    i.cipher         = tls::aead_cipher::aes_128_gcm;
    i.buffered_bytes = _buffered;
    return i;
  }
  std::error_code export_keys(tls::direction, tls::traffic_keys& out) noexcept override
  {
    ++exports;
    out.version      = tls::protocol_version::tls1_2;
    out.cipher       = tls::aead_cipher::aes_128_gcm;
    out.cipher_suite = 0xC02F;
    out.key_len      = 16;
    out.iv_len       = 4;
    out.sequence     = 1;
    return {};
  }
  std::error_code detach_from_socket() noexcept override
  {
    detached = true;
    return {};
  }
  tls::record_opener* opener() noexcept override { return &_opener; }
  tls::record_sealer* sealer() noexcept override { return &_sealer; }
  std::size_t describe_error(std::span<char>) const noexcept override { return 0; }

  int exports{0};
  bool detached{false};

 private:
  std::size_t _buffered;
  fake_opener _opener;
  fake_sealer _sealer;
};

class fake_kdf final : public tls::kdf {
 public:
  std::error_code hkdf_expand_label(std::size_t,
                                    std::span<std::uint8_t const>,
                                    std::string_view,
                                    std::span<std::uint8_t const>,
                                    std::span<std::uint8_t>) const noexcept override
  {
    return {};
  }
  std::error_code tls12_prf(std::size_t,
                            std::span<std::uint8_t const>,
                            std::string_view,
                            std::span<std::uint8_t const>,
                            std::span<std::uint8_t const>,
                            std::span<std::uint8_t>) const noexcept override
  {
    return {};
  }
};

class fake_provider final : public tls::provider {
 public:
  explicit fake_provider(tls::config cfg) : _cfg(std::move(cfg)) {}
  std::unique_ptr<tls::session> new_session(int,
                                            tls::server_identity const&,
                                            std::error_code&) const noexcept override
  {
    return std::make_unique<fake_session>();
  }
  /// No AEAD: the built-in record layer is unavailable, so user space uses the library's.
  std::unique_ptr<tls::aead> new_aead(tls::aead_cipher) const noexcept override { return nullptr; }
  tls::kdf const& kdf() const noexcept override { return _kdf; }
  tls::config const& policy() const noexcept override { return _cfg; }
  tls::protocol_version max_version() const noexcept override
  {
    return tls::protocol_version::tls1_2;
  }
  std::string_view name() const noexcept override { return "fake"; }
  std::string_view trust_source() const noexcept override { return "none"; }

 private:
  tls::config _cfg;
  fake_kdf _kdf;
};

/// What the fake kernel accepts, and how often it was asked.
std::atomic<int> g_install_calls{0};
std::atomic<int> g_install_mode{0};  // 0: nothing, 1: RX and TX, 2: TX only (RX refused)

tls::ktls::install_result fake_install(int,
                                       tls::traffic_keys const&,
                                       tls::traffic_keys const&,
                                       bool) noexcept
{
  ++g_install_calls;
  tls::ktls::install_result r;
  switch (g_install_mode.load()) {
    case 1:
      r.rx = true;
      r.tx = true;
      break;
    case 2:
      r.tx    = true;
      r.error = make_error_code(tls::errc::ktls_unavailable);
      break;
    default: r.error = std::make_error_code(std::errc::no_protocol_option); break;
  }
  return r;
}

struct install_guard {
  explicit install_guard(int mode)
  {
    g_install_calls = 0;
    g_install_mode  = mode;
    net::tls_support::set_ktls_install_for_testing(&fake_install);
  }
  ~install_guard() { net::tls_support::set_ktls_install_for_testing(nullptr); }
  install_guard(install_guard const&)            = delete;
  install_guard& operator=(install_guard const&) = delete;
};

net::tls_support::setup_result finish_with(tls::ktls_mode mode,
                                           int install_mode,
                                           fake_session& session,
                                           int& install_calls)
{
  tls::config cfg;
  cfg.ktls = mode;
  fake_provider const provider{cfg};
  install_guard const guard{install_mode};
  // A real (unconnected) socket: the user-space stream only records it.
  int const fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  REQUIRE(fd >= 0);
  auto r        = net::tls_support::finish(fd, session, provider, {}, "fake.host", 443);
  install_calls = g_install_calls.load();
  ::close(fd);
  return r;
}

// ---- loopback echo server ---------------------------------------------------------------------

/// Answers every request with `200 OK` and its body echoed (Content-Length); sends
/// `100 Continue` first when the request asks for it.  Keep-alive.
class echo_server {
 public:
  /// @p rcvbuf > 0: the accepted connections' SO_RCVBUF (a small one makes a slow reader pace
  /// the sender instead of the kernel absorbing the whole body).
  explicit echo_server(int rcvbuf = 0)
  {
    _fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    REQUIRE(_fd >= 0);
    if (rcvbuf > 0) {
      REQUIRE(::setsockopt(_fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof rcvbuf) == 0);
    }
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    REQUIRE(::bind(_fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0);
    REQUIRE(::listen(_fd, 16) == 0);
    socklen_t len = sizeof addr;
    REQUIRE(::getsockname(_fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0);
    _port   = ntohs(addr.sin_port);
    _thread = std::thread([this] { accept_loop(); });
  }
  ~echo_server()
  {
    _stop = true;
    ::shutdown(_fd, SHUT_RDWR);
    _thread.join();
    for (int const c : _open) {
      ::shutdown(c, SHUT_RDWR);
    }
    for (auto& t : _connections) {
      t.join();
    }
    ::close(_fd);
  }
  echo_server(echo_server const&)            = delete;
  echo_server& operator=(echo_server const&) = delete;

  [[nodiscard]] std::string url(std::string_view path = "/echo") const
  {
    return "http://127.0.0.1:" + std::to_string(_port) + std::string(path);
  }
  [[nodiscard]] std::size_t continues() const noexcept { return _continues.load(); }
  /// Respond with this raw response instead of an echo (one request), after @p delay.
  void answer_next_with(std::string response, std::chrono::milliseconds delay = {})
  {
    std::lock_guard lock{_mutex};
    _canned       = std::move(response);
    _canned_delay = delay;
  }
  /// Read the next request's whole body, then close without any response.
  void drop_next_after_body() { _drop_next = true; }
  /// Never answer `Expect: 100-continue` (wait for the body as if it were not there).
  void ignore_expect(bool on) { _ignore_expect = on; }
  /// Sleep this long after every receive (a slow reader).
  void pace_reads(std::chrono::milliseconds pause) { _pace_ms = pause.count(); }
  /// When the last request's body was complete, relative to when its head was.
  [[nodiscard]] std::chrono::milliseconds last_body_delay() const noexcept
  {
    return std::chrono::milliseconds{_body_delay_ms.load()};
  }

 private:
  void accept_loop()
  {
    while (!_stop) {
      int const conn = ::accept4(_fd, nullptr, nullptr, SOCK_CLOEXEC);
      if (conn < 0) {
        if (errno == EINTR) { continue; }
        return;
      }
      std::lock_guard lock{_mutex};
      _open.push_back(conn);
      _connections.emplace_back([this, conn] { serve(conn); });
    }
  }

  static bool send_all(int conn, std::string_view bytes)
  {
    std::size_t sent = 0;
    while (sent < bytes.size()) {
      auto const n = ::send(conn, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL);
      if (n < 0 && errno == EINTR) { continue; }
      if (n <= 0) { return false; }
      sent += static_cast<std::size_t>(n);
    }
    return true;
  }

  void serve(int conn)
  {
    std::string pending;
    std::array<char, 65536> buf{};
    auto fill = [&]() {
      auto const n = ::recv(conn, buf.data(), buf.size(), 0);
      if (n <= 0) { return false; }
      pending.append(buf.data(), static_cast<std::size_t>(n));
      if (auto const pace = _pace_ms.load(); pace > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds{pace});
      }
      return true;
    };
    for (;;) {
      while (pending.find("\r\n\r\n") == std::string::npos) {
        if (!fill()) { return; }
      }
      auto const end  = pending.find("\r\n\r\n") + 4;
      auto const head = pending.substr(0, end);
      pending.erase(0, end);
      std::string lower = head;
      for (auto& ch : lower) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
      }
      std::size_t length = 0;
      if (auto const at = lower.find("\r\ncontent-length:"); at != std::string::npos) {
        length = std::stoull(lower.substr(at + 17));
      }
      std::string canned;
      std::chrono::milliseconds delay{};
      {
        std::lock_guard lock{_mutex};
        canned.swap(_canned);
        delay = _canned_delay;
      }
      if (!canned.empty()) {
        std::this_thread::sleep_for(delay);
        (void)send_all(conn, canned);
        ::shutdown(conn, SHUT_RDWR);
        return;
      }
      if (lower.find("\r\nexpect: 100-continue") != std::string::npos && !_ignore_expect) {
        ++_continues;
        if (!send_all(conn, "HTTP/1.1 100 Continue\r\n\r\n")) { return; }
      }
      auto const head_at = std::chrono::steady_clock::now();
      while (pending.size() < length) {
        if (!fill()) { return; }
      }
      _body_delay_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - head_at)
                         .count();
      auto const body = pending.substr(0, length);
      pending.erase(0, length);
      if (_drop_next.exchange(false)) {
        ::shutdown(conn, SHUT_RDWR);
        return;
      }
      if (!send_all(conn,
                    "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(body.size()) +
                      "\r\n\r\n" + body)) {
        return;
      }
    }
  }

  int _fd{-1};
  std::uint16_t _port{0};
  std::atomic<bool> _stop{false};
  std::atomic<std::size_t> _continues{0};
  std::atomic<bool> _drop_next{false};
  std::atomic<bool> _ignore_expect{false};
  std::atomic<long> _pace_ms{0};
  std::atomic<long> _body_delay_ms{0};
  std::mutex _mutex;
  std::string _canned;
  std::chrono::milliseconds _canned_delay{};
  std::vector<int> _open;
  std::vector<std::thread> _connections;
  std::thread _thread;
};

/// Run @p x on slot 0 of @p t until it completes.
net::exchange_result run_exchange(net::uring_transport& t, net::exchange x)
{
  t.submit(0, std::move(x));
  t.flush();
  auto const deadline = std::chrono::steady_clock::now() + 20s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto& done = t.poll();
    if (!done.empty()) {
      REQUIRE(done.size() == 1);
      return std::move(done.front().result);
    }
    t.flush();
    std::this_thread::sleep_for(200us);
  }
  FAIL("the exchange did not complete");
  return {};
}

std::string pattern(std::size_t n)
{
  std::string s(n, '\0');
  for (std::size_t i = 0; i < n; ++i) {
    s[i] = static_cast<char>('a' + (i * 7 + i / 251) % 26);
  }
  return s;
}

}  // namespace

TEST_CASE("net: parse_url splits scheme, authority, port and target", "[rest][net][transport]")
{
  auto const a = net::parse_url("https://bucket.s3.amazonaws.com/key/a%20b?x-id=GetObject");
  CHECK(a.tls);
  CHECK(a.host == "bucket.s3.amazonaws.com");
  CHECK(a.port == 443);
  CHECK(a.authority == "bucket.s3.amazonaws.com");
  CHECK(a.target == "/key/a%20b?x-id=GetObject");
  auto const b = net::parse_url("http://127.0.0.1:9000");
  CHECK_FALSE(b.tls);
  CHECK(b.port == 9000);
  CHECK(b.authority == "127.0.0.1:9000");
  CHECK(b.target == "/");
  auto const c = net::parse_url("https://[::1]:8443/b?list-type=2#frag");
  CHECK(c.host == "::1");
  CHECK(c.authority == "[::1]:8443");
  CHECK(c.target == "/b?list-type=2");
  auto const d = net::parse_url("http://host?q=1");
  CHECK(d.target == "/?q=1");
  CHECK(a.origin() != net::parse_url("http://bucket.s3.amazonaws.com/").origin());
  CHECK_THROWS_AS(net::parse_url("s3://bucket/key"), std::invalid_argument);
  CHECK_THROWS_AS(net::parse_url("http://user@host/"), std::invalid_argument);
  CHECK_THROWS_AS(net::parse_url("http://host:99999/"), std::invalid_argument);
  CHECK_THROWS_AS(net::parse_url("http://host/a b"), std::invalid_argument);
}

TEST_CASE("net: the effective MTU adds the IP and TCP headers to the MSS", "[rest][net][mtu]")
{
  CHECK(net::effective_mtu({.ok = true, .mss = 1460}) == 1500);
  CHECK(net::effective_mtu({.ok = true, .mss = 1448, .timestamps = true}) == 1500);
  CHECK(net::effective_mtu({.ok = true, .mss = 1440, .ipv6 = true}) == 1500);
  CHECK(net::effective_mtu({.ok = true, .mss = 8948, .timestamps = true}) == 9000);
  // The real probe of a loopback connection.
  int const fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  REQUIRE(fd >= 0);
  auto const unconnected = net::mtu_probe(fd);
  CHECK((unconnected.ok ? unconnected.mss >= 0 : true));
  ::close(fd);
}

TEST_CASE("net: kTLS receive results are classified by their record-type cmsg", "[rest][net][ktls]")
{
  std::array<std::byte, 64> data{};
  iovec iov{data.data(), data.size()};
  net::ktls_cmsg_buffer cm;
  msghdr msg{};
  msg.msg_iov        = &iov;
  msg.msg_iovlen     = 1;
  msg.msg_control    = cm.bytes;
  msg.msg_controllen = sizeof(cm.bytes);
  auto set_type      = [&](unsigned char type) {
    auto* c       = CMSG_FIRSTHDR(&msg);
    c->cmsg_level = SOL_TLS;
    c->cmsg_type  = TLS_GET_RECORD_TYPE;
    c->cmsg_len   = CMSG_LEN(sizeof(unsigned char));
    *CMSG_DATA(c) = type;
  };
  SECTION("application data")
  {
    set_type(23);
    auto const r = net::interpret_ktls_recv(40, msg);
    CHECK(r.kind == net::rx_kind::data);
    CHECK(r.bytes == 40);
    CHECK(r.type == 23);
  }
  SECTION("a handshake record (ticket / KeyUpdate) and an alert are control records")
  {
    set_type(22);
    auto const r = net::interpret_ktls_recv(5, msg);
    CHECK(r.kind == net::rx_kind::control);
    CHECK(r.type == 22);
    CHECK(r.bytes == 5);
    set_type(21);
    CHECK(net::interpret_ktls_recv(2, msg).kind == net::rx_kind::control);
  }
  SECTION("a truncated cmsg is a protocol error")
  {
    set_type(22);
    msg.msg_flags = MSG_CTRUNC;
    auto const r  = net::interpret_ktls_recv(5, msg);
    CHECK(r.kind == net::rx_kind::error);
    CHECK(r.err == EPROTO);
  }
  SECTION("no cmsg: data, or the end of the stream")
  {
    msg.msg_controllen = 0;
    CHECK(net::interpret_ktls_recv(7, msg).kind == net::rx_kind::data);
    CHECK(net::interpret_ktls_recv(0, msg).kind == net::rx_kind::eof);
  }
  SECTION("errors")
  {
    CHECK(net::interpret_ktls_recv(-EAGAIN, msg).kind == net::rx_kind::again);
    auto const r = net::interpret_ktls_recv(-ECONNRESET, msg);
    CHECK(r.kind == net::rx_kind::error);
    CHECK(r.err == ECONNRESET);
  }
}

TEST_CASE("net: kTLS receive placement follows the policy and the EC2 hint", "[rest][net][ktls]")
{
  CHECK(net::ktls_recv_async(tls::recv_placement::iowq, false));
  CHECK_FALSE(net::ktls_recv_async(tls::recv_placement::inline_, true));
  CHECK(net::ktls_recv_async(tls::recv_placement::automatic, true));
  CHECK_FALSE(net::ktls_recv_async(tls::recv_placement::automatic, false));

  auto const root = std::filesystem::temp_directory_path() /
                    ("cuc-sysfs-" + std::to_string(::getpid()) + "-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(root / "devices/virtual/dmi/id");
  CHECK_FALSE(net::ec2_platform(root.string()));
  std::ofstream(root / "devices/virtual/dmi/id/sys_vendor") << "Dell Inc.\n";
  CHECK_FALSE(net::ec2_platform(root.string()));
  std::ofstream(root / "devices/virtual/dmi/id/board_vendor") << "Amazon EC2\n";
  CHECK(net::ec2_platform(root.string()));
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root / "hypervisor");
  std::ofstream(root / "hypervisor/uuid") << "ec2e1916-9099-7caf-fd21-012345abcdef\n";
  CHECK(net::ec2_platform(root.string()));
  std::filesystem::remove_all(root);
}

TEST_CASE("net: after the handshake kTLS is installed, refused or replaced by user space",
          "[rest][net][ktls]")
{
  int calls = 0;
  SECTION("prefer, kernel accepts RX and TX: kTLS, no user-space stream")
  {
    fake_session s;
    auto const r = finish_with(tls::ktls_mode::prefer, 1, s, calls);
    CHECK(r.path == net::tls_path::ktls);
    CHECK(r.stream == nullptr);
    CHECK_FALSE(r.user_space);
    CHECK(calls == 1);
    CHECK_FALSE(s.detached);
    net::tls_support::release_rekey(r.rekey);
  }
  SECTION("prefer, kernel refuses: the user-space stream, counted as a fallback")
  {
    fake_session s;
    auto const r = finish_with(tls::ktls_mode::prefer, 0, s, calls);
    REQUIRE(r.path == net::tls_path::user_space);
    CHECK(r.fallback);
    CHECK(r.user_space);
    CHECK_FALSE(r.record_layer);  // the fake provider has no AEAD: the library's layer
    CHECK(s.detached);
    CHECK(calls == 1);
    net::tls_stream::destroy(r.stream);
  }
  SECTION("require, kernel refuses: the connection fails with ktls_unavailable")
  {
    fake_session s;
    auto const r = finish_with(tls::ktls_mode::require, 0, s, calls);
    CHECK(r.path == net::tls_path::fail);
    CHECK(r.ec == make_error_code(rest::errc::ktls_unavailable));
    CHECK(r.stream == nullptr);
  }
  SECTION("one direction installed: unusable either way")
  {
    fake_session s;
    auto const r = finish_with(tls::ktls_mode::prefer, 2, s, calls);
    CHECK(r.path == net::tls_path::fail);
    CHECK(r.ec == make_error_code(rest::errc::ktls_unavailable));
  }
  SECTION("disable: user space without asking the kernel, not a fallback")
  {
    fake_session s;
    auto const r = finish_with(tls::ktls_mode::disable, 1, s, calls);
    REQUIRE(r.path == net::tls_path::user_space);
    CHECK_FALSE(r.fallback);
    CHECK(calls == 0);
    net::tls_stream::destroy(r.stream);
  }
  SECTION("bytes buffered by the library: no key export, user space keeps them")
  {
    fake_session s{17};
    auto const r = finish_with(tls::ktls_mode::prefer, 1, s, calls);
    REQUIRE(r.path == net::tls_path::user_space);
    CHECK(calls == 0);
    CHECK(s.exports == 0);
    net::tls_stream::destroy(r.stream);
  }
}

TEST_CASE("net: exchanges send request bodies (iovecs, generator, 100-continue)",
          "[rest][net][transport][body]")
{
  echo_server server;
  net::transport_options opts;
  opts.max_connections = 1;
  opts.queue_depth     = 8;
  net::uring_transport t{opts, net::transport_services{}};
  auto const payload = pattern((300UL << 10) + 7);

  auto put = [&](net::exchange x) {
    x.method       = http::method::put;
    x.url          = server.url();
    x.has_body     = true;
    x.body_length  = payload.size();
    x.target       = net::body_target::buffer;
    x.buffer_limit = payload.size();
    return x;
  };
  SECTION("fixed iovecs, scattered")
  {
    net::exchange x;
    x.body       = {iovec{const_cast<char*>(payload.data()), 1000},
                    iovec{const_cast<char*>(payload.data()) + 1000, 200000},
                    iovec{const_cast<char*>(payload.data()) + 201000, payload.size() - 201000}};
    auto const r = run_exchange(t, put(std::move(x)));
    REQUIRE_FALSE(r.ec);
    CHECK(r.meta.status == 200);
    CHECK(r.body_complete);
    CHECK(std::string(r.body.begin(), r.body.end()) == payload);
    // The connection stays for the next exchange.
    auto const again = run_exchange(
      t, put(net::exchange{.body = {iovec{const_cast<char*>(payload.data()), payload.size()}}}));
    CHECK(again.reused);
    CHECK(std::string(again.body.begin(), again.body.end()) == payload);
  }
  SECTION("a generator, five batches")
  {
    std::size_t offset = 0;
    std::vector<iovec> batch(1);
    net::exchange x;
    x.body_next = [&]() -> std::span<iovec const> {
      if (offset == payload.size()) { return {}; }
      auto const n = std::min<std::size_t>(64UL << 10, payload.size() - offset);
      batch[0]     = iovec{const_cast<char*>(payload.data()) + offset, n};
      offset += n;
      return batch;
    };
    auto const r = run_exchange(t, put(std::move(x)));
    REQUIRE_FALSE(r.ec);
    CHECK(std::string(r.body.begin(), r.body.end()) == payload);
  }
  SECTION("Expect: 100-continue sends the body only after the interim response")
  {
    net::exchange x;
    x.expect_continue = true;
    x.body            = {iovec{const_cast<char*>(payload.data()), payload.size()}};
    auto const r      = run_exchange(t, put(std::move(x)));
    REQUIRE_FALSE(r.ec);
    CHECK(server.continues() == 1);
    CHECK(std::string(r.body.begin(), r.body.end()) == payload);
  }
  SECTION("a final response before 100 Continue ends the exchange; the body is never sent")
  {
    server.answer_next_with("HTTP/1.1 417 Expectation Failed\r\nContent-Length: 0\r\n\r\n");
    net::exchange x;
    x.expect_continue = true;
    x.body            = {iovec{const_cast<char*>(payload.data()), payload.size()}};
    auto const r      = run_exchange(t, put(std::move(x)));
    REQUIRE_FALSE(r.ec);
    CHECK(r.meta.status == 417);
    CHECK_FALSE(t.established(0));  // not reusable: the request was never completed
  }
  t.shutdown();
}

TEST_CASE("net: buffered exchanges decode chunked bodies; data GETs refuse them",
          "[rest][net][transport][framing]")
{
  echo_server server;
  net::transport_options opts;
  opts.max_connections = 1;
  opts.queue_depth     = 8;
  net::uring_transport t{opts, net::transport_services{}};
  std::string const chunked =
    "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
  SECTION("buffer target")
  {
    server.answer_next_with(chunked);
    net::exchange x;
    x.url          = server.url("/list");
    x.target       = net::body_target::buffer;
    x.buffer_limit = 1024;
    auto const r   = run_exchange(t, std::move(x));
    REQUIRE_FALSE(r.ec);
    CHECK(r.body_complete);
    CHECK(std::string(r.body.begin(), r.body.end()) == "hello world");
  }
  SECTION("a length-framed exchange reports unsupported framing")
  {
    server.answer_next_with(chunked);
    std::array<char, 64> dst{};
    std::array<iovec, 1> sink{iovec{dst.data(), dst.size()}};
    net::exchange x;
    x.url            = server.url("/data");
    x.range          = std::pair<std::uint64_t, std::uint64_t>{0, 10};
    x.target         = net::body_target::sink;
    x.sink           = sink;
    x.require_length = true;
    auto const r     = run_exchange(t, std::move(x));
    CHECK(r.ec == make_error_code(rest::errc::http_unsupported_framing));
    CHECK(r.meta.status == 200);
  }
  t.shutdown();
}

TEST_CASE("net: the stall detector counts request body bytes sent",
          "[rest][net][transport][body][stall]")
{
  echo_server server{64 * 1024};
  net::transport_options opts;
  opts.max_connections = 1;
  opts.queue_depth     = 8;
  opts.so_sndbuf       = 64 * 1024;  // keep the body in user space: the reader paces the send
  net::uring_transport t{opts, net::transport_services{}};
  auto const payload = pattern(8UL << 20);
  auto put           = [&] {
    net::exchange x;
    x.method            = http::method::put;
    x.url               = server.url();
    x.has_body          = true;
    x.body_length       = payload.size();
    x.body              = {iovec{const_cast<char*>(payload.data()), payload.size()}};
    x.target            = net::body_target::discard;
    x.stall_bytes_per_s = 256 * 1024;
    x.stall_time        = 1s;
    return x;
  };
  SECTION("a slowly read upload is not cancelled while it is sending")
  {
    // <= 6 MiB/s for 8 MiB: the upload takes well over one stall window, and no response byte
    // arrives before its end; only the sent bytes keep the exchange alive.
    server.pace_reads(10ms);
    auto const r = run_exchange(t, put());
    REQUIRE_FALSE(r.ec);
    CHECK(r.meta.status == 200);
    CHECK(r.request_sent);
    CHECK(t.counters().stalls == 0);
    CHECK(server.last_body_delay() >= 1s);
  }
  SECTION("a server that never answers still stalls")
  {
    server.answer_next_with("HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n", 2500ms);
    auto x        = put();
    x.has_body    = false;
    x.body_length = 0;
    x.body        = {};
    x.method      = http::method::get;
    auto const r  = run_exchange(t, std::move(x));
    CHECK(r.ec == make_error_code(rest::errc::request_timeout));
    CHECK(t.counters().stalls == 1);
  }
  t.shutdown();
}

TEST_CASE("net: exchange results tell whether the request went out and why it failed",
          "[rest][net][transport][body]")
{
  echo_server server;
  net::transport_options opts;
  opts.max_connections = 1;
  opts.queue_depth     = 8;
  net::uring_transport t{opts, net::transport_services{}};
  auto const payload = pattern(100'000);
  auto put           = [&](std::string url) {
    net::exchange x;
    x.method       = http::method::put;
    x.url          = std::move(url);
    x.has_body     = true;
    x.body_length  = payload.size();
    x.body         = {iovec{const_cast<char*>(payload.data()), payload.size()}};
    x.target       = net::body_target::buffer;
    x.buffer_limit = 1024;
    return x;
  };
  SECTION("a refused connection: nothing was sent")
  {
    // A port nobody listens on (bound, then closed).
    int const fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    REQUIRE(::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0);
    socklen_t len = sizeof addr;
    REQUIRE(::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0);
    ::close(fd);
    auto const r = run_exchange(t, put("http://127.0.0.1:" + std::to_string(ntohs(addr.sin_port))));
    CHECK(r.ec);
    CHECK_FALSE(r.request_sent);
    CHECK_FALSE(r.response_started);
  }
  SECTION("a dropped response: the request went out, no answer came back")
  {
    server.drop_next_after_body();
    auto const r = run_exchange(t, put(server.url()));
    CHECK(r.ec);
    CHECK(r.request_sent);
    CHECK_FALSE(r.response_started);
    CHECK_FALSE(t.established(0));
  }
  SECTION("an S3 error body is drained and its code extracted")
  {
    std::string const body =
      "<?xml version=\"1.0\"?>\n<Error><Code>NoSuchUpload</Code><Message>gone</Message></Error>";
    server.answer_next_with("HTTP/1.1 404 Not Found\r\nContent-Length: " +
                            std::to_string(body.size()) + "\r\n\r\n" + body);
    auto const r = run_exchange(t, put(server.url()));
    REQUIRE_FALSE(r.ec);
    CHECK(r.meta.status == 404);
    CHECK(r.meta.amz_error_code.view() == "NoSuchUpload");
    CHECK(r.error_body_complete);
    CHECK_FALSE(r.body_complete);
  }
  SECTION("a cut error body is reported as incomplete")
  {
    server.answer_next_with(
      "HTTP/1.1 404 Not Found\r\nContent-Length: 500\r\n\r\n<Error><Code>NoSuchKey</Code>");
    auto const r = run_exchange(t, put(server.url()));
    REQUIRE_FALSE(r.ec);
    CHECK(r.meta.status == 404);
    CHECK_FALSE(r.error_body_complete);
  }
  SECTION("a 2xx answer is the caller's: no error body, no code")
  {
    auto const r = run_exchange(t, put(server.url()));
    REQUIRE_FALSE(r.ec);
    CHECK(r.request_sent);
    CHECK_FALSE(r.error_body_complete);
    CHECK(r.meta.amz_error_code.view().empty());
  }
  t.shutdown();
}

TEST_CASE("net: a held body goes out after continue_timeout when Expect is ignored",
          "[rest][net][transport][body]")
{
  echo_server server;
  server.ignore_expect(true);
  net::transport_options opts;
  opts.max_connections = 1;
  opts.queue_depth     = 8;
  net::uring_transport t{opts, net::transport_services{}};
  auto const payload = pattern(200'000);
  net::exchange x;
  x.method           = http::method::put;
  x.url              = server.url();
  x.has_body         = true;
  x.body_length      = payload.size();
  x.body             = {iovec{const_cast<char*>(payload.data()), payload.size()}};
  x.expect_continue  = true;
  x.continue_timeout = 300ms;
  x.target           = net::body_target::buffer;
  x.buffer_limit     = payload.size();
  auto const start   = std::chrono::steady_clock::now();
  auto const r       = run_exchange(t, std::move(x));
  REQUIRE_FALSE(r.ec);
  CHECK(r.meta.status == 200);
  CHECK(r.request_sent);
  CHECK(server.continues() == 0);
  CHECK(std::chrono::steady_clock::now() - start >= 300ms);
  CHECK(std::string(r.body.begin(), r.body.end()) == payload);
  // The connection is clean: the next exchange (with a 100 Continue this time) reuses it.
  server.ignore_expect(false);
  net::exchange y;
  y.method           = http::method::put;
  y.url              = server.url();
  y.has_body         = true;
  y.body_length      = payload.size();
  y.body             = {iovec{const_cast<char*>(payload.data()), payload.size()}};
  y.expect_continue  = true;
  y.continue_timeout = 10s;
  y.target           = net::body_target::buffer;
  y.buffer_limit     = payload.size();
  auto const again   = run_exchange(t, std::move(y));
  REQUIRE_FALSE(again.ec);
  CHECK(again.reused);
  CHECK(server.continues() == 1);
  CHECK(std::string(again.body.begin(), again.body.end()) == payload);
  t.shutdown();
}

namespace {

/// Records the record budget of every seal and "encrypts" by copying.
class counting_sealer final : public tls::record_sealer {
 public:
  tls::seal_result seal(std::span<iovec const> in,
                        std::size_t skip,
                        std::span<std::byte> out,
                        unsigned max_records) noexcept override
  {
    last_max_records = max_records;
    tls::seal_result r;
    for (auto const& v : in) {
      if (skip >= v.iov_len) {
        skip -= v.iov_len;
        continue;
      }
      auto const n = std::min(v.iov_len - skip, out.size() - r.ciphertext);
      std::memcpy(out.data() + r.ciphertext, static_cast<char const*>(v.iov_base) + skip, n);
      r.ciphertext += n;
      r.plaintext += n;
      skip = 0;
    }
    return r;
  }
  [[nodiscard]] std::size_t record_overhead() const noexcept override { return 64; }
  std::error_code request_key_update_reply() noexcept override { return {}; }
  unsigned last_max_records{0};
};

class sealing_session final : public tls::session {
 public:
  tls::handshake_status handshake(std::error_code&) noexcept override
  {
    return tls::handshake_status::done;
  }
  [[nodiscard]] tls::session_info info() const noexcept override { return {}; }
  std::error_code export_keys(tls::direction, tls::traffic_keys&) noexcept override { return {}; }
  std::error_code detach_from_socket() noexcept override { return {}; }
  tls::record_opener* opener() noexcept override { return &_opener; }
  tls::record_sealer* sealer() noexcept override { return &sealer_; }
  std::size_t describe_error(std::span<char>) const noexcept override { return 0; }
  counting_sealer sealer_;

 private:
  fake_opener _opener;
};

}  // namespace

TEST_CASE("net: the user-space TLS stream seals request bodies in configurable batches",
          "[rest][net][tls]")
{
  auto const configured = GENERATE(0U, 3U, 16U, 64U);
  CAPTURE(configured);
  sealing_session session;
  net::stream_options opts;
  if (configured != 0) { opts.tx_batch_records = configured; }
  int const fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  REQUIRE(fd >= 0);
  std::error_code ec;
  auto* stream = net::tls_stream::create(fd, session, nullptr, opts, ec);
  REQUIRE(stream != nullptr);
  std::vector<char> body(1 << 20, 'x');
  iovec const v{body.data(), body.size()};
  io_uring_sqe sqe{};
  stream->prep_send(&sqe, std::span<iovec const>{&v, 1});
  auto const expected = configured == 0 ? 16U : std::min(configured, 16U);
  CHECK(session.sealer_.last_max_records == expected);
  CHECK(net::tls_stream::tx_bytes >= 16 * tls::max_record_ciphertext);
  net::tls_stream::destroy(stream);
  ::close(fd);
}

TEST_CASE("net: a response that landed by the deadline completes instead of timing out",
          "[rest][net][transport][deadline]")
{
  // The response arrives at 100 ms; the exchange's deadline is 400 ms; the loop is not polled
  // between 50 ms and 700 ms (a slow pass).  The CQE is then already in the ring: it must be
  // reaped before deadlines are checked, not discarded by a cancel.
  echo_server server;
  server.answer_next_with("HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok", 100ms);
  net::transport_options opts;
  opts.max_connections = 1;
  opts.queue_depth     = 8;
  net::uring_transport t{opts, net::transport_services{}};
  net::exchange x;
  x.url          = server.url("/late");
  x.target       = net::body_target::buffer;
  x.buffer_limit = 16;
  x.timeout      = 400ms;
  t.submit(0, std::move(x));
  t.flush();
  auto const until = std::chrono::steady_clock::now() + 50ms;
  std::optional<net::exchange_result> result;
  while (std::chrono::steady_clock::now() < until && !result) {
    auto& done = t.poll();
    if (!done.empty()) { result = std::move(done.front().result); }
    std::this_thread::sleep_for(1ms);
  }
  REQUIRE_FALSE(result.has_value());  // the request is out, the response not yet
  std::this_thread::sleep_for(650ms);
  auto const deadline = std::chrono::steady_clock::now() + 5s;
  while (!result && std::chrono::steady_clock::now() < deadline) {
    auto& done = t.poll();
    if (!done.empty()) { result = std::move(done.front().result); }
  }
  REQUIRE(result.has_value());
  CHECK_FALSE(result->ec);
  CHECK(result->meta.status == 200);
  CHECK(std::string(result->body.begin(), result->body.end()) == "ok");
  CHECK(t.counters().timeouts == 0);
  t.shutdown();
}

namespace {

std::atomic<int> g_flush_calls{0};
std::atomic<int> g_flush_would_block{0};

std::error_code fake_flush(int, tls::ktls::rekey_state*) noexcept
{
  ++g_flush_calls;
  if (g_flush_would_block.fetch_sub(1) > 0) {
    return std::make_error_code(std::errc::operation_would_block);
  }
  return {};
}

std::error_code broken_flush(int, tls::ktls::rekey_state*) noexcept
{
  return std::make_error_code(std::errc::broken_pipe);
}

}  // namespace

TEST_CASE("net: a kTLS KeyUpdate reply that would block stays owed instead of failing",
          "[rest][net][ktls]")
{
  net::ktls_reply reply;
  g_flush_calls       = 0;
  g_flush_would_block = 2;
  // The socket buffer is full twice: the reply stays owed (wait for POLLOUT), not an error.
  CHECK(reply.flush(-1, nullptr, &fake_flush) == net::ktls_reply::step::wait_writable);
  CHECK(reply.owed);
  CHECK(reply.flush(-1, nullptr, &fake_flush) == net::ktls_reply::step::wait_writable);
  CHECK(reply.owed);
  CHECK(reply.flush(-1, nullptr, &fake_flush) == net::ktls_reply::step::done);
  CHECK_FALSE(reply.owed);
  CHECK(g_flush_calls == 3);
  // Any other error retires the connection.
  CHECK(reply.flush(-1, nullptr, &broken_flush) == net::ktls_reply::step::failed);
}

namespace {

/// A receive layer that never sees a complete record: every pass is starved.
class starved_opener final : public tls::record_opener {
 public:
  tls::open_result open(tls::ciphertext_view,
                        std::span<iovec const>,
                        std::size_t,
                        bool) noexcept override
  {
    return {0, 0, tls::open_stop::starved};
  }
  [[nodiscard]] bool progress_possible(tls::ciphertext_view) const noexcept override
  {
    return false;
  }
  [[nodiscard]] std::size_t pending() const noexcept override { return 0; }
  [[nodiscard]] std::error_code error() const noexcept override { return {}; }
};

}  // namespace

TEST_CASE("net: an idle user-space TLS connection with half a control record stays reusable",
          "[rest][net][tls]")
{
  std::array<int, 2> fds{};
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds.data()) == 0);
  fake_session session;
  std::error_code ec;
  auto* stream =
    net::tls_stream::create(fds[0], session, std::make_unique<starved_opener>(), {}, ec);
  REQUIRE(stream != nullptr);
  // Half of a NewSessionTicket arrived while the connection was idle.
  std::array<char, 10> const half{22, 3, 3, 0, 64, 4, 0, 0, 60, 0};
  REQUIRE(::send(fds[1], half.data(), half.size(), MSG_NOSIGNAL) ==
          static_cast<ssize_t>(half.size()));
  CHECK(stream->drain_idle_socket());  // buffered for the next receive, not a dead connection
  CHECK(stream->buffered_ciphertext() == half.size());
  // EOF still retires it.
  ::shutdown(fds[1], SHUT_WR);
  CHECK_FALSE(stream->drain_idle_socket());
  net::tls_stream::destroy(stream);
  ::close(fds[0]);
  ::close(fds[1]);
}
