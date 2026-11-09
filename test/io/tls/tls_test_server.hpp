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

#pragma once

#include "tls_test_pki.hpp"

#include <cucascade/io/tls/tls.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

/**
 * @file tls_test_server.hpp
 * @brief A blocking loopback TLS server (one OpenSSL server thread per connection, driven by a
 * test script), plus the client-side pieces the TLS tests share: deterministic payloads, a
 * non-blocking loopback connect, a poll(2) handshake driver for `io::tls::session`, a small
 * wrapping ciphertext ring and user-space receive / send loops over `record_opener` /
 * `record_sealer`.  Header-only.
 *
 * The server reads with `SSL_read` on its socket and WRITES THROUGH A MEMORY BIO: every action
 * (`write`, `key_update`, `new_session_ticket`, `close_notify`, `send_tampered`) produces
 * ciphertext that the script then sends with one `send(2)` (or tampers with first).  Server
 * threads block SIGPIPE for themselves, so a client that hangs up never kills the test process.
 */
namespace cucascade::test::tls {

//===----------------------------------------------------------------------===//
// Payloads and sockets
//===----------------------------------------------------------------------===//

inline std::uint64_t splitmix64(std::uint64_t x) noexcept
{
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

/// Deterministic byte at absolute offset @p o of stream @p seed.
inline std::byte pattern_byte(std::uint64_t seed, std::uint64_t o) noexcept
{
  return static_cast<std::byte>((splitmix64(seed ^ (o >> 3)) >> ((o & 7U) * 8U)) & 0xffU);
}

inline void fill_pattern(std::span<std::byte> out, std::uint64_t seed, std::uint64_t offset = 0)
{
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = pattern_byte(seed, offset + i);
  }
}

inline std::vector<std::byte> pattern(std::size_t n, std::uint64_t seed, std::uint64_t offset = 0)
{
  std::vector<std::byte> v(n);
  fill_pattern(v, seed, offset);
  return v;
}

/// @return the first offset where @p data differs from the pattern, or npos.
inline std::size_t check_pattern(std::span<std::byte const> data,
                                 std::uint64_t seed,
                                 std::uint64_t offset = 0)
{
  for (std::size_t i = 0; i < data.size(); ++i) {
    if (data[i] != pattern_byte(seed, offset + i)) { return i; }
  }
  return std::string_view::npos;
}

/// poll() one fd; false on timeout or error.
inline bool wait_fd(int fd, short events, int timeout_ms)
{
  pollfd p{fd, events, 0};
  for (;;) {
    int const r = ::poll(&p, 1, timeout_ms);
    if (r > 0) { return true; }
    if (r == 0) { return false; }
    if (errno != EINTR) { return false; }
  }
}

/// Non-blocking TCP socket connected to the loopback address; -1 on failure.
inline int connect_loopback(std::uint16_t port, bool ipv6 = false, int timeout_ms = 3000)
{
  int const fd = ::socket(ipv6 ? AF_INET6 : AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (fd < 0) { return -1; }
  int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  sockaddr_storage ss{};
  socklen_t len = 0;
  if (ipv6) {
    auto* a        = reinterpret_cast<sockaddr_in6*>(&ss);
    a->sin6_family = AF_INET6;
    a->sin6_port   = htons(port);
    a->sin6_addr   = in6addr_loopback;
    len            = sizeof(sockaddr_in6);
  } else {
    auto* a            = reinterpret_cast<sockaddr_in*>(&ss);
    a->sin_family      = AF_INET;
    a->sin_port        = htons(port);
    a->sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    len                = sizeof(sockaddr_in);
  }
  if (::connect(fd, reinterpret_cast<sockaddr*>(&ss), len) != 0) {
    int err        = 0;
    socklen_t elen = sizeof err;
    if (errno != EINPROGRESS || !wait_fd(fd, POLLOUT, timeout_ms) ||
        ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &elen) != 0 || err != 0) {
      ::close(fd);
      return -1;
    }
  }
  return fd;
}

/// Owning fd.
class unique_fd {
 public:
  explicit unique_fd(int fd = -1) noexcept : _fd(fd) {}
  ~unique_fd() { reset(); }
  unique_fd(unique_fd&& o) noexcept : _fd(std::exchange(o._fd, -1)) {}
  unique_fd& operator=(unique_fd&& o) noexcept
  {
    if (this != &o) {
      reset();
      _fd = std::exchange(o._fd, -1);
    }
    return *this;
  }
  unique_fd(unique_fd const&)            = delete;
  unique_fd& operator=(unique_fd const&) = delete;
  [[nodiscard]] int get() const noexcept { return _fd; }
  void reset() noexcept
  {
    if (_fd >= 0) { ::close(_fd); }
    _fd = -1;
  }

 private:
  int _fd;
};

/// send() all of @p data on a (possibly non-blocking) socket; false on error or timeout.
inline bool send_all(int fd, std::span<std::byte const> data, int timeout_ms = 10000)
{
  std::size_t off = 0;
  while (off < data.size()) {
    ssize_t const n = ::send(fd, data.data() + off, data.size() - off, MSG_NOSIGNAL);
    if (n > 0) {
      off += static_cast<std::size_t>(n);
      continue;
    }
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      if (!wait_fd(fd, POLLOUT, timeout_ms)) { return false; }
      continue;
    }
    if (n < 0 && errno == EINTR) { continue; }
    return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Server
//===----------------------------------------------------------------------===//

/// One server connection, handed to the test script on its own thread.
class server_conn {
 public:
  server_conn(SSL* ssl, int fd) noexcept : _ssl(ssl), _fd(fd) {}
  server_conn(server_conn const&)            = delete;
  server_conn& operator=(server_conn const&) = delete;

  [[nodiscard]] SSL* ssl() const noexcept { return _ssl; }
  [[nodiscard]] int fd() const noexcept { return _fd; }
  /// KeyUpdate messages the server received from the client.
  [[nodiscard]] unsigned key_updates_received() const noexcept { return _key_updates_rx.load(); }

  /// Read exactly @p n plaintext bytes (the socket's 10 s receive timeout bounds it).
  bool read_exact(std::size_t n, std::vector<std::byte>& out)
  {
    out.resize(n);
    std::size_t got = 0;
    while (got < n) {
      std::size_t r = 0;
      if (SSL_read_ex(_ssl, out.data() + got, n - got, &r) != 1) { return false; }
      got += r;
    }
    return true;
  }
  /// Read @p n bytes and compare them with pattern @p seed.
  bool expect_pattern(std::size_t n, std::uint64_t seed, std::uint64_t offset = 0)
  {
    std::vector<std::byte> got;
    return read_exact(n, got) && check_pattern(got, seed, offset) == std::string_view::npos;
  }
  /// Encrypt @p data (records of up to 16 KiB) and send it.
  bool write(std::span<std::byte const> data)
  {
    std::size_t off = 0;
    while (off < data.size()) {
      std::size_t w = 0;
      if (SSL_write_ex(_ssl, data.data() + off, data.size() - off, &w) != 1) { return false; }
      off += w;
    }
    return flush();
  }
  bool send_pattern(std::size_t n, std::uint64_t seed, std::uint64_t offset = 0)
  {
    return write(pattern(n, seed, offset));
  }
  /// Read @p n bytes and send them back.
  bool echo(std::size_t n)
  {
    std::vector<std::byte> b;
    return read_exact(n, b) && write(b);
  }
  /// Send a KeyUpdate (`SSL_key_update` + `SSL_do_handshake`); the server re-keys its TX.
  bool key_update(bool requested)
  {
    int const type = requested ? SSL_KEY_UPDATE_REQUESTED : SSL_KEY_UPDATE_NOT_REQUESTED;
    return SSL_key_update(_ssl, type) == 1 && SSL_do_handshake(_ssl) == 1 && flush();
  }
  /// Send one TLS 1.3 NewSessionTicket.
  bool new_session_ticket()
  {
    return SSL_new_session_ticket(_ssl) == 1 && SSL_do_handshake(_ssl) == 1 && flush();
  }
  /// Send close_notify.
  bool close_notify()
  {
    (void)SSL_shutdown(_ssl);
    return flush();
  }
  /// Encrypt @p data, flip bit 0 of ciphertext byte @p at (clamped) and send the result.
  bool send_tampered(std::span<std::byte const> data, std::size_t at)
  {
    std::size_t w = 0;
    if (SSL_write_ex(_ssl, data.data(), data.size(), &w) != 1) { return false; }
    std::vector<std::byte> ct = take();
    if (ct.empty()) { return false; }
    at = std::min(at, ct.size() - 1);
    ct[at] ^= std::byte{1};
    return send_all(_fd, ct);
  }
  /// The ciphertext produced since the last send, not sent.
  std::vector<std::byte> take()
  {
    char* p      = nullptr;
    BIO* const b = SSL_get_wbio(_ssl);
    long const n = BIO_get_mem_data(b, &p);
    std::vector<std::byte> out(reinterpret_cast<std::byte*>(p),
                               reinterpret_cast<std::byte*>(p) + std::max(0L, n));
    (void)BIO_reset(b);
    return out;
  }
  bool send_raw(std::span<std::byte const> b) { return send_all(_fd, b); }
  /// Send what the actions so far produced.
  bool flush() { return send_all(_fd, take()); }
  /// Read until the peer closes (or 10 s); the script's usual last step.
  void wait_peer_close()
  {
    std::array<char, 4096> b{};
    for (;;) {
      std::size_t r = 0;
      if (SSL_read_ex(_ssl, b.data(), b.size(), &r) != 1) { return; }
    }
  }

 private:
  friend class loopback_server;
  SSL* _ssl;
  int _fd;
  std::atomic<unsigned> _key_updates_rx{0};  ///< written by the server's message callback
};

/**
 * @brief Loopback TLS server: accepts on 127.0.0.1 (or ::1), one thread per connection runs
 * `SSL_accept` and then the test @p script.  The destructor stops accepting and joins.
 */
class loopback_server {
 public:
  using script = std::function<void(server_conn&)>;

  /// @param ctx  server context (owned: freed by the destructor)
  loopback_server(SSL_CTX* ctx, script s, bool ipv6 = false) : _ctx(ctx), _script(std::move(s))
  {
    SSL_CTX_set_app_data(_ctx, this);
    SSL_CTX_set_keylog_callback(_ctx, &loopback_server::on_keylog);
    _lfd    = ::socket(ipv6 ? AF_INET6 : AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    int one = 1;
    ::setsockopt(_lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_storage ss{};
    socklen_t len = 0;
    if (ipv6) {
      auto* a        = reinterpret_cast<sockaddr_in6*>(&ss);
      a->sin6_family = AF_INET6;
      a->sin6_addr   = in6addr_loopback;
      len            = sizeof(sockaddr_in6);
    } else {
      auto* a            = reinterpret_cast<sockaddr_in*>(&ss);
      a->sin_family      = AF_INET;
      a->sin_addr.s_addr = htonl(INADDR_LOOPBACK);
      len                = sizeof(sockaddr_in);
    }
    if (::bind(_lfd, reinterpret_cast<sockaddr*>(&ss), len) != 0 || ::listen(_lfd, 64) != 0 ||
        ::getsockname(_lfd, reinterpret_cast<sockaddr*>(&ss), &len) != 0) {
      throw std::runtime_error("loopback_server: bind/listen failed");
    }
    _port     = ntohs(ipv6 ? reinterpret_cast<sockaddr_in6*>(&ss)->sin6_port
                       : reinterpret_cast<sockaddr_in*>(&ss)->sin_port);
    _acceptor = std::thread([this] { accept_loop(); });
  }

  ~loopback_server()
  {
    _stop.store(true);
    ::shutdown(_lfd, SHUT_RDWR);
    if (_acceptor.joinable()) { _acceptor.join(); }
    join();
    ::close(_lfd);
    SSL_CTX_free(_ctx);
  }
  loopback_server(loopback_server const&)            = delete;
  loopback_server& operator=(loopback_server const&) = delete;

  [[nodiscard]] std::uint16_t port() const noexcept { return _port; }
  [[nodiscard]] unsigned accepted() const noexcept { return _accepted.load(); }
  [[nodiscard]] unsigned handshake_failures() const noexcept { return _failed.load(); }

  /// Wait for every connection thread started so far (their scripts finished).
  void join()
  {
    std::vector<std::thread> t;
    {
      std::lock_guard l(_mu);
      t.swap(_workers);
    }
    for (auto& w : t) {
      if (w.joinable()) { w.join(); }
    }
  }
  /// Server-side NSS key log lines (CLIENT_TRAFFIC_SECRET_0 ...).
  [[nodiscard]] std::vector<std::string> keylog() const
  {
    std::lock_guard l(_mu);
    return _keylog;
  }
  /// The secret of key log @p label of the last connection, hex; empty when absent.
  [[nodiscard]] std::string keylog_secret(std::string_view label) const
  {
    std::lock_guard l(_mu);
    for (auto it = _keylog.rbegin(); it != _keylog.rend(); ++it) {
      if (it->rfind(std::string{label} + " ", 0) == 0) { return it->substr(it->rfind(' ') + 1); }
    }
    return {};
  }
  /// The TLS 1.2 master secret of the last accepted connection.
  [[nodiscard]] std::vector<std::uint8_t> master_key() const
  {
    std::lock_guard l(_mu);
    return _master_key;
  }
  /// The hello randoms of the last accepted TLS 1.2 connection.
  [[nodiscard]] std::array<std::uint8_t, 32> client_random() const
  {
    std::lock_guard l(_mu);
    return _client_random;
  }
  [[nodiscard]] std::array<std::uint8_t, 32> server_random() const
  {
    std::lock_guard l(_mu);
    return _server_random;
  }

 private:
  static void on_keylog(SSL const* ssl, char const* line)
  {
    auto* self = static_cast<loopback_server*>(SSL_CTX_get_app_data(SSL_get_SSL_CTX(ssl)));
    if (self == nullptr || line == nullptr) { return; }
    std::lock_guard l(self->_mu);
    self->_keylog.emplace_back(line);
  }

  static void on_msg(
    int write_p, int, int content_type, void const* buf, std::size_t len, SSL*, void* arg)
  {
    auto* c = static_cast<server_conn*>(arg);
    if (c == nullptr || write_p != 0 || content_type != SSL3_RT_HANDSHAKE || len < 1) { return; }
    if (*static_cast<unsigned char const*>(buf) == SSL3_MT_KEY_UPDATE) { ++c->_key_updates_rx; }
  }

  void accept_loop()
  {
    while (!_stop.load()) {
      int const fd = ::accept4(_lfd, nullptr, nullptr, SOCK_CLOEXEC);
      if (fd < 0) {
        if (errno == EINTR) { continue; }
        return;
      }
      std::lock_guard l(_mu);
      _workers.emplace_back([this, fd] { serve(fd); });
    }
  }

  void serve(int fd)
  {
    // The client may hang up while the server writes: EPIPE, never a process-wide SIGPIPE.
    sigset_t block;
    sigemptyset(&block);
    sigaddset(&block, SIGPIPE);
    pthread_sigmask(SIG_BLOCK, &block, nullptr);
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    timeval tv{10, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    SSL* ssl = SSL_new(_ctx);
    SSL_set_fd(ssl, fd);
    server_conn conn(ssl, fd);
    SSL_set_msg_callback(ssl, &loopback_server::on_msg);
    SSL_set_msg_callback_arg(ssl, &conn);
    if (SSL_accept(ssl) == 1) {
      if (SSL_version(ssl) == TLS1_2_VERSION) {
        std::vector<std::uint8_t> mk(48);
        mk.resize(SSL_SESSION_get_master_key(SSL_get_session(ssl), mk.data(), mk.size()));
        std::array<std::uint8_t, 32> cr{};
        std::array<std::uint8_t, 32> sr{};
        (void)SSL_get_client_random(ssl, cr.data(), cr.size());
        (void)SSL_get_server_random(ssl, sr.data(), sr.size());
        std::lock_guard l(_mu);
        _master_key    = std::move(mk);
        _client_random = cr;
        _server_random = sr;
      }
      ++_accepted;
      SSL_set0_wbio(ssl, BIO_new(BIO_s_mem()));  // writes: memory BIO, then one send(2)
      if (_script) { _script(conn); }
    } else {
      ++_failed;
      ERR_clear_error();
    }
    SSL_set_msg_callback(ssl, nullptr);
    SSL_free(ssl);
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
  }

  SSL_CTX* _ctx;
  script _script;
  int _lfd{-1};
  std::uint16_t _port{0};
  std::atomic<bool> _stop{false};
  std::atomic<unsigned> _accepted{0};
  std::atomic<unsigned> _failed{0};
  std::thread _acceptor;
  mutable std::mutex _mu;
  std::vector<std::thread> _workers;
  std::vector<std::string> _keylog;
  std::vector<std::uint8_t> _master_key;
  std::array<std::uint8_t, 32> _client_random{};
  std::array<std::uint8_t, 32> _server_random{};
};

//===----------------------------------------------------------------------===//
// Client side
//===----------------------------------------------------------------------===//

/// Drive `session::handshake` to completion with poll(2), as the reactor does with POLL_ADD.
inline std::error_code drive_handshake(io::tls::session& s, int fd, int timeout_ms = 5000)
{
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  for (;;) {
    std::error_code ec;
    auto const st = s.handshake(ec);
    if (st == io::tls::handshake_status::done) { return {}; }
    if (st == io::tls::handshake_status::failed) { return ec; }
    auto const left = std::chrono::duration_cast<std::chrono::milliseconds>(
                        deadline - std::chrono::steady_clock::now())
                        .count();
    short const ev = st == io::tls::handshake_status::want_read ? POLLIN : POLLOUT;
    if (left <= 0 || !wait_fd(fd, ev, static_cast<int>(left))) {
      return std::make_error_code(std::errc::timed_out);
    }
  }
}

/// A wrapping ciphertext buffer, as the transport's ring: views come in one or two pieces.
class cipher_ring {
 public:
  explicit cipher_ring(std::size_t capacity) : _buf(capacity) {}

  [[nodiscard]] std::size_t size() const noexcept { return _size; }
  [[nodiscard]] bool empty() const noexcept { return _size == 0; }
  [[nodiscard]] std::size_t free_space() const noexcept { return _buf.size() - _size; }

  [[nodiscard]] io::tls::ciphertext_view view() const noexcept
  {
    std::size_t const end = _head + _size;
    if (end <= _buf.size()) { return {{_buf.data() + _head, _size}, {}}; }
    return {{_buf.data() + _head, _buf.size() - _head}, {_buf.data(), end - _buf.size()}};
  }
  void consume(std::size_t n) noexcept
  {
    _head = (_head + n) % _buf.size();
    _size -= n;
    if (_size == 0) { _head = 0; }
  }
  /// Append @p data (truncated to the free space); @return bytes appended.
  std::size_t append(std::span<std::byte const> data) noexcept
  {
    std::size_t const n = std::min(data.size(), free_space());
    for (std::size_t i = 0; i < n; ++i) {
      _buf[(_head + _size + i) % _buf.size()] = data[i];
    }
    _size += n;
    return n;
  }
  /// One readv() of the free space (tail, then front); @return its result.
  ssize_t recv_from(int fd) noexcept
  {
    std::array<iovec, 2> iov{};
    std::size_t const t    = (_head + _size) % _buf.size();
    std::size_t const free = free_space();
    if (free == 0) { return 0; }
    std::size_t const first = std::min(free, _buf.size() - t);
    iov[0]                  = iovec{_buf.data() + t, first};
    iov[1]                  = iovec{_buf.data(), free - first};
    ssize_t const r         = ::readv(fd, iov.data(), free > first ? 2 : 1);
    if (r > 0) { _size += static_cast<std::size_t>(r); }
    return r;
  }

 private:
  std::vector<std::byte> _buf;
  std::size_t _head{0};
  std::size_t _size{0};
};

/// User-space receive over a `record_opener` (tier 1 or tier 2) from a non-blocking socket.
struct user_rx {
  int fd{-1};
  cipher_ring ring{24 * 1024};  ///< small: records straddle the wrap often
  io::tls::record_opener* opener{nullptr};
  std::function<bool()> on_key_update{};  ///< e.g. request the reply and flush it
  unsigned key_updates{0};
  bool closed{false};
  bool failed{false};
  bool eof{false};

  /// Fill @p dst in windows of at most @p window bytes (split over two iovecs).
  bool receive(std::span<std::byte> dst, std::size_t window = 64 * 1024, int timeout_ms = 10000)
  {
    std::size_t got = 0;
    while (got < dst.size() || dst.empty()) {
      std::size_t const w = std::min(window, dst.size() - got);
      std::array<iovec, 2> iov{iovec{dst.data() + got, w / 2},
                               iovec{dst.data() + got + w / 2, w - w / 2}};
      auto const r = opener->open(ring.view(), iov, 0, false);
      ring.consume(r.consumed);
      got += r.plaintext;
      if (r.stop == io::tls::open_stop::failed) {
        failed = true;
        return false;
      }
      if (r.stop == io::tls::open_stop::closed) {
        closed = true;
        return got == dst.size();
      }
      if (r.stop == io::tls::open_stop::key_update) {
        ++key_updates;
        if (on_key_update && !on_key_update()) { return false; }
        continue;
      }
      if (dst.empty() && r.stop == io::tls::open_stop::starved) { return true; }
      if (got == dst.size()) { return true; }
      if (opener->progress_possible(ring.view())) { continue; }
      if (!wait_fd(fd, POLLIN, timeout_ms)) { return false; }
      ssize_t const n = ring.recv_from(fd);
      if (n == 0) {
        eof = true;
        return false;
      }
      if (n < 0 && errno != EAGAIN && errno != EINTR) { return false; }
    }
    return true;
  }

  /// Open with an empty window until close_notify: control records are consumed, any
  /// application byte stays pending (and fails the wait).
  bool wait_closed(int timeout_ms = 10000)
  {
    for (;;) {
      auto const r = opener->open(ring.view(), {}, 0, false);
      ring.consume(r.consumed);
      if (r.stop == io::tls::open_stop::closed) {
        closed = true;
        return true;
      }
      if (r.stop == io::tls::open_stop::failed) {
        failed = true;
        return false;
      }
      if (r.stop == io::tls::open_stop::key_update) {
        ++key_updates;
        if (on_key_update && !on_key_update()) { return false; }
        continue;
      }
      if (opener->pending() != 0) { return false; }  // application data before close_notify
      if (opener->progress_possible(ring.view())) { continue; }
      if (!wait_fd(fd, POLLIN, timeout_ms)) { return false; }
      ssize_t const n = ring.recv_from(fd);
      if (n == 0) {
        eof = true;
        return false;
      }
      if (n < 0 && errno != EAGAIN && errno != EINTR) { return false; }
    }
  }
};

/// Seal @p data through @p sealer and send it; an empty @p data runs one flush pass.
inline bool send_sealed(int fd,
                        io::tls::record_sealer& sealer,
                        std::span<std::byte const> data,
                        unsigned max_records     = 3,
                        std::size_t out_capacity = 64 * 1024)
{
  std::vector<std::byte> out(out_capacity);
  iovec const in{const_cast<std::byte*>(data.data()), data.size()};
  std::span<iovec const> const ins =
    data.empty() ? std::span<iovec const>{} : std::span<iovec const>(&in, 1);
  std::size_t done = 0;
  for (int i = 0; i < 100000; ++i) {
    auto const r = sealer.seal(ins, done, out, max_records);
    if (r.error) { return false; }
    if (r.ciphertext != 0 && !send_all(fd, std::span<std::byte const>(out.data(), r.ciphertext))) {
      return false;
    }
    done += r.plaintext;
    if (done == data.size() && !r.output_full) { return true; }
    if (r.plaintext == 0 && r.ciphertext == 0 && !r.output_full) { return false; }
  }
  return false;
}

}  // namespace cucascade::test::tls
