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

#include "net_test_support.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <openssl/err.h>
#include <openssl/pem.h>
#include <openssl/x509v3.h>
#include <poll.h>
#include <signal.h>  // NOLINT(modernize-deprecated-headers): POSIX SIGPIPE disposition (tests only)
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <system_error>

namespace net_test {

std::uint64_t now_ns() noexcept
{
  timespec ts{};
  ::clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
         static_cast<std::uint64_t>(ts.tv_nsec);
}

bool uring_available() noexcept
{
  return cucascade::io::net::cached_kernel_features().meets_floor();
}

// ================================================================================================
// test_loop
// ================================================================================================

test_loop::test_loop(cucascade::io::net::uring_loop_config cfg) : _loop(std::move(cfg))
{
  if (auto ec = _loop.start()) throw std::system_error(ec, "uring_loop::start");
}

test_loop::~test_loop()
{
  if (_loop.inflight_ops() != 0) {
    _loop.cancel_all();
    (void)_loop.drain([]() noexcept { return true; }, 2'000'000'000ULL);
  }
  _loop.close();
}

// ================================================================================================
// Payload
// ================================================================================================

void fill_pattern(std::span<std::byte> out, std::uint64_t seed, std::uint64_t offset) noexcept
{
  std::size_t i = 0;
  // Head up to 8-byte alignment of the absolute offset, then whole words.
  for (; i < out.size() && ((offset + i) & 7) != 0; ++i)
    out[i] = pattern_byte(seed, offset + i);
  for (; i + 8 <= out.size(); i += 8) {
    std::uint64_t x = ((offset + i) >> 3) * 0x9E3779B97F4A7C15ULL + seed;
    x ^= x >> 29;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 32;
    std::memcpy(out.data() + i, &x, 8);
  }
  for (; i < out.size(); ++i)
    out[i] = pattern_byte(seed, offset + i);
}

std::size_t check_pattern(std::span<std::byte const> data,
                          std::uint64_t seed,
                          std::uint64_t offset) noexcept
{
  std::byte chunk[65536];
  for (std::size_t i = 0; i < data.size(); i += sizeof chunk) {
    std::size_t const n = std::min(sizeof chunk, data.size() - i);
    fill_pattern(std::span<std::byte>(chunk, n), seed, offset + i);
    if (std::memcmp(chunk, data.data() + i, n) != 0) {
      for (std::size_t k = 0; k < n; ++k)
        if (chunk[k] != data[i + k]) return i + k;
    }
  }
  return static_cast<std::size_t>(-1);
}

// ================================================================================================
// Certificates
// ================================================================================================

namespace {

std::string bio_string(BIO* b)
{
  char* data   = nullptr;
  long const n = BIO_get_mem_data(b, &data);
  std::string s(data, static_cast<std::size_t>(n));
  BIO_free(b);
  return s;
}

void add_ext(X509* x, int nid, char const* value)
{
  X509V3_CTX ctx;
  X509V3_set_ctx_nodb(&ctx);
  X509V3_set_ctx(&ctx, x, x, nullptr, nullptr, 0);
  X509_EXTENSION* e = X509V3_EXT_conf_nid(nullptr, &ctx, nid, value);
  if (e != nullptr) {
    X509_add_ext(x, e, -1);
    X509_EXTENSION_free(e);
  }
}

test_cert make_cert()
{
  EVP_PKEY* key = EVP_RSA_gen(2048);
  if (key == nullptr) throw std::runtime_error("key generation failed");
  X509* x = X509_new();
  X509_set_version(x, X509_VERSION_3);
  ASN1_INTEGER_set(X509_get_serialNumber(x), 1313);
  X509_gmtime_adj(X509_getm_notBefore(x), -3600L);
  X509_gmtime_adj(X509_getm_notAfter(x), 7L * 86400L);
  X509_set_pubkey(x, key);
  X509_NAME* name = X509_get_subject_name(x);
  X509_NAME_add_entry_by_txt(
    name, "CN", MBSTRING_ASC, reinterpret_cast<unsigned char const*>("localhost"), -1, -1, 0);
  X509_set_issuer_name(x, name);
  add_ext(x, NID_basic_constraints, "critical,CA:TRUE");
  add_ext(x, NID_subject_key_identifier, "hash");
  add_ext(x, NID_subject_alt_name, "DNS:localhost,IP:127.0.0.1");
  if (X509_sign(x, key, EVP_sha256()) == 0) throw std::runtime_error("X509_sign failed");
  test_cert out;
  BIO* cb = BIO_new(BIO_s_mem());
  PEM_write_bio_X509(cb, x);
  out.cert_pem = bio_string(cb);
  BIO* kb      = BIO_new(BIO_s_mem());
  PEM_write_bio_PrivateKey(kb, key, nullptr, nullptr, 0, nullptr, nullptr);
  out.key_pem = bio_string(kb);
  X509_free(x);
  EVP_PKEY_free(key);
  return out;
}

}  // namespace

test_cert const& localhost_cert()
{
  static test_cert const c = make_cert();
  return c;
}

test_cert const& untrusted_cert()
{
  static test_cert const c = make_cert();
  return c;
}

bool ktls_available() { return ::access("/proc/net/tls_stat", R_OK) == 0; }

std::uint16_t closed_port()
{
  int const fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  sockaddr_in a{};
  a.sin_family      = AF_INET;
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  ::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof a);
  socklen_t al = sizeof a;
  ::getsockname(fd, reinterpret_cast<sockaddr*>(&a), &al);
  ::close(fd);
  return ntohs(a.sin_port);
}

// ================================================================================================
// test_server
// ================================================================================================

namespace {

bool write_all(int fd, SSL* ssl, char const* p, std::size_t n)
{
  while (n != 0) {
    if (ssl != nullptr) {
      std::size_t w = 0;
      if (SSL_write_ex(ssl, p, n, &w) != 1) return false;
      p += w;
      n -= w;
    } else {
      ssize_t const w = ::send(fd, p, n, MSG_NOSIGNAL);
      if (w <= 0) {
        if (w < 0 && errno == EINTR) continue;
        return false;
      }
      p += w;
      n -= static_cast<std::size_t>(w);
    }
  }
  return true;
}

/// @return bytes read (0 = EOF, < 0 = error).
long read_some(int fd, SSL* ssl, char* p, std::size_t n)
{
  if (ssl != nullptr) {
    std::size_t r = 0;
    if (SSL_read_ex(ssl, p, n, &r) == 1) return static_cast<long>(r);
    int const e = SSL_get_error(ssl, 0);
    return e == SSL_ERROR_ZERO_RETURN ? 0 : -1;
  }
  return static_cast<long>(::recv(fd, p, n, 0));
}

}  // namespace

test_server::test_server(server_options opts) : _opts(std::move(opts))
{
  ::signal(SIGPIPE, SIG_IGN);  // test process only (the library never does this)
  if (_opts.tls) {
    _ctx = SSL_CTX_new(TLS_server_method());
    SSL_CTX_set_min_proto_version(_ctx, _opts.min_version);
    SSL_CTX_set_max_proto_version(_ctx, _opts.max_version);
    if (!_opts.cipher_list.empty()) SSL_CTX_set_cipher_list(_ctx, _opts.cipher_list.c_str());
    if (!_opts.ciphersuites.empty()) SSL_CTX_set_ciphersuites(_ctx, _opts.ciphersuites.c_str());
    SSL_CTX_set_num_tickets(_ctx, static_cast<std::size_t>(std::max(0, _opts.num_tickets)));
    SSL_CTX_set_options(_ctx, SSL_OP_NO_RENEGOTIATION);
    test_cert const& cert = _opts.untrusted ? untrusted_cert() : localhost_cert();
    BIO* cb = BIO_new_mem_buf(cert.cert_pem.data(), static_cast<int>(cert.cert_pem.size()));
    X509* x = PEM_read_bio_X509(cb, nullptr, nullptr, nullptr);
    BIO_free(cb);
    BIO* kb     = BIO_new_mem_buf(cert.key_pem.data(), static_cast<int>(cert.key_pem.size()));
    EVP_PKEY* k = PEM_read_bio_PrivateKey(kb, nullptr, nullptr, nullptr);
    BIO_free(kb);
    SSL_CTX_use_certificate(_ctx, x);
    SSL_CTX_use_PrivateKey(_ctx, k);
    X509_free(x);
    EVP_PKEY_free(k);
    static unsigned char const alpn[] = "\x08http/1.1";
    SSL_CTX_set_alpn_select_cb(
      _ctx,
      [](SSL*,
         unsigned char const** out,
         unsigned char* outlen,
         unsigned char const* in,
         unsigned inlen,
         void*) -> int {
        unsigned char* o = nullptr;
        if (SSL_select_next_proto(&o, outlen, alpn, sizeof alpn - 1, in, inlen) !=
            OPENSSL_NPN_NEGOTIATED)
          return SSL_TLSEXT_ERR_NOACK;
        *out = o;
        return SSL_TLSEXT_ERR_OK;
      },
      nullptr);
  }
  _lfd    = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  int one = 1;
  ::setsockopt(_lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in a{};
  a.sin_family      = AF_INET;
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (::bind(_lfd, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0 || ::listen(_lfd, 256) != 0) {
    _error = "bind/listen failed";
    return;
  }
  socklen_t al = sizeof a;
  ::getsockname(_lfd, reinterpret_cast<sockaddr*>(&a), &al);
  _port     = ntohs(a.sin_port);
  _acceptor = std::thread([this] { accept_loop(); });
}

test_server::~test_server()
{
  _stop.store(true);
  if (_lfd >= 0) ::shutdown(_lfd, SHUT_RDWR);
  if (_acceptor.joinable()) _acceptor.join();
  close_all();
  std::vector<std::thread> w;
  {
    std::lock_guard g(_m);
    w.swap(_workers);
  }
  for (auto& t : w)
    t.join();
  if (_lfd >= 0) ::close(_lfd);
  if (_ctx != nullptr) SSL_CTX_free(_ctx);
}

std::string test_server::error() const { return _error; }

void test_server::close_all()
{
  std::lock_guard g(_m);
  for (int fd : _open)
    ::shutdown(fd, SHUT_RDWR);
}

void test_server::accept_loop()
{
  for (;;) {
    int const fd = ::accept4(_lfd, nullptr, nullptr, SOCK_CLOEXEC);
    if (fd < 0) {
      if (_stop.load()) return;
      if (errno == EINTR) continue;
      return;
    }
    if (_stop.load()) {
      ::close(fd);
      return;
    }
    unsigned const no = ++_accepted;
    std::lock_guard g(_m);
    _open.push_back(fd);
    _workers.emplace_back([this, fd, no] { serve(fd, no); });
  }
}

void test_server::serve(int fd, unsigned conn_no)
{
  (void)conn_no;
  int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  SSL* ssl    = nullptr;
  auto finish = [&] {
    if (ssl != nullptr) {
      SSL_set_quiet_shutdown(ssl, 1);
      SSL_free(ssl);
    }
    ::shutdown(fd, SHUT_RDWR);
    ++_closes;
    {
      std::lock_guard g(_m);
      std::erase(_open, fd);
    }
    ::close(fd);
  };
  if (_opts.silent) {
    char b[4096];
    while (!_stop.load() && ::recv(fd, b, sizeof b, 0) > 0) {}
    finish();
    return;
  }
  if (_opts.tls) {
    ssl = SSL_new(_ctx);
    SSL_set_fd(ssl, fd);
    if (SSL_accept(ssl) != 1) {
      finish();
      return;
    }
    ++_handshakes;
  }
  std::string req;
  std::vector<char> body;
  unsigned served = 0;
  char buf[16384];
  for (;;) {
    // Read one request head (idle_close: poll with timeout first).
    std::size_t end = std::string::npos;
    while ((end = req.find("\r\n\r\n")) == std::string::npos) {
      if (_opts.idle_close.count() > 0 && (ssl == nullptr || SSL_pending(ssl) == 0)) {
        pollfd p{fd, POLLIN, 0};
        int const r = ::poll(&p, 1, static_cast<int>(_opts.idle_close.count()));
        if (r == 0) {  // idle too long: close (FIN), like S3 after ~5 s
          finish();
          return;
        }
      }
      long const n = read_some(fd, ssl, buf, sizeof buf);
      if (n <= 0) {
        finish();
        return;
      }
      req.append(buf, static_cast<std::size_t>(n));
    }
    std::string const head = req.substr(0, end + 4);
    req.erase(0, end + 4);
    ++_requests;
    bool const is_head     = head.starts_with("HEAD ");
    std::size_t const sp1  = head.find(' ');
    std::size_t const sp2  = head.find(' ', sp1 + 1);
    std::string const path = head.substr(sp1 + 1, sp2 - sp1 - 1);
    std::uint64_t size     = 0;
    bool found             = path.size() > 1;
    for (std::size_t i = 1; i < path.size() && found; ++i) {
      if (path[i] < '0' || path[i] > '9')
        found = false;
      else
        size = size * 10 + static_cast<std::uint64_t>(path[i] - '0');
    }
    std::uint64_t first = 0, last = size == 0 ? 0 : size - 1;
    bool ranged = false;
    if (auto r = head.find("Range: bytes="); r != std::string::npos && found && size != 0) {
      ranged = true;
      first  = std::strtoull(head.c_str() + r + 13, nullptr, 10);
      auto d = head.find('-', r + 13);
      last   = std::strtoull(head.c_str() + d + 1, nullptr, 10);
      if (last >= size) last = size - 1;
    }
    if (_opts.head_delay.count() > 0) std::this_thread::sleep_for(_opts.head_delay);
    bool const limit_close = _opts.request_limit != 0 && served + 1 >= _opts.request_limit;
    bool const close_now =
      _opts.connection_close || limit_close ||
      (_opts.close_after_requests != 0 && served + 1 >= _opts.close_after_requests);
    std::string resp;
    for (unsigned i = 0; i < _opts.interim_heads; ++i)
      resp += (i % 2 == 0) ? "HTTP/1.1 100 Continue\r\n\r\n"
                           : "HTTP/1.1 102 Processing\r\nX-Interim: 1\r\n\r\n";
    std::uint64_t len = 0;
    std::string error_body;
    if (_opts.error_status != 0 && !is_head) {
      error_body = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<Error><Code>" + _opts.error_code +
                   "</Code><Message>scripted</Message><RequestId>T</RequestId></Error>";
      resp += "HTTP/1.1 " + std::to_string(_opts.error_status) +
              " Scripted\r\nContent-Type: application/xml\r\nContent-Length: " +
              std::to_string(error_body.size()) + "\r\n";
      found = false;
    } else if (!found) {
      resp += "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n";
    } else {
      len = size == 0 ? 0 : last - first + 1;
      if (ranged)
        resp += "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes " + std::to_string(first) +
                "-" + std::to_string(last) + "/" + std::to_string(size) + "\r\n";
      else
        resp += "HTTP/1.1 200 OK\r\n";
      resp +=
        "Content-Length: " + std::to_string(len + _opts.short_body_by) + "\r\nETag: \"t13\"\r\n";
    }
    // close_after_requests is SILENT (no Connection: close): the client finds out by reuse.
    if (_opts.connection_close || limit_close) resp += "Connection: close\r\n";
    resp += "\r\n";
    resp += error_body;
    if (!write_all(fd, ssl, resp.data(), resp.size())) {
      finish();
      return;
    }
    if (found && !is_head) {
      body.resize(_opts.write_chunk);
      std::uint64_t sent = 0;
      while (sent < len) {
        std::uint64_t n = std::min<std::uint64_t>(_opts.write_chunk, len - sent);
        // Stop at a scheduled control-record offset.
        for (std::uint64_t at : {_opts.ticket_at, _opts.key_update_at, _opts.fin_at})
          if (at > sent && at < sent + n) n = at - sent;
        bool const last_piece = sent + n == len;
        if (last_piece && ssl != nullptr && _opts.close_notify_at_end)
          ::setsockopt(fd, IPPROTO_TCP, TCP_CORK, &one, sizeof one);  // record + close_notify
        fill_pattern(std::span<std::byte>(reinterpret_cast<std::byte*>(body.data()), n),
                     _opts.seed,
                     first + sent);
        if (!write_all(fd, ssl, body.data(), n)) {
          finish();
          return;
        }
        sent += n;
        if (ssl != nullptr && sent == _opts.ticket_at) {
          SSL_new_session_ticket(ssl);
          SSL_do_handshake(ssl);  // flushes the ticket record now
        }
        if (ssl != nullptr && sent == _opts.key_update_at) {
          SSL_key_update(
            ssl,
            _opts.key_update_requested ? SSL_KEY_UPDATE_REQUESTED : SSL_KEY_UPDATE_NOT_REQUESTED);
          SSL_do_handshake(ssl);
        }
        if (sent == _opts.fin_at) {  // truncated body: bare FIN, no close_notify
          finish();
          return;
        }
      }
      if (_opts.short_body_by != 0) {  // fewer bytes than Content-Length, then FIN
        finish();
        return;
      }
      if (ssl != nullptr && _opts.close_notify_at_end) {
        (void)SSL_shutdown(ssl);  // close_notify behind the last record (corked)
        int zero = 0;
        ::setsockopt(fd, IPPROTO_TCP, TCP_CORK, &zero, sizeof zero);
        // No FIN until the client closes: the client must act on the close_notify itself.
        char b[512];
        while (!_stop.load() && ::recv(fd, b, sizeof b, 0) > 0) {}
        finish();
        return;
      }
    }
    ++served;
    if (close_now) {
      finish();
      return;
    }
  }
}

}  // namespace net_test
