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

#include "io/rest/net/uring_loop.hpp"

#include <openssl/ssl.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

/**
 * @file net_test_support.hpp
 * @brief Test support of the net layer (gpuio tests/unit/net/net_test_support, ported): a loop
 *        driven by the test thread, deterministic payloads, a self-signed localhost certificate,
 *        and an in-process HTTP(S) server with scripted misbehaviour (silent closes, idle closes,
 *        `Connection: close` after N requests, interim 1xx heads, short bodies, error bodies,
 *        mid-body TLS tickets / KeyUpdates, close_notify, silent handshakes).
 */

namespace net_test {

[[nodiscard]] std::uint64_t now_ns() noexcept;

/// @return true when this process can create the REST ring (io_uring allowed, DEFER_TASKRUN).
[[nodiscard]] bool uring_available() noexcept;

// --------------------------------------------------------------------------------- the loop ----

/// @brief A `uring_loop` owned and driven by the test thread.
class test_loop {
 public:
  explicit test_loop(cucascade::io::net::uring_loop_config cfg = {});
  ~test_loop();
  test_loop(test_loop const&)            = delete;
  test_loop& operator=(test_loop const&) = delete;

  [[nodiscard]] cucascade::io::net::uring_loop& loop() noexcept { return _loop; }
  [[nodiscard]] cucascade::io::net::loop_env env() noexcept
  {
    return cucascade::io::net::loop_env{_loop};
  }

  /// @brief One pass bounded by @p max_wait_ns.
  void run_once(std::uint64_t max_wait_ns = 5'000'000ULL) noexcept { _loop.run_pass(max_wait_ns); }
  /// @brief Passes until @p pred holds or @p timeout. @return pred().
  template <class Pred>
  bool run_until(Pred&& pred, std::chrono::milliseconds timeout = std::chrono::seconds(30))
  {
    auto const end = std::chrono::steady_clock::now() + timeout;
    while (!pred()) {
      if (std::chrono::steady_clock::now() > end) return pred();
      run_once();
    }
    return true;
  }

 private:
  cucascade::io::net::uring_loop _loop;
};

// ----------------------------------------------------------------------------------- payload ---

/// Deterministic byte at absolute offset @p o (any object served by the test server).
[[nodiscard]] inline std::byte pattern_byte(std::uint64_t seed, std::uint64_t o) noexcept
{
  std::uint64_t x = (o >> 3) * 0x9E3779B97F4A7C15ULL + seed;
  x ^= x >> 29;
  x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 32;
  return static_cast<std::byte>(x >> ((o & 7) * 8));
}
void fill_pattern(std::span<std::byte> out, std::uint64_t seed, std::uint64_t offset) noexcept;
/// @return the first mismatching offset or npos.
[[nodiscard]] std::size_t check_pattern(std::span<std::byte const> data,
                                        std::uint64_t seed,
                                        std::uint64_t offset) noexcept;

// ----------------------------------------------------------------------------- certificates ----

struct test_cert {
  std::string cert_pem, key_pem;
};
/// RSA-2048 self-signed CA:TRUE for CN=localhost, SAN DNS:localhost, IP:127.0.0.1 (cached).
[[nodiscard]] test_cert const& localhost_cert();
/// Another such certificate (a different key): clients trusting only `localhost_cert` reject it.
[[nodiscard]] test_cert const& untrusted_cert();

/// @return true when the kernel TLS module is loaded (/proc/net/tls_stat readable).
[[nodiscard]] bool ktls_available();

// ------------------------------------------------------------------------ in-process server ----

/// Scripted HTTP/1.1 server on 127.0.0.1:<ephemeral>, one thread per connection, blocking I/O.
/// `GET /<size>` (or `HEAD`) serves `size` pattern bytes (seed = server seed) with
/// Content-Length; `Range: bytes=a-b` yields 206. Everything else 404.
struct server_options {
  bool tls{false};
  int min_version{TLS1_2_VERSION};
  int max_version{TLS1_3_VERSION};
  std::string cipher_list{};   ///< TLS 1.2 (empty = OpenSSL default)
  std::string ciphersuites{};  ///< TLS 1.3 (empty = OpenSSL default)
  bool untrusted{false};       ///< present `untrusted_cert()` instead of `localhost_cert()`
  int num_tickets{2};          ///< TLS 1.3 tickets right after the handshake (nginx: 2)
  std::uint64_t seed{0x13};
  /// Mid-body TLS control records (body offset at which they are sent; ~0 = never).
  std::uint64_t ticket_at{~0ULL};
  std::uint64_t key_update_at{~0ULL};
  bool key_update_requested{true};
  /// Stop the body at this offset and close with a bare FIN (no close_notify). ~0 = never.
  std::uint64_t fin_at{~0ULL};
  /// Send close_notify right behind the last body record, then wait for the client to close.
  bool close_notify_at_end{false};
  /// Close the connection silently (no Connection: close) after this many responses (0 = never).
  unsigned close_after_requests{0};
  /// Answer the Nth request of every connection with `Connection: close`, then close (the S3
  /// per-connection request limit, 100 there; 0 = never).
  unsigned request_limit{0};
  /// Close an idle connection after this long (S3's ~5 s idle timeout; 0 = never).
  std::chrono::milliseconds idle_close{0};
  /// Accept TCP but never answer (handshake, header and connect-deadline tests).
  bool silent{false};
  /// Send `Connection: close` on every response.
  bool connection_close{false};
  /// Write the body with this chunk (record) size.
  std::size_t write_chunk{16384};
  /// Interim `100 Continue` / `102 Processing` heads sent before every final head.
  unsigned interim_heads{0};
  /// Promise this many body bytes more than are sent (Content-Length mismatch), then FIN.
  std::uint64_t short_body_by{0};
  /// Answer every GET with this status and an S3 XML error body (0 = off).
  int error_status{0};
  std::string error_code{"SlowDown"};
  /// Sleep before the response head / body (stall tests).
  std::chrono::milliseconds head_delay{0};
};

class test_server {
 public:
  explicit test_server(server_options opts);
  ~test_server();
  test_server(test_server const&)            = delete;
  test_server& operator=(test_server const&) = delete;

  [[nodiscard]] bool ok() const noexcept { return _port != 0; }
  [[nodiscard]] std::uint16_t port() const noexcept { return _port; }
  [[nodiscard]] std::uint64_t seed() const noexcept { return _opts.seed; }
  [[nodiscard]] unsigned connections() const noexcept { return _accepted.load(); }
  [[nodiscard]] unsigned requests() const noexcept { return _requests.load(); }
  [[nodiscard]] unsigned handshakes() const noexcept { return _handshakes.load(); }
  [[nodiscard]] unsigned closes() const noexcept { return _closes.load(); }
  /// Close every open connection now (FIN), as a killed worker would.
  void close_all();
  [[nodiscard]] std::string error() const;

 private:
  void accept_loop();
  void serve(int fd, unsigned conn_no);
  server_options _opts;
  SSL_CTX* _ctx{nullptr};
  int _lfd{-1};
  std::uint16_t _port{0};
  std::atomic<bool> _stop{false};
  std::thread _acceptor;
  std::mutex _m;
  std::vector<std::thread> _workers;
  std::vector<int> _open;
  std::atomic<unsigned> _accepted{0}, _requests{0}, _handshakes{0}, _closes{0};
  std::string _error;
};

/// @return a port on 127.0.0.1 nobody listens on (bound, then closed: connects are refused).
[[nodiscard]] std::uint16_t closed_port();

}  // namespace net_test
