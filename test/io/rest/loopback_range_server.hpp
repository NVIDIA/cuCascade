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

#include "../tls/tls_test_pki.hpp"

#include <cucascade/io/rest/authorizer.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <pthread.h>
#include <signal.h>  // NOLINT(modernize-deprecated-headers): POSIX pthread_sigmask
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * @file loopback_range_server.hpp
 * @brief Raw-socket HTTP/1.1 loopback server for the REST read tests (header-only).
 *
 * Ranged / suffix / full GET, HEAD and ListObjectsV2 of one fixture object (any key), with the
 * fault policies of @ref cucascade::test::range_fault_policy and per-key scripts (including a
 * chunked-framed GET body).  HTTPS: with @ref cucascade::test::loopback_tls enabled every
 * accepted socket is wrapped in an OpenSSL server `SSL*` (blocking `SSL_accept`) presenting the
 * test PKI's server certificate (`io/tls/tls_test_pki.hpp`); the same request handling runs over
 * `SSL_read` / `SSL_write`, and a body may carry a mid-body KeyUpdate or end without
 * close_notify.
 */

namespace cucascade::test {

class range_response_control {
 public:
  struct progress {
    std::size_t headers{0};
    std::size_t body_bytes{0};
    std::size_t body_writes{0};
    std::size_t held_responses{0};
    std::chrono::steady_clock::time_point first_body_write{};
    std::chrono::steady_clock::time_point last_body_write{};
  };

  [[nodiscard]] progress snapshot() const
  {
    std::scoped_lock lock{_mutex};
    return _progress;
  }

  bool wait_for_headers(std::size_t count, std::chrono::milliseconds timeout) const
  {
    return wait_for(timeout, [count](progress const& p) { return p.headers >= count; });
  }

  bool wait_for_body_bytes(std::size_t count, std::chrono::milliseconds timeout) const
  {
    return wait_for(timeout, [count](progress const& p) { return p.body_bytes >= count; });
  }

  bool wait_for_holds(std::size_t count, std::chrono::milliseconds timeout) const
  {
    return wait_for(timeout, [count](progress const& p) { return p.held_responses >= count; });
  }

  void headers_sent()
  {
    {
      std::scoped_lock lock{_mutex};
      ++_progress.headers;
    }
    _cv.notify_all();
  }

  void body_sent(std::size_t bytes)
  {
    if (bytes == 0) { return; }
    {
      std::scoped_lock lock{_mutex};
      auto const now = std::chrono::steady_clock::now();
      if (_progress.body_writes == 0) { _progress.first_body_write = now; }
      _progress.last_body_write = now;
      _progress.body_bytes += bytes;
      ++_progress.body_writes;
    }
    _cv.notify_all();
  }

  bool hold()
  {
    std::unique_lock lock{_mutex};
    ++_progress.held_responses;
    _cv.notify_all();
    _cv.wait(lock, [&] { return _released || _stopped; });
    --_progress.held_responses;
    return !_stopped;
  }

  void release_all()
  {
    {
      std::scoped_lock lock{_mutex};
      _released = true;
    }
    _cv.notify_all();
  }

  void stop()
  {
    {
      std::scoped_lock lock{_mutex};
      _stopped = true;
    }
    _cv.notify_all();
  }

 private:
  template <typename Predicate>
  bool wait_for(std::chrono::milliseconds timeout, Predicate ready) const
  {
    std::unique_lock lock{_mutex};
    _cv.wait_for(lock, timeout, [&] { return ready(_progress) || _stopped; });
    return ready(_progress);
  }

  mutable std::mutex _mutex;
  mutable std::condition_variable _cv;
  progress _progress;
  bool _released{false};
  bool _stopped{false};
};

struct response_release_guard {
  std::shared_ptr<range_response_control> control;
  ~response_release_guard()
  {
    if (control) { control->release_all(); }
  }
};

struct response_body_policy {
  /// Send the body in pieces of this many bytes; 0 sends it in one write.
  std::size_t chunk_bytes{0};
  /// Pause between pieces; requires a non-zero `chunk_bytes`.
  std::chrono::milliseconds chunk_delay{0};
  /// Stop after this many body bytes until `control->release_all()`; requires `control`.
  std::optional<std::size_t> hold_after_bytes;
  /// Observes header/body progress and owns the hold; the server rejects a
  /// policy that violates these rules in its constructor (`invalid_argument`).
  std::shared_ptr<range_response_control> control;
  /// Close the connection (no close_notify) once this many body bytes were sent.
  std::optional<std::size_t> drop_after_bytes;
  /// HTTPS (TLS 1.3): send a KeyUpdate (update_requested) once this many body bytes were sent.
  std::optional<std::size_t> key_update_after_bytes;
};

struct range_fault_policy {
  std::size_t fail_first_gets{0};
  bool fail_all_gets{false};
  int fail_status{503};
  std::size_t fail_first_heads{0};
  bool fail_all_heads{false};
  int head_fail_status{503};
  std::chrono::milliseconds response_delay{0};
  /// Hold GET responses until this many GET requests have arrived; 0 disables the barrier.
  std::size_t get_response_barrier{0};
  /// Answer every ranged data GET with a 200 carrying the whole object; its ETag
  /// comes from the script or `successful_get_etag`.
  bool ignore_range_with_200{false};
  bool fail_range_with_416{false};
  bool malformed_content_range{false};
  bool omit_successful_content_range{false};
  std::string failed_get_etag;
  std::string failed_get_retry_after;
  std::string failed_head_etag;
  std::string failed_head_retry_after;
  std::string successful_get_etag;
  std::string successful_head_etag;
  std::string interim_get_etag;
  std::string interim_get_content_range;
  std::string interim_get_retry_after;
  std::string interim_head_etag;
  std::string interim_head_retry_after;
  /// Failed GETs advertise a 64-byte error body, send 5 bytes, then close.
  bool truncate_error_body{false};
  std::optional<std::string> content_range;
  response_body_policy body;
};

struct listed_object {
  std::string key;
  std::uint64_t size{0};
};

struct scripted_response {
  /// Zero selects the server's normal successful response for this method.
  int status{0};
  std::chrono::milliseconds delay{0};
  std::optional<std::string> etag;
  std::optional<std::string> retry_after;
  bool malformed_content_range{false};
  /// XORed into every body byte of a successful GET at this step, so that
  /// successive versions of one key can carry different bytes.
  std::uint8_t body_xor{0};
  std::optional<std::string> content_range;
  std::optional<response_body_policy> body;
  /// Answer this ranged GET with a 200 carrying the whole object (as
  /// `range_fault_policy::ignore_range_with_200` does for every GET).
  bool ignore_range{false};
  /// Frame a successful GET body with `Transfer-Encoding: chunked` (one chunk, then the last
  /// chunk) instead of `Content-Length`, as CloudFront or S3 Object Lambda may.  GET only.
  bool chunked_transfer_encoding{false};
  /// Send this many extra bytes after a successful GET body (a desynchronising server).
  std::size_t trailing_garbage{0};
};

struct key_response_script {
  std::vector<scripted_response> gets;
  std::vector<scripted_response> heads;
};

/// HTTPS mode of @ref loopback_range_server.
struct loopback_tls {
  bool enabled{false};
  /// Server context (owned by the server, freed with it); null: the test PKI's default server
  /// context (`tls::make_server_ctx(tls::default_pki())`).
  SSL_CTX* ctx{nullptr};
};

/// One accepted connection: plain TCP, or TLS over it (HTTPS mode).
class loopback_connection {
 public:
  loopback_connection(int fd, SSL* ssl) noexcept : _fd(fd), _ssl(ssl) {}

  [[nodiscard]] int fd() const noexcept { return _fd; }
  [[nodiscard]] bool tls() const noexcept { return _ssl != nullptr; }

  /// @return bytes read (> 0), 0 at the end of the stream, < 0 on an error or a timeout.
  ssize_t recv_some(char* buf, std::size_t n)
  {
    if (_ssl == nullptr) {
      for (;;) {
        ssize_t const r = ::recv(_fd, buf, n, 0);
        if (r < 0 && errno == EINTR) { continue; }
        return r;
      }
    }
    ERR_clear_error();
    int const r = SSL_read(_ssl, buf, static_cast<int>(std::min<std::size_t>(n, INT_MAX)));
    if (r > 0) { return r; }
    return SSL_get_error(_ssl, r) == SSL_ERROR_ZERO_RETURN ? 0 : -1;
  }

  /// @return the bytes sent (fewer than @p size on an error).
  std::size_t send_all(std::uint8_t const* bytes, std::size_t size)
  {
    std::size_t sent = 0;
    if (_ssl == nullptr) {
      // Fault tests can disconnect mid-response; suppress SIGPIPE so send() reports the failure.
      while (sent < size) {
        ssize_t const n = ::send(_fd, bytes + sent, size - sent, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) { continue; }
        if (n <= 0) { break; }
        sent += static_cast<std::size_t>(n);
      }
      return sent;
    }
    while (sent < size) {
      ERR_clear_error();
      int const n = SSL_write(
        _ssl, bytes + sent, static_cast<int>(std::min<std::size_t>(size - sent, 1 << 30)));
      if (n <= 0) { break; }
      sent += static_cast<std::size_t>(n);
    }
    return sent;
  }

  /// A fault drops the connection: no close_notify.
  void drop() noexcept { _dropped = true; }
  [[nodiscard]] bool dropped() const noexcept { return _dropped; }

  /// TLS: send close_notify (one-way) before the socket is closed.
  void close_notify() noexcept
  {
    if (_ssl != nullptr) {
      ERR_clear_error();
      (void)SSL_shutdown(_ssl);
      ERR_clear_error();
    }
  }

  /// TLS 1.3: schedule a KeyUpdate asking the peer to update too (sent with the next write).
  /// @return false on plain TCP, TLS 1.2 or a library refusal.
  bool key_update() noexcept
  {
    if (_ssl == nullptr || SSL_version(_ssl) != TLS1_3_VERSION) { return false; }
    ERR_clear_error();
    bool const ok = SSL_key_update(_ssl, SSL_KEY_UPDATE_REQUESTED) == 1;
    ERR_clear_error();
    return ok;
  }

 private:
  int _fd;
  SSL* _ssl;
  bool _dropped{false};
};

class loopback_range_server {
 public:
  struct get_record {
    std::vector<std::string> if_match;
    std::vector<std::string> ranges;
    bool header_authorized{false};
    bool presigned{false};
  };

  [[nodiscard]] std::vector<get_record> get_requests() const
  {
    std::scoped_lock lock{_get_records_mutex};
    return _get_records;
  }

  explicit loopback_range_server(std::vector<std::uint8_t> object,
                                 range_fault_policy fault                                     = {},
                                 std::vector<listed_object> listed                            = {},
                                 std::unordered_map<std::string, key_response_script> scripts = {},
                                 loopback_tls tls                                             = {})
    : _object(std::move(object)),
      _fault(std::move(fault)),
      _listed(std::move(listed)),
      _scripts(std::move(scripts))
  {
    if (_object.empty()) { throw std::runtime_error("loopback object must be non-empty"); }
    add_body_control(_fault.body);
    for (auto const& [key, script] : _scripts) {
      for (auto const& response : script.gets) {
        if (response.body) { add_body_control(*response.body); }
      }
    }
    if (tls.enabled) {
      _tls_ctx = tls.ctx != nullptr ? tls.ctx : tls::make_server_ctx(tls::default_pki());
    }

    _listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (_listen_fd < 0) { throw std::runtime_error("socket failed: " + errno_message()); }

    int one = 1;
    if (::setsockopt(_listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0) {
      throw std::runtime_error("setsockopt failed: " + errno_message());
    }

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = 0;
    if (::bind(_listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      throw std::runtime_error("bind failed: " + errno_message());
    }
    if (::listen(_listen_fd, 64) != 0) {
      throw std::runtime_error("listen failed: " + errno_message());
    }

    socklen_t len = sizeof(addr);
    if (::getsockname(_listen_fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
      throw std::runtime_error("getsockname failed: " + errno_message());
    }
    _port               = ntohs(addr.sin_port);
    int const listen_fd = _listen_fd;
    _thread             = std::thread([this, listen_fd] { accept_loop(listen_fd); });
  }

  ~loopback_range_server()
  {
    stop();
    if (_tls_ctx != nullptr) { SSL_CTX_free(_tls_ctx); }
  }

  void stop()
  {
    std::call_once(_stop_once, [this] {
      {
        std::scoped_lock lock{_get_response_mutex, _data_get_mutex};
        _stop.store(true, std::memory_order_relaxed);
      }
      _get_response_cv.notify_all();
      _data_get_cv.notify_all();
      for (auto const& control : _body_controls) {
        control->stop();
      }
      if (_listen_fd >= 0) {
        // shutdown() wakes the blocked accept(); close() alone does not reliably interrupt it.
        ::shutdown(_listen_fd, SHUT_RDWR);
      }
      if (_thread.joinable()) { _thread.join(); }
      {
        std::scoped_lock lock{_workers_mutex};
        for (int fd : _client_fds) {
          ::shutdown(fd, SHUT_RDWR);
        }
      }
      if (_listen_fd >= 0) {
        ::close(_listen_fd);
        _listen_fd = -1;
      }
      for (auto& worker : _workers) {
        if (worker.joinable()) { worker.join(); }
      }
    });
  }

  loopback_range_server(loopback_range_server const&)            = delete;
  loopback_range_server& operator=(loopback_range_server const&) = delete;

  /// "http://127.0.0.1:<port>", or "https://..." in HTTPS mode.
  [[nodiscard]] std::string endpoint() const
  {
    return std::string{_tls_ctx != nullptr ? "https" : "http"} +
           "://127.0.0.1:" + std::to_string(_port);
  }
  [[nodiscard]] bool https() const noexcept { return _tls_ctx != nullptr; }
  [[nodiscard]] std::uint16_t port() const noexcept { return _port; }

  [[nodiscard]] std::size_t head_count() const noexcept { return _head_count.load(); }
  [[nodiscard]] std::size_t get_count() const noexcept { return _get_count.load(); }
  [[nodiscard]] std::size_t list_count() const noexcept { return _list_count.load(); }
  [[nodiscard]] std::size_t accepted_connection_count() const noexcept
  {
    return _accepted_connection_count.load();
  }
  [[nodiscard]] std::size_t head_count(std::string_view key) const
  {
    return request_count(_head_counts_by_key, key);
  }
  [[nodiscard]] std::size_t get_count(std::string_view key) const
  {
    return request_count(_get_counts_by_key, key);
  }
  /// Data GETs in flight / at peak; suffix-range (`bytes=-N`) footer probes are
  /// not counted here (nor by wait_for_active_gets / wait_for_idle), while
  /// get_count() counts every GET.
  [[nodiscard]] std::size_t active_data_gets() const noexcept { return _active_data_gets.load(); }
  [[nodiscard]] std::size_t peak_active_gets() const noexcept { return _peak_active_gets.load(); }
  [[nodiscard]] std::size_t body_bytes_sent() const noexcept { return _body_bytes_sent.load(); }
  [[nodiscard]] std::size_t tls_handshake_count() const noexcept { return _tls_handshakes.load(); }
  [[nodiscard]] std::size_t tls_handshake_failure_count() const noexcept
  {
    return _tls_handshake_failures.load();
  }
  /// KeyUpdates sent mid-body (response_body_policy::key_update_after_bytes).
  [[nodiscard]] std::size_t key_updates_sent() const noexcept { return _key_updates_sent.load(); }
  /// Responses cut off mid-body (response_body_policy::drop_after_bytes).
  [[nodiscard]] std::size_t body_drops() const noexcept { return _body_drops.load(); }

  bool wait_for_active_gets(std::size_t count, std::chrono::milliseconds timeout)
  {
    std::unique_lock lock{_data_get_mutex};
    _data_get_cv.wait_for(
      lock, timeout, [&] { return _active_data_gets.load() >= count || _stop.load(); });
    return _active_data_gets.load() >= count;
  }

  bool wait_for_idle(std::chrono::milliseconds timeout)
  {
    std::unique_lock lock{_data_get_mutex};
    return _data_get_cv.wait_for(lock, timeout, [&] { return _active_data_gets.load() == 0; });
  }

 private:
  class data_get_guard {
   public:
    data_get_guard(loopback_range_server& server, bool active) : _server(server), _active(active)
    {
      if (!_active) { return; }
      {
        std::scoped_lock lock{_server._data_get_mutex};
        auto const current = _server._active_data_gets.fetch_add(1) + 1;
        if (current > _server._peak_active_gets.load()) {
          _server._peak_active_gets.store(current);
        }
      }
      _server._data_get_cv.notify_all();
    }

    ~data_get_guard()
    {
      if (!_active) { return; }
      {
        std::scoped_lock lock{_server._data_get_mutex};
        _server._active_data_gets.fetch_sub(1);
      }
      _server._data_get_cv.notify_all();
    }

   private:
    loopback_range_server& _server;
    bool _active;
  };

  void add_body_control(response_body_policy const& policy)
  {
    if (policy.chunk_delay.count() < 0 ||
        (policy.chunk_delay.count() > 0 && policy.chunk_bytes == 0) ||
        (policy.hold_after_bytes && !policy.control)) {
      throw std::invalid_argument("invalid response body policy");
    }
    if (policy.control) { _body_controls.push_back(policy.control); }
  }

  static std::string errno_message() { return std::strerror(errno); }

  static std::vector<std::string> header_values(std::string_view request, std::string_view name)
  {
    std::vector<std::string> values;
    auto position = request.find("\r\n");
    while (position != std::string_view::npos) {
      position += 2;
      auto const end = request.find("\r\n", position);
      if (end == std::string_view::npos || end == position) { break; }
      auto const line  = request.substr(position, end - position);
      auto const colon = line.find(':');
      if (colon == name.size() &&
          std::equal(name.begin(), name.end(), line.begin(), [](unsigned char a, unsigned char b) {
            return std::tolower(a) == std::tolower(b);
          })) {
        auto value       = line.substr(colon + 1);
        auto const begin = value.find_first_not_of(" \t");
        auto const last  = value.find_last_not_of(" \t");
        values.emplace_back(begin == std::string_view::npos
                              ? std::string_view{}
                              : value.substr(begin, last - begin + 1));
      }
      position = end;
    }
    return values;
  }

  static void append_etag_header(std::string& response, std::string const& etag)
  {
    if (!etag.empty()) { response += "\r\nETag: " + etag; }
  }

  static void append_header(std::string& response, std::string_view name, std::string const& value)
  {
    if (value.empty()) { return; }
    response += "\r\n";
    response.append(name);
    response += ": " + value;
  }

  static void send_interim_headers(loopback_connection& conn,
                                   std::string const& etag,
                                   std::string const& content_range,
                                   std::string const& retry_after)
  {
    if (etag.empty() && content_range.empty() && retry_after.empty()) { return; }
    std::string response{"HTTP/1.1 100 Continue"};
    append_etag_header(response, etag);
    append_header(response, "Content-Range", content_range);
    append_header(response, "Retry-After", retry_after);
    response += "\r\n\r\n";
    send_all(conn, response);
  }

  static void append_connection_header(std::string& response, bool close_connection)
  {
    response += close_connection ? "\r\nConnection: close" : "\r\nConnection: keep-alive";
  }

  void accept_loop(int listen_fd)
  {
    while (!_stop.load(std::memory_order_relaxed)) {
      sockaddr_in client{};
      socklen_t len = sizeof(client);
      int fd        = ::accept(listen_fd, reinterpret_cast<sockaddr*>(&client), &len);
      if (fd < 0) {
        if (_stop.load(std::memory_order_relaxed)) { return; }
        continue;
      }
      _accepted_connection_count.fetch_add(1, std::memory_order_relaxed);
      std::scoped_lock lock{_workers_mutex};
      _client_fds.push_back(fd);
      _workers.emplace_back([this, fd] {
        handle_client(fd);
        std::scoped_lock lock{_workers_mutex};
        std::erase(_client_fds, fd);
        ::close(fd);
      });
    }
  }

  void handle_client(int fd)
  {
    timeval timeout{};
    timeout.tv_sec = 5;
    (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    (void)::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

    SSL* ssl = nullptr;
    if (_tls_ctx != nullptr) {
      // The client may hang up while the server writes: EPIPE, never a process-wide SIGPIPE
      // (SSL_write has no MSG_NOSIGNAL).
      sigset_t block;
      sigemptyset(&block);
      sigaddset(&block, SIGPIPE);
      pthread_sigmask(SIG_BLOCK, &block, nullptr);
      ssl = SSL_new(_tls_ctx);
      if (ssl == nullptr || SSL_set_fd(ssl, fd) != 1 || SSL_accept(ssl) != 1) {
        _tls_handshake_failures.fetch_add(1, std::memory_order_relaxed);
        ERR_clear_error();
        if (ssl != nullptr) { SSL_free(ssl); }
        return;
      }
      _tls_handshakes.fetch_add(1, std::memory_order_relaxed);
    }
    loopback_connection conn{fd, ssl};
    std::string pending;
    std::string request;
    while (!_stop.load(std::memory_order_relaxed) && read_request(conn, pending, request)) {
      if (handle_request(conn, request)) { break; }
    }
    if (ssl != nullptr) {
      if (!conn.dropped()) { conn.close_notify(); }
      SSL_free(ssl);
      ERR_clear_error();
    }
  }

  bool handle_request(loopback_connection& conn, std::string const& request)
  {
    bool const close_connection = requests_connection_close(request);
    bool const is_head          = request.rfind("HEAD ", 0) == 0;
    bool const is_get           = request.rfind("GET ", 0) == 0;
    bool const is_list = is_get && request_target(request).find("list-type=2") != std::string::npos;
    auto const key     = request_key(request);

    if (is_head) {
      auto const head_idx = _head_count.fetch_add(1, std::memory_order_relaxed);
      auto const key_idx  = increment_request_count(_head_counts_by_key, key);
      auto const scripted = scripted_step(key, false, key_idx);
      delay_response(scripted);
      send_interim_headers(conn, _fault.interim_head_etag, {}, _fault.interim_head_retry_after);
      bool const scripted_failure = scripted && scripted->status >= 400;
      bool const policy_failure =
        !scripted && (_fault.fail_all_heads || head_idx < _fault.fail_first_heads);
      if (scripted_failure || policy_failure) {
        auto const status    = scripted_failure ? scripted->status : _fault.head_fail_status;
        std::string response = "HTTP/1.1 " + std::to_string(status) + " Error\r\nContent-Length: 0";
        append_etag_header(response, response_etag(scripted, _fault.failed_head_etag));
        append_header(
          response, "Retry-After", response_retry_after(scripted, _fault.failed_head_retry_after));
        append_connection_header(response, close_connection);
        response += "\r\n\r\n";
        send_all(conn, response);
        return close_connection;
      }
      std::string response = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(_object.size());
      append_etag_header(response, response_etag(scripted, _fault.successful_head_etag));
      append_connection_header(response, close_connection);
      response += "\r\n\r\n";
      send_all(conn, response);
      return close_connection;
    }

    if (is_list) {
      if (!pause_response(_fault.response_delay)) { return true; }
      _list_count.fetch_add(1, std::memory_order_relaxed);
      auto const body = list_xml();
      std::string response =
        "HTTP/1.1 200 OK\r\nContent-Type: application/xml\r\nContent-Length: " +
        std::to_string(body.size());
      append_connection_header(response, close_connection);
      response += "\r\n\r\n";
      bool const sent = send_response(
        conn, response, {reinterpret_cast<std::uint8_t const*>(body.data()), body.size()}, {});
      return close_connection || !sent;
    }

    if (!is_get) {
      std::string response{"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0"};
      append_connection_header(response, close_connection);
      response += "\r\n\r\n";
      send_all(conn, response);
      return close_connection;
    }

    auto const ranges = header_values(request, "range");
    bool const suffix = std::any_of(ranges.begin(), ranges.end(), [](std::string const& range) {
      std::string lower = range;
      std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
      return lower.starts_with("bytes=-");
    });
    data_get_guard active{*this, !suffix};
    {
      std::scoped_lock lock{_get_records_mutex};
      _get_records.push_back(
        {header_values(request, "if-match"),
         ranges,
         !header_values(request, "authorization").empty(),
         request_target(request).find("X-Amz-Signature=") != std::string::npos});
    }
    std::size_t get_idx;
    {
      std::scoped_lock lock{_get_response_mutex};
      get_idx = _get_count.fetch_add(1, std::memory_order_relaxed);
    }
    _get_response_cv.notify_all();
    auto const key_idx = increment_request_count(_get_counts_by_key, key);
    wait_at_get_response_barrier();
    auto const scripted = scripted_step(key, true, key_idx);
    delay_response(scripted);
    auto const& body_policy   = scripted && scripted->body ? *scripted->body : _fault.body;
    auto const body_xor       = scripted ? scripted->body_xor : std::uint8_t{0};
    bool const chunked        = scripted && scripted->chunked_transfer_encoding;
    std::size_t const garbage = scripted ? scripted->trailing_garbage : 0;
    send_interim_headers(conn,
                         _fault.interim_get_etag,
                         _fault.interim_get_content_range,
                         _fault.interim_get_retry_after);
    bool const scripted_failure = scripted && scripted->status >= 400;
    bool const policy_failure =
      !scripted && (_fault.fail_all_gets || get_idx < _fault.fail_first_gets);
    if (scripted_failure || policy_failure) {
      auto const status    = scripted_failure ? scripted->status : _fault.fail_status;
      bool const truncate  = _fault.truncate_error_body;
      std::string response = "HTTP/1.1 " + std::to_string(status) +
                             " Error\r\nContent-Length: " + (truncate ? "64" : "0");
      append_etag_header(response, response_etag(scripted, _fault.failed_get_etag));
      append_header(
        response, "Retry-After", response_retry_after(scripted, _fault.failed_get_retry_after));
      append_connection_header(response, close_connection || truncate);
      response += "\r\n\r\n";
      std::string_view const body = truncate ? "short" : "";
      bool const sent =
        send_response(conn,
                      response,
                      {reinterpret_cast<std::uint8_t const*>(body.data()), body.size()},
                      body_policy);
      return close_connection || truncate || !sent;
    }

    if (auto range = parse_range(request)) {
      if (_fault.fail_range_with_416) {
        std::string response{"HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0"};
        append_etag_header(response, _fault.failed_get_etag);
        append_connection_header(response, close_connection);
        response += "\r\n\r\n";
        bool const sent = send_response(conn, response, {}, body_policy);
        return close_connection || !sent;
      }
      if (_fault.ignore_range_with_200 || (scripted && scripted->ignore_range)) {
        std::string response =
          "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(_object.size());
        append_etag_header(response, response_etag(scripted, _fault.successful_get_etag));
        append_connection_header(response, close_connection);
        response += "\r\n\r\n";
        bool const sent =
          send_body(conn, response, 0, _object.size(), body_xor, body_policy, chunked, garbage);
        return close_connection || !sent;
      }
      auto const [start, end] = *range;
      auto const size         = end - start + 1;
      std::string response =
        "HTTP/1.1 206 Partial Content\r\nContent-Length: " + std::to_string(size);
      if (_fault.malformed_content_range || (scripted && scripted->malformed_content_range)) {
        response += "\r\nContent-Range: bytes malformed";
        append_etag_header(response, _fault.failed_get_etag);
      } else {
        auto const& content_range =
          scripted && scripted->content_range ? scripted->content_range : _fault.content_range;
        if (content_range) {
          response += "\r\nContent-Range: " + *content_range;
        } else if (!_fault.omit_successful_content_range) {
          response += "\r\nContent-Range: bytes " + std::to_string(start) + "-" +
                      std::to_string(end) + "/" + std::to_string(_object.size());
        }
        append_etag_header(response, response_etag(scripted, _fault.successful_get_etag));
      }
      append_connection_header(response, close_connection);
      response += "\r\n\r\n";
      bool const sent =
        send_body(conn, response, start, size, body_xor, body_policy, chunked, garbage);
      return close_connection || !sent;
    }

    std::string response = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(_object.size());
    append_etag_header(response, response_etag(scripted, _fault.successful_get_etag));
    append_connection_header(response, close_connection);
    response += "\r\n\r\n";
    bool const sent =
      send_body(conn, response, 0, _object.size(), body_xor, body_policy, chunked, garbage);
    return close_connection || !sent;
  }

  static bool read_request(loopback_connection& conn, std::string& pending, std::string& request)
  {
    constexpr std::size_t max_header_bytes = 64 * 1024;
    while (true) {
      auto const end = pending.find("\r\n\r\n");
      if (end != std::string::npos) {
        request = pending.substr(0, end + 4);
        pending.erase(0, end + 4);
        return true;
      }
      if (pending.size() >= max_header_bytes) { return false; }

      char bytes[8192];
      ssize_t const n = conn.recv_some(bytes, sizeof(bytes));
      if (n <= 0) { return false; }
      pending.append(bytes, static_cast<std::size_t>(n));
    }
  }

  static bool requests_connection_close(std::string const& request)
  {
    std::string lower = request;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    return lower.find("\r\nconnection: close") != std::string::npos;
  }

  static std::string request_target(std::string const& request)
  {
    auto const first = request.find(' ');
    if (first == std::string::npos) { return {}; }
    auto const second = request.find(' ', first + 1);
    if (second == std::string::npos) { return {}; }
    return request.substr(first + 1, second - first - 1);
  }

  static std::string request_key(std::string const& request)
  {
    auto target = request_target(request);
    if (auto const query = target.find('?'); query != std::string::npos) { target.resize(query); }
    if (!target.empty() && target.front() == '/') { target.erase(0, 1); }
    auto const bucket_end = target.find('/');
    return bucket_end == std::string::npos ? target : target.substr(bucket_end + 1);
  }

  [[nodiscard]] std::optional<scripted_response> scripted_step(std::string const& key,
                                                               bool get,
                                                               std::size_t index) const
  {
    auto const script = _scripts.find(key);
    if (script == _scripts.end()) { return std::nullopt; }
    auto const& steps = get ? script->second.gets : script->second.heads;
    if (steps.empty()) { return std::nullopt; }
    return steps[std::min(index, steps.size() - 1)];
  }

  bool pause_response(std::chrono::milliseconds delay)
  {
    std::unique_lock lock{_get_response_mutex};
    if (delay.count() > 0) {
      _get_response_cv.wait_for(lock, delay, [&] { return _stop.load(std::memory_order_relaxed); });
    }
    return !_stop.load(std::memory_order_relaxed);
  }

  void delay_response(std::optional<scripted_response> const& scripted)
  {
    auto delay = _fault.response_delay;
    if (scripted) { delay += scripted->delay; }
    pause_response(delay);
  }

  void wait_at_get_response_barrier()
  {
    if (_fault.get_response_barrier == 0) { return; }
    _get_response_cv.notify_all();
    std::unique_lock lock{_get_response_mutex};
    _get_response_cv.wait(lock, [&] {
      return _stop.load(std::memory_order_relaxed) ||
             _get_count.load(std::memory_order_relaxed) >= _fault.get_response_barrier;
    });
  }

  static std::string response_etag(std::optional<scripted_response> const& scripted,
                                   std::string const& fallback)
  {
    return scripted && scripted->etag ? *scripted->etag : fallback;
  }

  static std::string response_retry_after(std::optional<scripted_response> const& scripted,
                                          std::string const& fallback)
  {
    return scripted && scripted->retry_after ? *scripted->retry_after : fallback;
  }

  std::size_t increment_request_count(std::unordered_map<std::string, std::size_t>& counts,
                                      std::string const& key)
  {
    std::scoped_lock lock{_request_counts_mutex};
    auto& count = counts[key];
    return count++;
  }

  [[nodiscard]] std::size_t request_count(
    std::unordered_map<std::string, std::size_t> const& counts, std::string_view key) const
  {
    std::scoped_lock lock{_request_counts_mutex};
    auto const found = counts.find(std::string{key});
    return found == counts.end() ? 0 : found->second;
  }

  static std::string xml_escape(std::string_view value)
  {
    std::string out;
    for (char c : value) {
      switch (c) {
        case '&': out += "&amp;"; break;
        case '<': out += "&lt;"; break;
        case '>': out += "&gt;"; break;
        case '\"': out += "&quot;"; break;
        case '\'': out += "&apos;"; break;
        default: out.push_back(c); break;
      }
    }
    return out;
  }

  [[nodiscard]] std::string list_xml() const
  {
    std::string body =
      "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
      "<ListBucketResult><IsTruncated>false</IsTruncated>";
    for (auto const& object : _listed) {
      body += "<Contents><Key>" + xml_escape(object.key) + "</Key><Size>" +
              std::to_string(object.size) + "</Size></Contents>";
    }
    body += "</ListBucketResult>";
    return body;
  }

  [[nodiscard]] std::optional<std::pair<std::size_t, std::size_t>> parse_range(
    std::string const& request) const
  {
    std::string lower = request;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    std::string const prefix{"range: bytes="};
    auto pos = lower.find(prefix);
    if (pos == std::string::npos) { return std::nullopt; }
    pos += prefix.size();
    auto const eol     = lower.find("\r\n", pos);
    auto const end_pos = eol == std::string::npos ? lower.size() : eol;
    auto const spec    = lower.substr(pos, end_pos - pos);
    auto const dash    = spec.find('-');
    if (dash == std::string::npos) { return std::nullopt; }

    try {
      std::size_t start = 0;
      std::size_t end   = _object.size() - 1;
      if (dash == 0) {
        auto const suffix = static_cast<std::size_t>(std::stoull(spec.substr(1)));
        if (suffix == 0) { return std::nullopt; }
        start = suffix >= _object.size() ? 0 : _object.size() - suffix;
      } else {
        start = static_cast<std::size_t>(std::stoull(spec.substr(0, dash)));
        if (dash + 1 < spec.size()) {
          end = static_cast<std::size_t>(std::stoull(spec.substr(dash + 1)));
        }
      }
      if (start >= _object.size()) { return std::nullopt; }
      end = std::min(end, _object.size() - 1);
      if (end < start) { return std::nullopt; }
      return std::pair{start, end};
    } catch (...) {
      return std::nullopt;
    }
  }

  bool send_body(loopback_connection& conn,
                 std::string_view headers,
                 std::size_t start,
                 std::size_t size,
                 std::uint8_t mask,
                 response_body_policy const& policy,
                 bool chunked        = false,
                 std::size_t garbage = 0)
  {
    std::span<std::uint8_t const> const object{_object};
    if (chunked) { return send_chunked(conn, headers, object.subspan(start, size), mask); }
    bool sent = false;
    if (mask == 0) {
      sent = send_response(conn, headers, object.subspan(start, size), policy);
    } else {
      std::vector<std::uint8_t> masked(object.begin() + static_cast<std::ptrdiff_t>(start),
                                       object.begin() + static_cast<std::ptrdiff_t>(start + size));
      for (auto& byte : masked) {
        byte = static_cast<std::uint8_t>(byte ^ mask);
      }
      sent = send_response(conn, headers, masked, policy);
    }
    if (sent && garbage > 0) {
      std::string const junk(garbage, 'x');
      sent = send_all(conn, junk) == junk.size();
    }
    return sent;
  }

  /// @p headers (which carry `Content-Length`) re-framed as `Transfer-Encoding: chunked`, and
  /// @p body sent as one chunk followed by the last chunk.
  bool send_chunked(loopback_connection& conn,
                    std::string_view headers,
                    std::span<std::uint8_t const> body,
                    std::uint8_t mask)
  {
    std::string head{headers};
    auto const at = head.find("\r\nContent-Length: ");
    if (at != std::string::npos) {
      auto const eol = head.find("\r\n", at + 2);
      head.replace(at, eol - at, "\r\nTransfer-Encoding: chunked");
    }
    std::string framed = head;
    if (!body.empty()) {
      std::array<char, 32> size_line{};
      std::snprintf(size_line.data(), size_line.size(), "%zx\r\n", body.size());
      framed += size_line.data();
      for (auto const byte : body) {
        framed.push_back(static_cast<char>(byte ^ mask));
      }
      framed += "\r\n";
    }
    framed += "0\r\n\r\n";
    auto const sent = send_all(conn, framed);
    _body_bytes_sent.fetch_add(body.size(), std::memory_order_relaxed);
    return sent == framed.size();
  }

  bool send_response(loopback_connection& conn,
                     std::string_view headers,
                     std::span<std::uint8_t const> body,
                     response_body_policy const& policy)
  {
    if (send_all(conn, headers) != headers.size()) { return false; }
    if (policy.control) { policy.control->headers_sent(); }
    if (!policy.control && policy.chunk_bytes == 0 && policy.chunk_delay.count() == 0 &&
        !policy.drop_after_bytes && !policy.key_update_after_bytes) {
      auto const sent = send_all(conn, body.data(), body.size());
      _body_bytes_sent.fetch_add(sent, std::memory_order_relaxed);
      return sent == body.size();
    }

    std::size_t sent = 0;
    bool held        = false;
    bool key_updated = false;
    while (true) {
      if (_stop.load(std::memory_order_relaxed)) { return false; }
      if (!held && policy.hold_after_bytes && sent == *policy.hold_after_bytes) {
        held = true;
        if (!policy.control->hold()) { return false; }
      }
      if (policy.drop_after_bytes && sent == *policy.drop_after_bytes && sent < body.size()) {
        _body_drops.fetch_add(1, std::memory_order_relaxed);
        conn.drop();
        return false;
      }
      if (!key_updated && policy.key_update_after_bytes && sent == *policy.key_update_after_bytes) {
        key_updated = true;
        if (conn.key_update()) { _key_updates_sent.fetch_add(1, std::memory_order_relaxed); }
      }
      if (sent == body.size()) { return true; }
      if (sent != 0 && !pause_response(policy.chunk_delay)) { return false; }
      auto bytes = policy.chunk_bytes == 0 ? body.size() - sent
                                           : std::min(policy.chunk_bytes, body.size() - sent);
      if (!held && policy.hold_after_bytes && *policy.hold_after_bytes > sent) {
        bytes = std::min(bytes, *policy.hold_after_bytes - sent);
      }
      if (policy.drop_after_bytes && *policy.drop_after_bytes > sent) {
        bytes = std::min(bytes, *policy.drop_after_bytes - sent);
      }
      if (!key_updated && policy.key_update_after_bytes && *policy.key_update_after_bytes > sent) {
        bytes = std::min(bytes, *policy.key_update_after_bytes - sent);
      }
      auto const wrote = send_all(conn, body.data() + sent, bytes);
      _body_bytes_sent.fetch_add(wrote, std::memory_order_relaxed);
      if (policy.control) { policy.control->body_sent(wrote); }
      sent += wrote;
      if (wrote != bytes) { return false; }
    }
  }

  static std::size_t send_all(loopback_connection& conn, std::string_view bytes)
  {
    return conn.send_all(reinterpret_cast<std::uint8_t const*>(bytes.data()), bytes.size());
  }

  static std::size_t send_all(loopback_connection& conn,
                              std::uint8_t const* bytes,
                              std::size_t size)
  {
    return conn.send_all(bytes, size);
  }

  int _listen_fd{-1};
  std::uint16_t _port{0};
  std::vector<std::uint8_t> _object;
  range_fault_policy _fault;
  std::vector<listed_object> _listed;
  std::unordered_map<std::string, key_response_script> _scripts;
  std::vector<std::shared_ptr<range_response_control>> _body_controls;
  std::once_flag _stop_once;
  std::atomic<bool> _stop{false};
  std::atomic<std::size_t> _accepted_connection_count{0};
  std::atomic<std::size_t> _head_count{0};
  std::atomic<std::size_t> _get_count{0};
  std::atomic<std::size_t> _list_count{0};
  std::atomic<std::size_t> _active_data_gets{0};
  std::atomic<std::size_t> _peak_active_gets{0};
  std::atomic<std::size_t> _body_bytes_sent{0};
  std::atomic<std::size_t> _tls_handshakes{0};
  std::atomic<std::size_t> _tls_handshake_failures{0};
  std::atomic<std::size_t> _key_updates_sent{0};
  std::atomic<std::size_t> _body_drops{0};
  SSL_CTX* _tls_ctx{nullptr};
  std::mutex _data_get_mutex;
  std::condition_variable _data_get_cv;
  std::mutex _get_response_mutex;
  std::condition_variable _get_response_cv;
  mutable std::mutex _request_counts_mutex;
  std::unordered_map<std::string, std::size_t> _head_counts_by_key;
  std::unordered_map<std::string, std::size_t> _get_counts_by_key;
  mutable std::mutex _get_records_mutex;
  std::vector<get_record> _get_records;
  std::thread _thread;
  std::mutex _workers_mutex;
  std::vector<int> _client_fds;
  std::vector<std::thread> _workers;
};

class list_capable_mock_authorizer final : public io::rest::request_authorizer {
 public:
  explicit list_capable_mock_authorizer(std::string endpoint) : _endpoint(std::move(endpoint)) {}

  void set_object_exception(std::string key, std::exception_ptr error)
  {
    std::scoped_lock lock{_exceptions_mutex};
    _object_exceptions.insert_or_assign(std::move(key), std::move(error));
  }

  io::rest::authorized_request authorize(io::rest::object_ref const& obj,
                                         io::rest::request_method,
                                         std::chrono::seconds) override
  {
    _object_calls.fetch_add(1, std::memory_order_relaxed);
    std::exception_ptr error;
    {
      std::scoped_lock lock{_exceptions_mutex};
      if (auto const found = _object_exceptions.find(obj.key); found != _object_exceptions.end()) {
        error = found->second;
      }
    }
    if (error) { std::rethrow_exception(error); }
    return {_endpoint + "/" + obj.bucket + "/" + obj.key, {}};
  }

  io::rest::authorized_request authorize_list(std::string_view bucket,
                                              std::string_view canonical_query,
                                              std::chrono::seconds) override
  {
    _list_calls.fetch_add(1, std::memory_order_relaxed);
    return {_endpoint + "/" + std::string{bucket} + "?" + std::string{canonical_query}, {}};
  }

  [[nodiscard]] int object_calls() const noexcept { return _object_calls.load(); }
  [[nodiscard]] int list_calls() const noexcept { return _list_calls.load(); }

 private:
  std::string _endpoint;
  std::atomic<int> _object_calls{0};
  std::atomic<int> _list_calls{0};
  std::mutex _exceptions_mutex;
  std::unordered_map<std::string, std::exception_ptr> _object_exceptions;
};

}  // namespace cucascade::test
