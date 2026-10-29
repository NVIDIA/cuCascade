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
#include <signal.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
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
#include <map>
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
 * @brief Raw-socket HTTP/1.1 loopback server for the REST tests (header-only).
 *
 * Reads: ranged / suffix / full GET, HEAD and ListObjectsV2 of one fixture object (any key), with
 * the fault policies of @ref cucascade::test::range_fault_policy and per-key scripts.
 *
 * Writes (S3, path style `/<bucket>/<key>`): an in-memory object store. PutObject (with
 * `If-None-Match: *` -> 412 when the key exists), CreateMultipartUpload (`POST ?uploads`),
 * UploadPart (`PUT ?partNumber=N&uploadId=U`, ETag header), CompleteMultipartUpload (`POST
 * ?uploadId=U`: parts assembled in part-number order; order, ETags and the minimum part size of
 * every part but the last are validated), AbortMultipartUpload (`DELETE ?uploadId=U`) and
 * DeleteObject. Written objects are what later GET / HEAD / LIST of their key see; other keys
 * keep returning the fixture object. Fault injection for writes (@ref
 * cucascade::test::write_fault), `Expect: 100-continue` handling and request records.
 *
 * HTTPS: with @ref cucascade::test::loopback_tls enabled every accepted socket is wrapped in an
 * OpenSSL server `SSL*` (blocking `SSL_accept`) presenting the test PKI's server certificate
 * (`io/tls/tls_test_pki.hpp`), and the same request handling runs over `SSL_read` /
 * `SSL_write`.
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
};

struct key_response_script {
  std::vector<scripted_response> gets;
  std::vector<scripted_response> heads;
};

/// S3 write request kinds served by @ref loopback_range_server.
enum class write_kind : std::uint8_t {
  put_object,       ///< PUT /b/k
  create_upload,    ///< POST /b/k?uploads
  upload_part,      ///< PUT /b/k?partNumber=N&uploadId=U
  complete_upload,  ///< POST /b/k?uploadId=U
  abort_upload,     ///< DELETE /b/k?uploadId=U
  delete_object,    ///< DELETE /b/k
};

/**
 * @brief One injected write fault: matches the requests of @c kind whose arrival index (0-based,
 *        among the requests of that kind, or among the UploadParts of @c part_number when set)
 *        lies in [nth, nth + count).
 */
struct write_fault {
  enum class action : std::uint8_t {
    status,           ///< answer @c status with an S3 error document (@c code); body read first
    ok_with_error,    ///< `200 OK` whose body is an S3 `<Error>` (@c code), e.g. on Complete
    drop_mid_body,    ///< read about half of the body, then close the connection, no response
    drop_after_body,  ///< read the body (perform it when @c apply), close without a response
    hold,             ///< hold the response on @c control until released, then answer normally
  };
  write_kind kind{write_kind::upload_part};
  std::size_t nth{0};
  std::size_t count{1};
  std::optional<unsigned> part_number;  ///< upload_part only: count this part number's requests
  action what{action::status};
  int status{503};
  std::string code{"SlowDown"};
  std::string retry_after;  ///< Retry-After header of a `status` answer (empty: none)
  bool apply{false};        ///< drop_after_body: the request takes effect (a lost response)
  std::shared_ptr<range_response_control> control;  ///< hold
};

/// How a write request carrying `Expect: 100-continue` is answered.
enum class expect_continue_mode : std::uint8_t {
  honor,   ///< `100 Continue`, then read the body
  ignore,  ///< no interim response; read the body when it comes
  reject,  ///< final @c expect_reject_status at once, without reading the body; close
};

/// S3 write behaviour of @ref loopback_range_server.
struct s3_write_policy {
  /// CompleteMultipartUpload: every part but the last must be at least this large
  /// (EntityTooSmall otherwise); S3 uses 5 MiB, tests may lower it.
  std::uint64_t min_part_size{std::uint64_t{5} << 20};
  /// GET / HEAD of a key that was never written: 404 NoSuchKey instead of the fixture object.
  bool missing_key_404{false};
  expect_continue_mode expect{expect_continue_mode::honor};
  int expect_reject_status{403};
  /// Spaces before the CreateMultipartUpload / CompleteMultipartUpload XML (S3 sends keep-alive
  /// whitespace while it assembles).
  std::size_t create_whitespace_prefix{0};
  std::size_t complete_whitespace_prefix{0};
  std::vector<write_fault> faults;
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

  /// One write request as the server saw it.
  struct write_record {
    write_kind kind{write_kind::put_object};
    std::string bucket;
    std::string key;    ///< percent-decoded
    std::string query;  ///< as sent (without '?')
    unsigned part_number{0};
    std::string upload_id;  ///< percent-decoded
    std::size_t content_length{0};
    bool if_none_match{false};    ///< `If-None-Match: *`
    bool expect_continue{false};  ///< `Expect: 100-continue`
    bool header_authorized{false};
    bool presigned{false};
    std::string content_sha256;  ///< x-amz-content-sha256
    int status{0};               ///< 0: no response was sent (dropped connection)
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
                                 s3_write_policy write                                        = {},
                                 loopback_tls tls                                             = {})
    : _object(std::move(object)),
      _fault(std::move(fault)),
      _listed(std::move(listed)),
      _scripts(std::move(scripts)),
      _write(std::move(write))
  {
    if (_object.empty()) { throw std::runtime_error("loopback object must be non-empty"); }
    add_body_control(_fault.body);
    for (auto const& [key, script] : _scripts) {
      for (auto const& response : script.gets) {
        if (response.body) { add_body_control(*response.body); }
      }
    }
    for (auto const& f : _write.faults) {
      check_write_fault(f);
      if (f.control) { _body_controls.push_back(f.control); }
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
      {
        std::scoped_lock lock{_store_mutex};
        for (auto const& control : _body_controls) {
          control->stop();
        }
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

  //===--------------------------------------------------------------------===//
  // S3 writes: counters, records, the object store
  //===--------------------------------------------------------------------===//

  [[nodiscard]] std::size_t put_object_count() const noexcept { return _put_object_count.load(); }
  [[nodiscard]] std::size_t create_upload_count() const noexcept
  {
    return _create_upload_count.load();
  }
  [[nodiscard]] std::size_t upload_part_count() const noexcept { return _upload_part_count.load(); }
  [[nodiscard]] std::size_t complete_upload_count() const noexcept
  {
    return _complete_upload_count.load();
  }
  [[nodiscard]] std::size_t abort_upload_count() const noexcept
  {
    return _abort_upload_count.load();
  }
  [[nodiscard]] std::size_t delete_object_count() const noexcept
  {
    return _delete_object_count.load();
  }
  /// Write requests that carried `Expect: 100-continue` (with a body).
  [[nodiscard]] std::size_t expect_continue_count() const noexcept
  {
    return _expect_continue_count.load();
  }
  /// Request body bytes the server read (all write kinds).
  [[nodiscard]] std::size_t write_body_bytes() const noexcept { return _write_body_bytes.load(); }
  [[nodiscard]] std::size_t active_part_uploads() const noexcept
  {
    return _active_part_uploads.load();
  }
  [[nodiscard]] std::size_t peak_part_uploads() const noexcept { return _peak_part_uploads.load(); }
  [[nodiscard]] std::size_t tls_handshake_count() const noexcept { return _tls_handshakes.load(); }
  [[nodiscard]] std::size_t tls_handshake_failure_count() const noexcept
  {
    return _tls_handshake_failures.load();
  }
  /// KeyUpdates sent mid-body (response_body_policy::key_update_after_bytes).
  [[nodiscard]] std::size_t key_updates_sent() const noexcept { return _key_updates_sent.load(); }
  /// Responses cut off mid-body (response_body_policy::drop_after_bytes).
  [[nodiscard]] std::size_t body_drops() const noexcept { return _body_drops.load(); }

  [[nodiscard]] std::vector<write_record> write_requests() const
  {
    std::scoped_lock lock{_store_mutex};
    return _write_records;
  }

  /// @return the bytes stored under @p key (percent-decoded key), if any.
  [[nodiscard]] std::optional<std::vector<std::uint8_t>> stored_object(std::string_view key) const
  {
    std::scoped_lock lock{_store_mutex};
    auto const it = _store.find(std::string{key});
    if (it == _store.end()) { return std::nullopt; }
    return it->second->data;
  }
  /// @return the ETag (quoted) of the object stored under @p key; empty when none.
  [[nodiscard]] std::string stored_etag(std::string_view key) const
  {
    std::scoped_lock lock{_store_mutex};
    auto const it = _store.find(std::string{key});
    return it == _store.end() ? std::string{} : it->second->etag;
  }
  [[nodiscard]] std::vector<std::string> stored_keys() const
  {
    std::scoped_lock lock{_store_mutex};
    std::vector<std::string> keys;
    for (auto const& [key, object] : _store) {
      keys.push_back(key);
    }
    return keys;
  }
  /// Store @p data under @p key as if it had been PUT. @return its ETag.
  std::string put_stored_object(std::string key, std::vector<std::uint8_t> data)
  {
    auto object  = std::make_shared<stored_object_t>();
    object->etag = "\"" + digest_hex(data.data(), data.size()) + "\"";
    object->data = std::move(data);
    std::scoped_lock lock{_store_mutex};
    auto etag = object->etag;
    _store.insert_or_assign(std::move(key), std::move(object));
    return etag;
  }

  /// Multipart uploads neither completed nor aborted.
  [[nodiscard]] std::size_t active_upload_count() const
  {
    std::scoped_lock lock{_store_mutex};
    return _uploads.size();
  }
  [[nodiscard]] std::vector<std::string> upload_ids() const
  {
    std::scoped_lock lock{_store_mutex};
    std::vector<std::string> ids;
    for (auto const& [id, upload] : _uploads) {
      ids.push_back(id);
    }
    return ids;
  }
  /// @return the part numbers uploaded so far to @p upload_id (empty when it does not exist).
  [[nodiscard]] std::vector<unsigned> upload_parts(std::string_view upload_id) const
  {
    std::scoped_lock lock{_store_mutex};
    std::vector<unsigned> parts;
    auto const it = _uploads.find(std::string{upload_id});
    if (it != _uploads.end()) {
      for (auto const& [n, part] : it->second.parts) {
        parts.push_back(n);
      }
    }
    return parts;
  }

  void set_min_part_size(std::uint64_t bytes)
  {
    std::scoped_lock lock{_store_mutex};
    _write.min_part_size = bytes;
  }
  void set_expect_continue(expect_continue_mode mode, int reject_status = 403)
  {
    std::scoped_lock lock{_store_mutex};
    _write.expect               = mode;
    _write.expect_reject_status = reject_status;
  }
  /// Add a fault; it counts requests from the server's start (see write_fault).
  void add_write_fault(write_fault fault)
  {
    check_write_fault(fault);
    std::scoped_lock lock{_store_mutex};
    if (fault.control) { _body_controls.push_back(fault.control); }
    _write.faults.push_back(std::move(fault));
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
      if (handle_request(conn, request, pending)) { break; }
    }
    if (ssl != nullptr) {
      if (!conn.dropped()) { conn.close_notify(); }
      SSL_free(ssl);
      ERR_clear_error();
    }
  }

  bool handle_request(loopback_connection& conn, std::string const& request, std::string& pending)
  {
    bool const close_connection = requests_connection_close(request);
    bool const is_head          = request.rfind("HEAD ", 0) == 0;
    bool const is_get           = request.rfind("GET ", 0) == 0;
    bool const is_list = is_get && request_target(request).find("list-type=2") != std::string::npos;
    auto const key     = request_key(request);

    if (is_write_request(request)) { return handle_write(conn, request, pending); }

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
      auto const stored = find_stored(request);
      if (!stored && missing_key_404()) {
        return send_no_such_key(conn, close_connection, /*with_body=*/false);
      }
      std::size_t const object_size = stored ? stored->data.size() : _object.size();
      auto etag                     = response_etag(scripted, _fault.successful_head_etag);
      if (etag.empty() && stored) { etag = stored->etag; }
      std::string response = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(object_size);
      append_etag_header(response, etag);
      append_connection_header(response, close_connection);
      response += "\r\n\r\n";
      send_all(conn, response);
      return close_connection;
    }

    if (is_list) {
      if (!pause_response(_fault.response_delay)) { return true; }
      _list_count.fetch_add(1, std::memory_order_relaxed);
      auto const body = list_xml(query_param(request, "prefix"));
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
    auto const& body_policy = scripted && scripted->body ? *scripted->body : _fault.body;
    auto const body_xor     = scripted ? scripted->body_xor : std::uint8_t{0};
    bool const chunked      = scripted && scripted->chunked_transfer_encoding;
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

    // A key written through the store is served from it; other keys get the fixture object.
    auto const stored = find_stored(request);
    if (!stored && missing_key_404()) {
      return send_no_such_key(conn, close_connection, /*with_body=*/true);
    }
    std::span<std::uint8_t const> const object =
      stored ? std::span<std::uint8_t const>{stored->data} : std::span<std::uint8_t const>{_object};
    auto const ok_etag = [&] {
      auto etag = response_etag(scripted, _fault.successful_get_etag);
      if (etag.empty() && stored) { etag = stored->etag; }
      return etag;
    };
    if (stored && range_unsatisfiable(request, object.size())) {
      // S3: 416 with `Content-Range: bytes */<size>` (an empty object answers any range so).
      std::string response = "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0";
      response += "\r\nContent-Range: bytes */" + std::to_string(object.size());
      append_etag_header(response, ok_etag());
      append_connection_header(response, close_connection);
      response += "\r\n\r\n";
      bool const sent = send_response(conn, response, {}, body_policy);
      return close_connection || !sent;
    }

    if (auto range = parse_range(request, object.size())) {
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
          "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(object.size());
        append_etag_header(response, ok_etag());
        append_connection_header(response, close_connection);
        response += "\r\n\r\n";
        bool const sent =
          send_body(conn, response, object, 0, object.size(), body_xor, body_policy, chunked);
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
                      std::to_string(end) + "/" + std::to_string(object.size());
        }
        append_etag_header(response, ok_etag());
      }
      append_connection_header(response, close_connection);
      response += "\r\n\r\n";
      bool const sent =
        send_body(conn, response, object, start, size, body_xor, body_policy, chunked);
      return close_connection || !sent;
    }

    std::string response = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(object.size());
    append_etag_header(response, ok_etag());
    append_connection_header(response, close_connection);
    response += "\r\n\r\n";
    bool const sent =
      send_body(conn, response, object, 0, object.size(), body_xor, body_policy, chunked);
    return close_connection || !sent;
  }

  //===--------------------------------------------------------------------===//
  // S3 writes
  //===--------------------------------------------------------------------===//

  struct stored_object_t {
    std::vector<std::uint8_t> data;
    std::string etag;  ///< quoted
  };
  struct upload_part_t {
    std::string etag;
    std::vector<std::uint8_t> data;
  };
  struct upload_t {
    std::string key;
    std::map<unsigned, upload_part_t> parts;
  };
  /// One response of the write path.
  struct write_response {
    int status{200};
    std::string reason{"OK"};
    std::string headers;  ///< extra header lines, each ending in "\r\n"
    std::string body;
  };

  class part_upload_guard {
   public:
    explicit part_upload_guard(loopback_range_server& server) : _server(server)
    {
      auto const current = _server._active_part_uploads.fetch_add(1) + 1;
      auto peak          = _server._peak_part_uploads.load();
      while (current > peak && !_server._peak_part_uploads.compare_exchange_weak(peak, current)) {}
    }
    ~part_upload_guard() { _server._active_part_uploads.fetch_sub(1); }
    part_upload_guard(part_upload_guard const&)            = delete;
    part_upload_guard& operator=(part_upload_guard const&) = delete;

   private:
    loopback_range_server& _server;
  };

  static void check_write_fault(write_fault const& f)
  {
    if (f.count == 0 || (f.what == write_fault::action::hold && !f.control)) {
      throw std::invalid_argument("invalid write fault");
    }
  }

  /// 128-bit FNV-1a (two 64-bit lanes) as 32 lowercase hex digits: the fixture's "MD5".
  static std::string digest_hex(void const* data, std::size_t size)
  {
    std::uint64_t a = 0xcbf29ce484222325ULL;
    std::uint64_t b = 0x84222325cbf29ce4ULL;
    auto const* p   = static_cast<unsigned char const*>(data);
    for (std::size_t i = 0; i < size; ++i) {
      a = (a ^ p[i]) * 0x100000001b3ULL;
      b = (b ^ p[i] ^ 0x5aU) * 0x100000001b3ULL;
    }
    static constexpr char digits[] = "0123456789abcdef";
    std::string out;
    for (std::uint64_t v : {a, b}) {
      for (int shift = 60; shift >= 0; shift -= 4) {
        out += digits[(v >> shift) & 0xF];
      }
    }
    return out;
  }

  static std::string pct_decode(std::string_view s)
  {
    auto const hex = [](char c) -> int {
      if (c >= '0' && c <= '9') { return c - '0'; }
      if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
      if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
      return -1;
    };
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
      if (s[i] == '%' && i + 2 < s.size()) {
        int const hi = hex(s[i + 1]);
        int const lo = hex(s[i + 2]);
        if (hi >= 0 && lo >= 0) {
          out += static_cast<char>((hi << 4) | lo);
          i += 2;
          continue;
        }
      }
      out += s[i];
    }
    return out;
  }

  static std::string request_method(std::string const& request)
  {
    return request.substr(0, request.find(' '));
  }

  static std::string request_query(std::string const& request)
  {
    auto const target = request_target(request);
    auto const q      = target.find('?');
    return q == std::string::npos ? std::string{} : target.substr(q + 1);
  }

  /// Percent-decoded query parameters (a bare key maps to "").
  static std::map<std::string, std::string> query_params(std::string const& request)
  {
    std::map<std::string, std::string> params;
    auto const query = request_query(request);
    std::string_view rest{query};
    while (!rest.empty()) {
      auto const amp  = rest.find('&');
      auto const pair = rest.substr(0, amp);
      if (!pair.empty()) {
        auto const eq = pair.find('=');
        params[pct_decode(pair.substr(0, eq))] =
          eq == std::string_view::npos ? std::string{} : pct_decode(pair.substr(eq + 1));
      }
      if (amp == std::string_view::npos) { break; }
      rest.remove_prefix(amp + 1);
    }
    return params;
  }

  static std::string query_param(std::string const& request, std::string const& name)
  {
    auto const params = query_params(request);
    auto const it     = params.find(name);
    return it == params.end() ? std::string{} : it->second;
  }

  static std::string request_bucket(std::string const& request)
  {
    auto target = request_target(request);
    if (auto const query = target.find('?'); query != std::string::npos) { target.resize(query); }
    if (!target.empty() && target.front() == '/') { target.erase(0, 1); }
    return pct_decode(target.substr(0, target.find('/')));
  }

  /// The object-store key of a request: its key, percent-decoded.
  static std::string store_key(std::string const& request)
  {
    return pct_decode(request_key(request));
  }

  static bool is_write_request(std::string const& request)
  {
    return request.rfind("PUT ", 0) == 0 || request.rfind("POST ", 0) == 0 ||
           request.rfind("DELETE ", 0) == 0;
  }

  static bool header_is(std::string const& request, std::string_view name, std::string_view value)
  {
    for (auto const& v : header_values(request, name)) {
      if (std::equal(v.begin(), v.end(), value.begin(), value.end(), [](char a, char b) {
            return std::tolower(static_cast<unsigned char>(a)) ==
                   std::tolower(static_cast<unsigned char>(b));
          })) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] std::shared_ptr<stored_object_t const> find_stored(std::string const& request) const
  {
    std::scoped_lock lock{_store_mutex};
    auto const it = _store.find(store_key(request));
    return it == _store.end() ? nullptr : it->second;
  }

  [[nodiscard]] bool missing_key_404() const
  {
    std::scoped_lock lock{_store_mutex};
    return _write.missing_key_404;
  }

  /// A syntactically valid `Range: bytes=` header that the object cannot satisfy.
  static bool range_unsatisfiable(std::string const& request, std::size_t object_size)
  {
    auto const ranges = header_values(request, "range");
    if (ranges.empty()) { return false; }
    std::string spec = ranges.front();
    std::transform(spec.begin(), spec.end(), spec.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (!spec.starts_with("bytes=")) { return false; }
    spec.erase(0, 6);
    auto const dash = spec.find('-');
    if (dash == std::string::npos) { return false; }
    try {
      if (dash == 0) { return object_size == 0 && std::stoull(spec.substr(1)) > 0; }
      return static_cast<std::size_t>(std::stoull(spec.substr(0, dash))) >= object_size;
    } catch (...) {
      return false;
    }
  }

  static std::string s3_error_body(std::string const& code, std::string const& message)
  {
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<Error><Code>" + code + "</Code><Message>" +
           message + "</Message><RequestId>LOOPBACK</RequestId></Error>";
  }

  static write_response s3_error(int status, std::string const& code, std::string const& message)
  {
    write_response r;
    r.status  = status;
    r.reason  = "Error";
    r.headers = "Content-Type: application/xml\r\n";
    r.body    = s3_error_body(code, message);
    return r;
  }

  bool send_write_response(loopback_connection& conn,
                           write_response const& r,
                           bool close_connection)
  {
    std::string head = "HTTP/1.1 " + std::to_string(r.status) + " " + r.reason +
                       "\r\nx-amz-request-id: LOOPBACK\r\n" + r.headers;
    if (r.status != 204) { head += "Content-Length: " + std::to_string(r.body.size()) + "\r\n"; }
    head += close_connection ? "Connection: close\r\n\r\n" : "Connection: keep-alive\r\n\r\n";
    if (send_all(conn, head) != head.size()) { return false; }
    return send_all(conn, r.body) == r.body.size();
  }

  bool send_no_such_key(loopback_connection& conn, bool close_connection, bool with_body)
  {
    auto r = s3_error(404, "NoSuchKey", "The specified key does not exist.");
    if (!with_body) {
      // HEAD: no body (Content-Length 0).
      r.headers.clear();
      r.body.clear();
    }
    r.reason = "Not Found";
    return !send_write_response(conn, r, close_connection) || close_connection;
  }

  /// Read @p n body bytes (after the head): first what @p pending holds, then the connection.
  static bool read_body(loopback_connection& conn,
                        std::string& pending,
                        std::size_t n,
                        std::vector<std::uint8_t>& body)
  {
    body.clear();
    body.reserve(n);
    std::size_t const take = std::min(n, pending.size());
    body.insert(body.end(), pending.begin(), pending.begin() + static_cast<std::ptrdiff_t>(take));
    pending.erase(0, take);
    std::vector<char> buf(std::min<std::size_t>(n, 256 * 1024) + 1);
    while (body.size() < n) {
      auto const r = conn.recv_some(buf.data(), std::min(buf.size(), n - body.size()));
      if (r <= 0) { return false; }
      body.insert(body.end(), buf.begin(), buf.begin() + r);
    }
    return true;
  }

  /// The fault matching this request, if any (each request matches at most one fault).
  [[nodiscard]] std::optional<write_fault> match_fault(write_kind kind,
                                                       std::size_t index,
                                                       unsigned part_number,
                                                       std::size_t part_index) const
  {
    std::scoped_lock lock{_store_mutex};
    for (auto const& f : _write.faults) {
      if (f.kind != kind) { continue; }
      std::size_t const i = f.part_number ? part_index : index;
      if (f.part_number && *f.part_number != part_number) { continue; }
      if (i >= f.nth && i - f.nth < f.count) { return f; }
    }
    return std::nullopt;
  }

  bool handle_write(loopback_connection& conn, std::string const& request, std::string& pending)
  {
    bool const close_connection = requests_connection_close(request);
    auto const method           = request_method(request);
    auto const params           = query_params(request);
    write_record rec;
    rec.bucket            = request_bucket(request);
    rec.key               = store_key(request);
    rec.query             = request_query(request);
    rec.if_none_match     = header_is(request, "if-none-match", "*");
    rec.header_authorized = !header_values(request, "authorization").empty();
    rec.presigned         = rec.query.find("X-Amz-Signature=") != std::string::npos;
    if (auto const sha = header_values(request, "x-amz-content-sha256"); !sha.empty()) {
      rec.content_sha256 = sha.front();
    }
    if (auto const id = params.find("uploadId"); id != params.end()) { rec.upload_id = id->second; }

    if (method == "POST" && params.contains("uploads")) {
      rec.kind = write_kind::create_upload;
    } else if (method == "POST" && params.contains("uploadId")) {
      rec.kind = write_kind::complete_upload;
    } else if (method == "PUT" && params.contains("partNumber") && params.contains("uploadId")) {
      rec.kind = write_kind::upload_part;
      try {
        auto const n    = std::stoul(params.at("partNumber"));
        rec.part_number = n > 100000 ? 0 : static_cast<unsigned>(n);
      } catch (...) {
        rec.part_number = 0;
      }
    } else if (method == "PUT") {
      rec.kind = write_kind::put_object;
    } else if (method == "DELETE" && params.contains("uploadId")) {
      rec.kind = write_kind::abort_upload;
    } else if (method == "DELETE") {
      rec.kind = write_kind::delete_object;
    } else {
      std::string response{"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0"};
      append_connection_header(response, close_connection);
      response += "\r\n\r\n";
      send_all(conn, response);
      return close_connection;
    }
    if (!header_values(request, "transfer-encoding").empty()) {
      // Only Content-Length bodies are supported.
      record_write(std::move(rec), 501);
      send_write_response(conn, s3_error(501, "NotImplemented", "chunked bodies"), true);
      return true;
    }
    std::size_t length = 0;
    if (auto const cl = header_values(request, "content-length"); !cl.empty()) {
      try {
        length = static_cast<std::size_t>(std::stoull(cl.front()));
      } catch (...) {
        record_write(std::move(rec), 400);
        send_write_response(conn, s3_error(400, "InvalidArgument", "Content-Length"), true);
        return true;
      }
    }
    rec.content_length  = length;
    rec.expect_continue = header_is(request, "expect", "100-continue") && length != 0;

    auto const [index, part_index] = next_write_index(rec.kind, rec.part_number);
    write_counter(rec.kind).fetch_add(1, std::memory_order_relaxed);
    auto const fault = match_fault(rec.kind, index, rec.part_number, part_index);
    std::optional<part_upload_guard> part_guard;
    if (rec.kind == write_kind::upload_part) { part_guard.emplace(*this); }

    if (rec.expect_continue) {
      _expect_continue_count.fetch_add(1, std::memory_order_relaxed);
      expect_continue_mode mode;
      int reject_status = 403;
      {
        std::scoped_lock lock{_store_mutex};
        mode          = _write.expect;
        reject_status = _write.expect_reject_status;
      }
      if (mode == expect_continue_mode::reject) {
        record_write(std::move(rec), reject_status);
        send_write_response(
          conn, s3_error(reject_status, "AccessDenied", "rejected before the body"), true);
        return true;
      }
      if (mode == expect_continue_mode::honor) {
        if (send_all(conn, std::string_view{"HTTP/1.1 100 Continue\r\n\r\n"}) == 0) { return true; }
      }
    }

    std::vector<std::uint8_t> body;
    if (fault && fault->what == write_fault::action::drop_mid_body) {
      (void)read_body(conn, pending, length / 2, body);
      _write_body_bytes.fetch_add(body.size(), std::memory_order_relaxed);
      record_write(std::move(rec), 0);
      conn.drop();
      return true;
    }
    if (!read_body(conn, pending, length, body)) {
      record_write(std::move(rec), 0);
      return true;
    }
    _write_body_bytes.fetch_add(body.size(), std::memory_order_relaxed);

    if (fault) {
      switch (fault->what) {
        case write_fault::action::status: {
          auto r = s3_error(fault->status, fault->code, "injected fault");
          if (!fault->retry_after.empty()) {
            r.headers += "Retry-After: " + fault->retry_after + "\r\n";
          }
          record_write(std::move(rec), fault->status);
          return !send_write_response(conn, r, close_connection) || close_connection;
        }
        case write_fault::action::ok_with_error: {
          write_response r;
          r.headers = "Content-Type: application/xml\r\n";
          r.body    = s3_error_body(fault->code, "injected 200 with an error");
          record_write(std::move(rec), 200);
          return !send_write_response(conn, r, close_connection) || close_connection;
        }
        case write_fault::action::drop_after_body:
          if (fault->apply) { (void)perform_write(rec, request, std::move(body)); }
          record_write(std::move(rec), 0);
          conn.drop();
          return true;
        case write_fault::action::hold:
          if (!fault->control->hold()) { return true; }
          break;
        case write_fault::action::drop_mid_body: break;  // handled above
      }
    }
    auto const response = perform_write(rec, request, std::move(body));
    record_write(std::move(rec), response.status);
    if (fault && fault->control) { fault->control->headers_sent(); }
    return !send_write_response(conn, response, close_connection) || close_connection;
  }

  /// Apply one write request to the store. @return the response to send.
  write_response perform_write(write_record const& rec,
                               std::string const& request,
                               std::vector<std::uint8_t> body)
  {
    std::scoped_lock lock{_store_mutex};
    switch (rec.kind) {
      case write_kind::put_object: {
        if (rec.if_none_match && _store.contains(rec.key)) {
          return s3_error(412, "PreconditionFailed", "At least one of the pre-conditions failed.");
        }
        auto object  = std::make_shared<stored_object_t>();
        object->etag = "\"" + digest_hex(body.data(), body.size()) + "\"";
        object->data = std::move(body);
        write_response r;
        r.headers = "ETag: " + object->etag + "\r\n";
        _store.insert_or_assign(rec.key, std::move(object));
        return r;
      }
      case write_kind::create_upload: {
        auto const id    = "loopback-upload/" + std::to_string(++_next_upload_id) + "+id=";
        _uploads[id].key = rec.key;
        write_response r;
        r.headers = "Content-Type: application/xml\r\n";
        r.body    = std::string(_write.create_whitespace_prefix, ' ') +
                 "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<InitiateMultipartUploadResult "
                 "xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\"><Bucket>" +
                 xml_escape(rec.bucket) + "</Bucket><Key>" + xml_escape(rec.key) +
                 "</Key><UploadId>" + xml_escape(id) +
                 "</UploadId></InitiateMultipartUploadResult>";
        return r;
      }
      case write_kind::upload_part: {
        auto const it = _uploads.find(rec.upload_id);
        if (it == _uploads.end() || it->second.key != rec.key) {
          return s3_error(404, "NoSuchUpload", "The specified upload does not exist.");
        }
        if (rec.part_number < 1 || rec.part_number > 10000) {
          return s3_error(400, "InvalidArgument", "Part number must be between 1 and 10000.");
        }
        auto const etag                   = "\"" + digest_hex(body.data(), body.size()) + "\"";
        it->second.parts[rec.part_number] = {etag, std::move(body)};
        write_response r;
        r.headers = "ETag: " + etag + "\r\n";
        return r;
      }
      case write_kind::complete_upload: return complete_upload(rec, request, body);
      case write_kind::abort_upload: {
        if (_uploads.erase(rec.upload_id) == 0) {
          return s3_error(404, "NoSuchUpload", "The specified upload does not exist.");
        }
        write_response r;
        r.status = 204;
        r.reason = "No Content";
        return r;
      }
      case write_kind::delete_object: {
        _store.erase(rec.key);
        write_response r;
        r.status = 204;
        r.reason = "No Content";
        return r;
      }
    }
    return s3_error(400, "InvalidRequest", "unknown write");
  }

  /// CompleteMultipartUpload (store mutex held).
  write_response complete_upload(write_record const& rec,
                                 std::string const& request,
                                 std::vector<std::uint8_t> const& body)
  {
    (void)request;
    auto const it = _uploads.find(rec.upload_id);
    if (it == _uploads.end() || it->second.key != rec.key) {
      return s3_error(404, "NoSuchUpload", "The specified upload does not exist.");
    }
    if (rec.if_none_match && _store.contains(rec.key)) {
      return s3_error(412, "PreconditionFailed", "At least one of the pre-conditions failed.");
    }
    std::string_view const xml{reinterpret_cast<char const*>(body.data()), body.size()};
    auto const tag = [](std::string_view doc, std::string_view name) {
      std::string const open  = "<" + std::string{name} + ">";
      std::string const close = "</" + std::string{name} + ">";
      auto const a            = doc.find(open);
      if (a == std::string_view::npos) { return std::string_view{}; }
      auto const b = doc.find(close, a + open.size());
      if (b == std::string_view::npos) { return std::string_view{}; }
      return doc.substr(a + open.size(), b - a - open.size());
    };
    std::vector<std::pair<unsigned, std::string>> listed;
    for (std::size_t at = 0; (at = xml.find("<Part>", at)) != std::string_view::npos;) {
      auto const end = xml.find("</Part>", at);
      if (end == std::string_view::npos) { break; }
      auto const part = xml.substr(at, end - at);
      unsigned number = 0;
      try {
        number = static_cast<unsigned>(std::stoul(std::string{tag(part, "PartNumber")}));
      } catch (...) {
        return s3_error(400, "MalformedXML", "bad PartNumber");
      }
      listed.emplace_back(number, xml_unescape(tag(part, "ETag")));
      at = end;
    }
    if (listed.empty()) { return s3_error(400, "MalformedXML", "no parts"); }
    // Validated in passes: order, then every part and its ETag, then the part sizes.
    for (std::size_t i = 1; i < listed.size(); ++i) {
      if (listed[i].first <= listed[i - 1].first) {
        return s3_error(400, "InvalidPartOrder", "parts out of order");
      }
    }
    for (auto const& [number, etag] : listed) {
      auto const part = it->second.parts.find(number);
      if (part == it->second.parts.end() || part->second.etag != etag) {
        return s3_error(400, "InvalidPart", "a part was not found or its ETag did not match");
      }
    }
    for (std::size_t i = 0; i + 1 < listed.size(); ++i) {
      if (it->second.parts.at(listed[i].first).data.size() < _write.min_part_size) {
        return s3_error(400, "EntityTooSmall", "a part other than the last is too small");
      }
    }
    std::vector<std::uint8_t> data;
    std::string etags;
    for (auto const& [number, etag] : listed) {
      auto const& part = it->second.parts.at(number);
      data.insert(data.end(), part.data.begin(), part.data.end());
      etags += part.etag;
    }
    auto object = std::make_shared<stored_object_t>();
    object->etag =
      "\"" + digest_hex(etags.data(), etags.size()) + "-" + std::to_string(listed.size()) + "\"";
    object->data = std::move(data);
    write_response r;
    r.headers = "Content-Type: application/xml\r\n";
    r.body    = std::string(_write.complete_whitespace_prefix, ' ') +
             "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<CompleteMultipartUploadResult "
             "xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\"><Location>http://127.0.0.1/" +
             xml_escape(rec.bucket) + "/" + xml_escape(rec.key) + "</Location><Bucket>" +
             xml_escape(rec.bucket) + "</Bucket><Key>" + xml_escape(rec.key) + "</Key><ETag>" +
             xml_escape(object->etag) + "</ETag></CompleteMultipartUploadResult>";
    _store.insert_or_assign(rec.key, std::move(object));
    _uploads.erase(it);
    return r;
  }

  static std::string xml_unescape(std::string_view s)
  {
    std::string out;
    for (std::size_t i = 0; i < s.size(); ++i) {
      if (s[i] == '&') {
        auto const rest = s.substr(i);
        bool hit        = false;
        for (auto const& [entity, c] : std::array<std::pair<std::string_view, char>, 5>{
               {{"&quot;", '"'}, {"&amp;", '&'}, {"&lt;", '<'}, {"&gt;", '>'}, {"&apos;", '\''}}}) {
          if (rest.starts_with(entity)) {
            out += c;
            i += entity.size() - 1;
            hit = true;
            break;
          }
        }
        if (hit) { continue; }
      }
      out += s[i];
    }
    return out;
  }

  std::atomic<std::size_t>& write_counter(write_kind kind)
  {
    switch (kind) {
      case write_kind::put_object: return _put_object_count;
      case write_kind::create_upload: return _create_upload_count;
      case write_kind::upload_part: return _upload_part_count;
      case write_kind::complete_upload: return _complete_upload_count;
      case write_kind::abort_upload: return _abort_upload_count;
      case write_kind::delete_object: return _delete_object_count;
    }
    return _put_object_count;
  }

  /// @return (index among the requests of @p kind, index among the UploadParts of @p part).
  std::pair<std::size_t, std::size_t> next_write_index(write_kind kind, unsigned part)
  {
    std::scoped_lock lock{_store_mutex};
    auto const k                 = static_cast<std::size_t>(kind);
    std::size_t const index      = _write_index[k]++;
    std::size_t const part_index = kind == write_kind::upload_part ? _part_index[part]++ : 0;
    return {index, part_index};
  }

  void record_write(write_record rec, int status)
  {
    rec.status = status;
    std::scoped_lock lock{_store_mutex};
    _write_records.push_back(std::move(rec));
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

  /// The fixture's listed objects (in their order, unfiltered), then the stored objects whose
  /// key starts with @p prefix (key order).
  [[nodiscard]] std::string list_xml(std::string const& prefix = {}) const
  {
    std::string body =
      "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
      "<ListBucketResult><IsTruncated>false</IsTruncated>";
    for (auto const& object : _listed) {
      body += "<Contents><Key>" + xml_escape(object.key) + "</Key><Size>" +
              std::to_string(object.size) + "</Size></Contents>";
    }
    {
      std::scoped_lock lock{_store_mutex};
      for (auto const& [key, object] : _store) {
        if (!key.starts_with(prefix)) { continue; }
        bool const listed = std::any_of(
          _listed.begin(), _listed.end(), [&](listed_object const& l) { return l.key == key; });
        if (listed) { continue; }
        body += "<Contents><Key>" + xml_escape(key) + "</Key><Size>" +
                std::to_string(object->data.size()) + "</Size><ETag>" + xml_escape(object->etag) +
                "</ETag></Contents>";
      }
    }
    body += "</ListBucketResult>";
    return body;
  }

  [[nodiscard]] static std::optional<std::pair<std::size_t, std::size_t>> parse_range(
    std::string const& request, std::size_t object_size)
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
      std::size_t end   = object_size - 1;
      if (dash == 0) {
        auto const suffix = static_cast<std::size_t>(std::stoull(spec.substr(1)));
        if (suffix == 0) { return std::nullopt; }
        start = suffix >= object_size ? 0 : object_size - suffix;
      } else {
        start = static_cast<std::size_t>(std::stoull(spec.substr(0, dash)));
        if (dash + 1 < spec.size()) {
          end = static_cast<std::size_t>(std::stoull(spec.substr(dash + 1)));
        }
      }
      if (start >= object_size) { return std::nullopt; }
      end = std::min(end, object_size - 1);
      if (end < start) { return std::nullopt; }
      return std::pair{start, end};
    } catch (...) {
      return std::nullopt;
    }
  }

  bool send_body(loopback_connection& conn,
                 std::string_view headers,
                 std::span<std::uint8_t const> object,
                 std::size_t start,
                 std::size_t size,
                 std::uint8_t mask,
                 response_body_policy const& policy,
                 bool chunked = false)
  {
    if (chunked) { return send_chunked(conn, headers, object.subspan(start, size), mask); }
    if (mask == 0) { return send_response(conn, headers, object.subspan(start, size), policy); }
    std::vector<std::uint8_t> masked(object.begin() + static_cast<std::ptrdiff_t>(start),
                                     object.begin() + static_cast<std::ptrdiff_t>(start + size));
    for (auto& byte : masked) {
      byte = static_cast<std::uint8_t>(byte ^ mask);
    }
    return send_response(conn, headers, masked, policy);
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
      char size_line[32];
      std::snprintf(size_line, sizeof(size_line), "%zx\r\n", body.size());
      framed += size_line;
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
  // S3 writes (the store and its policy are guarded by _store_mutex).
  s3_write_policy _write;
  mutable std::mutex _store_mutex;
  std::map<std::string, std::shared_ptr<stored_object_t const>> _store;
  std::map<std::string, upload_t> _uploads;
  std::size_t _next_upload_id{0};
  std::array<std::size_t, 6> _write_index{};
  std::map<unsigned, std::size_t> _part_index;
  std::vector<write_record> _write_records;
  std::atomic<std::size_t> _put_object_count{0};
  std::atomic<std::size_t> _create_upload_count{0};
  std::atomic<std::size_t> _upload_part_count{0};
  std::atomic<std::size_t> _complete_upload_count{0};
  std::atomic<std::size_t> _abort_upload_count{0};
  std::atomic<std::size_t> _delete_object_count{0};
  std::atomic<std::size_t> _expect_continue_count{0};
  std::atomic<std::size_t> _write_body_bytes{0};
  std::atomic<std::size_t> _active_part_uploads{0};
  std::atomic<std::size_t> _peak_part_uploads{0};
  // HTTPS mode.
  SSL_CTX* _tls_ctx{nullptr};
  std::atomic<std::size_t> _tls_handshakes{0};
  std::atomic<std::size_t> _tls_handshake_failures{0};
  std::atomic<std::size_t> _key_updates_sent{0};
  std::atomic<std::size_t> _body_drops{0};
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
