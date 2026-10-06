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

#include <cucascade/io/rest/authorizer.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cucascade::test {

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
};

struct key_response_script {
  std::vector<scripted_response> gets;
  std::vector<scripted_response> heads;
};

class loopback_range_server {
 public:
  explicit loopback_range_server(std::vector<std::uint8_t> object,
                                 range_fault_policy fault                                     = {},
                                 std::vector<listed_object> listed                            = {},
                                 std::unordered_map<std::string, key_response_script> scripts = {})
    : _object(std::move(object)),
      _fault(std::move(fault)),
      _listed(std::move(listed)),
      _scripts(std::move(scripts))
  {
    if (_object.empty()) { throw std::runtime_error("loopback object must be non-empty"); }

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
    _port   = ntohs(addr.sin_port);
    _thread = std::thread([this] { accept_loop(); });
  }

  ~loopback_range_server()
  {
    _stop.store(true, std::memory_order_relaxed);
    _get_response_cv.notify_all();
    if (_listen_fd >= 0) {
      // shutdown() wakes the blocked accept(); close() alone does not reliably interrupt it.
      ::shutdown(_listen_fd, SHUT_RDWR);
      ::close(_listen_fd);
      _listen_fd = -1;
    }
    if (_thread.joinable()) { _thread.join(); }
    for (auto& worker : _workers) {
      if (worker.joinable()) { worker.join(); }
    }
  }

  loopback_range_server(loopback_range_server const&)            = delete;
  loopback_range_server& operator=(loopback_range_server const&) = delete;

  [[nodiscard]] std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(_port); }
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

 private:
  static std::string errno_message() { return std::strerror(errno); }

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

  static void send_interim_headers(int fd,
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
    send_all(fd, response);
  }

  static void append_connection_header(std::string& response, bool close_connection)
  {
    response += close_connection ? "\r\nConnection: close" : "\r\nConnection: keep-alive";
  }

  void accept_loop()
  {
    while (!_stop.load(std::memory_order_relaxed)) {
      sockaddr_in client{};
      socklen_t len = sizeof(client);
      int fd        = ::accept(_listen_fd, reinterpret_cast<sockaddr*>(&client), &len);
      if (fd < 0) {
        if (_stop.load(std::memory_order_relaxed)) { return; }
        continue;
      }
      _accepted_connection_count.fetch_add(1, std::memory_order_relaxed);
      std::scoped_lock lock{_workers_mutex};
      _workers.emplace_back([this, fd] {
        handle_client(fd);
        ::close(fd);
      });
    }
  }

  void handle_client(int fd)
  {
    timeval timeout{};
    timeout.tv_sec = 5;
    (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    std::string pending;
    std::string request;
    while (!_stop.load(std::memory_order_relaxed) && read_request(fd, pending, request)) {
      if (handle_request(fd, request)) { return; }
    }
  }

  bool handle_request(int fd, std::string const& request)
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
      send_interim_headers(fd, _fault.interim_head_etag, {}, _fault.interim_head_retry_after);
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
        send_all(fd, response);
        return close_connection;
      }
      std::string response = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(_object.size());
      append_etag_header(response, response_etag(scripted, _fault.successful_head_etag));
      append_connection_header(response, close_connection);
      response += "\r\n\r\n";
      send_all(fd, response);
      return close_connection;
    }

    if (is_list) {
      if (_fault.response_delay.count() > 0) { std::this_thread::sleep_for(_fault.response_delay); }
      _list_count.fetch_add(1, std::memory_order_relaxed);
      auto const body = list_xml();
      std::string response =
        "HTTP/1.1 200 OK\r\nContent-Type: application/xml\r\nContent-Length: " +
        std::to_string(body.size());
      append_connection_header(response, close_connection);
      response += "\r\n\r\n" + body;
      send_all(fd, response);
      return close_connection;
    }

    if (!is_get) {
      std::string response{"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0"};
      append_connection_header(response, close_connection);
      response += "\r\n\r\n";
      send_all(fd, response);
      return close_connection;
    }

    auto const get_idx = _get_count.fetch_add(1, std::memory_order_relaxed);
    auto const key_idx = increment_request_count(_get_counts_by_key, key);
    wait_at_get_response_barrier();
    auto const scripted = scripted_step(key, true, key_idx);
    delay_response(scripted);
    send_interim_headers(fd,
                         _fault.interim_get_etag,
                         _fault.interim_get_content_range,
                         _fault.interim_get_retry_after);
    bool const scripted_failure = scripted && scripted->status >= 400;
    bool const policy_failure =
      !scripted && (_fault.fail_all_gets || get_idx < _fault.fail_first_gets);
    if (scripted_failure || policy_failure) {
      auto const status    = scripted_failure ? scripted->status : _fault.fail_status;
      std::string response = "HTTP/1.1 " + std::to_string(status) + " Error\r\nContent-Length: 0";
      append_etag_header(response, response_etag(scripted, _fault.failed_get_etag));
      append_header(
        response, "Retry-After", response_retry_after(scripted, _fault.failed_get_retry_after));
      append_connection_header(response, close_connection);
      response += "\r\n\r\n";
      send_all(fd, response);
      return close_connection;
    }

    if (auto range = parse_range(request)) {
      if (_fault.fail_range_with_416) {
        std::string response{"HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0"};
        append_etag_header(response, _fault.failed_get_etag);
        append_connection_header(response, close_connection);
        response += "\r\n\r\n";
        send_all(fd, response);
        return close_connection;
      }
      if (_fault.ignore_range_with_200) {
        std::string response =
          "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(_object.size());
        append_etag_header(response, _fault.failed_get_etag);
        append_connection_header(response, close_connection);
        response += "\r\n\r\n";
        send_all(fd, response);
        send_all(fd, _object.data(), _object.size());
        return close_connection;
      }
      auto const [start, end] = *range;
      auto const size         = end - start + 1;
      std::string response =
        "HTTP/1.1 206 Partial Content\r\nContent-Length: " + std::to_string(size);
      if (_fault.malformed_content_range || (scripted && scripted->malformed_content_range)) {
        response += "\r\nContent-Range: bytes malformed";
        append_etag_header(response, _fault.failed_get_etag);
      } else {
        if (!_fault.omit_successful_content_range) {
          response += "\r\nContent-Range: bytes " + std::to_string(start) + "-" +
                      std::to_string(end) + "/" + std::to_string(_object.size());
        }
        append_etag_header(response, response_etag(scripted, _fault.successful_get_etag));
      }
      append_connection_header(response, close_connection);
      response += "\r\n\r\n";
      send_all(fd, response);
      send_all(fd, _object.data() + start, size);
      return close_connection;
    }

    std::string response = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(_object.size());
    append_etag_header(response, response_etag(scripted, _fault.successful_get_etag));
    append_connection_header(response, close_connection);
    response += "\r\n\r\n";
    send_all(fd, response);
    send_all(fd, _object.data(), _object.size());
    return close_connection;
  }

  static bool read_request(int fd, std::string& pending, std::string& request)
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
      ssize_t const n = ::recv(fd, bytes, sizeof(bytes), 0);
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

  void delay_response(std::optional<scripted_response> const& scripted) const
  {
    auto delay = _fault.response_delay;
    if (scripted) { delay += scripted->delay; }
    if (delay.count() > 0) { std::this_thread::sleep_for(delay); }
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

  static void send_all(int fd, std::string_view bytes)
  {
    send_all(fd, reinterpret_cast<std::uint8_t const*>(bytes.data()), bytes.size());
  }

  static void send_all(int fd, std::uint8_t const* bytes, std::size_t size)
  {
    std::size_t sent = 0;
    // Fault tests can disconnect mid-response; suppress SIGPIPE so send() reports the failure.
    while (sent < size) {
      ssize_t const n = ::send(fd, bytes + sent, size - sent, MSG_NOSIGNAL);
      if (n <= 0) { return; }
      sent += static_cast<std::size_t>(n);
    }
  }

  int _listen_fd{-1};
  std::uint16_t _port{0};
  std::vector<std::uint8_t> _object;
  range_fault_policy _fault;
  std::vector<listed_object> _listed;
  std::unordered_map<std::string, key_response_script> _scripts;
  std::atomic<bool> _stop{false};
  std::atomic<std::size_t> _accepted_connection_count{0};
  std::atomic<std::size_t> _head_count{0};
  std::atomic<std::size_t> _get_count{0};
  std::atomic<std::size_t> _list_count{0};
  std::mutex _get_response_mutex;
  std::condition_variable _get_response_cv;
  mutable std::mutex _request_counts_mutex;
  std::unordered_map<std::string, std::size_t> _head_counts_by_key;
  std::unordered_map<std::string, std::size_t> _get_counts_by_key;
  std::thread _thread;
  std::mutex _workers_mutex;
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
