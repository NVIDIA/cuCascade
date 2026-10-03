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

// Minimal in-memory S3-compatible object store on a loopback socket, for the
// REST write tests: GET (ranged) / HEAD / PUT objects, CreateMultipartUpload,
// UploadPart, CompleteMultipartUpload and AbortMultipartUpload, with fault
// injection.  HTTP/1.1 keep-alive, Content-Length bodies only, path-style
// addressing ("/<bucket>/<key>"), one thread per connection.

#include <cucascade/io/rest/authorizer.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace cucascade::test {

/// Fault injection knobs of @ref loopback_object_store (all off by default).
struct object_store_faults {
  /// The first N UploadPart requests answer @c part_fail_status.
  std::size_t fail_first_part_puts{0};
  /// Every UploadPart answers @c part_fail_status.
  bool fail_all_part_puts{false};
  int part_fail_status{503};
  /// The first N CompleteMultipartUpload requests answer 200 with an <Error> body.
  std::size_t complete_error_body_first{0};
  /// The first N CompleteMultipartUpload requests that succeed assemble the
  /// object but answer 503 (the success response is "lost"): the client's
  /// retry then finds the upload gone (404 NoSuchUpload).
  std::size_t lose_first_complete_responses{0};
  /// Like @c lose_first_complete_responses, but the object is also replaced
  /// by @c lost_complete_replacement_size bytes afterwards (someone else
  /// overwrote the key): verification must not accept it.
  bool replace_after_lost_complete{false};
  std::size_t lost_complete_replacement_size{1};
  /// The first N single-object PUTs answer @c put_fail_status.
  std::size_t fail_first_puts{0};
  int put_fail_status{503};
  /// Delay before answering an UploadPart (the body has been received).
  std::chrono::milliseconds part_delay{0};
  /// Smallest accepted non-last part (S3: 5 MiB).
  std::size_t min_part_size{5UL << 20};
};

class loopback_object_store {
 public:
  explicit loopback_object_store(object_store_faults faults = {}) : _faults(faults)
  {
    _listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (_listen_fd < 0) throw std::runtime_error("socket failed");
    int one = 1;
    ::setsockopt(_listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = 0;
    if (::bind(_listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
        ::listen(_listen_fd, 64) != 0) {
      ::close(_listen_fd);
      throw std::runtime_error("bind/listen failed");
    }
    socklen_t len = sizeof(addr);
    ::getsockname(_listen_fd, reinterpret_cast<sockaddr*>(&addr), &len);
    _port   = ntohs(addr.sin_port);
    _thread = std::thread([this] { accept_loop(); });
  }

  ~loopback_object_store()
  {
    _stop.store(true);
    ::shutdown(_listen_fd, SHUT_RDWR);
    ::close(_listen_fd);
    if (_thread.joinable()) _thread.join();
    std::vector<std::thread> workers;
    {
      std::lock_guard lock{_workers_mutex};
      workers = std::move(_workers);
    }
    for (auto& worker : workers) {
      if (worker.joinable()) worker.join();
    }
  }

  loopback_object_store(loopback_object_store const&)            = delete;
  loopback_object_store& operator=(loopback_object_store const&) = delete;

  [[nodiscard]] std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(_port); }

  /// Store @p bytes as object @p path ("bucket/key").
  void put_object(std::string const& path, std::vector<std::uint8_t> bytes)
  {
    std::lock_guard lock{_mutex};
    _objects[path] = std::move(bytes);
  }

  /// Committed object @p path ("bucket/key"), if any.
  [[nodiscard]] std::optional<std::vector<std::uint8_t>> object(std::string const& path) const
  {
    std::lock_guard lock{_mutex};
    auto const it = _objects.find(path);
    if (it == _objects.end()) return std::nullopt;
    return it->second;
  }

  /// Multipart uploads created and neither completed nor aborted.
  [[nodiscard]] std::size_t live_uploads() const
  {
    std::lock_guard lock{_mutex};
    return _uploads.size();
  }

  [[nodiscard]] std::size_t gets() const noexcept { return _gets.load(); }
  [[nodiscard]] std::size_t puts() const noexcept { return _puts.load(); }
  [[nodiscard]] std::size_t initiates() const noexcept { return _initiates.load(); }
  /// UploadPart requests received (incl. failed ones), counted when the body arrived.
  [[nodiscard]] std::size_t part_puts() const noexcept { return _part_puts.load(); }
  [[nodiscard]] std::size_t completes() const noexcept { return _completes.load(); }
  [[nodiscard]] std::size_t aborts() const noexcept { return _aborts.load(); }

 private:
  struct http_request {
    std::string method;
    std::string path;  // "bucket/key" (decoded)
    std::map<std::string, std::string> query;
    std::map<std::string, std::string> headers;  // lower-case names
    std::string body;
  };

  struct upload {
    std::string path;
    std::map<std::uint32_t, std::pair<std::string, std::vector<std::uint8_t>>> parts;
  };

  static std::string lower(std::string value)
  {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    return value;
  }

  static std::string url_decode(std::string_view value)
  {
    std::string out;
    for (std::size_t i = 0; i < value.size(); ++i) {
      if (value[i] == '%' && i + 2 < value.size()) {
        out.push_back(
          static_cast<char>(std::stoi(std::string(value.substr(i + 1, 2)), nullptr, 16)));
        i += 2;
      } else {
        out.push_back(value[i]);
      }
    }
    return out;
  }

  static std::string xml_unescape(std::string value)
  {
    auto replace_all = [&](std::string const& from, std::string const& to) {
      for (std::size_t pos = 0; (pos = value.find(from, pos)) != std::string::npos;) {
        value.replace(pos, from.size(), to);
        pos += to.size();
      }
    };
    replace_all("&quot;", "\"");
    replace_all("&apos;", "'");
    replace_all("&lt;", "<");
    replace_all("&gt;", ">");
    replace_all("&amp;", "&");
    return value;
  }

  static std::string etag_of(std::vector<std::uint8_t> const& bytes)
  {
    std::uint64_t hash = 1469598103934665603ULL;
    for (auto const byte : bytes) {
      hash ^= byte;
      hash *= 1099511628211ULL;
    }
    char text[24];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(hash));
    return "\"" + std::string(text) + "\"";
  }

  static void send_all(int fd, std::string_view bytes)
  {
    std::size_t sent = 0;
    while (sent < bytes.size()) {
      auto const n = ::send(fd, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL);
      if (n <= 0) return;
      sent += static_cast<std::size_t>(n);
    }
  }

  static void respond(int fd,
                      int status,
                      std::string const& body                                  = {},
                      std::vector<std::pair<std::string, std::string>> headers = {})
  {
    std::string out = "HTTP/1.1 " + std::to_string(status) +
                      " X\r\nContent-Length: " + std::to_string(body.size()) +
                      "\r\nConnection: keep-alive";
    for (auto const& [name, value] : headers) {
      out += "\r\n" + name + ": " + value;
    }
    out += "\r\n\r\n";
    out += body;
    send_all(fd, out);
  }

  static std::string error_xml(std::string const& code)
  {
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Error><Code>" + code +
           "</Code><Message>injected</Message><RequestId>r</RequestId></Error>";
  }

  /// Read until @p pending holds @p bytes bytes (or the peer / stop ends it).
  bool fill(int fd, std::string& pending, std::size_t bytes)
  {
    char buffer[65536];
    while (pending.size() < bytes) {
      if (_stop.load()) return false;
      pollfd pfd{fd, POLLIN, 0};
      auto const ready = ::poll(&pfd, 1, 100);
      if (ready < 0 && errno != EINTR) return false;
      if (ready <= 0) continue;
      auto const n = ::recv(fd, buffer, sizeof(buffer), 0);
      if (n <= 0) return false;
      pending.append(buffer, static_cast<std::size_t>(n));
    }
    return true;
  }

  bool read_request(int fd, std::string& pending, http_request& out)
  {
    std::size_t end = std::string::npos;
    char buffer[65536];
    while ((end = pending.find("\r\n\r\n")) == std::string::npos) {
      if (_stop.load()) return false;
      pollfd pfd{fd, POLLIN, 0};
      auto const ready = ::poll(&pfd, 1, 100);
      if (ready < 0 && errno != EINTR) return false;
      if (ready <= 0) continue;
      auto const n = ::recv(fd, buffer, sizeof(buffer), 0);
      if (n <= 0) return false;
      pending.append(buffer, static_cast<std::size_t>(n));
    }
    auto const head = pending.substr(0, end);
    pending.erase(0, end + 4);

    out                 = http_request{};
    auto const line_end = head.find("\r\n");
    auto const line     = head.substr(0, line_end);
    auto const sp1      = line.find(' ');
    auto const sp2      = line.find(' ', sp1 + 1);
    out.method          = line.substr(0, sp1);
    auto target         = line.substr(sp1 + 1, sp2 - sp1 - 1);
    if (auto const q = target.find('?'); q != std::string::npos) {
      std::string_view query{target};
      query.remove_prefix(q + 1);
      while (!query.empty()) {
        auto const amp  = query.find('&');
        auto const pair = query.substr(0, amp);
        auto const eq   = pair.find('=');
        out.query[url_decode(pair.substr(0, eq))] =
          eq == std::string_view::npos ? std::string{} : url_decode(pair.substr(eq + 1));
        if (amp == std::string_view::npos) break;
        query.remove_prefix(amp + 1);
      }
      target.resize(q);
    }
    if (!target.empty() && target.front() == '/') target.erase(0, 1);
    out.path = url_decode(target);

    std::size_t pos = line_end == std::string::npos ? head.size() : line_end + 2;
    while (pos < head.size()) {
      auto next = head.find("\r\n", pos);
      if (next == std::string::npos) next = head.size();
      auto const header = head.substr(pos, next - pos);
      if (auto const colon = header.find(':'); colon != std::string::npos) {
        auto value = header.substr(colon + 1);
        while (!value.empty() && value.front() == ' ')
          value.erase(0, 1);
        out.headers[lower(header.substr(0, colon))] = value;
      }
      pos = next + 2;
    }

    if (auto const expect = out.headers.find("expect");
        expect != out.headers.end() && lower(expect->second) == "100-continue") {
      send_all(fd, "HTTP/1.1 100 Continue\r\n\r\n");
    }
    std::size_t length = 0;
    if (auto const cl = out.headers.find("content-length"); cl != out.headers.end()) {
      length = static_cast<std::size_t>(std::stoull(cl->second));
    }
    if (!fill(fd, pending, length)) return false;
    out.body = pending.substr(0, length);
    pending.erase(0, length);
    return true;
  }

  void accept_loop()
  {
    while (!_stop.load()) {
      int const fd = ::accept(_listen_fd, nullptr, nullptr);
      if (fd < 0) {
        if (_stop.load()) return;
        continue;
      }
      int nodelay = 1;
      ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
      std::lock_guard lock{_workers_mutex};
      _workers.emplace_back([this, fd] {
        std::string pending;
        http_request request;
        while (!_stop.load() && read_request(fd, pending, request)) {
          handle(fd, request);
        }
        ::close(fd);
      });
    }
  }

  void handle(int fd, http_request const& request)
  {
    auto const& q       = request.query;
    bool const has_id   = q.count("uploadId") != 0;
    bool const has_part = q.count("partNumber") != 0;

    if (request.method == "GET" && q.count("list-type") != 0) {
      respond(fd, 200, "<ListBucketResult><IsTruncated>false</IsTruncated></ListBucketResult>");
      return;
    }
    if (request.method == "GET" || request.method == "HEAD") {
      ++_gets;
      std::vector<std::uint8_t> bytes;
      {
        std::lock_guard lock{_mutex};
        auto const it = _objects.find(request.path);
        if (it == _objects.end()) {
          respond(fd, 404, request.method == "GET" ? error_xml("NoSuchKey") : std::string{});
          return;
        }
        bytes = it->second;
      }
      auto const etag = etag_of(bytes);
      if (request.method == "HEAD") {
        send_all(fd,
                 "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(bytes.size()) +
                   "\r\nETag: " + etag + "\r\nConnection: keep-alive\r\n\r\n");
        return;
      }
      if (auto const range = request.headers.find("range"); range != request.headers.end()) {
        auto const spec  = range->second.substr(range->second.find('=') + 1);
        auto const dash  = spec.find('-');
        auto const first = static_cast<std::size_t>(std::stoull(spec.substr(0, dash)));
        auto last        = static_cast<std::size_t>(std::stoull(spec.substr(dash + 1)));
        if (first >= bytes.size()) {
          respond(fd, 416);
          return;
        }
        last = std::min(last, bytes.size() - 1);
        std::string body(bytes.begin() + static_cast<std::ptrdiff_t>(first),
                         bytes.begin() + static_cast<std::ptrdiff_t>(last + 1));
        respond(fd,
                206,
                body,
                {{"Content-Range",
                  "bytes " + std::to_string(first) + "-" + std::to_string(last) + "/" +
                    std::to_string(bytes.size())},
                 {"ETag", etag}});
        return;
      }
      respond(fd, 200, std::string(bytes.begin(), bytes.end()), {{"ETag", etag}});
      return;
    }

    if (request.method == "PUT" && !has_id) {
      auto const n = _puts.fetch_add(1);
      if (n < _faults.fail_first_puts) {
        respond(fd, _faults.put_fail_status, error_xml("SlowDown"));
        return;
      }
      std::vector<std::uint8_t> bytes(request.body.begin(), request.body.end());
      auto const etag = etag_of(bytes);
      put_object(request.path, std::move(bytes));
      respond(fd, 200, {}, {{"ETag", etag}});
      return;
    }

    if (request.method == "POST" && q.count("uploads") != 0) {
      ++_initiates;
      std::string id;
      {
        std::lock_guard lock{_mutex};
        id           = "upload" + std::to_string(++_next_upload);
        _uploads[id] = upload{request.path, {}};
      }
      auto const slash = request.path.find('/');
      respond(fd,
              200,
              "<?xml version=\"1.0\" encoding=\"UTF-8\"?><InitiateMultipartUploadResult><Bucket>" +
                request.path.substr(0, slash) + "</Bucket><Key>" + request.path.substr(slash + 1) +
                "</Key><UploadId>" + id + "</UploadId></InitiateMultipartUploadResult>");
      return;
    }

    if (request.method == "PUT" && has_id && has_part) {
      auto const n = _part_puts.fetch_add(1);
      if (_faults.part_delay.count() > 0) std::this_thread::sleep_for(_faults.part_delay);
      if (_faults.fail_all_part_puts || n < _faults.fail_first_part_puts) {
        respond(fd, _faults.part_fail_status, error_xml("InjectedFailure"));
        return;
      }
      auto const number = static_cast<std::uint32_t>(std::stoul(q.at("partNumber")));
      std::vector<std::uint8_t> bytes(request.body.begin(), request.body.end());
      auto const etag = etag_of(bytes);
      {
        std::lock_guard lock{_mutex};
        auto const it = _uploads.find(q.at("uploadId"));
        if (it == _uploads.end()) {
          respond(fd, 404, error_xml("NoSuchUpload"));
          return;
        }
        it->second.parts[number] = {etag, std::move(bytes)};
      }
      respond(fd, 200, {}, {{"ETag", etag}});
      return;
    }

    if (request.method == "POST" && has_id) {
      auto const n = _completes.fetch_add(1);
      if (n < _faults.complete_error_body_first) {
        respond(fd, 200, error_xml("InternalError"));
        return;
      }
      std::lock_guard lock{_mutex};
      auto const it = _uploads.find(q.at("uploadId"));
      if (it == _uploads.end()) {
        respond(fd, 404, error_xml("NoSuchUpload"));
        return;
      }
      std::vector<std::pair<std::uint32_t, std::string>> listed;
      for (std::size_t pos = 0; (pos = request.body.find("<Part>", pos)) != std::string::npos;) {
        auto const end      = request.body.find("</Part>", pos);
        auto const part     = request.body.substr(pos, end - pos);
        auto const number_b = part.find("<PartNumber>") + 12;
        auto const number_e = part.find("</PartNumber>");
        auto const etag_b   = part.find("<ETag>") + 6;
        auto const etag_e   = part.find("</ETag>");
        listed.emplace_back(
          static_cast<std::uint32_t>(std::stoul(part.substr(number_b, number_e - number_b))),
          xml_unescape(part.substr(etag_b, etag_e - etag_b)));
        pos = end;
      }
      std::vector<std::uint8_t> assembled;
      for (std::size_t i = 0; i < listed.size(); ++i) {
        auto const found     = it->second.parts.find(listed[i].first);
        bool const ascending = i == 0 || listed[i].first > listed[i - 1].first;
        if (found == it->second.parts.end() || found->second.first != listed[i].second ||
            !ascending) {
          respond(fd, 400, error_xml("InvalidPart"));
          return;
        }
        if (i + 1 < listed.size() && found->second.second.size() < _faults.min_part_size) {
          respond(fd, 400, error_xml("EntityTooSmall"));
          return;
        }
        assembled.insert(assembled.end(), found->second.second.begin(), found->second.second.end());
      }
      if (listed.empty()) {
        respond(fd, 400, error_xml("MalformedXML"));
        return;
      }
      auto const path = it->second.path;
      _objects[path]  = std::move(assembled);
      _uploads.erase(it);
      if (_lost_completes < _faults.lose_first_complete_responses) {
        ++_lost_completes;
        if (_faults.replace_after_lost_complete) {
          _objects[path] = std::vector<std::uint8_t>(_faults.lost_complete_replacement_size, 0);
        }
        respond(fd, 503, error_xml("ServiceUnavailable"));
        return;
      }
      respond(fd,
              200,
              "<?xml version=\"1.0\" encoding=\"UTF-8\"?><CompleteMultipartUploadResult><ETag>"
              "&quot;done&quot;</ETag></CompleteMultipartUploadResult>");
      return;
    }

    if (request.method == "DELETE" && has_id) {
      ++_aborts;
      std::lock_guard lock{_mutex};
      auto const erased = _uploads.erase(q.at("uploadId"));
      respond(fd, erased != 0 ? 204 : 404, erased != 0 ? std::string{} : error_xml("NoSuchUpload"));
      return;
    }

    respond(fd, 405, error_xml("MethodNotAllowed"));
  }

  object_store_faults const _faults;
  int _listen_fd{-1};
  std::uint16_t _port{0};
  std::atomic<bool> _stop{false};
  std::thread _thread;
  std::mutex _workers_mutex;
  std::vector<std::thread> _workers;

  mutable std::mutex _mutex;
  std::map<std::string, std::vector<std::uint8_t>> _objects;  // guarded by _mutex
  std::map<std::string, upload> _uploads;                     // guarded by _mutex
  std::size_t _next_upload{0};                                // guarded by _mutex

  std::atomic<std::size_t> _gets{0};
  std::atomic<std::size_t> _puts{0};
  std::atomic<std::size_t> _initiates{0};
  std::atomic<std::size_t> _part_puts{0};
  std::atomic<std::size_t> _completes{0};
  std::size_t _lost_completes{0};  // guarded by _mutex
  std::atomic<std::size_t> _aborts{0};
};

/// Authorizer routing every request of @ref loopback_object_store by
/// bucket / key, appending the request's canonical query.
class object_store_authorizer final : public io::rest::request_authorizer {
 public:
  explicit object_store_authorizer(std::string endpoint) : _endpoint(std::move(endpoint)) {}

  io::rest::authorized_request authorize(io::rest::object_ref const& obj,
                                         io::rest::request_method,
                                         std::chrono::seconds) override
  {
    return {_endpoint + "/" + obj.bucket + "/" + obj.key, {}};
  }

  io::rest::authorized_request authorize_list(std::string_view bucket,
                                              std::string_view canonical_query,
                                              std::chrono::seconds) override
  {
    return {_endpoint + "/" + std::string{bucket} + "?" + std::string{canonical_query}, {}};
  }

  io::rest::authorized_request authorize_request(io::rest::request_spec const& spec,
                                                 std::chrono::seconds) override
  {
    ++_requests;
    auto url = _endpoint + "/" + spec.object.bucket + "/" + spec.object.key;
    if (!spec.canonical_query.empty()) url += "?" + spec.canonical_query;
    return {std::move(url), spec.extra_headers};
  }

  [[nodiscard]] std::size_t requests() const noexcept { return _requests.load(); }

 private:
  std::string _endpoint;
  std::atomic<std::size_t> _requests{0};
};

}  // namespace cucascade::test
