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

#include "mock_authorizer.hpp"

#include <cucascade/io/io_errors.hpp>
#include <cucascade/io/rest/config.hpp>
#include <cucascade/io/rest/details/sync_request.hpp>
#include <cucascade/io/rest/s3/xml_utils.hpp>

#include <arpa/inet.h>
#include <catch2/catch_all.hpp>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using cucascade::io::credential_error;
using cucascade::io::rest::authorized_request;
using cucascade::io::rest::config;
using cucascade::io::rest::mock_authorizer;
using cucascade::io::rest::request_method;
using cucascade::io::rest::request_spec;
using cucascade::io::rest::detail::perform_sync;
using cucascade::io::rest::detail::sync_request_options;

namespace {

/// One request as seen by the scripted server.
struct seen_request {
  std::string method;
  std::string target;
  std::vector<std::pair<std::string, std::string>> headers;
  std::string body;

  [[nodiscard]] std::string header(std::string_view name) const
  {
    for (auto const& [k, v] : headers) {
      if (k.size() == name.size() &&
          std::equal(k.begin(), k.end(), name.begin(), [](unsigned char a, unsigned char b) {
            return std::tolower(a) == std::tolower(b);
          })) {
        return v;
      }
    }
    return "<absent>";
  }
};

/// Scripted reply; once the script is exhausted the server repeats the last one.
struct scripted_reply {
  int status{200};
  std::string body;
  std::vector<std::pair<std::string, std::string>> headers;
};

/// Minimal HTTP/1.1 loopback server: one request per connection
/// (Connection: close), Content-Length bodies only.
class scripted_http_server {
 public:
  explicit scripted_http_server(std::vector<scripted_reply> script) : _script(std::move(script))
  {
    if (_script.empty()) { _script.push_back({}); }
    _fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (_fd < 0) { throw std::runtime_error("socket failed"); }
    int one = 1;
    ::setsockopt(_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = 0;
    if (::bind(_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
        ::listen(_fd, 16) != 0) {
      ::close(_fd);
      throw std::runtime_error("bind/listen failed");
    }
    socklen_t len = sizeof(addr);
    ::getsockname(_fd, reinterpret_cast<sockaddr*>(&addr), &len);
    _port   = ntohs(addr.sin_port);
    _thread = std::thread([this] { serve(); });
  }

  ~scripted_http_server()
  {
    _stop.store(true);
    ::shutdown(_fd, SHUT_RDWR);
    ::close(_fd);
    if (_thread.joinable()) { _thread.join(); }
  }

  scripted_http_server(scripted_http_server const&)            = delete;
  scripted_http_server& operator=(scripted_http_server const&) = delete;

  [[nodiscard]] std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(_port); }

  [[nodiscard]] std::vector<seen_request> requests() const
  {
    std::scoped_lock lk{_m};
    return _seen;
  }

 private:
  void serve()
  {
    while (!_stop.load()) {
      int const c = ::accept(_fd, nullptr, nullptr);
      if (c < 0) { return; }
      handle(c);
      ::close(c);
    }
  }

  void handle(int c)
  {
    std::string buf;
    char chunk[64 << 10];
    std::size_t header_end = std::string::npos;
    while (header_end == std::string::npos) {
      auto const n = ::recv(c, chunk, sizeof(chunk), 0);
      if (n <= 0) { return; }
      buf.append(chunk, static_cast<std::size_t>(n));
      header_end = buf.find("\r\n\r\n");
    }
    seen_request req;
    std::string_view head{buf.data(), header_end};
    auto const line_end        = head.find("\r\n");
    auto const line            = head.substr(0, line_end);
    auto const sp1             = line.find(' ');
    auto const sp2             = line.find(' ', sp1 + 1);
    req.method                 = std::string{line.substr(0, sp1)};
    req.target                 = std::string{line.substr(sp1 + 1, sp2 - sp1 - 1)};
    std::size_t content_length = 0;
    for (std::size_t p = line_end + 2; p < head.size();) {
      auto e = head.find("\r\n", p);
      if (e == std::string_view::npos) { e = head.size(); }
      auto const h     = head.substr(p, e - p);
      auto const colon = h.find(':');
      if (colon != std::string_view::npos) {
        auto v = h.substr(colon + 1);
        while (!v.empty() && v.front() == ' ') {
          v.remove_prefix(1);
        }
        req.headers.emplace_back(std::string{h.substr(0, colon)}, std::string{v});
      }
      p = e + 2;
    }
    if (auto const cl = req.header("Content-Length"); cl != "<absent>") {
      content_length = std::stoul(cl);
    }
    req.body = buf.substr(header_end + 4);
    while (req.body.size() < content_length) {
      auto const n = ::recv(c, chunk, sizeof(chunk), 0);
      if (n <= 0) { break; }
      req.body.append(chunk, static_cast<std::size_t>(n));
    }

    scripted_reply reply;
    {
      std::scoped_lock lk{_m};
      reply = _script[std::min(_seen.size(), _script.size() - 1)];
      _seen.push_back(std::move(req));
    }
    bool const is_head = _seen_last_method_is_head();
    std::string out    = "HTTP/1.1 " + std::to_string(reply.status) + " Scripted\r\n";
    for (auto const& [k, v] : reply.headers) {
      out += k + ": " + v + "\r\n";
    }
    out += "Content-Length: " + std::to_string(reply.body.size()) + "\r\nConnection: close\r\n\r\n";
    if (!is_head) { out += reply.body; }
    std::size_t sent = 0;
    while (sent < out.size()) {
      auto const n = ::send(c, out.data() + sent, out.size() - sent, MSG_NOSIGNAL);
      if (n <= 0) { break; }
      sent += static_cast<std::size_t>(n);
    }
  }

  bool _seen_last_method_is_head() const
  {
    std::scoped_lock lk{_m};
    return !_seen.empty() && _seen.back().method == "HEAD";
  }

  std::vector<scripted_reply> _script;
  int _fd{-1};
  std::uint16_t _port{0};
  std::atomic<bool> _stop{false};
  mutable std::mutex _m;
  std::vector<seen_request> _seen;
  std::thread _thread;
};

config fast_retry_config()
{
  config cfg;
  cfg.request_timeout_s       = 10;
  cfg.max_retry_attempts      = 4;
  cfg.max_auth_retry_attempts = 2;
  cfg.retry_backoff_base      = std::chrono::milliseconds{1};
  cfg.retry_jitter            = std::chrono::milliseconds{0};
  cfg.honor_retry_after       = false;
  return cfg;
}

request_spec make_spec(request_method method, std::string query = {})
{
  request_spec spec;
  spec.method          = method;
  spec.object          = {"bucket", "key"};
  spec.canonical_query = std::move(query);
  return spec;
}

std::string const k_error_body =
  "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
  "<Error><Code>InternalError</Code><Message>retry</Message></Error>";

}  // namespace

TEST_CASE("perform_sync uploads a PUT body and captures the ETag", "[rest][sync]")
{
  scripted_http_server server({{200, "", {{"ETag", "\"0123abcd\""}}}});
  mock_authorizer auth(authorized_request{server.endpoint() + "/bucket/key", {{"x-auth", "a"}}});

  std::string body(3UL << 20, '\0');
  for (std::size_t i = 0; i < body.size(); ++i) {
    body[i] = static_cast<char>(i * 131 % 251);
  }
  auto const resp = perform_sync(make_spec(request_method::PUT, "partNumber=1&uploadId=u1"),
                                 auth,
                                 fast_retry_config(),
                                 body,
                                 {.data_transfer = true});
  CHECK(resp.status == 200);
  CHECK(resp.etag == "\"0123abcd\"");
  CHECK(resp.attempts == 1);

  auto const seen = server.requests();
  REQUIRE(seen.size() == 1);
  CHECK(seen[0].method == "PUT");
  CHECK(seen[0].target == "/bucket/key?partNumber=1&uploadId=u1");
  CHECK(seen[0].header("x-auth") == "a");
  CHECK(seen[0].header("Expect") == "<absent>");
  CHECK(seen[0].header("Content-Length") == std::to_string(body.size()));
  CHECK(seen[0].body == body);
  CHECK(auth.request_count() == 1);
  CHECK(auth.last_method() == request_method::PUT);
}

TEST_CASE("perform_sync POSTs CreateMultipartUpload and returns the XML body", "[rest][sync]")
{
  scripted_http_server server(
    {{200,
      "<InitiateMultipartUploadResult><Bucket>bucket</Bucket><Key>key"
      "</Key><UploadId>upload-1</UploadId></InitiateMultipartUploadResult>",
      {}}});
  mock_authorizer auth(authorized_request{server.endpoint() + "/bucket/key", {}});

  auto const resp =
    perform_sync(make_spec(request_method::POST, "uploads="), auth, fast_retry_config());
  CHECK(cucascade::io::rest::s3::parse_initiate_multipart_upload(resp.body) == "upload-1");

  auto const seen = server.requests();
  REQUIRE(seen.size() == 1);
  CHECK(seen[0].method == "POST");
  CHECK(seen[0].target == "/bucket/key?uploads=");
  CHECK(seen[0].body.empty());
  CHECK(seen[0].header("Content-Type") == "<absent>");  // no curl form default
}

TEST_CASE("perform_sync retries a 200 response carrying an S3 <Error> body", "[rest][sync]")
{
  std::string const ok_body =
    "<CompleteMultipartUploadResult><ETag>\"x-2\"</ETag>"
    "</CompleteMultipartUploadResult>";
  scripted_http_server server({{200, k_error_body, {}}, {200, ok_body, {}}});
  mock_authorizer auth(authorized_request{server.endpoint() + "/bucket/key", {}});

  std::string const complete =
    "<CompleteMultipartUpload><Part><PartNumber>1</PartNumber><ETag>\"e\"</ETag></Part>"
    "</CompleteMultipartUpload>";
  auto spec = make_spec(request_method::POST, "uploadId=u1");
  spec.extra_headers.emplace_back("Content-Type", "application/xml");
  auto const resp =
    perform_sync(spec, auth, fast_retry_config(), complete, {.retry_on_error_body = true});
  CHECK(resp.attempts == 2);
  CHECK(resp.body == ok_body);
  CHECK(auth.request_count() == 2);  // re-authorized per attempt

  auto const seen = server.requests();
  REQUIRE(seen.size() == 2);
  CHECK(seen[0].body == complete);
  CHECK(seen[1].body == complete);
  CHECK(seen[1].header("Content-Type") == "application/xml");

  SECTION("without retry_on_error_body the 200 is accepted as is")
  {
    scripted_http_server lenient({{200, k_error_body, {}}});
    mock_authorizer auth2(authorized_request{lenient.endpoint() + "/bucket/key", {}});
    auto const r = perform_sync(spec, auth2, fast_retry_config(), complete);
    CHECK(r.attempts == 1);
    CHECK(cucascade::io::rest::s3::parse_s3_error(r.body).has_value());
  }
}

TEST_CASE("perform_sync retries transient statuses and S3 RequestTimeout", "[rest][sync]")
{
  scripted_http_server server({{503, "", {}},
                               {400, "<Error><Code>RequestTimeout</Code></Error>", {}},
                               {200, "", {{"ETag", "\"p\""}}}});
  mock_authorizer auth(authorized_request{server.endpoint() + "/bucket/key", {}});

  auto const resp =
    perform_sync(make_spec(request_method::PUT), auth, fast_retry_config(), "payload");
  CHECK(resp.attempts == 3);
  CHECK(resp.etag == "\"p\"");
  auto const seen = server.requests();
  REQUIRE(seen.size() == 3);
  for (auto const& r : seen) {
    CHECK(r.body == "payload");  // body re-sent in full on every attempt
  }
}

TEST_CASE("perform_sync fails fast on a non-retriable status with the S3 code", "[rest][sync]")
{
  scripted_http_server server({{404, "<Error><Code>NoSuchUpload</Code></Error>", {}}});
  mock_authorizer auth(authorized_request{server.endpoint() + "/bucket/key", {}});

  CHECK_THROWS_WITH(
    perform_sync(make_spec(request_method::DELETE_, "uploadId=gone"), auth, fast_retry_config()),
    Catch::Matchers::ContainsSubstring("DELETE HTTP 404: NoSuchUpload for bucket/key"));
  CHECK(server.requests().size() == 1);
}

TEST_CASE("perform_sync DELETE accepts a configured status", "[rest][sync]")
{
  scripted_http_server server({{204, "", {}}});
  mock_authorizer auth(authorized_request{server.endpoint() + "/bucket/key", {}});
  auto const resp = perform_sync(make_spec(request_method::DELETE_, "uploadId=u1"),
                                 auth,
                                 fast_retry_config(),
                                 {},
                                 {.accepted_statuses = {200, 204}});
  CHECK(resp.status == 204);
  auto const seen = server.requests();
  REQUIRE(seen.size() == 1);
  CHECK(seen[0].method == "DELETE");
  CHECK(seen[0].target == "/bucket/key?uploadId=u1");
}

TEST_CASE("perform_sync GET and HEAD", "[rest][sync]")
{
  scripted_http_server server({{200, "hello", {{"ETag", "\"g\""}}}});
  mock_authorizer auth(authorized_request{server.endpoint() + "/bucket/key", {}});
  auto const get = perform_sync(make_spec(request_method::GET), auth, fast_retry_config());
  CHECK(get.body == "hello");
  auto const head = perform_sync(make_spec(request_method::HEAD), auth, fast_retry_config());
  CHECK(head.body.empty());
  CHECK(head.etag == "\"g\"");
  auto const seen = server.requests();
  REQUIRE(seen.size() == 2);
  CHECK(seen[0].method == "GET");
  CHECK(seen[1].method == "HEAD");
}

TEST_CASE("perform_sync bounds retries", "[rest][sync]")
{
  SECTION("exhausted transient retries")
  {
    scripted_http_server server({{500, "", {}}});
    mock_authorizer auth(authorized_request{server.endpoint() + "/bucket/key", {}});
    CHECK_THROWS_WITH(perform_sync(make_spec(request_method::PUT), auth, fast_retry_config(), "x"),
                      Catch::Matchers::ContainsSubstring("exhausted retries (HTTP 500)"));
    CHECK(server.requests().size() == fast_retry_config().max_retry_attempts);
  }

  SECTION("403 is retried only max_auth_retry_attempts times")
  {
    scripted_http_server server({{403, "<Error><Code>SignatureDoesNotMatch</Code></Error>", {}}});
    mock_authorizer auth(authorized_request{server.endpoint() + "/bucket/key", {}});
    CHECK_THROWS_WITH(perform_sync(make_spec(request_method::PUT), auth, fast_retry_config(), "x"),
                      Catch::Matchers::ContainsSubstring("HTTP 403: SignatureDoesNotMatch"));
    CHECK(server.requests().size() == fast_retry_config().max_auth_retry_attempts);
  }

  SECTION("connection refused is retried then reported")
  {
    std::string endpoint;
    {
      scripted_http_server closed({});
      endpoint = closed.endpoint();
    }  // port now closed
    mock_authorizer auth(authorized_request{endpoint + "/bucket/key", {}});
    CHECK_THROWS_WITH(perform_sync(make_spec(request_method::GET), auth, fast_retry_config()),
                      Catch::Matchers::ContainsSubstring("exhausted retries"));
    CHECK(auth.request_count() == static_cast<int>(fast_retry_config().max_retry_attempts));
  }
}

TEST_CASE("perform_sync rejects invalid input and propagates authorizer errors", "[rest][sync]")
{
  mock_authorizer auth(authorized_request{"http://127.0.0.1:1/bucket/key", {}});
  CHECK_THROWS_AS(perform_sync(make_spec(request_method::DELETE_), auth, fast_retry_config(), "x"),
                  std::invalid_argument);
  CHECK_THROWS_AS(perform_sync(make_spec(request_method::GET), auth, fast_retry_config(), "x"),
                  std::invalid_argument);
  auth.set_throw("no creds");
  CHECK_THROWS_AS(perform_sync(make_spec(request_method::PUT), auth, fast_retry_config(), "x"),
                  credential_error);
}
