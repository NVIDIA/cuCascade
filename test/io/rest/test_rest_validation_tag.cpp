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

#include "loopback_range_server.hpp"

#include <cucascade/io/io_context.hpp>
#include <cucascade/io/rest/mock_authorizer.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>

#include <arpa/inet.h>
#include <catch2/catch_all.hpp>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using cucascade::io::open_hint;
using cucascade::io::rest::config;
using cucascade::io::rest::mock_authorizer;
using cucascade::io::rest::rest_ioctx;
using cucascade::io::rest::rest_reactor;
using cucascade::test::key_response_script;
using cucascade::test::loopback_range_server;
using cucascade::test::range_fault_policy;
using cucascade::test::scripted_response;
using namespace std::chrono_literals;

constexpr std::size_t object_size{4096};
constexpr std::size_t probe_size{512};
constexpr std::string_view object_uri{"s3://bucket/object.bin"};

std::vector<std::uint8_t> test_payload()
{
  std::vector<std::uint8_t> bytes(object_size);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>((i * 37U + 11U) & 0xffU);
  }
  return bytes;
}

config test_config()
{
  config cfg{};
  cfg.request_timeout_s       = 5;
  cfg.tls_verify              = false;
  cfg.max_retry_attempts      = 2;
  cfg.max_auth_retry_attempts = 1;
  cfg.retry_backoff_base      = 1ms;
  cfg.retry_jitter            = 0ms;
  cfg.honor_retry_after       = false;
  cfg.footer_probe_bytes      = probe_size;
  return cfg;
}

std::unique_ptr<rest_reactor> make_reactor(loopback_range_server const& server, config cfg)
{
  auto authorizer = std::make_shared<mock_authorizer>(
    cucascade::io::rest::authorized_request{server.endpoint() + "/bucket/object.bin", {}});
  auto context =
    std::make_shared<rest_reactor::reactor_context>(cfg, std::move(authorizer), nullptr);
  return std::make_unique<rest_reactor>(std::move(context), "validation-tag-test");
}

std::shared_ptr<rest_ioctx> make_ioctx(loopback_range_server const& server, config cfg)
{
  auto authorizer = std::make_shared<mock_authorizer>(
    cucascade::io::rest::authorized_request{server.endpoint() + "/bucket/object.bin", {}});
  auto context =
    std::make_shared<rest_reactor::reactor_context>(cfg, std::move(authorizer), nullptr);
  return std::make_shared<rest_ioctx>(1, std::move(context));
}

struct raw_http_response {
  int status{0};
  std::string headers;
  std::vector<std::uint8_t> body;

  [[nodiscard]] std::optional<std::string> header(std::string_view name) const
  {
    auto const prefix = "\r\n" + std::string{name} + ": ";
    auto const begin  = headers.find(prefix);
    if (begin == std::string::npos) { return std::nullopt; }
    auto const value_begin = begin + prefix.size();
    auto const end         = headers.find("\r\n", value_begin);
    return headers.substr(value_begin, end - value_begin);
  }
};

class raw_http_connection {
 public:
  explicit raw_http_connection(loopback_range_server const& server)
  {
    _fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (_fd < 0) { throw std::runtime_error("socket failed"); }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port   = htons(server.port());
    if (::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) != 1 ||
        ::connect(_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
      ::close(_fd);
      _fd = -1;
      throw std::runtime_error("connect failed");
    }
  }

  ~raw_http_connection()
  {
    if (_fd >= 0) { ::close(_fd); }
  }

  raw_http_connection(raw_http_connection const&)            = delete;
  raw_http_connection& operator=(raw_http_connection const&) = delete;

  raw_http_response request(std::string_view method,
                            std::string_view target,
                            bool close_connection = false)
  {
    std::string request{method};
    request += " ";
    request += target;
    request += " HTTP/1.1\r\nHost: 127.0.0.1\r\n";
    if (method == "GET") { request += "Range: bytes=-16\r\n"; }
    request += close_connection ? "Connection: close\r\n\r\n" : "Connection: keep-alive\r\n\r\n";
    send_all(request);
    return read_response(method != "HEAD");
  }

  [[nodiscard]] bool peer_closed()
  {
    timeval timeout{};
    timeout.tv_sec = 1;
    (void)::setsockopt(_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    char byte{};
    return ::recv(_fd, &byte, 1, 0) == 0;
  }

 private:
  void send_all(std::string_view bytes)
  {
    std::size_t sent = 0;
    while (sent < bytes.size()) {
      auto const n = ::send(_fd, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL);
      if (n <= 0) { throw std::runtime_error("send failed"); }
      sent += static_cast<std::size_t>(n);
    }
  }

  void read_at_least(std::size_t bytes)
  {
    while (_pending.size() < bytes) {
      std::array<char, 8192> buffer{};
      auto const n = ::recv(_fd, buffer.data(), buffer.size(), 0);
      if (n <= 0) { throw std::runtime_error("unexpected end of HTTP response"); }
      _pending.append(buffer.data(), static_cast<std::size_t>(n));
    }
  }

  raw_http_response read_response(bool has_body)
  {
    auto header_end = _pending.find("\r\n\r\n");
    while (header_end == std::string::npos) {
      read_at_least(_pending.size() + 1);
      header_end = _pending.find("\r\n\r\n");
    }

    raw_http_response response;
    response.headers = _pending.substr(0, header_end + 4);
    _pending.erase(0, header_end + 4);
    response.status = std::stoi(response.headers.substr(response.headers.find(' ') + 1, 3));

    std::size_t body_size = 0;
    if (has_body) {
      auto const content_length = response.header("Content-Length");
      REQUIRE(content_length.has_value());
      body_size = static_cast<std::size_t>(std::stoull(*content_length));
      read_at_least(body_size);
      response.body.assign(_pending.begin(), _pending.begin() + body_size);
      _pending.erase(0, body_size);
    }
    return response;
  }

  int _fd{-1};
  std::string _pending;
};

}  // namespace

TEST_CASE("loopback range server keeps connections alive and counts requests per key",
          "[rest][loopback_server]")
{
  loopback_range_server server(test_payload());
  raw_http_connection connection(server);

  auto const first  = connection.request("GET", "/bucket/first.parquet");
  auto const second = connection.request("HEAD", "/bucket/second.parquet", true);

  CHECK(first.status == 206);
  CHECK(first.body.size() == 16);
  CHECK(second.status == 200);
  CHECK(server.accepted_connection_count() == 1);
  CHECK(server.get_count() == 1);
  CHECK(server.head_count() == 1);
  CHECK(server.get_count("first.parquet") == 1);
  CHECK(server.get_count("second.parquet") == 0);
  CHECK(server.head_count("first.parquet") == 0);
  CHECK(server.head_count("second.parquet") == 1);
  CHECK(connection.peer_closed());
}

TEST_CASE("loopback range server scripts responses independently per key",
          "[rest][loopback_server]")
{
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["retry.parquet"].gets   = {scripted_response{.status = 503, .etag = "\"retry-v1\""},
                                     scripted_response{.etag = "\"retry-v2\""}};
  scripts["missing.parquet"].gets = {scripted_response{.status = 404}};
  scripts["denied.parquet"].heads = {scripted_response{.status = 403}};
  scripts["tagged.parquet"].gets  = {scripted_response{.etag = "\"tag-v1\""},
                                     scripted_response{.etag = "\"tag-v2\""}};
  scripts["slow.parquet"].gets    = {scripted_response{.delay = 20ms}};

  loopback_range_server server(test_payload(), {}, {}, std::move(scripts));
  raw_http_connection connection(server);

  CHECK(connection.request("GET", "/bucket/retry.parquet").status == 503);
  auto const retry = connection.request("GET", "/bucket/retry.parquet");
  CHECK(retry.status == 206);
  CHECK(retry.header("ETag") == "\"retry-v2\"");
  CHECK(connection.request("GET", "/bucket/missing.parquet").status == 404);
  CHECK(connection.request("HEAD", "/bucket/denied.parquet").status == 403);

  auto const tag_v1 = connection.request("GET", "/bucket/tagged.parquet");
  auto const tag_v2 = connection.request("GET", "/bucket/tagged.parquet");
  CHECK(tag_v1.header("ETag") == "\"tag-v1\"");
  CHECK(tag_v2.header("ETag") == "\"tag-v2\"");

  auto const start = std::chrono::steady_clock::now();
  CHECK(connection.request("GET", "/bucket/slow.parquet", true).status == 206);
  CHECK(std::chrono::steady_clock::now() - start >= 15ms);

  CHECK(server.get_count("retry.parquet") == 2);
  CHECK(server.get_count("missing.parquet") == 1);
  CHECK(server.head_count("denied.parquet") == 1);
  CHECK(server.get_count("tagged.parquet") == 2);
  CHECK(server.get_count("slow.parquet") == 1);
  CHECK(server.accepted_connection_count() == 1);
}

TEST_CASE("footer probes capture only the verified response validation tag",
          "[rest][validation_tag]")
{
  SECTION("quoted ETag is preserved")
  {
    range_fault_policy fault{};
    fault.successful_get_etag = "\"footer-v1\"";
    loopback_range_server server(test_payload(), fault);
    auto reactor = make_reactor(server, test_config());

    auto const probe = reactor->fetch_footer_suffix("bucket", "object.bin", probe_size);

    REQUIRE(probe.bytes != nullptr);
    CHECK(probe.etag == "\"footer-v1\"");
    CHECK(server.get_count() == 1);
  }

  SECTION("missing ETag stays empty on the probe and io object")
  {
    loopback_range_server server(test_payload());
    auto reactor     = make_reactor(server, test_config());
    auto const probe = reactor->fetch_footer_suffix("bucket", "object.bin", probe_size);
    REQUIRE(probe.bytes != nullptr);
    CHECK(probe.etag.empty());

    auto ioctx  = make_ioctx(server, test_config());
    auto object = ioctx->open_io_object(std::string{object_uri}, open_hint::parquet_footer_probe);
    CHECK(object->validation_tag().empty());
  }

  SECTION("weak ETag is preserved verbatim")
  {
    range_fault_policy fault{};
    fault.successful_get_etag = "W/\"weak\"";
    loopback_range_server server(test_payload(), fault);
    auto ioctx  = make_ioctx(server, test_config());
    auto object = ioctx->open_io_object(std::string{object_uri}, open_hint::parquet_footer_probe);
    CHECK(object->validation_tag() == "W/\"weak\"");
  }

  SECTION("empty entity-tag is distinct from a missing header")
  {
    range_fault_policy fault{};
    fault.successful_get_etag = "\"\"";
    loopback_range_server server(test_payload(), fault);
    auto ioctx  = make_ioctx(server, test_config());
    auto object = ioctx->open_io_object(std::string{object_uri}, open_hint::parquet_footer_probe);
    CHECK(object->validation_tag() == "\"\"");
  }
}

TEST_CASE("HEAD returns size and validation tag without changing the size wrapper",
          "[rest][validation_tag]")
{
  SECTION("quoted ETag is preserved")
  {
    range_fault_policy fault{};
    fault.successful_head_etag = "\"head-v1\"";
    loopback_range_server server(test_payload(), fault);
    auto reactor = make_reactor(server, test_config());

    auto const result = reactor->head_object("bucket", "object.bin");

    CHECK(result.object_size == object_size);
    CHECK(result.etag == "\"head-v1\"");
    CHECK(reactor->head_object_size("bucket", "object.bin") == object_size);
    CHECK(server.head_count() == 2);
  }

  SECTION("missing ETag does not affect size discovery")
  {
    loopback_range_server server(test_payload());
    auto reactor      = make_reactor(server, test_config());
    auto const result = reactor->head_object("bucket", "object.bin");
    CHECK(result.object_size == object_size);
    CHECK(result.etag.empty());
    CHECK(server.head_count() == 1);
  }
}

TEST_CASE("validation tags are isolated between retry attempts", "[rest][validation_tag]")
{
  SECTION("HEAD uses the successful attempt tag")
  {
    range_fault_policy fault{};
    fault.fail_first_heads     = 1;
    fault.failed_head_etag     = "\"stale-head\"";
    fault.successful_head_etag = "\"fresh-head\"";
    loopback_range_server server(test_payload(), fault);
    auto reactor = make_reactor(server, test_config());
    CHECK(reactor->head_object("bucket", "object.bin").etag == "\"fresh-head\"");
    CHECK(server.head_count() == 2);
  }

  SECTION("HEAD does not leak a failed attempt tag")
  {
    range_fault_policy fault{};
    fault.fail_first_heads = 1;
    fault.failed_head_etag = "\"stale-head\"";
    loopback_range_server server(test_payload(), fault);
    auto reactor = make_reactor(server, test_config());
    CHECK(reactor->head_object("bucket", "object.bin").etag.empty());
    CHECK(server.head_count() == 2);
  }

  SECTION("footer probe uses the successful attempt tag")
  {
    range_fault_policy fault{};
    fault.fail_first_gets     = 1;
    fault.failed_get_etag     = "\"stale-probe\"";
    fault.successful_get_etag = "\"fresh-probe\"";
    loopback_range_server server(test_payload(), fault);
    auto reactor     = make_reactor(server, test_config());
    auto const probe = reactor->fetch_footer_suffix("bucket", "object.bin", probe_size);
    REQUIRE(probe.bytes != nullptr);
    CHECK(probe.etag == "\"fresh-probe\"");
    CHECK(server.get_count() == 2);
  }

  SECTION("footer probe does not leak a failed attempt tag")
  {
    range_fault_policy fault{};
    fault.fail_first_gets = 1;
    fault.failed_get_etag = "\"stale-probe\"";
    loopback_range_server server(test_payload(), fault);
    auto reactor     = make_reactor(server, test_config());
    auto const probe = reactor->fetch_footer_suffix("bucket", "object.bin", probe_size);
    REQUIRE(probe.bytes != nullptr);
    CHECK(probe.etag.empty());
    CHECK(server.get_count() == 2);
  }
}

TEST_CASE("unusable footer responses discard their tag before HEAD fallback",
          "[rest][validation_tag]")
{
  auto check_fallback = [](range_fault_policy fault) {
    fault.failed_get_etag      = "\"discarded-probe\"";
    fault.successful_head_etag = "\"fallback-head\"";
    loopback_range_server server(test_payload(), fault);
    auto ioctx  = make_ioctx(server, test_config());
    auto object = ioctx->open_io_object(std::string{object_uri}, open_hint::parquet_footer_probe);

    CHECK(object->size() == object_size);
    CHECK(object->validation_tag() == "\"fallback-head\"");
    CHECK(server.get_count() == 1);
    CHECK(server.head_count() == 1);
  };

  SECTION("200 full-body response")
  {
    range_fault_policy fault{};
    fault.ignore_range_with_200 = true;
    check_fallback(fault);
  }

  SECTION("416 response")
  {
    range_fault_policy fault{};
    fault.fail_range_with_416 = true;
    check_fallback(fault);
  }

  SECTION("malformed 206 response")
  {
    range_fault_policy fault{};
    fault.malformed_content_range = true;
    check_fallback(fault);
  }
}

TEST_CASE("interim response tags do not leak into the final response", "[rest][validation_tag]")
{
  SECTION("HEAD")
  {
    range_fault_policy fault{};
    fault.interim_head_etag = "\"interim-head\"";
    loopback_range_server server(test_payload(), fault);
    auto reactor = make_reactor(server, test_config());
    CHECK(reactor->head_object("bucket", "object.bin").etag.empty());
    CHECK(server.head_count() == 1);
  }

  SECTION("footer probe")
  {
    range_fault_policy fault{};
    fault.interim_get_etag = "\"interim-probe\"";
    loopback_range_server server(test_payload(), fault);
    auto reactor     = make_reactor(server, test_config());
    auto const probe = reactor->fetch_footer_suffix("bucket", "object.bin", probe_size);
    REQUIRE(probe.bytes != nullptr);
    CHECK(probe.etag.empty());
    CHECK(server.get_count() == 1);
  }
}

TEST_CASE("footer probes do not reuse Content-Range from an interim response",
          "[rest][validation_tag]")
{
  range_fault_policy fault{};
  fault.interim_get_content_range = "bytes " + std::to_string(object_size - probe_size) + "-" +
                                    std::to_string(object_size - 1) + "/" +
                                    std::to_string(object_size);
  fault.omit_successful_content_range = true;
  fault.successful_get_etag           = "\"discarded-probe\"";
  fault.successful_head_etag          = "\"fallback-head\"";
  loopback_range_server server(test_payload(), fault);
  auto ioctx  = make_ioctx(server, test_config());
  auto object = ioctx->open_io_object(std::string{object_uri}, open_hint::parquet_footer_probe);

  CHECK(object->size() == object_size);
  CHECK(object->validation_tag() == "\"fallback-head\"");
  CHECK(server.get_count() == 1);
  CHECK(server.head_count() == 1);
}

TEST_CASE("retry backoff uses only the final response block", "[rest][validation_tag]")
{
  auto retry_config = [](std::chrono::milliseconds fallback) {
    auto cfg               = test_config();
    cfg.honor_retry_after  = true;
    cfg.retry_backoff_base = fallback;
    return cfg;
  };

  SECTION("HEAD ignores interim Retry-After when the final block omits it")
  {
    range_fault_policy fault{};
    fault.fail_first_heads         = 1;
    fault.interim_head_retry_after = "2";
    loopback_range_server server(test_payload(), fault);
    auto reactor = make_reactor(server, retry_config(1ms));

    auto const start   = std::chrono::steady_clock::now();
    auto const result  = reactor->head_object("bucket", "object.bin");
    auto const elapsed = std::chrono::steady_clock::now() - start;

    CHECK(result.object_size == object_size);
    CHECK(elapsed < 1s);
    CHECK(server.head_count() == 2);
  }

  SECTION("HEAD uses Retry-After from the final block")
  {
    range_fault_policy fault{};
    fault.fail_first_heads         = 1;
    fault.interim_head_retry_after = "2";
    fault.failed_head_retry_after  = "0";
    loopback_range_server server(test_payload(), fault);
    auto reactor = make_reactor(server, retry_config(1500ms));

    auto const start   = std::chrono::steady_clock::now();
    auto const result  = reactor->head_object("bucket", "object.bin");
    auto const elapsed = std::chrono::steady_clock::now() - start;

    CHECK(result.object_size == object_size);
    CHECK(elapsed < 1s);
    CHECK(server.head_count() == 2);
  }

  SECTION("footer probe ignores interim Retry-After when the final block omits it")
  {
    range_fault_policy fault{};
    fault.fail_first_gets         = 1;
    fault.interim_get_retry_after = "2";
    loopback_range_server server(test_payload(), fault);
    auto reactor = make_reactor(server, retry_config(1ms));

    auto const start   = std::chrono::steady_clock::now();
    auto const probe   = reactor->fetch_footer_suffix("bucket", "object.bin", probe_size);
    auto const elapsed = std::chrono::steady_clock::now() - start;

    REQUIRE(probe.bytes != nullptr);
    CHECK(elapsed < 1s);
    CHECK(server.get_count() == 2);
  }

  SECTION("footer probe uses Retry-After from the final block")
  {
    range_fault_policy fault{};
    fault.fail_first_gets         = 1;
    fault.interim_get_retry_after = "2";
    fault.failed_get_retry_after  = "0";
    loopback_range_server server(test_payload(), fault);
    auto reactor = make_reactor(server, retry_config(1500ms));

    auto const start   = std::chrono::steady_clock::now();
    auto const probe   = reactor->fetch_footer_suffix("bucket", "object.bin", probe_size);
    auto const elapsed = std::chrono::steady_clock::now() - start;

    REQUIRE(probe.bytes != nullptr);
    CHECK(elapsed < 1s);
    CHECK(server.get_count() == 2);
  }
}

TEST_CASE("rest ioctx threads validation tags through every network open path",
          "[rest][validation_tag]")
{
  SECTION("HEAD open")
  {
    range_fault_policy fault{};
    fault.successful_head_etag = "\"head-open\"";
    loopback_range_server server(test_payload(), fault);
    auto ioctx  = make_ioctx(server, test_config());
    auto object = ioctx->open_io_object(std::string{object_uri});
    CHECK(object->validation_tag() == "\"head-open\"");
  }

  SECTION("footer probe open")
  {
    range_fault_policy fault{};
    fault.successful_get_etag = "\"probe-open\"";
    loopback_range_server server(test_payload(), fault);
    auto ioctx  = make_ioctx(server, test_config());
    auto object = ioctx->open_io_object(std::string{object_uri}, open_hint::parquet_footer_probe);
    CHECK(object->validation_tag() == "\"probe-open\"");
  }

  SECTION("footer fallback open")
  {
    range_fault_policy fault{};
    fault.fail_range_with_416  = true;
    fault.failed_get_etag      = "\"discarded-probe\"";
    fault.successful_head_etag = "\"fallback-open\"";
    loopback_range_server server(test_payload(), fault);
    auto ioctx  = make_ioctx(server, test_config());
    auto object = ioctx->open_io_object(std::string{object_uri}, open_hint::parquet_footer_probe);
    CHECK(object->validation_tag() == "\"fallback-open\"");
  }
}

TEST_CASE("validation tag capture adds no network requests", "[rest][validation_tag]")
{
  SECTION("HEAD open remains one HEAD")
  {
    range_fault_policy fault{};
    fault.successful_head_etag = "\"head-v1\"";
    loopback_range_server server(test_payload(), fault);
    auto ioctx = make_ioctx(server, test_config());
    (void)ioctx->open_io_object(std::string{object_uri});
    CHECK(server.head_count() == 1);
    CHECK(server.get_count() == 0);
  }

  SECTION("footer probe open remains one GET")
  {
    range_fault_policy fault{};
    fault.successful_get_etag = "\"probe-v1\"";
    loopback_range_server server(test_payload(), fault);
    auto ioctx = make_ioctx(server, test_config());
    (void)ioctx->open_io_object(std::string{object_uri}, open_hint::parquet_footer_probe);
    CHECK(server.head_count() == 0);
    CHECK(server.get_count() == 1);
  }
}

TEST_CASE("known-size open stays tagless and performs no requests", "[rest][validation_tag]")
{
  range_fault_policy fault{};
  fault.successful_get_etag  = "\"unused-get\"";
  fault.successful_head_etag = "\"unused-head\"";
  loopback_range_server server(test_payload(), fault);
  auto ioctx  = make_ioctx(server, test_config());
  auto object = ioctx->open_io_object(std::string{object_uri}, object_size);

  CHECK(object->size() == object_size);
  CHECK(object->validation_tag().empty());
  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == 0);
}
