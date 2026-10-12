/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <cucascade/io/rest/authorizer.hpp>
#include <cucascade/io/rest/config.hpp>
#include <cucascade/io/rest/object_store_lister.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>

#include <arpa/inet.h>
#include <catch2/catch_all.hpp>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using cucascade::io::rest::authorized_request;
using cucascade::io::rest::config;
using cucascade::io::rest::object_ref;
using cucascade::io::rest::object_store_lister;
using cucascade::io::rest::request_authorizer;
using cucascade::io::rest::request_method;
using cucascade::io::rest::rest_ioctx;
using cucascade::io::rest::rest_reactor;
using cucascade::io::rest::s3::list_objects_v2_page;
using namespace std::chrono_literals;

struct listed_object {
  std::string key;
  std::uint64_t size;
};

struct scripted_page {
  std::string request_token;
  std::vector<listed_object> objects;
  bool truncated{false};
  std::string next_token;
};

struct observed_query {
  std::string max_keys;
  std::string continuation_token;
  std::string prefix;
};

class scripted_list_server {
 public:
  explicit scripted_list_server(std::vector<scripted_page> pages) : _pages(std::move(pages))
  {
    if (_pages.empty()) { throw std::invalid_argument("scripted LIST server needs a page"); }

    _listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (_listen_fd < 0) { throw std::runtime_error("socket failed: " + errno_message()); }

    int one = 1;
    if (::setsockopt(_listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0) {
      close_listener();
      throw std::runtime_error("setsockopt failed: " + errno_message());
    }

    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port        = 0;
    if (::bind(_listen_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
      close_listener();
      throw std::runtime_error("bind failed: " + errno_message());
    }
    if (::listen(_listen_fd, 16) != 0) {
      close_listener();
      throw std::runtime_error("listen failed: " + errno_message());
    }

    socklen_t length = sizeof(address);
    if (::getsockname(_listen_fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
      close_listener();
      throw std::runtime_error("getsockname failed: " + errno_message());
    }
    _port               = ntohs(address.sin_port);
    int const listen_fd = _listen_fd;
    _thread             = std::thread([this, listen_fd] { accept_loop(listen_fd); });
  }

  ~scripted_list_server()
  {
    _stop.store(true, std::memory_order_relaxed);
    if (_listen_fd >= 0) { (void)::shutdown(_listen_fd, SHUT_RDWR); }
    if (_thread.joinable()) { _thread.join(); }
    close_listener();
  }

  scripted_list_server(scripted_list_server const&)            = delete;
  scripted_list_server& operator=(scripted_list_server const&) = delete;

  [[nodiscard]] std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(_port); }

  [[nodiscard]] std::size_t request_count() const noexcept
  {
    return _request_count.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::vector<observed_query> observations() const
  {
    std::scoped_lock lock{_observations_mutex};
    return _observations;
  }

 private:
  static std::string errno_message() { return std::strerror(errno); }

  void close_listener() noexcept
  {
    if (_listen_fd < 0) { return; }
    (void)::close(_listen_fd);
    _listen_fd = -1;
  }

  void accept_loop(int listen_fd)
  {
    while (!_stop.load(std::memory_order_relaxed)) {
      sockaddr_in client{};
      socklen_t length = sizeof(client);
      int const fd     = ::accept(listen_fd, reinterpret_cast<sockaddr*>(&client), &length);
      if (fd < 0) {
        if (_stop.load(std::memory_order_relaxed)) { return; }
        continue;
      }
      handle_client(fd);
      (void)::close(fd);
    }
  }

  void handle_client(int fd)
  {
    timeval timeout{};
    timeout.tv_sec = 5;
    (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    std::string request;
    std::array<char, 4096> buffer{};
    while (request.find("\r\n\r\n") == std::string::npos && request.size() < 64 * 1024) {
      ssize_t const received = ::recv(fd, buffer.data(), buffer.size(), 0);
      if (received <= 0) { return; }
      request.append(buffer.data(), static_cast<std::size_t>(received));
    }

    std::string const target = request_target(request);
    if (request.rfind("GET ", 0) != 0 || target.find("list-type=2") == std::string::npos) {
      send_all(fd,
               "HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\nConnection: "
               "close\r\n\r\n");
      return;
    }

    observed_query observation{.max_keys           = query_value(target, "max-keys"),
                               .continuation_token = query_value(target, "continuation-token"),
                               .prefix             = query_value(target, "prefix")};
    {
      std::scoped_lock lock{_observations_mutex};
      _observations.push_back(observation);
    }
    _request_count.fetch_add(1, std::memory_order_relaxed);

    auto const* page = find_page(observation.continuation_token);
    if (page == nullptr) {
      send_all(fd, "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
      return;
    }

    std::string const body = page_xml(*page);
    send_all(fd,
             "HTTP/1.1 200 OK\r\nContent-Type: application/xml\r\nContent-Length: " +
               std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body);
  }

  [[nodiscard]] scripted_page const* find_page(std::string_view request_token) const noexcept
  {
    for (auto const& page : _pages) {
      if (page.request_token == request_token) { return &page; }
    }
    return nullptr;
  }

  static std::string request_target(std::string const& request)
  {
    auto const first_space = request.find(' ');
    if (first_space == std::string::npos) { return {}; }
    auto const second_space = request.find(' ', first_space + 1);
    if (second_space == std::string::npos) { return {}; }
    return request.substr(first_space + 1, second_space - first_space - 1);
  }

  static int hex_value(char c) noexcept
  {
    if (c >= '0' && c <= '9') { return c - '0'; }
    if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
    if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
    return -1;
  }

  static std::string url_decode(std::string_view encoded)
  {
    std::string decoded;
    decoded.reserve(encoded.size());
    for (std::size_t i = 0; i < encoded.size(); ++i) {
      if (encoded[i] == '%' && i + 2 < encoded.size()) {
        int const high = hex_value(encoded[i + 1]);
        int const low  = hex_value(encoded[i + 2]);
        if (high >= 0 && low >= 0) {
          decoded.push_back(static_cast<char>((high << 4) | low));
          i += 2;
          continue;
        }
      }
      decoded.push_back(encoded[i] == '+' ? ' ' : encoded[i]);
    }
    return decoded;
  }

  static std::string query_value(std::string_view target, std::string_view wanted_key)
  {
    auto const question = target.find('?');
    if (question == std::string_view::npos) { return {}; }
    std::string_view query = target.substr(question + 1);
    while (!query.empty()) {
      auto const ampersand = query.find('&');
      auto const part      = query.substr(0, ampersand);
      auto const equals    = part.find('=');
      if (equals != std::string_view::npos && part.substr(0, equals) == wanted_key) {
        return url_decode(part.substr(equals + 1));
      }
      if (ampersand == std::string_view::npos) { break; }
      query.remove_prefix(ampersand + 1);
    }
    return {};
  }

  static std::string xml_escape(std::string_view value)
  {
    std::string escaped;
    for (char c : value) {
      switch (c) {
        case '&': escaped += "&amp;"; break;
        case '<': escaped += "&lt;"; break;
        case '>': escaped += "&gt;"; break;
        case '\"': escaped += "&quot;"; break;
        case '\'': escaped += "&apos;"; break;
        default: escaped.push_back(c); break;
      }
    }
    return escaped;
  }

  static std::string page_xml(scripted_page const& page)
  {
    std::string body =
      "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
      "<ListBucketResult><IsTruncated>";
    body += page.truncated ? "true" : "false";
    body += "</IsTruncated>";
    if (!page.next_token.empty()) {
      body += "<NextContinuationToken>" + xml_escape(page.next_token) + "</NextContinuationToken>";
    }
    for (auto const& object : page.objects) {
      body += "<Contents><Key>" + xml_escape(object.key) + "</Key><Size>" +
              std::to_string(object.size) + "</Size></Contents>";
    }
    body += "</ListBucketResult>";
    return body;
  }

  static void send_all(int fd, std::string_view response)
  {
    std::size_t sent = 0;
    while (sent < response.size()) {
      ssize_t const written =
        ::send(fd, response.data() + sent, response.size() - sent, MSG_NOSIGNAL);
      if (written <= 0) { return; }
      sent += static_cast<std::size_t>(written);
    }
  }

  int _listen_fd{-1};
  std::uint16_t _port{0};
  std::vector<scripted_page> _pages;
  std::atomic<bool> _stop{false};
  std::atomic<std::size_t> _request_count{0};
  mutable std::mutex _observations_mutex;
  std::vector<observed_query> _observations;
  std::thread _thread;
};

class loopback_list_authorizer final : public request_authorizer {
 public:
  explicit loopback_list_authorizer(std::string endpoint) : _endpoint(std::move(endpoint)) {}

  authorized_request authorize(object_ref const& object,
                               request_method,
                               std::chrono::seconds) override
  {
    return {_endpoint + "/" + object.bucket + "/" + object.key, {}};
  }

  authorized_request authorize_list(std::string_view bucket,
                                    std::string_view canonical_query,
                                    std::chrono::seconds) override
  {
    return {_endpoint + "/" + std::string{bucket} + "?" + std::string{canonical_query}, {}};
  }

 private:
  std::string _endpoint;
};

config listing_config(std::size_t list_max_matches = 100'000)
{
  config cfg{};
  cfg.request_timeout_s       = 5;
  cfg.tls_verify              = false;
  cfg.max_connections         = 1;
  cfg.max_retry_attempts      = 1;
  cfg.max_auth_retry_attempts = 1;
  cfg.retry_backoff_base      = 1ms;
  cfg.retry_jitter            = 0ms;
  cfg.honor_retry_after       = false;
  cfg.list_max_matches        = list_max_matches;
  return cfg;
}

std::string direct_page_xml(std::vector<listed_object> objects,
                            bool truncated,
                            std::optional<std::string_view> next_token = std::nullopt)
{
  std::string body = "<ListBucketResult><IsTruncated>";
  body += truncated ? "true" : "false";
  body += "</IsTruncated>";
  if (next_token.has_value()) {
    body += "<NextContinuationToken>" + std::string{*next_token} + "</NextContinuationToken>";
  }
  for (auto const& object : objects) {
    body += "<Contents><Key>" + object.key + "</Key><Size>" + std::to_string(object.size) +
            "</Size></Contents>";
  }
  body += "</ListBucketResult>";
  return body;
}

class stub_page_fetch {
 public:
  explicit stub_page_fetch(std::vector<std::string> responses) : _responses(std::move(responses)) {}

  std::string fetch(std::string_view, std::string_view, std::string_view)
  {
    if (_next == _responses.size()) {
      throw std::runtime_error("stub page fetch exhausted its responses");
    }
    return _responses[_next++];
  }

 private:
  std::vector<std::string> _responses;
  std::size_t _next{0};
};

object_store_lister make_direct_lister(std::shared_ptr<stub_page_fetch> fetch,
                                       std::size_t max_scanned = 100,
                                       std::size_t max_matches = 100)
{
  return object_store_lister{
    [fetch = std::move(fetch)](
      std::string_view bucket, std::string_view prefix, std::string_view canonical_query) {
      return fetch->fetch(bucket, prefix, canonical_query);
    },
    max_scanned,
    max_matches,
    "test_lister::list_objects"};
}

struct listing_fixture {
  explicit listing_fixture(std::vector<scripted_page> pages, std::size_t list_max_matches = 100'000)
    : server(std::move(pages)),
      authorizer(std::make_shared<loopback_list_authorizer>(server.endpoint()))
  {
    auto context = std::make_shared<rest_reactor::reactor_context>(
      listing_config(list_max_matches), authorizer, nullptr);
    ioctx = std::make_shared<rest_ioctx>(1, std::move(context));
    ioctx->start();
  }

  scripted_list_server server;
  std::shared_ptr<loopback_list_authorizer> authorizer;
  std::shared_ptr<rest_ioctx> ioctx;
};

}  // namespace

TEST_CASE("rest ioctx delegates paged listing to its composed lister", "[rest][listing]")
{
  listing_fixture fixture{
    {scripted_page{.request_token = "",
                   .objects       = {{"prefix/a.parquet", 11}, {"prefix/b.parquet", 22}},
                   .truncated     = true,
                   .next_token    = "page/2"},
     scripted_page{.request_token = "page/2",
                   .objects       = {{"prefix/c.parquet", 33}},
                   .truncated     = false,
                   .next_token    = ""}}};
  std::vector<list_objects_v2_page> delivered;

  fixture.ioctx->list_objects_paged("bucket", "prefix/", 2, [&](list_objects_v2_page const& page) {
    delivered.push_back(page);
    return true;
  });

  REQUIRE(delivered.size() == 2);
  REQUIRE(delivered[0].entries.size() == 2);
  REQUIRE(delivered[1].entries.size() == 1);
  CHECK(delivered[0].entries[0].key == "prefix/a.parquet");
  CHECK(delivered[0].entries[0].size == 11);
  CHECK(delivered[0].entries[1].key == "prefix/b.parquet");
  CHECK(delivered[1].entries[0].key == "prefix/c.parquet");
  CHECK(delivered[1].entries[0].size == 33);

  auto const observations = fixture.server.observations();
  REQUIRE(observations.size() == 2);
  CHECK(observations[0].max_keys == "2");
  CHECK(observations[0].prefix == "prefix/");
  CHECK(observations[0].continuation_token.empty());
  CHECK(observations[1].continuation_token == "page/2");
  CHECK(observations[1].prefix == "prefix/");

  SECTION("whole-list delegation preserves order and enforces max_keys")
  {
    auto const objects = fixture.ioctx->list_objects("bucket", "prefix/", 2);

    REQUIRE(objects.size() == 3);
    CHECK(objects[0].key == "prefix/a.parquet");
    CHECK(objects[0].size == 11);
    CHECK(objects[1].key == "prefix/b.parquet");
    CHECK(objects[1].size == 22);
    CHECK(objects[2].key == "prefix/c.parquet");
    CHECK(objects[2].size == 33);

    CHECK_THROWS_WITH(fixture.ioctx->list_objects("bucket", "prefix/", 2, 2),
                      Catch::Matchers::ContainsSubstring("rest_ioctx::list_objects:") &&
                        Catch::Matchers::ContainsSubstring("more than 2 objects"));
  }
}

TEST_CASE("a listing sink can stop before the next page request", "[rest][listing]")
{
  listing_fixture fixture{
    {scripted_page{
       .request_token = "", .objects = {{"prefix/a", 1}}, .truncated = true, .next_token = "next"},
     scripted_page{.request_token = "next",
                   .objects       = {{"prefix/b", 2}},
                   .truncated     = false,
                   .next_token    = ""}}};
  std::size_t pages_seen = 0;

  fixture.ioctx->list_objects_paged("bucket", "prefix/", 1, [&](list_objects_v2_page const&) {
    ++pages_seen;
    return false;
  });

  CHECK(pages_seen == 1);
  CHECK(fixture.server.request_count() == 1);
}

TEST_CASE("listing page size is clamped on the wire", "[rest][listing]")
{
  listing_fixture fixture{{scripted_page{
    .request_token = "", .objects = {{"key", 1}}, .truncated = false, .next_token = ""}}};
  auto const consume = [](list_objects_v2_page const&) { return true; };

  fixture.ioctx->list_objects_paged("bucket", "", 0, consume);
  fixture.ioctx->list_objects_paged("bucket", "", 1001, consume);

  auto const observations = fixture.server.observations();
  REQUIRE(observations.size() == 2);
  CHECK(observations[0].max_keys == "1000");
  CHECK(observations[1].max_keys == "1000");
}

TEST_CASE("listing throws when the scanned object cap is exceeded", "[rest][listing]")
{
  listing_fixture fixture{{scripted_page{.request_token = "",
                                         .objects       = {{"prefix/a", 1}, {"prefix/b", 2}},
                                         .truncated     = false,
                                         .next_token    = ""}}};
  std::size_t pages_seen = 0;

  CHECK_THROWS_WITH(fixture.ioctx->list_objects_paged(
                      "bucket",
                      "prefix/",
                      1000,
                      [&](list_objects_v2_page const&) {
                        ++pages_seen;
                        return true;
                      },
                      1),
                    Catch::Matchers::ContainsSubstring("rest_ioctx::list_objects:") &&
                      Catch::Matchers::ContainsSubstring("scanned more than 1 objects"));
  CHECK(pages_seen == 0);
  CHECK(fixture.server.request_count() == 1);
}

TEST_CASE("rest ioctx exposes the configured listing match cap", "[rest][listing]")
{
  constexpr std::size_t configured_cap = 37;
  listing_fixture fixture{
    {scripted_page{.request_token = "", .objects = {}, .truncated = false, .next_token = ""}},
    configured_cap};

  CHECK(fixture.ioctx->list_max_matches() == configured_cap);
  CHECK(fixture.server.request_count() == 0);
}

TEST_CASE("object store lister rejects unsafe pagination and result growth", "[rest][listing]")
{
  auto const consume = [](list_objects_v2_page const&) { return true; };

  SECTION("a truncated page requires a non-empty continuation token")
  {
    // A missing element is rejected by the parser; an empty element reaches the lister guard.
    auto fetch  = std::make_shared<stub_page_fetch>(std::vector<std::string>{direct_page_xml(
      {{"prefix/a", 1}}, true, std::optional<std::string_view>{std::string_view{}})});
    auto lister = make_direct_lister(std::move(fetch));

    CHECK_THROWS_WITH(lister.list_objects_paged("bucket", "prefix/", 1000, consume),
                      Catch::Matchers::ContainsSubstring("test_lister::list_objects:") &&
                        Catch::Matchers::ContainsSubstring("without a continuation token"));
  }

  SECTION("a truncated page cannot be empty")
  {
    auto fetch = std::make_shared<stub_page_fetch>(
      std::vector<std::string>{direct_page_xml({}, true, std::string_view{"next"})});
    auto lister = make_direct_lister(std::move(fetch));

    CHECK_THROWS_WITH(lister.list_objects_paged("bucket", "prefix/", 1000, consume),
                      Catch::Matchers::ContainsSubstring("test_lister::list_objects:") &&
                        Catch::Matchers::ContainsSubstring("with no entries"));
  }

  SECTION("a continuation token must advance")
  {
    auto fetch = std::make_shared<stub_page_fetch>(
      std::vector<std::string>{direct_page_xml({{"prefix/a", 1}}, true, std::string_view{"next"}),
                               direct_page_xml({{"prefix/b", 2}}, true, std::string_view{"next"})});
    auto lister = make_direct_lister(std::move(fetch));

    CHECK_THROWS_WITH(lister.list_objects_paged("bucket", "prefix/", 1000, consume),
                      Catch::Matchers::ContainsSubstring("test_lister::list_objects:") &&
                        Catch::Matchers::ContainsSubstring("continuation token did not advance"));
  }

  SECTION("the scanned object cap is enforced")
  {
    auto fetch = std::make_shared<stub_page_fetch>(
      std::vector<std::string>{direct_page_xml({{"prefix/a", 1}, {"prefix/b", 2}}, false)});
    auto lister = make_direct_lister(std::move(fetch), 1);

    CHECK_THROWS_WITH(lister.list_objects_paged("bucket", "prefix/", 1000, consume),
                      Catch::Matchers::ContainsSubstring("test_lister::list_objects:") &&
                        Catch::Matchers::ContainsSubstring("scanned more than 1 objects"));
  }

  SECTION("the whole-list match cap is enforced")
  {
    auto fetch = std::make_shared<stub_page_fetch>(
      std::vector<std::string>{direct_page_xml({{"prefix/a", 1}, {"prefix/b", 2}}, false)});
    auto lister = make_direct_lister(std::move(fetch), 100, 1);

    CHECK_THROWS_WITH(lister.list_objects("bucket", "prefix/"),
                      Catch::Matchers::ContainsSubstring("test_lister::list_objects:") &&
                        Catch::Matchers::ContainsSubstring("more than 1 objects"));
  }
}
