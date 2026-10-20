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

// The cloud module: the minimal blocking HTTP/1.1 client, the IMDSv2 client against a loopback
// fake metadata server (token flow, 401 -> new token, token TTL lapse, chunked replies, timeouts,
// the disable switch and the endpoint variable), EC2 detection from a fake DMI tree, and the
// controller's link seed (C4). Ported from gpuio's tests/unit/auth/test_imds.cpp (the credential
// cases become instance-type cases) and tests/unit/s3/test_link_seed.cpp (the s3_client cases
// need the worker and are not ported). The real link-local IMDS address is never contacted: every
// case sets an endpoint, AWS_EC2_METADATA_DISABLED or a non-EC2 DMI tree.

#include "io/rest/cloud/http_client.hpp"
#include "io/rest/cloud/imds.hpp"
#include "io/rest/errc.hpp"
#include "io/rest/planner/planner_remote.hpp"

#include <arpa/inet.h>
#include <catch2/catch_all.hpp>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <ranges>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using cucascade::io::rest::errc;
using cucascade::io::rest::planner::link_seed_config;
namespace cloud = cucascade::io::rest::cloud;

namespace {

/// Sets/unsets variables and restores the previous values on destruction. Construction clears
/// every IMDS variable the client reads so the developer's own setup cannot leak into a test.
class env_guard {
 public:
  env_guard()
  {
    for (char const* n : {"AWS_EC2_METADATA_DISABLED",
                          "AWS_EC2_METADATA_SERVICE_ENDPOINT",
                          "AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE",
                          "AWS_METADATA_SERVICE_TIMEOUT",
                          "AWS_METADATA_SERVICE_NUM_ATTEMPTS"}) {
      unset(n);
    }
  }
  env_guard(env_guard const&)            = delete;
  env_guard& operator=(env_guard const&) = delete;
  env_guard(env_guard&&)                 = delete;
  env_guard& operator=(env_guard&&)      = delete;
  ~env_guard()
  {
    for (auto& it : std::ranges::reverse_view(_saved)) {
      if (it.second) {
        ::setenv(it.first.c_str(), it.second->c_str(), 1);
      } else {
        ::unsetenv(it.first.c_str());
      }
    }
  }
  void set(std::string const& name, std::string const& value)
  {
    save(name);
    ::setenv(name.c_str(), value.c_str(), 1);
  }
  void unset(std::string const& name)
  {
    save(name);
    ::unsetenv(name.c_str());
  }

 private:
  void save(std::string const& name)
  {
    char const* v = std::getenv(name.c_str());
    _saved.emplace_back(name, v != nullptr ? std::optional<std::string>{v} : std::nullopt);
  }
  std::vector<std::pair<std::string, std::optional<std::string>>> _saved;
};

/// A listening loopback socket on an ephemeral port.
int listen_loopback(std::uint16_t& port)
{
  int const fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  int one      = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in a{};
  a.sin_family      = AF_INET;
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  a.sin_port        = 0;
  if (::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0 && ::listen(fd, 16) == 0) {
    socklen_t len = sizeof a;
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len);
    port = ntohs(a.sin_port);
  }
  return fd;
}

/// Read one request head (until CRLFCRLF); false on EOF / timeout.
bool read_head(int fd, std::string& req)
{
  std::array<char, 2048> buf{};
  while (req.find("\r\n\r\n") == std::string::npos) {
    pollfd p{fd, POLLIN, 0};
    if (::poll(&p, 1, 2000) <= 0) { return false; }
    auto const n = ::recv(fd, buf.data(), buf.size(), 0);
    if (n <= 0) { return false; }
    req.append(buf.data(), static_cast<std::size_t>(n));
  }
  return true;
}

void send_all(int fd, std::string const& data)
{
  std::size_t off = 0;
  while (off < data.size()) {
    auto const n = ::send(fd, data.data() + off, data.size() - off, MSG_NOSIGNAL);
    if (n <= 0) { return; }
    off += static_cast<std::size_t>(n);
  }
}

/// IMDSv2 fake on 127.0.0.1:<ephemeral>: PUT /latest/api/token -> "TOKEN-<n>"; GET
/// /latest/meta-data/instance-type with the current token -> the configured type; 401 otherwise.
class fake_imds {
 public:
  struct state {
    int token_puts{0}, type_gets{0}, unauthorized{0}, requests{0};
    unsigned last_ttl{0};
    std::string token;  // currently valid token
    std::string type{"g7e.8xlarge"};
    int type_status{200};
    int force_401{0};  // answer the next N token-bearing GETs with 401
    bool chunked{false};
    bool hang{false};             // accept, never answer
    bool token_forbidden{false};  // 403 on PUT (IMDSv2 disabled / hop limit)
  };

  fake_imds()
  {
    _listen = listen_loopback(_port);
    _thread = std::thread{[this] { run(); }};
  }
  fake_imds(fake_imds const&)            = delete;
  fake_imds& operator=(fake_imds const&) = delete;
  fake_imds(fake_imds&&)                 = delete;
  fake_imds& operator=(fake_imds&&)      = delete;
  ~fake_imds()
  {
    _stop.store(true);
    _thread.join();
    ::close(_listen);
  }

  [[nodiscard]] std::uint16_t port() const noexcept { return _port; }
  [[nodiscard]] std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(_port); }

  template <class F>
  auto with(F&& f)
  {
    std::lock_guard lock{_mutex};
    return f(_st);
  }

 private:
  void run()
  {
    while (!_stop.load()) {
      pollfd p{_listen, POLLIN, 0};
      if (::poll(&p, 1, 20) <= 0) { continue; }
      int const fd = ::accept4(_listen, nullptr, nullptr, SOCK_CLOEXEC);
      if (fd < 0) { continue; }
      serve(fd);
      ::close(fd);
    }
  }

  void serve(int fd)
  {
    std::string req;
    if (!read_head(fd, req)) { return; }
    auto header = [&](std::string const& name) -> std::string {
      auto const pos = req.find("\r\n" + name + ": ");
      if (pos == std::string::npos) { return {}; }
      auto const b = pos + name.size() + 4;
      return req.substr(b, req.find("\r\n", b) - b);
    };
    auto const sp1    = req.find(' ');
    auto const method = req.substr(0, sp1);
    auto const path   = req.substr(sp1 + 1, req.find(' ', sp1 + 1) - sp1 - 1);

    bool hang = false;
    {
      std::lock_guard lock{_mutex};
      ++_st.requests;
      hang = _st.hang;
    }
    if (hang) {  // accept and never answer (outside the lock: the test may inspect state)
      std::this_thread::sleep_for(std::chrono::milliseconds{400});
      return;
    }
    int status = 404;
    std::string body;
    bool chunked = false;
    {
      std::lock_guard lock{_mutex};
      if (method == "PUT" && path == "/latest/api/token") {
        auto const ttl   = header("X-aws-ec2-metadata-token-ttl-seconds");
        unsigned const v = ttl.empty() ? 0 : static_cast<unsigned>(std::stoul(ttl));
        if (_st.token_forbidden) {
          status = 403;
        } else if (v < 1 || v > 21600) {
          status = 400;
        } else {
          ++_st.token_puts;
          _st.last_ttl = v;
          _st.token    = "TOKEN-" + std::to_string(_st.token_puts);
          status       = 200;
          body         = _st.token;
        }
      } else if (method == "GET") {
        auto const tok = header("X-aws-ec2-metadata-token");
        if (tok.empty() || tok != _st.token || _st.force_401 > 0) {
          if (_st.force_401 > 0) {
            --_st.force_401;
            _st.token.clear();  // the token "expired"
          }
          ++_st.unauthorized;
          status = 401;
        } else if (path == "/latest/meta-data/instance-type") {
          ++_st.type_gets;
          status  = _st.type_status;
          body    = status == 200 ? _st.type + "\n" : "";
          chunked = _st.chunked;
        }
      }
    }
    std::string resp = "HTTP/1.1 " + std::to_string(status) + " X\r\nConnection: close\r\n";
    if (chunked) {
      resp += "Transfer-Encoding: chunked\r\n\r\n";
      for (std::size_t i = 0; i < body.size(); i += 4) {
        auto const part = body.substr(i, 4);
        std::array<char, 32> hex{};
        std::snprintf(hex.data(), hex.size(), "%zx\r\n", part.size());
        resp += hex.data() + part + "\r\n";
      }
      resp += "0\r\n\r\n";
    } else {
      resp += "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    }
    send_all(fd, resp);
  }

  int _listen{-1};
  std::uint16_t _port{0};
  std::atomic<bool> _stop{false};
  std::mutex _mutex;
  state _st;
  std::thread _thread;
};

/// Serves one fixed raw response per connection (after reading the request head).
class raw_server {
 public:
  explicit raw_server(std::string response) : _response(std::move(response))
  {
    _listen = listen_loopback(_port);
    _thread = std::thread{[this] {
      while (!_stop.load()) {
        pollfd p{_listen, POLLIN, 0};
        if (::poll(&p, 1, 20) <= 0) { continue; }
        int const fd = ::accept4(_listen, nullptr, nullptr, SOCK_CLOEXEC);
        if (fd < 0) { continue; }
        std::string req;
        if (read_head(fd, req)) { send_all(fd, _response); }
        ::close(fd);
      }
    }};
  }
  raw_server(raw_server const&)            = delete;
  raw_server& operator=(raw_server const&) = delete;
  raw_server(raw_server&&)                 = delete;
  raw_server& operator=(raw_server&&)      = delete;
  ~raw_server()
  {
    _stop.store(true);
    _thread.join();
    ::close(_listen);
  }
  [[nodiscard]] std::uint16_t port() const noexcept { return _port; }

 private:
  std::string _response;
  int _listen{-1};
  std::uint16_t _port{0};
  std::atomic<bool> _stop{false};
  std::thread _thread;
};

/// A fake sysfs tree with a DMI vendor and / or a hypervisor uuid (empty: no file).
struct fake_sysfs {
  explicit fake_sysfs(std::string const& vendor, std::string const& uuid = {})
  {
    auto tmpl = (std::filesystem::temp_directory_path() / "cucascade-sysfs-XXXXXX").string();
    root      = ::mkdtemp(tmpl.data()) != nullptr ? tmpl : std::string{};
    if (!vendor.empty()) {
      std::filesystem::create_directories(root + "/devices/virtual/dmi/id");
      std::ofstream(root + "/devices/virtual/dmi/id/sys_vendor") << vendor << "\n";
    }
    if (!uuid.empty()) {
      std::filesystem::create_directories(root + "/hypervisor");
      std::ofstream(root + "/hypervisor/uuid") << uuid << "\n";
    }
  }
  fake_sysfs(fake_sysfs const&)            = delete;
  fake_sysfs& operator=(fake_sysfs const&) = delete;
  fake_sysfs(fake_sysfs&&)                 = delete;
  fake_sysfs& operator=(fake_sysfs&&)      = delete;
  ~fake_sysfs()
  {
    std::error_code ec;
    if (!root.empty()) { std::filesystem::remove_all(root, ec); }
  }
  std::string root;
};

link_seed_config config_for(fake_imds const& s)
{
  link_seed_config c;
  c.imds_endpoint = s.endpoint();
  c.imds_timeout  = std::chrono::milliseconds{2000};
  c.imds_attempts = 1;
  return c;
}

std::string get_request(std::uint16_t port)
{
  return "GET / HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(port) +
         "\r\nConnection: close\r\n\r\n";
}

}  // namespace

TEST_CASE("cloud.http_client: framing, interim heads and errors", "[rest][planner][cloud]")
{
  using std::chrono::milliseconds;
  SECTION("Content-Length after a 100 Continue")
  {
    raw_server srv(
      "HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello");
    auto r =
      cloud::http_exchange("127.0.0.1", srv.port(), get_request(srv.port()), milliseconds{2000});
    REQUIRE(r.has_value());
    CHECK(r->status == 200);
    CHECK(r->body == "hello");
  }
  SECTION("chunked body (Transfer-Encoding overrides Content-Length)")
  {
    raw_server srv(
      "HTTP/1.1 200 OK\r\nContent-Length: 99\r\nTransfer-Encoding: chunked\r\n\r\n"
      "3\r\nabc\r\n2;ext=1\r\nde\r\n0\r\n\r\n");
    auto r =
      cloud::http_exchange("127.0.0.1", srv.port(), get_request(srv.port()), milliseconds{2000});
    REQUIRE(r.has_value());
    CHECK(r->body == "abcde");
  }
  SECTION("read until close without a length")
  {
    raw_server srv("HTTP/1.0 404 Not Found\r\nServer: x\r\n\r\nnot here");
    auto r =
      cloud::http_exchange("127.0.0.1", srv.port(), get_request(srv.port()), milliseconds{2000});
    REQUIRE(r.has_value());
    CHECK(r->status == 404);
    CHECK(r->body == "not here");
  }
  SECTION("short body")
  {
    raw_server srv("HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nabc");
    auto r =
      cloud::http_exchange("127.0.0.1", srv.port(), get_request(srv.port()), milliseconds{2000});
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error() == errc::http_length_mismatch);
  }
  SECTION("malformed status line")
  {
    raw_server srv("HTTP/1.1 2x0 OK\r\n\r\n");
    auto r =
      cloud::http_exchange("127.0.0.1", srv.port(), get_request(srv.port()), milliseconds{2000});
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error() == errc::http_bad_message);
  }
  SECTION("head above 16 KiB")
  {
    raw_server srv("HTTP/1.1 200 OK\r\nX-Big: " + std::string(20000, 'a'));  // never terminated
    auto r =
      cloud::http_exchange("127.0.0.1", srv.port(), get_request(srv.port()), milliseconds{2000});
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error() == errc::http_header_too_large);
  }
  SECTION("connection refused")
  {
    auto r = cloud::http_exchange("127.0.0.1", 1, get_request(1), milliseconds{2000});
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error() == errc::connect_failed);
  }
}

TEST_CASE("cloud.imds: token flow and the instance type", "[rest][planner][cloud]")
{
  env_guard env;
  fake_imds server;
  REQUIRE(server.port() != 0);
  cloud::imds_client c(config_for(server));
  auto r = c.get("/latest/meta-data/instance-type");
  REQUIRE(r.has_value());
  CHECK(*r == "g7e.8xlarge");  // trailing newline trimmed
  server.with([](auto& s) {
    CHECK(s.token_puts == 1);
    CHECK(s.last_ttl == 21600);
    CHECK(s.type_gets == 1);
    CHECK(s.unauthorized == 0);
  });
  // The token is reused.
  REQUIRE(c.get("/latest/meta-data/instance-type").has_value());
  server.with([](auto& s) {
    CHECK(s.token_puts == 1);
    CHECK(s.type_gets == 2);
  });
  // A non-200 answer is a refusal.
  server.with([](auto& s) { s.type_status = 404; });
  auto const missing = c.get("/latest/meta-data/instance-type");
  REQUIRE_FALSE(missing.has_value());
  CHECK(missing.error() == errc::credentials_unavailable);
}

TEST_CASE("cloud.imds: 401 on an expired token fetches a new token", "[rest][planner][cloud]")
{
  env_guard env;
  fake_imds server;
  cloud::imds_client c(config_for(server));
  REQUIRE(c.get("/latest/meta-data/instance-type").has_value());
  server.with([](auto& s) { s.force_401 = 1; });
  auto r = c.get("/latest/meta-data/instance-type");
  REQUIRE(r.has_value());
  CHECK(*r == "g7e.8xlarge");
  server.with([](auto& s) {
    CHECK(s.unauthorized == 1);
    CHECK(s.token_puts == 2);
  });
  server.with([](auto& s) { s.force_401 = 2; });  // twice in a row: give up
  auto const twice = c.get("/latest/meta-data/instance-type");
  REQUIRE_FALSE(twice.has_value());
  CHECK(twice.error() == errc::credentials_unavailable);
}

TEST_CASE("cloud.imds: token TTL lapse triggers a new PUT (chunked replies)",
          "[rest][planner][cloud]")
{
  env_guard env;
  fake_imds server;
  server.with([](auto& s) { s.chunked = true; });  // also exercises chunked framing
  auto cfg           = config_for(server);
  cfg.imds_token_ttl = std::chrono::seconds{1};  // re-fetched 250 ms before it lapses
  cloud::imds_client c(cfg);
  auto r = c.get("/latest/meta-data/instance-type");
  REQUIRE(r.has_value());
  CHECK(*r == "g7e.8xlarge");
  std::this_thread::sleep_for(std::chrono::milliseconds{900});
  REQUIRE(c.get("/latest/meta-data/instance-type").has_value());
  server.with([](auto& s) {
    CHECK(s.last_ttl == 1);
    CHECK(s.token_puts == 2);
    CHECK(s.unauthorized == 0);
  });
}

TEST_CASE("cloud.imds: failures", "[rest][planner][cloud]")
{
  env_guard env;
  fake_imds server;
  SECTION("IMDSv2 token forbidden (403)")
  {
    server.with([](auto& s) { s.token_forbidden = true; });
    auto const r = cloud::imds_get(config_for(server), "/latest/meta-data/instance-type");
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error() == errc::credentials_unavailable);
  }
  SECTION("connection refused")
  {
    auto cfg          = config_for(server);
    cfg.imds_endpoint = "http://127.0.0.1:1";
    auto const r      = cloud::imds_get(cfg, "/latest/meta-data/instance-type");
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error() == errc::connect_failed);
  }
  SECTION("timeout, retried per attempt")
  {
    server.with([](auto& s) { s.hang = true; });
    auto cfg          = config_for(server);
    cfg.imds_timeout  = std::chrono::milliseconds{150};
    cfg.imds_attempts = 2;
    auto const t0     = std::chrono::steady_clock::now();
    auto const r      = cloud::imds_get(cfg, "/latest/meta-data/instance-type");
    auto const took   = std::chrono::steady_clock::now() - t0;
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error() == errc::request_timeout);
    CHECK(took >= std::chrono::milliseconds{250});
    CHECK(took < std::chrono::milliseconds{1500});
  }
  SECTION("malformed endpoint")
  {
    for (char const* bad : {"https://127.0.0.1:1", "http://", "http://h:0", "http://h/x", "x"}) {
      INFO(bad);
      auto cfg          = config_for(server);
      cfg.imds_endpoint = bad;
      auto const r      = cloud::imds_get(cfg, "/latest/meta-data/instance-type");
      REQUIRE_FALSE(r.has_value());
      CHECK(r.error() == errc::invalid_argument);
    }
  }
  server.with([](auto& s) { CHECK(s.type_gets == 0); });
}

TEST_CASE("cloud.imds: environment switches", "[rest][planner][cloud]")
{
  env_guard env;
  fake_imds server;
  SECTION("AWS_EC2_METADATA_DISABLED=true never contacts the service")
  {
    env.set("AWS_EC2_METADATA_DISABLED", " TRUE ");
    auto const r = cloud::imds_get(config_for(server), "/latest/meta-data/instance-type");
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error() == errc::not_supported);
    server.with([](auto& s) { CHECK(s.requests == 0); });
    env.set("AWS_EC2_METADATA_DISABLED", "false");
    CHECK(cloud::imds_get(config_for(server), "/latest/meta-data/instance-type").has_value());
  }
  SECTION("AWS_EC2_METADATA_SERVICE_ENDPOINT when the config has none")
  {
    env.set("AWS_EC2_METADATA_SERVICE_ENDPOINT", server.endpoint() + "/");
    link_seed_config cfg;
    cfg.imds_timeout = std::chrono::milliseconds{2000};
    auto const r     = cloud::imds_get(cfg, "/latest/meta-data/instance-type");
    REQUIRE(r.has_value());
    CHECK(*r == "g7e.8xlarge");
  }
}

TEST_CASE("cloud.ec2: DMI classification", "[rest][planner][cloud]")
{
  CHECK(cloud::ec2_platform_hint(fake_sysfs("Amazon EC2").root) == cloud::ec2_hint::ec2);
  CHECK(cloud::ec2_platform_hint(fake_sysfs("Dell Inc.").root) == cloud::ec2_hint::not_ec2);
  CHECK(cloud::ec2_platform_hint(fake_sysfs("", "ec2a1b2c3d-0000").root) == cloud::ec2_hint::ec2);
  CHECK(cloud::ec2_platform_hint(fake_sysfs("", "4c4c4544-0000").root) == cloud::ec2_hint::not_ec2);
  CHECK(cloud::ec2_platform_hint(fake_sysfs("").root) == cloud::ec2_hint::unknown);
  // the cached answer agrees with the real tree (no network either way)
  CHECK(cloud::running_on_ec2() == (cloud::ec2_platform_hint() == cloud::ec2_hint::ec2));
  CHECK(cloud::running_on_ec2() == cloud::running_on_ec2());
}

TEST_CASE("cloud.link_seed: IMDS instance type -> link seed for every table family (C4)",
          "[rest][planner][cloud]")
{
  env_guard env;
  fake_imds imds;
  fake_sysfs not_ec2("QEMU");  // an explicit IMDS endpoint is asked regardless of DMI
  auto cfg       = config_for(imds);
  cfg.sysfs_root = not_ec2.root;
  struct row {
    char const* type;
    double gbps;
  };
  for (row const r : {row{"c6in.32xlarge", 200},
                      row{"c7gn.16xlarge", 200},
                      row{"g7e.8xlarge", 100},
                      row{"g6e.48xlarge", 400},
                      row{"m6a.48xlarge", 50},
                      row{"m7i.48xlarge", 50},
                      row{"p5.48xlarge", 3200},
                      row{"trn1.32xlarge", 800},
                      row{"trn2.48xlarge", 3200},
                      row{"hpc7g.16xlarge", 200},
                      row{"z99.metal", 25}}) {  // an EC2 type missing from the table
    INFO(r.type);
    imds.with([&](auto& s) { s.type = r.type; });
    std::string type;
    CHECK(cloud::resolve_link_seed(cfg, &type) == r.gbps);
    CHECK(type == r.type);
  }
  imds.with([](auto& s) {
    CHECK(s.token_puts >= 1);
    CHECK(s.type_gets == 11);
  });
}

TEST_CASE("cloud.link_seed: IMDS disabled, unreachable or a non-EC2 host -> 10 Gbps",
          "[rest][planner][cloud]")
{
  using namespace std::chrono_literals;
  fake_imds imds;
  {
    env_guard env;  // disabled: IMDS never contacted
    env.set("AWS_EC2_METADATA_DISABLED", "true");
    std::string type = "x";
    CHECK(cloud::resolve_link_seed(config_for(imds), &type) == 10.0);
    CHECK(type.empty());
  }
  {
    env_guard env;  // unreachable (nothing listens on port 1): 10
    link_seed_config cfg;
    cfg.imds_endpoint = "http://127.0.0.1:1";
    CHECK(cloud::resolve_link_seed(cfg) == 10.0);
  }
  {
    env_guard env;  // DMI says not EC2 and no endpoint is configured: no IMDS round trip at all
    fake_sysfs not_ec2("QEMU");
    link_seed_config cfg;
    cfg.sysfs_root = not_ec2.root;
    auto const t0  = std::chrono::steady_clock::now();
    CHECK(cloud::resolve_link_seed(cfg) == 10.0);
    CHECK(std::chrono::steady_clock::now() - t0 < 200ms);
  }
  {
    env_guard env;  // the endpoint variable overrides a non-EC2 DMI answer
    env.set("AWS_EC2_METADATA_SERVICE_ENDPOINT", imds.endpoint());
    fake_sysfs not_ec2("QEMU");
    link_seed_config cfg;
    cfg.sysfs_root = not_ec2.root;
    CHECK(cloud::resolve_link_seed(cfg) == 100.0);
  }
  imds.with([](auto& s) { CHECK(s.type_gets == 1); });
}

TEST_CASE("cloud.link_seed: the process-wide seed resolves once, off the caller's thread",
          "[rest][planner][cloud]")
{
  using namespace std::chrono_literals;
  env_guard env;
  env.set("AWS_EC2_METADATA_DISABLED", "true");  // never the real IMDS from a test
  bool const started = cloud::start_link_seed();
  CHECK_FALSE(cloud::start_link_seed());  // idempotent
  auto const t0 = std::chrono::steady_clock::now();
  while (!cloud::link_seed_ready() && std::chrono::steady_clock::now() - t0 < 10s) {
    std::this_thread::sleep_for(1ms);
  }
  REQUIRE(cloud::link_seed_ready());
  auto const seed = cloud::link_gbps_seed();
  REQUIRE(seed.has_value());
  CHECK(*seed > 0.0);
  if (started) { CHECK(*seed == cucascade::io::rest::planner::default_link_gbps); }
}
