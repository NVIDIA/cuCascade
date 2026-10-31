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

/// @file test_rest_pool_parity.cpp
/// @brief The REST reactor's connection pool against Sirius' libcurl behaviour (curl parity):
/// idle-age expiry is lazy (one slot at a time, when a request needs it), a FIN'd idle connection
/// is detected before reuse, a pooled connection that dies before any response byte is retried
/// once for free, warm-up issues exactly max_connections zero-key LIST exchanges once per
/// conn_max_age on fresh connections, HTTP/1.1 upkeep is off by default and staggered when on;
/// plus the same-pass start invariant of the worker loop, the request_timeout bound of
/// control-plane calls, and config validation.

#include "io/rest/reactor_test_hooks.hpp"
#include "rest_contract_test_utils.hpp"
#include "utils/child_test_runner.hpp"

#include <cucascade/io/io_context.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>

#include <arpa/inet.h>
#include <catch2/catch_all.hpp>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using cucascade::io::slice;
using cucascade::io::rest::rest_ioctx;
using cucascade::io::rest::rest_reactor;
using cucascade::io::rest::transport_stats;
using cucascade::test::key_response_script;
using cucascade::test::list_capable_mock_authorizer;
using cucascade::test::loopback_range_server;
using cucascade::test::range_fault_policy;
using cucascade::test::require_rest_bytes_equal;
using cucascade::test::rest_contract_config;
using cucascade::test::rest_shutdown_guard;
using cucascade::test::rest_test_payload;
using cucascade::test::scripted_response;
using namespace std::chrono_literals;
namespace rest = cucascade::io::rest;

/// A @c rest_ioctx whose reactors' counters the test can read.
class probed_ioctx final : public rest_ioctx {
 public:
  using rest_ioctx::rest_ioctx;
  [[nodiscard]] transport_stats reactor_stats(std::size_t i = 0) const
  {
    return _reactors.at(i)->stats();
  }
  [[nodiscard]] std::size_t reactor_count() const noexcept { return _reactors.size(); }
};

/// Records when each LIST was authorized (the worker authorizes every exchange as it issues it).
class timed_authorizer final : public rest::request_authorizer {
 public:
  explicit timed_authorizer(std::string endpoint) : _endpoint(std::move(endpoint)) {}
  rest::authorized_request authorize(rest::object_ref const& obj,
                                     rest::request_method,
                                     std::chrono::seconds) override
  {
    return {_endpoint + "/" + obj.bucket + "/" + obj.key, {}};
  }
  rest::authorized_request authorize_list(std::string_view bucket,
                                          std::string_view canonical_query,
                                          std::chrono::seconds) override
  {
    {
      std::lock_guard lock{_mutex};
      _lists.push_back(std::chrono::steady_clock::now());
    }
    return {_endpoint + "/" + std::string{bucket} + "?" + std::string{canonical_query}, {}};
  }
  [[nodiscard]] std::vector<std::chrono::steady_clock::time_point> lists() const
  {
    std::lock_guard lock{_mutex};
    return _lists;
  }

 private:
  std::string _endpoint;
  mutable std::mutex _mutex;
  std::vector<std::chrono::steady_clock::time_point> _lists;
};

std::shared_ptr<probed_ioctx> make_ioctx(std::string const& endpoint,
                                         rest::config cfg,
                                         std::size_t reactors                           = 1,
                                         std::shared_ptr<rest::request_authorizer> auth = {})
{
  if (!auth) { auth = std::make_shared<list_capable_mock_authorizer>(endpoint); }
  auto context = std::make_shared<rest_reactor::reactor_context>(cfg, std::move(auth), nullptr);
  auto ioctx   = std::make_shared<probed_ioctx>(reactors, std::move(context));
  ioctx->start();
  return ioctx;
}

template <typename Predicate>
transport_stats wait_for_stats(probed_ioctx const& ioctx,
                               Predicate ready,
                               std::size_t reactor               = 0,
                               std::chrono::milliseconds timeout = 5s)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  auto stats          = ioctx.reactor_stats(reactor);
  while (!ready(stats) && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(10ms);
    stats = ioctx.reactor_stats(reactor);
  }
  return stats;
}

template <typename Predicate>
bool wait_until(Predicate&& ready, std::chrono::milliseconds timeout = 5s)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (!ready() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(5ms);
  }
  return ready();
}

std::vector<std::uint8_t> read_range(probed_ioctx& ioctx,
                                     cucascade::io::io_object const& object,
                                     std::size_t offset,
                                     std::size_t size)
{
  std::vector<std::uint8_t> out(size);
  std::vector<slice> slices{slice{offset, size, out.data()}};
  REQUIRE(std::move(ioctx.host_readv_async_io(object, slices)).get(30s) == size);
  return out;
}

/// Raw keep-alive HTTP server: answers every request head with @c response; with
/// @c close_after_response it then closes (FIN), with @c die_on_second_request it reads the
/// second request of a connection and closes without answering (a stale pooled connection).
class scripted_server {
 public:
  struct options {
    std::string response;
    bool close_after_response{false};
    bool die_on_second_request{false};
  };

  explicit scripted_server(options opts) : _opts(std::move(opts))
  {
    _fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    REQUIRE(_fd >= 0);
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    REQUIRE(::bind(_fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0);
    REQUIRE(::listen(_fd, 16) == 0);
    socklen_t len = sizeof addr;
    REQUIRE(::getsockname(_fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0);
    _port   = ntohs(addr.sin_port);
    _thread = std::thread([this] { accept_loop(); });
  }
  ~scripted_server()
  {
    _stop = true;
    ::shutdown(_fd, SHUT_RDWR);
    _thread.join();
    {
      std::lock_guard lock{_mutex};
      for (int const c : _open) {
        ::shutdown(c, SHUT_RDWR);
      }
    }
    for (auto& t : _connections) {
      t.join();
    }
    ::close(_fd);
  }
  scripted_server(scripted_server const&)            = delete;
  scripted_server& operator=(scripted_server const&) = delete;

  [[nodiscard]] std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(_port); }
  [[nodiscard]] std::size_t requests() const noexcept { return _requests.load(); }
  [[nodiscard]] std::size_t answered() const noexcept { return _answered.load(); }
  [[nodiscard]] std::size_t connections() const noexcept { return _accepted.load(); }

 private:
  void accept_loop()
  {
    while (!_stop) {
      int const conn = ::accept4(_fd, nullptr, nullptr, SOCK_CLOEXEC);
      if (conn < 0) {
        if (errno == EINTR) { continue; }
        return;
      }
      ++_accepted;
      std::lock_guard lock{_mutex};
      _open.push_back(conn);
      _connections.emplace_back([this, conn] { serve(conn); });
    }
  }

  void serve(int conn)
  {
    std::string pending;
    std::array<char, 4096> buf{};
    for (std::size_t served = 0;; ++served) {
      while (pending.find("\r\n\r\n") == std::string::npos) {
        auto const n = ::recv(conn, buf.data(), buf.size(), 0);
        if (n < 0 && errno == EINTR) { continue; }
        if (n <= 0) {
          ::close(conn);
          return;
        }
        pending.append(buf.data(), static_cast<std::size_t>(n));
      }
      pending.erase(0, pending.find("\r\n\r\n") + 4);
      ++_requests;
      if (_opts.die_on_second_request && served == 1) {
        ::close(conn);
        return;
      }
      std::size_t sent = 0;
      while (sent < _opts.response.size()) {
        auto const n =
          ::send(conn, _opts.response.data() + sent, _opts.response.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) { break; }
        sent += static_cast<std::size_t>(n);
      }
      ++_answered;
      if (_opts.close_after_response) {
        ::shutdown(conn, SHUT_WR);
        std::this_thread::sleep_for(20ms);
        ::close(conn);
        return;
      }
    }
  }

  options _opts;
  int _fd{-1};
  std::uint16_t _port{0};
  std::atomic<bool> _stop{false};
  std::atomic<std::size_t> _requests{0};
  std::atomic<std::size_t> _answered{0};
  std::atomic<std::size_t> _accepted{0};
  std::mutex _mutex;
  std::vector<int> _open;
  std::vector<std::thread> _connections;
  std::thread _thread;
};

}  // namespace

// ---- pool rule 1: reuse only while young and healthy, replaced lazily -----------------------

TEST_CASE(
  "rest pool: an idle connection past conn_max_age is replaced only when a request needs it",
  "[rest][pool]")
{
  auto payload = rest_test_payload(256UL << 10);
  loopback_range_server server(payload);
  auto cfg            = rest_contract_config();
  cfg.max_connections = 1;
  cfg.conn_max_age    = 1s;
  auto ioctx          = make_ioctx(server.endpoint(), cfg);
  rest_shutdown_guard cleanup{*ioctx};
  auto object = ioctx->open_io_object("s3://age-bucket/object.bin", payload.size());

  require_rest_bytes_equal(read_range(*ioctx, *object, 0, 4096), {payload.data(), 4096});
  require_rest_bytes_equal(read_range(*ioctx, *object, 8192, 4096), {payload.data() + 8192, 4096});
  CHECK(server.accepted_connection_count() == 1);  // reused while young

  std::this_thread::sleep_for(1300ms);
  // Idle past the cutoff, but nothing was swept: no reconnect, no close while idle.
  CHECK(server.accepted_connection_count() == 1);
  auto idle = ioctx->reactor_stats();
  CHECK(idle.closed == 0);
  CHECK(idle.connects == 1);

  require_rest_bytes_equal(read_range(*ioctx, *object, 100, 4096), {payload.data() + 100, 4096});
  CHECK(server.accepted_connection_count() == 2);
  auto const after = wait_for_stats(*ioctx, [](transport_stats const& s) { return s.gets >= 3; });
  CHECK(after.idle_closes == 1);
  CHECK(after.connects == 2);
  CHECK(after.retries == 0);
}

TEST_CASE("rest pool: a FIN'd idle connection is detected before reuse", "[rest][pool]")
{
  scripted_server server(
    {.response             = "HTTP/1.1 200 OK\r\nContent-Length: 4096\r\nETag: \"e\"\r\n\r\n",
     .close_after_response = true});
  auto cfg               = rest_contract_config();
  cfg.max_connections    = 1;
  cfg.max_retry_attempts = 1;  // no ordinary retry may hide a failure
  auto ioctx             = make_ioctx(server.endpoint(), cfg);
  rest_shutdown_guard cleanup{*ioctx};
  for (int i = 0; i < 3; ++i) {
    auto object = ioctx->open_io_object("s3://fin-bucket/object.bin");
    CHECK(object->size() == 4096);
    std::this_thread::sleep_for(100ms);  // the FIN arrives while the connection is idle
  }
  CHECK(server.requests() == 3);
  CHECK(server.connections() == 3);
  auto const stats = wait_for_stats(*ioctx, [](transport_stats const& s) { return s.heads >= 3; });
  CHECK(stats.dead_closes == 2);
  CHECK(stats.stale_retries == 0);
  CHECK(stats.retries == 0);
}

// ---- pool rule 2: the stale-connection retry is free ---------------------------------------

TEST_CASE("rest pool: a pooled connection that dies before the response is retried for free",
          "[rest][pool]")
{
  auto const via = GENERATE(std::string{"head"}, std::string{"get"});
  CAPTURE(via);
  std::string const head = "HTTP/1.1 200 OK\r\nContent-Length: 16\r\nETag: \"e\"\r\n\r\n";
  scripted_server server(
    {.response = via == "head" ? head : head + "0123456789abcdef", .die_on_second_request = true});
  auto cfg               = rest_contract_config();
  cfg.max_connections    = 1;
  cfg.max_retry_attempts = 1;  // the stale retry does not count against this
  auto ioctx             = make_ioctx(server.endpoint(), cfg);
  rest_shutdown_guard cleanup{*ioctx};
  if (via == "head") {
    for (int i = 0; i < 2; ++i) {
      auto object = ioctx->open_io_object("s3://stale-bucket/object.bin");
      CHECK(object->size() == 16);
    }
  } else {
    auto object = ioctx->open_io_object("s3://stale-bucket/object.bin", 16);
    for (int i = 0; i < 2; ++i) {
      auto const got = read_range(*ioctx, *object, 0, 16);
      CHECK(std::string(got.begin(), got.end()) == "0123456789abcdef");
    }
  }
  // Request 2 died on the pooled connection; its free retry ran on a new one.
  CHECK(server.requests() == 3);
  CHECK(server.connections() == 2);
  auto const stats =
    wait_for_stats(*ioctx, [](transport_stats const& s) { return s.stale_retries >= 1; });
  CHECK(stats.stale_retries == 1);
  CHECK(stats.stale_failures == 1);
  CHECK(stats.retries == 0);
  CHECK(stats.failures == 0);
}

TEST_CASE("rest pool: the free stale retry runs on a fresh connection, not another pooled one",
          "[rest][pool]")
{
  // Every connection serves one request and dies on its second.  Warm-up leaves two pooled
  // connections that both look healthy; a read on one of them dies, and its free retry must open
  // a new connection: on the other pooled one it would die again and, with one attempt allowed,
  // fail the read.
  scripted_server server(
    {.response = "HTTP/1.1 200 OK\r\nContent-Length: 16\r\nETag: \"e\"\r\n\r\n0123456789abcdef",
     .die_on_second_request = true});
  auto cfg               = rest_contract_config();
  cfg.max_connections    = 2;
  cfg.max_retry_attempts = 1;
  auto ioctx             = make_ioctx(server.endpoint(), cfg);
  rest_shutdown_guard cleanup{*ioctx};
  ioctx->warmup("s3://stale-bucket");
  REQUIRE(wait_until([&] { return server.answered() == 2; }));
  REQUIRE(wait_until([&] {
    return ioctx->reactor_stats().established == 2 && ioctx->reactor_stats().warmups == 2;
  }));
  auto object    = ioctx->open_io_object("s3://stale-bucket/object.bin", 16);
  auto const got = read_range(*ioctx, *object, 0, 16);
  CHECK(std::string(got.begin(), got.end()) == "0123456789abcdef");
  CHECK(server.connections() == 3);  // the two warm ones and one fresh for the retry
  auto const stats = wait_for_stats(*ioctx, [](transport_stats const& s) { return s.gets >= 1; });
  CHECK(stats.stale_retries == 1);
  CHECK(stats.retries == 0);
  CHECK(stats.failures == 0);
}

// ---- pool rule 3: warm-up -------------------------------------------------------------------

TEST_CASE("rest pool: warm-up issues max_connections LIST exchanges once per conn_max_age",
          "[rest][pool][warmup]")
{
  auto payload = rest_test_payload(64UL << 10);
  loopback_range_server server(payload);
  auto cfg            = rest_contract_config();
  cfg.max_connections = 4;
  cfg.conn_max_age    = 2s;
  auto auth           = std::make_shared<timed_authorizer>(server.endpoint());
  auto ioctx          = make_ioctx(server.endpoint(), cfg, 1, auth);
  rest_shutdown_guard cleanup{*ioctx};

  ioctx->warmup("s3://warm-bucket");
  REQUIRE(wait_until([&] { return server.list_count() == 4; }));
  CHECK(server.accepted_connection_count() == 4);  // one fresh connection per slot
  // Rate limited per bucket: an immediate second call (and an object URL of the same bucket) is
  // a no-op.
  ioctx->warmup("s3://warm-bucket");
  ioctx->warmup("s3://warm-bucket/some/key.parquet");
  std::this_thread::sleep_for(300ms);
  CHECK(server.list_count() == 4);
  CHECK(auth->lists().size() == 4);

  // Reads find the pool hot: no new connection.
  auto object = ioctx->open_io_object("s3://warm-bucket/object.bin", payload.size());
  require_rest_bytes_equal(read_range(*ioctx, *object, 0, 1000), {payload.data(), 1000});
  CHECK(server.accepted_connection_count() == 4);

  // Once conn_max_age passed, the next warm-up runs again on fresh connections.
  std::this_thread::sleep_for(2100ms);
  ioctx->warmup("s3://warm-bucket");
  REQUIRE(wait_until([&] { return server.list_count() == 8; }));
  CHECK(server.accepted_connection_count() == 8);
  auto const stats =
    wait_for_stats(*ioctx, [](transport_stats const& s) { return s.warmups >= 8; });
  CHECK(stats.warmups == 8);
  CHECK(stats.upkeeps == 0);
}

TEST_CASE("rest pool: warm-up failures never fail a read", "[rest][pool][warmup]")
{
  auto payload = rest_test_payload(16UL << 10);
  loopback_range_server server(payload);
  auto cfg = rest_contract_config();
  // An authorizer without LIST support: every warm-up exchange fails to sign.
  class no_list final : public rest::request_authorizer {
   public:
    explicit no_list(std::string e) : _e(std::move(e)) {}
    rest::authorized_request authorize(rest::object_ref const& o,
                                       rest::request_method,
                                       std::chrono::seconds) override
    {
      return {_e + "/" + o.bucket + "/" + o.key, {}};
    }

   private:
    std::string _e;
  };
  auto ioctx = make_ioctx(server.endpoint(), cfg, 1, std::make_shared<no_list>(server.endpoint()));
  rest_shutdown_guard cleanup{*ioctx};
  ioctx->warmup("s3://warm-bucket");
  auto object = ioctx->open_io_object("s3://warm-bucket/object.bin", payload.size());
  require_rest_bytes_equal(read_range(*ioctx, *object, 0, payload.size()), payload);
  CHECK(server.list_count() == 0);
}

TEST_CASE("rest pool: idle upkeep never spins the worker while a GET is in flight",
          "[rest][pool][upkeep]")
{
  auto const enabled = GENERATE(false, true);
  CAPTURE(enabled);
  auto control = std::make_shared<cucascade::test::range_response_control>();
  range_fault_policy fault{};
  fault.body.chunk_bytes      = 1024;
  fault.body.hold_after_bytes = 1024;
  fault.body.control          = control;
  loopback_range_server server(rest_test_payload(64UL << 10), fault);
  auto cfg            = rest_contract_config();
  cfg.max_connections = 1;
  cfg.upkeep_http1    = enabled;
  cfg.upkeep_interval = 200ms;
  auto ioctx          = make_ioctx(server.endpoint(), cfg);
  rest_shutdown_guard cleanup{*ioctx, control};
  ioctx->warmup("s3://spin-bucket");  // gives upkeep a bucket
  REQUIRE(wait_until([&] { return server.list_count() == 1; }));
  auto object = ioctx->open_io_object("s3://spin-bucket/object.bin", 64UL << 10);
  std::vector<std::uint8_t> out(64UL << 10);
  std::vector<slice> slices{slice{0, out.size(), out.data()}};
  auto future = ioctx->host_readv_async_io(*object, slices);
  REQUIRE(control->wait_for_holds(1, 5s));
  // The GET is held: the worker has nothing to do but wait for the network.
  auto const before = ioctx->reactor_stats().passes;
  std::this_thread::sleep_for(1s);
  auto const passes = ioctx->reactor_stats().passes - before;
  control->release_all();
  REQUIRE(std::move(future).get(10s) == out.size());
  CHECK(passes < 100);
}

// ---- pool rule 5: HTTP/1.1 upkeep -------------------------------------------------------------

TEST_CASE("rest pool: idle upkeep is off by default and staggered when enabled",
          "[rest][pool][upkeep]")
{
  auto const enabled = GENERATE(false, true);
  CAPTURE(enabled);
  loopback_range_server server(rest_test_payload(4096));
  auto cfg            = rest_contract_config();
  cfg.max_connections = 2;
  cfg.conn_max_age    = 60s;
  cfg.upkeep_http1    = enabled;
  cfg.upkeep_interval = 400ms;
  auto auth           = std::make_shared<timed_authorizer>(server.endpoint());
  auto ioctx          = make_ioctx(server.endpoint(), cfg, 1, auth);
  rest_shutdown_guard cleanup{*ioctx};
  ioctx->warmup("s3://upkeep-bucket");
  REQUIRE(wait_until([&] { return server.list_count() == 2; }));
  std::this_thread::sleep_for(1500ms);
  auto const lists = auth->lists();
  if (!enabled) {
    CHECK(lists.size() == 2);  // nothing is sent while idle
    CHECK(ioctx->reactor_stats().upkeeps == 0);
    return;
  }
  REQUIRE(lists.size() >= 2 + 3);
  // One connection per interval / connections: never two upkeeps at once.
  for (std::size_t i = 3; i < lists.size(); ++i) {
    CHECK(lists[i] - lists[i - 1] >= 150ms);
  }
  CHECK(server.accepted_connection_count() == 2);  // the warm connections are what is kept
  CHECK(ioctx->reactor_stats().upkeeps == lists.size() - 2);
}

// ---- same-pass start --------------------------------------------------------------------------

namespace {

/// Installs reactor events for the guard's life.
struct events_guard {
  explicit events_guard(rest::testing::reactor_events const* events)
  {
    rest::testing::set_reactor_events(events);
  }
  ~events_guard() { rest::testing::set_reactor_events(nullptr); }
  events_guard(events_guard const&)            = delete;
  events_guard& operator=(events_guard const&) = delete;
};

}  // namespace

TEST_CASE("rest work queued during a worker pass starts in the same pass", "[rest][same_pass]")
{
  auto payload = rest_test_payload(64UL << 10);
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["retry.bin"].gets = {scripted_response{.status = 503}, scripted_response{}};
  loopback_range_server server(payload, {}, {}, scripts);
  auto cfg               = rest_contract_config();
  cfg.max_connections    = 2;
  cfg.max_retry_attempts = 3;
  cfg.retry_backoff_base = 0ms;  // the retry is due at once
  cfg.retry_jitter       = 0ms;

  std::mutex mutex;
  std::unordered_map<std::string, std::uint64_t> finished_at;  // key#attempt -> waits
  std::unordered_map<std::string, std::uint64_t> started_at;
  std::shared_ptr<probed_ioctx> ioctx;
  std::shared_ptr<cucascade::io::io_object> follow_object;
  std::vector<std::uint8_t> follow(4096);
  std::optional<cucascade::exec::semi_future<std::size_t>> follow_future;
  rest::testing::reactor_events events;
  events.on_get_started = [&](std::string_view key, std::size_t attempt, std::uint64_t waits) {
    std::lock_guard lock{mutex};
    started_at[std::string(key) + "#" + std::to_string(attempt)] = waits;
  };
  events.on_get_finished =
    [&](std::string_view key, std::size_t attempt, unsigned, std::uint64_t waits) {
      std::lock_guard lock{mutex};
      finished_at[std::string(key) + "#" + std::to_string(attempt)] = waits;
      // A request enqueued from a completion (on the worker thread itself).
      if (key == "first.bin" && !follow_future) {
        follow_future = ioctx->host_read_async_io(*follow_object, 0, follow.size(), follow.data());
      }
    };
  events_guard const guard{&events};
  ioctx = make_ioctx(server.endpoint(), cfg);
  rest_shutdown_guard cleanup{*ioctx};
  follow_object = ioctx->open_io_object("s3://pass-bucket/follow.bin", payload.size());

  SECTION("a request enqueued by a completion")
  {
    auto first = ioctx->open_io_object("s3://pass-bucket/first.bin", payload.size());
    require_rest_bytes_equal(read_range(*ioctx, *first, 0, 4096), {payload.data(), 4096});
    REQUIRE(wait_until([&] {
      std::lock_guard lock{mutex};
      return follow_future.has_value();
    }));
    REQUIRE(std::move(*follow_future).get(10s) == follow.size());
    std::lock_guard lock{mutex};
    REQUIRE(finished_at.contains("first.bin#0"));
    REQUIRE(started_at.contains("follow.bin#0"));
    // No blocking wait between the completion and the start.
    CHECK(started_at["follow.bin#0"] == finished_at["first.bin#0"]);
  }
  SECTION("a retry due now")
  {
    auto object = ioctx->open_io_object("s3://pass-bucket/retry.bin", payload.size());
    require_rest_bytes_equal(read_range(*ioctx, *object, 0, 4096), {payload.data(), 4096});
    std::lock_guard lock{mutex};
    REQUIRE(finished_at.contains("retry.bin#0"));
    REQUIRE(started_at.contains("retry.bin#1"));
    CHECK(started_at["retry.bin#1"] == finished_at["retry.bin#0"]);
    CHECK(server.get_count("retry.bin") == 2);
  }
  ioctx->shutdown();
}

TEST_CASE("rest control-plane batches do not starve demand reads", "[rest][fairness]")
{
  // 2 reactors x 2 connections, a 100-entry footer batch against a 20 ms server: a 4 KiB demand
  // read issued meanwhile waits for at most a request or two, not for the batch.
  range_fault_policy fault{};
  fault.response_delay = 20ms;
  auto payload         = rest_test_payload(1UL << 20);
  loopback_range_server server(payload, fault);
  auto cfg               = rest_contract_config();
  cfg.max_connections    = 2;
  cfg.footer_probe_bytes = 4096;
  auto ioctx             = make_ioctx(server.endpoint(), cfg, 2);
  rest_shutdown_guard cleanup{*ioctx};
  auto object = ioctx->open_io_object("s3://fair-bucket/object.bin", payload.size());
  std::vector<std::string> paths;
  for (int i = 0; i < 100; ++i) {
    paths.push_back("s3://fair-bucket/f" + std::to_string(i) + ".parquet");
  }
  std::atomic<std::size_t> delivered{0};
  auto const batch_started = std::chrono::steady_clock::now();
  std::thread batch([&] {
    ioctx->resolve_footer_objects(paths, [&](rest::footer_resolve_result) { ++delivered; });
  });
  std::this_thread::sleep_for(100ms);  // the batch is under way
  std::chrono::milliseconds worst{0};
  for (int i = 0; i < 6; ++i) {
    auto const t0 = std::chrono::steady_clock::now();
    std::vector<std::uint8_t> out(4096);
    std::vector<slice> one{slice{0, out.size(), out.data()}};
    REQUIRE(std::move(ioctx->host_readv_async_io(*object, one)).get(30s) == out.size());
    worst = std::max(
      worst,
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0));
  }
  batch.join();
  auto const batch_time = std::chrono::steady_clock::now() - batch_started;
  CHECK(delivered == paths.size());
  CAPTURE(worst.count());
  CHECK(worst < 400ms);
  CHECK(batch_time > 600ms);  // the scenario really overlapped a long batch
}

TEST_CASE("rest control-plane calls from a reactor thread do not deadlock", "[rest][control]")
{
  using namespace std::chrono_literals;
  if (cucascade::test::run_in_child("CUCASCADE_REST_CONTROL_ON_WORKER_CHILD", 30s)) { return; }
  // A completion callback (here: the worker's GET-finished event) opens another object, which
  // needs a HEAD.  Posted to its own worker it would wait on itself.
  auto payload = rest_test_payload(4096);
  loopback_range_server server(payload);
  auto cfg = rest_contract_config();
  std::shared_ptr<probed_ioctx> ioctx;
  std::mutex mutex;
  std::optional<std::size_t> nested_size;
  std::string nested_error;
  rest::testing::reactor_events events;
  events.on_get_finished = [&](std::string_view key, std::size_t, unsigned, std::uint64_t) {
    if (key != "outer.bin") { return; }
    try {
      auto inner = ioctx->open_io_object("s3://worker-bucket/inner.bin");
      std::lock_guard lock{mutex};
      nested_size = inner->size();
    } catch (std::exception const& e) {
      std::lock_guard lock{mutex};
      nested_error = e.what();
    }
  };
  events_guard const guard{&events};
  ioctx      = make_ioctx(server.endpoint(), cfg);
  auto outer = ioctx->open_io_object("s3://worker-bucket/outer.bin", payload.size());
  require_rest_bytes_equal(read_range(*ioctx, *outer, 0, 100), {payload.data(), 100});
  std::lock_guard lock{mutex};
  INFO(nested_error);
  REQUIRE(nested_size.has_value());
  CHECK(*nested_size == payload.size());
  ioctx->shutdown();
}

TEST_CASE("rest in-flight accounting survives a failing transport submit", "[rest][control]")
{
  // A data GET whose hand-off to the transport throws fails, and leaves nothing counted in
  // flight: the peak stays the one GET that did run.
  auto payload = rest_test_payload(4096);
  loopback_range_server server(payload);
  auto cfg               = rest_contract_config();
  cfg.max_connections    = 1;
  cfg.max_retry_attempts = 1;
  std::atomic<int> throws{3};
  rest::testing::reactor_events events;
  events.before_get_submit = [&](std::string_view) {
    if (throws.fetch_sub(1) > 0) { throw std::runtime_error("injected submit failure"); }
  };
  events_guard const guard{&events};
  auto ioctx  = make_ioctx(server.endpoint(), cfg);
  auto object = ioctx->open_io_object("s3://submit-bucket/object.bin", payload.size());
  std::vector<std::uint8_t> out(100);
  for (int i = 0; i < 3; ++i) {
    CHECK_THROWS_WITH(ioctx->host_read(*object, 0, out.size(), out.data()),
                      Catch::Matchers::ContainsSubstring("injected submit failure"));
  }
  CHECK(ioctx->host_read(*object, 0, out.size(), out.data()) == out.size());
  auto const stats = wait_for_stats(*ioctx, [](transport_stats const& s) { return s.gets >= 1; });
  CHECK(stats.max_inflight == 1);
  ioctx->shutdown();
}

// ---- time bounds -------------------------------------------------------------------------------

TEST_CASE("rest control-plane calls are bounded by request_timeout_s", "[rest][timeout]")
{
  range_fault_policy fault{};
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["slow.bin"].heads = {scripted_response{.delay = 3000ms}};
  loopback_range_server server(rest_test_payload(4096), fault, {}, scripts);
  auto cfg               = rest_contract_config();
  cfg.request_timeout_s  = 1;
  cfg.max_retry_attempts = 1;
  auto ioctx             = make_ioctx(server.endpoint(), cfg);
  rest_shutdown_guard cleanup{*ioctx};
  auto const started = std::chrono::steady_clock::now();
  CHECK_THROWS_WITH(ioctx->open_io_object("s3://timeout-bucket/slow.bin"),
                    Catch::Matchers::ContainsSubstring("timed out", Catch::CaseSensitive::No) ||
                      Catch::Matchers::ContainsSubstring("exhausted"));
  auto const elapsed = std::chrono::steady_clock::now() - started;
  CHECK(elapsed >= 900ms);
  CHECK(elapsed < 2500ms);
}

// ---- configuration -----------------------------------------------------------------------------

TEST_CASE("rest config validates every field", "[rest][config]")
{
  CHECK_NOTHROW(rest::config{}.validate());
  auto bad = [](auto mutate) {
    rest::config c{};
    mutate(c);
    return c;
  };
  CHECK_THROWS_AS(bad([](rest::config& c) { c.max_connections = 0; }).validate(),
                  std::invalid_argument);
  CHECK_THROWS_AS(bad([](rest::config& c) { c.max_connections = 1024; }).validate(),
                  std::invalid_argument);  // queue_depth 1024 < 2 x 1024
  CHECK_THROWS_AS(bad([](rest::config& c) { c.min_segment_bytes = 0; }).validate(),
                  std::invalid_argument);
  CHECK_THROWS_AS(bad([](rest::config& c) {
                    c.min_segment_bytes = 8UL << 20;
                    c.max_segment_bytes = 4UL << 20;
                  }).validate(),
                  std::invalid_argument);
  CHECK_THROWS_AS(bad([](rest::config& c) { c.receive_buffer_bytes = 1024; }).validate(),
                  std::invalid_argument);
  CHECK_THROWS_AS(bad([](rest::config& c) { c.max_recv_iovecs = 0; }).validate(),
                  std::invalid_argument);
  CHECK_THROWS_AS(bad([](rest::config& c) { c.defer_taskrun = true; }).validate(),
                  std::invalid_argument);  // needs wait_mode uring
  CHECK_NOTHROW(bad([](rest::config& c) {
                  c.defer_taskrun = true;
                  c.wait_mode     = rest::reactor_wait_mode::uring;
                }).validate());
  CHECK_THROWS_AS(bad([](rest::config& c) {
                    c.defer_taskrun = true;
                    c.sqpoll        = true;
                    c.wait_mode     = rest::reactor_wait_mode::uring;
                  }).validate(),
                  std::invalid_argument);
  CHECK_THROWS_AS(bad([](rest::config& c) { c.dns_max_addresses = 0; }).validate(),
                  std::invalid_argument);
  CHECK_THROWS_AS(bad([](rest::config& c) { c.connect_timeout = 0ms; }).validate(),
                  std::invalid_argument);
  CHECK_THROWS_AS(bad([](rest::config& c) { c.keepalive_idle_s = 0; }).validate(),
                  std::invalid_argument);
  CHECK_THROWS_AS(bad([](rest::config& c) { c.request_timeout_s = -1; }).validate(),
                  std::invalid_argument);
  CHECK_THROWS_AS(bad([](rest::config& c) { c.max_retry_attempts = 0; }).validate(),
                  std::invalid_argument);
  CHECK_THROWS_AS(bad([](rest::config& c) { c.max_auth_retry_attempts = 0; }).validate(),
                  std::invalid_argument);
  CHECK_THROWS_AS(bad([](rest::config& c) {
                    c.tls.min_version = cucascade::io::tls::version_bound::tls1_3;
                    c.tls.max_version = cucascade::io::tls::version_bound::tls1_2;
                  }).validate(),
                  std::invalid_argument);
  // The deprecated aliases fill the TLS policy.
  rest::config aliases{};
  aliases.ca_bundle_path = "/etc/ssl/custom.pem";
  aliases.tls_verify     = false;
  auto const effective   = aliases.effective_tls();
  CHECK(effective.ca_file == "/etc/ssl/custom.pem");
  CHECK_FALSE(effective.verify);
  // A reactor refuses an invalid config.
  auto invalid            = rest_contract_config();
  invalid.max_connections = 0;
  CHECK_THROWS_AS(rest_reactor(std::make_shared<rest_reactor::reactor_context>(
                    invalid, std::make_shared<list_capable_mock_authorizer>("http://x"))),
                  std::invalid_argument);
}

TEST_CASE("rest reactor serves every wait mode and ring setup", "[rest][config]")
{
  auto const mode = GENERATE(0, 1, 2);
  CAPTURE(mode);
  auto payload = rest_test_payload(2UL << 20);
  loopback_range_server server(payload);
  auto cfg = rest_contract_config();
  if (mode >= 1) { cfg.wait_mode = rest::reactor_wait_mode::uring; }
  if (mode == 2) {
    cfg.single_issuer = true;
    cfg.defer_taskrun = true;
  }
  cfg.recv_waitall = mode == 1;
  auto ioctx       = make_ioctx(server.endpoint(), cfg);
  rest_shutdown_guard cleanup{*ioctx};
  auto object = ioctx->open_io_object("s3://mode-bucket/object.bin");
  require_rest_bytes_equal(read_range(*ioctx, *object, 0, payload.size()), payload);
  require_rest_bytes_equal(read_range(*ioctx, *object, 777, 9999), {payload.data() + 777, 9999});
}
