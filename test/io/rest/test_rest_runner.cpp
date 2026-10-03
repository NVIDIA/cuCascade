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

// Runner-model tests of the REST backend (rest_engine) against the loopback
// HTTP server: run / run_for / run_until, start/shutdown cycles, several
// runners sharing one queue, ranged-GET semantics (splitting, retries,
// footer stash, staged device reads) and warm-up.

#include "loopback_range_server.hpp"

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/rest/authorizer.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stop_token>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {

using cucascade::io::open_hint;
using cucascade::io::rest::authorized_request;
using cucascade::io::rest::config;
using cucascade::io::rest::object_ref;
using cucascade::io::rest::request_authorizer;
using cucascade::io::rest::request_method;
using cucascade::io::rest::rest_ioctx;
using cucascade::io::rest::rest_reactor;
using cucascade::test::loopback_range_server;
using cucascade::test::range_fault_policy;
using namespace std::chrono_literals;

constexpr std::string_view object_uri{"s3://bucket/object.bin"};

/// Routes every object request to the loopback object and every bucket LIST
/// (the warm-up request) to the loopback LIST endpoint.
class loopback_authorizer final : public request_authorizer {
 public:
  explicit loopback_authorizer(std::string endpoint) : _endpoint(std::move(endpoint)) {}

  authorized_request authorize(object_ref const& /*obj*/,
                               request_method /*method*/,
                               std::chrono::seconds /*timeout*/) override
  {
    return authorized_request{_endpoint + "/bucket/object.bin", {}};
  }

  authorized_request authorize_list(std::string_view bucket,
                                    std::string_view canonical_query,
                                    std::chrono::seconds /*timeout*/) override
  {
    return authorized_request{
      _endpoint + "/" + std::string{bucket} + "?" + std::string{canonical_query}, {}};
  }

 private:
  std::string _endpoint;
};

std::vector<std::uint8_t> make_payload(std::size_t size)
{
  std::vector<std::uint8_t> bytes(size);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>((i * 131U + (i >> 12) * 7U + 3U) & 0xffU);
  }
  return bytes;
}

config test_config()
{
  config cfg{};
  cfg.request_timeout_s       = 10;
  cfg.tls_verify              = false;
  cfg.max_connections         = 8;
  cfg.max_retry_attempts      = 4;
  cfg.max_auth_retry_attempts = 2;
  cfg.retry_backoff_base      = 1ms;
  cfg.retry_jitter            = 0ms;
  cfg.honor_retry_after       = false;
  cfg.footer_probe_bytes      = 4096;
  return cfg;
}

std::shared_ptr<rest_ioctx> make_ioctx(
  loopback_range_server const& server,
  std::size_t n_runner_threads,
  config cfg                                                  = test_config(),
  cucascade::memory::fixed_size_host_memory_resource* host_mr = nullptr)
{
  auto context = std::make_shared<rest_reactor::reactor_context>(
    cfg, std::make_shared<loopback_authorizer>(server.endpoint()), host_mr);
  return std::make_shared<rest_ioctx>(n_runner_threads, std::move(context));
}

std::shared_ptr<cucascade::io::io_object> open_object(rest_ioctx& ioctx, std::size_t size)
{
  return ioctx.open_io_object(std::string{object_uri}, static_cast<std::uint64_t>(size));
}

bool wait_for(std::function<bool()> const& predicate, std::chrono::milliseconds limit = 5s)
{
  auto const until = std::chrono::steady_clock::now() + limit;
  while (std::chrono::steady_clock::now() < until) {
    if (predicate()) return true;
    std::this_thread::sleep_for(1ms);
  }
  return predicate();
}

bool is_canceled(std::system_error const& error)
{
  return error.code() == std::errc::operation_canceled;
}

bool range_matches(std::vector<std::uint8_t> const& payload,
                   std::size_t offset,
                   std::vector<std::uint8_t> const& got)
{
  return std::equal(got.begin(), got.end(), payload.begin() + static_cast<std::ptrdiff_t>(offset));
}

/// Read [offset, offset + size) through the async host path and verify it.
bool read_and_verify(rest_ioctx& ioctx,
                     cucascade::io::io_object const& object,
                     std::vector<std::uint8_t> const& payload,
                     std::size_t offset,
                     std::size_t size)
{
  std::vector<std::uint8_t> buffer(size);
  auto const got = ioctx.host_read_async(object, offset, size, buffer.data()).get();
  return got == size && range_matches(payload, offset, buffer);
}

}  // namespace

TEST_CASE("rest runner: run_for without work returns after the deadline", "[rest][runner]")
{
  loopback_range_server server(make_payload(4096));
  auto ioctx = make_ioctx(server, 0);

  auto const t0     = std::chrono::steady_clock::now();
  auto const served = ioctx->run_for(100ms);
  auto const took   = std::chrono::steady_clock::now() - t0;
  CHECK(served == 0);
  CHECK(took >= 95ms);
  CHECK(took < 2s);
  CHECK(ioctx->active_runners() == 0);
  CHECK(server.get_count() == 0);
}

TEST_CASE("rest runner: run_until serves work submitted while it runs", "[rest][runner]")
{
  auto const payload = make_payload(2UL << 20);
  loopback_range_server server(payload);
  auto ioctx  = make_ioctx(server, 0);
  auto object = open_object(*ioctx, payload.size());

  std::atomic<std::size_t> served{0};
  std::jthread runner(
    [&] { served.store(ioctx->run_until(std::chrono::steady_clock::now() + 1500ms)); });
  REQUIRE(wait_for([&] { return ioctx->active_runners() == 1; }));

  // Latency-class (small) and read-class (large) requests, concurrently.
  std::vector<std::vector<std::uint8_t>> buffers;
  std::vector<cucascade::exec::semi_future<std::size_t>> futures;
  std::vector<std::pair<std::size_t, std::size_t>> ranges{
    {0, 4096}, {12345, 70000}, {300000, 1UL << 20}, {(2UL << 20) - 777, 777}, {5, 1}};
  for (auto const& [offset, size] : ranges) {
    buffers.emplace_back(size);
    futures.push_back(ioctx->host_read_async(*object, offset, size, buffers.back().data()));
  }
  for (std::size_t i = 0; i < futures.size(); ++i) {
    CHECK(std::move(futures[i]).get() == ranges[i].second);
    CHECK(range_matches(payload, ranges[i].first, buffers[i]));
  }
  runner.join();
  CHECK(served.load() == ranges.size());
  CHECK(ioctx->active_runners() == 0);
}

TEST_CASE("rest runner: external runner is stopped and joined by shutdown", "[rest][runner]")
{
  auto const payload = make_payload(64UL << 10);
  loopback_range_server server(payload);
  auto ioctx  = make_ioctx(server, 0);
  auto object = open_object(*ioctx, payload.size());

  std::atomic<std::size_t> served{0};
  std::atomic<bool> returned{false};
  std::jthread runner([&] {
    served.store(ioctx->run(std::stop_token{}));
    returned.store(true);
  });
  REQUIRE(wait_for([&] { return ioctx->active_runners() == 1; }));

  for (std::size_t i = 0; i < 8; ++i) {
    CHECK(read_and_verify(*ioctx, *object, payload, i * 8000, 4000));
  }
  // The synchronous host path goes through the runner as well.
  std::vector<std::uint8_t> buffer(10000);
  CHECK(ioctx->host_read(*object, 1000, buffer.size(), buffer.data()) == buffer.size());
  CHECK(range_matches(payload, 1000, buffer));

  ioctx->shutdown();
  // shutdown() returns once the runner unregistered; run() returns right after.
  CHECK(wait_for([&] { return returned.load(); }));
  CHECK(served.load() == 9);
  CHECK(ioctx->active_runners() == 0);

  // Admission is closed after shutdown.
  CHECK_THROWS_MATCHES(ioctx->host_read(*object, 0, 16, buffer.data()),
                       std::system_error,
                       Catch::Matchers::Predicate<std::system_error>(is_canceled));
}

TEST_CASE("rest runner: request before start fails with operation_canceled", "[rest][runner]")
{
  auto const payload = make_payload(4096);
  loopback_range_server server(payload);
  auto ioctx  = make_ioctx(server, 1);
  auto object = open_object(*ioctx, payload.size());

  std::vector<std::uint8_t> buffer(128);
  CHECK_THROWS_MATCHES(ioctx->host_read(*object, 0, buffer.size(), buffer.data()),
                       std::system_error,
                       Catch::Matchers::Predicate<std::system_error>(is_canceled));
  CHECK_THROWS_MATCHES(ioctx->host_read_async(*object, 0, buffer.size(), buffer.data()).get(),
                       std::system_error,
                       Catch::Matchers::Predicate<std::system_error>(is_canceled));
  CHECK(server.get_count() == 0);
}

TEST_CASE("rest runner: start, shutdown and start again", "[rest][runner]")
{
  auto const payload = make_payload(256UL << 10);
  loopback_range_server server(payload);
  auto ioctx  = make_ioctx(server, 2);
  auto object = open_object(*ioctx, payload.size());

  for (int round = 0; round < 3; ++round) {
    ioctx->start();
    CHECK(wait_for([&] { return ioctx->active_runners() == 2; }));
    CHECK(read_and_verify(*ioctx, *object, payload, 1024, 100000));
    CHECK(read_and_verify(*ioctx, *object, payload, 0, payload.size()));
    ioctx->shutdown();
    CHECK(ioctx->active_runners() == 0);
  }
}

TEST_CASE("rest runner: two external runners share the queue", "[rest][runner]")
{
  auto const payload = make_payload(1UL << 20);
  range_fault_policy fault{};
  fault.response_delay = 20ms;
  loopback_range_server server(payload, fault);
  auto ioctx  = make_ioctx(server, 0);
  auto object = open_object(*ioctx, payload.size());

  std::atomic<std::size_t> served_a{0};
  std::atomic<std::size_t> served_b{0};
  std::jthread a([&](std::stop_token stop) { served_a.store(ioctx->run(stop)); });
  std::jthread b([&](std::stop_token stop) { served_b.store(ioctx->run(stop)); });
  REQUIRE(wait_for([&] { return ioctx->active_runners() == 2; }));
  CHECK(ioctx->stats().active_runners == 2);

  constexpr std::size_t n_requests = 64;
  constexpr std::size_t size       = 4096;
  std::vector<std::vector<std::uint8_t>> buffers(n_requests, std::vector<std::uint8_t>(size));
  std::vector<cucascade::exec::semi_future<std::size_t>> futures;
  for (std::size_t i = 0; i < n_requests; ++i) {
    futures.push_back(ioctx->host_read_async(*object, i * 15000, size, buffers[i].data()));
  }
  for (std::size_t i = 0; i < n_requests; ++i) {
    CHECK(std::move(futures[i]).get() == size);
    CHECK(range_matches(payload, i * 15000, buffers[i]));
  }

  a.request_stop();
  b.request_stop();
  a.join();
  b.join();
  CHECK(served_a.load() > 0);
  CHECK(served_b.load() > 0);
  CHECK(served_a.load() + served_b.load() == n_requests);
  CHECK(ioctx->active_runners() == 0);
}

TEST_CASE("rest runner: one runner keeps several groups in flight", "[rest][runner]")
{
  auto const payload = make_payload(8UL << 20);
  range_fault_policy fault{};
  // GET responses are held until four GETs arrived: only an engine that
  // expands several grouped requests concurrently gets past this.
  fault.get_response_barrier = 4;
  loopback_range_server server(payload, fault);
  auto ioctx  = make_ioctx(server, 1);
  auto object = open_object(*ioctx, payload.size());
  ioctx->start();

  constexpr std::size_t size = 1UL << 20;  // read class (> latency limit)
  std::vector<std::vector<std::uint8_t>> buffers(4, std::vector<std::uint8_t>(size));
  std::vector<cucascade::exec::semi_future<std::size_t>> futures;
  for (std::size_t i = 0; i < 4; ++i) {
    futures.push_back(ioctx->host_read_async(*object, i * 2 * size, size, buffers[i].data()));
  }
  for (std::size_t i = 0; i < 4; ++i) {
    CHECK(std::move(futures[i]).get() == size);
    CHECK(range_matches(payload, i * 2 * size, buffers[i]));
  }
  ioctx->shutdown();
}

TEST_CASE("rest runner: one runner keeps more single-GET reads in flight than its group limit",
          "[rest][runner]")
{
  // Default limits: 4 bulk + 2 latency expanding groups per runner.  A read
  // group whose only GET is on a connection must not hold one of those, so a
  // single runner reaches the barrier with 12 single-GET requests in flight
  // (bounded by its 16 connections, not by the group limit).
  constexpr std::size_t n_requests = 12;
  static_assert(n_requests > 4 + 2);
  auto const payload = make_payload(32UL << 20);
  range_fault_policy fault{};
  fault.get_response_barrier = n_requests;
  loopback_range_server server(payload, fault);
  auto cfg            = test_config();
  cfg.max_connections = 16;
  auto ioctx          = make_ioctx(server, 1, cfg);
  auto object         = open_object(*ioctx, payload.size());
  ioctx->start();

  constexpr std::size_t size = 1UL << 20;  // read class, one GET
  std::vector<std::vector<std::uint8_t>> buffers(n_requests, std::vector<std::uint8_t>(size));
  std::vector<cucascade::exec::semi_future<std::size_t>> futures;
  for (std::size_t i = 0; i < n_requests; ++i) {
    futures.push_back(ioctx->host_read_async(*object, i * 2 * size, size, buffers[i].data()));
  }
  // Without the fix the runner stalls at 6 GETs until the barrier times out.
  CHECK(wait_for([&] { return server.get_count() >= n_requests; }));
  for (std::size_t i = 0; i < n_requests; ++i) {
    CHECK(std::move(futures[i]).get(30s) == size);
    CHECK(range_matches(payload, i * 2 * size, buffers[i]));
  }
  ioctx->shutdown();
}

TEST_CASE("rest runner: ranged GET semantics are preserved", "[rest][runner]")
{
  SECTION("a large read is split into several ranged GETs")
  {
    auto const payload = make_payload(24UL << 20);
    loopback_range_server server(payload);
    auto ioctx  = make_ioctx(server, 1);
    auto object = open_object(*ioctx, payload.size());
    ioctx->start();
    CHECK(read_and_verify(*ioctx, *object, payload, 333, (20UL << 20) + 17));
    CHECK(server.get_count() >= 2);
    ioctx->shutdown();
  }

  SECTION("transient 503s are retried with a fresh authorization")
  {
    auto const payload = make_payload(64UL << 10);
    range_fault_policy fault{};
    fault.fail_first_gets = 2;
    loopback_range_server server(payload, fault);
    auto ioctx  = make_ioctx(server, 1);
    auto object = open_object(*ioctx, payload.size());
    ioctx->start();
    CHECK(read_and_verify(*ioctx, *object, payload, 100, 5000));
    CHECK(server.get_count() == 3);
    ioctx->shutdown();
  }

  SECTION("exhausted retries fail the read")
  {
    auto const payload = make_payload(64UL << 10);
    range_fault_policy fault{};
    fault.fail_all_gets = true;
    loopback_range_server server(payload, fault);
    auto ioctx  = make_ioctx(server, 1);
    auto object = open_object(*ioctx, payload.size());
    ioctx->start();
    std::vector<std::uint8_t> buffer(100);
    CHECK_THROWS_AS(ioctx->host_read_async(*object, 0, buffer.size(), buffer.data()).get(),
                    std::runtime_error);
    CHECK(server.get_count() == test_config().max_retry_attempts);
    // The runner keeps serving after a failed request.
    ioctx->shutdown();
  }

  SECTION("a server ignoring Range fails the read")
  {
    auto const payload = make_payload(64UL << 10);
    range_fault_policy fault{};
    fault.ignore_range_with_200 = true;
    loopback_range_server server(payload, fault);
    auto ioctx  = make_ioctx(server, 1);
    auto object = open_object(*ioctx, payload.size());
    ioctx->start();
    std::vector<std::uint8_t> buffer(100);
    CHECK_THROWS_AS(ioctx->host_read_async(*object, 10, buffer.size(), buffer.data()).get(),
                    std::runtime_error);
    ioctx->shutdown();
  }

  SECTION("a malformed Content-Range fails the read")
  {
    auto const payload = make_payload(64UL << 10);
    range_fault_policy fault{};
    fault.malformed_content_range = true;
    loopback_range_server server(payload, fault);
    auto ioctx  = make_ioctx(server, 1);
    auto object = open_object(*ioctx, payload.size());
    ioctx->start();
    std::vector<std::uint8_t> buffer(100);
    CHECK_THROWS_AS(ioctx->host_read_async(*object, 10, buffer.size(), buffer.data()).get(),
                    std::runtime_error);
    ioctx->shutdown();
  }

  SECTION("reads inside the footer stash are served without a GET")
  {
    auto const payload = make_payload(64UL << 10);
    loopback_range_server server(payload);
    auto ioctx  = make_ioctx(server, 1);
    auto object = ioctx->open_io_object(std::string{object_uri}, open_hint::parquet_footer_probe);
    REQUIRE(object->size() == payload.size());
    auto const gets_after_probe = server.get_count();
    ioctx->start();
    CHECK(read_and_verify(*ioctx, *object, payload, payload.size() - 1000, 1000));
    CHECK(server.get_count() == gets_after_probe);
    CHECK(read_and_verify(*ioctx, *object, payload, 0, 1000));
    CHECK(server.get_count() == gets_after_probe + 1);
    ioctx->shutdown();
  }
}

TEST_CASE("rest runner: shutdown cancels in-flight transfers", "[rest][runner]")
{
  auto const payload = make_payload(64UL << 10);
  range_fault_policy fault{};
  fault.response_delay = 3s;
  loopback_range_server server(payload, fault);
  auto ioctx  = make_ioctx(server, 1);
  auto object = open_object(*ioctx, payload.size());
  ioctx->start();

  std::vector<std::uint8_t> buffer(1000);
  auto future = ioctx->host_read_async(*object, 0, buffer.size(), buffer.data());
  REQUIRE(wait_for([&] { return server.get_count() >= 1; }));

  auto const t0 = std::chrono::steady_clock::now();
  ioctx->shutdown();
  CHECK(std::chrono::steady_clock::now() - t0 < 2s);
  CHECK_THROWS_MATCHES(std::move(future).get(),
                       std::system_error,
                       Catch::Matchers::Predicate<std::system_error>(is_canceled));
}

TEST_CASE("rest runner: a retiring runner drains its transfers and leaves the queue",
          "[rest][runner]")
{
  auto const payload = make_payload(16UL << 20);
  range_fault_policy fault{};
  fault.response_delay = 300ms;
  loopback_range_server server(payload, fault);
  // Two connections: the runner holds at most 2 GETs in flight plus
  // max_active_groups groups with GETs still to dispatch.
  auto cfg            = test_config();
  cfg.max_connections = 2;
  auto ioctx          = make_ioctx(server, 1, cfg);
  auto object         = open_object(*ioctx, payload.size());

  std::atomic<std::size_t> served{0};
  std::jthread runner([&] { served.store(ioctx->run_for(100ms)); });
  REQUIRE(wait_for([&] { return ioctx->active_runners() == 1; }));

  // More read-class groups than one runner holds at once.
  constexpr std::size_t n_requests = 8;
  constexpr std::size_t size       = 1UL << 20;
  std::vector<std::vector<std::uint8_t>> buffers(n_requests, std::vector<std::uint8_t>(size));
  std::vector<cucascade::exec::semi_future<std::size_t>> futures;
  for (std::size_t i = 0; i < n_requests; ++i) {
    futures.push_back(ioctx->host_read_async(*object, i * 2 * size, size, buffers[i].data()));
  }
  runner.join();
  // The runner finished what it had started (past its deadline) and left the
  // rest queued for others.
  CHECK(served.load() > 0);
  CHECK(served.load() < n_requests);

  ioctx->start();
  for (std::size_t i = 0; i < n_requests; ++i) {
    CHECK(std::move(futures[i]).get() == size);
    CHECK(range_matches(payload, i * 2 * size, buffers[i]));
  }
  ioctx->shutdown();
}

TEST_CASE("rest runner: warm-up primes every runner's connection pool", "[rest][runner]")
{
  auto const payload = make_payload(4096);
  loopback_range_server server(payload);
  auto cfg            = test_config();
  cfg.max_connections = 4;
  auto ioctx          = make_ioctx(server, 2, cfg);

  // Requested before any runner exists: honored by the engines start() builds.
  ioctx->warmup("s3://bucket");
  CHECK(server.list_count() == 0);
  ioctx->start();
  CHECK(wait_for([&] { return server.list_count() >= 2 * cfg.max_connections; }));
  // Let the engines retire the warm-up transfers: a request arriving while a
  // round is still in flight is coalesced into it (as before the runner model).
  std::this_thread::sleep_for(300ms);

  // Requested while running (another bucket, so the rate limiter lets it
  // through): every runner is woken and primes again.
  ioctx->warmup("s3://other-bucket");
  CHECK(wait_for([&] { return server.list_count() >= 4 * cfg.max_connections; }));
  ioctx->shutdown();
}

TEST_CASE("rest runner: staged device reads copy through pinned staging", "[rest][runner][gpu]")
{
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { SKIP("no CUDA device"); }

  constexpr std::size_t block_size = 64UL << 10;
  rmm::mr::pinned_host_memory_resource pinned_mr;
  cucascade::memory::fixed_size_host_memory_resource host_mr{
    0, pinned_mr, 64UL << 20, 64UL << 20, block_size, 16, 1};

  auto const payload = make_payload(6UL << 20);
  loopback_range_server server(payload);
  auto ioctx  = make_ioctx(server, 1, test_config(), &host_mr);
  auto object = open_object(*ioctx, payload.size());
  ioctx->start();

  constexpr std::size_t offset = 4097;
  constexpr std::size_t size   = (5UL << 20) + 3;
  void* device                 = nullptr;
  REQUIRE(cudaMalloc(&device, size) == cudaSuccess);
  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  auto const got =
    ioctx
      ->device_read_async(
        *object, offset, size, static_cast<std::uint8_t*>(device), ::cuda::stream_ref{stream})
      .get();
  CHECK(got == size);
  std::vector<std::uint8_t> host(size);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  REQUIRE(cudaMemcpy(host.data(), device, size, cudaMemcpyDeviceToHost) == cudaSuccess);
  CHECK(range_matches(payload, offset, host));

  ioctx->shutdown();
  CHECK(cudaStreamDestroy(stream) == cudaSuccess);
  CHECK(cudaFree(device) == cudaSuccess);
}
