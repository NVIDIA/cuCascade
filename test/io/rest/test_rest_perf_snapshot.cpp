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
#include "mock_authorizer.hpp"

#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuda_runtime.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using cucascade::io::open_hint;
using cucascade::io::rest::config;
using cucascade::io::rest::mock_authorizer;
using cucascade::io::rest::rest_ioctx;
using cucascade::io::rest::rest_perf_snapshot;
using cucascade::io::rest::rest_reactor;
using cucascade::test::list_capable_mock_authorizer;
using cucascade::test::listed_object;
using cucascade::test::loopback_range_server;
using cucascade::test::range_fault_policy;
using namespace std::chrono_literals;

std::vector<std::uint8_t> deterministic_payload(std::size_t size)
{
  std::vector<std::uint8_t> bytes(size);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>((i * 131U + 17U) & 0xffU);
  }
  return bytes;
}

void require_bytes_equal(std::span<const std::uint8_t> actual,
                         std::span<const std::uint8_t> expected)
{
  REQUIRE(actual.size() == expected.size());
  CHECK(std::equal(actual.begin(), actual.end(), expected.begin(), expected.end()));
}

config test_config(bool instrumentation = true)
{
  config cfg{};
  cfg.request_timeout_s       = 5;
  cfg.tls_verify              = false;
  cfg.max_connections         = 2;
  cfg.chunk_size              = 64 * 1024;
  cfg.max_read_split          = 1;
  cfg.max_retry_attempts      = 3;
  cfg.max_auth_retry_attempts = 2;
  cfg.retry_backoff_base      = 1ms;
  cfg.retry_jitter            = 0ms;
  cfg.honor_retry_after       = false;
  cfg.perf_instrumentation    = instrumentation;
  cfg.footer_probe_bytes      = 512;
  return cfg;
}

struct direct_ioctx {
  std::shared_ptr<mock_authorizer> authorizer;
  std::shared_ptr<rest_ioctx> ioctx;
};

direct_ioctx make_ioctx(loopback_range_server const& server,
                        config cfg,
                        std::size_t reactors                                        = 1,
                        cucascade::memory::fixed_size_host_memory_resource* host_mr = nullptr)
{
  auto authorizer = std::make_shared<mock_authorizer>(
    cucascade::io::rest::authorized_request{server.endpoint() + "/bucket/object.bin", {}});
  auto context = std::make_shared<rest_reactor::reactor_context>(cfg, authorizer, host_mr);
  auto ioctx   = std::make_shared<rest_ioctx>(reactors, std::move(context));
  ioctx->start();
  return {std::move(authorizer), std::move(ioctx)};
}

std::unique_ptr<rest_reactor> make_reactor(
  loopback_range_server const& server,
  config cfg,
  std::shared_ptr<mock_authorizer>* authorizer_out = nullptr)
{
  auto authorizer = std::make_shared<mock_authorizer>(
    cucascade::io::rest::authorized_request{server.endpoint() + "/bucket/object.bin", {}});
  auto context = std::make_shared<rest_reactor::reactor_context>(cfg, authorizer, nullptr);
  if (authorizer_out != nullptr) { *authorizer_out = authorizer; }
  return std::make_unique<rest_reactor>(std::move(context), "rest-perf-test");
}

std::shared_ptr<cucascade::io::io_object> known_object(rest_ioctx& ioctx, std::size_t size)
{
  return ioctx.open_io_object("s3://bucket/object.bin", static_cast<std::uint64_t>(size));
}

void check_micro_counters_zero(rest_perf_snapshot const& snapshot)
{
  CHECK(snapshot.chunk_get_ns_total == 0);
  CHECK(snapshot.chunk_get_count == 0);
  CHECK(snapshot.chunk_get_ns_max == 0);
  CHECK(snapshot.queue_wait_ns_total == 0);
  CHECK(snapshot.queue_wait_count == 0);
  CHECK(snapshot.ttfb_ns == 0);
  CHECK(snapshot.h2d_observed_ns_total == 0);
  CHECK(snapshot.h2d_observed_count == 0);
  CHECK(snapshot.h2d_observed_ns_max == 0);
  CHECK(snapshot.blocking_host_get_count == 0);
  CHECK(snapshot.blocking_host_get_wall_ns_total == 0);
  CHECK(snapshot.blocking_host_get_wall_ns_max == 0);
}

class device_allocation {
 public:
  explicit device_allocation(std::size_t size)
  {
    if (cudaMalloc(reinterpret_cast<void**>(&_data), size) != cudaSuccess) {
      throw std::runtime_error("cudaMalloc failed");
    }
  }
  ~device_allocation()
  {
    if (_data != nullptr) { (void)cudaFree(_data); }
  }

  device_allocation(device_allocation const&)            = delete;
  device_allocation& operator=(device_allocation const&) = delete;

  [[nodiscard]] std::uint8_t* data() noexcept { return _data; }

 private:
  std::uint8_t* _data{nullptr};
};

struct staging_resource {
  static constexpr std::size_t block_size = 64 * 1024;
  static constexpr std::size_t capacity   = 4 * 1024 * 1024;

  rmm::mr::pinned_host_memory_resource pinned;
  cucascade::memory::fixed_size_host_memory_resource blocks{
    0, pinned, capacity, capacity, block_size, 4, 1};
};

}  // namespace

TEST_CASE("default perf snapshot is zero", "[rest][perf]")
{
  rest_perf_snapshot snapshot{};
  check_micro_counters_zero(snapshot);
  CHECK(snapshot.retries_total == 0);
  CHECK(snapshot.terminal_failures_total == 0);
  CHECK(snapshot.device_stream_sync_total == 0);
  CHECK(snapshot.payload_bytes_read_total == 0);
}

TEST_CASE("perf instrumentation defaults off", "[rest][perf]")
{
  CHECK_FALSE(config{}.perf_instrumentation);
}

TEST_CASE("perf snapshot readouts are noexcept", "[rest][perf]")
{
  STATIC_REQUIRE(noexcept(std::declval<rest_reactor const&>().perf_snapshot()));
  STATIC_REQUIRE(noexcept(std::declval<rest_ioctx const&>().perf_snapshot()));
}

TEST_CASE("ranged get feeds chunk counters", "[rest][perf]")
{
  auto payload = deterministic_payload(4096);
  loopback_range_server server(payload);
  auto fixture = make_ioctx(server, test_config());
  auto object  = known_object(*fixture.ioctx, payload.size());
  std::array<std::uint8_t, 257> out{};

  auto future = fixture.ioctx->host_read_async_io(*object, 17, out.size(), out.data());
  REQUIRE(std::move(future).get(5s) == out.size());
  require_bytes_equal(out, std::span<const std::uint8_t>(payload.data() + 17, out.size()));

  auto const snapshot = fixture.ioctx->perf_snapshot();
  CHECK(snapshot.chunk_get_count == 1);
  CHECK(snapshot.chunk_get_ns_total > 0);
  CHECK(snapshot.chunk_get_ns_max > 0);
  CHECK(snapshot.payload_bytes_read_total == out.size());
  CHECK(server.get_count() == 1);
}

TEST_CASE("head request feeds retry and terminal counters", "[rest][perf]")
{
  auto payload = deterministic_payload(4096);

  SECTION("clean head leaves data counters unchanged")
  {
    loopback_range_server server(payload);
    auto reactor = make_reactor(server, test_config());
    CHECK(reactor->head_object_size("bucket", "object.bin") == payload.size());
    auto const snapshot = reactor->perf_snapshot();
    CHECK(snapshot.chunk_get_count == 0);
    CHECK(snapshot.payload_bytes_read_total == 0);
    CHECK(snapshot.retries_total == 0);
    CHECK(snapshot.terminal_failures_total == 0);
    CHECK(server.head_count() == 1);
  }

  SECTION("transient head failure is retried")
  {
    range_fault_policy fault;
    fault.fail_first_heads = 1;
    loopback_range_server server(payload, fault);
    auto reactor = make_reactor(server, test_config(false));
    CHECK(reactor->head_object_size("bucket", "object.bin") == payload.size());
    auto const snapshot = reactor->perf_snapshot();
    CHECK(snapshot.retries_total == 1);
    CHECK(snapshot.terminal_failures_total == 0);
    CHECK(server.head_count() == 2);
  }
}

TEST_CASE("footer probe attributes as chunk get", "[rest][perf]")
{
  auto payload = deterministic_payload(4096);
  loopback_range_server server(payload);
  auto reactor = make_reactor(server, test_config());

  auto const probe = reactor->fetch_footer_suffix("bucket", "object.bin", 512);
  REQUIRE(probe.bytes != nullptr);
  CHECK(probe.object_size == payload.size());
  CHECK(probe.window_lo == payload.size() - 512);

  auto const snapshot = reactor->perf_snapshot();
  CHECK(snapshot.chunk_get_count == 1);
  CHECK(snapshot.blocking_host_get_count == 0);
  CHECK(snapshot.payload_bytes_read_total == 512);
  CHECK(server.get_count() == 1);
}

TEST_CASE("blocking host read is counted additively", "[rest][perf]")
{
  auto payload = deterministic_payload(4096);
  loopback_range_server server(payload);
  auto fixture = make_ioctx(server, test_config());
  auto object  = known_object(*fixture.ioctx, payload.size());
  std::array<std::uint8_t, 128> out{};

  REQUIRE(fixture.ioctx->host_read_io(*object, 99, out.size(), out.data()) == out.size());
  auto const snapshot = fixture.ioctx->perf_snapshot();
  CHECK(snapshot.chunk_get_count == 1);
  CHECK(snapshot.blocking_host_get_count == 1);
  CHECK(snapshot.blocking_host_get_wall_ns_total > 0);
  CHECK(snapshot.blocking_host_get_wall_ns_max > 0);
}

TEST_CASE("list bytes stay out of payload counters", "[rest][perf]")
{
  auto payload = deterministic_payload(128);
  loopback_range_server server(payload, {}, {listed_object{"prefix/a.parquet", 17}});
  auto authorizer = std::make_shared<list_capable_mock_authorizer>(server.endpoint());
  auto context =
    std::make_shared<rest_reactor::reactor_context>(test_config(), authorizer, nullptr);
  auto ioctx = std::make_shared<rest_ioctx>(1, std::move(context));

  auto const listed = ioctx->list_objects("bucket", "prefix/");
  REQUIRE(listed.size() == 1);
  CHECK(listed[0].key == "prefix/a.parquet");
  CHECK(listed[0].size == 17);
  auto const snapshot = ioctx->perf_snapshot();
  CHECK(snapshot.payload_bytes_read_total == 0);
  CHECK(snapshot.chunk_get_count == 0);
  CHECK(authorizer->list_calls() == 1);
  CHECK(server.list_count() == 1);
}

TEST_CASE("gate off keeps safety counters live", "[rest][perf]")
{
  auto payload = deterministic_payload(4096);

  SECTION("retry and payload counters remain active")
  {
    range_fault_policy fault;
    fault.fail_first_gets = 1;
    loopback_range_server server(payload, fault);
    auto fixture = make_ioctx(server, test_config(false));
    auto object  = known_object(*fixture.ioctx, payload.size());
    std::array<std::uint8_t, 128> out{};
    auto future = fixture.ioctx->host_read_async_io(*object, 0, out.size(), out.data());
    REQUIRE(std::move(future).get(5s) == out.size());

    auto const snapshot = fixture.ioctx->perf_snapshot();
    check_micro_counters_zero(snapshot);
    CHECK(snapshot.payload_bytes_read_total == out.size());
    CHECK(snapshot.retries_total == 1);
    CHECK(snapshot.terminal_failures_total == 0);
    CHECK(snapshot.device_stream_sync_total == 0);
  }

  SECTION("terminal counters remain active")
  {
    range_fault_policy fault;
    fault.fail_all_gets = true;
    fault.fail_status   = 404;
    loopback_range_server server(payload, fault);
    auto fixture = make_ioctx(server, test_config(false));
    auto object  = known_object(*fixture.ioctx, payload.size());
    std::array<std::uint8_t, 128> out{};
    auto future = fixture.ioctx->host_read_async_io(*object, 0, out.size(), out.data());
    CHECK_THROWS(std::move(future).get(5s));

    auto const snapshot = fixture.ioctx->perf_snapshot();
    check_micro_counters_zero(snapshot);
    CHECK(snapshot.retries_total == 0);
    CHECK(snapshot.terminal_failures_total == 1);
  }
}

TEST_CASE("gate on records micro timings", "[rest][perf]")
{
  auto payload = deterministic_payload(4096);
  loopback_range_server server(payload);
  auto fixture = make_ioctx(server, test_config());
  auto object  = known_object(*fixture.ioctx, payload.size());
  std::array<std::uint8_t, 128> out{};

  REQUIRE(fixture.ioctx->host_read_io(*object, 0, out.size(), out.data()) == out.size());
  auto const snapshot = fixture.ioctx->perf_snapshot();
  CHECK(snapshot.chunk_get_count == 1);
  CHECK(snapshot.chunk_get_ns_total > 0);
  CHECK(snapshot.chunk_get_ns_max > 0);
  CHECK(snapshot.queue_wait_count == 1);
  CHECK(snapshot.queue_wait_ns_total > 0);
  CHECK(snapshot.ttfb_ns > 0);
  CHECK(snapshot.blocking_host_get_count == 1);
}

TEST_CASE("pool snapshot aggregates across reactors", "[rest][perf]")
{
  auto payload = deterministic_payload(4096);
  loopback_range_server server(payload);
  auto cfg            = test_config();
  cfg.max_connections = 1;
  auto fixture        = make_ioctx(server, cfg, 2);
  auto object         = known_object(*fixture.ioctx, payload.size());
  std::array<std::uint8_t, 64> first{};
  std::array<std::uint8_t, 64> second{};

  auto first_future = fixture.ioctx->host_read_async_io(*object, 0, first.size(), first.data());
  REQUIRE(std::move(first_future).get(5s) == first.size());
  auto second_future =
    fixture.ioctx->host_read_async_io(*object, 128, second.size(), second.data());
  REQUIRE(std::move(second_future).get(5s) == second.size());

  auto const snapshot = fixture.ioctx->perf_snapshot();
  CHECK(snapshot.chunk_get_count == 2);
  CHECK(snapshot.chunk_get_ns_total >= snapshot.chunk_get_ns_max);
  CHECK(snapshot.chunk_get_ns_max > 0);
  CHECK(snapshot.queue_wait_count == 2);
  CHECK(snapshot.ttfb_ns > 0);
  CHECK(snapshot.payload_bytes_read_total == first.size() + second.size());
}

TEST_CASE("transient retry is counted", "[rest][perf]")
{
  auto payload = deterministic_payload(4096);
  range_fault_policy fault;
  fault.fail_first_gets = 1;
  loopback_range_server server(payload, fault);
  auto fixture = make_ioctx(server, test_config(false));
  auto object  = known_object(*fixture.ioctx, payload.size());
  std::array<std::uint8_t, 128> out{};

  auto future = fixture.ioctx->host_read_async_io(*object, 0, out.size(), out.data());
  REQUIRE(std::move(future).get(5s) == out.size());
  auto const snapshot = fixture.ioctx->perf_snapshot();
  CHECK(snapshot.retries_total == 1);
  CHECK(snapshot.terminal_failures_total == 0);
  CHECK(fixture.authorizer->get_count() == 2);
}

TEST_CASE("exhausted retries count terminal", "[rest][perf]")
{
  auto payload = deterministic_payload(4096);
  range_fault_policy fault;
  fault.fail_all_gets = true;
  loopback_range_server server(payload, fault);
  auto cfg               = test_config(false);
  cfg.max_retry_attempts = 3;
  auto fixture           = make_ioctx(server, cfg);
  auto object            = known_object(*fixture.ioctx, payload.size());
  std::array<std::uint8_t, 128> out{};

  auto future = fixture.ioctx->host_read_async_io(*object, 0, out.size(), out.data());
  CHECK_THROWS(std::move(future).get(5s));
  auto const snapshot = fixture.ioctx->perf_snapshot();
  CHECK(snapshot.retries_total == 2);
  CHECK(snapshot.terminal_failures_total == 1);
  CHECK(fixture.authorizer->get_count() == 3);
}

TEST_CASE("auth retry re-authorizes and is counted", "[rest][perf]")
{
  auto payload = deterministic_payload(4096);
  range_fault_policy fault;
  fault.fail_first_gets = 1;
  fault.fail_status     = 403;
  loopback_range_server server(payload, fault);
  auto cfg                    = test_config(false);
  cfg.max_auth_retry_attempts = 2;
  auto fixture                = make_ioctx(server, cfg);
  auto object                 = known_object(*fixture.ioctx, payload.size());
  std::array<std::uint8_t, 128> out{};

  auto future = fixture.ioctx->host_read_async_io(*object, 0, out.size(), out.data());
  REQUIRE(std::move(future).get(5s) == out.size());
  auto const snapshot = fixture.ioctx->perf_snapshot();
  CHECK(snapshot.retries_total == 1);
  CHECK(snapshot.terminal_failures_total == 0);
  CHECK(fixture.authorizer->get_count() == 2);
}

TEST_CASE("not found counts terminal", "[rest][perf]")
{
  auto payload = deterministic_payload(4096);
  range_fault_policy fault;
  fault.fail_all_gets = true;
  fault.fail_status   = 404;
  loopback_range_server server(payload, fault);
  auto fixture = make_ioctx(server, test_config(false));
  auto object  = known_object(*fixture.ioctx, payload.size());
  std::array<std::uint8_t, 128> out{};

  auto future = fixture.ioctx->host_read_async_io(*object, 0, out.size(), out.data());
  CHECK_THROWS(std::move(future).get(5s));
  auto const snapshot = fixture.ioctx->perf_snapshot();
  CHECK(snapshot.retries_total == 0);
  CHECK(snapshot.terminal_failures_total == 1);
  CHECK(fixture.authorizer->get_count() == 1);
}

TEST_CASE("retry events count exactly once", "[rest][perf]")
{
  auto payload = deterministic_payload(4096);
  range_fault_policy fault;
  fault.fail_all_gets = true;
  fault.fail_status   = 503;
  loopback_range_server server(payload, fault);
  auto cfg               = test_config(false);
  cfg.max_retry_attempts = 3;
  auto fixture           = make_ioctx(server, cfg);
  auto object            = known_object(*fixture.ioctx, payload.size());
  std::array<std::uint8_t, 64> out{};

  for (int i = 0; i < 2; ++i) {
    auto future = fixture.ioctx->host_read_async_io(*object, 0, out.size(), out.data());
    CHECK_THROWS(std::move(future).get(5s));
  }
  auto const snapshot = fixture.ioctx->perf_snapshot();
  CHECK(snapshot.retries_total == 4);
  CHECK(snapshot.terminal_failures_total == 2);
  CHECK(fixture.authorizer->get_count() == 6);
}

TEST_CASE("reactor teardown resolves all futures", "[rest][perf]")
{
  auto payload = deterministic_payload(4096);
  range_fault_policy fault;
  fault.response_delay = 500ms;
  loopback_range_server server(payload, fault);
  auto fixture = make_ioctx(server, test_config(false));
  auto object  = known_object(*fixture.ioctx, payload.size());
  std::array<std::uint8_t, 128> out{};

  auto future = fixture.ioctx->host_read_async_io(*object, 0, out.size(), out.data());
  fixture.ioctx->shutdown();
  CHECK_THROWS(std::move(future).get(2s));
}

TEST_CASE("device read records h2d timings", "[rest][perf][gpu]")
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    SKIP("CUDA device is unavailable");
  }
  REQUIRE(cudaSetDevice(0) == cudaSuccess);

  auto payload = deterministic_payload(32 * 1024);
  loopback_range_server server(payload);
  staging_resource staging;
  auto cfg              = test_config();
  cfg.max_connections   = 1;
  cfg.bounce_block_size = staging_resource::block_size;
  auto fixture          = make_ioctx(server, cfg, 1, &staging.blocks);
  auto object           = known_object(*fixture.ioctx, payload.size());
  device_allocation device(payload.size());
  rmm::cuda_stream stream;

  auto future =
    fixture.ioctx->device_read_async_io(*object, 0, payload.size(), device.data(), stream.view());
  REQUIRE(std::move(future).get(5s) == payload.size());
  stream.synchronize();

  std::vector<std::uint8_t> actual(payload.size());
  REQUIRE(cudaMemcpy(actual.data(), device.data(), actual.size(), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  require_bytes_equal(actual, payload);

  auto const snapshot = fixture.ioctx->perf_snapshot();
  CHECK(snapshot.h2d_observed_count == 1);
  CHECK(snapshot.h2d_observed_ns_total > 0);
  CHECK(snapshot.h2d_observed_ns_max > 0);
  CHECK(snapshot.device_stream_sync_total == 0);
}

TEST_CASE("queued get records queue wait", "[rest][perf]")
{
  auto payload = deterministic_payload(4096);
  range_fault_policy fault;
  fault.response_delay = 75ms;
  loopback_range_server server(payload, fault);
  auto cfg            = test_config();
  cfg.max_connections = 1;
  auto fixture        = make_ioctx(server, cfg);
  auto object         = known_object(*fixture.ioctx, payload.size());
  std::array<std::uint8_t, 128> first{};
  std::array<std::uint8_t, 128> second{};

  auto first_future = fixture.ioctx->host_read_async_io(*object, 0, first.size(), first.data());
  auto second_future =
    fixture.ioctx->host_read_async_io(*object, 256, second.size(), second.data());
  REQUIRE(std::move(first_future).get(5s) == first.size());
  REQUIRE(std::move(second_future).get(5s) == second.size());

  auto const snapshot = fixture.ioctx->perf_snapshot();
  CHECK(snapshot.queue_wait_count == 2);
  // With one connection, the second GET queues behind the 75 ms delay; 50 ms leaves scheduler
  // slack.
  CHECK(
    snapshot.queue_wait_ns_total >=
    static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(50ms).count()));
  CHECK(snapshot.chunk_get_count == 2);
}

TEST_CASE("stash hit moves no counters", "[rest][perf]")
{
  auto payload = deterministic_payload(4096);
  loopback_range_server server(payload);
  auto cfg               = test_config();
  cfg.footer_probe_bytes = 512;
  auto fixture           = make_ioctx(server, cfg);
  auto object =
    fixture.ioctx->open_io_object("s3://bucket/object.bin", open_hint::parquet_footer_probe);
  auto const before = fixture.ioctx->perf_snapshot();
  std::array<std::uint8_t, 64> out{};

  REQUIRE(fixture.ioctx->host_read_io(
            *object, payload.size() - out.size(), out.size(), out.data()) == out.size());
  require_bytes_equal(
    out, std::span<const std::uint8_t>(payload.data() + payload.size() - out.size(), out.size()));
  auto const after = fixture.ioctx->perf_snapshot();
  CHECK(after.chunk_get_count == before.chunk_get_count);
  CHECK(after.payload_bytes_read_total == before.payload_bytes_read_total);
  CHECK(after.blocking_host_get_count == before.blocking_host_get_count);
  CHECK(after.retries_total == before.retries_total);
  CHECK(after.terminal_failures_total == before.terminal_failures_total);
  CHECK(server.get_count() == 1);
}
