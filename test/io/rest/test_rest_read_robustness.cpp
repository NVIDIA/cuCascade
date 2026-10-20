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

/// @file test_rest_read_robustness.cpp
/// @brief End-to-end read-path behaviour of @c rest_ioctx over the io_uring transport against
/// the plain-HTTP loopback server: priority (demand ahead of prefetch), the data-path 403
/// re-sign, Content-Range mismatch, a server that ignores Range, resume after a mid-body drop,
/// Retry-After (a numeric value replaces the backoff), pinned-staging exhaustion of device reads
/// and the transport counters (docs/io/rest-uring-design.md section 9).

#include "rest_contract_test_utils.hpp"

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using cucascade::io::io_priority;
using cucascade::io::slice;
using cucascade::io::rest::rest_ioctx;
using cucascade::io::rest::rest_reactor;
using cucascade::io::rest::transport_stats;
using cucascade::test::key_response_script;
using cucascade::test::list_capable_mock_authorizer;
using cucascade::test::loopback_range_server;
using cucascade::test::range_fault_policy;
using cucascade::test::require_rest_bytes_equal;
using cucascade::test::response_body_policy;
using cucascade::test::rest_contract_config;
using cucascade::test::rest_device_bytes;
using cucascade::test::rest_shutdown_guard;
using cucascade::test::rest_test_payload;
using cucascade::test::scripted_response;
using namespace std::chrono_literals;

/// A @c rest_ioctx whose reactors' transport counters the test can read.
class probed_ioctx final : public rest_ioctx {
 public:
  using rest_ioctx::rest_ioctx;
  [[nodiscard]] transport_stats reactor_stats(std::size_t i = 0) const
  {
    return _reactors.at(i)->stats();
  }
};

struct read_fixture {
  std::shared_ptr<list_capable_mock_authorizer> authorizer;
  std::shared_ptr<probed_ioctx> ioctx;
};

read_fixture make_ioctx(loopback_range_server const& server,
                        cucascade::io::rest::config cfg = rest_contract_config(),
                        cucascade::memory::fixed_size_host_memory_resource* host = nullptr)
{
  auto authorizer = std::make_shared<list_capable_mock_authorizer>(server.endpoint());
  auto context    = std::make_shared<rest_reactor::reactor_context>(cfg, authorizer, host);
  auto ioctx      = std::make_shared<probed_ioctx>(1, std::move(context));
  ioctx->start();
  return {std::move(authorizer), std::move(ioctx)};
}

/// The worker publishes its counters every 100 ms: poll until @p ready holds.
template <typename Predicate>
transport_stats wait_for_stats(probed_ioctx const& ioctx,
                               Predicate ready,
                               std::chrono::milliseconds timeout = 5s)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  auto stats          = ioctx.reactor_stats();
  while (!ready(stats) && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(20ms);
    stats = ioctx.reactor_stats();
  }
  return stats;
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

/// "bytes=<first>-<last>" -> first.
std::size_t range_first(std::string const& range)
{
  REQUIRE(range.starts_with("bytes="));
  return std::stoull(range.substr(6, range.find('-', 6) - 6));
}

bool has_cuda_device()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

}  // namespace

TEST_CASE("rest demand reads run ahead of a queued prefetch backlog", "[rest][e2e][priority]")
{
  // One connection, a paced server: every GET takes ~40 ms, so the order in which the reactor
  // starts queued GETs is visible in when they complete.
  constexpr std::size_t n_low = 12;
  constexpr std::size_t piece = 256UL << 10;
  auto payload                = rest_test_payload((n_low + 1) * piece);
  range_fault_policy fault{};
  fault.body = response_body_policy{.chunk_bytes = 64UL << 10, .chunk_delay = 10ms};
  loopback_range_server server(payload, fault);
  auto cfg            = rest_contract_config();
  cfg.max_connections = 1;
  auto fx             = make_ioctx(server, cfg);
  rest_shutdown_guard cleanup{*fx.ioctx};
  auto object = fx.ioctx->open_io_object("s3://priority-bucket/object.bin", payload.size());

  std::vector<std::vector<std::uint8_t>> low_out(n_low, std::vector<std::uint8_t>(piece));
  std::vector<cucascade::exec::semi_future<std::size_t>> lows;
  for (std::size_t i = 0; i < n_low; ++i) {
    lows.push_back(
      fx.ioctx->host_read_async_io(*object, i * piece, piece, low_out[i].data(), io_priority::low));
  }
  std::vector<std::uint8_t> high_out(piece);
  auto high =
    fx.ioctx->host_read_async_io(*object, n_low * piece, piece, high_out.data(), io_priority::high);
  // Count the low reads that finished before the demand read did.
  std::size_t lows_before_high = 0;
  REQUIRE(std::move(high).get(30s) == piece);
  for (auto& low : lows) {
    if (low.is_ready()) { ++lows_before_high; }
  }
  for (std::size_t i = 0; i < n_low; ++i) {
    REQUIRE(std::move(lows[i]).get(30s) == piece);
    require_rest_bytes_equal(low_out[i], {payload.data() + i * piece, piece});
  }
  require_rest_bytes_equal(high_out, {payload.data() + n_low * piece, piece});
  // Strict FIFO would finish all twelve first; the demand read waits for at most the GETs that
  // were already on the wire (or chosen) when it arrived.
  CHECK(lows_before_high <= n_low / 2);
}

TEST_CASE("rest data GET re-signs after a 403 and retries", "[rest][e2e][auth]")
{
  auto payload = rest_test_payload(256UL << 10);
  std::unordered_map<std::string, key_response_script> scripts;
  SECTION("one 403, then the object")
  {
    scripts["auth.bin"].gets = {scripted_response{.status = 403}, scripted_response{}};
    loopback_range_server server(payload, {}, {}, scripts);
    auto cfg                    = rest_contract_config();
    cfg.max_auth_retry_attempts = 1;
    auto fx                     = make_ioctx(server, cfg);
    rest_shutdown_guard cleanup{*fx.ioctx};
    auto object             = fx.ioctx->open_io_object("s3://auth-bucket/auth.bin", payload.size());
    auto const calls_before = fx.authorizer->object_calls();
    auto const got          = read_range(*fx.ioctx, *object, 4096, 64UL << 10);
    require_rest_bytes_equal(got, {payload.data() + 4096, 64UL << 10});
    CHECK(server.get_count("auth.bin") == 2);
    // Every attempt is authorized afresh (a presigned URL that expired in the queue).
    CHECK(fx.authorizer->object_calls() - calls_before == 2);
    auto const stats =
      wait_for_stats(*fx.ioctx, [](transport_stats const& s) { return s.auth_retries >= 1; });
    CHECK(stats.auth_retries == 1);
    CHECK(stats.responses_4xx == 1);
  }
  SECTION("403s beyond max_auth_retry_attempts fail the read")
  {
    scripts["auth.bin"].gets = {scripted_response{.status = 403},
                                scripted_response{.status = 403},
                                scripted_response{.status = 403},
                                scripted_response{}};
    loopback_range_server server(payload, {}, {}, scripts);
    auto cfg                    = rest_contract_config();
    cfg.max_retry_attempts      = 5;
    cfg.max_auth_retry_attempts = 1;
    auto fx                     = make_ioctx(server, cfg);
    rest_shutdown_guard cleanup{*fx.ioctx};
    auto object = fx.ioctx->open_io_object("s3://auth-bucket/auth.bin", payload.size());
    std::vector<std::uint8_t> out(4096);
    CHECK_THROWS(fx.ioctx->host_read(*object, 0, out.size(), out.data()));
    CHECK(server.get_count("auth.bin") == 2);
  }
}

TEST_CASE("rest data GET fails a 206 whose Content-Range does not match without retrying",
          "[rest][e2e][content_range]")
{
  auto payload = rest_test_payload(256UL << 10);
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["shifted.bin"].gets = {
    scripted_response{.content_range = "bytes 0-4095/" + std::to_string(payload.size())}};
  loopback_range_server server(payload, {}, {}, scripts);
  auto cfg               = rest_contract_config();
  cfg.max_retry_attempts = 4;
  auto fx                = make_ioctx(server, cfg);
  rest_shutdown_guard cleanup{*fx.ioctx};
  auto object = fx.ioctx->open_io_object("s3://range-bucket/shifted.bin", payload.size());
  std::vector<std::uint8_t> out(4096);
  CHECK_THROWS_WITH(fx.ioctx->host_read(*object, 8192, out.size(), out.data()),
                    Catch::Matchers::ContainsSubstring("Content-Range mismatch"));
  CHECK(server.get_count("shifted.bin") == 1);
}

TEST_CASE("rest data GET against a server that ignores Range", "[rest][e2e][range_ignored]")
{
  range_fault_policy fault{};
  fault.ignore_range_with_200 = true;
  auto cfg                    = rest_contract_config();
  cfg.max_retry_attempts      = 4;

  SECTION("a small object is sliced to the requested window")
  {
    auto payload = rest_test_payload(64UL << 10);
    loopback_range_server server(payload, fault);
    auto fx = make_ioctx(server, cfg);
    rest_shutdown_guard cleanup{*fx.ioctx};
    auto object    = fx.ioctx->open_io_object("s3://ignored-bucket/small.bin", payload.size());
    auto const got = read_range(*fx.ioctx, *object, 1000, 40000);
    require_rest_bytes_equal(got, {payload.data() + 1000, 40000});
    CHECK(server.get_count() == 1);
  }
  SECTION("a whole-object body far larger than the window fails without retrying")
  {
    auto payload = rest_test_payload(4UL << 20);
    loopback_range_server server(payload, fault);
    auto fx = make_ioctx(server, cfg);
    rest_shutdown_guard cleanup{*fx.ioctx};
    auto object = fx.ioctx->open_io_object("s3://ignored-bucket/large.bin", payload.size());
    std::vector<std::uint8_t> out(64UL << 10);
    CHECK_THROWS_WITH(fx.ioctx->host_read(*object, 4096, out.size(), out.data()),
                      Catch::Matchers::ContainsSubstring("ignored Range"));
    CHECK(server.get_count() == 1);
    auto const stats =
      wait_for_stats(*fx.ioctx, [](transport_stats const& s) { return s.range_ignored >= 1; });
    CHECK(stats.range_ignored == 1);
  }
}

TEST_CASE("rest data GET resumes after a mid-body connection drop", "[rest][e2e][resume]")
{
  auto payload                 = rest_test_payload(2UL << 20);
  constexpr std::size_t offset = 4096;
  constexpr std::size_t size   = 1UL << 20;
  constexpr std::size_t cut    = 300000;
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["drop.bin"].gets = {
    scripted_response{.body = response_body_policy{.drop_after_bytes = cut}}, scripted_response{}};
  loopback_range_server server(payload, {}, {}, scripts);
  auto cfg               = rest_contract_config();
  cfg.max_connections    = 1;
  cfg.max_retry_attempts = 3;
  auto fx                = make_ioctx(server, cfg);
  rest_shutdown_guard cleanup{*fx.ioctx};
  auto object = fx.ioctx->open_io_object("s3://resume-bucket/drop.bin", payload.size());

  auto const got = read_range(*fx.ioctx, *object, offset, size);
  require_rest_bytes_equal(got, {payload.data() + offset, size});
  CHECK(server.body_drops() == 1);
  auto const requests = server.get_requests();
  REQUIRE(requests.size() == 2);
  REQUIRE(requests[0].ranges.size() == 1);
  REQUIRE(requests[1].ranges.size() == 1);
  CHECK(range_first(requests[0].ranges.front()) == offset);
  // The retry asks only for what the dropped response did not deliver.
  CHECK(range_first(requests[1].ranges.front()) == offset + cut);
  auto const stats =
    wait_for_stats(*fx.ioctx, [](transport_stats const& s) { return s.resumes >= 1; });
  CHECK(stats.resumes == 1);
  CHECK(stats.retries == 1);
}

TEST_CASE("rest data GET: a numeric Retry-After replaces the computed backoff",
          "[rest][e2e][retry_after]")
{
  auto payload           = rest_test_payload(64UL << 10);
  auto cfg               = rest_contract_config();
  cfg.retry_jitter       = 0ms;
  cfg.max_retry_attempts = 3;
  std::string retry_after;
  auto min_elapsed = 0ms;
  auto max_elapsed = 0ms;
  SECTION("Retry-After: 0 retries at once although the backoff is long")
  {
    cfg.honor_retry_after  = true;
    cfg.retry_backoff_base = 2s;  // a 503 backs off from 40 s (throttle class)
    retry_after            = "0";
    max_elapsed            = 1500ms;
  }
  SECTION("Retry-After: 1 waits a second although the backoff is tiny")
  {
    cfg.honor_retry_after  = true;
    cfg.retry_backoff_base = 1ms;
    retry_after            = "1";
    min_elapsed            = 900ms;
    max_elapsed            = 5s;
  }
  SECTION("honor_retry_after off ignores the header")
  {
    cfg.honor_retry_after  = false;
    cfg.retry_backoff_base = 1ms;
    retry_after            = "3";
    max_elapsed            = 1500ms;
  }
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["throttled.bin"].gets = {scripted_response{.status = 503, .retry_after = retry_after},
                                   scripted_response{}};
  loopback_range_server server(payload, {}, {}, scripts);
  auto fx = make_ioctx(server, cfg);
  rest_shutdown_guard cleanup{*fx.ioctx};
  auto object = fx.ioctx->open_io_object("s3://throttle-bucket/throttled.bin", payload.size());
  auto const started = std::chrono::steady_clock::now();
  auto const got     = read_range(*fx.ioctx, *object, 128, 4096);
  auto const elapsed = std::chrono::steady_clock::now() - started;
  require_rest_bytes_equal(got, {payload.data() + 128, 4096});
  CHECK(server.get_count("throttled.bin") == 2);
  CHECK(elapsed >= min_elapsed);
  CHECK(elapsed < max_elapsed);
}

TEST_CASE("rest device read fails fast when pinned staging is exhausted",
          "[rest][e2e][staging][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  constexpr std::size_t block_size = 256UL << 10;
  constexpr std::size_t capacity   = 1UL << 20;
  cucascade::memory::numa_region_pinned_host_memory_resource upstream{0};
  cucascade::memory::fixed_size_host_memory_resource mr{
    0, upstream, capacity, capacity, block_size, 4, 1};
  auto payload = rest_test_payload(4UL << 20);
  loopback_range_server server(payload);
  auto fx = make_ioctx(server, rest_contract_config(), &mr);
  rest_shutdown_guard cleanup{*fx.ioctx};
  auto object = fx.ioctx->open_io_object("s3://staging-bucket/object.bin", payload.size());
  rmm::cuda_stream stream;
  rmm::device_buffer destination(payload.size(), stream);

  // More than the whole resource: the read fails as an out-of-memory error (staging is taken
  // per GET, so at most the one GET that fit was sent), and every staging block comes back.
  auto too_big = fx.ioctx->device_read_async_io(*object,
                                                0,
                                                payload.size(),
                                                static_cast<std::uint8_t*>(destination.data()),
                                                ::cuda::stream_ref{stream.value()});
  CHECK_THROWS_AS(std::move(too_big).get(10s), rmm::out_of_memory);
  CHECK(server.get_count() <= 1);
  CHECK(mr.get_available_memory() == capacity);

  // The reactor stays usable: a read that fits is staged and lands.
  constexpr std::size_t fits = 512UL << 10;
  auto ok                    = fx.ioctx->device_read_async_io(*object,
                                           8192,
                                           fits,
                                           static_cast<std::uint8_t*>(destination.data()),
                                           ::cuda::stream_ref{stream.value()});
  REQUIRE(std::move(ok).get(10s) == fits);
  auto const got = rest_device_bytes(destination.data(), fits, stream.value());
  require_rest_bytes_equal(got, {payload.data() + 8192, fits});
}

TEST_CASE("rest transport counters follow the reads and survive shutdown", "[rest][e2e][stats]")
{
  auto payload = rest_test_payload(3UL << 20);
  loopback_range_server server(payload);
  auto cfg              = rest_contract_config();
  cfg.max_connections   = 1;
  cfg.max_segment_bytes = 1UL << 20;
  auto fx               = make_ioctx(server, cfg);
  auto object           = fx.ioctx->open_io_object("s3://stats-bucket/object.bin", payload.size());
  auto const first      = read_range(*fx.ioctx, *object, 0, payload.size());
  require_rest_bytes_equal(first, payload);
  auto const second = read_range(*fx.ioctx, *object, 1000, 5000);
  require_rest_bytes_equal(second, {payload.data() + 1000, 5000});

  auto const gets = server.get_count();
  auto const live =
    wait_for_stats(*fx.ioctx, [&](transport_stats const& s) { return s.responses_2xx >= gets; });
  CHECK(live.requests == gets);
  CHECK(live.responses_2xx == gets);
  CHECK(live.bytes_landed == payload.size() + 5000);
  CHECK(live.bytes_received >= live.bytes_landed);
  CHECK(live.segments_le_1mib == gets);
  CHECK(live.connects == 1);
  CHECK(live.reused_checkouts == gets - 1);
  CHECK(live.retries == 0);

  fx.ioctx->shutdown();
  auto const after = fx.ioctx->reactor_stats();
  CHECK(after.requests == live.requests);
  CHECK(after.bytes_landed == live.bytes_landed);
  CHECK(after.connects == live.connects);
  CHECK(after.reused_checkouts == live.reused_checkouts);
}
