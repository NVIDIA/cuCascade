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
/// re-sign, Content-Range mismatch, chunked GET bodies (unsupported framing, final), a server
/// that ignores Range, a retry after a mid-body drop, Retry-After (a numeric value replaces the
/// backoff), pinned-staging exhaustion of device reads, the reactor counters, device
/// destinations on every GPU and on a null (legacy) stream, and framing hygiene of the control
/// plane (incomplete 2xx bodies fail; bytes past a response retire the connection).

#include "io/rest/errc.hpp"
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

#include <arpa/inet.h>
#include <catch2/catch_all.hpp>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
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
  [[nodiscard]] std::size_t queued(std::size_t i = 0) const
  {
    return _reactors.at(i)->queued_bytes();
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

/// The worker publishes its counters once per loop pass: poll until @p ready holds.
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

/// The worker drops a finished operation (and the pinned staging it holds) just after settling
/// its credit, so a future can resolve a moment before its blocks are back: wait for them.
std::size_t available_after_release(cucascade::memory::fixed_size_host_memory_resource const& mr,
                                    std::size_t expected)
{
  auto const deadline = std::chrono::steady_clock::now() + 5s;
  while (mr.get_available_memory() != expected && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  return mr.get_available_memory();
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
    auto cfg = rest_contract_config();
    // Attempts, as in Sirius' libcurl reactor: 2 = one re-signed retry.
    cfg.max_auth_retry_attempts = 2;
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
    cfg.max_auth_retry_attempts = 2;
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

TEST_CASE("rest data GET with a chunked 206 body fails as unsupported framing without retrying",
          "[rest][e2e][framing]")
{
  // libcurl decoded chunked GET bodies; this transport does not.  The failure names the cause
  // and is final: a retry cannot succeed, so the second (Content-Length) step of the script is
  // never served (the regression test of errc::http_unsupported_framing).
  auto payload = rest_test_payload(64UL << 10);
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["chunked.bin"].gets = {scripted_response{.chunked_transfer_encoding = true},
                                 scripted_response{}};
  loopback_range_server server(payload, {}, {}, scripts);
  auto cfg               = rest_contract_config();
  cfg.max_retry_attempts = 4;
  auto fx                = make_ioctx(server, cfg);
  rest_shutdown_guard cleanup{*fx.ioctx};
  auto object = fx.ioctx->open_io_object("s3://range-bucket/chunked.bin", payload.size());
  std::vector<std::uint8_t> out(4096);
  CHECK_THROWS_WITH(fx.ioctx->host_read(*object, 8192, out.size(), out.data()),
                    Catch::Matchers::ContainsSubstring("chunked or close-delimited GET body"));
  CHECK(server.get_count("chunked.bin") == 1);

  auto const message =
    make_error_code(cucascade::io::rest::errc::http_unsupported_framing).message();
  CHECK_THAT(message, Catch::Matchers::ContainsSubstring("chunked"));
  CHECK_THAT(message, !Catch::Matchers::ContainsSubstring("unknown"));
}

TEST_CASE("rest data GET against a server that ignores Range", "[rest][e2e][range_ignored]")
{
  // Sirius' libcurl reactor: a 200 is accepted only for a whole-object range; for a sub-range it
  // is a final "server ignored Range" (no slicing, no retry), and the body is never written.
  range_fault_policy fault{};
  fault.ignore_range_with_200 = true;
  auto cfg                    = rest_contract_config();
  cfg.max_retry_attempts      = 4;
  auto payload                = rest_test_payload(64UL << 10);
  loopback_range_server server(payload, fault);
  auto fx = make_ioctx(server, cfg);
  rest_shutdown_guard cleanup{*fx.ioctx};
  auto object = fx.ioctx->open_io_object("s3://ignored-bucket/object.bin", payload.size());
  SECTION("a sub-range fails without retrying")
  {
    std::vector<std::uint8_t> out(40000, 0xEE);
    CHECK_THROWS_WITH(fx.ioctx->host_read(*object, 1000, out.size(), out.data()),
                      Catch::Matchers::ContainsSubstring("ignored Range"));
    CHECK(server.get_count() == 1);
    CHECK(std::all_of(out.begin(), out.end(), [](std::uint8_t b) { return b == 0xEE; }));
  }
  SECTION("the whole object is accepted from a 200")
  {
    auto const got = read_range(*fx.ioctx, *object, 0, payload.size());
    require_rest_bytes_equal(got, payload);
    CHECK(server.get_count() == 1);
  }
}

TEST_CASE("rest data GET retries after a mid-body connection drop", "[rest][e2e][retry]")
{
  // The dropped response is retried as a whole on a new connection (uring_ultra: no resume);
  // the destination ends byte-exact.
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
  auto object = fx.ioctx->open_io_object("s3://retry-bucket/drop.bin", payload.size());

  auto const got = read_range(*fx.ioctx, *object, offset, size);
  require_rest_bytes_equal(got, {payload.data() + offset, size});
  CHECK(server.body_drops() == 1);
  auto const requests = server.get_requests();
  REQUIRE(requests.size() == 2);
  REQUIRE(requests[1].ranges.size() == 1);
  CHECK(range_first(requests[1].ranges.front()) == offset);
  CHECK(server.accepted_connection_count() == 2);
  auto const stats =
    wait_for_stats(*fx.ioctx, [](transport_stats const& s) { return s.gets >= 1; });
  CHECK(stats.retries == 1);
  CHECK(stats.gets == 1);
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

TEST_CASE("rest device reads park on exhausted pinned staging until this reactor returns it",
          "[rest][e2e][staging][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  // Staging is taken per GET.  While GETs (or parked copies) of this reactor hold blocks, an
  // exhausted pool parks the next GET until one of them returns its blocks; only when the reactor
  // holds nothing (no block can come back from it) does a read fail with out_of_memory.
  constexpr std::size_t block_size = 1UL << 20;
  SECTION("segments that fit, a pool smaller than the demand in flight: the read completes")
  {
    constexpr std::size_t capacity = 2UL << 20;  // two blocks for an 8 MiB read in 1 MiB GETs
    cucascade::memory::numa_region_pinned_host_memory_resource upstream{0};
    cucascade::memory::fixed_size_host_memory_resource mr{
      0, upstream, capacity, capacity, block_size, 4, 1};
    auto payload = rest_test_payload(8UL << 20);
    range_fault_policy fault{};
    fault.response_delay = 50ms;  // GETs are still in flight when the next one wants staging
    loopback_range_server server(payload, fault);
    auto cfg              = rest_contract_config();
    cfg.max_connections   = 4;
    cfg.min_segment_bytes = 1UL << 20;
    cfg.max_segment_bytes = 1UL << 20;
    auto fx               = make_ioctx(server, cfg, &mr);
    rest_shutdown_guard cleanup{*fx.ioctx};
    auto object = fx.ioctx->open_io_object("s3://staging-bucket/object.bin", payload.size());
    rmm::cuda_stream stream;
    rmm::device_buffer destination(payload.size(), stream);
    auto read = fx.ioctx->device_read_async_io(*object,
                                               0,
                                               payload.size(),
                                               static_cast<std::uint8_t*>(destination.data()),
                                               ::cuda::stream_ref{stream.value()});
    REQUIRE(std::move(read).get(30s) == payload.size());
    require_rest_bytes_equal(rest_device_bytes(destination.data(), payload.size(), stream.value()),
                             payload);
    auto const stats = fx.ioctx->reactor_stats();
    CHECK(stats.staging_waits > 0);
    CHECK(stats.failures == 0);
    CHECK(server.get_count() == 8);
    CHECK(available_after_release(mr, capacity) == capacity);
  }
  SECTION("a park keeps the backlog queued and drains once the parked copy returns its block")
  {
    constexpr std::size_t capacity = 1UL << 20;  // one block
    cucascade::memory::numa_region_pinned_host_memory_resource upstream{0};
    cucascade::memory::fixed_size_host_memory_resource mr{
      0, upstream, capacity, capacity, block_size, 4, 1};
    auto payload = rest_test_payload(16UL << 20);
    loopback_range_server server(payload);
    auto cfg              = rest_contract_config();
    cfg.min_segment_bytes = 1UL << 20;
    cfg.max_segment_bytes = 1UL << 20;
    auto fx               = make_ioctx(server, cfg, &mr);
    rest_shutdown_guard cleanup{*fx.ioctx};
    auto object = fx.ioctx->open_io_object("s3://staging-bucket/object.bin", payload.size());
    rmm::cuda_stream stream;
    rmm::device_buffer destination(payload.size(), stream);
    // The first read's copy parks behind a held stream and keeps the only block.
    auto* const hold = new std::chrono::milliseconds{1000ms};
    REQUIRE(cudaLaunchHostFunc(
              stream.value(),
              [](void* arg) {
                std::unique_ptr<std::chrono::milliseconds> const t{
                  static_cast<std::chrono::milliseconds*>(arg)};
                std::this_thread::sleep_for(*t);
              },
              hold) == cudaSuccess);
    auto first = fx.ioctx->device_read_async_io(*object,
                                                0,
                                                1UL << 20,
                                                static_cast<std::uint8_t*>(destination.data()),
                                                ::cuda::stream_ref{stream.value()});
    REQUIRE(wait_for_stats(*fx.ioctx, [](transport_stats const& s) {
              return s.h2d_copies >= 1;
            }).h2d_copies == 1);
    std::vector<slice> slices;
    for (std::size_t i = 0; i < 16; ++i) {
      slices.push_back(
        slice{i << 20, 1UL << 20, static_cast<std::uint8_t*>(destination.data()) + (i << 20)});
    }
    auto rest =
      fx.ioctx->device_readv_async_io(*object, slices, ::cuda::stream_ref{stream.value()});
    std::this_thread::sleep_for(300ms);  // the first slice parks; the rest must stay queued
    CHECK(fx.ioctx->queued(0) >= (8UL << 20));
    REQUIRE(std::move(first).get(30s) == (1UL << 20));
    REQUIRE(std::move(rest).get(30s) == payload.size());
    require_rest_bytes_equal(rest_device_bytes(destination.data(), payload.size(), stream.value()),
                             payload);
    auto const stats = fx.ioctx->reactor_stats();
    CHECK(stats.staging_waits >= 1);
    CHECK(stats.failures == 0);
  }
  SECTION("a GET larger than the whole pool fails when the reactor holds no staging")
  {
    constexpr std::size_t capacity = 1UL << 20;
    cucascade::memory::numa_region_pinned_host_memory_resource upstream{0};
    cucascade::memory::fixed_size_host_memory_resource mr{
      0, upstream, capacity, capacity, 256UL << 10, 4, 1};
    auto payload = rest_test_payload(4UL << 20);
    loopback_range_server server(payload);
    auto fx = make_ioctx(server, rest_contract_config(), &mr);
    rest_shutdown_guard cleanup{*fx.ioctx};
    auto object = fx.ioctx->open_io_object("s3://staging-bucket/object.bin", payload.size());
    rmm::cuda_stream stream;
    rmm::device_buffer destination(payload.size(), stream);
    auto too_big = fx.ioctx->device_read_async_io(*object,
                                                  0,
                                                  payload.size(),
                                                  static_cast<std::uint8_t*>(destination.data()),
                                                  ::cuda::stream_ref{stream.value()});
    CHECK_THROWS_AS(std::move(too_big).get(10s), rmm::out_of_memory);
    CHECK(server.get_count() == 0);
    CHECK(available_after_release(mr, capacity) == capacity);
    // The reactor stays usable: a read that fits is staged and lands.
    constexpr std::size_t fits = 512UL << 10;
    auto ok                    = fx.ioctx->device_read_async_io(*object,
                                             8192,
                                             fits,
                                             static_cast<std::uint8_t*>(destination.data()),
                                             ::cuda::stream_ref{stream.value()});
    REQUIRE(std::move(ok).get(10s) == fits);
    require_rest_bytes_equal(rest_device_bytes(destination.data(), fits, stream.value()),
                             {payload.data() + 8192, fits});
  }
}

TEST_CASE("rest reactor counters follow the reads and survive shutdown", "[rest][e2e][stats]")
{
  auto payload = rest_test_payload(3UL << 20);
  loopback_range_server server(payload);
  auto cfg              = rest_contract_config();
  cfg.max_connections   = 1;
  cfg.min_segment_bytes = 1UL << 20;
  cfg.max_segment_bytes = 1UL << 20;
  auto fx               = make_ioctx(server, cfg);
  auto object           = fx.ioctx->open_io_object("s3://stats-bucket/object.bin", payload.size());
  auto const first      = read_range(*fx.ioctx, *object, 0, payload.size());
  require_rest_bytes_equal(first, payload);
  auto const second = read_range(*fx.ioctx, *object, 1000, 5000);
  require_rest_bytes_equal(second, {payload.data() + 1000, 5000});

  auto const gets = server.get_count();
  CHECK(gets == 4);
  auto const live =
    wait_for_stats(*fx.ioctx, [&](transport_stats const& s) { return s.responses_2xx >= gets; });
  CHECK(live.gets == gets);
  CHECK(live.responses_2xx == gets);
  CHECK(live.get_bytes == payload.size() + 5000);
  CHECK(live.recv_direct_bytes + live.recv_scratch_bytes >= live.get_bytes);
  CHECK(live.connects == 1);
  CHECK(live.reused == gets - 1);
  CHECK(live.retries == 0);
  CHECK(live.failures == 0);
  CHECK(live.userspace_tls_sessions == 0);
  CHECK(live.ktls_sessions == 0);
  CHECK(live.established == 1);
  CHECK(fx.ioctx->stats().gets == gets);  // the pool aggregate

  fx.ioctx->shutdown();
  auto const after = fx.ioctx->reactor_stats();
  CHECK(after.gets == live.gets);
  CHECK(after.get_bytes == live.get_bytes);
  CHECK(after.connects == live.connects);
  CHECK(after.reused == live.reused);
}

namespace {

/// Hold @p stream on the host for @p duration (a host function that sleeps).
void hold_stream(cudaStream_t stream, std::chrono::milliseconds duration)
{
  auto* const heap = new std::chrono::milliseconds{duration};
  REQUIRE(cudaLaunchHostFunc(
            stream,
            [](void* arg) {
              std::unique_ptr<std::chrono::milliseconds> const time{
                static_cast<std::chrono::milliseconds*>(arg)};
              std::this_thread::sleep_for(*time);
            },
            heap) == cudaSuccess);
}

/// Copy @p bytes of device memory back on a separate stream, so the read does not wait for work
/// still queued on the stream the device bytes were written on.
[[nodiscard]] std::vector<std::uint8_t> device_bytes_elsewhere(void const* device,
                                                               std::size_t bytes)
{
  cudaStream_t other{};
  REQUIRE(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking) == cudaSuccess);
  auto out = rest_device_bytes(device, bytes, other);
  REQUIRE(cudaStreamDestroy(other) == cudaSuccess);
  return out;
}

}  // namespace

TEST_CASE("rest device reads on a null stream order after the legacy default stream",
          "[rest][e2e][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }
  // A null stream handle means the legacy default stream to the caller; the worker names it
  // explicitly (cudaStreamLegacy), so its copy is ordered after the caller's queued work.
  auto const null_handle = GENERATE(true, false);
  CAPTURE(null_handle);
  constexpr std::size_t block_size = 1UL << 20;
  constexpr std::size_t capacity   = 16UL << 20;
  cucascade::memory::numa_region_pinned_host_memory_resource upstream{0};
  cucascade::memory::fixed_size_host_memory_resource mr{
    0, upstream, capacity, capacity, block_size, 4, 1};
  auto payload = rest_test_payload(4UL << 20);
  loopback_range_server server(payload);
  auto fx = make_ioctx(server, rest_contract_config(), &mr);
  rest_shutdown_guard cleanup{*fx.ioctx};
  auto object = fx.ioctx->open_io_object("s3://stream-bucket/object.bin", payload.size());

  void* destination = nullptr;
  REQUIRE(cudaMalloc(&destination, payload.size()) == cudaSuccess);
  // Hold the legacy stream, then queue a memset of the destination behind it: a copy not
  // ordered after it lands first and is then zeroed.
  hold_stream(cudaStreamLegacy, 300ms);
  REQUIRE(cudaMemsetAsync(destination, 0, payload.size(), cudaStreamLegacy) == cudaSuccess);
  cudaStream_t const handle = null_handle ? cudaStream_t{nullptr} : cudaStreamLegacy;
  CHECK(fx.ioctx
          ->device_read_async_io(*object,
                                 0,
                                 payload.size(),
                                 static_cast<std::uint8_t*>(destination),
                                 ::cuda::stream_ref{handle})
          .get(30s) == payload.size());
  REQUIRE(cudaStreamSynchronize(cudaStreamLegacy) == cudaSuccess);
  require_rest_bytes_equal(device_bytes_elsewhere(destination, payload.size()), payload);
  REQUIRE(cudaFree(destination) == cudaSuccess);
}

TEST_CASE("rest device reads land on every GPU", "[rest][e2e][gpu][multi_gpu]")
{
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
    std::ignore = cudaGetLastError();
    SKIP("requires a CUDA device");
  }
  // One read per device, all in flight at once from the same reactor: each copy, its event and
  // its staging run with the destination's device current, whatever the worker's own device.
  constexpr std::size_t block_size = 1UL << 20;
  constexpr std::size_t capacity   = 64UL << 20;
  cucascade::memory::numa_region_pinned_host_memory_resource upstream{0};
  cucascade::memory::fixed_size_host_memory_resource mr{
    0, upstream, capacity, capacity, block_size, 4, 1};
  auto payload = rest_test_payload((5UL << 20) + 333);
  loopback_range_server server(payload);
  auto cfg = rest_contract_config();
  auto fx  = make_ioctx(server, cfg, &mr);
  rest_shutdown_guard cleanup{*fx.ioctx};
  auto object = fx.ioctx->open_io_object("s3://gpus-bucket/object.bin", payload.size());

  int previous = 0;
  REQUIRE(cudaGetDevice(&previous) == cudaSuccess);
  std::vector<void*> destinations(static_cast<std::size_t>(devices), nullptr);
  std::vector<cudaStream_t> streams(static_cast<std::size_t>(devices), nullptr);
  std::vector<cucascade::exec::semi_future<std::size_t>> futures;
  for (int d = 0; d < devices; ++d) {
    auto const i = static_cast<std::size_t>(d);
    REQUIRE(cudaSetDevice(d) == cudaSuccess);
    REQUIRE(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking) == cudaSuccess);
    REQUIRE(cudaMalloc(&destinations[i], payload.size()) == cudaSuccess);
    REQUIRE(cudaMemset(destinations[i], 0, payload.size()) == cudaSuccess);
    futures.push_back(fx.ioctx->device_read_async_io(*object,
                                                     0,
                                                     payload.size(),
                                                     static_cast<std::uint8_t*>(destinations[i]),
                                                     ::cuda::stream_ref{streams[i]}));
  }
  REQUIRE(cudaSetDevice(previous) == cudaSuccess);
  for (auto& future : futures) {
    CHECK(std::move(future).get(60s) == payload.size());
  }
  for (int d = 0; d < devices; ++d) {
    auto const i = static_cast<std::size_t>(d);
    CAPTURE(d);
    REQUIRE(cudaSetDevice(d) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(streams[i]) == cudaSuccess);
    require_rest_bytes_equal(device_bytes_elsewhere(destinations[i], payload.size()), payload);
    REQUIRE(cudaFree(destinations[i]) == cudaSuccess);
    REQUIRE(cudaStreamDestroy(streams[i]) == cudaSuccess);
  }
  REQUIRE(cudaSetDevice(previous) == cudaSuccess);
  CHECK(available_after_release(mr, capacity) == capacity);
}

namespace {

/// Raw HTTP server on 127.0.0.1: every request (a head without a body) gets @c response
/// verbatim.  One-shot mode closes the connection after it; keep-alive mode answers every request
/// of a connection.  For framings the loopback server never produces.
class raw_response_server {
 public:
  explicit raw_response_server(std::string response, bool keep_alive = false)
    : _response(std::move(response)), _keep_alive(keep_alive)
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
  ~raw_response_server()
  {
    _stop = true;
    ::shutdown(_fd, SHUT_RDWR);
    _thread.join();
    {
      std::lock_guard lock(_mutex);
      for (int const conn : _open) {
        ::shutdown(conn, SHUT_RDWR);
      }
    }
    for (auto& t : _connections) {
      t.join();
    }
    ::close(_fd);
  }
  raw_response_server(raw_response_server const&)            = delete;
  raw_response_server& operator=(raw_response_server const&) = delete;

  [[nodiscard]] std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(_port); }
  [[nodiscard]] std::size_t requests() const noexcept { return _requests.load(); }
  /// The most requests any one connection carried.
  [[nodiscard]] std::size_t max_requests_per_connection() const noexcept
  {
    return _max_per_connection.load();
  }

 private:
  void accept_loop()
  {
    while (!_stop) {
      int const conn = ::accept4(_fd, nullptr, nullptr, SOCK_CLOEXEC);
      if (conn < 0) {
        if (errno == EINTR) continue;
        return;
      }
      std::lock_guard lock(_mutex);
      _open.push_back(conn);
      _connections.emplace_back([this, conn] { serve(conn); });
    }
  }

  void serve(int conn)
  {
    std::string pending;
    std::size_t served = 0;
    char buf[4096];
    for (;;) {
      while (pending.find("\r\n\r\n") == std::string::npos) {
        auto const n = ::recv(conn, buf, sizeof buf, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
          ::close(conn);
          return;
        }
        pending.append(buf, static_cast<std::size_t>(n));
      }
      pending.erase(0, pending.find("\r\n\r\n") + 4);
      ++_requests;
      ++served;
      auto seen = _max_per_connection.load();
      while (served > seen && !_max_per_connection.compare_exchange_weak(seen, served)) {}
      std::size_t sent = 0;
      while (sent < _response.size()) {
        auto const n = ::send(conn, _response.data() + sent, _response.size() - sent, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        sent += static_cast<std::size_t>(n);
      }
      if (!_keep_alive) {
        ::shutdown(conn, SHUT_WR);
        std::this_thread::sleep_for(20ms);
        ::close(conn);
        return;
      }
    }
  }

  std::string _response;
  bool _keep_alive{false};
  int _fd{-1};
  std::uint16_t _port{0};
  std::atomic<bool> _stop{false};
  std::atomic<std::size_t> _requests{0};
  std::atomic<std::size_t> _max_per_connection{0};
  std::mutex _mutex;
  std::vector<int> _open;
  std::vector<std::thread> _connections;
  std::thread _thread;
};

}  // namespace

TEST_CASE("rest control-plane calls never report an incomplete 2xx body as success",
          "[rest][e2e][framing]")
{
  // A captured 2xx body (LIST pages, S3 write responses) must arrive whole: a close-delimited
  // body cannot be told from a cut, and a broken chunked body is incomplete.  Either fails the
  // call instead of handing back a prefix (a truncated CompleteMultipartUpload answer would
  // otherwise read as committed).
  auto const response = GENERATE(
    std::string{"HTTP/1.1 200 OK\r\nContent-Type: application/xml\r\n\r\n<ListBucketResult><Na"},
    std::string{"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n10\r\n<ListBucketResult>"
                "\r\nzz\r\n"});
  raw_response_server server(response);
  auto cfg               = rest_contract_config();
  cfg.max_retry_attempts = 2;
  auto authorizer        = std::make_shared<list_capable_mock_authorizer>(server.endpoint());
  auto context           = std::make_shared<rest_reactor::reactor_context>(cfg, authorizer);
  rest_reactor reactor(context);
  reactor.start();
  CHECK_THROWS(reactor.list_page("bucket", "prefix/", "list-type=2&prefix=prefix%2F"));
  CHECK(server.requests() >= 1);
  reactor.shutdown();
}

TEST_CASE("rest connections with bytes past the response are not reused", "[rest][e2e][framing]")
{
  // Bytes after a complete response (a HEAD answered with a body, a body longer than its
  // Content-Length, data after the last chunk) are dropped; the connection they came on must be
  // closed, not handed to the next request, which could read the rest as its own response.
  auto const [response, size] = GENERATE(
    std::pair{std::string{"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nETag: \"e\"\r\n\r\nhello"},
              std::size_t{5}},
    std::pair{std::string{"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nETag: \"e\"\r\n\r\nextra bytes"},
              std::size_t{0}});
  raw_response_server server(response, /*keep_alive=*/true);
  auto cfg        = rest_contract_config();
  auto authorizer = std::make_shared<list_capable_mock_authorizer>(server.endpoint());
  auto context    = std::make_shared<rest_reactor::reactor_context>(cfg, authorizer);
  rest_reactor reactor(context);
  reactor.start();
  for (int i = 0; i < 3; ++i) {
    CHECK(reactor.head_object("bucket", "key").object_size == size);
  }
  CHECK(server.requests() == 3);
  CHECK(server.max_requests_per_connection() == 1);
  reactor.shutdown();
}
