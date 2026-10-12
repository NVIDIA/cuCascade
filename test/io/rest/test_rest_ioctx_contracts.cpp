/*
 * Copyright 2026, Sirius Contributors.
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

#include "rest_contract_test_utils.hpp"
#include "utils/child_test_runner.hpp"

#include <cucascade/io/io_context.hpp>
#include <cucascade/io/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using cucascade::io::open_hint;
using cucascade::io::slice;
using cucascade::io::rest::rest_io_object;
using cucascade::test::child_test_runner;
using cucascade::test::loopback_range_server;
using cucascade::test::make_rest_contract_ioctx;
using cucascade::test::make_rest_contract_reactor;
using cucascade::test::range_fault_policy;
using cucascade::test::range_response_control;
using cucascade::test::require_rest_bytes_equal;
using cucascade::test::response_body_policy;
using cucascade::test::response_release_guard;
using cucascade::test::rest_contract_config;
using cucascade::test::rest_device_bytes;
using cucascade::test::rest_shutdown_guard;
using cucascade::test::rest_test_footer_bytes;
using cucascade::test::rest_test_parquet_bytes;
using cucascade::test::rest_test_payload;
using namespace std::chrono_literals;

struct server_stop_guard {
  loopback_range_server& server;
  ~server_stop_guard() { server.stop(); }
};

struct cuda_host_deleter {
  void operator()(std::uint8_t* bytes) const noexcept
  {
    if (bytes) { (void)cudaFreeHost(bytes); }
  }
};

class delayed_stream_gate {
 public:
  explicit delayed_stream_gate(cudaStream_t stream) : _stream(stream) {}
  ~delayed_stream_gate()
  {
    release();
    if (_timer.joinable()) { _timer.join(); }
    (void)cudaStreamSynchronize(_stream);
  }

  void arm()
  {
    REQUIRE(cudaLaunchHostFunc(
              _stream,
              [](void* opaque) {
                auto& gate = *static_cast<delayed_stream_gate*>(opaque);
                std::unique_lock lock{gate._mutex};
                gate._entered = true;
                gate._cv.notify_all();
                gate._cv.wait(lock, [&] { return gate._released; });
              },
              this) == cudaSuccess);
  }

  bool wait_entered(std::chrono::milliseconds timeout)
  {
    std::unique_lock lock{_mutex};
    return _cv.wait_for(lock, timeout, [&] { return _entered; });
  }

  void release_after(std::chrono::milliseconds delay)
  {
    _timer = std::jthread([this, delay] {
      std::unique_lock lock{_mutex};
      _cv.wait_for(lock, delay, [&] { return _released; });
      _released = true;
      lock.unlock();
      _cv.notify_all();
    });
  }

  void release()
  {
    {
      std::scoped_lock lock{_mutex};
      _released = true;
    }
    _cv.notify_all();
  }

 private:
  cudaStream_t _stream;
  std::mutex _mutex;
  std::condition_variable _cv;
  bool _entered{false};
  bool _released{false};
  std::jthread _timer;
};

void check_rest_pool(std::size_t reactors, std::size_t connections, bool multiple)
{
  auto payload = rest_test_payload(512 * 1024);
  auto control = std::make_shared<range_response_control>();
  range_fault_policy fault{};
  fault.body = response_body_policy{.hold_after_bytes = 0, .control = control};
  loopback_range_server server(payload, fault);
  auto config            = rest_contract_config();
  config.max_connections = connections;
  config.merge_max_gap   = 0;
  auto ioctx             = make_rest_contract_ioctx(server, config, reactors);
  auto object = ioctx->open_io_object("s3://concurrency-bucket/object.bin", payload.size());
  std::vector<std::vector<std::uint8_t>> buffers;
  std::vector<slice> segments;
  buffers.reserve(8);
  segments.reserve(8);
  for (std::size_t i = 0; i < 8; ++i) {
    buffers.emplace_back(512);
    segments.emplace_back(i * 4096 + 13, 512, buffers.back().data());
  }
  rest_shutdown_guard cleanup{*ioctx, control};
  auto future = ioctx->host_readv_async_io(*object, segments);
  REQUIRE(control->wait_for_holds(multiple ? 2 : 1, 5s));
  control->release_all();
  REQUIRE(std::move(future).get(10s) == 8 * 512);
  REQUIRE(server.wait_for_idle(5s));
  CHECK(server.peak_active_gets() <= reactors * connections);
  if (multiple) {
    CHECK(server.peak_active_gets() > 1);
  } else {
    CHECK(server.peak_active_gets() >= 1);
  }
  for (std::size_t i = 0; i < segments.size(); ++i) {
    require_rest_bytes_equal(
      buffers[i],
      std::span<std::uint8_t const>{payload.data() + segments[i].offset(), segments[i].size()});
  }
}

}  // namespace

TEST_CASE("rest_ioctx opens LIST-sized objects without a HEAD round trip", "[rest][data]")
{
  auto payload = rest_test_payload(4096);
  loopback_range_server server(payload);
  auto context = make_rest_contract_ioctx(server);
  auto object  = context->open_io_object("s3://bucket/list-sized.bin", payload.size());
  REQUIRE(object != nullptr);
  CHECK(object->size() == payload.size());
  CHECK(server.head_count() == 0);
  std::vector<std::uint8_t> out(payload.size());
  rest_shutdown_guard cleanup{*context};
  CHECK(context->host_read(*object, 0, out.size(), out.data()) == out.size());
  CHECK(out == payload);
  CHECK(server.get_count() == 1);
}

TEST_CASE("rest reactor dynamically splits one logical read below the physical GET maximum",
          "[rest][data]")
{
  constexpr std::size_t max_rest_segment = 16UL << 20;
  auto payload                           = rest_test_payload((33UL << 20) + 17);
  loopback_range_server server(payload);
  auto config            = rest_contract_config();
  config.max_connections = 4;
  auto ioctx             = make_rest_contract_ioctx(server, config);
  auto object            = ioctx->open_io_object("s3://split-bucket/object.bin", payload.size());
  std::vector<std::uint8_t> destination(payload.size());
  std::vector<slice> slices{slice{0, destination.size(), destination.data()}};
  rest_shutdown_guard cleanup{*ioctx};
  auto future = ioctx->host_readv_async_io(*object, slices);
  REQUIRE(std::move(future).get(10s) == payload.size());
  CHECK(server.get_count() > 1);
  auto const requests = server.get_requests();
  REQUIRE(requests.size() > 1);
  for (auto const& request : requests) {
    REQUIRE(request.ranges.size() == 1);
    auto const& range = request.ranges.front();
    REQUIRE(range.starts_with("bytes="));
    auto const dash = range.find('-', 6);
    REQUIRE(dash != std::string::npos);
    REQUIRE(dash > 6);
    auto const first = std::stoull(range.substr(6, dash - 6));
    auto const last  = std::stoull(range.substr(dash + 1));
    REQUIRE(last >= first);
    CHECK(last - first + 1 <= max_rest_segment);
  }
  require_rest_bytes_equal(destination, payload);
}

TEST_CASE("rest data GET outlives request_timeout_s on a slow but healthy link", "[rest][data]")
{
  auto payload = rest_test_payload(2UL << 20);
  auto control = std::make_shared<range_response_control>();
  range_fault_policy fault{};
  fault.body =
    response_body_policy{.chunk_bytes = 128UL << 10, .chunk_delay = 100ms, .control = control};
  loopback_range_server server(payload, fault);
  auto config                    = rest_contract_config();
  config.max_connections         = 1;
  config.request_timeout_s       = 1;
  config.stall_speed_limit_bytes = 1024;
  config.stall_time_s            = 5;
  auto ioctx                     = make_rest_contract_ioctx(server, config);
  auto object = ioctx->open_io_object("s3://slow-link-bucket/object.bin", payload.size());
  std::vector<std::uint8_t> destination(payload.size());
  std::vector<slice> slices{slice{0, destination.size(), destination.data()}};
  rest_shutdown_guard cleanup{*ioctx, control};
  auto const started = std::chrono::steady_clock::now();
  auto future        = ioctx->host_readv_async_io(*object, slices);
  REQUIRE(control->wait_for_headers(1, 5s));
  REQUIRE(control->wait_for_body_bytes(128UL << 10, 5s));
  REQUIRE(std::move(future).get(60s) == payload.size());
  CHECK(std::chrono::steady_clock::now() - started > 1s);
  CHECK(server.get_count() == 1);
  require_rest_bytes_equal(destination, payload);
}

TEST_CASE("rest data GET on a stalled connection fails under the stall detector", "[rest][data]")
{
  auto payload = rest_test_payload(2UL << 20);
  auto control = std::make_shared<range_response_control>();
  range_fault_policy fault{};
  fault.body = response_body_policy{.hold_after_bytes = 0, .control = control};
  loopback_range_server server(payload, fault);
  auto config                    = rest_contract_config();
  config.max_connections         = 1;
  config.request_timeout_s       = 0;
  config.stall_speed_limit_bytes = 1024;
  config.stall_time_s            = 1;
  config.max_retry_attempts      = 2;
  auto ioctx                     = make_rest_contract_ioctx(server, config);
  auto object = ioctx->open_io_object("s3://stalled-bucket/object.bin", payload.size());
  std::vector<std::uint8_t> destination(payload.size());
  std::vector<slice> slices{slice{0, destination.size(), destination.data()}};
  rest_shutdown_guard cleanup{*ioctx, control};
  auto const started = std::chrono::steady_clock::now();
  auto future        = ioctx->host_readv_async_io(*object, slices);
  REQUIRE(control->wait_for_holds(1, 5s));
  CHECK_THROWS(std::move(future).get(60s));
  CHECK(std::chrono::steady_clock::now() - started < 15s);
  CHECK(server.get_count() == 2);
  CHECK(control->snapshot().headers == 2);
  CHECK(control->snapshot().body_bytes == 0);
}

TEST_CASE("rest footer suffix parses Content-Range totals", "[rest][footer]")
{
  using cucascade::io::rest::content_range_total;
  CHECK(content_range_total("bytes 42-99/12345") == std::optional<std::size_t>{12345});
  CHECK(content_range_total("Bytes 0-7/8") == std::optional<std::size_t>{8});
  CHECK_FALSE(content_range_total("bytes 42-99/*").has_value());
  CHECK_FALSE(content_range_total("bytes */12345").has_value());
  CHECK_FALSE(content_range_total("not-a-content-range").has_value());
}

TEST_CASE("rest footer suffix probe discovers size without HEAD", "[rest][footer]")
{
  auto const parquet = rest_test_parquet_bytes();
  loopback_range_server server(parquet);
  auto config               = rest_contract_config();
  config.max_retry_attempts = 1;
  auto reactor              = make_rest_contract_reactor(server, config);
  auto probe = reactor->fetch_footer_suffix("footer-bucket", "nation.parquet", 1UL << 20);
  CHECK(probe.object_size == parquet.size());
  CHECK(probe.window_lo == 0);
  REQUIRE(probe.bytes != nullptr);
  CHECK(probe.bytes->size() == parquet.size());
  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == 1);
}

TEST_CASE("footer outside the suffix window falls back to one body GET", "[rest][footer]")
{
  auto const parquet                 = rest_test_parquet_bytes();
  auto const footer_len              = rest_test_footer_bytes;
  auto const footer_off              = parquet.size() - 8 - footer_len;
  constexpr std::size_t suffix_bytes = 8;
  REQUIRE(footer_len + 8 > suffix_bytes);
  loopback_range_server server(parquet);
  auto config               = rest_contract_config();
  config.max_retry_attempts = 1;
  auto reactor              = make_rest_contract_reactor(server, config);
  reactor->start();
  auto probe = reactor->fetch_footer_suffix("footer-bucket", "nation.parquet", suffix_bytes);
  CHECK(probe.object_size == parquet.size());
  CHECK(probe.window_lo == parquet.size() - suffix_bytes);
  REQUIRE(probe.bytes != nullptr);
  auto object = std::make_shared<rest_io_object>("s3://footer-bucket/nation.parquet",
                                                 "footer-bucket",
                                                 "nation.parquet",
                                                 probe.object_size,
                                                 probe.window_lo,
                                                 probe.bytes);
  CHECK(server.get_count() == 1);
  std::vector<std::uint8_t> footer(footer_len);
  rest_shutdown_guard cleanup{*reactor};
  REQUIRE(reactor->host_read(*object, footer_off, footer.size(), footer.data()) == footer.size());
  require_rest_bytes_equal(footer, {parquet.data() + footer_off, footer_len});
  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == 2);
}

TEST_CASE("footer suffix probe falls back safely on unusable suffix responses", "[rest][footer]")
{
  auto const parquet = rest_test_parquet_bytes();
  range_fault_policy fault{};
  bool large       = false;
  int error_status = 0;
  SECTION("missing Content-Range") { fault.omit_successful_content_range = true; }
  SECTION("unknown Content-Range total")
  {
    fault.content_range = "bytes 0-" + std::to_string(parquet.size() - 1) + "/*";
  }
  SECTION("server ignores Range with 200 full-body")
  {
    fault.full_object_with_200 = true;
    fault.successful_get_etag  = "\"discarded-200-tag\"";
  }
  SECTION("server ignores Range with 200 full-body on a large object")
  {
    large                      = true;
    fault.full_object_with_200 = true;
    fault.body.chunk_bytes     = 64UL << 10;
  }
  SECTION("suffix 416 falls back to HEAD") { fault.fail_range_with_416 = true; }
  SECTION("missing object fails the footer-probe open after HEAD fallback") { error_status = 404; }
  SECTION("forbidden object fails the footer-probe open after HEAD fallback")
  {
    error_status = 403;
  }
  if (error_status != 0) {
    fault.fail_all_gets    = true;
    fault.fail_all_heads   = true;
    fault.fail_status      = error_status;
    fault.head_fail_status = error_status;
  }
  auto const payload = large ? rest_test_payload(8UL << 20) : parquet;
  loopback_range_server server(payload, fault);
  auto ioctx = make_rest_contract_ioctx(server);
  if (error_status != 0) {
    CHECK_THROWS_WITH(
      ioctx->open_io_object("s3://footer-bucket/object.parquet", open_hint::parquet_footer_probe),
      Catch::Matchers::ContainsSubstring("HTTP " + std::to_string(error_status)));
  } else {
    auto object =
      ioctx->open_io_object("s3://footer-bucket/object.parquet", open_hint::parquet_footer_probe);
    REQUIRE(object != nullptr);
    CHECK(object->size() == payload.size());
    if (!fault.successful_get_etag.empty()) { CHECK(object->validation_tag().empty()); }
    CHECK(server.head_count() == 1);
    CHECK(server.get_count() == 1);
    if (large) {
      server.stop();
      CHECK(server.body_bytes_sent() < payload.size());
    }
  }
}

TEST_CASE("footer suffix probe retries transient GET failures", "[rest][footer]")
{
  auto const parquet        = rest_test_parquet_bytes();
  auto config               = rest_contract_config();
  config.max_retry_attempts = 3;
  range_fault_policy fault{};
  std::size_t gets = 1;
  std::string expected_tag;
  std::string error;
  bool exhausted = false;
  SECTION("transient 503s are retried and the final 206 produces a probe")
  {
    fault.fail_first_gets = 2;
    fault.fail_status     = 503;
    fault.failed_get_etag = "\"stale-retry-tag\"";
    gets                  = 3;
  }
  SECTION("exhausted transient 503s throw after the retry budget")
  {
    fault.fail_all_gets       = true;
    fault.fail_status         = 503;
    config.max_retry_attempts = 2;
    gets                      = 2;
    exhausted                 = true;
    error                     = "HTTP 503";
  }
  SECTION("hard non-retriable errors fail without retrying")
  {
    fault.fail_all_gets = true;
    fault.fail_status   = 403;
    error               = "HTTP 403";
  }
  SECTION("clean 206 succeeds without retries")
  {
    fault.successful_get_etag = "\"footer-v1\"";
    expected_tag              = fault.successful_get_etag;
  }
  loopback_range_server server(parquet, fault);
  auto reactor = make_rest_contract_reactor(server, config);
  if (!error.empty()) {
    try {
      (void)reactor->fetch_footer_suffix("footer-bucket", "nation.parquet", 1UL << 20);
      FAIL("fetch_footer_suffix should throw on the scripted failure");
    } catch (std::runtime_error const& exception) {
      auto const message = std::string{exception.what()};
      CHECK(message.find(error) != std::string::npos);
      if (exhausted) { CHECK(message.find("exhausted retries") != std::string::npos); }
    }
  } else {
    auto probe = reactor->fetch_footer_suffix("footer-bucket", "nation.parquet", 1UL << 20);
    CHECK(probe.object_size == parquet.size());
    CHECK(probe.window_lo == 0);
    CHECK(probe.etag == expected_tag);
    REQUIRE(probe.bytes != nullptr);
    CHECK(probe.bytes->size() == parquet.size());
  }
  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == gets);
}

TEST_CASE("rest_reactor HEAD retries transient 503s and stops on exhaustion or a hard error",
          "[rest][footer]")
{
  auto const payload        = rest_test_payload(4096);
  auto config               = rest_contract_config();
  config.max_retry_attempts = 3;
  range_fault_policy fault{};
  std::size_t heads = 1;
  std::string expected_tag;
  std::string error;
  bool exhausted = false;
  SECTION("transient 503s are retried and the final HEAD returns the size")
  {
    fault.fail_first_heads     = 2;
    fault.head_fail_status     = 503;
    fault.successful_head_etag = "\"head-v2\"";
    expected_tag               = fault.successful_head_etag;
    heads                      = 3;
  }
  SECTION("exhausted transient 503s are reported as one terminal failure")
  {
    fault.fail_all_heads      = true;
    fault.head_fail_status    = 503;
    config.max_retry_attempts = 2;
    heads                     = 2;
    exhausted                 = true;
    error                     = "HTTP 503";
  }
  SECTION("hard non-retriable HEAD errors fail without retrying")
  {
    fault.fail_all_heads   = true;
    fault.head_fail_status = 403;
    error                  = "HTTP 403";
  }
  SECTION("clean HEAD succeeds without retries") {}
  loopback_range_server server(payload, fault);
  auto reactor = make_rest_contract_reactor(server, config);
  if (!error.empty()) {
    try {
      (void)reactor->head_object("head-bucket", "object.bin");
      FAIL("head_object should throw on the scripted failure");
    } catch (std::runtime_error const& exception) {
      auto const message = std::string{exception.what()};
      CHECK(message.find(error) != std::string::npos);
      if (exhausted) { CHECK(message.find("exhausted retries") != std::string::npos); }
    }
  } else {
    auto const result = reactor->head_object("head-bucket", "object.bin");
    CHECK(result.object_size == payload.size());
    CHECK(result.etag == expected_tag);
  }
  CHECK(server.head_count() == heads);
  CHECK(server.get_count() == 0);
}

TEST_CASE("small objects can be resolved by one suffix probe", "[rest][footer]")
{
  auto payload = rest_test_payload(512);
  loopback_range_server server(payload);
  auto config               = rest_contract_config();
  config.max_retry_attempts = 1;
  auto reactor              = make_rest_contract_reactor(server, config);
  auto probe                = reactor->fetch_footer_suffix("footer-bucket", "small.bin", 1UL << 20);
  CHECK(probe.object_size == payload.size());
  CHECK(probe.window_lo == 0);
  REQUIRE(probe.bytes != nullptr);
  CHECK(probe.bytes->size() == payload.size());
  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == 1);
}

TEST_CASE("concurrent footer probes each get an object-local suffix stash", "[rest][footer]")
{
  auto const parquet = rest_test_parquet_bytes();
  loopback_range_server server(parquet);
  auto config               = rest_contract_config();
  config.max_retry_attempts = 1;
  auto ioctx                = make_rest_contract_ioctx(server, config);
  std::vector<std::future<std::size_t>> futures;
  server_stop_guard cleanup{server};
  for (int i = 0; i < 8; ++i) {
    futures.push_back(std::async(std::launch::async, [ioctx] {
      auto object =
        ioctx->open_io_object("s3://footer-bucket/nation.parquet", open_hint::parquet_footer_probe);
      return object->size();
    }));
  }
  for (auto& future : futures) {
    REQUIRE(future.wait_for(10s) == std::future_status::ready);
    CHECK(future.get() == parquet.size());
  }
  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == futures.size());
}

TEST_CASE("rest_ioctx fans out host_readv_async_io ranges against the SeaweedFS medium fixture",
          "[rest][data]")
{
  auto const medium = rest_test_payload(4UL << 20);
  loopback_range_server server(medium);
  auto config            = rest_contract_config();
  config.max_connections = 4;
  config.merge_max_gap   = 0;
  auto ioctx             = make_rest_contract_ioctx(server, config);
  auto object            = ioctx->open_io_object("s3://bucket/medium.bin", medium.size());
  std::vector<std::pair<std::size_t, std::size_t>> const ranges{
    {17, 257}, {64 * 1024 + 9, 1024}, {2 * 1024 * 1024 + 11, 4096}, {medium.size() - 333, 333}};
  std::vector<std::vector<std::uint8_t>> buffers;
  std::vector<slice> segments;
  buffers.reserve(ranges.size());
  segments.reserve(ranges.size());
  std::size_t total = 0;
  for (auto const& [offset, size] : ranges) {
    buffers.emplace_back(size);
    segments.emplace_back(offset, size, buffers.back().data());
    total += size;
  }
  rest_shutdown_guard cleanup{*ioctx};
  REQUIRE(std::move(ioctx->host_readv_async_io(*object, segments)).get(5s) == total);
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    auto const [offset, size] = ranges[i];
    require_rest_bytes_equal(buffers[i], {medium.data() + offset, size});
  }
}

TEST_CASE("rest_ioctx keeps caller host buffers until the device event completes",
          "[rest][data][gpu]")
{
  int count = 0;
  REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
  if (count == 0) { SKIP("requires a CUDA device"); }
  auto payload = rest_test_payload(256 * 1024);
  loopback_range_server server(payload);
  auto ioctx     = make_rest_contract_ioctx(server);
  auto object    = ioctx->open_io_object("s3://device-event-bucket/object.bin", payload.size());
  void* raw_host = nullptr;
  REQUIRE(cudaMallocHost(&raw_host, payload.size()) == cudaSuccess);
  std::unique_ptr<std::uint8_t, cuda_host_deleter> host{static_cast<std::uint8_t*>(raw_host)};
  rmm::cuda_stream stream;
  rmm::device_buffer destination(payload.size(), stream);
  rest_shutdown_guard cleanup{*ioctx};
  delayed_stream_gate gate{stream.value()};
  gate.arm();
  REQUIRE(gate.wait_entered(5s));
  gate.release_after(500ms);
  std::vector<cucascade::io::prepared_io_slice> slices;
  slices.emplace_back(
    cucascade::io::range{0, payload.size()},
    cucascade::io::host_buffer{host.get()},
    cucascade::io::device_buffer{static_cast<std::uint8_t*>(destination.data()), stream});
  auto future = ioctx->host_device_readv_async_io(*object, std::move(slices));
  REQUIRE(std::move(future).get(5s) == payload.size());
  std::memset(host.get(), 0, payload.size());
  auto got = rest_device_bytes(destination.data(), payload.size(), stream.value());
  require_rest_bytes_equal(got, payload);
}

TEST_CASE("rest_ioctx retries transient fake-server failures and reports terminal failures",
          "[rest][data]")
{
  auto payload = rest_test_payload(64 * 1024);
  range_fault_policy fault{};
  auto config    = rest_contract_config();
  bool exhausted = false;
  SECTION("transient 503s are retried")
  {
    fault.fail_first_gets     = 2;
    fault.fail_status         = 503;
    config.max_retry_attempts = 4;
  }
  SECTION("exhausted retries surface as an error")
  {
    fault.fail_all_gets       = true;
    fault.fail_status         = 503;
    config.max_retry_attempts = 2;
    exhausted                 = true;
  }
  loopback_range_server server(payload, fault);
  auto ioctx  = make_rest_contract_ioctx(server, config);
  auto object = ioctx->open_io_object("s3://retry-bucket/object.bin");
  std::vector<std::uint8_t> bytes(exhausted ? 128 : 4096);
  rest_shutdown_guard cleanup{*ioctx};
  if (exhausted) {
    CHECK_THROWS(ioctx->host_read(*object, 0, bytes.size(), bytes.data()));
    CHECK(server.get_count() >= 2);
  } else {
    REQUIRE(ioctx->host_read(*object, 128, bytes.size(), bytes.data()) == bytes.size());
    require_rest_bytes_equal(bytes, {payload.data() + 128, bytes.size()});
    CHECK(server.get_count() >= 3);
  }
}

TEST_CASE("rest_ioctx honors max_connections under concurrent fake range reads", "[rest][data]")
{
  check_rest_pool(1, 2, false);
}

TEST_CASE("rest_ioctx spreads one batched range read across the reactor pool", "[rest][data]")
{
  check_rest_pool(4, 1, true);
}

TEST_CASE("rest_ioctx teardown resolves an in-flight async read without hanging",
          "[rest][lifecycle]")
{
  if (auto const* child = std::getenv("CUCASCADE_REST_TEARDOWN_CHILD");
      child == nullptr || std::string_view{child} != "1") {
    child_test_runner process{
      "rest_ioctx teardown resolves an in-flight async read without hanging",
      {{"CUCASCADE_REST_TEARDOWN_CHILD", std::string{"1"}}}};
    bool const completed = process.wait_for(30s);
    if (!completed) { process.stop(); }
    INFO(process.output());
    REQUIRE(completed);
    REQUIRE(WIFEXITED(process.status()));
    CHECK(WEXITSTATUS(process.status()) == 0);
    return;
  }

  auto payload = rest_test_payload(128 * 1024);
  auto control = std::make_shared<range_response_control>();
  range_fault_policy fault{};
  fault.body             = response_body_policy{.hold_after_bytes = 0, .control = control};
  auto server            = std::make_shared<loopback_range_server>(payload, fault);
  auto buffer            = std::make_shared<std::vector<std::uint8_t>>(4096);
  auto config            = rest_contract_config();
  config.max_connections = 1;
  auto context           = make_rest_contract_ioctx(*server, config);
  auto object = context->open_io_object("s3://lifecycle-bucket/object.bin", payload.size());
  response_release_guard initial_release{control};
  auto future = context->host_read_async(*object, 0, buffer->size(), buffer->data());
  REQUIRE(control->wait_for_holds(1, 5s));
  std::promise<void> completed;
  auto teardown = completed.get_future();
  std::jthread worker([context = std::move(context),
                       object  = std::move(object),
                       server,
                       buffer,
                       completed = std::move(completed)]() mutable {
    object.reset();
    context.reset();
    completed.set_value();
  });
  response_release_guard release_before_join{control};
  REQUIRE(teardown.wait_for(10s) == std::future_status::ready);
  worker.join();
  REQUIRE(future.valid());
  REQUIRE(std::move(future).wait(5s));
  try {
    auto const bytes = std::move(future).get();
    if (bytes != 0) {
      REQUIRE(bytes == buffer->size());
      require_rest_bytes_equal(*buffer, {payload.data(), buffer->size()});
    }
  } catch (std::exception const& error) {
    INFO("teardown completed in-flight request with exception: " << error.what());
    SUCCEED("future resolved with an exception during teardown");
  }
}
