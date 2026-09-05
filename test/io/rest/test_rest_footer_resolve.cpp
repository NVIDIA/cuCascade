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

#include <cucascade/exec/scoped_dispatcher.hpp>
#include <cucascade/exec/thread_pool.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using cucascade::exec::scoped_dispatcher;
using cucascade::exec::static_thread_pool;
using cucascade::io::open_hint;
using cucascade::io::rest::config;
using cucascade::io::rest::footer_resolve_result;
using cucascade::io::rest::rest_io_object;
using cucascade::io::rest::rest_ioctx;
using cucascade::io::rest::rest_perf_snapshot;
using cucascade::io::rest::rest_reactor;
using cucascade::io::rest::shared_byte_span;
using cucascade::test::key_response_script;
using cucascade::test::list_capable_mock_authorizer;
using cucascade::test::loopback_range_server;
using cucascade::test::range_fault_policy;
using cucascade::test::scripted_response;
using namespace std::chrono_literals;

constexpr std::size_t object_size{4096};
constexpr std::size_t probe_size{256};

std::vector<std::uint8_t> test_payload()
{
  std::vector<std::uint8_t> bytes(object_size);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>((i * 73U + 19U) & 0xffU);
  }
  return bytes;
}

std::string object_uri(std::string_view key) { return "s3://bucket/" + std::string{key}; }

config test_config(std::size_t max_inflight = 2,
                   std::size_t budget       = 4 * probe_size,
                   bool instrumentation     = true)
{
  config cfg{};
  cfg.request_timeout_s           = 5;
  cfg.tls_verify                  = false;
  cfg.max_connections             = 2;
  cfg.max_retry_attempts          = 3;
  cfg.max_auth_retry_attempts     = 1;
  cfg.retry_backoff_base          = 10ms;
  cfg.retry_jitter                = 0ms;
  cfg.honor_retry_after           = false;
  cfg.perf_instrumentation        = instrumentation;
  cfg.footer_probe_bytes          = probe_size;
  cfg.footer_resolve_max_inflight = max_inflight;
  cfg.footer_resolve_stash_budget = budget;
  return cfg;
}

struct footer_fixture {
  std::shared_ptr<list_capable_mock_authorizer> authorizer;
  std::shared_ptr<rest_ioctx> ioctx;
};

footer_fixture make_ioctx(loopback_range_server const& server, config cfg, std::size_t reactors = 1)
{
  auto authorizer = std::make_shared<list_capable_mock_authorizer>(server.endpoint());
  auto context    = std::make_shared<rest_reactor::reactor_context>(cfg, authorizer, nullptr);
  auto ioctx      = std::make_shared<rest_ioctx>(reactors, std::move(context));
  return {std::move(authorizer), std::move(ioctx)};
}

std::vector<footer_resolve_result> resolve(rest_ioctx& ioctx,
                                           std::vector<std::string> const& paths,
                                           std::stop_token stop = {})
{
  std::vector<footer_resolve_result> results;
  results.reserve(paths.size());
  ioctx.resolve_footer_objects(
    paths, [&](footer_resolve_result result) { results.push_back(std::move(result)); }, stop);
  return results;
}

footer_resolve_result const& result_at(std::vector<footer_resolve_result> const& results,
                                       std::size_t index)
{
  auto const found = std::find_if(
    results.begin(), results.end(), [index](auto const& result) { return result.index == index; });
  REQUIRE(found != results.end());
  return *found;
}

void require_success(footer_resolve_result const& result)
{
  CHECK_FALSE(result.error);
  REQUIRE(result.object != nullptr);
}

void require_failure(footer_resolve_result const& result)
{
  REQUIRE(result.error != nullptr);
  CHECK(result.object == nullptr);
  CHECK_FALSE(result.footer);
}

bool is_success(footer_resolve_result const& result)
{
  return !result.error && result.object != nullptr;
}

void require_span_equal(shared_byte_span const& lhs, shared_byte_span const& rhs)
{
  REQUIRE(static_cast<bool>(lhs) == static_cast<bool>(rhs));
  if (!lhs) { return; }
  REQUIRE(lhs->size() == rhs->size());
  CHECK(std::equal(lhs->begin(), lhs->end(), rhs->begin(), rhs->end()));
}

bool is_operation_canceled(std::exception_ptr const& error)
{
  if (!error) { return false; }
  try {
    std::rethrow_exception(error);
  } catch (std::system_error const& e) {
    return e.code() == std::make_error_code(std::errc::operation_canceled);
  } catch (...) {
    return false;
  }
}

template <typename Predicate>
bool wait_until(Predicate&& predicate, std::chrono::milliseconds timeout)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) { return true; }
    std::this_thread::sleep_for(1ms);
  }
  return predicate();
}

class async_call {
 public:
  template <typename Function>
  explicit async_call(Function function)
  {
    auto promise = std::make_shared<std::promise<std::exception_ptr>>();
    _done        = promise->get_future();
    _worker      = std::thread([promise, function = std::move(function)]() mutable {
      try {
        function();
        promise->set_value(nullptr);
      } catch (...) {
        promise->set_value(std::current_exception());
      }
    });
  }

  ~async_call()
  {
    if (_worker.joinable()) { _worker.detach(); }
  }

  async_call(async_call const&)            = delete;
  async_call& operator=(async_call const&) = delete;

  [[nodiscard]] bool ready_within(std::chrono::milliseconds timeout)
  {
    return _done.wait_for(timeout) == std::future_status::ready;
  }

  std::exception_ptr finish()
  {
    auto error = _done.get();
    _worker.join();
    return error;
  }

 private:
  std::future<std::exception_ptr> _done;
  std::thread _worker;
};

void require_ready(async_call& call, std::chrono::milliseconds timeout = 3s)
{
  REQUIRE(call.ready_within(timeout));
  auto const error = call.finish();
  if (error) { std::rethrow_exception(error); }
}

struct payload_gate {
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<shared_byte_span> payloads;
  std::size_t resident_bytes{0};
  std::size_t peak_resident_bytes{0};

  void retain(shared_byte_span payload)
  {
    std::scoped_lock lock{mutex};
    resident_bytes += payload ? payload->size() : 0;
    peak_resident_bytes = std::max(peak_resident_bytes, resident_bytes);
    payloads.push_back(std::move(payload));
    cv.notify_one();
  }

  bool wait_for_payload(std::chrono::milliseconds timeout)
  {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, timeout, [&] { return !payloads.empty(); });
  }

  void release_one()
  {
    shared_byte_span payload;
    {
      std::scoped_lock lock{mutex};
      if (payloads.empty()) { return; }
      payload = std::move(payloads.front());
      payloads.erase(payloads.begin());
      resident_bytes -= payload ? payload->size() : 0;
    }
    payload.reset();
    cv.notify_all();
  }

  void release_all()
  {
    while (true) {
      {
        std::scoped_lock lock{mutex};
        if (payloads.empty()) { return; }
      }
      release_one();
    }
  }

  [[nodiscard]] std::size_t resident()
  {
    std::scoped_lock lock{mutex};
    return resident_bytes;
  }

  [[nodiscard]] std::size_t peak()
  {
    std::scoped_lock lock{mutex};
    return peak_resident_bytes;
  }
};

rest_perf_snapshot snapshot_after_single_opens(loopback_range_server const& server,
                                               config cfg,
                                               std::vector<std::string> const& paths)
{
  auto fixture = make_ioctx(server, cfg);
  for (auto const& path : paths) {
    try {
      (void)fixture.ioctx->open_io_object(path, open_hint::parquet_footer_probe);
    } catch (...) {
    }
  }
  return fixture.ioctx->perf_snapshot();
}

class per_key_authorizer_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace

TEST_CASE("batched footer resolve matches single-probe bytes, size, and validation tag",
          "[rest][footer_resolve]")
{
  SECTION("verified 206")
  {
    range_fault_policy fault{};
    fault.successful_get_etag = "W/\"footer-v1\"";
    loopback_range_server server(test_payload(), fault);
    auto fixture    = make_ioctx(server, test_config());
    auto const path = object_uri("verified.parquet");

    auto single_base = fixture.ioctx->open_io_object(path, open_hint::parquet_footer_probe);
    auto single      = std::dynamic_pointer_cast<rest_io_object>(single_base);
    REQUIRE(single != nullptr);

    auto const results = resolve(*fixture.ioctx, {path});
    auto const& batch  = result_at(results, 0);
    require_success(batch);
    auto batch_object = std::dynamic_pointer_cast<rest_io_object>(batch.object);
    REQUIRE(batch_object != nullptr);

    CHECK(batch.path == path);
    CHECK(batch.object->size() == single->size());
    CHECK(batch.object->validation_tag() == single->validation_tag());
    CHECK(batch.window_lo == single->stash_window_lo());
    require_span_equal(batch.footer, single->stash());
    CHECK_FALSE(batch_object->stash());
  }

  SECTION("unusable probe falls back to HEAD")
  {
    range_fault_policy fault{};
    fault.malformed_content_range = true;
    fault.failed_get_etag         = "\"discarded\"";
    fault.successful_head_etag    = "\"head-v1\"";
    loopback_range_server server(test_payload(), fault);
    auto fixture    = make_ioctx(server, test_config());
    auto const path = object_uri("fallback.parquet");

    auto single_base = fixture.ioctx->open_io_object(path, open_hint::parquet_footer_probe);
    auto single      = std::dynamic_pointer_cast<rest_io_object>(single_base);
    REQUIRE(single != nullptr);

    auto const results = resolve(*fixture.ioctx, {path});
    auto const& batch  = result_at(results, 0);
    require_success(batch);
    auto batch_object = std::dynamic_pointer_cast<rest_io_object>(batch.object);
    REQUIRE(batch_object != nullptr);

    CHECK(batch.path == path);
    CHECK(batch.object->size() == single->size());
    CHECK(batch.object->validation_tag() == single->validation_tag());
    CHECK(batch.object->validation_tag() == "\"head-v1\"");
    CHECK(batch.window_lo == single->stash_window_lo());
    require_span_equal(batch.footer, single->stash());
    CHECK_FALSE(batch_object->stash());
    CHECK(server.get_count("fallback.parquet") == 2);
    CHECK(server.head_count("fallback.parquet") == 2);
  }
}

TEST_CASE("batched footer resolve uses HEAD only when the probe window is zero",
          "[rest][footer_resolve]")
{
  constexpr std::size_t path_count = 3;
  range_fault_policy fault{};
  fault.successful_head_etag = "\"head-only\"";
  loopback_range_server server(test_payload(), fault);
  auto cfg               = test_config(path_count);
  cfg.footer_probe_bytes = 0;
  auto fixture           = make_ioctx(server, cfg);
  std::vector<std::string> paths;
  paths.reserve(path_count);
  for (std::size_t i = 0; i < path_count; ++i) {
    paths.push_back(object_uri("head-only-" + std::to_string(i) + ".parquet"));
  }

  std::vector<footer_resolve_result> results;
  results.reserve(paths.size());
  std::uint64_t max_reserved = 0;
  fixture.ioctx->resolve_footer_objects(paths, [&](footer_resolve_result result) {
    max_reserved =
      std::max(max_reserved, fixture.ioctx->perf_snapshot().footer_stash_reserved_bytes);
    results.push_back(std::move(result));
  });

  REQUIRE(results.size() == paths.size());
  for (std::size_t i = 0; i < paths.size(); ++i) {
    auto const& result = result_at(results, i);
    require_success(result);
    CHECK(result.object->size() == object_size);
    CHECK(result.object->validation_tag() == "\"head-only\"");
    CHECK_FALSE(result.footer);
    CHECK(result.window_lo == 0);
    CHECK(server.get_count("head-only-" + std::to_string(i) + ".parquet") == 0);
    CHECK(server.head_count("head-only-" + std::to_string(i) + ".parquet") == 1);
  }
  CHECK(server.get_count() == 0);
  CHECK(server.head_count() == path_count);
  CHECK(max_reserved == 0);
  CHECK(fixture.ioctx->perf_snapshot().footer_stash_reserved_bytes == 0);
}

TEST_CASE("batched footer resolve isolates per-object authorization and not-found errors",
          "[rest][footer_resolve]")
{
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["missing.parquet"].gets = {scripted_response{.status = 404}};
  scripts["denied.parquet"].gets  = {scripted_response{.status = 403}};
  loopback_range_server server(test_payload(), {}, {}, std::move(scripts));
  auto fixture = make_ioctx(server, test_config(3));
  std::vector<std::string> paths{object_uri("a.parquet"),
                                 object_uri("missing.parquet"),
                                 object_uri("b.parquet"),
                                 object_uri("denied.parquet")};

  auto const results = resolve(*fixture.ioctx, paths);

  REQUIRE(results.size() == paths.size());
  for (std::size_t index = 0; index < paths.size(); ++index) {
    auto const& result = result_at(results, index);
    CHECK(result.path == paths[index]);
    if (index == 1 || index == 3) {
      require_failure(result);
    } else {
      require_success(result);
    }
  }
  CHECK(server.get_count("a.parquet") == 1);
  CHECK(server.get_count("missing.parquet") == 1);
  CHECK(server.get_count("b.parquet") == 1);
  CHECK(server.get_count("denied.parquet") == 1);
}

TEST_CASE("batched footer resolve releases malformed probe buffers before HEAD fallback",
          "[rest][footer_resolve]")
{
  constexpr std::size_t malformed_count = 6;
  std::unordered_map<std::string, key_response_script> scripts;
  std::vector<std::string> paths;
  for (std::size_t i = 0; i < malformed_count; ++i) {
    auto key          = "malformed-" + std::to_string(i) + ".parquet";
    scripts[key].gets = {scripted_response{.malformed_content_range = true}};
    paths.push_back(object_uri(key));
  }
  paths.push_back(object_uri("normal.parquet"));

  loopback_range_server server(test_payload(), {}, {}, std::move(scripts));
  auto fixture = make_ioctx(server, test_config(2, 2 * probe_size));
  auto results = resolve(*fixture.ioctx, paths);

  REQUIRE(results.size() == paths.size());
  for (std::size_t i = 0; i < malformed_count; ++i) {
    auto const& result = result_at(results, i);
    require_success(result);
    CHECK_FALSE(result.footer);
    auto const key = "malformed-" + std::to_string(i) + ".parquet";
    CHECK(server.get_count(key) == 1);
    CHECK(server.head_count(key) == 1);
  }
  auto const& normal = result_at(results, malformed_count);
  require_success(normal);
  REQUIRE(normal.footer);
  CHECK(normal.footer->size() == probe_size);
  CHECK(server.get_count("normal.parquet") == 1);
  CHECK(server.head_count("normal.parquet") == 0);

  for (auto& result : results) {
    result.footer.reset();
  }
  CHECK(fixture.ioctx->perf_snapshot().footer_stash_reserved_bytes == 0);
}

TEST_CASE("batched footer resolve isolates a per-key authorizer exception",
          "[rest][footer_resolve]")
{
  loopback_range_server server(test_payload());
  auto fixture = make_ioctx(server, test_config(3));
  fixture.authorizer->set_object_exception(
    "auth-error.parquet",
    std::make_exception_ptr(per_key_authorizer_error{"original authorizer error"}));
  std::vector<std::string> paths{
    object_uri("a.parquet"), object_uri("auth-error.parquet"), object_uri("b.parquet")};

  auto const results = resolve(*fixture.ioctx, paths);

  REQUIRE(results.size() == paths.size());
  require_success(result_at(results, 0));
  auto const& failed = result_at(results, 1);
  require_failure(failed);
  try {
    std::rethrow_exception(failed.error);
    FAIL("expected the original authorizer exception");
  } catch (per_key_authorizer_error const& error) {
    CHECK(std::string_view{error.what()} == "original authorizer error");
  } catch (...) {
    FAIL("authorizer exception type changed");
  }
  require_success(result_at(results, 2));
  CHECK(server.get_count("a.parquet") == 1);
  CHECK(server.get_count("auth-error.parquet") == 0);
  CHECK(server.get_count("b.parquet") == 1);
  CHECK(fixture.authorizer->object_calls() == 3);
}

TEST_CASE("batched footer resolve isolates malformed paths and rejects an undersized budget",
          "[rest][footer_resolve]")
{
  loopback_range_server server(test_payload());
  auto fixture = make_ioctx(server, test_config(2));
  std::vector<std::string> paths{"not-a-uri", object_uri("valid.parquet")};

  auto const results = resolve(*fixture.ioctx, paths);

  REQUIRE(results.size() == paths.size());
  auto const& malformed = result_at(results, 0);
  require_failure(malformed);
  CHECK_THROWS_AS(std::rethrow_exception(malformed.error), std::invalid_argument);
  require_success(result_at(results, 1));
  CHECK(server.get_count("valid.parquet") == 1);

  auto undersized = make_ioctx(server, test_config(1, probe_size - 1));
  std::vector<std::string> one{object_uri("undersized.parquet")};
  CHECK_THROWS(undersized.ioctx->resolve_footer_objects(one, [](footer_resolve_result) {}));
  CHECK(server.get_count("undersized.parquet") == 0);
}

TEST_CASE("batched footer resolve streams fast siblings while another entry delays and retries",
          "[rest][footer_resolve]")
{
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["slow.parquet"].gets  = {scripted_response{.delay = 300ms}};
  scripts["retry.parquet"].gets = {scripted_response{.status = 503}, scripted_response{}};
  loopback_range_server server(test_payload(), {}, {}, std::move(scripts));
  auto fixture = make_ioctx(server, test_config(3));
  std::vector<std::string> paths{
    object_uri("slow.parquet"), object_uri("retry.parquet"), object_uri("fast.parquet")};
  std::vector<std::size_t> delivery_order;
  std::vector<footer_resolve_result> results;

  fixture.ioctx->resolve_footer_objects(paths, [&](footer_resolve_result result) {
    delivery_order.push_back(result.index);
    results.push_back(std::move(result));
  });

  REQUIRE(results.size() == 3);
  CHECK(std::all_of(results.begin(), results.end(), is_success));
  REQUIRE(delivery_order.size() == 3);
  auto sorted_indices = delivery_order;
  std::sort(sorted_indices.begin(), sorted_indices.end());
  CHECK((sorted_indices == std::vector<std::size_t>{0, 1, 2}));
  auto const position = [&](std::size_t index) {
    return std::distance(delivery_order.begin(),
                         std::find(delivery_order.begin(), delivery_order.end(), index));
  };
  CHECK(position(2) < position(1));
  CHECK(position(1) < position(0));
  CHECK(delivery_order.back() == 0);
  CHECK(server.get_count("retry.parquet") == 2);
  CHECK(fixture.authorizer->object_calls() == 4);
}

TEST_CASE("batched footer resolve delivers duplicate inputs exactly once on the caller thread",
          "[rest][footer_resolve]")
{
  loopback_range_server server(test_payload());
  auto fixture    = make_ioctx(server, test_config(2));
  auto const path = object_uri("duplicate.parquet");
  std::vector<std::string> paths{path, path, path};
  auto const caller = std::this_thread::get_id();
  std::vector<footer_resolve_result> results;
  std::vector<std::thread::id> callback_threads;

  fixture.ioctx->resolve_footer_objects(paths, [&](footer_resolve_result result) {
    callback_threads.push_back(std::this_thread::get_id());
    results.push_back(std::move(result));
  });

  REQUIRE(results.size() == paths.size());
  std::vector<std::size_t> counts(paths.size());
  for (auto const& result : results) {
    require_success(result);
    REQUIRE(result.index < counts.size());
    ++counts[result.index];
    CHECK(result.path == path);
  }
  CHECK((counts == std::vector<std::size_t>{1, 1, 1}));
  CHECK(std::all_of(callback_threads.begin(), callback_threads.end(), [&](auto thread) {
    return thread == caller;
  }));
  CHECK(server.get_count("duplicate.parquet") == paths.size());
  std::vector<std::string> empty;
  CHECK_THROWS(fixture.ioctx->resolve_footer_objects(empty, [](footer_resolve_result) {}));
}

TEST_CASE("batched footer resolve cancels every undelivered input exactly once",
          "[rest][footer_resolve]")
{
  SECTION("before submission")
  {
    loopback_range_server server(test_payload());
    auto fixture = make_ioctx(server, test_config());
    std::stop_source stop;
    stop.request_stop();

    auto const results =
      resolve(*fixture.ioctx, {object_uri("a.parquet"), object_uri("b.parquet")}, stop.get_token());

    REQUIRE(results.size() == 2);
    CHECK(std::all_of(results.begin(), results.end(), [](auto const& result) {
      return is_operation_canceled(result.error);
    }));
    CHECK(result_at(results, 0).index == 0);
    CHECK(result_at(results, 1).index == 1);
    require_failure(result_at(results, 0));
    require_failure(result_at(results, 1));
    CHECK(server.get_count() == 0);
  }

  SECTION("while GETs are in flight")
  {
    std::unordered_map<std::string, key_response_script> scripts;
    scripts["a.parquet"].gets = {scripted_response{.delay = 500ms}};
    scripts["b.parquet"].gets = {scripted_response{.delay = 500ms}};
    auto server               = std::make_shared<loopback_range_server>(
      test_payload(), range_fault_policy{}, std::vector<cucascade::test::listed_object>{}, scripts);
    auto fixture = make_ioctx(*server, test_config(2));
    auto results = std::make_shared<std::vector<footer_resolve_result>>();
    auto mutex   = std::make_shared<std::mutex>();
    std::stop_source stop;
    std::vector<std::string> paths{object_uri("a.parquet"), object_uri("b.parquet")};
    async_call call(
      [server, ioctx = fixture.ioctx, results, mutex, paths, token = stop.get_token()] {
        ioctx->resolve_footer_objects(
          paths,
          [&](footer_resolve_result result) {
            std::scoped_lock lock{*mutex};
            results->push_back(std::move(result));
          },
          token);
      });

    REQUIRE(wait_until([&] { return server->get_count() == 2; }, 1s));
    stop.request_stop();
    require_ready(call, 250ms);

    std::scoped_lock lock{*mutex};
    REQUIRE(results->size() == paths.size());
    CHECK(std::all_of(results->begin(), results->end(), [](auto const& result) {
      return is_operation_canceled(result.error);
    }));
    CHECK(result_at(*results, 0).index == 0);
    CHECK(result_at(*results, 1).index == 1);
    require_failure(result_at(*results, 0));
    require_failure(result_at(*results, 1));
  }

  SECTION("during retry backoff")
  {
    std::unordered_map<std::string, key_response_script> scripts;
    scripts["retry.parquet"].gets = {scripted_response{.status = 503}};
    auto server                   = std::make_shared<loopback_range_server>(
      test_payload(), range_fault_policy{}, std::vector<cucascade::test::listed_object>{}, scripts);
    auto cfg               = test_config(1);
    cfg.retry_backoff_base = 500ms;
    auto fixture           = make_ioctx(*server, cfg);
    auto results           = std::make_shared<std::vector<footer_resolve_result>>();
    std::stop_source stop;
    auto const path = object_uri("retry.parquet");
    std::vector<std::string> paths{path};
    async_call call([server, ioctx = fixture.ioctx, results, paths, token = stop.get_token()] {
      ioctx->resolve_footer_objects(
        paths, [&](footer_resolve_result result) { results->push_back(std::move(result)); }, token);
    });

    REQUIRE(wait_until([&] { return server->get_count("retry.parquet") == 1; }, 1s));
    stop.request_stop();
    require_ready(call, 250ms);
    REQUIRE(results->size() == 1);
    CHECK(is_operation_canceled(results->front().error));
    require_failure(results->front());
    CHECK(server->get_count("retry.parquet") == 1);
  }
}

namespace {
class first_callback_error : public std::runtime_error {
 public:
  first_callback_error() : std::runtime_error("first callback failure") {}
};

class later_callback_error : public std::runtime_error {
 public:
  later_callback_error() : std::runtime_error("later callback failure") {}
};
}  // namespace

TEST_CASE(
  "batched footer resolve drains cancellation callbacks and rethrows the first callback error",
  "[rest][footer_resolve]")
{
  struct callback_state {
    std::mutex mutex;
    std::size_t successful{0};
    std::size_t canceled{0};
    std::size_t invalid_index{0};
    std::array<std::size_t, 3> deliveries{};
    bool all_canceled_errors{true};
  };

  auto server  = std::make_shared<loopback_range_server>(test_payload());
  auto fixture = make_ioctx(*server, test_config(1));
  std::vector<std::string> paths{
    object_uri("a.parquet"), object_uri("b.parquet"), object_uri("c.parquet")};
  auto state        = std::make_shared<callback_state>();
  bool caught_first = false;

  async_call call([server, ioctx = fixture.ioctx, paths, state] {
    ioctx->resolve_footer_objects(paths, [state](footer_resolve_result result) {
      {
        std::scoped_lock lock{state->mutex};
        if (result.index < state->deliveries.size()) {
          ++state->deliveries[result.index];
        } else {
          ++state->invalid_index;
        }
      }
      if (result.error) {
        {
          std::scoped_lock lock{state->mutex};
          state->all_canceled_errors &= is_operation_canceled(result.error);
          ++state->canceled;
        }
        throw later_callback_error{};
      }
      {
        std::scoped_lock lock{state->mutex};
        ++state->successful;
      }
      throw first_callback_error{};
    });
  });
  REQUIRE(call.ready_within(3s));
  auto const error = call.finish();
  try {
    if (error) { std::rethrow_exception(error); }
  } catch (first_callback_error const&) {
    caught_first = true;
  }

  CHECK(caught_first);
  std::size_t callbacks_at_return;
  {
    std::scoped_lock lock{state->mutex};
    CHECK(state->successful == 1);
    CHECK(state->canceled == paths.size() - 1);
    CHECK(state->invalid_index == 0);
    CHECK((state->deliveries == std::array<std::size_t, 3>{1, 1, 1}));
    CHECK(state->all_canceled_errors);
    callbacks_at_return = state->successful + state->canceled;
  }
  std::this_thread::sleep_for(25ms);
  std::scoped_lock lock{state->mutex};
  CHECK(state->successful + state->canceled == callbacks_at_return);
}

TEST_CASE("batched footer resolve cancels concurrent completions after the first callback error",
          "[rest][footer_resolve]")
{
  constexpr std::size_t max_inflight = 3;
  constexpr std::size_t path_count   = 6;
  range_fault_policy fault;
  fault.get_response_barrier = max_inflight;
  loopback_range_server server(test_payload(), fault);
  auto fixture = make_ioctx(server, test_config(max_inflight));
  std::vector<std::string> paths;
  paths.reserve(path_count);
  for (std::size_t i = 0; i < path_count; ++i) {
    paths.push_back(object_uri("callback-" + std::to_string(i) + ".parquet"));
  }

  std::array<std::size_t, path_count> deliveries{};
  std::size_t successes = 0;
  std::size_t canceled  = 0;
  bool all_canceled_errors{true};
  bool caught_first{false};
  try {
    fixture.ioctx->resolve_footer_objects(paths, [&](footer_resolve_result result) {
      REQUIRE(result.index < deliveries.size());
      ++deliveries[result.index];
      if (result.error) {
        all_canceled_errors &= is_operation_canceled(result.error);
        ++canceled;
        return;
      }
      ++successes;
      throw first_callback_error{};
    });
  } catch (first_callback_error const&) {
    caught_first = true;
  }

  CHECK(caught_first);
  CHECK(successes == 1);
  CHECK(canceled == path_count - 1);
  CHECK(all_canceled_errors);
  CHECK(std::all_of(deliveries.begin(), deliveries.end(), [](auto count) { return count == 1; }));
  CHECK(server.get_count() == max_inflight);
  for (std::size_t i = 0; i < path_count; ++i) {
    auto const key = "callback-" + std::to_string(i) + ".parquet";
    CHECK(server.get_count(key) == (i < max_inflight ? 1 : 0));
  }
  CHECK(server.head_count() == 0);
  CHECK(fixture.authorizer->object_calls() == max_inflight);

  auto const callbacks_at_return = successes + canceled;
  std::this_thread::sleep_for(25ms);
  CHECK(successes + canceled == callbacks_at_return);
}

TEST_CASE("batched footer resolve rejects an explicit zero payload budget",
          "[rest][footer_resolve]")
{
  loopback_range_server server(test_payload());
  auto fixture = make_ioctx(server, test_config(2, 0));
  std::vector<std::string> paths{object_uri("zero-budget.parquet")};

  CHECK_THROWS_AS(fixture.ioctx->resolve_footer_objects(paths, [](footer_resolve_result) {}),
                  std::invalid_argument);
  CHECK(server.get_count() == 0);
  CHECK(server.head_count() == 0);
  CHECK(fixture.authorizer->object_calls() == 0);
}

TEST_CASE("batched footer resolve bounds retained payloads across concurrent batches",
          "[rest][footer_resolve]")
{
  constexpr std::size_t budget = probe_size;
  auto server                  = std::make_shared<loopback_range_server>(test_payload());
  auto fixture                 = make_ioctx(*server, test_config(4, budget));
  auto retained                = std::make_shared<payload_gate>();
  auto errors                  = std::make_shared<std::atomic<std::size_t>>(0);
  auto first_delivered         = std::make_shared<std::atomic<std::size_t>>(0);
  auto second_delivered        = std::make_shared<std::atomic<std::size_t>>(0);
  std::vector<std::string> first_paths(4, object_uri("first.parquet"));
  std::vector<std::string> second_paths(4, object_uri("second.parquet"));

  auto first_callback = [retained, errors, first_delivered](footer_resolve_result result) {
    if (!is_success(result)) {
      errors->fetch_add(1);
      return;
    }
    first_delivered->fetch_add(1);
    retained->retain(std::move(result.footer));
  };
  auto second_callback = [retained, errors, second_delivered](footer_resolve_result result) {
    if (!is_success(result)) {
      errors->fetch_add(1);
      return;
    }
    second_delivered->fetch_add(1);
    retained->retain(std::move(result.footer));
  };
  async_call first([server, ioctx = fixture.ioctx, first_paths, first_callback] {
    ioctx->resolve_footer_objects(first_paths, first_callback);
  });
  REQUIRE(wait_until([&] { return server->get_count("first.parquet") == 1; }, 1s));
  async_call second([server, ioctx = fixture.ioctx, second_paths, second_callback] {
    ioctx->resolve_footer_objects(second_paths, second_callback);
  });

  bool stayed_within_budget = true;
  while (!first.ready_within(0ms)) {
    if (!retained->wait_for_payload(1s)) {
      throw std::runtime_error("timed out waiting for first-batch footer payload");
    }
    stayed_within_budget &= retained->resident() <= budget;
    if (first_delivered->load() == first_paths.size()) { break; }
    retained->release_one();
  }
  require_ready(first);

  REQUIRE(retained->wait_for_payload(1s));
  CHECK(retained->resident() == budget);
  CHECK(server->get_count("second.parquet") == 0);
  auto const held_snapshot = fixture.ioctx->perf_snapshot();
  CHECK(held_snapshot.footer_stash_reserved_bytes == budget);
  CHECK(held_snapshot.footer_stash_reserved_peak_bytes == budget);
  retained->release_one();
  REQUIRE(wait_until([&] { return server->get_count("second.parquet") == 1; }, 1s));

  while (!second.ready_within(0ms)) {
    if (!retained->wait_for_payload(1s)) {
      throw std::runtime_error("timed out waiting for second-batch footer payload");
    }
    stayed_within_budget &= retained->resident() <= budget;
    if (second_delivered->load() == second_paths.size()) { break; }
    retained->release_one();
  }
  require_ready(second);
  retained->release_all();

  auto const drained_snapshot = fixture.ioctx->perf_snapshot();
  CHECK(errors->load() == 0);
  CHECK(stayed_within_budget);
  CHECK(retained->resident() == 0);
  CHECK(retained->peak() <= budget);
  CHECK(drained_snapshot.footer_stash_reserved_bytes == 0);
  CHECK(drained_snapshot.footer_stash_reserved_peak_bytes == budget);
  CHECK(server->get_count() == first_paths.size() + second_paths.size());

  auto disabled = make_ioctx(*server, test_config(0, budget));
  std::vector<std::string> one{object_uri("disabled.parquet")};
  CHECK_THROWS(disabled.ioctx->resolve_footer_objects(one, [](footer_resolve_result) {}));
}

TEST_CASE("batched footer resolve makes progress with a one-worker consumer",
          "[rest][footer_resolve]")
{
  auto server     = std::make_shared<loopback_range_server>(test_payload());
  auto fixture    = make_ioctx(*server, test_config(2, 2 * probe_size));
  auto pool       = std::make_shared<static_thread_pool>(1, "footer-one-worker");
  auto dispatcher = std::make_shared<scoped_dispatcher>(*pool, 1);
  auto parsed     = std::make_shared<std::atomic<std::size_t>>(0);
  std::vector<std::string> paths(12, object_uri("one-worker.parquet"));

  async_call call([server, ioctx = fixture.ioctx, pool, dispatcher, parsed, paths] {
    ioctx->resolve_footer_objects(paths, [dispatcher, parsed](footer_resolve_result result) {
      dispatcher->enqueue([result = std::move(result), parsed] {
        if (!result.error && result.footer) { parsed->fetch_add(1); }
      });
    });
    dispatcher->wait_for_all();
  });

  require_ready(call);
  CHECK(parsed->load() == paths.size());
}

TEST_CASE("batched footer resolve completes saturated producers and preserves FIFO fairness",
          "[rest][footer_resolve]")
{
  auto server                     = std::make_shared<loopback_range_server>(test_payload());
  auto fixture                    = make_ioctx(*server, test_config(2, 4 * probe_size));
  auto completed                  = std::make_shared<std::atomic<std::size_t>>(0);
  constexpr std::size_t producers = 4;
  auto pool       = std::make_shared<static_thread_pool>(producers, "footer-saturation");
  auto dispatcher = std::make_shared<scoped_dispatcher>(*pool, producers);
  std::vector<std::unique_ptr<async_call>> calls;

  for (std::size_t producer = 0; producer < producers; ++producer) {
    std::vector<std::string> paths(4, object_uri("batch-" + std::to_string(producer) + ".parquet"));
    calls.push_back(std::make_unique<async_call>(
      [server, ioctx = fixture.ioctx, pool, dispatcher, completed, paths = std::move(paths)] {
        ioctx->resolve_footer_objects(paths, [dispatcher, completed](footer_resolve_result result) {
          dispatcher->enqueue([result = std::move(result), completed] {
            if (!result.error && result.footer) { completed->fetch_add(1); }
          });
        });
      }));
  }

  for (auto& call : calls) {
    require_ready(*call);
  }
  dispatcher->wait_for_all();
  CHECK(completed->load() == producers * 4);

  std::unordered_map<std::string, key_response_script> scripts;
  scripts["active.parquet"].gets = {scripted_response{.delay = 100ms}};
  auto fifo_server               = std::make_shared<loopback_range_server>(
    test_payload(), range_fault_policy{}, std::vector<cucascade::test::listed_object>{}, scripts);
  auto fifo_fixture = make_ioctx(*fifo_server, test_config(1));
  auto order        = std::make_shared<std::vector<char>>();
  auto order_mutex  = std::make_shared<std::mutex>();
  std::vector<std::string> active{object_uri("active.parquet")};
  std::vector<std::string> queued{object_uri("queued.parquet")};
  async_call first([fifo_server, ioctx = fifo_fixture.ioctx, order, order_mutex, active] {
    ioctx->resolve_footer_objects(active, [order, order_mutex](footer_resolve_result) {
      std::scoped_lock lock{*order_mutex};
      order->push_back('A');
    });
  });
  REQUIRE(wait_until([&] { return fifo_server->get_count("active.parquet") == 1; }, 1s));
  async_call second([fifo_server, ioctx = fifo_fixture.ioctx, order, order_mutex, queued] {
    ioctx->resolve_footer_objects(queued, [order, order_mutex](footer_resolve_result) {
      std::scoped_lock lock{*order_mutex};
      order->push_back('B');
    });
  });

  require_ready(first);
  require_ready(second);
  CHECK((*order == std::vector<char>{'A', 'B'}));
}

TEST_CASE("batched footer resolve stop unblocks a zero-active-transfer budget wait",
          "[rest][footer_resolve]")
{
  auto server  = std::make_shared<loopback_range_server>(test_payload());
  auto fixture = make_ioctx(*server, test_config(1, probe_size));
  auto held    = std::make_shared<std::vector<shared_byte_span>>();
  auto results = std::make_shared<std::vector<footer_resolve_result>>();
  auto mutex   = std::make_shared<std::mutex>();
  std::stop_source stop;
  std::vector<std::string> paths{object_uri("first.parquet"), object_uri("blocked.parquet")};
  async_call call(
    [server, ioctx = fixture.ioctx, held, results, mutex, paths, token = stop.get_token()] {
      ioctx->resolve_footer_objects(
        paths,
        [&](footer_resolve_result result) {
          std::scoped_lock lock{*mutex};
          if (result.footer) { held->push_back(result.footer); }
          results->push_back(std::move(result));
        },
        token);
    });

  REQUIRE(wait_until(
    [&] {
      std::scoped_lock lock{*mutex};
      return held->size() == 1 && server->get_count("blocked.parquet") == 0;
    },
    1s));
  stop.request_stop();
  require_ready(call, 250ms);

  std::scoped_lock lock{*mutex};
  REQUIRE(results->size() == 2);
  require_success(result_at(*results, 0));
  CHECK(is_operation_canceled(result_at(*results, 1).error));
  held->clear();
  results->clear();
}

TEST_CASE("batched footer payload lease can outlive its rest ioctx", "[rest][footer_resolve]")
{
  loopback_range_server server(test_payload());
  shared_byte_span held;
  auto const expected_last_byte = test_payload().back();
  {
    auto fixture = make_ioctx(server, test_config());
    auto results = resolve(*fixture.ioctx, {object_uri("held.parquet")});
    require_success(results.front());
    held = std::move(results.front().footer);
    fixture.ioctx.reset();
  }

  REQUIRE(held != nullptr);
  REQUIRE_FALSE(held->empty());
  CHECK(held->back() == expected_last_byte);
  held.reset();
  SUCCEED();
}

TEST_CASE("batched footer resolve removes a canceled batch from the FIFO queue",
          "[rest][footer_resolve]")
{
  std::unordered_map<std::string, key_response_script> scripts;
  scripts["active.parquet"].gets = {scripted_response{.delay = 300ms}};
  auto server                    = std::make_shared<loopback_range_server>(
    test_payload(), range_fault_policy{}, std::vector<cucascade::test::listed_object>{}, scripts);
  auto fixture = make_ioctx(*server, test_config(1));
  std::vector<std::string> active{object_uri("active.parquet")};
  std::vector<std::string> queued{object_uri("queued-a.parquet"), object_uri("queued-b.parquet")};
  auto queued_results = std::make_shared<std::vector<footer_resolve_result>>();
  auto queued_started = std::make_shared<std::atomic<bool>>(false);
  std::stop_source queued_stop;

  async_call first([server, ioctx = fixture.ioctx, active] {
    ioctx->resolve_footer_objects(active, [](footer_resolve_result) {});
  });
  REQUIRE(wait_until([&] { return server->get_count("active.parquet") == 1; }, 1s));
  async_call second([server,
                     ioctx = fixture.ioctx,
                     queued,
                     queued_results,
                     queued_started,
                     token = queued_stop.get_token()] {
    queued_started->store(true);
    ioctx->resolve_footer_objects(
      queued,
      [&](footer_resolve_result result) { queued_results->push_back(std::move(result)); },
      token);
  });

  REQUIRE(wait_until([&] { return queued_started->load(); }, 1s));
  CHECK_FALSE(second.ready_within(20ms));
  queued_stop.request_stop();
  require_ready(second, 1s);
  CHECK_FALSE(first.ready_within(0ms));
  REQUIRE(queued_results->size() == queued.size());
  CHECK(std::all_of(queued_results->begin(), queued_results->end(), [](auto const& result) {
    return is_operation_canceled(result.error);
  }));
  CHECK(server->get_count("queued-a.parquet") == 0);
  CHECK(server->get_count("queued-b.parquet") == 0);
  require_failure(result_at(*queued_results, 0));
  require_failure(result_at(*queued_results, 1));
  require_ready(first);
}

TEST_CASE("batched footer resolve reuses at most the configured in-flight connections",
          "[rest][footer_resolve]")
{
  constexpr std::size_t inflight = 2;
  loopback_range_server server(test_payload());
  std::vector<std::string> paths;
  for (std::size_t i = 0; i < 12; ++i) {
    paths.push_back(object_uri("reuse-" + std::to_string(i) + ".parquet"));
  }
  auto fixture = make_ioctx(server, test_config(inflight, paths.size() * probe_size));

  auto const results = resolve(*fixture.ioctx, paths);

  REQUIRE(results.size() == paths.size());
  CHECK(
    std::all_of(results.begin(), results.end(), [](auto const& result) { return !result.error; }));
  for (std::size_t i = 0; i < paths.size(); ++i) {
    CHECK(server.get_count("reuse-" + std::to_string(i) + ".parquet") == 1);
  }
  CHECK(server.get_count() == paths.size());
  CHECK(server.accepted_connection_count() <= inflight);
}

TEST_CASE("batched footer resolve folds equivalent request outcomes into the ioctx snapshot",
          "[rest][footer_resolve]")
{
  auto scripts = [] {
    std::unordered_map<std::string, key_response_script> value;
    value["retry.parquet"].gets   = {scripted_response{.status = 503}, scripted_response{}};
    value["missing.parquet"].gets = {scripted_response{.status = 404}};
    return value;
  };
  std::vector<std::string> paths{
    object_uri("ok.parquet"), object_uri("retry.parquet"), object_uri("missing.parquet")};
  loopback_range_server single_server(test_payload(), {}, {}, scripts());
  auto const single = snapshot_after_single_opens(single_server, test_config(2), paths);

  loopback_range_server batch_server(test_payload(), {}, {}, scripts());
  auto batch_fixture = make_ioctx(batch_server, test_config(2));
  auto const results = resolve(*batch_fixture.ioctx, paths);
  auto const batch   = batch_fixture.ioctx->perf_snapshot();

  REQUIRE(results.size() == paths.size());
  require_success(result_at(results, 0));
  require_success(result_at(results, 1));
  require_failure(result_at(results, 2));
  CHECK(batch.chunk_get_count == single.chunk_get_count);
  CHECK(batch.blocking_host_get_count == single.blocking_host_get_count);
  CHECK(batch.payload_bytes_read_total == single.payload_bytes_read_total);
  CHECK(batch.retries_total == single.retries_total);
  CHECK(batch.terminal_failures_total == single.terminal_failures_total);
}
