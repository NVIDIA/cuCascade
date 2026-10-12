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

#include "cache_test_fixture.hpp"
#include "utils/child_test_runner.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <memory>
#include <string>
#include <string_view>

namespace {
using namespace std::chrono_literals;
using namespace cucascade::test;
using cucascade::io::prefetch_refusal;
using cucascade::io::prepare_result;
constexpr std::size_t chunk_size = 1U << 20;
constexpr std::size_t read_size  = 4096;

bool run_arbitration_child(std::string const& name)
{
  auto const* marker = std::getenv("CUCASCADE_CACHE_ARBITRATION_CHILD");
  if (marker != nullptr && std::string_view{marker} == "1") { return false; }
  child_test_runner child{name, {{"CUCASCADE_CACHE_ARBITRATION_CHILD", std::string{"1"}}}};
  bool const finished = child.wait_for(30s);
  if (!finished) { child.stop(); }
  INFO(child.output());
  REQUIRE(finished);
  REQUIRE(WIFEXITED(child.status()));
  REQUIRE(WEXITSTATUS(child.status()) == 0);
  return true;
}

struct read_cleanup {
  controlled_ioctx& context;
  std::array<held_request*, 2> held;
  ~read_cleanup()
  {
    for (auto* request : held) {
      if (request) { request->cancel(); }
    }
    context.shutdown();
  }
};

struct stream_cleanup {
  rmm::cuda_stream& stream;
  ~stream_cleanup() { stream.synchronize_no_throw(); }
};

struct arbitration_fixture : cache_test_fixture {
  arbitration_fixture() : cache_test_fixture(allocation_cache_config())
  {
    REQUIRE(context->cache() != nullptr);
    REQUIRE(context->cache()->chunk_size() == chunk_size);
  }
  std::unique_ptr<cucascade::io::datasource> datasource =
    open("controlled://arbitration", 4 * chunk_size);
  void advise(std::size_t offset)
  {
    advise_range(*datasource, offset, read_size);
    REQUIRE(datasource->prepare_prefetch(false) == prepare_result::prepared);
  }
};
}  // namespace

TEST_CASE("an overlapping cache read waits for prefetch and reuses its chunk",
          "[cache][cache_handle][read-arbitration]")
{
  if (run_arbitration_child("an overlapping cache read waits for prefetch and reuses its chunk")) {
    return;
  }
  arbitration_fixture fixture;
  fixture.advise(0);
  std::atomic<bool> prefetched{false};
  std::atomic<int> callbacks{0};
  std::array<std::uint8_t, read_size> destination{};
  held_request prefetch;
  std::future<std::future<std::size_t>> read_call;
  read_cleanup cleanup{*fixture.context, {&prefetch, nullptr}};
  REQUIRE(fixture.datasource->prefetch_async([&](bool ok) noexcept {
    prefetched.store(ok, std::memory_order_release);
    callbacks.fetch_add(1);
  }) == prefetch_refusal::issued);
  prefetch.request = fixture.context->backend().wait_take(2s);
  REQUIRE(prefetch.request != nullptr);
  read_call = std::async(std::launch::async, [&] {
    return fixture.datasource->host_read_async(0, destination.size(), destination.data());
  });
  CHECK(read_call.wait_for(50ms) == std::future_status::timeout);
  CHECK(fixture.context->backend().pending_count() == 0);
  prefetch.complete(0x5a);
  REQUIRE(read_call.wait_for(2s) == std::future_status::ready);
  auto read = read_call.get();
  REQUIRE(read.wait_for(2s) == std::future_status::ready);
  CHECK(read.get() == destination.size());
  CHECK(std::ranges::all_of(destination, [](auto byte) { return byte == 0x5a; }));
  CHECK(prefetched.load(std::memory_order_acquire));
  CHECK(callbacks.load() == 1);
  CHECK(fixture.context->backend().pending_count() == 0);
}

TEST_CASE("a cached device read retires its pin through the stream completion poll",
          "[cache][cache_handle][completion-poll][gpu_execution]")
{
  if (run_arbitration_child(
        "a cached device read retires its pin through the stream completion poll")) {
    return;
  }
  arbitration_fixture fixture;
  fixture.advise(0);
  std::atomic<bool> prefetched{false};
  held_request prefetch;
  read_cleanup cleanup{*fixture.context, {&prefetch, nullptr}};
  REQUIRE(fixture.datasource->prefetch_async([&](bool ok) noexcept {
    prefetched.store(ok, std::memory_order_release);
  }) == prefetch_refusal::issued);
  prefetch.request = fixture.context->backend().wait_take(2s);
  REQUIRE(prefetch.request != nullptr);
  prefetch.complete(0x6c);
  REQUIRE(prefetched.load(std::memory_order_acquire));
  std::array<std::uint8_t, read_size> host{};
  rmm::cuda_stream stream;
  rmm::device_buffer destination(read_size, stream);
  stream_cleanup drain{stream};
  auto read = fixture.datasource->device_read_async(
    0, read_size, static_cast<std::uint8_t*>(destination.data()), stream);
  REQUIRE(read.wait_for(2s) == std::future_status::ready);
  CHECK(read.get() == read_size);
  CHECK(fixture.context->backend().pending_count() == 0);
  REQUIRE(cudaMemcpyAsync(
            host.data(), destination.data(), host.size(), cudaMemcpyDeviceToHost, stream.value()) ==
          cudaSuccess);
  stream.synchronize();
  CHECK(std::ranges::all_of(host, [](auto byte) { return byte == 0x6c; }));
}

TEST_CASE("a cache read does not wait for unrelated chunks in the same handle",
          "[cache][cache_handle][read-arbitration]")
{
  if (run_arbitration_child("a cache read does not wait for unrelated chunks in the same handle")) {
    return;
  }
  arbitration_fixture fixture;
  fixture.advise(0);
  std::array<std::uint8_t, read_size> destination{};
  held_request prefetch;
  held_request demand;
  std::future<std::future<std::size_t>> read_call;
  read_cleanup cleanup{*fixture.context, {&prefetch, &demand}};
  REQUIRE(fixture.datasource->prefetch_async([](bool) noexcept {}) == prefetch_refusal::issued);
  prefetch.request = fixture.context->backend().wait_take(2s);
  REQUIRE(prefetch.request != nullptr);
  read_call           = std::async(std::launch::async, [&] {
    return fixture.datasource->host_read_async(
      2 * chunk_size, destination.size(), destination.data());
  });
  auto const returned = read_call.wait_for(100ms) == std::future_status::ready;
  CHECK(returned);
  if (!returned) { prefetch.complete(0x5a); }
  REQUIRE(read_call.wait_for(2s) == std::future_status::ready);
  auto read      = read_call.get();
  demand.request = fixture.context->backend().wait_take(2s);
  REQUIRE(demand.request != nullptr);
  CHECK(demand.request->front().h_buffer.is_contiguous());
  demand.complete(0x2b);
  if (prefetch.request) { prefetch.complete(0x5a); }
  REQUIRE(read.wait_for(2s) == std::future_status::ready);
  CHECK(read.get() == destination.size());
  CHECK(std::ranges::all_of(destination, [](auto byte) { return byte == 0x2b; }));
}

TEST_CASE("a demand-owned loading chunk uses reactor bounce staging",
          "[cache][cache_handle][read-arbitration]")
{
  if (run_arbitration_child("a demand-owned loading chunk uses reactor bounce staging")) { return; }
  arbitration_fixture fixture;
  fixture.advise(chunk_size);
  rmm::cuda_stream stream;
  rmm::device_buffer first_destination(read_size, stream);
  rmm::device_buffer second_destination(read_size, stream);
  stream_cleanup drain{stream};
  held_request cache_load;
  held_request bounced;
  std::future<std::future<std::size_t>> second_call;
  read_cleanup cleanup{*fixture.context, {&cache_load, &bounced}};
  auto first = fixture.datasource->device_read_async(
    chunk_size, read_size, static_cast<std::uint8_t*>(first_destination.data()), stream);
  cache_load.request = fixture.context->backend().wait_take(2s);
  REQUIRE(cache_load.request != nullptr);
  CHECK(cache_load.request->front().h_buffer.is_fragmented());
  second_call         = std::async(std::launch::async, [&] {
    return fixture.datasource->device_read_async(
      chunk_size, read_size, static_cast<std::uint8_t*>(second_destination.data()), stream);
  });
  auto const returned = second_call.wait_for(100ms) == std::future_status::ready;
  CHECK(returned);
  if (!returned) { cache_load.complete(0x5a); }
  REQUIRE(second_call.wait_for(2s) == std::future_status::ready);
  auto second     = second_call.get();
  bounced.request = fixture.context->backend().wait_take(2s);
  REQUIRE(bounced.request != nullptr);
  CHECK(bounced.request->front().needs_staging());
  CHECK_FALSE(bounced.request->front().is_fragmented());
  CHECK(bounced.request->front().has_device_request());
  bounced.complete(0x5a);
  if (cache_load.request) { cache_load.complete(0x5a); }
  REQUIRE(first.wait_for(2s) == std::future_status::ready);
  REQUIRE(second.wait_for(2s) == std::future_status::ready);
  CHECK(first.get() == read_size);
  CHECK(second.get() == read_size);
}

TEST_CASE("a cache whose chunk size differs from the reactor staging block is refused", "[cache]")
{
  auto memory       = identity_memory();
  auto const config = allocation_cache_config();
  auto probe = std::make_shared<controlled_ioctx>(controlled_reactor_config{.staging_bytes = 0});
  probe->initialize_cache(*memory, config, nullptr);
  REQUIRE(probe->cache() != nullptr);
  auto const actual_chunk = probe->cache()->chunk_size();
  REQUIRE(actual_chunk > 1);
  probe->shutdown_cache();
  auto mismatched = std::make_shared<controlled_ioctx>(
    controlled_reactor_config{.staging_bytes = actual_chunk / 2});
  mismatched->initialize_cache(*memory, config, nullptr);
  CHECK(mismatched->cache() == nullptr);
  auto matched =
    std::make_shared<controlled_ioctx>(controlled_reactor_config{.staging_bytes = actual_chunk});
  matched->initialize_cache(*memory, config, nullptr);
  CHECK(matched->cache() != nullptr);
  matched->shutdown_cache();
}
