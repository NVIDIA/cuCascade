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
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace {
using namespace std::chrono_literals;
using namespace cucascade::test;
using cucascade::io::prefetch_refusal;
using cucascade::io::prepare_result;
using cucascade::io::cache::prefetch_result;
constexpr std::size_t read_size = 4096;

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
    REQUIRE(context->cache()->chunk_size() == controlled_chunk_bytes);
  }
  std::unique_ptr<cucascade::io::datasource> datasource =
    open("controlled://arbitration", 4 * controlled_chunk_bytes);
  void advise(std::size_t offset)
  {
    advise_range(*datasource, offset, read_size);
    REQUIRE(datasource->prepare_prefetch(false) == prepare_result::prepared);
  }
  /// A datasource over a new open of @p path at @p version (see controlled_object).
  std::unique_ptr<cucascade::io::datasource> open_version(std::string path, std::string version)
  {
    return std::make_unique<cucascade::io::datasource>(
      context,
      std::make_shared<controlled_object>(
        std::move(path), 4 * controlled_chunk_bytes, std::move(version)));
  }
  /// Issue a prefetch of the advised range and take its fill off the backend.
  void prefetch(held_request& fill)
  {
    REQUIRE(datasource->prefetch_async([](prefetch_result) noexcept {}) ==
            prefetch_refusal::issued);
    fill.request = context->backend().wait_take(2s);
    REQUIRE(fill.request != nullptr);
  }
};

template <typename Predicate>
bool eventually(Predicate predicate)
{
  auto const deadline = std::chrono::steady_clock::now() + 3s;
  while (!predicate() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(10ms);
  }
  return predicate();
}
}  // namespace

TEST_CASE("an overlapping cache read waits for prefetch and reuses its chunk",
          "[cache][cache_handle][read-arbitration]")
{
  if (run_in_child("CUCASCADE_CACHE_ARBITRATION_CHILD", 30s)) { return; }
  arbitration_fixture fixture;
  CHECK(fixture.datasource->cache_chunk_count() == 0);
  fixture.advise(0);
  // One read_size range at offset 0 lies inside the first chunk.
  static_assert(read_size <= controlled_chunk_bytes);
  CHECK(fixture.datasource->cache_chunk_count() == 1);
  CHECK(fixture.datasource->demand_wait_ns() == 0);
  std::atomic<bool> prefetched{false};
  std::atomic<int> callbacks{0};
  std::array<std::uint8_t, read_size> destination{};
  held_request prefetch;
  std::future<std::future<std::size_t>> read_call;
  read_cleanup cleanup{*fixture.context, {&prefetch, nullptr}};
  REQUIRE(fixture.datasource->prefetch_async([&](prefetch_result result) noexcept {
    prefetched.store(result.ok, std::memory_order_release);
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
  // The read parked on the in-flight prefetch (it was still blocked after
  // 50ms above), and that wait is charged to the datasource's handle.
  CHECK(fixture.datasource->demand_wait_ns() > 0);
  CHECK(fixture.datasource->cache_chunk_count() == 1);
}

TEST_CASE("a cached device read retires its pin through the stream completion poll",
          "[cache][cache_handle][completion-poll][gpu_execution]")
{
  if (run_in_child("CUCASCADE_CACHE_ARBITRATION_CHILD", 30s)) { return; }
  arbitration_fixture fixture;
  fixture.advise(0);
  std::atomic<bool> prefetched{false};
  held_request prefetch;
  read_cleanup cleanup{*fixture.context, {&prefetch, nullptr}};
  REQUIRE(fixture.datasource->prefetch_async([&](prefetch_result result) noexcept {
    prefetched.store(result.ok, std::memory_order_release);
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
  // A pin the copy failed to retire would keep chunk 0 from being reclaimed.
  fixture.datasource.reset();
  auto& cache = *fixture.context->cache();
  CHECK(wait_for_cache(
    [&] {
      cache.evict_sync(controlled_chunk_bytes);
      return cache.claimed_bytes() == 0;
    },
    2s));
}

TEST_CASE("a cache read does not wait for unrelated chunks in the same handle",
          "[cache][cache_handle][read-arbitration]")
{
  if (run_in_child("CUCASCADE_CACHE_ARBITRATION_CHILD", 30s)) { return; }
  arbitration_fixture fixture;
  fixture.advise(0);
  std::array<std::uint8_t, read_size> destination{};
  held_request prefetch;
  held_request demand;
  std::future<std::future<std::size_t>> read_call;
  read_cleanup cleanup{*fixture.context, {&prefetch, &demand}};
  REQUIRE(fixture.datasource->prefetch_async([](prefetch_result) noexcept {}) ==
          prefetch_refusal::issued);
  prefetch.request = fixture.context->backend().wait_take(2s);
  REQUIRE(prefetch.request != nullptr);
  CHECK(prefetch.request->priority == cucascade::io::io_priority::low);
  read_call           = std::async(std::launch::async, [&] {
    return fixture.datasource->host_read_async(
      2 * controlled_chunk_bytes, destination.size(), destination.data());
  });
  auto const returned = read_call.wait_for(100ms) == std::future_status::ready;
  CHECK(returned);
  if (!returned) { prefetch.complete(0x5a); }
  REQUIRE(read_call.wait_for(2s) == std::future_status::ready);
  auto read      = read_call.get();
  demand.request = fixture.context->backend().wait_take(2s);
  REQUIRE(demand.request != nullptr);
  CHECK(demand.request->priority == cucascade::io::io_priority::high);
  CHECK(demand.request->front().h_buffer.is_contiguous());
  demand.complete(0x2b);
  if (prefetch.request) { prefetch.complete(0x5a); }
  REQUIRE(read.wait_for(2s) == std::future_status::ready);
  CHECK(read.get() == destination.size());
  CHECK(std::ranges::all_of(destination, [](auto byte) { return byte == 0x2b; }));
}

TEST_CASE("a cached multi-range host read sends its backend work on the high tier",
          "[cache][cache_handle][read-arbitration][priority]")
{
  if (run_in_child("CUCASCADE_CACHE_ARBITRATION_CHILD", 30s)) { return; }
  // A native decoder's vectored host read is demand work: left at automatic,
  // the part the cache cannot serve must not land on the low (prefetch) tier.
  arbitration_fixture fixture;
  fixture.advise(0);
  std::array<std::uint8_t, 2 * read_size> destination{};
  std::array<cucascade::io::slice, 2> const ranges{
    cucascade::io::slice{2 * controlled_chunk_bytes, read_size, destination.data()},
    cucascade::io::slice{3 * controlled_chunk_bytes, read_size, destination.data() + read_size}};
  held_request first;
  held_request second;
  read_cleanup cleanup{*fixture.context, {&first, &second}};
  auto read     = fixture.datasource->host_read_ranges_async(ranges);
  first.request = fixture.context->backend().wait_take(2s);
  REQUIRE(first.request != nullptr);
  CHECK(first.request->priority == cucascade::io::io_priority::high);
  // The two ranges may share one request or be split across two.
  second.request = fixture.context->backend().wait_take(100ms);
  if (second.request) { CHECK(second.request->priority == cucascade::io::io_priority::high); }
  first.complete(0x3c);
  if (second.request) { second.complete(0x3c); }
  REQUIRE(read.wait_for(2s) == std::future_status::ready);
  CHECK(read.get() == destination.size());
  CHECK(std::ranges::all_of(destination, [](auto byte) { return byte == 0x3c; }));
}

TEST_CASE("a demand-owned loading chunk uses reactor bounce staging",
          "[cache][cache_handle][read-arbitration]")
{
  if (run_in_child("CUCASCADE_CACHE_ARBITRATION_CHILD", 30s)) { return; }
  arbitration_fixture fixture;
  fixture.advise(controlled_chunk_bytes);
  rmm::cuda_stream stream;
  rmm::device_buffer first_destination(read_size, stream);
  rmm::device_buffer second_destination(read_size, stream);
  stream_cleanup drain{stream};
  held_request cache_load;
  held_request bounced;
  std::future<std::future<std::size_t>> second_call;
  read_cleanup cleanup{*fixture.context, {&cache_load, &bounced}};
  auto first =
    fixture.datasource->device_read_async(controlled_chunk_bytes,
                                          read_size,
                                          static_cast<std::uint8_t*>(first_destination.data()),
                                          stream);
  cache_load.request = fixture.context->backend().wait_take(2s);
  REQUIRE(cache_load.request != nullptr);
  CHECK(cache_load.request->front().h_buffer.is_fragmented());
  second_call         = std::async(std::launch::async, [&] {
    return fixture.datasource->device_read_async(
      controlled_chunk_bytes,
      read_size,
      static_cast<std::uint8_t*>(second_destination.data()),
      stream);
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
  // Distinct fills: the first read must carry the chunk's bytes, the second
  // the bounced bytes.
  bounced.complete(0x7e);
  if (cache_load.request) { cache_load.complete(0x5a); }
  REQUIRE(first.wait_for(2s) == std::future_status::ready);
  REQUIRE(second.wait_for(2s) == std::future_status::ready);
  CHECK(first.get() == read_size);
  CHECK(second.get() == read_size);
  std::array<std::uint8_t, read_size> host{};
  REQUIRE(
    cudaMemcpyAsync(
      host.data(), first_destination.data(), host.size(), cudaMemcpyDeviceToHost, stream.value()) ==
    cudaSuccess);
  stream.synchronize();
  CHECK(std::ranges::all_of(host, [](auto byte) { return byte == 0x5a; }));
  REQUIRE(cudaMemcpyAsync(host.data(),
                          second_destination.data(),
                          host.size(),
                          cudaMemcpyDeviceToHost,
                          stream.value()) == cudaSuccess);
  stream.synchronize();
  CHECK(std::ranges::all_of(host, [](auto byte) { return byte == 0x7e; }));
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

TEST_CASE("raw cache validates identities across opens", "[cache][object_identity]")
{
  if (run_in_child("CUCASCADE_CACHE_ARBITRATION_CHILD", 30s)) { return; }
  arbitration_fixture fixture;
  int const mode            = GENERATE(0, 1, 2);  // same version, changed version, no validator
  std::string const version = mode == 2 ? "" : "v1";
  held_request fill;
  held_request miss;
  read_cleanup cleanup{*fixture.context, {&fill, &miss}};
  fixture.datasource = fixture.open_version("controlled://versioned", version);
  fixture.advise(0);
  fixture.prefetch(fill);
  fill.complete(0x5a);
  auto reopened = fixture.open_version("controlled://versioned", mode == 1 ? "v2" : version);
  std::array<std::uint8_t, read_size> bytes{};
  auto read = reopened->host_read_async(0, bytes.size(), bytes.data());
  if (mode != 0) {
    miss.request = fixture.context->backend().wait_take(2s);
    REQUIRE(miss.request != nullptr);
    miss.complete(0x7b);
  }
  REQUIRE(read.wait_for(2s) == std::future_status::ready);
  CHECK(read.get() == bytes.size());
  CHECK(std::ranges::all_of(bytes, [&](auto b) { return b == (mode == 0 ? 0x5a : 0x7b); }));
  CHECK(fixture.context->backend().pending_count() == 0);
}

TEST_CASE("a late old raw-cache fill cannot overwrite the new version", "[cache][object_identity]")
{
  if (run_in_child("CUCASCADE_CACHE_ARBITRATION_CHILD", 30s)) { return; }
  arbitration_fixture fixture;
  held_request old_fill;
  held_request new_fill;
  read_cleanup cleanup{*fixture.context, {&old_fill, &new_fill}};
  fixture.datasource = fixture.open_version("controlled://versioned", "v1");
  fixture.advise(0);
  fixture.prefetch(old_fill);
  auto old_source    = std::move(fixture.datasource);
  fixture.datasource = fixture.open_version("controlled://versioned", "v2");
  fixture.advise(0);
  fixture.prefetch(new_fill);
  new_fill.complete(0x7b);
  old_fill.complete(0x5a);
  auto reopened = fixture.open_version("controlled://versioned", "v2");
  std::array<std::uint8_t, read_size> bytes{};
  auto read = reopened->host_read_async(0, bytes.size(), bytes.data());
  REQUIRE(read.wait_for(2s) == std::future_status::ready);
  CHECK(read.get() == bytes.size());
  CHECK(std::ranges::all_of(bytes, [](auto b) { return b == 0x7b; }));
  read = old_source->host_read_async(0, bytes.size(), bytes.data());
  REQUIRE(read.wait_for(2s) == std::future_status::ready);
  CHECK(read.get() == bytes.size());
  CHECK(std::ranges::all_of(bytes, [](auto b) { return b == 0x5a; }));
  CHECK(fixture.context->backend().pending_count() == 0);
}

TEST_CASE("superseded raw cache entries release their arenas and handles without pressure",
          "[cache][object_identity][raw_retirement]")
{
  if (run_in_child("CUCASCADE_CACHE_ARBITRATION_CHILD", 30s)) { return; }
  arbitration_fixture fixture;
  bool const validated = GENERATE(true, false);
  std::vector<std::weak_ptr<controlled_object>> versions;
  for (int i = 0; i < 24; ++i) {
    auto object = std::make_shared<controlled_object>(
      "controlled://retirement", 4 * controlled_chunk_bytes, validated ? std::to_string(i) : "");
    versions.emplace_back(object);
    fixture.datasource = std::make_unique<cucascade::io::datasource>(fixture.context, object);
    fixture.advise(0);
    held_request fill;
    fixture.prefetch(fill);
    fill.complete(0x11);
    CHECK(fixture.context->cache()->file_entry_count_for_testing() == 1);
  }
  fixture.datasource.reset();
  // No query, allocation pressure or explicit evict call is needed to retire old versions.
  CHECK(eventually([&] {
    return std::all_of(
      versions.begin(), versions.end() - 1, [](auto const& v) { return v.expired(); });
  }));
  CHECK_FALSE(versions.back().expired());
  CHECK(fixture.context->cache()->claimed_bytes() == controlled_chunk_bytes);
  fixture.context->shutdown_cache();
  CHECK(versions.back().expired());
}

TEST_CASE("retired raw arena survives an abandoned in-flight fill until physical completion",
          "[cache][object_identity][raw_retirement]")
{
  if (run_in_child("CUCASCADE_CACHE_ARBITRATION_CHILD", 30s)) { return; }
  arbitration_fixture fixture;
  held_request fill;
  read_cleanup cleanup{*fixture.context, {&fill, nullptr}};
  auto object = std::make_shared<controlled_object>(
    "controlled://retirement", 4 * controlled_chunk_bytes, "v1");
  std::weak_ptr<controlled_object> old = object;
  fixture.datasource = std::make_unique<cucascade::io::datasource>(fixture.context, object);
  object.reset();
  fixture.advise(0);
  fixture.prefetch(fill);
  fixture.datasource = fixture.open_version("controlled://retirement", "v2");
  fixture.advise(0);
  std::this_thread::sleep_for(250ms);  // let the evictor release the disposed v1 request
  CHECK_FALSE(old.expired());
  CHECK(fixture.context->cache()->claimed_bytes() == 2 * controlled_chunk_bytes);
  fill.complete(0x11);
  CHECK(eventually([&] { return old.expired(); }));
  CHECK(fixture.context->cache()->claimed_bytes() == controlled_chunk_bytes);
}

TEST_CASE("retired raw arena survives a cache-hit CUDA copy without a prefetch handle",
          "[cache][object_identity][raw_retirement][gpu_execution]")
{
  if (run_in_child("CUCASCADE_CACHE_ARBITRATION_CHILD", 30s)) { return; }
  arbitration_fixture fixture;
  held_request fill;
  read_cleanup cleanup{*fixture.context, {&fill, nullptr}};
  auto object = std::make_shared<controlled_object>(
    "controlled://retirement", 4 * controlled_chunk_bytes, "v1");
  std::weak_ptr<controlled_object> old = object;
  fixture.datasource = std::make_unique<cucascade::io::datasource>(fixture.context, object);
  object.reset();
  fixture.advise(0);
  fixture.prefetch(fill);
  fill.complete(0x6c);
  // A separate open of the same version reads the resident chunk without a handle.
  auto reader = fixture.open_version("controlled://retirement", "v1");
  rmm::cuda_stream stream;
  rmm::device_buffer destination{read_size, stream};
  struct gate {
    cudaStream_t stream;
    std::atomic<bool> released{false};
    ~gate()
    {
      released.store(true);
      std::ignore = cudaStreamSynchronize(stream);
    }
  } blocked{stream.value()};
  REQUIRE(cudaLaunchHostFunc(
            stream.value(),
            [](void* ptr) {
              auto& ready = static_cast<gate*>(ptr)->released;
              while (!ready.load()) {
                std::this_thread::sleep_for(1ms);
              }
            },
            &blocked) == cudaSuccess);
  auto read =
    reader->device_read_async(0, read_size, static_cast<std::uint8_t*>(destination.data()), stream);
  reader.reset();
  fixture.datasource = fixture.open_version("controlled://retirement", "v2");
  fixture.advise(0);
  std::this_thread::sleep_for(250ms);
  CHECK(read.wait_for(0ms) != std::future_status::ready);
  CHECK_FALSE(old.expired());
  blocked.released.store(true);
  REQUIRE(read.wait_for(2s) == std::future_status::ready);
  CHECK(read.get() == read_size);
  std::array<std::uint8_t, read_size> host{};
  REQUIRE(cudaMemcpyAsync(
            host.data(), destination.data(), host.size(), cudaMemcpyDeviceToHost, stream.value()) ==
          cudaSuccess);
  stream.synchronize();
  CHECK(std::ranges::all_of(host, [](auto byte) { return byte == 0x6c; }));
  CHECK(eventually([&] { return old.expired(); }));
}
