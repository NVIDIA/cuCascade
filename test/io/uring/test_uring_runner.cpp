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

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <cuda/stream_ref>
#include <cuda_runtime.h>

#include <catch2/catch_all.hpp>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <span>
#include <stop_token>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;
using cucascade::io::uring::uring_ioctx;
using cucascade::io::uring::uring_reactor;

constexpr std::size_t MiB = 1UL << 20;

[[nodiscard]] std::uint8_t pattern_at(std::size_t offset) noexcept
{
  return static_cast<std::uint8_t>((offset * 131U + (offset >> 12U) * 7U) & 0xFFU);
}

/// A temporary file filled with @ref pattern_at, removed on destruction.
class temp_file {
 public:
  explicit temp_file(std::size_t size)
  {
    static std::atomic<int> counter{0};
    _path = std::filesystem::temp_directory_path() /
            ("cucascade_uring_runner_" + std::to_string(::getpid()) + "_" +
             std::to_string(counter.fetch_add(1)) + ".bin");
    std::vector<char> data(size);
    for (std::size_t i = 0; i < size; ++i) {
      data[i] = static_cast<char>(pattern_at(i));
    }
    std::ofstream out(_path, std::ios::binary | std::ios::trunc);
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
  }
  ~temp_file()
  {
    std::error_code ec;
    std::filesystem::remove(_path, ec);
  }
  temp_file(temp_file const&)            = delete;
  temp_file& operator=(temp_file const&) = delete;

  [[nodiscard]] std::string path() const { return _path.string(); }

 private:
  std::filesystem::path _path;
};

/// Pinned staging resource (1 MiB blocks) shared by the engines of a context.
class staging_resource {
 public:
  explicit staging_resource(std::size_t capacity = 320 * MiB)
    : _upstream(0, /*make_portable=*/true), _mr(0, _upstream, capacity, capacity, 1 * MiB)
  {
  }

  [[nodiscard]] std::shared_ptr<uring_reactor::reactor_context> context(bool use_odirect = true)
  {
    cucascade::io::uring::config cfg{};
    cfg.use_odirect = use_odirect;
    return std::make_shared<uring_reactor::reactor_context>(cfg, &_mr);
  }

 private:
  cucascade::memory::numa_region_pinned_host_memory_resource _upstream;
  cucascade::memory::fixed_size_host_memory_resource _mr;
};

[[nodiscard]] bool matches(std::vector<std::uint8_t> const& data, std::size_t offset)
{
  for (std::size_t i = 0; i < data.size(); ++i) {
    if (data[i] != pattern_at(offset + i)) return false;
  }
  return true;
}

template <class Predicate>
bool wait_until(Predicate&& predicate, std::chrono::milliseconds timeout = 5000ms)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(1ms);
  }
  return true;
}

/// CPU time (user + system) consumed by the calling thread.
[[nodiscard]] std::chrono::microseconds thread_cpu_time()
{
  rusage usage{};
  ::getrusage(RUSAGE_THREAD, &usage);
  auto const to_us = [](timeval const& tv) {
    return std::chrono::seconds{tv.tv_sec} + std::chrono::microseconds{tv.tv_usec};
  };
  return std::chrono::duration_cast<std::chrono::microseconds>(to_us(usage.ru_utime) +
                                                               to_us(usage.ru_stime));
}

/// Read [offset, offset + size) and verify the content.
void check_read(uring_ioctx& ctx,
                cucascade::io::io_object const& object,
                std::size_t offset,
                std::size_t size)
{
  std::vector<std::uint8_t> data(size);
  auto const bytes = ctx.host_read_async(object, offset, size, data.data()).get(10s);
  CHECK(bytes == size);
  CHECK(matches(data, offset));
}

}  // namespace

TEST_CASE("uring run_for without work returns at the deadline without spinning",
          "[io][uring][runner]")
{
  staging_resource staging;
  temp_file file(256 * 1024);
  uring_ioctx ctx(0, staging.context());
  auto object = ctx.open_io_object(file.path());

  SECTION("idle")
  {
    auto const cpu_before = thread_cpu_time();
    auto const start      = std::chrono::steady_clock::now();
    CHECK(ctx.run_for(300ms) == 0);
    auto const elapsed = std::chrono::steady_clock::now() - start;
    auto const cpu     = thread_cpu_time() - cpu_before;

    CHECK(elapsed >= 300ms);
    CHECK(elapsed < 2s);
    // Idle: the runner sleeps in the CQE wait (engine construction is the bulk).
    CHECK(cpu < 150ms);
  }

  SECTION("woken for work, then idle again")
  {
    std::atomic<bool> read_ok{false};
    std::jthread client([&] {
      std::this_thread::sleep_for(50ms);
      std::vector<std::uint8_t> data(64 * 1024);
      auto const bytes = ctx.host_read_async(*object, 4096, data.size(), data.data()).get(5s);
      read_ok.store(bytes == data.size() && matches(data, 4096));
    });
    auto const cpu_before = thread_cpu_time();
    CHECK(ctx.run_for(400ms) == 1);
    auto const cpu = thread_cpu_time() - cpu_before;
    client.join();
    CHECK(read_ok.load());
    // The wakeup (eventfd poll) must not keep firing once the work is done.
    CHECK(cpu < 150ms);
  }
  CHECK(ctx.active_runners() == 0);
}

TEST_CASE("uring run_until serves requests submitted while it runs", "[io][uring][runner]")
{
  staging_resource staging;
  temp_file file(4 * MiB);
  uring_ioctx ctx(0, staging.context());
  auto object = ctx.open_io_object(file.path());

  std::atomic<std::size_t> served{0};
  auto const deadline = std::chrono::steady_clock::now() + 1500ms;
  std::jthread runner([&] { served.store(ctx.run_until(deadline)); });
  REQUIRE(wait_until([&] { return ctx.active_runners() == 1; }));

  check_read(ctx, *object, 0, 4 * MiB);
  check_read(ctx, *object, 12345, 100'000);  // latency class, unaligned
  runner.join();
  CHECK(std::chrono::steady_clock::now() >= deadline);
  CHECK(served.load() == 2);
}

TEST_CASE("uring external runner is stopped by shutdown", "[io][uring][runner]")
{
  staging_resource staging;
  temp_file file(8 * MiB);
  uring_ioctx ctx(0, staging.context());
  auto object = ctx.open_io_object(file.path());

  std::atomic<std::size_t> served{0};
  std::jthread runner([&] { served.store(ctx.run(std::stop_token{})); });
  REQUIRE(wait_until([&] { return ctx.active_runners() == 1; }));

  for (std::size_t i = 0; i < 8; ++i) {
    check_read(ctx, *object, i * MiB, MiB);
  }
  ctx.shutdown();  // wakes the runner blocked in its CQE wait and waits for it
  CHECK(ctx.active_runners() == 0);
  runner.join();
  CHECK(served.load() == 8);
}

TEST_CASE("uring start, shutdown and start again", "[io][uring][runner]")
{
  staging_resource staging;
  temp_file file(2 * MiB);
  uring_ioctx ctx(2, staging.context());
  auto object = ctx.open_io_object(file.path());

  ctx.start();
  CHECK(ctx.active_runners() == 2);
  check_read(ctx, *object, 0, 2 * MiB);

  ctx.shutdown();
  CHECK(ctx.active_runners() == 0);
  std::vector<std::uint8_t> data(4096);
  CHECK_THROWS_MATCHES(ctx.host_read_async(*object, 0, data.size(), data.data()).get(5s),
                       std::system_error,
                       Catch::Matchers::Predicate<std::system_error>(
                         [](auto const& e) { return e.code() == std::errc::operation_canceled; }));

  ctx.start();
  CHECK(ctx.active_runners() == 2);
  check_read(ctx, *object, 4096, MiB);
  ctx.shutdown();
}

TEST_CASE("uring request before start is cancelled", "[io][uring][runner]")
{
  staging_resource staging;
  temp_file file(64 * 1024);
  uring_ioctx ctx(1, staging.context());
  auto object = ctx.open_io_object(file.path());

  std::vector<std::uint8_t> data(4096);
  CHECK_THROWS_MATCHES(ctx.host_read_async(*object, 0, data.size(), data.data()).get(5s),
                       std::system_error,
                       Catch::Matchers::Predicate<std::system_error>(
                         [](auto const& e) { return e.code() == std::errc::operation_canceled; }));
  // The synchronous path does not need a runner.
  CHECK(ctx.host_read(*object, 0, data.size(), data.data()) == data.size());
  CHECK(matches(data, 0));
}

TEST_CASE("uring two external runners share the work", "[io][uring][runner]")
{
  staging_resource staging;
  temp_file file(64 * MiB);
  uring_ioctx ctx(0, staging.context());
  auto object = ctx.open_io_object(file.path());

  std::atomic<std::size_t> served_a{0};
  std::atomic<std::size_t> served_b{0};
  std::jthread a([&](std::stop_token stop) { served_a.store(ctx.run(stop)); });
  std::jthread b([&](std::stop_token stop) { served_b.store(ctx.run(stop)); });
  REQUIRE(wait_until([&] { return ctx.stats().idle_runners == 2; }));
  CHECK(ctx.stats().active_runners == 2);

  constexpr std::size_t n_requests = 64;
  std::vector<std::vector<std::uint8_t>> buffers(n_requests, std::vector<std::uint8_t>(MiB));
  std::vector<cucascade::exec::semi_future<std::size_t>> futures;
  for (std::size_t i = 0; i < n_requests; ++i) {
    futures.push_back(ctx.host_read_async(*object, i * MiB, MiB, buffers[i].data()));
  }
  for (std::size_t i = 0; i < n_requests; ++i) {
    CHECK(std::move(futures[i]).get(10s) == MiB);
    CHECK(matches(buffers[i], i * MiB));
  }

  a.request_stop();
  b.request_stop();
  a.join();
  b.join();
  CHECK(served_a.load() + served_b.load() == n_requests);
  CHECK(served_a.load() > 0);
  CHECK(served_b.load() > 0);
  CHECK(ctx.active_runners() == 0);
}

TEST_CASE("uring runner expands several groups of mixed classes concurrently",
          "[io][uring][runner]")
{
  staging_resource staging;
  temp_file file(48 * MiB);
  uring_ioctx ctx(1, staging.context());
  auto object = ctx.open_io_object(file.path());
  ctx.start();

  std::mt19937_64 rng(42);
  std::uniform_int_distribution<std::size_t> offsets(0, 40 * MiB);
  std::uniform_int_distribution<std::size_t> small(1, 200'000);
  std::uniform_int_distribution<std::size_t> large(300'000, 8 * MiB);

  struct pending_read {
    std::size_t offset;
    std::vector<std::uint8_t> data;
    cucascade::exec::semi_future<std::size_t> future;
  };
  std::vector<pending_read> reads;
  reads.reserve(48);
  for (std::size_t i = 0; i < 48; ++i) {
    auto const offset = offsets(rng);
    auto const size   = (i % 3 == 0) ? large(rng) : small(rng);
    pending_read entry{offset, std::vector<std::uint8_t>(size), {}};
    entry.future = ctx.host_read_async(*object, offset, size, entry.data.data());
    reads.push_back(std::move(entry));
  }
  for (auto& entry : reads) {
    CHECK(std::move(entry.future).get(20s) == entry.data.size());
    CHECK(matches(entry.data, entry.offset));
  }
  ctx.shutdown();
}

TEST_CASE("uring retiring runner hands untaken work to another runner", "[io][uring][runner]")
{
  staging_resource staging;
  constexpr std::size_t slice_bytes = 32 * 1024;
  constexpr std::size_t n_slices    = 512;  // more slices than staging slots (64)
  temp_file file(slice_bytes * n_slices);
  uring_ioctx ctx(0, staging.context());
  auto object = ctx.open_io_object(file.path());

  // Every slice completion is slowed down so the short-lived runner's deadline
  // passes while most of the group is still untaken: it must hand the group
  // back (requeue), finish what it already submitted, and a later runner must
  // complete the rest.
  std::atomic<std::size_t> completions{0};
  auto on_complete = std::make_shared<cucascade::io::prepared_io_completion>(
    [&](std::span<cucascade::io::cache::cached_chunk* const>, bool success) noexcept {
      std::this_thread::sleep_for(1ms);
      if (success) completions.fetch_add(1);
    });

  std::atomic<std::size_t> retired_short{0};
  std::jthread short_lived([&] { retired_short.store(ctx.run_for(300ms)); });
  // Parked = its engine exists and it waits for work.
  REQUIRE(wait_until([&] { return ctx.stats().idle_runners == 1; }));

  std::vector<std::uint8_t> data(slice_bytes * n_slices);
  std::vector<cucascade::io::prepared_io_slice> slices;
  for (std::size_t i = 0; i < n_slices; ++i) {
    slices.emplace_back(cucascade::io::range{i * slice_bytes, slice_bytes},
                        cucascade::io::host_buffer{data.data() + i * slice_bytes});
    slices.back().on_complete = on_complete;
  }
  auto future = ctx.mixed_readv_async_io(*object, std::move(slices));
  short_lived.join();
  CHECK(retired_short.load() == 0);  // the group was handed back, not retired
  auto const served_by_short = completions.load();
  CHECK(served_by_short > 0);
  CHECK(served_by_short < n_slices);

  std::atomic<std::size_t> retired_finisher{0};
  std::jthread finisher(
    [&](std::stop_token stop) { retired_finisher.store(ctx.run(std::move(stop))); });
  CHECK(std::move(future).get(30s) == slice_bytes * n_slices);
  CHECK(matches(data, 0));
  CHECK(completions.load() == n_slices);
  finisher.request_stop();
  finisher.join();
  CHECK(retired_finisher.load() == 1);  // the requeued group retired here
  CHECK(ctx.active_runners() == 0);
}

TEST_CASE("uring device read through pinned staging", "[io][uring][runner][gpu]")
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    static_cast<void>(cudaGetLastError());
    SKIP("no CUDA device");
  }

  staging_resource staging;
  temp_file file(24 * MiB);
  uring_ioctx ctx(2, staging.context());
  auto object = ctx.open_io_object(file.path());
  ctx.start();

  constexpr std::size_t offset = 4096 * 3 + 17;  // unaligned: exercises the aligned over-read
  constexpr std::size_t size   = 20 * MiB + 333;
  std::uint8_t* device         = nullptr;
  REQUIRE(cudaMalloc(&device, size) == cudaSuccess);
  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  auto const bytes =
    ctx.device_read_async(*object, offset, size, device, ::cuda::stream_ref{stream}).get(20s);
  CHECK(bytes == size);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

  std::vector<std::uint8_t> host(size);
  REQUIRE(cudaMemcpy(host.data(), device, size, cudaMemcpyDeviceToHost) == cudaSuccess);
  CHECK(matches(host, offset));

  ctx.shutdown();
  static_cast<void>(cudaStreamDestroy(stream));
  static_cast<void>(cudaFree(device));
}
