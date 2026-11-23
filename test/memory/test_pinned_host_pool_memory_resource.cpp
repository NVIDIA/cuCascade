/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cucascade/cuda/stream.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/pinned_host_pool_memory_resource.hpp>

#include <rmm/error.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using cucascade::memory::pinned_host_pool_config;
using cucascade::memory::pinned_host_pool_memory_resource;

namespace {

constexpr std::size_t pool_limit = 64ull << 20;
constexpr std::size_t mib        = 1ull << 20;
constexpr std::size_t pool_alignment =
  cucascade::memory::detail::pinned_host_pool_memory_resource_impl::pool_alignment;

[[nodiscard]] pinned_host_pool_config test_config()
{
  return pinned_host_pool_config{.numa_node = 0, .max_pool_bytes = pool_limit};
}

/// Owns a non-blocking CUDA stream on the current device.
class test_stream {
 public:
  test_stream()
  {
    REQUIRE(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking) == cudaSuccess);
  }
  ~test_stream() { (void)cudaStreamDestroy(_stream); }
  test_stream(test_stream const&)            = delete;
  test_stream& operator=(test_stream const&) = delete;

  [[nodiscard]] cudaStream_t get() const noexcept { return _stream; }
  [[nodiscard]] cuda::stream_ref ref() const noexcept { return cuda::stream_ref{_stream}; }
  void sync() const { REQUIRE(cudaStreamSynchronize(_stream) == cudaSuccess); }

 private:
  cudaStream_t _stream{};
};

/// Owns a CUDA event created without timing.
class test_event {
 public:
  test_event()
  {
    REQUIRE(cudaEventCreateWithFlags(&_event, cudaEventDisableTiming) == cudaSuccess);
  }
  ~test_event() { (void)cudaEventDestroy(_event); }
  test_event(test_event const&)            = delete;
  test_event& operator=(test_event const&) = delete;

  [[nodiscard]] cudaEvent_t get() const noexcept { return _event; }

 private:
  cudaEvent_t _event{};
};

/// Owns a device buffer from `cudaMalloc`.
class device_buffer {
 public:
  explicit device_buffer(std::size_t bytes) { REQUIRE(cudaMalloc(&_ptr, bytes) == cudaSuccess); }
  ~device_buffer() { (void)cudaFree(_ptr); }
  device_buffer(device_buffer const&)            = delete;
  device_buffer& operator=(device_buffer const&) = delete;

  [[nodiscard]] void* get() const noexcept { return _ptr; }

 private:
  void* _ptr{};
};

/**
 * @brief Checks that `pinned` holds data through a host -> device -> host trip on `stream`.
 *
 * The host writes a pattern derived from `seed`, the pattern is copied to a device buffer, the host
 * clears `pinned`, the device copy is copied back and the bytes are compared. The caller must have
 * made `pinned` usable by the host and by `stream` before the call.
 */
void require_round_trip(void* pinned, std::size_t bytes, cudaStream_t stream, std::uint8_t seed = 0)
{
  REQUIRE(pinned != nullptr);
  auto* const host = static_cast<std::uint8_t*>(pinned);
  std::vector<std::uint8_t> expected(bytes);
  for (std::size_t i = 0; i < bytes; ++i) {
    expected[i] = static_cast<std::uint8_t>((i * 131u + seed) & 0xFFu);
  }
  std::memcpy(host, expected.data(), bytes);

  device_buffer device{bytes};
  REQUIRE(cudaMemcpyAsync(device.get(), host, bytes, cudaMemcpyHostToDevice, stream) ==
          cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  std::memset(host, 0, bytes);
  REQUIRE(cudaMemcpyAsync(host, device.get(), bytes, cudaMemcpyDeviceToHost, stream) ==
          cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  REQUIRE(std::memcmp(host, expected.data(), bytes) == 0);
}

}  // namespace

TEST_CASE("HostNuma pool can be created", "[pinned_host_pool]")
{
  pinned_host_pool_memory_resource pool{test_config()};
  test_stream stream;
  void* const ptr = pool.allocate(stream.ref(), mib, pool_alignment);
  stream.sync();
  require_round_trip(ptr, mib, stream.get());
  pool.deallocate(stream.ref(), ptr, mib, pool_alignment);
  stream.sync();
}

TEST_CASE("GPU work can write pool memory directly", "[pinned_host_pool]")
{
  // A copy to or from pool memory works even if the GPU was never granted access to the pool;
  // `cudaMemsetAsync` makes the GPU write the pool pages itself, so it fails without that access.
  pinned_host_pool_memory_resource pool{test_config()};
  test_stream stream;
  void* const ptr = pool.allocate(stream.ref(), mib, pool_alignment);
  REQUIRE(cudaMemsetAsync(ptr, 0x5A, mib, stream.get()) == cudaSuccess);
  stream.sync();
  auto const* const bytes = static_cast<std::uint8_t const*>(ptr);
  REQUIRE(std::all_of(bytes, bytes + mib, [](std::uint8_t b) { return b == 0x5A; }));
  pool.deallocate(stream.ref(), ptr, mib, pool_alignment);
  stream.sync();
}

TEST_CASE("Invalid config throws logic_error", "[pinned_host_pool]")
{
  auto config = test_config();
  SECTION("negative NUMA node") { config.numa_node = -1; }
  SECTION("lowest NUMA node value") { config.numa_node = std::numeric_limits<int>::min(); }
  SECTION("zero byte limit") { config.max_pool_bytes = 0; }
  SECTION("default-constructed config") { config = pinned_host_pool_config{}; }
  CHECK_THROWS_AS(pinned_host_pool_memory_resource{config}, cucascade::logic_error);
}

TEST_CASE("Nonexistent NUMA node throws cuda_error", "[pinned_host_pool]")
{
  auto config      = test_config();
  config.numa_node = 1 << 20;
  try {
    pinned_host_pool_memory_resource pool{config};
    FAIL("pool creation on NUMA node " << config.numa_node << " did not throw");
  } catch (cucascade::cuda_error const& e) {
    std::string const what{e.what()};
    CHECK_THAT(what, Catch::Matchers::ContainsSubstring("1048576"));
    CHECK_THAT(what, Catch::Matchers::ContainsSubstring("cudaError"));
  }
}

// Hidden: needs a process that sees no GPU. The `cucascade_pinned_host_pool_without_devices` CTest
// entry runs it with `CUDA_VISIBLE_DEVICES=-1`.
TEST_CASE("CUDA initialization failure throws cuda_error", "[.][pinned_host_pool_no_device]")
{
  try {
    pinned_host_pool_memory_resource pool{test_config()};
    FAIL("pool creation without a visible GPU did not throw");
  } catch (cucascade::cuda_error const& e) {
    std::string const what{e.what()};
    CHECK_THAT(what, Catch::Matchers::ContainsSubstring("cudaErrorNoDevice"));
    CHECK_THAT(what, Catch::Matchers::ContainsSubstring("NUMA node 0"));
  }
}

TEST_CASE("Copies share one pool; distinct pools are unequal", "[pinned_host_pool]")
{
  test_stream stream;
  std::optional<pinned_host_pool_memory_resource> original{std::in_place, test_config()};
  pinned_host_pool_memory_resource copy{*original};
  CHECK(copy == *original);

  pinned_host_pool_memory_resource const other{test_config()};
  CHECK(other != *original);
  CHECK(other != copy);

  pinned_host_pool_memory_resource assigned{test_config()};
  CHECK(assigned != copy);
  assigned = copy;
  CHECK(assigned == copy);

  void* const ptr = original->allocate(stream.ref(), mib, pool_alignment);
  stream.sync();
  CHECK(copy.used_bytes() >= mib);
  original.reset();

  require_round_trip(ptr, mib, stream.get());
  copy.deallocate(stream.ref(), ptr, mib, pool_alignment);
  stream.sync();
  CHECK(assigned.used_bytes() == 0);
}

TEST_CASE("Satisfies resource_with and round-trips through any_resource", "[pinned_host_pool]")
{
  static_assert(cuda::mr::resource_with<pinned_host_pool_memory_resource,
                                        cuda::mr::device_accessible,
                                        cuda::mr::host_accessible>);
  static_assert(std::is_copy_constructible_v<pinned_host_pool_memory_resource>);
  static_assert(std::is_copy_assignable_v<pinned_host_pool_memory_resource>);
  static_assert(std::is_constructible_v<rmm::host_device_async_resource_ref,
                                        pinned_host_pool_memory_resource&>);

  pinned_host_pool_memory_resource pool{test_config()};
  cuda::mr::any_resource<cuda::mr::device_accessible, cuda::mr::host_accessible> erased{pool};
  test_stream stream;

  void* const ptr = erased.allocate(stream.ref(), mib, pool_alignment);
  stream.sync();
  CHECK(pool.used_bytes() >= mib);
  require_round_trip(ptr, mib, stream.get());
  erased.deallocate(stream.ref(), ptr, mib, pool_alignment);
  stream.sync();
  CHECK(pool.used_bytes() == 0);
}

TEST_CASE("Alignment is validated and pointers are pool-aligned", "[pinned_host_pool]")
{
  pinned_host_pool_memory_resource pool{test_config()};
  test_stream stream;

  for (std::size_t const alignment :
       {std::size_t{0}, std::size_t{3}, pool_alignment + pool_alignment / 2, 2 * pool_alignment}) {
    INFO("alignment " << alignment);
    CHECK_THROWS_AS(pool.allocate(stream.ref(), 64, alignment), cucascade::logic_error);
    CHECK_THROWS_AS(pool.allocate_sync(64, alignment), cucascade::logic_error);
  }

  for (std::size_t alignment = 1; alignment <= pool_alignment; alignment *= 2) {
    for (std::size_t const bytes :
         {std::size_t{1}, std::size_t{100}, std::size_t{4096}, std::size_t{70000}, mib}) {
      INFO("alignment " << alignment << ", bytes " << bytes);
      void* const ptr = pool.allocate(stream.ref(), bytes, alignment);
      stream.sync();
      CHECK(reinterpret_cast<std::uintptr_t>(ptr) % pool_alignment == 0);
      require_round_trip(ptr, bytes, stream.get());
      pool.deallocate(stream.ref(), ptr, bytes, alignment);
    }
  }
  stream.sync();
  CHECK(pool.used_bytes() == 0);
}

TEST_CASE("Zero bytes and null deallocate", "[pinned_host_pool]")
{
  pinned_host_pool_memory_resource pool{test_config()};
  test_stream stream;

  static_assert(noexcept(pool.deallocate(stream.ref(), nullptr, 0, pool_alignment)));
  static_assert(noexcept(pool.deallocate_sync(nullptr, 0)));

  CHECK(pool.allocate(stream.ref(), 0, pool_alignment) == nullptr);
  CHECK(pool.allocate_sync(0) == nullptr);
  pool.deallocate(stream.ref(), nullptr, 0, pool_alignment);
  pool.deallocate(stream.ref(), nullptr, 64, pool_alignment);
  pool.deallocate_sync(nullptr, 0);
  pool.deallocate_sync(nullptr, 64);
  stream.sync();
  CHECK(pool.used_bytes() == 0);
  CHECK(pool.reserved_bytes() == 0);
}

TEST_CASE("Zero bytes on a default stream make no driver call", "[pinned_host_pool]")
{
  // A new thread has no current CUDA context, so any driver call on the legacy or per-thread
  // default stream fails there. A zero-byte request that returns nullptr there never reached the
  // driver.
  pinned_host_pool_memory_resource pool{test_config()};

  for (cudaStream_t const handle : {cudaStream_t{}, cudaStreamPerThread}) {
    INFO("stream handle " << static_cast<void const*>(handle));
    void* zero_byte_ptr     = &pool;
    bool zero_byte_threw    = false;
    bool driver_call_failed = false;
    std::thread{[&] {
      cuda::stream_ref const stream{handle};
      try {
        zero_byte_ptr = pool.allocate(stream, 0, pool_alignment);
      } catch (...) {
        zero_byte_threw = true;
      }
      // Confirms the premise: a request that does reach the driver fails in this thread.
      try {
        void* const ptr = pool.allocate(stream, mib, pool_alignment);
        pool.deallocate(stream, ptr, mib, pool_alignment);
      } catch (...) {
        driver_call_failed = true;
      }
    }}.join();

    CHECK_FALSE(zero_byte_threw);
    CHECK(zero_byte_ptr == nullptr);
    CHECK(driver_call_failed);
  }
  CHECK(pool.used_bytes() == 0);
}

TEST_CASE("Live allocations have distinct pointers", "[pinned_host_pool]")
{
  pinned_host_pool_memory_resource pool{test_config()};
  test_stream stream;

  struct allocation {
    std::uint8_t* ptr;
    std::size_t bytes;
  };
  std::vector<allocation> live;
  for (std::size_t i = 0; i < 16; ++i) {
    std::size_t const bytes = (i % 4 == 0) ? (i + 1) * 64 * 1024 : 1 + i * 777;
    live.push_back(
      {static_cast<std::uint8_t*>(pool.allocate(stream.ref(), bytes, pool_alignment)), bytes});
  }
  stream.sync();

  for (std::size_t i = 0; i < live.size(); ++i) {
    for (std::size_t j = i + 1; j < live.size(); ++j) {
      INFO("allocations " << i << " and " << j);
      bool const disjoint =
        live[i].ptr + live[i].bytes <= live[j].ptr || live[j].ptr + live[j].bytes <= live[i].ptr;
      CHECK(disjoint);
    }
  }
  for (std::size_t i = 0; i < live.size(); ++i) {
    require_round_trip(live[i].ptr, live[i].bytes, stream.get(), static_cast<std::uint8_t>(i));
  }
  for (auto const& a : live) {
    pool.deallocate(stream.ref(), a.ptr, a.bytes, pool_alignment);
  }
  stream.sync();
  CHECK(pool.used_bytes() == 0);
}

namespace {

constexpr std::size_t oom_step = 8 * mib;

/**
 * @brief Allocates `oom_step` bytes at a time on `stream` until the pool throws.
 *
 * Requires the throw to be `rmm::out_of_memory` and to come within `pool_limit / oom_step + 1`
 * tries. Returns the allocations that succeeded; the caller frees them.
 */
[[nodiscard]] std::vector<void*> fill_until_out_of_memory(pinned_host_pool_memory_resource& pool,
                                                          cuda::stream_ref stream)
{
  std::vector<void*> live;
  constexpr std::size_t max_tries = pool_limit / oom_step + 1;
  bool threw                      = false;
  for (std::size_t i = 0; i < max_tries && !threw; ++i) {
    try {
      live.push_back(pool.allocate(stream, oom_step, pool_alignment));
    } catch (rmm::out_of_memory const&) {
      threw = true;
    }
  }
  CHECK(threw);
  CHECK(live.size() <= pool_limit / oom_step);
  return live;
}

/// Round-trips data through each allocation from `fill_until_out_of_memory`, then frees it.
void round_trip_and_free(pinned_host_pool_memory_resource& pool,
                         std::vector<void*> const& live,
                         test_stream const& stream)
{
  stream.sync();
  for (std::size_t i = 0; i < live.size(); ++i) {
    require_round_trip(live[i], oom_step, stream.get(), static_cast<std::uint8_t>(i));
    pool.deallocate(stream.ref(), live[i], oom_step, pool_alignment);
  }
}

}  // namespace

TEST_CASE("Exceeding the limit throws rmm::out_of_memory", "[pinned_host_pool]")
{
  pinned_host_pool_memory_resource pool{test_config()};
  test_stream stream;

  SECTION("many requests that together exceed the limit")
  {
    round_trip_and_free(pool, fill_until_out_of_memory(pool, stream.ref()), stream);
  }
  SECTION("one stream-ordered request above the limit")
  {
    CHECK_THROWS_AS(pool.allocate(stream.ref(), pool_limit + 1, pool_alignment),
                    rmm::out_of_memory);
  }
  SECTION("one synchronous request above the limit")
  {
    CHECK_THROWS_AS(pool.allocate_sync(pool_limit + 1), rmm::out_of_memory);
  }
  stream.sync();
  CHECK(pool.used_bytes() == 0);
}

TEST_CASE("One request above a limit that is not a growth-step multiple throws",
          "[pinned_host_pool]")
{
  // The driver grows the pool in steps (32 MiB has been seen), so a limit between two steps cannot
  // be enforced by the driver alone.
  constexpr std::size_t limit = 40 * mib;
  pinned_host_pool_memory_resource pool{
    pinned_host_pool_config{.numa_node = 0, .max_pool_bytes = limit}};
  test_stream stream;

  CHECK_THROWS_AS(pool.allocate(stream.ref(), limit + 1, pool_alignment), rmm::out_of_memory);
  CHECK_THROWS_AS(pool.allocate_sync(limit + 1), rmm::out_of_memory);
  stream.sync();
  CHECK(pool.used_bytes() == 0);
  CHECK(pool.reserved_bytes() == 0);
}

TEST_CASE("OOM is not sticky", "[pinned_host_pool]")
{
  pinned_host_pool_memory_resource pool{test_config()};
  test_stream stream;

  round_trip_and_free(pool, fill_until_out_of_memory(pool, stream.ref()), stream);
  stream.sync();

  constexpr std::size_t bytes = 32 * mib;
  void* const ptr             = pool.allocate(stream.ref(), bytes, pool_alignment);
  stream.sync();
  require_round_trip(ptr, bytes, stream.get());
  pool.deallocate(stream.ref(), ptr, bytes, pool_alignment);
  stream.sync();
  CHECK(pool.used_bytes() == 0);
}

TEST_CASE("Counters move as documented", "[pinned_host_pool]")
{
  pinned_host_pool_memory_resource pool{test_config()};
  test_stream stream;

  static_assert(noexcept(pool.max_pool_bytes()));
  CHECK(pool.max_pool_bytes() == pool_limit);
  CHECK(pool.used_bytes() == 0);

  constexpr std::size_t bytes = 16 * mib;
  void* const ptr             = pool.allocate(stream.ref(), bytes, pool_alignment);
  stream.sync();
  CHECK(pool.used_bytes() >= bytes);
  CHECK(pool.used_bytes() <= pool.reserved_bytes());
  require_round_trip(ptr, bytes, stream.get());

  pool.deallocate(stream.ref(), ptr, bytes, pool_alignment);
  stream.sync();
  CHECK(pool.used_bytes() == 0);
}

TEST_CASE("Free on one stream, reallocate on another", "[pinned_host_pool]")
{
  pinned_host_pool_memory_resource pool{test_config()};
  test_stream first;
  test_stream second;
  constexpr std::size_t bytes = 4 * mib;

  void* const a = pool.allocate(first.ref(), bytes, pool_alignment);
  first.sync();
  require_round_trip(a, bytes, first.get(), 1);
  pool.deallocate(first.ref(), a, bytes, pool_alignment);

  void* const b = pool.allocate(second.ref(), bytes, pool_alignment);
  second.sync();
  require_round_trip(b, bytes, second.get(), 2);
  pool.deallocate(second.ref(), b, bytes, pool_alignment);
  first.sync();
  second.sync();
  CHECK(pool.used_bytes() == 0);
}

TEST_CASE("Ordered cross-stream allocate, use and free", "[pinned_host_pool]")
{
  pinned_host_pool_memory_resource pool{test_config()};
  test_stream allocating;
  test_stream using_stream;
  test_stream freeing;
  test_event allocated;
  test_event used;
  constexpr std::size_t bytes = 2 * mib;

  void* const ptr = pool.allocate(allocating.ref(), bytes, pool_alignment);
  REQUIRE(cudaEventRecord(allocated.get(), allocating.get()) == cudaSuccess);
  REQUIRE(cudaStreamWaitEvent(using_stream.get(), allocated.get(), 0) == cudaSuccess);
  REQUIRE(cudaEventSynchronize(allocated.get()) == cudaSuccess);
  require_round_trip(ptr, bytes, using_stream.get());
  REQUIRE(cudaEventRecord(used.get(), using_stream.get()) == cudaSuccess);
  REQUIRE(cudaStreamWaitEvent(freeing.get(), used.get(), 0) == cudaSuccess);
  pool.deallocate(freeing.ref(), ptr, bytes, pool_alignment);
  freeing.sync();
  CHECK(pool.used_bytes() == 0);
}

TEST_CASE("allocate_sync memory is host-usable immediately", "[pinned_host_pool]")
{
  pinned_host_pool_memory_resource pool{test_config()};
  constexpr std::size_t bytes = 3 * mib;

  auto* const ptr = static_cast<std::uint8_t*>(pool.allocate_sync(bytes));
  REQUIRE(ptr != nullptr);
  CHECK(reinterpret_cast<std::uintptr_t>(ptr) % pool_alignment == 0);
  std::memset(ptr, 0xA5, bytes);
  CHECK(std::all_of(ptr, ptr + bytes, [](std::uint8_t v) { return v == 0xA5; }));

  test_stream stream;
  require_round_trip(ptr, bytes, stream.get());
  pool.deallocate_sync(ptr, bytes);
  CHECK(pool.used_bytes() == 0);
}

TEST_CASE("Mixed sync and stream-ordered pairs", "[pinned_host_pool]")
{
  pinned_host_pool_memory_resource pool{test_config()};
  test_stream stream;
  constexpr std::size_t bytes = mib;

  SECTION("allocate on a stream, then deallocate_sync")
  {
    void* const ptr = pool.allocate(stream.ref(), bytes, pool_alignment);
    stream.sync();
    require_round_trip(ptr, bytes, stream.get());
    pool.deallocate_sync(ptr, bytes);
  }
  SECTION("allocate_sync, then deallocate on a stream")
  {
    void* const ptr = pool.allocate_sync(bytes);
    require_round_trip(ptr, bytes, stream.get());
    pool.deallocate(stream.ref(), ptr, bytes, pool_alignment);
    stream.sync();
  }
  CHECK(pool.used_bytes() == 0);
}

TEST_CASE("Shared host pool handles tolerate concurrent allocation churn",
          "[pinned_host_pool][stress]")
{
  pinned_host_pool_memory_resource pool{test_config()};
  constexpr std::size_t thread_count = 8;
  constexpr std::size_t iterations   = 200;
  std::array<std::exception_ptr, thread_count> failures{};
  {
    std::vector<std::jthread> workers;
    workers.reserve(thread_count);
    for (std::size_t worker = 0; worker < thread_count; ++worker) {
      workers.emplace_back([copy = pool, worker, &failures]() mutable {
        try {
          CUCASCADE_CUDA_TRY(cudaSetDevice(0));
          // Each thread owns its stream; no Catch2 assertions run on worker threads.
          cuda::stream stream{cuda::device_ref{0}};
          for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            std::size_t const bytes = 1 + ((iteration * 7919 + worker * 997) % (64 * 1024));
            bool const synchronous  = iteration % 2 == 0;
            auto* const ptr         = static_cast<unsigned char*>(
              synchronous ? copy.allocate_sync(bytes)
                          : copy.allocate(stream, bytes, pool_alignment));
            if (!synchronous) { stream.sync(); }
            auto const pattern = static_cast<unsigned char>((iteration + worker) % 256);
            std::memset(ptr, pattern, bytes);
            bool const intact = std::all_of(
              ptr, ptr + bytes, [pattern](unsigned char value) { return value == pattern; });
            // Exercise the counters while other threads queue allocations and frees.
            (void)copy.used_bytes();
            (void)copy.reserved_bytes();
            if (synchronous) {
              copy.deallocate_sync(ptr, bytes);
            } else {
              copy.deallocate(stream, ptr, bytes, pool_alignment);
              stream.sync();
            }
            if (!intact) { throw std::runtime_error{"concurrent allocation data corruption"}; }
          }
        } catch (...) {
          failures[worker] = std::current_exception();
        }
      });
    }
  }
  for (auto const& failure : failures) {
    if (failure) { std::rethrow_exception(failure); }
  }
  CHECK(pool.used_bytes() == 0);
}
