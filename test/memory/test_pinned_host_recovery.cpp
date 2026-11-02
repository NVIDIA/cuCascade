/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pinned_host_test_hooks.hpp"

#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>

namespace {

struct recovery_fixture {
  rmm::mr::pinned_host_memory_resource pinned;
  cucascade::memory::fixed_size_host_memory_resource upstream{
    0, pinned, 1ULL << 20, 1ULL << 20, 64ULL << 10, 4, 1};
  cucascade::memory::small_pinned_host_memory_resource resource;

  explicit recovery_fixture(std::size_t limit = 64ULL << 10) : resource{upstream, limit} {}
};

struct allocation_trace {
  std::array<std::size_t, 8> sizes{};
  std::size_t count{};
  std::size_t failures{1};
  cudaError_t terminal_error{cudaSuccess};

  static std::optional<cudaError_t> call(cucascade::test::pinned_host_call const& call,
                                         void* context) noexcept
  {
    auto& trace = *static_cast<allocation_trace*>(context);
    if (trace.count < trace.sizes.size()) { trace.sizes[trace.count] = call.bytes; }
    ++trace.count;
    if (trace.count <= trace.failures) { return cudaErrorMemoryAllocation; }
    if (trace.terminal_error != cudaSuccess) { return trace.terminal_error; }
    return std::nullopt;
  }
};

void require_recovery_pointer_type(void* pointer, cudaMemoryType type)
{
  cudaPointerAttributes attributes{};
  REQUIRE(cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess);
  REQUIRE(attributes.type == type);
}

}  // namespace

TEST_CASE("Rounded pinned OOM uses uncached exact storage without reclaiming retained buffers",
          "[small_pinned][pinned_recovery][fault_injection]")
{
  recovery_fixture fixture;
  auto* seed = fixture.resource.allocate_sync(9000);
  fixture.resource.deallocate_sync(seed, 9000);
  allocation_trace trace;
  void* pointer = nullptr;
  {
    cucascade::test::scoped_pinned_host_hook allocation{
      cucascade::test::pinned_host_operation::host_alloc, allocation_trace::call, &trace};
    pointer = fixture.resource.allocate_sync(40000);
  }
  REQUIRE(trace.count == 2);
  CHECK(trace.sizes[0] == 65536);
  CHECK(trace.sizes[1] == 40000);
  CHECK(fixture.resource.large_cache_bytes() == 16384);
  std::memset(pointer, 0x37, 40000);
  fixture.resource.deallocate_sync(pointer, 40000);
  require_recovery_pointer_type(pointer, cudaMemoryTypeUnregistered);
  REQUIRE(fixture.resource.large_cache_bytes() == 16384);
  auto const statistics = fixture.resource.get_statistics();
  REQUIRE(statistics.large_requests == 2);
  REQUIRE(statistics.large_misses == 2);
  REQUIRE(statistics.large_evictions == 0);
  auto* reused = fixture.resource.allocate_sync(9000);
  REQUIRE(reused == seed);
  fixture.resource.deallocate_sync(reused, 9000);
}

TEST_CASE("Exact pinned OOM reclaims only enough oldest retained capacity and retries once",
          "[small_pinned][pinned_recovery][fault_injection]")
{
  recovery_fixture fixture{128ULL << 10};
  std::array<void*, 4> seeds{fixture.resource.allocate_sync(20000),
                             fixture.resource.allocate_sync(20000),
                             fixture.resource.allocate_sync(20000),
                             fixture.resource.allocate_sync(9000)};
  for (std::size_t index = 0; index < seeds.size(); ++index) {
    fixture.resource.deallocate_sync(seeds[index], index == 3 ? 9000 : 20000);
  }
  allocation_trace trace;
  trace.failures = 2;
  void* pointer  = nullptr;
  {
    cucascade::test::scoped_pinned_host_hook allocation{
      cucascade::test::pinned_host_operation::host_alloc, allocation_trace::call, &trace};
    REQUIRE_NOTHROW(pointer = fixture.resource.allocate_sync(50000));
  }
  REQUIRE(pointer != nullptr);
  REQUIRE(trace.count == 3);
  REQUIRE(trace.sizes[0] == 65536);
  REQUIRE(trace.sizes[1] == 50000);
  REQUIRE(trace.sizes[2] == 50000);
  REQUIRE(fixture.resource.large_cache_bytes() == 49152);
  REQUIRE(fixture.resource.get_statistics().large_evictions == 2);
  fixture.resource.deallocate_sync(pointer, 50000);
  require_recovery_pointer_type(pointer, cudaMemoryTypeUnregistered);
  auto* retained = fixture.resource.allocate_sync(20000);
  REQUIRE(retained == seeds[2]);
  fixture.resource.deallocate_sync(retained, 20000);
}

TEST_CASE(
  "Insufficient retained capacity preserves cached storage without duplicate exact attempts",
  "[small_pinned][pinned_recovery][fault_injection]")
{
  auto const [bytes, limit, first_size, attempts] =
    GENERATE(Catch::Generators::table<std::size_t, std::size_t, std::size_t, std::size_t>(
      {{50000, 32768, 50000, 1}, {65536, 65536, 65536, 1}, {50000, 65536, 65536, 2}}));
  recovery_fixture fixture{limit};
  auto* seed = fixture.resource.allocate_sync(9000);
  fixture.resource.deallocate_sync(seed, 9000);
  allocation_trace trace;
  trace.failures = std::numeric_limits<std::size_t>::max();
  {
    cucascade::test::scoped_pinned_host_hook allocation{
      cucascade::test::pinned_host_operation::host_alloc, allocation_trace::call, &trace};
    REQUIRE_THROWS_AS(fixture.resource.allocate_sync(bytes), rmm::out_of_memory);
  }
  REQUIRE(trace.count == attempts);
  REQUIRE(trace.sizes[0] == first_size);
  if (attempts == 2) { REQUIRE(trace.sizes[1] == 50000); }
  REQUIRE(fixture.resource.large_cache_bytes() == 16384);
  REQUIRE(fixture.resource.get_statistics().large_evictions == 0);
  auto* reused = fixture.resource.allocate_sync(9000);
  REQUIRE(reused == seed);
  fixture.resource.deallocate_sync(reused, 9000);
}

TEST_CASE("Recovery reuses a matching buffer returned during the exact attempt",
          "[small_pinned][pinned_recovery][fault_injection]")
{
  recovery_fixture fixture{128ULL << 10};
  auto* held = fixture.resource.allocate_sync(40000);
  struct returned_buffer {
    allocation_trace trace;
    cucascade::memory::small_pinned_host_memory_resource* resource;
    void* pointer;
    bool worker_failed{};
  } returned{{}, &fixture.resource, held};
  returned.trace.failures = 2;
  auto publish            = [](cucascade::test::pinned_host_call const& call,
                    void* context) noexcept -> std::optional<cudaError_t> {
    auto& state = *static_cast<returned_buffer*>(context);
    auto status = allocation_trace::call(call, &state.trace);
    if (state.trace.count == 2) {
      try {
        std::jthread worker{[&] {
          state.resource->deallocate(
            ::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, state.pointer, 40000);
        }};
      } catch (...) {
        state.worker_failed = true;
      }
    }
    return status;
  };
  void* pointer = nullptr;
  {
    cucascade::test::scoped_pinned_host_hook allocation{
      cucascade::test::pinned_host_operation::host_alloc, publish, &returned};
    REQUIRE_NOTHROW(pointer = fixture.resource.allocate_sync(40000));
  }
  REQUIRE(pointer == held);
  REQUIRE_FALSE(returned.worker_failed);
  REQUIRE(returned.trace.count == 2);
  REQUIRE(fixture.resource.large_cache_bytes() == 0);
  auto const statistics = fixture.resource.get_statistics();
  REQUIRE(statistics.large_requests == 2);
  REQUIRE(statistics.large_hits == 1);
  REQUIRE(statistics.large_misses == 2);
  REQUIRE(statistics.large_evictions == 0);
  fixture.resource.deallocate_sync(pointer, 40000);
  REQUIRE(fixture.resource.large_cache_bytes() == 65536);
}

TEST_CASE("Exact recovery distinguishes retained capacity below equal and above the request",
          "[small_pinned][pinned_recovery][fault_injection]")
{
  auto const [seed_count, retained_bytes, attempts, succeeds] =
    GENERATE(Catch::Generators::table<std::size_t, std::size_t, std::size_t, bool>(
      {{1, 32768, 1, false}, {2, 0, 2, true}, {3, 32768, 2, true}}));
  recovery_fixture fixture{128ULL << 10};
  std::array<void*, 3> seeds{};
  for (std::size_t index = 0; index < seed_count; ++index) {
    seeds[index] = fixture.resource.allocate_sync(20000);
  }
  for (std::size_t index = 0; index < seed_count; ++index) {
    fixture.resource.deallocate_sync(seeds[index], 20000);
  }
  allocation_trace trace;
  void* pointer = nullptr;
  {
    cucascade::test::scoped_pinned_host_hook allocation{
      cucascade::test::pinned_host_operation::host_alloc, allocation_trace::call, &trace};
    if (succeeds) {
      REQUIRE_NOTHROW(pointer = fixture.resource.allocate_sync(65536));
    } else {
      REQUIRE_THROWS_AS(fixture.resource.allocate_sync(65536), rmm::out_of_memory);
    }
  }
  REQUIRE(trace.count == attempts);
  REQUIRE(trace.sizes[0] == 65536);
  if (succeeds) { REQUIRE(trace.sizes[1] == 65536); }
  REQUIRE(fixture.resource.large_cache_bytes() == retained_bytes);
  REQUIRE(fixture.resource.get_statistics().large_evictions == (succeeds ? 2 : 0));
  if (pointer != nullptr) {
    fixture.resource.deallocate_sync(pointer, 65536);
    require_recovery_pointer_type(pointer, cudaMemoryTypeUnregistered);
  }
}

TEST_CASE("Recovery does not purge on non-OOM failures and stops after its final exact attempt",
          "[small_pinned][pinned_recovery][fault_injection]")
{
  recovery_fixture fixture{128ULL << 10};
  std::array<void*, 3> seeds{fixture.resource.allocate_sync(20000),
                             fixture.resource.allocate_sync(20000),
                             fixture.resource.allocate_sync(20000)};
  for (auto* seed : seeds) {
    fixture.resource.deallocate_sync(seed, 20000);
  }
  auto const failures = GENERATE(
    std::size_t{0}, std::size_t{1}, std::size_t{2}, std::numeric_limits<std::size_t>::max());
  allocation_trace trace;
  trace.failures       = failures;
  trace.terminal_error = cudaErrorInvalidValue;
  {
    cucascade::test::scoped_pinned_host_hook allocation{
      cucascade::test::pinned_host_operation::host_alloc, allocation_trace::call, &trace};
    if (failures == std::numeric_limits<std::size_t>::max()) {
      REQUIRE_THROWS_AS(fixture.resource.allocate_sync(50000), rmm::out_of_memory);
    } else {
      REQUIRE_THROWS_AS(fixture.resource.allocate_sync(50000), rmm::bad_alloc);
    }
  }
  REQUIRE(trace.count == (failures < 2 ? failures + 1 : 3));
  REQUIRE(fixture.resource.large_cache_bytes() == (failures < 2 ? 98304 : 32768));
  REQUIRE(fixture.resource.get_statistics().large_evictions == (failures < 2 ? 0 : 2));
  REQUIRE(cudaGetLastError() == cudaSuccess);
}

TEST_CASE("Dedicated OOM reclamation uses the padded backing size",
          "[small_pinned][pinned_recovery][fault_injection]")
{
  bool const sufficient = GENERATE(false, true);
  recovery_fixture fixture;
  std::array<void*, 2> seeds{fixture.resource.allocate_sync(9000),
                             fixture.resource.allocate_sync(9000)};
  fixture.resource.deallocate_sync(seeds[0], 9000);
  if (sufficient) { fixture.resource.deallocate_sync(seeds[1], 9000); }
  allocation_trace trace;
  void* pointer = nullptr;
  {
    cucascade::test::scoped_pinned_host_hook allocation{
      cucascade::test::pinned_host_operation::host_alloc, allocation_trace::call, &trace};
    if (sufficient) {
      REQUIRE_NOTHROW(pointer = fixture.resource.allocate_sync(9000, 8192));
    } else {
      REQUIRE_THROWS_AS(fixture.resource.allocate_sync(9000, 8192), rmm::out_of_memory);
    }
  }
  REQUIRE(trace.count == (sufficient ? 2 : 1));
  REQUIRE(trace.sizes[0] == 17191);
  if (sufficient) {
    REQUIRE(trace.sizes[1] == 17191);
    REQUIRE(reinterpret_cast<std::uintptr_t>(pointer) % 8192 == 0);
    std::memset(pointer, 0x48, 9000);
    fixture.resource.deallocate_sync(pointer, 9000, 8192);
    require_recovery_pointer_type(pointer, cudaMemoryTypeUnregistered);
  } else {
    REQUIRE(fixture.resource.large_cache_bytes() == 16384);
    fixture.resource.deallocate_sync(seeds[1], 9000);
  }
}

TEST_CASE("Fallback metadata exhaustion preserves cached capacity before recovery",
          "[small_pinned][pinned_recovery][fault_injection]")
{
  recovery_fixture fixture{128ULL << 10};
  std::array<void*, 2> seeds{fixture.resource.allocate_sync(20000),
                             fixture.resource.allocate_sync(20000)};
  for (auto* seed : seeds) {
    fixture.resource.deallocate_sync(seed, 20000);
  }
  allocation_trace trace;
  {
    cucascade::test::scoped_pinned_host_hook allocation{
      cucascade::test::pinned_host_operation::host_alloc, allocation_trace::call, &trace};
    cucascade::test::scoped_pinned_host_hook metadata{
      cucascade::test::pinned_host_operation::exceptional_allocation, cudaErrorMemoryAllocation};
    REQUIRE_THROWS_AS(fixture.resource.allocate_sync(65536), std::bad_alloc);
  }
  REQUIRE(trace.count == 1);
  REQUIRE(fixture.resource.large_cache_bytes() == 65536);
  REQUIRE(fixture.resource.get_statistics().large_evictions == 0);
  auto* reused = fixture.resource.allocate_sync(20000);
  REQUIRE(reused == seeds[0]);
  fixture.resource.deallocate_sync(reused, 20000);
}

TEST_CASE("Unconfirmed reclamation withholds its victim without extending the retry budget",
          "[small_pinned][pinned_recovery][fault_injection]")
{
  recovery_fixture fixture{128ULL << 10};
  std::array<void*, 3> seeds{fixture.resource.allocate_sync(20000),
                             fixture.resource.allocate_sync(20000),
                             fixture.resource.allocate_sync(20000)};
  for (auto* seed : seeds) {
    fixture.resource.deallocate_sync(seed, 20000);
  }
  allocation_trace trace;
  trace.failures = 2;
  void* pointer  = nullptr;
  {
    cucascade::test::scoped_pinned_host_hook allocation{
      cucascade::test::pinned_host_operation::host_alloc, allocation_trace::call, &trace};
    cucascade::test::scoped_pinned_host_hook completion{
      cucascade::test::pinned_host_operation::event_synchronize, cudaErrorInvalidResourceHandle};
    REQUIRE_NOTHROW(pointer = fixture.resource.allocate_sync(50000));
  }
  REQUIRE(trace.count == 3);
  REQUIRE(fixture.resource.large_cache_bytes() == 32768);
  require_recovery_pointer_type(seeds[0], cudaMemoryTypeHost);
  REQUIRE(cudaGetLastError() == cudaSuccess);
  fixture.resource.deallocate_sync(pointer, 50000);
  REQUIRE(cudaFreeHost(seeds[0]) == cudaSuccess);
}

TEST_CASE("Uncacheable exact recovery stays untracked and releases its original storage",
          "[small_pinned][pinned_recovery][fault_injection]")
{
  recovery_fixture fixture{96ULL << 10};
  std::array<void*, 3> seeds{fixture.resource.allocate_sync(20000),
                             fixture.resource.allocate_sync(20000),
                             fixture.resource.allocate_sync(20000)};
  for (auto* seed : seeds) {
    fixture.resource.deallocate_sync(seed, 20000);
  }
  allocation_trace trace;
  void* pointer = nullptr;
  {
    cucascade::test::scoped_pinned_host_hook allocation{
      cucascade::test::pinned_host_operation::host_alloc, allocation_trace::call, &trace};
    cucascade::test::scoped_pinned_host_hook metadata{
      cucascade::test::pinned_host_operation::exceptional_allocation, cudaErrorMemoryAllocation};
    REQUIRE_NOTHROW(pointer = fixture.resource.allocate_sync(70000));
  }
  REQUIRE(trace.count == 2);
  REQUIRE(trace.sizes[0] == 70000);
  REQUIRE(trace.sizes[1] == 70000);
  REQUIRE(fixture.resource.large_cache_bytes() == 0);
  auto const statistics = fixture.resource.get_statistics();
  REQUIRE(statistics.large_requests == 4);
  REQUIRE(statistics.large_uncacheable == 1);
  REQUIRE(statistics.large_evictions == 3);
  fixture.resource.deallocate_sync(pointer, 70000);
  require_recovery_pointer_type(pointer, cudaMemoryTypeUnregistered);
}
