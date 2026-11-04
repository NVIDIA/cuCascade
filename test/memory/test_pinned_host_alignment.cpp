/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fault_injection_hooks.hpp"
#include "pinned_host_test_utils.hpp"

#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <thread>

namespace {

using cucascade::test::pinned_host_test_fixture;
using cucascade::test::require_pointer_type;

std::optional<cudaError_t> count_allocations(cucascade::test::fault_point_call const&,
                                             void* context) noexcept
{
  ++*static_cast<std::size_t*>(context);
  return std::nullopt;
}

struct dedicated_allocation_observation {
  void* base{};
  void* freed{};
  std::size_t bytes{};

  static std::optional<cudaError_t> allocate(cucascade::test::fault_point_call const& call,
                                             void* context) noexcept
  {
    auto& observation = *static_cast<dedicated_allocation_observation*>(context);
    observation.bytes = call.bytes;
    auto const status =
      cudaHostAlloc(&observation.base, call.bytes, cudaHostAllocPortable | cudaHostAllocMapped);
    if (status == cudaSuccess) { *call.buffer_out = observation.base; }
    return status;
  }

  static std::optional<cudaError_t> release(cucascade::test::fault_point_call const& call,
                                            void* context) noexcept
  {
    static_cast<dedicated_allocation_observation*>(context)->freed = call.buffer;
    return std::nullopt;
  }
};

}  // namespace

TEST_CASE("Nonzero pinned requests reject invalid alignment",
          "[small_pinned][pinned_alignment][fault_injection]")
{
  pinned_host_test_fixture fixture;
  auto const alignment = GENERATE(std::size_t{0}, std::size_t{3}, std::size_t{24});
  void* pointer        = nullptr;
  CHECK_THROWS_AS(pointer = fixture.resource.allocate_sync(128, alignment), std::invalid_argument);
  if (pointer != nullptr) { fixture.resource.deallocate_sync(pointer, 128, alignment); }
}

TEST_CASE("Alignment padding overflow fails before acquiring pinned storage",
          "[small_pinned][pinned_alignment][fault_injection]")
{
  pinned_host_test_fixture fixture;
  std::size_t allocations = 0;
  cucascade::test::scoped_fault_hook observe{
    cucascade::test::fault_point::host_alloc, count_allocations, &allocations};
  REQUIRE_THROWS_AS(fixture.resource.allocate_sync(std::numeric_limits<std::size_t>::max(), 64),
                    rmm::bad_alloc);
  REQUIRE(allocations == 0);
}

TEST_CASE("Dedicated metadata exhaustion acquires no pinned storage",
          "[small_pinned][pinned_alignment][fault_injection]")
{
  pinned_host_test_fixture fixture;
  std::size_t allocations = 0;
  cucascade::test::scoped_fault_hook observe{
    cucascade::test::fault_point::host_alloc, count_allocations, &allocations};
  void* pointer = nullptr;
  {
    cucascade::test::scoped_fault_hook failure{cucascade::test::fault_point::dedicated_allocation,
                                               cudaErrorMemoryAllocation};
    CHECK_THROWS_AS(pointer = fixture.resource.allocate_sync(9000, 64), std::bad_alloc);
  }
  if (pointer != nullptr) { fixture.resource.deallocate_sync(pointer, 9000, 64); }
  REQUIRE(allocations == 0);

  pointer = fixture.resource.allocate_sync(9000, 64);
  fixture.resource.deallocate_sync(pointer, 9000, 64);
  require_pointer_type(pointer, cudaMemoryTypeUnregistered);
}

TEST_CASE("Dedicated storage uses alignment padding and frees its original allocation",
          "[small_pinned][pinned_alignment][fault_injection]")
{
  pinned_host_test_fixture fixture;
  auto const [alignment, allocation_bytes] =
    GENERATE(Catch::Generators::table<std::size_t, std::size_t>(
      {{64, 9063}, {4096, 13095}, {1048576, 1057575}}));
  dedicated_allocation_observation observation;
  cucascade::test::scoped_fault_hook allocate{cucascade::test::fault_point::host_alloc,
                                              dedicated_allocation_observation::allocate,
                                              &observation};
  cucascade::test::scoped_fault_hook release{cucascade::test::fault_point::host_free,
                                             dedicated_allocation_observation::release,
                                             &observation};

  auto* pointer = fixture.resource.allocate_sync(9000, alignment);
  REQUIRE(reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0);
  REQUIRE(observation.bytes == allocation_bytes);
  require_pointer_type(pointer, cudaMemoryTypeHost);
  std::memset(pointer, 0x42, 9000);
  fixture.resource.deallocate_sync(pointer, 9000, alignment);
  REQUIRE(fixture.resource.large_cache_bytes() == 0);
  REQUIRE(observation.freed == observation.base);
  require_pointer_type(observation.base, cudaMemoryTypeUnregistered);
}

TEST_CASE("Over-aligned small storage bypasses an awkward upstream block stride",
          "[small_pinned][pinned_alignment][fault_injection]")
{
  rmm::mr::pinned_host_memory_resource pinned;
  cucascade::memory::fixed_size_host_memory_resource upstream{
    0, pinned, 1ULL << 20, 1ULL << 20, 8208, 8, 1};
  auto const initial_blocks = upstream.get_free_blocks();
  {
    cucascade::memory::small_pinned_host_memory_resource resource{upstream};
    for (std::size_t const bytes : std::array<std::size_t, 3>{1, 513, 8192}) {
      auto* pointer = resource.allocate_sync(bytes, 64);
      REQUIRE(reinterpret_cast<std::uintptr_t>(pointer) % 64 == 0);
      std::memset(pointer, 0x53, bytes);
      resource.deallocate_sync(pointer, bytes, 64);
      require_pointer_type(pointer, cudaMemoryTypeUnregistered);
      REQUIRE(upstream.get_free_blocks() == initial_blocks);
    }

    auto* ordinary = resource.allocate_sync(513);
    REQUIRE(reinterpret_cast<std::uintptr_t>(ordinary) % alignof(std::max_align_t) == 0);
    REQUIRE(upstream.get_free_blocks() < initial_blocks);
    resource.deallocate_sync(ordinary, 513);
    require_pointer_type(ordinary, cudaMemoryTypeHost);
  }
  REQUIRE(upstream.get_free_blocks() == initial_blocks);
}

TEST_CASE("Ordinary pooled storage does not require dedicated ownership metadata",
          "[small_pinned][pinned_alignment][fault_injection]")
{
  pinned_host_test_fixture fixture;
  cucascade::test::scoped_fault_hook failure{cucascade::test::fault_point::dedicated_allocation,
                                             cudaErrorMemoryAllocation};
  auto* small = fixture.resource.allocate_sync(513);
  fixture.resource.deallocate_sync(small, 513);
  auto* large = fixture.resource.allocate_sync(9000);
  fixture.resource.deallocate_sync(large, 9000);
  REQUIRE(fixture.resource.large_cache_bytes() == 16384);
  auto* reused = fixture.resource.allocate_sync(9000);
  REQUIRE(reused == large);
  fixture.resource.deallocate_sync(reused, 9000);
  auto const statistics = fixture.resource.get_statistics();
  REQUIRE(statistics.large_requests == 2);
  REQUIRE(statistics.large_hits == 1);
  REQUIRE(statistics.large_misses == 1);
  REQUIRE(statistics.large_uncacheable == 0);
}

TEST_CASE("Dedicated storage release withholds unconfirmed backing and recovers cleanly",
          "[small_pinned][pinned_alignment][fault_injection]")
{
  enum class release_failure { none, wait, free };
  auto const failure =
    GENERATE(release_failure::none, release_failure::wait, release_failure::free);
  auto const bytes = GENERATE(std::size_t{513}, std::size_t{9000});
  pinned_host_test_fixture fixture;
  rmm::cuda_stream stream;
  dedicated_allocation_observation observation;
  cucascade::test::scoped_fault_hook allocate{cucascade::test::fault_point::host_alloc,
                                              dedicated_allocation_observation::allocate,
                                              &observation};
  auto* pointer = fixture.resource.allocate_sync(bytes, 64);
  {
    std::optional<cucascade::test::scoped_fault_hook> rejected;
    if (failure != release_failure::none) {
      rejected.emplace(failure == release_failure::wait
                         ? cucascade::test::fault_point::stream_synchronize
                         : cucascade::test::fault_point::host_free,
                       cudaErrorInvalidResourceHandle);
    }
    fixture.resource.deallocate(stream, pointer, bytes, 64);
  }
  REQUIRE(cudaGetLastError() == cudaSuccess);
  require_pointer_type(
    observation.base,
    failure == release_failure::none ? cudaMemoryTypeUnregistered : cudaMemoryTypeHost);
  REQUIRE(fixture.resource.large_cache_bytes() == 0);
  if (failure != release_failure::none) {
    stream.synchronize();
    REQUIRE(cudaFreeHost(observation.base) == cudaSuccess);
  }

  pointer = fixture.resource.allocate_sync(bytes, 64);
  fixture.resource.deallocate_sync(pointer, bytes, 64);
  require_pointer_type(observation.base, cudaMemoryTypeUnregistered);
}

TEST_CASE("Concurrent dedicated allocations preserve alignment and release ownership",
          "[small_pinned][pinned_alignment][fault_injection]")
{
  pinned_host_test_fixture fixture;
  constexpr std::size_t num_threads = 4;
  constexpr std::size_t iterations  = 12;
  std::array<std::array<void*, iterations>, num_threads> pointers{};
  std::array<std::exception_ptr, num_threads> errors{};
  {
    std::array<std::jthread, num_threads> workers;
    for (std::size_t thread = 0; thread < num_threads; ++thread) {
      workers[thread] = std::jthread{[&, thread] {
        try {
          for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            auto* pointer               = fixture.resource.allocate_sync(513, 64);
            pointers[thread][iteration] = pointer;
            std::memset(pointer, 0x64, 513);
            fixture.resource.deallocate_sync(pointer, 513, 64);
          }
        } catch (...) {
          errors[thread] = std::current_exception();
        }
      }};
    }
  }
  for (auto const& error : errors) {
    REQUIRE(error == nullptr);
  }
  for (auto const& row : pointers) {
    for (auto* pointer : row) {
      REQUIRE(pointer != nullptr);
      REQUIRE(reinterpret_cast<std::uintptr_t>(pointer) % 64 == 0);
      require_pointer_type(pointer, cudaMemoryTypeUnregistered);
    }
  }
  REQUIRE(fixture.resource.large_cache_bytes() == 0);
}
