/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fault_injection_hooks.hpp"
#include "pinned_host_test_utils.hpp"

#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch2/catch_all.hpp>

#include <exception>
#include <thread>

namespace {

using cucascade::test::pinned_host_test_fixture;
using cucascade::test::pinned_upstream_fixture;
using cucascade::test::require_pointer_type;

}  // namespace

TEST_CASE("Uncached pinned storage is released only after confirmed completion",
          "[small_pinned][fault_injection]")
{
  pinned_host_test_fixture fixture{0};
  rmm::cuda_stream stream;
  constexpr std::size_t bytes = 100000;
  void* pointer               = fixture.resource.allocate_sync(bytes);
  require_pointer_type(pointer, cudaMemoryTypeHost);

  SECTION("completed work releases its storage")
  {
    fixture.resource.deallocate(stream, pointer, bytes);
    require_pointer_type(pointer, cudaMemoryTypeUnregistered);
  }

  SECTION("unconfirmed completion withholds its storage")
  {
    {
      cucascade::test::scoped_fault_hook failure{cucascade::test::fault_point::stream_synchronize,
                                                 cudaErrorInvalidResourceHandle};
      fixture.resource.deallocate(stream, pointer, bytes);
    }
    require_pointer_type(pointer, cudaMemoryTypeHost);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    stream.synchronize();
    REQUIRE(cudaFreeHost(pointer) == cudaSuccess);

    auto* next = fixture.resource.allocate_sync(bytes);
    fixture.resource.deallocate(stream, next, bytes);
    require_pointer_type(next, cudaMemoryTypeUnregistered);
  }
}

TEST_CASE("Pinned allocation failures are scoped to one call and one thread",
          "[small_pinned][fault_injection]")
{
  pinned_host_test_fixture fixture{0};
  constexpr std::size_t bytes = 100000;

  SECTION("one failed call does not reject subsequent allocations")
  {
    cucascade::test::scoped_fault_hook failure{cucascade::test::fault_point::host_alloc,
                                               cudaErrorInvalidValue};
    REQUIRE_THROWS_AS(fixture.resource.allocate_sync(bytes), rmm::bad_alloc);
    auto* pointer = fixture.resource.allocate_sync(bytes);
    fixture.resource.deallocate_sync(pointer, bytes);
    require_pointer_type(pointer, cudaMemoryTypeUnregistered);
  }

  SECTION("a hook does not reject another thread's allocation")
  {
    void* pointer = nullptr;
    std::exception_ptr error;
    {
      cucascade::test::scoped_fault_hook failure{cucascade::test::fault_point::host_alloc,
                                                 cudaErrorInvalidValue};
      std::jthread worker{[&] {
        try {
          pointer = fixture.resource.allocate_sync(bytes);
        } catch (...) {
          error = std::current_exception();
        }
      }};
      worker.join();
      REQUIRE(error == nullptr);
      REQUIRE(pointer != nullptr);
      REQUIRE_THROWS_AS(fixture.resource.allocate_sync(bytes), rmm::bad_alloc);
    }
    fixture.resource.deallocate_sync(pointer, bytes);
    require_pointer_type(pointer, cudaMemoryTypeUnregistered);
  }
}

TEST_CASE("Cached pinned storage is released only after its event confirms completion",
          "[small_pinned][fault_injection]")
{
  pinned_upstream_fixture fixture;
  std::optional<cucascade::memory::small_pinned_host_memory_resource> resource{
    std::in_place, fixture.upstream, 64ULL << 10};
  constexpr std::size_t bytes = 9000;
  auto* pointer               = resource->allocate_sync(bytes);
  resource->deallocate_sync(pointer, bytes);

  SECTION("completed cached work releases its storage")
  {
    resource.reset();
    require_pointer_type(pointer, cudaMemoryTypeUnregistered);
  }

  SECTION("unconfirmed cached work withholds its storage")
  {
    {
      cucascade::test::scoped_fault_hook failure{cucascade::test::fault_point::event_synchronize,
                                                 cudaErrorInvalidResourceHandle};
      resource.reset();
    }
    require_pointer_type(pointer, cudaMemoryTypeHost);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaFreeHost(pointer) == cudaSuccess);
  }
}

TEST_CASE("Cache-entry metadata exhaustion releases incoming completed storage",
          "[small_pinned][fault_injection]")
{
  pinned_host_test_fixture fixture;
  auto& resource              = fixture.resource;
  constexpr std::size_t bytes = 9000;
  auto* pointer               = resource.allocate_sync(bytes);
  {
    cucascade::test::scoped_fault_hook failure{cucascade::test::fault_point::large_cache_entry,
                                               cudaErrorMemoryAllocation};
    resource.deallocate_sync(pointer, bytes);
    REQUIRE(resource.large_cache_bytes() == 0);
    require_pointer_type(pointer, cudaMemoryTypeUnregistered);

    auto* next = resource.allocate_sync(bytes);
    resource.deallocate_sync(next, bytes);
    REQUIRE(resource.large_cache_bytes() == 16384);
    require_pointer_type(next, cudaMemoryTypeHost);
  }
}

int main(int argc, char** argv) { return Catch::Session().run(argc, argv); }
