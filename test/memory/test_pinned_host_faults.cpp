/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pinned_host_test_hooks.hpp"

#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <catch2/catch_all.hpp>

#include <exception>
#include <thread>

namespace {

struct pinned_fixture {
  rmm::mr::pinned_host_memory_resource pinned;
  cucascade::memory::fixed_size_host_memory_resource upstream{
    0, pinned, 1ULL << 20, 1ULL << 20, 64ULL << 10, 4, 1};
  cucascade::memory::small_pinned_host_memory_resource resource{upstream, 0};
};

void require_pointer_type(void* pointer, cudaMemoryType type)
{
  cudaPointerAttributes attributes{};
  REQUIRE(cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess);
  REQUIRE(attributes.type == type);
}

}  // namespace

TEST_CASE("Uncached pinned storage is released only after confirmed completion",
          "[small_pinned][fault_injection]")
{
  pinned_fixture fixture;
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
      cucascade::test::scoped_pinned_host_hook failure{
        cucascade::test::pinned_host_operation::stream_synchronize, cudaErrorInvalidResourceHandle};
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
  pinned_fixture fixture;
  constexpr std::size_t bytes = 100000;

  SECTION("one failed call does not reject subsequent allocations")
  {
    cucascade::test::scoped_pinned_host_hook failure{
      cucascade::test::pinned_host_operation::host_alloc, cudaErrorInvalidValue};
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
      cucascade::test::scoped_pinned_host_hook failure{
        cucascade::test::pinned_host_operation::host_alloc, cudaErrorInvalidValue};
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
  pinned_fixture fixture;
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
      cucascade::test::scoped_pinned_host_hook failure{
        cucascade::test::pinned_host_operation::event_synchronize, cudaErrorInvalidResourceHandle};
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
  pinned_fixture fixture;
  cucascade::memory::small_pinned_host_memory_resource resource{fixture.upstream, 64ULL << 10};
  constexpr std::size_t bytes = 9000;
  auto* pointer               = resource.allocate_sync(bytes);
  {
    cucascade::test::scoped_pinned_host_hook failure{
      cucascade::test::pinned_host_operation::large_cache_entry, cudaErrorMemoryAllocation};
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
