/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pinned_host_test_hooks.hpp"

#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <catch2/catch_all.hpp>

namespace {

struct insertion_fixture {
  rmm::mr::pinned_host_memory_resource pinned;
  cucascade::memory::fixed_size_host_memory_resource upstream{
    0, pinned, 1ULL << 20, 1ULL << 20, 64ULL << 10, 4, 1};
  cucascade::memory::small_pinned_host_memory_resource resource{upstream, 32ULL << 10};
};

void require_pointer_type(void* pointer, cudaMemoryType type)
{
  cudaPointerAttributes attributes{};
  REQUIRE(cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess);
  REQUIRE(attributes.type == type);
}

}  // namespace

TEST_CASE("Failed cache insertion preserves retained large-buffer capacity",
          "[small_pinned][fault_injection][cache_insertion]")
{
  insertion_fixture fixture;
  rmm::cuda_stream stream;
  constexpr std::size_t small_bytes = 9000;
  constexpr std::size_t large_bytes = 20000;
  auto* oldest                      = fixture.resource.allocate_sync(small_bytes);
  auto* next                        = fixture.resource.allocate_sync(small_bytes);
  auto* incoming                    = fixture.resource.allocate_sync(large_bytes);
  fixture.resource.deallocate(stream, oldest, small_bytes);
  fixture.resource.deallocate(stream, next, small_bytes);
  REQUIRE(fixture.resource.large_cache_bytes() == 32768);

  bool const reject_completion = GENERATE(false, true);
  {
    cucascade::test::scoped_pinned_host_hook metadata_failure{
      cucascade::test::pinned_host_operation::large_cache_entry, cudaErrorMemoryAllocation};
    if (reject_completion) {
      cucascade::test::scoped_pinned_host_hook completion_failure{
        cucascade::test::pinned_host_operation::event_synchronize, cudaErrorInvalidResourceHandle};
      fixture.resource.deallocate(stream, incoming, large_bytes);
    } else {
      fixture.resource.deallocate(stream, incoming, large_bytes);
    }
  }

  REQUIRE(fixture.resource.large_cache_bytes() == 32768);
  REQUIRE(fixture.resource.get_statistics().large_evictions == 0);
  require_pointer_type(oldest, cudaMemoryTypeHost);
  require_pointer_type(next, cudaMemoryTypeHost);
  require_pointer_type(incoming,
                       reject_completion ? cudaMemoryTypeHost : cudaMemoryTypeUnregistered);
  REQUIRE(cudaGetLastError() == cudaSuccess);

  if (reject_completion) {
    stream.synchronize();
    REQUIRE(cudaFreeHost(incoming) == cudaSuccess);
  }

  auto* reused_oldest = fixture.resource.allocate_sync(small_bytes);
  auto* reused_next   = fixture.resource.allocate_sync(small_bytes);
  REQUIRE(reused_oldest == oldest);
  REQUIRE(reused_next == next);
  fixture.resource.deallocate_sync(reused_oldest, small_bytes);
  fixture.resource.deallocate_sync(reused_next, small_bytes);
}

TEST_CASE("Successful cache insertion releases every required completed victim",
          "[small_pinned][fault_injection][cache_insertion]")
{
  insertion_fixture fixture;
  rmm::cuda_stream stream;
  constexpr std::size_t small_bytes = 9000;
  constexpr std::size_t large_bytes = 20000;
  auto* oldest                      = fixture.resource.allocate_sync(small_bytes);
  auto* next                        = fixture.resource.allocate_sync(small_bytes);
  auto* incoming                    = fixture.resource.allocate_sync(large_bytes);
  fixture.resource.deallocate(stream, oldest, small_bytes);
  fixture.resource.deallocate(stream, next, small_bytes);
  REQUIRE(fixture.resource.large_cache_bytes() == 32768);

  fixture.resource.deallocate(stream, incoming, large_bytes);
  REQUIRE(fixture.resource.large_cache_bytes() == 32768);
  REQUIRE(fixture.resource.get_statistics().large_evictions == 2);
  require_pointer_type(oldest, cudaMemoryTypeUnregistered);
  require_pointer_type(next, cudaMemoryTypeUnregistered);
  require_pointer_type(incoming, cudaMemoryTypeHost);

  auto* reused = fixture.resource.allocate_sync(large_bytes);
  REQUIRE(reused == incoming);
  fixture.resource.deallocate_sync(reused, large_bytes);
}
