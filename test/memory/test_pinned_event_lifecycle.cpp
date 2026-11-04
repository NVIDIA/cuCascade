/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fault_injection_hooks.hpp"
#include "pinned_host_test_utils.hpp"

#include <cucascade/cuda/event.hpp>
#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch2/catch_all.hpp>

#include <optional>
#include <utility>

namespace {

using cucascade::test::pinned_upstream_fixture;
using cucascade::test::require_pointer_type;

struct destruction_failure {
  int attempts{};
  cucascade::cuda::cuda_event* owner{};
  bool owner_was_empty{true};
};

std::optional<cudaError_t> destroy_then_fail(cucascade::test::fault_point_call const& call,
                                             void* context) noexcept
{
  auto& failure = *static_cast<destruction_failure*>(context);
  ++failure.attempts;
  if (failure.owner != nullptr && failure.owner->get() != nullptr) {
    failure.owner_was_empty = false;
  }
  auto const result = cudaEventDestroy(call.event);
  if (result != cudaSuccess) { return result; }
  return cudaSetDevice(-1);
}

std::optional<cudaError_t> count_destruction(cucascade::test::fault_point_call const&,
                                             void* context) noexcept
{
  ++*static_cast<int*>(context);
  return std::nullopt;
}

}  // namespace

TEST_CASE("Event destruction clears its own CUDA failure",
          "[small_pinned][fault_injection][event_lifecycle]")
{
  destruction_failure failure;
  cucascade::test::scoped_fault_hook hook{
    cucascade::test::fault_point::event_destroy, destroy_then_fail, &failure};
  {
    cucascade::cuda::cuda_event event;
    REQUIRE(event.get() != nullptr);
  }
  REQUIRE(cudaGetLastError() == cudaSuccess);
  REQUIRE(failure.attempts == 1);
}

TEST_CASE("Event reset relinquishes ownership even when destruction fails",
          "[small_pinned][fault_injection][event_lifecycle]")
{
  destruction_failure failure;
  cucascade::test::scoped_fault_hook hook{
    cucascade::test::fault_point::event_destroy, destroy_then_fail, &failure};
  {
    cucascade::cuda::cuda_event event;
    failure.owner = &event;
    REQUIRE(event.reset_no_throw() == cudaErrorInvalidDevice);
    REQUIRE(event.get() == nullptr);
    REQUIRE(failure.owner_was_empty);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(event.reset_no_throw() == cudaSuccess);
  }
  REQUIRE(failure.attempts == 1);
}

TEST_CASE("An empty event reset preserves unrelated CUDA error state",
          "[small_pinned][fault_injection][event_lifecycle]")
{
  int destructions = 0;
  cucascade::test::scoped_fault_hook observed{
    cucascade::test::fault_point::event_destroy, count_destruction, &destructions};
  {
    cucascade::cuda::cuda_event event;
    REQUIRE(event.reset_no_throw() == cudaSuccess);
    REQUIRE(event.get() == nullptr);
    REQUIRE(cudaSetDevice(-1) == cudaErrorInvalidDevice);
    REQUIRE(event.reset_no_throw() == cudaSuccess);
    REQUIRE(cudaGetLastError() == cudaErrorInvalidDevice);
  }
  REQUIRE(destructions == 1);
}

TEST_CASE("Event move assignment safely replaces its previous ownership",
          "[small_pinned][fault_injection][event_lifecycle]")
{
  cucascade::cuda::cuda_event source;
  cucascade::cuda::cuda_event destination;
  auto const source_handle = source.get();
  destruction_failure failure{.owner = &destination};
  {
    cucascade::test::scoped_fault_hook hook{
      cucascade::test::fault_point::event_destroy, destroy_then_fail, &failure};
    destination = std::move(source);
    REQUIRE(source.get() == nullptr);
    REQUIRE(destination.get() == source_handle);
    REQUIRE(failure.owner_was_empty);
    REQUIRE(failure.attempts == 1);
    REQUIRE(cudaGetLastError() == cudaSuccess);
  }
  REQUIRE(destination.reset_no_throw() == cudaSuccess);
}

TEST_CASE("Eviction retires only events whose completion cannot be confirmed",
          "[small_pinned][fault_injection][event_lifecycle]")
{
  pinned_upstream_fixture fixture;
  cucascade::memory::small_pinned_host_memory_resource resource{fixture.upstream, 16384};
  rmm::cuda_stream stream;
  constexpr std::size_t bytes = 9000;
  auto* victim                = resource.allocate_sync(bytes);
  auto* incoming              = resource.allocate_sync(bytes);
  resource.deallocate(stream, victim, bytes);
  int destructions = 0;
  bool const fail  = GENERATE(false, true);
  {
    cucascade::test::scoped_fault_hook observed{
      cucascade::test::fault_point::event_destroy, count_destruction, &destructions};
    std::optional<cucascade::test::scoped_fault_hook> failed;
    if (fail) {
      failed.emplace(cucascade::test::fault_point::event_synchronize,
                     cudaErrorInvalidResourceHandle);
    }
    resource.deallocate(stream, incoming, bytes);
  }
  require_pointer_type(victim, fail ? cudaMemoryTypeHost : cudaMemoryTypeUnregistered);
  REQUIRE(cudaGetLastError() == cudaSuccess);
  if (fail) {
    stream.synchronize();
    REQUIRE(cudaFreeHost(victim) == cudaSuccess);
  }
  REQUIRE(destructions == (fail ? 1 : 0));
  REQUIRE(resource.large_cache_bytes() == 16384);
}

TEST_CASE("Unpublished buffers retire events after unconfirmed completion",
          "[small_pinned][fault_injection][event_lifecycle]")
{
  pinned_upstream_fixture fixture;
  cucascade::memory::small_pinned_host_memory_resource resource{fixture.upstream, 16384};
  rmm::cuda_stream stream;
  constexpr std::size_t bytes = 9000;
  auto* pointer               = resource.allocate_sync(bytes);
  int destructions            = 0;
  bool const fail             = GENERATE(false, true);
  {
    cucascade::test::scoped_fault_hook observed{
      cucascade::test::fault_point::event_destroy, count_destruction, &destructions};
    cucascade::test::scoped_fault_hook metadata_failure{
      cucascade::test::fault_point::large_cache_entry, cudaErrorMemoryAllocation};
    std::optional<cucascade::test::scoped_fault_hook> failed;
    if (fail) {
      failed.emplace(cucascade::test::fault_point::event_synchronize,
                     cudaErrorInvalidResourceHandle);
    }
    resource.deallocate(stream, pointer, bytes);
  }
  require_pointer_type(pointer, fail ? cudaMemoryTypeHost : cudaMemoryTypeUnregistered);
  REQUIRE(cudaGetLastError() == cudaSuccess);
  if (fail) {
    stream.synchronize();
    REQUIRE(cudaFreeHost(pointer) == cudaSuccess);
  }
  REQUIRE(destructions == (fail ? 1 : 0));
  REQUIRE(resource.large_cache_bytes() == 0);
}

TEST_CASE("Repeated event retirement reuses reserved return capacity",
          "[small_pinned][fault_injection][event_lifecycle]")
{
  pinned_upstream_fixture fixture;
  cucascade::memory::small_pinned_host_memory_resource resource{fixture.upstream, 16384};
  rmm::cuda_stream stream;
  constexpr std::size_t bytes = 9000;
  int destructions            = 0;
  for (int iteration = 0; iteration != 8; ++iteration) {
    auto* pointer = resource.allocate_sync(bytes);
    {
      cucascade::test::scoped_fault_hook observed{
        cucascade::test::fault_point::event_destroy, count_destruction, &destructions};
      cucascade::test::scoped_fault_hook metadata_failure{
        cucascade::test::fault_point::large_cache_entry, cudaErrorMemoryAllocation};
      cucascade::test::scoped_fault_hook completion_failure{
        cucascade::test::fault_point::event_synchronize, cudaErrorInvalidResourceHandle};
      std::optional<cucascade::test::scoped_fault_hook> growth_failure;
      if (iteration != 0) {
        growth_failure.emplace(cucascade::test::fault_point::event_pool_reserve,
                               cudaErrorMemoryAllocation);
      }
      resource.deallocate(stream, pointer, bytes);
    }
    require_pointer_type(pointer, cudaMemoryTypeHost);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    stream.synchronize();
    REQUIRE(cudaFreeHost(pointer) == cudaSuccess);
  }
  REQUIRE(destructions == 8);

  auto* healthy = resource.allocate_sync(bytes);
  resource.deallocate(stream, healthy, bytes);
  REQUIRE(resource.large_cache_bytes() == 16384);
}
