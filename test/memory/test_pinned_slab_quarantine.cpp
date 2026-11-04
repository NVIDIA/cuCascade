/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fault_injection_hooks.hpp"
#include "pinned_host_test_utils.hpp"

#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch2/catch_all.hpp>

#include <optional>

namespace {

using cucascade::test::pinned_upstream_fixture;

}  // namespace

TEST_CASE("Slab destruction retains only backing with unconfirmed completion",
          "[small_pinned][fault_injection][slab_quarantine]")
{
  pinned_upstream_fixture fixture;
  auto const initial_free = fixture.upstream.get_free_blocks();
  bool const fail         = GENERATE(false, true);
  std::optional<cucascade::memory::small_pinned_host_memory_resource> resource{std::in_place,
                                                                               fixture.upstream};
  auto* uncertain = resource->allocate_sync(512);
  auto* completed = resource->allocate_sync(8192);
  REQUIRE(fixture.upstream.get_free_blocks() == initial_free - 2);
  resource->deallocate_sync(uncertain, 512);
  resource->deallocate_sync(completed, 8192);

  {
    std::optional<cucascade::test::scoped_fault_hook> failure;
    if (fail) {
      failure.emplace(cucascade::test::fault_point::event_synchronize,
                      cudaErrorInvalidResourceHandle);
    }
    resource.reset();
  }

  REQUIRE(cudaGetLastError() == cudaSuccess);
  REQUIRE(fixture.upstream.get_free_blocks() == initial_free - (fail ? 1 : 0));
}

TEST_CASE("An unconfirmed slab deallocation retains only its backing owner",
          "[small_pinned][fault_injection][slab_quarantine]")
{
  pinned_upstream_fixture fixture;
  rmm::cuda_stream stream;
  auto const initial_free = fixture.upstream.get_free_blocks();
  bool const fail         = GENERATE(false, true);
  std::optional<cucascade::memory::small_pinned_host_memory_resource> resource{std::in_place,
                                                                               fixture.upstream};
  auto* uncertain = resource->allocate_sync(512);
  auto* completed = resource->allocate_sync(8192);
  REQUIRE(fixture.upstream.get_free_blocks() == initial_free - 2);
  resource->deallocate_sync(completed, 8192);

  {
    std::optional<cucascade::test::scoped_fault_hook> record_failure;
    std::optional<cucascade::test::scoped_fault_hook> wait_failure;
    if (fail) {
      record_failure.emplace(cucascade::test::fault_point::event_record,
                             cudaErrorInvalidResourceHandle);
      wait_failure.emplace(cucascade::test::fault_point::stream_synchronize,
                           cudaErrorInvalidResourceHandle);
    }
    resource->deallocate(stream, uncertain, 512);
  }

  REQUIRE(cudaGetLastError() == cudaSuccess);
  resource.reset();
  REQUIRE(fixture.upstream.get_free_blocks() == initial_free - (fail ? 1 : 0));
}

TEST_CASE("Slab return metadata failure retains only its backing owner",
          "[small_pinned][fault_injection][slab_quarantine]")
{
  pinned_upstream_fixture fixture;
  rmm::cuda_stream stream;
  auto const initial_free = fixture.upstream.get_free_blocks();
  bool const fail         = GENERATE(false, true);
  std::optional<cucascade::memory::small_pinned_host_memory_resource> resource{std::in_place,
                                                                               fixture.upstream};
  auto* withheld  = resource->allocate_sync(512);
  auto* completed = resource->allocate_sync(8192);
  REQUIRE(fixture.upstream.get_free_blocks() == initial_free - 2);
  resource->deallocate_sync(completed, 8192);

  {
    std::optional<cucascade::test::scoped_fault_hook> failure;
    if (fail) {
      failure.emplace(cucascade::test::fault_point::slab_return, cudaErrorMemoryAllocation);
    }
    resource->deallocate(stream, withheld, 512);
  }

  REQUIRE(cudaGetLastError() == cudaSuccess);
  resource.reset();
  REQUIRE(fixture.upstream.get_free_blocks() == initial_free - (fail ? 1 : 0));
}
