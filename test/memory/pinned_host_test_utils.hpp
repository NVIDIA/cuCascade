/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file pinned_host_test_utils.hpp
 * @brief Shared fixtures and checks for `small_pinned_host_memory_resource` tests
 *
 * Has no fault-injection dependency, so both `cucascade_tests` and
 * `cucascade_pinned_host_fault_tests` include it.
 */

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <cstddef>

namespace cucascade {
namespace test {

/**
 * @brief The legacy default stream, named explicitly instead of through the null handle
 */
inline ::cuda::stream_ref const legacy_default_stream{cudaStreamLegacy};

/**
 * @brief A 1 MiB pinned block pool for tests that construct their own
 * `small_pinned_host_memory_resource`
 */
struct pinned_upstream_fixture {
  rmm::mr::pinned_host_memory_resource pinned;
  memory::fixed_size_host_memory_resource upstream{
    0, pinned, 1ULL << 20, 1ULL << 20, 64ULL << 10, 4, 1};
};

/**
 * @brief A `small_pinned_host_memory_resource` over the `pinned_upstream_fixture` block pool
 */
struct pinned_host_test_fixture : pinned_upstream_fixture {
  memory::small_pinned_host_memory_resource resource;

  explicit pinned_host_test_fixture(std::size_t cache_limit = 64ULL << 10)
    : resource{upstream, cache_limit}
  {
  }
};

/**
 * @brief Returns the CUDA memory type of `pointer`, failing the test if the query fails
 */
[[nodiscard]] inline cudaMemoryType get_pointer_type(void const* pointer)
{
  cudaPointerAttributes attributes{};
  REQUIRE(cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess);
  return attributes.type;
}

inline void require_pointer_type(void const* pointer, cudaMemoryType type)
{
  REQUIRE(get_pointer_type(pointer) == type);
}

}  // namespace test
}  // namespace cucascade
