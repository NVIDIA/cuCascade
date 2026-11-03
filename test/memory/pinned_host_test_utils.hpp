/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <cstddef>

namespace cucascade {
namespace test {

struct pinned_host_test_fixture {
  rmm::mr::pinned_host_memory_resource pinned;
  memory::fixed_size_host_memory_resource upstream{
    0, pinned, 1ULL << 20, 1ULL << 20, 64ULL << 10, 4, 1};
  memory::small_pinned_host_memory_resource resource;

  explicit pinned_host_test_fixture(std::size_t cache_limit = 64ULL << 10)
    : resource{upstream, cache_limit}
  {
  }
};

inline void require_pointer_type(void* pointer, cudaMemoryType type)
{
  cudaPointerAttributes attributes{};
  REQUIRE(cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess);
  REQUIRE(attributes.type == type);
}

}  // namespace test
}  // namespace cucascade
