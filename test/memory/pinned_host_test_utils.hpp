/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

namespace cucascade {
namespace test {

inline void require_pointer_type(void* pointer, cudaMemoryType type)
{
  cudaPointerAttributes attributes{};
  REQUIRE(cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess);
  REQUIRE(attributes.type == type);
}

}  // namespace test
}  // namespace cucascade
