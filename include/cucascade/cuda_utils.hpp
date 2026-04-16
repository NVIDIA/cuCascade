/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <cassert>
#include <iostream>
#include <string>

// =============================================================================
// CUDA error-checking macros (replacements for rmm/detail/error.hpp macros)
// =============================================================================

#define CUCASCADE_STRINGIFY_DETAIL(x) #x
#define CUCASCADE_STRINGIFY(x)        CUCASCADE_STRINGIFY_DETAIL(x)

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * If the call does not return cudaSuccess, clears the error and throws
 * rmm::cuda_error with file/line context.
 */
#define CUCASCADE_CUDA_TRY(call)                                                                   \
  do {                                                                                             \
    cudaError_t const error = (call);                                                              \
    if (cudaSuccess != error) {                                                                    \
      cudaGetLastError();                                                                          \
      throw rmm::cuda_error{std::string{"CUDA error at: "} + __FILE__ + ":" +                      \
                            CUCASCADE_STRINGIFY(__LINE__) + ": " + cudaGetErrorName(error) + " " + \
                            cudaGetErrorString(error)};                                            \
    }                                                                                              \
  } while (0)

/**
 * @brief Error checking macro for CUDA memory allocation calls.
 *
 * Throws rmm::out_of_memory when cudaErrorMemoryAllocation is returned,
 * rmm::bad_alloc for other failures.
 *
 * Can be called with 1 or 2 arguments:
 * - CUCASCADE_CUDA_TRY_ALLOC(cuda_call)
 * - CUCASCADE_CUDA_TRY_ALLOC(cuda_call, num_bytes)
 */
#define CUCASCADE_CUDA_TRY_ALLOC(...)                                    \
  GET_CUCASCADE_CUDA_TRY_ALLOC_MACRO(                                    \
    __VA_ARGS__, CUCASCADE_CUDA_TRY_ALLOC_2, CUCASCADE_CUDA_TRY_ALLOC_1) \
  (__VA_ARGS__)
#define GET_CUCASCADE_CUDA_TRY_ALLOC_MACRO(_1, _2, NAME, ...) NAME

#define CUCASCADE_CUDA_TRY_ALLOC_2(_call, num_bytes)                                          \
  do {                                                                                        \
    cudaError_t const error = (_call);                                                        \
    if (cudaSuccess != error) {                                                               \
      cudaGetLastError();                                                                     \
      auto const msg = std::string{"CUDA error (failed to allocate "} +                       \
                       std::to_string(num_bytes) + " bytes) at: " + __FILE__ + ":" +          \
                       CUCASCADE_STRINGIFY(__LINE__) + ": " + cudaGetErrorName(error) + " " + \
                       cudaGetErrorString(error);                                             \
      if (cudaErrorMemoryAllocation == error) { throw rmm::out_of_memory{msg}; }              \
      throw rmm::bad_alloc{msg};                                                              \
    }                                                                                         \
  } while (0)

#define CUCASCADE_CUDA_TRY_ALLOC_1(_call)                                                     \
  do {                                                                                        \
    cudaError_t const error = (_call);                                                        \
    if (cudaSuccess != error) {                                                               \
      cudaGetLastError();                                                                     \
      auto const msg = std::string{"CUDA error at: "} + __FILE__ + ":" +                      \
                       CUCASCADE_STRINGIFY(__LINE__) + ": " + cudaGetErrorName(error) + " " + \
                       cudaGetErrorString(error);                                             \
      if (cudaErrorMemoryAllocation == error) { throw rmm::out_of_memory{msg}; }              \
      throw rmm::bad_alloc{msg};                                                              \
    }                                                                                         \
  } while (0)

/**
 * @brief Debug-only CUDA error check (like assert for CUDA calls).
 *
 * In Release builds, invokes the call without checking. In Debug builds,
 * asserts that the call returns cudaSuccess.
 */
#ifdef NDEBUG
#define CUCASCADE_ASSERT_CUDA_SUCCESS(_call) \
  do {                                       \
    (_call);                                 \
  } while (0);
#else
#define CUCASCADE_ASSERT_CUDA_SUCCESS(_call)                                    \
  do {                                                                          \
    cudaError_t const status__ = (_call);                                       \
    if (status__ != cudaSuccess) {                                              \
      std::cerr << "CUDA Error detected. " << cudaGetErrorName(status__) << " " \
                << cudaGetErrorString(status__) << std::endl;                   \
    }                                                                           \
    /* NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay) */   \
    assert(status__ == cudaSuccess);                                            \
  } while (0)
#endif

// =============================================================================
// NVTX range macro (replacement for rmm/detail/nvtx/ranges.hpp)
// =============================================================================

#if defined(CUCASCADE_NVTX)
#include <nvtx3/nvtx3.hpp>

namespace cucascade {
/**
 * @brief Tag type for cuCascade's NVTX domain.
 */
struct cucascade_domain {
  static constexpr char const* name{"cucascade"};
};
}  // namespace cucascade

/**
 * @brief Convenience macro for generating an NVTX range in the `cucascade` domain
 * from the lifetime of a function.
 */
#define CUCASCADE_FUNC_RANGE() NVTX3_FUNC_RANGE_IN(cucascade::cucascade_domain)
#else
#define CUCASCADE_FUNC_RANGE()
#endif
