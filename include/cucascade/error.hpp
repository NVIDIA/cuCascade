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
#include <stdexcept>
#include <string>

#if defined(CUCASCADE_NVTX)
#include <nvtx3/nvtx3.hpp>
#endif

namespace cucascade {

struct cuda_error : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct logic_error : public std::logic_error {
  using std::logic_error::logic_error;
};

#if defined(CUCASCADE_NVTX)
struct libcucascade_domain {
  static constexpr char const* name{"libcucascade"};
};
#endif

}  // namespace cucascade

// clang-format off

#define CUCASCADE_STRINGIFY_DETAIL(x) #x
#define CUCASCADE_STRINGIFY(x) CUCASCADE_STRINGIFY_DETAIL(x)

// ---------------------------------------------------------------------------
// CUCASCADE_CUDA_TRY – wraps a CUDA runtime call and throws on failure.
// ---------------------------------------------------------------------------

#define CUCASCADE_CUDA_TRY_2(_call, _exception_type)                                                  \
  do {                                                                                                \
    cudaError_t const error = (_call);                                                                \
    if (cudaSuccess != error) {                                                                       \
      cudaGetLastError();                                                                             \
      throw _exception_type{std::string{"CUDA error at: "} + __FILE__ + ":" +                         \
                            CUCASCADE_STRINGIFY(__LINE__) + ": " + cudaGetErrorName(error) + " " +     \
                            cudaGetErrorString(error)};                                               \
    }                                                                                                 \
  } while (0)

#define CUCASCADE_CUDA_TRY_1(_call) CUCASCADE_CUDA_TRY_2(_call, cucascade::cuda_error)

#define GET_CUCASCADE_CUDA_TRY_MACRO(_1, _2, NAME, ...) NAME
#define CUCASCADE_CUDA_TRY(...)                                                         \
  GET_CUCASCADE_CUDA_TRY_MACRO(__VA_ARGS__, CUCASCADE_CUDA_TRY_2, CUCASCADE_CUDA_TRY_1) \
  (__VA_ARGS__)

// ---------------------------------------------------------------------------
// CUCASCADE_CUDA_TRY_ALLOC – like CUCASCADE_CUDA_TRY but throws allocation-
// specific exceptions (rmm::out_of_memory / rmm::bad_alloc).
// ---------------------------------------------------------------------------

#define CUCASCADE_CUDA_TRY_ALLOC_2(_call, _num_bytes)                                                 \
  do {                                                                                                \
    cudaError_t const error = (_call);                                                                \
    if (cudaSuccess != error) {                                                                       \
      cudaGetLastError();                                                                             \
      auto const msg = std::string{"CUDA error (failed to allocate "} +                               \
                       std::to_string(_num_bytes) + " bytes) at: " + __FILE__ + ":" +                 \
                       CUCASCADE_STRINGIFY(__LINE__) + ": " + cudaGetErrorName(error) + " " +          \
                       cudaGetErrorString(error);                                                     \
      if (cudaErrorMemoryAllocation == error) { throw rmm::out_of_memory{msg}; }                      \
      throw rmm::bad_alloc{msg};                                                                      \
    }                                                                                                 \
  } while (0)

#define CUCASCADE_CUDA_TRY_ALLOC_1(_call)                                                             \
  do {                                                                                                \
    cudaError_t const error = (_call);                                                                \
    if (cudaSuccess != error) {                                                                       \
      cudaGetLastError();                                                                             \
      auto const msg = std::string{"CUDA error at: "} + __FILE__ + ":" +                              \
                       CUCASCADE_STRINGIFY(__LINE__) + ": " + cudaGetErrorName(error) + " " +          \
                       cudaGetErrorString(error);                                                     \
      if (cudaErrorMemoryAllocation == error) { throw rmm::out_of_memory{msg}; }                      \
      throw rmm::bad_alloc{msg};                                                                      \
    }                                                                                                 \
  } while (0)

#define GET_CUCASCADE_CUDA_TRY_ALLOC_MACRO(_1, _2, NAME, ...) NAME
#define CUCASCADE_CUDA_TRY_ALLOC(...)                                                                     \
  GET_CUCASCADE_CUDA_TRY_ALLOC_MACRO(__VA_ARGS__, CUCASCADE_CUDA_TRY_ALLOC_2, CUCASCADE_CUDA_TRY_ALLOC_1) \
  (__VA_ARGS__)

// ---------------------------------------------------------------------------
// CUCASCADE_FAIL – throws an exception with file/line context.
// One-argument form throws cucascade::logic_error; two-argument form throws
// the caller-specified exception type.
// ---------------------------------------------------------------------------

#define CUCASCADE_FAIL_2(_what, _exception_type)                                                              \
  throw _exception_type                                                                                      \
  {                                                                                                          \
    std::string{"cuCascade failure at: "} + __FILE__ + ":" + CUCASCADE_STRINGIFY(__LINE__) + ": " + _what     \
  }

#define CUCASCADE_FAIL_1(_what) CUCASCADE_FAIL_2(_what, cucascade::logic_error)

#define GET_CUCASCADE_FAIL_MACRO(_1, _2, NAME, ...) NAME
#define CUCASCADE_FAIL(...)                                             \
  GET_CUCASCADE_FAIL_MACRO(__VA_ARGS__, CUCASCADE_FAIL_2, CUCASCADE_FAIL_1) \
  (__VA_ARGS__)

// ---------------------------------------------------------------------------
// CUCASCADE_ASSERT_CUDA_SUCCESS – assert-based CUDA error check for use in
// noexcept / destructor contexts. In release builds the call is still
// executed but the error code is discarded.
// ---------------------------------------------------------------------------

#ifdef NDEBUG
#define CUCASCADE_ASSERT_CUDA_SUCCESS(_call) \
  do {                                       \
    (_call);                                 \
  } while (0)
#else
#define CUCASCADE_ASSERT_CUDA_SUCCESS(_call)                                            \
  do {                                                                                  \
    cudaError_t const status__ = (_call);                                               \
    if (status__ != cudaSuccess) {                                                      \
      std::cerr << "CUDA Error detected. " << cudaGetErrorName(status__) << " "         \
                << cudaGetErrorString(status__) << std::endl;                           \
    }                                                                                   \
    assert(status__ == cudaSuccess);                                                    \
  } while (0)
#endif

// ---------------------------------------------------------------------------
// CUCASCADE_FUNC_RANGE – NVTX function-level range for profiling.
// Enabled when CUCASCADE_NVTX is defined at compile time.
// ---------------------------------------------------------------------------

#if defined(CUCASCADE_NVTX)
#define CUCASCADE_FUNC_RANGE() NVTX3_FUNC_RANGE_IN(cucascade::libcucascade_domain)
#else
#define CUCASCADE_FUNC_RANGE()
#endif

// clang-format on
