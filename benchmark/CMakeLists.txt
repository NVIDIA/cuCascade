# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# The benchmarks exercise the cudf-backed representation converters, so they
# require the cucascade-cudf library. Skip them when it is not being built.
if(NOT CUCASCADE_BUILD_CUDF)
  message(
    STATUS
      "cuCascade benchmarks require CUCASCADE_BUILD_CUDF=ON (cudf-backed converters); skipping"
  )
  return()
endif()

# Fetch Google Benchmark
include(FetchContent)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3
  GIT_SHALLOW TRUE)

# Configure benchmark options
set(BENCHMARK_ENABLE_TESTING
    OFF
    CACHE BOOL "Disable benchmark tests" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS
    OFF
    CACHE BOOL "Disable gtest tests" FORCE)
set(BENCHMARK_ENABLE_INSTALL
    OFF
    CACHE BOOL "Disable install" FORCE)

FetchContent_MakeAvailable(benchmark)

# Collect all benchmark sources
set(BENCHMARK_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_representation_converter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_disk_converter.cpp)

# Create benchmark executable
add_executable(cucascade_benchmarks ${BENCHMARK_SOURCES})

# Set include directories
target_include_directories(
  cucascade_benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                               ${benchmark_SOURCE_DIR}/include)

# Apply warning flags to benchmark executable. Suppress -Wnull-dereference:
# triggered by RAPIDS/CCCL system headers
# (cuda/__utility/__basic_any/basic_any_value.h, virtcall.h) which we cannot
# fix.
target_compile_options(
  cucascade_benchmarks
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CUCASCADE_CXX_WARNING_FLAGS};-Wno-null-dereference>
    ${CUCASCADE_CUDA_WARNING_FLAGS})

# Link dependencies
target_link_libraries(
  cucascade_benchmarks PRIVATE cucascade_cudf benchmark::benchmark
                               benchmark::benchmark_main CUDA::cudart_static)

# Add custom target to run benchmarks
add_custom_target(
  run_benchmarks
  COMMAND cucascade_benchmarks
  DEPENDS cucascade_benchmarks
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running cuCascade benchmarks")

# =============================================================================
# Standalone parquet read benchmark (CLI, not Google Benchmark): cudf's default
# datasource vs the cucascade-io uring datasource. Requires the io library.
# =============================================================================
if(CUCASCADE_BUILD_IO)
  add_executable(cucascade_parquet_benchmark
                 ${CMAKE_CURRENT_SOURCE_DIR}/parquet_benchmark.cpp)

  target_compile_options(
    cucascade_parquet_benchmark
    PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:${CUCASCADE_CXX_WARNING_FLAGS};-Wno-null-dereference>
      ${CUCASCADE_CUDA_WARNING_FLAGS})

  target_link_libraries(cucascade_parquet_benchmark
                        PRIVATE cucascade_cudf cucascade_io CUDA::cudart_static)
endif()

# =============================================================================
# S3 parquet benchmark (CLI): cucascade-io REST datasource (AWS-SDK presigned
# URLs) vs kvikIO's remote S3 handle. Opt-in via CUCASCADE_BUILD_S3_BENCHMARK so
# aws-sdk-cpp never becomes a dependency of cuCascade itself (use the `s3-bench`
# pixi environment, which provides it).
# =============================================================================
if(CUCASCADE_BUILD_S3_BENCHMARK)
  find_package(AWSSDK REQUIRED COMPONENTS s3)
  find_package(kvikio REQUIRED CONFIG)

  add_executable(cucascade_s3_parquet_benchmark
                 ${CMAKE_CURRENT_SOURCE_DIR}/s3_parquet_benchmark.cpp)

  target_compile_options(
    cucascade_s3_parquet_benchmark
    PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:${CUCASCADE_CXX_WARNING_FLAGS};-Wno-null-dereference>
      ${CUCASCADE_CUDA_WARNING_FLAGS})

  target_link_libraries(
    cucascade_s3_parquet_benchmark
    PRIVATE cucascade_cudf cucascade_io kvikio::kvikio ${AWSSDK_LINK_LIBRARIES}
            CUDA::cudart_static)
endif()
