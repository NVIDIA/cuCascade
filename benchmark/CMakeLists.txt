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

# =============================================================================
# IoBufFragmentedSink micro-benchmark: header-only and cudf-free, so it is built
# ahead of the cudf gate below (it needs only the public include directory).
# =============================================================================
add_executable(cucascade_iobuf_sink_benchmark
               ${CMAKE_CURRENT_SOURCE_DIR}/iobuf_fragmented_sink_bench.cpp)
target_include_directories(cucascade_iobuf_sink_benchmark
                           PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(cucascade_iobuf_sink_benchmark PRIVATE cxx_std_20)
target_compile_options(
  cucascade_iobuf_sink_benchmark
  PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CUCASCADE_CXX_WARNING_FLAGS}>)
target_link_libraries(cucascade_iobuf_sink_benchmark
                      PRIVATE benchmark::benchmark)

# The remaining benchmarks exercise the cudf-backed representation converters,
# so they require the cucascade-cudf library. Skip them when it is not built.
if(NOT CUCASCADE_BUILD_CUDF)
  message(
    STATUS
      "cuCascade benchmarks require CUCASCADE_BUILD_CUDF=ON (cudf-backed converters); skipping"
  )
  return()
endif()

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

  # Range-read I/O benchmark: isolates raw column-chunk range reads (io_context
  # vector-io vs cudf datasource) into host/device, without parquet decode.
  add_executable(cucascade_parquet_io_benchmark
                 ${CMAKE_CURRENT_SOURCE_DIR}/parquet_io_benchmark.cpp)

  target_compile_options(
    cucascade_parquet_io_benchmark
    PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:${CUCASCADE_CXX_WARNING_FLAGS};-Wno-null-dereference>
      ${CUCASCADE_CUDA_WARNING_FLAGS})

  target_link_libraries(cucascade_parquet_io_benchmark
                        PRIVATE cucascade_cudf cucascade_io CUDA::cudart_static)
endif()

# =============================================================================
# S3 parquet benchmark (CLI): cucascade-io REST datasource (AWS-SDK presigned
# URLs). Opt-in via CUCASCADE_BUILD_S3_BENCHMARK so aws-sdk-cpp never becomes a
# dependency of cuCascade itself (use the `s3-bench` pixi environment, which
# provides it).
# =============================================================================
if(CUCASCADE_BUILD_S3_BENCHMARK)
  find_package(AWSSDK REQUIRED COMPONENTS s3)

  add_executable(cucascade_s3_parquet_benchmark
                 ${CMAKE_CURRENT_SOURCE_DIR}/s3_parquet_benchmark.cpp)

  target_compile_options(
    cucascade_s3_parquet_benchmark
    PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:${CUCASCADE_CXX_WARNING_FLAGS};-Wno-null-dereference>
      ${CUCASCADE_CUDA_WARNING_FLAGS})

  target_link_libraries(
    cucascade_s3_parquet_benchmark
    PRIVATE cucascade_cudf cucascade_io ${AWSSDK_LINK_LIBRARIES}
            CUDA::cudart_static)

  # S3 range-read I/O benchmark: raw column-chunk range reads (rest vector-io /
  # host->device) into host/device, without parquet decode.
  add_executable(cucascade_s3_parquet_io_benchmark
                 ${CMAKE_CURRENT_SOURCE_DIR}/parquet_s3_io_benchmark.cpp)

  target_compile_options(
    cucascade_s3_parquet_io_benchmark
    PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:${CUCASCADE_CXX_WARNING_FLAGS};-Wno-null-dereference>
      ${CUCASCADE_CUDA_WARNING_FLAGS})

  target_link_libraries(
    cucascade_s3_parquet_io_benchmark
    PRIVATE cucascade_cudf cucascade_io ${AWSSDK_LINK_LIBRARIES}
            CUDA::cudart_static)
endif()
