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

# Set include directories for benchmarks
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Collect all benchmark sources
set(BENCHMARK_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_representation_converter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_main.cpp)

# Create benchmark executable
add_executable(cucascade_benchmarks ${BENCHMARK_SOURCES})

# Set include directories
target_include_directories(
  cucascade_benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                               ${benchmark_SOURCE_DIR}/include)

# Apply warning flags to benchmark executable
target_compile_options(
  cucascade_benchmarks
  PRIVATE $<$<COMPILE_LANGUAGE:C>:${CUCASCADE_CXX_WARNING_FLAGS}>
          $<$<COMPILE_LANGUAGE:CXX>:${CUCASCADE_CXX_WARNING_FLAGS}>
          ${CUCASCADE_CUDA_WARNING_FLAGS})

# Link dependencies
target_link_libraries(cucascade_benchmarks
                      PRIVATE cucascade benchmark::benchmark CUDA::cudart)

# Add custom target to run benchmarks
add_custom_target(
  run_benchmarks
  COMMAND cucascade_benchmarks
  DEPENDS cucascade_benchmarks
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running cuCascade benchmarks")
