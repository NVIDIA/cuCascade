# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

# Minimal downstream project that consumes an installed cuCascade through
# find_package. Driven by run.cmake (pixi run test-install), which configures
# and builds one variant per package component.

cmake_minimum_required(VERSION 4.0)

project(cucascade_install_consumer LANGUAGES CXX)

set(CUCASCADE_CONSUMER_VARIANT
    "core"
    CACHE STRING "cuCascade package component to consume: core, io or cudf")
set_property(CACHE CUCASCADE_CONSUMER_VARIANT PROPERTY STRINGS core io cudf)

set(CUCASCADE_CONSUMER_MOODYCAMEL_INCLUDE_DIR
    ""
    CACHE
      PATH
      "Directory containing the moodycamel concurrentqueue headers cuCascade was built against"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The installed io headers include moodycamel's concurrentqueue headers, which
# cuCascade does not ship: like any downstream project, provide the same copy
# cuCascade was built against. Adding it to CMAKE_INCLUDE_PATH before
# find_package lets the package config's header check find it.
if(NOT CUCASCADE_CONSUMER_VARIANT STREQUAL "core")
  if(NOT CUCASCADE_CONSUMER_MOODYCAMEL_INCLUDE_DIR)
    message(FATAL_ERROR "Set CUCASCADE_CONSUMER_MOODYCAMEL_INCLUDE_DIR")
  endif()
  list(APPEND CMAKE_INCLUDE_PATH "${CUCASCADE_CONSUMER_MOODYCAMEL_INCLUDE_DIR}")
endif()

if(CUCASCADE_CONSUMER_VARIANT STREQUAL "core")
  find_package(cuCascade CONFIG REQUIRED)
  set(_consumer_target cuCascade::cucascade)
elseif(CUCASCADE_CONSUMER_VARIANT STREQUAL "io")
  find_package(cuCascade CONFIG REQUIRED COMPONENTS io)
  set(_consumer_target cuCascade::cucascade_io)
elseif(CUCASCADE_CONSUMER_VARIANT STREQUAL "cudf")
  find_package(cuCascade CONFIG REQUIRED COMPONENTS cudf)
  set(_consumer_target cuCascade::cucascade_cudf)
else()
  message(
    FATAL_ERROR
      "Unknown CUCASCADE_CONSUMER_VARIANT '${CUCASCADE_CONSUMER_VARIANT}'")
endif()

# The cuCascade package config does not look up kvikio, which the io targets
# link in installs built with CUCASCADE_BUILD_CUDF=ON. Resolve it here when no
# other dependency has defined it; QUIET keeps this a no-op where kvikio is not
# needed or not installed.
if(NOT CUCASCADE_CONSUMER_VARIANT STREQUAL "core"
   AND cuCascade_cudf_FOUND
   AND NOT TARGET kvikio::kvikio)
  find_package(kvikio CONFIG QUIET)
endif()

add_executable(cucascade_install_consumer ${CUCASCADE_CONSUMER_VARIANT}.cpp)
target_link_libraries(cucascade_install_consumer PRIVATE ${_consumer_target})
if(NOT CUCASCADE_CONSUMER_VARIANT STREQUAL "core")
  target_include_directories(
    cucascade_install_consumer
    PRIVATE "${CUCASCADE_CONSUMER_MOODYCAMEL_INCLUDE_DIR}")
endif()
