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

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

# Set project name and languages
project(
  cuCascade
  VERSION 0.1.0
  LANGUAGES CXX CUDA
  DESCRIPTION "GPU memory management and data representation library")

# Options
option(CUCASCADE_BUILD_TESTS "Build the test suite" ON)
option(CUCASCADE_BUILD_BENCHMARKS "Build the benchmark suite" ON)
option(CUCASCADE_BUILD_SHARED_LIBS "Build shared library" ON)
option(CUCASCADE_BUILD_STATIC_LIBS "Build static library" ON)
option(CUCASCADE_NVTX "Enable NVTX range annotations" OFF)
option(CUCASCADE_TOPOLOGY_ONLY
       "Build only the topology discovery library (no cudf/rmm dependency)" OFF)
option(
  CUCASCADE_BUILD_CUDF
  "Build the cudf-coupled cucascade-cudf library (representations, converters, profiler)"
  ON)

# At least one library type must be built
if(NOT CUCASCADE_BUILD_SHARED_LIBS AND NOT CUCASCADE_BUILD_STATIC_LIBS)
  message(
    FATAL_ERROR
      "At least one of CUCASCADE_BUILD_SHARED_LIBS or CUCASCADE_BUILD_STATIC_LIBS must be ON"
  )
endif()

# When topology-only, force off benchmarks (they require the full library)
if(CUCASCADE_TOPOLOGY_ONLY)
  set(CUCASCADE_BUILD_BENCHMARKS
      OFF
      CACHE BOOL "Build the benchmark suite" FORCE)
endif()

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

# TODO: Remove vendored FindCUDAToolkit once we require CMake 4.4

# Find required packages
find_package(CUDAToolkit REQUIRED)
if(NOT CUCASCADE_TOPOLOGY_ONLY)
  find_package(Threads REQUIRED)

  # Find RMM (the cudf-free core depends only on RMM, not cudf)
  find_package(rmm REQUIRED CONFIG)

  # Find cudf only when building the cudf-coupled cucascade-cudf library
  if(CUCASCADE_BUILD_CUDF)
    find_package(cudf REQUIRED CONFIG)
  endif()

  # Find numa (provided by numactl-devel or libnuma-dev depending on the package
  # manager). Uses the bundled FindNuma module, which produces the Numa::Numa
  # imported target. The same module is installed with the package so consumers
  # resolve libnuma on their own system rather than inheriting a hardcoded path.
  find_package(Numa REQUIRED)
endif()

# Set CUDA architectures
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 90)
endif()

# =============================================================================
# Compiler warnings
# =============================================================================
option(CUCASCADE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)

# Warning flags for C++
set(CUCASCADE_CXX_WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Wcast-align
    -Wunused
    -Wconversion
    -Wsign-conversion
    -Wdouble-promotion
    -Wformat=2
    -Wimplicit-fallthrough)

if(CUCASCADE_WARNINGS_AS_ERRORS)
  list(APPEND CUCASCADE_CXX_WARNING_FLAGS -Werror)
endif()

# Warning flags for CUDA (passed to host compiler via -Xcompiler)
set(CUCASCADE_CUDA_WARNING_FLAGS)
foreach(flag ${CUCASCADE_CXX_WARNING_FLAGS})
  list(APPEND CUCASCADE_CUDA_WARNING_FLAGS
       "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>")
endforeach()

# =============================================================================
# NVTX support
# =============================================================================
if(CUCASCADE_NVTX AND NOT CUCASCADE_TOPOLOGY_ONLY)
  find_package(nvtx3 REQUIRED)
endif()

# =============================================================================
# Object library (compiles sources once, used by both static and shared libs)
# =============================================================================
add_library(cucascade_topology_discovery_objects OBJECT)

if(NOT CUCASCADE_TOPOLOGY_ONLY)
  add_library(cucascade_objects OBJECT)

  # Apply warning flags to object library
  target_compile_options(
    cucascade_objects
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CUCASCADE_CXX_WARNING_FLAGS}>
            ${CUCASCADE_CUDA_WARNING_FLAGS})
endif()

if(NOT CUCASCADE_TOPOLOGY_ONLY AND CUCASCADE_BUILD_CUDF)
  add_library(cucascade_cudf_objects OBJECT)

  # Apply warning flags to the cudf object library
  target_compile_options(
    cucascade_cudf_objects
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CUCASCADE_CXX_WARNING_FLAGS}>
            ${CUCASCADE_CUDA_WARNING_FLAGS})
endif()

target_compile_options(
  cucascade_topology_discovery_objects
  PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CUCASCADE_CXX_WARNING_FLAGS}>
          ${CUCASCADE_CUDA_WARNING_FLAGS})

# NVTX compile definition
if(CUCASCADE_NVTX AND NOT CUCASCADE_TOPOLOGY_ONLY)
  target_compile_definitions(cucascade_objects PUBLIC CUCASCADE_NVTX)
  target_link_libraries(cucascade_objects PUBLIC nvtx3::nvtx3-cpp)
endif()

# Add source directories (they will add sources to cucascade_objects)
if(NOT CUCASCADE_TOPOLOGY_ONLY)
  add_subdirectory(src/cuda)
endif()
add_subdirectory(src/memory)
if(NOT CUCASCADE_TOPOLOGY_ONLY)
  add_subdirectory(src/data)
endif()
if(NOT CUCASCADE_TOPOLOGY_ONLY AND CUCASCADE_BUILD_CUDF)
  add_subdirectory(src/cudf)
endif()

# Common usage requirements shared by the object library and installable targets
set(CUCASCADE_PUBLIC_INCLUDE_DIRS
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

if(NOT CUCASCADE_TOPOLOGY_ONLY)
  set(CUCASCADE_PUBLIC_LINK_LIBS rmm::rmm CUDA::cudart_static Threads::Threads
                                 Numa::Numa)

  # Set include directories for the object library
  target_include_directories(cucascade_objects
                             PUBLIC ${CUCASCADE_PUBLIC_INCLUDE_DIRS})

  # Link dependencies to object library
  target_link_libraries(cucascade_objects PUBLIC ${CUCASCADE_PUBLIC_LINK_LIBS})

  # cxx_std_20 is PUBLIC so consumers know they need C++20 to use our headers.
  # cuda_std_20 is kept separate and applied PRIVATE to cucascade_objects to
  # avoid propagating it to consumers that have no CUDA compiler requirement.
  target_compile_features(cucascade_objects PUBLIC cxx_std_20)
  target_compile_features(cucascade_objects PRIVATE cuda_std_20)

  set_target_properties(
    cucascade_objects PROPERTIES CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE
                                                    ON)
endif()

if(NOT CUCASCADE_TOPOLOGY_ONLY AND CUCASCADE_BUILD_CUDF)
  # The cudf object library inherits the core usage requirements (include dirs,
  # rmm/cuda/numa, C++20, NVTX) by linking the core object library, and adds
  # cudf.
  target_link_libraries(cucascade_cudf_objects PUBLIC cucascade_objects
                                                      cudf::cudf)
  target_include_directories(cucascade_cudf_objects
                             PUBLIC ${CUCASCADE_PUBLIC_INCLUDE_DIRS})
  target_compile_features(cucascade_cudf_objects PUBLIC cxx_std_20)
  target_compile_features(cucascade_cudf_objects PRIVATE cuda_std_20)

  set_target_properties(
    cucascade_cudf_objects PROPERTIES CXX_EXTENSIONS OFF
                                      POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(cucascade_topology_discovery_objects
                           PUBLIC ${CUCASCADE_PUBLIC_INCLUDE_DIRS})
target_link_libraries(cucascade_topology_discovery_objects
                      PUBLIC CUDA::nvml_static)
target_compile_features(cucascade_topology_discovery_objects PUBLIC cxx_std_20)
target_compile_features(cucascade_topology_discovery_objects
                        PRIVATE cuda_std_20)
set_target_properties(
  cucascade_topology_discovery_objects PROPERTIES CXX_EXTENSIONS OFF
                                                  POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# Static library
# =============================================================================
if(CUCASCADE_BUILD_STATIC_LIBS)
  add_library(cucascade_topology_discovery_static STATIC
              $<TARGET_OBJECTS:cucascade_topology_discovery_objects>)
  add_library(cuCascade::cucascade_topology_discovery_static ALIAS
              cucascade_topology_discovery_static)

  target_link_libraries(cucascade_topology_discovery_static
                        PRIVATE CUDA::nvml_static)
  target_include_directories(cucascade_topology_discovery_static
                             PUBLIC ${CUCASCADE_PUBLIC_INCLUDE_DIRS})
  target_compile_features(cucascade_topology_discovery_static PUBLIC cxx_std_20)
  # Work around a bug in libnvidia-ml.so - it attempts to call back into the
  # stub library (https://nvbugspro.nvidia.com/bug/6174166)
  target_link_options(cucascade_topology_discovery_static INTERFACE
                      "LINKER:--exclude-libs,libnvidia-ml")

  set_target_properties(
    cucascade_topology_discovery_static
    PROPERTIES OUTPUT_NAME cucascade_topology_discovery
               EXPORT_NAME cucascade_topology_discovery_static)

  if(NOT CUCASCADE_TOPOLOGY_ONLY)
    add_library(cucascade_static STATIC $<TARGET_OBJECTS:cucascade_objects>)
    add_library(cuCascade::cucascade_static ALIAS cucascade_static)

    target_include_directories(cucascade_static
                               PUBLIC ${CUCASCADE_PUBLIC_INCLUDE_DIRS})
    target_link_libraries(
      cucascade_static PUBLIC ${CUCASCADE_PUBLIC_LINK_LIBS}
                              cucascade_topology_discovery_static)
    target_compile_features(cucascade_static PUBLIC cxx_std_20)

    set_target_properties(
      cucascade_static PROPERTIES OUTPUT_NAME cucascade EXPORT_NAME
                                                        cucascade_static)

    if(CUCASCADE_BUILD_CUDF)
      add_library(cucascade_cudf_static STATIC
                  $<TARGET_OBJECTS:cucascade_cudf_objects>)
      add_library(cuCascade::cucascade_cudf_static ALIAS cucascade_cudf_static)

      target_include_directories(cucascade_cudf_static
                                 PUBLIC ${CUCASCADE_PUBLIC_INCLUDE_DIRS})
      target_link_libraries(cucascade_cudf_static PUBLIC cudf::cudf
                                                         cucascade_static)
      target_compile_features(cucascade_cudf_static PUBLIC cxx_std_20)

      set_target_properties(
        cucascade_cudf_static PROPERTIES OUTPUT_NAME cucascade_cudf
                                         EXPORT_NAME cucascade_cudf_static)
    endif()
  endif()
endif()

# =============================================================================
# Shared library
# =============================================================================
if(CUCASCADE_BUILD_SHARED_LIBS)
  add_library(cucascade_topology_discovery_shared SHARED
              $<TARGET_OBJECTS:cucascade_topology_discovery_objects>)
  add_library(cuCascade::cucascade_topology_discovery_shared ALIAS
              cucascade_topology_discovery_shared)

  target_link_libraries(cucascade_topology_discovery_shared
                        PRIVATE CUDA::nvml_static)
  target_include_directories(cucascade_topology_discovery_shared
                             PUBLIC ${CUCASCADE_PUBLIC_INCLUDE_DIRS})
  target_compile_features(cucascade_topology_discovery_shared PUBLIC cxx_std_20)
  # Work around a bug in libnvidia-ml.so - it attempts to call back into the
  # stub library (https://nvbugspro.nvidia.com/bug/6174166)
  target_link_options(cucascade_topology_discovery_shared PRIVATE
                      "LINKER:--exclude-libs,libnvidia-ml")

  set_target_properties(cucascade_topology_discovery_shared
                        PROPERTIES OUTPUT_NAME cucascade_topology_discovery)

  if(NOT CUCASCADE_TOPOLOGY_ONLY)
    add_library(cucascade_shared SHARED $<TARGET_OBJECTS:cucascade_objects>)
    add_library(cuCascade::cucascade_shared ALIAS cucascade_shared)

    target_include_directories(cucascade_shared
                               PUBLIC ${CUCASCADE_PUBLIC_INCLUDE_DIRS})
    target_link_libraries(
      cucascade_shared PUBLIC ${CUCASCADE_PUBLIC_LINK_LIBS}
                              cucascade_topology_discovery_shared)
    target_compile_features(cucascade_shared PUBLIC cxx_std_20)

    set_target_properties(cucascade_shared PROPERTIES OUTPUT_NAME cucascade)

    if(CUCASCADE_BUILD_CUDF)
      add_library(cucascade_cudf_shared SHARED
                  $<TARGET_OBJECTS:cucascade_cudf_objects>)
      add_library(cuCascade::cucascade_cudf_shared ALIAS cucascade_cudf_shared)

      target_include_directories(cucascade_cudf_shared
                                 PUBLIC ${CUCASCADE_PUBLIC_INCLUDE_DIRS})
      target_link_libraries(cucascade_cudf_shared PUBLIC cudf::cudf
                                                         cucascade_shared)
      target_compile_features(cucascade_cudf_shared PUBLIC cxx_std_20)

      set_target_properties(cucascade_cudf_shared PROPERTIES OUTPUT_NAME
                                                             cucascade_cudf)
    endif()
  endif()
endif()

# =============================================================================
# Default target alias (prefer shared if available, else static)
# =============================================================================
if(CUCASCADE_BUILD_SHARED_LIBS)
  if(NOT CUCASCADE_TOPOLOGY_ONLY)
    add_library(cuCascade::cucascade ALIAS cucascade_shared)
    # For backwards compatibility, also create 'cucascade' as an alias
    add_library(cucascade ALIAS cucascade_shared)
    if(CUCASCADE_BUILD_CUDF)
      add_library(cuCascade::cucascade_cudf ALIAS cucascade_cudf_shared)
      add_library(cucascade_cudf ALIAS cucascade_cudf_shared)
    endif()
  endif()
  add_library(cuCascade::cucascade_topology_discovery ALIAS
              cucascade_topology_discovery_shared)
  add_library(cucascade_topology_discovery ALIAS
              cucascade_topology_discovery_shared)
elseif(CUCASCADE_BUILD_STATIC_LIBS)
  if(NOT CUCASCADE_TOPOLOGY_ONLY)
    add_library(cuCascade::cucascade ALIAS cucascade_static)
    add_library(cucascade ALIAS cucascade_static)
    if(CUCASCADE_BUILD_CUDF)
      add_library(cuCascade::cucascade_cudf ALIAS cucascade_cudf_static)
      add_library(cucascade_cudf ALIAS cucascade_cudf_static)
    endif()
  endif()
  add_library(cuCascade::cucascade_topology_discovery ALIAS
              cucascade_topology_discovery_static)
  add_library(cucascade_topology_discovery ALIAS
              cucascade_topology_discovery_static)
endif()

# =============================================================================
# Installation
# =============================================================================
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install headers
if(CUCASCADE_TOPOLOGY_ONLY)
  install(FILES include/cucascade/memory/topology_discovery.hpp
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cucascade/memory)
elseif(CUCASCADE_BUILD_CUDF)
  install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h")
else()
  # cudf disabled: install all headers except the cudf-coupled subtree, which is
  # unusable without the cucascade-cudf library and its libcudf dependency.
  install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h"
    PATTERN "cucascade/cudf" EXCLUDE)
endif()

# Collect targets to install
set(CUCASCADE_INSTALL_TARGETS)
if(CUCASCADE_BUILD_STATIC_LIBS)
  list(APPEND CUCASCADE_INSTALL_TARGETS cucascade_topology_discovery_static)
  if(NOT CUCASCADE_TOPOLOGY_ONLY)
    list(APPEND CUCASCADE_INSTALL_TARGETS cucascade_static)
    if(CUCASCADE_BUILD_CUDF)
      list(APPEND CUCASCADE_INSTALL_TARGETS cucascade_cudf_static)
    endif()
  endif()
endif()
if(CUCASCADE_BUILD_SHARED_LIBS)
  list(APPEND CUCASCADE_INSTALL_TARGETS cucascade_topology_discovery_shared)
  if(NOT CUCASCADE_TOPOLOGY_ONLY)
    list(APPEND CUCASCADE_INSTALL_TARGETS cucascade_shared)
    if(CUCASCADE_BUILD_CUDF)
      list(APPEND CUCASCADE_INSTALL_TARGETS cucascade_cudf_shared)
    endif()
  endif()
endif()

# Install library targets
install(
  TARGETS ${CUCASCADE_INSTALL_TARGETS}
  EXPORT cuCascadeTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install export targets
install(
  EXPORT cuCascadeTargets
  FILE cuCascadeTargets.cmake
  NAMESPACE cuCascade::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuCascade)

# Generate package config file
if(CUCASCADE_TOPOLOGY_ONLY)
  set(CUCASCADE_CONFIG_TEMPLATE
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuCascadeTopologyOnlyConfig.cmake.in)
else()
  set(CUCASCADE_CONFIG_TEMPLATE
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuCascadeConfig.cmake.in)
endif()

configure_package_config_file(
  ${CUCASCADE_CONFIG_TEMPLATE} ${CMAKE_CURRENT_BINARY_DIR}/cuCascadeConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuCascade)

# Generate version file
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/cuCascadeConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

# Install config files
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cuCascadeConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/cuCascadeConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuCascade)

# Bundle the FindNuma module so consumers can rerun the same find logic (the
# full library links the Numa::Numa target it produces). Not needed for
# topology-only builds, which do not depend on libnuma.
if(NOT CUCASCADE_TOPOLOGY_ONLY)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/FindNuma.cmake
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuCascade/Modules)
endif()

# =============================================================================
# Tests
# =============================================================================
if(CUCASCADE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

# =============================================================================
# Benchmarks
# =============================================================================
if(CUCASCADE_BUILD_BENCHMARKS)
  add_subdirectory(benchmark)
endif()
