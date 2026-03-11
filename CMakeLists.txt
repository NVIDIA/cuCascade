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

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

# Set project name and languages
project(
  cuCascade
  VERSION 0.1.0
  LANGUAGES C CXX CUDA
  DESCRIPTION "GPU memory management and data representation library")

# Options
option(CUCASCADE_BUILD_TESTS "Build the test suite" ON)
option(CUCASCADE_BUILD_BENCHMARKS "Build the benchmark suite" ON)
option(CUCASCADE_BUILD_SHARED_LIBS "Build shared library" ON)
option(CUCASCADE_BUILD_STATIC_LIBS "Build static library" ON)

# At least one library type must be built
if(NOT CUCASCADE_BUILD_SHARED_LIBS AND NOT CUCASCADE_BUILD_STATIC_LIBS)
  message(
    FATAL_ERROR
      "At least one of CUCASCADE_BUILD_SHARED_LIBS or CUCASCADE_BUILD_STATIC_LIBS must be ON"
  )
endif()

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Find RMM from libcudf installation
find_package(rmm REQUIRED CONFIG)

# Find cudf for data representation support
find_package(cudf REQUIRED CONFIG)

# Find numa (provided by numactl-devel or libnuma-dev depending on the package
# manager)
find_library(NUMA_LIB numa REQUIRED)

# Set CUDA architectures
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 90)
endif()

# =============================================================================
# Compiler warnings
# =============================================================================
option(CUCASCADE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)

# Warning flags for C/C++
set(CUCASCADE_CXX_WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Wcast-align
    -Wunused
    -Wconversion
    -Wsign-conversion
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
    -Wimplicit-fallthrough)

if(WARNINGS_AS_ERRORS)
  list(APPEND CUCASCADE_CXX_WARNING_FLAGS -Werror)
endif()

# Warning flags for CUDA (passed to host compiler via -Xcompiler)
set(CUCASCADE_CUDA_WARNING_FLAGS)
foreach(flag ${CUCASCADE_CXX_WARNING_FLAGS})
  list(APPEND CUCASCADE_CUDA_WARNING_FLAGS
       "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>")
endforeach()

# =============================================================================
# Object library (compiles sources once, used by both static and shared libs)
# =============================================================================
add_library(cucascade_objects OBJECT)

# Apply warning flags to object library
target_compile_options(
  cucascade_objects
  PRIVATE $<$<COMPILE_LANGUAGE:C>:${CUCASCADE_CXX_WARNING_FLAGS}>
          $<$<COMPILE_LANGUAGE:CXX>:${CUCASCADE_CXX_WARNING_FLAGS}>
          ${CUCASCADE_CUDA_WARNING_FLAGS})

# Add source directories (they will add sources to cucascade_objects)
add_subdirectory(src/memory)
add_subdirectory(src/data)

# Common usage requirements shared by the object library and installable targets
set(CUCASCADE_PUBLIC_INCLUDE_DIRS
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

set(CUCASCADE_PUBLIC_LINK_LIBS rmm::rmm cudf::cudf CUDA::cudart CUDA::nvml
                               Threads::Threads ${NUMA_LIB})

# Set include directories for the object library
target_include_directories(cucascade_objects
                           PUBLIC ${CUCASCADE_PUBLIC_INCLUDE_DIRS})

# Link dependencies to object library
target_link_libraries(cucascade_objects PUBLIC ${CUCASCADE_PUBLIC_LINK_LIBS})

# cxx_std_20 is PUBLIC so consumers know they need C++20 to use our headers.
# cuda_std_20 is kept separate and applied PRIVATE to cucascade_objects to avoid
# propagating it to consumers that have no CUDA compiler requirement.
target_compile_features(cucascade_objects PUBLIC cxx_std_20)
target_compile_features(cucascade_objects PRIVATE cuda_std_20)

set_target_properties(cucascade_objects PROPERTIES CXX_EXTENSIONS OFF
                                                   POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# Static library
# =============================================================================
if(CUCASCADE_BUILD_STATIC_LIBS)
  add_library(cucascade_static STATIC $<TARGET_OBJECTS:cucascade_objects>)
  add_library(cuCascade::cucascade_static ALIAS cucascade_static)

  target_include_directories(cucascade_static
                             PUBLIC ${CUCASCADE_PUBLIC_INCLUDE_DIRS})
  target_link_libraries(cucascade_static PUBLIC ${CUCASCADE_PUBLIC_LINK_LIBS})
  target_compile_features(cucascade_static PUBLIC cxx_std_20)

  set_target_properties(
    cucascade_static PROPERTIES OUTPUT_NAME cucascade EXPORT_NAME
                                                      cucascade_static)
endif()

# =============================================================================
# Shared library
# =============================================================================
if(CUCASCADE_BUILD_SHARED_LIBS)
  add_library(cucascade_shared SHARED $<TARGET_OBJECTS:cucascade_objects>)
  add_library(cuCascade::cucascade_shared ALIAS cucascade_shared)

  target_include_directories(cucascade_shared
                             PUBLIC ${CUCASCADE_PUBLIC_INCLUDE_DIRS})
  target_link_libraries(cucascade_shared PUBLIC ${CUCASCADE_PUBLIC_LINK_LIBS})
  target_compile_features(cucascade_shared PUBLIC cxx_std_20)

  set_target_properties(
    cucascade_shared
    PROPERTIES OUTPUT_NAME cucascade
               VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR})
endif()

# =============================================================================
# Default target alias (prefer shared if available, else static)
# =============================================================================
if(CUCASCADE_BUILD_SHARED_LIBS)
  add_library(cuCascade::cucascade ALIAS cucascade_shared)
  # For backwards compatibility, also create 'cucascade' as an alias
  add_library(cucascade ALIAS cucascade_shared)
elseif(CUCASCADE_BUILD_STATIC_LIBS)
  add_library(cuCascade::cucascade ALIAS cucascade_static)
  add_library(cucascade ALIAS cucascade_static)
endif()

# =============================================================================
# Installation
# =============================================================================
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install headers
install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h")

# Collect targets to install
set(CUCASCADE_INSTALL_TARGETS)
if(CUCASCADE_BUILD_STATIC_LIBS)
  list(APPEND CUCASCADE_INSTALL_TARGETS cucascade_static)
endif()
if(CUCASCADE_BUILD_SHARED_LIBS)
  list(APPEND CUCASCADE_INSTALL_TARGETS cucascade_shared)
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
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuCascadeConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cuCascadeConfig.cmake
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
