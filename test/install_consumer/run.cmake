# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

# Installs a cuCascade build tree into a scratch prefix and builds the consumer
# project in this directory against it, once per package component.
#
# Usage: cmake [-D<VAR>=<value>...] -P test/install_consumer/run.cmake
#
# Optional variables: BUILD_DIR is the built cuCascade tree (default
# build/release), WORK_DIR the scratch directory under build/ that is wiped
# first (default build/install-consumer), VARIANTS the consumer variants to
# build (default core;io;cudf), and MOODYCAMEL_INCLUDE_DIR the moodycamel
# headers for the io and cudf consumers (default: the copy BUILD_DIR used).

cmake_minimum_required(VERSION 4.0)

if(NOT BUILD_DIR)
  set(BUILD_DIR build/release)
endif()
if(NOT WORK_DIR)
  set(WORK_DIR build/install-consumer)
endif()
if(NOT VARIANTS)
  set(VARIANTS core io cudf)
endif()

cmake_path(ABSOLUTE_PATH BUILD_DIR NORMALIZE)
cmake_path(ABSOLUTE_PATH WORK_DIR NORMALIZE)
cmake_path(SET _scratch_dir NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../../build")
cmake_path(IS_PREFIX _scratch_dir "${WORK_DIR}" _work_dir_in_scratch)
cmake_path(IS_PREFIX WORK_DIR "${BUILD_DIR}" _work_dir_contains_build)
cmake_path(IS_PREFIX BUILD_DIR "${WORK_DIR}" _work_dir_in_build)
if(NOT _work_dir_in_scratch
   OR WORK_DIR STREQUAL _scratch_dir
   OR _work_dir_contains_build
   OR _work_dir_in_build)
  message(FATAL_ERROR "WORK_DIR must be a directory under ${_scratch_dir} "
                      "that neither contains nor is inside BUILD_DIR")
endif()
set(_prefix "${WORK_DIR}/prefix")

# Installed consumers supply moodycamel themselves; use the copy BUILD_DIR was
# built against: a host-provided include dir, else the fetched copy. A build
# using CUCASCADE_MOODYCAMEL_TARGET needs MOODYCAMEL_INCLUDE_DIR.
if(NOT MOODYCAMEL_INCLUDE_DIR)
  foreach(entry IN ITEMS CUCASCADE_MOODYCAMEL_INCLUDE_DIR
                         concurrentqueue_SOURCE_DIR)
    file(STRINGS "${BUILD_DIR}/CMakeCache.txt" _line
         REGEX "^${entry}:[A-Z]+=.+$")
    if(_line)
      string(REGEX REPLACE "^[^=]*=" "" MOODYCAMEL_INCLUDE_DIR "${_line}")
      break()
    endif()
  endforeach()
endif()
if(NOT MOODYCAMEL_INCLUDE_DIR AND NOT VARIANTS STREQUAL "core")
  message(FATAL_ERROR "Pass -DMOODYCAMEL_INCLUDE_DIR=<moodycamel headers>")
endif()

file(REMOVE_RECURSE "${WORK_DIR}")

message(STATUS "Installing ${BUILD_DIR} into ${_prefix}")
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix
                        "${_prefix}" OUTPUT_QUIET COMMAND_ERROR_IS_FATAL ANY)

# Dependencies fetched at build time must not leak into the install.
file(
  GLOB_RECURSE _leaked_files
  RELATIVE "${_prefix}"
  "${_prefix}/*")
list(FILTER _leaked_files INCLUDE REGEX "(^|/)(concurrentqueue|moodycamel)/")
if(_leaked_files)
  message(FATAL_ERROR "cmake --install installed fetched dependencies: "
                      "${_leaked_files}")
endif()

foreach(variant IN LISTS VARIANTS)
  set(_binary_dir "${WORK_DIR}/${variant}")
  message(STATUS "Building ${variant} consumer in ${_binary_dir}")
  # Drop the environment's compiler and linker flags (pixi adds its prefix's
  # include and library dirs), so only the package's usage requirements apply.
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -E env --unset=CFLAGS --unset=CXXFLAGS --unset=CPPFLAGS
      --unset=LDFLAGS "${CMAKE_COMMAND}" -S "${CMAKE_CURRENT_LIST_DIR}" -B
      "${_binary_dir}" -G Ninja -DCMAKE_BUILD_TYPE=Release
      "-DCMAKE_PREFIX_PATH=${_prefix}" -DCUCASCADE_CONSUMER_VARIANT=${variant}
      "-DCUCASCADE_CONSUMER_MOODYCAMEL_INCLUDE_DIR=${MOODYCAMEL_INCLUDE_DIR}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _output)
  message("${_output}")
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "Configuring the ${variant} consumer failed")
  endif()
  # A warning from the installed package config means a dependency it checks for
  # is missing.
  if(_output MATCHES "CMake Warning at [^\n]*cuCascadeConfig\\.cmake")
    message(FATAL_ERROR "The cuCascade package config warned for the "
                        "${variant} consumer")
  endif()
  execute_process(COMMAND "${CMAKE_COMMAND}" --build "${_binary_dir}"
                          COMMAND_ERROR_IS_FATAL ANY)
endforeach()
