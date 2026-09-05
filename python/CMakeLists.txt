# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

project(
  cucascade_python
  VERSION 0.1.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBURING REQUIRED IMPORTED_TARGET liburing)
pkg_check_modules(CURL REQUIRED IMPORTED_TARGET libcurl)
find_package(OpenSSL REQUIRED)

find_package(cudf REQUIRED)
find_package(cuCascade REQUIRED COMPONENTS cudf io)

set(rapids-cmake-dir
    ""
    CACHE
      PATH
      "Optional path to an already-fetched rapids-cmake source tree (skips download)"
)
if(rapids-cmake-dir)
  list(APPEND CMAKE_MODULE_PATH "${rapids-cmake-dir}/rapids-cmake")
else()
  set(rapids-cmake-version "24.12")
  include(FetchContent)
  FetchContent_Declare(
    rapids-cmake
    GIT_REPOSITORY https://github.com/rapidsai/rapids-cmake.git
    GIT_TAG "branch-${rapids-cmake-version}"
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(rapids-cmake)
  list(APPEND CMAKE_MODULE_PATH "${rapids-cmake_SOURCE_DIR}/rapids-cmake")
endif()

execute_process(
  COMMAND
    "${Python_EXECUTABLE}" -c
    "import pylibcudf, pathlib; print(pathlib.Path(pylibcudf.__file__).parent.parent)"
  OUTPUT_VARIABLE PYLIBCUDF_SOURCE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)

include(rapids-cython-core)
rapids_cython_init()

set(CYTHON_FLAGS
    "${CYTHON_FLAGS} -I${PYLIBCUDF_SOURCE_DIR}"
    CACHE STRING "" FORCE)
message(STATUS "cucascade: pylibcudf source dir: ${PYLIBCUDF_SOURCE_DIR}")

add_subdirectory(cucascade)
