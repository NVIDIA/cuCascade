# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

cmake_minimum_required(VERSION 3.26.4)

project(
  cucascade-python
  LANGUAGES CXX
  VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module
                                             Development.SABIModule)
find_package(Cython 3.0 REQUIRED)
# cuCascade's exported io target links kvikio::kvikio, which its package config
# does not find.
find_package(kvikio REQUIRED CONFIG)
# Built against an installed cuCascade that has the io and cudf libraries.
find_package(cuCascade REQUIRED COMPONENTS cudf io)
find_package(cudf REQUIRED CONFIG)

# pylibcudf's .pxd files, which the extension cimports.
execute_process(
  COMMAND
    ${Python_EXECUTABLE} -c
    "import os, pylibcudf; print(os.path.dirname(os.path.dirname(pylibcudf.__file__)))"
  OUTPUT_VARIABLE PYLIBCUDF_PARENT_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)

include(UseCython)
cython_transpile(
  cucascade/_io.pyx
  LANGUAGE
  CXX
  CYTHON_ARGS
  -I
  ${PYLIBCUDF_PARENT_DIR}
  OUTPUT_VARIABLE
  io_source)

# pylibcudf is built against Python's limited API, and this extension subclasses
# its Datasource, so it must be built the same way.
python_add_library(
  _io
  MODULE
  USE_SABI
  3.10
  WITH_SOABI
  "${io_source}"
  src/session.cpp)
target_compile_definitions(_io PRIVATE CYTHON_LIMITED_API=1
                                       Py_LIMITED_API=0x030A0000)
# The installed cuCascade io headers include moodycamel's concurrentqueue, which
# it does not ship.
find_path(CUCASCADE_MOODYCAMEL_HEADER_DIR NAMES blockingconcurrentqueue.h
                                                REQUIRED)
target_include_directories(_io PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
                                       ${CUCASCADE_MOODYCAMEL_HEADER_DIR})
target_link_libraries(
  _io PRIVATE cuCascade::cucascade_cudf cuCascade::cucascade_io
              cuCascade::cucascade cudf::cudf)
# Find the libraries it linked against (cuCascade, libcudf, ...) without
# LD_LIBRARY_PATH. cuCascade_DIR is <prefix>/lib/cmake/cuCascade.
cmake_path(
  ABSOLUTE_PATH
  cuCascade_DIR
  BASE_DIRECTORY
  /
  NORMALIZE
  OUTPUT_VARIABLE
  cucascade_config_dir)
cmake_path(GET cucascade_config_dir PARENT_PATH cucascade_cmake_dir)
cmake_path(GET cucascade_cmake_dir PARENT_PATH cucascade_library_dir)
set_target_properties(
  _io PROPERTIES INSTALL_RPATH "$ORIGIN;${cucascade_library_dir}"
                 INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS _io DESTINATION cucascade)
