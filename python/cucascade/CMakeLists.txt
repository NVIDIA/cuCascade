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

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module
                                             Development.SABIModule)
find_package(Cython 3.0 REQUIRED)
include(UseCython)

# cuCascade's exported io target links kvikio::kvikio, which its package config
# does not find.
find_package(kvikio REQUIRED CONFIG)
# Built against an installed cuCascade that has the io and cudf libraries.
find_package(cuCascade REQUIRED COMPONENTS cudf io)
find_package(cudf REQUIRED CONFIG)

# The installed cuCascade io headers include moodycamel's concurrentqueue, which
# it does not ship.
find_path(CUCASCADE_MOODYCAMEL_HEADER_DIR NAMES blockingconcurrentqueue.h
                                                REQUIRED)

# cuCascade_DIR is <prefix>/lib/cmake/cuCascade.
cmake_path(
  ABSOLUTE_PATH
  cuCascade_DIR
  BASE_DIRECTORY
  /
  NORMALIZE
  OUTPUT_VARIABLE
  config_dir)
cmake_path(GET config_dir PARENT_PATH cmake_dir)
cmake_path(GET cmake_dir PARENT_PATH CUCASCADE_LIBRARY_DIR)

# pylibcudf's .pxd files, which the modules cimport.
execute_process(
  COMMAND
    ${Python_EXECUTABLE} -c
    "import os, pylibcudf; print(os.path.dirname(os.path.dirname(pylibcudf.__file__)))"
  OUTPUT_VARIABLE PYLIBCUDF_PARENT_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)

set(CUCASCADE_PYTHON_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
include(cmake/cucascade_cython.cmake)

set(CUCASCADE_PYTHON_LIBRARIES
    cuCascade::cucascade_cudf cuCascade::cucascade_io cuCascade::cucascade
    cudf::cudf)

add_subdirectory(cucascade)
