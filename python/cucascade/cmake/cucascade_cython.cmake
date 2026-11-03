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

# Create one Python extension module for each Cython source file of the calling
# directory.
#
# Mirrors rapids_cython_create_modules, which pylibcudf uses: each .pyx becomes
# its own module, so that Cython modules can cimport one another, and the module
# is installed at the path of its directory relative to the package root.
#
# cucascade_create_cython_modules( SOURCE_FILES <pyx files...> MODULE_PREFIX
# <prefix for the target names> LINKED_LIBRARIES <libraries...>)
function(cucascade_create_cython_modules)
  cmake_parse_arguments(_CCM "" "MODULE_PREFIX" "SOURCE_FILES;LINKED_LIBRARIES"
                        ${ARGN})

  cmake_path(RELATIVE_PATH CMAKE_CURRENT_SOURCE_DIR BASE_DIRECTORY
             ${CUCASCADE_PYTHON_ROOT} OUTPUT_VARIABLE install_dir)

  foreach(source IN LISTS _CCM_SOURCE_FILES)
    cmake_path(GET source STEM module_name)
    set(target ${_CCM_MODULE_PREFIX}${module_name})

    cython_transpile(
      ${source}
      LANGUAGE
      CXX
      CYTHON_ARGS
      -I
      ${CUCASCADE_PYTHON_ROOT}
      -I
      ${PYLIBCUDF_PARENT_DIR}
      OUTPUT_VARIABLE
      cxx_source)

    # pylibcudf is built against Python's limited API, and these modules
    # subclass its types, so they must be built the same way.
    python_add_library(${target} MODULE USE_SABI 3.12 WITH_SOABI
                       "${cxx_source}")
    target_compile_definitions(${target} PRIVATE CYTHON_LIMITED_API=1
                                                 Py_LIMITED_API=0x030C0000)
    target_include_directories(${target}
                               PRIVATE ${CUCASCADE_MOODYCAMEL_HEADER_DIR})
    target_link_libraries(${target} PRIVATE ${_CCM_LINKED_LIBRARIES})
    set_target_properties(
      ${target}
      PROPERTIES OUTPUT_NAME ${module_name}
                 INSTALL_RPATH "$ORIGIN;${CUCASCADE_LIBRARY_DIR}"
                 INSTALL_RPATH_USE_LINK_PATH ON)
    install(TARGETS ${target} DESTINATION ${install_dir})
  endforeach()
endfunction()
