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

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

# cuFile parameter declarations vary between installed headers. Probe the header
# without referencing a function at link time or requiring libcufile.
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_INCLUDES "${CUCASCADE_CUFILE_INCLUDE_DIR}"
                            ${CUDAToolkit_INCLUDE_DIRS})
# These definitions describe implementation capabilities only. Public structs
# have unconditional ABI slots and never depend on these definitions.
set(CUCASCADE_CUFILE_FEATURE_DEFINITIONS "")

foreach(
  cucascade_cufile_function IN
  ITEMS cuFileSetParameterSizeT
        cuFileSetParameterBool
        cuFileSetParameterString
        cuFileGetParameterSizeT
        cuFileGetParameterBool
        cuFileGetParameterString
        cuFileGetParameterMinMaxValue
        cuFileSetParameterPosixPoolSlabArray
        cuFileGetParameterPosixPoolSlabArray
        cuFileSetParameterGpuBounceBufferSlabArray
        cuFileGetParameterGpuBounceBufferSlabArray)
  string(TOUPPER "${cucascade_cufile_function}" cucascade_cufile_feature)
  string(PREPEND cucascade_cufile_feature "CUCASCADE_CUFILE_HAS_")

  # Recheck on every configure, including header changes at the same path and
  # toolchain switches, rather than retaining stale cached probe results. This
  # intentionally costs 11 try-compiles per configure, even when nothing
  # changed.
  unset(${cucascade_cufile_feature} CACHE)
  unset(${cucascade_cufile_feature})
  check_cxx_source_compiles(
    "#include <cufile.h>
using function_type = decltype(&${cucascade_cufile_function});
int main() { return 0; }
"
    ${cucascade_cufile_feature})
  if(${cucascade_cufile_feature})
    list(APPEND CUCASCADE_CUFILE_FEATURE_DEFINITIONS
         "${cucascade_cufile_feature}")
  endif()
endforeach()

# Validate enum identifiers separately: a declared setter alone is insufficient.
foreach(
  cucascade_cufile_id IN
  ITEMS CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE
        CUFILE_PARAM_FORCE_COMPAT_MODE
        CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB
        CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB
        CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB
        CUFILE_PARAM_GPU_BOUNCE_BUFFER_SLAB_SIZE_KB
        CUFILE_PARAM_GPU_BOUNCE_BUFFER_SLAB_COUNT
        CUFILE_PARAM_PROFILE_STATS
        CUFILE_PARAM_LOGGING_LEVEL)
  set(cucascade_cufile_feature "CUCASCADE_CUFILE_HAS_${cucascade_cufile_id}")
  unset(${cucascade_cufile_feature} CACHE)
  check_cxx_source_compiles(
    "#include <cufile.h>
int main() { auto id = ${cucascade_cufile_id}; (void)id; return 0; }
" ${cucascade_cufile_feature})
  if(${cucascade_cufile_feature})
    list(APPEND CUCASCADE_CUFILE_FEATURE_DEFINITIONS
         "${cucascade_cufile_feature}")
  endif()
endforeach()
unset(cucascade_cufile_id)
cmake_pop_check_state()
unset(cucascade_cufile_function)
unset(cucascade_cufile_feature)
