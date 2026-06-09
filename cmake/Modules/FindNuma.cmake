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

#[=======================================================================[.rst:
FindNuma
--------

Finds libnuma (the NUMA policy library, provided by ``numactl-devel`` or
``libnuma-dev`` depending on the package manager).

This module is bundled with the installed cuCascade CMake package so that the
same find logic runs for downstream consumers, locating libnuma at whatever
path it lives on the consumer's system rather than embedding the build
machine's absolute path into the exported targets.

Imported Targets
^^^^^^^^^^^^^^^^^

``Numa::Numa``
  The libnuma library, if found. Carries the include directory and library
  location as usage requirements.

Result Variables
^^^^^^^^^^^^^^^^^

``Numa_FOUND``
  True if libnuma was found.
``Numa_INCLUDE_DIRS``
  Include directories needed to use libnuma.
``Numa_LIBRARIES``
  Libraries needed to link against libnuma.

#]=======================================================================]

find_path(Numa_INCLUDE_DIR NAMES numa.h)
find_library(Numa_LIBRARY NAMES numa)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Numa REQUIRED_VARS Numa_LIBRARY Numa_INCLUDE_DIR)

if(Numa_FOUND)
  set(Numa_LIBRARIES ${Numa_LIBRARY})
  set(Numa_INCLUDE_DIRS ${Numa_INCLUDE_DIR})

  if(NOT TARGET Numa::Numa)
    add_library(Numa::Numa UNKNOWN IMPORTED)
    set_target_properties(
      Numa::Numa
      PROPERTIES IMPORTED_LOCATION "${Numa_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${Numa_INCLUDE_DIR}")
  endif()
endif()

mark_as_advanced(Numa_INCLUDE_DIR Numa_LIBRARY)
