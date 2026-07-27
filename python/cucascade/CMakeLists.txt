# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources datasource.pyx)
set(linked_libraries cuCascade::cucascade_cudf cudf::cudf)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}"
  MODULE_PREFIX cucascade_
  ASSOCIATED_TARGETS cucascade_cudf_shared
)

get_target_property(_cudf_loc cudf::cudf LOCATION)
get_filename_component(_cudf_lib_dir "${_cudf_loc}" DIRECTORY)
set_target_properties(cucascade_datasource
  PROPERTIES INSTALL_RPATH "${_cudf_lib_dir}"
)

set(concurrentqueue_dir
    ""
    CACHE PATH "Path to the moodycamel concurrentqueue source directory"
)
if(concurrentqueue_dir)
  target_include_directories(cucascade_datasource PRIVATE "${concurrentqueue_dir}")
endif()
