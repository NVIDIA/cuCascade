# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

if(NOT DEFINED BINARY OR NOT EXISTS "${BINARY}")
  message(FATAL_ERROR "BINARY must name an existing ELF file")
endif()

if(NOT DEFINED READELF OR READELF STREQUAL "")
  message(
    FATAL_ERROR "READELF must name the tool used to inspect ELF dependencies")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env LC_ALL=C "${READELF}" -d "${BINARY}"
  RESULT_VARIABLE readelf_result
  OUTPUT_VARIABLE dynamic_section
  ERROR_VARIABLE readelf_error)

if(NOT readelf_result EQUAL 0)
  message(FATAL_ERROR "Failed to inspect ${BINARY}: ${readelf_error}")
endif()

if(dynamic_section MATCHES "\\(NEEDED\\).*\\[libcuda\\.so(\\.[0-9]+)*\\]")
  message(
    FATAL_ERROR "${BINARY} has an unexpected runtime dependency on libcuda")
endif()
