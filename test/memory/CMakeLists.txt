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

set(TEST_SOURCES
    ${TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_memory_reservation_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_small_pinned_host_memory_resource.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_gpu_kernels.cu
    PARENT_SCOPE)

# Create topology discovery test executable
add_executable(cucascade_topology_discovery_tests test_topology_discovery.cpp
                                                  ../unittest.cpp)
set_target_properties(cucascade_topology_discovery_tests
                      PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

# Set include directories
target_include_directories(
  cucascade_tests
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../utils
          ${Catch2_SOURCE_DIR}/single_include)

# Link dependencies
target_link_libraries(cucascade_topology_discovery_tests
                      PRIVATE cucascade Catch2::Catch2 CUDA::cudart_static)

# Register tests with CTest
add_test(NAME cucascade_topology_discovery_tests
         COMMAND cucascade_topology_discovery_tests)
