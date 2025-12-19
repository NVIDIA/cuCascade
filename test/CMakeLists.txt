# =============================================================================
# Copyright 2025, cuCascade Contributors.
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

# Fetch Catch2 for testing
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v2.13.10)
FetchContent_MakeAvailable(Catch2)

# Set include directories for tests
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)

# Collect all test sources directly (CMake PARENT_SCOPE has scoping issues)
set(TEST_SOURCES
    # Utils
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/cudf_test_utils.cpp
    # Memory tests
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/test_memory_reservation_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/test_topology_discovery.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/test_gpu_kernels.cu
    # Data tests
    ${CMAKE_CURRENT_SOURCE_DIR}/data/test_data_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/test_data_repository.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/test_data_repository_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/test_data_representation.cpp
    # Main test runner
    ${CMAKE_CURRENT_SOURCE_DIR}/unittest.cpp)

# Create test executable
add_executable(cucascade_tests ${TEST_SOURCES})

# Set include directories
target_include_directories(
  cucascade_tests
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/utils
          ${Catch2_SOURCE_DIR}/single_include)

# Link dependencies
target_link_libraries(cucascade_tests PRIVATE cucascade Catch2::Catch2
                                              CUDA::cudart)

# Register tests with CTest
add_test(NAME cucascade_tests COMMAND cucascade_tests)
