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

# Fetch Catch2 for testing
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.8.1
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(Catch2)

if(NOT CUCASCADE_TOPOLOGY_ONLY)
  # Core (cudf-free) test executable - links only the cudf-free cucascade core.
  add_executable(
    cucascade_tests
    # Memory tests
    memory/test_memory_reservation_manager.cpp
    memory/test_reservation_aware_resource_adaptor.cpp
    memory/test_reservation_manager_configurator.cpp
    memory/test_small_pinned_host_memory_resource.cpp
    memory/test_gpu_kernels.cu
    # Data tests
    data/test_data_repository.cpp
    data/test_data_repository_manager.cpp
    data/test_disk_io_backend.cpp
    data/test_io_worker.cpp
    # Main test runner
    unittest.cpp)
  set_target_properties(cucascade_tests PROPERTIES CUDA_STANDARD 20
                                                   CUDA_STANDARD_REQUIRED ON)

  # Set include directories
  target_include_directories(
    cucascade_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                            ${CMAKE_CURRENT_SOURCE_DIR}/utils)

  # Link dependencies
  target_link_libraries(cucascade_tests PRIVATE cucascade Catch2::Catch2
                                                CUDA::cudart_static)

  # Register tests with CTest
  add_test(NAME cucascade_tests COMMAND cucascade_tests)
endif()

if(NOT CUCASCADE_TOPOLOGY_ONLY AND CUCASCADE_BUILD_IO)
  # IO test executable - links the cudf-free cucascade-io datasource layer.
  add_executable(
    cucascade_io_tests
    io/test_datasource_registry.cpp
    io/test_dispatch_failure_hook.cpp
    io/test_uri_parser.cpp
    io/cache/test_metadata_store.cpp
    io/kvikio/test_kvikio_config.cpp
    io/rest/test_object_store_lister.cpp
    io/rest/test_rest_footer_resolve.cpp
    io/rest/test_rest_perf_snapshot.cpp
    io/rest/test_rest_validation_tag.cpp
    io/rest/test_shared_byte_span.cpp
    io/rest/s3/test_sigv4.cpp
    io/rest/s3/test_sigv4_authorizer.cpp
    io/rest/s3/test_static_credentials.cpp
    # Main test runner
    unittest.cpp)
  set_target_properties(cucascade_io_tests PROPERTIES CUDA_STANDARD 20
                                                      CUDA_STANDARD_REQUIRED ON)

  target_include_directories(
    cucascade_io_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/utils
            ${Catch2_SOURCE_DIR}/single_include)

  target_link_libraries(cucascade_io_tests PRIVATE cucascade_io Catch2::Catch2
                                                   CUDA::cudart_static)

  add_test(NAME cucascade_io_tests COMMAND cucascade_io_tests)
endif()

if(NOT CUCASCADE_TOPOLOGY_ONLY AND CUCASCADE_BUILD_CUDF)
  # cudf-coupled test executable - links the cucascade-cudf library.
  add_executable(
    cucascade_cudf_tests
    # Utils
    utils/cudf_test_utils.cpp
    # Data tests (cudf-backed representations / converters / profiler)
    data/test_bandwidth_profiler.cpp
    data/test_data_batch.cpp
    data/test_data_representation.cpp
    data/test_disk_host_converters.cpp
    data/test_gpu_disk_converters.cpp
    data/test_representation_converter.cpp
    cudf/test_release_table_stream.cpp
    # Main test runner
    unittest.cpp)
  set_target_properties(cucascade_cudf_tests
                        PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

  # Set include directories
  target_include_directories(
    cucascade_cudf_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/utils)

  # Link dependencies
  target_link_libraries(
    cucascade_cudf_tests PRIVATE cucascade_cudf Catch2::Catch2
                                 CUDA::cudart_static)

  # Register tests with CTest
  add_test(NAME cucascade_cudf_tests COMMAND cucascade_cudf_tests)
endif()

# Create topology discovery test executable
add_executable(
  cucascade_topology_discovery_tests memory/test_topology_discovery.cpp
                                     unittest_topology_discovery.cpp)
set_target_properties(cucascade_topology_discovery_tests
                      PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

# Set include directories
target_include_directories(
  cucascade_topology_discovery_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                             ${CMAKE_CURRENT_SOURCE_DIR}/utils)

# Link dependencies
target_link_libraries(cucascade_topology_discovery_tests
                      PRIVATE cucascade_topology_discovery Catch2::Catch2)

# Register tests with CTest
add_test(NAME cucascade_topology_discovery_tests
         COMMAND cucascade_topology_discovery_tests)
