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

target_sources(
  cucascade_objects
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/disk_access_limiter.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/fixed_size_host_memory_resource.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/memory_reservation.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/memory_reservation_manager.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/memory_space.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/null_device_memory_resource.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/reservation_aware_resource_adaptor.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/reservation_manager_configurator.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/stream_pool.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/topology_discovery.cpp)
