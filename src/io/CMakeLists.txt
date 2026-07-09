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

target_sources(
  cucascade_io_objects
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/uri_parser.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/io_context.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/datasource_factory.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/rest/curl_handle.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/rest/rest_ioctx.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/rest/rest_reactor.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/uring/uring_ioctx.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/uring/uring_reactor.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/rest/s3/sigv4.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/rest/s3/sigv4_authorizer.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/cache/types.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/cache/metadata_store.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/cache/prefetching_cache.cpp)
