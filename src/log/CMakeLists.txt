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

# Record delivery (vemit), sink publication and the built-in stderr sink. Only
# the level check and the sink slot stay inline in
# include/cucascade/log/logging.hpp, so the hot path remains header-only while
# <format>, <chrono> and <ctime> stay out of every translation unit that logs.
#
# Logging therefore requires linking cuCascade, and is unavailable in a
# topology-only build, which does not produce cucascade_objects.
if(TARGET cucascade_objects)
  target_sources(cucascade_objects
                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/logging.cpp)
endif()
