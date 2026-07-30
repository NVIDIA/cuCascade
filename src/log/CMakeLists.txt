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

# Only the built-in stderr sink lives here. The logging machinery itself (level
# check, sink slot, formatting) stays header-only in
# include/cucascade/log/logging.hpp so that logging never imposes a link
# dependency -- including on the topology-only build, which does not produce
# cucascade_objects and therefore has no stderr sink available.
if(TARGET cucascade_objects)
  target_sources(cucascade_objects
                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/stderr_sink.cpp)
endif()
