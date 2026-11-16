/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Must NOT compile. Built only by the ctest case cucascade_log_pruned_site_checks_format, which
// asserts that a call site removed by the floor still has its format string checked against its
// arguments (CUCASCADE_LOG_NOOP evaluates detail::check_only in a never-taken branch).

#undef CUCASCADE_MIN_LOG_LEVEL
#define CUCASCADE_MIN_LOG_LEVEL 4  // level::error: the debug site below is pruned

#include <cucascade/log/logging.hpp>

void pruned_site_with_a_bad_format_string()
{
  CUCASCADE_LOG_DEBUG("{} {}", 1);  // two placeholders, one argument
}
