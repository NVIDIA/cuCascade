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

#pragma once

/**
 * @file
 * @brief Internals of the cuFile reactor exposed for its unit tests only.  Not
 *        part of the supported API; nothing outside the reactor and its tests
 *        should include this header.
 */

namespace cucascade::io::cufile {
struct cufile_api;
}  // namespace cucascade::io::cufile

namespace cucascade::io::cufile::detail {

/// Serve every cuFile call of the process from @p api instead of libcufile
/// (null restores libcufile), so tests can script reads, failures and short
/// returns.  Process-wide; install it before opening the objects it should
/// serve and remove it after they are closed, since handles registered by one
/// table mean nothing to the other.  @c cufile_api::open_driver() then reports
/// @p api's @c driver_open outcome.
void set_cufile_api_for_testing(cufile_api const* api) noexcept;

}  // namespace cucascade::io::cufile::detail
