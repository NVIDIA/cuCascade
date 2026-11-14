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
 * @brief Internals of GPUDirect Storage setup discovery, exposed for its unit
 *        tests only.  Not part of the supported API.
 */

#include <cucascade/memory/gds_topology.hpp>

#include <string>
#include <string_view>
#include <vector>

namespace cucascade::memory::detail {

/**
 * @brief Parse the text of a cuFile configuration file (JSON with @c // and
 *        @c /&lowast; &lowast;/ comments; trailing commas are tolerated) into
 *        @p out.
 *
 * Keys cuCascade does not know are ignored; a known key holding a value of the
 * wrong type is ignored too (cuFile would reject or default it).
 *
 * @return Whether the text parsed; on failure @p error says where and why and
 *         @p out is left unchanged.
 */
[[nodiscard]] bool parse_cufile_json(std::string_view text,
                                     cufile_json_settings& out,
                                     std::string& error) noexcept;

/// Fill the stats-file fields of @p out (GDS version, driver version when not
/// already known, PeerDirect support, I/O stats state, per-GPU lines) from the
/// text of @c /proc/driver/nvidia-fs/stats.
void parse_nvidia_fs_stats(std::string_view text, nvidia_fs_info& out);

/// The driver version in the text of @c /proc/driver/nvidia-fs/version or
/// @c /sys/module/nvidia_fs/version ("version: 2.20.5" or "2.20.5"); empty if none.
[[nodiscard]] std::string parse_nvidia_fs_version(std::string_view text);

/// Names of the modules @c /proc/modules lists in the @c Live state.
[[nodiscard]] std::vector<std::string> live_modules(std::string_view proc_modules);

/// Whether a @c CUFILE_FORCE_COMPAT_MODE value forces compatibility mode the way
/// cuFile reads it: its first four characters are "true", any case.
[[nodiscard]] bool force_compat_value(char const* value) noexcept;

}  // namespace cucascade::memory::detail
