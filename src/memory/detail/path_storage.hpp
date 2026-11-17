/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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
 * @brief Internals of per-path storage discovery, exposed for its unit tests
 *        only (they substitute fixture procfs / sysfs roots).  Not part of the
 *        supported API.
 */

#include <cucascade/memory/topology_discovery.hpp>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cucascade::memory::detail {

/// Where discovery reads the kernel's views from.  Production uses the real
/// locations; tests point them at fixture files and directory trees.
struct path_storage_roots {
  std::string mountinfo{"/proc/self/mountinfo"};
  std::string sys_dev_block{"/sys/dev/block"};
};

/// One line of @c /proc/self/mountinfo, with octal escapes decoded.
struct mount_entry {
  int mount_id{-1};
  int parent_id{-1};
  unsigned int major{0};
  unsigned int minor{0};
  std::string root;
  std::string mount_point;
  std::string mount_options;
  std::string fs_type;
  std::string source;
  std::string super_options;
};

/// What discovery learned from the opened file itself.
struct opened_file_facts {
  std::string resolved_path;
  unsigned int major{0};
  unsigned int minor{0};
  std::optional<int> mount_id;
};

/// Decode the octal escapes (@c \\040 and friends) mountinfo uses in paths.
[[nodiscard]] std::string unescape_mountinfo(std::string_view field);

/// Parse mountinfo text; malformed lines are skipped.
[[nodiscard]] std::vector<mount_entry> parse_mountinfo(std::string_view text);

/// Whether @p mount_point is @p path or a path-component prefix of it.
[[nodiscard]] bool mount_point_covers(std::string_view mount_point, std::string_view path) noexcept;

/**
 * @brief The mount holding a file.
 *
 * The entry with @p facts.mount_id when known and present; otherwise the
 * longest covering mount of the file's device; only when no mount ID is known,
 * the longest covering mount of any device (sets @p fallback). The last entry
 * wins a tie: later mounts shadow earlier ones. Null when no mount covers the
 * path, or the known mount ID is absent and no mount of the file's device
 * covers it.
 */
[[nodiscard]] mount_entry const* find_mount(std::vector<mount_entry> const& mounts,
                                            opened_file_facts const& facts,
                                            bool& fallback) noexcept;

/// Classify a filesystem type that is not backed by a local block device:
/// @c network, @c memory, or @c unknown (block filesystems also map to unknown).
[[nodiscard]] storage_kind classify_filesystem(std::string_view fs_type) noexcept;

/**
 * @brief Fill @p info's mount and backing from @p facts and the views under
 *        @p roots.  @p info.status must already be @c ok (the file was opened);
 *        it is lowered to @c unknown when the mount or backing cannot be
 *        determined.
 */
void resolve_path_storage(path_storage_info& info,
                          opened_file_facts const& facts,
                          path_storage_roots const& roots);

}  // namespace cucascade::memory::detail
