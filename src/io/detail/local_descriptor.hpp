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

#include <fcntl.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <string_view>
#include <system_error>

namespace cucascade::io::detail {

/// Open the file that is open as @p fd again, with @p flags (a local reactor
/// adds @c O_DIRECT this way).  Goes through @c /proc/self/fd so a concurrent
/// rename of @p path cannot pair @p fd with a different file; only when that
/// entry does not exist (@c ENOENT: procfs is not mounted) does it fall back to
/// opening @p path.  Either way the caller must compare the two descriptors'
/// device and inode before trusting the pair.  Returns the new descriptor
/// (owned by the caller), or -1 with @c errno set.
[[nodiscard]] inline int reopen_local_descriptor(int fd, char const* path, int flags) noexcept
{
  constexpr std::string_view prefix = "/proc/self/fd/";
  std::array<char, prefix.size() + 16> proc_path{};
  char* const out      = std::copy(prefix.begin(), prefix.end(), proc_path.data());
  auto const [end, ec] = std::to_chars(out, proc_path.data() + proc_path.size() - 1, fd);
  if (ec == std::errc{}) {
    *end             = '\0';
    int const direct = ::open(proc_path.data(), flags);
    if (direct >= 0 || errno != ENOENT) return direct;
  }
  return ::open(path, flags);
}

}  // namespace cucascade::io::detail
