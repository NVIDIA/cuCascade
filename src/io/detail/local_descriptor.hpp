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
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <string_view>
#include <system_error>

namespace cucascade::io::detail {

/// Test seam: a hook run by @ref reopen_local_descriptor just before it
/// reopens; with @c bypass_procfs it then takes the pathname fallback as if
/// procfs were not mounted.  Lets a test rename the path at exactly that point.
struct reopen_hook_for_testing {
  void (*hook)(void*){nullptr};
  void* arg{nullptr};
  bool bypass_procfs{false};
};

/// Install @p hook process-wide (a default-constructed one removes it).  Tests only.
void set_reopen_hook_for_testing(reopen_hook_for_testing hook) noexcept;

/// The installed hook, or null when none is.
[[nodiscard]] reopen_hook_for_testing const* reopen_hook_for_testing_installed() noexcept;

/// Open the file that is open as @p fd again, with @p flags (a local reactor
/// adds @c O_DIRECT this way).  Goes through @c /proc/self/fd so a concurrent
/// rename of @p path cannot pair @p fd with a different file; only when that
/// entry does not exist (@c ENOENT: procfs is not mounted) does it fall back to
/// opening @p path.  Either way the caller must compare the two descriptors'
/// device and inode before trusting the pair.  Returns the new descriptor
/// (owned by the caller), or -1 with @c errno set.
[[nodiscard]] inline int reopen_local_descriptor(int fd, char const* path, int flags) noexcept
{
  bool bypass_procfs = false;
  if (auto const* seam = reopen_hook_for_testing_installed(); seam != nullptr) {
    seam->hook(seam->arg);
    bypass_procfs = seam->bypass_procfs;
  }
  if (bypass_procfs) return ::open(path, flags);
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

/// Whether descriptors @p first and @p second name the same file (device and
/// inode).  False when either cannot be stat'ed.
[[nodiscard]] inline bool same_local_file(int first, int second) noexcept
{
  struct stat a{};
  struct stat b{};
  if (::fstat(first, &a) != 0 || ::fstat(second, &b) != 0) return false;
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino;
}

/**
 * @brief Open the O_DIRECT companion of a writable descriptor @p buffered.
 *
 * A write open creates or truncates @p path through @p buffered first; that
 * operation is destructive and is never retried.  The O_DIRECT descriptor is
 * then anchored to @p buffered with @ref reopen_local_descriptor, so a
 * concurrent rename of @p path cannot pair the two with different files.  When
 * the reopen had to fall back to the pathname and the result does not provably
 * name the same file, the open fails closed with @c EXDEV (the caller reports a
 * descriptor mismatch) rather than write half of the bytes elsewhere.
 *
 * @return The descriptor (owned by the caller); -1 with @c errno set when the
 *         reopen failed (e.g. @c EINVAL: no O_DIRECT here) or, with @c EXDEV,
 *         when the descriptors do not name the same file.
 */
[[nodiscard]] inline int reopen_writable_direct(int buffered, char const* path, int flags) noexcept
{
  int const direct = reopen_local_descriptor(buffered, path, flags);
  if (direct < 0) return direct;
  if (!same_local_file(buffered, direct)) {
    ::close(direct);
    errno = EXDEV;
    return -1;
  }
  return direct;
}

}  // namespace cucascade::io::detail
