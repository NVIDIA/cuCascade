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

#include <sys/stat.h>

#include <cerrno>
#include <cstdint>
#include <string>
#include <string_view>

namespace cucascade::io {

/// What an @ref object_identity was built from.  The backend chooses its
/// evidence:
///  - @c opened_handle: no reusable evidence; the identity is scoped to one
///    opened object (its @c version is that object's open generation), so it
///    never authorizes reuse through a separately opened handle.
///  - @c local_stat: device, inode, size and ctime of an opened local file.
///  - @c object_tag: a strong validator of a remote object (its ETag).
enum class identity_kind : std::uint8_t { opened_handle, local_stat, object_tag };

/// Evidence that two opens of a path see the same version of the object.  Two
/// identities are equal only when every field is: a caller may reuse what it
/// derived from one open (cached bytes, parsed metadata) through another open
/// with an equal identity.
struct object_identity {
  identity_kind kind = identity_kind::opened_handle;
  std::uint64_t size = 0;
  std::string version;
  std::uint64_t device   = 0;
  std::uint64_t inode    = 0;
  std::uint64_t ctime_ns = 0;

  bool operator==(object_identity const& other) const noexcept = default;

  /// A key naming this identity of the object at @p locator.  The length
  /// prefix makes arbitrary paths and tags unambiguous; an ETag is never parsed.
  [[nodiscard]] std::string cache_key(std::string_view locator) const
  {
    std::string key = std::to_string(locator.size());
    key.push_back(':');
    key.append(locator);
    key.push_back(':');
    key.append(std::to_string(static_cast<unsigned>(kind)));
    key.push_back(':');
    key.append(std::to_string(size));
    key.push_back(':');
    key.append(std::to_string(device));
    key.push_back(':');
    key.append(std::to_string(inode));
    key.push_back(':');
    key.append(std::to_string(ctime_ns));
    key.push_back(':');
    key.append(version);
    return key;
  }
};

/// A process-wide, strictly increasing number for each opened object (never 0):
/// it orders the publications of different opens of the same path.  One
/// counter per process: it is defined in the cucascade_io library.
[[nodiscard]] std::uint64_t next_open_generation() noexcept;

/// Capture the @c local_stat identity of the file open as @p fd into @p result.
/// Reads the open descriptor, never the path: a second path lookup could stat a
/// replacement.  Returns false (leaving @p result unchanged, @c errno set) when
/// @p fd cannot be stat'ed; callers then fall back to open-local identity.
[[nodiscard]] inline bool capture_local_identity(int fd, object_identity& result) noexcept
{
  struct stat info{};
  if (fd < 0) {
    errno = EBADF;
    return false;
  }
  if (::fstat(fd, &info) != 0) return false;
  if (info.st_size < 0) {
    errno = EOVERFLOW;
    return false;
  }
  result.kind     = identity_kind::local_stat;
  result.size     = static_cast<std::uint64_t>(info.st_size);
  result.device   = static_cast<std::uint64_t>(info.st_dev);
  result.inode    = static_cast<std::uint64_t>(info.st_ino);
  result.ctime_ns = static_cast<std::uint64_t>(info.st_ctim.tv_sec) * 1'000'000'000ULL +
                    static_cast<std::uint64_t>(info.st_ctim.tv_nsec);
  // ctime_ns is already part of the structured identity; do not duplicate it in
  // the free-form version field used by object-store validators.
  result.version.clear();
  return true;
}

}  // namespace cucascade::io
