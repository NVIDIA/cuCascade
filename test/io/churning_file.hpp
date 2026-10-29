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

#include <cucascade/io/object_identity.hpp>

#include <catch2/catch_all.hpp>
#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stop_token>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>

namespace cucascade::test {

/// A path that a background thread keeps replacing atomically, alternating
/// between two versions of different sizes and contents, so that opening the
/// path races a replacement.  Each version is a stable file: a replacement
/// hard-links it to a staging name and renames that over the path.
class churning_file {
 public:
  static constexpr std::array<std::size_t, 2> sizes{4096, 3 * 4096};
  static constexpr std::array<char, 2> fills{'a', 'b'};

  churning_file()
    : _dir(std::filesystem::temp_directory_path() /
           ("cucascade_churn_" + std::to_string(::getpid()) + "_" +
            std::to_string(next_id.fetch_add(1))))
  {
    std::filesystem::create_directory(_dir);
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      std::string const data(sizes[i], fills[i]);
      std::ofstream out(version(i), std::ios::binary);
      out.write(data.data(), static_cast<std::streamsize>(data.size()));
      out.close();
      REQUIRE(out);
    }
    std::filesystem::create_hard_link(version(0), _dir / "file");
    _thread = std::jthread([this](std::stop_token const& stop) { churn(stop); });
  }

  ~churning_file()
  {
    _thread.request_stop();
    _thread.join();
    std::error_code ec;
    std::filesystem::remove_all(_dir, ec);
  }

  churning_file(churning_file const&)            = delete;
  churning_file& operator=(churning_file const&) = delete;

  [[nodiscard]] std::string path() const { return (_dir / "file").string(); }

  /// Completed replacements so far.
  [[nodiscard]] std::size_t replacements() const noexcept { return _replacements.load(); }

  /// Check that what an object opened from this path reports -- its
  /// @p identity and @p size -- describes the version its buffered descriptor
  /// @p buffered holds, and that @p direct (when open) holds that version too.
  static void check_opened(cucascade::io::object_identity const& identity,
                           std::size_t size,
                           int buffered,
                           int direct)
  {
    using cucascade::io::capture_local_identity;
    using cucascade::io::identity_kind;
    using cucascade::io::object_identity;
    REQUIRE(identity.kind == identity_kind::local_stat);
    object_identity held;
    REQUIRE(capture_local_identity(buffered, held));
    // ctime is not compared: linking and renaming a version changes it.
    CHECK(identity.device == held.device);
    CHECK(identity.inode == held.inode);
    char first = 0;
    REQUIRE(::pread(buffered, &first, 1, 0) == 1);
    auto const held_version = first == fills[0] ? 0U : 1U;
    REQUIRE(first == fills[held_version]);
    CHECK(identity.size == sizes[held_version]);
    CHECK(size == sizes[held_version]);
    if (direct >= 0) {
      object_identity direct_identity;
      REQUIRE(capture_local_identity(direct, direct_identity));
      CHECK(direct_identity.device == identity.device);
      CHECK(direct_identity.inode == identity.inode);
    }
  }

 private:
  [[nodiscard]] std::filesystem::path version(std::size_t i) const
  {
    return _dir / ("v" + std::to_string(i));
  }

  void churn(std::stop_token const& stop)
  {
    auto const staged = (_dir / "staged").string();
    auto const target = path();
    for (std::size_t i = 1; !stop.stop_requested(); ++i) {
      // Consecutive versions differ, so each rename really replaces the path.
      auto const source = version(i % 2).string();
      if (::link(source.c_str(), staged.c_str()) != 0) { continue; }
      if (::rename(staged.c_str(), target.c_str()) == 0) { _replacements.fetch_add(1); }
      std::ignore = ::unlink(staged.c_str());  // left over only if the rename failed
    }
  }

  static inline std::atomic<int> next_id{0};
  std::filesystem::path _dir;
  std::atomic<std::size_t> _replacements{0};
  std::jthread _thread;
};

}  // namespace cucascade::test
