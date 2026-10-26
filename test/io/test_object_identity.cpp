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

#include <cucascade/io/object_identity.hpp>
#include <cucascade/io/rest/rest_reactor.hpp>
#include <cucascade/io/types.hpp>

#include <catch2/catch_all.hpp>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

using cucascade::io::capture_local_identity;
using cucascade::io::file_descriptor;
using cucascade::io::identity_kind;
using cucascade::io::io_object;
using cucascade::io::next_open_generation;
using cucascade::io::object_identity;

namespace {

/// An io_object with only the base class's identity: a validation tag, no
/// backend override.
class plain_object final : public io_object {
 public:
  plain_object(std::string path, std::string tag) : _path(std::move(path)), _tag(std::move(tag)) {}

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return 7; }
  [[nodiscard]] std::string_view validation_tag() const noexcept override { return _tag; }

 private:
  std::string _path;
  std::string _tag;
};

/// A temporary file holding @p bytes copies of @p fill, removed on scope exit;
/// @ref replace atomically renames a new file of the same size over it (with
/// the access and modification @p times given, if any).
class identity_file {
 public:
  explicit identity_file(std::size_t bytes, char fill = 'a')
    : _path(std::filesystem::temp_directory_path() /
            ("cucascade_identity_" + std::to_string(::getpid()) + "_" +
             std::to_string(next_id.fetch_add(1)) + ".bin")),
      _bytes(bytes)
  {
    write(_path, fill);
  }

  ~identity_file()
  {
    std::error_code ec;
    std::filesystem::remove(_path, ec);
    std::filesystem::remove(staged(), ec);
  }

  identity_file(identity_file const&)            = delete;
  identity_file& operator=(identity_file const&) = delete;

  void replace(char fill, timespec const* times = nullptr) const
  {
    auto const next = staged();
    write(next, fill);
    if (times != nullptr) { REQUIRE(::utimensat(AT_FDCWD, next.c_str(), times, 0) == 0); }
    std::filesystem::rename(next, _path);
  }

  [[nodiscard]] std::string path() const { return _path.string(); }
  [[nodiscard]] std::size_t bytes() const noexcept { return _bytes; }

 private:
  void write(std::filesystem::path const& target, char fill) const
  {
    std::string const data(_bytes, fill);
    std::ofstream out(target, std::ios::binary);
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    out.close();
    REQUIRE(out);
  }

  [[nodiscard]] std::filesystem::path staged() const
  {
    auto next = _path;
    next += ".next";
    return next;
  }

  static inline std::atomic<int> next_id{0};
  std::filesystem::path _path;
  std::size_t _bytes;
};

}  // namespace

TEST_CASE("Object identity separates version kinds and unambiguous locators", "[object_identity]")
{
  object_identity a{identity_kind::object_tag, 100, "tag"};
  auto b = a;
  CHECK(a == b);
  b.version = "new";
  CHECK(a.cache_key("file") != b.cache_key("file"));
  b = a;
  b.size++;
  CHECK(a != b);
  b      = a;
  b.kind = identity_kind::local_stat;
  CHECK(a != b);
  CHECK(a.cache_key("file:1") != a.cache_key("file"));
  CHECK(next_open_generation() != next_open_generation());
}

TEST_CASE("Local identity separates replacement with preserved mtime", "[object_identity]")
{
  // Owned by RAII from the start, in the platform's temporary directory.
  identity_file const file{3, 'a'};
  timespec const times[2]{{123, 456}, {123, 456}};
  object_identity first;
  {
    file_descriptor const first_fd{::open(file.path().c_str(), O_RDWR | O_CLOEXEC)};
    REQUIRE(first_fd);
    REQUIRE(::futimens(first_fd.get(), times) == 0);
    REQUIRE(capture_local_identity(first_fd.get(), first));
  }
  file.replace('x', times);

  file_descriptor const owned{::open(file.path().c_str(), O_RDONLY | O_CLOEXEC)};
  REQUIRE(owned);
  int const fd = owned.get();
  object_identity replaced;
  REQUIRE(capture_local_identity(fd, replaced));
  CHECK(replaced.kind == identity_kind::local_stat);
  CHECK(replaced.size == first.size);
  CHECK(replaced != first);
  CHECK((replaced.inode != first.inode || replaced.ctime_ns != first.ctime_ns));
  REQUIRE(::unlink(file.path().c_str()) == 0);
  object_identity held;
  REQUIRE(capture_local_identity(fd, held));
  // Removing a directory entry can change ctime as the link count drops. The open descriptor
  // still holds the same object, but its captured version need not match the pre-unlink one.
  CHECK(held.device == replaced.device);
  CHECK(held.inode == replaced.inode);
  CHECK(held.size == replaced.size);
  // The local reactors reopen the owned descriptor for O_DIRECT, so pathname
  // removal cannot switch the data descriptor to another object.
  auto proc_path = "/proc/self/fd/" + std::to_string(fd);
  file_descriptor const reopened{::open(proc_path.c_str(), O_RDONLY | O_CLOEXEC)};
  REQUIRE(reopened);
  object_identity reopened_identity;
  CHECK(capture_local_identity(reopened.get(), reopened_identity));
  CHECK(reopened_identity == held);
  errno = 0;
  CHECK_FALSE(capture_local_identity(-1, held));
  CHECK(errno == EBADF);
}

TEST_CASE("REST identity only reuses strong ETags", "[object_identity]")
{
  using cucascade::io::rest::rest_io_object;

  rest_io_object weak("s3://bucket/key", "bucket", "key", 3, "W/\"v1\"");
  CHECK(weak.identity().kind == identity_kind::opened_handle);
  CHECK(weak.validation_tag() == "W/\"v1\"");

  rest_io_object invalid("s3://bucket/key", "bucket", "key", 3, "v1");
  CHECK(invalid.identity().kind == identity_kind::opened_handle);

  rest_io_object absent("s3://bucket/key", "bucket", "key", 3);
  CHECK(absent.identity().kind == identity_kind::opened_handle);

  rest_io_object strong("s3://bucket/key", "bucket", "key", 3, "\"v1\"");
  CHECK(strong.identity().kind == identity_kind::object_tag);
  CHECK(strong.identity().version == "\"v1\"");
  CHECK(strong.identity().size == 3);

  // A strong tag is reusable across opens; an open without one is not, even
  // with the same (weak) tag.
  rest_io_object strong_again("s3://bucket/key", "bucket", "key", 3, "\"v1\"");
  CHECK(strong_again.identity() == strong.identity());
  CHECK(strong_again.identity_cache_key() == strong.identity_cache_key());
  rest_io_object weak_again("s3://bucket/key", "bucket", "key", 3, "W/\"v1\"");
  CHECK(weak_again.identity() != weak.identity());
  CHECK(weak_again.identity_cache_key() != weak.identity_cache_key());

  // A new version of the key is a new identity.
  rest_io_object next("s3://bucket/key", "bucket", "key", 3, "\"v2\"");
  CHECK(next.identity() != strong.identity());
  CHECK(next.identity_cache_key() != strong.identity_cache_key());
}

TEST_CASE("Default identity is a tag or scoped to the open", "[object_identity]")
{
  plain_object const tagged("file", "\"t\"");
  auto const tag_identity = tagged.identity();
  CHECK(tag_identity.kind == identity_kind::object_tag);
  CHECK(tag_identity.version == "\"t\"");
  CHECK(tag_identity.size == 7);
  CHECK(plain_object("file", "\"t\"").identity() == tag_identity);

  plain_object const untagged("file", "");
  plain_object const reopened("file", "");
  CHECK(untagged.identity().kind == identity_kind::opened_handle);
  CHECK(untagged.identity().version == std::to_string(untagged.open_generation()));
  CHECK(reopened.open_generation() > untagged.open_generation());
  CHECK(reopened.identity() != untagged.identity());
  CHECK(reopened.identity_cache_key() != untagged.identity_cache_key());
  // Stable for one open.
  CHECK(untagged.identity() == untagged.identity());
  CHECK(untagged.identity_cache_key() == untagged.identity_cache_key());
}
