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

#include "io/churning_file.hpp"
#include "io/detail/local_descriptor.hpp"

#include <cucascade/io/object_identity.hpp>
#include <cucascade/io/rest/rest_reactor.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>

#include <catch2/catch_all.hpp>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
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
class plain_object : public io_object {
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

class strong_object final : public plain_object {
 public:
  using plain_object::plain_object;
  [[nodiscard]] bool validation_tag_is_strong() const noexcept override { return true; }
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

/// The @c local_stat identity of the file open as @p fd.
object_identity identity_of(int fd)
{
  object_identity identity;
  REQUIRE(capture_local_identity(fd, identity));
  return identity;
}

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

TEST_CASE("Reopening a local descriptor reopens the open file, not its path", "[object_identity]")
{
  using cucascade::io::detail::reopen_local_descriptor;
  identity_file const file{4096, 'a'};
  file_descriptor const original{::open(file.path().c_str(), O_RDONLY | O_CLOEXEC)};
  REQUIRE(original);
  auto const held = identity_of(original.get());
  // Another file now owns the path.  It was created while the original still
  // existed, so the two inodes differ.
  file.replace('b');
  object_identity replacement;
  {
    file_descriptor const by_path{::open(file.path().c_str(), O_RDONLY | O_CLOEXEC)};
    REQUIRE(by_path);
    replacement = identity_of(by_path.get());
  }
  REQUIRE(replacement.inode != held.inode);

  SECTION("an open descriptor is reopened through /proc/self/fd")
  {
    file_descriptor const reopened{
      reopen_local_descriptor(original.get(), file.path().c_str(), O_RDONLY | O_CLOEXEC)};
    REQUIRE(reopened);
    auto const identity = identity_of(reopened.get());
    CHECK(identity.device == held.device);
    CHECK(identity.inode == held.inode);
    char byte = 0;
    REQUIRE(::pread(reopened.get(), &byte, 1, 0) == 1);
    CHECK(byte == 'a');
  }
  SECTION("only a descriptor without a /proc/self/fd entry falls back to the path")
  {
    // No /proc/self/fd entry (ENOENT), as on a host without procfs.  The
    // caller's dev/inode cross-check is what catches a replaced path then.
    file_descriptor const reopened{
      reopen_local_descriptor(-1, file.path().c_str(), O_RDONLY | O_CLOEXEC)};
    REQUIRE(reopened);
    CHECK(identity_of(reopened.get()).inode == replacement.inode);
  }
  SECTION("any other failure is reported, never retried on the path")
  {
    // A directory cannot be reopened for writing (EISDIR); the path, a regular
    // file, could be.
    auto const directory_path = std::filesystem::temp_directory_path();
    file_descriptor const directory{
      ::open(directory_path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC)};
    REQUIRE(directory);
    errno = 0;
    file_descriptor const reopened{
      reopen_local_descriptor(directory.get(), file.path().c_str(), O_WRONLY | O_CLOEXEC)};
    int const error = errno;
    CHECK_FALSE(reopened);
    CHECK(error == EISDIR);
  }
}

TEST_CASE("REST identity only reuses strong ETags", "[object_identity]")
{
  using cucascade::io::rest::rest_io_object;

  rest_io_object weak("s3://bucket/key", "bucket", "key", 3, "W/\"v1\"");
  CHECK(weak.identity().kind == identity_kind::opened_handle);
  CHECK(weak.validation_tag() == "W/\"v1\"");
  CHECK_FALSE(weak.validation_tag_is_strong());

  rest_io_object invalid("s3://bucket/key", "bucket", "key", 3, "v1");
  CHECK(invalid.identity().kind == identity_kind::opened_handle);
  CHECK_FALSE(invalid.validation_tag_is_strong());

  rest_io_object absent("s3://bucket/key", "bucket", "key", 3);
  CHECK(absent.identity().kind == identity_kind::opened_handle);

  rest_io_object strong("s3://bucket/key", "bucket", "key", 3, "\"v1\"");
  CHECK(strong.identity().kind == identity_kind::object_tag);
  CHECK(strong.validation_tag_is_strong());
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

TEST_CASE("Default identity ignores a validation tag the backend has not declared strong",
          "[object_identity]")
{
  for (auto const* tag : {"W/\"v1\"", "\"t\""}) {
    plain_object const a("p", tag), b("p", tag);
    CHECK(a.identity().kind == identity_kind::opened_handle);
    CHECK(a.identity() != b.identity());
    CHECK(a.identity_cache_key() != b.identity_cache_key());
  }

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

TEST_CASE("A backend that declares its tag strong shares an object_tag identity",
          "[object_identity]")
{
  strong_object const a("file", "\"t\""), b("file", "\"t\"");
  CHECK(a.identity().kind == identity_kind::object_tag);
  CHECK(a.identity().version == "\"t\"");
  CHECK(a.identity().size == 7);
  CHECK(a.identity() == b.identity());
  CHECK(a.identity_cache_key() == b.identity_cache_key());
  strong_object const empty("file", "");
  CHECK(empty.identity().kind == identity_kind::opened_handle);
}

TEST_CASE("uring opens carry the stat identity of their descriptors", "[object_identity][uring]")
{
  using cucascade::io::uring::uring_reactor;
  identity_file const file{12345};

  auto const first    = uring_reactor::create_io_object(file.path());
  auto const identity = first->identity();
  CHECK(identity.kind == identity_kind::local_stat);
  CHECK(identity.size == file.bytes());
  CHECK(first->size() == file.bytes());
  struct stat info{};
  REQUIRE(::stat(file.path().c_str(), &info) == 0);
  CHECK(identity.device == static_cast<std::uint64_t>(info.st_dev));
  CHECK(identity.inode == static_cast<std::uint64_t>(info.st_ino));
  // The O_DIRECT descriptor, when the filesystem has one, names the same file.
  if (first->fd_direct() >= 0) {
    object_identity direct;
    REQUIRE(capture_local_identity(first->fd_direct(), direct));
    CHECK(direct == identity);
  }

  // A second open of the unchanged file is the same version: equal identity and key.
  auto const second = uring_reactor::create_io_object(file.path());
  CHECK(second->open_generation() > first->open_generation());
  CHECK(second->identity() == identity);
  CHECK(second->identity_cache_key() == first->identity_cache_key());

  // An atomic replacement of the same size is a new version; the object opened
  // before it still describes the file it holds.
  file.replace('b');
  auto const third = uring_reactor::create_io_object(file.path());
  CHECK(third->identity().kind == identity_kind::local_stat);
  CHECK(third->identity() != identity);
  CHECK(third->identity_cache_key() != first->identity_cache_key());
  CHECK(first->identity() == identity);
}

TEST_CASE("uring unstatable objects have no reusable identity", "[object_identity][uring]")
{
  using cucascade::io::uring::local_io_object;

  // Without a descriptor to stat, an object keeps its given size and falls back
  // to the open-local identity.
  local_io_object const detached("/dev/null", file_descriptor{}, file_descriptor{}, 1);
  CHECK(detached.identity().kind == identity_kind::opened_handle);
  CHECK(detached.size() == 1);
}

TEST_CASE("uring opens of a path being replaced report the file they opened",
          "[object_identity][uring]")
{
  // Every open races an atomic replacement of its path.  The object's
  // identity and size must describe the file its buffered descriptor holds,
  // its O_DIRECT descriptor must hold that file too, and reopening the
  // descriptor (not the path) means no open ever sees a mismatch to retry or
  // throw on.
  using cucascade::io::uring::uring_reactor;
  cucascade::test::churning_file const file;
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  for (int opens = 0;
       opens < 200 || (file.replacements() < 200 && std::chrono::steady_clock::now() < deadline);
       ++opens) {
    auto const object = uring_reactor::create_io_object(file.path());
    cucascade::test::churning_file::check_opened(
      object->identity(), object->size(), object->fd(), object->fd_direct());
  }
}
