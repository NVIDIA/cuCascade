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
#include <cucascade/io/uring/uring_reactor.hpp>

#include <catch2/catch_all.hpp>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
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
using cucascade::io::identity_kind;
using cucascade::io::io_object;
using cucascade::io::next_open_generation;
using cucascade::io::object_identity;

namespace {

/// An io_object with only the base class's identity: a validation tag and a
/// writable flag, no backend override.
class plain_object final : public io_object {
 public:
  plain_object(std::string path, std::string tag, bool writable = false)
    : _path(std::move(path)), _tag(std::move(tag)), _writable(writable)
  {
  }

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return 7; }
  [[nodiscard]] std::string_view validation_tag() const noexcept override { return _tag; }
  [[nodiscard]] bool writable() const noexcept override { return _writable; }

 private:
  std::string _path;
  std::string _tag;
  bool _writable;
};

/// A temporary file holding @p bytes copies of @p fill, removed on scope exit;
/// @ref replace atomically renames a new file of the same size over it.
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
  }

  identity_file(identity_file const&)            = delete;
  identity_file& operator=(identity_file const&) = delete;

  void replace(char fill) const
  {
    auto staged = _path;
    staged += ".next";
    write(staged, fill);
    std::filesystem::rename(staged, _path);
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
  char name[]             = "/tmp/cucascade-identity-XXXXXX";
  char replacement_name[] = "/tmp/cucascade-identity-replacement-XXXXXX";
  int first_fd            = ::mkstemp(name);
  REQUIRE(first_fd >= 0);
  REQUIRE(::write(first_fd, "abc", 3) == 3);
  timespec times[2]{{123, 456}, {123, 456}};
  REQUIRE(::futimens(first_fd, times) == 0);
  object_identity first;
  REQUIRE(capture_local_identity(first_fd, first));
  ::close(first_fd);

  int replacement_fd = ::mkstemp(replacement_name);
  REQUIRE(replacement_fd >= 0);
  REQUIRE(::write(replacement_fd, "xyz", 3) == 3);
  REQUIRE(::futimens(replacement_fd, times) == 0);
  ::close(replacement_fd);
  REQUIRE(::rename(replacement_name, name) == 0);

  int fd = ::open(name, O_RDONLY);
  REQUIRE(fd >= 0);
  struct cleanup {
    int fd;
    char* name;
    char* replacement;
    ~cleanup()
    {
      ::close(fd);
      ::unlink(name);
      ::unlink(replacement);
    }
  } owner{fd, name, replacement_name};
  object_identity replaced;
  REQUIRE(capture_local_identity(fd, replaced));
  CHECK(replaced.kind == identity_kind::local_stat);
  CHECK(replaced.size == first.size);
  CHECK(replaced != first);
  CHECK((replaced.inode != first.inode || replaced.ctime_ns != first.ctime_ns));
  REQUIRE(::unlink(name) == 0);
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
  int reopened   = ::open(proc_path.c_str(), O_RDONLY);
  REQUIRE(reopened >= 0);
  object_identity reopened_identity;
  bool captured = capture_local_identity(reopened, reopened_identity);
  ::close(reopened);
  CHECK(captured);
  CHECK(reopened_identity == held);
  CHECK_FALSE(capture_local_identity(-1, held));
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

TEST_CASE("A writable object never carries a reusable identity", "[object_identity]")
{
  plain_object const writer("file", "\"t\"", /*writable=*/true);
  auto const identity = writer.identity();
  CHECK(identity.kind == identity_kind::opened_handle);
  CHECK(identity.size == 0);
  CHECK(identity.version == std::to_string(writer.open_generation()));
  CHECK(identity != plain_object("file", "\"t\"").identity());
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

TEST_CASE("uring writable and unstatable objects have no reusable identity",
          "[object_identity][uring]")
{
  using cucascade::io::file_descriptor;
  using cucascade::io::uring::local_io_object;
  using cucascade::io::uring::uring_reactor;
  identity_file const file{4096};

  auto const writer = uring_reactor::create_io_object_for_write(
    file.path(), {.mode = cucascade::io::write_mode::open_existing});
  REQUIRE(writer->writable());
  CHECK(writer->identity().kind == identity_kind::opened_handle);
  CHECK(writer->identity().size == 0);
  CHECK(writer->size() == file.bytes());
  auto const writer_key = writer->identity_cache_key();
  writer->note_written(2 * file.bytes());
  CHECK(writer->size() == 2 * file.bytes());
  CHECK(writer->identity_cache_key() == writer_key);
  CHECK(writer->identity_cache_key() !=
        uring_reactor::create_io_object(file.path())->identity_cache_key());

  // Without a descriptor to stat, an object keeps its given size and falls back
  // to the open-local identity.
  local_io_object const detached("/dev/null", file_descriptor{}, file_descriptor{}, 1);
  CHECK(detached.identity().kind == identity_kind::opened_handle);
  CHECK(detached.size() == 1);
}
