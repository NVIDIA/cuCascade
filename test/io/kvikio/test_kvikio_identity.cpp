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

#include <cucascade/io/io_errors.hpp>
#include <cucascade/io/kvikio/kvikio_context.hpp>
#include <cucascade/io/object_identity.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>

#include <catch2/catch_all.hpp>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>

namespace {
namespace io = cucascade::io;
using io::identity_kind;
using io::object_identity;

/// A temporary file of @p bytes copies of @p fill, removed on scope exit;
/// @ref replace atomically renames a new file of the same size over it.
class kvikio_identity_file {
 public:
  explicit kvikio_identity_file(std::size_t bytes, char fill = 'a')
    : _path(std::filesystem::temp_directory_path() /
            ("cucascade_kvikio_identity_" + std::to_string(::getpid()) + "_" +
             std::to_string(next_id.fetch_add(1)) + ".bin")),
      _bytes(bytes)
  {
    write(_path, fill);
  }

  ~kvikio_identity_file()
  {
    std::error_code ec;
    std::filesystem::remove(_path, ec);
    std::filesystem::remove(staged(), ec);
  }

  kvikio_identity_file(kvikio_identity_file const&)            = delete;
  kvikio_identity_file& operator=(kvikio_identity_file const&) = delete;

  void replace(char fill) const
  {
    write(staged(), fill);
    std::filesystem::rename(staged(), _path);
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
    if (!out) { throw std::runtime_error("creating kvikIO identity test file failed"); }
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

io::kvikio_io_object const& as_local(std::shared_ptr<io::io_object> const& object)
{
  auto const* local = dynamic_cast<io::kvikio_io_object const*>(object.get());
  REQUIRE(local != nullptr);
  return *local;
}

}  // namespace

TEST_CASE("kvikIO opens carry the stat identity of their descriptor", "[object_identity][kvikio]")
{
  kvikio_identity_file const file{12345};
  auto context = std::make_shared<io::kvikio_context>(io::kvikio_config{});

  auto const first    = context->open_io_object(file.path());
  auto const identity = first->identity();
  CHECK(identity.kind == identity_kind::local_stat);
  CHECK(identity.size == file.bytes());
  CHECK(first->size() == file.bytes());
  struct stat info{};
  REQUIRE(::stat(file.path().c_str(), &info) == 0);
  CHECK(identity.device == static_cast<std::uint64_t>(info.st_dev));
  CHECK(identity.inode == static_cast<std::uint64_t>(info.st_ino));
  // The O_DIRECT descriptor, when kvikIO opened one, names the same file.
  auto const& handle = as_local(first).handle();
  if (handle.fd(true) >= 0) {
    object_identity direct;
    REQUIRE(io::capture_local_identity(handle.fd(true), direct));
    CHECK(direct.device == identity.device);
    CHECK(direct.inode == identity.inode);
  }

  // A second open of the unchanged file is the same version, and so is an open
  // of it through the uring reactor: the identity names the file, not the
  // backend that opened it.
  auto const second = context->open_io_object(file.path());
  CHECK(second->open_generation() > first->open_generation());
  CHECK(second->identity() == identity);
  CHECK(second->identity_cache_key() == first->identity_cache_key());
  auto const uring_open = io::uring::uring_reactor::create_io_object(file.path());
  CHECK(uring_open->identity_cache_key() == first->identity_cache_key());

  // An atomic replacement of the same size is a new version; the object opened
  // before it still describes the file it holds.
  file.replace('b');
  auto const third = context->open_io_object(file.path());
  CHECK(third->identity().kind == identity_kind::local_stat);
  CHECK(third->identity() != identity);
  CHECK(third->identity_cache_key() != first->identity_cache_key());
  CHECK(first->identity() == identity);
}

TEST_CASE("kvikIO opens of a path being replaced report the file they opened",
          "[object_identity][kvikio]")
{
  // kvikIO opens its buffered and O_DIRECT descriptors from the path, so an
  // open can pair two versions; create_io_object retries once and then
  // reports descriptor_mismatch_error.  Every open that succeeds must describe
  // the file its buffered descriptor holds, with the O_DIRECT one on it too.
  cucascade::test::churning_file const file;
  auto context        = std::make_shared<io::kvikio_context>(io::kvikio_config{});
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  int opened          = 0;
  for (int opens = 0;
       opens < 200 || (file.replacements() < 200 && std::chrono::steady_clock::now() < deadline);
       ++opens) {
    std::shared_ptr<io::io_object> object;
    try {
      object = context->open_io_object(file.path());
    } catch (io::descriptor_mismatch_error const&) {
      continue;
    }
    ++opened;
    auto const& handle = as_local(object).handle();
    cucascade::test::churning_file::check_opened(
      object->identity(), object->size(), handle.fd(false), handle.fd(true));
  }
  CHECK(opened > 0);
}
