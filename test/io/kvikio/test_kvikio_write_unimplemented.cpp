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

// The kvikIO context stays read-only: every write and flush entry point must
// fail with io::not_implemented_error and leave the file system untouched.

#include <cucascade/io/io_errors.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/kvikio/kvikio_context.hpp>
#include <cucascade/io/types.hpp>

#include <catch2/catch_all.hpp>
#include <unistd.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

namespace io = cucascade::io;

template <class Call>
void require_not_implemented(Call&& call, std::string_view operation)
{
  CAPTURE(operation);
  try {
    std::forward<Call>(call)();
    FAIL("expected not_implemented_error");
  } catch (io::not_implemented_error const& error) {
    std::system_error const& base = error;
    CHECK(base.code() == std::errc::not_supported);
    std::string_view const what = error.what();
    CHECK(what.find("not yet implemented") != std::string_view::npos);
    CHECK(what.find("kvikio") != std::string_view::npos);
    CHECK(what.find(operation) != std::string_view::npos);
  }
}

/// A scratch directory holding one 4 KiB file of a known pattern.
class scratch {
 public:
  scratch()
    : _dir(std::filesystem::temp_directory_path() /
           ("cucascade_kvikio_write_" + std::to_string(::getpid())))
  {
    std::filesystem::create_directories(_dir);
    std::ofstream out(existing(), std::ios::binary);
    for (std::size_t i = 0; i < bytes; ++i) {
      out.put(static_cast<char>(i * 7));
    }
  }
  ~scratch()
  {
    std::error_code ignored;
    std::filesystem::remove_all(_dir, ignored);
  }
  scratch(scratch const&)            = delete;
  scratch& operator=(scratch const&) = delete;

  [[nodiscard]] std::string existing() const { return (_dir / "existing.bin").string(); }
  [[nodiscard]] std::string missing() const { return (_dir / "missing.bin").string(); }

  [[nodiscard]] std::vector<char> content() const
  {
    std::ifstream in(existing(), std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  }

  static constexpr std::size_t bytes = 4096;

 private:
  std::filesystem::path _dir;
};

}  // namespace

TEST_CASE("the kvikIO context advertises no writes", "[io][kvikio][write][unimplemented]")
{
  auto context    = std::make_shared<io::kvikio_context>();
  io::ioctx& base = *context;
  CHECK_FALSE(base.supports_write());
  CHECK_FALSE(base.supports_device_write());
}

TEST_CASE("every kvikIO write entry point is not yet implemented",
          "[io][kvikio][write][unimplemented]")
{
  scratch files;
  auto const before = files.content();
  auto context      = std::make_shared<io::kvikio_context>();
  io::ioctx& base   = *context;
  auto object       = base.open_io_object(files.existing());
  REQUIRE(object != nullptr);

  std::array<std::uint8_t, 16> host{};
  host.fill(0xee);
  auto* device       = reinterpret_cast<std::uint8_t const*>(std::uintptr_t{0x10000});
  auto const stream  = ::cuda::stream_ref{cudaStream_t{nullptr}};
  auto const segment = std::vector<io::write_slice>{{0, host.size(), host.data()}};
  auto const dev_seg = std::vector<io::write_slice>{{64, 16, device}};

  SECTION("opening for writing touches no file")
  {
    for (auto const mode : {io::write_mode::create_or_truncate,
                            io::write_mode::create_or_open,
                            io::write_mode::open_existing,
                            io::write_mode::create_new}) {
      require_not_implemented(
        [&] { std::ignore = base.open_io_object_for_write(files.missing(), {.mode = mode}); },
        "open_io_object_for_write");
      require_not_implemented(
        [&] { std::ignore = base.open_io_object_for_write(files.existing(), {.mode = mode}); },
        "open_io_object_for_write");
    }
    CHECK_FALSE(std::filesystem::exists(files.missing()));
  }

  SECTION("front wrappers through ioctx&")
  {
    require_not_implemented([&] { base.host_write(*object, 0, host.size(), host.data()); },
                            "host_write");
    require_not_implemented(
      [&] { std::ignore = base.host_write_async(*object, 0, host.size(), host.data()).get(); },
      "write");
    require_not_implemented(
      [&] { std::ignore = base.device_write_async(*object, 0, 16, device, stream).get(); },
      "write");
    require_not_implemented([&] { std::ignore = base.host_writev_async(*object, segment).get(); },
                            "write");
    require_not_implemented(
      [&] { std::ignore = base.device_writev_async(*object, dev_seg, stream).get(); }, "write");
    require_not_implemented(
      [&] { std::ignore = base.mixed_writev_async(*object, segment, dev_seg, stream).get(); },
      "write");
    require_not_implemented([&] { base.flush(*object); }, "flush");
    require_not_implemented([&] { base.flush_async(*object).get(); }, "flush_async");
  }

  SECTION("direct backend hooks")
  {
    require_not_implemented([&] { context->host_write_io(*object, 0, host.size(), host.data()); },
                            "host_write");
    std::vector<io::prepared_io_slice> slices;
    slices.push_back(io::make_host_write_slice(io::range{0, host.size()}, host.data()));
    require_not_implemented(
      [&] { std::ignore = context->mixed_writev_async_io(*object, std::move(slices)).get(); },
      "mixed_writev_async");
    require_not_implemented([&] { context->flush_io(*object); }, "flush");
    require_not_implemented([&] { context->flush_async_io(*object, io::io_priority::high).get(); },
                            "flush_async");
  }

  SECTION("zero-size writes stay no-ops")
  {
    CHECK(base.host_write(*object, 0, 0, nullptr) == 0);
    CHECK(base.host_write_async(*object, 0, 0, nullptr).get() == 0);
    CHECK(base.device_write_async(*object, 0, 0, nullptr, stream).get() == 0);
    CHECK(base.host_writev_async(*object, {}).get() == 0);
    CHECK(base.mixed_writev_async(*object, {}, {}, stream).get() == 0);
  }

  // The file's bytes and size are unchanged.
  CHECK(files.content() == before);
  CHECK(object->size() == scratch::bytes);
}
