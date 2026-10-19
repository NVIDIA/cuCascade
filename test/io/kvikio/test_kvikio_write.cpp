/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <cucascade/io/kvikio/kvikio_context.hpp>
#include <cucascade/io/types.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

using cucascade::io::kvikio_context;
using cucascade::io::write_mode;
using cucascade::io::write_slice;

namespace {

[[nodiscard]] std::vector<std::uint8_t> pattern(std::size_t bytes, std::size_t seed)
{
  std::vector<std::uint8_t> out(bytes);
  for (std::size_t i = 0; i < bytes; ++i) {
    out[i] = static_cast<std::uint8_t>(((i + seed * 131U) * 2654435761ULL) >> 9);
  }
  return out;
}

class scratch_path {
 public:
  scratch_path()
    : _path(std::filesystem::temp_directory_path() /
            ("cucascade_kvikio_write_" + std::to_string(::getpid()) + "_" +
             std::to_string(next_id.fetch_add(1)) + ".bin"))
  {
  }
  ~scratch_path()
  {
    std::error_code ec;
    std::filesystem::remove(_path, ec);
  }
  scratch_path(scratch_path const&)            = delete;
  scratch_path& operator=(scratch_path const&) = delete;

  [[nodiscard]] std::string str() const { return _path.string(); }

  [[nodiscard]] std::vector<std::uint8_t> contents() const
  {
    std::ifstream in(_path, std::ios::binary);
    REQUIRE(in);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  }

 private:
  static inline std::atomic<int> next_id{0};
  std::filesystem::path _path;
};

}  // namespace

TEST_CASE("kvikio_context writes host and device sources", "[kvikio][write]")
{
  kvikio_context ctx;
  CHECK(ctx.supports_write());
  CHECK(ctx.supports_device_write());

  scratch_path path;
  auto object = ctx.open_io_object_for_write(path.str());
  REQUIRE(object->writable());
  CHECK(object->size() == 0);

  auto const header = pattern(100, 1);
  auto const body   = pattern((3UL << 20) + 5, 2);
  auto const tail   = pattern(7, 3);

  void* device = nullptr;
  REQUIRE(cudaMalloc(&device, body.size()) == cudaSuccess);
  REQUIRE(cudaMemcpy(device, body.data(), body.size(), cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  CHECK(ctx.host_write(*object, 0, header.size(), header.data()) == header.size());
  std::vector<write_slice> device_slices{
    {4096, body.size(), static_cast<std::uint8_t const*>(device)}};
  std::vector<write_slice> host_slices{{4096 + body.size() + 10, tail.size(), tail.data()}};
  CHECK(
    ctx.mixed_writev_async(*object, host_slices, device_slices, ::cuda::stream_ref{stream}).get() ==
    body.size() + tail.size());
  CHECK_NOTHROW(ctx.flush(*object));
  CHECK_NOTHROW(ctx.flush_async(*object).get());

  auto const end = 4096 + body.size() + 10 + tail.size();
  CHECK(object->size() == end);
  auto const contents = path.contents();
  REQUIRE(contents.size() == end);
  CHECK(std::equal(header.begin(), header.end(), contents.begin()));
  CHECK(
    std::all_of(contents.begin() + 100, contents.begin() + 4096, [](auto b) { return b == 0; }));
  CHECK(std::equal(body.begin(), body.end(), contents.begin() + 4096));
  CHECK(std::equal(
    tail.begin(), tail.end(), contents.end() - static_cast<std::ptrdiff_t>(tail.size())));

  // Reads through the writable object see the grown size.
  std::vector<std::uint8_t> back(end);
  CHECK(ctx.host_read_io(*object, 0, back.size(), back.data()) == end);
  CHECK(back == contents);

  std::ignore = cudaStreamDestroy(stream);
  std::ignore = cudaFree(device);
}

TEST_CASE("kvikio_context rejects writes to read-only and object-store objects", "[kvikio][write]")
{
  kvikio_context ctx;
  scratch_path path;
  {
    std::ofstream out(path.str(), std::ios::binary);
    out << "abc";
  }
  auto object      = ctx.open_io_object(path.str());
  auto const bytes = pattern(3, 4);
  try {
    std::ignore = ctx.host_write(*object, 0, bytes.size(), bytes.data());
    FAIL("wrote through a read-only object");
  } catch (std::system_error const& error) {
    CHECK(error.code() == std::errc::bad_file_descriptor);
  }
  CHECK_THROWS_AS(ctx.host_write_async(*object, 0, bytes.size(), bytes.data()).get(),
                  std::system_error);

  try {
    std::ignore = ctx.open_io_object_for_write("s3://bucket/key");
    FAIL("opened an object-store URI for writing");
  } catch (std::system_error const& error) {
    CHECK(error.code() == std::errc::not_supported);
  }

  try {
    std::ignore = ctx.open_io_object_for_write(path.str(), {.mode = write_mode::create_new});
    FAIL("create_new opened an existing file");
  } catch (std::system_error const& error) {
    CHECK(error.code() == std::errc::file_exists);
  }
}

TEST_CASE("kvikio_context device writes observe the work queued on their stream", "[kvikio][write]")
{
  kvikio_context ctx;
  scratch_path path;
  auto object = ctx.open_io_object_for_write(path.str());

  constexpr std::size_t bytes = (8UL << 20) + 3;
  void* device                = nullptr;
  REQUIRE(cudaMalloc(&device, bytes) == cudaSuccess);
  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  // Queue the producer, then the write, without synchronizing in between.
  REQUIRE(cudaMemsetAsync(device, 0x11, bytes, stream) == cudaSuccess);
  REQUIRE(cudaMemsetAsync(device, 0xCD, bytes, stream) == cudaSuccess);
  CHECK(ctx
          .device_write_async(
            *object, 0, bytes, static_cast<std::uint8_t const*>(device), ::cuda::stream_ref{stream})
          .get() == bytes);

  auto const contents = path.contents();
  REQUIRE(contents.size() == bytes);
  CHECK(std::all_of(contents.begin(), contents.end(), [](std::uint8_t b) { return b == 0xCD; }));
  std::ignore = cudaStreamDestroy(stream);
  std::ignore = cudaFree(device);
}

TEST_CASE("kvikio_context open honours size hints and reports open errors", "[kvikio][write]")
{
  kvikio_context ctx;
  scratch_path path;
  {
    auto object = ctx.open_io_object_for_write(path.str(), {.size_hint = 1UL << 20});
    CHECK(object->size() == 0);
  }
  CHECK(std::filesystem::file_size(path.str()) == 0);

  // An existing file: create_or_open keeps it, create_or_truncate empties it.
  {
    std::ofstream out(path.str(), std::ios::binary);
    out << "0123456789";
  }
  CHECK(ctx.open_io_object_for_write(path.str(), {.mode = write_mode::create_or_open})->size() ==
        10);
  CHECK(std::filesystem::file_size(path.str()) == 10);
  CHECK(
    ctx.open_io_object_for_write(path.str(), {.mode = write_mode::create_or_truncate})->size() ==
    0);
  CHECK(std::filesystem::file_size(path.str()) == 0);

  try {
    std::ignore =
      ctx.open_io_object_for_write(path.str() + ".missing/x", {.mode = write_mode::create_new});
    FAIL("created a file under a missing directory");
  } catch (std::system_error const& error) {
    CHECK(error.code() == std::errc::no_such_file_or_directory);
  }
  CHECK_FALSE(std::filesystem::exists(path.str() + ".missing"));
}
