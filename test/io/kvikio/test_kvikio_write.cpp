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

#include <cucascade/io/kvikio/kvikio_context.hpp>
#include <cucascade/io/types.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

using namespace cucascade::io;

namespace {

/// Unique temp file path, removed on destruction.
struct temp_path {
  std::filesystem::path path;

  temp_path()
  {
    static std::atomic<int> counter{0};
    path = std::filesystem::temp_directory_path() /
           ("cucascade_kvikio_write_" + std::to_string(::getpid()) + "_" +
            std::to_string(counter.fetch_add(1)) + ".bin");
    std::filesystem::remove(path);
  }
  ~temp_path()
  {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
  temp_path(temp_path const&)            = delete;
  temp_path& operator=(temp_path const&) = delete;
};

std::vector<std::uint8_t> pattern(std::size_t size, std::uint8_t seed)
{
  std::vector<std::uint8_t> out(size);
  for (std::size_t i = 0; i < size; ++i) {
    out[i] = static_cast<std::uint8_t>((i * 131U + seed) & 0xFFU);
  }
  return out;
}

std::vector<std::uint8_t> file_bytes(std::filesystem::path const& path)
{
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

bool has_gpu()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

}  // namespace

TEST_CASE("kvikio_context advertises write support", "[io][kvikio][write]")
{
  auto ctx = std::make_shared<kvikio_context>();
  CHECK(ctx->supports_write());
  CHECK(ctx->supports_device_write());
}

TEST_CASE("kvikio_context host write round-trips", "[io][kvikio][write]")
{
  temp_path tmp;
  auto ctx = std::make_shared<kvikio_context>();
  auto obj = ctx->open_io_object_for_write(tmp.path.string());
  REQUIRE(obj->size() == 0);

  // Unaligned size and offset; no padding may appear in the file.
  constexpr std::size_t size = (3UL << 20) + 123;
  auto const data            = pattern(size, 7);
  REQUIRE(ctx->host_write(*obj, 0, size, data.data()) == size);
  CHECK(obj->size() == size);

  SECTION("readable through the same object")
  {
    std::vector<std::uint8_t> back(size);
    REQUIRE(ctx->host_read(*obj, 0, size, back.data()) == size);
    CHECK(back == data);
  }
  SECTION("visible to a fresh reader")
  {
    auto reader = ctx->open_io_object(tmp.path.string());
    REQUIRE(reader->size() == size);
    std::vector<std::uint8_t> back(size);
    REQUIRE(ctx->host_read(*reader, 0, size, back.data()) == size);
    CHECK(back == data);
    CHECK(file_bytes(tmp.path) == data);
  }
}

TEST_CASE("kvikio_context async and vectored host writes", "[io][kvikio][write]")
{
  temp_path tmp;
  auto ctx = std::make_shared<kvikio_context>();
  auto obj = ctx->open_io_object_for_write(tmp.path.string(), {.size_hint = 1UL << 20});

  auto const a = pattern(4096 + 17, 1);
  auto const b = pattern(5000, 2);
  auto const c = pattern(1UL << 20, 3);

  REQUIRE(ctx->host_write_async(*obj, 100, a.size(), a.data()).get() == a.size());

  // Segments out of order, leaving a hole between them.
  std::vector<write_segment> segments;
  std::size_t const c_offset = 3UL << 20;
  segments.push_back(write_segment{range{c_offset, c.size()}, host_source{c.data()}});
  segments.push_back(write_segment{range{100 + a.size(), b.size()}, host_source{b.data()}});
  REQUIRE(ctx->writev_async(*obj, std::move(segments), {.durability = write_durability::data_sync})
            .get() == b.size() + c.size());
  CHECK(obj->size() == c_offset + c.size());

  ctx->flush_async(*obj).get();
  ctx->commit_async(*obj, write_durability::data_sync).get();

  auto const bytes = file_bytes(tmp.path);
  REQUIRE(bytes.size() == c_offset + c.size());
  auto const is_zero = [](std::uint8_t v) { return v == 0; };
  CHECK(std::all_of(bytes.begin(), bytes.begin() + 100, is_zero));
  CHECK(std::equal(a.begin(), a.end(), bytes.begin() + 100));
  CHECK(
    std::equal(b.begin(), b.end(), bytes.begin() + static_cast<std::ptrdiff_t>(100 + a.size())));
  CHECK(std::all_of(bytes.begin() + static_cast<std::ptrdiff_t>(100 + a.size() + b.size()),
                    bytes.begin() + static_cast<std::ptrdiff_t>(c_offset),
                    is_zero));
  CHECK(std::equal(c.begin(), c.end(), bytes.begin() + static_cast<std::ptrdiff_t>(c_offset)));
}

TEST_CASE("kvikio_context write open modes", "[io][kvikio][write]")
{
  temp_path tmp;
  auto ctx           = std::make_shared<kvikio_context>();
  auto const initial = pattern(8192, 9);
  {
    auto obj = ctx->open_io_object_for_write(tmp.path.string());
    REQUIRE(ctx->host_write(*obj, 0, initial.size(), initial.data()) == initial.size());
  }

  SECTION("create_or_open keeps content and overwrites in place")
  {
    auto obj =
      ctx->open_io_object_for_write(tmp.path.string(), {.mode = write_mode::create_or_open});
    CHECK(obj->size() == initial.size());
    auto const patch = pattern(10, 200);
    REQUIRE(ctx->host_write(*obj, 50, patch.size(), patch.data()) == patch.size());
    auto expected = initial;
    std::copy(patch.begin(), patch.end(), expected.begin() + 50);
    CHECK(file_bytes(tmp.path) == expected);
  }
  SECTION("open_existing keeps content")
  {
    auto obj =
      ctx->open_io_object_for_write(tmp.path.string(), {.mode = write_mode::open_existing});
    CHECK(obj->size() == initial.size());
  }
  SECTION("create_or_truncate truncates")
  {
    auto obj = ctx->open_io_object_for_write(tmp.path.string());
    CHECK(obj->size() == 0);
    CHECK(std::filesystem::file_size(tmp.path) == 0);
  }
}

TEST_CASE("kvikio_context write errors", "[io][kvikio][write]")
{
  temp_path tmp;
  auto ctx           = std::make_shared<kvikio_context>();
  auto const payload = pattern(64, 5);

  SECTION("open_existing on a missing file is ENOENT")
  {
    try {
      static_cast<void>(
        ctx->open_io_object_for_write(tmp.path.string(), {.mode = write_mode::open_existing}));
      FAIL("expected std::system_error");
    } catch (std::system_error const& e) {
      CHECK(e.code() == std::error_code(ENOENT, std::generic_category()));
    }
  }
  SECTION("s3 URIs are not supported")
  {
    try {
      static_cast<void>(ctx->open_io_object_for_write("s3://bucket/key"));
      FAIL("expected std::system_error");
    } catch (std::system_error const& e) {
      CHECK(e.code() == std::make_error_code(std::errc::not_supported));
    }
  }
  SECTION("read-only objects reject writes")
  {
    {
      auto obj = ctx->open_io_object_for_write(tmp.path.string());
      REQUIRE(ctx->host_write(*obj, 0, payload.size(), payload.data()) == payload.size());
    }
    auto reader = ctx->open_io_object(tmp.path.string());
    CHECK_THROWS_AS(ctx->host_write(*reader, 0, payload.size(), payload.data()),
                    std::invalid_argument);
    CHECK_THROWS_AS(ctx->host_write_async(*reader, 0, payload.size(), payload.data()).get(),
                    std::invalid_argument);
    CHECK_THROWS_AS(ctx->commit_async(*reader).get(), std::invalid_argument);
  }
  SECTION("committed objects reject writes")
  {
    auto obj = ctx->open_io_object_for_write(tmp.path.string());
    REQUIRE(ctx->host_write(*obj, 0, payload.size(), payload.data()) == payload.size());
    ctx->commit_async(*obj).get();
    CHECK_THROWS_AS(ctx->host_write(*obj, 0, payload.size(), payload.data()),
                    std::invalid_argument);
    CHECK_THROWS_AS(ctx->commit_async(*obj).get(), std::invalid_argument);
    // Still readable after commit.
    std::vector<std::uint8_t> back(payload.size());
    REQUIRE(ctx->host_read(*obj, 0, back.size(), back.data()) == back.size());
    CHECK(back == payload);
  }
  SECTION("overlapping segments are rejected")
  {
    auto obj = ctx->open_io_object_for_write(tmp.path.string());
    std::vector<write_segment> segments;
    segments.push_back(write_segment{range{0, 32}, host_source{payload.data()}});
    segments.push_back(write_segment{range{16, 32}, host_source{payload.data()}});
    CHECK_THROWS_AS(ctx->writev_async(*obj, std::move(segments)).get(), std::invalid_argument);
  }
}

TEST_CASE("kvikio_context device writes round-trip", "[io][kvikio][write][gpu]")
{
  if (!has_gpu()) { SKIP("no CUDA device"); }

  temp_path tmp;
  auto ctx = std::make_shared<kvikio_context>();
  auto obj = ctx->open_io_object_for_write(tmp.path.string());

  constexpr std::size_t size = (2UL << 20) + 4097;
  auto const host            = pattern(size, 42);
  auto const tail            = pattern(3000, 77);

  cudaStream_t stream{};
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
  void* device = nullptr;
  REQUIRE(cudaMalloc(&device, size) == cudaSuccess);
  // Fill asynchronously on the same stream: the write must observe it.
  REQUIRE(cudaMemcpyAsync(device, host.data(), size, cudaMemcpyHostToDevice, stream) ==
          cudaSuccess);

  auto const* src = static_cast<const std::uint8_t*>(device);
  REQUIRE(ctx->device_write_async(*obj, 0, size, src, ::cuda::stream_ref{stream}).get() == size);

  // Mixed host + device vectored write, extending the file.
  std::vector<write_segment> segments;
  segments.push_back(write_segment{range{size, tail.size()}, host_source{tail.data()}});
  segments.push_back(write_segment{range{size + tail.size(), 4096},
                                   device_source{src, ::cuda::stream_ref{stream}, -1}});
  REQUIRE(ctx->writev_async(*obj, std::move(segments)).get() == tail.size() + 4096);
  ctx->commit_async(*obj, write_durability::data_sync).get();

  CHECK(cudaFree(device) == cudaSuccess);
  CHECK(cudaStreamDestroy(stream) == cudaSuccess);

  auto expected = host;
  expected.insert(expected.end(), tail.begin(), tail.end());
  expected.insert(expected.end(), host.begin(), host.begin() + 4096);
  CHECK(obj->size() == expected.size());
  CHECK(file_bytes(tmp.path) == expected);
}
