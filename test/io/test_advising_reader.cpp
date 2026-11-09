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

#include "io/rest/loopback_range_server.hpp"

#include <cucascade/io/scan_cache.hpp>

#include <cuda_runtime.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

using cucascade::io::scan_cache;
using cucascade::io::scan_cache_config;
using cucascade::test::loopback_range_server;

constexpr char const* object_path  = "s3://bucket/object.bin";
constexpr std::size_t object_bytes = 8UL << 20;
constexpr std::size_t read_bytes   = 1UL << 20;
constexpr std::size_t small_bytes  = 4UL << 10;

std::vector<std::uint8_t> payload(std::size_t n)
{
  std::vector<std::uint8_t> bytes(n);
  for (std::size_t i = 0; i < n; ++i) {
    bytes[i] = static_cast<std::uint8_t>((i * 131U + 7U) & 0xffU);
  }
  return bytes;
}

/// A small cache against the loopback server: creating it pins host memory.
scan_cache_config config_for(loopback_range_server const& server)
{
  return {.cache_bytes = 256UL << 20,
          .endpoint    = server.endpoint(),
          .region      = "us-east-1",
          .access_key  = "test",
          .secret_key  = "test"};
}

/// One column that covers the first half of the object.
void register_first_half(scan_cache& cache)
{
  std::vector<std::string> const columns{"a"};
  std::array<std::uint32_t, 1> const column_of_chunk{0};
  std::array<std::uint64_t, 1> const offsets{0};
  std::array<std::uint64_t, 1> const sizes{object_bytes / 2};
  cache.register_layout(object_path, "s3://bucket/t", columns, column_of_chunk, offsets, sizes);
}

bool matches_object(std::vector<std::uint8_t> const& bytes, std::size_t offset)
{
  auto const expected = payload(object_bytes);
  return std::equal(
    bytes.begin(), bytes.end(), expected.begin() + static_cast<std::ptrdiff_t>(offset));
}

}  // namespace

TEST_CASE("an advising reader knows the size of its object", "[io][advising_reader]")
{
  loopback_range_server server(payload(object_bytes));
  scan_cache cache(config_for(server));
  auto reader = cache.open(object_path);
  CHECK(reader->size() == object_bytes);
  CHECK(reader->supports_device_read());
}

TEST_CASE("a read the policy admits is served from the cache the next time",
          "[io][advising_reader]")
{
  loopback_range_server server(payload(object_bytes));
  scan_cache cache(config_for(server));
  register_first_half(cache);
  auto reader = cache.open(object_path);

  std::vector<std::uint8_t> first(read_bytes);
  std::vector<std::uint8_t> second(read_bytes);
  auto const after_open = server.get_count();

  CHECK(reader->host_read(0, read_bytes, first.data()) == read_bytes);
  auto const after_first = server.get_count();
  CHECK(after_first > after_open);
  CHECK(matches_object(first, 0));

  CHECK(reader->host_read(0, read_bytes, second.data()) == read_bytes);
  CHECK(server.get_count() == after_first);
  CHECK(second == first);
}

TEST_CASE("a read the policy refuses is read again, not cached", "[io][advising_reader]")
{
  loopback_range_server server(payload(object_bytes));
  scan_cache_config config = config_for(server);
  config.admission_budget  = 0.0;
  scan_cache cache(config);
  register_first_half(cache);
  auto reader = cache.open(object_path);

  std::vector<std::uint8_t> bytes(read_bytes);
  CHECK(reader->host_read(0, read_bytes, bytes.data()) == read_bytes);
  auto const after_first = server.get_count();
  CHECK(matches_object(bytes, 0));

  CHECK(reader->host_read(0, read_bytes, bytes.data()) == read_bytes);
  CHECK(server.get_count() > after_first);
  CHECK(matches_object(bytes, 0));
}

TEST_CASE("a read smaller than min_advise_bytes is not cached", "[io][advising_reader]")
{
  loopback_range_server server(payload(object_bytes));
  scan_cache cache(config_for(server));
  register_first_half(cache);
  auto reader = cache.open(object_path);

  std::vector<std::uint8_t> bytes(small_bytes);
  CHECK(reader->host_read(0, small_bytes, bytes.data()) == small_bytes);
  auto const after_first = server.get_count();

  CHECK(reader->host_read(0, small_bytes, bytes.data()) == small_bytes);
  CHECK(server.get_count() > after_first);
  CHECK(matches_object(bytes, 0));
}

TEST_CASE("an asynchronous host read returns the bytes of its range", "[io][advising_reader]")
{
  loopback_range_server server(payload(object_bytes));
  scan_cache cache(config_for(server));
  register_first_half(cache);
  auto reader = cache.open(object_path);

  constexpr std::size_t offset = 2UL << 20;
  std::vector<std::uint8_t> bytes(read_bytes);
  CHECK(reader->host_read_async(offset, read_bytes, bytes.data()).get() == read_bytes);
  CHECK(matches_object(bytes, offset));
}

TEST_CASE("a device read returns the bytes of its range", "[io][advising_reader]")
{
  loopback_range_server server(payload(object_bytes));
  scan_cache cache(config_for(server));
  register_first_half(cache);
  auto reader = cache.open(object_path);

  cudaStream_t stream{};
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
  void* device = nullptr;
  REQUIRE(cudaMalloc(&device, read_bytes) == cudaSuccess);

  // The first read goes through the cache; the second is served from it.
  for (int pass = 0; pass < 2; ++pass) {
    REQUIRE(cudaMemsetAsync(device, 0, read_bytes, stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(reader
            ->device_read_async(
              0, read_bytes, static_cast<std::uint8_t*>(device), ::cuda::stream_ref{stream})
            .get() == read_bytes);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<std::uint8_t> bytes(read_bytes);
    REQUIRE(cudaMemcpy(bytes.data(), device, read_bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(matches_object(bytes, 0));
  }

  CHECK(cudaFree(device) == cudaSuccess);
  CHECK(cudaStreamDestroy(stream) == cudaSuccess);
}
