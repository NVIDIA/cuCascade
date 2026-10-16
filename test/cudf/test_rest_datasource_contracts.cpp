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

#include "cache_test_fixture.hpp"
#include "io/rest/rest_contract_test_utils.hpp"

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

using cucascade::io::open_datasource;
using cucascade::io::open_hint;
using cucascade::io::rest::rest_io_object;
using cucascade::test::identity_memory;
using cucascade::test::loopback_range_server;
using cucascade::test::make_rest_contract_ioctx;
using cucascade::test::require_rest_bytes_equal;
using cucascade::test::rest_contract_config;
using cucascade::test::rest_device_bytes;
using cucascade::test::rest_shutdown_guard;
using cucascade::test::rest_test_footer_bytes;
using cucascade::test::rest_test_parquet_bytes;
using cucascade::test::rest_test_payload;

}  // namespace

TEST_CASE("parquet footer bind is served by one suffix GET and then the stash", "[rest][footer]")
{
  auto const parquet    = rest_test_parquet_bytes();
  auto const footer_len = rest_test_footer_bytes;
  auto const footer_off = parquet.size() - 8 - footer_len;
  loopback_range_server server(parquet);
  auto ioctx = make_rest_contract_ioctx(server);
  auto datasource =
    open_datasource(ioctx, "s3://footer-bucket/nation.parquet", open_hint::parquet_footer_probe);
  REQUIRE(datasource != nullptr);
  CHECK(datasource->size() == parquet.size());
  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == 1);
  std::array<std::uint8_t, 8> trailer{};
  std::vector<std::uint8_t> footer(footer_len);
  rest_shutdown_guard cleanup{*ioctx};
  REQUIRE(datasource->host_read(parquet.size() - trailer.size(), trailer.size(), trailer.data()) ==
          trailer.size());
  require_rest_bytes_equal(trailer, {parquet.data() + parquet.size() - 8, 8});
  REQUIRE(datasource->host_read(footer_off, footer.size(), footer.data()) == footer.size());
  require_rest_bytes_equal(footer, {parquet.data() + footer_off, footer_len});
  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == 1);
}

TEST_CASE("footer probe open uses the configured suffix window", "[rest][footer]")
{
  auto const parquet       = rest_test_parquet_bytes();
  auto const footer_len    = rest_test_footer_bytes;
  auto const footer_off    = parquet.size() - 8 - footer_len;
  std::size_t suffix_bytes = 0;
  bool covered             = false;
  SECTION("tiny configured window misses the footer body and re-GETs it")
  {
    suffix_bytes = 8;
    REQUIRE(footer_len + 8 > suffix_bytes);
  }
  SECTION("configured window covering the footer serves both cudf footer reads from stash")
  {
    suffix_bytes = footer_len + 8;
    covered      = true;
    REQUIRE(suffix_bytes <= parquet.size());
  }
  loopback_range_server server(parquet);
  auto cfg               = rest_contract_config();
  cfg.footer_probe_bytes = suffix_bytes;
  auto ioctx             = make_rest_contract_ioctx(server, cfg);
  auto datasource =
    open_datasource(ioctx, "s3://footer-bucket/nation.parquet", open_hint::parquet_footer_probe);
  auto const* object = dynamic_cast<rest_io_object const*>(&datasource->get_io_object());
  REQUIRE(object != nullptr);
  CHECK(object->stash_window_lo() == parquet.size() - suffix_bytes);
  REQUIRE(object->stash() != nullptr);
  CHECK(object->stash()->size() == suffix_bytes);
  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == 1);
  std::array<std::uint8_t, 8> trailer{};
  std::vector<std::uint8_t> footer(footer_len);
  rest_shutdown_guard cleanup{*ioctx};
  if (covered) {
    REQUIRE(datasource->host_read(
              parquet.size() - trailer.size(), trailer.size(), trailer.data()) == trailer.size());
    require_rest_bytes_equal(trailer, {parquet.data() + parquet.size() - 8, 8});
  }
  REQUIRE(datasource->host_read(footer_off, footer.size(), footer.data()) == footer.size());
  require_rest_bytes_equal(footer, {parquet.data() + footer_off, footer_len});
  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == (covered ? 1 : 2));
}

TEST_CASE("rest_ioctx reads exact host ranges and clips EOF on SeaweedFS fixtures", "[rest][data]")
{
  auto const small = rest_test_payload(64UL << 10);
  loopback_range_server server(small);
  auto ioctx      = make_rest_contract_ioctx(server);
  auto datasource = open_datasource(ioctx, "s3://bucket/small.bin");
  REQUIRE(datasource->size() == small.size());
  std::vector<std::uint8_t> full(small.size());
  std::vector<std::uint8_t> mid(4096);
  std::vector<std::uint8_t> crossing(64, std::uint8_t{0xaa});
  std::array<std::uint8_t, 8> eof{};
  rest_shutdown_guard cleanup{*ioctx};
  REQUIRE(datasource->host_read(0, full.size(), full.data()) == full.size());
  require_rest_bytes_equal(full, small);
  constexpr std::size_t mid_offset = 1234;
  REQUIRE(datasource->host_read(mid_offset, mid.size(), mid.data()) == mid.size());
  require_rest_bytes_equal(mid, {small.data() + mid_offset, mid.size()});
  auto const tail_offset = small.size() - 17;
  REQUIRE(datasource->host_read(tail_offset, crossing.size(), crossing.data()) == 17);
  require_rest_bytes_equal({crossing.data(), 17}, {small.data() + tail_offset, 17});
  eof.fill(std::uint8_t{0xcc});
  CHECK(datasource->host_read(small.size(), eof.size(), eof.data()) == 0);
  CHECK(datasource->host_read(small.size() + 99, eof.size(), eof.data()) == 0);
  CHECK(std::all_of(eof.begin(), eof.end(), [](std::uint8_t byte) { return byte == 0xcc; }));
}

TEST_CASE(
  "rest_ioctx stages device reads through the fixed-size host memory resource for single and multi "
  "chunk SeaweedFS reads",
  "[rest][data][gpu]")
{
  int count = 0;
  REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
  if (count == 0) { SKIP("requires a CUDA device"); }
  auto memory       = identity_memory();
  auto const spaces = memory->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE_FALSE(spaces.empty());
  auto* host = const_cast<cucascade::memory::memory_space*>(spaces.front())
                 ->get_memory_resource_of<cucascade::memory::Tier::HOST>();
  REQUIRE(host != nullptr);
  REQUIRE(host->get_block_size() == (1U << 20));

  SECTION("single chunk")
  {
    auto const small = rest_test_payload(64UL << 10);
    REQUIRE(small.size() <= host->get_block_size());
    loopback_range_server server(small);
    auto ioctx      = make_rest_contract_ioctx(server, rest_contract_config(), 1, host);
    auto datasource = open_datasource(ioctx, "s3://bucket/small.bin");
    rmm::cuda_stream stream;
    rmm::device_buffer destination(small.size(), stream);
    rest_shutdown_guard cleanup{*ioctx};
    REQUIRE(datasource->device_read(
              0, small.size(), static_cast<std::uint8_t*>(destination.data()), stream) ==
            small.size());
    auto bytes = rest_device_bytes(destination.data(), small.size(), stream.value());
    require_rest_bytes_equal(bytes, small);
  }
  SECTION("multi chunk")
  {
    auto const medium = rest_test_payload(4UL << 20);
    loopback_range_server server(medium);
    auto ioctx      = make_rest_contract_ioctx(server, rest_contract_config(), 1, host);
    auto datasource = open_datasource(ioctx, "s3://bucket/medium.bin");
    constexpr std::size_t offset = 123;
    constexpr std::size_t size   = (3UL << 20) + 17;
    REQUIRE(size > host->get_block_size());
    rmm::cuda_stream stream;
    rmm::device_buffer destination(size, stream);
    rest_shutdown_guard cleanup{*ioctx};
    REQUIRE(datasource->device_read(
              offset, size, static_cast<std::uint8_t*>(destination.data()), stream) == size);
    auto bytes = rest_device_bytes(destination.data(), size, stream.value());
    require_rest_bytes_equal(bytes, {medium.data() + offset, size});
  }
}
