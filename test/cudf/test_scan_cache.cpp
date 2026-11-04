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

#include <cucascade/cudf/scan_cache.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cucascade::io::scan_cache;
using cucascade::io::scan_cache_options;

/// A small cache against an endpoint that nothing listens on: creating it pins host memory but
/// makes no request.
scan_cache_options small_cache()
{
  return {.cache_bytes = 256UL << 20,
          .endpoint    = "http://127.0.0.1:9",
          .region      = "us-east-1",
          .access_key  = "test",
          .secret_key  = "test"};
}

void register_two_columns(scan_cache& cache, std::string const& path)
{
  std::vector<std::string> const columns{"a", "b"};
  std::array<std::uint32_t, 2> const column_of_chunk{0, 1};
  std::array<std::uint64_t, 2> const offsets{0, 4000};
  std::array<std::uint64_t, 2> const sizes{4000, 4000};
  cache.register_layout(path, "s3://bucket/t", columns, column_of_chunk, offsets, sizes);
}

}  // namespace

TEST_CASE("a scan cache needs a size", "[cache][scan-cache]")
{
  scan_cache_options options = small_cache();
  options.cache_bytes        = 0;
  CHECK_THROWS_AS(scan_cache(options), std::invalid_argument);
}

TEST_CASE("a registered layout shows up in the column statistics", "[cache][scan-cache]")
{
  scan_cache cache(small_cache());
  CHECK(cache.stats().empty());
  register_two_columns(cache, "s3://bucket/t/f1.parquet");

  auto const stats = cache.stats();
  REQUIRE(stats.size() == 2);
  CHECK(stats[0].name == "s3://bucket/t/a");
  CHECK(stats[0].registered == 4000);
  CHECK(stats[0].requested == 0);
  CHECK(stats[1].name == "s3://bucket/t/b");
}

TEST_CASE("a malformed layout is rejected", "[cache][scan-cache]")
{
  scan_cache cache(small_cache());
  std::vector<std::string> const columns{"a"};
  std::array<std::uint32_t, 1> const column_of_chunk{3};
  std::array<std::uint64_t, 1> const offsets{0};
  std::array<std::uint64_t, 1> const sizes{10};
  CHECK_THROWS_AS(
    cache.register_layout("s3://bucket/t/f.parquet", "t", columns, column_of_chunk, offsets, sizes),
    std::invalid_argument);
  CHECK(cache.stats().empty());
}

TEST_CASE("the summary reports the prefetching cache", "[cache][scan-cache]")
{
  scan_cache cache(small_cache());
  CHECK(cache.summary().starts_with("fs_cache:"));
}

TEST_CASE("the cache can be placed on a NUMA node or left to the driver", "[cache][scan-cache]")
{
  for (int node : {0, -1}) {
    scan_cache_options options = small_cache();
    options.numa_node          = node;
    scan_cache cache(options);
    CHECK(cache.stats().empty());
  }
}

TEST_CASE("a scan cache can be created again after one is destroyed", "[cache][scan-cache]")
{
  {
    scan_cache first(small_cache());
  }
  scan_cache second(small_cache());
  CHECK(second.stats().empty());
}
