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

#include <cucascade/memory/memory_space.hpp>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

namespace {
using namespace cucascade::test;
using namespace std::chrono_literals;
using cucascade::io::prepare_result;
using cucascade::io::cache::eviction_policy;
constexpr std::size_t mib    = 1ULL << 20;
constexpr char object_path[] = "controlled://explicit-eviction";

cucascade::io::cache::fs_cache& checked_cache(cache_test_fixture& fixture)
{
  auto* cache = fixture.context->cache();
  REQUIRE(cache != nullptr);
  REQUIRE(cache->is_armed());
  return *cache;
}
}  // namespace

TEST_CASE("claimed_bytes tracks the staging memory the cache is holding",
          "[cache][eviction][explicit]")
{
  cache_test_fixture fixture{allocation_cache_config()};
  auto& cache = checked_cache(fixture);
  CHECK(cache.claimed_bytes() == 0);
  auto source = fixture.open(object_path, 8 * mib);
  REQUIRE(source != nullptr);
  advise_range(*source, 0, 4 * mib);
  CHECK(cache.claimed_bytes() == 0);
  REQUIRE(source->prepare_prefetch(false) == prepare_result::prepared);
  INFO(cache.summary());
  CHECK(cache.claimed_bytes() == 4 * mib);
}

TEST_CASE("an explicit evict reclaims a disposed request the pressure rule would not",
          "[cache][eviction][explicit]")
{
  cache_test_fixture fixture{allocation_cache_config()};
  auto& cache = checked_cache(fixture);
  {
    auto source = fixture.open(object_path, 8 * mib);
    REQUIRE(source != nullptr);
    advise_range(*source, 0, 4 * mib);
    REQUIRE(source->prepare_prefetch(false) == prepare_result::prepared);
  }
  auto const claimed = cache.claimed_bytes();
  REQUIRE(claimed > 0);
  std::this_thread::sleep_for(300ms);
  REQUIRE(cache.claimed_bytes() == claimed);
  cache.evict(claimed);
  INFO(cache.summary());
  CHECK(wait_for_cache([&] { return cache.claimed_bytes() == 0; }, 2s));
}

TEST_CASE("blocking prepare waits for earlier asynchronous eviction before retrying",
          "[cache][eviction][explicit][prepare]")
{
  constexpr std::size_t request_bytes = 32 * mib;
  cache_test_fixture fixture{allocation_cache_config(eviction_policy::lru, true)};
  auto& cache        = checked_cache(fixture);
  std::size_t offset = 0;
  std::unique_ptr<cucascade::io::datasource> target;
  for (std::size_t attempt = 0; attempt < 32; ++attempt) {
    auto filler = fixture.open(object_path, 1ULL << 30);
    REQUIRE(filler != nullptr);
    advise_range(*filler, offset, request_bytes);
    auto const result = filler->prepare_prefetch(false);
    if (result == prepare_result::allocation_failed) {
      target = std::move(filler);
      break;
    }
    REQUIRE(result == prepare_result::prepared);
    offset += request_bytes;
  }
  INFO(cache.summary());
  REQUIRE(target != nullptr);
  REQUIRE(target->prepare_prefetch(true) == prepare_result::prepared);
}

TEST_CASE("failed nonblocking preparation starts eviction without waiting",
          "[cache][eviction][explicit][prepare]")
{
  cache_test_fixture fixture{allocation_cache_config(eviction_policy::lru, true)};
  auto& cache = checked_cache(fixture);
  {
    auto victim = fixture.open(object_path, 1ULL << 30);
    REQUIRE(victim != nullptr);
    advise_range(*victim, 0, 16 * mib);
    REQUIRE(victim->prepare_prefetch(false) == prepare_result::prepared);
  }
  std::vector<std::unique_ptr<cucascade::io::datasource>> fillers;
  bool exhausted = false;
  for (std::size_t i = 0; i < 32; ++i) {
    auto filler = fixture.open(object_path, 1ULL << 30);
    REQUIRE(filler != nullptr);
    advise_range(*filler, (16 + 32 * i) * mib, 32 * mib);
    auto const before = cache.claimed_bytes();
    auto const result = filler->prepare_prefetch(false);
    if (result == prepare_result::allocation_failed) {
      exhausted = true;
      CHECK(wait_for_cache([&] { return cache.claimed_bytes() < before; }, 2s));
      break;
    }
    REQUIRE(result == prepare_result::prepared);
    fillers.push_back(std::move(filler));
  }
  REQUIRE(exhausted);
}

TEST_CASE("prepare does not publish a request the retry's eviction left short of buffers",
          "[cache][eviction][explicit][prepare]")
{
  cache_test_fixture fixture{allocation_cache_config(eviction_policy::lru, true)};
  auto& cache = checked_cache(fixture);
  REQUIRE(cache.chunk_size() == controlled_chunk_bytes);
  auto spaces = fixture.memory->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(spaces.size() == 1);
  REQUIRE(spaces.front()->get_max_memory() == 256 * mib);
  auto* host = const_cast<cucascade::memory::memory_space*>(spaces.front())
                 ->get_memory_resource_of<cucascade::memory::Tier::HOST>();
  REQUIRE(host != nullptr);
  REQUIRE(host->get_block_size() == controlled_chunk_bytes);
  REQUIRE(host->get_total_reserved_bytes() == 64 * mib);
  auto victim = fixture.open(object_path, 2ULL << 30);
  REQUIRE(victim != nullptr);
  advise_range(*victim, 0, 36 * mib);
  REQUIRE(victim->prepare_prefetch(false) == prepare_result::prepared);
  auto target = fixture.open(object_path, 2ULL << 30);
  REQUIRE(target != nullptr);
  std::array<cudf::io::text::byte_range_info, 2> ranges{
    cudf::io::text::byte_range_info{0, static_cast<std::int64_t>(32 * mib)},
    cudf::io::text::byte_range_info{static_cast<std::int64_t>(36 * mib),
                                    static_cast<std::int64_t>(32 * mib)}};
  target->fadvise(ranges, 0);
  auto const pressure_bytes = host->get_available_memory();
  REQUIRE(pressure_bytes > 0);
  REQUIRE(pressure_bytes % mib == 0);
  auto pressure = host->allocate_multiple_blocks(pressure_bytes);
  REQUIRE(pressure != nullptr);
  REQUIRE(pressure->size_bytes() == pressure_bytes);
  REQUIRE(host->get_available_memory() == 0);
  cache.evict_sync(mib);
  REQUIRE(cache.claimed_bytes() == 36 * mib);
  victim.reset();
  INFO(cache.summary());
  CHECK(target->prepare_prefetch(true) == prepare_result::allocation_failed);
}

TEST_CASE("prepare abandons immediately after the consumer reaches the split",
          "[cache][eviction][explicit][prepare]")
{
  cache_test_fixture fixture{allocation_cache_config()};
  auto& cache = checked_cache(fixture);
  auto source = fixture.open(object_path, 8 * mib);
  REQUIRE(source != nullptr);
  advise_range(*source, 0, 4 * mib);
  source->update(cucascade::io::cache::scan_stage::preparing);
  CHECK(source->prepare_prefetch(true) == prepare_result::fallen_behind);
  CHECK(cache.claimed_bytes() == 0);
}

TEST_CASE("an explicit evict frees at least what was asked for", "[cache][eviction][explicit]")
{
  cache_test_fixture fixture{allocation_cache_config()};
  auto& cache = checked_cache(fixture);
  {
    auto source = fixture.open(object_path, 8 * mib);
    REQUIRE(source != nullptr);
    advise_range(*source, 0, 4 * mib);
    REQUIRE(source->prepare_prefetch(false) == prepare_result::prepared);
  }
  auto const claimed = cache.claimed_bytes();
  REQUIRE(claimed > 0);
  cache.evict_sync(2 * mib);
  CHECK(cache.claimed_bytes() <= claimed - std::min(claimed, 2 * mib));
}

TEST_CASE("a zero-byte evict is a no-op", "[cache][eviction][explicit]")
{
  cache_test_fixture fixture{allocation_cache_config()};
  auto& cache = checked_cache(fixture);
  {
    auto source = fixture.open(object_path, 8 * mib);
    REQUIRE(source != nullptr);
    advise_range(*source, 0, 4 * mib);
    REQUIRE(source->prepare_prefetch(false) == prepare_result::prepared);
  }
  auto const claimed = cache.claimed_bytes();
  REQUIRE(claimed > 0);
  cache.evict(0);
  std::this_thread::sleep_for(300ms);
  CHECK(cache.claimed_bytes() == claimed);
}

TEST_CASE("the evictor retires disposed requests while the pool is under its threshold",
          "[cache][eviction][explicit]")
{
  cache_test_fixture fixture{allocation_cache_config()};
  auto& cache            = checked_cache(fixture);
  constexpr int requests = 64;
  for (int i = 0; i < requests; ++i) {
    auto source = fixture.open(object_path, 8 * mib);
    REQUIRE(source != nullptr);
    advise_range(*source, 0, 4 * mib);
    std::this_thread::sleep_for(2ms);
  }
  constexpr std::size_t max_tracked = 8;
  INFO(cache.summary());
  CHECK(wait_for_cache([&] { return cache.eviction_batch_size_for_testing() <= max_tracked; }, 2s));
}
