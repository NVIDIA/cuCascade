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
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <regex>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace cucascade::test;
using namespace std::chrono_literals;
using cucascade::io::prepare_result;
using cucascade::io::cache::eviction_policy;
using cucascade::io::cache::reclaim_mode;
constexpr std::size_t mib    = 1ULL << 20;
constexpr char object_path[] = "controlled://explicit-eviction";

cucascade::io::cache::fs_cache& checked_cache(cache_test_fixture& fixture)
{
  auto* cache = fixture.context->cache();
  REQUIRE(cache != nullptr);
  REQUIRE(cache->is_armed());
  return *cache;
}

std::uint64_t evictions_from(std::string const& summary)
{
  std::smatch match;
  static std::regex const expression{R"(global\[[^\]]*evictions=(\d+))"};
  if (!std::regex_search(summary, match, expression)) { return 0; }
  return std::stoull(match[1].str());
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

// ===========================================================================
// Synchronous reclaim (the cache side of sirius-db/sirius#1933)
// ===========================================================================

TEST_CASE("a synchronous reclaim spares subscribed chunks when idle and takes them when forced",
          "[cache][eviction][explicit][reclaim]")
{
  using access               = cucascade::io::cache::fs_cache_test_access;
  using byte_range           = cucascade::io::cache::fs_cache::byte_range;
  using cache_prepare_result = cucascade::io::cache::prepare_result;
  cache_test_fixture fixture{allocation_cache_config()};
  auto& cache       = checked_cache(fixture);
  auto const chunk  = cache.chunk_size();
  auto const source = fixture.open(object_path, 16 * chunk);
  REQUIRE(source != nullptr);
  auto const& object = source->get_io_object();
  auto const ranges  = [](std::size_t offset, std::size_t bytes) {
    return std::vector<byte_range>{
      byte_range{static_cast<std::int64_t>(offset), static_cast<std::int64_t>(bytes)}};
  };

  // A live request holds two chunks; a disposed one held two others.
  auto const live_ranges = ranges(0, 2 * chunk);
  auto live              = access::request(cache, object, live_ranges);
  REQUIRE(access::prepare(cache, live) == cache_prepare_result::prepared);
  auto const gone_ranges = ranges(4 * chunk, 2 * chunk);
  auto gone              = access::request(cache, object, gone_ranges);
  REQUIRE(access::prepare(cache, gone) == cache_prepare_result::prepared);
  auto const gone_chunks = gone.chunks();
  gone                   = {};
  // A new request wakes the evictor, which hands back the disposed request's
  // subscriptions; nothing is under pressure, so it reclaims nothing.
  auto const poke_ranges = ranges(8 * chunk, 1);
  auto const poke        = access::request(cache, object, poke_ranges);
  REQUIRE(wait_for_cache(
    [&] {
      return std::ranges::all_of(*gone_chunks,
                                 [](auto const* c) { return c->state.get_subscribers() == 0; });
    },
    2s));
  REQUIRE(cache.claimed_bytes() == 4 * chunk);
  auto const evictions_before = evictions_from(cache.summary());

  // idle takes only what no live request names.
  CHECK(cache.reclaim(reclaim_mode::idle) == 2 * chunk);
  CHECK(cache.claimed_bytes() == 2 * chunk);
  CHECK(cache.reclaim(reclaim_mode::idle) == 0);
  CHECK(cache.claimed_bytes() == 2 * chunk);

  // forced takes the live request's chunks too, a whole chunk at a time: a
  // target of one byte costs one chunk, and a target past what is left takes
  // only what is left.
  CHECK(cache.reclaim(reclaim_mode::forced, 1) == chunk);
  CHECK(cache.claimed_bytes() == chunk);
  CHECK(cache.reclaim(reclaim_mode::forced, 3 * chunk) == chunk);
  CHECK(cache.claimed_bytes() == 0);
  CHECK(cache.reclaim(reclaim_mode::forced) == 0);
  CHECK(evictions_from(cache.summary()) == evictions_before + 4);
}

TEST_CASE("a synchronous reclaim never takes a loading or pinned chunk",
          "[cache][eviction][explicit][reclaim]")
{
  using access               = cucascade::io::cache::fs_cache_test_access;
  using byte_range           = cucascade::io::cache::fs_cache::byte_range;
  using cache_prepare_result = cucascade::io::cache::prepare_result;
  cache_test_fixture fixture{allocation_cache_config()};
  auto& cache       = checked_cache(fixture);
  auto const chunk  = cache.chunk_size();
  auto const source = fixture.open(object_path, 4 * chunk);
  REQUIRE(source != nullptr);
  std::vector<byte_range> const first{byte_range{0, 1}};
  held_request fill;
  auto handle = access::request(cache, source->get_io_object(), first);
  REQUIRE(access::prepare(cache, handle) == cache_prepare_result::prepared);
  auto* const resident = handle.chunks()->front();

  std::atomic<bool> done{false};
  REQUIRE(cache.prefetch(handle, [&](cucascade::io::cache::prefetch_result) noexcept {
    done.store(true, std::memory_order_release);
  }));
  fill.request = fixture.context->backend().wait_take(2s);
  REQUIRE(fill.request != nullptr);
  // Loading: neither mode may take the buffer the fill is writing into.
  CHECK(cache.reclaim(reclaim_mode::forced) == 0);
  CHECK(cache.reclaim(reclaim_mode::idle) == 0);
  fill.complete(0x5a);
  REQUIRE(wait_for_cache([&] { return done.load(std::memory_order_acquire); }, 2s));

  // Pinned by a reader: still untouchable, whatever the mode.
  REQUIRE(resident->state.try_pin_covering(resident->offset, chunk, 0, 1));
  CHECK(cache.reclaim(reclaim_mode::forced) == 0);
  CHECK(cache.claimed_bytes() == chunk);
  std::ignore = resident->state.release_read();
  // Unpinned but still subscribed: idle spares it, forced takes it.
  CHECK(cache.reclaim(reclaim_mode::idle) == 0);
  CHECK(cache.reclaim(reclaim_mode::forced) == chunk);
  CHECK(cache.claimed_bytes() == 0);
}

TEST_CASE("a synchronous reclaim of zero bytes is a no-op", "[cache][eviction][explicit][reclaim]")
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
  CHECK(cache.reclaim(reclaim_mode::forced, 0) == 0);
  CHECK(cache.reclaim(reclaim_mode::idle, 0) == 0);
  CHECK(cache.claimed_bytes() == claimed);
}
