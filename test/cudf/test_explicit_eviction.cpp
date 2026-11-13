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
#include <limits>
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
// A 1.5-chunk request rounds up to a reservation of exactly two chunks.
struct reserved_reclaim_fixture : cache_test_fixture {
  static cucascade::io::cache::config config()
  {
    auto cfg = allocation_cache_config();
    cfg.min_prefetching_budget_fraction =
      static_cast<double>(2 * controlled_chunk_bytes - controlled_chunk_bytes / 2) / (256 * mib);
    return cfg;
  }

  reserved_reclaim_fixture() : cache_test_fixture(config())
  {
    auto const spaces = memory->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    REQUIRE(spaces.size() == 1);
    host = spaces.front();
    REQUIRE(host->get_max_memory() == 256 * mib);
    source = open(object_path, 8 * controlled_chunk_bytes);
    REQUIRE(source != nullptr);
    advise_range(*source, 0, 4 * controlled_chunk_bytes);
    REQUIRE(source->prepare_prefetch(false) == prepare_result::prepared);
    REQUIRE(checked_cache(*this).claimed_bytes() == 4 * controlled_chunk_bytes);
  }

  cucascade::memory::memory_space const* host{nullptr};
  std::unique_ptr<cucascade::io::datasource> source;
};
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

TEST_CASE("an explicit evict of every byte frees every disposed chunk",
          "[cache][eviction][explicit]")
{
  constexpr auto all = std::numeric_limits<std::size_t>::max();
  cache_test_fixture fixture{allocation_cache_config()};
  auto& cache = checked_cache(fixture);
  // A live request keeps its chunks: the evictor takes only chunks of disposed requests, and the
  // pressure rule does not fire at this occupancy.
  auto live = fixture.open(object_path, 8 * mib);
  REQUIRE(live != nullptr);
  advise_range(*live, 4 * mib, 2 * mib);
  REQUIRE(live->prepare_prefetch(false) == prepare_result::prepared);
  auto const kept = cache.claimed_bytes();
  REQUIRE(kept == 2 * mib);
  {
    auto disposed = fixture.open(object_path, 8 * mib);
    REQUIRE(disposed != nullptr);
    advise_range(*disposed, 0, 4 * mib);
    REQUIRE(disposed->prepare_prefetch(false) == prepare_result::prepared);
  }
  REQUIRE(cache.claimed_bytes() == kept + 4 * mib);

  SECTION("synchronously")
  {
    cache.evict_sync(all);
    INFO(cache.summary());
    CHECK(cache.claimed_bytes() == kept);
  }
  SECTION("asynchronously")
  {
    cache.evict(all);
    INFO(cache.summary());
    CHECK(wait_for_cache([&] { return cache.claimed_bytes() == kept; }, 2s));
  }
  SECTION("with another demand that may join it in the same round")
  {
    // Taken in one round, the two demands must not add up past SIZE_MAX and wrap to nothing.
    cache.evict(all);
    cache.evict_sync(1);
    INFO(cache.summary());
    CHECK(cache.claimed_bytes() == kept);
  }
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
// Synchronous reclaim
// ===========================================================================

TEST_CASE("releasable bytes exclude the per-NUMA prefetch reservation",
          "[cache][eviction][explicit][reclaim]")
{
  reserved_reclaim_fixture fixture;
  auto& cache      = checked_cache(fixture);
  auto const chunk = cache.chunk_size();
  CHECK(cache.releasable_bytes() == 2 * chunk);
  CHECK(cache.releasable_bytes(fixture.host->get_device_id()) == 2 * chunk);
}

TEST_CASE("a reclaim scoped to a NUMA node with no arena takes nothing",
          "[cache][eviction][explicit][reclaim]")
{
  reserved_reclaim_fixture fixture;
  auto& cache        = checked_cache(fixture);
  auto const numa    = fixture.host->get_device_id();
  auto const claimed = cache.claimed_bytes();
  CHECK(cache.releasable_bytes(numa + 99) == 0);
  CHECK(cache.reclaim(reclaim_mode::forced, std::numeric_limits<std::size_t>::max(), numa + 99) ==
        0);
  CHECK(cache.claimed_bytes() == claimed);
}

TEST_CASE("reclaiming the releasable bytes returns only the overflow to the HOST space",
          "[cache][eviction][explicit][reclaim]")
{
  reserved_reclaim_fixture fixture;
  auto& cache          = checked_cache(fixture);
  auto const chunk     = cache.chunk_size();
  auto const available = fixture.host->get_available_memory();
  CHECK(cache.reclaim(reclaim_mode::forced, 2 * chunk, fixture.host->get_device_id()) == 2 * chunk);
  CHECK(cache.releasable_bytes() == 0);
  CHECK(cache.releasable_bytes(fixture.host->get_device_id()) == 0);
  CHECK(cache.claimed_bytes() == 2 * chunk);
  CHECK(fixture.host->get_available_memory() - available == 2 * chunk);
}

// Two configured HOST arenas: each one's reservation is its own, and a reclaim
// scoped to one node leaves the other's buffers and budget alone.  The layout
// below puts B's chunk first in the sweep, so a reclaim that ignored its node
// would reach B before any of A's chunks.

TEST_CASE("releasable bytes are summed per HOST arena, not from combined usage",
          "[cache][eviction][explicit][reclaim][two_arena]")
{
  using fixture_type   = two_arena_cache_fixture;
  constexpr auto chunk = fixture_type::chunk;
  constexpr auto a     = fixture_type::node_a;
  constexpr auto b     = fixture_type::node_b;
  fixture_type fixture;
  auto& cache = fixture.cache();
  // Before any request each arena is charged exactly its prefetch reservation.
  REQUIRE(fixture.arena(a).get_total_allocated_bytes() == 2 * chunk);
  REQUIRE(fixture.arena(b).get_total_allocated_bytes() == 2 * chunk);

  // B holds one chunk, inside its 2-chunk reservation; A holds four, two past its own.
  auto const& on_b = fixture.hold(b, 0, 1);
  auto const& on_a = fixture.hold(a, 1, 4);
  for (auto const* c : *on_b.chunks()) {
    CHECK(fixture_type::node_of(*c) == b);
  }
  for (auto const* c : *on_a.chunks()) {
    CHECK(fixture_type::node_of(*c) == a);
  }
  CHECK(cache.claimed_bytes() == 5 * chunk);

  CHECK(cache.releasable_bytes(a) == 2 * chunk);
  CHECK(cache.releasable_bytes(b) == 0);
  // Subtracting both reservations from combined usage would give 5C - 4C = C.
  CHECK(cache.releasable_bytes() == 2 * chunk);
  // The HOST spaces agree: A is charged two chunks past its reservation, B nothing past its own.
  CHECK(fixture.available(a) == (fixture_type::arena_chunks - 4) * chunk);
  CHECK(fixture.available(b) == (fixture_type::arena_chunks - 2) * chunk);

  // A reclaim with no node takes the chunks of both arenas and returns each buffer to the arena
  // that supplied it: all four to A, which gets back the two past its reservation, and B's one to
  // B, which gets no budget back.
  auto const a_available = fixture.available(a);
  auto const b_available = fixture.available(b);
  auto const a_free      = fixture.arena(a).get_free_blocks();
  auto const b_free      = fixture.arena(b).get_free_blocks();
  CHECK(cache.reclaim(reclaim_mode::forced, std::numeric_limits<std::size_t>::max()) == 5 * chunk);
  CHECK(fixture.available(a) - a_available == 2 * chunk);
  CHECK(fixture.available(b) == b_available);
  CHECK(fixture.arena(a).get_free_blocks() == a_free + 4);
  CHECK(fixture.arena(b).get_free_blocks() == b_free + 1);
  CHECK(cache.claimed_bytes() == 0);
  CHECK(cache.releasable_bytes() == 0);
}

TEST_CASE("reclaiming one arena's excess frees only that arena's HOST memory",
          "[cache][eviction][explicit][reclaim][two_arena]")
{
  using fixture_type   = two_arena_cache_fixture;
  using chunk_state    = cucascade::io::cache::chunk_state;
  constexpr auto chunk = fixture_type::chunk;
  constexpr auto a     = fixture_type::node_a;
  constexpr auto b     = fixture_type::node_b;
  constexpr auto all   = std::numeric_limits<std::size_t>::max();
  fixture_type fixture;
  auto& cache       = fixture.cache();
  auto const& on_b  = fixture.hold(b, 0, 1);
  auto const& on_a  = fixture.hold(a, 1, 4);
  auto* const b_one = on_b.chunks()->front();
  auto* const b_buf = b_one->data;
  REQUIRE(b_buf != nullptr);
  REQUIRE(b_one->state.get_state() == chunk_state::allocated);
  auto const a_available = fixture.available(a);
  auto const b_available = fixture.available(b);
  auto const b_free      = fixture.arena(b).get_free_blocks();
  auto const held_on_a   = [&] {
    return std::ranges::count_if(*on_a.chunks(), [&](auto const* c) {
      return c->data != nullptr && fixture_type::node_of(*c) == a &&
             c->state.get_state() == chunk_state::allocated;
    });
  };

  // A's excess, scoped to A.
  REQUIRE(cache.releasable_bytes(a) == 2 * chunk);
  CHECK(cache.reclaim(reclaim_mode::forced, cache.releasable_bytes(a), a) == 2 * chunk);
  CHECK(fixture.available(a) - a_available == 2 * chunk);
  CHECK(fixture.available(b) == b_available);
  CHECK(cache.releasable_bytes(a) == 0);
  CHECK(cache.releasable_bytes(b) == 0);
  CHECK(cache.releasable_bytes() == 0);
  CHECK(cache.claimed_bytes() == 3 * chunk);
  CHECK(held_on_a() == 2);
  // B's chunk keeps its buffer, its state and its subscriber; B's pool is untouched.
  CHECK(b_one->data == b_buf);
  CHECK(fixture_type::node_of(*b_one) == b);
  CHECK(b_one->state.get_state() == chunk_state::allocated);
  CHECK(b_one->state.get_subscribers() == 1);
  CHECK(fixture.arena(b).get_free_blocks() == b_free);

  // B's chunk sat inside B's reservation: returning it gives B's pool a block back and the
  // HOST space no budget.  A is not touched by B's reclaim.
  CHECK(cache.reclaim(reclaim_mode::forced, all, b) == chunk);
  CHECK(b_one->state.get_state() == chunk_state::empty);
  CHECK(fixture.arena(b).get_free_blocks() == b_free + 1);
  CHECK(fixture.available(b) == b_available);
  CHECK(fixture.available(a) - a_available == 2 * chunk);
  CHECK(held_on_a() == 2);
  CHECK(cache.claimed_bytes() == 2 * chunk);

  // A's last two chunks are inside A's reservation as well.
  CHECK(cache.reclaim(reclaim_mode::forced, all, a) == 2 * chunk);
  CHECK(fixture.available(a) - a_available == 2 * chunk);
  CHECK(fixture.available(b) == b_available);
  CHECK(cache.claimed_bytes() == 0);
}

TEST_CASE("buffers from the fallback arena are charged to, and reclaimed from, that arena",
          "[cache][eviction][explicit][reclaim][two_arena]")
{
  using fixture_type   = two_arena_cache_fixture;
  constexpr auto chunk = fixture_type::chunk;
  constexpr auto a     = fixture_type::node_a;
  constexpr auto b     = fixture_type::node_b;
  constexpr auto all   = std::numeric_limits<std::size_t>::max();
  fixture_type fixture;
  auto& cache = fixture.cache();

  // B holds one chunk inside its reservation; then the rest of B's HOST budget is taken outside
  // the cache.
  auto const& own = fixture.hold(b, 0, 1);
  fixture.exhaust(b);
  auto const a_available = fixture.available(a);

  // Four more chunks for B: its reservation has room for one, so B cannot supply the request
  // and the pool takes all four from A.
  auto const& spilled = fixture.hold(b, 1, 4);
  auto const on_a     = [&] {
    return std::ranges::count_if(*spilled.chunks(), [&](auto const* c) {
      return c->data != nullptr && fixture_type::node_of(*c) == a;
    });
  };
  CHECK(on_a() == 4);
  CHECK(fixture_type::node_of(*own.chunks()->front()) == b);
  // Two of the four fallback chunks fit A's reservation; only the other two cost A budget.
  CHECK(a_available - fixture.available(a) == 2 * chunk);
  CHECK(fixture.available(b) == 0);
  CHECK(cache.releasable_bytes(a) == 2 * chunk);
  CHECK(cache.releasable_bytes(b) == 0);
  CHECK(cache.releasable_bytes() == 2 * chunk);

  // Scoped to B, a reclaim takes only B's own chunk, not the ones a request for B got from A,
  // and returning it credits B nothing past its reservation.
  CHECK(cache.reclaim(reclaim_mode::forced, all, b) == chunk);
  CHECK(fixture.available(b) == 0);
  CHECK(on_a() == 4);
  CHECK(cache.releasable_bytes(a) == 2 * chunk);

  // Scoped to A, it takes the fallback buffers and A gets its two excess chunks back.
  CHECK(cache.reclaim(reclaim_mode::forced, cache.releasable_bytes(a), a) == 2 * chunk);
  CHECK(fixture.available(a) == a_available);
  CHECK(fixture.available(b) == 0);
  CHECK(on_a() == 2);
  CHECK(cache.releasable_bytes() == 0);
  CHECK(cache.claimed_bytes() == 2 * chunk);
}

// The node check of a node-scoped reclaim must be about the buffer the reclaim claims.  The sweep
// pauses after it accepted a node-A chunk and before claiming it; meanwhile the real evictor
// empties the slot and a real preparation refills it with a node-B buffer.  That buffer must
// survive the A-scoped sweep and must not count toward A's target.
TEST_CASE("a node-scoped reclaim spares another node's buffer put in a slot before its claim",
          "[cache][eviction][explicit][reclaim][two_arena]")
{
  using fixture_type         = two_arena_cache_fixture;
  using chunk_state          = cucascade::io::cache::chunk_state;
  using cached_chunk         = cucascade::io::cache::cached_chunk;
  using cache_prepare_result = cucascade::io::cache::prepare_result;
  constexpr auto chunk       = fixture_type::chunk;
  constexpr auto a           = fixture_type::node_a;
  constexpr auto b           = fixture_type::node_b;
  // A's request covers slots s < t: the sweep reaches s first, and t is what lets the evictor
  // take s at all (see below).
  constexpr std::size_t s = 0;
  constexpr std::size_t t = 1;
  static_assert(s < t);
  fixture_type fixture;
  auto& cache = fixture.cache();

  // Fill B's prefetch reservation, so any further buffer from B costs B budget.
  fixture.hold(b, 20, 2);

  auto a_request = fixture.request(a, s, t - s + 1);
  REQUIRE(fixture.prepare(a_request) == cache_prepare_result::prepared);
  auto* const slot = a_request.chunks()->front();
  REQUIRE(slot->offset == s * chunk);
  REQUIRE(slot->data != nullptr);
  REQUIRE(fixture_type::node_of(*slot) == a);
  // B names s too, but is prepared only once s is empty: prepared now, it would find s holding
  // A's buffer, attach nothing, and never attach anything later.
  auto b_request = fixture.request(b, s, 1);
  REQUIRE(b_request.chunks()->front() == slot);
  // A's consumer is done with its request; B's still subscribes to s.  The evictor releases A's
  // subscriptions on its next round, but frees nothing until it is asked to.
  a_request = {};

  auto const a_available = fixture.available(a);
  auto const b_available = fixture.available(b);
  auto const claimed     = cache.claimed_bytes();
  REQUIRE(claimed == 4 * chunk);

  // Runs while the sweep holds the generation lock, after its node check accepted s and before
  // its claim.  It must not throw (the sweep is noexcept), so it only records what it saw.
  struct {
    bool fired{false};
    bool threw{false};
    bool emptied{false};
    cache_prepare_result prepared{cache_prepare_result::unavailable};
    chunk_state::value state{chunk_state::empty};
    int node{-1};
    std::uint8_t* data{nullptr};
  } seen;
  cache.set_before_reclaim_claim_for_testing([&](cached_chunk const& c) {
    if (&c != slot || seen.fired) { return; }
    seen.fired = true;
    try {
      // The evictor releases A's subscriptions; its subscriber-respecting pass takes t (B names
      // s), and its last-resort pass takes s.  Without t, every chunk of A would be named by B
      // once A's subscriptions were released, so the evictor would retire A before its passes
      // and never evict s.  The last-resort pass runs only for a demand the first pass (t alone)
      // cannot meet, so 2 chunks is the least that empties s.
      cache.evict_sync(2 * chunk);
      seen.emptied = slot->state.get_state() == chunk_state::empty && slot->data == nullptr;
      // B's preparation finds s empty and attaches a buffer from B's arena.
      seen.prepared = fixture.prepare(b_request);
      seen.state    = slot->state.get_state();
      seen.node     = fixture_type::node_of(*slot);
      seen.data     = slot->data;
    } catch (...) {
      seen.threw = true;
    }
  });
  auto const reclaimed = cache.reclaim(reclaim_mode::forced, chunk, a);
  cache.set_before_reclaim_claim_for_testing({});

  // The interleaving happened: the sweep paused on s, which was emptied and then refilled from B.
  REQUIRE(seen.fired);
  REQUIRE_FALSE(seen.threw);
  REQUIRE(seen.emptied);
  REQUIRE(seen.prepared == cache_prepare_result::prepared);
  REQUIRE(seen.state == chunk_state::allocated);
  REQUIRE(seen.node == b);
  REQUIRE(seen.data != nullptr);

  // B's buffer survived the A-scoped sweep, in the state B's preparation left it, still named by
  // B's request, and was not counted toward A's target.
  CHECK(reclaimed == 0);
  CHECK(slot->data == seen.data);
  CHECK(fixture_type::node_of(*slot) == b);
  CHECK(slot->state.get_state() == chunk_state::allocated);
  CHECK(slot->state.get_pin_count() == 0);
  CHECK(slot->state.get_subscribers() == 1);
  // A's two chunks went back inside A's reservation; B's buffer on s is charged to B and stays so.
  CHECK(fixture.available(a) == a_available);
  CHECK(fixture.available(b) == b_available - chunk);
  CHECK(cache.claimed_bytes() == claimed - chunk);
}

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

TEST_CASE("a resident chunk forcibly reclaimed during a prefetch is reported uncovered",
          "[cache][eviction][explicit][reclaim][prefetch-coverage]")
{
  bool reclaim = false;
  SECTION("resident chunk is reclaimed before completion") { reclaim = true; }
  SECTION("resident chunk remains available") {}

  using access               = cucascade::io::cache::fs_cache_test_access;
  using byte_range           = cucascade::io::cache::fs_cache::byte_range;
  using cache_prepare_result = cucascade::io::cache::prepare_result;
  using cucascade::io::cache::chunk_state;
  using cucascade::io::cache::prefetch_result;
  cache_test_fixture fixture{allocation_cache_config()};
  auto& cache       = checked_cache(fixture);
  auto const n      = cache.chunk_size();
  auto const source = fixture.open("controlled://coverage", 4 * n);
  REQUIRE(source != nullptr);

  std::vector<byte_range> const first_range{byte_range{0, 1}};
  auto first_handle = access::request(cache, source->get_io_object(), first_range);
  REQUIRE(access::prepare(cache, first_handle) == cache_prepare_result::prepared);
  prefetch_result first_result;
  std::atomic<bool> first_done{false};
  held_request first;
  REQUIRE(cache.prefetch(first_handle, [&](prefetch_result result) noexcept {
    first_result = result;
    first_done.store(true, std::memory_order_release);
  }));
  first.request = fixture.context->backend().wait_take(2s);
  REQUIRE(first.request != nullptr);
  first.complete(0x5a);
  REQUIRE(wait_for_cache([&] { return first_done.load(std::memory_order_acquire); }, 2s));
  REQUIRE(first_result.ok);
  REQUIRE(first_result.covered);

  std::vector<byte_range> const both_ranges{byte_range{0, 1},
                                            byte_range{static_cast<std::int64_t>(n), 1}};
  auto second_handle = access::request(cache, source->get_io_object(), both_ranges);
  REQUIRE(access::prepare(cache, second_handle) == cache_prepare_result::prepared);
  prefetch_result second_result;
  std::atomic<bool> second_done{false};
  held_request second;
  REQUIRE(cache.prefetch(second_handle, [&](prefetch_result result) noexcept {
    second_result = result;
    second_done.store(true, std::memory_order_release);
  }));
  second.request = fixture.context->backend().wait_take(2s);
  REQUIRE(second.request != nullptr);
  if (reclaim) {
    CHECK(cache.reclaim(reclaim_mode::forced) == n);
    CHECK(second_handle.chunks()->front()->state.get_state() == chunk_state::empty);
  }
  second.complete(0x7b);
  REQUIRE(wait_for_cache([&] { return second_done.load(std::memory_order_acquire); }, 2s));
  CHECK(second_result.ok);
  CHECK(second_result.covered == !reclaim);
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
