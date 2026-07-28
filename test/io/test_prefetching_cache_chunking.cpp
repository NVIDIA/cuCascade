/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Unit tests for the partial-chunk-caching primitives in
// cucascade::io::cache (types.hpp): the request->cache_from mapping
// (needed_cache_from), the cache_from->fill-span mapping (chunk_fill_span),
// the populated-extent predicate (chunk_covers), the subsequent-insert merge
// rule (merge_cache_from), and the lock-based entry_state machine.
//
// The central invariant these tests guard is that, for any cache_from value a
// chunk can hold, the bytes that will actually be READ into its buffer
// (chunk_fill_span) are a SUPERSET of the bytes that chunk_covers will later
// advertise as populated.  A violation of that invariant is the exact
// data-corruption class (a partially-populated chunk served as a full hit)
// that this layer must never allow.

#include <cucascade/io/cache/types.hpp>
#include <cucascade/io/types.hpp>

#include <catch2/catch.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

using cucascade::io::cache::align_down;
using cucascade::io::cache::align_up;
using cucascade::io::cache::cached_chunk;
using cucascade::io::cache::chunk_covers;
using cucascade::io::cache::chunk_fill_span;
using cucascade::io::cache::entry_state;
using cucascade::io::cache::merge_cache_from;
using cucascade::io::cache::needed_cache_from;

namespace {

constexpr std::size_t chunk_size = 1ull << 20;                    // 1 MiB, the default cache chunk
constexpr std::size_t page       = cucascade::io::IO_BLOCK_SIZE;  // 4096

// Set a chunk's populated extent the way the cache does (relaxed store).
void set_cache_from(cached_chunk& c, std::int32_t cf)
{
  c.cache_from.store(cf, std::memory_order_relaxed);
}

// The core safety invariant: everything chunk_covers() claims as populated for
// `cf` must lie inside the span chunk_fill_span() would actually read for `cf`.
// Checked page-by-page across the whole chunk.
void require_fill_superset_of_cover(std::size_t off, std::int32_t cf)
{
  cached_chunk c(off);
  set_cache_from(c, cf);
  auto const [seg_lo, seg_hi] = chunk_fill_span(off, chunk_size, cf, page);
  REQUIRE(seg_lo >= off);
  REQUIRE(seg_hi <= off + chunk_size);
  REQUIRE(seg_lo <= seg_hi);
  for (std::size_t p = off; p < off + chunk_size; p += page) {
    if (chunk_covers(c, chunk_size, p, p + page)) {
      // Any page advertised as covered must be inside the filled span.
      REQUIRE(p >= seg_lo);
      REQUIRE(p + page <= seg_hi);
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// align_up / align_down
// ---------------------------------------------------------------------------

TEST_CASE("align helpers round to page boundaries", "[cache][chunking]")
{
  REQUIRE(align_down(0, page) == 0);
  REQUIRE(align_down(1, page) == 0);
  REQUIRE(align_down(page, page) == page);
  REQUIRE(align_down(page + 1, page) == page);
  REQUIRE(align_down(5000, page) == page);  // 5000 -> 4096

  REQUIRE(align_up(0, page) == 0);
  REQUIRE(align_up(1, page) == page);
  REQUIRE(align_up(page, page) == page);
  REQUIRE(align_up(page + 1, page) == 2 * page);
  REQUIRE(align_up(5000, page) == 2 * page);  // 5000 -> 8192
}

// ---------------------------------------------------------------------------
// needed_cache_from — request pattern -> signed page-aligned extent
// ---------------------------------------------------------------------------

TEST_CASE("needed_cache_from maps read patterns to cache_from", "[cache][chunking]")
{
  SECTION("request spanning the whole chunk -> full (0)")
  {
    REQUIRE(needed_cache_from(0, chunk_size, 0, chunk_size) == 0);
  }

  SECTION("request overhanging both edges -> full (0)")
  {
    REQUIRE(needed_cache_from(0, chunk_size, 0, 4 * chunk_size) == 0);
  }

  SECTION("non-overlapping request -> 0 (no-op sentinel)")
  {
    // Entirely below and entirely above the chunk both clamp to empty.
    REQUIRE(needed_cache_from(chunk_size, chunk_size, 0, chunk_size) == 0);
  }

  SECTION("left-anchored partial -> positive, page-aligned up")
  {
    REQUIRE(needed_cache_from(0, chunk_size, 0, page) == static_cast<std::int32_t>(page));
    REQUIRE(needed_cache_from(0, chunk_size, 0, 5000) == static_cast<std::int32_t>(2 * page));
  }

  SECTION("right-anchored partial -> negative, page-aligned down")
  {
    // [1044480, chunk_end): 1044480 is page-aligned -> suffix of exactly one page.
    REQUIRE(needed_cache_from(0, chunk_size, chunk_size - page, chunk_size) ==
            -static_cast<std::int32_t>(page));
    // [1040000, chunk_end): floor-aligns 1040000 -> 1036288, suffix = 12288 (3 pages).
    REQUIRE(needed_cache_from(0, chunk_size, 1040000, chunk_size) ==
            -static_cast<std::int32_t>(3 * page));
  }

  SECTION("interior read conservatively becomes a right suffix to the chunk end")
  {
    // [524288, 532480) touches neither edge; the edge-anchored representation
    // rounds it to a right suffix running to the chunk end.
    REQUIRE(needed_cache_from(0, chunk_size, 524288, 532480) ==
            -static_cast<std::int32_t>(chunk_size - 524288));
  }

  SECTION("second chunk of a boundary-spanning read -> left prefix")
  {
    // Chunk [1MiB, 2MiB), request tail [1MiB, 1MiB+100KiB-ish).
    std::size_t const off = chunk_size;
    REQUIRE(needed_cache_from(off, chunk_size, off, off + 102400) ==
            static_cast<std::int32_t>(102400));  // already a page multiple (25 pages)
  }
}

// ---------------------------------------------------------------------------
// chunk_fill_span vs chunk_covers — the safety invariant
// ---------------------------------------------------------------------------

TEST_CASE("chunk_fill_span always covers what chunk_covers advertises", "[cache][chunking][safety]")
{
  // Full, left prefixes, right suffixes, and the interior->suffix case, at
  // both a zero offset and a shifted offset.
  for (std::size_t off : {std::size_t{0}, chunk_size, 7 * chunk_size}) {
    require_fill_superset_of_cover(off, 0);
    require_fill_superset_of_cover(off, static_cast<std::int32_t>(page));
    require_fill_superset_of_cover(off, static_cast<std::int32_t>(8 * page));
    require_fill_superset_of_cover(off, -static_cast<std::int32_t>(page));
    require_fill_superset_of_cover(off, -static_cast<std::int32_t>(3 * page));
    require_fill_superset_of_cover(off, -static_cast<std::int32_t>(chunk_size - 524288));
  }
}

TEST_CASE("chunk_fill_span produces exact edge-anchored spans", "[cache][chunking]")
{
  SECTION("full chunk reads the whole extent")
  {
    auto const [lo, hi] = chunk_fill_span(0, chunk_size, 0, page);
    REQUIRE(lo == 0);
    REQUIRE(hi == chunk_size);
  }

  SECTION("left prefix reads from offset")
  {
    auto const [lo, hi] = chunk_fill_span(0, chunk_size, static_cast<std::int32_t>(8 * page), page);
    REQUIRE(lo == 0);
    REQUIRE(hi == 8 * page);
  }

  SECTION("right suffix reads to chunk end")
  {
    auto const [lo, hi] =
      chunk_fill_span(0, chunk_size, -static_cast<std::int32_t>(3 * page), page);
    REQUIRE(lo == chunk_size - 3 * page);
    REQUIRE(hi == chunk_size);
  }

  SECTION("round-trip: needed_cache_from -> chunk_fill_span reads the requested bytes")
  {
    // A boundary-spanning read [946176, 1150976) split across two 1 MiB chunks.
    std::size_t const req_lo = 946176;   // page-aligned
    std::size_t const req_hi = 1150976;  // page-aligned

    // Chunk 0 gets the right suffix.
    std::int32_t const cf0 = needed_cache_from(0, chunk_size, req_lo, req_hi);
    auto const [lo0, hi0]  = chunk_fill_span(0, chunk_size, cf0, page);
    REQUIRE(lo0 == req_lo);  // exactly the requested bytes in chunk 0
    REQUIRE(hi0 == chunk_size);

    // Chunk 1 gets the left prefix.
    std::size_t const off1 = chunk_size;
    std::int32_t const cf1 = needed_cache_from(off1, chunk_size, req_lo, req_hi);
    auto const [lo1, hi1]  = chunk_fill_span(off1, chunk_size, cf1, page);
    REQUIRE(lo1 == off1);
    REQUIRE(hi1 == req_hi);  // exactly the requested bytes in chunk 1
  }
}

// ---------------------------------------------------------------------------
// chunk_covers — hit / miss on partially-populated chunks
// ---------------------------------------------------------------------------

TEST_CASE("chunk_covers gates hits on the populated extent", "[cache][chunking]")
{
  SECTION("full chunk covers everything")
  {
    cached_chunk c(0);
    set_cache_from(c, 0);
    REQUIRE(chunk_covers(c, chunk_size, 0, chunk_size));
    REQUIRE(chunk_covers(c, chunk_size, 12345, 54321));
  }

  SECTION("left prefix: covers the prefix, misses beyond it")
  {
    cached_chunk c(0);
    set_cache_from(c, static_cast<std::int32_t>(2 * page));  // [0, 8192)
    REQUIRE(chunk_covers(c, chunk_size, 0, page));
    REQUIRE(chunk_covers(c, chunk_size, page, 2 * page));
    REQUIRE(chunk_covers(c, chunk_size, 0, 2 * page));
    REQUIRE_FALSE(chunk_covers(c, chunk_size, 2 * page, 3 * page));  // just past the edge
    REQUIRE_FALSE(chunk_covers(c, chunk_size, 0, 3 * page));         // straddles the edge
  }

  SECTION("right suffix: covers the suffix, misses before it")
  {
    cached_chunk c(0);
    set_cache_from(c, -static_cast<std::int32_t>(2 * page));  // [chunk-8192, chunk)
    std::size_t const edge = chunk_size - 2 * page;
    REQUIRE(chunk_covers(c, chunk_size, edge, chunk_size));
    REQUIRE(chunk_covers(c, chunk_size, edge + page, chunk_size));
    REQUIRE_FALSE(chunk_covers(c, chunk_size, edge - page, edge));        // just before the edge
    REQUIRE_FALSE(chunk_covers(c, chunk_size, edge - page, chunk_size));  // straddles the edge
  }

  SECTION("regression: interior fill does not false-hit the uncovered side")
  {
    // Reproduces the interior-read scenario: a chunk loaded for [524288, ...)
    // (stored as a right suffix to the chunk end) must MISS a request for the
    // left side it never populated.
    cached_chunk c(0);
    set_cache_from(c, needed_cache_from(0, chunk_size, 524288, 532480));
    // The far-right request that shares the suffix is a legitimate hit and is
    // genuinely inside the fill span (fill == cover for this cf).
    REQUIRE(chunk_covers(c, chunk_size, 786432, 794624));
    // The left side was never read -> must miss.
    REQUIRE_FALSE(chunk_covers(c, chunk_size, 0, page));
    REQUIRE_FALSE(chunk_covers(c, chunk_size, 262144, 262144 + page));
  }
}

// ---------------------------------------------------------------------------
// merge_cache_from — subsequent-insert widening rules
// ---------------------------------------------------------------------------

namespace {

// Apply the merge the way the cache does: under the entry lock.
std::int32_t merged(std::int32_t cur, std::int32_t want)
{
  cached_chunk c(0);
  set_cache_from(c, cur);
  auto lk = c.state.get_lock();
  merge_cache_from(c, want);
  return c.cache_from.load(std::memory_order_relaxed);
}

}  // namespace

TEST_CASE("merge_cache_from widens coverage per the insert rules", "[cache][chunking]")
{
  std::int32_t const left_small  = static_cast<std::int32_t>(page);
  std::int32_t const left_big    = static_cast<std::int32_t>(4 * page);
  std::int32_t const right_small = -static_cast<std::int32_t>(page);
  std::int32_t const right_big   = -static_cast<std::int32_t>(4 * page);

  SECTION("already full stays full")
  {
    REQUIRE(merged(0, left_big) == 0);
    REQUIRE(merged(0, right_big) == 0);
  }

  SECTION("merging full request makes the chunk full")
  {
    REQUIRE(merged(left_small, 0) == 0);
    REQUIRE(merged(right_small, 0) == 0);
  }

  SECTION("same side: keep the wider extent")
  {
    REQUIRE(merged(left_small, left_big) == left_big);     // grow
    REQUIRE(merged(left_big, left_small) == left_big);     // already covered -> ignore
    REQUIRE(merged(right_small, right_big) == right_big);  // grow
    REQUIRE(merged(right_big, right_small) == right_big);  // already covered -> ignore
  }

  SECTION("opposite sides together span the chunk -> full")
  {
    REQUIRE(merged(left_small, right_small) == 0);
    REQUIRE(merged(right_big, left_big) == 0);
  }
}

// ---------------------------------------------------------------------------
// entry_state — lock-based state machine
// ---------------------------------------------------------------------------

TEST_CASE("entry_state follows the allocate/load/read lifecycle", "[cache][state_machine]")
{
  entry_state s;
  REQUIRE(s.get_state() == entry_state::empty);
  REQUIRE(s.get_pin_count() == 0);

  SECTION("happy path empty -> queued -> allocated -> loading -> cached")
  {
    REQUIRE(s.mark_queued());
    REQUIRE(s.get_state() == entry_state::queued);
    REQUIRE(s.mark_allocated());
    REQUIRE(s.get_state() == entry_state::allocated);
    REQUIRE(s.mark_loading());
    REQUIRE(s.get_state() == entry_state::loading);
    REQUIRE(s.mark_cached());
    REQUIRE(s.get_state() == entry_state::cached);
  }

  SECTION("preconditions reject out-of-order transitions")
  {
    REQUIRE_FALSE(s.mark_allocated());  // empty, not queued
    REQUIRE_FALSE(s.mark_loading());    // empty, not allocated
    REQUIRE_FALSE(s.mark_cached());     // empty, not loading
    REQUIRE(s.mark_queued());
    REQUIRE_FALSE(s.mark_queued());  // no longer empty
  }

  SECTION("read pins: acquire nests, release unwinds to cached")
  {
    REQUIRE(s.mark_queued());
    REQUIRE(s.mark_allocated());
    REQUIRE(s.mark_loading());
    REQUIRE(s.mark_cached());

    REQUIRE(s.acquire_read());
    REQUIRE(s.get_state() == entry_state::in_use);
    REQUIRE(s.get_pin_count() == 1);
    REQUIRE(s.acquire_read());
    REQUIRE(s.get_pin_count() == 2);

    REQUIRE_FALSE(s.release_read());  // still pinned
    REQUIRE(s.get_pin_count() == 1);
    REQUIRE(s.release_read());  // last reader
    REQUIRE(s.get_state() == entry_state::cached);
    REQUIRE(s.get_pin_count() == 0);
  }

  SECTION("eviction only from unpinned allocated/cached")
  {
    REQUIRE(s.mark_queued());
    REQUIRE(s.mark_allocated());
    REQUIRE(s.mark_loading());
    REQUIRE(s.mark_cached());
    REQUIRE(s.acquire_read());
    REQUIRE_FALSE(s.mark_evicting());  // pinned in_use -> rejected
    REQUIRE(s.release_read());
    REQUIRE(s.mark_evicting());  // unpinned cached -> ok
    REQUIRE(s.get_state() == entry_state::evicting);
    REQUIRE(s.mark_empty());
    REQUIRE(s.get_state() == entry_state::empty);
  }

  SECTION("load failure reverts loading -> allocated")
  {
    REQUIRE(s.mark_queued());
    REQUIRE(s.mark_allocated());
    REQUIRE(s.mark_loading());
    REQUIRE(s.mark_load_failed());
    REQUIRE(s.get_state() == entry_state::allocated);
  }

  SECTION("mark_loading_in_use pins directly out of loading")
  {
    REQUIRE(s.mark_queued());
    REQUIRE(s.mark_allocated());
    REQUIRE(s.mark_loading());
    REQUIRE(s.mark_loading_in_use());
    REQUIRE(s.get_state() == entry_state::in_use);
    REQUIRE(s.get_pin_count() == 1);
  }
}

TEST_CASE("entry_state get_lock/state_locked expose the guarded state", "[cache][state_machine]")
{
  entry_state s;
  REQUIRE(s.mark_queued());
  {
    auto lk = s.get_lock();
    REQUIRE(s.state_locked() == entry_state::queued);
  }
  REQUIRE(s.mark_allocated());
  {
    auto lk = s.get_lock();
    REQUIRE(s.state_locked() == entry_state::allocated);
  }
}

TEST_CASE("entry_state read pins are consistent under concurrent acquire/release",
          "[cache][state_machine]")
{
  // Drives the spin_lock under contention: many threads each acquire and
  // release a read pin repeatedly.  The pin count must always return to 0 and
  // the entry must settle back to `cached`, with no lost or double counts.
  entry_state s;
  REQUIRE(s.mark_queued());
  REQUIRE(s.mark_allocated());
  REQUIRE(s.mark_loading());
  REQUIRE(s.mark_cached());

  constexpr int n_threads        = 8;
  constexpr int iters_per_thread = 5000;
  // Catch2's REQUIRE is not thread-safe, so workers record violations into
  // atomics and the main thread asserts on them after the join.
  std::atomic<int> acquire_failures{0};
  std::atomic<int> state_violations{0};

  std::vector<std::thread> workers;
  workers.reserve(n_threads);
  for (int t = 0; t < n_threads; ++t) {
    workers.emplace_back([&] {
      for (int i = 0; i < iters_per_thread; ++i) {
        if (!s.acquire_read()) {
          acquire_failures.fetch_add(1, std::memory_order_relaxed);
          continue;
        }
        // Holding a pin, the entry must be readable and pinned.
        if (s.get_state() != entry_state::in_use || s.get_pin_count() < 1) {
          state_violations.fetch_add(1, std::memory_order_relaxed);
        }
        s.release_read();
      }
    });
  }
  for (auto& w : workers) {
    w.join();
  }

  // A readable entry never rejects acquire_read(); every acquire must succeed.
  REQUIRE(acquire_failures.load() == 0);
  REQUIRE(state_violations.load() == 0);
  // All pins released -> back to cached with pin count 0.
  REQUIRE(s.get_pin_count() == 0);
  REQUIRE(s.get_state() == entry_state::cached);
}
