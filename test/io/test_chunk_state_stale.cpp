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

// Write invalidation of chunk_state (STALE bit).  Header-only, no GPU.

#include <cucascade/io/cache/types.hpp>

#include <catch2/catch_all.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <tuple>
#include <vector>

namespace {

using cucascade::io::cache::chunk_fill;
using cucascade::io::cache::chunk_state;
using invalidation = chunk_state::invalidation;

constexpr std::size_t chunk_bytes = 64 * 1024;

/// Drive a fresh chunk to `allocated` with @p fill recorded.
void to_allocated(chunk_state& s, chunk_fill fill = chunk_fill::whole())
{
  REQUIRE(s.mark_queued());
  REQUIRE(s.mark_allocated());
  REQUIRE(s.merge_fill(fill));
}

void to_cached(chunk_state& s, chunk_fill fill = chunk_fill::whole())
{
  to_allocated(s, fill);
  chunk_fill out;
  REQUIRE(s.take_loading(out));
  REQUIRE(s.mark_cached());
}

}  // namespace

TEST_CASE("invalidate leaves states without readable bytes alone", "[cache][chunk_state]")
{
  chunk_state s;
  CHECK(s.invalidate() == invalidation::none);  // empty
  CHECK(s.get_state() == chunk_state::empty);

  REQUIRE(s.mark_queued());
  CHECK(s.invalidate() == invalidation::none);
  CHECK(s.get_state() == chunk_state::queued);

  REQUIRE(s.mark_allocated());
  CHECK(s.invalidate() == invalidation::none);
  CHECK(s.get_state() == chunk_state::allocated);
  CHECK_FALSE(s.load().is_stale());

  REQUIRE(s.mark_evicting());
  CHECK(s.invalidate() == invalidation::none);
  CHECK(s.get_state() == chunk_state::evicting);
  REQUIRE(s.mark_empty());
  CHECK_FALSE(s.load().is_stale());
}

TEST_CASE("invalidate drops a cached chunk but keeps its buffer and extent", "[cache][chunk_state]")
{
  chunk_state s;
  auto const fill = chunk_fill::prefix_of(3);
  to_cached(s, fill);
  s.add_subscriber();

  CHECK(s.invalidate() == invalidation::dropped);
  auto const snap = s.load();
  CHECK(snap.state() == chunk_state::allocated);
  CHECK_FALSE(snap.is_stale());
  CHECK(snap.fill() == fill);
  CHECK(snap.subscribers() == 1);
  CHECK(snap.is_reclaimable());

  // Unreadable until reloaded; the reload publishes again.
  CHECK_FALSE(s.acquire_read());
  CHECK_FALSE(s.try_pin_covering(0, chunk_bytes, 0, 4096));
  chunk_fill out;
  REQUIRE(s.take_loading(out));
  CHECK(out == fill);
  CHECK(s.mark_cached());
  CHECK(s.try_pin_covering(0, chunk_bytes, 0, 4096));
  CHECK(s.release_read());
  CHECK(s.get_state() == chunk_state::cached);

  // Idempotent.
  CHECK(s.invalidate() == invalidation::dropped);
  CHECK(s.invalidate() == invalidation::none);
}

TEST_CASE("invalidate of a loading chunk refuses to publish the load", "[cache][chunk_state]")
{
  chunk_state s;
  to_allocated(s);
  chunk_fill out;
  REQUIRE(s.take_loading(out));

  CHECK(s.invalidate() == invalidation::deferred);
  CHECK(s.invalidate() == invalidation::deferred);  // idempotent
  CHECK(s.get_state() == chunk_state::loading);
  CHECK(s.load().is_stale());

  CHECK_FALSE(s.mark_cached());  // stale bytes are not published
  auto const snap = s.load();
  CHECK(snap.state() == chunk_state::allocated);
  CHECK_FALSE(snap.is_stale());
  CHECK(snap.fill() == chunk_fill::whole());
  CHECK_FALSE(s.acquire_read());

  // A fresh load (issued after the write) publishes normally.
  REQUIRE(s.take_loading(out));
  CHECK(s.mark_cached());
  CHECK(s.acquire_read());
  CHECK(s.release_read());
}

TEST_CASE("a failed load of a stale chunk clears the flag", "[cache][chunk_state]")
{
  chunk_state s;
  to_allocated(s);
  chunk_fill out;
  REQUIRE(s.take_loading(out));
  CHECK(s.invalidate() == invalidation::deferred);
  CHECK(s.mark_load_failed());
  CHECK(s.get_state() == chunk_state::allocated);
  CHECK_FALSE(s.load().is_stale());

  REQUIRE(s.take_loading(out));
  CHECK(s.mark_cached());
  CHECK(s.get_state() == chunk_state::cached);
}

TEST_CASE("invalidate of a pinned chunk unpublishes it at the last unpin", "[cache][chunk_state]")
{
  chunk_state s;
  to_cached(s);
  REQUIRE(s.acquire_read());
  REQUIRE(s.acquire_read());

  CHECK(s.invalidate() == invalidation::deferred);
  CHECK(s.get_state() == chunk_state::in_use);
  CHECK(s.get_pin_count() == 2);
  CHECK_FALSE(s.mark_evicting(false));  // pinned chunks are never reclaimed

  // No new reader may pin the old bytes.
  CHECK_FALSE(s.acquire_read());
  CHECK_FALSE(s.try_pin_covering(0, chunk_bytes, 0, chunk_bytes));

  CHECK_FALSE(s.release_read());
  CHECK(s.get_state() == chunk_state::in_use);
  CHECK(s.load().is_stale());
  CHECK(s.release_read());
  auto const snap = s.load();
  CHECK(snap.state() == chunk_state::allocated);
  CHECK_FALSE(snap.is_stale());
  CHECK(snap.pins() == 0);
  CHECK(snap.is_reclaimable());

  // Reclaimable through the normal eviction path.
  REQUIRE(s.mark_evicting(false));
  REQUIRE(s.mark_empty());
  CHECK(s.load().fill().is_unset());
}

TEST_CASE("concurrent pins, loads and invalidations keep chunk_state consistent",
          "[cache][chunk_state]")
{
  // Every thread repeatedly pins / loads a chunk while others invalidate it.
  // Invariants: a pin only ever succeeds on a non-stale chunk, the word never
  // reaches an impossible combination, and after the dust settles the chunk is
  // in a quiescent state with no pins and no STALE bit.
  chunk_state s;
  to_cached(s);

  constexpr int n_readers      = 4;
  constexpr int n_invalidators = 2;
  constexpr int iterations     = 200000;
  std::atomic<bool> bad_pin{false};
  std::atomic<std::uint64_t> pins_taken{0};
  std::atomic<std::uint64_t> publishes{0};
  std::vector<std::thread> threads;

  for (int r = 0; r < n_readers; ++r) {
    threads.emplace_back([&] {
      for (int i = 0; i < iterations; ++i) {
        if (s.acquire_read()) {
          pins_taken.fetch_add(1, std::memory_order_relaxed);
          auto const snap = s.load();
          if (snap.state() != chunk_state::in_use || snap.pins() == 0) { bad_pin = true; }
          s.release_read();
        } else {
          chunk_fill out;
          if (s.take_loading(out)) {
            if (s.mark_cached()) { publishes.fetch_add(1, std::memory_order_relaxed); }
          }
        }
      }
    });
  }
  for (int t = 0; t < n_invalidators; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < iterations; ++i) {
        std::ignore = s.invalidate();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  CHECK_FALSE(bad_pin.load());
  CHECK(pins_taken.load() > 0);
  auto const snap = s.load();
  CHECK(snap.pins() == 0);
  CHECK_FALSE(snap.is_stale());
  CHECK((snap.state() == chunk_state::cached || snap.state() == chunk_state::allocated));
  CHECK(snap.fill() == chunk_fill::whole());
}
