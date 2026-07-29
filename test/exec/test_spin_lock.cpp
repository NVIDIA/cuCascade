/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
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
 *
 * ---------------------------------------------------------------------------
 * ATTRIBUTION
 *
 * The correctness and try-lock tests below are ports of folly's SpinLockTest
 * (folly/test/SpinLockTest.cpp, Meta Platforms, Inc., Apache-2.0), adapted to
 * cucascade::exec::spin_lock and the Catch2 framework.  Because Catch2's
 * assertion macros are not thread-safe, worker threads record invariant
 * violations into atomics and the main thread asserts on them after joining,
 * in place of folly's in-thread gtest EXPECT_* calls.
 * ---------------------------------------------------------------------------
 */

#include <cucascade/exec/spin_lock.hpp>

#include <catch2/catch_all.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <random>
#include <thread>
#include <type_traits>
#include <vector>

using cucascade::exec::spin_lock;
using cucascade::exec::spin_lock_array;
using cucascade::exec::spin_lock_guard;

namespace {

std::size_t next_pow_two(std::size_t v) noexcept
{
  std::size_t p = 1;
  while (p < v) {
    p <<= 1;
  }
  return p;
}

std::size_t worker_count() noexcept
{
  unsigned hw = std::thread::hardware_concurrency();
  return hw == 0 ? 4u : static_cast<std::size_t>(hw);
}

// ---------------------------------------------------------------------------
// Correctness: N threads mutate a shared array under the lock; while holding
// the lock every element must be identical (no torn interleaving), then the
// holder memsets the array to a fresh random byte.
// ---------------------------------------------------------------------------

struct locked_val {
  static constexpr std::size_t k_len = 1024;
  int ar[k_len];
  // spin_lock is a POD with no constructor; value-initialize to the FREE state.
  spin_lock lock{};

  locked_val() { std::memset(ar, 0, sizeof(ar)); }
};

void spinlock_test_thread(std::size_t nthrs, locked_val* v, std::atomic<int>* violations)
{
  std::size_t const max = (1u << 16) / next_pow_two(nthrs);
  std::mt19937 rng(std::random_device{}());
  for (std::size_t i = 0; i < max; i++) {
    cucascade::exec::detail::spin_cpu_relax();
    std::unique_lock<spin_lock> g(v->lock);

    // Invariant under mutual exclusion: all elements equal ar[0].
    int const first = v->ar[0];
    for (std::size_t j = 0; j < locked_val::k_len; j++) {
      if (v->ar[j] != first) {
        violations->fetch_add(1, std::memory_order_relaxed);
        break;
      }
    }

    int const byte = static_cast<int>(rng() & 0xffu);
    std::memset(v->ar, byte, sizeof(v->ar));
  }
}

// ---------------------------------------------------------------------------
// TryLock: threads contend on lock2 via try_lock() while serializing bookkeeping
// under lock1.  Exactly one thread may hold lock2 at a time (locked flips), and
// every successful acquire waits for at least one other thread to fail before
// releasing, so failures must accumulate.
// ---------------------------------------------------------------------------

struct try_lock_state {
  spin_lock lock1{};
  spin_lock lock2{};
  bool locked{false};
  std::uint64_t obtained{0};
  std::uint64_t failed{0};
};

void trylock_test_thread(try_lock_state* state, std::uint64_t count, std::atomic<int>* violations)
{
  while (true) {
    cucascade::exec::detail::spin_cpu_relax();
    bool ret = state->lock2.try_lock();
    std::unique_lock<spin_lock> g(state->lock1);
    if (state->obtained >= count) {
      if (ret) { state->lock2.unlock(); }
      break;
    }

    if (ret) {
      // We got lock2 — no other thread must believe it holds it.
      if (state->locked) { violations->fetch_add(1, std::memory_order_relaxed); }
      ++state->obtained;
      state->locked = true;

      // Release lock1 and wait until at least one other thread fails to obtain
      // lock2 before continuing.
      auto old_failed = state->failed;
      while (state->failed == old_failed && state->obtained < count) {
        state->lock1.unlock();
        cucascade::exec::detail::spin_cpu_relax();
        state->lock1.lock();
      }

      state->locked = false;
      state->lock2.unlock();
    } else {
      ++state->failed;
    }
  }
}

}  // namespace

TEST_CASE("spin_lock basic acquire/release and try_lock", "[spin_lock][exec]")
{
  // Zero-initialized spin_lock starts FREE (documented POD contract).
  spin_lock lock{};
  REQUIRE(lock.try_lock());        // acquired
  REQUIRE_FALSE(lock.try_lock());  // already held -> fails
  lock.unlock();
  REQUIRE(lock.try_lock());  // free again
  lock.unlock();

  SECTION("spin_lock_guard (std::lock_guard alias) is RAII")
  {
    {
      spin_lock_guard g(lock);
      REQUIRE_FALSE(lock.try_lock());
    }
    REQUIRE(lock.try_lock());  // released by guard destructor
    lock.unlock();
  }
}

TEST_CASE("spin_lock is a POD / trivially usable in packed structs", "[spin_lock][exec]")
{
  STATIC_REQUIRE(std::is_standard_layout<spin_lock>::value);
  STATIC_REQUIRE(std::is_trivial<spin_lock>::value);
}

TEST_CASE("spin_lock Correctness under contention", "[spin_lock][exec]")
{
  std::size_t const nthrs = worker_count() * std::size_t{2};
  std::atomic<int> violations{0};
  locked_val v;

  std::vector<std::thread> threads;
  threads.reserve(nthrs);
  for (std::size_t i = 0; i < nthrs; ++i) {
    threads.emplace_back(spinlock_test_thread, nthrs, &v, &violations);
  }
  for (auto& t : threads) {
    t.join();
  }

  REQUIRE(violations.load() == 0);
}

TEST_CASE("spin_lock TryLock ping-pong", "[spin_lock][exec]")
{
  std::size_t const nthrs   = worker_count() + std::size_t{4};
  std::uint64_t const count = 100;
  std::atomic<int> violations{0};
  try_lock_state state;

  std::vector<std::thread> threads;
  threads.reserve(nthrs);
  for (std::size_t i = 0; i < nthrs; ++i) {
    threads.emplace_back(trylock_test_thread, &state, count, &violations);
  }
  for (auto& t : threads) {
    t.join();
  }

  REQUIRE(violations.load() == 0);
  REQUIRE(state.obtained == count);
  // Every successful acquire waits for another thread to fail, except possibly
  // the very last one when no other threads remain.
  REQUIRE(state.failed + 1u >= state.obtained);
}

TEST_CASE("spin_lock_array shards lock independently", "[spin_lock][exec]")
{
  spin_lock_array<spin_lock, 8> locks;
  REQUIRE(locks.size() == 8);

  // Each shard is independent: locking one leaves the others free.
  REQUIRE(locks[3].try_lock());
  REQUIRE(locks[4].try_lock());
  REQUIRE_FALSE(locks[3].try_lock());
  locks[3].unlock();
  REQUIRE(locks[3].try_lock());
  locks[3].unlock();
  locks[4].unlock();
}
