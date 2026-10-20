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

#include <cucascade/exec/invocable.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace {

using cucascade::exec::invocable;

/// Counts live instances and moves of a callable.
struct probe {
  static inline int live  = 0;
  static inline int moves = 0;
  int* calls;
  explicit probe(int* c) : calls(c) { ++live; }
  probe(probe&& other) noexcept : calls(other.calls)
  {
    ++live;
    ++moves;
  }
  probe(probe const&)            = delete;
  probe& operator=(probe const&) = delete;
  probe& operator=(probe&&)      = delete;
  ~probe() { --live; }
  int operator()(int v) { return ++*calls + v; }
};

/// A callable whose move may throw: always stored on the heap.
struct throwing_move {
  int value{3};
  throwing_move() = default;
  throwing_move(throwing_move&& other) noexcept(false) : value(other.value) {}
  int operator()() const { return value; }
};

int add_one(int v) { return v + 1; }

}  // namespace

TEST_CASE("invocable stores small nothrow-movable targets in place", "[exec][invocable]")
{
#ifndef CUCASCADE_USE_ABSEIL_INVOCABLE
  using cucascade::exec::detail::invocable_stores_inline;
  CHECK(invocable_stores_inline<probe>);
  CHECK(invocable_stores_inline<std::unique_ptr<int>>);
  CHECK_FALSE(invocable_stores_inline<std::array<std::byte, 64>>);
  CHECK_FALSE(invocable_stores_inline<throwing_move>);
#endif
  probe::live  = 0;
  probe::moves = 0;
  int calls    = 0;
  {
    invocable<int(int)> f{probe{&calls}};
    CHECK(f(1) == 2);
    invocable<int(int)> g = std::move(f);
    CHECK_FALSE(f);  // NOLINT(bugprone-use-after-move): moved-from is empty
    CHECK(g(10) == 12);
    CHECK(probe::live == 1);
    g = nullptr;
    CHECK(probe::live == 0);
  }
  CHECK(probe::live == 0);
}

TEST_CASE("invocable handles heap targets, move-only captures and signatures", "[exec][invocable]")
{
  std::array<std::uint64_t, 16> big{};
  big[15] = 42;
  invocable<std::uint64_t()> large{[big] { return big[15]; }};
  invocable<std::uint64_t()> moved = std::move(large);
  CHECK(moved() == 42);

  invocable<int()> throwing{throwing_move{}};
  invocable<int()> throwing_moved = std::move(throwing);
  CHECK(throwing_moved() == 3);

  auto owned = std::make_unique<std::string>("owned");
  invocable<std::string() &&> once{[p = std::move(owned)]() mutable { return std::move(*p); }};
  CHECK(std::move(once)() == "owned");

  invocable<int(int) noexcept> nothrow{[](int v) noexcept { return v * 2; }};
  CHECK(nothrow(4) == 8);

  invocable<int(int)> fn_ptr{&add_one};
  CHECK(fn_ptr(1) == 2);
  int (*null_fn)(int) = nullptr;
  invocable<int(int)> empty_fn{null_fn};
  CHECK_FALSE(empty_fn);
  CHECK(empty_fn == nullptr);

  // A target whose result is dropped by a void signature.
  int calls = 0;
  invocable<void()> discard{[&calls] { return ++calls; }};
  discard();
  CHECK(calls == 1);

  // Swap through moves keeps both targets.
  invocable<int()> a{[] { return 1; }};
  invocable<int()> b{[big] { return static_cast<int>(big[15]); }};
  invocable<int()> tmp = std::move(a);
  a                    = std::move(b);
  b                    = std::move(tmp);
  CHECK(a() == 42);
  CHECK(b() == 1);
}
