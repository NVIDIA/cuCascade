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

// The cache and the cudf datasource bridge build their completion terminal as an
// `invocable<void(try_t<T>&&) &&>` before they publish I/O and install it after
// dispatch. The terminal slot must therefore take that one-shot callable by move,
// without an adapter or an allocation: an allocation that fails after dispatch
// would skip the completion bookkeeping the in-flight read depends on.

#include "utils/allocation_counter.hpp"

#include <cucascade/exec/invocable.hpp>
#include <cucascade/exec/semi_future.hpp>
#include <cucascade/exec/try.hpp>

#include <catch2/catch_all.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

using cucascade::exec::invocable;
using cucascade::exec::make_semi_future;
using cucascade::exec::promise;
using cucascade::exec::try_t;
using cucascade::test::scoped_allocation_counter;

using terminal_type = invocable<void(try_t<std::size_t>&&) &&>;

struct observed {
  std::size_t fired = 0;
  std::size_t value = 0;
  bool failed       = false;
};

terminal_type make_terminal(std::shared_ptr<observed> const& sink)
{
  return terminal_type{[sink](try_t<std::size_t>&& t) mutable {
    ++sink->fired;
    if (t.has_exception()) {
      sink->failed = true;
    } else {
      sink->value = std::move(t).value();
    }
  }};
}

}  // namespace

TEST_CASE("install_callback moves a prepared one-shot terminal in without allocating",
          "[exec][semi_future]")
{
  static_assert(
    std::is_constructible_v<cucascade::exec::detail::state<std::size_t>::callback, terminal_type&&>,
    "the terminal slot must accept the one-shot invocable flavour by move");

  auto sink = std::make_shared<observed>();

  SECTION("pending future: the terminal fires when the promise is fulfilled")
  {
    promise<std::size_t> p;
    auto future   = p.get_semi_future();
    auto terminal = make_terminal(sink);

    std::size_t allocations = 0;
    {
      scoped_allocation_counter counter;
      std::move(future).install_callback(std::move(terminal));
      allocations = counter.count();
    }
    CHECK(allocations == 0);
    CHECK(sink->fired == 0);

    p.set_value(std::size_t{42});
    CHECK(sink->fired == 1);
    CHECK(sink->value == 42);
    CHECK_FALSE(sink->failed);
  }

  SECTION("ready future: the terminal fires inline")
  {
    auto future   = make_semi_future(std::size_t{7});
    auto terminal = make_terminal(sink);

    std::size_t allocations = 0;
    {
      scoped_allocation_counter counter;
      std::move(future).install_callback(std::move(terminal));
      allocations = counter.count();
    }
    CHECK(allocations == 0);
    CHECK(sink->fired == 1);
    CHECK(sink->value == 7);
  }

  SECTION("a failed future reaches the terminal as an exception")
  {
    auto future = make_semi_future<std::size_t>(
      try_t<std::size_t>(std::make_exception_ptr(std::runtime_error("backend failed"))));
    std::move(future).install_callback(make_terminal(sink));
    CHECK(sink->fired == 1);
    CHECK(sink->failed);
  }

  SECTION("a plain lambda still installs")
  {
    promise<std::size_t> p;
    auto future      = p.get_semi_future();
    std::size_t seen = 0;
    std::move(future).install_callback(
      [&seen](try_t<std::size_t>&& t) { seen = std::move(t).value(); });
    p.set_value(std::size_t{3});
    CHECK(seen == 3);
  }
}
