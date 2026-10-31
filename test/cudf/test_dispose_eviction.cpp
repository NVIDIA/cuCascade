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

#include <catch2/catch_all.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <new>
#include <optional>
#include <regex>
#include <string>
#include <vector>

namespace {
using namespace cucascade::test;
using namespace std::chrono_literals;
using cucascade::io::prepare_result;
using cucascade::io::cache::eviction_policy;

std::uint64_t evictions_from(std::string const& summary)
{
  std::smatch match;
  static std::regex const expression{R"(global\[[^\]]*evictions=(\d+))"};
  if (!std::regex_search(summary, match, expression)) { return 0; }
  return std::stoull(match[1].str());
}
}  // namespace

TEST_CASE("dispose_on_idle reclaims a disposed request's chunks on the next sweep",
          "[cache][eviction][dispose]")
{
  cache_test_fixture fixture{allocation_cache_config(eviction_policy::idle)};
  auto* cache = fixture.context->cache();
  REQUIRE(cache != nullptr);
  REQUIRE(cache->is_armed());
  {
    auto source = fixture.open("controlled://dispose", 8ULL << 20);
    REQUIRE(source != nullptr);
    advise_range(*source, 0, 4ULL << 20);
    REQUIRE(source->prepare_prefetch(false) == prepare_result::prepared);
    REQUIRE(cache->claimed_bytes() > 0);
  }
  auto other = fixture.open("controlled://dispose", 8ULL << 20);
  REQUIRE(other != nullptr);
  advise_range(*other, 6ULL << 20, 1ULL << 20);
  INFO(cache->summary());
  CHECK(wait_for_cache([&] { return evictions_from(cache->summary()) > 0; }, 2s));
}

TEST_CASE("dispose_on_idle keeps a chunk a live request still shares", "[cache][eviction][dispose]")
{
  constexpr std::size_t file_bytes = 32ULL << 20;
  cache_test_fixture fixture{allocation_cache_config(eviction_policy::idle)};
  auto* cache = fixture.context->cache();
  REQUIRE(cache != nullptr);
  REQUIRE(cache->is_armed());
  auto probe = fixture.open("controlled://shared-dispose", file_bytes);
  REQUIRE(probe != nullptr);
  advise_range(*probe, 0, 1);
  REQUIRE(probe->prepare_prefetch(false) == prepare_result::prepared);
  auto const chunk = cache->claimed_bytes();
  REQUIRE(chunk > 0);
  REQUIRE(6 * chunk <= file_bytes);
  auto first = fixture.open("controlled://shared-dispose", file_bytes);
  REQUIRE(first != nullptr);
  advise_range(*first, chunk, 2 * chunk);
  REQUIRE(first->prepare_prefetch(false) == prepare_result::prepared);
  auto second = fixture.open("controlled://shared-dispose", file_bytes);
  REQUIRE(second != nullptr);
  advise_range(*second, 2 * chunk, 2 * chunk);
  REQUIRE(second->prepare_prefetch(false) == prepare_result::prepared);
  REQUIRE(cache->claimed_bytes() == 4 * chunk);
  first.reset();
  auto untouched = fixture.open("controlled://shared-dispose", file_bytes);
  REQUIRE(untouched != nullptr);
  advise_range(*untouched, 5 * chunk, chunk);
  INFO(cache->summary());
  CHECK(wait_for_cache([&] { return cache->claimed_bytes() < 4 * chunk; }, 2s));
  CHECK(cache->claimed_bytes() == 3 * chunk);
}

TEST_CASE("a failed prefetch request hand-off rolls back only its own subscriptions",
          "[cache][cache_subscription]")
{
  // sirius-db/sirius#2314: a request that subscribed to its chunks but never
  // reached the evictor (allocating or enqueueing it failed) must drop those
  // subscriptions exactly once, and only those.  The failure is injected at
  // the hand-off; a failure part-way through chunk acquisition takes the same
  // rollback guard but has no seam to inject it.
  using access = cucascade::io::cache::fs_cache_test_access;
  using cucascade::io::cache::cache_handle;
  using byte_range = cucascade::io::cache::fs_cache::byte_range;
  cache_test_fixture fixture{allocation_cache_config(eviction_policy::idle)};
  auto* cache = fixture.context->cache();
  REQUIRE(cache != nullptr);
  REQUIRE(cache->is_armed());
  auto const chunk_bytes = cache->chunk_size();
  auto source            = fixture.open("controlled://subscription", 3 * chunk_bytes);
  auto const& object     = source->get_io_object();
  std::vector<byte_range> const first{{0, 1}};
  std::vector<byte_range> const all{{0, static_cast<std::int64_t>(3 * chunk_bytes)}};

  // A live request shares the first chunk with the one that fails.
  auto survivor = access::request(*cache, object, first);
  REQUIRE(survivor.chunks()->size() == 1);
  CHECK(survivor.chunks()->front()->state.get_subscribers() == 1);

  struct hook_reset {
    cucascade::io::cache::fs_cache& cache;
    ~hook_reset() { cache.set_before_request_enqueue_for_testing({}); }
  } reset{*cache};
  int failures = 0;
  cache->set_before_request_enqueue_for_testing([&failures] {
    ++failures;
    throw std::bad_alloc();
  });
  std::optional<cache_handle> rejected;
  CHECK_THROWS_AS(rejected.emplace(access::request(*cache, object, all)), std::bad_alloc);
  CHECK(failures == 1);
  CHECK_FALSE(rejected.has_value());
  cache->set_before_request_enqueue_for_testing({});
  // The survivor keeps its own subscription; the failed request left none.
  CHECK(survivor.chunks()->front()->state.get_subscribers() == 1);

  // A retry subscribes once to each chunk, on top of the survivor's.
  auto request = access::request(*cache, object, all);
  auto chunks  = request.chunks();
  REQUIRE(chunks->size() == 3);
  for (std::size_t i = 0; i < chunks->size(); ++i) {
    INFO("chunk " << i);
    CHECK((*chunks)[i]->state.get_subscribers() == (i == 0 ? 2U : 1U));
  }

  // Disposing the retry and driving the evictor releases exactly its
  // subscriptions: the survivor (which keeps the generation alive) still
  // holds the first chunk.
  request = {};
  cache->evict_sync(1);
  for (std::size_t i = 0; i < chunks->size(); ++i) {
    INFO("chunk " << i);
    CHECK((*chunks)[i]->state.get_subscribers() == (i == 0 ? 1U : 0U));
  }
}
