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

// Tests for the io_config cache_level enum and its derivations: the single
// cache_level knob replaces the former use_odirect / enable_prefetch_cache
// pair, so these pin down the (O_DIRECT, prefetching-cache) mapping and the
// default.

#include <cucascade/io/config.hpp>

#include <catch2/catch.hpp>

using cucascade::io::cache_level;
using cucascade::io::io_config;
using cucascade::io::odirect_enabled;
using cucascade::io::prefetch_enabled;

TEST_CASE("cache_level maps to O_DIRECT and prefetching-cache flags", "[io_config]")
{
  // none     : O_DIRECT on,  prefetch off
  STATIC_REQUIRE(odirect_enabled(cache_level::none));
  STATIC_REQUIRE_FALSE(prefetch_enabled(cache_level::none));

  // os       : O_DIRECT off, prefetch off
  STATIC_REQUIRE_FALSE(odirect_enabled(cache_level::os));
  STATIC_REQUIRE_FALSE(prefetch_enabled(cache_level::os));

  // prefetch : O_DIRECT on,  prefetch on
  STATIC_REQUIRE(odirect_enabled(cache_level::prefetch));
  STATIC_REQUIRE(prefetch_enabled(cache_level::prefetch));
}

TEST_CASE("io_config defaults to cache_level::none (direct, uncached)", "[io_config]")
{
  io_config cfg;
  REQUIRE(cfg.caching == cache_level::none);
  // Preserves the pre-refactor defaults: O_DIRECT on, prefetching cache off.
  REQUIRE(odirect_enabled(cfg.caching));
  REQUIRE_FALSE(prefetch_enabled(cfg.caching));
}
