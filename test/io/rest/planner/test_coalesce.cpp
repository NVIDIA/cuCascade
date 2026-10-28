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

// `coalesce` (ported from gpuio's tests/unit/planner/coalesce_test.cpp): ordering, gap,
// amplification cap, span cap, scatter map, rejection of overlapping / empty input, and a
// randomized property test of the greedy rules (gpuio ARCHITECTURE §7.1 producer pseudocode;
// s3_line_rate §2.6). The entries carry only the logical range and the caller's index here
// (cuCascade keeps destinations in its own slices), so the scatter-map checks use `range_index`.

#include "io/rest/planner/planner_common.hpp"
#include "planner_test_support.hpp"

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

using namespace cucascade::io::rest;
using namespace cucascade::io::rest::planner;

namespace {

/// Destination entries; entry i carries range_index i.
struct dst_set {
  std::vector<dst_entry> dsts;

  explicit dst_set(std::vector<file_range> const& ranges)
  {
    for (std::size_t i = 0; i < ranges.size(); ++i) {
      dst_entry e;
      e.logical     = ranges[i];
      e.range_index = static_cast<std::uint32_t>(i);
      dsts.push_back(e);
    }
  }
};

std::vector<span_entry> run(dst_set& s, coalesce_params p)
{
  std::vector<span_entry> out(s.dsts.size());
  std::size_t const n = coalesce(s.dsts, p, out);
  out.resize(n);
  return out;
}

constexpr coalesce_params params(std::uint64_t gap, std::uint64_t span, double amp)
{
  return coalesce_params{gap, span, amp};
}

}  // namespace

TEST_CASE("planner.coalesce: adjacent ranges and holes up to the gap merge",
          "[rest][planner][coalesce]")
{
  dst_set s({{0, 100}, {100, 100}, {230, 70}, {400, 100}});
  auto const spans = run(s, params(30, 0, 10.0));
  REQUIRE(spans.size() == 2);
  CHECK(spans[0].physical == file_range{0, 300});  // adjacent + hole of 30 (== gap)
  CHECK(spans[0].first_dst == 0);
  CHECK(spans[0].n_dst == 3);
  CHECK(spans[1].physical == file_range{400, 100});  // hole of 100 > gap
  CHECK(spans[1].first_dst == 3);
  CHECK(spans[1].n_dst == 1);
}

TEST_CASE("planner.coalesce: gap 0 merges only adjacent ranges", "[rest][planner][coalesce]")
{
  dst_set s({{0, 10}, {10, 10}, {21, 10}});
  auto const spans = run(s, params(0, 0, 10.0));
  REQUIRE(spans.size() == 2);
  CHECK(spans[0].physical == file_range{0, 20});
  CHECK(spans[1].physical == file_range{21, 10});
}

TEST_CASE("planner.coalesce: read-amplification cap", "[rest][planner][coalesce]")
{
  // Two 100-byte ranges, amp 1.25: merged waste must be <= 0.25 x 200 = 50 bytes.
  dst_set ok({{0, 100}, {150, 100}});
  CHECK(run(ok, params(1000, 0, 1.25)).size() == 1);
  dst_set too_much({{0, 100}, {151, 100}});
  CHECK(run(too_much, params(1000, 0, 1.25)).size() == 2);
  // amp < 1 (and NaN) admit no waste: only adjacent ranges merge.
  dst_set adjacent({{0, 100}, {100, 100}, {201, 1}});
  CHECK(run(adjacent, params(1000, 0, 0.5)).size() == 2);
  dst_set nan({{0, 100}, {101, 100}});
  CHECK(run(nan, params(1000, 0, std::numeric_limits<double>::quiet_NaN())).size() == 2);
  // The cap is cumulative: a big useful range buys waste for a later hole (parquet column
  // chunks: 3 of 16 with a 2.7 MiB gap would triple the bytes without it; s3_line_rate §2.6).
  dst_set cumulative({{0, 1000}, {1200, 10}});
  CHECK(run(cumulative, params(1000, 0, 1.25)).size() == 1);  // 200 waste <= 0.25 x 1010
  // P11: the remote default 1.05 merges a 5% hole, not a 6% one.
  dst_set remote_ok({{0, 1000}, {1100, 1000}});  // 100 waste <= 0.05 x 2000
  CHECK(run(remote_ok, params(1000, 0, 1.05)).size() == 1);
  dst_set remote_no({{0, 1000}, {1101, 1000}});
  CHECK(run(remote_no, params(1000, 0, 1.05)).size() == 2);
}

TEST_CASE("planner.coalesce: span cap, and an oversized range stays alone",
          "[rest][planner][coalesce]")
{
  dst_set s({{0, 40}, {40, 40}, {80, 40}, {120, 500}, {620, 10}});
  auto const spans = run(s, params(100, 100, 10.0));
  REQUIRE(spans.size() == 4);
  CHECK(spans[0].physical == file_range{0, 80});
  CHECK(spans[1].physical == file_range{80, 40});
  CHECK(spans[2].physical == file_range{120, 500});  // > max_span on its own: the cutter splits it
  CHECK(spans[3].physical == file_range{620, 10});
  // max_span 0 = no cap.
  dst_set u({{0, 40}, {40, 40}, {80, 40}});
  CHECK(run(u, params(0, 0, 1.0)).size() == 1);
}

TEST_CASE("planner.coalesce: sorts unsorted input and keeps the scatter map",
          "[rest][planner][coalesce]")
{
  dst_set s({{500, 10}, {0, 10}, {20, 10}, {505 + 5, 10}, {10, 10}});
  auto const original = s.dsts;
  auto const spans    = run(s, params(10, 0, 10.0));
  REQUIRE(spans.size() == 2);
  CHECK(spans[0].physical == file_range{0, 30});
  CHECK(spans[0].n_dst == 3);
  CHECK(spans[1].physical == file_range{500, 20});
  CHECK(spans[1].n_dst == 2);
  for (std::size_t i = 1; i < s.dsts.size(); ++i) {
    CHECK(s.dsts[i - 1].logical.offset < s.dsts[i].logical.offset);
  }
  for (auto const& e : s.dsts) {  // every entry keeps its caller index
    REQUIRE(e.range_index < original.size());
    CHECK(e.logical == original[e.range_index].logical);
  }
}

TEST_CASE("planner.coalesce: invalid input and small output return 0", "[rest][planner][coalesce]")
{
  std::vector<span_entry> out(8);
  SECTION("empty") { CHECK(coalesce({}, params(0, 0, 1.0), out) == 0); }
  SECTION("overlap")
  {
    dst_set s({{0, 100}, {99, 10}});
    CHECK(coalesce(s.dsts, params(0, 0, 1.0), out) == 0);
  }
  SECTION("identical ranges (two destinations for one range)")
  {
    dst_set s({{0, 100}, {0, 100}});
    CHECK(coalesce(s.dsts, params(0, 0, 1.0), out) == 0);
  }
  SECTION("empty range")
  {
    dst_set s({{0, 100}, {200, 0}});
    CHECK(coalesce(s.dsts, params(0, 0, 1.0), out) == 0);
  }
  SECTION("range past 2^64")
  {
    constexpr auto max = std::numeric_limits<std::uint64_t>::max();
    dst_set s({{max - 10, 100}});
    CHECK(coalesce(s.dsts, params(0, 0, 1.0), out) == 0);
  }
  SECTION("output too small")
  {
    dst_set s({{0, 1}, {10, 1}, {20, 1}});
    std::vector<span_entry> small(2);
    CHECK(coalesce(s.dsts, params(0, 0, 1.0), small) == 0);
    std::vector<span_entry> exact(3);
    CHECK(coalesce(s.dsts, params(0, 0, 1.0), exact) == 3);
  }
}

TEST_CASE("planner.coalesce: property test (random ranges, gaps, caps)",
          "[rest][planner][coalesce]")
{
  planner_test::splitmix64 rng{0xC0A1E5CE};
  constexpr int trials = 3000;
  for (int t = 0; t < trials; ++t) {
    auto const n            = static_cast<std::size_t>(rng.between(1, 64));
    std::uint64_t const gap = rng.between(0, 3) == 0 ? 0 : rng.log_between(1, 1 << 20);
    std::uint64_t const cap = rng.between(0, 4) == 0 ? 0 : rng.log_between(1, 1 << 24);
    double const amps[]     = {0.5, 1.0, 1.05, 1.1, 1.25, 2.0, 8.0};
    double const amp        = amps[rng.between(0, 6)];
    std::vector<file_range> ranges;
    std::uint64_t off = rng.between(0, 1 << 30);
    for (std::size_t i = 0; i < n; ++i) {
      off += rng.between(0, 2) == 0 ? 0 : rng.log_between(1, 2 << 20);
      std::uint64_t const size = rng.log_between(1, 4 << 20);
      ranges.push_back({off, size});
      off += size;
    }
    for (std::size_t i = n; i > 1; --i) {  // shuffle
      std::swap(ranges[i - 1], ranges[static_cast<std::size_t>(rng.between(0, i - 1))]);
    }
    dst_set s(ranges);
    auto const original = s.dsts;
    auto const spans    = run(s, params(gap, cap, amp));
    INFO("trial " << t << " n=" << n << " gap=" << gap << " cap=" << cap << " amp=" << amp);
    REQUIRE(!spans.empty());

    // Sorted, scatter map intact.
    for (std::size_t i = 1; i < n; ++i) {
      REQUIRE(s.dsts[i - 1].logical.end() <= s.dsts[i].logical.offset);
    }
    for (auto const& e : s.dsts) {
      REQUIRE(e.logical == original[e.range_index].logical);
    }

    double const slack       = amp >= 1.0 ? amp - 1.0 : 0.0;
    std::uint64_t const scap = cap == 0 ? std::numeric_limits<std::uint64_t>::max() : cap;
    std::uint32_t next_dst   = 0;
    for (std::size_t k = 0; k < spans.size(); ++k) {
      span_entry const& sp = spans[k];
      REQUIRE(sp.first_dst == next_dst);  // spans partition the entries, in order
      REQUIRE(sp.n_dst >= 1);
      next_dst += sp.n_dst;
      auto const& first = s.dsts[sp.first_dst].logical;
      auto const& last  = s.dsts[sp.first_dst + sp.n_dst - 1].logical;
      REQUIRE(sp.physical.offset == first.offset);
      REQUIRE(sp.physical.end() == last.end());
      std::uint64_t useful = 0;
      for (std::uint32_t i = sp.first_dst; i < sp.first_dst + sp.n_dst; ++i) {
        useful += s.dsts[i].logical.size;
        if (i > sp.first_dst) {
          REQUIRE(s.dsts[i].logical.offset - s.dsts[i - 1].logical.end() <= gap);
        }
      }
      if (sp.n_dst > 1) {
        REQUIRE(sp.physical.size <= scap);
        REQUIRE(static_cast<double>(sp.physical.size - useful) <=
                slack * static_cast<double>(useful) + 1e-6);
      }
      if (k + 1 < spans.size()) {  // greedy: the next entry could not have joined this span
        auto const& nxt          = s.dsts[spans[k + 1].first_dst].logical;
        std::uint64_t const hole = nxt.offset - sp.physical.end();
        std::uint64_t const size = nxt.end() - sp.physical.offset;
        std::uint64_t const use  = useful + nxt.size;
        bool const could_merge =
          hole <= gap && size <= scap &&
          static_cast<double>(size - use) <= slack * static_cast<double>(use);
        REQUIRE_FALSE(could_merge);
      }
    }
    REQUIRE(next_dst == n);
  }
}
