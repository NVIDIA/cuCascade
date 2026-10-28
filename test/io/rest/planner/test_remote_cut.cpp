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

// `cut_remote_chunks` and `recut_chunk_for_parts` (ported from gpuio's tests/unit/planner/
// remote_cut_test.cpp; gpuio ARCHITECTURE §7.1): equal split with no runt (P7), part alignment
// when P in [s/2, 2s] (P8), every piece <= max_segment_bytes, exact tiling, output sizing, hint
// acceptance, late alignment k values and the credit rule.

#include "io/rest/planner/planner_remote.hpp"
#include "planner_test_support.hpp"

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <vector>

using namespace cucascade::io::rest;
using namespace cucascade::io::rest::planner;

namespace {

std::vector<file_range> cut(file_range span,
                            std::uint64_t s,
                            std::optional<std::uint64_t> p,
                            remote_planner_config const& cfg = {})
{
  std::vector<file_range> out(remote_chunk_capacity(span.size, cfg));
  std::size_t const n = cut_remote_chunks(span, s, p, cfg, out);
  out.resize(n);
  return out;
}

std::vector<file_range> recut(file_range chunk,
                              std::uint64_t p,
                              remote_planner_config const& cfg = {})
{
  std::vector<file_range> out(remote_chunk_capacity(chunk.size, cfg));
  std::size_t const n = recut_chunk_for_parts(chunk, p, cfg, out);
  out.resize(n);
  return out;
}

bool tiles(std::vector<file_range> const& chunks, file_range span)
{
  if (chunks.empty() || chunks.front().offset != span.offset) return false;
  for (std::size_t i = 1; i < chunks.size(); ++i) {
    if (chunks[i - 1].end() != chunks[i].offset) return false;
  }
  return chunks.back().end() == span.end();
}

std::uint64_t spread(std::vector<file_range> const& chunks, std::size_t a, std::size_t b)
{
  std::uint64_t lo = std::numeric_limits<std::uint64_t>::max(), hi = 0;
  for (std::size_t i = a; i < b; ++i) {
    lo = std::min(lo, chunks[i].size);
    hi = std::max(hi, chunks[i].size);
  }
  return hi - lo;
}

/// Checks the full part-aligned contract: every multiple of P inside the span is a chunk boundary
/// except the first (dropped iff the head fragment < min) and the last (dropped iff the tail
/// fragment < min); between kept boundaries the chunks are an equal split into ceil(size / max).
void check_part_aligned(std::vector<file_range> const& chunks,
                        file_range span,
                        std::uint64_t p,
                        std::uint64_t lo,
                        std::uint64_t hi)
{
  std::uint64_t const end     = span.end();
  std::uint64_t const first_k = span.offset / p + 1;
  std::uint64_t const last_k  = (end - 1) / p;
  std::vector<std::uint64_t> kept{span.offset};
  if (first_k <= last_k) {
    bool const drop_first = first_k * p - span.offset < lo;
    bool const drop_last  = end - last_k * p < lo;
    for (std::uint64_t k = first_k; k <= last_k; ++k) {
      if (k == first_k && drop_first) continue;
      if (k == last_k && drop_last) continue;
      kept.push_back(k * p);
    }
  }
  kept.push_back(end);
  std::set<std::uint64_t> boundaries;
  for (auto const& c : chunks)
    boundaries.insert(c.offset);
  std::size_t ci = 0;
  for (std::size_t r = 0; r + 1 < kept.size(); ++r) {
    std::uint64_t const a = kept[r], b = kept[r + 1];
    REQUIRE(boundaries.count(a) == 1);
    std::uint64_t const k   = (b - a + hi - 1) / hi;  // pieces of this region
    std::size_t const begin = ci;
    while (ci < chunks.size() && chunks[ci].offset < b)
      ++ci;
    REQUIRE(ci - begin == k);
    REQUIRE(chunks[ci - 1].end() == b);
    REQUIRE(spread(chunks, begin, ci) <= 1);
  }
  REQUIRE(ci == chunks.size());
}

}  // namespace

TEST_CASE("planner.remote.cut: equal split, no runt", "[rest][planner][remote][cut]")
{
  remote_planner_config const cfg{};
  SECTION("100 MiB at 16 MiB: 7 equal pieces, not 6 x 16 + a 4 MiB runt")
  {
    auto const c = cut({0, 100 * MiB}, 16 * MiB, std::nullopt);
    REQUIRE(c.size() == 7);
    CHECK(tiles(c, {0, 100 * MiB}));
    CHECK(spread(c, 0, c.size()) <= 1);
    CHECK(c.front().size == 100 * MiB / 7 + 1);  // remainder spread from the front
  }
  SECTION("16 MiB + 1 byte at 16 MiB: two halves")
  {
    auto const c = cut({5, 16 * MiB + 1}, 16 * MiB, std::nullopt);
    REQUIRE(c.size() == 2);
    CHECK(c[0].size == 8 * MiB + 1);
    CHECK(c[1].size == 8 * MiB);
  }
  SECTION("1.5 MiB at 1 MiB: one piece (two would be below min_segment_bytes)")
  {
    auto const c = cut({0, 3 * MiB / 2}, 1 * MiB, std::nullopt);
    REQUIRE(c.size() == 1);
    CHECK(c[0].size == 3 * MiB / 2);
  }
  SECTION("span below min_segment_bytes: one chunk")
  {
    auto const c = cut({777, 100 * KiB}, 16 * MiB, std::nullopt);
    REQUIRE(c.size() == 1);
    CHECK(c[0] == file_range{777, 100 * KiB});
  }
  SECTION("s above max_segment_bytes is clamped")
  {
    auto const c = cut({0, 1 * GiB}, 1 * GiB, std::nullopt);
    REQUIRE(c.size() == 32);
    for (auto const& x : c)
      CHECK(x.size == 32 * MiB);
  }
  SECTION("s = 0 is read as min_segment_bytes")
  {
    CHECK(cut({0, 4 * MiB}, 0, std::nullopt).size() == 4);
  }
  SECTION("empty span, wrapping span and small output return 0")
  {
    std::vector<file_range> out(64);
    CHECK(cut_remote_chunks({0, 0}, 8 * MiB, std::nullopt, cfg, out) == 0);
    CHECK(cut_remote_chunks({~0ull - 10, 100}, 8 * MiB, std::nullopt, cfg, out) == 0);
    std::vector<file_range> small(3);
    CHECK(cut_remote_chunks({0, 64 * MiB}, 16 * MiB, std::nullopt, cfg, small) == 0);
    std::vector<file_range> exact(4);
    CHECK(cut_remote_chunks({0, 64 * MiB}, 16 * MiB, std::nullopt, cfg, exact) == 4);
  }
}

TEST_CASE("planner.remote.cut: segment_bytes_hint bounds", "[rest][planner][remote][cut]")
{
  remote_planner_config const cfg{};
  endpoint_estimate const e{30, 30, 75e6, 0, 20, 1 * GiB};
  auto const big = cut({0, 256 * MiB}, segment_target(cfg, e, 100 * MiB), std::nullopt);
  REQUIRE(big.size() == 8);
  for (auto const& c : big)
    CHECK(c.size == cfg.max_segment_bytes);
  auto const small = cut({0, 8 * MiB}, segment_target(cfg, e, 1), std::nullopt);
  REQUIRE(small.size() == 8);
  for (auto const& c : small)
    CHECK(c.size == cfg.min_segment_bytes);
}

TEST_CASE("planner.remote.cut: part alignment", "[rest][planner][remote][cut]")
{
  remote_planner_config cfg{};
  SECTION("P = s: cut at part boundaries, head and tail fragments kept when >= min")
  {
    file_range const span{3 * MiB, 40 * MiB};
    auto const c = cut(span, 8 * MiB, 8 * MiB);
    std::vector<file_range> const expect{{3 * MiB, 5 * MiB},
                                         {8 * MiB, 8 * MiB},
                                         {16 * MiB, 8 * MiB},
                                         {24 * MiB, 8 * MiB},
                                         {32 * MiB, 8 * MiB},
                                         {40 * MiB, 3 * MiB}};
    CHECK(c == expect);
  }
  SECTION("runt head below min merges into the next piece")
  {
    file_range const span{8 * MiB - 100 * KiB, 24 * MiB};
    auto const c = cut(span, 8 * MiB, 8 * MiB);
    REQUIRE(c.size() == 3);
    CHECK(c[0] == file_range{8 * MiB - 100 * KiB, 8 * MiB + 100 * KiB});
    CHECK(c[1] == file_range{16 * MiB, 8 * MiB});
    CHECK(c[2] == file_range{24 * MiB, 8 * MiB - 100 * KiB});
  }
  SECTION("runt tail below min merges into the previous piece")
  {
    file_range const span{0, 16 * MiB + 1};
    auto const c = cut(span, 8 * MiB, 8 * MiB);
    REQUIRE(c.size() == 2);
    CHECK(c[0] == file_range{0, 8 * MiB});
    CHECK(c[1] == file_range{8 * MiB, 8 * MiB + 1});
  }
  SECTION("P = 2s and P = s/2 are inside the window")
  {
    CHECK(cut({0, 64 * MiB}, 8 * MiB, 16 * MiB).size() == 4);  // 4 parts of 16 MiB
    CHECK(cut({0, 64 * MiB}, 8 * MiB, 4 * MiB).size() == 16);  // 16 parts of 4 MiB
  }
  SECTION("P outside [s/2, 2s]: equal split")
  {
    auto const c = cut({MiB, 64 * MiB}, 8 * MiB, 64 * MiB);
    REQUIRE(c.size() == 8);
    CHECK(spread(c, 0, c.size()) == 0);
    CHECK(cut({0, 64 * MiB}, 8 * MiB, 3 * MiB).size() == 8);
  }
  SECTION("P above max_segment_bytes: each part is split again")
  {
    auto const c = cut({0, 96 * MiB}, 32 * MiB, 48 * MiB);  // s' = 32 MiB, P = 48 in [16, 64]
    REQUIRE(c.size() == 4);
    for (auto const& x : c)
      CHECK(x.size == 24 * MiB);
  }
  SECTION("P below min_segment_bytes, unknown P, or align_to_parts off: equal split")
  {
    CHECK(cut({0, 8 * MiB}, 1 * MiB, 512 * KiB).size() == 8);  // not 16 sub-min parts
    CHECK(cut({MiB, 64 * MiB}, 8 * MiB, std::nullopt).size() == 8);
    cfg.align_to_parts = false;
    CHECK(cut({3 * MiB, 40 * MiB}, 8 * MiB, 8 * MiB, cfg).size() == 5);
  }
}

TEST_CASE("planner.remote.cut: property test (random spans, s, P, bounds)",
          "[rest][planner][remote][cut]")
{
  planner_test::splitmix64 rng{0x5E6E47};
  constexpr int trials = 20000;
  int aligned_trials   = 0;
  for (int t = 0; t < trials; ++t) {
    remote_planner_config cfg{};
    if (rng.between(0, 3) == 0) {  // other (sane: max >= 2 min) bounds
      cfg.min_segment_bytes = rng.log_between(64 * KiB, 8 * MiB);
      cfg.max_segment_bytes = cfg.min_segment_bytes * rng.between(2, 64);
    }
    std::uint64_t const lo = cfg.min_segment_bytes, hi = cfg.max_segment_bytes;
    std::uint64_t const len =
      rng.between(0, 3) == 0 ? rng.log_between(1, 2 * GiB) : rng.log_between(lo, 2 * GiB);
    file_range const span{rng.between(0, std::uint64_t{1} << 40), len};
    std::uint64_t const s = rng.log_between(1, 64 * MiB);
    // Mirror of the documented eligibility rule (s' from the equal split).
    std::uint64_t const sc = std::clamp(s, lo, hi);
    std::uint64_t n        = (span.size + sc - 1) / sc;
    n =
      std::max((span.size + hi - 1) / hi, std::min(n, std::max<std::uint64_t>(1, span.size / lo)));
    std::uint64_t const s_eq = (span.size + n - 1) / n;
    std::optional<std::uint64_t> p;
    switch (rng.between(0, 3)) {
      case 0: break;
      case 1: p = rng.log_between(256 * KiB, 256 * MiB); break;
      case 2: p = rng.between(s_eq / 2, 2 * s_eq); break;  // inside the window (if >= min)
      default: p = 5 * MiB * rng.between(1, 8); break;     // S3-like (5 MiB minimum part)
    }
    if (p && *p == 0) p.reset();
    auto const c = cut(span, s, p, cfg);
    INFO("trial " << t << " span " << span.offset << "+" << span.size << " s " << s << " P "
                  << (p ? *p : 0) << " min " << lo << " max " << hi);
    REQUIRE(!c.empty());
    REQUIRE(c.size() <= remote_chunk_capacity(span.size, cfg));
    REQUIRE(tiles(c, span));
    for (auto const& x : c) {
      REQUIRE(x.size >= 1);
      REQUIRE(x.size <= hi);
      if (span.size >= lo) REQUIRE(x.size >= lo);  // no runt unless the span itself is one
    }
    if (span.size < lo) REQUIRE(c.size() == 1);
    bool const aligned = p && *p >= lo && 2 * *p >= s_eq && *p <= 2 * s_eq;
    if (aligned) {
      ++aligned_trials;
      check_part_aligned(c, span, *p, lo, hi);
    } else {
      REQUIRE(c.size() == n);
      REQUIRE(spread(c, 0, c.size()) <= 1);  // sizes differ by at most one byte
    }
  }
  CHECK(aligned_trials > trials / 10);  // both paths exercised
}

TEST_CASE("planner.remote.recut: late part alignment k values", "[rest][planner][remote][recut]")
{
  remote_planner_config cfg{};
  struct row {
    char const* what;
    file_range chunk;
    std::uint64_t p;
    std::size_t k;
  };
  std::vector<row> const rows{
    {"chunk inside one part", {0, 8 * MiB}, 8 * MiB, 1},
    {"crosses one boundary, both sides >= min", {4 * MiB, 8 * MiB}, 8 * MiB, 2},
    {"crosses one boundary, tail fragment < min", {MiB / 2, 8 * MiB}, 8 * MiB, 1},
    {"crosses one boundary, head fragment < min", {8 * MiB - MiB / 2, 8 * MiB}, 8 * MiB, 1},
    {"16 MiB chunk over 8 MiB parts, unaligned", {4 * MiB, 16 * MiB}, 8 * MiB, 3},
    {"P = 2c: inside the window", {12 * MiB, 8 * MiB}, 16 * MiB, 2},
    {"P > 2c: outside the window", {60 * MiB, 8 * MiB}, 64 * MiB, 1},
    {"P < c/2: outside the window", {1 * MiB, 24 * MiB}, 8 * MiB, 1},
    {"P below min_segment_bytes", {0, 1 * MiB}, 512 * KiB, 1},
    {"oversized chunk (old plan), P outside window: split by max", {0, 48 * MiB}, 1 * GiB, 2},
  };
  for (auto const& r : rows) {
    INFO(r.what);
    auto const pieces = recut(r.chunk, r.p, cfg);
    CHECK(pieces.size() == r.k);
    CHECK(tiles(pieces, r.chunk));
    for (auto const& x : pieces) {
      CHECK(x.size <= cfg.max_segment_bytes);
      bool const aligned = r.p <= 2 * r.chunk.size;  // rows inside the window
      if (aligned && x.offset != r.chunk.offset) CHECK(x.offset % r.p == 0);
    }
    if (r.k == 1) CHECK(pieces.front() == r.chunk);
  }
  cfg.align_to_parts = false;
  CHECK(recut({4 * MiB, 8 * MiB}, 8 * MiB, cfg).size() == 1);
  std::vector<file_range> none;
  CHECK(recut_chunk_for_parts({4 * MiB, 8 * MiB}, 8 * MiB, cfg, none) == 0);
  cfg.align_to_parts = true;
  std::vector<file_range> one(1);
  CHECK(recut_chunk_for_parts({4 * MiB, 8 * MiB}, 8 * MiB, cfg, one) == 0);  // needs 2
}

TEST_CASE("planner.remote.recut: credit arithmetic (add k - 1 before the first piece)",
          "[rest][planner][remote][recut]")
{
  // A request planned without P: a 40 MiB span at offset 4 MiB cut into 8 MiB chunks. The first
  // response's ETag reveals P = 8 MiB, so each chunk is re-cut at the reactor (case (c)).
  remote_planner_config const cfg{};
  file_range const span{4 * MiB, 40 * MiB};
  auto const chunks = cut(span, 8 * MiB, std::nullopt);
  REQUIRE(chunks.size() == 5);

  std::uint64_t credits        = chunks.size();  // request::credits = n_chunks
  std::uint64_t settled_pieces = 0, settled_bytes = 0;
  bool fulfilled_early = false;
  for (std::size_t ci = 0; ci < chunks.size(); ++ci) {
    auto const pieces = recut(chunks[ci], 8 * MiB, cfg);
    REQUIRE(!pieces.empty());
    credits += pieces.size() - 1;  // add_credits(k - 1) while the chunk's own credit is held
    for (std::size_t pi = 0; pi < pieces.size(); ++pi) {
      REQUIRE(credits > 0);
      --credits;  // each piece settles one credit
      ++settled_pieces;
      settled_bytes += pieces[pi].size;
      bool const last = ci + 1 == chunks.size() && pi + 1 == pieces.size();
      if (credits == 0 && !last) fulfilled_early = true;
    }
  }
  CHECK_FALSE(fulfilled_early);
  CHECK(credits == 0);
  CHECK(settled_bytes == span.size);
  CHECK(settled_pieces == 10);  // every 8 MiB chunk straddles a part boundary: 4 + 4 MiB pieces

  // The wrong order (settle the first piece, then add) completes the request early when this is
  // the last outstanding chunk: why ARCHITECTURE §7.1 says "before the first piece is prepared".
  auto const pieces = recut(chunks.back(), 8 * MiB, cfg);
  REQUIRE(pieces.size() == 2);
  std::uint64_t wrong = 1;  // only the last chunk's credit is left
  --wrong;                  // first piece settles ...
  CHECK(wrong == 0);        // ... and the future would be fulfilled with a piece still in flight
}
