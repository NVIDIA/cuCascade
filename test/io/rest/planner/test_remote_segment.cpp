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

// The utilisation model and the producer-side sizing functions (ported from gpuio's
// tests/unit/planner/remote_segment_test.cpp; `segment_target`, `auto_coalesce_gap`):
// s3_line_rate §2.1 / §2.6 tables (cases A/B/C/X, connection counts, GET-rate), the gpuio
// ARCHITECTURE §7.1 pseudocode on the documented seeds, 6 ms TTFB seeds, hint clamping, and the
// uring_ultra cold-start pitfall (P2-P7, P10, P11).

#include "io/rest/planner/planner_remote.hpp"
#include "planner_test_support.hpp"

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace cucascade::io::rest;
using namespace cucascade::io::rest::planner;
using Catch::Approx;

namespace {

constexpr double mib = 1024.0 * 1024.0;

endpoint_estimate est(double ttfb_ms,
                      double bc,
                      double util           = 0.0,
                      unsigned conns        = 100,
                      std::uint64_t backlog = 100 * GiB)
{
  endpoint_estimate e;
  e.ttfb_p50_ms   = ttfb_ms;
  e.ttfb_p99_ms   = ttfb_ms;
  e.bc_bytes_s    = bc;
  e.link_util     = util;
  e.conn_target   = conns;
  e.backlog       = backlog;
  e.util_measured = true;  // these estimates stand for measured state (P10)
  return e;
}

/// s3_line_rate §2.1 "Request-size efficiency" table (cases A, B, C, X).
struct model_row {
  char const* name;
  double bc;                 // bytes/s
  double ttfb_ms;            // ms
  double d_mib;              // table column D
  std::array<int, 6> u_pct;  // u(s) at 1, 4, 8, 16, 32, 64 MiB, rounded
  double s85_mib;            // s at u = 85%
};

constexpr std::array<model_row, 4> model_table{{
  {"A: B_c 90 MB/s, TTFB 30 ms", 90e6, 30, 2.6, {28, 61, 76, 86, 93, 96}, 14.6},
  {"B: B_c 90 MB/s, TTFB 50 ms", 90e6, 50, 4.3, {19, 48, 65, 79, 88, 94}, 24.3},
  {"C: AnyBlob median 50 MiB/s, 30 ms", 50 * mib, 30, 1.5, {40, 73, 84, 91, 96, 98}, 8.5},
  {"X: Express 90 MB/s, 6 ms", 90e6, 6, 0.5, {66, 89, 94, 97, 98, 99}, 2.9},
}};

}  // namespace

TEST_CASE("planner.remote.model: s3_line_rate 2.6 cases A/B/C/X utilisation table",
          "[rest][planner][remote][model]")
{
  constexpr std::array<double, 6> sizes_mib{1, 4, 8, 16, 32, 64};
  for (auto const& row : model_table) {
    INFO(row.name);
    auto const e   = est(row.ttfb_ms, row.bc);
    double const d = delay_bytes(e);
    CHECK(std::abs(d / mib - row.d_mib) < 0.05);
    for (std::size_t i = 0; i < sizes_mib.size(); ++i) {
      double const u = connection_utilisation(sizes_mib[i] * mib, d);
      CHECK(std::lround(u * 100) == row.u_pct[i]);
    }
    double const s85 = utilisation_segment(d, 0.85);
    CHECK(std::abs(s85 / mib - row.s85_mib) < 0.05);
    CHECK(connection_utilisation(s85, d) == Approx(0.85));
  }
}

TEST_CASE("planner.remote.model: chosen segment sizes for cases A/B/C/X",
          "[rest][planner][remote][model]")
{
  // 10 Gbps fallback: s_rate = 1.15e9 / 2750 = 418 KB. P2: the seeded D (assumed TTFB x B_c) is
  // a floor under the measured one; a 1 ms seed keeps it below every row so the rows test s_util.
  remote_planner_config cfg{};
  cfg.assumed_ttfb_ms = 1;
  for (auto const& row : model_table) {
    INFO(row.name);
    // Large backlog, idle link: s = s_util (connection-bound regime), within [1, 32] MiB.
    auto const s_idle = segment_target(cfg, est(row.ttfb_ms, row.bc), 0);
    CHECK(std::abs(static_cast<double>(s_idle) / mib - row.s85_mib) < 0.05);
    // Saturated link (util > 0.9): never below the 4 MiB floor (only X is below it).
    auto const s_sat = segment_target(cfg, est(row.ttfb_ms, row.bc, 0.95), 0);
    CHECK(s_sat == std::max<std::uint64_t>(s_idle, 4 * MiB));
  }
  CHECK(segment_target(cfg, est(6, 90e6, 0.95), 0) == 4 * MiB);
}

TEST_CASE("planner.remote.model: s3_line_rate 2.1 connection-count table",
          "[rest][planner][remote][model]")
{
  // N = ceil(R / (B_c x u(s))) with R the raw link rate (headroom 1).
  struct row {
    double gbps;
    unsigned a16, a4, c16, c4;
  };
  constexpr std::array<row, 6> table{{{10, 17, 23, 27, 33},
                                      {25, 41, 58, 66, 82},
                                      {50, 81, 115, 131, 164},
                                      {100, 162, 229, 261, 328},
                                      {200, 323, 457, 522, 656},
                                      {400, 645, 914, 1044, 1312}}};
  double const d_a = 0.030 * 90e6;
  double const d_c = 0.030 * 50 * mib;
  for (auto const& r : table) {
    INFO(r.gbps << " Gbps");
    double const rate = r.gbps * 1e9 / 8;
    CHECK(connections_for(rate, 90e6, 16 * mib, d_a, 1.0) == r.a16);
    CHECK(connections_for(rate, 90e6, 4 * mib, d_a, 1.0) == r.a4);
    CHECK(connections_for(rate, 50 * mib, 16 * mib, d_c, 1.0) == r.c16);
    CHECK(connections_for(rate, 50 * mib, 4 * mib, d_c, 1.0) == r.c4);
  }
}

TEST_CASE("planner.remote.model: GET-rate pressure s_rate", "[rest][planner][remote][model]")
{
  // s3_line_rate §2.6 pressure 3: "with a 50% budget of 2,750 GET/s it is 4.5 MB at 100 Gbps and
  // 18 MB at 400 Gbps" (raw link rate; the planner uses the goodput target, x 0.92).
  remote_planner_config cfg{};
  cfg.goodput_fraction = 1.0;
  auto const tiny      = est(30, 75e6, 0.0, 100, 0);  // no backlog: s_par = 0, s_rate decides
  cfg.link_gbps        = 100;
  CHECK(segment_target(cfg, tiny, 0) == static_cast<std::uint64_t>(std::llround(12.5e9 / 2750)));
  cfg.link_gbps = 400;
  CHECK(segment_target(cfg, tiny, 0) == static_cast<std::uint64_t>(std::llround(50e9 / 2750)));
  cfg.goodput_fraction = 0.92;
  cfg.link_gbps        = 100;
  CHECK(target_bytes_per_s(cfg) == Approx(11.5e9));  // s3_line_rate §3.2 "11.5 GB/s"
  CHECK(segment_target(cfg, tiny, 0) == static_cast<std::uint64_t>(std::llround(11.5e9 / 2750)));
  cfg.prefix_get_rate_budget = 0;  // off
  CHECK(segment_target(cfg, tiny, 0) == cfg.min_segment_bytes);
}

TEST_CASE("planner.remote.segment: 7.1 pseudocode on the documented seeds",
          "[rest][planner][remote][segment]")
{
  remote_planner_config const cfg{};
  endpoint_estimator const ep(cfg);
  auto const seed = ep.snapshot();
  // Seeds: TTFB 30 ms, B_c 0.6 Gbps = 75 MB/s => D = 2.25 MB, s_util = 5.67 D = 12.75 MB.
  CHECK(seed.ttfb_p50_ms == 30);
  CHECK(seed.bc_bytes_s == Approx(75e6));
  CHECK(seed.link_util == 0);
  CHECK(delay_bytes(seed) == Approx(2.25e6));
  CHECK(utilisation_segment(delay_bytes(seed), cfg.target_conn_util) == Approx(12.75e6));
  // With B_c = 75 MiB/s the same numbers read in MiB: D = 2.25 MiB, s_util = 12.75 MiB.
  auto const mib_seed = est(30, 75 * mib);
  CHECK(delay_bytes(mib_seed) / mib == Approx(2.25));
  CHECK(utilisation_segment(delay_bytes(mib_seed), 0.85) / mib == Approx(12.75));
  CHECK(segment_target(cfg, mib_seed, 0) == static_cast<std::uint64_t>(12.75 * mib));
  // Seeded connection target at the 10 Gbps fallback: ceil(1.15e9 / (75e6 x 0.85) x 1.1) = 20.
  CHECK(seed.conn_target == 20);
  CHECK(max_conn_target(cfg) == 30);  // max_conns_total 0 => 1.5 x initial target

  struct row {
    char const* what;
    std::uint64_t backlog;
    double util;
    std::uint64_t hint;
    std::uint64_t expect;
  };
  std::array<row, 9> const rows{{
    {"deep backlog: s_util", 1 * GiB, 0.0, 0, 12'750'000},
    {"64 MiB backlog: s_par = 64 MiB / 20", 64 * MiB, 0.0, 0, 64 * MiB / 20},
    {"10 MiB backlog: floor 1 MiB", 10 * MiB, 0.0, 0, 1 * MiB},
    {"no backlog: floor 1 MiB", 0, 0.0, 0, 1 * MiB},
    {"saturated link: 4 MiB floor", 10 * MiB, 0.95, 0, 4 * MiB},
    {"saturated, deep backlog: s_util", 1 * GiB, 0.95, 0, 12'750'000},
    {"hint 100 MiB: max_segment_bytes", 0, 0.0, 100 * MiB, 32 * MiB},
    {"hint 1 byte: min_segment_bytes", 1 * GiB, 0.95, 1, 1 * MiB},
    {"hint 8 MiB: as given", 1 * GiB, 0.0, 8 * MiB, 8 * MiB},
  }};
  for (auto const& r : rows) {
    INFO(r.what);
    auto e      = seed;
    e.backlog   = r.backlog;
    e.link_util = r.util;
    CHECK(segment_target(cfg, e, r.hint) == r.expect);
  }
}

TEST_CASE("planner.remote.segment: 6 ms TTFB seeds", "[rest][planner][remote][segment]")
{
  // gpuio's `express_planner_config` (S3 Express, S2) is not ported; the same overrides by hand
  // keep the case: seed TTFB 6 ms, prefix GET-rate budget 100k/s, per-part cap 75.
  remote_planner_config cfg{};
  cfg.assumed_ttfb_ms        = 6;
  cfg.prefix_get_rate_budget = 100000;
  cfg.max_inflight_per_part  = 75;
  CHECK(max_inflight_per_part(cfg) == 75);
  endpoint_estimator ep(cfg);
  auto const e = ep.snapshot_for_request(1 * GiB);
  // D = 6 ms x 75 MB/s = 450 KB; s_util = 5.67 D = 2.55 MB ("~3 MiB Express", gpuio ARCHITECTURE
  // §7.1).
  CHECK(delay_bytes(e) == Approx(450e3));
  CHECK(segment_target(cfg, e, 0) == 2'550'000);
  CHECK(auto_coalesce_gap(cfg, e) == 64 * KiB);  // P10: cold = saturated: the floor
  auto measured          = e;
  measured.util_measured = true;
  CHECK(auto_coalesce_gap(cfg, measured) == 450'000);  // D x (1 - 0)
  auto sat      = measured;
  sat.link_util = 0.95;
  CHECK(segment_target(cfg, sat, 0) == 4 * MiB);
}

TEST_CASE("planner.remote.segment: the uring_ultra cold-start pitfall",
          "[rest][planner][remote][segment]")
{
  // uring_ultra dynamic_segment_target = clamp(backlog / FREE connections, 4 MiB, 16 MiB): a cold
  // 1 GiB backlog on 256 free connections plans 4 MiB GETs (48-61% utilisation, s3_line_rate
  // §3.3). the planner divides by the seeded connection TARGET and never sees the free count.
  auto const uring_ultra = [](std::uint64_t backlog, std::uint64_t free) {
    return std::clamp<std::uint64_t>(backlog / std::max<std::uint64_t>(free, 1), 4 * MiB, 16 * MiB);
  };
  REQUIRE(uring_ultra(1 * GiB, 256) == 4 * MiB);

  for (double gbps : {10.0, 25.0, 50.0, 100.0, 200.0, 400.0}) {
    for (bool controller : {true, false}) {
      remote_planner_config cfg{};
      cfg.link_gbps         = gbps;
      cfg.enable_controller = controller;
      endpoint_estimator ep(cfg);  // cold: seeds only, nothing measured
      auto const e          = ep.snapshot_for_request(1 * GiB);
      std::uint64_t const s = segment_target(cfg, e, 0);
      INFO(gbps << " Gbps: conn_target " << e.conn_target << ", s = " << s << " bytes");
      CHECK(e.backlog == 1 * GiB);
      CHECK(s > 4 * MiB);
      // The cut of that request: no 4 MiB GETs, and far fewer GETs than uring_ultra's 256.
      std::vector<file_range> chunks(remote_chunk_capacity(1 * GiB, cfg));
      std::size_t const n = cut_remote_chunks({0, 1 * GiB}, s, std::nullopt, cfg, chunks);
      REQUIRE(n > 0);
      CHECK(n < 256);
      for (std::size_t i = 0; i < n; ++i)
        CHECK(chunks[i].size > 4 * MiB);
    }
  }
  // Without the request's own bytes in the backlog a cold endpoint would plan 1 MiB GETs; that
  // is why the producer must use snapshot_for_request (documented on endpoint_estimate::backlog).
  remote_planner_config const cfg{};
  endpoint_estimator const ep(cfg);
  CHECK(segment_target(cfg, ep.snapshot(), 0) == 1 * MiB);
}

TEST_CASE("planner.remote.gap: auto_coalesce_gap = clamp(D x (1 - util), min, max)",
          "[rest][planner][remote][gap]")
{
  remote_planner_config cfg{};
  CHECK(auto_coalesce_gap(cfg, est(30, 75e6)) == 2'250'000);      // seeds: D
  CHECK(auto_coalesce_gap(cfg, est(30, 90e6, 0.8)) == 540'000);   // case A at util 0.8 ~ 512 KiB
  CHECK(auto_coalesce_gap(cfg, est(30, 90e6, 1.0)) == 64 * KiB);  // saturated: floor
  CHECK(auto_coalesce_gap(cfg, est(30, 90e6, 1.7)) == 64 * KiB);  // util > 1 read as 1
  CHECK(auto_coalesce_gap(cfg, est(200, 90e6)) == 4 * MiB);       // cap
  CHECK(auto_coalesce_gap(cfg, est(0, 0)) == 64 * KiB);           // nothing known: floor
  CHECK(auto_coalesce_gap(cfg, est(std::nan(""), 90e6)) == 64 * KiB);
  cfg.coalesce_gap_fixed = 512 * KiB;  // sirius' fixed gap
  CHECK(auto_coalesce_gap(cfg, est(30, 90e6)) == 512 * KiB);
  cfg.coalesce_gap_fixed = 0;
  auto const p           = remote_coalesce_params(cfg, est(30, 75e6), 0);
  CHECK(p.gap == 2'250'000);
  CHECK(p.max_span == 64 * MiB);
  CHECK(p.max_amplification == 1.05);  // P11: remote default (local planner keeps 1.25)
  CHECK(remote_coalesce_params(cfg, est(30, 75e6), 7).gap == 7);  // the caller's gap wins
}

TEST_CASE("planner.remote.segment: degenerate configs and estimates",
          "[rest][planner][remote][segment]")
{
  remote_planner_config cfg{};
  auto e = est(std::nan(""), -5, std::nan(""), 0, 0);
  CHECK(segment_target(cfg, e, 0) == cfg.min_segment_bytes);
  cfg.min_segment_bytes = 64 * MiB;  // inverted bounds: max wins
  cfg.max_segment_bytes = 16 * MiB;
  CHECK(segment_target(cfg, est(30, 75e6), 0) == 16 * MiB);
  CHECK(segment_target(cfg, est(30, 75e6), 1) == 16 * MiB);
  cfg                 = remote_planner_config{};
  cfg.saturated_floor = 64 * MiB;  // floor above max: still capped at max
  CHECK(segment_target(cfg, est(30, 75e6, 0.99, 100, 0), 0) == 32 * MiB);
  cfg                  = remote_planner_config{};
  cfg.target_conn_util = 1.0;  // u clamped to 0.999: huge s_util, capped at max
  CHECK(segment_target(cfg, est(30, 75e6), 0) == 32 * MiB);
}
