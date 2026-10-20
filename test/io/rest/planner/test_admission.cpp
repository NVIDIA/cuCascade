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

// The planner pieces that were members of gpuio's remote backend and client: the admission cap
// and its throttle guard (C5, C6), admission waits (C7), connect-ahead (C8), the DRR weights
// (P12), the part size from an ETag (P8; gpuio tests/unit/s3/test_builder.cpp), the per-endpoint
// `endpoint_control` hand-off (C1, C4), and the request flow the REST worker runs (P4: the
// backlog sees the request before sizing and returns to zero after its GETs).

#include "io/rest/planner/planner_remote.hpp"

#include <catch2/catch_all.hpp>

#include <chrono>
#include <cstdint>
#include <limits>
#include <vector>

using namespace cucascade::io::rest;
using namespace cucascade::io::rest::planner;

namespace {
constexpr std::uint64_t ms_ns = 1'000'000ULL;
}  // namespace

TEST_CASE("planner.parts: part count and size from a multipart ETag", "[rest][planner][parts]")
{
  CHECK(etag_part_count("\"abc-12\"") == 12U);
  CHECK(etag_part_count("abc-3") == 3U);
  CHECK_FALSE(etag_part_count("\"abc\"").has_value());
  CHECK_FALSE(etag_part_count("\"abc-\"").has_value());
  CHECK_FALSE(etag_part_count("\"abc-0\"").has_value());
  CHECK_FALSE(etag_part_count("\"abc-1x\"").has_value());
  CHECK(estimate_part_size(100ULL << 20, 13) == (8ULL << 20));
  CHECK(estimate_part_size((1ULL << 30) + 1, 17) == (64ULL << 20));
  CHECK(estimate_part_size(1ULL << 30, 16) == (64ULL << 20));
  CHECK(estimate_part_size(100, 1) == 0);
  CHECK(estimate_part_size(0, 4) == 0);
  // no power-of-two MiB fits: ceil(size / parts) rounded up to a MiB
  CHECK(estimate_part_size(30ULL << 20, 3) == (10ULL << 20));
}

TEST_CASE("planner.admission: conn_cap is the reactor's share of conn_target",
          "[rest][planner][admission]")
{
  CHECK(conn_cap(20, 64, 1) == 20);
  CHECK(conn_cap(199, 64, 2) == 64);  // ceil(199 / 2) = 100, bounded by the connection limit
  CHECK(conn_cap(199, 128, 2) == 100);
  CHECK(conn_cap(7, 64, 4) == 2);  // ceil
  CHECK(conn_cap(0, 64, 1) == 1);
  CHECK(conn_cap(5, 0, 0) == 1);
  unsigned const big = std::numeric_limits<unsigned>::max();
  CHECK(conn_cap(big, big, 1) == big);  // no overflow in the ceiling
}

TEST_CASE("planner.admission: the cap binds only after a throttle (C5)",
          "[rest][planner][admission]")
{
  // gpuio's mock: 12 warm connections -> 12 GETs in flight before the first 503, 4 after.
  CHECK(admit_cap(4, true, false, false, 12) == 12);
  CHECK(admit_cap(4, true, true, false, 12) == 4);    // estimator: within the caution window
  CHECK(admit_cap(4, true, false, true, 12) == 4);    // this reactor saw a 503 this period
  CHECK(admit_cap(4, false, false, false, 12) == 4);  // controller off: conn_cap
  CHECK(admit_cap(16, true, false, false, 12) == 16);

  // The estimator's binding flag through a real throttle decrease and its caution window.
  remote_planner_config cfg{};
  endpoint_estimator ep(cfg);
  unsigned const cap0 = conn_cap(ep.snapshot().conn_target, 64, 1);
  CHECK(admit_cap(cap0, true, ep.throttle_binding(), false, 64) == 64);
  for (int i = 0; i < 100; ++i) {
    ep.record_response(i < 10);
  }
  (void)ep.control(0.25);
  REQUIRE(ep.throttle_binding());
  unsigned const cap1 = conn_cap(ep.snapshot().conn_target, 64, 1);
  CHECK(cap1 == 15);  // floor(20 x 0.75)
  CHECK(admit_cap(cap1, true, ep.throttle_binding(), false, 64) == 15);
  for (unsigned i = 0; i < throttle_caution_periods; ++i) {
    (void)ep.control(0.25);
  }
  CHECK_FALSE(ep.throttle_binding());
  CHECK(admit_cap(conn_cap(ep.snapshot().conn_target, 64, 1), true, false, false, 64) == 64);
}

TEST_CASE("planner.admission: the fast per-reactor throttle guard (C6)",
          "[rest][planner][admission]")
{
  using std::chrono::milliseconds;
  throttle_guard g;
  std::uint64_t const t0 = 1'000 * ms_ns;
  CHECK_FALSE(g.active(t0));
  CHECK_FALSE(g.holds(1000, t0));
  g.on_throttle(5, t0, milliseconds{250});
  CHECK(g.cap == 5);
  CHECK(g.active(t0 + 249 * ms_ns));
  CHECK(g.holds(5, t0 + 100 * ms_ns));
  CHECK_FALSE(g.holds(4, t0 + 100 * ms_ns));
  g.on_throttle(7, t0 + 100 * ms_ns, milliseconds{250});  // repeated: never raised in the window
  CHECK(g.cap == 5);
  g.on_throttle(3, t0 + 200 * ms_ns, milliseconds{250});  // repeated: lowered further
  CHECK(g.cap == 3);
  CHECK(g.until_ns == t0 + 450 * ms_ns);
  CHECK_FALSE(g.holds(3, t0 + 450 * ms_ns));  // one control period after the last throttle
  CHECK_FALSE(g.active(t0 + 450 * ms_ns));
  g.on_throttle(9, t0 + 500 * ms_ns, milliseconds{250});  // a new window starts at the new busy
  CHECK(g.cap == 9);
  throttle_guard z;
  z.on_throttle(0, t0, milliseconds{0});  // max(1, busy); a period of at least 1 ms
  CHECK(z.cap == 1);
  CHECK(z.until_ns == t0 + ms_ns);
}

TEST_CASE("planner.admission: over_cap, latency reserve and admission waits (C5, C7, P12)",
          "[rest][planner][admission]")
{
  remote_planner_config cfg{};
  throttle_guard g;
  std::uint64_t const now = 5'000 * ms_ns;
  CHECK_FALSE(over_cap(cfg, g, now, 63, 64, false));
  CHECK(over_cap(cfg, g, now, 64, 64, false));
  // the latency class adds max(2, ceil(5% x 64)) = 4 reserved connections on top of the cap
  CHECK_FALSE(over_cap(cfg, g, now, 67, 64, true));
  CHECK(over_cap(cfg, g, now, 68, 64, true));
  g.on_throttle(3, now, cfg.control_period);
  CHECK(over_cap(cfg, g, now, 3, 64, true));  // the guard holds even for the latency class
  CHECK_FALSE(over_cap(cfg, g, now, 2, 64, false));
  cfg.enable_controller = false;
  CHECK_FALSE(over_cap(cfg, g, now, 1000, 4, false));  // controller off: no admission cap

  CHECK(counts_as_admission_wait(4, 64));         // the controller's cap binds: probe evidence
  CHECK_FALSE(counts_as_admission_wait(64, 64));  // the hard connection limit: never
  CHECK_FALSE(counts_as_admission_wait(100, 64));
  CHECK_FALSE(counts_as_admission_wait(1, 0));
}

TEST_CASE("planner.admission: connect-ahead sized by demand and conn_cap (C8)",
          "[rest][planner][admission]")
{
  CHECK(connect_ahead_count(10, 2, 1, 3, 8) == 2);     // min(12, 8) - (2 + 1 + 3)
  CHECK(connect_ahead_count(100, 0, 0, 0, 64) == 64);  // a cold wave opens in parallel
  CHECK(connect_ahead_count(3, 0, 0, 0, 64) == 3);     // never more than the demand
  CHECK(connect_ahead_count(5, 5, 4, 2, 64) == 0);     // enough connections already
  CHECK(connect_ahead_count(0, 0, 0, 0, 64) == 0);
  constexpr auto u32 = std::numeric_limits<std::uint32_t>::max();
  CHECK(connect_ahead_count(u32, 0, 0, 0, 16) == 16);  // no overflow
  CHECK(connect_ahead_count(u32, u32, 0, 0, 16) == 0);
}

TEST_CASE("planner.fairness: DRR weights high : normal : bulk = 4 : 2 : 1 (P12)",
          "[rest][planner][fairness]")
{
  remote_scheduler_config cfg{};
  CHECK(drr_weight(priority::high, cfg) == 4);
  CHECK(drr_weight(priority::normal, cfg) == 2);
  CHECK(drr_weight(priority::bulk, cfg) == 1);
  CHECK(drr_weight(priority::urgent, cfg) == 1);  // rides the latency list; weight unused
  cfg.remote_weights = {0, 0, 0};                 // zero is read as 1
  CHECK(drr_weight(priority::high, cfg) == 1);
  CHECK(drr_quantum >= remote_planner_config{}.max_segment_bytes);
}

TEST_CASE("planner.endpoint_control: reactor hand-off and the link seed (C1, C4)",
          "[rest][planner][controller]")
{
  remote_planner_config cfg{};
  retry_config const rcfg{};
  SECTION("controller on: merge, conn_limit, one step per control period")
  {
    endpoint_control ep(cfg, rcfg);
    CHECK(ep.budget.tokens() == 500);
    estimator_samples s;
    for (int i = 0; i < 100; ++i) {
      s.record_ttfb(20);
      s.record_response(i < 10);  // 10% throttled
    }
    std::uint64_t const t0 = 1'000 * ms_ns;
    CHECK_FALSE(ep.publish(s, 64, t0).has_value());  // starts the clock
    CHECK_FALSE(ep.publish(s, 64, t0 + 100 * ms_ns).has_value());
    auto const mv = ep.publish(s, 64, t0 + 250 * ms_ns);
    REQUIRE(mv.has_value());
    CHECK(*mv == controller_move::throttle_decrease);
    CHECK(ep.estimator.steps() == 1);
    CHECK(ep.estimator.throttle_binding());
  }
  SECTION("controller off: samples are ignored, seeds stay pinned")
  {
    cfg.enable_controller = false;
    endpoint_control ep(cfg, rcfg);
    auto const before = ep.estimator.snapshot();
    estimator_samples s;
    s.record_ttfb(500);
    s.record_response(true);
    CHECK_FALSE(ep.publish(s, 64, 0).has_value());
    CHECK_FALSE(ep.publish(s, 64, 10'000 * ms_ns).has_value());
    CHECK(ep.estimator.steps() == 0);
    CHECK(ep.estimator.snapshot().conn_target == before.conn_target);
  }
  SECTION("auto link: the seed re-seeds; an explicit link wins")
  {
    endpoint_control ep(cfg, rcfg);
    CHECK(ep.estimator.snapshot().link_gbps == 10);
    ep.apply_link_seed(100);
    CHECK(ep.estimator.snapshot().link_gbps == 100);
    CHECK(ep.estimator.snapshot().conn_target == 199);
    cfg.link_gbps = 25;
    endpoint_control ex(cfg, rcfg);
    ex.apply_link_seed(100);
    CHECK(ex.estimator.snapshot().link_gbps == 25);
  }
}

TEST_CASE("planner.flow: one request through the worker's planner calls (P1-P4, P7)",
          "[rest][planner][segment]")
{
  remote_planner_config const cfg{};
  endpoint_control ep(cfg, retry_config{});
  // A 1 GiB slice on a cold endpoint: its bytes enter the backlog before sizing (P4).
  file_range const slice{0, 1 * GiB};
  auto const est = ep.estimator.snapshot_for_request(slice.size);
  CHECK(est.backlog == slice.size);
  std::uint64_t const s = segment_target(cfg, est, 0);
  CHECK(s == 12'750'000);  // s_util from the seeds, never the 1 MiB cold-start GET
  std::vector<file_range> chunks(remote_chunk_capacity(slice.size, cfg));
  std::size_t const n = cut_remote_chunks(slice, s, std::nullopt, cfg, chunks);
  REQUIRE(n == 85);  // ceil(1 GiB / 12.75 MB), equal split
  std::uint64_t sum = 0;
  for (std::size_t i = 0; i < n; ++i) {
    CHECK(chunks[i].size <= cfg.max_segment_bytes);
    CHECK(chunks[i].size >= cfg.min_segment_bytes);
    sum += chunks[i].size;
    ep.estimator.sub_backlog(chunks[i].size);  // per GET completion or drop
  }
  CHECK(sum == slice.size);
  CHECK(ep.estimator.backlog() == 0);
  // A request that never reached the backlog would have been planned at the 1 MiB floor.
  CHECK(segment_target(cfg, ep.estimator.snapshot(), 0) == 1 * MiB);
}
