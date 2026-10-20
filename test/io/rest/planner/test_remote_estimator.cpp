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

// Endpoint estimator (ported from gpuio's tests/unit/planner/remote_estimator_test.cpp; gpuio
// ARCHITECTURE §7.1 "estimator per endpoint"; s3_line_rate §3.3 `control()`): quantile sketch
// accuracy vs exact quantiles on 10^5 samples, EWMA B_c, seeds, backlog accounting (P4), the
// AIMD transitions (C2), and `enable_controller == false` pinning the seeds.

#include "io/rest/planner/planner_remote.hpp"
#include "planner_test_support.hpp"

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

using namespace cucascade::io::rest;
using namespace cucascade::io::rest::planner;
using Catch::Approx;

namespace {

double exact_quantile(std::vector<double> v, double q)
{
  std::sort(v.begin(), v.end());
  auto const rank = static_cast<std::size_t>(std::ceil(q * static_cast<double>(v.size())));
  return v[std::clamp<std::size_t>(rank, 1, v.size()) - 1];
}

}  // namespace

TEST_CASE("planner.remote.sketch: p50/p99 within 2% of exact on 10^5 samples",
          "[rest][planner][remote][estimator]")
{
  CHECK(quantile_sketch::ln_gamma == Approx(std::log(quantile_sketch::gamma)).epsilon(1e-15));
  CHECK(std::ceil(std::log(quantile_sketch::max_value / quantile_sketch::min_value) /
                  std::log(quantile_sketch::gamma)) +
          2 ==
        static_cast<double>(quantile_sketch::buckets));

  std::mt19937_64 gen(12345);
  struct dist {
    char const* name;
    std::vector<double> samples;
  };
  std::vector<dist> dists;
  {
    // TTFB-like: log-normal, median 30 ms, long tail (AnyBlob: < 5% above 200 ms).
    std::lognormal_distribution<double> d(std::log(30.0), 0.6);
    dist x{"lognormal(30 ms, 0.6)", {}};
    for (int i = 0; i < 100000; ++i)
      x.samples.push_back(d(gen));
    dists.push_back(std::move(x));
  }
  {
    std::exponential_distribution<double> d(1.0 / 50.0);
    dist x{"20 ms + exponential(50 ms)", {}};
    for (int i = 0; i < 100000; ++i)
      x.samples.push_back(20.0 + d(gen));
    dists.push_back(std::move(x));
  }
  {
    std::uniform_real_distribution<double> d(1.0, 1000.0);
    dist x{"uniform(1, 1000 ms)", {}};
    for (int i = 0; i < 100000; ++i)
      x.samples.push_back(d(gen));
    dists.push_back(std::move(x));
  }
  {
    std::lognormal_distribution<double> d(std::log(6.0), 0.3);  // Express-like
    dist x{"lognormal(6 ms, 0.3)", {}};
    for (int i = 0; i < 100000; ++i)
      x.samples.push_back(d(gen));
    dists.push_back(std::move(x));
  }
  auto sketch = std::make_unique<quantile_sketch>();
  for (auto const& d : dists) {
    INFO(d.name);
    sketch->clear();
    for (double v : d.samples)
      sketch->add(v);
    REQUIRE(sketch->count() == d.samples.size());
    for (double q : {0.5, 0.9, 0.99, 0.999}) {
      double const exact = exact_quantile(d.samples, q);
      double const est   = sketch->quantile(q);
      INFO("q " << q << " exact " << exact << " sketch " << est);
      CHECK(std::abs(est - exact) <= 0.02 * exact);
    }
  }
}

TEST_CASE("planner.remote.sketch: merge, decay, edges", "[rest][planner][remote][estimator]")
{
  auto a   = std::make_unique<quantile_sketch>();
  auto b   = std::make_unique<quantile_sketch>();
  auto all = std::make_unique<quantile_sketch>();
  planner_test::splitmix64 rng{3};
  for (int i = 0; i < 20000; ++i) {
    double const v = 1.0 + 500.0 * rng.unit();
    (i % 2 == 0 ? *a : *b).add(v);
    all->add(v);
  }
  a->merge(*b);  // reactor-local sketches merge exactly
  CHECK(a->count() == all->count());
  for (double q : {0.1, 0.5, 0.99})
    CHECK(a->quantile(q) == all->quantile(q));
  all->decay();
  CHECK(all->count() <= 10000);
  CHECK(all->count() >= 10000 - quantile_sketch::buckets);

  quantile_sketch e;
  CHECK(e.quantile(0.5) == 0.0);
  e.add(0.0);
  CHECK(e.quantile(0.5) == quantile_sketch::min_value);
  e.clear();
  e.add(1e9);
  CHECK(e.quantile(0.5) == quantile_sketch::max_value);
  e.add(std::nan(""));  // counted in the underflow bucket, never UB
  CHECK(e.count() == 2);
}

TEST_CASE("planner.remote.estimator: seeds, backlog, EWMA B_c",
          "[rest][planner][remote][estimator]")
{
  remote_planner_config cfg{};
  endpoint_estimator ep(cfg);
  auto s = ep.snapshot();
  CHECK(s.ttfb_p50_ms == 30);
  CHECK(s.ttfb_p99_ms == 30);
  CHECK(s.bc_bytes_s == Approx(75e6));
  CHECK(s.link_util == 0);
  CHECK_FALSE(s.util_measured);  // P10: the auto gap reads it as saturated
  CHECK(s.conn_target == 20);    // 10 Gbps fallback
  CHECK(s.backlog == 0);

  auto const r = ep.snapshot_for_request(1 * GiB);
  CHECK(r.backlog == 1 * GiB);
  ep.add_backlog(1 * MiB);
  CHECK(ep.backlog() == 1 * GiB + 1 * MiB);
  ep.sub_backlog(1 * GiB);
  ep.sub_backlog(2 * MiB);  // racing completions may undershoot: reads as 0
  CHECK(ep.backlog() == 0);

  ep.record_body(512 * KiB, 0.001);  // < 1 MiB: TTFB-dominated, ignored
  CHECK(ep.bc_ewma() == Approx(75e6));
  ep.record_body(100 * MiB, 1.0);  // 104.9 MB/s
  CHECK(ep.bc_ewma() == Approx(0.1 * 104857600.0 + 0.9 * 75e6));
  ep.record_body(100 * MiB, 0.0);  // invalid duration: ignored
  CHECK(ep.bc_ewma() == Approx(0.1 * 104857600.0 + 0.9 * 75e6));

  cfg.link_gbps = 100;
  CHECK(endpoint_estimator(cfg).snapshot().conn_target == 199);
  cfg.max_conns_total = 64;
  CHECK(endpoint_estimator(cfg).snapshot().conn_target == 64);
  cfg.link_gbps       = 1;
  cfg.max_conns_total = 0;
  CHECK(endpoint_estimator(cfg).snapshot().conn_target == cfg.min_conns_total);  // floor 8
}

TEST_CASE("planner.remote.estimator: enable_controller == false pins the seeds",
          "[rest][planner][remote][estimator]")
{
  remote_planner_config cfg{};
  cfg.enable_controller = false;
  endpoint_estimator ep(cfg);
  auto const before = ep.snapshot();
  for (int i = 0; i < 1000; ++i) {
    ep.record_ttfb(80);
    ep.record_body(16 * MiB, 0.1);
    ep.record_response(i % 10 == 0);  // 10% throttled
  }
  ep.record_delivered(1 * GiB);
  CHECK(ep.control(0.25) == controller_move::hold);
  auto const after = ep.snapshot();
  CHECK(after.ttfb_p50_ms == before.ttfb_p50_ms);
  CHECK(after.bc_bytes_s == before.bc_bytes_s);
  CHECK(after.link_util == before.link_util);
  CHECK(after.conn_target == before.conn_target);
}

TEST_CASE("planner.remote.estimator: control() publishes measurements",
          "[rest][planner][remote][estimator]")
{
  remote_planner_config const cfg{};  // controller on, 10 Gbps => target 1.15 GB/s
  endpoint_estimator ep(cfg);
  for (int i = 0; i < 1000; ++i) {
    ep.record_ttfb(50 + (i % 2 == 0 ? 0.0 : 0.1));
    ep.record_body(16 * MiB, 16.0 * 1048576.0 / 100e6);  // 100 MB/s bodies
    ep.record_response(false);
  }
  ep.record_delivered(static_cast<std::uint64_t>(0.5 * 1.15e9 * 0.25));  // half the target
  (void)ep.snapshot_for_request(100 * GiB);                              // deep backlog
  CHECK(ep.control(0.25) == controller_move::additive_increase);
  auto const s = ep.snapshot();
  CHECK(s.ttfb_p50_ms == Approx(50).epsilon(0.01));
  CHECK(s.bc_bytes_s > 99e6);
  CHECK(s.link_util == Approx(0.5).epsilon(0.01));
  CHECK(s.conn_target == 20 + 2);  // + max(1, 20 / 8)
  // A period with > 0.5% throttled responses: x 0.75.
  for (int i = 0; i < 200; ++i)
    ep.record_response(i < 2);
  CHECK(ep.control(0.25) == controller_move::throttle_decrease);
  CHECK(ep.snapshot().conn_target == 16);  // floor(22 x 0.75)
}

TEST_CASE("planner.remote.controller: AIMD transitions", "[rest][planner][remote][estimator]")
{
  remote_planner_config cfg{};
  cfg.link_gbps       = 100;  // target 11.5 GB/s
  cfg.max_conns_total = 400;
  double const target = target_bytes_per_s(cfg);
  controller_inputs in;
  in.target_bytes_s = target;
  in.bc_bytes_s     = 75e6;
  in.delay_bytes    = 2.25e6;
  in.backlog        = 100 * GiB;
  in.responses      = 1000;

  SECTION("throttle: x0.75 above 0.5% 503/429, floored at min_conns_total")
  {
    controller_state st;
    st.conn_target = 100;
    in.throttled   = 6;  // 0.6%
    CHECK(conn_target_step(st, in, cfg) == controller_move::throttle_decrease);
    CHECK(st.conn_target == 75);
    in.throttled       = 5;  // exactly 0.5% is not "> 0.5%"
    in.goodput_bytes_s = target;
    CHECK(conn_target_step(st, in, cfg) != controller_move::throttle_decrease);
    st.conn_target = 9;
    in.throttled   = 100;
    CHECK(conn_target_step(st, in, cfg) == controller_move::throttle_decrease);
    CHECK(st.conn_target == cfg.min_conns_total);  // floor(6.75) -> 8
  }
  SECTION("growth: + max(1, n/8) while under 0.9 x target with backlog, capped at max")
  {
    controller_state st;
    st.conn_target     = 100;
    in.goodput_bytes_s = 0.5 * target;
    CHECK(conn_target_step(st, in, cfg) == controller_move::additive_increase);
    CHECK(st.conn_target == 112);
    in.goodput_bytes_s *= 1.10;  // a strong gain keeps growing
    CHECK(conn_target_step(st, in, cfg) == controller_move::additive_increase);
    CHECK(st.conn_target == 126);
    CHECK(st.weak_increases == 0);
    controller_state small;
    small.conn_target   = 5;
    cfg.min_conns_total = 1;
    CHECK(conn_target_step(small, in, cfg) == controller_move::additive_increase);
    CHECK(small.conn_target == 6);  // +1 below 8
    controller_state top;
    top.conn_target = 395;
    CHECK(conn_target_step(top, in, cfg) == controller_move::additive_increase);
    CHECK(top.conn_target == 400);
    // Too little backlog to feed more connections: no growth, retarget instead.
    controller_state lean;
    lean.conn_target = 100;
    in.backlog       = 100 * 10 * MiB;  // < 100 x s_typ (12.16 MiB)
    CHECK(conn_target_step(lean, in, cfg) == controller_move::retarget);
  }
  SECTION("freeze after two increases that gained < 2%, cleared by a retarget")
  {
    controller_state st;
    st.conn_target     = 100;
    in.goodput_bytes_s = 0.5 * target;
    CHECK(conn_target_step(st, in, cfg) == controller_move::additive_increase);  // 112
    in.goodput_bytes_s *= 1.01;                                                  // weak #1
    CHECK(conn_target_step(st, in, cfg) == controller_move::additive_increase);  // 126
    CHECK(st.weak_increases == 1);
    in.goodput_bytes_s *= 1.01;  // weak #2 => frozen
    CHECK(conn_target_step(st, in, cfg) == controller_move::frozen_hold);
    CHECK(st.frozen);
    CHECK(st.conn_target == 126);
    CHECK(conn_target_step(st, in, cfg) == controller_move::frozen_hold);
    in.goodput_bytes_s = target;  // at target: follow the model, unfreeze
    CHECK(conn_target_step(st, in, cfg) == controller_move::retarget);
    CHECK_FALSE(st.frozen);
    // want = ceil(11.5e9 / (75e6 x 0.85) x 1.1) = 199
    CHECK(st.conn_target == 199);
  }
}
