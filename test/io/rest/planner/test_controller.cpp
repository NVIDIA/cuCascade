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

// The periodic controller step over merged reactor-local samples (ported from gpuio's
// tests/unit/planner/test_controller.cpp; `estimator_samples`, `endpoint_estimator::merge` /
// `control_at`): seeds until the first step, AIMD convergence against a simulated throttling
// server (no storm, recovery), link discovery (C3), the EC2 NIC table and the link seed (C4), the
// admission cap's binding window (C5), segment sizing at the tail (P3) and part alignment never
// extending a read (P8). gpuio ARCHITECTURE §7.1. Set `print_traces` to print the traces.
// gpuio's hedge cases (X8) and its "allocates nothing" case are not ported.

#include "io/rest/planner/planner_remote.hpp"
#include "planner_test_support.hpp"

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

using namespace cucascade::io::rest;
using namespace cucascade::io::rest::planner;
using Catch::Approx;

namespace {

constexpr std::uint64_t ms = 1'000'000ULL;

/// Set to true to print the controller traces (gpuio read an environment variable; the port keeps
/// no environment knob for it).
constexpr bool print_traces = false;

bool trace() noexcept { return print_traces; }

bool seeds_equal(endpoint_estimate const& a, endpoint_estimate const& b) noexcept
{
  return a.ttfb_p50_ms == b.ttfb_p50_ms && a.ttfb_p99_ms == b.ttfb_p99_ms &&
         a.bc_bytes_s == b.bc_bytes_s && a.link_util == b.link_util &&
         a.conn_target == b.conn_target;
}

}  // namespace

TEST_CASE("planner.remote.controller: estimator_samples merge exactly",
          "[rest][planner][remote][controller]")
{
  // Four "reactors" record disjoint shares of the same samples; merged == recorded in one place.
  auto one  = std::make_unique<estimator_samples>();
  auto agg  = std::make_unique<estimator_samples>();
  auto part = std::make_unique<std::array<estimator_samples, 4>>();
  planner_test::splitmix64 rng{7};
  for (int i = 0; i < 20000; ++i) {
    double const t        = 1.0 + static_cast<double>(rng() % 100000) / 1000.0;  // 1..101 ms
    std::uint64_t const b = (1 + rng() % 16) * MiB;
    double const secs     = static_cast<double>(b) / (50e6 + static_cast<double>(rng() % 50) * 1e6);
    bool const thr        = rng() % 100 == 0;
    auto& p               = (*part)[static_cast<std::size_t>(i % 4)];
    for (estimator_samples* s : {one.get(), &p}) {
      s->record_ttfb(t);
      s->record_body(b, secs);
      s->record_body(512 * KiB, 0.001);  // below 1 MiB: ignored by both
      s->record_response(thr);
      s->record_delivered(b);
    }
  }
  for (auto const& p : *part)
    agg->merge(p);
  CHECK(agg->ttfb.count() == one->ttfb.count());
  CHECK(agg->ttfb.quantile(0.5) == one->ttfb.quantile(0.5));
  CHECK(agg->ttfb.quantile(0.99) == one->ttfb.quantile(0.99));
  CHECK(agg->bodies == one->bodies);
  CHECK(agg->body_rate_sum == Approx(one->body_rate_sum));
  CHECK(agg->responses == one->responses);
  CHECK(agg->throttled == one->throttled);
  CHECK(agg->delivered == one->delivered);
  agg->clear();
  CHECK(agg->empty());
  CHECK(agg->ttfb.count() == 0);
}

TEST_CASE("planner.remote.controller: merge moves B_c like sequential EWMA steps",
          "[rest][planner][remote][controller]")
{
  remote_planner_config const cfg{};
  endpoint_estimator seq(cfg), batch(cfg);
  auto s = std::make_unique<estimator_samples>();
  for (int i = 0; i < 7; ++i) {
    seq.record_body(32 * MiB, 32.0 * 1048576.0 / 200e6);  // 200 MB/s
    s->record_body(32 * MiB, 32.0 * 1048576.0 / 200e6);
  }
  batch.merge(*s);
  CHECK(batch.bc_ewma() == Approx(seq.bc_ewma()).epsilon(1e-12));
  CHECK(batch.bc_ewma() > 75e6);
}

TEST_CASE("planner.remote.controller: snapshots equal the seeds until the first control step",
          "[rest][planner][remote][controller]")
{
  remote_planner_config cfg{};
  cfg.control_period = std::chrono::milliseconds(250);
  endpoint_estimator ep(cfg);
  endpoint_estimate const seeds = ep.snapshot();
  auto s                        = std::make_unique<estimator_samples>();
  for (int i = 0; i < 1000; ++i) {
    s->record_ttfb(5);
    s->record_body(8 * MiB, 0.01);    // 839 MB/s
    s->record_response(i % 20 == 0);  // 5% throttled
    s->record_delivered(8 * MiB);
  }
  std::uint64_t const t0 = 1'000 * ms;
  ep.merge(*s);
  CHECK_FALSE(ep.control_at(t0).has_value());  // starts the clock only
  CHECK(seeds_equal(ep.snapshot(), seeds));
  CHECK_FALSE(ep.control_at(t0 + 249 * ms).has_value());  // not due yet
  CHECK(seeds_equal(ep.snapshot(), seeds));
  CHECK(ep.steps() == 0);
  auto const mv = ep.control_at(t0 + 250 * ms);
  REQUIRE(mv.has_value());
  CHECK(*mv == controller_move::throttle_decrease);  // 5% > 0.5%
  CHECK(ep.steps() == 1);
  endpoint_estimate const after = ep.snapshot();
  CHECK(after.ttfb_p50_ms == Approx(5).epsilon(0.01));
  CHECK(after.conn_target == 15);  // floor(20 x 0.75)
  CHECK(after.bc_bytes_s > seeds.bc_bytes_s);
  CHECK_FALSE(ep.control_at(t0 + 400 * ms).has_value());  // 150 ms after the step
  CHECK(ep.control_at(t0 + 500 * ms).has_value());

  SECTION("enable_controller == false: merged samples and steps never publish")
  {
    cfg.enable_controller = false;
    endpoint_estimator off(cfg);
    endpoint_estimate const s0 = off.snapshot();
    off.merge(*s);
    (void)off.control_at(t0);
    auto const m = off.control_at(t0 + 300 * ms);
    REQUIRE(m.has_value());
    CHECK(*m == controller_move::hold);
    CHECK(seeds_equal(off.snapshot(), s0));
  }
}

namespace {

/// A server that throttles every response beyond `capacity` concurrent connections, with
/// `per_conn` responses per connection per period at B_c 75 MB/s.
struct throttling_server {
  unsigned capacity;
  unsigned per_conn{4};
  controller_inputs period(unsigned conns, double target) const noexcept
  {
    controller_inputs in;
    in.target_bytes_s  = target;
    in.bc_bytes_s      = 75e6;
    in.delay_bytes     = 30e-3 * 75e6;
    in.backlog         = 100 * GiB;
    in.responses       = std::uint64_t{conns} * per_conn;
    in.throttled       = conns > capacity ? std::uint64_t{conns - capacity} * per_conn : 0;
    in.goodput_bytes_s = static_cast<double>(std::min(conns, capacity)) * 75e6;
    return in;
  }
};

}  // namespace

TEST_CASE("planner.remote.controller: AIMD converges under simulated throttling without a storm",
          "[rest][planner][remote][controller]")
{
  remote_planner_config cfg{};  // 10 Gbps: target 1.15 GB/s, seed 20, max 30
  cfg.min_conns_total = 2;
  double const target = target_bytes_per_s(cfg);
  controller_state st;
  st.conn_target = seed_conn_target(cfg);
  REQUIRE(st.conn_target == 20);
  throttling_server srv{12};
  std::vector<unsigned> trace_ct;
  std::uint64_t responses = 0, throttled = 0, throttled_periods = 0;
  unsigned hi = 0, lo = ~0u;
  // Phase 1: 100 periods (25 s at 250 ms) against a server that takes 12 connections.
  for (int i = 0; i < 100; ++i) {
    controller_inputs const in = srv.period(st.conn_target, target);
    responses += in.responses;
    throttled += in.throttled;
    throttled_periods += in.throttled != 0;
    (void)conn_target_step(st, in, cfg);
    trace_ct.push_back(st.conn_target);
    if (i >= 10) {
      hi = std::max(hi, st.conn_target);
      lo = std::min(lo, st.conn_target);
    }
  }
  // Converged: a bounded sawtooth around the capacity (AIMD), never back at the seed.
  CHECK(hi <= srv.capacity + 1);
  CHECK(lo >= static_cast<unsigned>(std::floor((srv.capacity + 1) * 0.75)));
  double const share = static_cast<double>(throttled) / static_cast<double>(responses);
  CHECK(share < 0.03);                  // throttled responses stay a few percent
  CHECK(throttled_periods <= 100 / 3);  // at most one period in three sees any 503
  // Phase 2: the throttling stops; the target recovers additively to the model's want (20); once
  // the caution window has passed, a server that sustains more than the seeded 10 Gbps lets the
  // link discovery (C3) climb further, never past the discovered ceiling.
  srv.capacity     = 1000;
  int recovered_at = -1;
  for (int i = 0; i < 40; ++i) {
    (void)conn_target_step(st, srv.period(st.conn_target, target), cfg);
    trace_ct.push_back(st.conn_target);
    CHECK(st.conn_target <= max_conn_target(cfg, st.target_scale, 0));
    if (recovered_at < 0 && st.conn_target >= 20) recovered_at = i;
  }
  CHECK(recovered_at >= 0);
  CHECK(recovered_at <= 12);
  CHECK(st.conn_target >= 20);
  if (trace()) {
    std::printf("AIMD trace (conn_target per 250 ms period; capacity 12 for 100, then free):");
    for (std::size_t i = 0; i < trace_ct.size(); ++i)
      std::printf("%s%u", i % 25 == 0 ? "\n  " : " ", trace_ct[i]);
    std::printf("\n  throttled share %.4f, throttled periods %llu/100, recovered after %d\n",
                share,
                static_cast<unsigned long long>(throttled_periods),
                recovered_at + 1);
  }
}

namespace {

/// A link-limited server: every connection moves `per_conn` bytes/s, the link `link` bytes/s in
/// all; it throttles nothing. One controller step per 250 ms period.
struct link_server {
  double per_conn, link;
  controller_inputs period(unsigned conns, double target, unsigned limit) const noexcept
  {
    controller_inputs in;
    in.target_bytes_s  = target;
    in.bc_bytes_s      = per_conn;
    in.delay_bytes     = 1e-3 * per_conn;  // TTFB 1 ms
    in.backlog         = 100 * GiB;
    in.responses       = std::uint64_t{conns} * 4;
    in.goodput_bytes_s = std::min(static_cast<double>(conns) * per_conn, link);
    in.conn_limit      = limit;
    return in;
  }
};

/// Run @p periods steps; @return the first period whose goodput is >= 90% of the link (-1).
int run_link(controller_state& st,
             link_server const& srv,
             remote_planner_config const& cfg,
             int periods,
             unsigned limit,
             std::vector<double>* goodput_out,
             std::vector<unsigned>* ct_out)
{
  int reached = -1;
  for (int i = 0; i < periods; ++i) {
    controller_inputs const in = srv.period(st.conn_target, target_bytes_per_s(cfg), limit);
    if (reached < 0 && in.goodput_bytes_s >= 0.9 * srv.link) reached = i;
    goodput_out->push_back(in.goodput_bytes_s);
    (void)conn_target_step(st, in, cfg);
    ct_out->push_back(st.conn_target);
  }
  return reached;
}

}  // namespace

TEST_CASE("planner.remote.controller: link discovery climbs past the seeded 10 Gbps",
          "[rest][planner][remote][controller]")
{
  remote_planner_config const cfg{};  // seed: 10 Gbps, B_c 75 MB/s => 20 connections
  unsigned const limit = 64 * 4;      // max_connections_per_reactor x 4 live reactors

  SECTION("100 Gbps link at 75 MB/s per connection: >= 90% within 2 s, then steady")
  {
    link_server const srv{75e6, 100e9 / 8};
    controller_state st;
    std::vector<double> gp;
    std::vector<unsigned> ct;
    int const reached = run_link(st, srv, cfg, 48, limit, &gp, &ct);
    CHECK(reached >= 0);
    CHECK(reached <= 8);  // 8 periods of 250 ms
    for (std::size_t i = static_cast<std::size_t>(std::max(reached, 0)); i < gp.size(); ++i)
      CHECK(gp[i] >= 0.9 * srv.link);  // stays there (and the server throttles nothing)
    CHECK(ct.back() <= limit);
    CHECK(st.target_scale > 8.0);
    if (trace()) {
      std::printf("C3 100 Gbps link, 10 Gbps seed (conn_target / goodput GB/s per period):\n ");
      for (std::size_t i = 0; i < 16; ++i)
        std::printf(" %u/%.1f", ct[i], gp[i] / 1e9);
      std::printf("\n  reached 90%% at period %d; scale %.2f\n", reached, st.target_scale);
    }
  }
  SECTION("loopback-style: 50 Gbps server, 300 MB/s per connection")
  {
    link_server const srv{300e6, 50e9 / 8};
    controller_state st;
    st.conn_target = cfg.min_conns_total;  // where a measured fast B_c leaves the seed (8)
    std::vector<double> gp;
    std::vector<unsigned> ct;
    int const reached = run_link(st, srv, cfg, 40, limit, &gp, &ct);
    CHECK(reached >= 0);
    CHECK(reached <= 8);
    CHECK(ct.back() >= 21);  // 6.25 GB/s / 300 MB/s
    CHECK(gp.back() >= 0.9 * srv.link);
    CHECK(st.target_scale * target_bytes_per_s(cfg) >= 0.9 * srv.link);  // discovered
  }
  SECTION("the probing stops at the connection ceiling and never over-shoots it")
  {
    link_server const srv{75e6, 1e12};
    controller_state st;
    std::vector<double> gp;
    std::vector<unsigned> ct;
    (void)run_link(st, srv, cfg, 60, 64, &gp, &ct);
    for (unsigned c : ct)
      CHECK(c <= 64);
    CHECK(ct.back() == 64);
  }
  SECTION("a throttle stops the probing for the caution window")
  {
    link_server const srv{75e6, 100e9 / 8};
    controller_state st;
    controller_inputs in = srv.period(st.conn_target = 20, target_bytes_per_s(cfg), limit);
    in.throttled         = in.responses;  // all throttled
    CHECK(conn_target_step(st, in, cfg) == controller_move::throttle_decrease);
    double const scale = st.target_scale;
    for (unsigned i = 0; i + 1 < throttle_caution_periods; ++i) {
      CHECK(conn_target_step(st, srv.period(st.conn_target, target_bytes_per_s(cfg), limit), cfg) !=
            controller_move::probe_up);
    }
    CHECK(st.target_scale == scale);
    CHECK(conn_target_step(st, srv.period(st.conn_target, target_bytes_per_s(cfg), limit), cfg) ==
          controller_move::probe_up);
  }
}

TEST_CASE("planner.remote.controller: cross-reactor aggregation drives the endpoint controller",
          "[rest][planner][remote][controller]")
{
  // Four reactors publish every 100 ms; the endpoint steps every 250 ms on the merged samples.
  remote_planner_config cfg{};
  cfg.min_conns_total = 2;
  endpoint_estimator ep(cfg);
  auto rs = std::make_unique<std::array<estimator_samples, 4>>();
  (void)ep.snapshot_for_request(100 * GiB);
  unsigned capacity = 8;
  std::vector<unsigned> ct;
  std::uint64_t now = 0;
  for (int tick = 0; tick < 200; ++tick) {
    now += 100 * ms;
    unsigned const conns = ep.snapshot().conn_target;
    for (std::size_t r = 0; r < rs->size(); ++r) {
      auto& s = (*rs)[r];
      // this reactor's share of the endpoint's connections, 1 response each per 100 ms
      unsigned const mine = conns / 4 + (r < conns % 4 ? 1u : 0u);
      for (unsigned c = 0; c < mine; ++c) {
        unsigned const global_index = static_cast<unsigned>(r) + 4 * c;
        s.record_ttfb(20);
        s.record_response(global_index >= capacity);
        if (global_index < capacity) {
          s.record_body(8 * MiB, 8.0 * 1048576.0 / 75e6);
          s.record_delivered(8 * MiB);
        }
      }
      ep.merge(s);  // remote_client::publish under its mutex
      s.clear();
      (void)ep.control_at(now);
    }
    ct.push_back(ep.snapshot().conn_target);
    if (tick == 100) capacity = 1000;
  }
  CHECK(ep.steps() >= 60);  // 20 s at one step per >= 250 ms (ticks of 100 ms: every 300 ms)
  unsigned hi = 0;
  for (int i = 30; i < 100; ++i)
    hi = std::max(hi, ct[static_cast<std::size_t>(i)]);
  CHECK(hi <= 8 + 1);  // throttled: a sawtooth around the capacity (8), never back at the seed
  CHECK(ct[99] < 20);
  CHECK(ct.back() >= 20);  // recovered once the throttling stopped
  if (trace()) {
    std::printf("4-reactor trace (conn_target per 100 ms; capacity 8 until 10 s):");
    for (std::size_t i = 0; i < ct.size(); i += 2)
      std::printf("%s%u", i % 50 == 0 ? "\n  " : " ", ct[i]);
    std::printf("\n");
  }
}

TEST_CASE("planner.remote.controller: EC2 NIC table and the link seed",
          "[rest][planner][remote][controller]")
{
  struct row {
    char const* type;
    double gbps;
  };
  for (row const r :
       {row{"c6in.32xlarge", 200},   row{"c6in.large", 25},      row{"c7gn.16xlarge", 200},
        row{"c7gn.medium", 25},      row{"g7e.8xlarge", 100},    row{"g7e.48xlarge", 1600},
        row{"g6e.48xlarge", 400},    row{"g6e.xlarge", 20},      row{"m6a.48xlarge", 50},
        row{"m6a.large", 12.5},      row{"m7i.48xlarge", 50},    row{"m7i.8xlarge", 12.5},
        row{"p5.48xlarge", 3200},    row{"p5.4xlarge", 100},     row{"trn1.32xlarge", 800},
        row{"trn1n.32xlarge", 1600}, row{"trn2.48xlarge", 3200}, row{"hpc7g.16xlarge", 200},
        row{"i4i.32xlarge", 75},     row{"i7ie.48xlarge", 200},  row{"im4gn.16xlarge", 100},
        row{"c7i.48xlarge", 50},     row{"c7a.48xlarge", 50},    row{"m6i.32xlarge", 50},
        row{"m7a.48xlarge", 50},     row{"r7i.48xlarge", 50}}) {
    INFO(r.type);
    auto const g = ec2_instance_link_gbps(r.type);
    REQUIRE(g.has_value());
    CHECK(*g == r.gbps);
    CHECK(link_seed_gbps(std::string_view(r.type)) == r.gbps);
  }
  CHECK(ec2_instance_link_gbps("  G7E.8XLARGE\n") == 100.0);  // blanks, case
  CHECK_FALSE(ec2_instance_link_gbps("x9z.large").has_value());
  CHECK_FALSE(ec2_instance_link_gbps("g7e.16xlarge").has_value());                // no such size
  CHECK(link_seed_gbps(std::string_view("x9z.large")) == unknown_ec2_link_gbps);  // 25
  CHECK(link_seed_gbps(std::nullopt) == default_link_gbps);                       // off EC2: 10
  CHECK(link_seed_gbps(std::string_view("")) == default_link_gbps);
}

TEST_CASE("planner.remote.controller: set_link_gbps is a seed change; an explicit link wins",
          "[rest][planner][remote][controller]")
{
  remote_planner_config cfg{};
  endpoint_estimator ep(cfg);
  CHECK(ep.snapshot().link_gbps == 10);
  CHECK(ep.snapshot().conn_target == 20);
  ep.set_link_gbps(100);  // before the first step: re-seeded
  CHECK(ep.snapshot().link_gbps == 100);
  CHECK(ep.snapshot().conn_target == 199);
  ep.set_link_gbps(0);  // invalid: ignored
  CHECK(ep.snapshot().link_gbps == 100);

  cfg.link_gbps = 25;  // explicit
  endpoint_estimator ex(cfg);
  ex.set_link_gbps(100);
  CHECK(ex.snapshot().link_gbps == 25);
  CHECK(ex.snapshot().conn_target == seed_conn_target(cfg));

  // after steps: conn_target is raised to the new seed, the discovered target kept
  remote_planner_config c2{};
  endpoint_estimator st(c2);
  (void)st.control(0.25);
  (void)st.control(0.25);
  REQUIRE(st.steps() == 2);
  st.set_link_gbps(100);
  CHECK(st.snapshot().conn_target >= 199);
  CHECK(st.snapshot().link_gbps == 100);
}

TEST_CASE(
  "planner.remote.controller: admission queueing below the link probes up; a CPU cap stops it",
  "[rest][planner][remote][controller]")
{
  // 100 Gbps seed (IMDS); bodies arrive at 400 MB/s once flowing (the model's B_c) but each
  // connection only nets 40 MB/s (TTFB 100 ms of queueing), and the consumer is CPU-bound at
  // 6 GB/s: the model's want is ~70, the CPU allows ~150 connections' worth.
  remote_planner_config cfg{};
  cfg.link_gbps         = 100;
  double const target   = target_bytes_per_s(cfg);
  double const per_conn = 40e6, cpu = 6e9;
  auto run = [&](bool queued, std::vector<unsigned>* trace_out) {
    controller_state st;
    st.conn_target = 70;
    double gp      = 0;
    for (int i = 0; i < 60; ++i) {
      controller_inputs in;
      in.target_bytes_s  = target;
      in.bc_bytes_s      = 400e6;
      in.delay_bytes     = 0.1 * 400e6;
      in.backlog         = 500 * MiB;
      in.responses       = std::uint64_t{st.conn_target} * 2;
      in.conn_limit      = 256;
      in.admission_waits = queued ? 1000 : 0;
      gp = in.goodput_bytes_s = std::min(static_cast<double>(st.conn_target) * per_conn, cpu);
      (void)conn_target_step(st, in, cfg);
      if (trace_out != nullptr) trace_out->push_back(st.conn_target);
    }
    return std::pair<unsigned, double>{st.conn_target, gp};
  };
  std::vector<unsigned> tr;
  auto const [ct_q, gp_q] = run(true, &tr);
  auto const [ct_n, gp_n] = run(false, nullptr);
  CHECK(ct_n <= 80);  // no evidence: the model's want (~70) stays
  CHECK(gp_n < 0.5 * cpu);
  CHECK(ct_q >= 150);  // queueing: climbs to what the CPU can use
  CHECK(gp_q >= 0.95 * cpu);
  unsigned top = 0;
  for (unsigned c : tr)
    top = std::max(top, c);
  CHECK(top <= 256);     // the ceiling
  CHECK(ct_q < 200);     // weak probes beyond the CPU cap were reverted
  unsigned settled = 0;  // periods spent at the reverted level (~150 connections)
  for (std::size_t i = 10; i < tr.size(); ++i)
    settled += tr[i] < 200 ? 1u : 0u;
  CHECK(settled * 2 >= tr.size() - 10);
  if (trace()) {
    std::printf("C7 queued probe-up (conn_target per period):\n ");
    for (unsigned int i : tr)
      std::printf(" %u", i);
    std::printf("\n  queued: %u conns %.2f GB/s; not queued: %u conns %.2f GB/s\n",
                ct_q,
                gp_q / 1e9,
                ct_n,
                gp_n / 1e9);
  }
}

TEST_CASE("planner.remote.controller: the admission cap binds only within the caution window",
          "[rest][planner][remote][controller]")
{
  remote_planner_config cfg{};
  endpoint_estimator ep(cfg);
  CHECK_FALSE(ep.throttle_binding());  // before the first 503/429
  for (int i = 0; i < 100; ++i)
    ep.record_response(i < 10);
  (void)ep.control(0.25);
  CHECK(ep.throttle_binding());
  for (unsigned i = 0; i + 1 < throttle_caution_periods; ++i) {
    (void)ep.control(0.25);
    CHECK(ep.throttle_binding());
  }
  (void)ep.control(0.25);
  CHECK_FALSE(ep.throttle_binding());
}

TEST_CASE(
  "planner.remote.controller: segment size: s_util with a deep backlog, s_par only at the tail",
  "[rest][planner][remote][segment]")
{
  remote_planner_config cfg{};
  cfg.link_gbps = 100;
  endpoint_estimate e{};
  e.ttfb_p50_ms     = 72;  // the EC2 A2 figures: D = 72 ms x 60 MB/s = 4.3 MB, s_util = 24.5 MB
  e.bc_bytes_s      = 60e6;
  e.conn_target     = 256;
  e.link_gbps       = 100;
  e.backlog         = 16ull * GiB;  // deep: 256 x 24.5 MB = 6.3 GB < 16 GiB
  auto const s_deep = segment_target(cfg, e, 0);
  CHECK(static_cast<double>(s_deep) == Approx(utilisation_segment(4.32e6, 0.85)).epsilon(0.01));
  CHECK(s_deep > 16 * MiB);  // was backlog / conn_target = 64 MiB? no: the old rule gave s_par
  e.backlog         = 2ull * GiB;  // tail: 2 GiB / 256 = 8 MiB
  auto const s_tail = segment_target(cfg, e, 0);
  CHECK(s_tail == 8 * MiB);
  // a smaller measured D never shrinks s below the seeded D (30 ms x 75 MB/s -> 12.75 MB)
  e.ttfb_p50_ms = 1;
  e.bc_bytes_s  = 2e9;
  e.backlog     = 16ull * GiB;
  CHECK(segment_target(cfg, e, 0) == 12'750'000);
}

TEST_CASE("planner.remote.controller: part alignment never extends a read beyond the span",
          "[rest][planner][remote][segment]")
{
  // P8: cut_remote_chunks / recut_chunk_for_parts tile their input exactly (no byte
  // before the start or after the end, no overlap, no hole) for any span, segment and part size.
  remote_planner_config const cfg{};
  planner_test::splitmix64 rng{2026};
  std::vector<file_range> out(4096), pieces(256);
  for (int i = 0; i < 20000; ++i) {
    file_range const span{rng.between(0, 64 * GiB), rng.log_between(1, 256 * MiB)};
    std::uint64_t const s = rng.log_between(1 * MiB, 32 * MiB);
    std::optional<std::uint64_t> part;
    if (rng() % 2 == 0) part = rng.log_between(1 * MiB, 64 * MiB);
    std::size_t const n = cut_remote_chunks(span, s, part, cfg, std::span(out));
    REQUIRE(n != 0);
    std::uint64_t pos = span.offset;
    for (std::size_t k = 0; k < n; ++k) {
      CHECK(out[k].offset == pos);
      pos = out[k].end();
      // each chunk re-cut for late part alignment stays inside the chunk too
      if (part && k % 7 == 0) {
        std::size_t const m = recut_chunk_for_parts(out[k], *part, cfg, std::span(pieces));
        REQUIRE(m != 0);
        std::uint64_t q = out[k].offset;
        for (std::size_t j = 0; j < m; ++j) {
          CHECK(pieces[j].offset == q);
          q = pieces[j].end();
        }
        CHECK(q == out[k].end());
      }
    }
    CHECK(pos == span.end());
  }
}
