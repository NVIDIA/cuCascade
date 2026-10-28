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

// Fairness and pick order (ported from gpuio's tests/unit/planner/remote_fairness_test.cpp; gpuio
// ARCHITECTURE §2.4, §7.1; s3_line_rate §3 items 3 and 6): latency class and reserved
// connections, DRR / FIFO, a discrete-event simulation of one 100 GB scan plus Poisson 64 KiB
// reads, block-cyclic order and per-part in-flight caps (P9), weighted DRR 4:2:1 with `drr_wait`
// under a binding admission cap (P12).

#include "io/rest/planner/planner_remote.hpp"
#include "planner_test_support.hpp"

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

using namespace cucascade::io::rest;
using namespace cucascade::io::rest::planner;

namespace {

struct flow {
  drr_hook hook;  // first member: pointer-interconvertible with the flow
  std::uint64_t chunk{0};
  std::uint64_t served{0};
  std::uint64_t picks{0};
  bool blocked{false};
};
static_assert(std::is_standard_layout_v<flow>);

flow& as_flow(drr_hook& h) { return *reinterpret_cast<flow*>(&h); }

std::uint64_t flow_cost(drr_hook& h) noexcept
{
  flow& f = as_flow(h);
  return f.blocked ? drr_skip : f.chunk;
}

flow* pick(drr_scheduler& s)
{
  drr_hook* const h = s.next(flow_cost);
  if (h == nullptr) return nullptr;
  flow& f = as_flow(*h);
  f.served += f.chunk;
  ++f.picks;
  return &f;
}

}  // namespace

TEST_CASE("planner.remote.fairness: latency class and reserved connections",
          "[rest][planner][remote][fairness]")
{
  remote_planner_config const cfg{};
  // gpuio passed `read_options` (prio normal, not explicit by default); its two fields here.
  CHECK(request_class(priority::normal, false, 64 * KiB, cfg) ==
        remote_class::latency);  // footer-sized
  CHECK(request_class(priority::normal, false, 1 * MiB, cfg) ==
        remote_class::latency);  // <= small_request_bytes
  CHECK(request_class(priority::normal, false, 1 * MiB + 1, cfg) == remote_class::bulk);
  CHECK(request_class(priority::urgent, false, 10 * GiB, cfg) == remote_class::latency);
  CHECK(request_class(priority::urgent, true, 10 * GiB, cfg) == remote_class::latency);
  // explicit: no size inference; `high` is not the latency class
  CHECK(request_class(priority::bulk, true, 4 * KiB, cfg) == remote_class::bulk);
  CHECK(request_class(priority::high, true, 4 * KiB, cfg) == remote_class::bulk);

  // max(2, ceil(5% x conns)), always leaving bulk one connection.
  CHECK(latency_reserved_connections(cfg, 64) == 4);
  CHECK(latency_reserved_connections(cfg, 32) == 2);
  CHECK(latency_reserved_connections(cfg, 8) == 2);
  CHECK(latency_reserved_connections(cfg, 200) == 10);
  CHECK(latency_reserved_connections(cfg, 2) == 1);
  CHECK(latency_reserved_connections(cfg, 1) == 0);
  CHECK(latency_reserved_connections(cfg, 0) == 0);
}

TEST_CASE("planner.remote.drr: byte fairness across chunk sizes",
          "[rest][planner][remote][fairness]")
{
  drr_scheduler s(32 * MiB);
  flow a, b, c;
  a.chunk = 16 * MiB;
  b.chunk = 4 * MiB;
  c.chunk = 1 * MiB;
  s.push(a.hook, remote_class::bulk);
  s.push(b.hook, remote_class::bulk);
  s.push(c.hook, remote_class::bulk);
  for (int i = 0; i < 20000; ++i)
    REQUIRE(pick(s) != nullptr);
  std::uint64_t const lo = std::min({a.served, b.served, c.served});
  std::uint64_t const hi = std::max({a.served, b.served, c.served});
  CHECK(hi - lo <= 2 * s.quantum());  // equal bytes, not equal GETs
  CHECK(c.picks > 10 * a.picks);
}

TEST_CASE("planner.remote.drr: FIFO order, latency bypass, skipping, removal",
          "[rest][planner][remote][fairness]")
{
  SECTION("fifo: the oldest bulk request is served until it is removed")
  {
    drr_scheduler s(32 * MiB, remote_planner_config::order::fifo);
    flow a, b;
    a.chunk = b.chunk = 8 * MiB;
    s.push(a.hook, remote_class::bulk);
    s.push(b.hook, remote_class::bulk);
    for (int i = 0; i < 10; ++i)
      CHECK(pick(s) == &a);
    a.blocked = true;  // a cannot send now: b is next
    CHECK(pick(s) == &b);
    a.blocked = false;
    s.remove(a.hook);
    CHECK(pick(s) == &b);
  }
  SECTION("latency requests bypass bulk, FIFO among themselves")
  {
    drr_scheduler s(32 * MiB);
    flow bulk, l1, l2;
    bulk.chunk = 16 * MiB;
    l1.chunk = l2.chunk = 64 * KiB;
    s.push(bulk.hook, remote_class::bulk);
    CHECK(pick(s) == &bulk);
    s.push(l1.hook, remote_class::latency);
    s.push(l2.hook, remote_class::latency);
    CHECK(s.latency_size() == 2);
    CHECK(pick(s) == &l1);
    s.remove(l1.hook);
    CHECK(pick(s) == &l2);
    l2.blocked = true;  // a latency request at its cap does not block bulk
    CHECK(pick(s) == &bulk);
    s.remove(l2.hook);
    CHECK(s.latency_size() == 0);
  }
  SECTION("skipped requests lose their deficit; all blocked => nullptr")
  {
    drr_scheduler s(32 * MiB);
    flow a, b;
    a.chunk = b.chunk = 8 * MiB;
    s.push(a.hook, remote_class::bulk);
    s.push(b.hook, remote_class::bulk);
    CHECK(pick(s) == &a);  // a holds 24 MiB of deficit now
    a.blocked = true;
    CHECK(pick(s) == &b);
    CHECK(a.hook.deficit == 0);
    b.blocked = true;
    CHECK(pick(s) == nullptr);
    a.blocked = b.blocked = false;
    CHECK(pick(s) != nullptr);
  }
  SECTION("removing the cursor's request, double push / remove")
  {
    drr_scheduler s(8 * MiB);
    flow a, b, c;
    a.chunk = b.chunk = c.chunk = 8 * MiB;
    s.push(a.hook, remote_class::bulk);
    s.push(a.hook, remote_class::bulk);  // no-op
    s.push(b.hook, remote_class::bulk);
    s.push(c.hook, remote_class::bulk);
    CHECK(s.bulk_size() == 3);
    CHECK(pick(s) == &a);
    CHECK(pick(s) == &b);  // quantum == chunk: one GET per round
    s.remove(b.hook);      // b was the cursor: c inherits it
    s.remove(b.hook);      // no-op
    CHECK(pick(s) == &c);
    CHECK(pick(s) == &a);
    s.remove(a.hook);
    s.remove(c.hook);
    CHECK(s.empty());
    CHECK(pick(s) == nullptr);
    s.push(b.hook, remote_class::bulk);
    CHECK(pick(s) == &b);
  }
}

namespace {

/// Discrete-event model of one reactor's 64 connections to S3 (GET = TTFB + bytes / B_c,
/// s3_line_rate §2.6): one 100 GB scan in 16 MiB GETs, plus Poisson 64 KiB reads. TTFB is drawn
/// per GET, log-normal with median 30 ms and sigma 0.5 (p95 68 ms, p99 96 ms; AnyBlob: "less than
/// 5% have a first byte latency above 200ms", Quickwit: "80ms is rather common"). A fixed TTFB
/// would keep the 60 bulk GETs that start together in lock step forever, which real S3 does not.
struct sim_policy {
  char const* name;
  remote_planner_config::order order;
  bool latency_class;  // small reads use the latency class (bypass)
  bool reserve;        // bulk may not take the latency-reserved connections
};

struct sim_result {
  std::vector<double> queue_delay;  // small reads: dispatch - arrival (s)
  std::vector<double> latency;      // small reads: completion - arrival (s)
  std::vector<double> service;      // small reads: their own GET time (s), no queueing
  double scan_done{0};
};

struct sim_req {
  drr_hook hook;
  bool small{false};
  std::uint64_t chunk{0};
  std::uint64_t remaining{0};
  double arrival{0};
  double dispatched{-1};
  double service{0};
};

constexpr unsigned sim_conns      = 64;
constexpr double sim_ttfb         = 0.030;  // s, median TTFB (cases A/C)
constexpr double sim_sigma        = 0.5;    // log-normal shape of the TTFB
constexpr double sim_bc           = 75e6;   // B/s, the seed
constexpr double sim_scan         = 100e9;  // bytes
constexpr std::uint64_t sim_get   = 16 * MiB;
constexpr std::uint64_t sim_small = 64 * KiB;
constexpr double sim_rate         = 40;  // small reads per second
constexpr double sim_window       = 20;  // seconds of arrivals (the scan lasts ~25 s)

/// Median GET time for `bytes`.
double get_time(std::uint64_t bytes) { return sim_ttfb + static_cast<double>(bytes) / sim_bc; }

/// The time of one GET with a log-normal TTFB (Box-Muller from two uniforms).
double draw_get_time(std::uint64_t bytes, planner_test::splitmix64& rng)
{
  double const u1 = 1.0 - rng.unit(), u2 = rng.unit();
  double const z = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
  return sim_ttfb * std::exp(sim_sigma * z) + static_cast<double>(bytes) / sim_bc;
}

sim_result simulate(sim_policy const& pol)
{
  remote_planner_config const cfg{};
  unsigned const reserved = pol.reserve ? latency_reserved_connections(cfg, sim_conns) : 0;
  drr_scheduler sched(cfg.max_segment_bytes, pol.order);
  planner_test::splitmix64 rng{2026};
  planner_test::splitmix64 ttfb_rng{77};

  // Arrival times first, so the request vector never reallocates (hooks must not move).
  std::vector<double> arrivals;
  for (double t = 1.0;;) {
    t += -std::log(1.0 - rng.unit()) / sim_rate;
    if (t > 1.0 + sim_window) break;
    arrivals.push_back(t);
  }
  auto const reqs = std::make_unique<sim_req[]>(arrivals.size() + 1);  // stable addresses
  sim_req& scan   = reqs[0];
  scan.chunk      = sim_get;
  scan.remaining  = static_cast<std::uint64_t>(std::ceil(sim_scan / static_cast<double>(sim_get)));
  sched.push(scan.hook, remote_class::bulk);

  using event = std::pair<double, std::size_t>;  // time, request index (+ arrival flag)
  constexpr std::size_t arrival_flag = std::size_t{1} << 62;
  std::priority_queue<event, std::vector<event>, std::greater<>> q;
  for (std::size_t i = 0; i < arrivals.size(); ++i)
    q.emplace(arrivals[i], (i + 1) | arrival_flag);

  unsigned free = sim_conns, bulk_inflight = 0;
  sim_result res;
  auto dispatch = [&](double now) {
    while (free > 0) {
      bool const bulk_ok = bulk_inflight < sim_conns - reserved;
      drr_hook* const h  = sched.next([&](drr_hook& hk) noexcept -> std::uint64_t {
        auto const& r = *reinterpret_cast<sim_req*>(&hk);
        if (!hk.latency && !bulk_ok) return drr_skip;
        return r.chunk;
      });
      if (h == nullptr) return;
      auto& r = *reinterpret_cast<sim_req*>(h);
      --free;
      if (!h->latency) ++bulk_inflight;
      if (r.dispatched < 0) r.dispatched = now;
      auto const idx  = static_cast<std::size_t>(&r - reqs.get());
      double const dt = draw_get_time(r.chunk, ttfb_rng);
      if (r.small) r.service = dt;
      q.emplace(now + dt, idx | (h->latency ? std::size_t{1} << 61 : 0));
      if (--r.remaining == 0) sched.remove(*h);
    }
  };
  dispatch(0.0);
  while (!q.empty()) {
    auto const [now, tag] = q.top();
    q.pop();
    std::size_t const idx = tag & ((std::size_t{1} << 61) - 1);
    if ((tag & arrival_flag) != 0) {
      sim_req& r  = reqs[idx];
      r.small     = true;
      r.chunk     = sim_small;
      r.remaining = 1;
      r.arrival   = now;
      sched.push(r.hook, pol.latency_class ? remote_class::latency : remote_class::bulk);
    } else {
      ++free;
      if ((tag & (std::size_t{1} << 61)) == 0) --bulk_inflight;
      sim_req const& r = reqs[idx];
      if (r.small) {
        res.queue_delay.push_back(r.dispatched - r.arrival);
        res.latency.push_back(now - r.arrival);
        res.service.push_back(r.service);
      } else if (idx == 0 && r.remaining == 0) {
        res.scan_done = std::max(res.scan_done, now);
      }
    }
    dispatch(now);
  }
  return res;
}

double quantile(std::vector<double> v, double q)
{
  std::sort(v.begin(), v.end());
  auto const i = static_cast<std::size_t>(std::ceil(q * static_cast<double>(v.size()))) - 1;
  return v[std::min(i, v.size() - 1)];
}

}  // namespace

TEST_CASE("planner.remote.drr: 100 GB scan + Poisson 64 KiB reads, small-read delay ~ one GET",
          "[rest][planner][remote][fairness][sim]")
{
  double const one_get  = get_time(sim_small);  // 30.9 ms
  double const bulk_get = get_time(sim_get);    // 254 ms
  sim_policy const ours{
    "latency class + reserved + DRR", remote_planner_config::order::drr, true, true};
  sim_policy const drr_only{
    "DRR, no latency class", remote_planner_config::order::drr, false, false};
  sim_policy const fifo{"uring_ultra: FIFO", remote_planner_config::order::fifo, false, false};

  auto const g = simulate(ours);
  auto const d = simulate(drr_only);
  auto const f = simulate(fifo);
  REQUIRE(g.queue_delay.size() > 500);
  REQUIRE(g.queue_delay.size() == f.queue_delay.size());
  for (auto const* r : {&g, &d, &f})
    REQUIRE(r->scan_done > 20.0);

  std::printf(
    "[planner sim] one 64 KiB GET %.1f ms; small reads %zu over %.0f s of a %.0f GB scan\n",
    one_get * 1e3,
    g.queue_delay.size(),
    sim_window,
    sim_scan / 1e9);
  for (auto const& [name, r] : {std::pair{ours.name, &g}, {drr_only.name, &d}, {fifo.name, &f}}) {
    std::printf(
      "[planner sim] %-40s queue p50 %9.2f ms p99 %9.2f ms | latency p99 %9.2f ms | scan %.2f s\n",
      name,
      quantile(r->queue_delay, 0.5) * 1e3,
      quantile(r->queue_delay, 0.99) * 1e3,
      quantile(r->latency, 0.99) * 1e3,
      r->scan_done);
  }
  // The planner: a small read waits about nothing and completes in about one GET of its own
  // (P12: "small-read p99 ~ TTFB"; s3_line_rate §3 item 6).
  double const own_p99 = quantile(g.service, 0.99);
  std::printf("[planner sim] a 64 KiB GET on an idle connection: p50 %.1f ms p99 %.1f ms\n",
              quantile(g.service, 0.5) * 1e3,
              own_p99 * 1e3);
  CHECK(quantile(g.queue_delay, 0.50) == 0.0);
  CHECK(quantile(g.queue_delay, 0.99) <= 0.5 * one_get);
  CHECK(quantile(g.latency, 0.50) <= 1.1 * one_get);
  CHECK(quantile(g.latency, 0.99) <= own_p99 + 0.5 * one_get);
  // The reservation costs the scan at most 64/60 of its time.
  CHECK(g.scan_done <= 1.1 * f.scan_done);
  // DRR alone: the next free connection (well under one bulk GET).
  CHECK(quantile(d.queue_delay, 0.99) <= bulk_get);
  // FIFO behind the scan: seconds (the whole scan must be dispatched first).
  CHECK(quantile(f.queue_delay, 0.50) > 100 * one_get);
}

TEST_CASE("planner.remote.parts: block-cyclic order", "[rest][planner][remote][parts]")
{
  // Bijection for every (n, block).
  for (std::size_t n = 1; n <= 40; ++n) {
    for (std::size_t block = 0; block <= n + 2; ++block) {
      std::vector<bool> seen(n, false);
      for (std::size_t p = 0; p < n; ++p) {
        std::size_t const i = block_cyclic_index(p, n, block);
        REQUIRE(i < n);
        REQUIRE_FALSE(seen[i]);
        seen[i] = true;
      }
    }
  }
  std::vector<std::size_t> order;
  for (std::size_t p = 0; p < 10; ++p)
    order.push_back(block_cyclic_index(p, 10, 3));
  CHECK(order == std::vector<std::size_t>{0, 3, 6, 9, 1, 4, 7, 2, 5, 8});
  CHECK(chunks_per_part(8 * MiB, 32 * MiB) == 4);
  CHECK(chunks_per_part(16 * MiB, 24 * MiB) == 2);  // 1.5 rounds up
  CHECK(chunks_per_part(32 * MiB, 8 * MiB) == 1);
  CHECK(chunks_per_part(8 * MiB, 0) == 1);
  CHECK(part_index(0, 8 * MiB) == 0);
  CHECK(part_index(8 * MiB - 1, 8 * MiB) == 0);
  CHECK(part_index(8 * MiB, 8 * MiB) == 1);
  CHECK(part_index(123 * GiB, 0) == 0);
  remote_planner_config cfg{};
  CHECK(max_inflight_per_part(cfg) == 24);
  // P9: one reactor's share of the per-part cap, minimum 1 (gpuio remote_backend part_cap).
  CHECK(part_cap(cfg, 1) == 24);
  CHECK(part_cap(cfg, 0) == 24);
  CHECK(part_cap(cfg, 4) == 6);
  CHECK(part_cap(cfg, 64) == 1);
  cfg.max_inflight_per_part = 0;
  CHECK(max_inflight_per_part(cfg) == std::numeric_limits<unsigned>::max());
  CHECK(part_cap(cfg, 4) == std::numeric_limits<unsigned>::max());
}

TEST_CASE("planner.remote.parts: per-part cap with block-cyclic picks",
          "[rest][planner][remote][parts]")
{
  // One 512 MiB object in 32 MiB parts, cut into 64 chunks of 8 MiB; 40 connections; cap 2
  // GETs per part (small, to exercise skipping; S3 uses 24).
  constexpr std::size_t n       = 64;
  constexpr std::uint64_t chunk = 8 * MiB, part = 32 * MiB, object = 7;
  constexpr unsigned cap = 2, conns = 40;
  std::size_t const block = chunks_per_part(chunk, part);
  REQUIRE(block == 4);
  part_inflight_table<> table;
  std::vector<bool> taken(n, false);
  std::vector<std::size_t> inflight;
  std::vector<std::uint64_t> first_parts;
  planner_test::splitmix64 rng{11};
  std::size_t cursor = 0, done = 0, skips = 0;
  auto part_of = [&](std::size_t i) { return part_index(i * chunk, part); };
  while (done < n) {
    std::size_t idx = n;
    if (inflight.size() < conns) {
      idx = pick_block_cyclic(n, block, cursor, [&](std::size_t i) noexcept {
        return !taken[i] && table.inflight(object, part_of(i)) < cap;
      });
    }
    if (idx == n) {  // nothing pickable: a GET completes
      REQUIRE(!inflight.empty());
      ++skips;
      auto const k = static_cast<std::size_t>(rng.between(0, inflight.size() - 1));
      table.release(object, part_of(inflight[k]));
      inflight.erase(inflight.begin() + static_cast<std::ptrdiff_t>(k));
      ++done;
      continue;
    }
    REQUIRE(table.try_acquire(object, part_of(idx), cap));
    REQUIRE(table.inflight(object, part_of(idx)) <= cap);
    taken[idx] = true;
    inflight.push_back(idx);
    if (first_parts.size() < 16) first_parts.push_back(part_of(idx));
  }
  CHECK(std::all_of(taken.begin(), taken.end(), [](bool b) { return b; }));
  CHECK(std::set<std::uint64_t>(first_parts.begin(), first_parts.end()).size() == 16);
  CHECK(skips > 0);
  CHECK(table.size() == 0);
}

TEST_CASE("planner.remote.parts: part_inflight_table vs a reference map",
          "[rest][planner][remote][parts]")
{
  part_inflight_table<128> table;
  std::map<std::pair<std::uint64_t, std::uint64_t>, unsigned> ref;
  planner_test::splitmix64 rng{5};
  for (int i = 0; i < 200000; ++i) {
    std::uint64_t const obj = rng.between(0, 9), part = rng.between(0, 9);  // 100 keys <= 128
    auto& r = ref[{obj, part}];
    if (rng.between(0, 1) == 0) {
      bool const ok = table.try_acquire(obj, part, 3);
      REQUIRE(ok == (r < 3));
      if (ok) ++r;
    } else if (r > 0) {
      table.release(obj, part);
      --r;
    }
    REQUIRE(table.inflight(obj, part) == r);
    if (r == 0) ref.erase({obj, part});
    REQUIRE(table.size() == ref.size());
  }
  // Full table: further keys are admitted untracked (never blocks a GET).
  part_inflight_table<2> tiny;
  CHECK(tiny.try_acquire(1, 1, 1));
  CHECK(tiny.try_acquire(2, 2, 1));
  CHECK(tiny.try_acquire(3, 3, 1));
  CHECK(tiny.inflight(3, 3) == 0);
  CHECK_FALSE(tiny.try_acquire(1, 1, 1));  // tracked keys still capped
  CHECK_FALSE(tiny.try_acquire(9, 9, 0));  // cap 0 never admits
}

namespace {

/// P12: three streams (high, normal, bulk) with weights 4:2:1 and per-flow GET sizes; when
/// @p cap is non-zero, at most `cap` GETs are in flight together (one endpoint's admission cap)
/// and a capped flow answers @p capped.
std::array<std::uint64_t, 3> weighted_run(std::uint64_t quantum,
                                          std::array<std::uint64_t, 3> chunk,
                                          unsigned cap,
                                          std::uint64_t capped)
{
  drr_scheduler s(quantum);
  std::array<flow, 3> f{};
  std::array<std::uint32_t, 3> const w{4, 2, 1};
  for (std::size_t i = 0; i < 3; ++i) {
    f[i].chunk       = chunk[i];
    f[i].hook.weight = w[i];
    s.push(f[i].hook, remote_class::bulk);
  }
  unsigned inflight = 0;
  std::queue<flow*> done;  // completion order = issue order
  auto cost = [&](drr_hook& h) noexcept -> std::uint64_t {
    if (cap != 0 && inflight >= cap) return capped;
    return as_flow(h).chunk;
  };
  std::uint64_t total = 0;
  while (total < 1000 * quantum) {
    drr_hook* const h = s.next(cost);
    if (h == nullptr) {
      REQUIRE(!done.empty());  // only the cap can block: complete the oldest GET
      done.pop();
      --inflight;
      continue;
    }
    flow& x = as_flow(*h);
    x.served += x.chunk;
    total += x.chunk;
    if (cap != 0) {
      ++inflight;
      done.push(&x);
    }
  }
  return {f[0].served, f[1].served, f[2].served};
}

bool near(double v, double want) { return v >= want * 0.9 && v <= want * 1.1; }

}  // namespace

TEST_CASE("planner.remote.drr: weighted DRR serves bytes 4:2:1", "[rest][planner][remote][drr]")
{
  for (auto const chunk : {std::array<std::uint64_t, 3>{8 * MiB, 8 * MiB, 8 * MiB},
                           std::array<std::uint64_t, 3>{2 * MiB, 8 * MiB, 16 * MiB}}) {
    auto const b = weighted_run(16 * MiB, chunk, 0, drr_wait);
    INFO("served " << b[0] << " " << b[1] << " " << b[2]);
    CHECK(near(double(b[0]) / double(b[2]), 4.0));
    CHECK(near(double(b[1]) / double(b[2]), 2.0));
  }
}

TEST_CASE("planner.remote.drr: weights hold under a binding admission cap (drr_wait)",
          "[rest][planner][remote][drr]")
{
  for (unsigned cap : {1u, 3u, 8u}) {
    auto const b = weighted_run(16 * MiB, {8 * MiB, 8 * MiB, 8 * MiB}, cap, drr_wait);
    INFO("cap " << cap << " served " << b[0] << " " << b[1] << " " << b[2]);
    CHECK(near(double(b[0]) / double(b[2]), 4.0));
    CHECK(near(double(b[1]) / double(b[2]), 2.0));
  }
}

TEST_CASE("planner.remote.drr: drr_wait keeps the deficit, fifo keeps arrival order",
          "[rest][planner][remote][drr]")
{
  SECTION("a waiting request keeps its deficit and the cursor")
  {
    drr_scheduler s(32 * MiB);
    flow a, b;
    a.chunk = b.chunk = 8 * MiB;
    a.hook.weight     = 2;
    s.push(a.hook, remote_class::bulk);
    s.push(b.hook, remote_class::bulk);
    CHECK(pick(s) == &a);  // 64 MiB granted, 56 MiB left
    CHECK(a.hook.deficit == 56 * MiB);
    bool wait_all = true;
    auto cost     = [&](drr_hook& h) noexcept { return wait_all ? drr_wait : as_flow(h).chunk; };
    CHECK(s.next(cost) == nullptr);
    CHECK(a.hook.deficit == 56 * MiB);  // kept (drr_skip would reset it)
    CHECK(b.hook.deficit == 0);         // never granted while waiting
    wait_all = false;
    CHECK(s.next(cost) == &a.hook);  // the cursor did not move
    CHECK(a.hook.deficit == 48 * MiB);
  }
  SECTION("fifo ignores weights: the oldest request is served until it is removed")
  {
    drr_scheduler s(32 * MiB, remote_planner_config::order::fifo);
    flow bulk, high;
    bulk.chunk = high.chunk = 8 * MiB;
    high.hook.weight        = 4;
    s.push(bulk.hook, remote_class::bulk);
    s.push(high.hook, remote_class::bulk);
    for (int i = 0; i < 10; ++i)
      CHECK(pick(s) == &bulk);
    s.remove(bulk.hook);
    CHECK(pick(s) == &high);
  }
}
