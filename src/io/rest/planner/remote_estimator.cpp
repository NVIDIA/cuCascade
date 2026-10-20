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

// Per-endpoint estimator and connection-target controller (ported from gpuio's
// src/planner/remote_estimator.cpp; gpuio ARCHITECTURE §7.1 "estimator per endpoint";
// s3_line_rate §3.3 `control()`). Mergeable reactor-local samples (`estimator_samples`), `merge`,
// the periodic `control_at(now)`, the freeze re-probe and the post-throttle additive re-approach
// (C1, C2); link discovery with probe revert and the measured-B_c ceiling (C3); the link seed
// (C4); the admission-cap flag (C5); admission waits as probe evidence (C7).
// `enable_controller == false` pins the seeds. Also `endpoint_control`, the publication mutex of
// gpuio's `remote_client::publish` / `apply_link_seed` (src/s3/s3_client.cpp).

#include "io/rest/planner/planner_remote.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>

namespace cucascade::io::rest::planner {
namespace {

constexpr unsigned u_max = std::numeric_limits<unsigned>::max();

double nonneg(double v) noexcept { return std::isfinite(v) && v > 0.0 ? v : 0.0; }

/// B_c seed: assumed_conn_gbps (0.6 Gbps = 75 MB/s; s3_line_rate §2.1 per-connection table).
double seed_bc(remote_planner_config const& cfg) noexcept
{
  return nonneg(cfg.assumed_conn_gbps) * 1e9 / 8.0;
}

/// The typical GET size the controller reasons with: s_util clamped to the segment bounds.
double typical_segment(remote_planner_config const& cfg, double d) noexcept
{
  double const hi = cfg.max_segment_bytes == 0 ? std::numeric_limits<double>::max()
                                               : static_cast<double>(cfg.max_segment_bytes);
  double const lo =
    std::min(static_cast<double>(std::max<std::uint64_t>(cfg.min_segment_bytes, 1)), hi);
  return std::clamp(utilisation_segment(d, cfg.target_conn_util), lo, hi);
}

/// want = ceil(target / (B_c x s_typ / (s_typ + D)) x 1.1)  (gpuio ARCHITECTURE §7.1).
unsigned want_connections(remote_planner_config const& cfg, double bc, double d) noexcept
{
  return connections_for(target_bytes_per_s(cfg), bc, typical_segment(cfg, d), d, conn_headroom);
}

unsigned clamp_conns(unsigned v, unsigned lo, unsigned hi) noexcept
{
  return std::min(std::max(v, lo), std::max(lo, hi));
}

}  // namespace

// ---- quantile sketch --------------------------------------------------------------------------

std::size_t quantile_sketch::bucket_of(double v) noexcept
{
  if (!(v > min_value)) { return 0; }  // also NaN
  if (v >= max_value) { return buckets - 1; }
  // Interior bucket i (1-based) covers [min x gamma^(i-1), min x gamma^i).
  double const i = std::floor(std::log(v / min_value) / ln_gamma);
  return std::clamp<std::size_t>(static_cast<std::size_t>(i) + 1, 1, buckets - 2);
}

double quantile_sketch::value_of(std::size_t bucket) noexcept
{
  if (bucket == 0) { return min_value; }
  if (bucket >= buckets - 1) { return max_value; }
  // Geometric centre: relative error <= sqrt(gamma) - 1 ~ 0.5% for any sample in the bucket.
  return min_value * std::exp((static_cast<double>(bucket) - 0.5) * ln_gamma);
}

void quantile_sketch::add(double v) noexcept
{
  std::size_t const b = bucket_of(v);
  if (_counts[b] == std::numeric_limits<std::uint32_t>::max()) { decay(); }
  ++_counts[b];
  ++_total;
}

void quantile_sketch::merge(quantile_sketch const& o) noexcept
{
  for (std::size_t i = 0; i < buckets; ++i) {
    std::uint64_t const sum = std::uint64_t{_counts[i]} + o._counts[i];
    std::uint64_t const cap = std::numeric_limits<std::uint32_t>::max();
    _total += std::min(sum, cap) - _counts[i];
    _counts[i] = static_cast<std::uint32_t>(std::min(sum, cap));
  }
}

void quantile_sketch::decay() noexcept
{
  _total = 0;
  for (auto& c : _counts) {
    c >>= 1;
    _total += c;
  }
}

void quantile_sketch::clear() noexcept
{
  _counts.fill(0);
  _total = 0;
}

double quantile_sketch::quantile(double q) const noexcept
{
  if (_total == 0) { return 0.0; }
  double const qq   = std::isfinite(q) ? std::clamp(q, 0.0, 1.0) : 0.5;
  auto const rank_d = std::ceil(qq * static_cast<double>(_total));
  std::uint64_t const rank =
    std::clamp<std::uint64_t>(static_cast<std::uint64_t>(rank_d), 1, _total);
  std::uint64_t cum = 0;
  for (std::size_t i = 0; i < buckets; ++i) {
    cum += _counts[i];
    if (cum >= rank) { return value_of(i); }
  }
  return max_value;
}

// ---- controller -------------------------------------------------------------------------------

unsigned seed_conn_target(remote_planner_config const& cfg) noexcept
{
  double const bc   = seed_bc(cfg);
  double const d    = nonneg(cfg.assumed_ttfb_ms) * 1e-3 * bc;
  unsigned const lo = cfg.min_conns_total;
  unsigned const w  = std::max(want_connections(cfg, bc, d), lo);
  return cfg.max_conns_total != 0 ? clamp_conns(w, lo, cfg.max_conns_total) : w;
}

unsigned max_conn_target(remote_planner_config const& cfg) noexcept
{
  return max_conn_target(cfg, 1.0, 0);
}

unsigned max_conn_target(remote_planner_config const& cfg,
                         double target_scale,
                         unsigned conn_limit) noexcept
{
  if (cfg.max_conns_total != 0) { return std::max(cfg.max_conns_total, cfg.min_conns_total); }
  double const scale = std::isfinite(target_scale) ? std::max(1.0, target_scale) : 1.0;
  double const bc    = seed_bc(cfg);
  double const d     = nonneg(cfg.assumed_ttfb_ms) * 1e-3 * bc;
  unsigned const w   = std::max(
    connections_for(target_bytes_per_s(cfg) * scale, bc, typical_segment(cfg, d), d, conn_headroom),
    cfg.min_conns_total);
  double const cap = std::ceil(1.5 * static_cast<double>(w));  // "0 = 1.5 x initial target"
  unsigned m       = cap >= static_cast<double>(u_max) ? u_max : static_cast<unsigned>(cap);
  if (conn_limit != 0) { m = std::min(m, conn_limit); }  // C3: what the reactors can open
  return std::max(m, cfg.min_conns_total);
}

controller_move conn_target_step(controller_state& st,
                                 controller_inputs const& in,
                                 remote_planner_config const& cfg) noexcept
{
  unsigned const lo = cfg.min_conns_total;
  if (!(st.target_scale >= 1.0) || !std::isfinite(st.target_scale)) { st.target_scale = 1.0; }
  double const s_typ = typical_segment(cfg, nonneg(in.delay_bytes));
  // C3: the ceiling follows the current (discovered) target with the seeded B_c AND with the
  // measured one (a slow measured B_c needs more connections than the seed predicts), bounded by
  // the connection limit; max_conns_total, when set, is fixed.
  auto const ceiling = [&](double tgt) noexcept {
    unsigned h = max_conn_target(cfg, st.target_scale, in.conn_limit);
    if (cfg.max_conns_total == 0 && nonneg(in.bc_bytes_s) > 0.0) {
      unsigned const w =
        connections_for(tgt, nonneg(in.bc_bytes_s), s_typ, nonneg(in.delay_bytes), conn_headroom);
      double const c = std::ceil(1.5 * static_cast<double>(w));
      unsigned m     = c >= static_cast<double>(u_max) ? u_max : static_cast<unsigned>(c);
      if (in.conn_limit != 0) { m = std::min(m, in.conn_limit); }
      h = std::max(h, m);
    }
    return std::max(h, lo);
  };
  double const goodput = nonneg(in.goodput_bytes_s);
  // Gains are judged on the mean of the last two periods (one 250 ms period at GB/s rates moves
  // by a few percent on its own; two noisy periods must not freeze growth).
  double const smooth = st.goodput_prev > 0.0 ? 0.5 * (goodput + st.goodput_prev) : goodput;
  st.goodput_prev     = goodput;
  if (st.cautious != 0) { --st.cautious; }
  if (st.probe_hold != 0) { --st.probe_hold; }

  // Judge the previous increase: additive ones "freeze if two increases gained < 2%"; a weak
  // probe-up is reverted (it found nothing: CPU- or link-bound) and two in a row hold probing for
  // freeze_hold_periods (C3).
  if (st.pending_increase) {
    st.pending_increase = false;
    // probes (x 1.5) are judged on the period itself: saturation must show at once; additive
    // steps (+1/8) on the 2-period mean, where one noisy period must not count as a weak gain
    double const after = st.pending_probe ? goodput : smooth;
    bool const weak    = st.goodput_before_increase > 0.0
                           ? after < st.goodput_before_increase * (1.0 + freeze_min_gain)
                           : after <= 0.0;
    if (st.pending_probe) {
      st.pending_probe = false;
      if (weak) {
        st.target_scale = std::max(1.0, st.scale_before);
        st.conn_target  = std::max(lo, st.ct_before);
        if (++st.weak_probes >= freeze_after_weak) {
          st.weak_probes = 0;
          st.probe_hold  = freeze_hold_periods;
        }
      } else {
        st.weak_probes = 0;
      }
    } else {
      st.weak_increases = weak ? st.weak_increases + 1 : 0;
      if (st.weak_increases >= freeze_after_weak && !st.frozen) {
        st.frozen         = true;
        st.frozen_periods = 0;
      }
    }
  }
  double target = nonneg(in.target_bytes_s) * st.target_scale;  // discovered (C3)
  unsigned hi   = ceiling(target);
  if (st.conn_target == 0) { st.conn_target = clamp_conns(seed_conn_target(cfg), lo, hi); }

  // Multiplicative decrease on throttling: 503/429 share > 0.5% => x throttle_decrease (0.75).
  bool const throttling =
    in.responses > 0 &&
    static_cast<double>(in.throttled) > throttle_rate_threshold * static_cast<double>(in.responses);
  if (throttling) {
    double const dec = std::clamp(nonneg(cfg.throttle_decrease), 0.0, 1.0);
    auto const scaled =
      static_cast<unsigned>(std::floor(static_cast<double>(st.conn_target) * dec));
    st.conn_target      = clamp_conns(scaled, lo, hi);
    st.frozen           = false;
    st.weak_increases   = 0;
    st.cautious         = throttle_caution_periods;
    st.throttle_binding = true;
    return controller_move::throttle_decrease;
  }
  // C5: the admission cap binds only after a throttle (caution window); before the first 503/429
  // the reactors admit on every warm connection.
  st.throttle_binding = st.cautious != 0;

  double const need = static_cast<double>(st.conn_target) * s_typ;
  // Link discovery (C3): the seeded link is not a ceiling. Goodput keeps up with the target and
  // the backlog could feed more -- or GETs waited for admission while the link is below
  // goodput_fraction (a CPU- or latency-bound reactor cannot show goodput growth without more in
  // flight): raise the target x 1.5 and let want follow (never down).
  double const fraction = std::clamp(nonneg(cfg.goodput_fraction), 0.0, 1.0);
  bool const keeps_up   = goodput >= fraction * target && static_cast<double>(in.backlog) >= need;
  bool const queued     = in.admission_waits != 0 && goodput < fraction * target;
  if (target > 0.0 && (keeps_up || queued) && st.cautious == 0 && !st.frozen &&
      st.probe_hold == 0 && st.target_scale < link_probe_max_scale) {
    st.scale_before = st.target_scale;
    st.ct_before    = st.conn_target;
    st.target_scale = std::min(st.target_scale * link_probe_factor, link_probe_max_scale);
    target          = nonneg(in.target_bytes_s) * st.target_scale;
    hi              = ceiling(target);
    unsigned const want =
      connections_for(target, nonneg(in.bc_bytes_s), s_typ, nonneg(in.delay_bytes), conn_headroom);
    unsigned const step = std::max(1U, st.conn_target / 8);
    unsigned grown      = std::max(clamp_conns(want, lo, hi), st.conn_target);
    // queued: want may not move (it is the model's view); the cap itself must open up
    if (queued && grown == st.conn_target) {
      grown = st.conn_target > u_max - step ? u_max : st.conn_target + step;
    }
    st.conn_target = std::min(std::max(grown, std::min(st.conn_target, hi)), hi);
    // Only a probe that put more in flight is judged (and reverted when weak); a target raise
    // alone is the evidence itself (goodput kept up with it).
    if (st.conn_target > st.ct_before) {
      st.pending_increase        = true;
      st.pending_probe           = true;
      st.goodput_before_increase = goodput;
    }
    return controller_move::probe_up;
  }

  // Additive increase while under target with enough backlog to feed more connections.
  bool const under      = goodput < growth_goodput_fraction * target;
  bool const backlogged = static_cast<double>(in.backlog) > need;
  if (under && backlogged) {
    if (st.frozen) {
      if (++st.frozen_periods < freeze_hold_periods) { return controller_move::frozen_hold; }
      st.frozen         = false;  // re-probe (C2): the binding limit may have gone
      st.weak_increases = 0;
    }
    st.frozen_periods          = 0;
    unsigned const step        = std::max(1U, st.conn_target / 8);
    unsigned const grown       = st.conn_target > u_max - step ? u_max : st.conn_target + step;
    st.conn_target             = clamp_conns(grown, lo, hi);
    st.pending_increase        = true;
    st.goodput_before_increase = smooth;
    return controller_move::additive_increase;
  }

  // Otherwise follow the model: want = ceil(target / (B_c x u(s_typ)) x 1.1).
  unsigned const want =
    connections_for(target, nonneg(in.bc_bytes_s), s_typ, nonneg(in.delay_bytes), conn_headroom);
  unsigned next = clamp_conns(want, lo, hi);
  if (st.cautious != 0 && next > st.conn_target) {
    // AIMD after a throttle: re-approach the model's target additively (C2).
    unsigned const step = std::max(1U, st.conn_target / 8);
    next = std::min(next, st.conn_target > u_max - step ? u_max : st.conn_target + step);
  }
  st.conn_target    = next;
  st.frozen         = false;
  st.weak_increases = 0;
  return controller_move::retarget;
}

// ---- estimator --------------------------------------------------------------------------------

endpoint_estimator::endpoint_estimator(remote_planner_config const& cfg) noexcept
  : _cfg(cfg),
    _bc(seed_bc(cfg)),
    _p50(nonneg(cfg.assumed_ttfb_ms)),
    _p99(nonneg(cfg.assumed_ttfb_ms)),
    _bc_pub(seed_bc(cfg)),
    _conn_target(seed_conn_target(cfg)),
    _link_pub(cfg.link_gbps > 0.0 ? cfg.link_gbps : default_link_gbps),
    _link_explicit(cfg.link_gbps > 0.0)
{
  _ctl.conn_target = _conn_target.load(std::memory_order_relaxed);
}

endpoint_estimate endpoint_estimator::snapshot() const noexcept
{
  endpoint_estimate e;
  e.ttfb_p50_ms   = _p50.load(std::memory_order_relaxed);
  e.ttfb_p99_ms   = _p99.load(std::memory_order_relaxed);
  e.bc_bytes_s    = _bc_pub.load(std::memory_order_relaxed);
  e.link_util     = _util.load(std::memory_order_relaxed);
  e.conn_target   = _conn_target.load(std::memory_order_relaxed);
  e.link_gbps     = _link_pub.load(std::memory_order_relaxed);
  e.util_measured = _util_measured.load(std::memory_order_relaxed);
  e.backlog       = backlog();
  return e;
}

endpoint_estimate endpoint_estimator::snapshot_for_request(std::uint64_t request_bytes) noexcept
{
  auto const add = static_cast<std::int64_t>(
    std::min<std::uint64_t>(request_bytes, std::numeric_limits<std::int64_t>::max() / 2));
  std::int64_t const after = _backlog.fetch_add(add, std::memory_order_relaxed) + add;
  endpoint_estimate e      = snapshot();
  e.backlog                = after > 0 ? static_cast<std::uint64_t>(after) : 0;
  return e;
}

void endpoint_estimator::add_backlog(std::uint64_t bytes) noexcept
{
  _backlog.fetch_add(static_cast<std::int64_t>(std::min<std::uint64_t>(
                       bytes, std::numeric_limits<std::int64_t>::max() / 2)),
                     std::memory_order_relaxed);
}

void endpoint_estimator::sub_backlog(std::uint64_t bytes) noexcept
{
  _backlog.fetch_sub(static_cast<std::int64_t>(std::min<std::uint64_t>(
                       bytes, std::numeric_limits<std::int64_t>::max() / 2)),
                     std::memory_order_relaxed);
}

std::uint64_t endpoint_estimator::backlog() const noexcept
{
  std::int64_t const b = _backlog.load(std::memory_order_relaxed);  // racing subs may undershoot
  return b > 0 ? static_cast<std::uint64_t>(b) : 0;
}

void endpoint_estimator::record_ttfb(double ms) noexcept
{
  if (std::isfinite(ms) && ms >= 0.0) { _ttfb.add(ms); }
}

void endpoint_estimator::record_body(std::uint64_t bytes, double seconds) noexcept
{
  if (bytes < bc_min_body_bytes || !(seconds > 0.0) || !std::isfinite(seconds)) { return; }
  double const rate  = static_cast<double>(bytes) / seconds;
  double const alpha = std::clamp(nonneg(_cfg.ewma_alpha), 0.0, 1.0);
  _bc                = alpha * rate + (1.0 - alpha) * _bc;
}

void endpoint_estimator::record_response(bool is_throttled) noexcept
{
  ++_responses;
  if (is_throttled) { ++_throttled; }
}

void endpoint_estimator::record_delivered(std::uint64_t bytes) noexcept { _delivered += bytes; }

// ---- reactor-local samples (C1) ---------------------------------------------------------------

void estimator_samples::record_ttfb(double ms) noexcept
{
  if (std::isfinite(ms) && ms >= 0.0) { ttfb.add(ms); }
}

void estimator_samples::record_body(std::uint64_t bytes, double seconds) noexcept
{
  if (bytes < bc_min_body_bytes || !(seconds > 0.0) || !std::isfinite(seconds)) { return; }
  ++bodies;
  body_rate_sum += static_cast<double>(bytes) / seconds;
}

void estimator_samples::record_response(bool is_throttled) noexcept
{
  ++responses;
  if (is_throttled) { ++throttled; }
}

void estimator_samples::merge(estimator_samples const& o) noexcept
{
  if (o.ttfb.count() != 0) { ttfb.merge(o.ttfb); }
  bodies += o.bodies;
  body_rate_sum += o.body_rate_sum;
  responses += o.responses;
  throttled += o.throttled;
  delivered += o.delivered;
  admission_waits += o.admission_waits;
}

void estimator_samples::clear() noexcept
{
  if (ttfb.count() != 0) { ttfb.clear(); }
  bodies          = 0;
  body_rate_sum   = 0;
  responses       = 0;
  throttled       = 0;
  delivered       = 0;
  admission_waits = 0;
}

void endpoint_estimator::merge(estimator_samples const& s) noexcept
{
  if (s.ttfb.count() != 0) { _ttfb.merge(s.ttfb); }
  if (s.bodies != 0 && std::isfinite(s.body_rate_sum) && s.body_rate_sum > 0.0) {
    // n sequential EWMA steps with samples equal to the batch mean.
    double const alpha = std::clamp(nonneg(_cfg.ewma_alpha), 0.0, 1.0);
    double const mean  = s.body_rate_sum / static_cast<double>(s.bodies);
    double const keep  = std::pow(1.0 - alpha, static_cast<double>(s.bodies));
    _bc                = keep * _bc + (1.0 - keep) * mean;
  }
  _responses += s.responses;
  _throttled += s.throttled;
  _delivered += s.delivered;
  _admission_waits += s.admission_waits;
}

void endpoint_estimator::set_link_gbps(double gbps) noexcept
{
  if (_link_explicit || !(gbps > 0.0) || !std::isfinite(gbps)) { return; }  // explicit wins
  double const old_target = target_bytes_per_s(_cfg) * std::max(1.0, _ctl.target_scale);
  _cfg.link_gbps          = gbps;
  _link_pub.store(gbps, std::memory_order_relaxed);
  unsigned const seed = seed_conn_target(_cfg);
  if (_steps == 0) {  // still on the seeds: re-seed
    _ctl.conn_target = seed;
  } else {
    // a seed change, not a measurement: the discovered target stays (re-expressed), the
    // connection target is raised to the new seed unless a throttle is recent
    double const t    = target_bytes_per_s(_cfg);
    _ctl.target_scale = t > 0.0 ? std::max(1.0, old_target / t) : 1.0;
    if (_ctl.cautious == 0) { _ctl.conn_target = std::max(_ctl.conn_target, seed); }
  }
  _conn_target.store(_ctl.conn_target, std::memory_order_relaxed);
}

std::optional<controller_move> endpoint_estimator::control_at(std::uint64_t now_ns) noexcept
{
  if (!_clock_started) {
    _clock_started   = true;
    _last_control_ns = now_ns;
    return std::nullopt;
  }
  auto const period_ms          = _cfg.control_period.count() > 0 ? _cfg.control_period.count() : 1;
  std::uint64_t const period_ns = static_cast<std::uint64_t>(period_ms) * 1'000'000ULL;
  if (now_ns < _last_control_ns || now_ns - _last_control_ns < period_ns) { return std::nullopt; }
  double const period_s = static_cast<double>(now_ns - _last_control_ns) * 1e-9;
  _last_control_ns      = now_ns;
  return control(period_s);
}

controller_move endpoint_estimator::control(double period_s) noexcept
{
  ++_steps;
  auto const reset_period = [this]() noexcept {
    _delivered       = 0;
    _responses       = 0;
    _throttled       = 0;
    _admission_waits = 0;
  };
  if (!_cfg.enable_controller) {  // seeds pinned permanently
    reset_period();
    return controller_move::hold;
  }
  double p50 = _p50.load(std::memory_order_relaxed);
  double p99 = _p99.load(std::memory_order_relaxed);
  if (_ttfb.count() > 0) {
    p50 = _ttfb.quantile(0.50);
    p99 = _ttfb.quantile(0.99);
    _ttfb.decay();  // age: older periods weigh 1/2, 1/4, ...
  }
  double const target  = target_bytes_per_s(_cfg);
  double const goodput = period_s > 0.0 ? static_cast<double>(_delivered) / period_s : 0.0;
  // link_util against the discovered target (C3; the seeded one until the first probe-up)
  double const found = target * std::max(1.0, _ctl.target_scale);
  double const util  = found > 0.0 ? goodput / found : 0.0;

  controller_inputs in;
  in.goodput_bytes_s       = goodput;
  in.target_bytes_s        = target;
  in.bc_bytes_s            = _bc;
  in.delay_bytes           = p50 * 1e-3 * _bc;
  in.backlog               = backlog();
  in.responses             = _responses;
  in.throttled             = _throttled;
  in.conn_limit            = _conn_limit;
  in.admission_waits       = _admission_waits;
  controller_move const mv = conn_target_step(_ctl, in, _cfg);

  _p50.store(p50, std::memory_order_relaxed);
  _p99.store(p99, std::memory_order_relaxed);
  _bc_pub.store(_bc, std::memory_order_relaxed);
  _util.store(util, std::memory_order_relaxed);
  _conn_target.store(_ctl.conn_target, std::memory_order_relaxed);
  _binding.store(_ctl.throttle_binding, std::memory_order_relaxed);
  _util_measured.store(true, std::memory_order_relaxed);
  reset_period();
  return mv;
}

// ---- endpoint_control ---------------------------------------------------------------------------

std::optional<controller_move> endpoint_control::publish(estimator_samples const& samples,
                                                         unsigned conn_limit,
                                                         std::uint64_t now_ns) noexcept
{
  if (!planner.enable_controller) { return std::nullopt; }
  try {
    std::lock_guard const lock(_mutex);
    estimator.merge(samples);
    // C3: the discovered max_conn_target is bounded by what the reactors can open
    estimator.set_conn_limit(conn_limit);
    return estimator.control_at(now_ns);
  } catch (...) {  // std::mutex::lock failed: skip this hand-off (the next tick retries)
    return std::nullopt;
  }
}

void endpoint_control::apply_link_seed(double gbps) noexcept
{
  try {
    std::lock_guard const lock(_mutex);
    estimator.set_link_gbps(gbps);
  } catch (...) {  // std::mutex::lock failed: the estimator keeps its previous link seed
  }
}

}  // namespace cucascade::io::rest::planner
