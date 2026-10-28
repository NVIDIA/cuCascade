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

// Remote request planning (ported from gpuio's src/planner/remote_segment.cpp): the utilisation
// model, GET size, load-aware coalescing gap, equal-split / part-aligned chunk cutting, the
// reactor's late part alignment, and the multipart part size learned from an ETag (gpuio
// src/s3/s3_client.cpp `etag_part_count` / `estimate_part_size`). Formulas: gpuio ARCHITECTURE
// §7.1; s3_line_rate §2.6, §3.3. Decisions P2-P8, P10, P11.

#include "io/rest/planner/planner_remote.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <system_error>

namespace cucascade::io::rest::planner {
namespace {

constexpr std::uint64_t u64_max = std::numeric_limits<std::uint64_t>::max();

/// Finite and non-negative, else 0.
double nonneg(double v) noexcept { return std::isfinite(v) && v > 0.0 ? v : 0.0; }

std::uint64_t ceil_div(std::uint64_t a, std::uint64_t b) noexcept
{
  return a / b + (a % b != 0 ? 1 : 0);
}

/// [lo, hi] for every GET: hi = max_segment_bytes (0 = no cap), lo = min_segment_bytes (>= 1),
/// lowered to hi when the config inverts them ("every piece <= max_segment_bytes" wins).
struct seg_bounds {
  std::uint64_t lo, hi;
};
seg_bounds bounds(remote_planner_config const& cfg) noexcept
{
  std::uint64_t const hi = cfg.max_segment_bytes == 0 ? u64_max : cfg.max_segment_bytes;
  std::uint64_t const lo = std::min(std::max<std::uint64_t>(cfg.min_segment_bytes, 1), hi);
  return {lo, hi};
}

/// Clamp a byte count computed in floating point to [lo, hi] (NaN -> lo), rounded to the nearest
/// byte (12.75e6 must not become 12'749'999 through 0.85 / 0.15 rounding error).
std::uint64_t clamp_bytes(double s, std::uint64_t lo, std::uint64_t hi) noexcept
{
  if (!(s > static_cast<double>(lo))) { return lo; }
  if (s >= static_cast<double>(hi)) { return hi; }
  return std::clamp(static_cast<std::uint64_t>(std::floor(s + 0.5)), lo, hi);
}

/// P in [s/2, 2s], i.e. 2P >= s and P <= 2s, without overflow.
bool within_factor_two(std::uint64_t p, std::uint64_t s) noexcept
{
  return p >= s / 2 + (s & 1) && p / 2 + (p & 1) <= s;
}

/// Equal split, no runt (uring_ultra `physical_ranges`): n = ceil(L / s), but no piece below lo
/// (n <= floor(L / lo)) and none above hi (n >= ceil(L / hi), which wins).
std::uint64_t equal_split_count(std::uint64_t len,
                                std::uint64_t s,
                                std::uint64_t lo,
                                std::uint64_t hi) noexcept
{
  std::uint64_t const n_target = ceil_div(len, s);
  std::uint64_t const n_hi     = std::max<std::uint64_t>(1, len / lo);
  std::uint64_t const n_lo     = ceil_div(len, hi);
  return std::max(n_lo, std::min(n_target, n_hi));
}

/// Append `piece` split into k equal pieces (sizes differ by <= 1 byte; the remainder goes one
/// byte each to the first pieces). False when `out` is too small.
bool emit_equal(file_range piece,
                std::uint64_t k,
                std::span<file_range> out,
                std::size_t& n) noexcept
{
  if (k == 0 || k > out.size() - n) { return false; }
  std::uint64_t const base = piece.size / k;
  std::uint64_t const rem  = piece.size % k;
  std::uint64_t off        = piece.offset;
  for (std::uint64_t i = 0; i < k; ++i) {
    std::uint64_t const sz = base + (i < rem ? 1 : 0);
    out[n++]               = file_range{off, sz};
    off += sz;
  }
  return true;
}

/// Append `piece`, equal-split when it exceeds hi.
bool emit_capped(file_range piece,
                 std::uint64_t hi,
                 std::span<file_range> out,
                 std::size_t& n) noexcept
{
  return emit_equal(piece, ceil_div(piece.size, hi), out, n);
}

/// Cut [span) at the multiples of P inside it. A head (tail) fragment below lo is merged into the
/// next (previous) piece, so no GET is a runt; pieces above hi are equal-split.
bool cut_at_parts(file_range span,
                  std::uint64_t p,
                  std::uint64_t lo,
                  std::uint64_t hi,
                  std::span<file_range> out,
                  std::size_t& n) noexcept
{
  std::uint64_t const end     = span.offset + span.size;           // caller checked: no wrap
  std::uint64_t const first_k = span.offset / p + 1;               // first boundary strictly inside
  std::uint64_t const last_k  = (end - 1) / p;                     // last boundary strictly inside
  if (last_k < first_k) { return emit_capped(span, hi, out, n); }  // the span sits in one part

  std::uint64_t k_begin = first_k;
  std::uint64_t k_end   = last_k;
  if (first_k * p - span.offset < lo) { ++k_begin; }          // runt head: merge forward
  if (k_begin <= k_end && end - k_end * p < lo) { --k_end; }  // runt tail: merge backward
  std::uint64_t pos = span.offset;
  for (std::uint64_t k = k_begin; k <= k_end; ++k) {
    if (!emit_capped(file_range{pos, k * p - pos}, hi, out, n)) { return false; }
    pos = k * p;
  }
  return emit_capped(file_range{pos, end - pos}, hi, out, n);
}

}  // namespace

// ---- model ------------------------------------------------------------------------------------

double link_bytes_per_s(remote_planner_config const& cfg) noexcept
{
  double const gbps = cfg.link_gbps > 0.0 ? cfg.link_gbps : default_link_gbps;
  return gbps * 1e9 / 8.0;
}

double target_bytes_per_s(remote_planner_config const& cfg) noexcept
{
  return link_bytes_per_s(cfg) * nonneg(cfg.goodput_fraction);
}

double delay_bytes(endpoint_estimate const& est) noexcept
{
  return nonneg(est.ttfb_p50_ms) * 1e-3 * nonneg(est.bc_bytes_s);
}

double connection_utilisation(double s, double d) noexcept
{
  double const denom = nonneg(s) + nonneg(d);
  return denom > 0.0 ? nonneg(s) / denom : 1.0;
}

double utilisation_segment(double d, double u) noexcept
{
  double const uu = std::clamp(nonneg(u), 0.0, 0.999);
  return nonneg(d) * uu / (1.0 - uu);
}

unsigned connections_for(double rate, double bc, double s, double d, double headroom) noexcept
{
  double const per_conn = nonneg(bc) * connection_utilisation(s, d);
  if (!(nonneg(rate) > 0.0)) { return 0; }
  if (!(per_conn > 0.0)) { return std::numeric_limits<unsigned>::max(); }
  double const n = std::ceil(rate / per_conn * nonneg(headroom));
  if (n >= static_cast<double>(std::numeric_limits<unsigned>::max())) {
    return std::numeric_limits<unsigned>::max();
  }
  return static_cast<unsigned>(n);
}

// gpuio's `express_planner_config` (S3 Express One Zone seeds: TTFB 6 ms, 100k GET/s, per-part
// cap 75) is not ported: S3 Express sessions are out of scope (decision S2).

// ---- producer side ----------------------------------------------------------------------------

std::uint64_t segment_target(remote_planner_config const& cfg,
                             endpoint_estimate const& est,
                             std::uint64_t hint) noexcept
{
  auto const [lo, hi] = bounds(cfg);
  if (hint != 0) { return std::clamp(hint, lo, hi); }

  // P2/P3: the measured D replaces the seeded one (30 ms x 75 MB/s) but can only enlarge s;
  // s_par = backlog / conn_target only cuts the TAIL (backlog < conn_target x s_util), never a
  // deep backlog (it shrank 100 Gbps GETs to ~8 MiB on EC2 while s_util was ~26 MiB).
  double const d_seed =
    nonneg(cfg.assumed_ttfb_ms) * 1e-3 * nonneg(cfg.assumed_conn_gbps) * 1e9 / 8.0;
  double const d      = std::max(delay_bytes(est), d_seed);
  double const s_util = utilisation_segment(d, cfg.target_conn_util);
  double const conns  = static_cast<double>(std::max(1U, est.conn_target));
  double const s_par  = static_cast<double>(est.backlog) / conns;
  bool const tail     = static_cast<double>(est.backlog) < conns * s_util;
  // C4: an auto link (link_gbps 0) uses the estimator's seed (the IMDS table on EC2).
  double const target = cfg.link_gbps <= 0.0 && est.link_gbps > 0.0
                          ? est.link_gbps * 1e9 / 8.0 * nonneg(cfg.goodput_fraction)
                          : target_bytes_per_s(cfg);
  double const s_rate =
    cfg.prefix_get_rate_budget > 0.0 ? target / cfg.prefix_get_rate_budget : 0.0;
  std::uint64_t s = clamp_bytes(std::max(tail ? std::min(s_util, s_par) : s_util, s_rate), lo, hi);
  if (est.link_util > saturated_link_util) { s = std::max(s, std::min(cfg.saturated_floor, hi)); }
  return s;
}

std::uint64_t auto_coalesce_gap(remote_planner_config const& cfg,
                                endpoint_estimate const& est) noexcept
{
  if (cfg.coalesce_gap_fixed != 0) { return cfg.coalesce_gap_fixed; }
  // P10: an unmeasured link counts as saturated: the gap starts at coalesce_gap_min (a cold
  // gap of D, 2-4 MiB, over-read ~5% on parquet-like traces on EC2; 0.02% after the fix).
  double const util      = est.util_measured ? std::clamp(nonneg(est.link_util), 0.0, 1.0) : 1.0;
  std::uint64_t const lo = cfg.coalesce_gap_min;
  std::uint64_t const hi = std::max(lo, cfg.coalesce_gap_max);
  return clamp_bytes(delay_bytes(est) * (1.0 - util), lo, hi);
}

coalesce_params remote_coalesce_params(remote_planner_config const& cfg,
                                       endpoint_estimate const& est,
                                       std::uint64_t gap_override) noexcept
{
  return coalesce_params{gap_override != 0 ? gap_override : auto_coalesce_gap(cfg, est),
                         cfg.max_coalesced_span,
                         cfg.max_read_amplification};
}

std::size_t remote_chunk_capacity(std::uint64_t span_bytes,
                                  remote_planner_config const& cfg) noexcept
{
  auto const [lo, hi]   = bounds(cfg);
  std::uint64_t const m = std::min(lo, std::max<std::uint64_t>(1, hi / 2));
  std::uint64_t const n = std::max<std::uint64_t>(1, ceil_div(span_bytes, m));
  return n > std::numeric_limits<std::size_t>::max() ? std::numeric_limits<std::size_t>::max()
                                                     : static_cast<std::size_t>(n);
}

std::size_t cut_remote_chunks(file_range span,
                              std::uint64_t s,
                              std::optional<std::uint64_t> part_size,
                              remote_planner_config const& cfg,
                              std::span<file_range> out) noexcept
{
  if (span.size == 0 || span.size > u64_max - span.offset) { return 0; }
  auto const [lo, hi]      = bounds(cfg);
  std::uint64_t const n    = equal_split_count(span.size, std::clamp(s, lo, hi), lo, hi);
  std::uint64_t const s_eq = ceil_div(span.size, n);  // s' of gpuio ARCHITECTURE §7.1
  std::size_t written      = 0;
  bool ok                  = false;
  if (cfg.align_to_parts && part_size && *part_size >= lo && within_factor_two(*part_size, s_eq)) {
    ok = cut_at_parts(span, *part_size, lo, hi, out, written);
  } else {
    ok = emit_equal(span, n, out, written);
  }
  return ok ? written : 0;
}

std::size_t recut_chunk_for_parts(file_range chunk,
                                  std::uint64_t part_size,
                                  remote_planner_config const& cfg,
                                  std::span<file_range> out) noexcept
{
  if (out.empty()) { return 0; }
  if (chunk.size == 0 || chunk.size > u64_max - chunk.offset) {
    out[0] = chunk;
    return 1;
  }
  auto const [lo, hi] = bounds(cfg);
  std::size_t written = 0;
  bool const align =
    cfg.align_to_parts && part_size >= lo && within_factor_two(part_size, chunk.size);
  bool const ok = align ? cut_at_parts(chunk, part_size, lo, hi, out, written)
                        : emit_capped(chunk, hi, out, written);
  return ok ? written : 0;
}

// ---- parts --------------------------------------------------------------------------------------

std::optional<unsigned> etag_part_count(std::string_view etag) noexcept
{
  if (!etag.empty() && etag.back() == '"') { etag.remove_suffix(1); }
  auto const dash = etag.rfind('-');
  if (dash == std::string_view::npos || dash + 1 >= etag.size()) { return std::nullopt; }
  unsigned v    = 0;
  auto const rc = std::from_chars(etag.data() + dash + 1, etag.data() + etag.size(), v);
  if (rc.ec != std::errc{} || rc.ptr != etag.data() + etag.size() || v == 0) {
    return std::nullopt;
  }
  return v;
}

std::uint64_t estimate_part_size(std::uint64_t size, unsigned parts) noexcept
{
  if (parts <= 1 || size == 0) { return 0; }
  std::uint64_t const lo = (size + parts - 1) / parts;
  std::uint64_t const hi = (size - 1) / (parts - 1);
  for (std::uint64_t p = 1 * MiB; p != 0 && p <= hi; p <<= 1) {
    if (p >= lo) { return p; }
  }
  return (lo + MiB - 1) / MiB * MiB;
}

unsigned part_cap(remote_planner_config const& cfg, unsigned live_reactors) noexcept
{
  // The global per-part cap split evenly over the live reactors, min 1 (P9).
  unsigned const cap = max_inflight_per_part(cfg);
  if (cap == std::numeric_limits<unsigned>::max()) { return cap; }
  unsigned const live = std::max(1U, live_reactors);
  return std::max(1U, cap / live);
}

std::size_t chunks_per_part(std::uint64_t chunk_bytes, std::uint64_t part_size) noexcept
{
  if (part_size == 0 || chunk_bytes == 0) { return 1; }
  // round(P / chunk), half up, without overflow
  std::uint64_t const k =
    part_size / chunk_bytes + (part_size % chunk_bytes >= chunk_bytes - chunk_bytes / 2 ? 1 : 0);
  return k <= 1 ? 1 : static_cast<std::size_t>(k);
}

}  // namespace cucascade::io::rest::planner
