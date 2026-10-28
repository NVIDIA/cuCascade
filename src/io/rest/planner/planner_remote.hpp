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

#pragma once

#include "io/rest/common.hpp"
#include "io/rest/http/classify.hpp"
#include "io/rest/planner/config.hpp"
#include "io/rest/planner/planner_common.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>

/**
 * @file planner_remote.hpp
 * @brief Remote (S3) planner of the io_uring REST transport, ported from gpuio's
 *        `planner_remote.hpp`: GET sizing, load-aware coalescing gap, chunk cutting, part
 *        alignment, per-part caps, latency class and weighted DRR, the admission cap of the
 *        controller, two-class retries with full jitter and a token budget, deadlines, and the
 *        per-endpoint estimator / connection-target controller.
 *
 * Design: gpuio ARCHITECTURE §7.1 (planner, re-cut cases and their credit rules), §7.3 (retries,
 * deadlines), §2.4 (priorities, fairness); s3_line_rate §2.6 (utilisation model), §2.8 (retry
 * policy), §3.1-§3.3; cuCascade docs/io/rest-uring-design.md §3.1-§3.4. Decision ids (P*, C*,
 * X*) refer to the gpuio performance-decision catalogue.
 *
 * Everything here is pure: no reactor, no I/O, no allocation (outputs go to caller-provided
 * spans) and `noexcept`. The only shared mutable state is `endpoint_estimator` / `retry_budget`
 * (atomics) and `endpoint_control` (a mutex taken once per reactor tick, never per GET).
 *
 * @par How the REST worker drives the planner
 *
 * Ownership. One `endpoint_control` per endpoint (scheme, host, port, TLS policy) shared by all
 * reactors of an ioctx: it holds the endpoint's `remote_planner_config`, `endpoint_estimator`
 * (C1-C7) and `retry_budget` (X2). Each reactor keeps, per endpoint, an `estimator_samples`
 * (reactor-local, single writer), a `throttle_guard` (C6), its busy / idle / pending counts, a
 * `part_inflight_table` (P9) and one `drr_scheduler` (P12). The per-object multipart part size is
 * learned from the ETag (`etag_part_count` + `estimate_part_size`, P8) at open / HEAD / probe or
 * from the first response.
 *
 * Per request (a slice entering the worker; before any GET of it is cut):
 *   1. `auto est = ep.estimator.snapshot_for_request(bytes)`: the request's own bytes enter the
 *      endpoint backlog before sizing (P4); without that a cold endpoint plans 1 MiB GETs.
 *   2. Coalescing (when the caller hands several ranges, e.g. `align_and_coalesce`):
 *      `coalesce(dsts, remote_coalesce_params(cfg, est, caller_gap), spans)`; a non-zero caller
 *      gap wins, else the load-aware gap starting at `coalesce_gap_min` (P10, P11).
 *   3. `s = segment_target(cfg, est, 0)` once per request (P2, P3, P5, P6), then per span
 *      `cut_remote_chunks(span, s, part_size, cfg, out)` with `out` sized by
 *      `remote_chunk_capacity` (P7, P8). One chunk == one GET; credits = chunks. A fragmented
 *      cache fill keeps cuCascade's rule (one fill is never split; contiguous fills merge up to
 *      `s`, each GET <= `max_segment_bytes`).
 *   4. Scheduling: link the request's `drr_hook` (`weight = drr_weight(prio, sched)`) into the
 *      reactor's `drr_scheduler` with `request_class(...)` (latency class: FIFO first, may use the
 *      `latency_reserved_connections`).
 *
 * Per dispatch pass: `drr_scheduler::next(cost)` where `cost` returns `drr_skip` for a request
 * that cannot send (nothing pending, `max_inflight_per_request`, every chunk's part at its
 * `part_cap`), `drr_wait` when `over_cap(...)` holds (the endpoint cannot admit: the deficit is
 * kept so the 4:2:1 weights survive a binding cap), else the bytes of the next GET. When `over_cap`
 * defers a GET and `counts_as_admission_wait(admit_cap, max_conns)` holds, increment
 * `samples.admission_waits` (C7: only the controller's cap feeds the probe, never the hard
 * connection limit). Retries waiting in the deadline heap are re-admitted through the same
 * `over_cap` gate before new GETs. Connect-ahead per endpoint: `connect_ahead_count(pending, busy,
 * idle, connecting, conn_cap)` (C8). Late part alignment (P8 case (c)): when the part size became
 * known after the cut, `recut_chunk_for_parts` (at most `max_recut_pieces`) and `add_credits(k -
 * 1)` before the first piece is prepared.
 *
 * Per GET start: `part_inflight_table::try_acquire(object, part_index(off, P), part_cap(cfg,
 * live))`; arm `stall_deadline` (progress-based) and `attempt_deadline` in the reactor's deadline
 * heap (X7; never linked timeouts).
 *
 * Per GET completion (success or failure): `samples.record_response(503 || 429)`,
 * `samples.record_ttfb(ms)` when a first byte arrived, `samples.record_body(bytes, seconds)` (it
 * keeps only bodies >= 1 MiB) and `samples.record_delivered(bytes landed)`; on success
 * `budget.on_success(attempts == 1 && stale_retries == 0)`; on 503/429
 * `guard.on_throttle(busy_after_release, now, cfg.control_period)` (C6); release the part slot;
 * `estimator.sub_backlog(bytes)` once per GET completed or dropped (P4: the backlog must stay
 * exact, cancellations and build aborts included).
 *
 * Per reactor tick (gpuio: 100 ms while busy): `ep.publish(samples, max_conns x live_reactors,
 * now_ns)` then `samples.clear()`. `publish` merges the samples and runs `control_at(now)`, which
 * steps the controller every `control_period` (C1-C3, C5's binding flag) under the endpoint
 * mutex. The C4 seed arrives once from the helper thread: `ep.apply_link_seed(gbps)` (see
 * `io/rest/cloud/imds.hpp`).
 *
 * Per failure (after `http::classify`): `d = decide_retry(outcome, chunk, received,
 * retry_after_ms, op_retry_state, retry_cfg, ep.budget, rng())`, then `fail` / `retry_now`
 * (free stale retry X3 or re-sign) / `retry_later` after `d.delay` from the deadline heap with
 * `d.range` (resume X4: same credit, `If-Match`). cuCascade's budgets map onto `retry_config`:
 * `max_retry_attempts` -> `max_attempts` (total attempts, 0 read as 1), `max_auth_retry_attempts`
 * -> `max_auth_attempts`; control-plane requests (HEAD, probe, LIST) use `max_attempts = 3`.
 * `d.new_connection` / `d.different_ip` are computed but gpuio consumed neither (X11): the failed
 * connection is closed by `classify`'s reuse flag and DNS round robin spreads addresses.
 *
 * Credit rules for the reactor-side re-cuts (gpuio ARCHITECTURE §7.1 item 3: add before the
 * derived op can finish, while the chunk's own credit is still held):
 *   (a) resume (`decide_retry` with `resume_range`): same chunk, same credit, no `add_credits`;
 *   (b) hedge: not ported (X8 was off by default in gpuio);
 *   (c) late alignment (`recut_chunk_for_parts` returns k): add k - 1 credits BEFORE the first
 *       piece is prepared, then each piece settles one credit.
 */

namespace cucascade::io::rest::planner {

namespace http = ::cucascade::io::http;

// =================================================================================================
// Endpoint estimate and the utilisation model (s3_line_rate §2.6)
// =================================================================================================

/// One snapshot of an endpoint's estimator, read once per request.
struct endpoint_estimate {
  double ttfb_p50_ms{0};
  double ttfb_p99_ms{0};
  double bc_bytes_s{0};  ///< per-connection body rate B_c
  double link_util{0};   ///< goodput / target goodput, in [0, 1+]
  unsigned conn_target{0};
  /// Bytes queued or in flight for the endpoint. For `segment_target` it must include the bytes
  /// of the request being planned (`endpoint_estimator::snapshot_for_request`): otherwise a cold
  /// endpoint has backlog 0 and plans `min_segment_bytes` GETs (P4).
  std::uint64_t backlog{0};
  /// The seeded link in Gbps (explicit `link_gbps`, else the IMDS table / 10); 0 = unknown (a
  /// default-constructed estimate: `segment_target` then reads `cfg.link_gbps`).
  double link_gbps{0};
  /// link_util comes from a controller step (false: the seed). `auto_coalesce_gap` reads an
  /// unmeasured link as saturated (cold gap = coalesce_gap_min, widened only when measurements
  /// show slack); the saturated segment floor needs a measured one (P6, P10).
  bool util_measured{false};
};

/// 10 Gbps: the `link_gbps` fallback off EC2 (or when IMDS does not answer).
inline constexpr double default_link_gbps = 10.0;
/// C4: the seed for an EC2 instance type missing from the table (IMDS answered).
inline constexpr double unknown_ec2_link_gbps = 25.0;

/**
 * @brief C4: the NIC bandwidth of an EC2 instance type, Gbps, from the "Network specifications"
 * tables of the AWS instance type pages (fetched 2026-10-12):
 * https://docs.aws.amazon.com/ec2/latest/instancetypes/co.html (c6in, c7gn, c7i, c7a),
 * .../gp.html (m6a, m6i, m7i, m7a), .../mo.html (r7i), .../ac.html (g6e, g7e, p5, p5e, p5en,
 * trn1, trn1n, trn2, trn2u), .../so.html (i4i, i7ie, im4gn), .../hpc.html (hpc7g). An "Up to X"
 * burst figure counts as X; multi-card totals (p5.48xlarge 3200, trn2.48xlarge 16 x 200) are the
 * page totals. The table is remote_link_table.cpp.
 * @return nullopt for a type not in the table (case-insensitive, surrounding blanks ignored).
 */
[[nodiscard]] std::optional<double> ec2_instance_link_gbps(std::string_view instance_type) noexcept;
/// @return the controller's link seed: the table value, `unknown_ec2_link_gbps` (25) for an EC2
///         type missing from it, `default_link_gbps` (10) when there is no instance type (IMDS
///         disabled or unreachable: off EC2). An explicit `link_gbps > 0` is applied by the caller.
[[nodiscard]] double link_seed_gbps(std::optional<std::string_view> instance_type) noexcept;

/// Link utilisation above which the link is "saturated" and `saturated_floor` applies (P6;
/// gpuio ARCHITECTURE §7.1: "if est.link_util > 0.9: s = max(s, cfg.saturated_floor)").
inline constexpr double saturated_link_util = 0.9;
/// Controller constants (gpuio ARCHITECTURE §7.1 estimator block; s3_line_rate §3.3 `control()`).
inline constexpr double throttle_rate_threshold = 0.005;  ///< 503/429 share > 0.5% => decrease
inline constexpr double growth_goodput_fraction = 0.9;    ///< grow while goodput < 0.9 x target
inline constexpr double freeze_min_gain         = 0.02;   ///< an increase gaining < 2% is "weak"
inline constexpr unsigned freeze_after_weak     = 2;      ///< two weak increases freeze growth
/// A freeze is re-probed after this many periods (2 s at 250 ms; C2): the limit that made growth
/// useless (CPU, NIC, a throttling burst) may have gone, and nothing else would clear it while the
/// backlog stays deep.
inline constexpr unsigned freeze_hold_periods = 8;
/// After a throttle decrease the target is re-approached additively for this many periods (10 s
/// at 250 ms; C2): an upward retarget is limited to + max(1, conn_target / 8) per period, so a
/// throttled endpoint is never jumped back to the level that drew the 503s (AIMD; the backlog can
/// look shallow while throttled GETs sit in their back-off). Also the window in which the
/// admission cap binds (C5).
inline constexpr unsigned throttle_caution_periods = 40;
/// Link discovery (C3): `link_gbps` (or its 10 Gbps fallback) seeds the goodput target but does
/// not cap it. While goodput reaches `goodput_fraction` x the current target with a deep backlog
/// (and the state is neither cautious nor frozen), the target is scaled by this factor per period
/// and `want` follows; the probing stops by itself once goodput no longer keeps up.
inline constexpr double link_probe_factor = 1.5;
/// Upper bound of the discovered target scale (x 10^4 the seed: 100 Tbps from a 10 Gbps seed).
inline constexpr double link_probe_max_scale = 1.0e4;
inline constexpr double conn_headroom        = 1.1;  ///< want = ceil(target / (Bc u) x 1.1)
/// B_c samples come from bodies of at least this size only (C1; gpuio ARCHITECTURE §7.1:
/// "Bc = EWMA(body_rate for bodies >= 1 MiB)"): smaller bodies measure TTFB, not bandwidth.
inline constexpr std::uint64_t bc_min_body_bytes = 1 * MiB;

/// @return the raw link rate in bytes/s (`link_gbps`, 0 read as `default_link_gbps`).
[[nodiscard]] double link_bytes_per_s(remote_planner_config const& cfg) noexcept;
/// @return the goodput target in bytes/s: link rate x `goodput_fraction` (s3_line_rate §3.2:
///         100 Gbps x 0.92 = 11.5 GB/s).
[[nodiscard]] double target_bytes_per_s(remote_planner_config const& cfg) noexcept;
/// @return D = TTFB_p50 x B_c in bytes: the bytes a connection could have moved while waiting
///         for the first byte (s3_line_rate §2.1 "Request-size efficiency", §2.6). Non-finite or
///         negative inputs count as 0.
[[nodiscard]] double delay_bytes(endpoint_estimate const& est) noexcept;
/// @return u(s) = s / (s + D), the per-connection utilisation of an s-byte GET (s3_line_rate
///         §2.1, model "GET time = TTFB + s/B_c"). 1 when s + D == 0.
[[nodiscard]] double connection_utilisation(double s, double d) noexcept;
/// @return s_util = D x u / (1 - u): the GET size at which a connection is u busy
///         (s3_line_rate §2.6 pressure 1; u = 0.85 gives 5.7 x D). u is clamped to [0, 0.999].
[[nodiscard]] double utilisation_segment(double d, double u) noexcept;
/// @return N = ceil(rate / (B_c x u(s)) x headroom), the connection count that sustains `rate`
///         (s3_line_rate §2.1 "Concurrent connections needed" with headroom 1; the controller's
///         `want` with headroom 1.1, §3.3). 0 when rate <= 0; saturates at UINT_MAX.
[[nodiscard]] unsigned connections_for(
  double rate, double bc, double s, double d, double headroom) noexcept;

// =================================================================================================
// Producer side: gap, segment size, chunk cutting (gpuio ARCHITECTURE §7.1 item 1)
// =================================================================================================

/**
 * @brief GET size for one request, computed once (P2, P3, P5, P6).
 *
 * `hint != 0`: `clamp(hint, min_segment_bytes, max_segment_bytes)`. Otherwise (the steady-state
 * rule of P2/P3; gpuio's header comment still showed an earlier formula, its code is the rule):
 * @code
 *   D      = max(TTFB_p50 x B_c (measured), assumed_ttfb x assumed B_c (seed))
 *   s_util = D x u / (1 - u)                       u = target_conn_util (0.85 -> 5.7 D)
 *   s_par  = backlog / max(1, conn_target)          the TARGET, not the free connections
 *   tail   = backlog < conn_target x s_util         s_par applies only to the tail of a batch
 *   s_rate = goodput target / prefix_get_rate_budget   (0 when the budget is 0)
 *   s      = clamp(max(tail ? min(s_util, s_par) : s_util, s_rate), min_seg, max_seg)
 *   if link_util > 0.9: s = max(s, saturated_floor)      (never above max_segment_bytes)
 * @endcode
 * A measured D can only enlarge s. uring_ultra divided the backlog by the FREE connections, so a
 * cold 1 GiB backlog on 256 idle slots planned 4 MiB GETs at 48-61% connection utilisation; the
 * steady-state `min(s_util, s_par)` let a bounded caller window shrink GETs to ~8 MiB on EC2. An
 * auto link (`link_gbps` 0) takes the goodput target from `est.link_gbps` (the IMDS seed).
 * Bounds: a zero `max_segment_bytes` is read as "no cap"; a `min_segment_bytes` above the max is
 * lowered to it (every piece <= max is the hard rule).
 */
[[nodiscard]] std::uint64_t segment_target(remote_planner_config const& cfg,
                                           endpoint_estimate const& est,
                                           std::uint64_t hint) noexcept;

/// @brief Load-aware coalescing gap (P10; s3_line_rate §2.6 "Rule").
/// @return `coalesce_gap_fixed` when non-zero, else clamp(D x (1 - link_util), coalesce_gap_min,
///         coalesce_gap_max), with an unmeasured link read as saturated (so the cold gap is
///         `coalesce_gap_min`): merging saves connection time iff the hole is < D while
///         connections bind (Arrow's hole_size_limit = TTFB x BW), and shrinks toward the
///         per-request overhead as the link saturates.
[[nodiscard]] std::uint64_t auto_coalesce_gap(remote_planner_config const& cfg,
                                              endpoint_estimate const& est) noexcept;

/// @return the `coalesce` rules for a remote request: gap = `gap_override` (the caller's gap)
///         when non-zero, else `auto_coalesce_gap`; span cap `max_coalesced_span`; amplification
///         cap `max_read_amplification` (P10, P11). The caller's gap wins so a caller that
///         coalesced already is never merged a second time.
[[nodiscard]] coalesce_params remote_coalesce_params(remote_planner_config const& cfg,
                                                     endpoint_estimate const& est,
                                                     std::uint64_t gap_override) noexcept;

/// @return a size for `cut_remote_chunks` / `recut_chunk_for_parts` output arrays that always
///         suffices: ceil(span_bytes / m) with m = min(min_segment_bytes, max_segment_bytes / 2),
///         i.e. ceil(span_bytes / min_segment_bytes) for any config with max >= 2 x min (the
///         defaults: 1 MiB / 32 MiB). At least 1.
[[nodiscard]] std::size_t remote_chunk_capacity(std::uint64_t span_bytes,
                                                remote_planner_config const& cfg) noexcept;

/**
 * @brief Cut one coalesced span into GET-sized chunks (P7, P8; gpuio ARCHITECTURE §7.1).
 *
 * Equal split, no runt (uring_ultra `physical_ranges`): n = ceil(L / s), limited so that no
 * piece is below `min_segment_bytes` (n <= floor(L / min)) and none above `max_segment_bytes`
 * (n >= ceil(L / max), which wins); piece sizes differ by at most one byte (the remainder is
 * spread one byte at a time from the front). With s' = ceil(L / n): when `align_to_parts`,
 * P = `*part_size` is known, P >= min_segment_bytes and P in [s'/2, 2s'] (AWS "GET in the part
 * sizes"; s3_line_rate §2.6), the span is cut at the multiples of P inside it instead. A head or
 * tail fragment below `min_segment_bytes` is merged into its neighbour (no runt GET), and a piece
 * above `max_segment_bytes` is equal-split again. `s` is clamped to [min, max] first. Alignment
 * never extends a read: the chunks tile `span` exactly.
 * @param span the coalesced physical span (object offsets).
 * @param s the request's GET size (`segment_target`).
 * @param part_size P from the object's ETag "-N" (multipart), if known.
 * @param out receives the chunks in offset order; they tile `span` exactly.
 * @return number of chunks (one chunk == one GET); 0 when `span` is empty or ends past 2^64, or
 *         when `out` is too small (size it with `remote_chunk_capacity(span.size, cfg)`).
 */
[[nodiscard]] std::size_t cut_remote_chunks(file_range span,
                                            std::uint64_t s,
                                            std::optional<std::uint64_t> part_size,
                                            remote_planner_config const& cfg,
                                            std::span<file_range> out) noexcept;

/// Late-alignment pieces per chunk the reactor accepts (gpuio `remote_backend.cpp` `max_pieces`:
/// a 32 MiB chunk with 1 MiB parts is 32 pieces); a chunk whose `remote_chunk_capacity` exceeds
/// it is not re-cut.
inline constexpr std::size_t max_recut_pieces = 64;

/**
 * @brief Reactor side, P8 case (c) only: the part size P became known from the first response's
 * ETag after the request was cut. A chunk's bounds are fixed (its neighbours and destinations are
 * planned), so "snapping" cuts the chunk at the multiples of P inside it, with the producer's
 * rules: only when `align_to_parts`, P >= min_segment_bytes and P in [c/2, 2c] (c = chunk size),
 * fragments below `min_segment_bytes` merged into their neighbour, pieces above
 * `max_segment_bytes` equal-split. Outside the window the chunk stays whole unless it exceeds
 * `max_segment_bytes`, in which case it is equal-split.
 *
 * Credits: the caller must add k - 1 credits BEFORE preparing the first piece (while the chunk's
 * own credit is still held), then settle one credit per piece.
 * @return k, the number of pieces written to `out` (1 => no change, `out[0] == chunk`); 0 when
 *         `out` is too small (size it with `remote_chunk_capacity(chunk.size, cfg)`).
 */
[[nodiscard]] std::size_t recut_chunk_for_parts(file_range chunk,
                                                std::uint64_t part_size,
                                                remote_planner_config const& cfg,
                                                std::span<file_range> out) noexcept;

/// @return the remainder `[off + received, end)` that a resumed GET requests (with
///         `If-Match: <etag>`) after a mid-body failure, keeping the received bytes (X4; gpuio
///         ARCHITECTURE §7.3 row "mid-body failure after r bytes"; case (a): same credit).
///         Empty (at the chunk end) when everything was received.
[[nodiscard]] constexpr file_range resume_range(file_range chunk, std::uint64_t received) noexcept
{
  if (received >= chunk.size) { return file_range{chunk.offset + chunk.size, 0}; }
  return file_range{chunk.offset + received, chunk.size - received};
}

// =================================================================================================
// Parts: part size from the ETag, per-part in-flight caps, block-cyclic order (P8, P9)
// =================================================================================================

/// @return the multipart part count of an ETag `"<md5>-N"` (quotes kept), or nullopt (P8).
[[nodiscard]] std::optional<unsigned> etag_part_count(std::string_view etag) noexcept;

/// @return a part size consistent with @p size bytes in @p parts parts: the smallest power-of-two
///         MiB in [ceil(size / parts), floor((size - 1) / (parts - 1))] when one exists (8/16/64
///         MiB tool defaults), else ceil(size / parts) rounded up to a MiB; 0 for one part. The
///         ETag gives the count only (s3_line_rate §2.1 "Range semantics"); the estimate only
///         steers alignment and the per-part caps, never correctness (P8).
[[nodiscard]] std::uint64_t estimate_part_size(std::uint64_t size, unsigned parts) noexcept;

/// @return the index of the part holding `offset` for a part size P (0 when P is 0: the object is
///         one part, e.g. a single PUT). A GET is accounted to the part of its first byte.
[[nodiscard]] constexpr std::uint64_t part_index(std::uint64_t offset,
                                                 std::uint64_t part_size) noexcept
{
  return part_size == 0 ? 0 : offset / part_size;
}

/// @return the cap on concurrent GETs per (object, part): `cfg.max_inflight_per_part` (24 for
///         general-purpose S3; aws-c-s3 s3_util.c:120, s3_line_rate §2.1 "Per-part / per-object
///         server-side limits"); 0 in the config means unlimited.
[[nodiscard]] constexpr unsigned max_inflight_per_part(remote_planner_config const& cfg) noexcept
{
  return cfg.max_inflight_per_part == 0 ? std::numeric_limits<unsigned>::max()
                                        : cfg.max_inflight_per_part;
}

/// @return P9: one reactor's share of the per-part cap, `max_inflight_per_part / live_reactors`
///         (minimum 1; unlimited stays unlimited). gpuio `remote_backend::part_cap`.
[[nodiscard]] unsigned part_cap(remote_planner_config const& cfg, unsigned live_reactors) noexcept;

/// @return how many consecutive chunks of size `chunk_bytes` fall into one part of size P
///         (the block of the block-cyclic order): max(1, round(P / chunk)); 1 when P is unknown.
[[nodiscard]] std::size_t chunks_per_part(std::uint64_t chunk_bytes,
                                          std::uint64_t part_size) noexcept;

/**
 * @brief Block-cyclic order over n chunks in blocks of `block` (aws-c-s3 PR #682 part spreading;
 * s3_line_rate §3 item 3): position p visits the r-th chunk of every block before the (r+1)-th,
 * so consecutive GETs land in different parts. A bijection of [0, n); the last block may be
 * short. `block` 0 or >= n gives the identity.
 * @note gpuio's backend picked in list order (scanning up to 16 chunks and skipping parts at their
 *       cap); the block-cyclic order was used only by its tests. Kept for the worker to choose.
 * @return the chunk index visited at position `pos` (`pos` < n).
 */
[[nodiscard]] constexpr std::size_t block_cyclic_index(std::size_t pos,
                                                       std::size_t n,
                                                       std::size_t block) noexcept
{
  if (block == 0 || block >= n) { return pos; }
  std::size_t const full   = n / block;  // complete blocks
  std::size_t const rem    = n % block;  // chunks in the short last block
  std::size_t const blocks = full + (rem != 0 ? 1 : 0);
  std::size_t round        = 0;
  std::size_t j            = 0;
  if (pos < rem * blocks) {  // rounds where every block (the short one too) has a chunk
    round = pos / blocks;
    j     = pos % blocks;
  } else {  // later rounds: only the complete blocks
    std::size_t const q = pos - rem * blocks;
    round               = rem + q / full;
    j                   = q % full;
  }
  return j * block + round;
}

/// @brief Next chunk in block-cyclic order, skipping unavailable ones (gpuio ARCHITECTURE §7.1:
/// "a chunk whose part is at the cap is skipped this pass and taken later").
/// @param n number of chunks; @param block chunks per part (`chunks_per_part`).
/// @param cursor in/out pick position; advanced past the returned chunk.
/// @param available `bool(std::size_t chunk_index) noexcept`: not yet taken and its part has
///        in-flight capacity.
/// @return the chunk index, or n when none is available in a full lap.
template <class Available>
[[nodiscard]] std::size_t pick_block_cyclic(std::size_t n,
                                            std::size_t block,
                                            std::size_t& cursor,
                                            Available&& available) noexcept
{
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t const pos = (cursor + i) % n;
    std::size_t const idx = block_cyclic_index(pos, n, block);
    if (available(idx)) {
      cursor = (pos + 1) % n;
      return idx;
    }
  }
  return n;
}

/**
 * @brief In-flight GET counts per (object, part), reactor-local, fixed size (open addressing,
 * linear probing, backward-shift deletion; no allocation). Size `Capacity` above the GETs a
 * reactor can have in flight (connections per reactor, 64 by default): if the table is ever full,
 * `try_acquire` admits the GET untracked (degrades to "uncapped", never blocks). The global
 * per-part cap is split across the reactors serving an object by the caller (`part_cap`).
 */
template <std::size_t Capacity = 128>
class part_inflight_table {
  static_assert(Capacity >= 2 && (Capacity & (Capacity - 1)) == 0, "power of two");

 public:
  /// @return true (and count the GET) when (object, part) has fewer than `cap` GETs in flight.
  [[nodiscard]] bool try_acquire(std::uint64_t object, std::uint64_t part, unsigned cap) noexcept
  {
    std::size_t i = home(object, part);
    for (std::size_t probe = 0; probe < Capacity; ++probe, i = (i + 1) & mask) {
      slot& s = _slots[i];
      if (s.count == 0) {  // empty: the key is not tracked yet
        if (cap == 0) { return false; }
        s = slot{object, part, 1};
        ++_used;
        return true;
      }
      if (s.object == object && s.part == part) {
        if (s.count >= cap) { return false; }
        ++s.count;
        return true;
      }
    }
    return cap != 0;  // full table: admit untracked (see class comment)
  }

  /// Release one GET of (object, part); a key that is not tracked is ignored.
  void release(std::uint64_t object, std::uint64_t part) noexcept
  {
    std::size_t i = home(object, part);
    for (std::size_t probe = 0; probe < Capacity; ++probe, i = (i + 1) & mask) {
      slot& s = _slots[i];
      if (s.count == 0) { return; }
      if (s.object == object && s.part == part) {
        if (--s.count == 0) { erase_at(i); }
        return;
      }
    }
  }

  /// @return GETs in flight for (object, part).
  [[nodiscard]] unsigned inflight(std::uint64_t object, std::uint64_t part) const noexcept
  {
    std::size_t i = home(object, part);
    for (std::size_t probe = 0; probe < Capacity; ++probe, i = (i + 1) & mask) {
      slot const& s = _slots[i];
      if (s.count == 0) { return 0; }
      if (s.object == object && s.part == part) { return s.count; }
    }
    return 0;
  }

  /// @return number of distinct (object, part) keys with GETs in flight.
  [[nodiscard]] std::size_t size() const noexcept { return _used; }

 private:
  struct slot {
    std::uint64_t object{0};
    std::uint64_t part{0};
    std::uint32_t count{0};  ///< 0 = empty slot
  };
  static constexpr std::size_t mask = Capacity - 1;

  [[nodiscard]] static std::size_t home(std::uint64_t object, std::uint64_t part) noexcept
  {
    std::uint64_t h = (object * 0x9E3779B97F4A7C15ULL) ^ (part + 0x632BE59BD9B4E019ULL);
    h ^= h >> 31;
    h *= 0xD6E8FEB86659FD93ULL;
    h ^= h >> 32;
    return static_cast<std::size_t>(h) & mask;
  }

  /// Backward-shift deletion keeps probe chains intact without tombstones.
  void erase_at(std::size_t hole) noexcept
  {
    --_used;
    std::size_t j = hole;
    for (;;) {
      j = (j + 1) & mask;
      if (_slots[j].count == 0) { break; }
      std::size_t const h = home(_slots[j].object, _slots[j].part);
      // Move slot j into the hole when its home is not in the cyclic range (hole, j].
      bool const in_range = hole <= j ? (hole < h && h <= j) : (hole < h || h <= j);
      if (!in_range) {
        _slots[hole] = _slots[j];
        hole         = j;
      }
    }
    _slots[hole] = slot{};
  }

  std::array<slot, Capacity> _slots{};
  std::size_t _used{0};
};

// =================================================================================================
// Fairness: latency class, reserved connections, weighted DRR / FIFO (P12)
// =================================================================================================

/// Remote scheduling class of a request (gpuio ARCHITECTURE §7.1:
/// `class = urgent || total <= small_request_bytes ? LATENCY : BULK`).
enum class remote_class : std::uint8_t { latency, bulk };

/// @return `latency` when `prio == urgent`, or when the priority is not explicit and the
///         request's total logical bytes are <= `small_request_bytes` (1 MiB; gpuio
///         ARCHITECTURE §2.4); `bulk` otherwise. `high` is not the latency class. (gpuio took a
///         `read_options`; its two fields are passed here.)
[[nodiscard]] remote_class request_class(priority prio,
                                         bool explicit_priority,
                                         std::uint64_t total_bytes,
                                         remote_planner_config const& cfg) noexcept;

/// @return connections of one reactor reserved for the latency class on an endpoint:
///         max(2, ceil(latency_reserved_fraction x conns)) (s3_line_rate §3.2 "5% reserved
///         (>= 2 per reactor)"), but always leaving bulk at least one (0 when conns <= 1).
[[nodiscard]] unsigned latency_reserved_connections(remote_planner_config const& cfg,
                                                    unsigned conns_per_reactor) noexcept;

/// DRR quantum: >= the largest GET (`max_segment_bytes`, 32 MiB by default), so a bulk request
/// sends at least one GET per round (gpuio `remote_backend.cpp` `drr_quantum`).
inline constexpr std::uint64_t drr_quantum = 64 * MiB;

/// @return the DRR weight of a priority: `remote_weights` {high, normal, bulk} (4 : 2 : 1 by
///         default; zero read as 1). `urgent` rides the latency list (its weight is unused).
///         gpuio `remote_backend::drr_weight`.
[[nodiscard]] std::uint32_t drr_weight(priority prio, remote_scheduler_config const& cfg) noexcept;

/// Intrusive scheduler hook, embedded in whatever represents an active request (no allocation).
struct drr_hook {
  drr_hook* prev{nullptr};
  drr_hook* next{nullptr};
  std::uint64_t deficit{0};  ///< DRR deficit counter, bytes
  /// Quanta per grant (weighted DRR: `deficit += quantum x weight`); set by the owner before
  /// `push` (0 is read as 1). 1 everywhere = plain DRR.
  std::uint32_t weight{1};
  bool latency{false};  ///< on the latency list (bypasses DRR)
  bool linked{false};
};

/// Cost-function result meaning "this request cannot send now" for a request-bound reason (e.g.
/// at `max_inflight_per_request`, nothing pending, every chunk's part at its cap): skipped without
/// charge and its deficit reset (an idle DRR flow keeps no credit).
inline constexpr std::uint64_t drr_skip = std::numeric_limits<std::uint64_t>::max();
/// Cost-function result meaning "the endpoint cannot admit now" (the admission cap, C5;
/// P12): skipped WITHOUT resetting the deficit, and when every request of a lap
/// is skipped this way (or by `drr_skip`) the cursor stays where it was, so a binding cap does not
/// collapse the weights into round-robin.
inline constexpr std::uint64_t drr_wait = std::numeric_limits<std::uint64_t>::max() - 1;

/**
 * @brief Chooses which active request sends the next GET (gpuio ARCHITECTURE §2.4; s3_line_rate
 * §3.3 `next_segment`: `latency_queue.pop_ready() ?: drr.next_bulk_request(...)`).
 *
 * Latency-class requests are served first, FIFO, without deficit accounting (they are small by
 * definition). Bulk requests are served by deficit round-robin (Shreedhar & Varghese): the
 * request at the cursor sends while its deficit covers the cost of its next GET; moving the
 * cursor grants the next request one quantum (times its `weight`: demand reads get more bytes
 * per round than prefetch, never starving it). A request that cannot send is skipped and its
 * deficit reset (`drr_skip`), or skipped keeping its deficit while its endpoint cannot admit
 * (`drr_wait`). With `order::fifo` bulk requests are served oldest first (weights unused).
 * Single-threaded (one per reactor). The scheduler keeps pointers to linked hooks: `remove` a
 * hook before its owner is destroyed.
 */
class drr_scheduler {
 public:
  using order = remote_planner_config::order;

  /// @param quantum bytes granted per round; use >= the largest GET (`max_segment_bytes`) so a
  ///        request sends at least one GET per round. 0 is read as 1.
  explicit drr_scheduler(std::uint64_t quantum, order o = order::drr) noexcept
    : _quantum(quantum == 0 ? 1 : quantum), _order(o)
  {
  }
  drr_scheduler(drr_scheduler const&)            = delete;
  drr_scheduler& operator=(drr_scheduler const&) = delete;
  drr_scheduler(drr_scheduler&&)                 = delete;
  drr_scheduler& operator=(drr_scheduler&&)      = delete;
  ~drr_scheduler()                               = default;

  /// Activate a request (tail of its class's list). No-op when already linked.
  void push(drr_hook& h, remote_class cls) noexcept;
  /// Deactivate a request (no chunk left, or finished). No-op when not linked.
  void remove(drr_hook& h) noexcept;

  /// @brief Pick the request that sends the next GET and charge it.
  /// @param cost `std::uint64_t(drr_hook&) noexcept`: bytes of that request's next GET, or
  ///        `drr_skip` / `drr_wait` when it cannot send now.
  /// @return the chosen hook (still linked; call `remove` when it has nothing left), or nullptr
  ///         when no active request can send.
  template <class Cost>
  [[nodiscard]] drr_hook* next(Cost&& cost) noexcept
  {
    for (drr_hook* h = _latency.head; h != nullptr; h = h->next) {
      if (cost(*h) < drr_wait) { return h; }
    }
    if (_bulk.size == 0) { return nullptr; }
    if (_order == order::fifo) {
      for (drr_hook* h = _bulk.head; h != nullptr; h = h->next) {
        if (cost(*h) < drr_wait) { return h; }
      }
      return nullptr;
    }
    if (_current == nullptr) {
      _current = _bulk.head;
      _fresh   = true;
    }
    drr_hook* const start  = _current;
    bool const start_fresh = _fresh;
    bool waited            = false;
    std::size_t skipped    = 0;
    for (;;) {
      drr_hook& h           = *_current;
      std::uint64_t const c = cost(h);
      if (c >= drr_wait) {
        if (c == drr_skip) {
          h.deficit = 0;  // request-bound: no credit kept (this visit's quantum is spent)
          _fresh    = false;
        } else {
          waited = true;  // endpoint-bound: the deficit (and an ungranted visit) is kept
        }
        if (++skipped >= _bulk.size) {
          if (waited) {  // a lap blocked by admission: resume at the same request later
            _current = start;
            _fresh   = start_fresh;
          }
          return nullptr;
        }
        advance();
        continue;
      }
      skipped = 0;
      if (_fresh) {  // the quantum of this visit (granted once the request can send)
        grant(h);
        _fresh = false;
      }
      if (h.deficit >= c) {
        h.deficit -= c;
        return &h;
      }
      advance();  // terminates: every lap grants each eligible request one more quantum
    }
  }

  [[nodiscard]] bool empty() const noexcept { return _latency.size == 0 && _bulk.size == 0; }
  [[nodiscard]] std::size_t latency_size() const noexcept { return _latency.size; }
  [[nodiscard]] std::size_t bulk_size() const noexcept { return _bulk.size; }
  [[nodiscard]] std::uint64_t quantum() const noexcept { return _quantum; }

 private:
  struct list {
    drr_hook* head{nullptr};
    drr_hook* tail{nullptr};
    std::size_t size{0};
    void push_back(drr_hook& h) noexcept;
    void erase(drr_hook& h) noexcept;
  };

  void grant(drr_hook& h) const noexcept
  {
    std::uint64_t const w = h.weight == 0 ? 1 : h.weight;
    std::uint64_t const q =
      _quantum > std::numeric_limits<std::uint64_t>::max() / w ? drr_wait : _quantum * w;
    h.deficit = h.deficit > drr_wait - 1 - q ? drr_wait - 1 : h.deficit + q;
  }
  /// Move the cursor to the next bulk request (wrapping); its quantum is granted at its first
  /// eligible evaluation (`next`).
  void advance() noexcept
  {
    _current = _current->next != nullptr ? _current->next : _bulk.head;
    _fresh   = true;
  }

  list _latency{};
  list _bulk{};
  drr_hook* _current{nullptr};
  bool _fresh{false};  ///< `_current` has not been granted its quantum for this visit
  std::uint64_t _quantum;
  order _order;
};

// =================================================================================================
// Admission: the controller's lever on the reactor (C5-C8; gpuio remote_backend.cpp)
// =================================================================================================

/// @return C5: one reactor's share of the endpoint's connection target,
///         `max(1, min(max_conns_per_reactor, ceil(conn_target / live_reactors)))`: the planner
///         sized the GETs for `conn_target` connections (s_par = backlog / conn_target).
[[nodiscard]] unsigned conn_cap(unsigned conn_target,
                                unsigned max_conns_per_reactor,
                                unsigned live_reactors) noexcept;

/**
 * @brief C5: the number of GETs (and retries) a reactor may have in flight on the endpoint.
 *
 * With the controller off it is `conn_cap`. With it on, the cap binds only after a throttle: while
 * `throttle_binding` (the estimator's flag: within the caution window of a throttle decrease or
 * while the throttle share is above 0.5%) or `throttle_recent` (this reactor saw a 503/429 within
 * a control period, `throttle_guard::active`) it is `conn_cap`; otherwise every live warm
 * connection may be used: `max(conn_cap, live_connections)`. On a never-throttled endpoint an
 * idle warm connection is pure loss (EC2 A/B: an always-binding cap ran at 73% of the reference).
 */
[[nodiscard]] unsigned admit_cap(unsigned conn_cap,
                                 bool enable_controller,
                                 bool throttle_binding,
                                 bool throttle_recent,
                                 unsigned live_connections) noexcept;

/**
 * @brief C6: fast per-reactor throttle reaction between controller steps.
 *
 * A 503/429 caps this reactor's GETs in flight on the endpoint at the concurrency that drew it
 * (`max(1, busy)`, the own slot of the failed GET already released) until one `control_period` has
 * passed (the controller has stepped by then); repeated throttles within the window lower the cap
 * further. Without it a slot one above the server's limit drained pending chunks into 503s
 * (converged 503 rate 10-20/s -> 2-3.6/s in gpuio's mock). Reactor-local, controller on only.
 */
struct throttle_guard {
  std::uint32_t cap{0};       ///< 0 = no cap
  std::uint64_t until_ns{0};  ///< monotonic ns; the cap holds while now < until_ns

  /// Record a 503/429 seen with @p busy GETs still in flight.
  void on_throttle(std::uint32_t busy,
                   std::uint64_t now_ns,
                   std::chrono::milliseconds control_period) noexcept;
  /// @return true within a control period of the last throttle (`admit_cap`'s throttle_recent).
  [[nodiscard]] bool active(std::uint64_t now_ns) const noexcept { return now_ns < until_ns; }
  /// @return true when the guard alone forbids one more GET at @p busy.
  [[nodiscard]] bool holds(std::uint32_t busy, std::uint64_t now_ns) const noexcept
  {
    return cap != 0 && now_ns < until_ns && busy >= cap;
  }
};

/// @return true when a reactor with @p busy GETs in flight on the endpoint must not start one
///         more (controller on only): the throttle guard holds (C6), or `busy >= admit_cap`, the
///         latency class adding its `latency_reserved_connections(cfg, admit_cap)` on top (P12).
[[nodiscard]] bool over_cap(remote_planner_config const& cfg,
                            throttle_guard const& guard,
                            std::uint64_t now_ns,
                            std::uint32_t busy,
                            unsigned admit_cap,
                            bool latency) noexcept;

/// @return C7: whether a GET deferred by `over_cap` counts as an admission wait (evidence for the
///         probe-up, `estimator_samples::admission_waits`): only when the controller's cap binds
///         (`admit_cap < max(1, max_conns_per_reactor)`), never at the hard connection limit,
///         where probing cannot help (counting those probed ~7x, read util ~0.12 and widened the
///         coalescing gap into a 5% over-read on EC2).
[[nodiscard]] bool counts_as_admission_wait(unsigned admit_cap,
                                            unsigned max_conns_per_reactor) noexcept;

/// @return C8: connections to open ahead now on one endpoint so a burst pays DNS + TCP + TLS once,
///         in parallel: `min(pending + busy, conn_cap) - (busy + idle + connecting)`, 0 when
///         negative. gpuio `remote_backend::connect_ahead_all`.
[[nodiscard]] std::uint32_t connect_ahead_count(std::uint32_t pending,
                                                std::uint32_t busy,
                                                std::uint32_t idle,
                                                std::uint32_t connecting,
                                                unsigned conn_cap) noexcept;

// =================================================================================================
// Retries: two classes with full jitter, budget, resume (X1-X4; gpuio ARCHITECTURE §7.3)
// =================================================================================================

/// Retry-After / x-amz-retry-after are honoured up to 30 s (gpuio ARCHITECTURE §7.3 row 3; the
/// codec caps `response_meta::retry_after_ms` the same way, and the ms header wins over the s one).
inline constexpr std::chrono::milliseconds max_retry_after{30000};

/// Inputs of `backoff_delay`.
struct backoff_inputs {
  http::outcome::retry_class cls;
  unsigned attempt;              ///< 1-based number of the attempt that just failed (0 read as 1)
  std::uint32_t retry_after_ms;  ///< `response_meta::retry_after_ms` (0 = none)
  retry_config const& cfg;
  std::uint64_t rng;  ///< one uniformly random 64-bit word from the caller (top 53 bits used)
};

/// @return the full-jitter ceiling min(backoff_cap, base x 2^(attempt-1)) with base
///         `transient_base` (50 ms) or `throttle_base` (1 s); 0 for the other classes.
[[nodiscard]] std::chrono::milliseconds backoff_ceiling(http::outcome::retry_class cls,
                                                        unsigned attempt,
                                                        retry_config const& cfg) noexcept;

/**
 * @brief Delay before the next attempt (X1; AWS SDK 2026 retry behaviour):
 *   - transient: U(0,1) x min(20 s, 50 ms x 2^k); throttle: U(0,1) x min(20 s, 1 s x 2^k),
 *     k = attempt - 1 ("full jitter", U from the top 53 bits of `rng`, floored to whole ms, so
 *     the delay is uniform over [0, ceiling) ms);
 *   - with `honor_retry_after` and a Retry-After value (transient/throttle only): never earlier
 *     than min(retry_after_ms, 30 s); the jittered delay still applies when it is longer, so GETs
 *     throttled together do not retry in lock step;
 *   - free_stale: 0 (immediate retry on a fresh connection); auth: 0 (re-sign now); none: 0.
 */
[[nodiscard]] std::chrono::milliseconds backoff_delay(backoff_inputs const& in) noexcept;

/// @return retry-budget tokens a retry of this class costs: `transient_cost` (14) or
///         `throttle_cost` (5); 0 for free stale-connection retries and re-signs (X2;
///         s3_line_rate §2.8 "Outage" row).
[[nodiscard]] constexpr unsigned retry_cost(http::outcome::retry_class cls,
                                            retry_config const& cfg) noexcept
{
  using rc = http::outcome::retry_class;
  return cls == rc::transient ? cfg.transient_cost : cls == rc::throttle ? cfg.throttle_cost : 0;
}

/**
 * @brief Per-endpoint retry token bucket (X2; gpuio ARCHITECTURE §7.3 last row; AWS SDK standard
 * mode): starts full at `retry_budget_tokens` (500); a retry spends its class's cost; a first-try
 * success returns one token (capped at the start value). Exhausted => fail fast with the last
 * error. Lock-free (CAS); safe from any reactor. One per endpoint, shared by every reactor (a
 * per-reactor budget would be N times larger).
 */
class retry_budget {
 public:
  explicit retry_budget(retry_config const& cfg) noexcept
    : _tokens(cfg.retry_budget_tokens), _max(cfg.retry_budget_tokens)
  {
  }
  retry_budget(retry_budget const&)            = delete;
  retry_budget& operator=(retry_budget const&) = delete;
  retry_budget(retry_budget&&)                 = delete;
  retry_budget& operator=(retry_budget&&)      = delete;
  ~retry_budget()                              = default;

  /// @return true and spend `cost` tokens when at least `cost` are left (cost 0 always succeeds).
  [[nodiscard]] bool try_spend(unsigned cost) noexcept;
  /// `try_spend(retry_cost(cls, cfg))`.
  [[nodiscard]] bool try_acquire(http::outcome::retry_class cls, retry_config const& cfg) noexcept
  {
    return try_spend(retry_cost(cls, cfg));
  }
  /// A GET succeeded; `first_try` successes return one token (retried successes return none).
  void on_success(bool first_try) noexcept;
  [[nodiscard]] unsigned tokens() const noexcept { return _tokens.load(std::memory_order_relaxed); }
  [[nodiscard]] unsigned capacity() const noexcept { return _max; }

 private:
  std::atomic<unsigned> _tokens;
  unsigned _max;
};

/// Per-chunk retry bookkeeping, kept in the reactor's op slot.
struct retry_state {
  unsigned attempts{1};       ///< attempts made, the failed one included (1 after the first try)
  unsigned stale_retries{0};  ///< free immediate retries used (X3, <= 5)
  unsigned auth_retries{0};   ///< re-signs used (<= max_auth_attempts)
};

enum class retry_action : std::uint8_t {
  fail,        ///< settle the chunk with the outcome's error
  retry_now,   ///< re-issue immediately (free stale retry, or re-sign)
  retry_later  ///< re-issue `range` after `delay` (reactor deadline heap, never a linked timeout)
};

/// What to do after a failed exchange.
struct retry_decision {
  retry_action action{retry_action::fail};
  std::chrono::milliseconds delay{0};
  /// What the next attempt requests: the whole chunk, or `[off + received, end)` (resume, with
  /// `If-Match: <etag>`, same credit). Equal to the chunk on `fail`.
  file_range range{};
  bool new_connection{false};    ///< do not reuse the failed connection for the retry (X11)
  bool different_ip{false};      ///< prefer another resolved address (transient; X11)
  bool resign{false};            ///< re-sign (auth class; fresh credentials / clock offset)
  bool budget_exhausted{false};  ///< `fail` because the retry budget is empty (log request ids)
};

/**
 * @brief `http::classify`-driven retry transition for one chunk (gpuio ARCHITECTURE §7.3 table).
 *   - none: fail.
 *   - free_stale (reused connection died before any byte): `retry_now` on a fresh connection,
 *     no attempt, no backoff, no budget, up to `stale_connection_free_retries` (5); beyond that
 *     it is an ordinary transient failure (X3).
 *   - auth: `retry_now` with `resign` up to `max_auth_attempts` (3) re-signs, then fail.
 *   - transient / throttle: fail when `attempts >= max_attempts` (10 for data; metadata callers
 *     pass a config with 3) or when the budget cannot pay the class cost (`budget_exhausted`);
 *     else `retry_later` with `backoff_delay`, `attempts + 1`. Transient retries use a new
 *     connection and another IP; throttle retries may reuse a connection the outcome kept.
 *     When `o.resume_allowed` and `received > 0`, `range = resume_range(chunk, received)` (X4).
 * @param received body bytes already placed for this chunk (the resume offset).
 * @param rng one uniformly random 64-bit word for the jitter.
 */
[[nodiscard]] retry_decision decide_retry(http::outcome const& o,
                                          file_range chunk,
                                          std::uint64_t received,
                                          std::uint32_t retry_after_ms,
                                          retry_state& st,
                                          retry_config const& cfg,
                                          retry_budget& budget,
                                          std::uint64_t rng) noexcept;

// =================================================================================================
// Deadlines (X7; they live in the reactor's deadline heap, never as linked timeouts)
// =================================================================================================

// gpuio's `hedge_ttfb_timeout` (clamp(3 x ttfb_p99, 200 ms, 2 s)) belonged to hedging (X8, off
// by default) and is not ported.

/// @return the stall deadline: no completed receive for max(stall_timeout, 3 x body_recv_cap /
///         B_c) => ASYNC_CANCEL + resume (X7; 3 receives' worth at the measured per-connection
///         rate, never below 5 s). Progress-based, so a slow-but-moving body is never cut.
[[nodiscard]] std::chrono::milliseconds stall_deadline(remote_planner_config const& cfg,
                                                       std::uint64_t body_recv_cap,
                                                       double bc_bytes_s) noexcept;
/// @return the attempt deadline 10 s + bytes / min_rate_bytes_s (1 MB/s; replaces uring_ultra's
///         fixed 30 s, s3_line_rate §2.8 "Last resort").
[[nodiscard]] std::chrono::milliseconds attempt_deadline(remote_planner_config const& cfg,
                                                         std::uint64_t bytes) noexcept;

// =================================================================================================
// Hedging (X8): not ported. gpuio's `hedge_probe`, `hedge_plan`, `plan_hedge`,
// `hedge_budget_allows` and `hedge_slow_body_fraction` (resume-split hedges, src/planner/
// remote_hedge.cpp) were off by default (`enable_hedging = false`) pending an EC2 decision; the
// credit rule they needed (case (b) above) is the only trace left here.
// =================================================================================================

// =================================================================================================
// Estimator: TTFB quantiles, EWMA B_c, goodput, conn_target controller (C1-C3)
// =================================================================================================

/**
 * @brief Fixed-size quantile sketch for TTFB (ms): a log-bucketed histogram, ratio 1.01 per bucket
 * from 10 us to 100 s, so any quantile is within 0.5% of a sample in its bucket (plus rank
 * granularity). Chosen over P^2 because reactor-local sketches merge exactly (counts add) and the
 * error bound does not depend on the distribution (C1). 6.5 KiB, no allocation.
 */
class quantile_sketch {
 public:
  static constexpr double min_value = 0.01;   ///< ms; smaller samples land in the first bucket
  static constexpr double max_value = 1.0e5;  ///< ms; larger samples land in the last bucket
  static constexpr double gamma     = 1.01;   ///< bucket ratio
  static constexpr double ln_gamma  = 0.00995033085316808285;  ///< ln(1.01)
  /// ceil(ln(max/min) / ln(gamma)) = 1620 interior buckets, plus underflow and overflow.
  static constexpr std::size_t buckets = 1622;

  void add(double v) noexcept;
  void merge(quantile_sketch const& o) noexcept;
  /// Halve every count (exponential ageing, once per control period).
  void decay() noexcept;
  void clear() noexcept;
  [[nodiscard]] std::uint64_t count() const noexcept { return _total; }
  /// @return the q-quantile (rank ceil(q x count), q clamped to [0, 1]) as the geometric centre
  ///         of its bucket; 0 when empty.
  [[nodiscard]] double quantile(double q) const noexcept;

 private:
  [[nodiscard]] static std::size_t bucket_of(double v) noexcept;
  [[nodiscard]] static double value_of(std::size_t bucket) noexcept;

  std::array<std::uint32_t, buckets> _counts{};
  std::uint64_t _total{0};
};

/// Per-period measurements fed to the connection-target controller.
struct controller_inputs {
  double goodput_bytes_s{0};  ///< delivered body bytes / period
  double target_bytes_s{0};   ///< `target_bytes_per_s(cfg)`
  double bc_bytes_s{0};       ///< EWMA B_c
  double delay_bytes{0};      ///< D = ttfb_p50 x B_c
  std::uint64_t backlog{0};
  std::uint64_t responses{0};  ///< responses completed in the period
  std::uint64_t throttled{0};  ///< of which 503 SlowDown / 429
  /// C3: connection ceiling of the endpoint (max_connections_per_reactor x live reactors); 0 =
  /// none. Bounds the discovered `max_conn_target`.
  unsigned conn_limit{0};
  /// C7: dispatch decisions in the period at which GETs waited for the controller's admission cap.
  std::uint64_t admission_waits{0};
};

/// Controller memory between periods.
struct controller_state {
  unsigned conn_target{0};
  unsigned weak_increases{0};  ///< consecutive increases that gained < 2%
  bool frozen{false};          ///< additive growth stopped (CPU-, NIC- or burst-limited)
  unsigned frozen_periods{0};  ///< periods held frozen; growth is re-probed at freeze_hold_periods
  unsigned cautious{0};        ///< periods left with additive-only growth after a throttle
  double target_scale{1.0};    ///< C3: discovered goodput target / the seeded one (>= 1)
  bool pending_probe{false};   ///< the pending increase is a probe-up (reverted when weak)
  double scale_before{1.0};    ///< target_scale before that probe
  unsigned ct_before{0};       ///< conn_target before that probe
  unsigned weak_probes{0};     ///< consecutive weak probe-ups
  unsigned probe_hold{0};      ///< periods left without probing (two weak probes: 2 s)
  double goodput_prev{0};      ///< previous period's goodput (2-period mean for the gain test)
  bool throttle_binding{false};  ///< the admission cap binds (see `endpoint_estimator`)
  bool pending_increase{false};
  double goodput_before_increase{0};
};

/// What a controller step did.
enum class controller_move : std::uint8_t {
  hold,               ///< controller disabled: seeds pinned
  throttle_decrease,  ///< x throttle_decrease (0.75), floor min_conns_total
  additive_increase,  ///< + max(1, conn_target / 8), cap max connections
  frozen_hold,        ///< would grow, but two weak increases froze growth
  retarget,           ///< conn_target = clamp(want, min, max)
  probe_up            ///< C3: goodput keeps up with the target: target x 1.5, want follows
};

/// @return the seeded connection target before any measurement: want = ceil(target /
///         (B_c x u(s_typ)) x 1.1) from the seeds (B_c = assumed_conn_gbps, TTFB =
///         assumed_ttfb_ms, s_typ = clamp(s_util)), clamped to [min_conns_total, max].
///         10 Gbps: 20; 100 Gbps: 199 (s3_line_rate §2.1 table, plus 10% headroom).
[[nodiscard]] unsigned seed_conn_target(remote_planner_config const& cfg) noexcept;
/// @return `max_conns_total`, or 1.5 x the initial target when it is 0.
[[nodiscard]] unsigned max_conn_target(remote_planner_config const& cfg) noexcept;
/// @return C3: the ceiling that follows the discovered target: `max_conns_total` when set, else
///         1.5 x the seeded want at `target_scale` x the seeded goodput target, bounded by
///         @p conn_limit (max_connections_per_reactor x live reactors; 0 = none) and never below
///         min_conns_total. `target_scale` 1 and no limit == `max_conn_target(cfg)`.
[[nodiscard]] unsigned max_conn_target(remote_planner_config const& cfg,
                                       double target_scale,
                                       unsigned conn_limit) noexcept;

/**
 * @brief One controller period (C2, C3; gpuio ARCHITECTURE §7.1 "conn_target: AIMD";
 * s3_line_rate §3.3 `control()`):
 * @code
 *   (first, judge the previous increase: gain < 2% => weak; two weak in a row => frozen; a weak
 *    probe-up is reverted, two weak probes hold probing for freeze_hold_periods)
 *   target = in.target x target_scale; max = ceiling(target) (seeded AND measured B_c, <= limit)
 *   if 503/429 share > 0.5%:                      conn_target = max(min, conn_target x 0.75)
 *   elif (goodput >= goodput_fraction x target and backlog >= conn_target x s_typ, or GETs
 *        waited for admission while goodput < goodput_fraction x target) and not cautious,
 *        frozen or in a probe hold:             target_scale x= 1.5, conn_target =
 *        max(conn_target, clamp(want(target), min, max))  (+ n/8 when queued and want is flat)
 *   elif goodput < 0.9 x target and backlog > conn_target x s_typ:
 *        frozen ? hold : conn_target = min(max, conn_target + max(1, conn_target / 8))
 *        (a freeze held for freeze_hold_periods is lifted and growth probed again)
 *   else conn_target = clamp(ceil(target / (B_c x s_typ / (s_typ + D)) x 1.1), min, max)
 *        (within throttle_caution_periods of a decrease, upward by at most max(1, n / 8))
 * @endcode
 * A throttle decrease or a retarget clears the freeze (the binding limit changed). TTFB is not a
 * control signal; it feeds D only.
 */
controller_move conn_target_step(controller_state& st,
                                 controller_inputs const& in,
                                 remote_planner_config const& cfg) noexcept;

/**
 * @brief Reactor-local observations of one (reactor, endpoint) since the last hand-off, merged
 * into the endpoint's estimator by the controller step (C1: cross-reactor aggregation). Everything
 * here merges exactly: the TTFB log-histogram counts add, the B_c samples travel as (count, sum of
 * rates), the response / throttle / delivered counters add. No allocation.
 */
struct estimator_samples {
  quantile_sketch ttfb{};            ///< ms
  std::uint64_t bodies{0};           ///< bodies of >= bc_min_body_bytes
  double body_rate_sum{0};           ///< sum of their rates, bytes/s
  std::uint64_t responses{0};        ///< responses (any status)
  std::uint64_t throttled{0};        ///< of which 503 SlowDown / 429
  std::uint64_t delivered{0};        ///< body bytes delivered to destinations
  std::uint64_t admission_waits{0};  ///< C7: dispatch decisions deferred by the admission cap

  void record_ttfb(double ms) noexcept;
  /// Same filter as `endpoint_estimator::record_body` (>= 1 MiB, positive finite duration).
  void record_body(std::uint64_t bytes, double seconds) noexcept;
  void record_response(bool is_throttled) noexcept;
  void record_delivered(std::uint64_t bytes) noexcept { delivered += bytes; }
  void merge(estimator_samples const& o) noexcept;
  void clear() noexcept;
  [[nodiscard]] bool empty() const noexcept
  {
    return ttfb.count() == 0 && bodies == 0 && responses == 0 && delivered == 0 &&
           admission_waits == 0;
  }
};

/**
 * @brief Per-endpoint estimator (C1; gpuio ARCHITECTURE §7.1 "estimator per endpoint").
 *
 * Threading: `snapshot`, `snapshot_for_request`, `add_backlog`, `sub_backlog`, `backlog` and
 * `throttle_binding` are safe from any thread (relaxed atomics; the snapshot's fields may come
 * from adjacent periods, which is fine for an estimate). `record_*`, `merge`, `control`,
 * `control_at`, `set_conn_limit` and `set_link_gbps` are single-writer: every reactor records into
 * its own `estimator_samples` and hands them to `endpoint_control::publish` from its tick, which
 * merges them and calls `control_at(now)` under the endpoint's mutex (cross-reactor aggregation).
 *
 * Seeds before any measurement: TTFB p50 = p99 = assumed_ttfb_ms (30), B_c = assumed_conn_gbps
 * (0.6 Gbps = 75 MB/s), link_util 0, conn_target = `seed_conn_target`. The snapshot stays on the
 * seeds until the first controller step (`control_at`'s first call only starts the clock);
 * `control()` publishes measured values and runs `conn_target_step`; with `enable_controller ==
 * false` it never changes the published seeds.
 */
class endpoint_estimator {
 public:
  explicit endpoint_estimator(remote_planner_config const& cfg) noexcept;
  endpoint_estimator(endpoint_estimator const&)            = delete;
  endpoint_estimator& operator=(endpoint_estimator const&) = delete;
  endpoint_estimator(endpoint_estimator&&)                 = delete;
  endpoint_estimator& operator=(endpoint_estimator&&)      = delete;
  ~endpoint_estimator()                                    = default;

  /// @return the published estimate with the current backlog.
  [[nodiscard]] endpoint_estimate snapshot() const noexcept;
  /// P4: add a request's coalesced bytes to the backlog (one relaxed RMW per request) and return
  /// the snapshot `segment_target` needs: its backlog includes this request.
  [[nodiscard]] endpoint_estimate snapshot_for_request(std::uint64_t request_bytes) noexcept;
  void add_backlog(std::uint64_t bytes) noexcept;
  /// Per GET completion (or drop) of `bytes` planned bytes.
  void sub_backlog(std::uint64_t bytes) noexcept;
  [[nodiscard]] std::uint64_t backlog() const noexcept;

  /// Time to first byte of one GET, in ms.
  void record_ttfb(double ms) noexcept;
  /// One finished body: bytes over `seconds` from first to last byte. Only bodies of at least
  /// `bc_min_body_bytes` update B_c (EWMA with `ewma_alpha`).
  void record_body(std::uint64_t bytes, double seconds) noexcept;
  /// One finished response; `is_throttled` for 503 SlowDown / 429.
  void record_response(bool is_throttled) noexcept;
  /// Body bytes delivered to destinations (goodput numerator).
  void record_delivered(std::uint64_t bytes) noexcept;

  /// @brief Controller tick, every `control_period` (250 ms) with the real elapsed time.
  /// Publishes TTFB quantiles (then ages the sketch), B_c, goodput / link_util, and the
  /// `conn_target_step` result; resets the period counters.
  /// @return the controller move (`hold` when `enable_controller` is false).
  controller_move control(double period_s) noexcept;

  /// @brief C1: fold one batch of reactor-local samples in (single writer, like `record_*`):
  /// TTFB counts merge into the sketch; the batch's n bodies of mean rate m move B_c as n
  /// sequential EWMA steps would, B_c = (1 - a)^n B_c + (1 - (1 - a)^n) m; counters add.
  void merge(estimator_samples const& s) noexcept;
  /// @brief C1: the periodic controller step, on the caller's monotonic clock (ns). The first
  /// call only starts the clock; afterwards `control(period)` runs once at least
  /// `control_period` (250 ms) elapsed since the previous step.
  /// @return the move, or nullopt when no step was due.
  std::optional<controller_move> control_at(std::uint64_t now_ns) noexcept;
  /// @return controller steps taken so far (`control` / `control_at`).
  [[nodiscard]] std::uint64_t steps() const noexcept { return _steps; }
  /// @brief C3: the endpoint's connection ceiling (max_connections_per_reactor x live reactors;
  /// 0 = none) for the discovered `max_conn_target`. Single writer, like `control`.
  void set_conn_limit(unsigned limit) noexcept { _conn_limit = limit; }
  /// @brief C4: a seed change (not a measurement) of an auto link (`link_gbps` 0 at
  /// construction; an explicit value is never replaced): the target follows @p gbps; before the
  /// first step the snapshot is re-seeded, afterwards conn_target is raised to the new seed
  /// (unless a throttle is recent) and the discovered scale re-expressed against the new seed.
  /// Single writer, like `control`.
  void set_link_gbps(double gbps) noexcept;
  /// @return C5: true while the admission cap binds: within throttle_caution_periods of a
  ///         throttle decrease or while the last period's throttle share exceeded the threshold.
  ///         Before the first 503/429 it is false (the reactors then admit on every warm
  ///         connection). Any thread.
  [[nodiscard]] bool throttle_binding() const noexcept
  {
    return _binding.load(std::memory_order_relaxed);
  }

  [[nodiscard]] controller_state const& state() const noexcept { return _ctl; }
  [[nodiscard]] remote_planner_config const& config() const noexcept { return _cfg; }
  /// @return the current (unpublished) B_c EWMA.
  [[nodiscard]] double bc_ewma() const noexcept { return _bc; }

 private:
  remote_planner_config _cfg;
  quantile_sketch _ttfb{};
  double _bc;
  std::uint64_t _delivered{0};
  std::uint64_t _responses{0};
  std::uint64_t _throttled{0};
  std::uint64_t _steps{0};
  std::uint64_t _last_control_ns{0};
  bool _clock_started{false};
  unsigned _conn_limit{0};
  controller_state _ctl{};
  std::atomic<double> _p50;
  std::atomic<double> _p99;
  std::atomic<double> _bc_pub;
  std::atomic<double> _util{0};
  std::atomic<bool> _util_measured{false};
  std::atomic<unsigned> _conn_target;
  std::atomic<double> _link_pub;
  std::atomic<bool> _binding{false};
  bool _link_explicit{false};
  std::uint64_t _admission_waits{0};
  std::atomic<std::int64_t> _backlog{0};
};

/**
 * @brief The per-endpoint planner state shared by every reactor of an ioctx (gpuio
 * `remote_endpoint`: planner config, estimator, retry budget; and `remote_client::publish`, the
 * publication mutex that makes the estimator's writers single).
 *
 * Producers read `estimator.snapshot_for_request` / `sub_backlog` and the reactors use `budget`
 * lock-free; `publish` and `apply_link_seed` serialise the estimator's single-writer calls.
 */
class endpoint_control {
 public:
  endpoint_control(remote_planner_config const& planner_cfg, retry_config const& retry_cfg) noexcept
    : planner(planner_cfg), estimator(planner_cfg), budget(retry_cfg)
  {
  }
  endpoint_control(endpoint_control const&)            = delete;
  endpoint_control& operator=(endpoint_control const&) = delete;
  endpoint_control(endpoint_control&&)                 = delete;
  endpoint_control& operator=(endpoint_control&&)      = delete;
  ~endpoint_control()                                  = default;

  /**
   * @brief C1: one reactor's tick hand-off. With the controller on, merges @p samples, sets the
   * connection ceiling (C3) and runs `control_at(now_ns)`; with it off the samples are ignored
   * (gpuio did not even merge them: the seeds stay pinned). The caller clears @p samples after.
   * @param conn_limit max_connections_per_reactor x live reactors.
   * @return the controller move when a step was due.
   */
  std::optional<controller_move> publish(estimator_samples const& samples,
                                         unsigned conn_limit,
                                         std::uint64_t now_ns) noexcept;

  /// @brief C4: apply the resolved link seed (an explicit `link_gbps` always wins). Any thread.
  void apply_link_seed(double gbps) noexcept;

  remote_planner_config const planner;  ///< the endpoint's config (immutable)
  endpoint_estimator estimator;         ///< producer side: snapshot_for_request / sub_backlog
  retry_budget budget;                  ///< X2, per endpoint

 private:
  std::mutex _mutex;  ///< single writer of `estimator`'s record / merge / control calls
};

}  // namespace cucascade::io::rest::planner
