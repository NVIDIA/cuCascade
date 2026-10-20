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

// Retry policy and deadlines (ported from gpuio's src/planner/remote_retry.cpp; gpuio
// ARCHITECTURE §7.3 table; s3_line_rate §2.8, AWS SDK 2026 retry behaviour: two classes with full
// jitter (X1), a token budget (X2), free retries of dead pooled connections (X3), resume with
// If-Match (X4), stall and attempt deadlines (X7)). gpuio's `hedge_ttfb_timeout` lived here; it
// belonged to hedging (X8, off by default) and is not ported.

#include "io/rest/planner/planner_remote.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>

namespace cucascade::io::rest::planner {
namespace {

using rc = http::outcome::retry_class;
using std::chrono::milliseconds;

/// Whole milliseconds, rounded up, saturating; NaN/negative -> 0.
milliseconds ms_ceil(double ms) noexcept
{
  if (!(ms > 0.0)) { return milliseconds{0}; }
  constexpr std::int64_t half_range = std::numeric_limits<std::int64_t>::max() / 2;
  constexpr auto max                = static_cast<double>(half_range);
  return milliseconds{static_cast<std::int64_t>(std::ceil(std::min(ms, max)))};
}

}  // namespace

milliseconds backoff_ceiling(rc cls, unsigned attempt, retry_config const& cfg) noexcept
{
  std::int64_t base = 0;
  switch (cls) {
    case rc::transient: base = cfg.transient_base.count(); break;  // 50 ms
    case rc::throttle: base = cfg.throttle_base.count(); break;    // 1 s
    default: return milliseconds{0};
  }
  std::int64_t const cap = std::max<std::int64_t>(0, cfg.backoff_cap.count());  // 20 s
  if (base <= 0) { return milliseconds{0}; }
  unsigned const k = attempt > 0 ? attempt - 1 : 0;
  // base x 2^k > cap  <=>  base > floor(cap / 2^k); checked before shifting, so no overflow.
  if (k >= 62 || base > (cap >> k)) { return milliseconds{cap}; }
  return milliseconds{base << k};
}

milliseconds backoff_delay(backoff_inputs const& in) noexcept
{
  std::int64_t const ceiling = backoff_ceiling(in.cls, in.attempt, in.cfg).count();
  // Full jitter: U(0,1) x ceiling, U from the top 53 bits (exactly representable in a double).
  double const u = static_cast<double>(in.rng >> 11) * 0x1.0p-53;
  auto delay     = static_cast<std::int64_t>(u * static_cast<double>(ceiling));
  if (ceiling > 0 && delay >= ceiling) { delay = ceiling - 1; }  // floating-point rounding guard
  bool const backs_off = in.cls == rc::transient || in.cls == rc::throttle;
  if (backs_off && in.cfg.honor_retry_after && in.retry_after_ms != 0) {
    std::int64_t const after = std::min<std::int64_t>(in.retry_after_ms, max_retry_after.count());
    delay                    = std::max(delay, after);
  }
  return milliseconds{delay};
}

bool retry_budget::try_spend(unsigned cost) noexcept
{
  if (cost == 0) { return true; }
  unsigned cur = _tokens.load(std::memory_order_relaxed);
  do {
    if (cur < cost) { return false; }
  } while (!_tokens.compare_exchange_weak(
    cur, cur - cost, std::memory_order_relaxed, std::memory_order_relaxed));
  return true;
}

void retry_budget::on_success(bool first_try) noexcept
{
  if (!first_try) { return; }
  unsigned cur = _tokens.load(std::memory_order_relaxed);
  do {
    if (cur >= _max) { return; }
  } while (!_tokens.compare_exchange_weak(
    cur, cur + 1, std::memory_order_relaxed, std::memory_order_relaxed));
}

retry_decision decide_retry(http::outcome const& o,
                            file_range chunk,
                            std::uint64_t received,
                            std::uint32_t retry_after_ms,
                            retry_state& st,
                            retry_config const& cfg,
                            retry_budget& budget,
                            std::uint64_t rng) noexcept
{
  retry_decision d;
  d.range = chunk;
  rc cls  = o.retryable ? o.cls : rc::none;
  if (cls == rc::none) { return d; }

  // §7.3 row 1: a pooled connection died before any response byte. Immediate, free, fresh
  // connection (curl transfer.c:600-660), at most stale_connection_free_retries times (X3).
  if (cls == rc::free_stale) {
    if (st.stale_retries < cfg.stale_connection_free_retries) {
      ++st.stale_retries;
      d.action         = retry_action::retry_now;
      d.new_connection = true;
      return d;
    }
    cls = rc::transient;
  }

  // §7.3 row "403": bounded re-sign (fresh credentials, or the clock offset from Date).
  if (cls == rc::auth) {
    if (st.auth_retries < cfg.max_auth_attempts) {
      ++st.auth_retries;
      d.action         = retry_action::retry_now;
      d.resign         = true;
      d.new_connection = !o.reuse_connection;
      return d;
    }
    return d;
  }

  // Transient / throttle: attempt cap, then the budget (fail fast during an outage, X2).
  if (st.attempts >= cfg.max_attempts) { return d; }
  if (!budget.try_spend(retry_cost(cls, cfg))) {
    d.budget_exhausted = true;
    return d;
  }
  d.delay = backoff_delay(backoff_inputs{cls, st.attempts, retry_after_ms, cfg, rng});
  ++st.attempts;
  d.action         = retry_action::retry_later;
  d.new_connection = cls == rc::transient || !o.reuse_connection;  // "new connection, other IP"
  d.different_ip   = cls == rc::transient;
  // §7.3 row "mid-body failure after r bytes": resume the remainder, keep the received bytes.
  if (o.resume_allowed && received > 0) { d.range = resume_range(chunk, received); }
  return d;
}

milliseconds stall_deadline(remote_planner_config const& cfg,
                            std::uint64_t body_recv_cap,
                            double bc_bytes_s) noexcept
{
  milliseconds const floor = cfg.stall_timeout;
  if (!(bc_bytes_s > 0.0) || !std::isfinite(bc_bytes_s)) { return floor; }
  return std::max(floor, ms_ceil(3.0 * static_cast<double>(body_recv_cap) / bc_bytes_s * 1e3));
}

milliseconds attempt_deadline(remote_planner_config const& cfg, std::uint64_t bytes) noexcept
{
  constexpr milliseconds base{10000};  // 10 s: s3_line_rate §2.8 "Last resort"
  if (!(cfg.min_rate_bytes_s > 0.0)) { return base; }
  return base + ms_ceil(static_cast<double>(bytes) / cfg.min_rate_bytes_s * 1e3);
}

}  // namespace cucascade::io::rest::planner
