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

// The controller's lever on the reactor (C5-C8). In gpuio these rules were member functions of
// the remote backend (src/s3/remote_backend.cpp: `conn_cap`, `admit_cap`, `over_cap`, the 503/429
// branch of the completion path, the admission-wait test of `cost`, `connect_ahead_all`); they are
// pure functions of a few counters, so the port keeps them here, next to the controller they
// serve, and the REST worker supplies the counters.

#include "io/rest/planner/planner_remote.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>

namespace cucascade::io::rest::planner {

unsigned conn_cap(unsigned conn_target,
                  unsigned max_conns_per_reactor,
                  unsigned live_reactors) noexcept
{
  // min(max_connections_per_reactor, ceil(conn_target / live reactors)): the planner sized the
  // GETs for conn_target connections (s_par = backlog / conn_target, gpuio ARCHITECTURE §7.1).
  unsigned const live   = std::max(1U, live_reactors);
  unsigned const target = std::max(1U, conn_target);
  unsigned const share  = target / live + (target % live != 0 ? 1U : 0U);  // no overflow
  return std::max(1U, std::min(std::max(1U, max_conns_per_reactor), share));
}

unsigned admit_cap(unsigned conn_cap,
                   bool enable_controller,
                   bool throttle_binding,
                   bool throttle_recent,
                   unsigned live_connections) noexcept
{
  if (!enable_controller) { return conn_cap; }
  if (throttle_binding || throttle_recent) { return conn_cap; }
  // C5: before a throttle every warm connection is used (an idle warm connection on a
  // never-throttled endpoint is pure loss).
  return std::max(conn_cap, live_connections);
}

void throttle_guard::on_throttle(std::uint32_t busy,
                                 std::uint64_t now_ns,
                                 std::chrono::milliseconds control_period) noexcept
{
  // C6 fast reaction: no more GETs in flight here than drew this 503/429 (the throttled
  // op's own slot is already released), until the controller has had a period to step.
  std::uint32_t const at = std::max<std::uint32_t>(1, busy);
  cap                    = now_ns < until_ns && cap != 0 ? std::min(cap, at) : at;
  auto const period_ms   = std::max<std::int64_t>(control_period.count(), 1);
  until_ns               = now_ns + static_cast<std::uint64_t>(period_ms) * 1'000'000ULL;
}

bool over_cap(remote_planner_config const& cfg,
              throttle_guard const& guard,
              std::uint64_t now_ns,
              std::uint32_t busy,
              unsigned admit_cap,
              bool latency) noexcept
{
  if (!cfg.enable_controller) { return false; }
  if (guard.holds(busy, now_ns)) { return true; }  // C6: hold the concurrency that drew a 503
  // P12: the latency class adds its reserve on top of the cap.
  unsigned const lim =
    latency ? admit_cap + latency_reserved_connections(cfg, admit_cap) : admit_cap;
  return busy >= lim;
}

bool counts_as_admission_wait(unsigned admit_cap, unsigned max_conns_per_reactor) noexcept
{
  // C7: an admission wait is the CONTROLLER's cap binding; at the hard connection limit
  // the GET waits for a connection and a probe-up cannot help.
  return admit_cap < std::max(1U, max_conns_per_reactor);
}

std::uint32_t connect_ahead_count(std::uint32_t pending,
                                  std::uint32_t busy,
                                  std::uint32_t idle,
                                  std::uint32_t connecting,
                                  unsigned conn_cap) noexcept
{
  // C8: open up to min(pending + busy, conn_cap) connections in parallel when demand arrives.
  std::uint64_t const demand = std::uint64_t{pending} + busy;
  std::uint64_t const want   = std::min<std::uint64_t>(demand, conn_cap);
  std::uint64_t const have   = std::uint64_t{busy} + idle + connecting;
  return want > have ? static_cast<std::uint32_t>(want - have) : 0;
}

}  // namespace cucascade::io::rest::planner
