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

// Fairness across remote requests (ported from gpuio's src/planner/remote_fairness.cpp and the
// `drr_weight` of src/s3/remote_backend.cpp; gpuio ARCHITECTURE §2.4, §7.1; s3_line_rate §3 item
// 6, §3.3 `next_segment`): latency class with reserved connections, weighted deficit round-robin
// across bulk requests, FIFO as a policy (P12).

#include "io/rest/planner/planner_remote.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace cucascade::io::rest::planner {

remote_class request_class(priority prio,
                           bool explicit_priority,
                           std::uint64_t total_bytes,
                           remote_planner_config const& cfg) noexcept
{
  if (prio == priority::urgent) { return remote_class::latency; }
  if (!explicit_priority && total_bytes <= cfg.small_request_bytes) {
    return remote_class::latency;
  }
  return remote_class::bulk;
}

unsigned latency_reserved_connections(remote_planner_config const& cfg,
                                      unsigned conns_per_reactor) noexcept
{
  if (conns_per_reactor <= 1) { return 0; }
  double const frac = std::isfinite(cfg.latency_reserved_fraction)
                        ? std::clamp(cfg.latency_reserved_fraction, 0.0, 1.0)
                        : 0.0;
  auto const by_fraction =
    static_cast<unsigned>(std::ceil(frac * static_cast<double>(conns_per_reactor)));
  return std::min(std::max(2U, by_fraction), conns_per_reactor - 1);
}

std::uint32_t drr_weight(priority prio, remote_scheduler_config const& cfg) noexcept
{
  std::size_t const i = prio == priority::high ? 0 : prio == priority::normal ? 1 : 2;
  return std::max<std::uint32_t>(1, cfg.remote_weights[i]);  // urgent rides the latency list
}

void drr_scheduler::list::push_back(drr_hook& h) noexcept
{
  h.prev = tail;
  h.next = nullptr;
  if (tail != nullptr) {
    tail->next = &h;
  } else {
    head = &h;
  }
  tail = &h;
  ++size;
}

void drr_scheduler::list::erase(drr_hook& h) noexcept
{
  if (h.prev != nullptr) {
    h.prev->next = h.next;
  } else {
    head = h.next;
  }
  if (h.next != nullptr) {
    h.next->prev = h.prev;
  } else {
    tail = h.prev;
  }
  h.prev = h.next = nullptr;
  --size;
}

void drr_scheduler::push(drr_hook& h, remote_class cls) noexcept
{
  if (h.linked) { return; }
  h.latency = cls == remote_class::latency;
  h.deficit = 0;
  h.linked  = true;
  (h.latency ? _latency : _bulk).push_back(h);
}

void drr_scheduler::remove(drr_hook& h) noexcept
{
  if (!h.linked) { return; }
  if (h.latency) {
    _latency.erase(h);
  } else if (_current == &h) {
    // Hand the cursor (and a fresh quantum) to the successor before unlinking.
    drr_hook* const succ = h.next != nullptr ? h.next : _bulk.head;
    _bulk.erase(h);
    _current = succ == &h ? nullptr : succ;
    _fresh   = _current != nullptr;  // the successor's quantum is granted at its evaluation
  } else {
    _bulk.erase(h);
  }
  h.deficit = 0;
  h.linked  = false;
}

}  // namespace cucascade::io::rest::planner
