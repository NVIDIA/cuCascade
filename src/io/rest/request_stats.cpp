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

#include <cucascade/io/rest/request_stats.hpp>

#include <algorithm>
#include <cstddef>

namespace cucascade::io::rest {

namespace {

std::size_t bucket_of(std::uint64_t ms) noexcept
{
  auto const it = std::ranges::upper_bound(request_latency_edges_ms, ms);
  return static_cast<std::size_t>(it - request_latency_edges_ms.begin());
}

}  // namespace

request_stats_snapshot& request_stats_snapshot::operator+=(request_stats_snapshot const& other)
{
  attempts += other.attempts;
  succeeded += other.succeeded;
  throttled += other.throttled;
  http_client_errors += other.http_client_errors;
  http_server_errors += other.http_server_errors;
  timed_out += other.timed_out;
  transport_errors += other.transport_errors;
  bytes += other.bytes;
  max_total_ms = std::max(max_total_ms, other.max_total_ms);
  for (std::size_t i = 0; i < total_ms_histogram.size(); ++i) {
    total_ms_histogram[i] += other.total_ms_histogram[i];
    first_byte_ms_histogram[i] += other.first_byte_ms_histogram[i];
  }
  return *this;
}

std::vector<std::uint32_t> request_stats_snapshot::latency_edges_ms()
{
  return {request_latency_edges_ms.begin(), request_latency_edges_ms.end()};
}

void request_stats::record(request_outcome outcome,
                           std::uint64_t total_ms,
                           std::uint64_t first_byte_ms,
                           std::uint64_t bytes) noexcept
{
  constexpr auto relaxed = std::memory_order_relaxed;
  _attempts.fetch_add(1, relaxed);
  _outcomes[static_cast<std::size_t>(outcome)].fetch_add(1, relaxed);
  _bytes.fetch_add(bytes, relaxed);
  _total_ms[bucket_of(total_ms)].fetch_add(1, relaxed);
  _first_byte_ms[bucket_of(first_byte_ms)].fetch_add(1, relaxed);
  auto seen = _max_total_ms.load(relaxed);
  while (seen < total_ms && !_max_total_ms.compare_exchange_weak(seen, total_ms, relaxed)) {}
}

request_stats_snapshot request_stats::snapshot() const
{
  constexpr auto relaxed = std::memory_order_relaxed;
  request_stats_snapshot out;
  out.attempts  = _attempts.load(relaxed);
  out.succeeded = _outcomes[static_cast<std::size_t>(request_outcome::succeeded)].load(relaxed);
  out.throttled = _outcomes[static_cast<std::size_t>(request_outcome::throttled)].load(relaxed);
  out.http_client_errors =
    _outcomes[static_cast<std::size_t>(request_outcome::http_client_error)].load(relaxed);
  out.http_server_errors =
    _outcomes[static_cast<std::size_t>(request_outcome::http_server_error)].load(relaxed);
  out.timed_out = _outcomes[static_cast<std::size_t>(request_outcome::timed_out)].load(relaxed);
  out.transport_errors =
    _outcomes[static_cast<std::size_t>(request_outcome::transport_error)].load(relaxed);
  out.bytes        = _bytes.load(relaxed);
  out.max_total_ms = _max_total_ms.load(relaxed);
  for (std::size_t i = 0; i < buckets; ++i) {
    out.total_ms_histogram[i]      = _total_ms[i].load(relaxed);
    out.first_byte_ms_histogram[i] = _first_byte_ms[i].load(relaxed);
  }
  return out;
}

}  // namespace cucascade::io::rest
