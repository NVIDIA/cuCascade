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

#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

namespace cucascade::io::rest {

/// How one attempt of a ranged GET ended.
enum class request_outcome : std::uint8_t {
  succeeded,          ///< 200 or 206.
  throttled,          ///< 429 or 503: the store asked the client to slow down.
  http_client_error,  ///< Any other status below 500 that is not a success.
  http_server_error,  ///< 500 or above, except 503.
  timed_out,          ///< The transfer timed out, or stalled below the stall detector's rate.
  transport_error,    ///< Any other transfer error: reset connection, TLS, DNS.
};

/// Upper edges, in milliseconds, of the latency histogram buckets. A request falls in the first
/// bucket whose edge it is under; the last bucket holds everything slower than the last edge.
inline constexpr std::array<std::uint32_t, 9> request_latency_edges_ms{
  10, 20, 50, 100, 200, 500, 1000, 2000, 5000};

/// A copy of the counters of a @ref request_stats, which can be added to another.
struct request_stats_snapshot {
  std::uint64_t attempts{0};  ///< Every GET attempt, including retries.
  std::uint64_t succeeded{0};
  std::uint64_t throttled{0};
  std::uint64_t http_client_errors{0};
  std::uint64_t http_server_errors{0};
  std::uint64_t timed_out{0};
  std::uint64_t transport_errors{0};
  std::uint64_t bytes{0};         ///< Bytes received.
  std::uint64_t max_total_ms{0};  ///< Slowest attempt.
  /// Attempts per latency bucket: the whole attempt, then the time to the first byte. Each has one
  /// more entry than @ref request_latency_edges_ms.
  std::vector<std::uint64_t> total_ms_histogram =
    std::vector<std::uint64_t>(request_latency_edges_ms.size() + 1, 0);
  std::vector<std::uint64_t> first_byte_ms_histogram =
    std::vector<std::uint64_t>(request_latency_edges_ms.size() + 1, 0);

  request_stats_snapshot& operator+=(request_stats_snapshot const& other);

  /// @ref request_latency_edges_ms, for callers that cannot see the constant.
  [[nodiscard]] static std::vector<std::uint32_t> latency_edges_ms();
};

/**
 * @brief Counters for the ranged GETs a REST reactor makes: how many attempts, how they ended, how
 * many bytes, and how long they took.
 *
 * The reactor records every attempt, so a slow run can be told from a fast one by what the object
 * store did, not just by the total time. Counters only grow; take two snapshots and subtract to
 * see an interval.
 *
 * Thread-safe. Recording is a handful of relaxed atomic increments.
 */
class request_stats {
 public:
  /// Record one finished attempt.
  void record(request_outcome outcome,
              std::uint64_t total_ms,
              std::uint64_t first_byte_ms,
              std::uint64_t bytes) noexcept;

  [[nodiscard]] request_stats_snapshot snapshot() const;

 private:
  static constexpr std::size_t buckets = request_latency_edges_ms.size() + 1;

  std::atomic<std::uint64_t> _attempts{0};
  std::atomic<std::uint64_t> _outcomes[6]{};
  std::atomic<std::uint64_t> _bytes{0};
  std::atomic<std::uint64_t> _max_total_ms{0};
  std::atomic<std::uint64_t> _total_ms[buckets]{};
  std::atomic<std::uint64_t> _first_byte_ms[buckets]{};
};

}  // namespace cucascade::io::rest
