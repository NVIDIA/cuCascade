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

#include <catch2/catch_all.hpp>

#include <cstdint>
#include <numeric>
#include <thread>
#include <vector>

namespace {

using cucascade::io::rest::request_latency_edges_ms;
using cucascade::io::rest::request_outcome;
using cucascade::io::rest::request_stats;
using cucascade::io::rest::request_stats_snapshot;

std::uint64_t sum(std::vector<std::uint64_t> const& v)
{
  return std::accumulate(v.begin(), v.end(), std::uint64_t{0});
}

}  // namespace

TEST_CASE("a new request_stats has seen nothing", "[io][rest][request-stats]")
{
  request_stats stats;
  auto const s = stats.snapshot();
  CHECK(s.attempts == 0);
  CHECK(s.bytes == 0);
  CHECK(s.max_total_ms == 0);
  CHECK(s.total_ms_histogram.size() == request_latency_edges_ms.size() + 1);
  CHECK(sum(s.total_ms_histogram) == 0);
}

TEST_CASE("the latency edges can be read back", "[io][rest][request-stats]")
{
  auto const edges = request_stats_snapshot::latency_edges_ms();
  REQUIRE(edges.size() == request_latency_edges_ms.size());
  CHECK(edges.front() == 10);
  CHECK(edges.back() == 5000);
}

TEST_CASE("each outcome is counted separately", "[io][rest][request-stats]")
{
  request_stats stats;
  stats.record(request_outcome::succeeded, 5, 1, 100);
  stats.record(request_outcome::succeeded, 5, 1, 100);
  stats.record(request_outcome::throttled, 5, 1, 0);
  stats.record(request_outcome::http_client_error, 5, 1, 0);
  stats.record(request_outcome::http_server_error, 5, 1, 0);
  stats.record(request_outcome::timed_out, 5, 1, 0);
  stats.record(request_outcome::transport_error, 5, 1, 0);
  auto const s = stats.snapshot();
  CHECK(s.attempts == 7);
  CHECK(s.succeeded == 2);
  CHECK(s.throttled == 1);
  CHECK(s.http_client_errors == 1);
  CHECK(s.http_server_errors == 1);
  CHECK(s.timed_out == 1);
  CHECK(s.transport_errors == 1);
  CHECK(s.bytes == 200);
}

TEST_CASE("latencies fall in the bucket under their upper edge", "[io][rest][request-stats]")
{
  request_stats stats;
  stats.record(request_outcome::succeeded, 0, 0, 0);    // under 10 ms: bucket 0
  stats.record(request_outcome::succeeded, 9, 9, 0);    // bucket 0
  stats.record(request_outcome::succeeded, 10, 10, 0);  // not under 10: bucket 1
  stats.record(
    request_outcome::succeeded, 4999, 49, 0);  // last edge is 5000: bucket 8; ttfb bucket 2
  stats.record(request_outcome::succeeded, 5000, 5000, 0);  // beyond the last edge: bucket 9
  stats.record(request_outcome::succeeded, 90000, 1, 0);
  auto const s = stats.snapshot();
  CHECK(s.total_ms_histogram[0] == 2);
  CHECK(s.total_ms_histogram[1] == 1);
  CHECK(s.total_ms_histogram[8] == 1);
  CHECK(s.total_ms_histogram[9] == 2);
  CHECK(sum(s.total_ms_histogram) == 6);
  CHECK(s.first_byte_ms_histogram[2] == 1);
  CHECK(s.first_byte_ms_histogram[9] == 1);
  CHECK(s.max_total_ms == 90000);
}

TEST_CASE("snapshots add, and the slowest request is the maximum", "[io][rest][request-stats]")
{
  request_stats a, b;
  a.record(request_outcome::succeeded, 30, 10, 10);
  b.record(request_outcome::throttled, 700, 600, 5);
  auto total = a.snapshot();
  total += b.snapshot();
  CHECK(total.attempts == 2);
  CHECK(total.succeeded == 1);
  CHECK(total.throttled == 1);
  CHECK(total.bytes == 15);
  CHECK(total.max_total_ms == 700);
  CHECK(sum(total.total_ms_histogram) == 2);
}

TEST_CASE("concurrent recording loses nothing", "[io][rest][request-stats]")
{
  request_stats stats;
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&stats] {
      for (int i = 0; i < 10000; ++i) {
        stats.record(request_outcome::succeeded, static_cast<std::uint64_t>(i % 300), 1, 1);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  auto const s = stats.snapshot();
  CHECK(s.attempts == 80000);
  CHECK(s.succeeded == 80000);
  CHECK(s.bytes == 80000);
  CHECK(sum(s.total_ms_histogram) == 80000);
}
