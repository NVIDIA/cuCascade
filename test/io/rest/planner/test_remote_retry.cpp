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

// Retry policy (ported from gpuio's tests/unit/planner/remote_retry_test.cpp; gpuio ARCHITECTURE
// §7.3; s3_line_rate §2.8): full-jitter distributions per class (10^5 samples), caps, Retry-After
// precedence through the real response parser (X1), the token budget (X2), classify-driven
// transitions (X3, free stale retries; X4, resume), deadlines (X7).

#include "io/rest/errc.hpp"
#include "io/rest/http/classify.hpp"
#include "io/rest/http/response_parser.hpp"
#include "io/rest/planner/planner_remote.hpp"
#include "planner_test_support.hpp"

#include <catch2/catch_all.hpp>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

using namespace cucascade::io::rest;
using namespace cucascade::io::rest::planner;
using rc = http::outcome::retry_class;
using std::chrono::milliseconds;

namespace {

http::response_meta parse(std::string_view raw)
{
  http::response_parser p;
  http::response_meta meta;
  std::size_t consumed = 0;
  std::error_code ec;
  REQUIRE(p.parse_header(std::span<char const>(raw.data(), raw.size()), consumed, meta, ec) ==
          http::parse_status::ok);
  return meta;
}

}  // namespace

TEST_CASE("planner.remote.backoff: ceilings per class and attempt",
          "[rest][planner][remote][retry]")
{
  retry_config const cfg{};
  // transient: min(20 s, 50 ms x 2^(attempt-1)); throttle: min(20 s, 1 s x 2^(attempt-1)).
  std::array<std::pair<unsigned, std::int64_t>, 8> const transient{
    {{0, 50}, {1, 50}, {2, 100}, {3, 200}, {5, 800}, {9, 12800}, {10, 20000}, {1000, 20000}}};
  for (auto [attempt, ms] : transient) {
    CHECK(backoff_ceiling(rc::transient, attempt, cfg).count() == ms);
  }
  std::array<std::pair<unsigned, std::int64_t>, 6> const throttle{
    {{1, 1000}, {2, 2000}, {3, 4000}, {5, 16000}, {6, 20000}, {64, 20000}}};
  for (auto [attempt, ms] : throttle) {
    CHECK(backoff_ceiling(rc::throttle, attempt, cfg).count() == ms);
  }
  for (rc c : {rc::none, rc::free_stale, rc::auth}) {
    CHECK(backoff_ceiling(c, 3, cfg).count() == 0);
    CHECK(backoff_delay({c, 3, 5000, cfg, ~0ull}).count() == 0);  // no Retry-After either
  }
}

TEST_CASE("planner.remote.backoff: full-jitter distributions (10^5 samples per class)",
          "[rest][planner][remote][retry]")
{
  retry_config const cfg{};
  planner_test::splitmix64 rng{42};
  constexpr int samples = 100000;
  for (rc cls : {rc::transient, rc::throttle}) {
    for (unsigned attempt : {1u, 2u, 4u, 8u, 12u}) {
      std::int64_t const ceiling = backoff_ceiling(cls, attempt, cfg).count();
      INFO("class " << static_cast<int>(cls) << " attempt " << attempt << " ceiling " << ceiling);
      double sum      = 0;
      std::int64_t lo = ceiling, hi = 0;
      std::array<int, 10> hist{};
      for (int i = 0; i < samples; ++i) {
        std::int64_t const d = backoff_delay({cls, attempt, 0, cfg, rng()}).count();
        REQUIRE(d >= 0);
        REQUIRE(d < ceiling);  // U(0,1) x ceiling, floored to whole ms
        sum += static_cast<double>(d);
        lo = std::min(lo, d);
        hi = std::max(hi, d);
        ++hist[static_cast<std::size_t>(d * 10 / ceiling)];
      }
      double const mean   = sum / samples;
      double const expect = static_cast<double>(ceiling - 1) / 2.0;  // uniform on {0..C-1}
      CHECK(std::abs(mean - expect) <= 0.05 * expect);
      CHECK(lo <= ceiling / 100);       // reaches near 0 ...
      CHECK(hi >= ceiling * 99 / 100);  // ... and near the ceiling
      for (int h : hist)
        CHECK(std::abs(h - samples / 10) <= samples / 10 / 20);  // flat, +-5%
    }
  }
}

TEST_CASE("planner.remote.backoff: Retry-After precedence and cap",
          "[rest][planner][remote][retry]")
{
  retry_config cfg{};
  // Both headers: the ms form wins, in either order (the codec resolves precedence; T5).
  auto const both = parse(
    "HTTP/1.1 503 Slow Down\r\nRetry-After: 7\r\nx-amz-retry-after: 1500\r\n"
    "Content-Length: 0\r\n\r\n");
  CHECK(both.retry_after_ms == 1500);
  auto const both_rev = parse(
    "HTTP/1.1 503 Slow Down\r\nx-amz-retry-after: 1500\r\nRetry-After: 7\r\n"
    "Content-Length: 0\r\n\r\n");
  CHECK(both_rev.retry_after_ms == 1500);
  auto const secs = parse("HTTP/1.1 503 Slow Down\r\nRetry-After: 7\r\nContent-Length: 0\r\n\r\n");
  CHECK(secs.retry_after_ms == 7000);
  auto const huge =
    parse("HTTP/1.1 503 Slow Down\r\nRetry-After: 120\r\nContent-Length: 0\r\n\r\n");
  CHECK(huge.retry_after_ms == 30000);

  planner_test::splitmix64 rng{7};
  for (int i = 0; i < 10000; ++i) {
    std::uint64_t const r = rng();
    // Throttle attempt 1: jitter < 1 s, so the server's value decides.
    CHECK(backoff_delay({rc::throttle, 1, both.retry_after_ms, cfg, r}).count() == 1500);
    CHECK(backoff_delay({rc::throttle, 1, secs.retry_after_ms, cfg, r}).count() == 7000);
    // A raw value above 30 s is capped (max_retry_after) even if the codec did not cap it.
    CHECK(backoff_delay({rc::throttle, 1, 45000, cfg, r}).count() == 30000);
    // Never earlier than Retry-After; a longer jittered delay still applies.
    auto const t = backoff_delay({rc::transient, 9, 100, cfg, r}).count();
    CHECK(t >= 100);
    CHECK(t < 12800);
  }
  cfg.honor_retry_after = false;
  for (int i = 0; i < 1000; ++i) {
    CHECK(backoff_delay({rc::throttle, 1, 7000, cfg, rng()}).count() < 1000);
  }
}

TEST_CASE("planner.remote.budget: costs, exhaustion and replenishment",
          "[rest][planner][remote][retry]")
{
  retry_config const cfg{};
  CHECK(retry_cost(rc::transient, cfg) == 14);
  CHECK(retry_cost(rc::throttle, cfg) == 5);
  CHECK(retry_cost(rc::free_stale, cfg) == 0);
  CHECK(retry_cost(rc::auth, cfg) == 0);
  CHECK(retry_cost(rc::none, cfg) == 0);

  retry_budget b(cfg);
  CHECK(b.tokens() == 500);
  b.on_success(true);  // capped at the initial 500
  CHECK(b.tokens() == 500);
  for (int i = 0; i < 35; ++i)
    REQUIRE(b.try_acquire(rc::transient, cfg));  // 35 x 14 = 490
  CHECK(b.tokens() == 10);
  CHECK_FALSE(b.try_acquire(rc::transient, cfg));  // 10 < 14: exhausted for transient ...
  CHECK(b.tokens() == 10);
  CHECK(b.try_acquire(rc::throttle, cfg));  // ... a throttle retry still fits twice
  CHECK(b.try_acquire(rc::throttle, cfg));
  CHECK_FALSE(b.try_acquire(rc::throttle, cfg));
  CHECK(b.tokens() == 0);
  CHECK(b.try_acquire(rc::free_stale, cfg));  // free retries never need tokens
  CHECK(b.try_spend(0));
  b.on_success(false);  // a retried success returns nothing
  CHECK(b.tokens() == 0);
  for (int i = 0; i < 13; ++i)
    b.on_success(true);
  CHECK_FALSE(b.try_acquire(rc::transient, cfg));
  b.on_success(true);  // 14th first-try success
  CHECK(b.try_acquire(rc::transient, cfg));
  CHECK(b.tokens() == 0);
}

TEST_CASE("planner.remote.budget: concurrent spend and replenish stay in bounds",
          "[rest][planner][remote][retry][threads]")
{
  // Lock-free bucket shared by the reactors of an endpoint; run under TSAN as well. Catch2
  // assertions are not thread-safe, so the threads only count violations.
  retry_config const cfg{};
  retry_budget b(cfg);
  std::atomic<long> spends{0}, over_capacity{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&, t] {
      planner_test::splitmix64 rng{static_cast<std::uint64_t>(t) + 1};
      for (int i = 0; i < 20000; ++i) {
        if (rng() % 3 == 0) {
          if (b.try_spend(14)) spends.fetch_add(1, std::memory_order_relaxed);
        } else {
          b.on_success(true);
        }
        if (b.tokens() > b.capacity()) over_capacity.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& th : threads)
    th.join();
  CHECK(over_capacity.load() == 0);
  CHECK(b.tokens() <= 500);
  CHECK(spends.load() > 0);
}

TEST_CASE("planner.remote.retry: classify-driven transitions", "[rest][planner][remote][retry]")
{
  retry_config const cfg{};
  file_range const chunk{64 * MiB, 8 * MiB};
  planner_test::splitmix64 rng{99};

  SECTION("non-retriable outcome fails")
  {
    retry_budget b(cfg);
    retry_state st;
    http::response_meta m;
    m.status     = 404;
    auto const o = http::classify(m, {}, http::method::get, 0, 10, 100, 0, true);
    auto const d = decide_retry(o, chunk, 0, 0, st, cfg, b, rng());
    CHECK(d.action == retry_action::fail);
    CHECK(b.tokens() == 500);
  }
  SECTION("stale pooled connection: 5 free immediate retries, then transient")
  {
    retry_budget b(cfg);
    retry_state st;
    http::response_meta const none;
    auto const o = http::classify(none,
                                  std::error_code(ECONNRESET, std::system_category()),
                                  http::method::get,
                                  0,
                                  10,
                                  100,
                                  0,
                                  true);
    REQUIRE(o.cls == rc::free_stale);
    for (int i = 0; i < 5; ++i) {
      auto const d = decide_retry(o, chunk, 0, 0, st, cfg, b, rng());
      CHECK(d.action == retry_action::retry_now);
      CHECK(d.delay.count() == 0);
      CHECK(d.new_connection);
      CHECK(d.range == chunk);
    }
    CHECK(st.attempts == 1);   // no attempt charged
    CHECK(b.tokens() == 500);  // no budget charged
    auto const d = decide_retry(o, chunk, 0, 0, st, cfg, b, rng());
    CHECK(d.action == retry_action::retry_later);  // the 6th is an ordinary transient failure
    CHECK(d.different_ip);
    CHECK(b.tokens() == 486);
    CHECK(st.attempts == 2);
  }
  SECTION("transient: max_attempts, growing ceilings, new connection on another IP")
  {
    retry_budget b(cfg);
    retry_state st;
    http::response_meta m;
    m.status     = 500;
    auto const o = http::classify(m, {}, http::method::get, 0, 10, 100, 0, true);
    REQUIRE(o.cls == rc::transient);
    int retries = 0;
    for (;;) {
      unsigned const attempt = st.attempts;
      auto const d           = decide_retry(o, chunk, 0, 0, st, cfg, b, rng());
      if (d.action == retry_action::fail) break;
      ++retries;
      CHECK(d.action == retry_action::retry_later);
      CHECK(d.delay < backoff_ceiling(rc::transient, attempt, cfg));
      CHECK(d.new_connection);
      CHECK(d.different_ip);
      CHECK(d.range == chunk);
    }
    CHECK(retries == 9);  // max_attempts = 10 attempts in total
    CHECK(st.attempts == 10);
    CHECK(b.tokens() == 500 - 9 * 14);
  }
  SECTION("throttle: Retry-After honoured, cost 5, connection may be reused")
  {
    retry_budget b(cfg);
    retry_state st;
    auto m =
      parse("HTTP/1.1 503 Slow Down\r\nx-amz-retry-after: 2500\r\nContent-Length: 0\r\n\r\n");
    auto const o = http::classify(m, {}, http::method::get, 0, 10, 100, 0, true);
    REQUIRE(o.cls == rc::throttle);
    auto const d = decide_retry(o, chunk, 0, m.retry_after_ms, st, cfg, b, rng());
    CHECK(d.action == retry_action::retry_later);
    CHECK(d.delay.count() == 2500);
    CHECK(d.new_connection == !o.reuse_connection);
    CHECK_FALSE(d.different_ip);
    CHECK(b.tokens() == 495);
  }
  SECTION("mid-body reset after r bytes: resume [off + r, end), same chunk credit")
  {
    retry_budget b(cfg);
    retry_state st;
    http::response_meta m;
    m.status         = 206;
    m.content_length = chunk.size;
    m.range          = http::content_range{chunk.offset, chunk.end() - 1, 1 * GiB, false};
    auto const o     = http::classify(m,
                                  std::error_code(ECONNRESET, std::system_category()),
                                  http::method::get,
                                  chunk.offset,
                                  chunk.end() - 1,
                                  1 * GiB,
                                  3 * MiB,
                                  true);
    REQUIRE(o.cls == rc::transient);
    REQUIRE(o.resume_allowed);
    auto const d = decide_retry(o, chunk, 3 * MiB, 0, st, cfg, b, rng());
    CHECK(d.action == retry_action::retry_later);
    CHECK(d.range == file_range{chunk.offset + 3 * MiB, 5 * MiB});
    CHECK(resume_range(chunk, 0) == chunk);
    CHECK(resume_range(chunk, chunk.size) == file_range{chunk.end(), 0});
  }
  SECTION("auth: three re-signs, then fail")
  {
    retry_budget b(cfg);
    retry_state st;
    http::response_meta m;
    m.status     = 403;
    auto const o = http::classify(m, {}, http::method::get, 0, 10, 100, 0, true);
    REQUIRE(o.cls == rc::auth);
    for (int i = 0; i < 3; ++i) {
      auto const d = decide_retry(o, chunk, 0, 0, st, cfg, b, rng());
      CHECK(d.action == retry_action::retry_now);
      CHECK(d.resign);
    }
    CHECK(decide_retry(o, chunk, 0, 0, st, cfg, b, rng()).action == retry_action::fail);
    CHECK(b.tokens() == 500);
  }
  SECTION("budget exhausted: fail fast")
  {
    retry_budget b(cfg);
    while (b.try_spend(14)) {}
    retry_state st;
    http::response_meta m;
    m.status     = 502;
    auto const o = http::classify(m, {}, http::method::get, 0, 10, 100, 0, true);
    auto const d = decide_retry(o, chunk, 0, 0, st, cfg, b, rng());
    CHECK(d.action == retry_action::fail);
    CHECK(d.budget_exhausted);
    CHECK(st.attempts == 1);
  }
}

TEST_CASE("planner.remote.deadlines: stall and attempt", "[rest][planner][remote][retry]")
{
  remote_planner_config const cfg{};
  // gpuio's hedge_ttfb_timeout rows are not ported (hedging, X8).
  // 3 x 4 MiB at 75 MB/s = 168 ms < 5 s floor; at 1 MB/s = 12.6 s.
  CHECK(stall_deadline(cfg, 4 * MiB, 75e6).count() == 5000);
  CHECK(stall_deadline(cfg, 4 * MiB, 1e6).count() == 12583);
  CHECK(stall_deadline(cfg, 4 * MiB, 0).count() == 5000);
  // 10 s + size / 1 MB/s: 16 MiB -> 26.78 s (uring_ultra: fixed 30 s for any size).
  CHECK(attempt_deadline(cfg, 16 * MiB).count() == 10000 + 16778);
  CHECK(attempt_deadline(cfg, 0).count() == 10000);
}
