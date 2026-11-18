/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include "engine_test_utils.hpp"

using namespace cucascade::io;
using namespace cucascade::io::cufile;
using namespace cucascade::io::cufile::engine_test;

TEST_CASE("engine large slice fills and slides the effective storage window",
          "[cufile][engine][sched]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(64 * mib, configuration(8));
  transport_simulator::fault fault;
  fault.completion_latency = 5ms;
  f.transport->add_fault(fault);
  f.start();
  auto group  = std::make_shared<grouped_coordinator>(f.bytes, 1);
  auto future = group->get_future();
  f.engine->hand_off(f.slice(group, {0, f.bytes}));
  REQUIRE(std::move(future).get(10s) == f.bytes);
  CHECK(f.contents() == f.image);
  REQUIRE(f.engine->drain(5s));
  auto const stats = f.engine->gauges();
  // Idle mounts surrender their grants after the last operation retires.
  CHECK(stats.max_storage_qd == 8);
  INFO("backlog time-weighted QD = " << stats.backlog_mean_storage_qd);
  INFO("reaper active ns/op = " << stats.reaper_active_ns / stats.completed_ops);
  CHECK(stats.reaper_iterations > 0);
  CHECK(stats.reaper_active_ns > 0);
  // A host-scheduler-dependent QD mean belongs in Phase 7 performance gates.
  CHECK(stats.backlog_mean_storage_qd > 0);
  CHECK(f.transport->snapshot().max_inflight == 8);
  CHECK(f.transport->snapshot().read_calls == 64);
  CHECK(stats.async_registered_bytes == f.bytes);
  CHECK(stats.submit.count == 64);
  CHECK(stats.completion.count == 64);
  CHECK(stats.residence.count == 64);
}

TEST_CASE("engine drained submission tail is not storage eligible backlog",
          "[cufile][engine][sched]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(mib, configuration(1));
  transport_simulator::fault fault;
  fault.stuck_completion = true;
  f.transport->add_fault(fault);
  f.start();
  auto group  = std::make_shared<grouped_coordinator>(mib, 1);
  auto future = group->get_future();
  f.engine->hand_off(f.slice(group, {0, mib}));
  REQUIRE(until([&] { return f.transport->snapshot().blocked_completions == 1; }));
  std::this_thread::sleep_for(10ms);
  auto before = f.engine->gauges().backlog_ns;
  std::this_thread::sleep_for(20ms);
  auto after = f.engine->gauges().backlog_ns;
  CHECK(after == before);
  f.transport->unblock_all();
  CHECK(std::move(future).get(5s) == mib);
}

TEST_CASE("engine priority is high first with bounded low service", "[cufile][engine][sched]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(12 * mib, configuration(1));
  auto control = std::make_shared<held_executor>();
  f.start([&](auto& opts) { opts.control = control; });
  auto group   = std::make_shared<grouped_coordinator>(f.bytes, 2);
  auto future  = group->get_future();
  auto low     = f.slice(group, {10 * mib, 2 * mib}, 10 * mib);
  auto high    = f.slice(group, {0, 10 * mib});
  low.priority = io_priority::low;
  f.engine->hand_off(std::move(low));
  f.engine->hand_off(std::move(high));
  REQUIRE(until([&] { return control->size() > 0; }));
  REQUIRE(until([&] {
    control->run();
    return future.is_ready();
  }));
  CHECK(std::move(future).get() == f.bytes);
  CHECK(f.contents() == f.image);
  auto const calls = f.transport->submissions();
  REQUIRE(calls.size() == 12);
  CHECK(calls[0].file_offset == 0);
  CHECK(calls[4].file_offset == static_cast<off_t>(10 * mib));
  CHECK(calls[9].file_offset == static_cast<off_t>(11 * mib));
}

TEST_CASE("engine requests round robin within one mount", "[cufile][engine][sched]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(8 * mib, configuration(1));
  auto control = std::make_shared<held_executor>();
  f.start([&](auto& opts) { opts.control = control; });
  auto group  = std::make_shared<grouped_coordinator>(f.bytes, 2);
  auto future = group->get_future();
  auto first  = f.slice(group, {0, 4 * mib});
  auto second = f.slice(group, {4 * mib, 4 * mib}, 4 * mib);
  f.engine->hand_off(std::move(first));
  f.engine->hand_off(std::move(second));
  REQUIRE(until([&] {
    control->run();
    return future.is_ready();
  }));
  CHECK(std::move(future).get() == f.bytes);
  CHECK(f.contents() == f.image);
  auto const calls = f.transport->submissions();
  REQUIRE(calls.size() == 8);
  CHECK(calls[0].file_offset == 0);
  CHECK(calls[1].file_offset == static_cast<off_t>(4 * mib));
  CHECK(calls[2].file_offset == static_cast<off_t>(mib));
}

TEST_CASE("engine byte deficit is shared by all slices of a logical request",
          "[cufile][engine][sched]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(3 * mib, configuration(1));
  auto control = std::make_shared<held_executor>();
  f.start([&](auto& opts) { opts.control = control; });
  auto first  = std::make_shared<grouped_coordinator>(mib, 4);
  auto second = std::make_shared<grouped_coordinator>(2 * mib, 1);
  auto a      = first->get_future();
  auto b      = second->get_future();
  std::vector<engine_slice> pieces;
  for (std::size_t i = 0; i < 4; ++i)
    pieces.push_back(f.slice(first, {i * mib / 4, mib / 4}, i * mib / 4));
  pieces.push_back(f.slice(second, {mib, 2 * mib}, mib));
  for (auto& s : pieces)
    f.engine->hand_off(std::move(s));
  REQUIRE(until([&] {
    control->run();
    return a.is_ready() && b.is_ready();
  }));
  CHECK(std::move(a).get() == mib);
  CHECK(std::move(b).get() == 2 * mib);
  CHECK(std::ranges::equal(f.contents(), f.image));
  auto const calls = f.transport->submissions();
  REQUIRE(calls.size() == 6);
  // Four small pieces cost ONE quantum, not four. A slice-count round robin fails this.
  for (std::size_t i = 0; i < 4; ++i)
    CHECK(calls[i].file_offset < static_cast<off_t>(mib));
  CHECK(calls[4].file_offset == static_cast<off_t>(mib));
}

TEST_CASE("engine mounts share capacity and obey lane caps", "[cufile][engine][sched]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto config                  = configuration(4);
  config.device_lane_cap       = 10;
  config.arena_bytes           = 32 * mib;
  config.submitters_per_device = GENERATE(1U, 2U);
  fixture f(32 * mib, config);
  transport_simulator::fault fault;
  fault.completion_latency = 2ms;
  f.transport->add_fault(fault);
  auto control = std::make_shared<held_executor>();
  f.start([&](auto& opts) { opts.control = control; });
  auto group         = std::make_shared<grouped_coordinator>(f.bytes, 2);
  auto future        = group->get_future();
  auto a             = f.slice(group, {0, 16 * mib});
  auto b             = f.slice(group, {16 * mib, 16 * mib}, 16 * mib);
  auto second_policy = f.make_policy(2);
  b.policy           = second_policy;
  f.engine->hand_off(std::move(a));
  f.engine->hand_off(std::move(b));
  REQUIRE(until([&] {
    control->run();
    return future.is_ready();
  }));
  CHECK(std::move(future).get() == f.bytes);
  CHECK(f.contents() == f.image);
  REQUIRE(until([&] {
    control->run();
    return f.engine->drain(1ms);
  }));
  auto const stats = f.engine->gauges();
  REQUIRE(stats.mounts.size() == 2);
  for (auto const& m : stats.mounts) {
    CHECK((m.effective.qd == 4 || m.effective.qd == 0));
    CHECK(m.max_storage_qd <= 4);
    CHECK(m.max_occupied_lanes <= 5);
    CHECK(m.max_storage_qd >= 3);
  }
  auto const calls       = f.transport->submissions();
  auto const first_other = std::find_if(calls.begin(), calls.end(), [](auto const& call) {
    return call.file_offset >= static_cast<off_t>(16 * mib);
  });
  CHECK(std::distance(calls.begin(), first_other) < 8);
  CHECK(f.transport->snapshot().max_submit_concurrency <= config.submitters_per_device);
}

TEST_CASE("engine zero async grants use bounded sync staging instead of stalling",
          "[cufile][engine][sched]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(4 * mib, configuration(1));  // Two total lanes, one slab, two active mounts.
  auto control = std::make_shared<held_executor>();
  f.start([&](auto& opts) { opts.control = control; });
  auto group  = std::make_shared<grouped_coordinator>(f.bytes, 2);
  auto future = group->get_future();
  auto a      = f.slice(group, {0, 2 * mib});
  auto b      = f.slice(group, {2 * mib, 2 * mib}, 2 * mib);
  b.policy    = f.make_policy(2);
  f.engine->hand_off(std::move(a));
  f.engine->hand_off(std::move(b));
  REQUIRE(until([&] {
    control->run();
    return future.is_ready();
  }));
  CHECK(std::move(future).get() == f.bytes);
  CHECK(std::ranges::equal(f.contents(), f.image));
  REQUIRE(until([&] {
    control->run();
    return f.engine->drain(1ms);
  }));
  CHECK(f.engine->gauges().sync_bytes > 0);
  CHECK(group->tasks_remaining() == 0);
}

TEST_CASE("engine metadata exhaustion rejects without waiting for resources",
          "[cufile][engine][sched]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto config                          = configuration(1);
  config.max_pending_slices_per_device = 1;
  fixture f(mib, config);
  f.start();
  gate input(f.stream);
  auto first         = std::make_shared<grouped_coordinator>(mib, 1);
  auto first_future  = first->get_future();
  auto second        = std::make_shared<grouped_coordinator>(mib, 1);
  auto second_future = second->get_future();
  auto a             = f.slice(first, {0, mib});
  auto b             = f.slice(second, {0, mib});
  f.engine->hand_off(std::move(a));
  REQUIRE(until([&] { return first_future.is_ready(); }));
  auto const start = policy_clock::now();
  f.engine->hand_off(std::move(b));
  CHECK(policy_clock::now() - start < 20ms);
  REQUIRE_THROWS_AS(std::move(second_future).get(1s), std::system_error);
  CHECK(second->tasks_remaining() == 0);
  input.release();
  CHECK(std::move(first_future).get() == mib);
  REQUIRE(f.engine->drain(5s));
  CHECK(f.engine->gauges().metadata_rejections == 1);
}

TEST_CASE("engine adaptive event polling backs off beneath configured caps",
          "[cufile][engine][sched]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(mib, configuration(1));
  std::vector<std::pair<cudaEvent_t, unsigned>> polls;
  f.start([&](auto& opts) {
    opts.query_event = [&](cudaEvent_t event) {
      auto p = std::find_if(
        polls.begin(), polls.end(), [&](auto const& item) { return item.first == event; });
      if (p == polls.end()) {
        polls.emplace_back(event, 0);
        p = std::prev(polls.end());
      }
      if (++p->second <= 4) return cudaErrorNotReady;
      return cudaEventQuery(event);
    };
  });
  auto group  = std::make_shared<grouped_coordinator>(mib, 1);
  auto future = group->get_future();
  f.engine->hand_off(f.slice(group, {0, mib}));
  REQUIRE(std::move(future).get(5s) == mib);
  REQUIRE(f.engine->drain(5s));
  auto const stats = f.engine->gauges();
  CHECK(stats.io_poll_delay.buckets[14] > 0);    // 10 us.
  CHECK(stats.io_poll_delay.buckets[15] > 0);    // 25 us.
  CHECK(stats.io_poll_delay.buckets[16] > 0);    // 50 us.
  CHECK(stats.io_poll_delay.buckets[17] > 0);    // 100 us cap.
  CHECK(stats.copy_poll_delay.buckets[11] > 0);  // 2 us.
  CHECK(stats.copy_poll_delay.buckets[13] > 0);  // 5 us.
  CHECK(stats.copy_poll_delay.buckets[14] > 0);  // 10 us cap.
  for (std::size_t i = 18; i < 64; ++i)
    CHECK(stats.io_poll_delay.buckets[i] == 0);
  // Long-pending copy tails share the latency-scaled ceiling (AM-15 hardening).
  for (std::size_t i = 21; i < 64; ++i)
    CHECK(stats.copy_poll_delay.buckets[i] == 0);
  CHECK(stats.queries >= 10);
}

TEST_CASE("engine slot exhaustion is retryable backpressure", "[cufile][engine][sched]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(mib, configuration(1));
  static_cast<void>(f.arena->set_quota(99, 1, 0, slot_class::mib1));
  std::vector<staging_lease> borrowed;
  for (unsigned i = 0; i < 16; ++i) {
    auto lease = f.arena->try_lease(99, mib, true);
    REQUIRE(lease.has_value());
    borrowed.push_back(std::move(*lease));
  }
  f.start();
  auto group  = std::make_shared<grouped_coordinator>(mib, 1);
  auto future = group->get_future();
  f.engine->hand_off(f.slice(group, {0, mib}));
  REQUIRE(until([&] { return f.engine->gauges().slot_waits > 0; }));
  CHECK_FALSE(future.is_ready());
  borrowed.clear();
  f.engine->notify();
  CHECK(std::move(future).get(5s) == mib);
  CHECK(std::ranges::equal(f.contents(), f.image));
  REQUIRE(f.engine->drain(5s));
  CHECK(f.engine->gauges().slot_wait_ns > 0);
}

TEST_CASE("engine lane exhaustion backpressures without rejecting handoff",
          "[cufile][engine][sched]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(8 * mib, configuration(1));
  f.start();
  gate input(f.stream);
  auto group       = std::make_shared<grouped_coordinator>(f.bytes, 1);
  auto future      = group->get_future();
  auto s           = f.slice(group, {0, f.bytes});
  auto const start = policy_clock::now();
  f.engine->hand_off(std::move(s));
  CHECK(policy_clock::now() - start < 20ms);
  REQUIRE(until([&] { return f.transport->snapshot().completed == 1; }));
  CHECK_FALSE(future.is_ready());
  CHECK(f.arena->gauges().live_leases == 1);
  input.release();
  CHECK(std::move(future).get(5s) == f.bytes);
  CHECK(f.contents() == f.image);
  CHECK(f.engine->drain(5s));
}

TEST_CASE("engine multi-piece busy request leaves capacity for another caller",
          "[cufile][engine][sched]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const qd         = GENERATE(1U, 4U);
  bool const busy_first = GENERATE(false, true);
  fixture f(17 * mib, configuration(qd));
  f.start();
  cudaStream_t other{};
  REQUIRE(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking) == cudaSuccess);
  gate input(f.stream);
  auto a        = std::make_shared<grouped_coordinator>(16 * mib, 1);
  auto b        = std::make_shared<grouped_coordinator>(mib, 1);
  auto blocked  = a->get_future();
  auto ready    = b->get_future();
  auto busy     = f.slice(a, {0, 16 * mib});
  auto original = f.stream;
  f.stream      = other;
  auto idle     = f.slice(b, {16 * mib, mib}, 16 * mib);
  if (GENERATE(false, true)) idle.priority = io_priority::low;
  f.stream = original;
  if (busy_first) {
    f.engine->hand_off(std::move(busy));
    REQUIRE(until([&] { return f.transport->snapshot().completed == 1; }));
    f.engine->hand_off(std::move(idle));
  } else {
    f.engine->hand_off(std::move(idle));
    f.engine->hand_off(std::move(busy));
  }
  REQUIRE(std::move(ready).get(5s) == mib);
  REQUIRE(cudaStreamSynchronize(other) == cudaSuccess);
  CHECK_FALSE(blocked.is_ready());
  CHECK(cudaStreamQuery(f.stream) == cudaErrorNotReady);
  input.release();
  REQUIRE(std::move(blocked).get(5s) == 16 * mib);
  CHECK(f.contents() == f.image);
  REQUIRE(f.engine->drain(5s));
  REQUIRE(cudaStreamDestroy(other) == cudaSuccess);
}

TEST_CASE("engine short-read retry falls back when a second mount removes its QD",
          "[cufile][engine][sched][fault]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(2 * mib, configuration(1));
  transport_simulator::fault fault;
  fault.index             = 0;
  fault.stuck_completion  = true;
  fault.completion_result = 4097;
  f.transport->add_fault(fault);
  auto control = std::make_shared<held_executor>();
  f.start([&](auto& opts) { opts.control = control; });
  auto a       = std::make_shared<grouped_coordinator>(mib, 1);
  auto b       = std::make_shared<grouped_coordinator>(mib, 1);
  auto first   = a->get_future();
  auto second  = b->get_future();
  auto initial = f.slice(a, {0, mib});
  auto other   = f.slice(b, {mib, mib}, mib);
  other.policy = f.make_policy(2);
  f.engine->hand_off(std::move(initial));
  REQUIRE(until([&] {
    control->run();
    return f.transport->snapshot().submitted == 1;
  }));
  f.engine->hand_off(std::move(other));
  REQUIRE(until([&] {
    control->run();
    auto gauges = f.engine->gauges();
    return gauges.mounts.size() == 2 && gauges.mounts[0].effective.qd == 0;
  }));
  f.transport->unblock_all();
  REQUIRE(until([&] {
    control->run();
    return first.is_ready() && second.is_ready();
  }));
  CHECK(std::move(first).get() == mib);
  CHECK(std::move(second).get() == mib);
  CHECK(f.contents() == f.image);
  REQUIRE(until([&] {
    control->run();
    return f.engine->drain(1ms);
  }));
  CHECK(f.engine->gauges().retries == 1);
  CHECK(f.engine->gauges().sync_bytes >= mib - 4096);
}
