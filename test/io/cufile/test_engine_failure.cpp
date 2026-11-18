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

#include <fcntl.h>
#include <unistd.h>

using namespace cucascade::io;
using namespace cucascade::io::cufile;
using namespace cucascade::io::cufile::engine_test;

TEST_CASE("engine short reads retry aligned on the same slot", "[cufile][engine][fault]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(mib, configuration(1));
  transport_simulator::fault fault;
  fault.index             = 0;
  fault.completion_result = 4097;
  f.transport->add_fault(fault);
  f.start();
  auto group  = std::make_shared<grouped_coordinator>(mib, 1);
  auto future = group->get_future();
  f.engine->hand_off(f.slice(group, {0, mib}));
  REQUIRE(std::move(future).get(5s) == mib);
  CHECK(f.contents() == f.image);
  REQUIRE(f.engine->drain(5s));
  auto const calls = f.transport->submissions();
  REQUIRE(calls.size() == 2);
  CHECK(calls[1].file_offset == 4096);
  CHECK(calls[1].buffer_offset == 4096);
  CHECK(f.engine->gauges().retries == 1);
  CHECK(group->tasks_remaining() == 0);
}

TEST_CASE("engine invalid completion counts never deliver success", "[cufile][engine][fault]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const scenario    = GENERATE(0, 1, 2, 3);
  auto config            = configuration(1);
  config.max_short_reads = 2;
  fixture f(mib, config);
  transport_simulator::fault fault;
  if (scenario == 0) fault.completion_result = 0;
  if (scenario == 1) fault.completion_result = -ENOSPC;
  if (scenario == 2) fault.preserve_bytes_done = true;
  if (scenario == 3) fault.completion_result = 4097;
  f.transport->add_fault(fault);
  f.start();
  auto group  = std::make_shared<grouped_coordinator>(mib, 1);
  auto future = group->get_future();
  f.engine->hand_off(f.slice(group, {0, mib}));
  REQUIRE_THROWS(std::move(future).get(5s));
  REQUIRE(f.engine->drain(5s));
  CHECK(group->tasks_remaining() == 0);
  CHECK(f.transport->snapshot().copy_calls == 0);
  CHECK(f.arena->gauges().live_leases == 0);
  CHECK(f.transport->snapshot().read_calls == (scenario == 3 ? 3 : 1));
}

TEST_CASE("engine enqueue failures recover the owned lane and settle exactly once",
          "[cufile][engine][fault]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const failure = GENERATE(0, 1, 2, 3, 4, 5, 6);
  fixture f(mib, configuration(1));
  if (failure == 0) {
    transport_simulator::fault fault;
    fault.submit_status = {-EIO};
    f.transport->add_fault(fault);
  }
  if (failure == 1) f.transport->fail_record(0);  // I/O record.
  if (failure == 2) f.transport->fail_copy(0);
  if (failure == 3) f.transport->fail_record(1);  // Copy record, after copy enqueue.
  if (failure == 4) f.transport->fail_wait(0);    // Fence wait.
  if (failure == 5) f.transport->fail_wait(1);    // User join, after copy record.
  if (failure == 6) {
    transport_simulator::fault fault;
    fault.prepare_error = cudaErrorUnknown;  // Partial transport enqueue.
    f.transport->add_fault(fault);
  }
  std::atomic<unsigned> synchronizations{};
  std::atomic<bool> owned_lane_only{true};
  f.start([&](auto& opts) {
    opts.synchronize_lane = [&](cudaStream_t stream) {
      if (stream == f.stream) owned_lane_only.store(false);
      ++synchronizations;
      return f.transport->recover_stream(stream);
    };
  });
  auto group  = std::make_shared<grouped_coordinator>(mib, 1);
  auto future = group->get_future();
  f.engine->hand_off(f.slice(group, {0, mib}));
  if (failure == 0 || failure == 6) {
    REQUIRE(std::move(future).get(5s) == mib);
    CHECK(f.contents() == f.image);
  } else {
    REQUIRE_THROWS(std::move(future).get(5s));
  }
  REQUIRE(f.engine->drain(5s));
  CHECK(group->tasks_remaining() == 0);
  CHECK(synchronizations.load() == 1);
  CHECK(owned_lane_only.load());
  CHECK(f.engine->gauges().recoveries == 1);
  CHECK(f.arena->gauges().live_leases == 0);
  CHECK(f.lanes->available() == f.lanes->capacity());
}

TEST_CASE("engine failed join retains staging through partial-copy recovery",
          "[cufile][engine][fault]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(mib, configuration(1));
  f.transport->fail_wait(1);
  f.start();
  gate input(f.stream);
  auto group  = std::make_shared<grouped_coordinator>(mib, 1);
  auto future = group->get_future();
  f.engine->hand_off(f.slice(group, {0, mib}));
  REQUIRE(until([&] { return f.engine->gauges().recoveries == 1; }));
  CHECK_FALSE(future.is_ready());
  CHECK(f.arena->gauges().live_leases == 1);
  CHECK(f.lanes->quarantined() == 1);
  CHECK_FALSE(f.engine->drain(1ms));
  input.release();
  REQUIRE_THROWS(std::move(future).get(5s));
  REQUIRE(f.engine->drain(5s));
  CHECK(group->tasks_remaining() == 0);
}

TEST_CASE("engine cancellation distinguishes pending and submitted work", "[cufile][engine][fault]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  bool const submitted = GENERATE(false, true);
  fixture f(mib, configuration(1));
  transport_simulator::fault fault;
  fault.stuck_completion = true;
  f.transport->add_fault(fault);
  auto control = std::make_shared<held_executor>();
  f.start([&](auto& opts) { opts.control = control; });
  auto cancel = std::make_shared<std::atomic<bool>>(false);
  auto group  = std::make_shared<grouped_coordinator>(mib, 1);
  auto future = group->get_future();
  auto s      = f.slice(group, {0, mib});
  s.cancel    = cancel;
  f.engine->hand_off(std::move(s));
  if (submitted) {
    REQUIRE(until([&] {
      control->run();
      return f.transport->snapshot().submitted == 1;
    }));
  }
  cancel->store(true);
  f.engine->notify();
  if (submitted) {
    std::this_thread::sleep_for(1ms);
    CHECK_FALSE(future.is_ready());
    CHECK(f.arena->gauges().live_leases == 1);
    f.transport->unblock_all();
  }
  REQUIRE_THROWS(std::move(future).get(5s));
  REQUIRE(until([&] {
    control->run();
    return f.engine->drain(1ms);
  }));
  CHECK(f.transport->snapshot().read_calls == (submitted ? 1 : 0));
  CHECK(f.transport->snapshot().copy_calls == 0);
  CHECK(group->tasks_remaining() == 0);
}

TEST_CASE("engine shutdown cancels pending pieces and drains submitted IO",
          "[cufile][engine][fault]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(8 * mib, configuration(1));
  transport_simulator::fault fault;
  fault.stuck_completion = true;
  f.transport->add_fault(fault);
  f.start();
  auto group  = std::make_shared<grouped_coordinator>(f.bytes, 1);
  auto future = group->get_future();
  f.engine->hand_off(f.slice(group, {0, f.bytes}));
  REQUIRE(until([&] { return f.transport->snapshot().submitted == 1; }));
  f.engine->request_stop();
  CHECK_FALSE(f.engine->drain(5ms));
  CHECK_FALSE(future.is_ready());
  f.transport->unblock_all();
  REQUIRE_THROWS(std::move(future).get(5s));
  REQUIRE(f.engine->drain(5s));
  CHECK(group->tasks_remaining() == 0);
  CHECK(f.transport->snapshot().read_calls == 1);
}

TEST_CASE("engine watchdog refuses queued calls and falls back on blocking pool",
          "[cufile][engine][fault]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto config              = configuration(4);
  config.demotion.stuck_ms = 2;
  fixture f(8 * mib, config);
  transport_simulator::fault fault;
  fault.index        = 0;
  fault.stuck_submit = true;
  f.transport->add_fault(fault);
  f.start();
  auto group  = std::make_shared<grouped_coordinator>(f.bytes, 1);
  auto future = group->get_future();
  f.engine->hand_off(f.slice(group, {0, f.bytes}));
  REQUIRE(until([&] { return f.transport->snapshot().blocked_submissions == 1; }));
  REQUIRE(until([&] { return f.policy->state() == mount_state::circuit_open; }));
  CHECK_FALSE(future.is_ready());
  REQUIRE(until([&] { return f.engine->gauges().sync_bytes == f.bytes - mib; }));
  CHECK(f.transport->snapshot().read_calls == 1);  // Unstarted queue drained while call is stuck.
  f.transport->unblock_all();
  CHECK(std::move(future).get(5s) == f.bytes);
  CHECK(f.contents() == f.image);
  REQUIRE(f.engine->drain(5s));
  CHECK(f.engine->gauges().not_submitted > 0);
  CHECK(f.engine->gauges().sync_bytes > 0);
  CHECK(f.transport->snapshot().read_calls == 1);
  CHECK(group->tasks_remaining() == 0);
}

TEST_CASE("engine migrates another mount away from a stuck submitter", "[cufile][engine][fault]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto config                  = configuration(1);
  config.device_lane_cap       = 6;
  config.arena_bytes           = 48 * mib;
  config.submitters_per_device = 2;
  config.demotion.stuck_ms     = 2;
  fixture f(3 * mib, config);
  transport_simulator::fault fault;
  fault.file_offset  = 0;
  fault.stuck_submit = true;
  f.transport->add_fault(fault);
  auto control = std::make_shared<held_executor>();
  f.start([&](auto& opts) { opts.control = control; });
  std::vector<std::shared_ptr<grouped_coordinator>> groups;
  std::vector<cucascade::exec::semi_future<std::size_t>> futures;
  std::vector<engine_slice> slices;
  for (std::size_t i = 0; i < 3; ++i) {
    groups.push_back(std::make_shared<grouped_coordinator>(mib, 1));
    futures.push_back(groups.back()->get_future());
    auto s   = f.slice(groups.back(), {i * mib, mib}, i * mib);
    s.policy = i == 0 ? f.policy : f.make_policy(i + 1);
    slices.push_back(std::move(s));
  }
  for (auto& s : slices)
    f.engine->hand_off(std::move(s));
  REQUIRE(until([&] {
    control->run();
    return f.transport->snapshot().blocked_submissions == 1;
  }));
  REQUIRE(until([&] {
    control->run();
    return futures[2].is_ready();
  }));
  CHECK_FALSE(futures[0].is_ready());
  CHECK(std::move(futures[2]).get() == mib);
  f.transport->unblock_all();
  for (std::size_t i = 0; i < 2; ++i)
    CHECK(std::move(futures[i]).get(5s) == mib);
  CHECK(std::ranges::equal(f.contents(), f.image));
  REQUIRE(until([&] {
    control->run();
    return f.engine->drain(1ms);
  }));
}

TEST_CASE("engine cleanup transfers the final file owner off the reaper", "[cufile][engine][fault]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  struct slow_file final : io_object {
    std::atomic<bool>& entered;
    std::atomic<bool>& release;
    std::atomic<std::size_t>& thread;
    std::string path{"slow-engine-owner"};
    slow_file(std::atomic<bool>& a, std::atomic<bool>& b, std::atomic<std::size_t>& t)
      : entered(a), release(b), thread(t)
    {
    }
    ~slow_file() override
    {
      thread.store(std::hash<std::thread::id>{}(std::this_thread::get_id()));
      entered.store(true);
      while (!release.load())
        std::this_thread::sleep_for(50us);
    }
    std::size_t size() const noexcept override { return mib; }
    std::string const& object_path() const noexcept override { return path; }
    std::string const& raw_file_cache_id() const noexcept override { return path; }
  };
  std::atomic<bool> entered{}, release{};
  std::atomic<std::size_t> destructor_thread{}, reaper_thread{};
  fixture f(2 * mib, configuration(2));
  struct unblock {
    std::atomic<bool>& flag;
    ~unblock() { flag.store(true); }
  } guard{release};
  f.start([&](auto& opts) {
    opts.query_event = [&](cudaEvent_t event) {
      reaper_thread.store(std::hash<std::thread::id>{}(std::this_thread::get_id()));
      return cudaEventQuery(event);
    };
  });
  auto first  = std::make_shared<grouped_coordinator>(mib, 1);
  auto future = first->get_future();
  auto s      = f.slice(first, {0, mib});
  s.file      = std::make_shared<slow_file>(entered, release, destructor_thread);
  f.engine->hand_off(std::move(s));
  REQUIRE(std::move(future).get(5s) == mib);
  REQUIRE(until([&] { return entered.load(); }));
  auto second = std::make_shared<grouped_coordinator>(mib, 1);
  auto other  = second->get_future();
  f.engine->hand_off(f.slice(second, {mib, mib}, mib));
  CHECK(std::move(other).get(5s) == mib);  // Progresses while a destructor is blocked.
  CHECK(destructor_thread.load() != reaper_thread.load());
  release.store(true);
  REQUIRE(f.engine->drain(5s));
}

TEST_CASE("engine failed recovery quarantines without settling a fictitious completion",
          "[cufile][engine][fault]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  // A failed quiescence proof is deliberately process-lifetime, just like production quarantine.
  // Keep ALL injected dependencies alive. Actual CUDA is synchronized first so the test itself
  // does not poison the device or leave a live kernel behind.
  auto* f = new fixture(mib, configuration(1));
  f->transport->fail_copy(0);
  f->start([](auto& opts) {
    opts.synchronize_lane = [](cudaStream_t stream) {
      static_cast<void>(cudaStreamSynchronize(stream));
      return cudaErrorUnknown;
    };
  });
  auto group  = std::make_shared<grouped_coordinator>(mib, 1);
  auto future = group->get_future();
  f->engine->hand_off(f->slice(group, {0, mib}));
  REQUIRE(until([&] { return f->engine->gauges().quarantines == 1; }));
  f->engine->request_stop();
  CHECK_FALSE(f->engine->drain(5ms));
  CHECK_FALSE(future.is_ready());
  CHECK(group->tasks_remaining() == 1);
  CHECK(f->arena->gauges().live_leases == 1);
  CHECK(f->lanes->quarantined() == 1);
}

TEST_CASE("engine recoverable first completions use sync fallback exactly once",
          "[cufile][engine][fault]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const result = GENERATE(-EINVAL, -EOPNOTSUPP, -ENOSYS, -1, -EIO);
  fixture f(mib, configuration(1));
  transport_simulator::fault fault;
  fault.index             = 0;
  fault.completion_result = result;
  f.transport->add_fault(fault);
  f.start();
  auto group  = std::make_shared<grouped_coordinator>(mib, 1);
  auto future = group->get_future();
  f.engine->hand_off(f.slice(group, {0, mib}));
  REQUIRE(std::move(future).get(5s) == mib);
  CHECK(f.contents() == f.image);
  REQUIRE(f.engine->drain(5s));
  CHECK(f.engine->gauges().sync_bytes == mib);
  CHECK(f.engine->gauges().unsupported_fallbacks == 1);
  CHECK(f.transport->snapshot().read_calls == 1);
}

TEST_CASE("engine repeated short reads preserve nonzero physical-piece offsets",
          "[cufile][engine][fault]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(3 * mib, configuration(1));
  transport_simulator::fault first;
  first.index              = 0;
  first.completion_result  = 4097;
  auto second              = first;
  second.index             = 1;
  second.completion_result = 8193;
  f.transport->add_fault(first);
  f.transport->add_fault(second);
  f.start();
  auto group  = std::make_shared<grouped_coordinator>(mib - 37, 1);
  auto future = group->get_future();
  f.engine->hand_off(f.slice(group, {mib + 37, mib - 37}));
  REQUIRE(std::move(future).get(5s) == mib - 37);
  auto contents = f.contents(mib - 37);
  CHECK(std::equal(contents.begin(), contents.end(), f.image.begin() + mib + 37));
  REQUIRE(f.engine->drain(5s));
  auto calls = f.transport->submissions();
  REQUIRE(calls.size() == 3);
  CHECK(calls[1].file_offset == static_cast<off_t>(mib + 4096));
  CHECK(calls[1].buffer_offset == 4096);
  CHECK(calls[2].file_offset == static_cast<off_t>(mib + 12288));
  CHECK(calls[2].buffer_offset == 12288);
}

TEST_CASE("engine post-success copy-query error is counted and recovered",
          "[cufile][engine][fault]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(mib, configuration(1));
  std::atomic<bool> injected{};
  f.start([&](auto& opts) {
    opts.query_event = [&](cudaEvent_t event) {
      if (f.transport->snapshot().copy_calls != 0 && !injected.exchange(true))
        return cudaErrorUnknown;
      return cudaEventQuery(event);
    };
  });
  auto group  = std::make_shared<grouped_coordinator>(mib, 1);
  auto future = group->get_future();
  f.engine->hand_off(f.slice(group, {0, mib}));
  REQUIRE(std::move(future).get(5s) == mib);  // Cannot revoke an enqueued success.
  REQUIRE(f.engine->drain(5s));
  CHECK(injected.load());
  CHECK(f.engine->gauges().late_copy_failures == 1);
  CHECK(f.engine->gauges().recoveries == 1);
  CHECK(group->tasks_remaining() == 0);
}

namespace {
struct backing_file {
  char path[64]{"/tmp/cucascade_engine_fault_XXXXXX"};
  int fd{mkstemp(path)};
  backing_file(std::vector<std::byte> const& image)
  {
    REQUIRE(fd >= 0);
    REQUIRE(pwrite(fd, image.data(), image.size(), 0) == static_cast<ssize_t>(image.size()));
  }
  ~backing_file()
  {
    static_cast<void>(close(fd));
    static_cast<void>(unlink(path));
  }
};
}  // namespace

TEST_CASE("engine short read detects a file shrunk after open", "[cufile][engine][fault]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(mib, configuration(1));
  backing_file disk(f.image);
  transport_simulator::fault fault;
  fault.index             = 0;
  fault.completion_result = 4097;
  fault.stuck_completion  = true;
  f.transport->add_fault(fault);
  f.start([&](auto& opts) { opts.buffered_fd = [&](io_object const&) { return disk.fd; }; });
  auto group  = std::make_shared<grouped_coordinator>(mib, 1);
  auto future = group->get_future();
  f.engine->hand_off(f.slice(group, {0, mib}));
  REQUIRE(until([&] { return f.transport->snapshot().submitted == 1; }));
  REQUIRE(ftruncate(disk.fd, 4097) == 0);
  f.transport->unblock_all();
  REQUIRE_THROWS(std::move(future).get(5s));
  REQUIRE(f.engine->drain(5s));
  CHECK(f.transport->snapshot().read_calls == 1);
  CHECK(group->tasks_remaining() == 0);
}

TEST_CASE("engine buffered fallback handles retry refusal and scratch recovery",
          "[cufile][engine][fault]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const scenario = GENERATE(0, 1, 2);
  fixture f(mib, configuration(1));
  backing_file disk(f.image);
  if (scenario == 2) {
    auto fd = open(disk.path, O_WRONLY);
    REQUIRE(fd >= 0);
    REQUIRE(close(disk.fd) == 0);
    disk.fd = fd;  // fstat succeeds but pread fails; never re-enter fallback on that error.
  }
  transport_simulator::fault first;
  first.index              = 0;
  first.completion_result  = 4097;
  auto second              = first;
  second.index             = 1;
  second.completion_result = -EINVAL;
  f.transport->add_fault(first);
  f.transport->add_fault(second);
  f.transport->set_sync_result(-EINVAL);
  std::atomic<unsigned> syncs{};
  f.start([&](auto& opts) {
    opts.buffered_fd      = [&](io_object const&) { return disk.fd; };
    opts.synchronize_lane = [&](cudaStream_t stream) {
      auto status = cudaStreamSynchronize(stream);
      if (++syncs == 1 && scenario == 1) return cudaErrorUnknown;
      return status;
    };
  });
  auto group  = std::make_shared<grouped_coordinator>(mib, 1);
  auto future = group->get_future();
  f.engine->hand_off(f.slice(group, {0, mib}));
  if (scenario == 0) {
    REQUIRE(std::move(future).get(5s) == mib);
    CHECK(f.contents() == f.image);
  } else {
    REQUIRE_THROWS(std::move(future).get(5s));
  }
  REQUIRE(f.engine->drain(5s));
  CHECK(f.transport->snapshot().read_calls == 2);
  CHECK(group->tasks_remaining() == 0);
  CHECK(f.arena->gauges().live_leases == 0);
  if (scenario == 1) CHECK(syncs.load() == 2);
}

TEST_CASE("engine cleanup progresses while sync fallback executor is held",
          "[cufile][engine][fault]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto config            = configuration(2);
  config.device_lane_cap = 6;
  config.arena_bytes     = 32 * mib;
  fixture f(2 * mib, config);
  transport_simulator::fault fault;
  fault.index             = 0;
  fault.completion_result = -EINVAL;
  f.transport->add_fault(fault);
  auto blocking = std::make_shared<held_executor>();
  f.start([&](auto& opts) { opts.blocking = blocking; });
  auto first   = std::make_shared<grouped_coordinator>(mib, 1);
  auto second  = std::make_shared<grouped_coordinator>(mib, 1);
  auto pending = first->get_future();
  auto ready   = second->get_future();
  // Prepare fences before any caller-stream join.
  auto a   = f.slice(first, {0, mib});
  auto b   = f.slice(second, {mib, mib}, mib);
  b.policy = f.make_policy(2);
  f.engine->hand_off(std::move(a));
  REQUIRE(until([&] { return blocking->size() > 0; }));
  CHECK_FALSE(pending.is_ready());
  f.engine->hand_off(std::move(b));
  REQUIRE(std::move(ready).get(5s) == mib);
  // An unrelated mount's operation retires while fallback remains held.
  REQUIRE(until([&] { return f.engine->gauges().completed_ops == 1; }));
  CHECK_FALSE(pending.is_ready());
  REQUIRE(until([&] {
    blocking->run();
    return pending.is_ready();
  }));
  CHECK(std::move(pending).get() == mib);
  REQUIRE(f.engine->drain(5s));  // Cursor/operation cleanup needs no more blocking->run().
  CHECK(f.contents() == f.image);
}
