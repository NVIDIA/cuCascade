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

#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/runtime.hpp>

#include <fcntl.h>
#include <unistd.h>

#include <future>

using namespace cucascade::io;
using namespace cucascade::io::cufile;
using namespace cucascade::io::cufile::engine_test;

namespace {
async_engine_config write_config(unsigned qd = 4)
{
  auto c                                 = configuration(qd);
  c.async_writes                         = true;
  c.mount_overrides.front().allow_writes = true;
  c.mount_overrides.push_back(
    {mount_match::mount_prefix, "/", mount_override_state::operator_qualified, qd, mib, true});
  return c;
}
struct write_fixture {
  fixture r;
  std::string path;
  std::shared_ptr<cufile_io_object> native;
  std::shared_ptr<file_write_state> state;
  explicit write_fixture(std::size_t bytes = 4 * mib, unsigned qd = 4) : r(bytes, write_config(qd))
  {
    char name[] = "/tmp/cucascade_engine_write_XXXXXX";
    auto fd     = mkstemp(name);
    REQUIRE(fd >= 0);
    close(fd);
    path   = name;
    native = cufile_reactor::create_io_object_for_write(
      path, {write_mode::open_existing}, true, compat::automatic);
    r.object = native;
    state    = native->write_state();
    r.transport->mirror_writes_to(native->buffered_fd());
    REQUIRE(
      cudaMemcpyAsync(r.destination, r.image.data(), bytes, cudaMemcpyHostToDevice, r.stream) ==
      cudaSuccess);
  }
  engine_slice slice(std::shared_ptr<grouped_coordinator> const& g,
                     range rng,
                     std::size_t source_offset = 0)
  {
    auto s         = r.slice(g, rng, source_offset);
    s.write        = true;
    s.write_ticket = state->begin();
    return s;
  }
  std::vector<std::byte> disk(std::size_t bytes)
  {
    std::vector<std::byte> result(bytes);
    REQUIRE(pread(native->buffered_fd(), result.data(), bytes, 0) == static_cast<ssize_t>(bytes));
    return result;
  }
  auto write(range rng, std::size_t source = 0)
  {
    auto g      = std::make_shared<grouped_coordinator>(rng.size, 1);
    auto future = g->get_future();
    r.engine->hand_off(slice(g, rng, source));
    return future;
  }
  ~write_fixture()
  {
    r.transport->unblock_all();
    if (r.engine) {
      r.engine->request_stop();
      static_cast<void>(r.engine->drain(5s));
    }
    unlink(path.c_str());
  }
};
}  // namespace

TEST_CASE("inode intervals permit same-mode disjoint bytes but exclude cross-mode granules",
          "[cufile][engine][write]")
{
  char name[] = "/tmp/cucs_write_intervals_XXXXXX";
  file_descriptor fd{mkstemp(name)};
  REQUIRE(fd.get() >= 0);
  unlink(name);
  auto state = file_write_state::get(fd.get());
  auto first = state->try_acquire({37, 100}, 4096, write_io_mode::buffered);
  REQUIRE(first);
  auto adjacent = state->try_acquire({200, 100}, 4096, write_io_mode::buffered);
  CHECK(adjacent);
  CHECK_FALSE(state->try_acquire({37, 100}, 4096, write_io_mode::buffered));
  CHECK_FALSE(state->try_acquire({0, 4096}, 4096, write_io_mode::direct));
  CHECK(state->try_acquire({4096, 4096}, 4096, write_io_mode::direct));
}

TEST_CASE("engine write aligned single and striped bodies reach the file",
          "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const bytes = GENERATE(mib, 7 * mib);
  write_fixture w(bytes);
  w.r.start();
  CHECK(w.write({0, bytes}).get(5s) == bytes);
  CHECK(w.disk(bytes) == w.r.image);
  CHECK(w.native->size() == bytes);
  REQUIRE(w.r.engine->drain(5s));
  CHECK(w.r.transport->snapshot().write_calls == bytes / mib);
  CHECK(w.r.transport->snapshot().read_calls == 0);
  CHECK(w.r.engine->gauges().async_write_bytes == bytes);
  for (auto const& s : w.r.transport->submissions())
    CHECK(s.buffer_offset == 0);
}

TEST_CASE("engine write unaligned head and tail are exact buffered operations",
          "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const bytes = GENERATE(std::size_t{77}, 2 * mib + 77);
  write_fixture w(bytes + 37);
  w.r.start();
  CHECK(w.write({37, bytes}).get(5s) == bytes);
  auto disk = w.disk(bytes + 37);
  CHECK(std::all_of(disk.begin(), disk.begin() + 37, [](auto b) { return b == std::byte{}; }));
  CHECK(std::equal(disk.begin() + 37, disk.end(), w.r.image.begin()));
  CHECK(w.native->size() == bytes + 37);
  REQUIRE(w.r.engine->drain(5s));
  CHECK(w.r.engine->gauges().buffered_write_bytes > 0);
}

TEST_CASE("engine write mixed slices keep exact independent credits", "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  write_fixture w;
  w.r.start();
  auto g      = std::make_shared<grouped_coordinator>(2 * mib + 107, 3);
  auto future = g->get_future();
  auto a      = w.slice(g, {0, mib});
  auto b      = w.slice(g, {mib, mib}, mib);
  auto c      = w.slice(g, {2 * mib, 107}, 2 * mib);
  w.r.engine->hand_off(std::move(a));
  w.r.engine->hand_off(std::move(b));
  w.r.engine->hand_off(std::move(c));
  CHECK(std::move(future).get(5s) == 2 * mib + 107);
  auto disk = w.disk(2 * mib + 107);
  CHECK(std::equal(disk.begin(), disk.end(), w.r.image.begin()));
  CHECK(g->tasks_remaining() == 0);
}

TEST_CASE("engine writes to disjoint ranges progress concurrently", "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  write_fixture w;
  transport_simulator::fault f;
  f.index            = 0;
  f.stuck_completion = true;
  w.r.transport->add_fault(f);
  w.r.start();
  auto first = w.write({0, mib});
  REQUIRE(until([&] { return w.r.transport->snapshot().blocked_completions == 1; }));
  auto second = w.write({mib, mib}, mib);
  CHECK(std::move(second).get(5s) == mib);
  CHECK_FALSE(std::move(first).wait(1ms));
  w.r.transport->unblock_all();
  CHECK(std::move(first).get(5s) == mib);
  CHECK(w.disk(2 * mib) == std::vector<std::byte>(w.r.image.begin(), w.r.image.begin() + 2 * mib));
}

TEST_CASE("engine buffered edge waits for a conflicting direct granule across handles",
          "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  write_fixture w;
  auto other = cufile_reactor::create_io_object_for_write(
    w.path, {write_mode::open_existing}, true, compat::automatic);
  CHECK(other->write_state() == w.state);
  transport_simulator::fault f;
  f.index            = 0;
  f.stuck_completion = true;
  w.r.transport->add_fault(f);
  w.r.start();
  auto first = w.write({0, mib});
  REQUIRE(until([&] { return w.r.transport->snapshot().blocked_completions == 1; }));
  auto g      = std::make_shared<grouped_coordinator>(100, 1);
  auto future = g->get_future();
  auto s      = w.slice(g, {37, 100}, mib);
  s.file      = std::move(other);
  w.r.engine->hand_off(std::move(s));
  CHECK_FALSE(std::move(future).wait(10ms));
  REQUIRE(until([&] { return w.r.engine->gauges().granule_conflicts > 0; }));
  w.r.transport->unblock_all();
  CHECK(std::move(first).get(5s) == mib);
  CHECK(std::move(future).get(5s) == 100);
  auto expected = std::vector<std::byte>(w.r.image.begin(), w.r.image.begin() + mib);
  std::copy_n(w.r.image.begin() + mib, 100, expected.begin() + 37);
  CHECK(w.disk(mib) == expected);
}

TEST_CASE("engine flush captures only prior inode sequences without blocking a worker",
          "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  write_fixture w;
  transport_simulator::fault f;
  f.index            = 0;
  f.stuck_completion = true;
  w.r.transport->add_fault(f);
  f.index = 1;
  w.r.transport->add_fault(f);
  w.r.start();
  auto first = w.write({0, mib});
  REQUIRE(until([&] { return w.r.transport->snapshot().blocked_completions == 1; }));
  auto flush = device_engine::flush_async(w.native, w.state);
  auto later = w.write({mib, mib}, mib);
  REQUIRE(until([&] { return w.r.transport->snapshot().blocked_completions == 2; }));
  CHECK_FALSE(std::move(flush).wait(5ms));
  REQUIRE(w.r.transport->unblock(0));
  CHECK(std::move(first).get(5s) == mib);
  CHECK(std::move(flush).wait(5s));
  CHECK_NOTHROW(std::move(flush).get());
  CHECK_FALSE(std::move(later).wait(1ms));
  w.r.transport->unblock_all();
  CHECK(std::move(later).get(5s) == mib);
}

TEST_CASE("engine positive write suffix retries aligned or buffers exact remainder",
          "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const short_count = GENERATE(ssize_t{4096}, ssize_t{4103});
  write_fixture w(mib);
  transport_simulator::fault f;
  f.index             = 0;
  f.completion_result = short_count;
  w.r.transport->add_fault(f);
  w.r.start();
  CHECK(w.write({0, mib}).get(5s) == mib);
  CHECK(w.disk(mib) == w.r.image);
  REQUIRE(w.r.engine->drain(5s));
  CHECK(w.r.transport->snapshot().write_calls == (short_count == 4096 ? 2 : 1));
  CHECK(w.r.engine->gauges().short_writes == 1);
  CHECK(w.r.engine->gauges().buffered_suffixes == (short_count == 4096 ? 0 : 1));
  if (short_count == 4096) {
    auto calls = w.r.transport->submissions();
    CHECK(calls.back().file_offset == 4096);
    CHECK(calls.back().buffer_offset == 4096);
  }
}

TEST_CASE("engine nonpositive and unknown write progress never replays", "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const done = GENERATE(ssize_t{0},
                             ssize_t{-EIO},
                             ssize_t{-ESTALE},
                             ssize_t{-EAGAIN},
                             ssize_t{-ETIMEDOUT},
                             ssize_t{-ENOTCONN},
                             pending_sentinel);
  write_fixture w(mib);
  transport_simulator::fault f;
  f.index = 0;
  if (done == pending_sentinel)
    f.preserve_bytes_done = true;
  else
    f.completion_result = done;
  w.r.transport->add_fault(f);
  w.r.start();
  CHECK_THROWS(w.write({0, mib}).get(5s));
  REQUIRE(w.r.engine->drain(5s));
  CHECK(w.r.transport->snapshot().write_calls == 1);
  CHECK(w.r.engine->gauges().direct_write_bytes == 0);
  CHECK(w.r.engine->gauges().buffered_write_bytes == 0);
  CHECK(w.r.engine->gauges().recoveries == (done == pending_sentinel ? 1 : 0));
  CHECK(w.native->size() == 0);
}

TEST_CASE("engine write submit failure and lost event recover without replay",
          "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const submit_error = GENERATE(false, true);
  write_fixture w(mib);
  transport_simulator::fault f;
  f.index = 0;
  if (submit_error)
    f.submit_status = {EIO, 0};
  else
    f.record_error = cudaErrorUnknown;
  w.r.transport->add_fault(f);
  w.r.start();
  CHECK_THROWS(w.write({0, mib}).get(5s));
  REQUIRE(w.r.engine->drain(5s));
  CHECK(w.r.transport->snapshot().write_calls == 1);
  CHECK(w.r.engine->gauges().recoveries == 1);
  CHECK(w.r.engine->gauges().direct_write_bytes == 0);
  CHECK(w.r.engine->gauges().buffered_write_bytes == 0);
  CHECK(w.native->size() == 0);
}

TEST_CASE("engine unsubmitted write falls back with its original credit", "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  write_fixture w(mib);
  w.r.policy->open_circuit();
  w.r.start();
  CHECK(w.write({0, mib}).get(5s) == mib);
  CHECK(w.disk(mib) == w.r.image);
  REQUIRE(w.r.engine->drain(5s));
  CHECK(w.r.transport->snapshot().write_calls == 0);
  CHECK(w.r.engine->gauges().direct_write_bytes == mib);
}

TEST_CASE("engine pending write cancellation issues no storage operation",
          "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  write_fixture w(mib);
  w.r.start();
  auto g      = std::make_shared<grouped_coordinator>(mib, 1);
  auto future = g->get_future();
  auto s      = w.slice(g, {0, mib});
  s.cancel    = std::make_shared<std::atomic<bool>>(true);
  w.r.engine->hand_off(std::move(s));
  CHECK_THROWS(std::move(future).get(5s));
  REQUIRE(w.r.engine->drain(5s));
  CHECK(w.r.transport->snapshot().write_calls == 0);
}

TEST_CASE("engine submitted write cancellation drains before source lifetime ends",
          "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  write_fixture w(mib);
  transport_simulator::fault f;
  f.index            = 0;
  f.stuck_completion = true;
  w.r.transport->add_fault(f);
  w.r.start();
  auto g      = std::make_shared<grouped_coordinator>(mib, 1);
  auto future = g->get_future();
  auto s      = w.slice(g, {0, mib});
  auto cancel = s.cancel = std::make_shared<std::atomic<bool>>(false);
  w.r.engine->hand_off(std::move(s));
  REQUIRE(until([&] { return w.r.transport->snapshot().blocked_completions == 1; }));
  cancel->store(true);
  w.r.engine->notify();
  CHECK_FALSE(std::move(future).wait(5ms));
  w.r.transport->unblock_all();
  CHECK(std::move(future).get(5s) == mib);  // Already submitted: drain, don't cancel or replay.
  CHECK(w.disk(mib) == w.r.image);
}

TEST_CASE("engine write fence gates the source staging copy", "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const edge = GENERATE(false, true);
  write_fixture w(mib);
  w.r.start();
  gate producer(w.r.stream);
  REQUIRE(cudaMemsetAsync(w.r.destination, 0x5a, mib, w.r.stream) == cudaSuccess);
  auto future = w.write({edge ? 37UL : 0UL, edge ? 100UL : mib});
  CHECK_FALSE(std::move(future).wait(10ms));
  CHECK(w.r.transport->snapshot().write_calls == 0);
  producer.release();
  auto count = edge ? 100UL : mib;
  CHECK(std::move(future).get(5s) == count);
  auto disk = w.disk(count + (edge ? 37UL : 0UL));
  CHECK(std::all_of(
    disk.begin() + (edge ? 37 : 0), disk.end(), [](auto b) { return b == std::byte{0x5a}; }));
}

TEST_CASE("ioctx write routing and flush preserve legacy opt outs",
          "[cufile][engine][write][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const route = GENERATE(0, 1, 2, 3, 4, 5);
  write_fixture w(3 * mib + 77);
  if (!w.native->cufile_registered()) SKIP("compat libcufile registration unavailable");
  config cfg;
  cfg.async_engine = w.r.config;
  if (route == 1) cfg.async_engine.mode = engine_mode::legacy;
  if (route == 2) cfg.async_engine.async_writes = false;
  if (route == 3)
    for (auto& entry : cfg.async_engine.mount_overrides)
      entry.allow_writes = false;
  if (route == 4)
    for (auto& entry : cfg.async_engine.mount_overrides) {
      entry.state        = mount_override_state::known_compat;
      entry.allow_writes = false;
    }
  // Installed shared engine models the first, reads-only context.
  if (route == 5) w.r.config.async_writes = false;
  auto shared = std::make_shared<cufile_reactor::reactor_context>(cfg);
  w.r.start();
  auto binding         = std::make_shared<cufile_reactor::reactor_context::engine_binding>();
  binding->engine      = w.r.engine.get();
  binding->arena       = w.r.arena.get();
  binding->transport   = w.r.transport;
  binding->simulated   = true;
  binding->device_uuid = "write-api";
  static std::atomic<std::uint64_t> generation{9000000};
  binding->generation = generation.fetch_add(1);
  shared->install_engine(0, std::move(binding));
  cufile_ioctx context(2, shared);
  context.start();
  std::vector<prepared_io_slice> slices;
  slices.push_back(make_device_write_slice({37, 2 * mib + 17},
                                           static_cast<std::uint8_t*>(w.r.destination),
                                           ::cuda::stream_ref{w.r.stream},
                                           0));
  slices.push_back(
    make_host_write_slice({3 * mib, 77}, reinterpret_cast<std::uint8_t const*>(w.r.image.data())));
  auto future = context.mixed_writev_async_io(*w.native, std::move(slices));
  auto flush  = context.flush_async_io(*w.native, io_priority::high);
  CHECK(std::move(future).get(5s) == 2 * mib + 94);
  CHECK_NOTHROW(std::move(flush).get(5s));
  auto disk = w.disk(3 * mib + 77);
  CHECK(std::equal(disk.begin() + 37, disk.begin() + 2 * mib + 54, w.r.image.begin()));
  CHECK(std::equal(disk.begin() + 3 * mib, disk.end(), w.r.image.begin()));
  CHECK(w.native->size() == 3 * mib + 77);
  REQUIRE(w.r.engine->drain(5s));
  CHECK((w.r.transport->snapshot().write_calls > 0) == (route == 0));
  context.shutdown();
}

TEST_CASE("engine breaker returns queued writes as not submitted and preserves their credits",
          "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  write_fixture w(3 * mib);
  w.r.config.demotion.stuck_ms = 30000;  // This test opens the breaker explicitly.
  w.r.policy                   = w.r.make_policy(1);
  transport_simulator::fault f;
  f.index        = 0;
  f.stuck_submit = true;
  w.r.transport->add_fault(f);
  w.r.start();
  auto future = w.write({0, 3 * mib});
  REQUIRE(until([&] { return w.r.transport->snapshot().blocked_submissions == 1; }));
  // Live operations may still be staging. Tokens prove both siblings reached
  // the submit queue before the explicit breaker transition.
  REQUIRE(until([&] { return w.r.engine->gauges().storage_qd >= 3; }));
  w.r.policy->open_circuit();
  w.r.engine->notify();
  REQUIRE(until([&] { return w.r.engine->gauges().not_submitted >= 2; }));
  CHECK_FALSE(std::move(future).wait(1ms));
  w.r.transport->unblock_all();
  CHECK(std::move(future).get(5s) == 3 * mib);
  CHECK(w.disk(3 * mib) == w.r.image);
  CHECK(w.r.transport->snapshot().write_calls == 1);
}

TEST_CASE("engine watchdog delayed write success validates progress without replay",
          "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  write_fixture w(mib);
  w.r.config.demotion.stuck_ms = 2;
  w.r.policy                   = w.r.make_policy(1);
  transport_simulator::fault f;
  f.index        = 0;
  f.stuck_submit = true;
  w.r.transport->add_fault(f);
  w.r.start();
  auto future = w.write({0, mib});
  REQUIRE(until([&] { return w.r.transport->snapshot().blocked_submissions == 1; }));
  REQUIRE(until([&] { return w.r.policy->state() == mount_state::circuit_open; }));
  CHECK_FALSE(std::move(future).wait(1ms));
  w.r.transport->unblock_all();
  CHECK(std::move(future).get(5s) == mib);
  REQUIRE(w.r.engine->drain(5s));
  CHECK(w.r.transport->snapshot().write_calls == 1);
  CHECK(w.disk(mib) == w.r.image);
  CHECK(w.native->size() == mib);
}

TEST_CASE("engine delayed five millisecond write submit succeeds", "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  write_fixture w(mib);
  w.r.config.demotion.stuck_ms = 2;
  w.r.policy                   = w.r.make_policy(1);
  transport_simulator::fault f;
  f.submit_latency = 5ms;
  w.r.transport->add_fault(f);
  w.r.start();
  CHECK(w.write({0, mib}).get(5s) == mib);
  REQUIRE(w.r.engine->drain(5s));
  CHECK(w.disk(mib) == w.r.image);
  CHECK(w.r.transport->snapshot().write_calls == 1);
}

TEST_CASE("engine stage polling backs off across a long caller fence", "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  write_fixture w(mib);
  w.r.start();
  // CUDA callback performs no CUDA calls; the fence follows 100 ms of application work.
  REQUIRE(cudaLaunchHostFunc(
            w.r.stream, [](void*) { std::this_thread::sleep_for(100ms); }, nullptr) == cudaSuccess);
  CHECK(w.write({0, mib}).get(5s) == mib);
  REQUIRE(w.r.engine->drain(5s));
  auto g = w.r.engine->gauges();
  INFO("queries per op=" << g.queries_per_op);
  CHECK(g.queries_per_op <= 70);
  CHECK(w.disk(mib) == w.r.image);
}

TEST_CASE("engine write failure waits for every submitted sibling to become quiescent",
          "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  write_fixture w(2 * mib);
  transport_simulator::fault f;
  f.index             = 0;
  f.completion_result = -EIO;
  f.stuck_completion  = true;
  w.r.transport->add_fault(f);
  f.index = 1;
  f.completion_result.reset();
  f.stuck_completion = true;
  w.r.transport->add_fault(f);
  w.r.start();
  auto future = w.write({0, 2 * mib});
  REQUIRE(until([&] { return w.r.transport->snapshot().blocked_completions == 2; }));
  REQUIRE(w.r.transport->unblock(0));
  REQUIRE(until([&] { return w.r.transport->snapshot().completed == 1; }));
  CHECK_FALSE(std::move(future).wait(10ms));
  w.r.transport->unblock_all();
  CHECK_THROWS(std::move(future).get(5s));
  CHECK(w.r.transport->snapshot().current_inflight == 0);
  REQUIRE(w.r.engine->drain(5s));
  CHECK(w.r.transport->snapshot().write_calls == 2);
}

TEST_CASE("engine staging failures recover before releasing write source lifetime",
          "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const which = GENERATE(0, 1, 2);
  write_fixture w(mib);
  if (which == 0) w.r.transport->fail_wait(0);
  if (which == 1) w.r.transport->fail_copy(0);
  if (which == 2) w.r.transport->fail_record(0);
  w.r.start();
  CHECK_THROWS(w.write({0, mib}).get(5s));
  REQUIRE(w.r.engine->drain(5s));
  CHECK(w.r.transport->snapshot().write_calls == 0);
  CHECK(w.r.engine->gauges().recoveries == 1);
  CHECK(w.r.engine->gauges().direct_write_bytes == 0);
  CHECK(w.r.engine->gauges().buffered_write_bytes == 0);
  CHECK(w.native->size() == 0);
}

TEST_CASE("engine public flush waits before a gated write is even admitted",
          "[cufile][engine][write][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  write_fixture w(mib);
  if (!w.native->cufile_registered()) SKIP("compat libcufile registration unavailable");
  config cfg;
  cfg.async_engine = w.r.config;
  auto shared      = std::make_shared<cufile_reactor::reactor_context>(cfg);
  w.r.start();
  auto binding         = std::make_shared<cufile_reactor::reactor_context::engine_binding>();
  binding->engine      = w.r.engine.get();
  binding->arena       = w.r.arena.get();
  binding->transport   = w.r.transport;
  binding->simulated   = true;
  binding->device_uuid = "gated-write-api";
  shared->install_engine(0, std::move(binding));
  cufile_ioctx context(1, shared);
  context.start();
  gate producer(w.r.stream);
  write_slice s{0, mib, static_cast<std::uint8_t*>(w.r.destination)};
  auto write =
    context.device_writev_async(*w.native, std::span{&s, 1}, ::cuda::stream_ref{w.r.stream});
  auto flush = context.flush_async(*w.native);
  CHECK_FALSE(std::move(write).wait(10ms));
  CHECK_FALSE(std::move(flush).wait(10ms));
  CHECK(w.r.engine->gauges().storage_qd == 0);  // Input staging never holds a storage token.
  producer.release();
  CHECK(std::move(write).get(5s) == mib);
  CHECK_NOTHROW(std::move(flush).get(5s));
  CHECK(w.disk(mib) == w.r.image);
  context.shutdown();
}

TEST_CASE("opt-in context buffered fallbacks serialize against engine granules and share flush",
          "[cufile][engine][write][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  write_fixture w(2 * mib);
  std::shared_ptr<cufile_io_object> other = cufile_reactor::create_io_object_for_write(
    w.path, {write_mode::open_existing}, true, compat::on);
  transport_simulator::fault f;
  f.index            = 0;
  f.stuck_completion = true;
  w.r.transport->add_fault(f);
  w.r.start();
  auto body = w.write({0, mib});
  REQUIRE(until([&] { return w.r.transport->snapshot().blocked_completions == 1; }));
  config cfg;
  cfg.async_engine.mode = engine_mode::async_auto;
  cfg.async_engine.async_writes = true;
  auto shared = std::make_shared<cufile_reactor::reactor_context>(cfg);
  cufile_ioctx legacy(1, shared);
  legacy.start();
  std::vector<prepared_io_slice> edges;
  edges.push_back(make_host_write_slice(
    {37, 100}, reinterpret_cast<std::uint8_t const*>(w.r.image.data() + mib)));
  auto edge  = legacy.mixed_writev_async_io(*other, std::move(edges));
  auto flush = legacy.flush_async(*other);
  CHECK_FALSE(std::move(edge).wait(10ms));
  CHECK_FALSE(std::move(flush).wait(1ms));
  w.r.transport->unblock_all();
  CHECK(std::move(body).get(5s) == mib);
  CHECK(std::move(edge).get(5s) == 100);
  CHECK_NOTHROW(std::move(flush).get(5s));
  auto expected = std::vector<std::byte>(w.r.image.begin(), w.r.image.begin() + mib);
  std::copy_n(w.r.image.begin() + mib, 100, expected.begin() + 37);
  CHECK(w.disk(mib) == expected);
  legacy.shutdown();
}

TEST_CASE("write registry evicts inodes and legacy peers never conflict", "[cufile][engine][write]")
{
  auto const baseline = file_write_state::registry_size_for_testing();
  for (int i = 0; i < 128; ++i) {
    char path[] = "/tmp/cucs_write_registry_XXXXXX";
    file_descriptor fd{mkstemp(path)};
    REQUIRE(fd.get() >= 0);
    unlink(path);
    auto state = file_write_state::get(fd.get());
    CHECK(file_write_state::get(fd.get()) == state);
    auto ticket = state->begin();
    std::byte byte{0x5a};
    REQUIRE(pwrite(fd.get(), &byte, 1, 0) == 1);
    std::vector<std::shared_ptr<file_write_state::interval>> peers(4);
    std::vector<std::jthread> threads;
    for (std::size_t j = 0; j < peers.size(); ++j)
      threads.emplace_back([&, j] {
        peers[j] =
          state->try_acquire({37 + j * 100, 100}, write_granule(), write_io_mode::mixed, {}, true);
      });
    threads.clear();
    CHECK(std::all_of(peers.begin(), peers.end(), [](auto const& p) { return bool(p); }));
    CHECK(state->conflicts() == 0);
    state->activate();
    std::atomic<bool> notified{};
    CHECK_FALSE(state->try_acquire(
      {0, write_granule()}, write_granule(), write_io_mode::direct, [&] { notified = true; }));
    peers.clear();
    CHECK(notified.load());
    CHECK(state->try_acquire({0, write_granule()}, write_granule(), write_io_mode::direct));
  }
  CHECK(file_write_state::registry_size_for_testing() == baseline);
}

TEST_CASE("default legacy writes across four reactors do not serialize shared granules",
          "[cufile][engine][write][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  constexpr auto piece = mib + 37;
  write_fixture w(4 * piece);
  auto shared = std::make_shared<cufile_reactor::reactor_context>(config{});
  cufile_ioctx legacy(4, shared);
  legacy.start();
  std::vector<prepared_io_slice> slices;
  for (std::size_t i = 0; i < 4; ++i)
    slices.push_back(
      make_device_write_slice({i * piece, piece},
                              static_cast<std::uint8_t*>(w.r.destination) + i * piece,
                              ::cuda::stream_ref{w.r.stream},
                              0));
  CHECK(legacy.mixed_writev_async_io(*w.native, std::move(slices)).get(5s) == 4 * piece);
  CHECK(w.disk(4 * piece) == w.r.image);
  CHECK_FALSE(w.state->has_engine_work());
  CHECK(w.state->conflicts() == 0);
  legacy.shutdown();
}

TEST_CASE("engine negative write completion revokes even an operator override",
          "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const result = GENERATE(-EOPNOTSUPP, -ENOSPC, -EACCES, -EDQUOT);
  write_fixture w(mib);
  transport_simulator::fault f;
  f.index             = 0;
  f.completion_result = result;
  w.r.transport->add_fault(f);
  w.r.start();
  CHECK_THROWS(w.write({0, mib}).get(5s));
  REQUIRE(w.r.engine->drain(5s));
  CHECK(w.r.policy->write_eligible());  // File refusal no longer revokes unrelated mount peers.
  CHECK(w.native->async_policy()->write_demoted.load() == (result == -EOPNOTSUPP));
  CHECK(w.write({0, mib}).get(5s) == mib);
  REQUIRE(w.r.engine->drain(5s));
  CHECK(w.disk(mib) == w.r.image);
  if (result == -EOPNOTSUPP) CHECK(w.r.transport->snapshot().write_calls == 1);
  // A transient error retains eligibility, but a temporary zero capacity grant
  // may still send the next piece to the engine's safe blocking fallback.
  auto gauges = w.r.engine->gauges();
  CHECK(gauges.async_write_bytes + gauges.direct_write_bytes + gauges.buffered_write_bytes == mib);
  CHECK(w.r.engine->gauges().pinned_write_scratch_bytes == write_granule() * w.r.lanes->capacity());
}

TEST_CASE("opt-in context host and legacy device fallbacks wait for engine granules",
          "[cufile][engine][write][ioctx]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const device = GENERATE(false, true);
  auto const count  = device ? mib + 100 : 100UL;
  write_fixture w(3 * mib);
  REQUIRE(pwrite(w.native->buffered_fd(), w.r.image.data() + 2 * mib, 100, 2 * mib) == 100);
  w.native->note_written(2 * mib + 100);
  transport_simulator::fault f;
  f.index            = 0;
  f.stuck_completion = true;
  w.r.transport->add_fault(f);
  w.r.start();
  auto body = w.write({0, mib});
  REQUIRE(until([&] { return w.r.transport->snapshot().blocked_completions == 1; }));
  config cfg;
  cfg.max_inflight = 1;
  cfg.async_engine.mode = engine_mode::async_auto;
  cfg.async_engine.async_writes = true;
  auto shared      = std::make_shared<cufile_reactor::reactor_context>(cfg);
  cufile_ioctx legacy(1, shared);
  legacy.start();
  auto pending = std::async(std::launch::async, [&] {
    if (!device)
      return legacy.host_write(
        *w.native, 37, 100, reinterpret_cast<std::uint8_t const*>(w.r.image.data() + mib));
    std::vector<prepared_io_slice> slices;
    slices.push_back(make_device_write_slice({37, count},
                                             static_cast<std::uint8_t*>(w.r.destination) + mib,
                                             ::cuda::stream_ref{w.r.stream},
                                             0));
    return legacy.mixed_writev_async_io(*w.native, std::move(slices)).get(5s);
  });
  REQUIRE(until([&] { return w.state->conflicts() > 0; }));
  CHECK(pending.wait_for(10ms) == std::future_status::timeout);
  if (device) {
    // One reactor and max_inflight=1: a parked mixed legacy slice must not
    // prevent an unrelated read from using that reactor's dispatch window.
    std::vector<std::byte> readback(100);
    CHECK(legacy
            .host_read_async_io(
              *w.native, 2 * mib, readback.size(), reinterpret_cast<std::uint8_t*>(readback.data()))
            .get(5s) == readback.size());
    CHECK(std::equal(readback.begin(), readback.end(), w.r.image.begin() + 2 * mib));
  }
  w.r.transport->unblock_all();
  CHECK(std::move(body).get(5s) == mib);
  CHECK(pending.get() == count);
  auto expected = std::vector<std::byte>(w.r.image.begin(), w.r.image.begin() + mib);
  expected.resize(std::max(mib, 37 + count));
  std::copy_n(w.r.image.begin() + mib, count, expected.begin() + 37);
  CHECK(w.disk(expected.size()) == expected);
  legacy.shutdown();
}

TEST_CASE("engine body waits behind a preactivation synchronous legacy guard",
          "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  write_fixture w(mib);
  auto guard = w.state->acquire_legacy({37, 100}, write_granule(), write_io_mode::buffered);
  w.r.start();
  auto future = w.write({0, mib});
  REQUIRE(until([&] { return w.r.engine->gauges().granule_conflicts != 0; }));
  CHECK_FALSE(std::move(future).wait(10ms));
  CHECK(w.r.transport->snapshot().write_calls == 0);
  guard.reset();
  CHECK(std::move(future).get(5s) == mib);
  CHECK(w.disk(mib) == w.r.image);
}

TEST_CASE("engine aligned short write fallback uses only the staged suffix",
          "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  write_fixture w(mib);
  transport_simulator::fault f;
  f.index             = 0;
  f.completion_result = 4096;
  f.stuck_completion  = true;
  w.r.transport->add_fault(f);
  w.r.start();
  auto future = w.write({0, mib});
  REQUIRE(until([&] { return w.r.transport->snapshot().blocked_completions == 1; }));
  w.r.policy->open_circuit();
  w.r.engine->notify();
  w.r.transport->unblock_all();
  CHECK(std::move(future).get(5s) == mib);
  REQUIRE(w.r.engine->drain(5s));
  CHECK(w.r.transport->snapshot().write_calls == 1);
  CHECK(w.r.engine->gauges().direct_write_bytes == mib - 4096);
  CHECK(w.disk(mib) == w.r.image);
}

TEST_CASE("engine not-submitted after a short write preserves the known prefix",
          "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  write_fixture w(2 * mib);
  w.r.config.demotion.stuck_ms = 30000;
  w.r.policy                   = w.r.make_policy(1);
  transport_simulator::fault short_write;
  short_write.index             = 0;
  short_write.completion_result = 4096;
  short_write.stuck_completion  = true;
  w.r.transport->add_fault(short_write);
  transport_simulator::fault blocked;
  blocked.index        = 1;
  blocked.stuck_submit = true;
  w.r.transport->add_fault(blocked);
  w.r.start();
  auto first = w.write({0, mib});
  REQUIRE(until([&] { return w.r.transport->snapshot().blocked_completions == 1; }));
  auto group = std::make_shared<grouped_coordinator>(mib, 1);
  auto other = group->get_future();
  auto slice = w.slice(group, {mib, mib}, mib);
  w.r.engine->hand_off(std::move(slice));
  REQUIRE(until([&] { return w.r.transport->snapshot().blocked_submissions == 1; }));
  REQUIRE(w.r.transport->unblock(0));
  REQUIRE(until([&] {
    auto g = w.r.engine->gauges();
    return g.short_writes == 1 && g.storage_qd == 2;
  }));
  w.r.policy->open_circuit();
  w.r.engine->notify();
  CHECK(std::move(first).get(5s) == mib);
  CHECK(w.r.engine->gauges().not_submitted == 1);
  w.r.transport->unblock_all();
  CHECK(std::move(other).get(5s) == mib);
  REQUIRE(w.r.engine->drain(5s));
  CHECK(w.r.engine->gauges().direct_write_bytes == mib - 4096);
  CHECK(w.disk(2 * mib) == w.r.image);
}

TEST_CASE("engine known direct progress continues buffered after EINVAL", "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const prefix = GENERATE(false, true);
  write_fixture w(mib);
  w.r.policy->open_circuit();
  unsigned calls{};
  w.r.start([&](auto& opts) {
    opts.write_at = [&](int fd, void const* ptr, std::size_t size, off_t offset) -> ssize_t {
      if (++calls == (prefix ? 2U : 1U)) {
        errno = EINVAL;
        return -1;
      }
      return pwrite(fd, ptr, size, offset);
    };
  });
  CHECK(w.write({0, mib}).get(5s) == mib);
  REQUIRE(w.r.engine->drain(5s));
  CHECK(w.r.engine->gauges().direct_write_bytes == (prefix ? write_granule() : 0));
  CHECK(w.r.engine->gauges().buffered_write_bytes == mib - (prefix ? write_granule() : 0));
  CHECK(w.disk(mib) == w.r.image);
}

TEST_CASE("engine quarantined write retains interval and flush sequence without polling",
          "[cufile][engine][write]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  // Intentional process-lifetime retention, as in the read quarantine test.
  auto* w = new write_fixture(mib);
  transport_simulator::fault f;
  f.index               = 0;
  f.preserve_bytes_done = true;
  w->r.transport->add_fault(f);
  w->r.start([](auto& opts) {
    opts.synchronize_lane = [](cudaStream_t stream) {
      static_cast<void>(cudaStreamSynchronize(stream));
      return cudaErrorUnknown;
    };
  });
  auto future = w->write({0, mib});
  REQUIRE(until([&] { return w->r.engine->gauges().quarantines == 1; }));
  auto flush    = device_engine::flush_async(w->native, w->state);
  auto conflict = w->write({37, 100});
  REQUIRE(until([&] { return w->r.engine->gauges().granule_conflicts != 0; }));
  CHECK_FALSE(std::move(flush).wait(5ms));
  CHECK_FALSE(std::move(conflict).wait(5ms));
  CHECK(w->state->quarantined_intervals() == 1);
  CHECK(w->r.engine->gauges().quarantined_write_intervals == 1);
  auto iterations = w->r.engine->gauges().reaper_iterations;
  std::this_thread::sleep_for(20ms);
  CHECK(w->r.engine->gauges().reaper_iterations - iterations < 30);
  w->r.engine->request_stop();
  CHECK_FALSE(w->r.engine->drain(5ms));
  CHECK_FALSE(future.is_ready());
  unlink(w->path.c_str());
}

TEST_CASE("engine real compat libcufile writes roundtrip slot registrations NOT GDS",
          "[cufile][engine][write][integration]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto report = runtime::instance().ensure_initialized();
  if (!report.driver_opened || !report.optional_groups.async_write) SKIP("async writes absent");
  write_fixture w(4 * mib + 77);
  if (!w.native->cufile_registered()) SKIP("file registration unavailable");
  w.r.arena->release_for_maintenance();
  w.r.arena = std::make_unique<device_arena>(0);
  if (w.r.arena->warm_up(w.r.config).registered_bytes == 0) SKIP("buffer registration unavailable");
  w.r.lanes = std::make_shared<lane_pool>(0, w.r.config);
  w.r.start([](auto& opts) { opts.transport = make_cufile_transport(); });
  CHECK(w.write({37, 4 * mib}).get(5s) == 4 * mib);
  auto disk = w.disk(4 * mib + 37);
  CHECK(std::equal(disk.begin() + 37, disk.end(), w.r.image.begin()));
}
