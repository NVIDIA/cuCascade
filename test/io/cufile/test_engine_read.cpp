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

#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/runtime.hpp>

#include <unistd.h>

using namespace cucascade::io;
using namespace cucascade::io::cufile;
using namespace cucascade::io::cufile::engine_test;

TEST_CASE("engine registry lookup never enables the default legacy mode", "[cufile][engine][read]")
{
  CHECK(device_engine::try_get(-1) == nullptr);
  CHECK_THROWS_AS(device_engine::for_device(-1), std::invalid_argument);
}

TEST_CASE("engine staged read joins exact aligned and unaligned slices", "[cufile][engine][read]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto const offset = GENERATE(std::size_t{0}, std::size_t{37});
  fixture f(3 * mib + 101);
  f.start();
  auto const count = f.bytes - offset;
  auto group       = std::make_shared<grouped_coordinator>(count, 1);
  auto future      = group->get_future();
  f.engine->hand_off(f.slice(group, {offset, count}));
  REQUIRE(std::move(future).get(5s) == count);
  auto result = f.contents(count);
  CHECK(std::equal(
    result.begin(), result.end(), f.image.begin() + static_cast<std::ptrdiff_t>(offset)));
  REQUIRE(f.engine->drain(5s));
  CHECK(group->tasks_remaining() == 0);
  CHECK(f.transport->snapshot().extent_errors == 0);
  CHECK(f.transport->snapshot().argument_changes == 0);
  CHECK(f.transport->snapshot().bounce == 0);
  for (auto const& call : f.transport->submissions()) {
    CHECK(call.buffer_offset == 0);  // AM-9: separately registered slot base, NOT arena+offset.
    CHECK(call.size <= mib);
    CHECK(call.thread != std::this_thread::get_id());
  }
}

TEST_CASE("engine AM1 future precedes execution and AM2 fence never gates storage",
          "[cufile][engine][read]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  std::atomic<bool> retired{false};
  fixture f(mib);
  f.start();
  gate input(f.stream);
  auto group             = std::make_shared<grouped_coordinator>(f.bytes, 1);
  auto future            = group->get_future();
  auto slice             = f.slice(group, {0, f.bytes});
  slice.retirement_owner = std::shared_ptr<void>(new int(0), [&](void* ptr) {
    delete static_cast<int*>(ptr);
    retired.store(true);
  });
  f.engine->hand_off(std::move(slice));
  REQUIRE(until([&] { return f.transport->snapshot().completed == 1; }));
  REQUIRE(std::move(future).wait(5s));  // The fence/copies have NOT executed.
  CHECK(std::move(future).get() == f.bytes);
  CHECK(until([&] { return f.engine->gauges().live_operations > 0; }));
  CHECK(f.arena->gauges().live_leases > 0);
  CHECK(cudaStreamQuery(f.stream) == cudaErrorNotReady);
  CHECK_FALSE(retired.load());
  input.release();
  CHECK(f.contents() == f.image);
  CHECK(f.engine->drain(5s));
  CHECK(f.arena->gauges().live_leases == 0);
  CHECK(retired.load());
}

TEST_CASE("engine PTDS completion is host complete", "[cufile][engine][read]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f;
  f.start();
  gate input(f.stream);
  auto group  = std::make_shared<grouped_coordinator>(f.bytes, 1);
  auto future = group->get_future();
  f.engine->hand_off(f.slice(group, {0, f.bytes}, 0, true));
  REQUIRE(until([&] { return f.transport->snapshot().completed == 1; }));
  CHECK_FALSE(std::move(future).wait(10ms));
  CHECK(f.transport->snapshot().wait_calls == 1);  // Fence only; no invalid PTDS stand-in joins.
  input.release();
  CHECK(std::move(future).get(5s) == f.bytes);
  std::vector<std::byte> result(f.bytes);
  REQUIRE(cudaMemcpy(result.data(), f.destination, f.bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
  CHECK(result == f.image);
}

TEST_CASE("engine several slices share one fan in", "[cufile][engine][read]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(8 * mib);
  f.start();
  auto group  = std::make_shared<grouped_coordinator>(f.bytes, 8);
  auto future = group->get_future();
  // Prepare all fences before the first possible join onto this stream.
  std::vector<engine_slice> slices;
  for (std::size_t i = 0; i < 8; ++i)
    slices.push_back(f.slice(group, {i * mib, mib}, i * mib));
  for (auto& slice : slices)
    f.engine->hand_off(std::move(slice));
  CHECK(std::move(future).get(5s) == f.bytes);
  CHECK(f.contents() == f.image);
  CHECK(group->tasks_remaining() == 0);
}

TEST_CASE("engine handoff is MPSC across concurrent requests", "[cufile][engine][read]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  fixture f(16 * mib, configuration(8));
  f.start();
  std::vector<std::shared_ptr<grouped_coordinator>> groups;
  std::vector<cucascade::exec::semi_future<std::size_t>> futures;
  std::vector<engine_slice> slices;
  for (std::size_t i = 0; i < 16; ++i) {
    groups.push_back(std::make_shared<grouped_coordinator>(mib, 1));
    futures.push_back(groups.back()->get_future());
    slices.push_back(f.slice(groups.back(), {i * mib, mib}, i * mib));
  }
  std::vector<std::thread> callers;
  for (std::size_t t = 0; t < 4; ++t)
    callers.emplace_back([&, t] {
      for (std::size_t i = t; i < slices.size(); i += 4)
        f.engine->hand_off(std::move(slices[i]));
    });
  for (auto& caller : callers)
    caller.join();
  for (auto& future : futures)
    CHECK(std::move(future).get(5s) == mib);
  CHECK(f.contents() == f.image);
  CHECK(f.engine->drain(5s));
  for (auto const& group : groups)
    CHECK(group->tasks_remaining() == 0);
}

TEST_CASE("engine real libcufile compat reads use separately registered slots",
          "[cufile][engine][read][integration]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto& runtime = cucascade::io::cufile::runtime::instance();
  auto report   = runtime.ensure_initialized();
  if (!report.driver_opened || !report.optional_groups.async_read) SKIP("async libcufile absent");
  auto config                          = configuration(4);
  config.probe.require_direct_evidence = false;
  fixture f(8 * mib + 37, config);
  char name[] = "/tmp/cucascade_engine_XXXXXX";
  file_descriptor fd{mkstemp(name)};
  REQUIRE(fd.get() >= 0);
  struct remove_file {
    char const* path;
    ~remove_file() { static_cast<void>(unlink(path)); }
  } remove{name};
  std::size_t written{};
  while (written < f.image.size()) {
    auto const count = pwrite(
      fd.get(), f.image.data() + written, f.image.size() - written, static_cast<off_t>(written));
    REQUIRE(count > 0);
    written += static_cast<std::size_t>(count);
  }
  auto file = cufile_reactor::create_io_object(name, true, compat::automatic);
  if (!file->cufile_registered()) SKIP("libcufile cannot register this test filesystem");
  f.object = std::move(file);
  f.arena->release_for_maintenance();
  f.arena         = std::make_unique<device_arena>(0);
  auto const warm = f.arena->warm_up(config);  // Default AM-9 slot registration.
  if (warm.registered_bytes == 0) SKIP("cuFile buffer registration unavailable");
  f.lanes = std::make_shared<lane_pool>(0, config);
  f.start([](auto& opts) { opts.transport = make_cufile_transport(); });
  for (unsigned round = 0; round < 3; ++round) {
    auto group  = std::make_shared<grouped_coordinator>(f.bytes, 1);
    auto future = group->get_future();
    f.engine->hand_off(f.slice(group, {0, f.bytes}));
    REQUIRE(std::move(future).get(5s) == f.bytes);
    CHECK(std::ranges::equal(f.contents(), f.image));
    REQUIRE(f.engine->drain(5s));
  }
}

TEST_CASE("engine default slot geometry caps filesystem preferred pieces", "[cufile][engine][read]")
{
  if (!has_device()) SKIP("CUDA unavailable");
  auto config                        = configuration();
  config.mount_overrides[0].op_bytes = 4 * mib;
  fixture f(8 * mib, config);
  f.start();
  auto group  = std::make_shared<grouped_coordinator>(f.bytes, 1);
  auto future = group->get_future();
  f.engine->hand_off(f.slice(group, {0, f.bytes}));
  CHECK(std::move(future).get(5s) == f.bytes);
  CHECK(f.contents() == f.image);
  auto const calls = f.transport->submissions();
  REQUIRE(calls.size() == 8);
  for (auto const& call : calls)
    CHECK(call.size == mib);
}
