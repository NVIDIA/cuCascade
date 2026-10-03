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

#include "io/stub_reactor.hpp"

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/types.hpp>

#include <catch2/catch_all.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <stop_token>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;
using cucascade::io::request_class;
using cucascade::io::request_class_index;
using cucascade::test::stub::dispatch_controls;
using cucascade::test::stub::stub_io_object;
using cucascade::test::stub::stub_ioctx;

std::shared_ptr<stub_io_object> make_object(std::shared_ptr<dispatch_controls> controls = nullptr,
                                            std::size_t size                            = 1UL << 20)
{
  if (controls == nullptr) controls = std::make_shared<dispatch_controls>();
  return std::make_shared<stub_io_object>(std::move(controls), "stub://runner", size);
}

/// @p n host slices of @p bytes each.
std::vector<cucascade::io::prepared_io_slice> slices_of(std::size_t n, std::size_t bytes = 64)
{
  static std::uint8_t sink{};
  std::vector<cucascade::io::prepared_io_slice> slices;
  for (std::size_t i = 0; i < n; ++i) {
    slices.emplace_back(cucascade::io::range{i * bytes, bytes}, cucascade::io::host_buffer{&sink});
  }
  return slices;
}

template <class Predicate>
bool wait_for(Predicate&& predicate, std::chrono::milliseconds timeout = 5000ms)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(1ms);
  }
  return true;
}

bool is_canceled(std::system_error const& error)
{
  return error.code() == std::errc::operation_canceled;
}

}  // namespace

TEST_CASE("run_for without work returns after the deadline", "[io][runner]")
{
  stub_ioctx ioctx(0);
  auto const start = std::chrono::steady_clock::now();
  CHECK(ioctx.run_for(50ms) == 0);
  auto const elapsed = std::chrono::steady_clock::now() - start;
  CHECK(elapsed >= 50ms);
  CHECK(elapsed < 2s);
  CHECK(ioctx.active_runners() == 0);
}

TEST_CASE("run returns immediately for a stopped token", "[io][runner]")
{
  stub_ioctx ioctx(0);
  std::stop_source source;
  source.request_stop();
  CHECK(ioctx.run(source.get_token()) == 0);
}

TEST_CASE("an external runner serves requests", "[io][runner]")
{
  stub_ioctx ioctx(0);
  auto object = make_object();
  std::atomic<std::size_t> served{0};
  std::jthread runner([&](std::stop_token stop) { served.store(ioctx.run(stop)); });
  REQUIRE(wait_for([&] { return ioctx.active_runners() == 1; }));

  std::vector<cucascade::exec::semi_future<std::size_t>> futures;
  for (int i = 0; i < 64; ++i) {
    futures.push_back(ioctx.mixed_readv_async_io(*object, slices_of(3)));
  }
  for (auto& future : futures) {
    CHECK(std::move(future).get() == 3 * 64);
  }
  CHECK(wait_for([&] { return ioctx.stats().idle_runners == 1; }));
  CHECK(ioctx.stats().in_flight_requests == 0);

  runner.request_stop();
  runner.join();
  CHECK(served.load() == 64);
  CHECK(ioctx.active_runners() == 0);
}

TEST_CASE("two external runners share the work", "[io][runner]")
{
  stub_ioctx ioctx(0);
  auto controls     = std::make_shared<dispatch_controls>();
  controls->on_unit = [] { std::this_thread::sleep_for(200us); };
  auto object       = make_object(controls);
  std::atomic<std::size_t> served_a{0};
  std::atomic<std::size_t> served_b{0};
  std::jthread a([&](std::stop_token stop) { served_a.store(ioctx.run(stop)); });
  std::jthread b([&](std::stop_token stop) { served_b.store(ioctx.run(stop)); });
  REQUIRE(wait_for([&] { return ioctx.stats().idle_runners == 2; }));

  std::vector<cucascade::exec::semi_future<std::size_t>> futures;
  for (int i = 0; i < 64; ++i) {
    futures.push_back(ioctx.mixed_readv_async_io(*object, slices_of(4)));
  }
  for (auto& future : futures) {
    CHECK(std::move(future).get() == 4 * 64);
  }
  a.request_stop();
  b.request_stop();
  a.join();
  b.join();
  CHECK(served_a.load() + served_b.load() == 64);
  CHECK(served_a.load() > 0);
  CHECK(served_b.load() > 0);
}

TEST_CASE("large reads fan out into several groups", "[io][runner]")
{
  stub_ioctx ioctx(0);
  ioctx.forced_fanout = 3;
  auto object         = make_object();
  std::atomic<std::size_t> served{0};
  std::jthread runner([&](std::stop_token stop) { served.store(ioctx.run(stop)); });
  REQUIRE(wait_for([&] { return ioctx.active_runners() == 1; }));

  CHECK(ioctx.mixed_readv_async_io(*object, slices_of(5)).get() == 5 * 64);
  runner.request_stop();
  runner.join();
  CHECK(served.load() == 3);
}

TEST_CASE("start, shutdown and start again", "[io][runner]")
{
  stub_ioctx ioctx(2);
  auto object = make_object();

  ioctx.start();
  ioctx.start();  // idempotent
  CHECK(wait_for([&] { return ioctx.active_runners() == 2; }));
  CHECK(ioctx.mixed_readv_async_io(*object, slices_of(2)).get() == 128);

  ioctx.shutdown();
  CHECK(ioctx.active_runners() == 0);
  CHECK_THROWS_MATCHES(ioctx.mixed_readv_async_io(*object, slices_of(1)).get(),
                       std::system_error,
                       Catch::Matchers::Predicate<std::system_error>(is_canceled));

  ioctx.start();
  CHECK(ioctx.mixed_readv_async_io(*object, slices_of(2)).get() == 128);
  ioctx.shutdown();
  ioctx.shutdown();  // idempotent
}

TEST_CASE("shutdown cancels queued requests and waits for external runners", "[io][runner]")
{
  stub_ioctx ioctx(0);
  auto object = make_object();

  ioctx.start();  // no threads: only opens admission
  auto queued = ioctx.mixed_readv_async_io(*object, slices_of(1));
  CHECK(ioctx.stats().per_class[request_class_index(request_class::latency)].queued_requests == 1);

  std::atomic<bool> returned{false};
  std::jthread external([&] {
    static_cast<void>(ioctx.run(std::stop_token{}));  // stops only through shutdown()
    returned.store(true);
  });
  REQUIRE(wait_for([&] { return ioctx.active_runners() == 1; }));
  // The runner may already have served the queued request; either outcome is valid.
  ioctx.shutdown();  // returns only after the external runner unregistered
  CHECK(ioctx.active_runners() == 0);
  external.join();
  CHECK(returned.load());

  try {
    CHECK(std::move(queued).get() == 64);
  } catch (std::system_error const& error) {
    CHECK(is_canceled(error));
  }

  // run*() works again after shutdown completed.
  CHECK(ioctx.run_for(10ms) == 0);
}

TEST_CASE("shutdown without runners cancels queued requests", "[io][runner]")
{
  stub_ioctx ioctx(0);
  auto object = make_object();
  ioctx.start();
  auto future = ioctx.mixed_readv_async_io(*object, slices_of(1));
  ioctx.shutdown();
  CHECK_THROWS_MATCHES(std::move(future).get(),
                       std::system_error,
                       Catch::Matchers::Predicate<std::system_error>(is_canceled));
}

TEST_CASE("a retiring runner hands untaken work back", "[io][runner]")
{
  stub_ioctx ioctx(0);
  auto controls = std::make_shared<dispatch_controls>();
  std::stop_source first_runner;
  controls->on_unit = [&] { first_runner.request_stop(); };
  auto object       = make_object(controls);

  ioctx.start();  // admission only
  auto future = ioctx.mixed_readv_async_io(*object, slices_of(3));

  CHECK(ioctx.run(first_runner.get_token()) == 0);  // processed one slice, then retired
  CHECK(controls->units.load() == 1);
  auto const stats = ioctx.stats();
  CHECK(stats.per_class[request_class_index(request_class::latency)].queued_requests == 1);
  CHECK(stats.in_flight_requests == 0);

  controls->on_unit = nullptr;
  CHECK(ioctx.run_for(200ms) == 1);
  CHECK(std::move(future).get() == 3 * 64);
  CHECK(controls->units.load() == 3);
}

TEST_CASE("nested run on a runner thread throws logic_error", "[io][runner]")
{
  stub_ioctx ioctx(0);
  auto controls = std::make_shared<dispatch_controls>();
  std::atomic<bool> threw_logic_error{false};
  controls->on_unit = [&] {
    try {
      static_cast<void>(ioctx.run_for(1ms));
    } catch (std::logic_error const&) {
      threw_logic_error.store(true);
    }
  };
  auto object = make_object(controls);
  std::jthread runner([&](std::stop_token stop) { static_cast<void>(ioctx.run(stop)); });
  REQUIRE(wait_for([&] { return ioctx.active_runners() == 1; }));
  CHECK(ioctx.mixed_readv_async_io(*object, slices_of(1)).get() == 64);
  CHECK(threw_logic_error.load());
}

TEST_CASE("start reports engine construction failures", "[io][runner]")
{
  stub_ioctx ioctx(2);
  ioctx.reactor().fail_make_engine.store(true);
  CHECK_THROWS_WITH(ioctx.start(), Catch::Matchers::ContainsSubstring("make_engine failure"));
  CHECK(ioctx.active_runners() == 0);
  CHECK_FALSE(ioctx.reactor().hub().accepting());

  ioctx.reactor().fail_make_engine.store(false);
  ioctx.start();
  auto object = make_object();
  CHECK(ioctx.mixed_readv_async_io(*object, slices_of(1)).get() == 64);
}

TEST_CASE("failed I/O fails only the affected request", "[io][runner]")
{
  stub_ioctx ioctx(1);
  ioctx.start();
  auto failing     = std::make_shared<dispatch_controls>();
  failing->fail_io = true;
  auto bad_object  = make_object(failing);
  auto good_object = make_object();
  auto bad         = ioctx.mixed_readv_async_io(*bad_object, slices_of(4));
  auto good        = ioctx.mixed_readv_async_io(*good_object, slices_of(4));
  CHECK_THROWS_AS(std::move(bad).get(), std::system_error);
  CHECK(std::move(good).get() == 4 * 64);
  // The failing group stopped after its first error.
  CHECK(failing->units.load() == 1);
}

TEST_CASE("writes, flushes and commits are published as grouped requests", "[io][runner]")
{
  stub_ioctx ioctx(1);
  CHECK(ioctx.supports_write());
  CHECK(ioctx.supports_device_write());
  ioctx.start();

  auto object = ioctx.open_io_object_for_write("stub://written");
  REQUIRE(object != nullptr);
  std::vector<std::uint8_t> data(1000, 7);
  CHECK(ioctx.host_write(*object, 0, data.size(), data.data()) == data.size());
  CHECK(ioctx.host_write_async(*object, 0, data.size(), data.data()).get() == data.size());

  std::vector<cucascade::io::write_segment> segments;
  segments.push_back({cucascade::io::range{0, 100}, cucascade::io::host_source{data.data()}});
  segments.push_back({cucascade::io::range{200, 300}, cucascade::io::host_source{data.data()}});
  CHECK(ioctx.writev_async(*object, std::move(segments)).get() == 400);

  CHECK_NOTHROW(ioctx.flush_async(*object).get());
  CHECK_NOTHROW(ioctx.commit_async(*object, cucascade::io::write_durability::data_sync).get());
  auto const write_lane = request_class_index(request_class::write);
  CHECK(ioctx.stats().per_class[write_lane].queued_requests == 0);
}

namespace {

/// An object whose first unit blocks its runner until released, then throws
/// out of the engine (a fatal engine error).
struct fatal_gate {
  std::atomic<bool> entered{false};
  std::atomic<bool> release{false};
  std::shared_ptr<dispatch_controls> controls = std::make_shared<dispatch_controls>();

  fatal_gate()
  {
    controls->on_unit = [this] {
      entered.store(true);
      while (!release.load()) {
        std::this_thread::sleep_for(1ms);
      }
      throw std::runtime_error("fatal engine boom");
    };
  }
};

bool is_boom(std::runtime_error const& error)
{
  return std::string_view(error.what()).find("fatal engine boom") != std::string_view::npos;
}

}  // namespace

TEST_CASE("queued requests fail when every started runner dies", "[io][runner]")
{
  stub_ioctx ioctx(1);
  ioctx.start();
  fatal_gate gate;
  auto fatal_object = make_object(gate.controls);
  auto good_object  = make_object();

  auto doomed = ioctx.mixed_readv_async_io(*fatal_object, slices_of(1));
  REQUIRE(wait_for([&] { return gate.entered.load(); }));
  auto queued = ioctx.mixed_readv_async_io(*good_object, slices_of(1));
  CHECK(ioctx.stats().per_class[request_class_index(request_class::latency)].queued_requests == 1);

  gate.release.store(true);
  auto const boom = Catch::Matchers::Predicate<std::runtime_error>(is_boom);
  CHECK_THROWS_MATCHES(std::move(doomed).get(), std::runtime_error, boom);
  CHECK_THROWS_MATCHES(std::move(queued).get(), std::runtime_error, boom);
  CHECK(ioctx.active_runners() == 0);
  CHECK_FALSE(ioctx.reactor().hub().accepting());

  // New submissions fail fast (no runner will ever serve them) ...
  CHECK_THROWS_MATCHES(
    ioctx.mixed_readv_async_io(*good_object, slices_of(1)).get(), std::runtime_error, boom);

  // ... until start() replaces the dead pool.
  ioctx.start();
  CHECK(ioctx.active_runners() == 1);
  CHECK(ioctx.mixed_readv_async_io(*good_object, slices_of(2)).get() == 128);

  // shutdown() resets the state as well.
  gate.entered.store(false);
  auto doomed_again = ioctx.mixed_readv_async_io(*fatal_object, slices_of(1));
  REQUIRE(wait_for([&] { return gate.entered.load(); }));
  CHECK_THROWS_MATCHES(std::move(doomed_again).get(), std::runtime_error, boom);
  REQUIRE(wait_for([&] { return !ioctx.reactor().hub().accepting(); }));
  ioctx.shutdown();
  CHECK_THROWS_MATCHES(ioctx.mixed_readv_async_io(*good_object, slices_of(1)).get(),
                       std::system_error,
                       Catch::Matchers::Predicate<std::system_error>(is_canceled));
  ioctx.start();
  CHECK(ioctx.mixed_readv_async_io(*good_object, slices_of(1)).get() == 64);
}

TEST_CASE("a surviving runner keeps serving after another runner dies", "[io][runner]")
{
  stub_ioctx ioctx(2);
  ioctx.start();
  fatal_gate gate;
  auto fatal_object = make_object(gate.controls);
  auto good_object  = make_object();

  auto doomed = ioctx.mixed_readv_async_io(*fatal_object, slices_of(1));
  REQUIRE(wait_for([&] { return gate.entered.load(); }));
  CHECK(ioctx.mixed_readv_async_io(*good_object, slices_of(1)).get() == 64);
  gate.release.store(true);
  CHECK_THROWS_AS(std::move(doomed).get(), std::runtime_error);
  REQUIRE(wait_for([&] { return ioctx.active_runners() == 1; }));
  CHECK(ioctx.reactor().hub().accepting());
  CHECK(ioctx.mixed_readv_async_io(*good_object, slices_of(3)).get() == 3 * 64);
}

TEST_CASE("the last external runner dying fails the queue of a started context", "[io][runner]")
{
  stub_ioctx ioctx(0);
  ioctx.start();  // admission only
  fatal_gate gate;
  auto fatal_object = make_object(gate.controls);
  auto good_object  = make_object();

  auto doomed = ioctx.mixed_readv_async_io(*fatal_object, slices_of(1));
  std::atomic<bool> run_threw{false};
  std::jthread external([&] {
    try {
      static_cast<void>(ioctx.run(std::stop_token{}));
    } catch (std::runtime_error const& error) {
      run_threw.store(is_boom(error));
    }
  });
  REQUIRE(wait_for([&] { return gate.entered.load(); }));
  auto queued = ioctx.mixed_readv_async_io(*good_object, slices_of(1));
  gate.release.store(true);
  external.join();
  CHECK(run_threw.load());
  auto const boom = Catch::Matchers::Predicate<std::runtime_error>(is_boom);
  CHECK_THROWS_MATCHES(std::move(doomed).get(), std::runtime_error, boom);
  CHECK_THROWS_MATCHES(std::move(queued).get(), std::runtime_error, boom);

  // A new external runner reopens admission and serves new work.
  auto later = ioctx.mixed_readv_async_io(*good_object, slices_of(1));
  CHECK_THROWS_AS(std::move(later).get(), std::runtime_error);
  std::stop_source stop;
  std::jthread revived([&] { static_cast<void>(ioctx.run(stop.get_token())); });
  REQUIRE(wait_for([&] { return ioctx.reactor().hub().accepting(); }));
  CHECK(ioctx.mixed_readv_async_io(*good_object, slices_of(1)).get() == 64);
  stop.request_stop();
}
