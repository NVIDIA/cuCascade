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

#include <cucascade/io/details/request_hub.hpp>
#include <cucascade/io/details/runner_registry.hpp>
#include <cucascade/io/io_request.hpp>

#include <catch2/catch_all.hpp>
#include <poll.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <random>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {

using cucascade::io::grouped_io_request;
using cucascade::io::request_class;
using cucascade::io::detail::request_hub;
using cucascade::io::detail::runner_registry;
using cucascade::io::detail::runner_slot;

bool readable(int fd, int timeout_ms = 0)
{
  pollfd entry{fd, POLLIN, 0};
  return ::poll(&entry, 1, timeout_ms) == 1 && (entry.revents & POLLIN) != 0;
}

}  // namespace

TEST_CASE("runner_registry registers one slot per thread", "[io][runner_registry]")
{
  runner_registry registry;
  auto slot = registry.register_runner();
  REQUIRE(slot != nullptr);
  CHECK(slot->id() > 0);
  CHECK(slot->thread_id() == std::this_thread::get_id());
  CHECK(slot->wake_fd() >= 0);
  CHECK(registry.size() == 1);
  CHECK(registry.is_registered(std::this_thread::get_id()));
  CHECK_THROWS_AS(registry.register_runner(), std::logic_error);

  std::shared_ptr<runner_slot> other;
  std::jthread([&] { other = registry.register_runner(); }).join();
  REQUIRE(other != nullptr);
  CHECK(other->id() != slot->id());
  CHECK(registry.size() == 2);

  registry.unregister_runner(*other);
  registry.unregister_runner(*slot);
  CHECK(registry.size() == 0);
  CHECK_FALSE(registry.is_registered(std::this_thread::get_id()));
}

TEST_CASE("runner_slot notify and consume", "[io][runner_registry]")
{
  runner_registry registry;
  auto slot = registry.register_runner();
  CHECK_FALSE(readable(slot->wake_fd()));
  slot->notify();
  slot->notify();
  CHECK(readable(slot->wake_fd()));
  CHECK(slot->consume_notifications() == 2);
  CHECK_FALSE(readable(slot->wake_fd()));
  CHECK(slot->consume_notifications() == 0);
  CHECK(slot->notifications_sent() == 2);
  registry.unregister_runner(*slot);
}

TEST_CASE("wake_one_parked wakes exactly one parked runner", "[io][runner_registry]")
{
  runner_registry registry;
  std::vector<std::shared_ptr<runner_slot>> slots;
  // Registering threads stay alive (and registered) until the end of the test:
  // thread ids of exited threads may be reused.
  std::atomic<bool> release{false};
  std::vector<std::jthread> threads;
  for (int i = 0; i < 3; ++i) {
    std::atomic<bool> registered{false};
    threads.emplace_back([&] {
      slots.push_back(registry.register_runner());
      registered.store(true);
      while (!release.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });
    while (!registered.load()) {
      std::this_thread::yield();
    }
  }

  CHECK_FALSE(registry.wake_one_parked());  // nobody parked
  registry.park(*slots[0]);
  registry.park(*slots[2]);
  registry.park(*slots[2]);  // idempotent
  CHECK(registry.parked_count() == 2);
  CHECK(registry.idle_count() == 2);

  CHECK(registry.wake_one_parked());
  CHECK(registry.parked_count() == 1);
  CHECK(registry.wake_one_parked());
  CHECK(registry.parked_count() == 0);
  CHECK_FALSE(registry.wake_one_parked());
  CHECK(readable(slots[0]->wake_fd()));
  CHECK_FALSE(readable(slots[1]->wake_fd()));
  CHECK(readable(slots[2]->wake_fd()));

  registry.unpark(*slots[0]);  // already claimed: no double decrement
  CHECK(registry.parked_count() == 0);

  registry.wake_all();
  CHECK(readable(slots[1]->wake_fd()));
  for (auto& slot : slots) {
    registry.unregister_runner(*slot);
  }
  release.store(true);
}

TEST_CASE("wait_until_at_most returns once runners left", "[io][runner_registry]")
{
  runner_registry registry;
  std::atomic<bool> release{false};
  std::atomic<bool> registered{false};
  std::jthread runner([&] {
    auto slot = registry.register_runner();
    registered.store(true);
    while (!release.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    registry.unregister_runner(*slot);
  });
  while (!registered.load()) {
    std::this_thread::yield();
  }
  CHECK(registry.size() == 1);
  release.store(true);
  registry.wait_until_empty();
  CHECK(registry.size() == 0);
}

TEST_CASE("park protocol loses no wakeup and causes no wake storm", "[io][runner_registry]")
{
  request_hub hub;
  hub.set_accepting(true);
  auto object = std::make_shared<cucascade::test::stub::stub_io_object>(
    std::make_shared<cucascade::test::stub::dispatch_controls>(), "stub://park", 1UL << 20);

  constexpr std::size_t total = 5000;
  std::atomic<std::size_t> completed{0};
  std::atomic<std::uint64_t> notifications{0};
  std::atomic<std::size_t> parks{0};
  std::atomic<bool> ready{false};

  // One runner following the engine idle protocol, with an hour-long wait
  // timeout so that a lost wakeup would hang the test.
  std::jthread runner([&](std::stop_token stop) {
    auto slot = hub.registry().register_runner();
    ready.store(true);
    while (!stop.stop_requested() && completed.load() < total) {
      bool progressed = false;
      for (auto const cls : {request_class::latency, request_class::read}) {
        while (auto request = hub.try_pull(cls, *slot)) {
          while (!request->empty()) {
            static_cast<void>(request->take_front());
            request->coordinator->on_complete();
          }
          hub.finish_group(*request, *slot);
          completed.fetch_add(1);
          progressed = true;
        }
      }
      if (progressed || completed.load() >= total) continue;
      if (!hub.try_park(*slot)) continue;
      parks.fetch_add(1);
      pollfd entry{slot->wake_fd(), POLLIN, 0};
      static_cast<void>(::poll(&entry, 1, 3'600'000));
      static_cast<void>(slot->consume_notifications());
      hub.unpark(*slot);
    }
    notifications.store(slot->notifications_sent());
    hub.registry().unregister_runner(*slot);
  });
  while (!ready.load()) {
    std::this_thread::yield();
  }

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> pause(0, 20);
  for (std::size_t i = 0; i < total; ++i) {
    std::vector<cucascade::io::prepared_io_slice> slices;
    static std::uint8_t sink{};
    slices.emplace_back(cucascade::io::range{0, 64}, cucascade::io::host_buffer{&sink});
    hub.enqueue(grouped_io_request::create(object, std::move(slices)));
    if (auto const us = pause(rng); us > 15) {
      std::this_thread::sleep_for(std::chrono::microseconds(us));
    }
  }

  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (completed.load() < total && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  bool const all_completed = completed.load() == total;
  if (!all_completed) {
    runner.request_stop();
    hub.registry().wake_all();
  }
  runner.join();
  REQUIRE(all_completed);
  // At most one wakeup per park (the hub never notifies an unparked runner).
  CHECK(notifications.load() <= parks.load());
  CHECK(notifications.load() <= total);
}

TEST_CASE("prepare_wait encodes the queued-but-refused waiting rule", "[io][runner_registry]")
{
  using cucascade::io::detail::wait_action;
  request_hub hub;
  hub.set_accepting(true);
  auto slot       = hub.registry().register_runner();
  bool asked      = false;
  auto would_pull = [&](bool answer) {
    return [&asked, answer] {
      asked = true;
      return answer;
    };
  };

  // No room: never park, never consult the policy.
  CHECK(hub.prepare_wait(*slot, false, would_pull(true)) == wait_action::wait_unparked);
  CHECK_FALSE(asked);
  CHECK_FALSE(slot->parked());

  // Empty queue: park (the policy is not consulted).
  CHECK(hub.prepare_wait(*slot, true, would_pull(true)) == wait_action::park);
  CHECK_FALSE(asked);
  CHECK(slot->parked());
  hub.unpark(*slot);

  // Something queued: pull now if the policy takes it, else a bounded wait.
  auto object = std::make_shared<cucascade::test::stub::stub_io_object>(
    std::make_shared<cucascade::test::stub::dispatch_controls>(), "stub://wait", 1UL << 20);
  static std::uint8_t sink{};
  std::vector<cucascade::io::prepared_io_slice> slices;
  slices.emplace_back(cucascade::io::range{0, 64}, cucascade::io::host_buffer{&sink});
  hub.enqueue(grouped_io_request::create(object, std::move(slices)));
  CHECK(hub.prepare_wait(*slot, true, would_pull(true)) == wait_action::pull_now);
  CHECK(asked);
  CHECK_FALSE(slot->parked());
  CHECK(hub.prepare_wait(*slot, true, would_pull(false)) == wait_action::wait_bounded);
  CHECK_FALSE(slot->parked());

  hub.set_accepting(false);
  hub.cancel_queued(std::make_error_code(std::errc::operation_canceled));
  hub.registry().unregister_runner(*slot);
}

TEST_CASE("close_admission rejects with its reason until admission reopens",
          "[io][runner_registry]")
{
  request_hub hub;
  hub.set_accepting(true);
  auto object = std::make_shared<cucascade::test::stub::stub_io_object>(
    std::make_shared<cucascade::test::stub::dispatch_controls>(), "stub://closed", 1UL << 20);
  auto submit = [&] {
    static std::uint8_t sink{};
    std::vector<cucascade::io::prepared_io_slice> slices;
    slices.emplace_back(cucascade::io::range{0, 64}, cucascade::io::host_buffer{&sink});
    auto coordinator = std::make_shared<cucascade::io::grouped_coordinator>(64, 1);
    auto future      = coordinator->get_future();
    hub.enqueue(grouped_io_request::create(object, std::move(slices), coordinator));
    return future;
  };

  auto queued = submit();
  hub.close_admission(std::make_exception_ptr(std::runtime_error("no runner left")));
  CHECK_FALSE(hub.accepting());
  REQUIRE(hub.rejection_reason().has_value());
  CHECK(hub.cancel_queued(*hub.rejection_reason()) == 1);
  CHECK_THROWS_WITH(std::move(queued).get(), "no runner left");
  CHECK_THROWS_WITH(submit().get(), "no runner left");

  hub.set_accepting(false);  // e.g. shutdown: plain cancellation again
  CHECK_FALSE(hub.rejection_reason().has_value());
  CHECK_THROWS_AS(submit().get(), std::system_error);
}
