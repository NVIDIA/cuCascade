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
#include <cucascade/io/details/request_queue.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/types.hpp>

#include <catch2/catch_all.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {

using cucascade::io::grouped_io_request;
using cucascade::io::io_options;
using cucascade::io::request_class;
using cucascade::io::request_state;
using cucascade::io::detail::request_hub;
using cucascade::io::detail::request_queue;
using cucascade::test::stub::dispatch_controls;
using cucascade::test::stub::stub_io_object;

std::shared_ptr<stub_io_object> make_object(std::size_t size = 1UL << 30)
{
  return std::make_shared<stub_io_object>(
    std::make_shared<dispatch_controls>(), "stub://queue", size);
}

/// A read request of @p n_slices slices of @p slice_bytes each, in lane @p cls.
std::unique_ptr<grouped_io_request> make_read(std::shared_ptr<stub_io_object> const& object,
                                              request_class cls,
                                              std::size_t slice_bytes = 4096,
                                              std::size_t n_slices    = 1)
{
  static std::uint8_t sink{};
  std::vector<cucascade::io::prepared_io_slice> slices;
  for (std::size_t i = 0; i < n_slices; ++i) {
    slices.emplace_back(cucascade::io::range{i * slice_bytes, slice_bytes},
                        cucascade::io::host_buffer{&sink});
  }
  io_options opts;
  opts.cls = cls;
  return grouped_io_request::create(object, std::move(slices), opts);
}

}  // namespace

TEST_CASE("request_queue pushes and pulls per class", "[io][queue]")
{
  request_queue queue;
  auto object = make_object();

  auto latency = make_read(object, request_class::latency, 100);
  auto read    = make_read(object, request_class::read, 1000, 2);
  REQUIRE(queue.push(latency));
  REQUIRE(queue.push(read));
  CHECK(latency == nullptr);
  CHECK(read == nullptr);

  CHECK(queue.approx_size(request_class::latency) == 1);
  CHECK(queue.approx_size(request_class::read) == 1);
  CHECK(queue.approx_size(request_class::write) == 0);
  CHECK(queue.approx_total() == 2);
  CHECK(queue.has_queued());
  CHECK(queue.queued_bytes(request_class::latency) == 100);
  CHECK(queue.queued_bytes(request_class::read) == 2000);
  CHECK(queue.total_queued_bytes() == 2100);

  std::unique_ptr<grouped_io_request> out;
  CHECK_FALSE(queue.try_pull(request_class::write, out));
  CHECK(out == nullptr);

  REQUIRE(queue.try_pull(request_class::read, out));
  REQUIRE(out != nullptr);
  CHECK(out->meta.cls == request_class::read);
  CHECK(out->meta.state.load() == request_state::assigned);
  CHECK(out->meta.assigned_at >= out->meta.enqueued_at);
  CHECK(queue.approx_size(request_class::read) == 0);
  CHECK(queue.queued_bytes(request_class::read) == 0);
  CHECK(queue.total_queued_bytes() == 100);

  out.reset();
  REQUIRE(queue.try_pull(request_class::latency, out));
  CHECK(out->meta.cls == request_class::latency);
  CHECK_FALSE(queue.has_queued());
  CHECK(queue.total_queued_bytes() == 0);

  // Settle the pulled request's credits so no promise is broken.
  out->cancel_remaining(std::make_error_code(std::errc::operation_canceled));
}

TEST_CASE("request_queue is FIFO within a lane for one producer", "[io][queue]")
{
  request_queue queue;
  auto object = make_object();
  std::vector<std::uint64_t> ids;
  for (int i = 0; i < 8; ++i) {
    auto request = make_read(object, request_class::write);
    ids.push_back(request->meta.id);
    REQUIRE(queue.push(request));
  }
  for (auto const id : ids) {
    std::unique_ptr<grouped_io_request> out;
    REQUIRE(queue.try_pull(request_class::write, out));
    CHECK(out->meta.id == id);
    out->cancel_remaining(std::make_error_code(std::errc::operation_canceled));
  }
}

TEST_CASE("request_queue oldest hint and queue-wait statistics", "[io][queue]")
{
  using clock = std::chrono::steady_clock;
  request_queue queue;
  auto object = make_object();

  CHECK(queue.oldest_hint(request_class::read) == clock::time_point::max());
  CHECK(queue.oldest_age(request_class::read, clock::now()).count() == 0);

  auto first = make_read(object, request_class::read);
  REQUIRE(queue.push(first));
  auto const after_first = clock::now();
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  auto second = make_read(object, request_class::read);
  REQUIRE(queue.push(second));

  auto const hint = queue.oldest_hint(request_class::read);
  CHECK(hint <= after_first);
  CHECK(queue.oldest_age(request_class::read, clock::now()) >= std::chrono::milliseconds(5));

  std::unique_ptr<grouped_io_request> out;
  REQUIRE(queue.try_pull(request_class::read, out));
  // After pulling the first, the hint moves forward but never past the second's enqueue time.
  CHECK(queue.oldest_hint(request_class::read) >= hint);
  CHECK(queue.last_wait(request_class::read) >= std::chrono::milliseconds(5));
  CHECK(queue.max_wait(request_class::read) >= queue.last_wait(request_class::read));
  out->cancel_remaining(std::make_error_code(std::errc::operation_canceled));

  REQUIRE(queue.try_pull(request_class::read, out));
  CHECK(queue.oldest_hint(request_class::read) == clock::time_point::max());
  out->cancel_remaining(std::make_error_code(std::errc::operation_canceled));
}

TEST_CASE("request_queue requeue preserves the enqueue time", "[io][queue]")
{
  using clock = std::chrono::steady_clock;
  request_queue queue;
  auto object = make_object();

  auto request = make_read(object, request_class::write);
  REQUIRE(queue.push(request));
  std::unique_ptr<grouped_io_request> out;
  REQUIRE(queue.try_pull(request_class::write, out));
  auto const enqueued = out->meta.enqueued_at;

  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  auto younger = make_read(object, request_class::write);
  REQUIRE(queue.push(younger));
  REQUIRE(queue.push(out, /*preserve_enqueue_time=*/true));
  CHECK(queue.oldest_hint(request_class::write) == enqueued);
  CHECK(queue.oldest_age(request_class::write, clock::now()) >= std::chrono::milliseconds(2));

  CHECK(queue.drain([](std::unique_ptr<grouped_io_request> r) noexcept {
    r->cancel_remaining(std::make_error_code(std::errc::operation_canceled));
  }) == 2);
}

TEST_CASE("request_queue drain hands out every request", "[io][queue]")
{
  request_queue queue;
  auto object = make_object();
  std::vector<cucascade::exec::semi_future<std::size_t>> futures;
  for (auto const cls : {request_class::latency,
                         request_class::read,
                         request_class::write,
                         request_class::background}) {
    for (int i = 0; i < 3; ++i) {
      auto request = make_read(object, cls);
      futures.push_back(request->coordinator->get_future());
      REQUIRE(queue.push(request));
    }
  }
  CHECK(queue.approx_total() == 12);

  std::size_t seen = 0;
  CHECK(queue.drain([&](std::unique_ptr<grouped_io_request> request) noexcept {
    ++seen;
    request->cancel_remaining(std::make_error_code(std::errc::operation_canceled));
  }) == 12);
  CHECK(seen == 12);
  CHECK(queue.approx_total() == 0);
  CHECK(queue.total_queued_bytes() == 0);
  for (auto& future : futures) {
    CHECK_THROWS_AS(std::move(future).get(), std::system_error);
  }
}

TEST_CASE("request_queue destructor cancels leftover requests", "[io][queue]")
{
  auto object = make_object();
  cucascade::exec::semi_future<std::size_t> future;
  {
    request_queue queue;
    auto request = make_read(object, request_class::read);
    future       = request->coordinator->get_future();
    REQUIRE(queue.push(request));
  }
  CHECK_THROWS_AS(std::move(future).get(), std::system_error);
}

TEST_CASE("request_queue is safe under concurrent producers and consumers", "[io][queue]")
{
  request_queue queue;
  auto object                          = make_object();
  constexpr std::size_t producers      = 4;
  constexpr std::size_t per_producer   = 2000;
  constexpr std::size_t total_requests = producers * per_producer;
  std::atomic<std::size_t> consumed{0};

  std::vector<std::jthread> threads;
  for (std::size_t p = 0; p < producers; ++p) {
    threads.emplace_back([&, p] {
      for (std::size_t i = 0; i < per_producer; ++i) {
        auto const cls = (i + p) % 2 == 0 ? request_class::read : request_class::write;
        auto request   = make_read(object, cls);
        while (!queue.push(request)) {}
      }
    });
  }
  for (std::size_t c = 0; c < 2; ++c) {
    threads.emplace_back([&, c] {
      auto const cls = c == 0 ? request_class::read : request_class::write;
      while (consumed.load() < total_requests) {
        std::unique_ptr<grouped_io_request> out;
        if (queue.try_pull(cls, out)) {
          out->cancel_remaining(std::make_error_code(std::errc::operation_canceled));
          consumed.fetch_add(1);
        } else {
          std::this_thread::yield();
        }
      }
    });
  }
  threads.clear();
  CHECK(consumed.load() == total_requests);
  CHECK(queue.approx_total() == 0);
  CHECK(queue.total_queued_bytes() == 0);
}

TEST_CASE("request_hub cancels requests while not accepting", "[io][queue]")
{
  request_hub hub;
  auto object  = make_object();
  auto request = make_read(object, request_class::read);
  auto future  = request->coordinator->get_future();

  CHECK_FALSE(hub.accepting());
  hub.enqueue(std::move(request));
  CHECK_FALSE(hub.has_queued());
  CHECK_THROWS_MATCHES(std::move(future).get(),
                       std::system_error,
                       Catch::Matchers::Predicate<std::system_error>([](auto const& error) {
                         return error.code() == std::errc::operation_canceled;
                       }));
}

TEST_CASE("request_hub pull / finish / requeue bookkeeping", "[io][queue]")
{
  request_hub hub;
  hub.set_accepting(true);
  auto slot   = hub.registry().register_runner();
  auto object = make_object();

  auto request = make_read(object, request_class::background, 4096, 3);
  auto future  = request->coordinator->get_future();
  hub.enqueue(std::move(request));
  CHECK(hub.stats()
          .per_class[cucascade::io::request_class_index(request_class::background)]
          .queued_requests == 1);

  auto pulled = hub.try_pull(request_class::background, *slot);
  REQUIRE(pulled != nullptr);
  CHECK(pulled->meta.runner_id == slot->id());
  CHECK(slot->active_groups() == 1);
  CHECK(hub.in_flight_requests() == 1);

  // Settle one slice, then hand the rest back (runner retirement).
  static_cast<void>(pulled->take_front());
  pulled->coordinator->on_complete();
  hub.requeue(std::move(pulled), *slot);
  CHECK(slot->active_groups() == 0);
  CHECK(hub.in_flight_requests() == 0);
  CHECK(hub.queued_bytes() == 2 * 4096);

  auto again = hub.try_pull(request_class::background, *slot);
  REQUIRE(again != nullptr);
  CHECK(again->remaining_slices() == 2);
  while (!again->empty()) {
    static_cast<void>(again->take_front());
    again->coordinator->on_complete();
  }
  hub.finish_group(*again, *slot);
  CHECK(again->meta.state.load() == request_state::completed);
  CHECK(again->meta.completed_at >= again->meta.assigned_at);
  CHECK(slot->retired_groups() == 1);
  CHECK(hub.in_flight_requests() == 0);
  CHECK(std::move(future).get() == 3 * 4096);

  // Requeue after admission closed cancels instead.
  auto late        = make_read(object, request_class::read, 4096, 2);
  auto late_future = late->coordinator->get_future();
  hub.enqueue(std::move(late));
  auto late_pulled = hub.try_pull(request_class::read, *slot);
  REQUIRE(late_pulled != nullptr);
  hub.set_accepting(false);
  hub.requeue(std::move(late_pulled), *slot);
  CHECK_FALSE(hub.has_queued());
  CHECK_THROWS_AS(std::move(late_future).get(), std::system_error);

  hub.registry().unregister_runner(*slot);
}

TEST_CASE("request_hub cancel_queued settles every queued request", "[io][queue]")
{
  request_hub hub;
  hub.set_accepting(true);
  auto object = make_object();
  std::vector<cucascade::exec::semi_future<std::size_t>> futures;
  for (int i = 0; i < 5; ++i) {
    auto request = make_read(object, i % 2 == 0 ? request_class::read : request_class::latency);
    futures.push_back(request->coordinator->get_future());
    hub.enqueue(std::move(request));
  }
  hub.set_accepting(false);
  CHECK(hub.cancel_queued(std::make_error_code(std::errc::operation_canceled)) == 5);
  for (auto& future : futures) {
    CHECK_THROWS_AS(std::move(future).get(), std::system_error);
  }
}
