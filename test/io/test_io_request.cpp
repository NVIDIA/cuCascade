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

#include <cucascade/io/io_request.hpp>

#include <catch2/catch_all.hpp>

#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

using cucascade::io::grouped_coordinator;
using cucascade::io::io_op_request;
using cucascade::io::prepared_io_completion;
using namespace std::chrono_literals;

TEST_CASE("an empty grouped coordinator is ready with zero bytes", "[io][coordinator]")
{
  grouped_coordinator coordinator{0, 0};
  auto future = coordinator.get_future();
  CHECK(future.is_ready());
  CHECK(std::move(future).get(2s) == 0);
}

TEST_CASE("slice expansion adds credits before children complete", "[io][coordinator]")
{
  grouped_coordinator coordinator{4096, 1};
  auto future = coordinator.get_future();
  coordinator.add_tasks(3);
  CHECK(coordinator.tasks_remaining() == 4);
  std::vector<std::jthread> completions;
  completions.reserve(4);
  for (int i = 0; i < 4; ++i) {
    completions.emplace_back([&] { coordinator.on_complete(); });
  }
  for (auto& completion : completions) {
    completion.join();
  }
  CHECK(future.is_ready());
  CHECK(std::move(future).get(2s) == 4096);
}

TEST_CASE("the first error stops dispatch but fulfillment waits for drain", "[io][coordinator]")
{
  grouped_coordinator coordinator{128, 2};
  auto future = coordinator.get_future();
  auto error  = std::make_exception_ptr(std::runtime_error("first physical read failed"));
  coordinator.report_error(error);
  CHECK_FALSE(coordinator.should_continue());
  CHECK(coordinator.has_error());
  CHECK(coordinator.tasks_remaining() == 1);
  CHECK_FALSE(future.is_ready());
  coordinator.on_complete();
  CHECK(future.is_ready());
  CHECK_THROWS_WITH(std::move(future).get(2s), "first physical read failed");
}

TEST_CASE("physical completion publishes cache state before the future", "[io][coordinator]")
{
  auto coordinator = std::make_shared<grouped_coordinator>(64, 1);
  auto future      = coordinator->get_future();
  std::atomic<bool> callback_ran{false};
  io_op_request operation;
  operation.coordinator = coordinator;
  operation.on_complete = std::make_shared<prepared_io_completion>(
    [&](std::span<cucascade::io::cache::cached_chunk* const>, bool success) noexcept {
      callback_ran.store(success, std::memory_order_release);
    });
  operation.finish_success();
  CHECK(callback_ran.load(std::memory_order_acquire));
  CHECK(future.is_ready());
  CHECK(std::move(future).get(2s) == 64);
  CHECK(operation.terminal());
}

TEST_CASE("host-valid data is publishable even when its device copy fails", "[io][coordinator]")
{
  auto coordinator = std::make_shared<grouped_coordinator>(64, 1);
  auto future      = coordinator->get_future();
  std::atomic<bool> host_data_valid{false};
  io_op_request operation;
  operation.coordinator = coordinator;
  operation.on_complete = std::make_shared<prepared_io_completion>(
    [&](std::span<cucascade::io::cache::cached_chunk* const>, bool success) noexcept {
      host_data_valid.store(success, std::memory_order_release);
    });
  operation.finish_error(std::make_exception_ptr(std::runtime_error("H2D event failed")), true);
  CHECK(host_data_valid.load(std::memory_order_acquire));
  CHECK(future.is_ready());
  CHECK_THROWS_WITH(std::move(future).get(2s), "H2D event failed");
}
