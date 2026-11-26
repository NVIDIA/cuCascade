/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "utils/detached_task.hpp"

#include <cucascade/memory/notification_channel.hpp>

#include <catch2/catch_all.hpp>

#include <future>
#include <memory>
#include <vector>

using cucascade::memory::notification_channel;
using cucascade::test::becomes_ready;
using cucascade::test::run_detached;
using cucascade::test::stays_blocked;
using wait_status = notification_channel::wait_status;

namespace {

std::future<wait_status> wait_detached(std::shared_ptr<notification_channel::scoped_waiter> waiter)
{
  return run_detached([waiter = std::move(waiter)] { return waiter->wait(); });
}

}  // namespace

TEST_CASE("notification_channel registers waiters in FIFO order", "[notification_channel]")
{
  auto channel = std::make_shared<notification_channel>();
  CHECK_FALSE(channel->has_waiters());
  {
    notification_channel::scoped_waiter first{*channel};
    CHECK(channel->has_waiters());
    CHECK(first.is_head());
    {
      notification_channel::scoped_waiter second{*channel};
      CHECK(first.is_head());
      CHECK_FALSE(second.is_head());
    }
    CHECK(first.is_head());
  }
  CHECK_FALSE(channel->has_waiters());

  notification_channel::scoped_waiter first{*channel};
  auto second = std::make_unique<notification_channel::scoped_waiter>(*channel);
  auto third  = std::make_unique<notification_channel::scoped_waiter>(*channel);
  // A departing non-head waiter leaves the order of the others intact.
  second.reset();
  CHECK(first.is_head());
  CHECK_FALSE(third->is_head());
}

TEST_CASE("notification_channel delivers a notification to the head waiter only",
          "[notification_channel][threading]")
{
  auto channel = std::make_shared<notification_channel>();
  // An active notifier keeps wait() from returning IDLE.
  auto notifier = channel->get_notifier();

  auto first         = std::make_shared<notification_channel::scoped_waiter>(*channel);
  auto second        = std::make_shared<notification_channel::scoped_waiter>(*channel);
  auto second_status = wait_detached(second);
  auto first_status  = wait_detached(first);
  CHECK(stays_blocked(first_status));
  CHECK(stays_blocked(second_status));

  notifier->post();
  REQUIRE(becomes_ready(first_status));
  CHECK(first_status.get() == wait_status::NOTIFIED);
  // The head consumed the notification and is still registered, so the second waiter sleeps.
  CHECK(stays_blocked(second_status));

  // Destroying the head passes the turn to the next waiter.
  first.reset();
  REQUIRE(becomes_ready(second_status));
  CHECK(second_status.get() == wait_status::NOTIFIED);
  CHECK(second->is_head());
}

TEST_CASE("notification_channel keeps a notification for the head waiter",
          "[notification_channel][threading]")
{
  auto channel  = std::make_shared<notification_channel>();
  auto notifier = channel->get_notifier();

  auto first         = std::make_shared<notification_channel::scoped_waiter>(*channel);
  auto second        = std::make_shared<notification_channel::scoped_waiter>(*channel);
  auto second_status = wait_detached(second);
  CHECK(stays_blocked(second_status));

  // Only a waiter behind the head is blocked when the notification arrives.
  notifier->post();
  CHECK(stays_blocked(second_status));

  auto first_status = wait_detached(first);
  REQUIRE(becomes_ready(first_status));
  CHECK(first_status.get() == wait_status::NOTIFIED);

  first.reset();
  REQUIRE(becomes_ready(second_status));
  CHECK(second_status.get() == wait_status::NOTIFIED);
}

TEST_CASE("notification_channel shutdown wakes every waiter", "[notification_channel][threading]")
{
  constexpr int num_waiters = 4;
  auto channel              = std::make_shared<notification_channel>();
  auto notifier             = channel->get_notifier();

  // The waiters stay registered after wait() returns, so a waiter that wakes does not pass
  // the turn on to the others.
  std::vector<std::shared_ptr<notification_channel::scoped_waiter>> waiters;
  std::vector<std::future<wait_status>> statuses;
  for (int i = 0; i < num_waiters; ++i) {
    waiters.push_back(std::make_shared<notification_channel::scoped_waiter>(*channel));
    statuses.push_back(wait_detached(waiters.back()));
  }
  for (auto& status : statuses) {
    CHECK(stays_blocked(status));
  }

  channel->shutdown();
  for (auto& status : statuses) {
    REQUIRE(becomes_ready(status));
    CHECK(status.get() == wait_status::SHUTDOWN);
  }
}

TEST_CASE("notification_channel wakes every waiter when the last notifier is released",
          "[notification_channel][threading]")
{
  auto channel  = std::make_shared<notification_channel>();
  auto notifier = channel->get_notifier();

  auto first         = std::make_shared<notification_channel::scoped_waiter>(*channel);
  auto second        = std::make_shared<notification_channel::scoped_waiter>(*channel);
  auto first_status  = wait_detached(first);
  auto second_status = wait_detached(second);
  CHECK(stays_blocked(first_status));
  CHECK(stays_blocked(second_status));

  notifier.reset();
  REQUIRE(becomes_ready(first_status));
  REQUIRE(becomes_ready(second_status));
  CHECK(first_status.get() == wait_status::IDLE);
  CHECK(second_status.get() == wait_status::IDLE);
}
