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

#pragma once

#include <chrono>
#include <exception>
#include <future>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>

namespace cucascade::test {

/// Upper bound for an operation that is expected to complete.
inline constexpr std::chrono::seconds expect_ready_timeout{10};

/// How long an operation that is expected to stay blocked is observed.
inline constexpr std::chrono::milliseconds expect_blocked_timeout{100};

/**
 * @brief Run `fn` on a detached thread and return a future for its result.
 *
 * Unlike std::async, the returned future does not block on destruction, so a test whose
 * bounded wait times out fails instead of hanging. `fn` should hold shared ownership of
 * whatever it uses, because a call that never returns outlives the test. `fn` is destroyed
 * before its result is published, so once the future is ready the thread holds no references.
 */
template <typename Fn>
[[nodiscard]] std::future<std::invoke_result_t<Fn&>> run_detached(Fn fn)
{
  using result_type = std::invoke_result_t<Fn&>;
  std::promise<result_type> promise;
  auto future = promise.get_future();
  std::thread([fn = std::optional<Fn>(std::move(fn)), promise = std::move(promise)]() mutable {
    try {
      if constexpr (std::is_void_v<result_type>) {
        (*fn)();
        fn.reset();
        promise.set_value();
      } else {
        auto result = (*fn)();
        fn.reset();
        promise.set_value(std::move(result));
      }
    } catch (...) {
      fn.reset();
      promise.set_exception(std::current_exception());
    }
  }).detach();
  return future;
}

/// True when `future` becomes ready within expect_ready_timeout.
template <typename T>
[[nodiscard]] bool becomes_ready(std::future<T> const& future)
{
  return future.wait_for(expect_ready_timeout) == std::future_status::ready;
}

/// True when `future` is still not ready after expect_blocked_timeout.
template <typename T>
[[nodiscard]] bool stays_blocked(std::future<T> const& future)
{
  return future.wait_for(expect_blocked_timeout) == std::future_status::timeout;
}

/// Poll `predicate` until it holds or expect_ready_timeout elapses; returns its final value.
template <typename Predicate>
[[nodiscard]] bool eventually(Predicate predicate)
{
  auto const deadline = std::chrono::steady_clock::now() + expect_ready_timeout;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) { return predicate(); }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return true;
}

}  // namespace cucascade::test
