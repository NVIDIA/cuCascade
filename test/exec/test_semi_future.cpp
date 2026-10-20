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

#include <cucascade/exec/semi_future.hpp>

#include <catch2/catch_all.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using cucascade::exec::broken_promise_error;
using cucascade::exec::future_already_retrieved_error;
using cucascade::exec::future_invalid_error;
using cucascade::exec::inline_executor;
using cucascade::exec::make_semi_future;
using cucascade::exec::promise;
using cucascade::exec::promise_already_satisfied_error;
using cucascade::exec::semi_future;
using cucascade::exec::try_t;
using namespace std::chrono_literals;

/// Counts live instances, so a test can check that every value a future carried was destroyed.
struct tracked {
  static inline std::atomic<int> live{0};
  int value{0};
  explicit tracked(int v) : value(v) { live.fetch_add(1); }
  tracked(tracked&& other) noexcept : value(other.value) { live.fetch_add(1); }
  tracked& operator=(tracked&&) noexcept = default;
  tracked(tracked const&)                = delete;
  tracked& operator=(tracked const&)     = delete;
  ~tracked() { live.fetch_sub(1); }
};

/// One worker thread draining a FIFO: an executor whose tasks run on another thread.
class worker_executor {
 public:
  worker_executor()
    : _thread([this] {
        for (;;) {
          cucascade::exec::executor_func task;
          {
            std::unique_lock lock(_mutex);
            _cv.wait(lock, [this] { return _stop || !_tasks.empty(); });
            if (_tasks.empty()) { return; }
            task = std::move(_tasks.front());
            _tasks.pop_front();
          }
          std::move(task)();
        }
      })
  {
  }
  worker_executor(worker_executor const&)            = delete;
  worker_executor& operator=(worker_executor const&) = delete;
  ~worker_executor()
  {
    {
      std::lock_guard lock(_mutex);
      _stop = true;
    }
    _cv.notify_all();
    _thread.join();
  }

  void enqueue(cucascade::exec::executor_func task)
  {
    {
      std::lock_guard lock(_mutex);
      _tasks.push_back(std::move(task));
    }
    _cv.notify_one();
  }

 private:
  std::mutex _mutex;
  std::condition_variable _cv;
  std::deque<cucascade::exec::executor_func> _tasks;
  bool _stop{false};
  std::thread _thread;
};

}  // namespace

TEST_CASE("semi_future delivers a value set before and after it is taken", "[exec][semi_future]")
{
  SECTION("set first")
  {
    promise<int> p;
    p.set_value(7);
    // A satisfied promise has let go of its slot (cuCascade's contract).
    CHECK_THROWS_AS(p.get_semi_future(), promise_already_satisfied_error);
  }
  SECTION("future first, pull")
  {
    promise<int> p;
    auto f = p.get_semi_future();
    CHECK_FALSE(f.is_ready());
    p.set_value(7);
    CHECK(f.is_ready());
    CHECK(std::move(f).get() == 7);
  }
  SECTION("future first, push")
  {
    promise<std::string> p;
    auto f = p.get_semi_future();
    std::string got;
    std::move(f).install_callback([&](try_t<std::string>&& t) { got = std::move(t).get(); });
    CHECK(got.empty());
    p.set_value("done");
    CHECK(got == "done");
  }
  SECTION("ready value and a second retrieval")
  {
    CHECK(make_semi_future(3).get_try().value() == 3);
    promise<void> p;
    auto f = p.get_semi_future();
    CHECK_THROWS_AS(p.get_semi_future(), future_already_retrieved_error);
    p.set_value();
    CHECK_NOTHROW(std::move(f).get());
    semi_future<int> invalid;
    CHECK_THROWS_AS(std::move(invalid).get(), future_invalid_error);
  }
}

TEST_CASE("semi_future propagates exceptions and broken promises", "[exec][semi_future]")
{
  {
    promise<int> p;
    auto f = p.get_semi_future();
    p.set_exception(std::runtime_error("boom"));
    CHECK_THROWS_AS(std::move(f).get(), std::runtime_error);
  }
  {
    semi_future<int> f;
    {
      promise<int> p;
      f = p.get_semi_future();
    }
    CHECK_THROWS_AS(std::move(f).get(), broken_promise_error);
  }
  {
    // Move-assigning over an unsatisfied promise breaks the promise it replaces.
    promise<int> a;
    auto f = a.get_semi_future();
    promise<int> b;
    a = std::move(b);
    CHECK_THROWS_AS(std::move(f).get(), broken_promise_error);
    CHECK(a.valid());
    CHECK_FALSE(b.valid());  // NOLINT(bugprone-use-after-move): moved-from state is specified
  }
  {
    // A promise dropped before its future was taken frees its slot.
    tracked::live = 0;
    {
      promise<tracked> p;
    }
    {
      promise<tracked> p;
      p.set_value(tracked{1});
    }
    CHECK(tracked::live == 0);
  }
}

TEST_CASE("semi_future frees the value on every path", "[exec][semi_future]")
{
  tracked::live = 0;
  {
    promise<tracked> p;
    auto f = p.get_semi_future();
    p.set_value(tracked{1});
  }  // future dropped unread
  {
    promise<tracked> p;
    auto f = p.get_semi_future();
    std::move(f).install_callback([](try_t<tracked>&&) {});
    p.set_value(tracked{2});
  }
  {
    promise<tracked> p;
    auto f = p.get_semi_future();
    p.set_value(tracked{3});
    CHECK(std::move(f).get().value == 3);
  }
  {
    promise<tracked> p;
    auto f = p.get_semi_future();
    CHECK_FALSE(std::move(f).wait(1ms));
  }  // consumer and producer both drop an unsatisfied slot
  CHECK(tracked::live == 0);
}

TEST_CASE("semi_future timed waits", "[exec][semi_future]")
{
  promise<int> p;
  auto f     = p.get_semi_future();
  auto start = std::chrono::steady_clock::now();
  CHECK_THROWS_AS(std::move(f).get(20ms), std::runtime_error);
  CHECK(std::chrono::steady_clock::now() - start >= 20ms);

  promise<int> q;
  auto g = q.get_semi_future();
  std::thread producer([&] {
    std::this_thread::sleep_for(20ms);
    q.set_value(5);
  });
  CHECK(std::move(g).get(10s) == 5);
  producer.join();
}

TEST_CASE("semi_future wakes a parked consumer", "[exec][semi_future]")
{
  // Long enough that the consumer is past its spin and parked on the futex.
  for (auto delay : {0ms, 1ms, 20ms}) {
    promise<int> p;
    auto f = p.get_semi_future();
    std::thread producer([&, delay] {
      std::this_thread::sleep_for(delay);
      p.set_value(11);
    });
    CHECK(std::move(f).get() == 11);
    producer.join();
  }
}

TEST_CASE("semi_future deferred and flattened chains", "[exec][semi_future]")
{
  promise<int> p;
  auto f = p.get_semi_future()
             .defer_value([](int v) { return v * 2; })
             .defer([](try_t<int>&& t) { return make_semi_future(std::move(t).get() + 1); })
             .defer_value([](int v) { return std::to_string(v); });
  p.set_value(20);
  CHECK(std::move(f).get() == "41");

  promise<int> q;
  bool ensured = false;
  auto g       = q.get_semi_future()
             .defer_value([](int) -> int { throw std::logic_error("x"); })
             .defer_error(cucascade::exec::tag<std::logic_error>,
                          [](std::logic_error const&) { return -1; })
             .defer_ensure([&] { ensured = true; });
  q.set_value(1);
  CHECK(std::move(g).get() == -1);
  CHECK(ensured);
}

TEST_CASE("future runs then-stages on its executor", "[exec][semi_future]")
{
  worker_executor ex;
  promise<int> p;
  auto const caller = std::this_thread::get_id();
  std::atomic<bool> off_caller{true};
  auto f = p.get_semi_future()
             .via(&ex)
             .then_value([&](int v) {
               off_caller = off_caller && std::this_thread::get_id() != caller;
               return v + 1;
             })
             .then_try([](try_t<int>&& t) { return make_semi_future(std::move(t).get() * 10); })
             .then_error([](std::exception_ptr const&) { return 0; });
  p.set_value(4);
  CHECK(std::move(f).get() == 50);
  CHECK(off_caller);

  auto inline_result =
    make_semi_future(1).via(inline_executor::instance()).then_value([](int v) { return v + 1; });
  CHECK(std::move(inline_result).get() == 2);
}

TEST_CASE("semi_future callback that throws leaves a finished slot", "[exec][semi_future]")
{
  tracked::live = 0;
  {
    promise<tracked> p;
    auto f = p.get_semi_future();
    std::move(f).install_callback([](try_t<tracked>&&) { throw std::runtime_error("cb"); });
    // The exception reaches the producer; the slot is finished and freed exactly once.
    CHECK_THROWS_AS(p.set_value(tracked{1}), std::runtime_error);
    CHECK_FALSE(p.valid());
  }
  {
    promise<tracked> p;
    auto f = p.get_semi_future();
    p.set_value(tracked{2});
    CHECK_THROWS_AS(
      std::move(f).install_callback([](try_t<tracked>&&) { throw std::runtime_error("cb"); }),
      std::runtime_error);
  }
  CHECK(tracked::live == 0);
}

TEST_CASE("semi_future races producers against pull and push consumers", "[exec][semi_future]")
{
  constexpr int rounds = 20000;
  tracked::live        = 0;
  std::atomic<long> sum{0};
  std::atomic<int> callbacks{0};
  for (int mode = 0; mode < 3; ++mode) {
    std::vector<promise<tracked>> promises(rounds);
    std::vector<semi_future<tracked>> futures;
    futures.reserve(rounds);
    for (auto& p : promises) {
      futures.push_back(p.get_semi_future());
    }
    std::thread producer([&] {
      for (int i = 0; i < rounds; ++i) {
        promises[static_cast<std::size_t>(i)].set_value(tracked{i});
      }
    });
    for (int i = 0; i < rounds; ++i) {
      auto& f = futures[static_cast<std::size_t>(i)];
      if (mode == 0) {
        sum += std::move(f).get().value;
      } else if (mode == 1) {
        std::move(f).install_callback([&](try_t<tracked>&& t) {
          sum += t.value().value;
          callbacks.fetch_add(1);
        });
      } else if (i % 2 == 0) {
        f.wait();
        sum += std::move(f).get().value;
      } else {
        f = semi_future<tracked>{};  // dropped while the producer races
      }
    }
    producer.join();
  }
  long const per_round = static_cast<long>(rounds) * (rounds - 1) / 2;
  long evens           = 0;
  for (int i = 0; i < rounds; i += 2) {
    evens += i;
  }
  CHECK(sum == 2 * per_round + evens);
  CHECK(callbacks == rounds);
  CHECK(tracked::live == 0);
}
