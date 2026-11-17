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
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace cucascade::test {

/**
 * @brief Thrown when the start gate of a worker_group will not open: a worker ended without
 * passing it, the deadline passed, or the group was destroyed first.
 */
class start_gate_broken : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/**
 * @brief Runs a fixed number of workers on threads of their own for a test that must fail rather
 * than hang.
 *
 * Each worker prepares, calls start_gate::arrive_and_wait() once, and then does the work that
 * should run at the same time as the other workers'. The gate opens when every worker has arrived.
 *
 * No wait is unbounded. All of them end at one deadline, set when the group is constructed:
 * - A worker that ends without passing the gate, by an exception or by returning, breaks the gate,
 *   so the workers waiting at it leave with start_gate_broken instead of waiting for it for ever.
 * - wait_until_started() and join() report the first failure of a worker as that worker's own
 *   exception.
 * - A worker that is still running two seconds after the deadline cannot be stopped, and it uses
 *   objects of the test that is about to end. join() and the destructor then write a message to
 *   stderr and abort the process.
 *
 * The destructor breaks a gate that is still closed and waits for the workers as join() does, so
 * a test may leave the group's scope at any point, also by an exception.
 *
 * @tparam Result What a worker returns; not void
 */
template <std::movable Result>
class worker_group {
 public:
  using clock = std::chrono::steady_clock;

  /**
   * @brief Where the workers of one group wait until all of them are ready.
   */
  class start_gate {
   public:
    /**
     * @brief Report that the calling worker is ready and wait until every worker is.
     *
     * @throws start_gate_broken if the gate will not open
     */
    void arrive_and_wait() { _group.arrive_and_wait(); }

   private:
    friend class worker_group;
    explicit start_gate(worker_group& group) noexcept : _group{group} {}

    worker_group& _group;
  };

  /**
   * @param worker_count Workers that must arrive before the start gate opens
   * @param timeout Time from now until the deadline of every wait of this group
   */
  worker_group(std::size_t worker_count, clock::duration timeout)
    : _worker_count{worker_count}, _deadline{clock::now() + timeout}, _results(worker_count)
  {
    _threads.reserve(worker_count);
  }

  worker_group(worker_group const&)            = delete;
  worker_group& operator=(worker_group const&) = delete;

  ~worker_group()
  {
    break_gate_if_closed();
    wait_for_workers_or_abort();
  }

  /**
   * @brief Start a worker on a new thread.
   *
   * @param work Called as `work(gate)` with this group's start gate; returns the worker's result
   * @throws std::logic_error if the group already has `worker_count` workers
   */
  template <std::invocable<start_gate&> Work>
    requires std::convertible_to<std::invoke_result_t<Work&, start_gate&>, Result>
  void spawn(Work work)
  {
    std::size_t const index = _threads.size();
    if (index == _worker_count) { throw std::logic_error{"worker_group: too many workers"}; }
    _threads.emplace_back([this, index, work = std::move(work)]() mutable { run(index, work); });
  }

  /**
   * @brief Wait until every worker has passed the start gate.
   *
   * @throws The first failure of a worker, if one failed before the gate opened; otherwise
   * start_gate_broken if the gate did not open
   */
  void wait_until_started()
  {
    std::unique_lock lock{_mutex};
    _changed.wait_until(lock, _deadline, [this] { return _gate != gate_state::closed; });
    if (_gate == gate_state::open) { return; }
    _gate = gate_state::broken;
    _changed.notify_all();
    if (_failure) { std::rethrow_exception(_failure); }
    throw start_gate_broken{"worker_group: the start gate did not open"};
  }

  /**
   * @brief Wait until every worker has ended.
   *
   * @return The workers' results, in the order the workers were spawned
   * @throws The first failure of a worker, if any worker failed
   */
  [[nodiscard]] std::vector<Result> join()
  {
    if (_threads.size() < _worker_count) { break_gate_if_closed(); }
    wait_for_workers_or_abort();

    std::scoped_lock lock{_mutex};
    if (_failure) { std::rethrow_exception(_failure); }
    std::vector<Result> results;
    results.reserve(_results.size());
    for (std::optional<Result>& result : _results) {
      results.push_back(std::move(result).value());
    }
    return results;
  }

 private:
  enum class gate_state { closed, open, broken };

  /// Time a worker gets to end once the deadline has passed and the gate is broken.
  static constexpr std::chrono::seconds exit_grace{2};

  void arrive_and_wait()
  {
    std::unique_lock lock{_mutex};
    if (_gate == gate_state::closed && ++_arrived == _worker_count) {
      _gate = gate_state::open;
      _changed.notify_all();
    }
    _changed.wait_until(lock, _deadline, [this] { return _gate != gate_state::closed; });
    if (_gate == gate_state::open) { return; }
    _gate = gate_state::broken;
    _changed.notify_all();
    throw start_gate_broken{"worker_group: the start gate did not open"};
  }

  template <typename Work>
  void run(std::size_t index, Work& work) noexcept
  {
    std::optional<Result> result;
    std::exception_ptr failure;
    try {
      start_gate gate{*this};
      result.emplace(work(gate));
    } catch (...) {
      failure = std::current_exception();
    }

    std::scoped_lock lock{_mutex};
    // The first failure is the cause; workers that then find the gate broken fail after it.
    if (failure && !_failure) { _failure = failure; }
    _results[index] = std::move(result);
    // A worker that ends while the gate is closed never arrived, so the gate cannot open any more.
    if (_gate == gate_state::closed) { _gate = gate_state::broken; }
    ++_finished;
    _changed.notify_all();
  }

  void break_gate_if_closed()
  {
    std::scoped_lock lock{_mutex};
    if (_gate == gate_state::closed) {
      _gate = gate_state::broken;
      if (!_failure) {
        _failure = std::make_exception_ptr(
          start_gate_broken{"worker_group: the start gate was given up before it opened"});
      }
    }
    _changed.notify_all();
  }

  void wait_for_workers_or_abort()
  {
    {
      std::unique_lock lock{_mutex};
      auto const all_ended = [this] { return _finished == _threads.size(); };
      if (!_changed.wait_until(lock, _deadline, all_ended)) {
        if (_gate == gate_state::closed) { _gate = gate_state::broken; }
        _changed.notify_all();
        if (!_changed.wait_for(lock, exit_grace, all_ended)) {
          std::fprintf(stderr,
                       "worker_group: %zu of %zu workers are still running after the deadline; "
                       "aborting, because they use objects of the test that started them\n",
                       _threads.size() - _finished,
                       _threads.size());
          std::abort();
        }
      }
    }
    // Every worker has left run(), so these joins return at once.
    _threads.clear();
    _finished = 0;
  }

  std::size_t const _worker_count;
  clock::time_point const _deadline;

  std::mutex _mutex;
  std::condition_variable _changed;  ///< Signals every change of the members below
  gate_state _gate{gate_state::closed};
  std::size_t _arrived{0};
  std::size_t _finished{0};
  std::exception_ptr _failure;  ///< First failure of a worker, or why the gate was given up
  std::vector<std::optional<Result>> _results;
  std::vector<std::jthread> _threads;
};

}  // namespace cucascade::test
