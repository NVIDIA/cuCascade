/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <thread>

namespace cucascade {
namespace detail {

/**
 * @brief Persistent single-thread task runner for disk I/O.
 *
 * Avoids per-chunk std::async thread creation overhead. Accepts one task at a
 * time; callers wait for completion via the returned std::future<void>.
 *
 * Owners that hold io_worker as a member alongside resources the submitted
 * tasks touch (pinned buffers, file descriptors, CUDA handles) must call
 * shutdown_and_join() before freeing those resources, otherwise the worker
 * thread may still be executing a task that references freed memory.
 */
class io_worker {
 public:
  io_worker() : _thread([this] { run(); }) {}

  ~io_worker() { shutdown_and_join(); }

  io_worker(const io_worker&)            = delete;
  io_worker& operator=(const io_worker&) = delete;
  io_worker(io_worker&&)                 = delete;
  io_worker& operator=(io_worker&&)      = delete;

  /// Submit a task and return a future that resolves when it completes.
  [[nodiscard]] std::future<void> submit(std::function<void()> work)
  {
    std::promise<void> promise;
    auto future = promise.get_future();
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _pending_work    = std::move(work);
      _pending_promise = std::move(promise);
      _has_task        = true;
    }
    _cv.notify_one();
    return future;
  }

  /**
   * @brief Stop the worker thread and wait for it to exit.
   *
   * Idempotent: safe to call from owning class destructors before the
   * io_worker member itself is destroyed, and from ~io_worker() if the owner
   * didn't call it explicitly. Returns only after any in-flight task has
   * finished — this is what owners rely on when freeing buffers/files the
   * task references.
   */
  void shutdown_and_join() noexcept
  {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      if (_shutdown) return;
      _shutdown = true;
    }
    _cv.notify_one();
    if (_thread.joinable()) { _thread.join(); }
  }

 private:
  void run()
  {
    while (true) {
      std::unique_lock<std::mutex> lock(_mutex);
      _cv.wait(lock, [this] { return _has_task || _shutdown; });
      if (_shutdown && !_has_task) return;

      auto work    = std::move(_pending_work);
      auto promise = std::move(_pending_promise);
      _has_task    = false;
      lock.unlock();

      try {
        work();
        promise.set_value();
      } catch (...) {
        promise.set_exception(std::current_exception());
      }
    }
  }

  std::mutex _mutex;
  std::condition_variable _cv;
  std::function<void()> _pending_work;
  std::promise<void> _pending_promise;
  bool _has_task{false};
  bool _shutdown{false};

  // Must be declared last so all other members are constructed before run()
  // starts, and (by reverse-destruction-order) joined before they're torn down
  // if a caller forgets to call shutdown_and_join() explicitly.
  std::thread _thread;
};

}  // namespace detail
}  // namespace cucascade
