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

// Tests for cucascade::detail::io_worker.
//
// These tests guard the contract that pipeline_io_backend relies on at
// shutdown: when shutdown_and_join() returns, any in-flight task has finished
// running. If that contract is violated, ~pipeline_io_backend would free
// pinned buffers / CUDA handles while a pwrite/pread is still running on the
// worker thread, producing the kernel-level hang observed in the field.

#include "../../src/data/io_worker.hpp"

#include <catch2/catch.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>

using cucascade::detail::io_worker;
using namespace std::chrono_literals;

TEST_CASE("io_worker basic submit and wait", "[io_worker]")
{
  io_worker worker;
  std::atomic<int> counter{0};
  auto fut = worker.submit([&] { counter.store(42); });
  fut.get();
  REQUIRE(counter.load() == 42);
}

TEST_CASE("io_worker propagates exceptions through future", "[io_worker]")
{
  io_worker worker;
  auto fut = worker.submit([] { throw std::runtime_error("boom"); });
  REQUIRE_THROWS_AS(fut.get(), std::runtime_error);
}

TEST_CASE("io_worker shutdown_and_join is idempotent", "[io_worker]")
{
  io_worker worker;
  worker.shutdown_and_join();
  // Calling again must not crash, deadlock, or throw — the io_worker's own
  // destructor will call it a third time below.
  worker.shutdown_and_join();
  worker.shutdown_and_join();
}

// =============================================================================
// Regression test for the pipeline_io_backend shutdown deadlock.
//
// Scenario reproduced (production failure mode):
//   1. A long-running task is submitted to io_worker (in the field this is a
//      pwrite/pread on a slow NVMe path).
//   2. The owning class is destroyed while that task is still running.
//   3. The owner's destructor body had already freed the buffer the task
//      references, and the io_worker is destroyed *after* that, so the worker
//      thread either dereferences freed memory or hangs.
//
// With the fix, owners explicitly call shutdown_and_join() at the top of their
// destructor — this returns only after the task completes, so freeing the
// buffer afterwards is safe.
//
// This test simulates the pattern with a test-owned "buffer" guarded by a
// latch. If shutdown_and_join() returned while the task was still running,
// the assertion `task_completed_before_shutdown_returned` would be false.
// =============================================================================
TEST_CASE("io_worker shutdown waits for in-flight task to complete", "[io_worker][regression]")
{
  std::mutex latch_mutex;
  std::condition_variable latch_cv;
  bool release_task{false};
  std::atomic<bool> task_running{false};
  std::atomic<bool> task_completed{false};

  std::atomic<bool> task_completed_before_shutdown_returned{false};

  {
    io_worker worker;

    auto fut = worker.submit([&] {
      task_running.store(true);
      std::unique_lock<std::mutex> lock(latch_mutex);
      latch_cv.wait(lock, [&] { return release_task; });
      task_completed.store(true);
    });

    // Wait until the task is actually running on the worker thread.
    while (!task_running.load()) {
      std::this_thread::sleep_for(1ms);
    }

    // Start a thread that triggers worker shutdown while the task is blocked.
    // This is the scenario where, in the buggy ordering, the owner would free
    // resources the task still references.
    std::thread shutdown_thread([&] {
      worker.shutdown_and_join();
      // If shutdown_and_join() respects the contract, the task has finished
      // before this point.
      task_completed_before_shutdown_returned.store(task_completed.load());
    });

    // Give the shutdown thread time to enter the join. shutdown_and_join must
    // not return yet — the task is still blocked.
    std::this_thread::sleep_for(50ms);
    REQUIRE(task_completed.load() == false);

    // Release the task. Now shutdown_and_join can return.
    {
      std::lock_guard<std::mutex> lock(latch_mutex);
      release_task = true;
    }
    latch_cv.notify_one();

    shutdown_thread.join();
    // The future was set inside the task; draining it must succeed.
    fut.get();
  }

  REQUIRE(task_completed.load() == true);
  REQUIRE(task_completed_before_shutdown_returned.load() == true);
}

// =============================================================================
// Regression test for owner-initiated shutdown ordering.
//
// Models the pipeline_io_backend destructor: the owner holds a buffer that
// the submitted task writes to. With the correct destruction sequence, the
// owner calls shutdown_and_join() *before* releasing the buffer, so the task
// finishes touching the buffer before it goes away.
// =============================================================================
TEST_CASE("io_worker lets owner safely free task-referenced state after shutdown",
          "[io_worker][regression]")
{
  // A heap-allocated buffer that the task writes to. With the correct
  // ordering, the owner's destruction frees this only after the worker has
  // joined, so the task's write has already completed.
  auto buffer      = std::make_unique<std::atomic<int>>(0);
  auto* buffer_ptr = buffer.get();

  std::atomic<bool> task_ran_to_completion{false};

  {
    io_worker worker;

    // Simulates pwrite on a pinned buffer: takes a non-trivial amount of time
    // and writes to the buffer the owner controls.
    auto fut = worker.submit([buffer_ptr, &task_ran_to_completion] {
      std::this_thread::sleep_for(20ms);
      buffer_ptr->store(123);
      task_ran_to_completion.store(true);
    });

    // Mimic an owner that wants to safely free the buffer: shutdown_and_join
    // first, then drop the buffer. The future-get isn't strictly required for
    // ordering (shutdown_and_join already waits), but mirrors real usage.
    worker.shutdown_and_join();
    fut.get();
  }

  // Now safe to free.
  REQUIRE(task_ran_to_completion.load() == true);
  REQUIRE(buffer->load() == 123);
  buffer.reset();
}
