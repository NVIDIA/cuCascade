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

#pragma once

#include <cucascade/io/details/event_fd.hpp>
#include <cucascade/io/types.hpp>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace cucascade::io::detail {

/**
 * @brief Per-runner state shared between the runner thread and everyone who
 *        needs to wake it.
 *
 * Created by @ref runner_registry::register_runner on the runner thread and
 * handed to the backend engine (@c Reactor::make_engine).  Owns the runner's
 * wakeup eventfd.  Wakeups are delivered by @ref notify (any thread):
 *  - the request hub, when it publishes work while this runner is parked;
 *  - the stop callbacks installed by @c templated_ioctx::run_impl (user token,
 *    context shutdown) -- so a stop is always observed promptly;
 *  - the engine itself or its CUDA host callbacks, if it wants to.
 *
 * The engine waits on @ref wake_fd together with its own completion source
 * (epoll set / an io_uring POLL_ADD -- not a read SQE: the eventfd is
 * non-blocking, so an io_uring read completes at once with -EAGAIN) and
 * clears it with @ref consume_notifications.
 */
class runner_slot {
 public:
  /// @throws std::system_error if the eventfd cannot be created.
  runner_slot(std::uint64_t id, std::thread::id tid);

  runner_slot(runner_slot const&)            = delete;
  runner_slot& operator=(runner_slot const&) = delete;

  /// Registry-unique runner id (> 0), stored in @c request_meta::runner_id.
  [[nodiscard]] std::uint64_t id() const noexcept { return _id; }

  /// Thread that registered (and drives) this runner.
  [[nodiscard]] std::thread::id thread_id() const noexcept { return _tid; }

  /// The wakeup eventfd (non-blocking, close-on-exec).  Readable when notified.
  [[nodiscard]] int wake_fd() const noexcept { return _wake_fd.get(); }

  /// Signal the wakeup eventfd.  Thread-safe; never blocks.
  void notify() noexcept;

  /// Reset the wakeup eventfd; returns the number of notifications collapsed.
  std::uint64_t consume_notifications() noexcept { return drain_event_fd(_wake_fd.get()); }

  /// Whether the runner announced it is (about to be) blocked and wants work.
  /// Managed by the request hub's park protocol; engines do not write it.
  [[nodiscard]] bool parked() const noexcept { return _parked.load(std::memory_order_seq_cst); }

  /// Grouped requests currently owned by this runner (maintained by the hub).
  [[nodiscard]] std::size_t active_groups() const noexcept
  {
    return _active_groups.load(std::memory_order_relaxed);
  }

  /// Grouped requests this runner retired (any terminal state).
  [[nodiscard]] std::size_t retired_groups() const noexcept
  {
    return _retired_groups.load(std::memory_order_relaxed);
  }

  /// Total @ref notify calls (observability; lets tests detect wake storms).
  [[nodiscard]] std::uint64_t notifications_sent() const noexcept
  {
    return _notifications.load(std::memory_order_relaxed);
  }

 private:
  friend class runner_registry;
  friend class request_hub;

  std::uint64_t const _id;
  std::thread::id const _tid;
  file_descriptor _wake_fd;
  std::atomic<bool> _parked{false};
  std::atomic<std::size_t> _active_groups{0};
  std::atomic<std::size_t> _retired_groups{0};
  std::atomic<std::uint64_t> _notifications{0};
};

/**
 * @brief The set of threads currently inside @c run*() of one context.
 *
 * Owned by the request hub of a reactor.  Provides registration (one slot per
 * runner thread; a thread may hold at most one), targeted wakeups of parked
 * runners, and a condition variable so @c shutdown() can wait for externally
 * driven runners to return.
 *
 * Park protocol (implemented by @c request_hub::try_park / @c unpark on top of
 * @ref park / @ref unpark / @ref wake_one_parked):
 *  - runner: @c park(slot) (publishes @c parked=true and bumps the parked
 *    counter, seq_cst), then re-checks the queue; if work is visible it
 *    @c unpark()s and continues, otherwise it blocks on its eventfd.
 *  - submitter: publishes the request (seq_cst counter), then
 *    @c wake_one_parked(): a lock-free check of the parked counter, and if it
 *    is non-zero, round-robin over the slots claiming one with a CAS on
 *    @c parked (true -> false) and notifying its eventfd.  At most one wakeup
 *    per parked runner, so no wake storms.
 *
 * Thread-safety: all members may be called concurrently.
 */
class runner_registry {
 public:
  runner_registry() = default;

  runner_registry(runner_registry const&)            = delete;
  runner_registry& operator=(runner_registry const&) = delete;

  /**
   * @brief Register the calling thread as a runner.
   *
   * @return The new slot (also retained by the registry until
   *         @ref unregister_runner).
   * @throws std::logic_error if the calling thread is already registered
   *         (nested @c run*() on the same context).
   * @throws std::system_error if the eventfd cannot be created.
   */
  [[nodiscard]] std::shared_ptr<runner_slot> register_runner();

  /// Remove @p slot (clears its parked flag) and notify @ref wait_until_at_most waiters.
  void unregister_runner(runner_slot& slot) noexcept;

  /// Mark @p slot parked (idempotent).  See the class comment.
  void park(runner_slot& slot) noexcept;

  /// Clear @p slot's parked flag if still set (idempotent).
  void unpark(runner_slot& slot) noexcept;

  /**
   * @brief Wake one parked runner, if any.
   *
   * Lock-free when nobody is parked.  The claimed runner's @c parked flag is
   * cleared before its eventfd is signalled.
   *
   * @return Whether a runner was woken.
   */
  bool wake_one_parked() noexcept;

  /// Notify every registered runner (parked or not), e.g. on shutdown.
  void wake_all() noexcept;

  /// Block until at most @p count runners remain registered.
  void wait_until_at_most(std::size_t count) noexcept;

  /// Block until no runner remains registered.
  void wait_until_empty() noexcept { wait_until_at_most(0); }

  /// Whether thread @p tid is a registered runner.
  [[nodiscard]] bool is_registered(std::thread::id tid) const noexcept;

  /// Number of registered runners.
  [[nodiscard]] std::size_t size() const noexcept;

  /// Number of parked runners (approximate).
  [[nodiscard]] std::size_t parked_count() const noexcept
  {
    return _parked_count.load(std::memory_order_seq_cst);
  }

  /// Number of parked runners that own no grouped request (truly idle).
  [[nodiscard]] std::size_t idle_count() const noexcept;

 private:
  mutable std::mutex _mutex;
  std::condition_variable _cv;
  std::vector<std::shared_ptr<runner_slot>> _slots;  // guarded by _mutex
  std::size_t _round_robin{0};                       // guarded by _mutex
  std::uint64_t _next_id{1};                         // guarded by _mutex
  std::atomic<std::size_t> _parked_count{0};
};

}  // namespace cucascade::io::detail
