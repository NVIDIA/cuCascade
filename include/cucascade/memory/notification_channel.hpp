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
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <utility>

namespace cucascade {
namespace memory {

struct notification_channel : std::enable_shared_from_this<notification_channel> {
  struct event_notifier {
    explicit event_notifier(notification_channel& channel);
    ~event_notifier();

    void post();

   private:
    std::shared_ptr<notification_channel> _channel;
  };

  enum class wait_status { IDLE, NOTIFIED, SHUTDOWN };

  /**
   * @brief RAII registration in the channel's FIFO wait list.
   *
   * A blocking caller registers ONCE for the whole wait (taking a ticket under the channel
   * lock) and then calls wait() as many times as its retry loop needs. wait() only returns
   * NOTIFIED to the waiter at the HEAD of the list, so a release notification is offered to
   * the longest-waiting caller first — never to whichever waiter happens to win the wake-up
   * race. A caller whose retry fails keeps its ticket (it stays head); a caller that is done
   * destroys the waiter, which removes the ticket and passes the baton to the next in line.
   *
   * IDLE and SHUTDOWN are delivered to every waiter regardless of position: they mean no
   * notification can be forthcoming, so queue order is moot.
   */
  class scoped_waiter {
   public:
    explicit scoped_waiter(notification_channel& channel);
    ~scoped_waiter();

    scoped_waiter(const scoped_waiter&)            = delete;
    scoped_waiter& operator=(const scoped_waiter&) = delete;
    scoped_waiter(scoped_waiter&&)                 = delete;
    scoped_waiter& operator=(scoped_waiter&&)      = delete;

    /// @brief True when no earlier-registered waiter is still unserved.
    [[nodiscard]] bool is_head() const;

    /**
     * @brief Block until this waiter is at the head of the list AND a notification arrived
     * (NOTIFIED), or no notification can be forthcoming (IDLE / SHUTDOWN).
     */
    wait_status wait();

   private:
    std::shared_ptr<notification_channel> _channel;
    std::uint64_t _ticket;
  };

  ~notification_channel();

  /**
   * @brief True when at least one scoped_waiter is registered.
   *
   * Used by blocking callers to keep a fresh arrival from barging past parked waiters via a
   * non-blocking fast path: when waiters exist, join the FIFO instead.
   */
  [[nodiscard]] bool has_waiters() const;

  std::unique_ptr<event_notifier> get_notifier();

  void shutdown();

 private:
  void notify();

  void release_notifier();

  void acquire_notifier();

  mutable std::mutex _mutex;
  std::condition_variable _cv;
  bool _has_been_notified{false};
  std::size_t _n_active_notifiers{0};
  bool _is_running{true};
  /// Live waiter tickets in arrival order; front() is the only waiter NOTIFIED may go to.
  std::deque<std::uint64_t> _wait_queue;
  std::uint64_t _next_ticket{0};
};

using event_notifier = notification_channel::event_notifier;

struct notify_on_exit {
  explicit notify_on_exit(std::unique_ptr<event_notifier> notifier);
  ~notify_on_exit() noexcept;

 private:
  std::unique_ptr<event_notifier> _notifier;
};

}  // namespace memory
}  // namespace cucascade
