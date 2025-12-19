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
#include <memory>
#include <mutex>
#include <utility>
namespace cucascade {
namespace memory {

struct notification_channel : std::enable_shared_from_this<notification_channel> {
  struct event_notifier {
    explicit event_notifier(notification_channel& channel) : _channel(channel.shared_from_this()) {}

    ~event_notifier() { _channel->release_notifier(); }

    void post() { _channel->notify(); }

   private:
    std::shared_ptr<notification_channel> _channel;
  };

  enum class wait_status { IDLE, NOTIFIED, SHUTDOWN };

  ~notification_channel() { shutdown(); }

  wait_status wait()
  {
    std::unique_lock lock(_mutex);
    bool notified = false;
    _cv.wait(lock, [&, self = shared_from_this()] {
      notified = std::exchange(_has_been_notified, false);
      return notified || (_n_active_notifiers == 0) || not _is_running;
    });
    return !_is_running ? wait_status::SHUTDOWN
           : (notified) ? wait_status::NOTIFIED
                        : wait_status::IDLE;
  }

  std::unique_ptr<event_notifier> get_notifier() { return std::make_unique<event_notifier>(*this); }

  void shutdown()
  {
    std::lock_guard lock(_mutex);
    _is_running = false;
    _cv.notify_one();
  }

 private:
  void notify()
  {
    std::lock_guard lock(_mutex);
    _has_been_notified = true;
    _cv.notify_one();
  }

  void release_notifier()
  {
    std::lock_guard lock(_mutex);
    _n_active_notifiers--;
    if (_n_active_notifiers == 0) { _cv.notify_all(); }
  }

  mutable std::mutex _mutex;
  std::condition_variable _cv;
  bool _has_been_notified{false};
  std::size_t _n_active_notifiers{0};
  bool _is_running{true};
};

using event_notifier = notification_channel::event_notifier;

struct notify_on_exit {
  explicit notify_on_exit(std::unique_ptr<event_notifier> notifier) : _notifier(std::move(notifier))
  {
  }

  ~notify_on_exit() noexcept
  {
    try {
      if (_notifier) _notifier->post();
    } catch (...) {
    }
  }

 private:
  std::unique_ptr<event_notifier> _notifier;
};

}  // namespace memory
}  // namespace cucascade
