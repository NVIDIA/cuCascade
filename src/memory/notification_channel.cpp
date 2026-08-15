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

#include <cucascade/memory/notification_channel.hpp>

#include <algorithm>

namespace cucascade {
namespace memory {

//===----------------------------------------------------------------------===//
// notification_channel::event_notifier
//===----------------------------------------------------------------------===//

notification_channel::event_notifier::event_notifier(notification_channel& channel)
  : _channel(channel.shared_from_this())
{
  _channel->acquire_notifier();
}

notification_channel::event_notifier::~event_notifier() { _channel->release_notifier(); }

void notification_channel::event_notifier::post() { _channel->notify(); }

void notification_channel::acquire_notifier()
{
  std::lock_guard lock(_mutex);
  _n_active_notifiers++;
}

//===----------------------------------------------------------------------===//
// notification_channel::scoped_waiter
//===----------------------------------------------------------------------===//

notification_channel::scoped_waiter::scoped_waiter(notification_channel& channel)
  : _channel(channel.shared_from_this())
{
  std::lock_guard lock(_channel->_mutex);
  _ticket = _channel->_next_ticket++;
  _channel->_wait_queue.push_back(_ticket);
}

notification_channel::scoped_waiter::~scoped_waiter()
{
  auto& ch = *_channel;
  {
    std::lock_guard lock(ch._mutex);
    auto it = std::find(ch._wait_queue.begin(), ch._wait_queue.end(), _ticket);
    if (it != ch._wait_queue.end()) { ch._wait_queue.erase(it); }
    if (ch._wait_queue.empty()) { return; }
    // Pass the baton: this waiter usually leaves because its reservation was just granted, so
    // the space's state changed and the new head must re-check it — a release that arrived
    // during our retry was coalesced into the single notified flag we may have consumed. A
    // fabricated notification costs the new head one cheap failed retry at worst; NOT waking it
    // costs a stall until the next release.
    ch._has_been_notified = true;
  }
  ch._cv.notify_all();
}

bool notification_channel::scoped_waiter::is_head() const
{
  std::lock_guard lock(_channel->_mutex);
  return !_channel->_wait_queue.empty() && _channel->_wait_queue.front() == _ticket;
}

notification_channel::wait_status notification_channel::scoped_waiter::wait()
{
  auto& ch = *_channel;
  std::unique_lock lock(ch._mutex);
  bool notified = false;
  ch._cv.wait(lock, [&] {
    // IDLE / SHUTDOWN break every waiter out regardless of position (nothing further can be
    // posted). Only the HEAD may consume a notification: consuming it from the middle of the
    // queue is precisely the wake-up race the FIFO exists to remove.
    if (!ch._is_running || ch._n_active_notifiers == 0) { return true; }
    if (!ch._wait_queue.empty() && ch._wait_queue.front() == _ticket) {
      notified = std::exchange(ch._has_been_notified, false);
      return notified;
    }
    return false;
  });
  return !ch._is_running ? wait_status::SHUTDOWN
         : (notified)    ? wait_status::NOTIFIED
                         : wait_status::IDLE;
}

//===----------------------------------------------------------------------===//
// notification_channel
//===----------------------------------------------------------------------===//

notification_channel::~notification_channel() { shutdown(); }

bool notification_channel::has_waiters() const
{
  std::lock_guard lock(_mutex);
  return !_wait_queue.empty();
}

std::unique_ptr<notification_channel::event_notifier> notification_channel::get_notifier()
{
  return std::make_unique<event_notifier>(*this);
}

void notification_channel::shutdown()
{
  std::lock_guard lock(_mutex);
  _is_running = false;
  // Every parked waiter must observe the shutdown, not just one.
  _cv.notify_all();
}

void notification_channel::notify()
{
  std::lock_guard lock(_mutex);
  _has_been_notified = true;
  // notify_all so the HEAD waiter definitely wakes: with one shared CV a notify_one can land on
  // a non-head waiter, which re-checks its position and goes back to sleep while the head never
  // learns memory was released. Waiter counts per space are small, so the extra wake-ups are
  // noise.
  _cv.notify_all();
}

void notification_channel::release_notifier()
{
  std::lock_guard lock(_mutex);
  _n_active_notifiers--;
  if (_n_active_notifiers == 0) { _cv.notify_all(); }
}

//===----------------------------------------------------------------------===//
// notify_on_exit
//===----------------------------------------------------------------------===//

notify_on_exit::notify_on_exit(std::unique_ptr<event_notifier> notifier)
  : _notifier(std::move(notifier))
{
}

notify_on_exit::~notify_on_exit() noexcept
{
  try {
    if (_notifier) _notifier->post();
  } catch (...) {
  }
}

}  // namespace memory
}  // namespace cucascade
