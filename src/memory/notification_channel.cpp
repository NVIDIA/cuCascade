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

namespace cucascade {
namespace memory {

//===----------------------------------------------------------------------===//
// notification_channel::event_notifier
//===----------------------------------------------------------------------===//

notification_channel::event_notifier::event_notifier(notification_channel& channel)
  : _channel(channel.shared_from_this())
{
}

notification_channel::event_notifier::~event_notifier() { _channel->release_notifier(); }

void notification_channel::event_notifier::post() { _channel->notify(); }

//===----------------------------------------------------------------------===//
// notification_channel
//===----------------------------------------------------------------------===//

notification_channel::~notification_channel() { shutdown(); }

notification_channel::wait_status notification_channel::wait()
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

std::unique_ptr<notification_channel::event_notifier> notification_channel::get_notifier()
{
  return std::make_unique<event_notifier>(*this);
}

void notification_channel::shutdown()
{
  std::lock_guard lock(_mutex);
  _is_running = false;
  _cv.notify_one();
}

void notification_channel::notify()
{
  std::lock_guard lock(_mutex);
  _has_been_notified = true;
  _cv.notify_one();
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
