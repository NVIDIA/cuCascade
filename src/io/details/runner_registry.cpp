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

#include <cucascade/io/details/event_fd.hpp>
#include <cucascade/io/details/runner_registry.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace cucascade::io::detail {

// ---------------------------------------------------------------------------
// runner_slot
// ---------------------------------------------------------------------------

runner_slot::runner_slot(std::uint64_t id, std::thread::id tid)
  : _id(id), _tid(tid), _wake_fd(make_event_fd())
{
}

void runner_slot::notify() noexcept
{
  _notifications.fetch_add(1, std::memory_order_relaxed);
  static_cast<void>(signal_event_fd(_wake_fd.get()));
}

// ---------------------------------------------------------------------------
// runner_registry
// ---------------------------------------------------------------------------

std::shared_ptr<runner_slot> runner_registry::register_runner()
{
  auto const tid = std::this_thread::get_id();
  std::lock_guard lock(_mutex);
  for (auto const& slot : _slots) {
    if (slot->thread_id() == tid) {
      throw std::logic_error("io: run*() called on a thread already running this context");
    }
  }
  auto slot = std::make_shared<runner_slot>(_next_id, tid);
  _slots.push_back(slot);
  ++_next_id;
  return slot;
}

void runner_registry::unregister_runner(runner_slot& slot) noexcept
{
  unpark(slot);
  {
    std::lock_guard lock(_mutex);
    auto const it = std::find_if(
      _slots.begin(), _slots.end(), [&](auto const& entry) { return entry.get() == &slot; });
    if (it != _slots.end()) _slots.erase(it);
  }
  _cv.notify_all();
}

void runner_registry::park(runner_slot& slot) noexcept
{
  if (!slot._parked.exchange(true, std::memory_order_seq_cst)) {
    _parked_count.fetch_add(1, std::memory_order_seq_cst);
  }
}

void runner_registry::unpark(runner_slot& slot) noexcept
{
  if (slot._parked.exchange(false, std::memory_order_seq_cst)) {
    _parked_count.fetch_sub(1, std::memory_order_seq_cst);
  }
}

bool runner_registry::wake_one_parked() noexcept
{
  if (_parked_count.load(std::memory_order_seq_cst) == 0) return false;

  std::lock_guard lock(_mutex);
  auto const count = _slots.size();
  for (std::size_t step = 0; step < count; ++step) {
    auto& slot     = *_slots[(_round_robin + step) % count];
    bool expected  = true;
    bool const won = slot._parked.compare_exchange_strong(
      expected, false, std::memory_order_seq_cst, std::memory_order_seq_cst);
    if (!won) continue;
    _parked_count.fetch_sub(1, std::memory_order_seq_cst);
    _round_robin = (_round_robin + step + 1) % count;
    slot.notify();
    return true;
  }
  return false;
}

void runner_registry::wake_all() noexcept
{
  std::lock_guard lock(_mutex);
  for (auto const& slot : _slots) {
    slot->notify();
  }
}

void runner_registry::wait_until_at_most(std::size_t count) noexcept
{
  std::unique_lock lock(_mutex);
  _cv.wait(lock, [&] { return _slots.size() <= count; });
}

bool runner_registry::is_registered(std::thread::id tid) const noexcept
{
  std::lock_guard lock(_mutex);
  return std::any_of(
    _slots.begin(), _slots.end(), [&](auto const& slot) { return slot->thread_id() == tid; });
}

std::size_t runner_registry::size() const noexcept
{
  std::lock_guard lock(_mutex);
  return _slots.size();
}

std::size_t runner_registry::idle_count() const noexcept
{
  std::lock_guard lock(_mutex);
  return static_cast<std::size_t>(std::count_if(_slots.begin(), _slots.end(), [](auto const& slot) {
    return slot->parked() && slot->active_groups() == 0;
  }));
}

}  // namespace cucascade::io::detail
