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

#include <cucascade/io/concurrent_queue.hpp>
#include <cucascade/io/types.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace cucascade::io {

/**
 * @brief Two-tier blocking MPMC queue: a dequeue always prefers the high tier.
 *
 * Two non-blocking moodycamel queues (one per @ref io_priority tier) share one
 * lightweight semaphore that counts the items across both, so a consumer can
 * block on "anything queued" yet still take high-priority items first.  The
 * semaphore is signalled only after an item is visible in its queue, so a
 * consumer holding a permit always finds an item in one of the tiers.
 *
 * Ordering is FIFO within a tier (per producer, as moodycamel guarantees) and
 * strictly high-before-low across tiers: a low item is taken only when the
 * high tier is observed empty.
 */
template <typename T>
class tiered_blocking_queue {
 public:
  /// Queue @p item on @p priority's tier (@c automatic counts as high).
  /// Returns false when moodycamel could not allocate; @p item is then intact.
  bool enqueue(T&& item, io_priority priority)
  {
    auto& tier = priority == io_priority::low ? _low : _high;
    if (!tier.enqueue(std::move(item))) return false;
    _items.signal();
    return true;
  }

  /// Take the next item, high tier first, without blocking.
  bool try_dequeue(T& item)
  {
    if (!_items.tryWait()) return false;
    take_any(item);
    return true;
  }

  /// Take the next item from the high tier only, without blocking; low items
  /// are left in place.
  bool try_dequeue_high(T& item)
  {
    if (_high.size_approx() == 0 || !_items.tryWait()) return false;
    if (_high.try_dequeue(item)) return true;
    // Another consumer won the high item; hand the permit back for whichever
    // item it now accounts for.
    _items.signal();
    return false;
  }

  /// Take the next item, high tier first, waiting up to @p timeout_us
  /// microseconds (negative waits forever).
  bool wait_dequeue_timed(T& item, std::int64_t timeout_us)
  {
    if (!_items.wait(timeout_us)) return false;
    take_any(item);
    return true;
  }

  [[nodiscard]] std::size_t size_approx() const { return _high.size_approx() + _low.size_approx(); }

  [[nodiscard]] std::size_t size_approx(io_priority priority) const
  {
    return priority == io_priority::low ? _low.size_approx() : _high.size_approx();
  }

 private:
  /// Spend a permit already acquired: the item it counts is visible in one of
  /// the tiers, though a racing consumer may shift which one, so retry both.
  void take_any(T& item)
  {
    while (!_high.try_dequeue(item) && !_low.try_dequeue(item)) {}
  }

  concurrent_queue<T> _high;
  concurrent_queue<T> _low;
  lightweight_semaphore _items;
};

}  // namespace cucascade::io
