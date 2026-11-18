/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <cucascade/io/cufile/mount_policy.hpp>
#include <cucascade/io/types.hpp>

#include <condition_variable>
#include <functional>
#include <list>
#include <map>
#include <set>

namespace cucascade::io::cufile {

enum class write_io_mode { buffered, direct, mixed };
class file_write_state;
struct file_write_ticket {
  std::shared_ptr<file_write_state> state;
  std::uint64_t sequence;
  ~file_write_ticket();
};

/** Process-wide inode ordering. Pending sequences include resource-free, queued writes.
 * Cross-mode granule overlaps and same-mode byte overlaps are excluded. A legacy
 * head/body/tail slice can hold a conservative mixed-mode interval.
 * Only real overlaps park; the short bookkeeping critical section uses a blocking lock.
 * Before inode activation legacy writes take only an activity guard, not an interval.
 */
class file_write_state : public std::enable_shared_from_this<file_write_state> {
 public:
  using ticket = file_write_ticket;
  ~file_write_state();
  [[nodiscard]] static std::size_t registry_size_for_testing();
  struct interval {
    std::shared_ptr<file_write_state> state;
    std::uint64_t id;
    ~interval();
  };
  [[nodiscard]] static std::shared_ptr<file_write_state> get(int fd);
  [[nodiscard]] std::shared_ptr<ticket> begin();
  [[nodiscard]] std::shared_ptr<interval> try_acquire(range bytes,
                                                      std::size_t granule,
                                                      write_io_mode mode = write_io_mode::mixed,
                                                      std::function<void()> ready = {},
                                                      bool legacy                 = false);
  [[nodiscard]] std::shared_ptr<interval> acquire_legacy(range bytes,
                                                         std::size_t granule,
                                                         write_io_mode mode);
  void activate() noexcept { _engine.store(true); }
  void quarantine_interval() noexcept { ++_quarantined; }
  [[nodiscard]] std::size_t quarantined_intervals() const noexcept { return _quarantined.load(); }
  [[nodiscard]] bool has_engine_work() const noexcept { return _engine.load(); }
  [[nodiscard]] std::uint64_t capture() const;
  /// Returns true when the callback had to wait. Callbacks run outside the registry mutex.
  bool when_retired(std::uint64_t sequence, std::function<void()> ready);
  [[nodiscard]] std::uint64_t conflicts() const noexcept { return _conflicts.load(); }

 private:
  friend struct file_write_ticket;
  std::pair<std::uint64_t, std::uint64_t> _identity{};
  void retire(std::uint64_t sequence);
  void release(std::uint64_t id);
  mutable std::mutex _mutex;
  std::uint64_t _last{}, _interval_id{};
  std::size_t _granule{}, _legacy_active{};
  std::condition_variable _changed;
  std::uint64_t _epoch{};
  struct interval_waiter {
    range bytes;
    std::function<void()> ready;
  };
  std::list<interval_waiter> _interval_waiters;
  std::atomic<std::size_t> _quarantined{};
  std::set<std::uint64_t> _pending;
  struct active_interval {
    range granules;
    range bytes;
    write_io_mode mode;
  };
  std::map<std::uint64_t, active_interval> _intervals;
  std::multimap<std::uint64_t, std::function<void()>> _waiters;
  std::atomic<bool> _engine{};
  std::atomic<std::uint64_t> _conflicts{};
};
}  // namespace cucascade::io::cufile
