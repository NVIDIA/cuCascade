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
#include <cucascade/io/cufile/file_write_state.hpp>

#include <sys/stat.h>

#include <limits>

namespace cucascade::io::cufile {
namespace {
struct registry {
  // Recursive only for allocation-unwind destruction during get().
  std::recursive_mutex mutex;
  struct entry {
    std::weak_ptr<file_write_state> weak;
    file_write_state* identity{};
  };
  std::map<std::pair<std::uint64_t, std::uint64_t>, entry> files;
};
registry& table()
{
  static auto* value = new registry;
  return *value;
}
}  // namespace
file_write_state::~file_write_state()
{
  auto& t = table();
  std::lock_guard lock(t.mutex);
  auto it = t.files.find(_identity);
  if (it != t.files.end() && it->second.identity == this) t.files.erase(it);
}
std::size_t file_write_state::registry_size_for_testing()
{
  auto& t = table();
  std::lock_guard lock(t.mutex);
  return t.files.size();
}
std::shared_ptr<file_write_state> file_write_state::get(int fd)
{
  struct stat st{};
  if (fstat(fd, &st) != 0)
    throw std::system_error(errno, std::generic_category(), "write identity");
  auto& t = table();
  std::lock_guard lock(t.mutex);
  auto const key =
    std::pair{static_cast<std::uint64_t>(st.st_dev), static_cast<std::uint64_t>(st.st_ino)};
  auto found = t.files.find(key);
  if (found != t.files.end())
    if (auto existing = found->second.weak.lock()) return existing;
  auto created       = std::make_shared<file_write_state>();
  created->_identity = key;
  t.files.insert_or_assign(key, registry::entry{created, created.get()});
  return created;
}
std::shared_ptr<file_write_state::ticket> file_write_state::begin()
{
  std::lock_guard lock(_mutex);
  if (_last == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("file write sequence exhausted");
  auto const seq = _last + 1;
  auto result    = std::shared_ptr<ticket>(new ticket{nullptr, seq});
  _pending.insert(seq);
  result->state = shared_from_this();
  _last         = seq;
  return result;
}
file_write_ticket::~file_write_ticket()
{
  if (state) state->retire(sequence);
}
file_write_state::interval::~interval()
{
  if (state) state->release(id);
}

std::shared_ptr<file_write_state::interval> file_write_state::try_acquire(
  range bytes, std::size_t granule, write_io_mode mode, std::function<void()> ready, bool legacy)
{
  if (bytes.empty() || granule == 0 ||
      bytes.size > std::numeric_limits<std::size_t>::max() - bytes.offset)
    throw std::invalid_argument("invalid write interval");
  std::unique_lock lock(_mutex);
  if (_granule && _granule != granule) throw std::invalid_argument("write granule changed");
  _granule = granule;
  // Activation and pre-engine activity registration are serialized by this mutex.
  // The first engine interval waits for these guards, without gating legacy peers.
  if (legacy && !_engine.load()) {
    auto result = std::shared_ptr<interval>(new interval{nullptr, 0});
    ++_legacy_active;
    result->state = shared_from_this();
    return result;
  }
  auto conflict = [&]() -> std::shared_ptr<interval> {
    ++_conflicts;
    if (ready) _interval_waiters.push_back({bytes, std::move(ready)});
    return {};
  };
  if (!legacy && _legacy_active) return conflict();
  auto const first = bytes.offset / granule;
  auto const last  = (bytes.end() - 1) / granule + 1;
  for (auto const& [id, r] : _intervals) {
    static_cast<void>(id);
    bool const same_mode     = mode == r.mode && mode != write_io_mode::mixed;
    bool const bytes_overlap = bytes.offset < r.bytes.end() && r.bytes.offset < bytes.end();
    if (first < r.granules.end() && r.granules.offset < last && (!same_mode || bytes_overlap)) {
      return conflict();
    }
  }
  auto result = std::shared_ptr<interval>(new interval{nullptr, ++_interval_id});
  _intervals.emplace(result->id, active_interval{{first, last - first}, bytes, mode});
  result->state = shared_from_this();
  return result;
}
std::shared_ptr<file_write_state::interval> file_write_state::acquire_legacy(range bytes,
                                                                             std::size_t granule,
                                                                             write_io_mode mode)
{
  for (;;) {
    std::unique_lock lock(_mutex);
    auto const epoch = _epoch;
    lock.unlock();
    if (auto result = try_acquire(bytes, granule, mode, {}, true)) return result;
    lock.lock();
    _changed.wait(lock, [&] { return _epoch != epoch; });
  }
}
void file_write_state::release(std::uint64_t id)
{
  std::list<interval_waiter> ready;
  {
    std::lock_guard lock(_mutex);
    range released{};
    if (id == 0) {
      --_legacy_active;
    } else {
      released = _intervals.at(id).granules;
      _intervals.erase(id);
    }
    for (auto it = _interval_waiters.begin(); it != _interval_waiters.end();) {
      auto current     = it++;
      auto const first = current->bytes.offset / _granule;
      auto const last  = (current->bytes.end() - 1) / _granule + 1;
      if (id == 0 || (first < released.end() && released.offset < last))
        ready.splice(ready.end(), _interval_waiters, current);
    }
    ++_epoch;
  }
  _changed.notify_all();
  for (auto& waiter : ready) {
    try {
      waiter.ready();
    } catch (...) { /* Notification fallback is the idle health wake. */
    }
  }
}
std::uint64_t file_write_state::capture() const
{
  std::lock_guard lock(_mutex);
  return _last;
}
bool file_write_state::when_retired(std::uint64_t seq, std::function<void()> ready)
{
  {
    std::lock_guard lock(_mutex);
    if (!_pending.empty() && *_pending.begin() <= seq) {
      _waiters.emplace(seq, std::move(ready));
      return true;
    }
  }
  ready();
  return false;
}
void file_write_state::retire(std::uint64_t sequence)
{
  std::unique_lock lock(_mutex);
  _pending.erase(sequence);
  while (!_waiters.empty() && (_pending.empty() || *_pending.begin() > _waiters.begin()->first)) {
    auto callback = std::move(_waiters.begin()->second);
    _waiters.erase(_waiters.begin());
    lock.unlock();
    callback();
    lock.lock();
  }
}
}  // namespace cucascade::io::cufile
