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

#include <cucascade/data/data_batch.hpp>

#include <memory>

namespace cucascade {

// ========== data_batch_core ==========

uint64_t data_batch_core::get_batch_id() const { return _batch_id; }

memory::Tier data_batch_core::get_current_tier() const { return _data->get_current_tier(); }

const idata_representation* data_batch_core::get_data() const { return _data.get(); }

const memory::memory_space* data_batch_core::get_memory_space() const
{
  return &(_data->get_memory_space());
}

[[nodiscard]] std::shared_ptr<data_batch> data_batch_core::clone(uint64_t new_batch_id,
                                                                 rmm::cuda_stream_view stream) const
{
  auto cloned_data = _data->clone(stream);
  return data_batch::make(new_batch_id, std::move(cloned_data));
}

data_batch_core::data_batch_core(uint64_t batch_id, std::unique_ptr<idata_representation> data)
  : _batch_id(batch_id), _data(std::move(data))
{
}

// ========== data_batch ==========

std::shared_ptr<data_batch> data_batch::make(uint64_t batch_id,
                                             std::unique_ptr<idata_representation> data)
{
  if (data == nullptr) { throw std::runtime_error("data is null in data_batch constructor"); }
  return std::shared_ptr<data_batch>(new data_batch(batch_id, std::move(data)));
}

read_only_data_batch data_batch::get_read_only()
{
  auto self = shared_from_this();
  std::shared_lock<std::shared_mutex> lock(_rw_mutex);
  return read_only_data_batch(std::move(self), std::move(lock));
}

mutable_data_batch data_batch::get_mutable()
{
  auto self = shared_from_this();
  std::unique_lock<std::shared_mutex> lock(_rw_mutex);
  return mutable_data_batch(std::move(self), std::move(lock));
}

std::optional<read_only_data_batch> data_batch::try_get_read_only()
{
  std::shared_lock<std::shared_mutex> lock(_rw_mutex, std::try_to_lock);
  if (!lock.owns_lock()) { return std::nullopt; }
  auto self = shared_from_this();
  return read_only_data_batch(std::move(self), std::move(lock));
}

std::optional<mutable_data_batch> data_batch::try_get_mutable()
{
  std::unique_lock<std::shared_mutex> lock(_rw_mutex, std::try_to_lock);
  if (!lock.owns_lock()) { return std::nullopt; }
  auto self = shared_from_this();
  return mutable_data_batch(std::move(self), std::move(lock));
}

void data_batch::subscribe() { _subscriber_count.fetch_add(1, std::memory_order_relaxed); }

void data_batch::unsubscribe()
{
  size_t current = _subscriber_count.load(std::memory_order_relaxed);
  while (true) {
    if (current == 0) {
      throw std::runtime_error("Cannot unsubscribe: subscriber count is already zero");
    }
    if (_subscriber_count.compare_exchange_weak(
          current, current - 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
      return;
    }
  }
}

data_batch::data_batch(uint64_t batch_id, std::unique_ptr<idata_representation> data)
  : _batch(batch_id, std::move(data))
{
}

// ========== read_only_data_batch ==========

read_only_data_batch::~read_only_data_batch()
{
  // Decrement the reader count. If we were the last reader, transition to idle.
  // NOTE: Do NOT call _lock.unlock() here — the _lock member destructor handles that.
  // The destructor body runs before member destructors, so _owner is still valid here.
  // After this function returns, _lock destructor fires first (declared after _owner,
  // destroyed in reverse order), releasing the shared lock. Then _owner destructor fires.
  if (_owner) {  // the read_only_data_batch may have been moved.
    size_t prev = _owner->_read_only_count.fetch_sub(1);
    if (prev == 1) { _owner->_state.store(batch_state::idle); }
  }
}

read_only_data_batch::read_only_data_batch(read_only_data_batch&& other) noexcept
  : _owner(std::move(other._owner)), _lock(std::move(other._lock))
{
  // other._owner is now nullptr — other's destructor will be a no-op.
}

read_only_data_batch& read_only_data_batch::operator=(read_only_data_batch&& other) noexcept
{
  if (this != &other) {
    // Release the current state (same logic as destructor)
    if (_owner) {
      size_t prev = _owner->_read_only_count.fetch_sub(1);
      if (prev == 1) { _owner->_state.store(batch_state::idle); }
      // _lock will be replaced below; its destructor fires when the old _lock is overwritten,
      // releasing the shared lock. We release _lock explicitly here so the sequence is:
      // decrement count -> set state (if last) -> release lock.
      _lock.unlock();
    }
    _owner = std::move(other._owner);
    _lock  = std::move(other._lock);
    // other._owner is now nullptr — other's destructor will be a no-op.
  }
  return *this;
}

read_only_data_batch::read_only_data_batch(std::shared_ptr<data_batch> owner,
                                           std::shared_lock<std::shared_mutex> lock)
  : _owner(std::move(owner)), _lock(std::move(lock))
{
  _owner->_read_only_count.fetch_add(1);
  _owner->_state.store(batch_state::read_only);
}

// ========== mutable_data_batch ==========

mutable_data_batch::~mutable_data_batch()
{
  if (_owner) {  // mutable_data_batch may have been moved
    // Transition state to idle. The _lock member destructor handles releasing the exclusive lock.
    _owner->_state.store(batch_state::idle);
  }
}

mutable_data_batch::mutable_data_batch(mutable_data_batch&& other) noexcept
  : _owner(std::move(other._owner)), _lock(std::move(other._lock))
{
  // other._owner is now nullptr — other's destructor will be a no-op.
}

mutable_data_batch& mutable_data_batch::operator=(mutable_data_batch&& other) noexcept
{
  if (this != &other) {
    // Release the current state (same logic as destructor)
    if (_owner) {
      _owner->_state.store(batch_state::idle);
      // Release the exclusive lock explicitly before taking ownership of the new one.
      _lock.unlock();
    }
    _owner = std::move(other._owner);
    _lock  = std::move(other._lock);
    // other._owner is now nullptr — other's destructor will be a no-op.
  }
  return *this;
}

mutable_data_batch::mutable_data_batch(std::shared_ptr<data_batch> owner,
                                       std::unique_lock<std::shared_mutex> lock)
  : _owner(std::move(owner)), _lock(std::move(lock))
{
  _owner->_state.store(batch_state::mutable_locked);
}

}  // namespace cucascade
