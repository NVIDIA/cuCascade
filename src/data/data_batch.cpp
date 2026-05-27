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

namespace cucascade {

// ========== data_batch implementation ==========

data_batch::data_batch(uint64_t batch_id, std::unique_ptr<idata_representation> data)
  : _batch_id(batch_id), _data(std::move(data))
{
  if (_data == nullptr) { throw std::runtime_error("data is null in data_batch constructor"); }
}

uint64_t data_batch::get_batch_id() const { return _batch_id; }

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

size_t data_batch::get_subscriber_count() const
{
  return _subscriber_count.load(std::memory_order_relaxed);
}

// ========== data_batch private data accessors ==========

memory::Tier data_batch::get_current_tier() const { return _data->get_current_tier(); }

idata_representation* data_batch::get_data() const { return _data.get(); }

memory::memory_space* data_batch::get_memory_space() const
{
  if (_data == nullptr) { return nullptr; }
  return &_data->get_memory_space();
}

void data_batch::set_data(std::unique_ptr<idata_representation> data) { _data = std::move(data); }

// ========== Static transition methods ==========

std::shared_ptr<data_batch> data_batch::to_idle(read_only_data_batch&& accessor)
{
  auto ptr = accessor._batch;
  {
    auto _ = std::move(accessor);
  }  // destroy accessor, releasing shared lock
  return ptr;
}

std::shared_ptr<data_batch> data_batch::to_idle(mutable_data_batch&& accessor)
{
  auto ptr = accessor._batch;
  {
    auto _ = std::move(accessor);
  }  // destroy accessor, releasing exclusive lock
  return ptr;
}

// ========== Non-static transition methods ==========

read_only_data_batch data_batch::to_read_only()
{
  auto self = shared_from_this();
  std::shared_lock<std::shared_mutex> lock(_rw_mutex);
  return read_only_data_batch(std::move(self), std::move(lock));
}

mutable_data_batch data_batch::to_mutable()
{
  auto self = shared_from_this();
  std::unique_lock<std::shared_mutex> lock(_rw_mutex);
  return mutable_data_batch(std::move(self), std::move(lock));
}

std::optional<read_only_data_batch> data_batch::try_to_read_only()
{
  std::shared_lock<std::shared_mutex> lock(_rw_mutex, std::try_to_lock);
  if (!lock.owns_lock()) { return std::nullopt; }
  auto self = shared_from_this();
  return read_only_data_batch(std::move(self), std::move(lock));
}

std::optional<mutable_data_batch> data_batch::try_to_mutable()
{
  std::unique_lock<std::shared_mutex> lock(_rw_mutex, std::try_to_lock);
  if (!lock.owns_lock()) { return std::nullopt; }
  auto self = shared_from_this();
  return mutable_data_batch(std::move(self), std::move(lock));
}

// ========== Locked-to-locked static transitions ==========

mutable_data_batch data_batch::readonly_to_mutable(read_only_data_batch&& accessor)
{
  auto ptr = accessor._batch;
  {
    // destructor decrements _read_only_count, releases the shared lock and sets state to idle
    auto _ = std::move(accessor);  // move into temporary, destroyed at }
  }
  std::unique_lock<std::shared_mutex> lock(ptr->_rw_mutex);
  return mutable_data_batch(std::move(ptr), std::move(lock));
}

read_only_data_batch data_batch::mutable_to_readonly(mutable_data_batch&& accessor)
{
  auto ptr = accessor._batch;
  {
    // destructor frees the exclusive lock and sets state to idle
    auto _ = std::move(accessor);  // move into temporary, destroyed at }
  }
  std::shared_lock<std::shared_mutex> lock(ptr->_rw_mutex);
  return read_only_data_batch(std::move(ptr), std::move(lock));
}

// ========== read_only_data_batch ==========

read_only_data_batch::read_only_data_batch(std::shared_ptr<data_batch> parent,
                                           std::shared_lock<std::shared_mutex> lock)
  : _batch(std::move(parent)), _lock(std::move(lock))
{
  _batch->_read_only_count.fetch_add(1);
  _batch->_state.store(batch_state::read_only);
}

read_only_data_batch::read_only_data_batch(read_only_data_batch&& other) noexcept
  : _batch(std::move(other._batch)), _lock(std::move(other._lock))
{
  // other._batch is now nullptr — other's destructor will be a no-op.
  // The read_only_count does NOT change: ownership was transferred, not a new reader created.
}

read_only_data_batch::read_only_data_batch(const read_only_data_batch& other)
  : _batch(other._batch),
    _lock(other._batch ? std::shared_lock<std::shared_mutex>(other._batch->_rw_mutex)
                       : std::shared_lock<std::shared_mutex>())
{
  if (_batch) { _batch->_read_only_count.fetch_add(1); }
}

read_only_data_batch& read_only_data_batch::operator=(read_only_data_batch&& other) noexcept
{
  if (this != &other) {
    // Release the current state (same logic as destructor)
    if (_batch) {
      auto prev = _batch->_read_only_count.fetch_sub(1);
      if (prev == 1) { _batch->_state.store(batch_state::idle); }
      // _lock will be replaced below; its destructor fires when the old _lock is overwritten,
      // releasing the shared lock. We release _lock explicitly here so the sequence is:
      // decrement count -> set state (if last) -> release lock.
      _lock.unlock();
    }
    _batch = std::move(other._batch);
    _lock  = std::move(other._lock);
  }
  return *this;
}

read_only_data_batch& read_only_data_batch::operator=(const read_only_data_batch& other)
{
  if (this != &other) {
    if (_batch) {
      auto prev = _batch->_read_only_count.fetch_sub(1);
      if (prev == 1) { _batch->_state.store(batch_state::idle); }
      _lock.unlock();
    }
    _batch = other._batch;
    if (_batch) {
      _lock = std::shared_lock<std::shared_mutex>(_batch->_rw_mutex);
      _batch->_read_only_count.fetch_add(1);
      _batch->_state.store(batch_state::read_only);
    } else {
      _lock = std::shared_lock<std::shared_mutex>();
    }
  }
  return *this;
}

read_only_data_batch::~read_only_data_batch()
{
  if (_batch) {
    // Decrement the reader count. If we were the last reader, transition to idle.
    // NOTE: Do NOT call _lock.unlock() here — the _lock member destructor handles that.
    // The destructor body runs before member destructors, so _batch is still valid here.
    // After this function returns, _lock destructor fires first (declared after _batch,
    // destroyed in reverse order), releasing the shared lock. Then _batch destructor fires.
    auto prev = _batch->_read_only_count.fetch_sub(1);
    if (prev == 1) { _batch->_state.store(batch_state::idle); }
  }
}

std::shared_ptr<data_batch> read_only_data_batch::clone(uint64_t new_batch_id,
                                                        rmm::cuda_stream_view stream) const
{
  if (_batch->_data == nullptr) { throw std::runtime_error("Cannot clone: data is null"); }
  auto cloned_data = _batch->_data->clone(stream);
  return std::make_shared<data_batch>(new_batch_id, std::move(cloned_data));
}

// ========== mutable_data_batch ==========

mutable_data_batch::mutable_data_batch(std::shared_ptr<data_batch> parent,
                                       std::unique_lock<std::shared_mutex> lock)
  : _batch(std::move(parent)), _lock(std::move(lock))
{
  _batch->_state.store(batch_state::mutable_locked);
}

mutable_data_batch::mutable_data_batch(mutable_data_batch&& other) noexcept
  : _batch(std::move(other._batch)), _lock(std::move(other._lock))
{
  // other._batch is now nullptr — other's destructor will be a no-op.
}

mutable_data_batch& mutable_data_batch::operator=(mutable_data_batch&& other) noexcept
{
  if (this != &other) {
    // Release the current state (same logic as destructor)
    if (_batch) {
      _batch->_state.store(batch_state::idle);
      // Release the exclusive lock explicitly before taking ownership of the new one.
      _lock.unlock();
    }
    _batch = std::move(other._batch);
    _lock  = std::move(other._lock);
  }
  return *this;
}

mutable_data_batch::~mutable_data_batch()
{
  if (_batch) {
    // Transition state to idle. The _lock member destructor handles releasing the exclusive lock.
    _batch->_state.store(batch_state::idle);
  }
}

std::shared_ptr<data_batch> mutable_data_batch::clone(uint64_t new_batch_id,
                                                      rmm::cuda_stream_view stream) const
{
  if (_batch->_data == nullptr) { throw std::runtime_error("Cannot clone: data is null"); }
  auto cloned_data = _batch->_data->clone(stream);
  return std::make_shared<data_batch>(new_batch_id, std::move(cloned_data));
}

}  // namespace cucascade
