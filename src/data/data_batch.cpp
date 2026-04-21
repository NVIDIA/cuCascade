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
}

uint64_t data_batch::get_batch_id() const { return _batch_id; }

bool data_batch::subscribe()
{
  _subscriber_count.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void data_batch::unsubscribe()
{
  size_t prev = _subscriber_count.fetch_sub(1, std::memory_order_relaxed);
  if (prev == 0) {
    _subscriber_count.fetch_add(1, std::memory_order_relaxed);
    throw std::runtime_error("Cannot unsubscribe: subscriber count is already zero");
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
  auto ptr = std::move(accessor._batch);
  ptr->_state.store(batch_state::idle, std::memory_order_relaxed);
  accessor._lock.unlock();
  return ptr;
}

std::shared_ptr<data_batch> data_batch::to_idle(mutable_data_batch&& accessor)
{
  auto ptr = std::move(accessor._batch);
  ptr->_state.store(batch_state::idle, std::memory_order_relaxed);
  accessor._lock.unlock();
  return ptr;
}

// ========== Non-static transition methods ==========

read_only_data_batch data_batch::to_read_only()
{
  auto self = shared_from_this();
  std::shared_lock<std::shared_mutex> lock(_rw_mutex);
  _state.store(batch_state::read_only, std::memory_order_relaxed);
  return read_only_data_batch(std::move(self), std::move(lock));
}

mutable_data_batch data_batch::to_mutable()
{
  auto self = shared_from_this();
  std::unique_lock<std::shared_mutex> lock(_rw_mutex);
  _state.store(batch_state::mutable_locked, std::memory_order_relaxed);
  return mutable_data_batch(std::move(self), std::move(lock));
}

std::optional<read_only_data_batch> data_batch::try_to_read_only()
{
  std::shared_lock<std::shared_mutex> lock(_rw_mutex, std::try_to_lock);
  if (!lock.owns_lock()) { return std::nullopt; }
  _state.store(batch_state::read_only, std::memory_order_relaxed);
  auto self = shared_from_this();
  return read_only_data_batch(std::move(self), std::move(lock));
}

std::optional<mutable_data_batch> data_batch::try_to_mutable()
{
  std::unique_lock<std::shared_mutex> lock(_rw_mutex, std::try_to_lock);
  if (!lock.owns_lock()) { return std::nullopt; }
  _state.store(batch_state::mutable_locked, std::memory_order_relaxed);
  auto self = shared_from_this();
  return mutable_data_batch(std::move(self), std::move(lock));
}

// ========== Locked-to-locked static transitions ==========

mutable_data_batch data_batch::readonly_to_mutable(read_only_data_batch&& accessor)
{
  auto ptr = std::move(accessor._batch);
  accessor._lock.unlock();
  std::unique_lock<std::shared_mutex> lock(ptr->_rw_mutex);
  ptr->_state.store(batch_state::mutable_locked, std::memory_order_relaxed);
  return mutable_data_batch(std::move(ptr), std::move(lock));
}

read_only_data_batch data_batch::mutable_to_readonly(mutable_data_batch&& accessor)
{
  auto ptr = std::move(accessor._batch);
  accessor._lock.unlock();
  std::shared_lock<std::shared_mutex> lock(ptr->_rw_mutex);
  ptr->_state.store(batch_state::read_only, std::memory_order_relaxed);
  return read_only_data_batch(std::move(ptr), std::move(lock));
}

// ========== read_only_data_batch ==========

read_only_data_batch::read_only_data_batch(std::shared_ptr<data_batch> parent,
                                           std::shared_lock<std::shared_mutex> lock)
  : _batch(std::move(parent)), _lock(std::move(lock))
{
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
}

std::shared_ptr<data_batch> mutable_data_batch::clone(uint64_t new_batch_id,
                                                      rmm::cuda_stream_view stream) const
{
  if (_batch->_data == nullptr) { throw std::runtime_error("Cannot clone: data is null"); }
  auto cloned_data = _batch->_data->clone(stream);
  return std::make_shared<data_batch>(new_batch_id, std::move(cloned_data));
}

}  // namespace cucascade
