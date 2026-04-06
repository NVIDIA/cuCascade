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
#include <cucascade/data/gpu_data_representation.hpp>

namespace cucascade {

// =============================================================================
// data_batch (inner) implementation
// =============================================================================

synchronized_data_batch::data_batch::data_batch(uint64_t batch_id,
                                                std::unique_ptr<idata_representation> data)
  : _batch_id(batch_id), _data(std::move(data))
{
}

uint64_t synchronized_data_batch::data_batch::get_batch_id() const { return _batch_id; }

memory::Tier synchronized_data_batch::data_batch::get_current_tier() const
{
  return _data->get_current_tier();
}

idata_representation* synchronized_data_batch::data_batch::get_data() const { return _data.get(); }

cucascade::memory::memory_space* synchronized_data_batch::data_batch::get_memory_space() const
{
  if (_data == nullptr) { return nullptr; }
  return &_data->get_memory_space();
}

void synchronized_data_batch::data_batch::set_data(std::unique_ptr<idata_representation> data)
{
  _data = std::move(data);
}

// =============================================================================
// read_only_data_batch implementation
// =============================================================================

synchronized_data_batch::read_only_data_batch
synchronized_data_batch::read_only_data_batch::from_mutable(mutable_data_batch&& rw)
{
  auto* parent = rw.parent_;
  auto* mtx    = rw.lock_.mutex();
  rw.lock_.unlock();
  std::shared_lock<std::shared_mutex> lock(*mtx);
  return read_only_data_batch(parent, std::move(lock));
}

// =============================================================================
// mutable_data_batch implementation
// =============================================================================

synchronized_data_batch::mutable_data_batch
synchronized_data_batch::mutable_data_batch::from_read_only(read_only_data_batch&& ro)
{
  auto* parent = ro.parent_;
  auto* mtx    = ro.lock_.mutex();
  ro.lock_.unlock();
  std::unique_lock<std::shared_mutex> lock(*mtx);
  return mutable_data_batch(parent, std::move(lock));
}

// =============================================================================
// synchronized_data_batch implementation
// =============================================================================

synchronized_data_batch::synchronized_data_batch(uint64_t batch_id,
                                                 std::unique_ptr<idata_representation> data)
  : _batch(batch_id, std::move(data))
{
}

synchronized_data_batch::synchronized_data_batch(synchronized_data_batch&& other)
  : _batch(std::move(other._batch))
{
}

synchronized_data_batch& synchronized_data_batch::operator=(synchronized_data_batch&& other)
{
  if (this != &other) { _batch = std::move(other._batch); }
  return *this;
}

// Lock acquisition (blocking)

synchronized_data_batch::read_only_data_batch synchronized_data_batch::get_read_only()
{
  std::shared_lock<std::shared_mutex> lock(_rw_mutex);
  return read_only_data_batch(this, std::move(lock));
}

synchronized_data_batch::mutable_data_batch synchronized_data_batch::get_mutable()
{
  std::unique_lock<std::shared_mutex> lock(_rw_mutex);
  return mutable_data_batch(this, std::move(lock));
}

// Lock acquisition (non-blocking)

std::optional<synchronized_data_batch::read_only_data_batch>
synchronized_data_batch::try_get_read_only()
{
  std::shared_lock<std::shared_mutex> lock(_rw_mutex, std::try_to_lock);
  if (!lock.owns_lock()) { return std::nullopt; }
  return read_only_data_batch(this, std::move(lock));
}

std::optional<synchronized_data_batch::mutable_data_batch>
synchronized_data_batch::try_get_mutable()
{
  std::unique_lock<std::shared_mutex> lock(_rw_mutex, std::try_to_lock);
  if (!lock.owns_lock()) { return std::nullopt; }
  return mutable_data_batch(this, std::move(lock));
}

// Immutable field (lock-free)

uint64_t synchronized_data_batch::get_batch_id() const { return _batch._batch_id; }

// Subscriber count

bool synchronized_data_batch::subscribe()
{
  _subscriber_count.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void synchronized_data_batch::unsubscribe()
{
  size_t prev = _subscriber_count.fetch_sub(1, std::memory_order_relaxed);
  if (prev == 0) {
    _subscriber_count.fetch_add(1, std::memory_order_relaxed);
    throw std::runtime_error("Cannot unsubscribe: subscriber count is already zero");
  }
}

size_t synchronized_data_batch::get_subscriber_count() const
{
  return _subscriber_count.load(std::memory_order_relaxed);
}

// Clone

std::shared_ptr<synchronized_data_batch> synchronized_data_batch::clone(
  uint64_t new_batch_id, rmm::cuda_stream_view stream)
{
  std::shared_lock<std::shared_mutex> lock(_rw_mutex);

  if (_batch._data == nullptr) {
    throw std::runtime_error("Cannot clone: data is null");
  }

  auto cloned_data = _batch._data->clone(stream);
  return std::make_shared<synchronized_data_batch>(new_batch_id, std::move(cloned_data));
}

}  // namespace cucascade
