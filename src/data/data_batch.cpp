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

#include "data/data_batch.hpp"

#include "data/gpu_data_representation.hpp"
#include "memory/memory_reservation_manager.hpp"

namespace cucascade {

// data_batch_processing_handle implementation

data_batch_processing_handle::~data_batch_processing_handle() { release(); }

void data_batch_processing_handle::release()
{
  if (_batch != nullptr) {
    _batch->decrement_processing_count();
    _batch = nullptr;
  }
}

// data_batch implementation

data_batch::data_batch(uint64_t batch_id, std::unique_ptr<idata_representation> data)
  : _batch_id(batch_id), _data(std::move(data))
{
}

data_batch::data_batch(data_batch&& other)
  : _batch_id(other._batch_id), _data(std::move(other._data))
{
  std::lock_guard<std::mutex> lock(other._mutex);
  size_t other_processing_count = other._processing_count;
  if (other_processing_count != 0) {
    throw std::runtime_error(
      "Cannot move data_batch with active processing (processing_count != 0)");
  }
  other._batch_id = 0;
  other._data     = nullptr;
}

data_batch& data_batch::operator=(data_batch&& other)
{
  if (this != &other) {
    std::lock_guard<std::mutex> lock(other._mutex);
    size_t other_processing_count = other._processing_count;
    if (other_processing_count != 0) {
      throw std::runtime_error(
        "Cannot move data_batch with active processing (processing_count != 0)");
    }
    _batch_id       = other._batch_id;
    _data           = std::move(other._data);
    other._batch_id = 0;
    other._data     = nullptr;
  }
  return *this;
}

memory::Tier data_batch::get_current_tier() const { return _data->get_current_tier(); }

uint64_t data_batch::get_batch_id() const { return _batch_id; }

batch_state data_batch::get_state() const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _state;
}

size_t data_batch::get_processing_count() const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _processing_count;
}

idata_representation* data_batch::get_data() const { return _data.get(); }

cucascade::memory::memory_space* data_batch::get_memory_space() const
{
  if (_data == nullptr) { return nullptr; }
  auto& manager = cucascade::memory::memory_reservation_manager::get_instance();
  auto* space   = manager.get_memory_space(_data->get_current_tier(), _data->get_device_id());
  return const_cast<cucascade::memory::memory_space*>(space);
}

void data_batch::set_data(std::unique_ptr<idata_representation> data)
{
  std::lock_guard<std::mutex> lock(_mutex);
  if (_processing_count != 0) {
    throw std::runtime_error("Cannot set data while there is active processing");
  }
  _data = std::move(data);
}

void data_batch::convert_to_memory_space(const cucascade::memory::memory_space* target_memory_space,
                                         rmm::cuda_stream_view stream)
{
  std::lock_guard<std::mutex> lock(_mutex);

  if (_processing_count != 0) {
    throw std::runtime_error("Cannot convert memory space while there is active processing");
  }
  auto new_representation = _data->convert_to_memory_space(target_memory_space, stream);
  _data                   = std::move(new_representation);
}

bool data_batch::try_to_lock_for_processing()
{
  std::lock_guard<std::mutex> lock(_mutex);
  if (_state == batch_state::at_rest || _state == batch_state::processing) {
    _processing_count += 1;
    _state = batch_state::processing;
    return true;
  }
  return false;
}

bool data_batch::try_to_lock_for_downgrade()
{
  std::lock_guard<std::mutex> lock(_mutex);
  if (_processing_count == 0 && _state == batch_state::at_rest) {
    _state = batch_state::downgrading;
    return true;
  }
  return false;
}

void data_batch::decrement_processing_count()
{
  std::lock_guard<std::mutex> lock(_mutex);
  if (_state != batch_state::processing) {
    throw std::runtime_error("Cannot decrement processing count: batch is not in processing state");
  }
  if (_processing_count == 0) {
    throw std::runtime_error("Cannot decrement processing count: processing count is already zero");
  }
  _processing_count -= 1;
  if (_processing_count == 0) { _state = batch_state::at_rest; }
}

}  // namespace cucascade
