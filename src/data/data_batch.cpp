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
  return &_data->get_memory_space();
}

void data_batch::set_state_change_cv(std::condition_variable* cv)
{
  std::lock_guard<std::mutex> lock(_mutex);
  _state_change_cv = cv;
}

void data_batch::set_data(std::unique_ptr<idata_representation> data)
{
  std::lock_guard<std::mutex> lock(_mutex);
  if (_processing_count != 0) {
    throw std::runtime_error("Cannot set data while there is active processing");
  }
  _data = std::move(data);
}

bool data_batch::try_to_create_task()
{
  std::condition_variable* cv_to_notify = nullptr;
  bool should_notify                    = false;
  bool success                          = false;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_state == batch_state::idle) {
      _state = batch_state::task_created;
      ++_task_created_count;
      should_notify = true;
      cv_to_notify  = _state_change_cv;
      success       = true;
    } else if (_state == batch_state::task_created) {
      ++_task_created_count;
      success = true;
    } else if (_state == batch_state::processing) {
      // Batch is already processing, increment counter but stay in processing state
      ++_task_created_count;
      success = true;
    }
  }
  if (should_notify && cv_to_notify) { cv_to_notify->notify_all(); }
  return success;
}

size_t data_batch::get_task_created_count() const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _task_created_count;
}

bool data_batch::try_to_cancel_task()
{
  std::condition_variable* cv_to_notify = nullptr;
  bool should_notify                    = false;
  bool success                          = false;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_state == batch_state::task_created || _state == batch_state::processing) {
      if (_task_created_count == 0) {
        throw std::runtime_error(
          "Cannot cancel task: task_created_count is zero. "
          "try_to_create_task() must be called before try_to_cancel_task()");
      }
      --_task_created_count;
      if (_task_created_count == 0 && _processing_count == 0) {
        _state        = batch_state::idle;
        should_notify = true;
        cv_to_notify  = _state_change_cv;
      }
      success = true;
    }
  }
  if (should_notify && cv_to_notify) { cv_to_notify->notify_all(); }
  return success;
}
lock_for_processing_result data_batch::try_to_lock_for_processing(
  memory::memory_space_id requested_memory_space)
{
  std::condition_variable* cv_to_notify = nullptr;
  bool should_notify                    = false;
  lock_for_processing_result result{
    false, data_batch_processing_handle{}, lock_for_processing_status::not_attempted};
  {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_data == nullptr) {
      result.status = lock_for_processing_status::missing_data;
      return result;
    }

    if (_data->get_memory_space().get_id() != requested_memory_space) {
      result.status = lock_for_processing_status::memory_space_mismatch;
      return result;
    }

    if (_task_created_count == 0) {
      result.status = lock_for_processing_status::task_not_created;
      return result;
    }

    if (!(_state == batch_state::task_created || _state == batch_state::processing)) {
      result.status = lock_for_processing_status::invalid_state;
      return result;
    }
    --_task_created_count;
    ++_processing_count;
    _state        = batch_state::processing;
    should_notify = true;
    cv_to_notify  = _state_change_cv;
    result        = {true, data_batch_processing_handle{this}, lock_for_processing_status::success};
  }
  if (should_notify && cv_to_notify) { cv_to_notify->notify_all(); }
  return result;
}

bool data_batch::try_to_lock_for_in_transit()
{
  std::condition_variable* cv_to_notify = nullptr;
  bool should_notify                    = false;
  bool success                          = false;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_processing_count == 0 &&
        ((_state == batch_state::idle) ||
         (_state == batch_state::task_created && _task_created_count > 0))) {
      _state        = batch_state::in_transit;
      should_notify = true;
      cv_to_notify  = _state_change_cv;
      success       = true;
    }
  }
  if (should_notify && cv_to_notify) { cv_to_notify->notify_all(); }
  return success;
}

bool data_batch::try_to_release_in_transit(std::optional<batch_state> target_state)
{
  std::condition_variable* cv_to_notify = nullptr;
  bool should_notify                    = false;
  bool success                          = false;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_state == batch_state::in_transit) {
      // Caller can explicitly choose the state to return to; default is idle.
      if (target_state.has_value()) {
        _state = *target_state;
      } else {
        _state = batch_state::idle;
      }
      should_notify = true;
      cv_to_notify  = _state_change_cv;
      success       = true;
    }
  }
  if (should_notify && cv_to_notify) { cv_to_notify->notify_all(); }
  return success;
}

void data_batch::decrement_processing_count()
{
  std::condition_variable* cv_to_notify = nullptr;
  bool should_notify                    = false;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_state != batch_state::processing) {
      throw std::runtime_error(
        "Cannot decrement processing count: batch is not in processing state");
    }
    if (_processing_count == 0) {
      throw std::runtime_error(
        "Cannot decrement processing count: processing count is already zero");
    }
    _processing_count -= 1;
    if (_processing_count == 0) {
      // Preserve pending task_created intent if any remain
      _state        = (_task_created_count > 0) ? batch_state::task_created : batch_state::idle;
      should_notify = true;
      cv_to_notify  = _state_change_cv;
    }
  }
  if (should_notify && cv_to_notify) { cv_to_notify->notify_all(); }
}

std::shared_ptr<data_batch> data_batch::clone(uint64_t new_batch_id)
{
  // Create a task and lock for processing to protect data during clone
  if (!try_to_create_task()) {
    throw std::runtime_error(
      "Cannot clone data_batch: failed to create task (batch may be in transit)");
  }

  auto space_id = _data->get_memory_space().get_id();
  auto result   = try_to_lock_for_processing(space_id);
  if (!result.success) {
    throw std::runtime_error("Cannot clone data_batch: failed to lock for processing");
  }

  // Clone the data while holding the processing lock
  auto cloned_data = _data->clone();

  // Handle destructor will decrement processing count when result goes out of scope
  return std::make_shared<data_batch>(new_batch_id, std::move(cloned_data));
}

}  // namespace cucascade
