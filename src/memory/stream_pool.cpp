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

#include "memory/stream_pool.hpp"

#include <rmm/cuda_device.hpp>

#include <functional>
#include <mutex>
#include <utility>

namespace cucascade {
namespace memory {

borrowed_stream::borrowed_stream(rmm::cuda_stream s,
                                 std::function<void(rmm::cuda_stream&&)> release_fn) noexcept
  : _stream(std::move(s)), _release_fn(std::move(release_fn))
{
}

borrowed_stream::~borrowed_stream() = default;

borrowed_stream::borrowed_stream(borrowed_stream&& other) noexcept
  : _stream(std::move(other._stream)), _release_fn(std::exchange(other._release_fn, nullptr))
{
}

borrowed_stream& borrowed_stream::operator=(borrowed_stream&& other) noexcept
{
  if (this != &other) {
    reset();
    _stream     = std::move(other._stream);
    _release_fn = std::exchange(other._release_fn, nullptr);
  }
  return *this;
}

borrowed_stream::operator rmm::cuda_stream_view() const { return _stream; }

void borrowed_stream::reset() noexcept
{
  if (_release_fn) { std::exchange(_release_fn, nullptr)(std::move(_stream)); }
}

rmm::cuda_stream_view borrowed_stream::get() const noexcept { return _stream; }
const rmm::cuda_stream* borrowed_stream::operator->() const noexcept { return &_stream; }
const rmm::cuda_stream* borrowed_stream::operator->() noexcept { return &_stream; }

exclusive_stream_pool::exclusive_stream_pool(rmm::cuda_device_id device_id,
                                             std::size_t pool_size,
                                             rmm::cuda_stream::flags flags)
  : _device_id(device_id), _flags(flags)
{
  rmm::cuda_set_device_raii set_device{_device_id};
  if (pool_size == 0) { throw std::logic_error("Stream pool size must be greater than zero"); }

  _streams.reserve(pool_size);
  for (std::size_t i = 0; i < pool_size; ++i) {
    _streams.emplace_back(rmm::cuda_stream(_flags));
  }
}

borrowed_stream exclusive_stream_pool::acquire_stream(stream_acquire_policy policy) noexcept
{
  std::unique_lock lock(_mutex);
  if (_streams.empty()) {
    if (policy == stream_acquire_policy::GROW) {
      rmm::cuda_set_device_raii set_device{_device_id};
      return borrowed_stream(rmm::cuda_stream(_flags),
                             std::bind_front(&exclusive_stream_pool::release_stream, this));
    } else {
      _cv.wait(lock, [this]() { return !_streams.empty(); });
    }
  }
  auto stream = std::move(_streams.back());
  _streams.pop_back();
  return borrowed_stream(std::move(stream),
                         std::bind_front(&exclusive_stream_pool::release_stream, this));
}

std::size_t exclusive_stream_pool::size() const noexcept
{
  std::lock_guard lock(_mutex);
  return _streams.size();
}

void exclusive_stream_pool::release_stream(rmm::cuda_stream&& s) noexcept
{
  std::lock_guard lock(_mutex);
  _streams.emplace_back(std::move(s));
  _cv.notify_one();
}

}  // namespace memory
}  // namespace cucascade
