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

#include <cucascade/memory/stream_pool.hpp>

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

borrowed_stream::~borrowed_stream() noexcept { reset(); }

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

  for (std::size_t i = 0; i < pool_size; ++i) {
    _streams.emplace_back(rmm::cuda_stream(_flags));
  }
}

borrowed_stream exclusive_stream_pool::acquire_stream(stream_acquire_policy policy) noexcept
{
  std::unique_lock lock(_mutex);
  if (policy == stream_acquire_policy::GROW) {
    // GROW never waits — and never takes a pooled stream a parked BLOCK waiter is owed
    // (_grant_ticket != _next_ticket means at least one waiter is still unserved). Minting a
    // fresh stream costs the same either way: the pool ends up one stream larger once it is
    // released.
    if (_streams.empty() || _grant_ticket != _next_ticket) {
      rmm::cuda_set_device_raii set_device{_device_id};
      return borrowed_stream(rmm::cuda_stream(_flags),
                             std::bind_front(&exclusive_stream_pool::release_stream, this));
    }
  } else {
    // FIFO ticket handoff: a released stream goes to the LONGEST-WAITING caller. The ticket is
    // drawn under the lock, so arrival order is the service order; a caller that releases and
    // immediately re-acquires draws a fresh ticket and queues behind every parked waiter
    // instead of winning the wake-up race against them. When the pool has streams and no
    // earlier waiter is unserved, the predicate is immediately true and nothing blocks.
    const std::uint64_t ticket = _next_ticket++;
    _cv.wait(lock, [&]() { return ticket == _grant_ticket && !_streams.empty(); });
    ++_grant_ticket;
  }
  // Acquire from the front; release_stream() returns to the back. This cycles through all
  // streams round-robin so every stream's prior async work has maximal time to drain before
  // it is handed out again, instead of hammering the most-recently-returned stream.
  auto stream = std::move(_streams.front());
  _streams.pop_front();
  // A single release wakes every waiter (they must all re-check the head ticket); if streams
  // remain for the next-in-line, pass the baton before leaving so it does not sleep until the
  // next release.
  if (!_streams.empty() && _grant_ticket != _next_ticket) { _cv.notify_all(); }
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
  // notify_all, not notify_one: only the head-ticket waiter may take the stream, and with one
  // shared CV a notify_one could wake a non-head waiter that just goes back to sleep while the
  // head never learns a stream arrived.
  _cv.notify_all();
}

}  // namespace memory
}  // namespace cucascade
