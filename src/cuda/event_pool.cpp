/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cucascade/cuda/event_pool.hpp>

#include <cassert>
#include <mutex>
#include <utility>

namespace cucascade {
namespace cuda {

void event_pool::record(rmm::cuda_stream_view stream)
{
  std::lock_guard<std::mutex> lock(mutex_);
  // Opportunistically recycle before taking from the pool so steady-state usage
  // converges to a handful of events instead of growing without bound.
  recycle_completed_locked();
  if (available_.empty()) { available_.emplace_back(cudaEventDisableTiming); }
  cuda_event event = std::move(available_.back());
  available_.pop_back();
  event.record(stream);
  outstanding_.push_back(std::move(event));
}

void event_pool::enqueue_waits(rmm::cuda_stream_view stream)
{
  std::lock_guard<std::mutex> lock(mutex_);
  recycle_completed_locked();
  // Still-pending events stay outstanding after the wait is enqueued: the wait only
  // snapshots the captured work, it does not retire the event. They are recycled by a
  // later call once cudaEventQuery reports completion.
  for (auto& event : outstanding_) {
    event.wait(stream);
  }
}

void event_pool::synchronize()
{
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& event : outstanding_) {
    event.synchronize();
  }
  for (auto& event : outstanding_) {
    available_.push_back(std::move(event));
  }
  outstanding_.clear();
}

void event_pool::synchronize_no_throw() noexcept
{
  std::lock_guard<std::mutex> lock(mutex_);
  // No recycling: pushing into available_ may allocate, which a noexcept
  // destructor context must not risk. Completed events stay outstanding and are
  // recycled by the next record() / enqueue_waits().
  for (auto& event : outstanding_) {
    // Destructor discipline (as CUCASCADE_ASSERT_CUDA_SUCCESS): assert in debug
    // builds, discard the error in release — never throw.
    [[maybe_unused]] cudaError_t const status = event.synchronize_no_throw();
    assert(status == cudaSuccess);
  }
}

bool event_pool::is_done() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& event : outstanding_) {
    if (event.query() != event::query_result::success) { return false; }
  }
  return true;
}

void event_pool::recycle_completed_locked()
{
  auto it = outstanding_.begin();
  while (it != outstanding_.end()) {
    if (it->query() == event::query_result::success) {
      available_.push_back(std::move(*it));
      it = outstanding_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace cuda
}  // namespace cucascade
