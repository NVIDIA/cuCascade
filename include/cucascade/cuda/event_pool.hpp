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

#pragma once

#include <cucascade/cuda/event.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <mutex>
#include <vector>

namespace cucascade {
namespace cuda {

/**
 * @brief Thread-safe pool of recorded CUDA events tracking outstanding asynchronous work.
 *
 * Callers record() an event on the stream where they just enqueued work; later, another
 * party either enqueues device-side waits on all outstanding events via enqueue_waits()
 * (no host sync) or blocks the host until they complete via synchronize().
 *
 * Events are created lazily with cudaEventDisableTiming and recycled once
 * cudaEventQuery reports completion, so steady-state record() calls perform no event
 * creation. A default-constructed pool holds no events and no heap allocations — users
 * that never record pay only the (empty) containers and the mutex.
 *
 * All public methods are internally synchronized by a single mutex. Note that
 * synchronize() blocks the host while holding that mutex, so concurrent record() calls
 * are delayed until it returns; callers coordinating record vs. reclaim through an
 * external lock (e.g. data_batch's reader-writer lock) never hit this case.
 */
class event_pool {
 public:
  event_pool()  = default;
  ~event_pool() = default;

  // Non-copyable / non-movable: the pool is referenced concurrently by recorders and
  // waiters, so its address must remain stable.
  event_pool(const event_pool&)            = delete;
  event_pool& operator=(const event_pool&) = delete;
  event_pool(event_pool&&)                 = delete;
  event_pool& operator=(event_pool&&)      = delete;

  /**
   * @brief Record an event on @p stream and track it as outstanding.
   *
   * Takes a recycled event from the pool when one is available, otherwise creates a new
   * timing-disabled event. Cheap: one mutex lock, at most one cudaEventCreateWithFlags
   * (amortized away by recycling), and one cudaEventRecord. Never host-syncs.
   *
   * @param stream The stream whose currently-enqueued work the event captures.
   */
  void record(rmm::cuda_stream_view stream);

  /**
   * @brief Enqueue device-side waits on @p stream for every outstanding event.
   *
   * Issues cudaStreamWaitEvent(@p stream, event) for each event that has not yet
   * completed; work enqueued on @p stream afterwards is ordered after all captured
   * work. Does NOT block the host. Events observed complete are recycled.
   *
   * @param stream The stream on which to enqueue the waits.
   */
  void enqueue_waits(rmm::cuda_stream_view stream);

  /**
   * @brief Block the host until every outstanding event has completed.
   *
   * All events are recycled on return. No-op (a mutex lock) when nothing is
   * outstanding.
   */
  void synchronize();

  /**
   * @brief True if no outstanding event is still pending on the device.
   *
   * Non-blocking: polls each outstanding event with cudaEventQuery. Only events whose
   * query reports success count as done (a query error is conservatively treated as
   * still pending).
   */
  [[nodiscard]] bool is_done() const;

 private:
  /// Move every completed outstanding event to the available pool. Must hold @c mutex_.
  void recycle_completed_locked();

  mutable std::mutex mutex_;
  std::vector<cuda_event> outstanding_;  ///< Recorded events, possibly still pending
  std::vector<cuda_event> available_;    ///< Completed events awaiting reuse
};

}  // namespace cuda
}  // namespace cucascade
