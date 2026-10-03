/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <stop_token>

namespace cucascade::io::detail {
class runner_slot;
}  // namespace cucascade::io::detail

namespace cucascade::io::uring {

class uring_reactor;

/**
 * @brief Per-runner io_uring event loop (the @c engine_type of @c uring_reactor).
 *
 * Created by @c uring_reactor::make_engine on the runner thread, driven by one
 * call to @ref run on that thread and destroyed there.  Owns everything that is
 * thread-confined: the io_uring (SINGLE_ISSUER | COOP_TASKRUN | DEFER_TASKRUN,
 * falling back to default flags), the pinned bounce staging (allocated from the
 * context's host memory resource in the constructor, registered as fixed
 * buffers, returned in the destructor), the in-flight operations and the
 * grouped requests this runner pulled from the reactor's request hub.
 *
 * Several grouped requests are expanded concurrently (bounded by the
 * @c detail::scheduling_policy group limits, which count only groups with
 * undispatched work; groups whose operations are all submitted are bounded by
 * the staging slots they hold); each keeps its own planned-but-unsubmitted
 * operations, so cancelling one group never touches another.  The runner's
 * wakeup eventfd is watched by a poll SQE on the ring, so new work, stop
 * requests and context shutdown wake a runner blocked in the CQE wait.
 *
 * See @c io_engine_c for the run / leave contract.
 */
class uring_engine {
 public:
  using clock = std::chrono::steady_clock;

  /// Longest single wait when nothing is in flight (new work and stop
  /// requests arrive through the eventfd; the timeout is only a safety net).
  static constexpr std::chrono::milliseconds idle_timeout{1000};
  /// While CUDA copies out of the staging blocks are outstanding (they signal
  /// no CQE) the wait is bounded: it starts at @ref copy_poll_min and doubles
  /// on every wait without a completed copy, up to @ref copy_poll_max.
  static constexpr std::chrono::microseconds copy_poll_min{4};
  static constexpr std::chrono::microseconds copy_poll_max{1000};
  /// Retry period while work is visible but cannot be started right now
  /// (scheduling budget, queued work of a class this runner may not pull).
  static constexpr std::chrono::milliseconds blocked_retry_interval{1};

  /**
   * @brief Build the engine: pinned staging, ring, fixed buffers, wake poll.
   *
   * @param owner The reactor (shared configuration, context and request hub).
   * @param slot The runner's slot (wakeup eventfd, hub bookkeeping).
   * @throws std::invalid_argument if the staging block size is zero.
   * @throws std::runtime_error if the pinned staging cannot be allocated.
   * @throws std::system_error if the ring cannot be created.
   */
  uring_engine(uring_reactor& owner, io::detail::runner_slot& slot);

  /// Releases the ring and returns the staging blocks.  Must run on the
  /// thread that drove @ref run, after it returned.
  ~uring_engine();

  uring_engine(uring_engine const&)            = delete;
  uring_engine& operator=(uring_engine const&) = delete;

  /**
   * @brief Pull and process grouped requests until @p stop or @p deadline.
   *
   * On exit, either hands untaken work back to the hub (runner retirement,
   * admission still open) or cancels it (context shutdown), and drains every
   * operation already planned or submitted.  See @c io_engine_c.
   *
   * @return Number of grouped requests this runner retired.
   * @throws The fatal engine error (ring failure), after the cleanup above.
   */
  [[nodiscard]] std::size_t run(std::stop_token stop, std::optional<clock::time_point> deadline);

 private:
  class impl;
  std::unique_ptr<impl> _impl;
};

}  // namespace cucascade::io::uring
