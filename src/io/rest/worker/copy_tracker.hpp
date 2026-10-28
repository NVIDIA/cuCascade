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

#include "io/rest/worker/op_request.hpp"

#include <cucascade/cuda/event.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace cucascade::io::rest::detail {

/**
 * @brief Device copies a REST worker waits for: H2D of a received GET into its device
 *        destination, D2H of a device write source into its part buffer.
 *
 * Copies run on the caller's stream (cuCascade's stream semantics), one event per copy from a
 * per-device pool.  The worker never blocks on them (gpuio G-4): it polls an event only once the
 * copy's ETA has passed, and the earliest ETA bounds its ring wait (gpuio R11 / G-5: a
 * remote-only ring gets no file CQE that would wake it when a copy lands).  The connection that
 * delivered a GET is checked in before its copy starts (gpuio M-10).  Worker-thread only.
 */
class copy_tracker {
 public:
  enum class direction : std::uint8_t { to_device, to_host };

  /// A copy that finished (successfully or not).
  struct done {
    std::unique_ptr<rest_io_op_request> op;
    cudaError_t status{cudaSuccess};
    direction dir{direction::to_device};
  };

  copy_tracker()                               = default;
  copy_tracker(copy_tracker const&)            = delete;
  copy_tracker& operator=(copy_tracker const&) = delete;
  ~copy_tracker();

  /**
   * @brief Enqueue the copy of @p op and track it.
   *
   * @return cudaSuccess and @p op moved into the tracker, or the enqueue error with @p op left
   *         with the caller.
   */
  [[nodiscard]] cudaError_t start(std::unique_ptr<rest_io_op_request>& op,
                                  direction dir,
                                  std::uint64_t now_ns);

  /// Move every copy that completed, or failed, into @p out.  Queries only copies whose ETA
  /// passed.
  void poll(std::uint64_t now_ns, std::vector<done>& out);

  [[nodiscard]] bool empty() const noexcept { return _active.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return _active.size(); }

  /// When the next copy is due for a query (the ring wait bound); max when nothing is tracked.
  [[nodiscard]] std::uint64_t next_due_ns() const noexcept;

  /// Shutdown: synchronize every tracked copy (a pinned block must never be released under a
  /// pending DMA) and move them into @p out.
  void drain(std::vector<done>& out) noexcept;

 private:
  struct entry {
    std::unique_ptr<rest_io_op_request> op;
    cucascade::cuda::cuda_event event;
    int device{-1};
    std::uint64_t due_ns{0};
    std::uint64_t backoff_ns{0};
    direction dir{direction::to_device};
  };

  [[nodiscard]] cucascade::cuda::cuda_event take_event(int device);
  void return_event(int device, cucascade::cuda::cuda_event event) noexcept;

  std::vector<entry> _active;
  std::unordered_map<int, std::vector<cucascade::cuda::cuda_event>> _free_events;
};

/// Make every REST worker report a failed CUDA event record after enqueuing a device copy (H2D
/// of a received GET, D2H of a write source), so tests can check the operation is not settled
/// -- and its pinned buffer not released -- while that copy may still run.  Process-wide; tests
/// only.
void set_copy_event_record_failure_for_testing(bool enabled) noexcept;

}  // namespace cucascade::io::rest::detail
