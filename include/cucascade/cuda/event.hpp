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

#pragma once

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>

namespace cucascade {
namespace cuda {

namespace event {
enum class query_result { success, in_progress, error };
}

/**
 * @brief Non-owning view of a CUDA event.
 *
 * Mirrors the rmm::cuda_stream / rmm::cuda_stream_view relationship: cuda_event owns
 * the underlying cudaEvent_t handle, while cuda_event_view is a copyable, non-owning
 * reference. Pass cuda_event_view by value through APIs that only need to query or
 * record an event but should not affect its lifetime.
 */
class cuda_event_view {
 public:
  cuda_event_view()                                           = default;
  ~cuda_event_view()                                          = default;
  cuda_event_view(cuda_event_view const&) noexcept            = default;
  cuda_event_view(cuda_event_view&&) noexcept                 = default;
  cuda_event_view& operator=(cuda_event_view const&) noexcept = default;
  cuda_event_view& operator=(cuda_event_view&&) noexcept      = default;

  // Disable construction from literal 0 / nullptr so callers can't silently pass an
  // invalid handle the way one might to a default-constructed view.
  cuda_event_view(int)            = delete;
  cuda_event_view(std::nullptr_t) = delete;

  /**
   * @brief Construct a view referring to an existing cudaEvent_t handle.
   */
  cuda_event_view(cudaEvent_t event) noexcept : event_{event} {}

  [[nodiscard]] cudaEvent_t value() const noexcept { return event_; }
  operator cudaEvent_t() const noexcept { return event_; }

  void record(rmm::cuda_stream_view stream = rmm::cuda_stream_default);
  void wait(rmm::cuda_stream_view stream = rmm::cuda_stream_default) const;
  void synchronize() const;

  /**
   * @brief Synchronize the calling thread on this event, without throwing.
   *
   * Like synchronize(), blocks the calling thread until the event has completed, but
   * does not throw on failure. Intended for noexcept / destructor contexts.
   *
   * @return cudaSuccess on success, otherwise the failing CUDA error code.
   */
  [[nodiscard]] cudaError_t synchronize_no_throw() const noexcept;

  /**
   * @brief Return elapsed time between `start` and this event.
   *
   * Both events must have been recorded, completed, and created with timing enabled.
   */
  [[nodiscard]] std::chrono::duration<float, std::milli> elapsed_time(cuda_event_view start) const;

  /**
   * @brief Query whether the event has completed without blocking.
   */
  [[nodiscard]] event::query_result query() const noexcept;

 private:
  cudaEvent_t event_{};
};

class cuda_event {
 public:
  explicit cuda_event(unsigned int flags = cudaEventDisableTiming);
  ~cuda_event() noexcept;

  cuda_event(cuda_event const&)            = delete;
  cuda_event& operator=(cuda_event const&) = delete;

  cuda_event(cuda_event&& other) noexcept;
  cuda_event& operator=(cuda_event&& other) noexcept;

  [[nodiscard]] cudaEvent_t get() const noexcept;
  [[nodiscard]] explicit operator cudaEvent_t() const noexcept;

  /**
   * @brief Return a non-owning view of this event.
   */
  [[nodiscard]] cuda_event_view view() const noexcept;

  /**
   * @brief Implicit conversion to cuda_event_view.
   */
  operator cuda_event_view() const noexcept;

  void record(rmm::cuda_stream_view stream = rmm::cuda_stream_default);
  void wait(rmm::cuda_stream_view stream = rmm::cuda_stream_default) const;
  void synchronize() const;

  /**
   * @brief Synchronize the calling thread on this event, without throwing.
   *
   * Like synchronize(), blocks the calling thread until the event has completed, but
   * does not throw on failure. Intended for noexcept / destructor contexts.
   *
   * @return cudaSuccess on success, otherwise the failing CUDA error code.
   */
  [[nodiscard]] cudaError_t synchronize_no_throw() const noexcept;

  /**
   * @brief Return elapsed time between `start` and this event.
   *
   * Both events must have been recorded, completed, and created with timing enabled.
   */
  [[nodiscard]] std::chrono::duration<float, std::milli> elapsed_time(
    cuda_event const& start) const;

  /**
   * @brief Query whether the event has completed without blocking.
   */
  [[nodiscard]] event::query_result query() const noexcept;

 private:
  cudaEvent_t event_{nullptr};
};

}  // namespace cuda
}  // namespace cucascade
