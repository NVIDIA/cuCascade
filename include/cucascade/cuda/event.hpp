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

namespace cucascade {
namespace cuda {

class cuda_event {
 public:
  enum class query_status { success, in_progress, error };

  explicit cuda_event(unsigned int flags = cudaEventDisableTiming);
  ~cuda_event() noexcept;

  cuda_event(cuda_event const&)            = delete;
  cuda_event& operator=(cuda_event const&) = delete;

  cuda_event(cuda_event&& other) noexcept;
  cuda_event& operator=(cuda_event&& other) noexcept;

  [[nodiscard]] cudaEvent_t get() const noexcept;
  [[nodiscard]] explicit operator cudaEvent_t() const noexcept;

  void record(rmm::cuda_stream_view stream = rmm::cuda_stream_default);
  void wait(rmm::cuda_stream_view stream = rmm::cuda_stream_default) const;
  void synchronize() const;

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
  [[nodiscard]] query_status query() const noexcept;

 private:
  cudaEvent_t event_{nullptr};
};

}  // namespace cuda
}  // namespace cucascade
