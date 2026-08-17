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

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <atomic>

namespace cucascade {
namespace test {

/**
 * @brief Parks a CUDA stream until the host releases it.
 *
 * Enqueue stream_gate_callback with cudaLaunchHostFunc to hold a stream open for as long as a
 * test needs, so work behind the gate stays observably in flight.
 */
struct stream_gate {
  std::atomic<bool> released{false};

  void release() noexcept
  {
    released.store(true, std::memory_order_release);
    released.notify_all();
  }
};

inline void CUDART_CB stream_gate_callback(void* user_data)
{
  auto* gate = static_cast<stream_gate*>(user_data);
  gate->released.wait(false, std::memory_order_acquire);
}

/**
 * @brief Releases and drains a parked callback during unwinding.
 *
 * Draining is required because the CUDA callback retains a pointer to the gate until the stream
 * has passed it.
 */
class stream_gate_release_guard {
 public:
  stream_gate_release_guard(stream_gate& gate, rmm::cuda_stream_view stream)
    : _gate(gate), _stream(stream)
  {
  }
  ~stream_gate_release_guard() noexcept
  {
    _gate.release();
    _stream.synchronize_no_throw();
  }

  stream_gate_release_guard(stream_gate_release_guard const&)            = delete;
  stream_gate_release_guard& operator=(stream_gate_release_guard const&) = delete;

 private:
  stream_gate& _gate;
  rmm::cuda_stream_view _stream;
};

}  // namespace test
}  // namespace cucascade
