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

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace cucascade::test {

/// Number of visible CUDA devices; a runtime that cannot enumerate any
/// (cudaErrorNoDevice, no driver) counts as zero so callers can SKIP.
inline int cuda_device_count()
{
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) { return 0; }
  return count;
}

/// Parks a stream in a host callback until released, so work queued behind it
/// (a read's H2D copy, a kernel) cannot run until then. The destructor releases
/// the gate and drains the stream so no callback outlives the gate.
class stream_gate {
 public:
  explicit stream_gate(cudaStream_t stream) : _stream(stream) {}
  ~stream_gate()
  {
    release();
    if (_timer.joinable()) { _timer.join(); }
    (void)cudaStreamSynchronize(_stream);
  }

  stream_gate(stream_gate const&)            = delete;
  stream_gate& operator=(stream_gate const&) = delete;

  void arm()
  {
    REQUIRE(cudaLaunchHostFunc(
              _stream,
              [](void* opaque) {
                auto& gate = *static_cast<stream_gate*>(opaque);
                std::unique_lock lock{gate._mutex};
                gate._entered = true;
                gate._cv.notify_all();
                gate._cv.wait(lock, [&] { return gate._released; });
              },
              this) == cudaSuccess);
  }

  /// True once the stream has reached the gate.
  bool wait_entered(std::chrono::milliseconds timeout)
  {
    std::unique_lock lock{_mutex};
    return _cv.wait_for(lock, timeout, [&] { return _entered; });
  }

  void release_after(std::chrono::milliseconds delay)
  {
    _timer = std::jthread([this, delay] {
      std::unique_lock lock{_mutex};
      _cv.wait_for(lock, delay, [&] { return _released; });
      _released = true;
      lock.unlock();
      _cv.notify_all();
    });
  }

  void release()
  {
    {
      std::scoped_lock lock{_mutex};
      _released = true;
    }
    _cv.notify_all();
  }

 private:
  cudaStream_t _stream;
  std::mutex _mutex;
  std::condition_variable _cv;
  bool _entered{false};
  bool _released{false};
  std::jthread _timer;
};

}  // namespace cucascade::test
