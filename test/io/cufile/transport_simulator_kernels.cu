/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "transport_simulator_kernels.hpp"

#include <cuda/atomic>
#include <cuda_runtime.h>

namespace cucascade::io::cufile {
namespace {

__device__ std::uint64_t global_timer()
{
  std::uint64_t result;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(result));
  return result;
}

// A one-thread device gate never blocks CUDA's host-function executor or consumes a full GPU.
__global__ void await_completion(simulator_copy_args* args)
{
  cuda::atomic_ref<unsigned, cuda::thread_scope_system> released(args->released);
  while (released.load(cuda::memory_order_acquire) == 0)
    __nanosleep(1000);
  auto const start = global_timer();
  while (global_timer() - start < args->completion_latency_ns)
    __nanosleep(1000);
}

__global__ void copy_image(simulator_copy_args const* args)
{
  auto const work   = *args;
  auto const stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < work.count;
       i += stride) {
    if (work.write) {
      work.image[work.file_offset + i] = work.buffer[work.buffer_offset + i];
    } else {
      work.buffer[work.buffer_offset + i] = work.image[work.file_offset + i];
    }
  }
}

}  // namespace

cudaError_t launch_simulator_gate(simulator_copy_args* args,
                                  cudaStream_t stream,
                                  bool* possibly_enqueued) noexcept
{
  if (possibly_enqueued != nullptr) *possibly_enqueued = false;
  // A stale thread-local error must not be mistaken for proof that this launch was rejected.
  auto const previous = cudaGetLastError();
  if (previous != cudaSuccess) return previous;
  if (possibly_enqueued != nullptr) *possibly_enqueued = true;
  await_completion<<<1, 1, 0, stream>>>(args);
  return cudaGetLastError();
}

cudaError_t launch_simulator_copy(simulator_copy_args const* args, cudaStream_t stream) noexcept
{
  copy_image<<<64, 256, 0, stream>>>(args);
  return cudaGetLastError();
}

}  // namespace cucascade::io::cufile
