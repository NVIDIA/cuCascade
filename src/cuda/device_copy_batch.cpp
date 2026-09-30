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

#include <cucascade/cuda/device_copy_batch.hpp>
#include <cucascade/error.hpp>

#include <algorithm>

namespace cucascade::cuda {

namespace {

/// Copies at or below this size are latency-bound small transfers that overlap
/// well with compute; any larger copy in the batch makes the whole batch use the
/// default flag.
constexpr std::size_t overlap_with_compute_max_bytes = 128ULL << 10;

/// cudaMemcpyBatchAsync rejects the legacy and per-thread default streams.
[[nodiscard]] bool is_default_stream(cudaStream_t stream) noexcept
{
  return stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread;
}

}  // namespace

void device_copy_batch::reserve(std::size_t n)
{
  _dsts.reserve(n);
  _srcs.reserve(n);
  _sizes.reserve(n);
}

void device_copy_batch::add(void* dst, void const* src, std::size_t bytes)
{
  if (bytes == 0 || dst == nullptr || src == nullptr) { return; }
  _dsts.push_back(dst);
  _srcs.push_back(src);
  _sizes.push_back(bytes);
  _bytes += bytes;
}

cudaError_t device_copy_batch::enqueue(::cuda::stream_ref stream) const
{
  if (_dsts.empty()) { return cudaSuccess; }

#if CUDART_VERSION >= 13000
  if (!is_default_stream(stream.get())) {
    cudaMemcpyAttributes attr{};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attr.flags          = std::ranges::all_of(
                   _sizes, [](std::size_t size) { return size <= overlap_with_compute_max_bytes; })
                            ? cudaMemcpyFlagPreferOverlapWithCompute
                            : cudaMemcpyFlagDefault;
    // Single-attribute overload (cuda_runtime.h): deduces direction from the pointer types.
    return cudaMemcpyBatchAsync(
      _dsts.data(), _srcs.data(), _sizes.data(), _dsts.size(), attr, stream.get());
  }
#endif

  for (std::size_t i = 0; i < _dsts.size(); ++i) {
    auto const status =
      cudaMemcpyAsync(_dsts[i], _srcs[i], _sizes[i], cudaMemcpyDefault, stream.get());
    if (status != cudaSuccess) { return status; }
  }
  return cudaSuccess;
}

void device_copy_batch::clear() noexcept
{
  _dsts.clear();
  _srcs.clear();
  _sizes.clear();
  _bytes = 0;
}

}  // namespace cucascade::cuda
