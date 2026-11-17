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

#include <cucascade/cuda/stream.hpp>
#include <cucascade/error.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace cucascade {

/**
 * @brief Accumulates (dst, src, size) copy operations for a single cudaMemcpyBatchAsync call.
 *
 * Collect all copies first, then call flush() once to submit them all to the GPU in one driver
 * call.
 */
class batch_copy_accumulator {
 public:
  void add(void* dst, const void* src, std::size_t size)
  {
    if (size == 0 || src == nullptr || dst == nullptr) { return; }
    _dsts.push_back(dst);
    _srcs.push_back(src);
    _sizes.push_back(size);
  }

  [[nodiscard]] std::size_t get_count() const noexcept { return _dsts.size(); }

  void flush(::cuda::stream_ref stream, cudaMemcpySrcAccessOrder src_order)
  {
    if (get_count() == 0) { return; }
#if CUDART_VERSION >= 12080
    cudaMemcpyAttributes attr{};
    attr.srcAccessOrder = src_order;
    attr.flags          = cudaMemcpyFlagDefault;
    // Single-attribute template overload (cuda_runtime.h): deduces direction from pointer types.
    // NOTE: cudaMemcpyBatchAsync requires a real (non-default) CUDA stream.
    // CUDA 12.x has a failIdx parameter that was removed in CUDA 13.
#if CUDART_VERSION < 13000
    CUCASCADE_CUDA_TRY(cudaMemcpyBatchAsync(
      _dsts.data(), _srcs.data(), _sizes.data(), get_count(), attr, nullptr, stream.get()));
#else
    CUCASCADE_CUDA_TRY(cudaMemcpyBatchAsync(
      _dsts.data(), _srcs.data(), _sizes.data(), get_count(), attr, stream.get()));
#endif
#else
    // cudaMemcpyBatchAsync requires CUDA 12.8+; fall back to individual copies.
    (void)src_order;
    for (std::size_t i = 0; i < get_count(); ++i) {
      CUCASCADE_CUDA_TRY(
        cudaMemcpyAsync(_dsts[i], _srcs[i], _sizes[i], cudaMemcpyDefault, stream.get()));
    }
#endif
    // Clear so subsequent add()+flush() cycles do not resubmit already-issued ops.
    _dsts.clear();
    _srcs.clear();
    _sizes.clear();
  }

 private:
  std::vector<void*> _dsts;
  std::vector<const void*> _srcs;
  std::vector<std::size_t> _sizes;
};

}  // namespace cucascade
