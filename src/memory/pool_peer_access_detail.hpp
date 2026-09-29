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

#include <cucascade/memory/common.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace cucascade {
namespace memory {
namespace detail {

enum class peer_dma_probe_status {
  SUPPORTED,            ///< The directional copy probe succeeded
  UNSUPPORTED,          ///< CUDA reports that required peer capability is unavailable
  VERIFICATION_FAILED,  ///< CUDA reports that the direction is supported, but the sentinel bytes
                        ///< did not arrive correctly
  CUDA_ERROR
};

struct peer_dma_probe_result {
  peer_dma_probe_status status{peer_dma_probe_status::CUDA_ERROR};
  cudaError_t error{cudaErrorUnknown};
};

struct peer_dma_probe_operations {
  cudaError_t (*get_device_count)(int* count);
  cudaError_t (*get_device)(int* device);
  cudaError_t (*set_device)(int device);
  cudaError_t (*can_access_peer)(int* can_access, int device, int peer_device);
  cudaError_t (*enable_peer_access)(int peer_device, unsigned int flags);
  cudaError_t (*disable_peer_access)(int peer_device);
  cudaError_t (*get_last_error)();
  peer_dma_probe_result (*probe_peer_dma)(int source_device, int destination_device);
};

/** @brief Run requests through a fresh probe cache with supplied CUDA operations. */
[[nodiscard]] std::vector<peer_dma_probe_result> probe_peer_dma_sequence(
  std::vector<std::pair<int, int>> const& requests,
  peer_dma_probe_operations const& operations,
  bool retry_errors = true);

struct pool_peer_access_operations {
  cudaError_t (*get_device_count)(int* count);
  cudaError_t (*can_access_peer)(int* can_access, int device, int peer_device);
  peer_dma_probe_result (*probe_peer_dma)(int source_device, int destination_device);
  cudaError_t (*get_pool_access)(cudaMemAccessFlags* flags,
                                 cudaMemPool_t pool,
                                 cudaMemLocation* location);
  cudaError_t (*set_pool_access)(cudaMemPool_t pool,
                                 cudaMemAccessDesc const* descriptors,
                                 std::size_t count);
};

/**
 * @brief Disable the CUDA peer-access direction used by one directional copy probe.
 *
 * A copy from source to destination uses destination as the accessing device and source as the
 * allocation owner.
 */
[[nodiscard]] cudaError_t disable_peer_access_for_failed_probe(
  int source_device,
  int destination_device,
  cudaError_t (*set_device)(int),
  cudaError_t (*disable_peer_access)(int)) noexcept;

/**
 * @brief Restore the device that was current before the peer DMA probe.
 */
[[nodiscard]] cudaError_t finish_peer_dma_probe(int saved_device,
                                                cudaError_t probe_error,
                                                cudaError_t (*set_device)(int)) noexcept;

}  // namespace detail
}  // namespace memory
}  // namespace cucascade
