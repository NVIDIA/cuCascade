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

/**
 * @file
 * The `*_operations` structs below are injection seams: production code binds them to the CUDA
 * runtime and tests substitute fakes. Each function pointer other than probe_peer_dma mirrors the
 * like-named CUDA runtime call (for example, get_pool_access mirrors `cudaMemPoolGetAccess`), and a
 * real implementation must clear CUDA's thread-local last error after a failure so that it cannot
 * leak into unrelated calls.
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
  CUDA_ERROR            ///< A CUDA runtime operation failed; the result carries the first error
};

struct peer_dma_probe_result {
  peer_dma_probe_status status{peer_dma_probe_status::CUDA_ERROR};
  cudaError_t error{cudaErrorUnknown};
};

/**
 * @brief Operations used by the process-wide peer DMA probe cache
 *
 * Production code binds probe_peer_dma to probe_peer_dma_with_private_pools(); tests drive the
 * cache through probe_peer_dma_sequence() and count_broken_directions().
 */
struct peer_dma_probe_operations {
  cudaError_t (*get_device_count)(int* count);
  cudaError_t (*get_device)(int* device);
  cudaError_t (*set_device)(int device);
  cudaError_t (*can_access_peer)(int* can_access, int device, int peer_device);
  cudaError_t (*get_last_error)();
  peer_dma_probe_result (*probe_peer_dma)(int source_device, int destination_device) noexcept;
};

/** @brief Run requests through a fresh probe cache with supplied CUDA operations. */
[[nodiscard]] std::vector<peer_dma_probe_result> probe_peer_dma_sequence(
  std::vector<std::pair<int, int>> const& requests,
  peer_dma_probe_operations const& operations,
  bool retry_errors = true);

/**
 * @brief Call the broken-direction count behind disable_peer_access_where_broken() @p calls times
 * on one fresh probe cache with supplied CUDA operations
 *
 * @return The count returned by each call, in order
 */
[[nodiscard]] std::vector<int> count_broken_directions(peer_dma_probe_operations const& operations,
                                                       std::size_t calls);

/**
 * @brief Operations used by grant_pool_peer_access()
 *
 * Production code binds probe_peer_dma to a lookup in the process-wide probe cache.
 */
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
 * @brief Verify one peer copy direction on memory pools that only this call can see
 *
 * Creates a non-blocking stream on @p destination_device and one private pool on each device,
 * grants each pool to the other device, and copies 64 bytes from the source pool to the destination
 * pool with `cudaMemcpyPeerAsync`. The guarantees that reach callers are documented on
 * probe_peer_dma_works(); in addition, this call releases every resource it created and clears the
 * calling thread's last CUDA error after a failure.
 *
 * The devices are expected to be distinct and peer capable in both directions; an invalid device
 * yields CUDA_ERROR. The probe leaves @p destination_device current, and the probe cache restores
 * the caller's device.
 *
 * @param source_device The device that owns the copied bytes
 * @param destination_device The device that receives the bytes and issues the copy
 * @return SUPPORTED when the bytes match, VERIFICATION_FAILED when they do not, or CUDA_ERROR with
 * the first CUDA runtime error, including a cleanup failure
 */
[[nodiscard]] peer_dma_probe_result probe_peer_dma_with_private_pools(
  int source_device, int destination_device) noexcept;

}  // namespace detail
}  // namespace memory
}  // namespace cucascade
