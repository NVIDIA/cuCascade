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

#include <cucascade/cuda/stream.hpp>

#include <rmm/error.hpp>
#include <rmm/version_config.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace cucascade {
namespace memory {

/** @brief Outcome of a CUDA memory-pool peer-access request. */
enum class pool_peer_access_status {
  GRANTED,              ///< Read/write pool access is established
  UNSUPPORTED,          ///< The device pair lacks required peer capability
  VERIFICATION_FAILED,  ///< Bidirectional byte-transfer verification was rejected
  CUDA_ERROR            ///< The request is invalid or a CUDA runtime operation failed
};

class pool_peer_access_result;

namespace detail {
struct pool_peer_access_operations;

[[nodiscard]] pool_peer_access_result grant_pool_peer_access(
  cudaMemPool_t pool,
  int owner_device,
  int accessing_device,
  pool_peer_access_operations const& operations) noexcept;
}  // namespace detail

/**
 * @brief Result of a targeted CUDA memory-pool peer-access request
 *
 * A granted cross-device result establishes that byte-transfer verification succeeded in both
 * directions and that the CUDA runtime accepted or already reported read/write access to the
 * requested pool for the accessing device.
 */
class pool_peer_access_result {
 public:
  /** @brief Returns the outcome category. */
  [[nodiscard]] pool_peer_access_status status() const noexcept { return _status; }

  /**
   * @brief Returns the associated CUDA error for CUDA_ERROR, otherwise cudaSuccess
   *
   * Errors returned by CUDA operations are preserved unchanged. Argument-validation failures use
   * the corresponding CUDA runtime error code.
   */
  [[nodiscard]] cudaError_t error() const noexcept { return _error; }

  /** @brief Returns true only when read/write access is established. */
  [[nodiscard]] bool granted() const noexcept
  {
    return _status == pool_peer_access_status::GRANTED;
  }

 private:
  constexpr pool_peer_access_result(pool_peer_access_status status, cudaError_t error) noexcept
    : _status(status), _error(error)
  {
  }

  friend pool_peer_access_result grant_pool_peer_access(cudaMemPool_t, int, int) noexcept;
  friend pool_peer_access_result detail::grant_pool_peer_access(
    cudaMemPool_t, int, int, detail::pool_peer_access_operations const&) noexcept;

  pool_peer_access_status _status;
  cudaError_t _error;
};

/**
 * Memory tier enumeration representing different types of memory storage.
 * Ordered roughly by performance (fastest to slowest access).
 */
enum class Tier : int32_t {
  GPU,   // GPU device memory (fastest but limited)
  HOST,  // Host system memory (fast, larger capacity)
  DISK,  // Disk/storage memory (slowest but largest capacity)
  SIZE   // Value = size of the enum, allows code to be more dynamic
};

/**
 * Memory space id, comprised of device id, and tier
 *
 */
class memory_space_id {
 public:
  Tier tier;
  int32_t device_id;

  explicit memory_space_id(Tier t, int32_t d_id) : tier(t), device_id(d_id) {}

  auto operator<=>(const memory_space_id&) const noexcept = default;

  std::size_t uuid() const noexcept
  {
    std::size_t key = 0;
    std::memcpy(&key, this, sizeof(key));
    return key;
  }
};

using DeviceMemoryResourceFactoryFn =
  std::function<::cuda::mr::any_resource<::cuda::mr::device_accessible>(int device_id,
                                                                        std::size_t capacity)>;

::cuda::mr::any_resource<::cuda::mr::device_accessible> make_default_gpu_memory_resource(
  int device_id, std::size_t capacity);

/**
 * @brief Grants one device persistent read/write access to allocations from one CUDA memory pool
 *
 * The pool is borrowed and remains owned by its creator. A successful grant persists until it is
 * changed through the CUDA runtime or the pool is destroyed. On first cross-device use, peer
 * verification synchronizes CUDA work, temporarily changes the caller thread's current device,
 * and enables or disables legacy peer access across visible device pairs. The original current
 * device is restored before return; a restoration failure is reported as
 * pool_peer_access_status::CUDA_ERROR. Verified peer results are cached process-wide;
 * CUDA errors are retried on a later request.
 *
 * The caller must supply a live non-null pool, valid visible CUDA device IDs, and the device that
 * actually owns the pool's allocations as owner_device.
 *
 * @param pool The actual pool backing the allocations to share
 * @param owner_device The device on which the pool's allocations reside
 * @param accessing_device The device that needs read/write access
 * @return The grant outcome and any associated CUDA runtime error
 */
[[nodiscard]] pool_peer_access_result grant_pool_peer_access(cudaMemPool_t pool,
                                                             int owner_device,
                                                             int accessing_device) noexcept;

/**
 * @brief Grant cross-device peer ReadWrite access on a cudaMallocAsync pool.
 *
 * This best-effort helper attempts access for every visible peer on both @p pool and the owner's
 * currently selected pool. It discards individual outcomes. On first use it also performs the
 * process-wide synchronization and legacy peer-state changes documented by
 * grant_pool_peer_access().
 *
 * @param pool The pool to configure
 * @param owner_device_id The device on which the pool's allocations reside
 */
void enable_pool_peer_access_for_all_visible_devices(cudaMemPool_t pool, int owner_device_id);

/**
 * @brief Report whether a peer copy moves bytes from one GPU to another
 *
 * A same-device request returns true. For distinct devices, the process-wide cache enables legacy
 * peer access for capable directions before testing a 64-byte copy. It disables the matching
 * direction only after a confirmed byte mismatch. CUDA errors are retried on later requests.
 *
 * @return True for a same-device request or a verified directional byte copy; false for unsupported
 * directions, failed verification, or CUDA errors
 */
[[nodiscard]] bool probe_peer_dma_works(int src_device, int dst_device);

/**
 * @brief Trigger cached peer verification and count verified fallback directions
 *
 * On first cache use, every visible direction is probed with legacy peer access enabled where
 * needed. Only a confirmed byte mismatch causes that direction to be disabled. Later calls retry
 * directions with CUDA errors. CUDA memory pool permissions are not changed.
 *
 * @param pools_by_device Ignored; retained for API compatibility
 * @return Number of verified mismatched directions whose legacy peer access was disabled or was
 * already disabled
 */
int disable_peer_access_where_broken(std::vector<cudaMemPool_t> const& pools_by_device = {});

::cuda::mr::any_resource<::cuda::mr::device_accessible, ::cuda::mr::host_accessible>
make_default_host_memory_resource(int device_id, std::size_t capacity);

::cuda::mr::any_resource<::cuda::mr::device_accessible, ::cuda::mr::host_accessible>
make_default_host_memory_resource(int device_id, std::size_t capacity, bool make_portable);

DeviceMemoryResourceFactoryFn make_default_allocator_for_tier(Tier tier);

// Forward declaration — fixed_size_host_memory_resource.hpp is heavy.
class fixed_size_host_memory_resource;

/**
 * @brief Register a HOST-tier `fixed_size_host_memory_resource` so cross-tier
 * code paths (notably the peer-DMA-broken host-staging branch in
 * representation_converter.cpp) can borrow blocks from the pre-pinned pool
 * instead of calling cudaHostAlloc per transfer.
 *
 * Called by memory_space's HOST constructor; unregistered by its destructor.
 * Multiple pools may be registered against the same numa_id (test fixtures
 * commonly create overlapping HOST memory_spaces); re-registering the same
 * pointer is a no-op.
 */
void register_host_pool(int numa_id, fixed_size_host_memory_resource* pool);

/**
 * @brief Unregister a previously registered HOST pool. No-op if not present.
 */
void unregister_host_pool(int numa_id, fixed_size_host_memory_resource* pool) noexcept;

/**
 * @brief Look up a registered HOST pool. Returns the most recently registered
 * pool for the requested numa_id; falls back to any registered pool from any
 * numa_id (cross-NUMA staging is suboptimal but correct). Returns nullptr if
 * no HOST pool has been registered.
 */
[[nodiscard]] fixed_size_host_memory_resource* find_host_pool(int numa_id) noexcept;

}  // namespace memory
}  // namespace cucascade

// Specialization for std::hash to enable use of std::pair<Tier, size_t> as key
namespace std {
template <>
struct hash<cucascade::memory::memory_space_id> {
  size_t operator()(const cucascade::memory::memory_space_id& p) const;
};

}  // namespace std
