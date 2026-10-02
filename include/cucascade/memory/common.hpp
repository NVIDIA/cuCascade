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
 * changed through the CUDA runtime or the pool is destroyed. A failed request leaves existing pool
 * permissions unchanged; callers must coordinate any revocation with other users of the pool.
 *
 * Access is granted only after the byte verification described for probe_peer_dma_works() passes in
 * both directions. The first request that needs verification, through this function,
 * probe_peer_dma_works(), or disable_peer_access_where_broken(), verifies every ordered pair of
 * visible devices while holding a process-wide lock, so concurrent requests wait for it. Results
 * are cached for the process lifetime; directions that ended in a CUDA error are verified again on
 * a later grant request. If both directional probes fail, a CUDA error takes precedence over a
 * verification failure or an unsupported result. Verification temporarily changes the caller
 * thread's current device and restores it before return; a restoration failure is reported as
 * pool_peer_access_status::CUDA_ERROR.
 *
 * The caller must supply a live non-null pool, valid visible CUDA device IDs, and the device that
 * actually owns the pool's allocations as owner_device. A request where owner_device equals
 * accessing_device runs no verification: it reports GRANTED when the pool already grants that
 * device read/write access, and pool_peer_access_status::CUDA_ERROR with cudaErrorInvalidValue
 * otherwise, which means owner_device does not describe this pool.
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
 * This best-effort helper calls grant_pool_peer_access() for every visible peer on both @p pool and
 * the owner's currently selected pool. It writes one line to stderr for each grant that fails with
 * a CUDA error and otherwise discards the outcomes, including unsupported pairs and failed
 * verification.
 *
 * @param pool The pool to configure
 * @param owner_device_id The device on which the pool's allocations reside
 */
void enable_pool_peer_access_for_all_visible_devices(cudaMemPool_t pool, int owner_device_id);

/**
 * @brief Report whether a peer copy between two GPUs' memory pools delivers correct bytes
 *
 * CUDA can report peer capability on hardware whose direct peer route delivers wrong bytes while
 * every call returns cudaSuccess, so a byte comparison, not `cudaDeviceCanAccessPeer`, decides
 * whether peer copies are used.
 *
 * A same-device request returns true. For distinct devices, verification copies 64 bytes with
 * `cudaMemcpyPeerAsync` between two private memory pools that are granted to each other, then
 * compares the bytes. Pool grants, not ordinary peer access, select the copy route for pool
 * allocations, so the result applies to pools created with the allocation properties of the pool
 * owned by `rmm::mr::cuda_async_memory_resource`; pools with other properties are not verified
 * separately. Verification changes neither ordinary peer access nor the permissions of any pool it
 * did not create, and waits only for its own streams.
 *
 * Results are cached as described for grant_pool_peer_access(). A cached CUDA error returns false
 * without repeating the probe on every copy; grant_pool_peer_access() or
 * disable_peer_access_where_broken() retries it.
 *
 * @param src_device The device that owns the copied bytes
 * @param dst_device The device that receives the copied bytes
 * @return True for a same-device request or a verified direction; false for unsupported directions,
 * failed verification, or CUDA errors
 */
[[nodiscard]] bool probe_peer_dma_works(int src_device, int dst_device);

/**
 * @brief Trigger cached peer verification and count the directions that failed it
 *
 * Despite its name, which is kept for API compatibility, this function changes no peer access and
 * no pool permission. It runs the verification described for probe_peer_dma_works() if it has not
 * run yet, and retries directions with cached CUDA errors.
 *
 * @param pools_by_device Ignored; retained for API compatibility
 * @return Number of directions whose copied bytes did not match, where directions with CUDA errors
 * are not counted; or -1 when verification could not run because the device count or current device
 * could not be queried or an internal error occurred. A completed verification is counted even if
 * restoring the caller's device afterwards fails.
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
