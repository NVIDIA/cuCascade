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

#include <rmm/error.hpp>
#include <rmm/version_config.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#if RMM_VERSION_MAJOR > 26 || (RMM_VERSION_MAJOR == 26 && RMM_VERSION_MINOR >= 6)
#define CUCASCADE_RMM_HAS_MOVABLE_ANY_RESOURCE 1
#else
#define CUCASCADE_RMM_HAS_MOVABLE_ANY_RESOURCE 0
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#endif

namespace cucascade {
namespace memory {
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
  std::function<cuda::mr::any_resource<cuda::mr::device_accessible>(int device_id,
                                                                    std::size_t capacity)>;

#if !CUCASCADE_RMM_HAS_MOVABLE_ANY_RESOURCE
namespace detail {

class legacy_rmm_resource_adapter {
 public:
  explicit legacy_rmm_resource_adapter(std::shared_ptr<rmm::mr::device_memory_resource> resource)
    : resource_(std::move(resource))
  {
  }

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 [[maybe_unused]] std::size_t alignment = alignof(std::max_align_t))
  {
    return resource_->allocate(rmm::cuda_stream_view{stream}, bytes);
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  [[maybe_unused]] std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    resource_->deallocate(rmm::cuda_stream_view{stream}, ptr, bytes);
  }

  void* allocate_sync(std::size_t bytes,
                      std::size_t alignment = alignof(std::max_align_t))
  {
    auto* ptr = allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
    rmm::cuda_stream_default.synchronize();
    return ptr;
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
    rmm::cuda_stream_default.synchronize_no_throw();
  }

  bool operator==(legacy_rmm_resource_adapter const& other) const noexcept
  {
    return resource_.get() == other.resource_.get();
  }

  friend void get_property(legacy_rmm_resource_adapter const&,
                           cuda::mr::device_accessible) noexcept
  {
  }

 private:
  std::shared_ptr<rmm::mr::device_memory_resource> resource_;
};

}  // namespace detail

inline cuda::mr::any_resource<cuda::mr::device_accessible> wrap_legacy_rmm_resource(
  std::shared_ptr<rmm::mr::device_memory_resource> resource)
{
  return cuda::mr::any_resource<cuda::mr::device_accessible>{
    detail::legacy_rmm_resource_adapter{std::move(resource)}};
}
#endif

cuda::mr::any_resource<cuda::mr::device_accessible> make_default_gpu_memory_resource(
  int device_id, std::size_t capacity);

/**
 * @brief Grant cross-device peer ReadWrite access on a cudaMallocAsync pool.
 *
 * cudaMallocAsync pools require explicit `cudaMemPoolSetAccess` for cross-device peer copy
 * (cudaMemcpyPeer*) to actually transfer bytes; `cudaDeviceEnablePeerAccess` alone only
 * governs legacy cudaMalloc memory. This helper iterates all visible CUDA devices and
 * declares that each peer (other than `owner_device_id`) may read/write the pool.
 * Best effort — non-P2P-capable peers are skipped silently.
 */
void enable_pool_peer_access_for_all_visible_devices(cudaMemPool_t pool, int owner_device_id);

/**
 * @brief Empirical probe: does direct peer DMA actually move bytes between two GPUs?
 *
 * `cudaDeviceCanAccessPeer`, `cudaDeviceEnablePeerAccess`, and `cudaMemPoolGetAccess`
 * report peer access as available on hardware that physically cannot do peer DMA
 * (notably consumer Intel chipsets — Core i9 / Core Ultra desktop platforms — where
 * GPUDirect P2P is not supported). The standard APIs return success, but
 * `cudaMemcpyPeer*` then silently no-ops: it returns success without moving bytes.
 *
 * This probe allocates a tiny (64-byte) test buffer on each device, writes a known
 * sentinel pattern on the source, peer-copies to a different sentinel on the
 * destination, and verifies the destination matches the source. Returns true iff
 * bytes actually moved.
 *
 * Caller policy: call this AFTER `cudaDeviceEnablePeerAccess` so the probe sees the
 * "lying enable" failure mode. With peer access disabled, the driver auto-stages
 * through host and the probe always passes — which is correct, but doesn't
 * distinguish "real peer DMA" from "host fallback".
 */
[[nodiscard]] bool probe_peer_dma_works(int src_device, int dst_device);

/**
 * @brief Run the empirical probe across every P2P-capable GPU pair and DISABLE
 * peer access wherever direct DMA does not actually work.
 *
 * On consumer platforms where peer DMA is broken, leaving peer access enabled
 * forces the driver onto a silent-no-op path. Disabling peer access (and resetting
 * pool access to ProtNone) tells the driver to fall back to its internal pinned
 * host-staging path for `cudaMemcpyPeer*` — slower than real peer DMA but correct.
 *
 * Intended call site: cucascade::register_builtin_converters() — runs once after
 * memory_spaces are constructed and the application has called
 * `cudaDeviceEnablePeerAccess` for each pair. Idempotent.
 *
 * @param pools_by_device A vector indexed by device id (0..N-1) of the cucascade
 *        pool to also reset access on. Pass an empty vector if cucascade pools
 *        don't need pool-level peer access reset (legacy memory only).
 * @return Number of (i, j) pairs where peer access was disabled because the probe
 *         failed.
 */
int disable_peer_access_where_broken(
  std::vector<cudaMemPool_t> const& pools_by_device = {});

cuda::mr::any_resource<cuda::mr::device_accessible, cuda::mr::host_accessible>
make_default_host_memory_resource(int device_id, std::size_t capacity);

DeviceMemoryResourceFactoryFn make_default_allocator_for_tier(Tier tier);

}  // namespace memory
}  // namespace cucascade

// Specialization for std::hash to enable use of std::pair<Tier, size_t> as key
namespace std {
template <>
struct hash<cucascade::memory::memory_space_id> {
  size_t operator()(const cucascade::memory::memory_space_id& p) const;
};

}  // namespace std
