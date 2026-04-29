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

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/null_device_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <mutex>

namespace cucascade {

namespace memory {

namespace {
// Cached per-pair result of an empirical peer-DMA probe. p2p_dma_supported(i,j)
// is true iff cudaMemcpyPeer from device i to device j actually moves bytes
// on this hardware. The probe is run lazily on first access and pinned for
// the lifetime of the process. On consumer Intel chipsets (Core Ultra etc.)
// the standard CUDA peer-access APIs all report success but the underlying
// PCIe/chipset hardware can't actually do peer DMA — cudaMemcpyPeer returns
// success without moving bytes. The probe catches this empirically.
constexpr int kMaxDevices = 16;
bool g_p2p_supported[kMaxDevices][kMaxDevices] = {};
bool g_p2p_probed                             = false;
std::mutex& p2p_probe_mutex()
{
  static std::mutex m;
  return m;
}

void run_p2p_probe_locked(int device_count)
{
  // Step 1: enable legacy peer access for all P2P-capable pairs. The probe
  // needs peer access enabled to detect the "lying enable" failure mode —
  // with peer access disabled, cudaMemcpyPeer auto-host-stages and the probe
  // would always report success, which would conflate working hardware with
  // the host-fallback path.
  for (int i = 0; i < device_count; ++i) {
    for (int j = 0; j < device_count; ++j) {
      if (i == j) continue;
      int can = 0;
      if (cudaDeviceCanAccessPeer(&can, i, j) != cudaSuccess || !can) {
        (void)cudaGetLastError();
        continue;
      }
      if (cudaSetDevice(i) != cudaSuccess) {
        (void)cudaGetLastError();
        continue;
      }
      cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
      if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
        (void)cudaGetLastError();
      }
    }
  }

  // Step 2: probe each direction. Allocate tiny test buffers, fill src with a
  // known sentinel, peer-copy, verify dst == src.
  constexpr std::size_t kProbeBytes      = 64;
  unsigned char src_pat[kProbeBytes]     = {};
  unsigned char dst_sentinel[kProbeBytes] = {};
  for (std::size_t k = 0; k < kProbeBytes; ++k) {
    src_pat[k]      = static_cast<unsigned char>(0x40 + (k & 0x3F));
    dst_sentinel[k] = 0xAA;
  }
  for (int i = 0; i < device_count; ++i) {
    for (int j = 0; j < device_count; ++j) {
      if (i == j) {
        g_p2p_supported[i][j] = true;
        continue;
      }
      int can = 0;
      if (cudaDeviceCanAccessPeer(&can, i, j) != cudaSuccess || !can) {
        (void)cudaGetLastError();
        g_p2p_supported[i][j] = false;
        continue;
      }
      void* src = nullptr;
      void* dst = nullptr;
      bool ok   = false;
      if (cudaSetDevice(i) == cudaSuccess && cudaMalloc(&src, kProbeBytes) == cudaSuccess &&
          cudaMemcpy(src, src_pat, kProbeBytes, cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaSetDevice(j) == cudaSuccess && cudaMalloc(&dst, kProbeBytes) == cudaSuccess &&
          cudaMemcpy(dst, dst_sentinel, kProbeBytes, cudaMemcpyHostToDevice) == cudaSuccess) {
        if (cudaMemcpyPeer(dst, j, src, i, kProbeBytes) == cudaSuccess &&
            cudaDeviceSynchronize() == cudaSuccess) {
          unsigned char readback[kProbeBytes] = {};
          if (cudaMemcpy(readback, dst, kProbeBytes, cudaMemcpyDeviceToHost) == cudaSuccess) {
            ok = std::memcmp(readback, src_pat, kProbeBytes) == 0;
          }
        }
      }
      if (dst) {
        cudaSetDevice(j);
        cudaFree(dst);
      }
      if (src) {
        cudaSetDevice(i);
        cudaFree(src);
      }
      g_p2p_supported[i][j] = ok;
    }
  }

  // Step 3: for any pair where the probe failed, disable legacy peer access so
  // subsequent cudaMemcpyPeer* calls fall back to the driver's host-stage path.
  for (int i = 0; i < device_count; ++i) {
    for (int j = 0; j < device_count; ++j) {
      if (i == j) continue;
      if (g_p2p_supported[i][j]) continue;
      int can = 0;
      if (cudaDeviceCanAccessPeer(&can, i, j) != cudaSuccess || !can) {
        (void)cudaGetLastError();
        continue;
      }
      cudaSetDevice(i);
      cudaError_t e = cudaDeviceDisablePeerAccess(j);
      if (e != cudaSuccess && e != cudaErrorPeerAccessNotEnabled) {
        (void)cudaGetLastError();
      }
    }
  }
  cudaSetDevice(0);
  (void)cudaGetLastError();

  // Report.
  int broken = 0;
  for (int i = 0; i < device_count; ++i) {
    for (int j = 0; j < device_count; ++j) {
      if (i != j && !g_p2p_supported[i][j]) {
        int can = 0;
        if (cudaDeviceCanAccessPeer(&can, i, j) == cudaSuccess && can) ++broken;
        (void)cudaGetLastError();
      }
    }
  }
  if (broken > 0) {
    fprintf(stderr,
            "[cucascade] direct GPU↔GPU peer DMA broken on %d direction(s); "
            "cudaMemcpyPeer* will host-stage automatically.\n",
            broken);
  }
}

bool ensure_p2p_probed()
{
  std::lock_guard<std::mutex> lk(p2p_probe_mutex());
  if (g_p2p_probed) return true;
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
    (void)cudaGetLastError();
    g_p2p_probed = true;  // pin the empty result
    return false;
  }
  if (device_count > kMaxDevices) device_count = kMaxDevices;
  run_p2p_probe_locked(device_count);
  g_p2p_probed = true;
  return true;
}

bool p2p_dma_works_cached(int src, int dst)
{
  if (src < 0 || dst < 0 || src >= kMaxDevices || dst >= kMaxDevices) return false;
  ensure_p2p_probed();
  return g_p2p_supported[src][dst];
}

void set_access_on_pool(cudaMemPool_t pool, int owner_device_id, int device_count)
{
  for (int peer = 0; peer < device_count; ++peer) {
    if (peer == owner_device_id) { continue; }
    int can_access = 0;
    if (cudaDeviceCanAccessPeer(&can_access, peer, owner_device_id) != cudaSuccess ||
        !can_access) {
      (void)cudaGetLastError();
      continue;
    }
    // Skip pairs where the empirical probe shows that direct peer DMA does NOT
    // actually move bytes on this hardware (consumer Intel platforms etc.).
    // Granting cudaMemPoolSetAccess(ProtReadWrite) on those pairs would force
    // cudaMemcpyPeer* down a silent-no-op path for pool-allocated memory; with
    // pool access left at the default ProtNone instead, the driver host-stages
    // automatically. cudaDeviceEnablePeerAccess for the broken pair has
    // already been disabled by the probe.
    if (!p2p_dma_works_cached(peer, owner_device_id)) continue;
    cudaMemAccessDesc desc{};
    desc.location.type = cudaMemLocationTypeDevice;
    desc.location.id   = peer;
    desc.flags         = cudaMemAccessFlagsProtReadWrite;
    if (cudaMemPoolSetAccess(pool, &desc, 1) != cudaSuccess) {
      (void)cudaGetLastError();  // best effort
    }
  }
}
}  // namespace

void enable_pool_peer_access_for_all_visible_devices(cudaMemPool_t pool, int owner_device_id)
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
    (void)cudaGetLastError();
    return;
  }
  // Make sure the empirical peer-DMA probe has run before we grant pool
  // access. The probe needs cudaDeviceEnablePeerAccess to be in its enabled
  // state to detect the "lying enable" failure mode, and disables peer access
  // for any pair where direct DMA doesn't actually move bytes. Subsequent
  // cudaMemPoolSetAccess calls then skip the broken pairs.
  ensure_p2p_probed();

  set_access_on_pool(pool, owner_device_id, device_count);

  // Also set access on the device's DEFAULT pool. cudf and other libraries may
  // route allocations through the default pool (e.g. cudf::concatenate when
  // its resource_ref points back to the device default), and without
  // peer-access set on the default pool, cudaMemcpyPeer* between two default
  // pools silently no-ops the same way it does for cudaMallocAsync pools.
  rmm::cuda_set_device_raii set_device(rmm::cuda_device_id{owner_device_id});
  cudaMemPool_t default_pool{};
  if (cudaDeviceGetMemPool(&default_pool, owner_device_id) == cudaSuccess) {
    set_access_on_pool(default_pool, owner_device_id, device_count);
  } else {
    (void)cudaGetLastError();
  }
}

cuda::mr::any_resource<cuda::mr::device_accessible> make_default_gpu_memory_resource(
  int device_id, [[maybe_unused]] size_t capacity)
{
  rmm::cuda_set_device_raii set_device(rmm::cuda_device_id{device_id});
  // Construct the cudaMallocAsync pool WITHOUT priming — passing the cucascade
  // capacity as initial_pool_size makes RMM allocate+immediately-deallocate
  // that many bytes upfront (RAPIDS 26.04 behavior, post-CCCL-MR migration #98),
  // which under cudaMemPoolAttrReleaseThreshold=total leaves all those bytes
  // RETAINED in the pool. With multiple memory_space instances per device
  // (e.g. shared_test_env constructs 3 envs upfront in unittest.cpp main),
  // the cumulative priming exhausts physical GPU memory before any test runs.
  // The reservation_aware_resource_adaptor enforces the cucascade-level budget
  // independently, so the underlying RMM pool can grow lazily without losing
  // budget enforcement.
#if CUCASCADE_RMM_HAS_MOVABLE_ANY_RESOURCE
  rmm::mr::cuda_async_memory_resource concrete_mr;
  enable_pool_peer_access_for_all_visible_devices(concrete_mr.pool_handle(), device_id);
  return {std::move(concrete_mr)};
#else
  auto concrete_mr = std::make_shared<rmm::mr::cuda_async_memory_resource>();
  enable_pool_peer_access_for_all_visible_devices(concrete_mr->pool_handle(), device_id);
  return wrap_legacy_rmm_resource(std::move(concrete_mr));
#endif
}

cuda::mr::any_resource<cuda::mr::device_accessible, cuda::mr::host_accessible>
make_default_host_memory_resource(int numa_node_id, [[maybe_unused]] size_t capacity)
{
  return {cucascade::memory::numa_region_pinned_host_memory_resource(numa_node_id)};
}

DeviceMemoryResourceFactoryFn make_default_allocator_for_tier(Tier tier)
{
  if (tier == Tier::GPU) {
    return make_default_gpu_memory_resource;
  } else if (tier == Tier::HOST) {
    return make_default_host_memory_resource;
  } else {
    return [](int, size_t) {
      return cuda::mr::any_resource<cuda::mr::device_accessible>{null_device_memory_resource{}};
    };
  }
}

bool probe_peer_dma_works(int src_device, int dst_device)
{
  if (src_device == dst_device) return true;
  return p2p_dma_works_cached(src_device, dst_device);
}

int disable_peer_access_where_broken(std::vector<cudaMemPool_t> const& pools_by_device)
{
  // Lazy probe runs the first time enable_pool_peer_access_for_all_visible_devices
  // is called from a memory_space ctor; broken pairs are disabled then. This
  // entry point is kept for API compatibility — it just triggers the probe (if
  // it hasn't run) and reports how many pairs ended up in the host-stage
  // fallback path. The pools_by_device argument is unused under the new
  // architecture (cucascade pools that exist before the probe runs would never
  // have peer access granted to broken peers in the first place).
  (void)pools_by_device;
  ensure_p2p_probed();
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  if (device_count > kMaxDevices) device_count = kMaxDevices;
  int disabled = 0;
  for (int i = 0; i < device_count; ++i) {
    for (int j = 0; j < device_count; ++j) {
      if (i == j) continue;
      int can = 0;
      if (cudaDeviceCanAccessPeer(&can, i, j) != cudaSuccess || !can) {
        (void)cudaGetLastError();
        continue;
      }
      if (!g_p2p_supported[j][i]) ++disabled;
    }
  }
  return disabled;
}

}  // namespace memory
}  // namespace cucascade

namespace std {
size_t hash<cucascade::memory::memory_space_id>::operator()(
  const cucascade::memory::memory_space_id& p) const
{
  return p.uuid();
}
}  // namespace std
