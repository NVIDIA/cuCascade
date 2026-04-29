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

namespace cucascade {

namespace memory {

void enable_pool_peer_access_for_all_visible_devices(cudaMemPool_t pool, int owner_device_id)
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
    (void)cudaGetLastError();
    return;
  }
  for (int peer = 0; peer < device_count; ++peer) {
    if (peer == owner_device_id) { continue; }
    int can_access = 0;
    if (cudaDeviceCanAccessPeer(&can_access, peer, owner_device_id) != cudaSuccess ||
        !can_access) {
      (void)cudaGetLastError();
      continue;
    }
    cudaMemAccessDesc desc{};
    desc.location.type = cudaMemLocationTypeDevice;
    desc.location.id   = peer;
    desc.flags         = cudaMemAccessFlagsProtReadWrite;
    if (cudaMemPoolSetAccess(pool, &desc, 1) != cudaSuccess) {
      (void)cudaGetLastError();  // best effort; non-fatal on systems without P2P
    }
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

}  // namespace memory
}  // namespace cucascade

namespace std {
size_t hash<cucascade::memory::memory_space_id>::operator()(
  const cucascade::memory::memory_space_id& p) const
{
  return p.uuid();
}
}  // namespace std
