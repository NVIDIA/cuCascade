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

#include "memory/common.hpp"

#include "memory/null_device_memory_resource.hpp"
#include "memory/numa_region_pinned_host_allocator.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>

namespace cucascade {

namespace memory {

std::unique_ptr<rmm::mr::device_memory_resource> make_default_gpu_memory_resource(int device_id,
                                                                                  size_t capacity)
{
  rmm::cuda_set_device_raii set_device(rmm::cuda_device_id{device_id});
  return std::make_unique<rmm::mr::cuda_async_memory_resource>(capacity);
}

std::unique_ptr<rmm::mr::device_memory_resource> make_default_host_memory_resource(
  int numa_node_id, [[maybe_unused]] size_t capacity)
{
  return std::make_unique<cucascade::memory::numa_region_pinned_host_memory_resource>(numa_node_id);
}

DeviceMemoryResourceFactoryFn make_default_allocator_for_tier(Tier tier)
{
  if (tier == Tier::GPU) {
    return make_default_gpu_memory_resource;
  } else if (tier == Tier::HOST) {
    return make_default_host_memory_resource;
  } else {
    return [](int, size_t) { return std::make_unique<null_device_memory_resource>(); };
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
