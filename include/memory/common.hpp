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

#include <rmm/mr/device/device_memory_resource.hpp>

#include <cstdint>
#include <cstring>
#include <functional>
#include <utility>

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
  std::function<std::unique_ptr<rmm::mr::device_memory_resource>(int device_id,
                                                                 std::size_t capacity)>;

std::unique_ptr<rmm::mr::device_memory_resource> make_default_gpu_memory_resource(
  int device_id, std::size_t capacity);

std::unique_ptr<rmm::mr::device_memory_resource> make_default_host_memory_resource(
  int device_id, std::size_t capacity);

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
