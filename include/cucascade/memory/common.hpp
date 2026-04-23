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
