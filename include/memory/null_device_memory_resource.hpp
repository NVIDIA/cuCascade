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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace cucascade {
namespace memory {

/**
 * A no-op device_memory_resource used for DISK tier to satisfy API requirements.
 * - allocate always returns nullptr
 * - deallocate is a no-op
 */
class null_device_memory_resource : public rmm::mr::device_memory_resource {
 public:
  null_device_memory_resource()           = default;
  ~null_device_memory_resource() override = default;

 protected:
  void* do_allocate([[maybe_unused]] std::size_t bytes,
                    [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    return nullptr;
  }
  void do_deallocate([[maybe_unused]] void* p,
                     [[maybe_unused]] std::size_t bytes,
                     [[maybe_unused]] rmm::cuda_stream_view stream) noexcept override
  {
  }
  [[nodiscard]] bool do_is_equal(
    const rmm::mr::device_memory_resource& other) const noexcept override
  {
    return this == &other;
  }
};

}  // namespace memory
}  // namespace cucascade
