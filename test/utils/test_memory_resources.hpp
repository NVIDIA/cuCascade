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

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

namespace cucascade::test {

class shared_device_resource {
 public:
  explicit shared_device_resource(rmm::device_async_resource_ref upstream) : upstream_(upstream) {}

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t))
  {
    return upstream_.allocate(stream, bytes, alignment);
  }

  void deallocate(cuda::stream_ref stream,
                  void* p,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    upstream_.deallocate(stream, p, bytes, alignment);
  }

  bool operator==(shared_device_resource const&) const noexcept { return false; }

  friend void get_property(shared_device_resource const&, cuda::mr::device_accessible) noexcept {}

 private:
  rmm::device_async_resource_ref upstream_;
};

inline cuda::mr::any_resource<cuda::mr::device_accessible> make_shared_current_device_resource(
  int, size_t)
{
  return cuda::mr::any_resource<cuda::mr::device_accessible>{
    shared_device_resource{rmm::mr::get_current_device_resource_ref()}};
}

}  // namespace cucascade::test
