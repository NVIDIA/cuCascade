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

#include <cstddef>

namespace cucascade {
namespace memory {

/**
 * @brief Opt-in mixin interface for memory resources that allocate in fixed-size chunks.
 *
 * A memory resource that hands out bounded-size chunks rather than arbitrary contiguous
 * ranges can inherit from this interface (alongside `rmm::mr::device_memory_resource`)
 * to advertise its chunk size. Probing code can then use `dynamic_cast` to detect
 * chunked allocators without coupling to specific allocator types.
 *
 * Allocators that do not inherit from this interface are treated as contiguous
 * (a single allocation of the requested size).
 *
 * @code
 * rmm::mr::device_memory_resource* mr = space.get_default_allocator();
 * if (auto* chunked = dynamic_cast<cucascade::memory::chunked_resource_info const*>(mr)) {
 *   auto chunk = chunked->max_chunk_bytes();
 *   // ... iterate chunks ...
 * } else {
 *   // ... single contiguous allocation ...
 * }
 * @endcode
 */
struct chunked_resource_info {
  virtual ~chunked_resource_info() = default;

  /**
   * @brief The maximum size in bytes of a single allocation from this chunked resource.
   *
   * Callers requesting more than this must issue multiple allocations.
   */
  [[nodiscard]] virtual std::size_t max_chunk_bytes() const = 0;
};

}  // namespace memory
}  // namespace cucascade
