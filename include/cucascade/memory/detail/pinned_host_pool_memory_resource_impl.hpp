/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda/memory_pool>
#include <cuda/memory_resource>
#include <cuda/stream>

#include <cstddef>

namespace cucascade {
namespace memory {

struct pinned_host_pool_config;

namespace detail {

/**
 * @brief The shared state behind every copy of a `pinned_host_pool_memory_resource` handle.
 *
 * `pinned_host_pool_memory_resource` holds one `pinned_host_pool_memory_resource_impl` inside a
 * `::cuda::mr::shared_resource` block, so all copies of a handle use the same object, which owns
 * the CUDA HostNuma memory pool and a private stream for the synchronous calls, and turns every
 * CCCL error into a cuCascade or RMM exception. Its state never changes after construction, so
 * every member is thread-safe.
 */
class pinned_host_pool_memory_resource_impl {
 public:
  /// Alignment of every pointer the pool returns, and the largest alignment a caller may request.
  static constexpr std::size_t pool_alignment = 256;

  /**
   * @brief Validates `config`, then creates the pool and the private stream.
   *
   * @param config The pool settings.
   * @throws cucascade::logic_error if `config` is invalid.
   * @throws cucascade::cuda_error if CUDA cannot be initialized or cannot create the pool or the
   *         private stream.
   */
  explicit pinned_host_pool_memory_resource_impl(pinned_host_pool_config const& config);

  ~pinned_host_pool_memory_resource_impl()                                            = default;
  pinned_host_pool_memory_resource_impl(pinned_host_pool_memory_resource_impl const&) = delete;
  pinned_host_pool_memory_resource_impl& operator=(pinned_host_pool_memory_resource_impl const&) =
    delete;
  pinned_host_pool_memory_resource_impl(pinned_host_pool_memory_resource_impl&&) = delete;
  pinned_host_pool_memory_resource_impl& operator=(pinned_host_pool_memory_resource_impl&&) =
    delete;

  /**
   * @brief Allocates `bytes` ordered on `stream`.
   *
   * @param stream The stream the allocation is ordered on.
   * @param bytes The size of the allocation.
   * @param alignment The requested alignment; a power of two no larger than `pool_alignment`.
   * @return The allocated memory, or `nullptr` if `bytes` is 0.
   * @throws cucascade::logic_error if `alignment` is invalid.
   * @throws rmm::out_of_memory if `bytes` exceeds the limit or the pool or host memory is
   *         exhausted.
   * @throws rmm::bad_alloc on any other CUDA failure.
   */
  [[nodiscard]] void* allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment);

  /**
   * @brief Frees `ptr` ordered on `stream`; does nothing for `nullptr` or 0 bytes.
   *
   * @param stream The stream the free is ordered on.
   * @param ptr The memory to free.
   * @param bytes The size passed to the matching allocation.
   * @param alignment The alignment passed to the matching allocation; checked only in debug builds.
   */
  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment) noexcept;

  /**
   * @brief Allocates `bytes` on the private stream and waits for it.
   *
   * @param bytes The size of the allocation.
   * @param alignment The requested alignment; a power of two no larger than `pool_alignment`.
   * @return The allocated memory, usable by the host at once, or `nullptr` if `bytes` is 0.
   * @throws cucascade::logic_error if `alignment` is invalid.
   * @throws rmm::out_of_memory if `bytes` exceeds the limit or the pool or host memory is
   *         exhausted.
   * @throws rmm::bad_alloc on any other CUDA failure.
   */
  [[nodiscard]] void* allocate_sync(std::size_t bytes, std::size_t alignment);

  /**
   * @brief Frees `ptr` on the private stream and waits for it; does nothing for `nullptr` or 0
   *        bytes.
   *
   * @param ptr The memory to free; every use of it must have finished.
   * @param bytes The size passed to the matching allocation.
   * @param alignment The alignment passed to the matching allocation; checked only in debug builds.
   */
  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept;

  /**
   * @brief Returns the configured byte limit.
   *
   * @return `pinned_host_pool_config::max_pool_bytes` as given at construction.
   */
  [[nodiscard]] std::size_t max_pool_bytes() const noexcept { return _max_pool_bytes; }

  /**
   * @brief Returns the bytes the pool holds from the operating system, in use or idle.
   *
   * @return The pool's current reserved size in bytes.
   * @throws cucascade::cuda_error if the driver query fails.
   */
  [[nodiscard]] std::size_t reserved_bytes() const;

  /**
   * @brief Returns the bytes of the pool allocated to callers.
   *
   * @return The pool's current used size in bytes.
   * @throws cucascade::cuda_error if the driver query fails.
   */
  [[nodiscard]] std::size_t used_bytes() const;

  /**
   * @brief Compares two implementation objects by identity.
   *
   * `::cuda::mr::shared_resource` calls this only for handles that hold different blocks, which
   * always own different pools.
   *
   * @param other The object to compare with.
   * @return `true` only if `other` is this object.
   */
  [[nodiscard]] bool operator==(pinned_host_pool_memory_resource_impl const& other) const noexcept
  {
    return this == &other;
  }

  /**
   * @brief Declares that every visible GPU may access the pool's memory.
   */
  friend constexpr void get_property(pinned_host_pool_memory_resource_impl const&,
                                     ::cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Declares that the host may access the pool's memory.
   */
  friend constexpr void get_property(pinned_host_pool_memory_resource_impl const&,
                                     ::cuda::mr::host_accessible) noexcept
  {
  }

 private:
  std::size_t const _max_pool_bytes;
  ::cuda::shared_pinned_memory_pool _pool;
  ::cuda::stream _private_stream;
};

}  // namespace detail
}  // namespace memory
}  // namespace cucascade
