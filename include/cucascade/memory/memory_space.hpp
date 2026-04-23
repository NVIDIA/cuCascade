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

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/disk_access_limiter.hpp>
#include <cucascade/memory/notification_channel.hpp>

#include <cuda/memory_resource>

#include <concepts>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <variant>

// RMM includes for memory resource management
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

namespace cucascade {

// Forward declaration for disk I/O backend owned by DISK-tier memory spaces
class idisk_io_backend;

namespace memory {

// Forward declaration
class reservation;
class reservation_aware_resource_adaptor;
class fixed_size_host_memory_resource;

/**
 * memory_space represents a specific memory location identified by a tier and device ID.
 * It manages memory reservations within that space and owns allocator resources.
 *
 * Each memory_space:
 * - Has a fixed memory limit
 * - Tracks active reservations
 * - Provides thread-safe reservation management
 * - Owns one or more RMM memory allocators
 */
class memory_space {
 public:
  /**
   * Construct a memory_space with the given parameters.
   *
   * @param config Configuration for the memory space
   */
  explicit memory_space(const gpu_memory_space_config& config);
  explicit memory_space(const host_memory_space_config& config);
  explicit memory_space(const disk_memory_space_config& config);

  /**
   * Construct a DISK-tier memory_space with a specific I/O backend.
   *
   * @param config Configuration for the disk memory space
   * @param io_backend The I/O backend to use for this disk space (must not be null)
   */
  memory_space(const disk_memory_space_config& config,
               std::shared_ptr<idisk_io_backend> io_backend);

  // Disable copy/move to ensure stable addresses for reservations
  memory_space(const memory_space&)            = delete;
  memory_space& operator=(const memory_space&) = delete;
  memory_space(memory_space&&)                 = delete;
  memory_space& operator=(memory_space&&)      = delete;

  ~memory_space();

  // Comparison operators
  bool operator==(const memory_space& other) const;
  bool operator!=(const memory_space& other) const;

  // Basic properties
  [[nodiscard]] memory_space_id get_id() const noexcept;
  [[nodiscard]] Tier get_tier() const noexcept;
  [[nodiscard]] int get_device_id() const noexcept;

  // Reservation management - these are the core methods that do the actual work
  std::unique_ptr<reservation> make_reservation_or_null(size_t size);
  std::unique_ptr<reservation> make_reservation_upto(size_t size);
  std::unique_ptr<reservation> make_reservation(size_t size);
  [[nodiscard]] rmm::cuda_stream_view acquire_stream() const;

  [[nodiscard]] std::size_t get_active_reservation_count() const;
  [[nodiscard]] bool should_downgrade_memory() const;
  [[nodiscard]] bool should_stop_downgrading_memory() const;
  [[nodiscard]] size_t get_amount_to_downgrade() const;

  // State queries
  [[nodiscard]] size_t get_available_memory(rmm::cuda_stream_view stream) const;
  [[nodiscard]] size_t get_available_memory() const;
  [[nodiscard]] size_t get_total_reserved_memory() const;
  [[nodiscard]] size_t get_max_memory() const noexcept;

  // Allocator management
  [[nodiscard]] rmm::device_async_resource_ref get_default_allocator() const noexcept;

  template <typename T>
    requires(cuda::mr::resource_with<T, cuda::mr::device_accessible> ||
             std::same_as<T, disk_access_limiter>)
  T* get_memory_resource_as() const noexcept
  {
    T* result = nullptr;
    std::visit(
      [&result](const auto& ptr) {
        using held_type = std::decay_t<decltype(*ptr)>;
        if constexpr (std::is_same_v<held_type, T>) { result = ptr.get(); }
      },
      _reservation_allocator);
    return result;
  }

  template <Tier TIER>
  auto* get_memory_resource_of() const noexcept
  {
    return get_memory_resource_as<typename tier_memory_resource_trait<TIER>::type>();
  }

  /** @brief Get the mount path for a DISK tier memory space. Throws if not DISK tier. */
  [[nodiscard]] std::string_view get_disk_mount_path() const;

  /** @brief Get the I/O backend for a DISK tier memory space. Throws if not DISK tier. */
  [[nodiscard]] idisk_io_backend& get_io_backend() const;

  // Utility methods
  std::string to_string() const;

  void shutdown();

 protected:
  friend class reservation;

  const memory_space_id _id;
  const size_t _capacity;
  const size_t _memory_limit;
  const size_t _start_downgrading_memory_threshold;
  const size_t _stop_downgrading_memory_threshold;
  using reserving_adaptor_type = std::variant<std::unique_ptr<reservation_aware_resource_adaptor>,
                                              std::unique_ptr<fixed_size_host_memory_resource>,
                                              std::unique_ptr<disk_access_limiter>>;

  std::shared_ptr<notification_channel> _notification_channel =
    std::make_shared<notification_channel>();

  // Memory resources owned by this memory_space
  cuda::mr::any_resource<cuda::mr::device_accessible> _allocator;
  reserving_adaptor_type _reservation_allocator;
  std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>>
    _reservation_allocator_resource;
  std::unique_ptr<rmm::cuda_stream_pool> _stream_pool;
  std::shared_ptr<idisk_io_backend> _io_backend;  ///< I/O backend for DISK tier (null for others)
};

/**
 * Hash function for memory_space to enable use in unordered containers.
 * Hash is based on tier and device_id combination.
 */
struct memory_space_hash {
  size_t operator()(const memory_space& ms) const;
};

template <typename T>
T* reservation::get_memory_resource_as() const noexcept
{
  return _space->get_memory_resource_as<T>();
}

template <Tier TIER>
auto* reservation::get_memory_resource_of() const noexcept
{
  return get_memory_resource_as<typename tier_memory_resource_trait<TIER>::type>();
}

}  // namespace memory
}  // namespace cucascade
