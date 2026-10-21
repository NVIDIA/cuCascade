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

#include <cucascade/memory/chunked_resource_info.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/disk_access_limiter.hpp>
#include <cucascade/memory/notification_channel.hpp>

#include <cuda/memory_resource>

#include <concepts>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <variant>

// RMM includes for memory resource management
#include <cucascade/cuda/stream.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/resource_ref.hpp>

namespace cucascade {

// Forward declaration: DISK-tier memory spaces hold the io context that serves their files. The
// core library only stores the pointer; all I/O through it happens in the cucascade-cudf layer.
namespace io {
class ioctx;
}  // namespace io

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
  /**
   * Construct a DISK-tier memory_space.
   *
   * The io context in @p config (if any) serves this space's files; without one, the DISK
   * converters adopt the process-shared default on first use (see get_or_init_io_context()).
   */
  explicit memory_space(const disk_memory_space_config& config);

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
  [[nodiscard]] ::cuda::stream_ref acquire_stream() const;

  [[nodiscard]] std::size_t get_active_reservation_count() const;
  [[nodiscard]] bool should_downgrade_memory() const;
  [[nodiscard]] bool should_stop_downgrading_memory() const;
  [[nodiscard]] size_t get_amount_to_downgrade() const;

  // State queries
  [[nodiscard]] size_t get_available_memory(::cuda::stream_ref stream) const;
  [[nodiscard]] size_t get_available_memory() const;
  [[nodiscard]] size_t get_total_reserved_memory() const;
  [[nodiscard]] size_t get_max_memory() const noexcept;

  /** @brief Configured capacity of this space in bytes (the base of the threshold fractions). */
  [[nodiscard]] size_t get_capacity() const noexcept;

  /**
   * @brief Consumed bytes (`get_max_memory() - get_available_memory()`) at or above which
   * `should_downgrade_memory()` reports true: capacity x downgrade trigger fraction.
   */
  [[nodiscard]] size_t get_downgrade_trigger_threshold() const noexcept;

  /**
   * @brief Consumed bytes at or below which `should_stop_downgrading_memory()` reports true:
   * capacity x downgrade stop fraction.
   */
  [[nodiscard]] size_t get_downgrade_stop_threshold() const noexcept;

  // Allocator management
  [[nodiscard]] rmm::device_async_resource_ref get_default_allocator() const noexcept;

  /**
   * @brief Probe the underlying allocator for the `chunked_resource_info` mixin.
   *
   * @return Non-null pointer to the mixin interface if the underlying allocator inherits from
   * `chunked_resource_info`; `nullptr` otherwise (contiguous allocator).
   */
  [[nodiscard]] const chunked_resource_info* get_chunked_resource_info() const noexcept;

  template <typename T>
    requires(::cuda::mr::resource_with<T, ::cuda::mr::device_accessible> ||
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

  /**
   * @brief Get the io context that serves this DISK space's files.
   *
   * @return The configured or adopted context; null if none has been set yet.
   * @throws std::logic_error if this is not a DISK-tier memory space.
   */
  [[nodiscard]] std::shared_ptr<io::ioctx> get_io_context() const;

  /**
   * @brief Replace the io context that serves this DISK space's files.
   *
   * Thread-safe. Conversions already in flight keep using the context they started with (they
   * hold their own reference to it). The context must be started and support writes.
   *
   * @param context The new context (must not be null).
   * @throws std::logic_error if this is not a DISK-tier memory space.
   * @throws std::invalid_argument if @p context is null.
   */
  void set_io_context(std::shared_ptr<io::ioctx> context);

  /**
   * @brief Get the io context, installing `make_default()` first when none is set.
   *
   * Thread-safe: `make_default` runs at most once per successful installation, under this space's
   * lock, so concurrent callers all observe the same context.
   *
   * @param make_default Factory for the context to adopt when none is set.
   * @return The (possibly just installed) context; never null.
   * @throws std::logic_error if this is not a DISK-tier memory space.
   * @throws std::runtime_error if `make_default` returns null (nothing is installed).
   */
  std::shared_ptr<io::ioctx> get_or_init_io_context(
    const std::function<std::shared_ptr<io::ioctx>()>& make_default) const;

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
  ::cuda::mr::any_resource<::cuda::mr::device_accessible> _allocator;
  reserving_adaptor_type _reservation_allocator;
  std::optional<::cuda::mr::any_resource<::cuda::mr::device_accessible>>
    _reservation_allocator_resource;
  std::unique_ptr<rmm::cuda_stream_pool> _stream_pool;
  /// io context serving a DISK space's files (null for other tiers and until set or adopted).
  /// Mutable: get_or_init_io_context() lazily adopts a default on a const space.
  mutable std::shared_ptr<io::ioctx> _io_context;
  mutable std::mutex _io_context_mutex;  ///< Guards _io_context
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
