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

#include <cucascade/memory/detail/reservation_aware_resource_adaptor_impl.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <memory>

namespace cucascade {
namespace memory {

/**
 * @brief A memory resource adaptor that tracks allocations on a per-stream basis.
 *
 * This adaptor wraps another device memory resource and provides detailed tracking
 * of allocations per CUDA stream. It maintains both current allocated bytes and
 * the maximum allocated bytes observed for each stream.
 *
 * Features:
 * - Per-stream allocation tracking
 * - Maximum allocated bytes tracking per stream
 * - Thread-safe operations using atomic operations and mutexes
 * - Reset capability for maximum allocated bytes
 * - Race condition handling for deallocations after reset
 *
 * Based on RMM's tracking_resource_adaptor but extended for per-stream tracking.
 *
 * This class inherits from cuda::mr::shared_resource, making it copyable and
 * movable via reference counting. Copies share the same underlying state.
 */
class reservation_aware_resource_adaptor
  : public cuda::mr::shared_resource<detail::reservation_aware_resource_adaptor_impl> {
  using shared_base = cuda::mr::shared_resource<detail::reservation_aware_resource_adaptor_impl>;
  using impl_type   = detail::reservation_aware_resource_adaptor_impl;

 public:
  // Re-export nested types from impl for backward compatibility
  using device_reserved_arena        = impl_type::device_reserved_arena;
  using stream_ordered_tracker_state = impl_type::stream_ordered_tracker_state;
  using allocation_tracker_iface     = impl_type::allocation_tracker_iface;
  using AllocationTrackingScope      = impl_type::AllocationTrackingScope;

  friend void get_property(reservation_aware_resource_adaptor const&,
                           cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Constructs a per-stream tracking resource adaptor.
   *
   * @param space_id The unique identifier for this memory space
   * @param upstream The upstream memory resource to wrap
   * @param capacity The total capacity for allocations
   * @param stream_reservation_policy The default reservation policy for streams
   * @param default_oom_policy The default OOM handling policy
   * @param tracking_scope [default: PER_STREAM] The scope of allocation tracking (per-stream,
   * per-thread)
   */
  explicit reservation_aware_resource_adaptor(
    memory_space_id space_id,
    rmm::device_async_resource_ref upstream,
    std::size_t capacity,
    std::unique_ptr<reservation_limit_policy> stream_reservation_policy = nullptr,
    std::unique_ptr<oom_handling_policy> default_oom_policy             = nullptr,
    AllocationTrackingScope tracking_scope = AllocationTrackingScope::PER_STREAM,
    cudaMemPool_t pool_handle              = nullptr);

  /**
   * @brief Constructs a per-stream tracking resource adaptor.
   *
   * @param space_id The unique identifier for this memory space
   * @param upstream The upstream memory resource to wrap
   * @param memory_limit The memory limit for reservations
   * @param capacity The total capacity for allocations
   * @param stream_reservation_policy The default reservation policy for streams
   * @param default_oom_policy The default OOM handling policy
   * @param tracking_scope [default: PER_STREAM] The scope of allocation tracking (per-stream,
   * per-thread)
   * @param pool_handle Optional CUDA memory pool handle for accurate OOM diagnostics
   */
  explicit reservation_aware_resource_adaptor(
    memory_space_id space_id,
    rmm::device_async_resource_ref upstream,
    std::size_t memory_limit,
    std::size_t capacity,
    std::unique_ptr<reservation_limit_policy> stream_reservation_policy = nullptr,
    std::unique_ptr<oom_handling_policy> default_oom_policy             = nullptr,
    AllocationTrackingScope tracking_scope = AllocationTrackingScope::PER_STREAM,
    cudaMemPool_t pool_handle              = nullptr);

  /**
   * @brief Gets the upstream memory resource.
   * @return Reference to the upstream resource
   */
  rmm::device_async_resource_ref get_upstream_resource() const noexcept;

  /**
   * @brief Returns the available memory left in the resource
   */
  std::size_t get_available_memory() const noexcept;

  /**
   * @brief Returns the available memory left in the resource
   */
  std::size_t get_available_memory(rmm::cuda_stream_view stream) const noexcept;

  std::size_t get_available_memory_print(rmm::cuda_stream_view stream) const noexcept;

  /**
   * @brief Gets the currently allocated bytes for a specific stream.
   * @param stream The CUDA stream to query
   * @return The allocated bytes for the stream
   */
  std::size_t get_allocated_bytes(rmm::cuda_stream_view stream) const;

  /**
   * @brief Gets the peak allocated bytes observed for a specific stream.
   * @param stream The CUDA stream to query
   * @return The peak allocated bytes for the stream
   */
  std::size_t get_peak_allocated_bytes(rmm::cuda_stream_view stream) const;

  /**
   * @brief Gets the total currently allocated bytes across all streams.
   * @return The total allocated bytes
   */
  std::size_t get_total_allocated_bytes() const;

  /**
   * @brief Gets the peak total allocated bytes across all streams.
   * @return The peak total allocated bytes
   */
  std::size_t get_peak_total_allocated_bytes() const;

  /**
   * @brief Resets the peak allocated bytes for a specific stream to 0.
   * @param stream The CUDA stream to reset
   */
  void reset_peak_allocated_bytes(rmm::cuda_stream_view stream);

  /**
   * @brief Gets the total reserved bytes across all streams.
   * @return The total reserved bytes
   */
  std::size_t get_total_reserved_bytes() const;

  /**
   * @brief Checks if a stream is currently being tracked.
   * @param stream The CUDA stream to check
   * @return true if the stream is tracked, false otherwise
   */
  bool is_stream_tracked(rmm::cuda_stream_view stream) const;

  //===----------------------------------------------------------------------===//
  // Reservation Management
  //===----------------------------------------------------------------------===//

  /**
   * @brief makes reservations
   * @param bytes the size of reservation
   * @param release_notifer used to hook callbacks for when the reservation is released
   */
  std::unique_ptr<reserved_arena> reserve(
    std::size_t bytes, std::unique_ptr<event_notifier> release_notifer = nullptr);

  /**
   * @brief makes reservations
   * @param bytes the size of reservation
   * @param release_notifer used to hook callbacks for when the reservation is released
   */
  std::unique_ptr<reserved_arena> reserve_upto(
    std::size_t bytes, std::unique_ptr<event_notifier> release_notifer = nullptr);

  /**
   * @brief Return number of activate reservation count
   */
  std::size_t get_active_reservation_count() const noexcept;

  /**
   * @brief Sets the memory reservation for a specific stream by requesting from the memory manager.
   * @param stream The CUDA stream to set reservation for
   * @param reserved_bytes The reservation object (0 = remove reservation)
   * @param stream_reservation_policy The reservation policy for the stream
   * @param stream_oom_policy The OOM policy for the stream
   * @return true if reservation was successfully set, false otherwise
   */
  bool attach_reservation_to_tracker(
    rmm::cuda_stream_view stream,
    std::unique_ptr<reservation> reserved_bytes,
    std::unique_ptr<reservation_limit_policy> stream_reservation_policy = nullptr,
    std::unique_ptr<oom_handling_policy> stream_oom_policy              = nullptr);

  /**
   * @brief Rests the reservation object for a specific stream.
   * @param stream The CUDA stream to query
   */
  void reset_stream_reservation(rmm::cuda_stream_view stream);

  /**
   * @brief Sets the default reservation policy for new streams.
   * @param policy The default policy to use (takes ownership)
   */
  void set_default_policy(std::unique_ptr<reservation_limit_policy> policy);

  /**
   * @brief Gets the default reservation policy.
   * @return Reference to the default policy
   */
  const reservation_limit_policy& get_default_reservation_policy() const;

  /**
   * @brief Gets the default reservation policy.
   * @return Reference to the default policy
   */
  const oom_handling_policy& get_default_oom_handling_policy() const;

  //===----------------------------------------------------------------------===//
  // Convenience allocate/deallocate with default alignment
  //===----------------------------------------------------------------------===//

  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    return get().allocate(stream, bytes, alignment);
  }

#if !CUCASCADE_RMM_HAS_MOVABLE_ANY_RESOURCE
  void* allocate(rmm::cuda_stream_view stream, std::size_t bytes, std::size_t alignment)
  {
    return allocate(cuda::stream_ref{stream}, bytes, alignment);
  }
#endif

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    get().deallocate(stream, ptr, bytes, alignment);
  }

#if !CUCASCADE_RMM_HAS_MOVABLE_ANY_RESOURCE
  void deallocate(rmm::cuda_stream_view stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    deallocate(cuda::stream_ref{stream}, ptr, bytes, alignment);
  }
#endif
};

}  // namespace memory
}  // namespace cucascade
