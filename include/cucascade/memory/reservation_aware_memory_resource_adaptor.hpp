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
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/oom_handling_policy.hpp>
#include <cucascade/memory/reservation_aware_memory_resource.hpp>

#include <rmm/aligned.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>

namespace cucascade {
namespace memory {

namespace detail {
struct adaptor_state;  // defined in src/memory/reservation_aware_memory_resource_adaptor.cpp
}  // namespace detail

/// @brief Attach/detach key: a CUDA stream. Written as `on_stream{stream}` where `stream` is a
/// `::cuda::stream_ref`, a `cudaStream_t` (e.g. `cudaStream_t{nullptr}` for the legacy default
/// stream) or an `rmm::cuda_stream` (implicit conversions). `on_stream{nullptr}` / `on_stream{0}`
/// are ill-formed on purpose.
struct on_stream {
  ::cuda::stream_ref stream;
  explicit on_stream(::cuda::stream_ref s) noexcept : stream(s) {}
};

/**
 * @brief Per-stream view over a reservation_aware_memory_resource.
 *
 * Binds a reservation to a stream with attach(); plain allocate/deallocate on that stream are then
 * routed through the bound reservation (and its policies). Streams without a binding use the
 * resource's untracked path. Copies share the same bindings (reference semantics, like
 * ::cuda::mr::shared_resource) and compare equal; this lets the adaptor be stored in
 * rmm::device_async_resource_ref and ::cuda::mr::any_resource (rmm::device_buffer keeps a copy of
 * the handle for its lifetime). A moved-from adaptor may only be destroyed or assigned to.
 *
 * Thread safety: all member functions are safe for concurrent use, including attach/detach racing
 * with allocate/deallocate on the same stream. detach() blocks until in-flight allocate/deallocate
 * calls on that stream have completed (it holds the adaptor's writer lock meanwhile); hence do not
 * call attach() or detach() from inside an OOM or overflow policy callback (it could wait for
 * itself). The allocate/deallocate path is lock-free apart from the reservation's own short
 * critical section and performs no heap allocation unless an OOM policy is consulted.
 *
 * Lifetime: the resource must outlive every allocate/deallocate/attach call and every call that
 * reports resource-wide figures (get_available_memory, get_upstream_resource, get_root_resource),
 * hence every rmm::device_buffer that holds a copy of this adaptor; destroying the last copy after
 * the resource is gone is safe (bound reservations are released against the shared accounting
 * core).
 *
 * Streams: bindings are keyed by the raw stream handle. detach() a stream before destroying it
 * (cudaStreamDestroy): CUDA may hand the same handle value to a later stream, which would then
 * inherit the stale binding.
 *
 * Known limitation: per-stream bookkeeping (about 128 bytes, two cache lines, per distinct stream
 * ever attached plus table entries) is kept until the last copy is destroyed; attaching many
 * short-lived streams grows it.
 *
 * Naming: get_upstream_resource() / get_root_resource() are the sketch's
 * upstream_memory_resource() / root_memory_resource() (repo `get_` convention).
 */
class reservation_aware_memory_resource_adaptor {
 public:
  using reservation = reservation_aware_memory_resource::reservation;

  /// @param upstream Resource this adaptor routes to (must outlive allocate/deallocate calls).
  explicit reservation_aware_memory_resource_adaptor(reservation_aware_memory_resource& upstream);

  reservation_aware_memory_resource_adaptor(
    reservation_aware_memory_resource_adaptor const&) noexcept;
  reservation_aware_memory_resource_adaptor& operator=(
    reservation_aware_memory_resource_adaptor const&) noexcept;
  reservation_aware_memory_resource_adaptor(reservation_aware_memory_resource_adaptor&&) noexcept;
  reservation_aware_memory_resource_adaptor& operator=(
    reservation_aware_memory_resource_adaptor&&) noexcept;
  ~reservation_aware_memory_resource_adaptor();

  /// @return the reservation-aware resource this adaptor routes to.
  [[nodiscard]] reservation_aware_memory_resource& get_upstream_resource() const noexcept;
  /// @return the root device resource wrapped by the reservation-aware resource.
  [[nodiscard]] rmm::device_async_resource_ref get_root_resource() const noexcept;

  //===----------------------------------------------------------------------===//
  // Bindings
  //===----------------------------------------------------------------------===//

  /**
   * @brief Binds @p res to @p key. Allocations on that stream draw from @p res afterwards.
   * @param key Stream to bind.
   * @param res Reservation to bind; consumed only on success.
   * @param oom Policy when the upstream is out of memory (nullptr = resource default).
   * @param overflow Policy when an allocation does not fit the reservation (nullptr = resource
   *        default).
   * @throws cucascade::logic_error if @p res is empty, belongs to another resource, or the stream
   *         is already bound. On any exception @p res is left untouched (strong guarantee).
   */
  void attach(on_stream key,
              reservation&& res,
              std::unique_ptr<oom_handling_policy> oom  = nullptr,
              std::unique_ptr<overflow_policy> overflow = nullptr);

  /**
   * @brief Unbinds the stream and returns its reservation (empty handle if none was bound).
   *
   * Live allocations made through the binding stay accounted inside the returned reservation;
   * freeing them later on the (now unbound) stream credits the global counter; freeing them through
   * the same reservation (re-attached, or via the resource's reservation-taking deallocate) credits
   * the reservation. Blocks until in-flight allocate/deallocate calls on @p key have completed;
   * must not be called from inside an OOM or overflow policy callback. Call it before the stream is
   * destroyed (a recycled handle value would otherwise inherit the binding).
   */
  [[nodiscard]] reservation detach(on_stream key);

  /// @return true if a reservation is bound to @p key.
  [[nodiscard]] bool is_attached(on_stream key) const noexcept;
  /// @return bytes allocated through the stream's reservation (0 if unbound).
  [[nodiscard]] std::size_t get_allocated_bytes(on_stream key) const noexcept;
  /// @return high-water mark of the stream reservation's allocated bytes (0 if unbound).
  [[nodiscard]] std::size_t get_peak_allocated_bytes(on_stream key) const noexcept;
  /// @return resource available memory + the stream reservation's available bytes.
  [[nodiscard]] std::size_t get_available_memory(on_stream key) const noexcept;
  /// @brief Resets the stream reservation's peak to 0 (no-op if unbound).
  void reset_peak_allocated_bytes(on_stream key) noexcept;

  //===----------------------------------------------------------------------===//
  // ::cuda::mr::resource interface
  //===----------------------------------------------------------------------===//

  /**
   * @brief Allocates @p bytes on @p stream: through the stream's reservation if one is bound
   * (with the binding's policies), otherwise through the resource's untracked path. As on the
   * resource, a request with @p bytes > capacity is rejected up front (LIMIT_EXCEEDED) without
   * consulting any policy.
   * @throws cucascade::memory::cucascade_out_of_memory (LIMIT_EXCEEDED / ALLOCATION_FAILED).
   * @throws rmm::out_of_memory from a fail/increase overflow policy.
   */
  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);
  /// @brief Returns @p ptr; credits the stream's reservation if one is bound, else the global
  /// counter.
  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;
  /// @brief allocate() on the legacy default stream, then synchronizes it; if the synchronization
  /// throws, the allocation is returned (deallocate() on the same stream) before it propagates.
  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);
  /// @brief deallocate() on the legacy default stream, then synchronizes it.
  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  /// @return true iff both handles alias the same adaptor state.
  bool operator==(reservation_aware_memory_resource_adaptor const& other) const noexcept;

  friend void get_property(reservation_aware_memory_resource_adaptor const&,
                           ::cuda::mr::device_accessible) noexcept
  {
  }

 private:
  std::shared_ptr<detail::adaptor_state> _state;
};

static_assert(::cuda::mr::resource_with<reservation_aware_memory_resource_adaptor,
                                        ::cuda::mr::device_accessible>);

}  // namespace memory
}  // namespace cucascade
