/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file fault_injection.hpp
 * @brief Library-private fault points that let tests replace selected CUDA calls and metadata
 * allocations
 *
 * Library sources wrap a fault-prone call in `CUCASCADE_FAULT_POINT_CUDA_CALL` and mark a metadata
 * allocation with `CUCASCADE_FAULT_POINT_METADATA_ALLOCATION`. Without
 * `CUCASCADE_ENABLE_FAULT_INJECTION`, which only the test build of the core defines, both macros
 * expand to the plain call and a no-op, and the context arguments are discarded. With it, every
 * fault point consults `intercept_fault_point`, which this library declares but never defines: the
 * test hooks in `test/memory/fault_injection_hooks.cpp` provide it. Nothing here is installed or
 * reachable from a production build.
 */

#if defined(CUCASCADE_ENABLE_FAULT_INJECTION)

#include <cuda_runtime_api.h>

#include <cstddef>
#include <new>
#include <optional>
#include <utility>

namespace cucascade {
namespace detail {

/**
 * @brief Identifies one CUDA call site or metadata allocation that a test may fail
 */
enum class fault_point {
  host_alloc,
  host_free,
  stream_synchronize,
  stream_device,
  current_device,
  set_device,
  event_create,
  event_record,
  event_wait,
  event_synchronize,
  event_destroy,
  stream_capture_status,
  large_cache_entry,
  dedicated_allocation,
  event_pool_reserve,
  slab_return,
  count
};

/**
 * @brief Arguments and output locations of the call at a fault point
 *
 * Each fault point fills only the fields its call uses. A replacement may write the output
 * locations to emulate a successful call.
 */
struct fault_point_call {
  std::size_t bytes{};
  void* buffer{};
  void** buffer_out{};
  int device{};
  int* device_out{};
  cudaStream_t stream{};
  cudaEvent_t event{};
  cudaEvent_t* event_out{};
  cudaStreamCaptureStatus* capture_status_out{};
};

/**
 * @brief Lets a test replace the call at a fault point on the calling thread
 *
 * Defined only by the test hooks.
 *
 * @param point Fault point being reached
 * @param call Arguments and output locations of the call
 * @return Status that replaces the call, or `std::nullopt` to make the actual call
 */
[[nodiscard]] std::optional<cudaError_t> intercept_fault_point(
  fault_point point, fault_point_call const& call) noexcept;

/**
 * @brief Makes a CUDA call unless a test replaces it
 *
 * @param point Fault point being reached
 * @param actual_call Non-throwing callable that makes the CUDA call
 * @param call Arguments and output locations passed to `intercept_fault_point`
 * @return The replacement status, or the status of `actual_call`
 */
template <typename Call>
[[nodiscard]] cudaError_t invoke_fault_point(fault_point point,
                                             Call&& actual_call,
                                             fault_point_call const& call = {}) noexcept
{
  if (auto replaced = intercept_fault_point(point, call)) { return *replaced; }
  return std::forward<Call>(actual_call)();
}

/**
 * @brief Emulates exhaustion before a metadata allocation when a test fails the fault point
 *
 * @param point Fault point being reached
 * @throws std::bad_alloc If the test replaces the point with any status other than `cudaSuccess`
 */
inline void before_metadata_allocation(fault_point point)
{
  if (auto replaced = intercept_fault_point(point, {}); replaced && *replaced != cudaSuccess) {
    throw std::bad_alloc{};
  }
}

}  // namespace detail
}  // namespace cucascade

#define CUCASCADE_FAULT_POINT_CUDA_CALL(point, call, ...)                                  \
  ::cucascade::detail::invoke_fault_point(::cucascade::detail::fault_point::point,         \
                                          [&]() noexcept { return (call); } __VA_OPT__(, ) \
                                            __VA_ARGS__)
#define CUCASCADE_FAULT_POINT_METADATA_ALLOCATION(point) \
  ::cucascade::detail::before_metadata_allocation(::cucascade::detail::fault_point::point)
#else
#define CUCASCADE_FAULT_POINT_CUDA_CALL(point, call, ...) (call)
#define CUCASCADE_FAULT_POINT_METADATA_ALLOCATION(point)  ((void)0)
#endif
