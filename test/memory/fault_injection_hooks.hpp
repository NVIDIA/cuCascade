/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file fault_injection_hooks.hpp
 * @brief Test-side implementation of the library fault points declared in `src/fault_injection.hpp`
 *
 * Only targets linked against the fault-injection build of the core
 * (`cucascade_fault_injection_core`) may include this header.
 */

#include "fault_injection.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <optional>

namespace cucascade {
namespace test {

using fault_point      = ::cucascade::detail::fault_point;
using fault_point_call = ::cucascade::detail::fault_point_call;

/**
 * @brief Replaces, or observes and forwards, the call at a fault point
 *
 * Returning `std::nullopt` makes the actual call; returning a status replaces it.
 */
using fault_hook_callback = std::optional<cudaError_t> (*)(fault_point_call const&,
                                                           void* context) noexcept;

/**
 * @brief Overrides one fault point on the calling thread until destruction
 *
 * The error constructor replaces one call after skipping the requested number. Nested hooks on the
 * same fault point restore the previous hook, and no hook affects another thread.
 */
class scoped_fault_hook {
 public:
  scoped_fault_hook(fault_point point, cudaError_t error, std::size_t skip = 0) noexcept;
  scoped_fault_hook(fault_point point, fault_hook_callback callback, void* context) noexcept;
  ~scoped_fault_hook();

  scoped_fault_hook(scoped_fault_hook const&)            = delete;
  scoped_fault_hook& operator=(scoped_fault_hook const&) = delete;
  scoped_fault_hook(scoped_fault_hook&&)                 = delete;
  scoped_fault_hook& operator=(scoped_fault_hook&&)      = delete;

 private:
  friend std::optional<cudaError_t> detail::intercept_fault_point(fault_point,
                                                                  fault_point_call const&) noexcept;

  static std::optional<cudaError_t> fail_once(fault_point_call const&, void* context) noexcept;

  fault_point _point;
  fault_hook_callback _callback;
  void* _context;
  scoped_fault_hook* _previous;
  cudaError_t _error{cudaSuccess};
  std::size_t _skip{};
  bool _armed{true};
};

}  // namespace test
}  // namespace cucascade
