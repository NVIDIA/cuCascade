/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fault_injection_hooks.hpp"

#include <array>
#include <utility>

namespace cucascade {
namespace test {
namespace {

constinit thread_local std::array<scoped_fault_hook*, static_cast<std::size_t>(fault_point::count)>
  active_hooks{};

}  // namespace

scoped_fault_hook::scoped_fault_hook(fault_point point,
                                     cudaError_t error,
                                     std::size_t skip) noexcept
  : scoped_fault_hook(point, fail_once, this)
{
  _error = error;
  _skip  = skip;
}

scoped_fault_hook::scoped_fault_hook(fault_point point,
                                     fault_hook_callback callback,
                                     void* context) noexcept
  : _point(point),
    _callback(callback),
    _context(context),
    _previous(std::exchange(active_hooks[static_cast<std::size_t>(point)], this))
{
}

scoped_fault_hook::~scoped_fault_hook()
{
  active_hooks[static_cast<std::size_t>(_point)] = _previous;
}

std::optional<cudaError_t> scoped_fault_hook::fail_once(fault_point_call const&,
                                                        void* context) noexcept
{
  auto& hook = *static_cast<scoped_fault_hook*>(context);
  if (!hook._armed) { return std::nullopt; }
  if (hook._skip != 0) {
    --hook._skip;
    return std::nullopt;
  }
  hook._armed = false;
  return hook._error;
}

}  // namespace test

namespace detail {

std::optional<cudaError_t> intercept_fault_point(fault_point point,
                                                 fault_point_call const& call) noexcept
{
  auto* hook = test::active_hooks[static_cast<std::size_t>(point)];
  return hook ? hook->_callback(call, hook->_context) : std::nullopt;
}

}  // namespace detail
}  // namespace cucascade
