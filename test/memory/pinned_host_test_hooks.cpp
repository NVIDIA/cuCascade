/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pinned_host_test_hooks.hpp"

#include <array>
#include <new>

namespace cucascade {
namespace test {
namespace {

constinit thread_local std::array<scoped_pinned_host_hook*,
                                  static_cast<std::size_t>(pinned_host_operation::count)>
  active_hooks{};

}  // namespace

scoped_pinned_host_hook::scoped_pinned_host_hook(pinned_host_operation operation,
                                                 cudaError_t error,
                                                 std::size_t skip) noexcept
  : scoped_pinned_host_hook(operation, fail_once, this)
{
  _error = error;
  _skip  = skip;
}

scoped_pinned_host_hook::scoped_pinned_host_hook(pinned_host_operation operation,
                                                 pinned_host_callback callback,
                                                 void* context) noexcept
  : _operation(operation),
    _callback(callback),
    _context(context),
    _previous(std::exchange(active_hooks[static_cast<std::size_t>(operation)], this))
{
}

scoped_pinned_host_hook::~scoped_pinned_host_hook()
{
  active_hooks[static_cast<std::size_t>(_operation)] = _previous;
}

std::optional<cudaError_t> scoped_pinned_host_hook::fail_once(pinned_host_call const&,
                                                              void* context) noexcept
{
  auto& hook = *static_cast<scoped_pinned_host_hook*>(context);
  if (!hook._armed) { return std::nullopt; }
  if (hook._skip != 0) {
    --hook._skip;
    return std::nullopt;
  }
  hook._armed = false;
  return hook._error;
}

std::optional<cudaError_t> intercept_pinned_host_call(pinned_host_operation operation,
                                                      pinned_host_call const& call) noexcept
{
  auto* hook = active_hooks[static_cast<std::size_t>(operation)];
  return hook ? hook->_callback(call, hook->_context) : std::nullopt;
}

void before_pinned_host_metadata(pinned_host_operation operation)
{
  if (auto error = intercept_pinned_host_call(operation, {}); error && *error != cudaSuccess) {
    throw std::bad_alloc{};
  }
}

}  // namespace test
}  // namespace cucascade
