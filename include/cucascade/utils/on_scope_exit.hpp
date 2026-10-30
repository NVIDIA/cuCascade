/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <type_traits>
#include <utility>

namespace cucascade {
namespace utils {

/**
 * @brief Run a noexcept callback once on destruction, including exception exits.
 *
 * For local completion notifications and cleanup. Stores the callable inline;
 * adds no allocation, locking, or synchronization. Captured references must
 * outlive the guard. Copying, moving, and cancellation are unsupported.
 * Construction may throw if moving the callable throws; no callback then runs.
 *
 * @code
 * auto completion = on_scope_exit{[&]() noexcept { notify(success); }};
 * @endcode
 */
template <typename Fn>
  requires std::is_nothrow_invocable_v<Fn&>
class on_scope_exit {
 public:
  explicit on_scope_exit(Fn fn) noexcept(std::is_nothrow_move_constructible_v<Fn>)
    : _fn(std::move(fn))
  {
  }

  ~on_scope_exit() noexcept { _fn(); }

  on_scope_exit(const on_scope_exit&)            = delete;
  on_scope_exit& operator=(const on_scope_exit&) = delete;
  on_scope_exit(on_scope_exit&&)                 = delete;
  on_scope_exit& operator=(on_scope_exit&&)      = delete;

 private:
  Fn _fn;
};

}  // namespace utils
}  // namespace cucascade
