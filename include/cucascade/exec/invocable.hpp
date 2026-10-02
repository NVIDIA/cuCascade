/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

/**
 * @file invocable.hpp
 * @brief Move-only type-erased callable used across the exec layer.
 *
 * `cucascade::exec::invocable<Signature>` is the single spelling used by the
 * thread pool, dispatcher, and future/promise continuations. Two backing
 * implementations are selected at configure time:
 *
 *   - Default (standard-library only): a minimal in-tree implementation, a
 *     stand-in for C++23 `std::move_only_function` on the C++20 toolchain.
 *   - `CUCASCADE_USE_ABSEIL_INVOCABLE` defined: an alias for
 *     `absl::AnyInvocable`. Enable via the `CUCASCADE_USE_ABSEIL_INVOCABLE`
 *     CMake option so the library reuses a host project's abseil (e.g. when
 *     embedded in a codebase that already depends on it).
 *
 * The macro must be defined consistently for the library build and every
 * consumer that includes this header; the CMake option propagates it as a
 * public compile definition to guarantee that.
 */

#ifdef CUCASCADE_USE_ABSEIL_INVOCABLE

#include <absl/functional/any_invocable.h>

namespace cucascade::exec {

/**
 * @brief Move-only type-erased callable, backed by `absl::AnyInvocable`.
 */
template <typename Signature>
using invocable = absl::AnyInvocable<Signature>;

}  // namespace cucascade::exec

#else  // CUCASCADE_USE_ABSEIL_INVOCABLE

#include <concepts>
#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace cucascade::exec {

/**
 * @brief Minimal move-only type-erased callable, standard-library only.
 *
 * A stand-in for C++23 std::move_only_function on the C++20 toolchain:
 * futures/promises capture move-only state in their continuations, which
 * std::function (copyable target required) cannot hold.
 *
 * Mirrors the signature flavours of `absl::AnyInvocable` that the exec and io
 * layers rely on: `R(Args...)`, `R(Args...) noexcept`, `R(Args...) &&` and
 * `R(Args...) && noexcept`.  A `noexcept` signature only accepts targets that
 * are nothrow-invocable; an `&&` signature may only be invoked on an rvalue.
 */
template <typename Signature>
class invocable;

namespace detail {

/// Type-erased move-only storage shared by every `invocable` flavour.
template <bool NoExcept, typename R, typename... Args>
class invocable_storage {
 public:
  invocable_storage() noexcept = default;

  template <typename F>
  explicit invocable_storage(F&& f)
    : _impl(std::make_unique<model<std::decay_t<F>>>(std::forward<F>(f)))
  {
  }

  [[nodiscard]] bool has_target() const noexcept { return _impl != nullptr; }

  void reset() noexcept { _impl.reset(); }

  R call(Args&&... args) noexcept(NoExcept) { return _impl->invoke(std::forward<Args>(args)...); }

 private:
  struct concept_t {
    virtual ~concept_t()                                = default;
    virtual R invoke(Args&&... args) noexcept(NoExcept) = 0;
  };

  template <typename F>
  struct model final : concept_t {
    explicit model(F f) : fn(std::move(f)) {}
    R invoke(Args&&... args) noexcept(NoExcept) override
    {
      return std::invoke(fn, std::forward<Args>(args)...);
    }
    F fn;
  };

  std::unique_ptr<concept_t> _impl;
};

/// A target is accepted when it is invocable as an lvalue, and additionally
/// nothrow-invocable for a `noexcept` signature.
template <bool NoExcept, typename F, typename... Args>
inline constexpr bool invocable_target_v =
  std::invocable<std::decay_t<F>&, Args...> &&
  (!NoExcept || std::is_nothrow_invocable_v<std::decay_t<F>&, Args...>);

}  // namespace detail

// clang-format off
#define CUCASCADE_EXEC_DEFINE_INVOCABLE(QUAL, NOEXCEPT_SPEC, NOEXCEPT_BOOL)                      \
  template <typename R, typename... Args>                                                        \
  class invocable<R(Args...) QUAL NOEXCEPT_SPEC> {                                               \
   public:                                                                                       \
    invocable() noexcept = default;                                                              \
    invocable(std::nullptr_t) noexcept {} /* NOLINT(google-explicit-constructor) */              \
                                                                                                 \
    template <typename F>                                                                        \
      requires(!std::same_as<std::remove_cvref_t<F>, invocable> &&                               \
               detail::invocable_target_v<NOEXCEPT_BOOL, F, Args...>)                            \
    invocable(F&& f) /* NOLINT(google-explicit-constructor) */                                   \
      : _storage(std::forward<F>(f))                                                             \
    {                                                                                            \
    }                                                                                            \
                                                                                                 \
    invocable(invocable&&) noexcept            = default;                                        \
    invocable& operator=(invocable&&) noexcept = default;                                        \
    invocable(const invocable&)                = delete;                                         \
    invocable& operator=(const invocable&)     = delete;                                         \
    ~invocable()                               = default;                                        \
                                                                                                 \
    invocable& operator=(std::nullptr_t) noexcept                                                \
    {                                                                                            \
      _storage.reset();                                                                          \
      return *this;                                                                              \
    }                                                                                            \
                                                                                                 \
    [[nodiscard]] explicit operator bool() const noexcept { return _storage.has_target(); }      \
                                                                                                 \
    friend bool operator==(const invocable& f, std::nullptr_t) noexcept { return !f; }           \
                                                                                                 \
    R operator()(Args... args) QUAL NOEXCEPT_SPEC                                                \
    {                                                                                            \
      return _storage.call(std::forward<Args>(args)...);                                         \
    }                                                                                            \
                                                                                                 \
   private:                                                                                      \
    detail::invocable_storage<NOEXCEPT_BOOL, R, Args...> _storage;                               \
  }

CUCASCADE_EXEC_DEFINE_INVOCABLE(, , false);
CUCASCADE_EXEC_DEFINE_INVOCABLE(, noexcept, true);
CUCASCADE_EXEC_DEFINE_INVOCABLE(&&, , false);
CUCASCADE_EXEC_DEFINE_INVOCABLE(&&, noexcept, true);
// clang-format on

#undef CUCASCADE_EXEC_DEFINE_INVOCABLE

}  // namespace cucascade::exec

#endif  // CUCASCADE_USE_ABSEIL_INVOCABLE
