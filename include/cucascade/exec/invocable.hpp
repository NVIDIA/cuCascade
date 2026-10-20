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
#include <cstring>
#include <functional>
#include <memory>
#include <new>
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
 *
 * A target of at most 48 bytes (alignment 16) whose move cannot throw is stored
 * in place, without allocating, as `absl::AnyInvocable` and folly::Function do;
 * a larger one is allocated once.  A null function pointer is no target.
 */
template <typename Signature>
class invocable;

namespace detail {

/// In-place buffer of `invocable` (folly::Function's size and alignment): a continuation that
/// captures a pointer or two and a small result is stored without allocating.
inline constexpr std::size_t invocable_inline_bytes = 48;
inline constexpr std::size_t invocable_inline_align = 16;

/// True when a target of type @p F is stored in place: it fits, and moving it cannot throw.
template <typename F>
inline constexpr bool invocable_stores_inline =
  sizeof(F) <= invocable_inline_bytes && alignof(F) <= invocable_inline_align &&
  std::is_nothrow_move_constructible_v<F>;

/// Type-erased move-only storage shared by every `invocable` flavour.
///
/// A target that fits (`invocable_stores_inline`) lives in the object (no allocation); anything
/// else is allocated once.  A call is one indirect call through a function pointer; a move is a
/// 48-byte copy for a trivially copyable target, else one indirect call.
template <bool NoExcept, typename R, typename... Args>
class invocable_storage {
 public:
  invocable_storage() noexcept = default;

  template <typename F>
  explicit invocable_storage(F&& f) noexcept(invocable_stores_inline<std::decay_t<F>> &&
                                             std::is_nothrow_constructible_v<std::decay_t<F>, F>)
  {
    using target_t = std::decay_t<F>;
    if constexpr (std::is_pointer_v<target_t> || std::is_member_pointer_v<target_t>) {
      if (f == nullptr) { return; }  // a null function pointer is no target
    }
    if constexpr (invocable_stores_inline<target_t>) {
      if constexpr (std::is_trivially_copyable_v<target_t> &&
                    sizeof(target_t) < invocable_inline_bytes) {
        // Relocated by a whole-buffer copy: give the bytes the target does not use a value.
        std::memset(_buf, 0, invocable_inline_bytes);
      }
      ::new (static_cast<void*>(_buf)) target_t(std::forward<F>(f));
      _invoke = &invoke_inline<target_t>;
      if constexpr (!std::is_trivially_copyable_v<target_t>) { _manage = &manage_inline<target_t>; }
    } else {
      auto* target = new target_t(std::forward<F>(f));
      ::new (static_cast<void*>(_buf)) target_t*(target);
      _invoke = &invoke_heap<target_t>;
      _manage = &manage_heap<target_t>;
    }
  }

  invocable_storage(invocable_storage&& other) noexcept { take(other); }

  invocable_storage& operator=(invocable_storage&& other) noexcept
  {
    if (this != &other) {
      reset();
      take(other);
    }
    return *this;
  }

  invocable_storage(invocable_storage const&)            = delete;
  invocable_storage& operator=(invocable_storage const&) = delete;

  ~invocable_storage() { reset(); }

  [[nodiscard]] bool has_target() const noexcept { return _invoke != nullptr; }

  void reset() noexcept
  {
    if (_manage != nullptr) { _manage(manage_op::destroy, _buf, nullptr); }
    _invoke = nullptr;
    _manage = nullptr;
  }

  R call(Args&&... args) noexcept(NoExcept)
  {
    return _invoke(static_cast<void*>(_buf), std::forward<Args>(args)...);
  }

 private:
  enum class manage_op : unsigned char { relocate, destroy };
  using invoke_fn = R (*)(void*, Args&&...) noexcept(NoExcept);
  /// Null for a trivially copyable in-place target (moved by copying the buffer).
  using manage_fn = void (*)(manage_op, void* src, void* dst) noexcept;

  void take(invocable_storage& other) noexcept
  {
    if (other._invoke == nullptr) { return; }
    if (other._manage != nullptr) {
      other._manage(manage_op::relocate, other._buf, _buf);
    } else {
      std::memcpy(_buf, other._buf, invocable_inline_bytes);
    }
    _invoke = std::exchange(other._invoke, nullptr);
    _manage = std::exchange(other._manage, nullptr);
  }

  template <typename T>
  static R invoke_target(T& target, Args&&... args) noexcept(NoExcept)
  {
    if constexpr (std::is_void_v<R>) {
      std::invoke(target, std::forward<Args>(args)...);
    } else {
      return std::invoke(target, std::forward<Args>(args)...);
    }
  }

  template <typename T>
  static R invoke_inline(void* buf, Args&&... args) noexcept(NoExcept)
  {
    return invoke_target(*std::launder(static_cast<T*>(buf)), std::forward<Args>(args)...);
  }

  template <typename T>
  static R invoke_heap(void* buf, Args&&... args) noexcept(NoExcept)
  {
    return invoke_target(**std::launder(static_cast<T**>(buf)), std::forward<Args>(args)...);
  }

  template <typename T>
  static void manage_inline(manage_op op, void* src, void* dst) noexcept
  {
    T* target = std::launder(static_cast<T*>(src));
    if (op == manage_op::relocate) { ::new (dst) T(std::move(*target)); }
    target->~T();
  }

  template <typename T>
  static void manage_heap(manage_op op, void* src, void* dst) noexcept
  {
    T** target = std::launder(static_cast<T**>(src));
    if (op == manage_op::relocate) {
      ::new (dst) T*(*target);
    } else {
      delete *target;
    }
  }

  alignas(invocable_inline_align) unsigned char _buf[invocable_inline_bytes];
  invoke_fn _invoke{nullptr};
  manage_fn _manage{nullptr};
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
