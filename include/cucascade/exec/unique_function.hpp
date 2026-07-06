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
 */
template <typename Signature>
class unique_function;

template <typename R, typename... Args>
class unique_function<R(Args...)> {
 public:
  unique_function() noexcept = default;
  unique_function(std::nullptr_t) noexcept {}  // NOLINT(google-explicit-constructor)

  template <typename F>
    requires(!std::same_as<std::remove_cvref_t<F>, unique_function> &&
             std::invocable<std::decay_t<F>&, Args...>)
  unique_function(F&& f)  // NOLINT(google-explicit-constructor)
    : _impl(std::make_unique<model<std::decay_t<F>>>(std::forward<F>(f)))
  {
  }

  unique_function(unique_function&&) noexcept            = default;
  unique_function& operator=(unique_function&&) noexcept = default;
  unique_function(const unique_function&)                = delete;
  unique_function& operator=(const unique_function&)     = delete;
  ~unique_function()                                     = default;

  unique_function& operator=(std::nullptr_t) noexcept
  {
    _impl.reset();
    return *this;
  }

  [[nodiscard]] explicit operator bool() const noexcept { return _impl != nullptr; }

  friend bool operator==(const unique_function& f, std::nullptr_t) noexcept { return !f; }

  R operator()(Args... args) { return _impl->invoke(std::forward<Args>(args)...); }

 private:
  struct concept_t {
    virtual ~concept_t()             = default;
    virtual R invoke(Args&&... args) = 0;
  };

  template <typename F>
  struct model final : concept_t {
    explicit model(F f) : fn(std::move(f)) {}
    R invoke(Args&&... args) override { return std::invoke(fn, std::forward<Args>(args)...); }
    F fn;
  };

  std::unique_ptr<concept_t> _impl;
};

}  // namespace cucascade::exec
