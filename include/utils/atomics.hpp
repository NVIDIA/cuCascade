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

#include <atomic>
#include <concepts>
#include <cstdint>
#include <utility>

namespace cucascade {
namespace utils {

template <std::integral T>
struct atomic_peak_tracker {
  explicit atomic_peak_tracker(T initial_peak = 0) noexcept : _peak(initial_peak) {}

  void reset(T new_peak = 0) noexcept { _peak.store(new_peak); }

  [[nodiscard]] T peak() const noexcept { return _peak.load(); }

  void update_peak(T current_value)
  {
    auto peak_value = _peak.load();
    while (current_value > peak_value && !_peak.compare_exchange_weak(peak_value, current_value)) {}
  }

 private:
  std::atomic<T> _peak{0};
};

template <std::integral T>
struct atomic_bounded_counter {
  explicit atomic_bounded_counter(T initial = 0) : _value(initial) {}

  std::atomic<T>& native_handle() noexcept { return _value; }

  std::atomic<T> const& native_handle() const noexcept { return _value; }

  std::atomic<T>& operator*() noexcept { return native_handle(); }

  std::atomic<T> const& operator*() const noexcept { return native_handle(); }

  std::atomic<T>* operator->() noexcept { return &_value; }

  std::atomic<T> const* operator->() const noexcept { return &_value; }

  [[nodiscard]] T value(std::memory_order order = std::memory_order_seq_cst) const noexcept
  {
    return _value.load(order);
  }

  [[nodiscard]] T load(std::memory_order order = std::memory_order_seq_cst) const noexcept
  {
    return _value.load(order);
  }

  /// \@brief return updated value
  T add(T diff, std::memory_order order = std::memory_order_seq_cst) noexcept
  {
    return _value.fetch_add(diff, order) + diff;
  }

  [[nodiscard]] std::pair<bool, T> try_add(T diff, T upper_bound)
  {
    auto current = _value.load();
    while (true) {
      if (upper_bound < current || (upper_bound - current) < diff) { return {false, current}; }
      if (_value.compare_exchange_weak(current, current + diff)) { return {true, current + diff}; }
    }
  }

  [[nodiscard]] T add_bounded(T& diff, T upper_bound)
  {
    auto current = _value.load();
    while (current < upper_bound) {
      T space_left = upper_bound - current;
      diff         = std::min(diff, space_left);
      if (_value.compare_exchange_weak(current, current + diff)) { return current + diff; }
    }
    diff = 0;
    return current;
  }

  T sub(T diff, std::memory_order order = std::memory_order_seq_cst) noexcept
  {
    return _value.fetch_sub(diff, order) - diff;
  }

  [[nodiscard]] std::pair<bool, T> try_sub(T diff, T lower_bound)
  {
    auto current = _value.load();
    while (true) {
      if (current < lower_bound || (current - lower_bound) < diff) { return {false, current}; }
      if (_value.compare_exchange_weak(current, current - diff)) { return {true, current - diff}; }
    }
  }

  [[nodiscard]] T sub_bounded(T& diff, T lower_bound)
  {
    auto current = _value.load();
    while (current > lower_bound) {
      T space_above = current - lower_bound;
      diff          = std::min(diff, space_above);

      if (_value.compare_exchange_weak(current, current - diff)) { return current - diff; }
    }
    diff = 0;
    return current;
  }

 private:
  std::atomic<T> _value{0};
};

}  // namespace utils
}  // namespace cucascade
