/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <memory>
#include <utility>

namespace cucascade::io::cufile::detail {
/** Bounded sequence-number MPMC ring. Producers never wait for space or allocate. */
template <typename T>
class engine_ring {
 public:
  explicit engine_ring(std::size_t capacity)
    : _capacity(std::bit_ceil(std::max(std::size_t{2}, capacity))),
      _cells(std::make_unique<cell[]>(_capacity))
  {
    for (std::size_t i = 0; i < _capacity; ++i)
      _cells[i].sequence.store(i);
  }
  bool push(T&& value) noexcept
  {
    auto pos = _write.load(std::memory_order_relaxed);
    cell* c;
    for (;;) {
      c                = &_cells[pos & (_capacity - 1)];
      auto const seq   = c->sequence.load(std::memory_order_acquire);
      auto const delta = static_cast<std::ptrdiff_t>(seq - pos);
      if (delta == 0) {
        if (_write.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) break;
      } else if (delta < 0) {
        return false;
      } else {
        pos = _write.load(std::memory_order_relaxed);
      }
    }
    c->value = std::move(value);
    c->sequence.store(pos + 1, std::memory_order_release);
    return true;
  }
  bool pop(T& value) noexcept
  {
    auto pos = _read.load(std::memory_order_relaxed);
    cell* c;
    for (;;) {
      c                = &_cells[pos & (_capacity - 1)];
      auto const seq   = c->sequence.load(std::memory_order_acquire);
      auto const delta = static_cast<std::ptrdiff_t>(seq - (pos + 1));
      if (delta == 0) {
        if (_read.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) break;
      } else if (delta < 0) {
        return false;
      } else {
        pos = _read.load(std::memory_order_relaxed);
      }
    }
    value = std::move(c->value);
    c->sequence.store(pos + _capacity, std::memory_order_release);
    return true;
  }

 private:
  struct cell {
    std::atomic<std::size_t> sequence{};
    T value{};
  };
  std::size_t const _capacity;
  std::unique_ptr<cell[]> _cells;
  alignas(64) std::atomic<std::size_t> _write{};
  alignas(64) std::atomic<std::size_t> _read{};
};
}  // namespace cucascade::io::cufile::detail
