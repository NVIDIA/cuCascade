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

/// @file deadline_heap.hpp
/// @brief Fixed-capacity indexed binary min-heap of deadlines for the reactor loop (API.md §B.4
/// `arm_deadline`). Deadlines bound the ring wait (`park_and_wait` timeout = min(head, copy ETA,
/// safety tick)); they replace per-op LINK_TIMEOUTs (io_uring_knobs §2.5.8: a linked timeout per
/// receive doubles task_work and triggers the DEFER_TASKRUN backlog of liburing#1645) and
/// uring_ultra's O(connections) deadline scan (S7).
///
/// Storage is allocated once (`init`); `arm`/`cancel`/`pop` never allocate. Handles carry a
/// generation so a stale handle (already fired or cancelled, node reused) is rejected.
/// Equal deadlines fire in arming order. Single-threaded (reactor thread only).

#pragma once

#include <cstdint>
#include <memory>
#include <new>

namespace cucascade::io::net {

/// Identifies one armed deadline. Default-constructed = invalid.
struct deadline_handle {
  std::uint32_t index{~0u};
  std::uint32_t generation{0};
  [[nodiscard]] constexpr bool valid() const noexcept { return index != ~0u; }
};

/// @tparam Fn callback pointer type stored per node (the reactor uses
/// `void (*)(reactor&, void*) noexcept`).
template <class Fn>
class deadline_heap {
 public:
  struct entry {
    std::uint64_t when{0};
    Fn fn{};
    void* arg{nullptr};
  };

  deadline_heap() noexcept                       = default;
  deadline_heap(deadline_heap const&)            = delete;
  deadline_heap& operator=(deadline_heap const&) = delete;
  ~deadline_heap()                               = default;

  /// @brief Allocate storage for @p capacity concurrent deadlines (the only allocation).
  /// @return false when out of memory.
  [[nodiscard]] bool init(std::uint32_t capacity) noexcept
  {
    _nodes.reset(new (std::nothrow) node[capacity]);
    _heap.reset(new (std::nothrow) std::uint32_t[capacity]);
    if (!_nodes || !_heap) {
      _nodes.reset();
      _heap.reset();
      _capacity = 0;
      return false;
    }
    _capacity = capacity;
    _size     = 0;
    _free     = ~0u;
    for (std::uint32_t i = capacity; i-- > 0;) {
      _nodes[i].next_free = _free;
      _free               = i;
    }
    return true;
  }

  /// @brief Arm a deadline at @p when (any clock; the reactor uses CLOCK_MONOTONIC ns).
  /// @return its handle, or an invalid handle when the heap is full.
  [[nodiscard]] deadline_handle arm(std::uint64_t when, Fn fn, void* arg) noexcept
  {
    if (_free == ~0u) return {};
    std::uint32_t const i = _free;
    node& n               = _nodes[i];
    _free                 = n.next_free;
    n.e                   = entry{when, fn, arg};
    n.seq                 = _seq++;
    n.pos                 = _size;
    _heap[_size++]        = i;
    sift_up(n.pos);
    return {i, n.generation};
  }

  /// @brief Disarm @p h. @return false when @p h already fired, was cancelled or is invalid.
  bool cancel(deadline_handle h) noexcept
  {
    if (!live(h)) return false;
    remove_at(_nodes[h.index].pos);
    return true;
  }

  /// @return true while @p h is armed.
  [[nodiscard]] bool live(deadline_handle h) const noexcept
  {
    return h.index < _capacity && _nodes[h.index].pos != npos &&
           _nodes[h.index].generation == h.generation;
  }

  /// @return the earliest deadline, or UINT64_MAX when empty.
  [[nodiscard]] std::uint64_t next() const noexcept
  {
    return _size == 0 ? ~std::uint64_t{0} : _nodes[_heap[0]].e.when;
  }

  /// @brief Remove the earliest entry if it is due (`when <= now`) and return it in @p out.
  /// Callbacks run by the caller after the pop may re-arm (the node is already free).
  [[nodiscard]] bool pop_due(std::uint64_t now, entry& out) noexcept
  {
    if (_size == 0 || _nodes[_heap[0]].e.when > now) return false;
    out = _nodes[_heap[0]].e;
    remove_at(0);
    return true;
  }

  [[nodiscard]] std::uint32_t size() const noexcept { return _size; }
  [[nodiscard]] std::uint32_t capacity() const noexcept { return _capacity; }
  [[nodiscard]] bool empty() const noexcept { return _size == 0; }

 private:
  static constexpr std::uint32_t npos = ~0u;
  struct node {
    entry e{};
    std::uint64_t seq{0};
    std::uint32_t pos{npos};  ///< position in _heap, npos when free
    std::uint32_t generation{0};
    std::uint32_t next_free{npos};
  };

  [[nodiscard]] bool less(std::uint32_t a, std::uint32_t b) const noexcept
  {
    node const& x = _nodes[a];
    node const& y = _nodes[b];
    return x.e.when < y.e.when || (x.e.when == y.e.when && x.seq < y.seq);
  }

  void place(std::uint32_t pos, std::uint32_t idx) noexcept
  {
    _heap[pos]      = idx;
    _nodes[idx].pos = pos;
  }

  void sift_up(std::uint32_t pos) noexcept
  {
    std::uint32_t const idx = _heap[pos];
    while (pos > 0) {
      std::uint32_t const parent = (pos - 1) / 2;
      if (!less(idx, _heap[parent])) break;
      place(pos, _heap[parent]);
      pos = parent;
    }
    place(pos, idx);
  }

  void sift_down(std::uint32_t pos) noexcept
  {
    std::uint32_t const idx = _heap[pos];
    for (;;) {
      std::uint32_t const l = 2 * pos + 1;
      if (l >= _size) break;
      std::uint32_t const r = l + 1;
      std::uint32_t const c = (r < _size && less(_heap[r], _heap[l])) ? r : l;
      if (!less(_heap[c], idx)) break;
      place(pos, _heap[c]);
      pos = c;
    }
    place(pos, idx);
  }

  void remove_at(std::uint32_t pos) noexcept
  {
    std::uint32_t const idx = _heap[pos];
    node& n                 = _nodes[idx];
    n.pos                   = npos;
    ++n.generation;
    n.next_free = _free;
    _free       = idx;
    --_size;
    if (pos == _size) return;
    std::uint32_t const last = _heap[_size];
    place(pos, last);
    if (pos > 0 && less(last, _heap[(pos - 1) / 2]))
      sift_up(pos);
    else
      sift_down(pos);
  }

  std::unique_ptr<node[]> _nodes;
  std::unique_ptr<std::uint32_t[]> _heap;
  std::uint32_t _capacity{0};
  std::uint32_t _size{0};
  std::uint32_t _free{npos};
  std::uint64_t _seq{0};
};

}  // namespace cucascade::io::net
