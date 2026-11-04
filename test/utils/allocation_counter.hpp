/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstddef>

namespace cucascade::test {

/**
 * @brief Counts the calling thread's global `operator new` calls while in scope.
 *
 * A test binary that links `utils/allocation_counter.cpp` replaces the global allocation
 * functions with malloc/free pass-throughs that also count calls made by a thread holding a
 * live scope, so a test can assert that a code path does not allocate, or make one of its
 * allocations fail (fail_at()).  Other threads are not counted.  Scopes do not nest.
 *
 * Only the unaligned scalar `operator new(std::size_t)` is replaced and counted.  The aligned
 * forms (`std::align_val_t`) are not counted.  The array and nothrow forms are counted only
 * because libstdc++'s default implementations forward to the replaced scalar form; that does
 * not hold where another allocator supplies all the forms itself (ASan, valgrind, tcmalloc).
 */
class scoped_allocation_counter {
 public:
  scoped_allocation_counter() noexcept;
  ~scoped_allocation_counter() noexcept;

  scoped_allocation_counter(scoped_allocation_counter const&)            = delete;
  scoped_allocation_counter& operator=(scoped_allocation_counter const&) = delete;

  /// Allocations this thread made since the scope began.
  [[nodiscard]] std::size_t count() const noexcept;

  /**
   * @brief Make this thread's @p n-th allocation in the scope throw `std::bad_alloc`.
   *
   * A nothrow form returns null instead.  Counted from the start of the scope, as count() is,
   * and the failed call counts too, so `fail_at(count() + 1)` fails the next allocation.  Only
   * that one call fails; later ones succeed.  0 disarms it, and the fault ends with the scope.
   * A test can sweep @p n from 1 until a call makes fewer than @p n allocations, to fail each
   * allocation of a path in turn.
   */
  void fail_at(std::size_t n) noexcept;
};

}  // namespace cucascade::test
