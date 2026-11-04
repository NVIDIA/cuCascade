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

#include "utils/allocation_counter.hpp"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace {

// Constant-initialised, so reading them needs no TLS initialisation from inside operator new.
thread_local bool counting       = false;
thread_local std::size_t counted = 0;

}  // namespace

// Replacements for the global allocation functions (the unaligned forms; the array and
// nothrow forms forward to these).  Same malloc/free behaviour as the defaults, plus the count.
void* operator new(std::size_t size)
{
  if (counting) { ++counted; }
  if (void* p = std::malloc(size == 0 ? 1 : size)) { return p; }
  throw std::bad_alloc{};
}

void operator delete(void* p) noexcept { std::free(p); }

void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace cucascade::test {

scoped_allocation_counter::scoped_allocation_counter() noexcept
{
  assert(!counting && "scopes do not nest");
  counted  = 0;
  counting = true;
}

scoped_allocation_counter::~scoped_allocation_counter() noexcept { counting = false; }

std::size_t scoped_allocation_counter::count() const noexcept { return counted; }

}  // namespace cucascade::test
