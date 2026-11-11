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

#include <cucascade/io/object_identity.hpp>

#include <atomic>
#include <cstdint>

namespace cucascade::io {

std::uint64_t next_open_generation() noexcept
{
  // Defined here, not inline in the header: every module linking cucascade_io
  // must share this one counter, whatever symbol visibility it is built with.
  static std::atomic<std::uint64_t> generation{0};
  return generation.fetch_add(1, std::memory_order_relaxed) + 1;
}

}  // namespace cucascade::io
