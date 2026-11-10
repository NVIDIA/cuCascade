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

#include <cucascade/memory/memory_reservation_manager.hpp>

#include <cstdint>

using cucascade::memory::memory_reservation_manager;
using cucascade::memory::memory_space;
using cucascade::memory::Tier;

int main()
{
  // Storing the address through a volatile forces the linker to resolve the symbol against the
  // installed library.
  memory_space* (memory_reservation_manager::* volatile get_memory_space)(Tier, std::int32_t) =
    &memory_reservation_manager::get_memory_space;
  return get_memory_space == nullptr ? 1 : 0;
}
