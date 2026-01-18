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

#include <cucascade/memory/error.hpp>

#include <system_error>

namespace cucascade {
namespace memory {

const char* memory_error_category::name() const noexcept { return "cuCascadeMemorySystem"; }

std::string memory_error_category::message(int ev) const
{
  switch (static_cast<MemoryError>(ev)) {
    case MemoryError::SUCCESS: return "Success";
    case MemoryError::ALLOCATION_FAILED: return "System allocation failed";
    case MemoryError::LIMIT_EXCEEDED: return "Reservation limit exceeded";
    case MemoryError::POOL_EXHAUSTED: return "Internal memory pool exhausted";
    default: return "Unknown memory error";
  }
}

const memory_error_category& memory_category()
{
  static const memory_error_category instance;
  return instance;
}

std::error_code make_error_code(MemoryError e)
{
  return std::error_code(static_cast<int>(e), memory_category());
}

cucascade_out_of_memory::cucascade_out_of_memory(std::string_view message,
                                                 std::size_t requested_bytes,
                                                 std::size_t global_usage)
  : rmm::out_of_memory(message.data()), requested_bytes(requested_bytes), global_usage(global_usage)
{
}

}  // namespace memory
}  // namespace cucascade
