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

#include <rmm/error.hpp>

#include <cstring>
#include <string_view>
#include <system_error>

namespace cucascade {
namespace memory {

enum class MemoryError { SUCCESS, ALLOCATION_FAILED, LIMIT_EXCEEDED, POOL_EXHAUSTED, SIZE };

struct memory_error_category : std::error_category {
  const char* name() const noexcept final;

  std::string message(int ev) const final;
};

const memory_error_category& memory_category();

inline std::error_code make_error_code(MemoryError e);

struct cucascade_out_of_memory : public rmm::out_of_memory {
  explicit cucascade_out_of_memory(std::string_view message,
                                   std::size_t requested_bytes,
                                   std::size_t global_usage);

  const std::size_t requested_bytes;
  const std::size_t global_usage;
};

}  // namespace memory
}  // namespace cucascade

namespace std {

template <>
struct is_error_code_enum<cucascade::memory::MemoryError> : true_type {};

}  // namespace std
