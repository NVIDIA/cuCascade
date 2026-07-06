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

#include <cstdint>

namespace cucascade::io {

/**
 * @brief A half-open byte range [offset, offset + size) within a file/object.
 *
 * cudf-free equivalent of @c cudf::io::text::byte_range_info, so the io core
 * can express range-based APIs (fadvise ranges, alignment/coalescing) without
 * depending on cudf.  The cudf-coupled datasource layer converts to/from the
 * cudf type at its boundary.
 */
class byte_range {
 public:
  byte_range() = default;

  constexpr byte_range(std::int64_t offset, std::int64_t size) noexcept
    : _offset(offset), _size(size)
  {
  }

  [[nodiscard]] constexpr std::int64_t offset() const noexcept { return _offset; }
  [[nodiscard]] constexpr std::int64_t size() const noexcept { return _size; }
  [[nodiscard]] constexpr bool is_empty() const noexcept { return _size == 0; }

 private:
  std::int64_t _offset{0};
  std::int64_t _size{0};
};

}  // namespace cucascade::io
