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

#include <cstddef>
#include <cstdint>
#include <limits>

/**
 * @file common.hpp
 * @brief Vocabulary shared by the modules of the io_uring REST transport (ported from gpuio's
 *        `types.hpp`).
 */

namespace cucascade::io::rest {

inline constexpr std::size_t KiB = std::size_t{1} << 10;
inline constexpr std::size_t MiB = std::size_t{1} << 20;
inline constexpr std::size_t GiB = std::size_t{1} << 30;

/**
 * @brief Half-open byte interval [offset, offset + size) within an object.
 *
 * gpuio's `byte_range`, renamed so it does not shadow cuCascade's public `io::byte_range` (a
 * signed offset/size class used by the cache planner) inside this namespace.
 */
struct file_range {
  std::uint64_t offset{0};
  std::uint64_t size{0};

  /// @return offset + size, saturating at the maximum representable value.
  [[nodiscard]] constexpr std::uint64_t end() const noexcept
  {
    constexpr auto max = std::numeric_limits<std::uint64_t>::max();
    return size > max - offset ? max : offset + size;
  }
  [[nodiscard]] constexpr bool empty() const noexcept { return size == 0; }

  friend constexpr bool operator==(file_range const&, file_range const&) noexcept = default;
};

/// @return the overlap of @p a and @p b; an empty range at max(a.offset, b.offset) when they do
/// not overlap.  Ends saturate (see @ref file_range::end).
[[nodiscard]] constexpr file_range intersect(file_range a, file_range b) noexcept
{
  auto const begin = a.offset > b.offset ? a.offset : b.offset;
  auto const end   = a.end() < b.end() ? a.end() : b.end();
  return end > begin ? file_range{begin, end - begin} : file_range{begin, 0};
}

/**
 * @brief Scheduling class inside the transport, ordered: a lower value is served first.
 *
 * cuCascade's `io_priority` maps onto it at the reactor boundary: `high` (demand reads, writes,
 * control-plane requests) → @c high, `low` (fs_cache prefetch) → @c bulk.  @c urgent is the
 * latency class that alone draws on the reserved connections (blocking control-plane calls:
 * HEAD, footer probe, LIST, sync reads); @c normal is unused by cuCascade's dispatch but kept so
 * the ported fairness tables keep their shape.
 */
enum class priority : std::uint8_t { urgent, high, normal, bulk };

}  // namespace cucascade::io::rest
