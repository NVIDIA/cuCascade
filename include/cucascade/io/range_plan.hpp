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

#include <cucascade/io/io_utils.hpp>
#include <cucascade/io/types.hpp>

#include <sys/types.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>

namespace cucascade::io::range_plan {

/**
 * @brief Widen a logical range to whole granules, clamping to EOF rounded up.
 * @note A zero granule leaves the range unchanged except for EOF clamping.
 *       This legacy arithmetic helper saturates at SIZE_MAX, where the physical end
 *       may be unaligned; use next_cut to reject unrepresentable file offsets.
 */
[[nodiscard]] inline range staged_physical_range(range logical,
                                                 std::size_t granule,
                                                 std::size_t file_size) noexcept
{
  auto const start = io_utils::align_down(logical.offset, granule);
  auto const end =
    std::min(io_utils::align_up(logical.end(), granule), io_utils::align_up(file_size, granule));
  return end > start ? range{start, end - start} : range{start, 0};
}

/**
 * @brief Split a write range at granule boundaries, preserving the legacy split.
 * @return {head, body, tail}, concatenating to exactly rng. Empty pieces have size zero.
 *         A range contained in one granule is all head.
 */
[[nodiscard]] inline std::array<range, 3> split_write_range(range rng, std::size_t granule) noexcept
{
  auto const begin = rng.offset;
  auto const end   = rng.end();
  if (granule == 0 || rng.empty()) return {rng, range{end, 0}, range{end, 0}};
  auto const first = begin % granule == 0 ? begin : begin - begin % granule + granule;
  auto const last  = end - end % granule;
  if (first >= end || last <= first) { return {rng, range{end, 0}, range{end, 0}}; }
  return {range{begin, first - begin}, range{first, last - first}, range{last, end - last}};
}

/**
 * @brief One logical credit and its granule-widened physical I/O window.
 *
 * EOF clamping alone may make physical end before logical, or leave physical empty
 * even when logical is not. Read callers must handle this as a short/EOF read and
 * must not submit empty physical windows. Geometry cuts need not be granule-aligned:
 * neighbouring pieces can share a granule, which write callers must serialize.
 */
struct piece_plan {
  range logical;
  range physical;
  bool last{false};
};

/**
 * @brief Cut on the absolute op-size grid, a geometry boundary, or the slice end.
 * @param geometry_boundaries Sorted absolute file offsets; boundaries at/before the cursor
 *                            are ignored.
 * @param file_size EOF used to clamp the physical window (independent of the slice end).
 * @note An exhausted cursor returns empty logical and physical ranges with last=true.
 *       The op size must be a granule multiple, so every physical window fits one op slot.
 * @throws std::invalid_argument For a zero size, non-multiple op size, reversed interval,
 *         or slice end beyond the maximum representable off_t.
 */
[[nodiscard]] inline piece_plan next_cut(std::size_t cursor_offset,
                                         std::size_t slice_end,
                                         std::size_t op_bytes,
                                         std::size_t granule,
                                         std::span<std::size_t const> geometry_boundaries,
                                         std::size_t file_size,
                                         std::size_t geometry_bytes = 0)
{
  if (op_bytes == 0 || granule == 0 || op_bytes % granule != 0 || cursor_offset > slice_end ||
      slice_end > static_cast<std::size_t>(std::numeric_limits<off_t>::max())) {
    throw std::invalid_argument("range_plan::next_cut: invalid interval or op/granule size");
  }
  assert(std::is_sorted(geometry_boundaries.begin(), geometry_boundaries.end()));
  if (cursor_offset == slice_end) {
    return {range{cursor_offset, 0}, range{io_utils::align_down(cursor_offset, granule), 0}, true};
  }
  auto end = std::min(slice_end,
                      io_utils::saturating_add(cursor_offset, op_bytes - cursor_offset % op_bytes));
  auto const boundary =
    std::upper_bound(geometry_boundaries.begin(), geometry_boundaries.end(), cursor_offset);
  if (boundary != geometry_boundaries.end()) { end = std::min(end, *boundary); }
  if (geometry_bytes)
    end = std::min(
      end,
      io_utils::saturating_add(cursor_offset, geometry_bytes - cursor_offset % geometry_bytes));
  range const logical{cursor_offset, end - cursor_offset};
  return {logical, staged_physical_range(logical, granule, file_size), end == slice_end};
}

/// Convenience form when the slice ends at EOF; use the six-argument form for an interior slice.
[[nodiscard]] inline piece_plan next_cut(std::size_t cursor_offset,
                                         std::size_t slice_end,
                                         std::size_t op_bytes,
                                         std::size_t granule,
                                         std::span<std::size_t const> geometry_boundaries = {})
{
  return next_cut(cursor_offset, slice_end, op_bytes, granule, geometry_boundaries, slice_end);
}

}  // namespace cucascade::io::range_plan
