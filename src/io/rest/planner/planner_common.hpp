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

#include "io/rest/common.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/**
 * @file planner_common.hpp
 * @brief Coalescing vocabulary of the remote planner: destination entries, coalesced spans and
 *        the coalescing pass (ported from gpuio's `planner_common.hpp`; design: gpuio
 *        ARCHITECTURE §7.1 remote gap rule, s3_line_rate §2.6 "Coalescing a gap g").
 *
 * gpuio's `dst_entry` also carried the destination of the range (host / device pointers,
 * registered-buffer slot, destination kind, multi-target index): that is gpuio's request block.
 * In cuCascade the destinations live in the `prepared_io_slice` / cache chunks the caller owns,
 * so an entry here carries only the logical range and the caller's index of it; `range_index`
 * travels through the sort and maps every coalesced entry back to the caller's array.
 */

namespace cucascade::io::rest::planner {

/// One logical range to read (sorted in place by `coalesce`).
struct dst_entry {
  file_range logical;  ///< clamped logical range (never empty)
  /// The caller's index of the range (its position before the sort); carried through the sort
  /// untouched. ~0 when the caller does not need it.
  std::uint32_t range_index{~0u};
};

/// A coalesced physical span and the destination entries it covers:
/// `dsts[first_dst, first_dst + n_dst)` of the (sorted) dst array, all inside `physical`.
struct span_entry {
  file_range physical;
  std::uint32_t first_dst{0}, n_dst{0};
};

/// Coalescing rules (gpuio ARCHITECTURE §7.1 producer pseudocode; s3_line_rate §2.6 "Rule").
struct coalesce_params {
  std::uint64_t gap;         ///< merge only holes of at most this many bytes
  std::uint64_t max_span;    ///< a merge never grows a span beyond this; 0 = no cap
  double max_amplification;  ///< span bytes <= this x useful bytes (values < 1 are read as 1)
};

/**
 * @brief Sort `dsts` by offset (in place) and merge them greedily into physical spans.
 *
 * Walking the sorted entries left to right, entry d joins the current span iff
 *   - the hole `d.offset - span.end` is <= `gap` (adjacent entries have a hole of 0),
 *   - the merged span is <= `max_span`, and
 *   - the merged waste (span bytes - useful bytes) is <= (max_amplification - 1) x the useful
 *     bytes of the merged span ("merged waste <= (amp - 1) x useful bytes", gpuio
 *     ARCHITECTURE §7.1).
 * Otherwise d starts a new span. A single entry larger than `max_span` is a span of its own (the
 * cutter splits it later). Hole bytes are discarded by the caller. Every entry keeps its
 * `range_index`, and each span names the contiguous run of entries it covers. No allocation
 * (`std::sort` is introsort, in place).
 *
 * @param dsts destination entries; sorted in place when they are not sorted already.
 * @param params gap / span / amplification rules (`remote_coalesce_params`).
 * @param out receives the spans in offset order; `dsts.size()` entries always suffice.
 * @return number of spans written; 0 when `dsts` is empty, when two entries overlap or an entry
 *         is empty or ends past 2^64 (the caller maps this to `errc::invalid_argument`), or when
 *         `out` is too small.
 */
[[nodiscard]] std::size_t coalesce(std::span<dst_entry> dsts,
                                   coalesce_params params,
                                   std::span<span_entry> out) noexcept;

}  // namespace cucascade::io::rest::planner
