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

// Coalescing pass of the remote planner (ported from gpuio's src/planner/coalesce.cpp).
// Rules: gpuio ARCHITECTURE §7.1 producer pseudocode ("merged waste <= (amp - 1) x useful bytes;
// span <= max_coalesced_span"); s3_line_rate §2.6 ("Coalescing a gap g"). Greedy left to right,
// like Arrow's CoalesceReadRanges. The remote defaults (P10, P11) come from
// `remote_coalesce_params`: gap_min cold, amplification 1.05, span 64 MiB, the caller's gap wins.

#include "io/rest/planner/planner_common.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace cucascade::io::rest::planner {
namespace {

constexpr std::uint64_t u64_max = std::numeric_limits<std::uint64_t>::max();

bool by_offset(dst_entry const& a, dst_entry const& b) noexcept
{
  return a.logical.offset < b.logical.offset ||
         (a.logical.offset == b.logical.offset && a.logical.size < b.logical.size);
}

/// Non-empty and representable: offset + size does not wrap.
bool valid(file_range r) noexcept { return r.size != 0 && r.size <= u64_max - r.offset; }

}  // namespace

std::size_t coalesce(std::span<dst_entry> dsts,
                     coalesce_params params,
                     std::span<span_entry> out) noexcept
{
  std::size_t const n = dsts.size();
  if (n == 0 || n > std::numeric_limits<std::uint32_t>::max()) { return 0; }
  // Producers usually hand ranges over in offset order: check first (O(n)), sort only if needed.
  if (!std::is_sorted(dsts.begin(), dsts.end(), by_offset)) {
    std::sort(dsts.begin(), dsts.end(), by_offset);
  }

  // max_amplification < 1 (or NaN) admits no waste: only adjacent entries merge.
  double const slack = params.max_amplification >= 1.0 ? params.max_amplification - 1.0 : 0.0;
  std::uint64_t const max_span = params.max_span == 0 ? u64_max : params.max_span;

  if (!valid(dsts[0].logical)) { return 0; }
  std::size_t n_out    = 0;
  std::uint64_t begin  = dsts[0].logical.offset;
  std::uint64_t end    = dsts[0].logical.end();
  std::uint64_t useful = dsts[0].logical.size;
  std::size_t first    = 0;

  auto emit = [&](std::size_t last_exclusive) noexcept {
    if (n_out == out.size()) { return false; }
    out[n_out++] = span_entry{file_range{begin, end - begin},
                              static_cast<std::uint32_t>(first),
                              static_cast<std::uint32_t>(last_exclusive - first)};
    return true;
  };

  for (std::size_t i = 1; i < n; ++i) {
    file_range const d = dsts[i].logical;
    if (!valid(d)) { return 0; }
    if (d.offset < end) { return 0; }  // overlapping destinations (sorted: the only case)
    std::uint64_t const hole       = d.offset - end;
    std::uint64_t const new_end    = d.end();
    std::uint64_t const new_size   = new_end - begin;
    std::uint64_t const new_useful = useful + d.size;  // <= new_size: no overflow
    std::uint64_t const waste      = new_size - new_useful;
    bool const merge               = hole <= params.gap && new_size <= max_span &&
                       static_cast<double>(waste) <= slack * static_cast<double>(new_useful);
    if (merge) {
      end    = new_end;
      useful = new_useful;
      continue;
    }
    if (!emit(i)) { return 0; }
    begin  = d.offset;
    end    = new_end;
    useful = d.size;
    first  = i;
  }
  if (!emit(n)) { return 0; }
  return n_out;
}

}  // namespace cucascade::io::rest::planner
