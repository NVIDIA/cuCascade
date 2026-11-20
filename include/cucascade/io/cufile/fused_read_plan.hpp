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

#include <cucascade/io/types.hpp>

#include <cstddef>
#include <vector>

namespace cucascade::io::cufile {

/**
 * @brief How @ref plan_fused_reads packs device slices into lane reads.
 */
struct fused_plan_options {
  /// Largest aligned window one lane read covers: the lane slot size.  A non-zero multiple
  /// of @c io::IO_BLOCK_SIZE.
  std::size_t window_bytes{0};
  /// Largest gap between two neighbouring slices still served by one read (the gap is read
  /// and dropped).  0 fuses only contiguous slices.  Smaller than @ref window_bytes.
  std::size_t gap_bytes{0};
  /// Fuse neighbouring slices into one read.  When false a slice is only split at window
  /// boundaries, and every slice costs at least one read.
  bool fuse{true};
  /// Windows whose aligned size is below this are not lane reads: their parts are emitted as
  /// ordinary slices.  0 = every window is a lane read.
  std::size_t min_read_bytes{0};
};

/**
 * @brief What one @ref plan_fused_reads call produced.
 *
 * Only lane reads count in @ref reads, @ref parts, @ref payload_bytes and @ref issued_bytes;
 * EOF tails and the parts of windows below @c min_read_bytes do not.
 */
struct fused_plan_stats {
  std::size_t reads{0};          ///< Lane reads produced.
  std::size_t parts{0};          ///< Caller pieces they serve.
  std::size_t split_slices{0};   ///< Input slices emitted as more than one piece.
  std::size_t tail_slices{0};    ///< Sub-block EOF tails peeled off as ordinary slices.
  std::size_t payload_bytes{0};  ///< Bytes the caller asked for, in lane reads.
  std::size_t issued_bytes{0};   ///< Bytes the lane reads cover (their aligned windows).

  bool operator==(fused_plan_stats const&) const = default;
};

/**
 * @brief Rewrite the lane-eligible device slices of one request, in place, as lane reads.
 *
 * An eligible slice is a read with a device destination only (no host buffer), no
 * @c on_complete, @c d_buffer.ready_gate_set, not already fused, non-empty and starting
 * inside the file.  Eligible slices are grouped by (device id, stream handle, gate pointer,
 * priority) and, inside a group, packed in file order into windows
 * @c [align_down(offset), ...) of at most @c window_bytes that end at or before
 * @c align_down(file_size): neighbours whose gap is at most @c gap_bytes share a window,
 * overlapping ones never do, and a slice longer than a window is split (its last piece may
 * share a window with what follows).  Slices are clamped to @p file_size.
 *
 * Each window is emitted as one slice:
 * - exactly one whole, unsplit input slice: that slice, moved as is;
 * - one piece of a split slice: a plain slice of just that piece;
 * - several parts: a fused slice (@c rng from the first part's offset to the last part's
 *   end, @c fused = the parts, @c d_buffer copied from the first part's slice with @c data
 *   = the first part's destination, the same priority, kind read);
 * - an aligned window below @c min_read_bytes: one ordinary slice per part instead.
 * The bytes at or past @c align_down(file_size) (the file's partial last block) become an
 * ordinary plain slice per input slice (an input slice wholly inside it is moved as is).
 *
 * Every emitted lane read satisfies @ref is_lane_read with @c window_bytes, and the emitted
 * payload equals the eligible input payload clamped to the file.
 *
 * Emit order: first every slice that is not eligible, in its original relative order; then
 * the planned slices group by group (groups ordered by their key), in file order inside a
 * group (a slice that overlaps an earlier one starts a new window where it begins, so
 * windows of overlapping inputs may interleave).  When no slice is eligible @p slices is not
 * touched.
 *
 * Strong guarantee: if anything throws, @p slices is unchanged.
 *
 * @param slices The prepared slices of one request.
 * @param file_size The file's size in bytes.
 * @param options Window size, gap, fusion switch and minimum read.
 * @return What was produced.
 * @throws std::invalid_argument when @c options.window_bytes is zero or not a multiple of
 *         @c io::IO_BLOCK_SIZE, or @c options.gap_bytes is not smaller than it.
 * @throws std::bad_alloc when the plan cannot be built (@p slices unchanged).
 */
fused_plan_stats plan_fused_reads(std::vector<prepared_io_slice>& slices,
                                  std::size_t file_size,
                                  fused_plan_options const& options);

/**
 * @brief Whether @p slice is exactly one lane read for a slot of @p window_bytes.
 *
 * True for a non-empty read with a device destination only and no @c on_complete whose
 * @ref lane_window is at most @p window_bytes and ends at or before
 * @c align_down(file_size); a fused slice's parts must also be non-empty, sorted,
 * non-overlapping, inside @c rng and have a destination.  Cheap: the reactor calls it per
 * slice.
 */
[[nodiscard]] bool is_lane_read(prepared_io_slice const& slice,
                                std::size_t file_size,
                                std::size_t window_bytes) noexcept;

/// The 4 KiB-aligned window a lane read issues: @c [align_down(offset), align_up(end)) of
/// @p slice's @c rng (the end saturates at the largest aligned @c std::size_t).
[[nodiscard]] range lane_window(prepared_io_slice const& slice) noexcept;

}  // namespace cucascade::io::cufile
