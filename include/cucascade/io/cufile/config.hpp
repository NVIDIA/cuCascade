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

#include <cucascade/exec/config.hpp>
#include <cucascade/io/types.hpp>

#include <cstddef>
#include <vector>

namespace cucascade::io::cufile {

/// Settings of the cuFile (GPUDirect Storage) backend; see @c cufile_reactor.
struct config {
  /// How many scan tasks the readahead manager may keep in flight against this
  /// backend at once.  Zero disables readahead for it entirely.  This backend
  /// never arms the prefetching cache, so a scan here only bounds how many
  /// pipeline threads may read through it concurrently.
  std::size_t n_max_concurrent_scans{
    static_cast<std::size_t>(exec::default_gpu_pipeline_num_threads)};

  /// Lanes in each of a reactor's two lane sets (one reads, one writes).  A lane
  /// is a thread blocked in one cuFile call at a time, so this is the read depth
  /// and the write depth of a GPU.
  std::size_t lanes{32};

  /// Bytes of a lane's slot of registered device memory, and so the most one
  /// cuFile call moves.  A positive multiple of the write granule.
  std::size_t lane_bytes{8UL << 20};

  /// Device reads of one request that lie at most this far apart are served by
  /// one cuFile read, the gap between them read and discarded.
  std::size_t coalesce_gap_bytes{1UL << 20};

  /// Most of such a read that may be gaps.
  double coalesce_gap_ratio{0.15};

  /// The GPUs to serve, one reactor each; empty serves every visible GPU.
  std::vector<int> devices;

  /// cuFile DMAs whole pages: a device read is widened to page boundaries, so
  /// reporting it lets the caller align once, up front.
  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept { return io::IO_BLOCK_SIZE; }

  /// Bridging a gap is worth it when the bridged bytes cost less than the extra
  /// request -- one page for a local NVMe syscall.
  [[nodiscard]] std::size_t merge_gap_size() const noexcept { return io::IO_BLOCK_SIZE; }

  /// Registered device memory one reactor (one GPU) holds: a slot per lane.
  [[nodiscard]] std::size_t arena_bytes() const noexcept { return 2 * lanes * lane_bytes; }
};

}  // namespace cucascade::io::cufile
