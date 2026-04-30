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

#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cstddef>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace cucascade {
namespace data {

/**
 * @brief A single timing measurement for a (pair, transfer-size) probe.
 */
struct bandwidth_sample {
  double gbps                   = 0.0;  ///< Achieved throughput in GB/s (10^9 bytes)
  double mean_seconds           = 0.0;  ///< Mean wall-clock time per iteration
  std::size_t bytes_transferred = 0;    ///< Effective bytes moved per iteration
  std::size_t iterations_timed  = 0;    ///< Number of timed iterations aggregated
};

/**
 * @brief Bandwidth measurement for one ordered (src -> dst) pair of memory spaces.
 *
 * Results are asymmetric — `src -> dst` and `dst -> src` are separate entries.
 * Disk pairs are only included when exactly one endpoint is DISK (no disk-to-disk).
 */
struct bandwidth_pair_result {
  /// Source memory space. Overwritten before use; default values exist only because
  /// `memory_space_id` has no default constructor.
  memory::memory_space_id src{memory::Tier::GPU, 0};
  /// Destination memory space. See `src` note.
  memory::memory_space_id dst{memory::Tier::GPU, 0};

  /// Chunk size advertised by the source's allocator (0 if contiguous / not a chunked resource).
  std::size_t src_max_chunk_bytes = 0;
  /// Chunk size advertised by the destination's allocator (0 if contiguous / not a chunked
  /// resource).
  std::size_t dst_max_chunk_bytes = 0;

  /// Per-test-size detail. Keyed by requested transfer size in bytes.
  std::map<std::size_t, bandwidth_sample> per_size;
  /// Aggregated summary across per_size entries (median gbps).
  bandwidth_sample summary;

  /// False when no converter was available for this pair, or a transient error suppressed it.
  bool converter_available = true;
  /// Human-readable detail when `converter_available == false`.
  std::string unavailable_reason;
};

/**
 * @brief Configuration for `measure_bandwidth`.
 */
struct bandwidth_profile_config {
  /// Transfer sizes to probe per pair. Defaults to a sweep from 1 MiB up to 256 MiB.
  std::vector<std::size_t> test_sizes_bytes{
    1ull << 20,    //   1 MiB
    16ull << 20,   //  16 MiB
    64ull << 20,   //  64 MiB
    256ull << 20,  // 256 MiB
  };
  /// Untimed converter invocations run before measurement to warm caches, JIT, file cache, etc.
  std::size_t warmup_iterations = 3;
  /// Timed converter invocations aggregated into the per-size sample.
  std::size_t timed_iterations = 10;
  /// Skip disk pairs (useful when you only care about GPU/HOST cells).
  bool measure_disk_pairs = true;
  /// When true, call `posix_fadvise(POSIX_FADV_DONTNEED)` on disk source files between timed
  /// iterations so every read hits the disk instead of the Linux page cache. Process-local;
  /// no sudo required. Disable to measure warm-cache behavior.
  bool drop_page_cache_between_iters = true;
};

/**
 * @brief Asymmetric bandwidth profile across a set of memory spaces.
 */
struct bandwidth_profile {
  std::vector<bandwidth_pair_result> pairs;

  /**
   * @brief Summary gbps for the given ordered pair, or 0 if not found / unavailable.
   */
  [[nodiscard]] double gbps(memory::memory_space_id src,
                            memory::memory_space_id dst) const noexcept;

  /**
   * @brief Per-size sample for the given ordered pair and transfer size.
   *
   * @return The sample if the pair exists and size was measured; `std::nullopt` otherwise.
   */
  [[nodiscard]] std::optional<bandwidth_sample> sample(memory::memory_space_id src,
                                                       memory::memory_space_id dst,
                                                       std::size_t size_bytes) const;

  /**
   * @brief Find the result for a pair, if any.
   */
  [[nodiscard]] const bandwidth_pair_result* find(memory::memory_space_id src,
                                                  memory::memory_space_id dst) const noexcept;
};

/**
 * @brief Measure pairwise transfer throughput between a set of memory spaces.
 *
 * Results are asymmetric — both `src -> dst` and `dst -> src` are measured.
 *
 * Pair rules:
 *   - GPU and HOST spaces are measured against each other in both directions, including
 *     self-to-self across distinct device ids.
 *   - DISK spaces are measured against each GPU and HOST space only (no disk-to-disk).
 *   - Pairs that reduce to the same memory space id are skipped.
 *
 * Transfers flow through the converter registry, using the built-in canonical
 * representation for each tier (GPU -> gpu_table_representation,
 * HOST -> host_data_representation, DISK -> disk_data_representation). Pairs without a
 * registered converter are marked `converter_available == false`.
 *
 * This function is pure and synchronous. It does not reserve through
 * `memory_reservation_manager`; callers are expected to run it at init time when the
 * target memory spaces are otherwise idle.
 *
 * Each DISK `memory_space` owns its own I/O backend, so the profile reflects whichever
 * backend each disk space was constructed with — no separate backend parameter is needed.
 * At least one GPU space must be present: it is used to materialize the canonical source
 * cudf table that seeds every pairwise transfer.
 *
 * @param spaces   The memory spaces to profile.
 * @param registry Converter registry to dispatch through. Pass a registry populated by
 *                 `register_builtin_converters` plus any user-supplied converters.
 * @param config   Measurement knobs; defaults probe a size sweep.
 *
 * @return `bandwidth_profile` containing one `bandwidth_pair_result` per ordered pair considered.
 *
 * @throws std::invalid_argument if no GPU space is present in `spaces`.
 */
[[nodiscard]] bandwidth_profile measure_bandwidth(std::span<memory::memory_space* const> spaces,
                                                  const representation_converter_registry& registry,
                                                  const bandwidth_profile_config& config = {});

}  // namespace data
}  // namespace cucascade
