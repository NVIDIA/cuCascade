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
#include <cucascade/io/details/scheduling_policy.hpp>
#include <cucascade/io/types.hpp>

#include <cstddef>

namespace cucascade::io::uring {

/// Largest accepted @ref config::slices_per_pass: an engine never owns more than
/// this many staging slots (every in-flight operation holds at least one), so a
/// larger cap could never take effect.
inline constexpr std::size_t max_slices_per_pass = 64;

struct config {
  /// How many scan tasks the readahead manager may keep in flight against this
  /// backend at once.  Zero disables readahead for it entirely.
  ///
  /// Local NVMe saturates at modest queue depth and every in-flight scan pins
  /// staging buffers, so one scan per pipeline executor thread is enough to
  /// keep the decoders fed without over-committing the pinned pool.
  std::size_t n_max_concurrent_scans{
    static_cast<std::size_t>(exec::default_gpu_pipeline_num_threads)};

  /// Whether the config named @c n_max_concurrent_scans explicitly. Needed
  /// because the derived default follows the configured pipeline width and can
  /// legitimately equal the struct default. Without this provenance, an
  /// explicit value equal to the struct default is silently overwritten.
  bool n_max_concurrent_scans_explicit{false};

  /// When false, worker-planned operations use the buffered page-cache handle.
  /// Defaults to O_DIRECT when a physical operation satisfies its constraints.
  bool use_odirect{true};

  /// How many slices (or write segments) of one grouped request a runner may
  /// turn into physical operations per loop pass before it moves on to the
  /// next group it holds; 0 means no cap.  A runner loops again at once while
  /// anything progressed, so this does not bound the queue depth of a request
  /// (it fills the free staging slots either way); it bounds how much of one
  /// pass a single request may claim, so the groups a runner holds share
  /// freed slots instead of being served strictly first come, first served.
  /// Valid: 0..@ref max_slices_per_pass; the uring reactor rejects other values.
  std::size_t slices_per_pass{8};

  /// Per-runner scheduling tunables (group limits, class shares, reservations);
  /// validated by the uring reactor.  Prefetch reads are background class
  /// (fs_cache::prefetch), so @c scheduling.max_background_groups,
  /// @c background_slot_fraction and @c reserved_background_slots are the
  /// prefetch-isolation knobs -- the analogue of sirius's dedicated prefetch
  /// reactors (K of N reactors ~ background_slot_fraction K/N).
  io::detail::scheduling_config scheduling{};

  /// O_DIRECT transfers whole pages, so a read is widened to a page boundary
  /// either way -- naming it lets the caller align once, up front, instead of
  /// every layer rediscovering it.  Reported even when @ref use_odirect is
  /// false: a buffered read of a page-aligned span costs no more than an
  /// unaligned one, and keeping the value constant keeps the two modes
  /// comparable.
  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept { return io::IO_BLOCK_SIZE; }

  /// A local read is a syscall against NVMe, so bridging is only worth it when
  /// the bridged bytes are cheaper than the extra request -- one page.
  [[nodiscard]] std::size_t merge_gap_size() const noexcept { return io::IO_BLOCK_SIZE; }
};

}  // namespace cucascade::io::uring
