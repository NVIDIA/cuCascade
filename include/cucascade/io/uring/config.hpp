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

namespace cucascade::io::uring {

/// Largest accepted @c config::slices_per_pass: a reactor never owns more than
/// this many staging slots (each in-flight operation holds at least one), so a
/// larger cap could never take effect.
inline constexpr std::size_t max_slices_per_pass = 64;

struct config {
  /// How many scan tasks the readahead manager may keep in flight against this
  /// backend at once.  Zero disables readahead for it entirely.
  ///
  /// Local NVMe has no round trip to hide, so a readahead competes with the
  /// executor's own reads for the same device and just reorders the queue
  /// rather than adding throughput.  Measured on SF1000 local-parquet, turning
  /// it off is a large net win, so the local backend defaults to 0 (off).  Set
  /// a positive value (or `max_readahead_scans`) to opt the local path back in.
  std::size_t n_max_concurrent_scans{0};

  /// Whether the config named @c n_max_concurrent_scans explicitly. Preserved
  /// for parity with the REST backend (whose budget is still derived from the
  /// pipeline width) so an explicit value is always distinguishable from the
  /// struct default -- including an explicit 0, which opts the local path out
  /// as deliberately as the default does.
  bool n_max_concurrent_scans_explicit{false};

  /// When false, worker-planned operations use the buffered page-cache handle.
  /// Defaults to O_DIRECT when a physical operation satisfies its constraints.
  bool use_odirect{true};

  /// How many slices of its active request a reactor may turn into physical
  /// reads per loop pass before it waits for a completion; 0 means no cap (keep
  /// going while every planned read finds a free staging slot).  With 1, a
  /// single many-slice request (a whole-split prefetch, a wide demand read)
  /// runs at a queue depth of 1-2 per reactor.  The default of 8 keeps such a
  /// request deep without letting one pass claim every free slot ahead of the
  /// requests queued behind it.
  std::size_t slices_per_pass{8};

  /// Most slices one queue entry may hold when a host-only multi-range read is
  /// dispatched (see templated_ioctx::mixed_readv_async_io): a local read does
  /// not prefer bulk I/O, so a wide range read (a whole-split prefetch, a
  /// decoder's column-chunk set) is queued as several smaller requests rather
  /// than one that holds a reactor until its last slice.  0 disables the split.
  std::size_t range_batch_slices{8};

  /// How many reactors, taken from the end of the pool, serve only prefetch
  /// (readahead) reads; the others serve only demand reads.  A reactor runs
  /// its queue in order, so without the split a small demand read can wait
  /// behind gigabytes of whole-split prefetch on every reactor.  The reactor's
  /// two-tier queue (see @ref io_priority) now keeps demand ahead of prefetch
  /// without reserving a reactor, so this is normally 0.  0 disables the
  /// split: every read ranks among all reactors (see
  /// templated_ioctx::next_reactor).  Must be below the reactor count; with a
  /// single reactor it is ignored.  When not named in the config it stays 0;
  /// an embedding application may derive it (e.g. 1 when uring readahead runs).
  std::size_t prefetch_reactors{0};

  /// Whether the config named @c prefetch_reactors explicitly, so an explicit
  /// value (0 included) is never replaced by the derived default.
  bool prefetch_reactors_explicit{false};

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
