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
#include <cstdint>
#include <optional>

namespace cucascade::io::cufile {

/**
 * @brief Per-file policy for using cuFile.
 */
enum class compat : std::uint8_t {
  automatic,  ///< try cuFile; a file that cannot be registered falls back to POSIX I/O
  on,         ///< never use cuFile: every transfer of the file goes through POSIX I/O
  off,        ///< enforce cuFile: opening a file that cannot be registered throws
};

/**
 * @brief Tunables for the cuFile / GPUDirect Storage local-file reactor.
 *
 * This backend registers each file with @c cuFileHandleRegister and serves
 * device destinations with synchronous @c cuFileRead calls issued from the
 * reactor's worker thread.  A
 * device read starts once the work already queued on the caller's CUDA stream
 * has finished and completes host-observed; with a real GDS stack cuFile DMAs
 * from NVMe straight into GPU memory with no host-side bounce of its own.
 *
 * Host-only reads go O_DIRECT (through the reactor's pinned scratch when not
 * block-aligned); mixed host+device requests use POSIX @c pread on the buffered
 * fd.
 *
 * @note Host-source writes whose aligned body is below 4 MiB or does not line up
 *       with its source address (e.g. the DISK tier's pinned pool blocks) are
 *       written through the page cache with writeback started at once
 *       (@c sync_file_range), which measured faster than O_DIRECT from a
 *       synchronous reactor.  Spilled host data can therefore transiently occupy
 *       the page cache; the pages are clean, hence reclaimable, once written
 *       back, and durability still needs a flush (@c fdatasync).  Larger
 *       congruent bodies go O_DIRECT from the caller's buffer.
 *
 * Writes of device sources use @c cuFileWrite for their aligned body only when
 * cuFile reports a real GDS mode; on a compat-only platform every device write
 * is staged through pinned host memory and @c pwrite, which measured faster
 * there (see @c cufile_reactor).  No setting changes that.
 *
 * @note libcufile is loaded at run time (@c cufile_api); a host without it runs
 *       every transfer through POSIX I/O.  The backend is built when @c cufile.h is
 *       found at configure time (@c CUCASCADE_HAS_CUFILE).
 */
struct config {
  /// How many scan tasks the readahead manager may keep in flight against this
  /// backend at once.  Zero disables readahead for it entirely.  Local NVMe /
  /// GDS saturates at modest queue depth, so one scan per pipeline thread is
  /// enough to keep the decoders fed without over-committing the device.
  std::size_t n_max_concurrent_scans{
    static_cast<std::size_t>(exec::default_gpu_pipeline_num_threads)};

  /// Whether the config named @c n_max_concurrent_scans explicitly.  Same role as
  /// in the uring config: distinguish an explicit value equal to the struct
  /// default from a defaulted one, so a derived default (which follows the
  /// pipeline width) can override the latter but not the former.
  bool n_max_concurrent_scans_explicit{false};

  /// Maximum asynchronous items per reactor: device slices parked until their
  /// stream is ready plus fallback host-to-device copies in flight.  cuFile and
  /// staged transfers are synchronous on the worker and never count; the reactor
  /// stops dispatching new slices once this many items await their CUDA event,
  /// so slices parked behind busy streams can fill the window; parked slices that
  /// became ready are served (in about @ref max_op_bytes portions) before new ones.
  std::size_t max_inflight{32};

  /// Largest single @c cuFileRead the reactor issues; a larger device slice is
  /// read in pieces of at most this size whose boundaries fall on 4 KiB file
  /// offsets.  One call this large lets cuFile split it internally across its
  /// own threads, which is what saturates a local NVMe from one reactor.  Must be
  /// at least @c io::IO_BLOCK_SIZE.
  ///
  /// @note 64 MiB was measured in cuFile compatibility mode (no nvidia-fs; one
  ///       thread reached 6.83 GB/s at 64 MiB vs 6.00 at 16 MiB against a
  ///       7.1 GB/s device; docs/io/gpuio-gds-practices.md).  Re-sweep it on a
  ///       host with a real GDS stack, where cuFile chunks by its own
  ///       @c max_direct_io_size (16 MiB) anyway.
  std::size_t max_op_bytes{64UL << 20};

  /// Register device buffers with @c cuFileBufRegister for the duration of one
  /// transfer when the transfer is at least this many bytes and the buffer (and,
  /// for a read, the file offset) is 4 KiB-aligned; 0 (the default) never
  /// registers.  The registration is dropped before the transfer's future
  /// resolves.
  ///
  /// Registrations are shared through a process-wide table keyed by base
  /// address: a transfer whose buffer starts at a live registration's base and
  /// fits inside it reuses that registration (the last user deregisters); one
  /// whose buffer otherwise overlaps a live registration takes the staged POSIX
  /// path, and a range an unregistered cuFile transfer is using is not registered
  /// until it completes, so no cuFile call ever addresses memory a concurrent
  /// transfer may deregister.
  ///
  /// @note Off by default on purpose: NVIDIA advises against registering a
  ///       one-shot destination (the cost is not amortised and BAR1 space is
  ///       shared), and in compatibility mode it measured no gain here
  ///       (docs/io/gpuio-gds-practices.md, G8).  It exists to A/B on a host with
  ///       a real GDS stack, where an aligned registered buffer is DMA'd directly
  ///       instead of through cuFile's GPU bounce buffers.
  std::size_t register_min_bytes{0};

  /// Write slices larger than this are cut by @c templated_ioctx into pieces of
  /// this size (rounded down to the write granule) and spread over the selected
  /// reactors; 0 = never cut.  cuFile prefers large submissions (one call splits
  /// internally across cuFile's own threads), so the default is no split.
  std::size_t write_split_bytes{0};

  /// Device reads below this size use POSIX pread (through the reactor's pinned
  /// scratch) instead of cuFile.  256 KiB follows gpuio's @c gds_min_bytes: it
  /// sits above the <= 128 KiB small-I/O regression NVIDIA documents for real GDS
  /// on kernels >= 6.11, and in compatibility mode a 256 KiB @c cuFileRead from
  /// four threads already reached 4 GB/s here (docs/io/gpuio-gds-practices.md).
  std::size_t gds_threshold{256UL << 10};

  /// Whether the reactor opens the O_DIRECT fd needed to register with cuFile
  /// (and used for aligned POSIX writes).  If false, cuFile registration is
  /// skipped and every read and write goes through POSIX and the page cache --
  /// effectively disabling the reactor's fast path.  Kept as a knob so a
  /// filesystem that rejects O_DIRECT can still exercise the reactor end-to-end
  /// via the fallback.
  bool use_odirect{true};

  /// Per-file cuFile policy (see @ref compat).  Unset means @c compat::automatic.
  /// Independently, @c CUFILE_FORCE_COMPAT_MODE=true in the environment makes
  /// cuFile itself run in compatibility mode (see @c probe_platform()).
  std::optional<compat> compat_mode;

  /// How long the worker sleeps between completion polls while fallback copies
  /// are in flight and it has nothing to submit.  A completion is only noticed at
  /// a poll, so this bounds the latency a finished copy adds; it is not a wait the
  /// reactor takes when it has work to dispatch.
  std::size_t poll_interval_us{200};

  /// Bounded service for the low-priority tier.  Priority is otherwise strict (a
  /// waiting high request is taken first and parks an active low one), so a
  /// demand read that waits on in-flight low work could starve behind a
  /// continuous stream of high requests.  After this many high requests have
  /// started while low work was waiting, the reactor grants the low tier one
  /// turn (at least @c max_inflight slices) even though high work is waiting.
  /// 0 restores strict priority.
  std::size_t low_service_interval{4};

  /// cuFile can only DMA whole pages, so a read is widened to a page boundary
  /// either way -- reporting it lets the caller align once, up front, instead of
  /// every layer rediscovering it.
  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept { return io::IO_BLOCK_SIZE; }

  /// A local read is a syscall against NVMe, so bridging is only worth it when
  /// the bridged bytes cost less than the extra request -- one page.
  [[nodiscard]] std::size_t merge_gap_size() const noexcept { return io::IO_BLOCK_SIZE; }
};

}  // namespace cucascade::io::cufile
