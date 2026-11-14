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
#include <string>
#include <vector>

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
 * @brief How device reads of a cuFile-registered file are staged.
 *
 * cuFile can DMA straight into GPU memory only when the device address, the file
 * offset and the length are 4 KiB aligned and the memory is registered
 * (@c cuFileBufRegister).  Parquet column chunks land at arbitrary offsets in
 * unregistered RMM buffers, so a direct read is bounced by cuFile through its own
 * small GPU buffers (@c per_buffer_cache_size_kb pieces, 1 MiB by default, out of
 * a @c max_device_cache_size_kb pool, 128 MiB per GPU by default).  A staged read
 * instead widens the range to 4 KiB boundaries, reads it with one call into a
 * registered, aligned slot the reactor owns, and scatters the wanted bytes into
 * the destination with a device-to-device copy.
 */
enum class staging : std::uint8_t {
  /// Stage when cuFile reports a real GDS mode (@ref gds_available()) and a
  /// registered slot can be had; read directly otherwise.  In compatibility mode
  /// cuFile bounces through host memory anyway, so staging would only add a copy.
  automatic,
  /// Stage whenever a slot can be had, registered or not (tests, diagnostics).
  always,
  /// Never stage: cuFile reads straight into the destination.
  never,
};

/**
 * @brief Tunables for the cuFile / GPUDirect Storage local-file reactor.
 *
 * This backend registers each file with @c cuFileHandleRegister and serves
 * device destinations with synchronous @c cuFileRead calls issued from the
 * reactor's worker threads, one read in flight per reactor: the queue depth a
 * GPU sees is the number of reactors serving it, so a pool is planned per GPU
 * (see @c plan_cufile_pool) and sized by Little's law -- bytes in flight =
 * throughput x latency (50 GB/s at 2-5 ms over RDMA needs 100-250 MB in flight
 * per GPU; local NVMe far less).  Device reads of a GDS-capable file are staged
 * through registered slots (@ref staging); the read itself is ordered after the
 * work queued on the caller's stream before the call only where it writes the
 * caller's memory (the scatter copy, or a direct read).  Reads complete
 * host-observed.
 *
 * Host-only reads go O_DIRECT (through the reactor's pinned scratch when not
 * block-aligned); mixed host+device requests use POSIX @c pread on the buffered
 * fd.  The backend is read-only.
 *
 * @note libcufile is loaded at run time (@c cufile_api); a host without it runs
 *       every transfer through POSIX I/O.  The backend is built when @c cufile.h is
 *       found at configure time (@c CUCASCADE_HAS_CUFILE).
 */
struct config {
  /// How many scan tasks the readahead manager may keep in flight against this
  /// backend at once.  Zero disables readahead for it entirely.  This backend
  /// never arms the prefetching cache, so a scan here only bounds how many
  /// pipeline threads may read through it concurrently.
  std::size_t n_max_concurrent_scans{
    static_cast<std::size_t>(exec::default_gpu_pipeline_num_threads)};

  /// Whether the config named @c n_max_concurrent_scans explicitly.  Same role as
  /// in the uring config: distinguish an explicit value equal to the struct
  /// default from a defaulted one, so a derived default (which follows the
  /// pipeline width) can override the latter but not the former.
  bool n_max_concurrent_scans_explicit{false};

  /// Maximum asynchronous items per reactor that hold copy resources: staged
  /// scatter copies still waiting for their caller's stream plus host-to-device
  /// copies of host+device reads.  Slices parked behind a busy stream hold no
  /// resource and do not count: they never stop the reactor from serving other
  /// requests.
  std::size_t max_inflight{32};

  /// Largest single direct (unstaged) @c cuFileRead the reactor issues; a larger
  /// device slice is read in pieces of at most this size whose boundaries fall
  /// on 4 KiB file offsets.  Also the byte quantum of one scheduling pass.  Must
  /// be at least @c io::IO_BLOCK_SIZE.
  ///
  /// @note 64 MiB was measured in cuFile compatibility mode (no nvidia-fs; one
  ///       thread reached 6.83 GB/s at 64 MiB vs 6.00 at 16 MiB against a
  ///       7.1 GB/s device).  With a real GDS stack an unregistered read is
  ///       bounced in @c per_buffer_cache_size_kb pieces whatever this is.
  std::size_t max_op_bytes{64UL << 20};

  /// Device reads below this size use POSIX pread (through the reactor's pinned
  /// scratch) instead of cuFile.  256 KiB
  /// sits above the <= 128 KiB small-I/O regression NVIDIA documents for real GDS
  /// on kernels >= 6.11, and in compatibility mode a 256 KiB @c cuFileRead from
  /// four threads already reached 4 GB/s here.
  std::size_t gds_threshold{256UL << 10};

  /// Whether the reactor opens the O_DIRECT fd needed to register with cuFile
  /// (and used for aligned host reads).  If false, cuFile registration is
  /// skipped and every read goes through POSIX and the page cache --
  /// effectively disabling the reactor's fast path.  Kept as a knob so a
  /// filesystem that rejects O_DIRECT can still exercise the reactor end-to-end
  /// via the fallback.
  bool use_odirect{true};

  /// Per-file cuFile policy (see @ref compat).  Unset means @c compat::automatic.
  /// Independently, @c CUFILE_FORCE_COMPAT_MODE=true in the environment makes
  /// cuFile itself run in compatibility mode (see @c probe_platform()).
  std::optional<compat> compat_mode;

  // -- staging (registered device slots) --------------------------------------

  /// How device reads of a registered file are staged (see @ref staging).
  staging staging_mode{staging::automatic};

  /// File bytes one staged @c cuFileRead covers at most.  A staging slot holds
  /// this plus the two partial blocks widening can add (rounded up to 64 KiB),
  /// so a piece of this size at any offset is one aligned call.  The cuFile ioctx
  /// also splits larger device reads into pieces of this size so that several
  /// reactors read them at once.  Must be a multiple of @c io::IO_BLOCK_SIZE.
  /// 8 MiB covers the largest Parquet column chunks in one call and stays below
  /// cuFile's default @c max_direct_io_size_kb (16 MiB).
  std::size_t staging_piece_bytes{8UL << 20};

  /// Registered slots a reactor keeps per device it stages for.  With two, a read
  /// whose caller's stream is still busy can complete into one slot (its scatter
  /// copy waits on the GPU for that stream) while the other stays free for reads
  /// that are ready; with one, such a read waits for its stream before it starts.
  std::size_t staging_slots_per_reactor{2};

  /// Process-wide cap on registered staging per device, in bytes; 0 = no cap
  /// beyond the device limits.  A pool built by @c make_cufile_ioctx (or the
  /// registry) also stays within half the GPU's free BAR1 and cuFile's
  /// @c max_device_pinned_mem_size less its bounce cache; a registration that
  /// fails anyway turns staging off on that device.  A reactor without a slot
  /// reads directly.
  /// Staging memory is allocated outside RMM and kept for the life of the
  /// process (freeing it would synchronize the device): account for it when
  /// sizing a GPU memory pool, or reserve it up front (@c reserve_staging).
  std::size_t max_staging_bytes_per_device{1UL << 30};

  // -- pool placement (used when the pool is planned per device) --------------

  /// Reactors serving each GPU when the pool is planned per device
  /// (@c io_config::cufile_n_reactors == 0).  0 = automatic: 32 when
  /// @ref storage_hint_path is on a network filesystem (GPFS, NFS, Lustre, ...),
  /// 8 when it is on local block storage, 16 otherwise.  Each reactor keeps one
  /// read in flight, so this is the GPU's queue depth.
  std::size_t reactors_per_device{0};

  /// Pin each reactor's worker to the CPUs local to its GPU (NVML's CPU affinity
  /// of the GPU, intersected with the process's affinity), so the I/O for a GPU
  /// is issued from its own socket, next to its NIC on an RDMA filesystem.
  bool pin_to_device_cpus{true};

  /// CUDA devices a planned pool serves, a group of reactors (and reserved
  /// staging) each; empty = every visible device.  List the GPUs the process
  /// reads into, so no reactors or staging are spent on others; a read into a
  /// device without a group is served by the whole pool.
  std::vector<int> devices{};

  /// A file on the storage the pool will mostly read, used only to size it
  /// (network vs local; see @ref reactors_per_device).  Empty = unknown.
  std::string storage_hint_path{};

  /// How long the worker sleeps between completion polls while copies are in
  /// flight and it has nothing to submit.  A completion is only noticed at
  /// a poll, so this bounds the latency a finished copy adds; it is not a wait the
  /// reactor takes when it has work to dispatch.
  std::size_t poll_interval_us{50};

  /// Bounded service for the low-priority tier.  Priority is otherwise strict (a
  /// waiting high request is taken first and parks an active low one), so a
  /// demand read that waits on in-flight low work could starve behind a
  /// continuous stream of high requests.  After this many high requests have
  /// started while low work was waiting, the reactor grants the low tier one
  /// turn (at least @c max_inflight slices) even though high work is waiting.
  /// 0 restores strict priority.
  std::size_t low_service_interval{4};

  /// cuFile DMAs whole pages: a direct read is widened to page boundaries, and a
  /// staged one too, so reporting it lets the caller align once, up front.
  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept { return io::IO_BLOCK_SIZE; }

  /// Bridging a gap is worth it when the bridged bytes cost less than the extra
  /// request -- one page for a local NVMe syscall.
  [[nodiscard]] std::size_t merge_gap_size() const noexcept { return io::IO_BLOCK_SIZE; }

  /// Bytes of one registered staging slot: @ref staging_piece_bytes plus the two
  /// partial blocks widening can add, rounded up to 64 KiB (the GPU page size
  /// BAR1 mappings use).
  [[nodiscard]] std::size_t staging_slot_bytes() const noexcept
  {
    constexpr std::size_t gpu_page = 64UL << 10;
    auto const raw                 = staging_piece_bytes + 2 * io::IO_BLOCK_SIZE;
    return (raw + gpu_page - 1) / gpu_page * gpu_page;
  }
};

}  // namespace cucascade::io::cufile
