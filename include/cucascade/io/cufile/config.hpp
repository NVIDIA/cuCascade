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

#include <array>
#include <limits>
#include <stdexcept>
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

/** @brief Opt-in transport mode; legacy remains the default until hardware qualification. */
enum class engine_mode : std::uint8_t { legacy, async_auto, async_simulated_for_testing };

/** @brief Which mount identity an operator override matches. */
enum class mount_match : std::uint8_t { fsid, mount_prefix, fs_type };

/** @brief An operator's routing decision, never evidence of verified direct I/O. */
enum class mount_override_state : std::uint8_t { operator_qualified, known_compat, legacy };

/** @brief GPU bounce pool sizes (KiB, ascending multiples of 4) and matching counts. */
struct gpu_bounce_slabs {
  std::vector<std::size_t> sizes_kib;
  std::vector<std::size_t> counts;
  bool operator==(gpu_bounce_slabs const&) const = default;
};

/** @brief Typed pre-driver-open process allowlist. Unset means no requested change. */
struct process_cufile_parameters {
  /// Set CUFILE_ENV_PATH_JSON on the control/bootstrap path, before other environment readers.
  std::optional<std::string> cufile_json_path;
  std::optional<bool> allow_compat_mode;
  std::optional<bool> force_compat_mode;
  std::optional<std::size_t> max_direct_io_size_kib;
  std::optional<std::size_t> max_device_pinned_mem_size_kib;
  std::optional<std::size_t> max_device_cache_size_kib;
  std::optional<std::vector<std::string>> rdma_dev_addr_list;
  std::optional<std::string> rdma_load_balancing_policy;
  std::optional<gpu_bounce_slabs> gpu_bounce_buffer_slabs;
  std::optional<std::size_t> stats_level;
  std::optional<std::string> logging_level;
  bool operator==(process_cufile_parameters const&) const = default;
  void validate() const
  {
    auto positive = [](auto const& v) { return !v || *v > 0; };
    if (!positive(max_direct_io_size_kib) || !positive(max_device_pinned_mem_size_kib) ||
        !positive(max_device_cache_size_kib))
      throw std::invalid_argument("cuFile size limits must be positive KiB");
    for (auto const* text : {&logging_level, &rdma_load_balancing_policy, &cufile_json_path})
      if (*text && ((*text)->empty() || (*text)->find('\0') != std::string::npos))
        throw std::invalid_argument("cuFile parameter string must be nonempty and contain no NUL");
    if (rdma_dev_addr_list) {
      if (rdma_dev_addr_list->empty()) throw std::invalid_argument("RDMA address list is empty");
      for (auto const& addr : *rdma_dev_addr_list)
        if (addr.empty() || addr.find('\0') != std::string::npos)
          throw std::invalid_argument("RDMA address is empty or contains NUL");
    }
    if (gpu_bounce_buffer_slabs) {
      auto const& v = *gpu_bounce_buffer_slabs;
      if (v.sizes_kib.empty() || v.sizes_kib.size() != v.counts.size() ||
          v.sizes_kib.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("GPU bounce slab arrays must have matching positive lengths");
      for (std::size_t i = 0; i < v.sizes_kib.size(); ++i)
        if (!v.sizes_kib[i] || v.sizes_kib[i] % 4 || !v.counts[i] ||
            (i && v.sizes_kib[i] <= v.sizes_kib[i - 1]))
          throw std::invalid_argument("GPU bounce slabs require ascending 4-KiB sizes and counts");
    }
  }
};

/**
 * @brief Mount routing and optional preferred profile, interpreted by the policy layer.
 *
 * The fsid is an opaque string, keeping the configuration independent of platform fsid types.
 * A mount prefix must already be a normalized absolute path: root is allowed, but trailing
 * slashes, repeated slashes, and @c . or @c .. components are not. Filesystem types must contain
 * no whitespace. Profile values are preferences: capacity fitting may reduce QD and choose a
 * smaller supported staging class than @c op_bytes.
 */
struct mount_override {
  mount_match match{mount_match::mount_prefix};
  std::string value;
  mount_override_state state{mount_override_state::legacy};
  std::optional<unsigned> qd;
  std::optional<std::size_t> op_bytes;
  bool allow_writes{false};  ///< Explicit write opt-in; qualification is reads-only by default.
  std::optional<unsigned> lanes{};   ///< Mount lane ceiling, still limited by device capacity.
  std::optional<std::string> nic{};  ///< RDMA device name or PCI BDF; placement only, not routing.
  std::vector<int> cpus{};           ///< Permitted submitter CPUs, intersected with process cpuset.
  bool operator==(mount_override const&) const = default;
};

/**
 * @brief Configuration for the opt-in per-device async engine.
 *
 * This is a configuration-only interface: construction and validation perform no probing,
 * CUDA work, or runtime initialization. Only non-legacy contexts bootstrap the engine.
 */
struct async_engine_config {
  engine_mode mode{engine_mode::legacy};
  bool async_reads{true};
  bool async_writes{false};
  bool unregistered_device_tier{true};
  bool direct_registered_io{false};
  bool stream_register{true};
  bool affinity{false};
  process_cufile_parameters process_params;
  bool strict_parameters{false};
  bool coalesce{true};  ///< Engine-only small-slice grouping; legacy routing is unchanged.
  unsigned coalesce_max_targets{64};

  /// Mounts are assigned round-robin to 1..8 submitters on each device.
  unsigned submitters_per_device{1};

  /// Per-device allocation target. Zero means auto, not disabled: round down to 16 MiB
  /// the minimum of 1 GiB, 5% of device memory, and reservable memory. Explicit nonzero
  /// targets are accepted as given; actual allocation/registration is a resource concern.
  std::size_t arena_bytes{0};
  /// Prewarmed MiB classes, unique members of {1,2,4,8}. Smaller pieces use larger slots.
  std::vector<unsigned> arena_slot_classes{4};
  unsigned device_lane_cap{160};
  std::size_t max_pending_slices_per_device{4096};
  unsigned low_service_interval{4};
  unsigned max_short_reads{64};

  /// Routing precedence: fsid, then longest matching mount_prefix, then fs_type. Prefixes
  /// match whole path components of normalized absolute paths; the first entry in list order
  /// wins any tie. Exact duplicate {match, value} selectors are invalid, regardless of state.
  std::vector<mount_override> mount_overrides;

  struct probe_config {
    unsigned submit_max_us{100};
    /// False permits non-blocking-submit-only qualification for simulator/compat tests.
    bool require_direct_evidence{true};
    std::size_t bytes{16UL << 20};
    bool operator==(probe_config const&) const = default;
  } probe;

  struct demotion_config {
    unsigned slow_floor_us{100};
    unsigned p99_multiple{4};
    unsigned slow_of_last_eight{3};
    unsigned stuck_ms{2};
    unsigned cooldown_s{30};
    unsigned cooldown_cap_s{960};
    bool operator==(demotion_config const&) const = default;
  } demotion;

  struct poll_config {
    /// Floors for latency-adaptive ceilings, bounded above by 1 ms (AM-15).
    unsigned io_cap_us{100};
    unsigned copy_cap_us{10};
    unsigned fence_hint_us{100};
    unsigned due_events_per_round{16};
    unsigned idle_health_ms{20};
    bool operator==(poll_config const&) const = default;
  } poll;

  bool operator==(async_engine_config const&) const = default;

  /**
   * @brief Check syntactic ranges without consulting device or filesystem capabilities.
   *
   * Flags are independently selectable, including in legacy mode. An override's QD need
   * not fit the lane cap: effective capacity is determined later by resource fitting.
   * Byte counts used for direct-I/O probes/profile preferences must be positive multiples
   * of @c io::IO_BLOCK_SIZE; device-specific granules are checked by the future policy layer.
   *
   * @throws std::invalid_argument naming the first invalid field.
   */
  void validate() const
  {
    auto const positive = [](auto value, char const* field) {
      if (value == 0) { throw std::invalid_argument(std::string{field} + " must be positive"); }
    };
    auto const aligned_bytes = [](std::size_t value, char const* field) {
      if (value == 0 || value % io::IO_BLOCK_SIZE != 0) {
        throw std::invalid_argument(std::string{field} +
                                    " must be a positive multiple of IO_BLOCK_SIZE");
      }
    };

    process_params.validate();
    switch (mode) {
      case engine_mode::legacy:
      case engine_mode::async_auto:
      case engine_mode::async_simulated_for_testing: break;
      default: throw std::invalid_argument("mode is invalid");
    }
    if (submitters_per_device < 1 || submitters_per_device > 8) {
      throw std::invalid_argument("submitters_per_device must be in [1, 8]");
    }
    if (arena_slot_classes.empty())
      throw std::invalid_argument("arena_slot_classes must be nonempty");
    unsigned class_mask = 0;
    for (auto cls : arena_slot_classes) {
      if ((cls != 1 && cls != 2 && cls != 4 && cls != 8) || (class_mask & cls))
        throw std::invalid_argument(
          "arena_slot_classes requires unique MiB classes from {1,2,4,8}");
      class_mask |= cls;
    }
    positive(device_lane_cap, "device_lane_cap");
    positive(max_pending_slices_per_device, "max_pending_slices_per_device");
    positive(low_service_interval, "low_service_interval");
    positive(max_short_reads, "max_short_reads");
    // coalesce is independently selectable, including in legacy mode.
    positive(coalesce_max_targets, "coalesce_max_targets");
    if (coalesce_max_targets > 4096)
      throw std::invalid_argument("coalesce_max_targets must be in [1, 4096]");
    if (device_lane_cap > 160 || max_pending_slices_per_device > (1UL << 20)) {
      throw std::invalid_argument("engine metadata/lane capacity exceeds supported limit");
    }
    positive(probe.submit_max_us, "probe.submit_max_us");
    aligned_bytes(probe.bytes, "probe.bytes");
    positive(demotion.slow_floor_us, "demotion.slow_floor_us");
    positive(demotion.p99_multiple, "demotion.p99_multiple");
    if (demotion.slow_of_last_eight < 1 || demotion.slow_of_last_eight > 8) {
      throw std::invalid_argument("demotion.slow_of_last_eight must be in [1, 8]");
    }
    positive(demotion.stuck_ms, "demotion.stuck_ms");
    positive(demotion.cooldown_s, "demotion.cooldown_s");
    if (demotion.cooldown_cap_s < demotion.cooldown_s) {
      throw std::invalid_argument("demotion.cooldown_cap_s must be >= demotion.cooldown_s");
    }
    positive(poll.io_cap_us, "poll.io_cap_us");
    if (poll.io_cap_us > 1000) throw std::invalid_argument("poll.io_cap_us must be <= 1000");
    positive(poll.copy_cap_us, "poll.copy_cap_us");
    if (poll.copy_cap_us > 1000) throw std::invalid_argument("poll.copy_cap_us must be <= 1000");
    positive(poll.fence_hint_us, "poll.fence_hint_us");
    if (poll.fence_hint_us > 1000)
      throw std::invalid_argument("poll.fence_hint_us must be <= 1000");
    positive(poll.due_events_per_round, "poll.due_events_per_round");
    positive(poll.idle_health_ms, "poll.idle_health_ms");

    for (std::size_t index = 0; index < mount_overrides.size(); ++index) {
      auto const& entry = mount_overrides[index];
      switch (entry.match) {
        case mount_match::fsid: break;
        case mount_match::fs_type:
          if (entry.value.find_first_of(" \t\n\r\f\v") != std::string::npos) {
            throw std::invalid_argument("mount_overrides.value fs_type must contain no whitespace");
          }
          break;
        case mount_match::mount_prefix:
          if (entry.value.empty() || entry.value.front() != '/' ||
              (entry.value.size() > 1 && entry.value.back() == '/') ||
              entry.value.find("//") != std::string::npos ||
              entry.value.find("/./") != std::string::npos ||
              entry.value.find("/../") != std::string::npos || entry.value.ends_with("/.") ||
              entry.value.ends_with("/..")) {
            throw std::invalid_argument(
              "mount_overrides.value must be a normalized absolute mount_prefix");
          }
          break;
        default: throw std::invalid_argument("mount_overrides.match is invalid");
      }
      if (entry.value.empty() || entry.value.find('\0') != std::string::npos) {
        throw std::invalid_argument("mount_overrides.value must be nonempty and contain no NUL");
      }
      switch (entry.state) {
        case mount_override_state::operator_qualified:
        case mount_override_state::known_compat:
        case mount_override_state::legacy: break;
        default: throw std::invalid_argument("mount_overrides.state is invalid");
      }
      if (entry.allow_writes && entry.state != mount_override_state::operator_qualified)
        throw std::invalid_argument("mount_overrides.allow_writes requires operator_qualified");
      if (entry.nic && (entry.nic->empty() || entry.nic->find('\0') != std::string::npos ||
                        entry.nic->find_first_of("/ \t\n\r\f\v") != std::string::npos))
        throw std::invalid_argument("mount_overrides.nic must be a device name or PCI BDF");
      for (auto cpu : entry.cpus)
        if (cpu < 0) throw std::invalid_argument("mount_overrides.cpus must be nonnegative");
      if (entry.lanes) positive(*entry.lanes, "mount_overrides.lanes");
      if (entry.qd) { positive(*entry.qd, "mount_overrides.qd"); }
      if (entry.op_bytes) { aligned_bytes(*entry.op_bytes, "mount_overrides.op_bytes"); }
      for (std::size_t earlier = 0; earlier < index; ++earlier) {
        if (mount_overrides[earlier].match == entry.match &&
            mount_overrides[earlier].value == entry.value) {
          throw std::invalid_argument(
            "mount_overrides contains duplicate {match, value} selectors");
        }
      }
    }
  }
};

/// Config/API ABI diagnostic compiled into the library.
[[nodiscard]] std::array<std::size_t, 6> cufile_config_library_abi_reference() noexcept;

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
  /// Opt-in engine; legacy retains the planned synchronous staging path.
  async_engine_config async_engine{};
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

  /// Write slices larger than this are cut by @c templated_ioctx into pieces of
  /// this size (rounded down to the write granule) and spread over the selected
  /// reactors; 0 = never cut.  A reactor serves a slice to completion, so an
  /// uncut multi-GiB write would hold it (and the reads queued behind it) for
  /// hundreds of milliseconds.  The default equals @c max_op_bytes, so no cuFile
  /// call gets smaller than an uncut slice's.
  std::size_t write_split_bytes{64UL << 20};

  /// Device reads below this size use POSIX pread (through the reactor's pinned
  /// scratch) instead of cuFile.  256 KiB
  /// sits above the <= 128 KiB small-I/O regression NVIDIA documents for real GDS
  /// on kernels >= 6.11, and in compatibility mode a 256 KiB @c cuFileRead from
  /// four threads already reached 4 GB/s here.
  std::size_t gds_threshold{256UL << 10};

  /// Whether the reactor opens the O_DIRECT fd needed to register with cuFile
  /// (and used for aligned host reads and aligned POSIX writes).  If false,
  /// cuFile registration is skipped and every read and write goes through POSIX
  /// and the page cache --
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

  /// Fuse neighbouring small device slices of one vectored read into a single
  /// staged @c cuFileRead: slices of at most half a piece whose gap is at most
  /// this many bytes, and which together fit one staging slot, are read by one
  /// aligned call and scattered to their destinations.  A read's cost is mostly
  /// its fixed latency, so many small reads (Parquet column chunks of a file
  /// with small row groups) cannot fill a fast link however deep the pool is;
  /// fusing trades the gap bytes for fewer, larger reads.  0 disables it.  Only
  /// staged reads are fused; a fused slice that cannot be staged is served as
  /// its separate slices.
  std::size_t fuse_gap_bytes{0};

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
