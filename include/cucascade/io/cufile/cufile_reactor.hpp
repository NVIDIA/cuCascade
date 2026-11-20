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

#include <cucascade/cuda/event.hpp>
#include <cucascade/io/concurrent_queue.hpp>
#include <cucascade/io/cufile/config.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/object_identity.hpp>
#include <cucascade/io/templated_ioctx.hpp>
#include <cucascade/io/tiered_queue.hpp>
#include <cucascade/io/types.hpp>

#include <cuda_runtime.h>

#include <cufile.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace cucascade::io::cufile {
struct mount_identity;
struct file_async_policy;
struct mount_policy_key;
class mount_policy;
class device_engine;
class file_write_state;
class device_arena;
class cufile_transport;



// ---------------------------------------------------------------------------
// cufile_io_object
// ---------------------------------------------------------------------------

/**
 * @brief @c io_object backed by a local filesystem path and a cuFile handle.
 *
 * Owns two @c file_descriptor s -- a buffered fd for POSIX I/O and, when
 * O_DIRECT is enabled and supported, an O_DIRECT fd registered with cuFile via
 * @c cuFileHandleRegister for GDS DMA.  Both are @c O_RDONLY for an object opened
 * for reading and @c O_RDWR for one opened for writing (@ref writable).  The
 * registered handle lives for the object's lifetime; the destructor deregisters
 * it before the fds close.  All I/O is performed by the reactor -- this class is
 * a passive bag of native handles plus the file size, which writes through a
 * writable object raise (@ref note_written).
 *
 * A read-only object opened by the reactor carries the @c local_stat identity
 * of its buffered fd and takes its size from it; a writable object, or one
 * constructed without such an identity, has the base class's open-local one.
 */
class cufile_io_object final : public io_object {
 public:
  /// @p cufile_handle is null when the file was not registered with cuFile; the
  /// reactor then serves every read and write of it through POSIX.  @p identity
  /// is the @c local_stat identity of @p buffered_fd; it is ignored for a
  /// writable object or when it is not a @c local_stat identity.
  cufile_io_object(std::string path,
                   file_descriptor buffered_fd,
                   file_descriptor direct_fd,
                   CUfileHandle_t cufile_handle,
                   std::size_t file_size,
                   io_access access         = io_access::read_only,
                   object_identity identity = {});

  ~cufile_io_object() override;

  cufile_io_object(cufile_io_object const&)            = delete;
  cufile_io_object& operator=(cufile_io_object const&) = delete;

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override
  {
    return _file_size.load(std::memory_order_acquire);
  }

  [[nodiscard]] object_identity identity() const override;

  [[nodiscard]] bool writable() const noexcept override { return _access == io_access::read_write; }

  void note_written(std::size_t end) const noexcept override
  {
    if (writable()) atomic_store_max(_file_size, end);
  }

  /// The buffered fd (page cache).  Always valid.
  [[nodiscard]] int buffered_fd() const noexcept { return _buffered_fd.get(); }

  /// The O_DIRECT fd cuFile was registered against, or -1 when none was opened.
  /// Reads through it are cuFile's; the reactor @c pwrite s aligned write
  /// pieces through it.
  [[nodiscard]] int direct_fd() const noexcept { return _direct_fd.get(); }

  /// The handle to pass to @c cuFileRead; null when the GDS fast path is
  /// unavailable for this file.
  [[nodiscard]] CUfileHandle_t cufile_handle() const noexcept { return _cufile_handle; }

  [[nodiscard]] std::shared_ptr<file_write_state> write_state() const;

  [[nodiscard]] bool cufile_registered() const noexcept { return _cufile_handle != nullptr; }

  /// Resolve descriptor-derived mount metadata once, only when engine routing requests it.
  /// Legacy opens never inspect mountinfo. This does not initialize CUDA or cuFile.
  [[nodiscard]] mount_identity const& mount_identity_info() const;
  /// Caller-side snapshot only; never performs discovery or waits on call_once.
  [[nodiscard]] mount_identity const* cached_mount_identity() const noexcept
  {
    return _mount_identity_ready.load(std::memory_order_acquire) ? _mount_identity.get() : nullptr;
  }
  [[nodiscard]] std::shared_ptr<file_async_policy> async_policy() const;
  [[nodiscard]] mount_policy_key policy_key(std::string device_uuid,
                                            std::uint64_t generation) const;

 private:
  std::string _path;
  file_descriptor _buffered_fd;
  file_descriptor _direct_fd;
  CUfileHandle_t _cufile_handle{nullptr};
  mutable std::atomic<std::size_t> _file_size{0};
  io_access _access{io_access::read_only};
  std::optional<object_identity> _identity;
  mutable std::once_flag _write_state_once;
  mutable std::shared_ptr<file_write_state> _write_state;
  mutable std::once_flag _mount_identity_once;
  mutable std::atomic<bool> _mount_identity_ready{false};
  mutable std::unique_ptr<mount_identity> _mount_identity;
  mutable std::shared_ptr<file_async_policy> _async_policy;

};

// ---------------------------------------------------------------------------
// cufile_reactor
// ---------------------------------------------------------------------------

/**
 * @brief Where a cuFile reactor runs: the GPU it mainly serves, the CPUs its
 *        worker is pinned to, and the NUMA node its host scratch comes from.
 */
struct reactor_placement {
  int device{-1};         ///< Home GPU (CUDA ordinal); -1 = serves any device.
  std::vector<int> cpus;  ///< CPUs the worker is pinned to; empty = not pinned.
  int numa_node{-1};      ///< The home GPU's NUMA node; -1 = unknown.
  /// May serve device reads through fused stream lanes; honoured only when
  /// @c config::fused_reads is enabled.  A flat pool leaves it on for every reactor; a
  /// planned pool sets it on its lane reactors only.
  bool fused_lanes{true};
  /// May serve host reads through cuFile's batch API; honoured only when
  /// @c config::host_batch is enabled.  A flat pool leaves it on for every reactor; a
  /// planned pool sets it on its batch reactors only.
  bool host_batch{true};
};

/**
 * @brief How a cuFile reactor (or a pool, summed) served its reads and the
 *        cuFile parts of its writes.
 *
 * Counted when a transfer settles successfully, except @c cufile_failures and
 * @c parked_slices.  A test or a deployment check can tell from them which path
 * served the bytes: a GDS-capable file whose device reads all land in
 * @c posix_device_bytes is not using cuFile at all.
 */
struct reactor_stats {
  std::uint64_t staged_reads{0};  ///< Device reads cuFile served into a staging slot (+ scatter).
  std::uint64_t staged_bytes{0};  ///< Their bytes.
  /// Of @c staged_reads, those read before the caller's stream had finished the
  /// work queued ahead of them (only their scatter copy waited for it).
  std::uint64_t early_staged_reads{0};
  /// Caller slices served through fused staged reads (several per read).
  std::uint64_t fused_slices{0};
  std::uint64_t direct_reads{0};        ///< Device reads cuFile served into the destination.
  std::uint64_t direct_bytes{0};        ///< Their bytes.
  std::uint64_t posix_device_reads{0};  ///< Device reads served by POSIX + host-to-device copy.
  std::uint64_t posix_device_bytes{0};  ///< Their bytes.
  std::uint64_t host_reads{0};          ///< Host-only reads.
  std::uint64_t host_bytes{0};          ///< Their bytes.
  std::uint64_t cufile_failures{0};     ///< cuFile errors that sent a transfer to POSIX.
  std::uint64_t parked_slices{0};       ///< Device slices that waited for their stream to start.
  /// Device-write bodies cuFile wrote from a staging slot (after a device copy).
  std::uint64_t staged_writes{0};
  std::uint64_t staged_write_bytes{0};  ///< Their bytes.
  /// Device-write bodies cuFile wrote from the caller's source directly.
  std::uint64_t direct_writes{0};
  std::uint64_t direct_write_bytes{0};  ///< Their bytes.
  /// Fused-lane reads (@c cuFileReadAsync into a lane slot + device copies) that succeeded.
  std::uint64_t lane_reads{0};
  std::uint64_t lane_bytes{0};         ///< Their caller payload bytes.
  std::uint64_t lane_issued_bytes{0};  ///< Bytes their aligned windows covered.
  std::uint64_t lane_parts{0};         ///< Caller pieces they served (one per unfused slice).
  std::uint64_t lane_resubmits{0};     ///< Short-read continuations they needed.
  /// Lane reads that failed and were re-served through the synchronous paths (also counted
  /// in @c cufile_failures).
  std::uint64_t lane_failures{0};
  /// Of @c lane_reads, those whose copies waited on the GPU for the caller's stream.
  std::uint64_t lane_gated_reads{0};
  std::uint64_t batch_reads{0};     ///< Host reads served by cuFile batch entries.
  std::uint64_t batch_bytes{0};     ///< Their bytes.
  std::uint64_t batch_failures{0};  ///< Batch entries that failed or came back short.
  std::uint64_t batch_waves{0};     ///< cuFileBatchIOSubmit calls (waves) of the batch readers.
  std::uint64_t batch_entries{0};   ///< Entries those waves carried (wave fill = entries / waves).
  std::uint64_t batch_rejected_submits{0};  ///< Waves cuFile rejected.
  /// Wave GetStatus calls that returned less than the wave (each poisons its reader).
  std::uint64_t batch_partial_status{0};
  std::uint64_t batch_handle_resets{0};  ///< Batch handles recreated after a rejection.
  std::uint64_t batch_deferred{0};       ///< Flushes that left entries staged for a later wave.
  std::uint64_t batch_submit_ns{0};      ///< Time the batch helpers spent in Submit.

  /// Every counter, so the arithmetic below cannot miss one.
  [[nodiscard]] static constexpr auto fields() noexcept
  {
    return std::array{&reactor_stats::staged_reads,       &reactor_stats::staged_bytes,
                      &reactor_stats::early_staged_reads, &reactor_stats::fused_slices,
                      &reactor_stats::direct_reads,       &reactor_stats::direct_bytes,
                      &reactor_stats::posix_device_reads, &reactor_stats::posix_device_bytes,
                      &reactor_stats::host_reads,         &reactor_stats::host_bytes,
                      &reactor_stats::cufile_failures,    &reactor_stats::parked_slices,
                      &reactor_stats::staged_writes,      &reactor_stats::staged_write_bytes,
                      &reactor_stats::direct_writes,      &reactor_stats::direct_write_bytes,
                      &reactor_stats::lane_reads,         &reactor_stats::lane_bytes,
                      &reactor_stats::lane_issued_bytes,  &reactor_stats::lane_parts,
                      &reactor_stats::lane_resubmits,     &reactor_stats::lane_failures,
                      &reactor_stats::lane_gated_reads,   &reactor_stats::batch_reads,
                      &reactor_stats::batch_bytes,        &reactor_stats::batch_failures,
                      &reactor_stats::batch_waves,        &reactor_stats::batch_entries,
                      &reactor_stats::batch_rejected_submits,
                      &reactor_stats::batch_partial_status,
                      &reactor_stats::batch_handle_resets, &reactor_stats::batch_deferred,
                      &reactor_stats::batch_submit_ns};
  }

  reactor_stats& operator+=(reactor_stats const& other) noexcept
  {
    for (auto const field : fields()) {
      this->*field += other.*field;
    }
    return *this;
  }

  /// Subtract @p other field by field, saturating at 0 (for per-interval deltas).
  reactor_stats& operator-=(reactor_stats const& other) noexcept
  {
    for (auto const field : fields()) {
      this->*field = this->*field > other.*field ? this->*field - other.*field : 0;
    }
    return *this;
  }

  [[nodiscard]] friend reactor_stats operator+(reactor_stats lhs, reactor_stats const& rhs) noexcept
  {
    lhs += rhs;
    return lhs;
  }

  /// @p lhs minus @p rhs field by field, saturating at 0.
  [[nodiscard]] friend reactor_stats operator-(reactor_stats lhs, reactor_stats const& rhs) noexcept
  {
    lhs -= rhs;
    return lhs;
  }

  bool operator==(reactor_stats const&) const = default;
};

/**
 * @brief Single-threaded local-file I/O reactor that speaks cuFile / GDS.
 *
 * Owns one worker thread and a two-tier request queue (see @ref io_priority).
 * The worker keeps one storage read in flight, so a pool's queue depth per GPU
 * is the number of reactors serving it (see @c cufile_ioctx and
 * @c plan_cufile_pool).
 *
 * Device destinations of at least @ref config::gds_threshold bytes of a file
 * registered with cuFile are read by synchronous @c cuFileRead calls:
 * - staged (see @ref staging): the range is widened to 4 KiB boundaries and read
 *   with one call per @ref config::staging_piece_bytes into a registered, aligned
 *   slot the reactor holds, then the wanted bytes are copied device-to-device into
 *   the destination.  Only that copy has to follow the work queued on the
 *   caller's stream before the read: when the stream is still busy, a one-piece
 *   read is issued at once into a spare slot and its copy waits on the GPU
 *   (@c cudaStreamWaitEvent), so the I/O overlaps the caller's compute.
 * - direct: cuFile reads into the destination itself, after the caller's
 *   stream has finished the work queued before the call; cuFile bounces an
 *   unaligned or unregistered read through its own buffers.
 * Everything else (host reads, small reads, dual host+device requests, files
 * cuFile does not serve) falls back to POSIX @c pread, with a host-to-device copy
 * when a device destination was also requested.  A cuFile read that fails
 * degrades to that fallback instead of failing the read.
 *
 * Writes (on objects from @ref create_io_object_for_write) are split at the
 * write granule like the uring reactor's: the granule-aligned body of a device
 * source of at least @ref config::gds_threshold bytes goes to synchronous
 * @c cuFileWrite calls (after the work queued on the source's stream) --
 * staged like a read when the reactor holds slots on the source's GPU: copied
 * device-to-device into a registered slot a piece at a time and written from
 * there, so cuFile can DMA it rather than bounce the unregistered source --
 * while
 * the sub-granule head and tail, smaller bodies and host sources are written
 * with @c pwrite -- O_DIRECT for aligned pieces, buffered otherwise -- device
 * bytes being staged through the reactor's pinned scratch by D2H copies.  A
 * host source that lines up with the file goes O_DIRECT straight from the
 * caller when its body is at least 4 MiB; any other host source is written
 * through the page cache with its writeback started at once
 * (@c sync_file_range), which overlaps the kernel's copy with the device write
 * and measured faster than staging it for O_DIRECT -- so spilled host data can
 * transiently occupy the page cache (clean and reclaimable once written back;
 * durability still needs a flush).
 * Host-only reads go O_DIRECT, through the scratch when not block-aligned.  A
 * flush is an @c fdatasync.  A transfer's pinned host scratch is borrowed from
 * the HOST memory space pool of the reactor's NUMA node when one is registered
 * (@c memory::find_host_pool), so it counts against that space only while the
 * transfer runs; otherwise the reactor keeps scratch of its own.  On a platform where cuFile
 * runs only in compatibility mode (@c probe_platform().compat_only) aligned
 * device bodies are staged too: cuFile then bounces through host memory itself,
 * and its 1 MiB writes measured about half as fast as the staged path when they
 * extend a file.
 *
 * The synchronous API is deliberate (it is what NVIDIA's own consumers use, with
 * one thread per read in flight): @c cuFileReadAsync serialises a stream's
 * operations, runs synchronously in the submitting thread in compatibility mode,
 * and keeps per-stream state keyed by caller stream handles the reactor neither
 * registers nor deregisters.
 *
 * Ordering against the caller's stream: @c cufile_ioctx records, on the
 * submitting thread, an event on the caller's stream (@c device_buffer::ready_gate)
 * and the worker waits on that event, never on the stream handle itself (a
 * per-thread default stream names a different stream on the worker).  A slice
 * without one (enqueued on the reactor directly) has its stream checked on the
 * worker at dispatch, the null stream meaning the legacy default stream.  A
 * device slice that must wait is parked rather than blocking the worker; parked
 * slices hold nothing and never stop the reactor from serving other requests.
 *
 * The worker takes a high-priority request before a low one and parks an active
 * low request at a slice boundary while high requests wait.  Priority is bounded
 * (see @ref config::low_service_interval) so waiting low work is never starved.
 *
 * Completion: a request's future settles only once every one of its slices has
 * stopped touching its destination (or, for a write, its source), failed and
 * cancelled siblings included.  cuFile and POSIX transfers are synchronous on
 * the worker; scatter and staging copies run on reactor-owned streams and are
 * host-complete (their event completed) before their slice settles; a parked or
 * cancelled slice settles without having issued anything.  The bytes of a
 * successful read are therefore visible from any stream without further CUDA
 * synchronization; the caller owns the destination (or source) memory until the
 * future settles.
 *
 * Fused stream lanes (@c config::fused_reads, on reactors whose placement allows them):
 * @c cufile_ioctx packs the device reads of a request into 4 KiB-aligned windows of at most
 * a lane slot (@c plan_fused_reads), and the reactor issues each such lane read as one
 * @c cuFileReadAsync into a free lane's registered slot on the lane's own registered stream,
 * followed on that stream by the device-to-device copies into the caller's destinations
 * (waiting on the GPU for the caller's stream when its gate is still pending) and a stream
 * callback.  The worker keeps up to a lane set's capacity of such reads in flight per device
 * and settles each when its callback reported it: a lane read's slice settles only once its
 * copies completed, so the contract is the synchronous path's (host-complete, the bytes
 * visible from any stream).  A short read is continued on its lane; a failed one is re-served
 * through the synchronous paths (and counted in @c lane_failures); after
 * @c fused_read_config::failure_limit consecutive failures the reactor stops using lanes for
 * @c fused_read_config::cooldown_s.  A lane read that finds no free lane waits in a
 * one-element backlog and the worker stops taking slices until a lane frees, so excess reads
 * wait in the queue, where priorities still apply.
 *
 * Host batch (@c config::host_batch, on reactors whose placement allows it): a host-only read
 * whose 4 KiB-aligned body is at least @c host_batch_config::min_bytes and whose destination is
 * congruent with the file offset reads its sub-block head and tail synchronously and its body
 * as cuFile batch entries straight into the destination.  The reactor's batch reader submits
 * and reaps on a helper thread (whole waves, see @c batch_submit_policy) and wakes the worker
 * when a wave completed; the worker never blocks in cuFile for it.  The read holds one
 * coordinator credit until its last entry settled; an entry that fails or comes back short is
 * finished synchronously (@c batch_failures).  Such reads count in @c batch_reads /
 * @c batch_bytes, never in @c host_reads / @c host_bytes.
 *
 * One wait point: the worker sleeps only on a semaphore that @ref enqueue, lane completion
 * callbacks, the batch helper, write-interval wakeups and @ref shutdown signal (coalesced by a
 * pending flag),
 * with a timeout that is the idle wait when nothing is in flight, the poll interval while
 * copies, parked slices or polled lanes need polling, and 1 ms while only callback-driven
 * lanes are in flight.
 *
 * Fragmented (cache-chunk) destinations are not served: the reactor's pinned
 * scratch is not shaped like cache chunks, so it opts out of the prefetching
 * cache (@c reactor_traits::supports_vector_host_read is false).
 *
 * @note Destinations and sources are never registered: registering a one-shot
 *       buffer costs about as much as reading it, and its 4 KiB phase rarely
 *       matches the file offset anyway.  The staging slots are registered once and reused by the
 *       whole process (see @c reserve_staging).
 *
 * @note Known gap: a device read below @ref config::gds_threshold, or of a file
 *       without a cuFile handle, is a synchronous @c pread plus H2D copy, so a
 *       reactor serves one at a time (64 KiB reads: about 1.3 GB/s from four
 *       reactors against about 6 GB/s for the uring reactor, compat mode).
 *
 * Models the reactor concept consumed by @c templated_ioctx.
 */
class cufile_reactor {
 public:
  /// A read here is a syscall against NVMe (POSIX fallback) or a synchronous
  /// cuFile call (GDS fast path); batching either buys little.
  static constexpr bool prefers_bulk_io = false;
  static constexpr bool wants_submit_fence = true;

  /// Shared, immutable config for a pool of reactors.  A dedicated type so a
  /// future iteration can add shared state (a registered-buffer cache, a stream
  /// registry, ...) without changing the reactor constructor's shape.
  class reactor_context : public std::enable_shared_from_this<reactor_context> {
   public:
    explicit reactor_context(config cfg) : _config(std::move(cfg)) {}

    /// @p staging_caps: registered-staging cap per device (bytes, indexed by
    /// device id, 0 = none), as a pool plan bounded it by BAR1 and cuFile's
    /// pinned-memory limit; devices past its end use the config's cap.
    reactor_context(config cfg, std::vector<std::size_t> staging_caps)
      : _config(std::move(cfg)), _staging_caps(std::move(staging_caps))
    {
    }

    /// As above, with @p lane_quotas: the fused-read lanes the plan reserved per device
    /// (@c device_group_plan::lane_slots, indexed by device id); devices past its end use
    /// @ref lane_quota's default.
    reactor_context(config cfg,
                    std::vector<std::size_t> staging_caps,
                    std::vector<std::size_t> lane_quotas)
      : _config(std::move(cfg)),
        _staging_caps(std::move(staging_caps)),
        _lane_quotas(std::move(lane_quotas))
    {
    }

    [[nodiscard]] const config& cfg() const noexcept { return _config; }

    struct engine_binding {
      device_engine* engine{};
      device_arena* arena{};
      std::shared_ptr<cufile_transport> transport;
      std::string device_uuid;
      std::uint64_t generation{};
      bool simulated{false};
      /// Injection only; production engines have process lifetime.
      std::shared_ptr<device_engine> owner;
    };
    /// Control-thread publication; injecting a simulator never changes process runtime state.
    void install_engine(int device, std::shared_ptr<engine_binding> binding);
    [[nodiscard]] std::shared_ptr<engine_binding> engine_for_device(int device) const noexcept;
    [[nodiscard]] bool has_simulated_engine() const noexcept { return _simulated_engine.load(); }
    /// Per-active-device asynchronous warm-up; rejected submissions are logged and retried.
    void ensure_engine(int device) noexcept;
    void notify_engines() const noexcept;

    /// The registered-staging cap of @p device, in bytes (0 = none), as the staging pool
    /// enforces it: the plan's cap for a planned device (it includes the fused-read lanes);
    /// otherwise @c config::max_staging_bytes_per_device plus the lanes' registered bytes
    /// (@c fused_lane_staging_bytes), 0 staying 0.
    [[nodiscard]] std::size_t staging_cap(int device) const noexcept;

    /// True for the first caller only: the reactors of one ioctx (which share this context)
    /// log the effective host-batch submit policy once.
    [[nodiscard]] bool claim_batch_policy_log() const noexcept
    {
      return !_batch_policy_logged.exchange(true, std::memory_order_relaxed);
    }

    /// The fused-read lanes the process may hold on @p device for this context: the plan's
    /// @c lane_slots for a planned device, else
    /// @c min(reactors_per_device x lanes_per_reactor, max_lanes_per_device) of
    /// @c config::fused_reads (what @c fused_lane_staging_bytes budgets).
    [[nodiscard]] std::size_t lane_quota(int device) const noexcept;

   private:
    config _config;
    std::vector<std::size_t> _staging_caps;
    mutable std::atomic<bool> _batch_policy_logged{false};
    std::vector<std::size_t> _lane_quotas;
    mutable std::mutex _engine_mutex;
    using engine_bindings = std::vector<std::shared_ptr<engine_binding>>;
    std::atomic<std::shared_ptr<const engine_bindings>> _engines;
    std::vector<int> _warming;
    std::atomic<bool> _simulated_engine{false};

  };

  using native_handle_type   = int;
  using io_object_type       = cufile_io_object;
  using reactor_config_type  = config;
  using reactor_context_type = reactor_context;

  explicit cufile_reactor(std::shared_ptr<reactor_context> ctx,
                          std::string_view tname      = "cufile_reactor",
                          reactor_placement placement = {});

  ~cufile_reactor();

  cufile_reactor(cufile_reactor const&)            = delete;
  cufile_reactor& operator=(cufile_reactor const&) = delete;

  [[nodiscard]] const reactor_config_type& get_config() const noexcept { return _config; }

  /// Zero: the reactor's pinned scratch is not a cache-chunk staging area (it
  /// does not serve the prefetching cache), so there is no chunk size for a
  /// cache to match.  @c ioctx::initialize_cache accepts 0 as "backend does not
  /// stage".
  [[nodiscard]] std::size_t staging_block_size() const noexcept { return 0; }

  /// Open the process-wide cuFile driver (once) and launch the worker thread.
  /// Idempotent: a second call while the worker is running is a no-op.
  void start();

  [[nodiscard]] bool submit_fence_enabled() const noexcept;
  void record_submit_fences(cufile_io_object const& file, std::vector<prepared_io_slice>& slices);
  /// Single-reactor batching is allowed only for currently eligible small device reads.
  [[nodiscard]] bool coalescing_candidate(cufile_io_object const& object,
                                          std::span<prepared_io_slice const> slices) const noexcept;
  [[nodiscard]] std::shared_ptr<reactor_context> context() const noexcept { return _ctx; }

  /// Present for reactor-concept symmetry with the uring reactor; does nothing.
  void interrupt() noexcept;

  /// Close admission, request stop, join the worker, and cancel anything that
  /// never made it out of the queue.  Slices parked behind a busy stream are
  /// cancelled at once; a transfer already running finishes its current piece.
  /// The pinned scratch goes back to a process-wide pool rather than to
  /// @c cudaFreeHost, which would synchronize the device and make shutdown wait
  /// for other callers' GPU work.  Idempotent.
  void shutdown();

  /// Synchronous buffered host read (pread on the buffered fd).  Blocks the
  /// caller; used by @c host_read_io.  Runs on the calling thread and never
  /// touches the worker's pinned scratch.
  std::size_t host_read(cufile_io_object const& file,
                        std::size_t offset,
                        std::size_t size,
                        std::uint8_t* destination);

  void enqueue(std::unique_ptr<grouped_io_request> request) noexcept;

  /// Approximate bytes queued on this reactor plus those of the transfer its worker is running
  /// (a transfer is synchronous on the worker, so the reactor is busy until it is done).  Slices
  /// parked behind a busy stream are not counted.
  [[nodiscard]] std::size_t queued_bytes() const noexcept
  {
    return _queued_bytes.load(std::memory_order_relaxed);
  }

  /// Whether the calling thread is this reactor's worker.
  [[nodiscard]] bool on_worker_thread() const noexcept
  {
    return _worker.get_id() == std::this_thread::get_id();
  }

  /// How this reactor has served its reads so far (relaxed snapshot).
  [[nodiscard]] reactor_stats stats() const noexcept;

  /// The GPU this reactor mainly serves and the CPUs its worker is pinned to.
  [[nodiscard]] reactor_placement const& placement() const noexcept { return _placement; }

  /// Whether this reactor can serve lane reads into @p device now: fused reads enabled, its
  /// placement allows lanes, not cooling down after failures, and its lane set for @p device
  /// has lanes (a device it has not tried yet counts as usable).  Any thread; the ioctx
  /// routes by it.
  [[nodiscard]] bool lanes_usable(int device) const noexcept;

  /// Whether @p path can be served by this reactor: true iff it names an
  /// existing, accessible regular file.  Routing between this and the uring
  /// reactor (both claim local files) is decided by @c io_context_registry.
  [[nodiscard]] static bool supports(std::string_view path);

  /// Open @p path with default options (O_DIRECT, @c compat::automatic).
  /// See the three-argument overload.
  static std::unique_ptr<cufile_io_object> create_io_object(std::string path);

  /**
   * @brief Open the buffered fd and, when @p use_odirect, an O_DIRECT fd, and
   *        register the latter with cuFile.
   *
   * @p compat_mode (default @c compat::automatic) selects what a file that cannot
   * be registered means: @c on never registers, @c automatic falls back to POSIX
   * for that file, @c off treats it as an error.  The O_DIRECT fd reopens the
   * buffered one through @c /proc/self/fd (the path only where procfs is not
   * mounted), and the object carries the buffered fd's @c local_stat identity
   * and size.
   *
   * @throws std::runtime_error / std::system_error when the file cannot be
   *         opened or stat'ed, or when @p compat_mode is @c off and cuFile
   *         registration fails; descriptor_mismatch_error when, on two
   *         attempts, the two fds do not name the same file, or when the
   *         O_DIRECT fd cannot be stat'ed.
   */
  static std::unique_ptr<cufile_io_object> create_io_object(std::string path,
                                                            bool use_odirect,
                                                            std::optional<compat> compat_mode);

  // -- Writes (the reactor write contract, templated_ioctx.hpp) -------------

  /// Open @p path for writing with default options (O_DIRECT,
  /// @c compat::automatic).  See the four-argument overload.
  static std::unique_ptr<cufile_io_object> create_io_object_for_write(std::string path,
                                                                      write_open_options options);

  /**
   * @brief Open @p path read+write per @p options and register its O_DIRECT fd
   *        with cuFile, with the same @p use_odirect / @p compat_mode rules as
   *        @ref create_io_object.
   *
   * @throws std::system_error for open failures (@c ENOENT, @c EEXIST, ...);
   *         std::invalid_argument / std::runtime_error as @ref create_io_object.
   */
  static std::unique_ptr<cufile_io_object> create_io_object_for_write(
    std::string path,
    write_open_options options,
    bool use_odirect,
    std::optional<compat> compat_mode);

  /// Synchronous host write on the calling thread: the write-granule-aligned
  /// body goes O_DIRECT when the source is block-aligned, head and tail through
  /// the page cache.  Raises the object's size; returns @p size.
  /// @throws std::system_error (@c EBADF for an object not opened for writing).
  std::size_t host_write(cufile_io_object const& file,
                         std::size_t offset,
                         std::size_t size,
                         std::uint8_t const* source);

  /// Nonblocking eligibility snapshot used to track queued engine writes before flush capture.
  bool engine_write_candidate(std::shared_ptr<const io_object> const& object,
                              prepared_io_slice const& slice) noexcept;

  /// @c fdatasync the file on the calling thread.  @throws std::system_error.
  static void flush(cufile_io_object const& file);

  /// Same alignment / coalescing policy as the uring reactor: outward to page
  /// boundaries, then merge overlapping / adjacent ranges.  @p alignment smaller
  /// than @c IO_BLOCK_SIZE is ignored in favour of @c IO_BLOCK_SIZE.
  [[nodiscard]] static std::vector<byte_range> align_and_coalesce(
    std::span<byte_range const> ranges,
    std::optional<std::size_t> alignment = std::nullopt) noexcept;

 private:
  /// A device-only slice parked until the work queued on its stream before the
  /// read has finished (@c gate fires): the submission-time event, or one the
  /// worker recorded at dispatch for a slice that came without one.
  struct pending_write {
    std::shared_ptr<const io_object> obj;
    prepared_io_slice slice;
    std::shared_ptr<grouped_coordinator> coordinator;
    std::shared_ptr<std::atomic<bool>> ready;
  };
  struct write_wakeup {
    std::mutex mutex;
    std::function<void()> notify;
  };
  std::shared_ptr<write_wakeup> _write_wakeup = std::make_shared<write_wakeup>();
  struct waiting_slice {
    std::shared_ptr<const io_object> obj;
    prepared_io_slice slice;
    std::shared_ptr<grouped_coordinator> coordinator;
    std::shared_ptr<cucascade::cuda::cuda_event> gate;
    /// Re-dispatched with lanes bypassed: the fallback of a failed lane read.
    bool no_lanes{false};
  };

  /// Registered staging slots and their copy streams for one device (defined in
  /// the source).
  struct device_staging;

  /// Where a slice stands against the work queued on its caller's stream.
  enum class gate_state : std::uint8_t { ready, pending, unchecked };

  /// A non-blocking stream owned by the reactor, destroyed with it.
  struct owned_stream {
    owned_stream() noexcept = default;
    owned_stream(owned_stream&& other) noexcept;
    owned_stream& operator=(owned_stream&& other) noexcept;
    owned_stream(owned_stream const&)            = delete;
    owned_stream& operator=(owned_stream const&) = delete;
    ~owned_stream();

    cudaStream_t stream{nullptr};
    int device_id{-1};
  };

  /// One record per in-flight H2D copy issued for a read that took the POSIX
  /// fallback and also had a device destination.
  struct inflight_copy {
    std::unique_ptr<io_op_request> request;
    cucascade::cuda::cuda_event event;
    std::size_t bytes{0};
  };

  void worker_loop(std::stop_token const& stop_token);

  /// Lazily opened only by real handoffs; legacy I/O publishes before blocking.
  struct engine_publication {
    reactor_context& context;
    std::shared_ptr<std::atomic<bool>> ready;
    std::shared_ptr<std::atomic<bool>> take() noexcept
    {
      try {
        if (!ready) ready = std::make_shared<std::atomic<bool>>(false);
      } catch (...) { /* Coalescing is optional under metadata pressure. */
      }
      return ready;
    }
    void publish() noexcept
    {
      if (!ready) return;
      ready->store(true, std::memory_order_release);
      context.notify_engines();
      ready.reset();
    }
    ~engine_publication() { publish(); }
  };

  /// Turn one slice into a lane read, cuFile reads or a POSIX pread.  Never throws --
  /// errors are reported through the slice's completion and coordinator.  A
  /// device-only slice whose stream still has queued work is parked (see
  /// @ref park_if_stream_busy) unless @p stream_ready says its gate passed.
  /// @p allow_lanes false keeps a slice off the fused lanes (a lane read's fallback);
  /// @p lane_owned is set when an asynchronous facility took the slice -- a lane (or the
  /// lane backlog) or the host batch reader (or its backlog) -- whose queued bytes then stay
  /// counted until it settles.  Without @p lane_owned the caller does not count the slice in
  /// @c queued_bytes, and neither does the facility.
  void dispatch_slice(std::shared_ptr<const io_object> const& obj,
                      prepared_io_slice slice,
                      std::shared_ptr<grouped_coordinator> const& coordinator,
                      bool stream_ready = false,
                      bool* handed_off = nullptr,
                      engine_publication* publication = nullptr,
                      bool allow_lanes = true,
                      bool* lane_owned = nullptr) noexcept;

  /// When the slice's stream still has queued work, record an event on it and
  /// park the slice in @ref _waiting; returns whether it was parked.  For a slice
  /// enqueued without a submission-time gate only.
  /// @throws cucascade::cuda_error when the stream cannot be queried / recorded.
  [[nodiscard]] bool park_if_stream_busy(std::shared_ptr<const io_object> const& obj,
                                         prepared_io_slice& slice,
                                         std::shared_ptr<grouped_coordinator> const& coordinator);

  /// Where @p slice stands against its submission-time gate (@c unchecked when it
  /// has none).  @throws cucascade::cuda_error when the gate cannot be queried.
  [[nodiscard]] static gate_state gate_of(prepared_io_slice const& slice);

  /// Park @p slice until its submission-time gate fires.
  void park_on_gate(std::shared_ptr<const io_object> const& obj,
                    prepared_io_slice&& slice,
                    std::shared_ptr<grouped_coordinator> const& coordinator);

  /// This reactor's staging for @p device, set up on first use; null when
  /// staging is off or no slot could be had.  The calling thread's current
  /// device must be @p device.
  [[nodiscard]] device_staging* staging_for(int device);

  /// Read @p request's range through registered staging (see the class
  /// comment).  @p gate_pending: the caller's stream had not reached the gate at
  /// dispatch, so the scatter copy must wait on the GPU.  Returns true when
  /// @p request was settled or handed to a pending copy, false when cuFile failed
  /// and the caller must take the POSIX path for the whole range (by then the
  /// gate has passed).
  [[nodiscard]] bool read_cufile_staged(cufile_io_object const& file,
                                        prepared_io_slice const& slice,
                                        std::unique_ptr<io_op_request>& request,
                                        device_staging& staging,
                                        bool gate_pending);

  /// The reactor-owned stream of @p device_id for staging copies, created on
  /// first use.  The calling thread's current device must be @p device_id.
  [[nodiscard]] cudaStream_t copy_stream(int device_id);

  /// The reactor-owned stream of @p device_id for the copies of host+device
  /// pairs, which wait there for their caller's gate: kept apart from
  /// @ref copy_stream, which the worker synchronizes on, so a busy caller stream
  /// never stalls the worker.  Created on first use, like @ref copy_stream.
  [[nodiscard]] cudaStream_t pair_stream(int device_id);

  /// The stream of @p device_id in @p streams, created non-blocking on first use.
  [[nodiscard]] static cudaStream_t device_stream(std::vector<owned_stream>& streams,
                                                  int device_id);

  /// Read @p request's range straight into the slice's device buffer with
  /// synchronous @c cuFileRead calls, once the slice's stream has drained the work
  /// queued before the read.  Returns true when @p request was settled (success or
  /// a definite error), false when cuFile failed and the caller must take the
  /// POSIX path for the whole range.
  [[nodiscard]] bool read_cufile(cufile_io_object const& file,
                                 prepared_io_slice const& slice,
                                 std::unique_ptr<io_op_request>& request);

  /// Serve a write slice: host sources by @c pwrite, device sources by
  /// @c cuFileWrite for the aligned body (from a registered staging slot when
  /// the reactor has one) and staged @c pwrite for the rest.
  void dispatch_write(cufile_io_object const& file,
                      prepared_io_slice const& slice,
                      io_op_request& request);

  /// Write the device-source pieces @p pieces of @p slice by D2H copies (on the
  /// reactor's copy stream) into the pinned scratch and @c pwrite, double-buffered.
  void write_device_staged(cufile_io_object const& file,
                           prepared_io_slice const& slice,
                           std::span<range const> pieces,
                           int device_id);

  /// Write @p piece from device memory @p source with @c cuFileWrite.  Returns
  /// false when cuFile failed and the caller must stage the piece instead.
  [[nodiscard]] bool write_cufile(cufile_io_object const& file,
                                  std::uint8_t const* source,
                                  range piece);

  /// Write the 4 KiB-aligned @p piece from device memory @p source through a
  /// free slot of @p staging: a device copy into the slot, then @c cuFileWrite
  /// from it, a slot at a time.  Writes directly (@ref write_cufile) when no slot
  /// is free.  Returns false when cuFile failed and the caller must stage the
  /// piece through POSIX instead.
  [[nodiscard]] bool write_cufile_staged(cufile_io_object const& file,
                                         std::uint8_t const* source,
                                         range piece,
                                         device_staging& staging);

  /// The reactor's own scratch, allocated on first use.
  [[nodiscard]] std::uint8_t* scratch();

  /// Pinned host memory one transfer stages through (see @ref lease_scratch).
  struct scratch_lease;

  /// Up to @p slots slots of at least @p bytes each (capped at the scratch slot
  /// size) for one transfer: borrowed from the HOST memory space pool of this
  /// reactor's NUMA node (@c memory::find_host_pool), so the bytes count against
  /// that space and are back in it before the transfer settles; or the reactor's
  /// own scratch when no pool is registered or it cannot lend them.
  [[nodiscard]] scratch_lease lease_scratch(std::size_t bytes, std::size_t slots);

  /// POSIX pread into the slice's host buffer, then the H2D copy when a device
  /// destination was asked for too; a device-only slice goes to
  /// @ref read_device_staged.
  void dispatch_posix(cufile_io_object const& file,
                      prepared_io_slice const& slice,
                      std::unique_ptr<io_op_request> request);

  /// O_DIRECT read of @p rng into host memory @p destination: straight into it
  /// when the range and the pointer are block-aligned, else whole blocks through
  /// the scratch, copied out.  Returns the bytes read (short only at EOF).
  [[nodiscard]] std::size_t read_host_direct(cufile_io_object const& file,
                                             std::uint8_t* destination,
                                             range rng);

  /// POSIX pread of a device-only slice through the reactor's reusable pinned
  /// scratch, in pieces of @ref scratch_slot_bytes whose H2D copies (on the
  /// reactor's copy stream) overlap the next read.  Settles @p request before
  /// returning.
  void read_device_staged(cufile_io_object const& file,
                          prepared_io_slice const& slice,
                          io_op_request& request);

  /// Dispatch every parked slice whose stream gate opened, then settle every
  /// in-flight H2D copy and staged scatter copy whose CUDA event has completed.
  void poll_completions() noexcept;

  /// Settle every staged scatter copy whose event completed; with @p wait, wait
  /// for each first and settle it with @p error (shutdown).  Then, with @p wait,
  /// give every slot back to the process-wide pool.
  void poll_staged(bool wait, grouped_coordinator::error_type const* error) noexcept;

  /// Synchronize and settle every outstanding copy with @p terminal_error, so no
  /// coordinator promise is stranded at shutdown.
  void drain_inflight(grouped_coordinator::error_type const& terminal_error) noexcept;

  /// Settle every parked slice with @p error; nothing was issued for them.
  void cancel_waiting(grouped_coordinator::error_type const& error) noexcept;

  /// Asynchronous items holding copy resources (see @ref config::max_inflight);
  /// parked slices hold none and are not counted.
  [[nodiscard]] std::size_t inflight_count() const noexcept
  {
    return _inflight_copy.size() + _staged_inflight;
  }

  struct engine_admission {
    std::shared_ptr<std::atomic<bool>> cancel{std::make_shared<std::atomic<bool>>(false)};
    std::atomic<std::size_t> outstanding{0};
  };
  struct engine_retirement_ticket;
  struct engine_route_result;
  [[nodiscard]] engine_route_result engine_route(std::shared_ptr<const io_object> const& obj,
                                                 prepared_io_slice const& slice) noexcept;
  /// Shared tickets, not early AM-1 futures, determine facade drain completion.
  std::shared_ptr<engine_admission> _engine_outstanding;

  std::shared_ptr<reactor_context> _ctx;
  reactor_config_type _config;
  std::string _tname;
  reactor_placement _placement;

  /// Transfer counters (see @ref reactor_stats); written by the worker, read by any
  /// thread.
  struct counters {
    std::atomic<std::uint64_t> staged_reads{0};
    std::atomic<std::uint64_t> staged_bytes{0};
    std::atomic<std::uint64_t> early_staged_reads{0};
    std::atomic<std::uint64_t> fused_slices{0};
    std::atomic<std::uint64_t> direct_reads{0};
    std::atomic<std::uint64_t> direct_bytes{0};
    std::atomic<std::uint64_t> posix_device_reads{0};
    std::atomic<std::uint64_t> posix_device_bytes{0};
    std::atomic<std::uint64_t> host_reads{0};
    std::atomic<std::uint64_t> host_bytes{0};
    std::atomic<std::uint64_t> cufile_failures{0};
    std::atomic<std::uint64_t> parked_slices{0};
    std::atomic<std::uint64_t> staged_writes{0};
    std::atomic<std::uint64_t> staged_write_bytes{0};
    std::atomic<std::uint64_t> direct_writes{0};
    std::atomic<std::uint64_t> direct_write_bytes{0};
    std::atomic<std::uint64_t> lane_reads{0};
    std::atomic<std::uint64_t> lane_bytes{0};
    std::atomic<std::uint64_t> lane_issued_bytes{0};
    std::atomic<std::uint64_t> lane_parts{0};
    std::atomic<std::uint64_t> lane_resubmits{0};
    std::atomic<std::uint64_t> lane_failures{0};
    std::atomic<std::uint64_t> lane_gated_reads{0};
    std::atomic<std::uint64_t> batch_reads{0};
    std::atomic<std::uint64_t> batch_bytes{0};
    std::atomic<std::uint64_t> batch_failures{0};
    std::atomic<std::uint64_t> batch_waves{0};
    std::atomic<std::uint64_t> batch_entries{0};
    std::atomic<std::uint64_t> batch_rejected_submits{0};
    std::atomic<std::uint64_t> batch_partial_status{0};
    std::atomic<std::uint64_t> batch_handle_resets{0};
    std::atomic<std::uint64_t> batch_deferred{0};
    std::atomic<std::uint64_t> batch_submit_ns{0};
  };
  counters _counters;

  std::stop_source _stop_source;
  std::jthread _worker;
  tiered_blocking_queue<std::unique_ptr<grouped_io_request>> _requests;
  mutable std::mutex _enqueue_mutex;
  std::atomic<std::size_t> _queued_bytes{0};
  std::atomic<bool> _accepting{false};

  /// Returns the pinned scratch to a process-wide pool instead of freeing it:
  /// @c cudaFreeHost synchronizes the whole device, so freeing it would make a
  /// reactor's shutdown wait for other callers' GPU work.
  struct pinned_deleter {
    void operator()(std::uint8_t* memory) const noexcept;
  };
  using pinned_scratch = std::unique_ptr<std::uint8_t, pinned_deleter>;

  /// Slots of the device-fallback scratch: two, so a read overlaps a copy.
  static constexpr std::size_t scratch_slots = 2;
  /// Bytes per scratch slot (the fallback's piece size).
  static constexpr std::size_t scratch_slot_bytes = 8UL << 20;

  // Worker-only state; touched exclusively from the reactor thread (and by
  // shutdown() once the worker has been joined).
  std::vector<inflight_copy> _inflight_copy;
  std::vector<waiting_slice> _waiting;
  std::vector<pending_write> _write_waiting;
  /// Staging-copy streams, indexed by device id.
  std::vector<owned_stream> _copy_streams;
  /// Gated pair-copy streams, indexed by device id.
  std::vector<owned_stream> _pair_streams;
  /// Pinned (portable) staging for device-only POSIX reads and writes and
  /// unaligned host reads when no HOST memory space pool can lend it (see
  /// @ref lease_scratch), taken from a process-wide pool on first use (so a
  /// reactor that never stages holds no pinned memory) and given back to it at
  /// shutdown.
  pinned_scratch _scratch;
  /// Registered device staging, indexed by device id (null until first used).
  std::vector<std::unique_ptr<device_staging>> _staging;
  /// Staged scatter copies still waiting for their caller's stream.
  std::size_t _staged_inflight{0};

  // -- The one wait point ----------------------------------------------------

  /// Wake the worker from any thread (enqueue, lane callbacks on CUDA's callback thread,
  /// write wakeups, shutdown).  The caller publishes its work first.
  void notify_worker() noexcept;

  /// Sleep until @ref notify_worker, at most @p timeout_us.  Returns at once when a source
  /// already has work: a stop request, completed asynchronous I/O, or -- with
  /// @p requests_count -- a queued request.
  void wait_for_work(std::int64_t timeout_us,
                     bool requests_count,
                     std::stop_token const& stop_token) noexcept;

  lightweight_semaphore _wake;
  std::atomic<bool> _wake_pending{false};

  // -- Asynchronous I/O facilities (fused lanes; the host batch plugs in here) --------

  /// Fused-lane state (defined in the source); null unless fused reads are enabled and the
  /// placement allows them.  Worker-only.
  struct lane_state;
  std::unique_ptr<lane_state> _lanes;

  /// Published for @ref lanes_usable (any thread): per device, 0 = not tried yet, 1 = has
  /// lanes, 2 = none; and the end of the current cooldown (steady clock, ns since epoch).
  static constexpr std::size_t max_lane_devices = 64;
  std::array<std::atomic<std::uint8_t>, max_lane_devices> _lane_device_state{};
  std::atomic<std::int64_t> _lane_cooldown_until_ns{0};

  // Lane-specific helpers (called from the generic async hooks below).
  void reap_lanes() noexcept;
  void poll_held_lanes() noexcept;
  void issue_lane_backlog() noexcept;
  /// Settle (or move to the held list) the lane read of reaped result @p result_index.
  void settle_lane_result(std::size_t result_index) noexcept;
  [[nodiscard]] bool issue_lane_op(std::uint32_t index) noexcept;
  void copy_out_lane_op(std::uint32_t index) noexcept;
  /// Serve a lane op's destinations through the synchronous paths of this reactor (parked
  /// on its gate first when the gate has not fired), then release the op.
  void serve_lane_op_synchronously(std::uint32_t index) noexcept;
  /// Release a lane op whose read was not served and settle it with @p error.
  void fail_lane_op(std::uint32_t index, grouped_coordinator::error_type const& error) noexcept;
  void publish_lane_state(int device, bool usable) noexcept;
  void drain_lanes(grouped_coordinator::error_type const& terminal_error) noexcept;

  /// Reap every asynchronous facility once (no blocking): settle finished lane reads,
  /// refill lanes from the backlog, run fallbacks.  Called every worker pass.
  void reap_async() noexcept;
  /// Whether asynchronous I/O is in flight (lane reads or batch entries, issued or staged,
  /// not yet reaped).
  [[nodiscard]] bool async_in_flight() const noexcept;
  /// Whether lane reads are in flight.
  [[nodiscard]] bool lanes_in_flight() const noexcept;
  /// Whether an asynchronous facility has completions to reap now.
  [[nodiscard]] bool async_completions_pending() const noexcept;
  /// Whether a slice waits for a free lane (the lane backlog): the worker then takes no more
  /// slices.  (The batch backlog holds back only further batch reads, see
  /// @ref batch_front_blocked.)
  [[nodiscard]] bool async_backlog_pending() const noexcept;
  /// How long the worker may sleep while asynchronous I/O is in flight (-1: no constraint).
  [[nodiscard]] std::int64_t async_wait_limit_us() const noexcept;
  /// Shutdown: cancel the backlog, wait for every read in flight and settle it (no
  /// fallback), and release the facilities.  Worker thread, after the loop.
  void drain_async(grouped_coordinator::error_type const& terminal_error) noexcept;

  /// What @ref try_lane_read did with a slice.
  enum class lane_dispatch : std::uint8_t {
    declined,  ///< not a lane read here: the synchronous paths serve it (slice untouched)
    parked,    ///< parked behind its caller's stream (see @ref park_on_gate)
    taken,     ///< issued on a lane, or waiting in the lane backlog
  };

  // -- Host batch (cuFile batch API for aligned host reads) -------------------------------

  /// Host-batch state (defined in the source); null unless the host batch is enabled and the
  /// placement allows it.  Its reader is created on the worker at the first eligible read.
  /// Worker-only.
  struct batch_state;
  std::unique_ptr<batch_state> _batch;

  /// What @ref try_batch_read did with a slice.
  enum class batch_dispatch : std::uint8_t {
    declined,  ///< not served by the batch: the synchronous paths serve it (slice untouched)
    served,    ///< settled synchronously (its head or tail read failed); not owned
    taken,     ///< its body is staged as batch entries (or waits in the batch backlog)
  };

  /// Whether @p slice (clamped to @p clamped bytes) is a batch read by its shape: a host-only,
  /// contiguous read without @c on_complete on a cuFile-registered O_DIRECT file, whose aligned
  /// body is at least @c host_batch_config::min_bytes and whose destination is congruent.
  [[nodiscard]] bool batch_eligible(cufile_io_object const& file,
                                    prepared_io_slice const& slice,
                                    std::size_t clamped) const noexcept;
  /// Whether the front slice of @p request must wait for the batch backlog: it is a batch read
  /// and another slice occupies the backlog.  Other slices dispatch past the backlog.
  [[nodiscard]] bool batch_front_blocked(grouped_io_request& request) const noexcept;
  /// Create the batch reader on first use (on the worker); false while it is unusable.  A reader
  /// that could not be set up is retried after a cooldown, a bounded number of times.
  [[nodiscard]] bool ensure_batch_reader() noexcept;
  /// Serve a batch read: its body as entries of at most @c host_batch_config::max_entry_bytes
  /// straight into the destination, its sub-block head and tail synchronously once the pass
  /// handed the wave over (they overlap it).  One coordinator credit per slice, given back when
  /// all of it settled.  @p count_queued: the caller counts the slice in @c queued_bytes until
  /// it settles.
  /// @throws only while the slice is still the caller's (before any I/O).
  [[nodiscard]] batch_dispatch try_batch_read(std::shared_ptr<const io_object> const& obj,
                                              cufile_io_object const& file,
                                              prepared_io_slice& slice,
                                              std::shared_ptr<grouped_coordinator> const& coordinator,
                                              std::size_t clamped,
                                              bool count_queued);
  /// Stage the next entries of batch slice @p index while the reader has room; it waits in
  /// the batch backlog otherwise.  Owes the rest synchronously when the reader became
  /// unavailable, and stops (settling with the cancellation) when its request stopped.
  void stage_batch_entries(std::uint32_t index) noexcept;
  /// Collect what the reader's helper published, settle what finished, continue the backlog.
  void poll_batch() noexcept;
  /// Turn the results the reader reported into slice progress; with @p terminal (shutdown) a
  /// failed entry fails its slice with that error instead of being re-read synchronously.
  void process_batch_results(grouped_coordinator::error_type const* terminal) noexcept;
  /// Owe the synchronous read of @p rng for batch slice @p index (head, tail, a failed entry's
  /// missing bytes, the rest of a body the reader can no longer take).
  void owe_batch_read(std::uint32_t index, range rng) noexcept;
  /// Run owed synchronous reads, at most @p budget bytes (a pass's worth; the rest waits).
  void run_batch_owed(std::size_t budget) noexcept;
  /// Shutdown: fail every owed read with @p error instead of running it.
  void fail_batch_owed(grouped_coordinator::error_type const& error) noexcept;
  /// Give batch slice @p index's credit back once nothing of it is outstanding.
  void maybe_settle_batch(std::uint32_t index) noexcept;
  /// Give batch slice @p index's credit back, with its error if it has one.
  void settle_batch(std::uint32_t index) noexcept;
  /// Add the reader's counter increments to the reactor's.
  void export_batch_counters() noexcept;
  /// Whether batch work is outstanding: entries staged, in flight or unreported, or owed
  /// synchronous reads.
  [[nodiscard]] bool batch_has_work() const noexcept;
  /// End of a pass: drop the staged entries of stopped requests, then hand what is staged
  /// (deferred and new together) to the helper.
  void flush_batch() noexcept;

  /// Serve @p slice through a lane of @p device_id when it is a lane read and lanes are
  /// usable.  The calling thread's current device must be @p device_id.
  /// @throws as @ref dispatch_slice's body does, only while the slice is still the caller's.
  [[nodiscard]] lane_dispatch try_lane_read(std::shared_ptr<const io_object> const& obj,
                                   cufile_io_object const& file,
                                   prepared_io_slice& slice,
                                   std::shared_ptr<grouped_coordinator> const& coordinator,
                                   int device_id,
                                   std::size_t clamped,
                                   bool stream_ready);
};

}  // namespace cucascade::io::cufile

namespace cucascade::io {

/// The reactor serves contiguous host destinations (POSIX pread), contiguous
/// device destinations (cuFileRead) and a caller-supplied contiguous
/// host+device pair (POSIX + H2D copy).  It cannot fill fragmented cache chunks.
///
/// @c supports_host_to_device_read is deliberately false even though the
/// contiguous pair works: @c ioctx::can_use_fs_cache() reads that flag (or
/// @c supports_vector_host_read) as "the prefetching cache may route through this
/// backend", and with the cache armed a device read that misses carries a
/// fragmented cache-chunk host buffer, which this reactor rejects.  Both flags
/// false keeps the cache from ever being armed here.
template <>
struct reactor_traits<cufile::cufile_reactor> {
  static constexpr bool supports_device_read         = true;
  static constexpr bool supports_host_to_device_read = false;
  static constexpr bool supports_vector_host_read    = false;
  static constexpr bool supports_device_range_read   = true;
  static constexpr bool prefers_bulk_io              = false;
};

}  // namespace cucascade::io
