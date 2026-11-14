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
#include <cucascade/io/cufile/config.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/object_identity.hpp>
#include <cucascade/io/templated_ioctx.hpp>
#include <cucascade/io/tiered_queue.hpp>
#include <cucascade/io/types.hpp>

#include <cuda_runtime.h>

#include <cufile.h>

#include <atomic>
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

// ---------------------------------------------------------------------------
// cufile_io_object
// ---------------------------------------------------------------------------

/**
 * @brief @c io_object backed by a local filesystem path and a cuFile handle.
 *
 * Owns two @c O_RDONLY @c file_descriptor s -- a buffered fd for POSIX I/O and,
 * when O_DIRECT is enabled and supported, an O_DIRECT fd registered with cuFile
 * via @c cuFileHandleRegister for GDS DMA.  The registered handle lives for the
 * object's lifetime; the destructor deregisters it before the fds close.  All
 * I/O is performed by the reactor -- this class is a passive bag of native
 * handles plus the file size.
 *
 * An object opened by the reactor carries the @c local_stat identity of its
 * buffered fd and takes its (immutable) size from it; one constructed without
 * such an identity has the base class's open-local one.
 */
class cufile_io_object final : public io_object {
 public:
  /// @p cufile_handle is null when the file was not registered with cuFile; the
  /// reactor then serves every read of it through POSIX.  @p identity is the
  /// @c local_stat identity of @p buffered_fd; when it is not a @c local_stat
  /// identity it is ignored and the object keeps @p file_size.
  cufile_io_object(std::string path,
                   file_descriptor buffered_fd,
                   file_descriptor direct_fd,
                   CUfileHandle_t cufile_handle,
                   std::size_t file_size,
                   object_identity identity = {});

  ~cufile_io_object() override;

  cufile_io_object(cufile_io_object const&)            = delete;
  cufile_io_object& operator=(cufile_io_object const&) = delete;

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return _file_size; }

  [[nodiscard]] object_identity identity() const override;

  /// The buffered fd (page cache).  Always valid.
  [[nodiscard]] int buffered_fd() const noexcept { return _buffered_fd.get(); }

  /// The O_DIRECT fd cuFile was registered against, or -1 when none was opened.
  /// Reads through it are cuFile's and the aligned host reads'.
  [[nodiscard]] int direct_fd() const noexcept { return _direct_fd.get(); }

  /// The handle to pass to @c cuFileRead; null when the GDS fast path is
  /// unavailable for this file.
  [[nodiscard]] CUfileHandle_t cufile_handle() const noexcept { return _cufile_handle; }

  [[nodiscard]] bool cufile_registered() const noexcept { return _cufile_handle != nullptr; }

 private:
  std::string _path;
  file_descriptor _buffered_fd;
  file_descriptor _direct_fd;
  CUfileHandle_t _cufile_handle{nullptr};
  std::size_t _file_size{0};
  std::optional<object_identity> _identity;
};

// ---------------------------------------------------------------------------
// cufile_reactor
// ---------------------------------------------------------------------------

/**
 * @brief Where a cuFile reactor runs: the GPU it mainly serves and the CPUs its
 *        worker is pinned to.
 */
struct reactor_placement {
  int device{-1};         ///< Home GPU (CUDA ordinal); -1 = serves any device.
  std::vector<int> cpus;  ///< CPUs the worker is pinned to; empty = not pinned.
};

/**
 * @brief How a cuFile reactor (or a pool, summed) served its reads.
 *
 * Counted when a read settles successfully, except @c cufile_failures and
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
  std::uint64_t direct_reads{0};        ///< Device reads cuFile served into the destination.
  std::uint64_t direct_bytes{0};        ///< Their bytes.
  std::uint64_t posix_device_reads{0};  ///< Device reads served by POSIX + host-to-device copy.
  std::uint64_t posix_device_bytes{0};  ///< Their bytes.
  std::uint64_t host_reads{0};          ///< Host-only reads.
  std::uint64_t host_bytes{0};          ///< Their bytes.
  std::uint64_t cufile_failures{0};     ///< cuFileRead errors that sent a read to POSIX.
  std::uint64_t parked_slices{0};       ///< Device slices that waited for their stream to start.

  reactor_stats& operator+=(reactor_stats const& other) noexcept
  {
    staged_reads += other.staged_reads;
    staged_bytes += other.staged_bytes;
    early_staged_reads += other.early_staged_reads;
    direct_reads += other.direct_reads;
    direct_bytes += other.direct_bytes;
    posix_device_reads += other.posix_device_reads;
    posix_device_bytes += other.posix_device_bytes;
    host_reads += other.host_reads;
    host_bytes += other.host_bytes;
    cufile_failures += other.cufile_failures;
    parked_slices += other.parked_slices;
    return *this;
  }
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
 * Host-only reads go O_DIRECT, through the scratch when not block-aligned.  The
 * reactor is read-only: it opens files @c O_RDONLY and has no write path.
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
 * stopped touching its destination, failed and cancelled siblings included.
 * cuFile and POSIX reads are synchronous on the worker; scatter and H2D copies
 * run on reactor-owned streams and are host-complete (their event completed)
 * before their slice settles; a parked or cancelled slice settles without
 * having issued anything.  The bytes of a successful read are therefore visible
 * from any stream without further CUDA synchronization; the caller owns the
 * destination memory until the future settles.
 *
 * Fragmented (cache-chunk) destinations are not served: the reactor's pinned
 * scratch is not shaped like cache chunks, so it opts out of the prefetching
 * cache (@c reactor_traits::supports_vector_host_read is false).
 *
 * @note Destinations are never registered: registering a one-shot buffer costs
 *       about as much as reading it, and its 4 KiB phase rarely matches the file
 *       offset anyway.  The staging slots are registered once and reused by the
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

  /// Shared, immutable config for a pool of reactors.  A dedicated type so a
  /// future iteration can add shared state (a registered-buffer cache, a stream
  /// registry, ...) without changing the reactor constructor's shape.
  class reactor_context {
   public:
    explicit reactor_context(config cfg) : _config(std::move(cfg)) {}

    /// @p staging_caps: registered-staging cap per device (bytes, indexed by
    /// device id, 0 = none), as a pool plan bounded it by BAR1 and cuFile's
    /// pinned-memory limit; devices past its end use the config's cap.
    reactor_context(config cfg, std::vector<std::size_t> staging_caps)
      : _config(std::move(cfg)), _staging_caps(std::move(staging_caps))
    {
    }

    [[nodiscard]] const config& cfg() const noexcept { return _config; }

    /// The registered-staging cap of @p device, in bytes (0 = none).
    [[nodiscard]] std::size_t staging_cap(int device) const noexcept
    {
      auto const index = static_cast<std::size_t>(device);
      return device >= 0 && index < _staging_caps.size() ? _staging_caps[index]
                                                         : _config.max_staging_bytes_per_device;
    }

   private:
    config _config;
    std::vector<std::size_t> _staging_caps;
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

  /// Approximate bytes not yet dispatched to a physical operation.
  [[nodiscard]] std::size_t queued_bytes() const noexcept
  {
    return _queued_bytes.load(std::memory_order_relaxed);
  }

  /// How this reactor has served its reads so far (relaxed snapshot).
  [[nodiscard]] reactor_stats stats() const noexcept;

  /// The GPU this reactor mainly serves and the CPUs its worker is pinned to.
  [[nodiscard]] reactor_placement const& placement() const noexcept { return _placement; }

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
  struct waiting_slice {
    std::shared_ptr<const io_object> obj;
    prepared_io_slice slice;
    std::shared_ptr<grouped_coordinator> coordinator;
    std::shared_ptr<cucascade::cuda::cuda_event> gate;
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

  /// Turn one slice into cuFile reads or a POSIX pread.  Never throws --
  /// errors are reported through the slice's completion and coordinator.  A
  /// device-only slice whose stream still has queued work is parked (see
  /// @ref park_if_stream_busy) unless @p stream_ready says its gate passed.
  void dispatch_slice(std::shared_ptr<const io_object> const& obj,
                      prepared_io_slice slice,
                      std::shared_ptr<grouped_coordinator> const& coordinator,
                      bool stream_ready = false) noexcept;

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

  /// The scratch, allocated on first use.
  [[nodiscard]] std::uint8_t* scratch();

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

  std::shared_ptr<reactor_context> _ctx;
  reactor_config_type _config;
  std::string _tname;
  reactor_placement _placement;

  /// Read counters (see @ref reactor_stats); written by the worker, read by any
  /// thread.
  struct counters {
    std::atomic<std::uint64_t> staged_reads{0};
    std::atomic<std::uint64_t> staged_bytes{0};
    std::atomic<std::uint64_t> early_staged_reads{0};
    std::atomic<std::uint64_t> direct_reads{0};
    std::atomic<std::uint64_t> direct_bytes{0};
    std::atomic<std::uint64_t> posix_device_reads{0};
    std::atomic<std::uint64_t> posix_device_bytes{0};
    std::atomic<std::uint64_t> host_reads{0};
    std::atomic<std::uint64_t> host_bytes{0};
    std::atomic<std::uint64_t> cufile_failures{0};
    std::atomic<std::uint64_t> parked_slices{0};
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
  /// Staging-copy streams, indexed by device id.
  std::vector<owned_stream> _copy_streams;
  /// Gated pair-copy streams, indexed by device id.
  std::vector<owned_stream> _pair_streams;
  /// Pinned (portable) staging for device-only POSIX reads and unaligned host
  /// reads, taken from a process-wide pool on first use (so a reactor that never
  /// stages holds no pinned memory) and given back to it at shutdown.
  pinned_scratch _scratch;
  /// Registered device staging, indexed by device id (null until first used).
  std::vector<std::unique_ptr<device_staging>> _staging;
  /// Staged scatter copies still waiting for their caller's stream.
  std::size_t _staged_inflight{0};
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
