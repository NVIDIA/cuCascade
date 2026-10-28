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
 * @brief Single-threaded local-file I/O reactor that speaks cuFile / GDS.
 *
 * Owns one worker thread and a two-tier request queue (see @ref io_priority).
 * A contiguous device destination of at least @ref config::gds_threshold bytes
 * is served by synchronous @c cuFileRead calls of at most
 * @ref config::max_op_bytes each, issued on the worker once the work already
 * queued on the caller's stream has finished (so a stream-ordered allocation or
 * a kernel still using the buffer is never overwritten); the read completes
 * host-observed, like every other backend.  Everything else (host reads, small
 * reads, dual host+device requests) falls back to POSIX @c pread on the
 * buffered fd, with a host-to-device copy when a device destination was also
 * requested.  A cuFile read that fails degrades to the same fallback instead of
 * failing the read.
 *
 * Host-only reads go O_DIRECT, through the scratch when not block-aligned.  The
 * reactor is read-only: it opens files @c O_RDONLY and has no write path.
 *
 * The synchronous API is deliberate (it is what NVIDIA's own consumers use):
 * @c cuFileReadAsync serialises a stream's operations, runs synchronously in
 * the submitting thread in compatibility mode, and keeps per-stream state keyed
 * by caller stream handles the reactor neither registers nor deregisters.  See
 * docs/io/gpuio-gds-practices.md (G1).
 *
 * The worker takes a high-priority request before a low one and parks an active
 * low request at a slice boundary while high requests wait.  Priority is bounded
 * (see @ref config::low_service_interval) so waiting low work is never starved.
 *
 * A device slice never blocks the worker on its caller's stream: when work is
 * still queued there, the slice is parked behind an event recorded at dispatch
 * and served once that event completed, while the worker keeps serving other
 * requests.  Staging copies then run on a reactor-owned stream per device.
 *
 * Completion: a request's future settles only once every one of its slices has
 * stopped touching its destination, failed and cancelled siblings included.
 * cuFile and POSIX reads are synchronous on the worker; H2D staging copies run
 * on a reactor-owned stream and are host-complete (their event completed)
 * before their slice settles; a parked or cancelled slice settles without
 * having issued anything.  The bytes of a successful read are therefore visible
 * from any stream without further CUDA synchronization; the caller owns the
 * destination memory until the future settles.
 *
 * Fragmented (cache-chunk) destinations are not served: the reactor's pinned
 * scratch is not shaped like cache chunks, so it opts out of the prefetching
 * cache (@c reactor_traits::supports_vector_host_read is false).
 *
 * @note The reactor registers each file (@c cuFileHandleRegister) but never a
 *       destination buffer (@c cuFileBufRegister): every @c cuFileRead gets the
 *       exact unregistered device pointer.  Per cuFile, an unregistered buffer
 *       is bounced through cuFile's own GPU buffers rather than DMA'd peer to
 *       peer; NVIDIA advises against registering a one-shot destination, and in
 *       compatibility mode registration buys nothing measurable
 *       (docs/io/gpuio-gds-practices.md).
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
    explicit reactor_context(config cfg) : _config(cfg) {}

    [[nodiscard]] const config& cfg() const noexcept { return _config; }

   private:
    config _config;
  };

  using native_handle_type   = int;
  using io_object_type       = cufile_io_object;
  using reactor_config_type  = config;
  using reactor_context_type = reactor_context;

  explicit cufile_reactor(std::shared_ptr<reactor_context> ctx,
                          std::string_view tname = "cufile_reactor");

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
  /// A device-only slice parked until the work queued on its stream before
  /// dispatch has finished (@c event fires).
  struct waiting_slice {
    std::shared_ptr<const io_object> obj;
    prepared_io_slice slice;
    std::shared_ptr<grouped_coordinator> coordinator;
    cucascade::cuda::cuda_event event;
  };

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
  /// park the slice in @ref _waiting; returns whether it was parked.
  /// @throws cucascade::cuda_error when the stream cannot be queried / recorded.
  [[nodiscard]] bool park_if_stream_busy(std::shared_ptr<const io_object> const& obj,
                                         prepared_io_slice& slice,
                                         std::shared_ptr<grouped_coordinator> const& coordinator);

  /// The reactor-owned stream of @p device_id for staging copies, created on
  /// first use.  The calling thread's current device must be @p device_id.
  [[nodiscard]] cudaStream_t copy_stream(int device_id);

  /// Read @p request's range into the slice's device buffer with synchronous
  /// @c cuFileRead calls, after the slice's stream has drained the work queued
  /// before this call.  Returns true when @p request was settled (success or a
  /// definite error), false when cuFile failed and the caller must take the
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
  /// in-flight H2D copy whose CUDA event has completed.
  void poll_completions() noexcept;

  /// Synchronize and settle every outstanding copy with @p terminal_error, so no
  /// coordinator promise is stranded at shutdown.
  void drain_inflight(grouped_coordinator::error_type const& terminal_error) noexcept;

  /// Settle every parked slice with @p error; nothing was issued for them.
  void cancel_waiting(grouped_coordinator::error_type const& error) noexcept;

  [[nodiscard]] std::size_t inflight_count() const noexcept
  {
    return _inflight_copy.size() + _waiting.size();
  }

  std::shared_ptr<reactor_context> _ctx;
  reactor_config_type _config;
  std::string _tname;

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
  /// Pinned (portable) staging for device-only POSIX reads and unaligned host
  /// reads, taken from a process-wide pool on first use (so a reactor that never
  /// stages holds no pinned memory) and given back to it at shutdown.
  pinned_scratch _scratch;
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
