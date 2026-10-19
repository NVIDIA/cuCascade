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
#include <cucascade/io/templated_ioctx.hpp>
#include <cucascade/io/tiered_queue.hpp>
#include <cucascade/io/types.hpp>

#include <kvikio/compat_mode.hpp>
#include <kvikio/shim/cufile_h_wrapper.hpp>

#include <cuda_runtime.h>

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
 * Owns two @c file_descriptor s -- a buffered fd for POSIX I/O and, when
 * O_DIRECT is enabled and supported, an O_DIRECT fd registered with cuFile via
 * @c cuFileHandleRegister for GDS DMA.  Both are @c O_RDONLY for an object opened
 * for reading and @c O_RDWR for one opened for writing (@ref writable).  The
 * registered handle lives for the object's lifetime; the destructor deregisters
 * it before the fds close.  All I/O is performed by the reactor -- this class is
 * a passive bag of native handles plus the file size, which writes through a
 * writable object raise (@ref note_written).
 */
class cufile_io_object final : public io_object {
 public:
  /// @p cufile_handle is null when the file was not registered with cuFile; the
  /// reactor then serves every read and write of it through POSIX.
  cufile_io_object(std::string path,
                   file_descriptor buffered_fd,
                   file_descriptor direct_fd,
                   CUfileHandle_t cufile_handle,
                   std::size_t file_size,
                   io_access access = io_access::read_only);

  ~cufile_io_object() override;

  cufile_io_object(cufile_io_object const&)            = delete;
  cufile_io_object& operator=(cufile_io_object const&) = delete;

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override
  {
    return _file_size.load(std::memory_order_acquire);
  }

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

  [[nodiscard]] bool cufile_registered() const noexcept { return _cufile_handle != nullptr; }

 private:
  std::string _path;
  file_descriptor _buffered_fd;
  file_descriptor _direct_fd;
  CUfileHandle_t _cufile_handle{nullptr};
  mutable std::atomic<std::size_t> _file_size{0};
  io_access _access{io_access::read_only};
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
 * Writes (on objects from @ref create_io_object_for_write) are split at the
 * write granule like the uring reactor's: the granule-aligned body of a device
 * source of at least @ref config::gds_threshold bytes goes to synchronous
 * @c cuFileWrite calls (after the work queued on the source's stream), while
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
 * flush is an @c fdatasync.  On a platform where cuFile
 * runs only in compatibility mode (@c probe_platform().compat_only) aligned
 * device bodies are staged too: cuFile then bounces through host memory itself,
 * and its 1 MiB writes measured about half as fast as the staged path when they
 * extend a file (docs/io/gpuio-gds-practices.md, section 6).
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
 * Fragmented (cache-chunk) destinations are not served: the reactor's pinned
 * scratch is not shaped like cache chunks, so it opts out of the prefetching
 * cache (@c reactor_traits::supports_vector_host_read is false).
 *
 * @note By default the reactor does not register device buffers with cuFile
 *       (see @ref config::register_min_bytes).  Per cuFile, an unregistered
 *       buffer is bounced through cuFile's own GPU buffers rather than DMA'd
 *       peer to peer; NVIDIA advises against registering a one-shot destination,
 *       and in compatibility mode registration buys nothing measurable
 *       (docs/io/gpuio-gds-practices.md, G8).
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
  /// caller; used by @c host_read_io.
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

  /// Open @p path with default options (O_DIRECT, kvikIO's default compat mode).
  /// See the three-argument overload.
  static std::unique_ptr<cufile_io_object> create_io_object(std::string path);

  /**
   * @brief Open the buffered fd and, when @p use_odirect, an O_DIRECT fd, and
   *        register the latter with cuFile.
   *
   * @p compat_mode (default: kvikIO's @c defaults::compat_mode()) selects what a
   * file that cannot be registered means: @c ON never registers, @c AUTO falls
   * back to POSIX for that file, @c OFF treats it as an error.
   *
   * @throws std::runtime_error / std::system_error when the file cannot be
   *         opened or stat'ed, or when @p compat_mode is @c OFF and cuFile
   *         registration fails.
   */
  static std::unique_ptr<cufile_io_object> create_io_object(
    std::string path, bool use_odirect, std::optional<kvikio::CompatMode> compat_mode);

  // -- Writes (docs/io/write-api-design.md section 6) ------------------------

  /// Open @p path for writing with default options (O_DIRECT, kvikIO's default
  /// compat mode).  See the four-argument overload.
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
    std::optional<kvikio::CompatMode> compat_mode);

  /// Synchronous host write on the calling thread: the write-granule-aligned
  /// body goes O_DIRECT when the source is block-aligned, head and tail through
  /// the page cache.  Raises the object's size; returns @p size.
  /// @throws std::system_error (@c EBADF for an object not opened for writing).
  std::size_t host_write(cufile_io_object const& file,
                         std::size_t offset,
                         std::size_t size,
                         std::uint8_t const* source);

  /// @c fdatasync the file on the calling thread.  @throws std::system_error.
  static void flush(cufile_io_object const& file);

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

  /// Serve a write slice: host sources by @c pwrite, device sources by
  /// @c cuFileWrite for the aligned body and staged @c pwrite for the rest.
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
  /// Pinned (portable) staging for device-only POSIX reads and writes, taken from
  /// a process-wide pool on first use (so a reactor that never stages holds no
  /// pinned memory) and given back at shutdown.
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
