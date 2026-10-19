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
 * Owns two @c file_descriptor s -- a buffered O_RDONLY fd for POSIX fallback
 * reads and, when O_DIRECT is enabled and supported, an O_DIRECT fd registered
 * with cuFile via @c cuFileHandleRegister for GDS DMA.  The registered handle
 * lives for the object's lifetime; the destructor deregisters it before the fds
 * close.  All I/O is performed by the reactor -- this class is a passive bag of
 * native handles.
 */
class cufile_io_object final : public io_object {
 public:
  /// @p cufile_handle is null when the file was not registered with cuFile; the
  /// reactor then serves every read of it through the POSIX fallback.
  cufile_io_object(std::string path,
                   file_descriptor buffered_fd,
                   file_descriptor direct_fd,
                   CUfileHandle_t cufile_handle,
                   std::size_t file_size);

  ~cufile_io_object() override;

  cufile_io_object(cufile_io_object const&)            = delete;
  cufile_io_object& operator=(cufile_io_object const&) = delete;

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return _file_size; }

  /// The buffered fd (page cache).  Always valid.
  [[nodiscard]] int buffered_fd() const noexcept { return _buffered_fd.get(); }

  /// The O_DIRECT fd cuFile was registered against, or -1 when none was opened.
  /// The reactor never reads through it directly -- cuFile does.
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
 * Fragmented (cache-chunk) destinations are not served: the reactor stages
 * nothing through pinned host memory, so it opts out of the prefetching cache
 * (@c reactor_traits::supports_vector_host_read is false).
 *
 * @note The reactor does not register device buffers with cuFile.  Per cuFile,
 *       an unregistered buffer is bounced through cuFile's own GPU buffers
 *       rather than DMA'd peer to peer; NVIDIA advises against registering a
 *       one-shot destination, and in compatibility mode registration buys
 *       nothing measurable (docs/io/gpuio-gds-practices.md, G8).
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

  /// Zero: this reactor stages nothing through pinned host memory (the GDS fast
  /// path DMAs direct to device), so there is no chunk size for a cache to match.
  /// @c ioctx::initialize_cache accepts 0 as "backend does not stage".
  [[nodiscard]] std::size_t staging_block_size() const noexcept { return 0; }

  /// Open the process-wide cuFile driver (once) and launch the worker thread.
  /// Idempotent: a second call while the worker is running is a no-op.
  void start();

  /// Present for reactor-concept symmetry with the uring reactor; does nothing.
  void interrupt() noexcept;

  /// Close admission, request stop, join the worker, and cancel anything that
  /// never made it out of the queue.  Idempotent.
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

  /// Same alignment / coalescing policy as the uring reactor: outward to page
  /// boundaries, then merge overlapping / adjacent ranges.  @p alignment smaller
  /// than @c IO_BLOCK_SIZE is ignored in favour of @c IO_BLOCK_SIZE.
  [[nodiscard]] static std::vector<byte_range> align_and_coalesce(
    std::span<byte_range const> ranges,
    std::optional<std::size_t> alignment = std::nullopt) noexcept;

 private:
  /// One record per in-flight H2D copy issued for a read that took the POSIX
  /// fallback and also had a device destination.
  struct inflight_copy {
    std::unique_ptr<io_op_request> request;
    cucascade::cuda::cuda_event event;
    std::size_t bytes{0};
  };

  void worker_loop(std::stop_token const& stop_token);

  /// Turn one slice into cuFile reads or a POSIX pread.  Never throws --
  /// errors are reported through the slice's completion and coordinator.
  void dispatch_slice(std::shared_ptr<const io_object> const& obj,
                      prepared_io_slice slice,
                      std::shared_ptr<grouped_coordinator> const& coordinator) noexcept;

  /// Read @p request's range into the slice's device buffer with synchronous
  /// @c cuFileRead calls, after the slice's stream has drained the work queued
  /// before this call.  Returns true when @p request was settled (success or a
  /// definite error), false when cuFile failed and the caller must take the
  /// POSIX path for the whole range.
  [[nodiscard]] bool read_cufile(cufile_io_object const& file,
                                 prepared_io_slice const& slice,
                                 std::unique_ptr<io_op_request>& request);

  /// POSIX pread into the slice's host buffer, then the H2D copy when a device
  /// destination was asked for too; a device-only slice goes to
  /// @ref read_device_staged.
  void dispatch_posix(cufile_io_object const& file,
                      prepared_io_slice const& slice,
                      std::unique_ptr<io_op_request> request);

  /// POSIX pread of a device-only slice through the reactor's reusable pinned
  /// scratch, in pieces of @ref scratch_slot_bytes whose H2D copies (on the
  /// slice's stream) overlap the next read.  Settles @p request before returning.
  void read_device_staged(cufile_io_object const& file,
                          prepared_io_slice const& slice,
                          io_op_request& request);

  /// Settle every in-flight H2D copy whose CUDA event has completed.
  void poll_completions() noexcept;

  /// Synchronize and settle every outstanding copy with @p terminal_error, so no
  /// coordinator promise is stranded at shutdown.
  void drain_inflight(grouped_coordinator::error_type const& terminal_error) noexcept;

  [[nodiscard]] std::size_t inflight_count() const noexcept { return _inflight_copy.size(); }

  std::shared_ptr<reactor_context> _ctx;
  reactor_config_type _config;
  std::string _tname;

  std::stop_source _stop_source;
  std::jthread _worker;
  tiered_blocking_queue<std::unique_ptr<grouped_io_request>> _requests;
  mutable std::mutex _enqueue_mutex;
  std::atomic<std::size_t> _queued_bytes{0};
  std::atomic<bool> _accepting{false};

  /// Frees the pinned scratch.
  struct pinned_deleter {
    void operator()(std::uint8_t* memory) const noexcept { std::ignore = cudaFreeHost(memory); }
  };
  using pinned_scratch = std::unique_ptr<std::uint8_t, pinned_deleter>;

  /// Slots of the device-fallback scratch: two, so a read overlaps a copy.
  static constexpr std::size_t scratch_slots = 2;
  /// Bytes per scratch slot (the fallback's piece size).
  static constexpr std::size_t scratch_slot_bytes = 8UL << 20;

  // Worker-only state; touched exclusively from the reactor thread.
  std::vector<inflight_copy> _inflight_copy;
  /// Pinned (portable) staging for device-only POSIX reads, allocated on first use
  /// so a reactor that never takes that path holds no pinned memory.
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
