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

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/cache/types.hpp>
#include <cucascade/io/concurrent_queue.hpp>
#include <cucascade/io/details/slot_pool.hpp>
#include <cucascade/io/reactor_placement.hpp>
#include <cucascade/io/tiered_queue.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/io/uring/config.hpp>
#include <cucascade/io/uring/types.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <cudf/io/text/byte_range_info.hpp>

#include <cuda_runtime.h>

#include <liburing.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace cucascade::io::uring {

// ---------------------------------------------------------------------------
// local_io_object
// ---------------------------------------------------------------------------

/**
 * @brief Concrete @c io_object backed by a filesystem path.
 *
 * Passive bag of native handles. The buffered fd and, when the filesystem
 * supports it, an optional @c O_DIRECT fd are produced by
 * @c uring_reactor::create_io_object (read-only, @c O_RDONLY) or
 * @c uring_reactor::create_io_object_for_write (@c O_RDWR). This class does no
 * I/O of its own.  A writable object's size grows as writes complete (see
 * @c io_object::note_written).
 *
 * A read-only object carries the @c local_stat identity of its buffered fd
 * (captured by the reactor, or from the fd here) and takes its size from it;
 * when the fd cannot be stat'ed, and for every writable object, the identity
 * is the base class's open-local one.
 */
class local_io_object : public io_object {
 public:
  /// A read-only object.  @p captured is the identity the reactor took from
  /// @p fd; when it is not a @c local_stat identity, it is captured from @p fd
  /// here.
  local_io_object(std::string path,
                  file_descriptor fd,
                  file_descriptor fd_direct,
                  size_t file_size,
                  std::string_view hash    = "",
                  object_identity captured = {})
    : local_io_object(std::move(path),
                      std::move(fd),
                      std::move(fd_direct),
                      file_size,
                      hash,
                      std::move(captured),
                      io_access::read_only)
  {
  }

  /// As above with an explicit @p access: @c io_access::read_write makes the
  /// object accept writes (its fds must then be open for writing).
  local_io_object(std::string path,
                  file_descriptor fd,
                  file_descriptor fd_direct,
                  size_t file_size,
                  io_access access)
    : local_io_object(
        std::move(path), std::move(fd), std::move(fd_direct), file_size, "", {}, access)
  {
  }

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] size_t size() const noexcept override
  {
    return _file_size.load(std::memory_order_acquire);
  }

  [[nodiscard]] object_identity identity() const override
  {
    return _has_identity ? _identity : io_object::identity();
  }

  [[nodiscard]] bool writable() const noexcept override { return _access == io_access::read_write; }

  void note_written(std::size_t end) const noexcept override
  {
    if (writable()) atomic_store_max(_file_size, end);
  }

  [[nodiscard]] int fd() const noexcept { return _fd.get(); }
  [[nodiscard]] int fd_direct() const noexcept { return _fd_direct.get(); }

  // ---- templated_ioctx / io_object_c requirements -----------------------
  [[nodiscard]] int buffered_handle() const noexcept { return _fd.get(); }
  [[nodiscard]] int odirect_handle() const noexcept { return _fd_direct.get(); }

 private:
  local_io_object(std::string path,
                  file_descriptor fd,
                  file_descriptor fd_direct,
                  size_t file_size,
                  std::string_view hash,
                  object_identity captured,
                  io_access access)
    : _path(std::move(path)),
      _hash(hash.empty() ? std::string_view{_path} : hash),
      _fd(std::move(fd)),
      _fd_direct(std::move(fd_direct)),
      _file_size(file_size),
      _access(access)
  {
    // A writable object's bytes change as it is written: it never carries a
    // reusable identity, and keeps the size it was opened with.  Tested on the
    // access mode itself rather than the virtual writable(), which a subclass
    // may override.
    if (_access == io_access::read_write) { return; }
    _has_identity = captured.kind == identity_kind::local_stat;
    if (_has_identity) {
      _identity = std::move(captured);
    } else {
      _has_identity = capture_local_identity(_fd.get(), _identity);
    }
    if (_has_identity) { _file_size.store(_identity.size, std::memory_order_relaxed); }
  }

  std::string _path;
  std::string _hash;
  file_descriptor _fd;
  file_descriptor _fd_direct;
  mutable std::atomic<size_t> _file_size{0};
  io_access _access{io_access::read_only};
  object_identity _identity;
  bool _has_identity{false};
};

// ---------------------------------------------------------------------------
// uring_reactor
// ---------------------------------------------------------------------------

/**
 * @brief Single-threaded local-file I/O reactor.
 *
 * Owns one @c io_uring, one worker thread, a fixed pool of pinned staging
 * blocks, and a two-tier request queue (see @ref io_priority). The worker
 * takes a high-priority request before a low one, and parks an active low
 * request at a slice boundary while high requests are waiting, so a demand
 * read does not wait behind a whole-split readahead.  Priority is bounded,
 * not strict: after @c config::low_service_interval high requests have
 * started while low work waited, the low tier gets one turn of at least one
 * quantum (@c config::slices_per_pass slices) even if high work is waiting, so
 * a demand read blocked on an in-flight prefetch is not starved by a stream of
 * other high requests (0 restores strict priority). Physical operations use
 * O_DIRECT only when the worker determines that the complete transfer is
 * compatible. Models the reactor concept consumed by @c templated_ioctx.
 *
 * Writes (@c io_write_reactor_c): each write slice is split at the write
 * granule into an unaligned head, an aligned body and an unaligned tail.  The
 * body goes O_DIRECT -- straight from a host source whose address is congruent
 * with its file offset, otherwise through the pinned staging blocks (a D2H
 * copy on the caller's stream for device sources, a memcpy for host ones);
 * head and tail are written through the page cache.  Short writes are
 * resubmitted; a flush is an @c IORING_OP_FSYNC with @c IORING_FSYNC_DATASYNC.
 */
class uring_reactor {
 public:
  /// A read here is a syscall against page cache or NVMe, cheap enough that
  /// batching buys little -- and demanding the whole range set up front forces
  /// the caller to materialise ranges it might never read.
  static constexpr bool prefers_bulk_io = false;

  /// Shared, immutable services for a pool of reactors.  One instance is built
  /// by @c uring_ioctx and shared (via shared_ptr) across every reactor in the
  /// pool — the natural home for things shared rather than per-reactor: the
  /// pinned bounce-staging resource (and, in future, GDS-registered buffers, a
  /// shared submission/poll thread, etc.).  Carries the primitive @c config too.
  class reactor_context {
   public:
    reactor_context(config cfg, cucascade::memory::fixed_size_host_memory_resource* mr)
      : _config(cfg), _mr(mr)
    {
    }

    [[nodiscard]] const config& cfg() const noexcept { return _config; }
    [[nodiscard]] cucascade::memory::fixed_size_host_memory_resource* host_memory_resource()
      const noexcept
    {
      return _mr;
    }

   private:
    config _config;
    cucascade::memory::fixed_size_host_memory_resource* _mr{nullptr};
  };

  using native_handle_type   = int;
  using io_object_type       = local_io_object;
  using reactor_config_type  = config;
  using reactor_context_type = reactor_context;

  /// Bounce slots are allocated by @c start() from @p placement's staging
  /// resource, or the context's host_memory_resource when it has none (one of
  /// the two must be non-null); their size is taken from its
  /// @c get_block_size() and their count from a fixed 64 MiB staging budget.  The
  /// reactor keeps the @c multiple_blocks_allocation alive until it is destroyed —
  /// blocks return to the resource then — and holds the shared @p ctx so it (and
  /// the resource) outlive the reactor.  The worker pins itself to
  /// @p placement's CPUs before it sets up its ring.
  explicit uring_reactor(std::shared_ptr<reactor_context> ctx,
                         std::string_view tname      = "uring_reactor",
                         reactor_placement placement = {});

  ~uring_reactor();

  uring_reactor(uring_reactor const&)            = delete;
  uring_reactor& operator=(uring_reactor const&) = delete;

  /// The reactor's effective config (copied from its context at construction).
  /// templated_ioctx reads its own _config from here so the config lives in one
  /// place — the context — rather than being passed in separately.
  [[nodiscard]] const reactor_config_type& get_config() const noexcept { return _config; }

  /// Bounce-slot size, read from the staging resource (0 when there is none)
  /// rather than from @c _bounce_slot_size, which start() derives from the same
  /// resource: the default ioctx initializes its cache
  /// before it starts its reactors, so the check below must not depend on
  /// start() having run.  INVARIANT: when a prefetching cache is present this
  /// MUST equal its chunk size — a fragmented fill's extent is computed as
  /// @c cache::fill_span(fill, chunk->offset, this value), so a larger staging
  /// block writes past the end of the pinned chunk and a smaller one marks a
  /// chunk cached while only part of it was read.  Checked once in
  /// @c ioctx::initialize_cache, which is the only place both sizes are known.
  [[nodiscard]] std::size_t staging_block_size() const noexcept
  {
    auto const* resource = staging_resource();
    return resource == nullptr ? std::size_t{0} : resource->get_block_size();
  }

  /// Where this reactor lives (home GPU, NUMA node, staging, worker CPUs).
  [[nodiscard]] reactor_placement const& placement() const noexcept { return _placement; }

  /// Allocate the pinned bounce slots and launch the worker thread.  Split out
  /// of the constructor so a reactor can be built cheaply (it only copies its
  /// config) and parked until it is actually needed — see @c ioctx::start.
  /// Idempotent: a second call (while the worker is already running) is a no-op.
  ///
  /// Waits until the worker has set up its io_uring.  When that fails (e.g.
  /// @c EPERM where a seccomp profile blocks io_uring: Docker's default since
  /// 25, gVisor, @c kernel.io_uring_disabled), the worker exits, the error is
  /// logged, and start() rethrows it (a @c std::system_error with the errno).
  /// Requests enqueued before or after the failure fail with that same error,
  /// not with @c operation_canceled.  start() may be called again to retry.
  void start();

  void interrupt();
  void shutdown();

  /// Synchronous buffered host read (pread on @p fd).  Blocks the caller.
  size_t host_read(const io_object_type& file, size_t offset, size_t size, uint8_t* dst);

  /// Synchronous write of @p size bytes of host memory @p src at @p offset on
  /// the calling thread, with the same head / body / tail placement as the
  /// asynchronous path (the body O_DIRECT only when @p src is congruent; no
  /// staging).  Raises @p file's size.  @return @p size.
  /// @throws std::system_error (@c EBADF for a read-only object, or the errno of
  ///         a failed write) or std::invalid_argument for a null @p src.
  std::size_t host_write(const io_object_type& file,
                         std::size_t offset,
                         std::size_t size,
                         std::uint8_t const* src);

  /// @c fdatasync the file on the calling thread.  @throws std::system_error.
  static void flush(const io_object_type& file);

  void enqueue(std::unique_ptr<grouped_io_request> request) noexcept;

  /// Approximate bytes not yet converted from logical slices to physical ops,
  /// plus the bytes of staged writes whose device-to-host copy is still pending
  /// on the caller's stream (they hold staging blocks without any I/O in flight).
  [[nodiscard]] std::size_t queued_bytes() const noexcept
  {
    return _queued_bytes.load(std::memory_order_relaxed);
  }

  /// Log2 histogram of queue delays: bucket b counts delays d (microseconds) with
  /// std::bit_width(d) == b, i.e. [2^(b-1), 2^b) us; the last bucket is open-ended.
  static constexpr std::size_t queue_delay_buckets = 26;

  /// One io_priority tier's queue-delay window (see @ref gauges::queue_delay).
  struct queue_delay_stats {
    std::uint64_t count{0};
    std::uint64_t sum_ns{0};
    std::uint64_t max_ns{0};
    std::array<std::uint32_t, queue_delay_buckets> histogram{};
  };

  /// Diagnostic view of the worker's depth and queueing.  The worker publishes
  /// these with relaxed stores once per loop pass (right before it waits for a
  /// completion), so every field is approximate and may be one pass stale.
  struct gauges {
    std::uint32_t inflight_ops{0};      ///< physical ops owned by the kernel
    std::uint32_t max_inflight_ops{0};  ///< peak since the previous take_gauges()
    std::uint32_t pending_ops{0};       ///< planned ops waiting for a staging slot
    /// Slices of the active request not yet expanded into physical ops.
    std::uint32_t active_remaining_slices{0};
    std::size_t queued_requests{0};      ///< requests queued behind the active one
    std::size_t queued_low_requests{0};  ///< of which on the low-priority tier
    std::size_t queued_bytes{0};         ///< see @ref queued_bytes
    std::uint64_t requests_started{0};   ///< cumulative requests taken off the queue
    std::uint64_t bytes_submitted{0};    ///< cumulative physical bytes submitted
    /// Cumulative times an active low request was parked for a high one.
    std::uint64_t preemptions{0};
    /// Cumulative times the low tier was served ahead of waiting high work by
    /// the bounded-service rule (see @c config::low_service_interval).
    std::uint64_t low_service_grants{0};
    /// Queue delay (enqueue -> first slice expanded) of requests whose first slice
    /// was expanded since the previous take_gauges(), per priority tier:
    /// [0] = @c io_priority::high, [1] = @c io_priority::low.
    queue_delay_stats queue_delay[2]{};
  };

  /// Snapshot the gauges and restart the @c max_inflight_ops window.
  [[nodiscard]] gauges take_gauges() noexcept;

  /// Whether @p path can be served by this reactor.  Local-disk only:
  /// returns true iff the path refers to an existing, accessible file.
  [[nodiscard]] static bool supports(std::string_view path);

  /// Open the buffered fd and opportunistically open an O_DIRECT fd for
  /// @p path. The buffered handle is always required; an unsupported direct
  /// handle simply makes worker-planned operations use buffered I/O.  The
  /// O_DIRECT fd reopens the buffered one through @c /proc/self/fd (the path
  /// only where procfs is not mounted), and the object carries the buffered
  /// fd's @c local_stat identity and size.
  /// @throws descriptor_mismatch_error when, on two attempts, the two fds do
  ///         not name the same file (the path is being replaced concurrently),
  ///         or when the O_DIRECT fd cannot be stat'ed.
  static std::unique_ptr<io_object_type> create_io_object(std::string path);

  /// Open (per @p options.mode, creating with @p options.permissions) a buffered
  /// @c O_RDWR fd and opportunistically an @c O_RDWR | @c O_DIRECT fd for
  /// @p path, preallocate @p options.size_hint bytes best-effort, and return a
  /// writable object sized at its current length.
  /// @throws std::system_error with the errno of the failed open.
  static std::unique_ptr<io_object_type> create_io_object_for_write(std::string path,
                                                                    write_open_options options);

  /// fstat the open fd to get the file's current size.
  static size_t size(int native_handle);

 public:
  /// O_DIRECT requires 4 KiB alignment of both file offset and length.
  static byte_range align_to_physical(byte_range logical, size_t file_size);

  /// Align every input range's ends outward to the effective alignment, then
  /// coalesce overlapping or adjacent results into a minimal set of aligned,
  /// non-overlapping ranges (sorted by offset).
  ///
  /// The reactor reads through O_DIRECT, so @c IO_BLOCK_SIZE is the minimum
  /// viable alignment and is used when @p alignment is unset.  A caller-supplied
  /// alignment is honored only when it is at least @c IO_BLOCK_SIZE; a smaller
  /// value is ignored in favor of the reactor's own alignment.
  static std::vector<byte_range> align_and_coalesce(
    std::span<const byte_range> ranges, std::optional<size_t> alignment = std::nullopt) noexcept;

 private:
  /// @p ready is fulfilled once the ring is set up, or failed with the setup error.
  void worker_loop(const std::stop_token& stop_token, std::promise<void>& ready);

  /// The placement's staging resource, else the context's (null when neither).
  [[nodiscard]] cucascade::memory::fixed_size_host_memory_resource* staging_resource()
    const noexcept
  {
    if (_placement.staging_mr != nullptr) return _placement.staging_mr;
    return _ctx == nullptr ? nullptr : _ctx->host_memory_resource();
  }

  // Shared services + tunables for the whole reactor pool, kept alive for this
  // reactor's lifetime (so the bounce-staging resource outlives _bounce_storage).
  std::shared_ptr<reactor_context> _ctx;

  reactor_config_type _config;
  // Thread name prefix captured at construction; applied to the worker in start().
  std::string _tname;
  // Home GPU / NUMA node, staging override and worker CPUs.
  reactor_placement _placement;
  // Keeps the bounce-slot blocks alive for the reactor's lifetime.  The
  // multiple_blocks_allocation destructor returns the blocks to the upstream
  // resource when the reactor is destroyed.
  cucascade::memory::fixed_multiple_blocks_allocation _bounce_storage;
  std::size_t _bounce_slot_size;
  std::stop_source _stop_source;
  std::jthread _worker;
  tiered_blocking_queue<std::unique_ptr<grouped_io_request>> _requests;
  mutable std::mutex _enqueue_mutex;
  std::atomic<std::size_t> _queued_bytes{0};
  std::atomic<bool> _accepting{false};
  /// Why the reactor stopped accepting work after its worker failed (ring setup
  /// or a fatal worker error); rejected requests fail with it.  Null after a
  /// clean shutdown, which rejects with @c operation_canceled.  Guarded by
  /// @c _enqueue_mutex.
  std::exception_ptr _rejection_error;

  // Gauges (see @ref gauges): written only by the worker, read by take_gauges().
  std::atomic<std::uint32_t> _gauge_inflight{0};
  std::atomic<std::uint32_t> _gauge_max_inflight{0};
  std::atomic<std::uint32_t> _gauge_pending{0};
  std::atomic<std::uint32_t> _gauge_active_slices{0};
  std::atomic<std::uint64_t> _gauge_requests_started{0};
  std::atomic<std::uint64_t> _gauge_bytes_submitted{0};
  std::atomic<std::uint64_t> _gauge_preemptions{0};
  std::atomic<std::uint64_t> _gauge_low_service_grants{0};

  /// Worker-side accumulators behind gauges::queue_delay; take_gauges() swaps them out.
  struct queue_delay_atomics {
    std::atomic<std::uint64_t> count{0};
    std::atomic<std::uint64_t> sum_ns{0};
    std::atomic<std::uint64_t> max_ns{0};
    std::array<std::atomic<std::uint32_t>, queue_delay_buckets> histogram{};
  };
  queue_delay_atomics _gauge_queue_delay[2];
  /// Record one request's queue delay (worker thread only).
  void record_queue_delay(grouped_io_request const& request) noexcept;

  /// config::slices_per_pass with 0 (no cap) mapped to SIZE_MAX; read once at
  /// construction.
  std::size_t _slices_per_pass{1};
  /// Slices a low request granted service must expand before it can be parked
  /// again: _slices_per_pass, capped at max_slices_per_pass.
  std::size_t _low_service_quantum{1};
};

}  // namespace cucascade::io::uring
