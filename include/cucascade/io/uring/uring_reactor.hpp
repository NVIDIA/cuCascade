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
#include <cucascade/io/details/request_hub.hpp>
#include <cucascade/io/details/runner_registry.hpp>
#include <cucascade/io/details/slot_pool.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/io/uring/config.hpp>
#include <cucascade/io/uring/types.hpp>
#include <cucascade/io/uring/uring_engine.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <cudf/io/text/byte_range_info.hpp>

#include <cuda_runtime.h>

#include <liburing.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
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
 * @c uring_reactor::create_io_object (both @c O_RDONLY; size frozen at open)
 * or @c uring_reactor::create_io_object_for_write (both @c O_RDWR).
 *
 * A *writable* object accepts writes until @ref mark_committed; its
 * @ref size is the live high-water mark (size at open, raised by every
 * completed write).  Reads of a writable object are allowed.  This class does
 * no I/O of its own.
 */
class local_io_object : public io_object {
 public:
  local_io_object(std::string path,
                  file_descriptor fd,
                  file_descriptor fd_direct,
                  size_t file_size,
                  std::string_view hash = "",
                  bool writable         = false)
    : _path(std::move(path)),
      _fd(std::move(fd)),
      _fd_direct(std::move(fd_direct)),
      _file_size(file_size),
      _writable(writable)
  {
    if (hash.empty()) {
      _hash = _path;
    } else {
      _hash = hash;
    }
  }

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] size_t size() const noexcept override
  {
    return _file_size.load(std::memory_order_acquire);
  }

  [[nodiscard]] int fd() const noexcept { return _fd.get(); }
  [[nodiscard]] int fd_direct() const noexcept { return _fd_direct.get(); }

  // ---- templated_ioctx / io_object_c requirements -----------------------
  [[nodiscard]] int buffered_handle() const noexcept { return _fd.get(); }
  [[nodiscard]] int odirect_handle() const noexcept { return _fd_direct.get(); }

  // ---- write support -----------------------------------------------------

  /// Whether writes are accepted (opened for write and not yet committed).
  [[nodiscard]] bool is_writable() const noexcept
  {
    return _writable.load(std::memory_order_acquire);
  }

  /// Raise the visible size to @p end (a write of [.., end) completed).
  void note_written(std::size_t end) const noexcept
  {
    auto current = _file_size.load(std::memory_order_relaxed);
    while (current < end && !_file_size.compare_exchange_weak(
                              current, end, std::memory_order_acq_rel, std::memory_order_relaxed)) {
    }
  }

  /// Flip the object to read-only.  @return Whether it was writable before.
  [[nodiscard]] bool mark_committed() const noexcept
  {
    return _writable.exchange(false, std::memory_order_acq_rel);
  }

 private:
  std::string _path;
  std::string _hash;
  file_descriptor _fd;
  file_descriptor _fd_direct;
  mutable std::atomic<size_t> _file_size{0};
  mutable std::atomic<bool> _writable{false};
};

// ---------------------------------------------------------------------------
// uring_reactor
// ---------------------------------------------------------------------------

/**
 * @brief Local-file I/O dispatcher shared by every runner of a @c uring_ioctx.
 *
 * Holds the shared, immutable configuration and @ref reactor_context, the
 * @c detail::request_hub (admission, shared request queue, runner registry)
 * and the synchronous caller-thread helpers.  The event loop itself lives in
 * @ref uring_engine: one per runner thread, built by @ref make_engine, owning
 * its own ring and pinned staging.  Physical operations use O_DIRECT only when
 * the engine determines that the complete transfer is compatible.  Models
 * @c io_writable_reactor_c: host and device-source writes, flush and commit
 * run on the engines; @ref host_write is a synchronous caller-thread helper.
 *
 * Thread-safety: every member may be called concurrently.
 */
class uring_reactor {
 public:
  /// A read here is a syscall against page cache or NVMe, cheap enough that
  /// batching buys little -- and demanding the whole range set up front forces
  /// the caller to materialise ranges it might never read.
  static constexpr bool prefers_bulk_io = false;
  /// Local files accept writes (see @ref create_io_object_for_write).
  static constexpr bool supports_write = true;
  /// Device sources are staged through the engines' pinned blocks (D2H on the
  /// caller's stream, then a write SQE).
  static constexpr bool supports_device_write = true;

  /// Shared, immutable services for every runner of the context: the pinned
  /// bounce-staging resource (each engine allocates its own staging blocks
  /// from it) and the primitive @c config.
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
  using engine_type          = uring_engine;

  /// Only copies the configuration; pinned staging is allocated per engine
  /// (@ref make_engine) from the context's host memory resource, which must
  /// be non-null and outlive every engine.
  explicit uring_reactor(std::shared_ptr<reactor_context> ctx,
                         std::string_view tname = "uring_reactor");

  ~uring_reactor();

  uring_reactor(uring_reactor const&)            = delete;
  uring_reactor& operator=(uring_reactor const&) = delete;

  /// The reactor's effective config (copied from its context at construction).
  /// templated_ioctx reads its own _config from here so the config lives in one
  /// place — the context — rather than being passed in separately.
  [[nodiscard]] const reactor_config_type& get_config() const noexcept { return _config; }

  /// The shared context (configuration + pinned staging resource).
  [[nodiscard]] reactor_context const& context() const noexcept { return *_ctx; }

  /// Name given at construction (diagnostics).
  [[nodiscard]] std::string const& name() const noexcept { return _tname; }

  /// Bounce-slot size, read from the context's host resource (0 when the
  /// context has none).  INVARIANT: when a prefetching cache is present this
  /// MUST equal its chunk size — a fragmented fill's extent is computed as
  /// @c cache::fill_span(fill, chunk->offset, this value), so a larger staging
  /// block writes past the end of the pinned chunk and a smaller one marks a
  /// chunk cached while only part of it was read.  Checked once in
  /// @c ioctx::initialize_cache, which is the only place both sizes are known.
  [[nodiscard]] std::size_t staging_block_size() const noexcept
  {
    return _ctx == nullptr || _ctx->host_memory_resource() == nullptr
             ? std::size_t{0}
             : _ctx->host_memory_resource()->get_block_size();
  }

  /// Admission, shared queue and runner registry of the context.
  [[nodiscard]] io::detail::request_hub& hub() noexcept { return _hub; }
  [[nodiscard]] io::detail::request_hub const& hub() const noexcept { return _hub; }

  /// Approximate bytes not yet taken of all queued requests.
  [[nodiscard]] std::size_t queued_bytes() const noexcept { return _hub.queued_bytes(); }

  /**
   * @brief Build the event loop of the runner owning @p slot (on its thread).
   *
   * @throws std::invalid_argument if the staging block size is zero.
   * @throws std::runtime_error if the pinned staging cannot be allocated.
   * @throws std::system_error if the io_uring cannot be created.
   */
  [[nodiscard]] std::unique_ptr<uring_engine> make_engine(io::detail::runner_slot& slot);

  /// Synchronous buffered host read (pread on @p fd).  Blocks the caller.
  size_t host_read(const io_object_type& file, size_t offset, size_t size, uint8_t* dst);

  /**
   * @brief Synchronous buffered host write (pwrite loop on the buffered fd).
   *
   * Blocks the caller; works without runners.  Retries @c EINTR and short
   * writes.  With @c write_durability::data_sync the file is @c fdatasync'ed
   * afterwards (a filesystem without sync support, @c EINVAL, is ignored).
   *
   * @return @p size.
   * @throws std::invalid_argument if @p file is read-only or committed.
   * @throws std::system_error on an I/O error.
   */
  std::size_t host_write(const io_object_type& file,
                         std::size_t offset,
                         std::size_t size,
                         const std::uint8_t* source,
                         write_options options);

  /**
   * @brief Open (create / truncate) @p path for writing.
   *
   * @c create_or_truncate: @c O_RDWR|O_CREAT|O_TRUNC, @c create_or_open:
   * @c O_RDWR|O_CREAT, @c open_existing: @c O_RDWR (all @c O_CLOEXEC, mode
   * @c options.permissions), plus an opportunistic @c O_RDWR|O_DIRECT fd.
   * @c size_hint > 0 reserves space with @c fallocate(FALLOC_FL_KEEP_SIZE)
   * (best effort).
   *
   * @throws std::system_error with the @c open errno (e.g. @c ENOENT for a
   *         missing file with @c open_existing).
   */
  static std::unique_ptr<io_object_type> create_io_object_for_write(std::string path,
                                                                    write_open_options options);

  /// Whether @p path can be served by this reactor.  Local-disk only:
  /// returns true iff the path refers to an existing, accessible file.
  [[nodiscard]] static bool supports(std::string_view path);

  /// Open the buffered fd and opportunistically open an O_DIRECT fd for
  /// @p path. The buffered handle is always required; an unsupported direct
  /// handle simply makes engine-planned operations use buffered I/O.
  static std::unique_ptr<io_object_type> create_io_object(std::string path);

  /// fstat the open fd to get the file's current size.
  static size_t size(int native_handle);

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
  // Shared services + tunables for every runner, kept alive for this reactor's
  // lifetime (so the bounce-staging resource outlives every engine).
  std::shared_ptr<reactor_context> _ctx;
  reactor_config_type _config;
  std::string _tname;
  io::detail::request_hub _hub;
};

}  // namespace cucascade::io::uring
