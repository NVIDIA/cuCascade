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

#include <cucascade/io/cufile/config.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/templated_ioctx.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>

#include <cuda_runtime_api.h>
#include <sys/types.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cucascade::io::cufile {

class device_arena;

/**
 * @brief Allocate and register the arena of @p device ahead of its reactors.
 *
 * The arena is @c cudaMalloc memory outside RMM (@c config::arena_bytes()), so a
 * process that hands the rest of the GPU to a memory pool reserves it first.
 * Every reactor of @p device with the same @c config::lanes and
 * @c config::lane_bytes shares it; the GPU keeps it for the life of the process.
 */
void reserve_arena(int device, config const& cfg);

/**
 * @brief A local file opened for cuFile: the descriptors of a
 *        @c uring::local_io_object plus a cuFile handle registered on them.
 *
 * Host I/O uses the descriptors, device I/O the handle.  When libcufile, its
 * driver or the registration is unavailable the object still opens (the reason
 * is logged) and serves host I/O only: device I/O on it fails with
 * @c std::errc::operation_not_supported.
 */
class cufile_io_object final : public uring::local_io_object {
 public:
  explicit cufile_io_object(uring::local_io_object&& file);
  ~cufile_io_object() override;

  [[nodiscard]] bool cufile_registered() const noexcept { return _handle != nullptr; }
  /// The @c CUfileHandle_t, null when not registered.
  [[nodiscard]] void* cufile_handle() const noexcept { return _handle; }

 private:
  void* _handle{nullptr};
};

/**
 * @brief The cuFile (GPUDirect Storage) reactor of one GPU.
 *
 * Device I/O runs on lanes.  A lane is a thread that takes one slot of the
 * GPU's arena -- @c cudaMalloc memory registered with cuFile once -- for each
 * segment it serves, and blocks in one synchronous cuFile call at a time, so
 * the lane count is the I/O depth.  There are two sets of @c config::lanes
 * lanes, one reading and one writing, each with its own queue, so reads never
 * wait behind writes; the arena holds a read slot and a write slot of
 * @c config::lane_bytes for every lane.  The arena belongs to the GPU: the
 * reactors of several contexts on one GPU share its slots, and so its depth.
 *
 * A lane serves high-priority segments first, and a waiting low-priority one
 * after every few high ones, so neither starves.
 *
 * A device request is cut into segments of at most one slot at submission:
 *  - read: the destinations of a request that lie close together share one
 *    segment (see @c config::coalesce_gap_bytes).  A lane reads the segment's
 *    page-aligned window into its slot, then queues the copies of each
 *    destination's bytes out of the slot.  With @c config::async_copies it hands
 *    the slot and the segment to the reactor's completion thread -- which gives
 *    the slot back and settles the segment once the copies have finished -- and
 *    takes another slot for its next read at once (the arena holds
 *    @c config::read_slots_per_lane read slots per lane).
 *  - write: a lane copies the source bytes into its slot and writes the
 *    granule-aligned body from there; an unaligned head and tail are written by
 *    the host reactor.
 * A lane never touches a caller's device buffer before the work queued on the
 * caller's stream at submission has finished, and settles a segment only once
 * its bytes are in the destination (read) or in the file (write).  The streams
 * and events this takes are the lanes' own or recycled; none is created per
 * request.
 *
 * Host I/O -- host reads, host writes and flushes -- is served by a host-only
 * @c uring::uring_reactor the reactor owns.
 *
 * Models @c io_reactor_c and @c io_write_reactor_c.
 */
class cufile_reactor {
 public:
  using io_object_type      = cufile_io_object;
  using reactor_config_type = config;

  /**
   * @brief How a lane moves bytes between a file and its arena slot.
   *
   * Both calls mirror @c cuFileRead / @c cuFileWrite: they block, and return the
   * bytes moved, -1 with @c errno set, or a negated @c CUfileOpError.  The one
   * implementation makes those synchronous calls; a transport built on cuFile's
   * asynchronous or batch APIs plugs in here.
   */
  class transport {
   public:
    virtual ~transport() = default;

    virtual ssize_t read(void* handle,
                         void* arena,
                         std::size_t size,
                         std::size_t file_offset,
                         std::size_t arena_offset) = 0;

    virtual ssize_t write(void* handle,
                          void* arena,
                          std::size_t size,
                          std::size_t file_offset,
                          std::size_t arena_offset) = 0;
  };

  /// A reactor for GPU @p device whose threads are pinned to @p cpus (the CPUs
  /// of the GPU's NUMA node; empty leaves them unpinned).
  /// @throws std::invalid_argument for no lanes, or a lane size that is not a
  ///         positive multiple of the write granule.
  cufile_reactor(config cfg, int device, std::vector<int> cpus = {});
  ~cufile_reactor();

  cufile_reactor(cufile_reactor const&)            = delete;
  cufile_reactor& operator=(cufile_reactor const&) = delete;

  [[nodiscard]] config const& get_config() const noexcept { return _config; }

  /// Zero: the arena is not a cache-chunk staging area (the reactor does not
  /// serve the prefetching cache), so there is no chunk size for a cache to
  /// match.
  [[nodiscard]] std::size_t staging_block_size() const noexcept { return 0; }

  /// Open the cuFile driver, take the arena and launch the lanes.  Idempotent,
  /// and a reactor that was shut down can be started again.
  /// @throws std::runtime_error when libcufile or its driver is unavailable.
  void start();

  /// Present for the reactor concept; does nothing.
  void interrupt() noexcept {}

  /// Stop the lanes (each finishes the segment it is in) and cancel what is
  /// still queued with @c operation_canceled.  Idempotent.
  void shutdown();

  /// What the reactor has queued and what it has moved since it was built.
  struct gauges {
    int device{-1};                  ///< the reactor's GPU
    std::size_t queued_bytes{0};     ///< see @ref queued_bytes
    std::uint64_t reads{0};          ///< cuFile reads completed (one per read segment)
    std::uint64_t bytes_read{0};     ///< file bytes those reads covered, gaps included
    std::uint64_t writes{0};         ///< device writes completed (one per write segment)
    std::uint64_t bytes_written{0};  ///< file bytes those writes covered
  };

  [[nodiscard]] gauges get_gauges() const noexcept;

  std::size_t host_read(io_object_type const& file,
                        std::size_t offset,
                        std::size_t size,
                        std::uint8_t* dst)
  {
    return _host.host_read(file, offset, size, dst);
  }

  std::size_t host_write(io_object_type const& file,
                         std::size_t offset,
                         std::size_t size,
                         std::uint8_t const* src)
  {
    return _host.host_write(file, offset, size, src);
  }

  static void flush(io_object_type const& file) { uring::uring_reactor::flush(file); }

  /// Queue @p request: a request of device slices on the read or the write
  /// lanes, any other on the host reactor.
  void enqueue(std::unique_ptr<grouped_io_request> request) noexcept;

  /// Bytes queued and not yet picked up by a lane or the host reactor.
  [[nodiscard]] std::size_t queued_bytes() const noexcept
  {
    return _queued_bytes.load(std::memory_order_relaxed) + _host.queued_bytes();
  }

  /// Whether the calling thread is one of this reactor's lanes or its host
  /// reactor's worker.
  [[nodiscard]] bool on_worker_thread() const noexcept;

  /// Whether @p path names a local file (see @c uring::uring_reactor::supports).
  /// Routing between this and the uring reactor, which both claim local files,
  /// is decided by @c io_context_registry.
  [[nodiscard]] static bool supports(std::string_view path)
  {
    return uring::uring_reactor::supports(path);
  }

  static std::unique_ptr<io_object_type> create_io_object(std::string path);
  static std::unique_ptr<io_object_type> create_io_object_for_write(std::string path,
                                                                    write_open_options options);

  [[nodiscard]] static std::vector<byte_range> align_and_coalesce(
    std::span<byte_range const> ranges,
    std::optional<std::size_t> alignment = std::nullopt) noexcept
  {
    return uring::uring_reactor::align_and_coalesce(ranges, alignment);
  }

 private:
  struct part;
  struct segment;
  struct gate_pool;
  struct lane;
  struct lane_set;
  struct copy_job;
  struct completer;

  /// Cut a device request into the segments its lanes will serve.
  [[nodiscard]] std::vector<segment> plan(grouped_io_request const& request) const;
  /// The loop of lane @p self: serve the segments of @p set until it stops.
  void run(lane_set& set, lane& self);
  /// Serve @p work through the arena slot at offset @p slot.  Returns true when it handed the
  /// slot and @p work (moved from) to the completion thread, false when the read is done.
  bool read_segment(lane& self, std::size_t slot, segment& work);
  void write_segment(lane& self, std::size_t slot, segment const& work);
  /// The loop of the completion thread: settle read segments as their copies finish.
  void complete_copies();
  /// Give @p job's slot back and settle (or, for a failed copy, fail) its segment.
  void finish_copy(copy_job& job, cudaError_t status) noexcept;

  config _config;
  int _device;
  std::vector<int> _cpus;
  uring::uring_reactor _host;
  std::shared_ptr<device_arena> _arena;
  std::unique_ptr<transport> _transport;
  std::unique_ptr<gate_pool> _gates;
  std::unique_ptr<lane_set> _readers;
  std::unique_ptr<lane_set> _writers;
  /// The copy-done events of read segments in flight, recycled like the gates.
  std::unique_ptr<gate_pool> _copy_events;
  std::unique_ptr<completer> _completer;
  std::atomic<std::size_t> _queued_bytes{0};
  std::atomic<std::uint64_t> _reads{0};
  std::atomic<std::uint64_t> _bytes_read{0};
  std::atomic<std::uint64_t> _writes{0};
  std::atomic<std::uint64_t> _bytes_written{0};
};

}  // namespace cucascade::io::cufile

namespace cucascade::io {

template <>
struct reactor_traits<cufile::cufile_reactor> {
  static constexpr bool supports_device_read         = true;
  static constexpr bool supports_host_to_device_read = false;
  static constexpr bool supports_vector_host_read    = false;
  static constexpr bool supports_device_range_read   = true;
  static constexpr bool prefers_bulk_io              = false;
};

}  // namespace cucascade::io
