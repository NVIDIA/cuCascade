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

#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/placement.hpp>
#include <cucascade/io/cufile/platform.hpp>
#include <cucascade/io/templated_ioctx.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace cucascade::io::cufile {

// ---------------------------------------------------------------------------
// cufile_ioctx
// ---------------------------------------------------------------------------

/**
 * @brief cuFile / GPUDirect Storage local-disk ioctx.  Specialisation of
 *        @c templated_ioctx<cufile_reactor>.
 *
 * A pool of @c cufile_reactor s serves device reads with synchronous
 * @c cuFileRead (staged through registered slots on a GDS-capable host, see
 * @ref staging; or, when cuFile cannot or should not serve them, POSIX reads
 * staged through each reactor's pinned scratch) and host reads with POSIX
 * @c pread; the pool shares one @c reactor_context.  Writes
 * (@c open_io_object_for_write and the write API) use @c cuFileWrite for large
 * aligned device bodies on a real GDS platform, staged through the same
 * registered slots, and staged / direct @c pwrite otherwise (see
 * @c cufile_reactor).  Selected by @c io_backend::cufile and registered from
 * the datasource factory like the uring / rest backends.
 *
 * Built from a @ref pool_plan (see @ref plan_cufile_pool), the pool has a group
 * of reactors per GPU, pinned to the CPUs local to that GPU: a device read goes
 * to the group of its destination's GPU, so each GPU gets its own queue depth
 * and its I/O is issued from its own socket.  Built from a reactor count, it is
 * one flat group serving every GPU.
 *
 * On the submitting thread, each device read is prepared before it is queued:
 * - its GPU is taken from the destination pointer (not from the calling
 *   thread's current device);
 * - an event is recorded on its stream when that stream still has work queued
 *   (@c device_buffer::ready_gate), which the reactors wait on instead of the
 *   stream handle -- a per-thread default stream names a different stream on a
 *   worker thread.  A stream being captured is refused;
 * - a device-only read larger than @c config::staging_piece_bytes is split into
 *   pieces on 4 KiB boundaries, so several reactors read it at once.
 *
 * With fused reads (@c config::fused_reads enabled, @c cuFileReadAsync available and the
 * object registered with cuFile) the last step is @c plan_fused_reads instead of that split
 * and the legacy fusion (@c config::fuse_gap_bytes): the device slices are packed, in file
 * order, into 4 KiB-aligned windows of at most a lane slot, each one lane read that a lane
 * reactor issues as one @c cuFileReadAsync plus device copies (see @c cufile_reactor).
 * A planned pool then has, per GPU, @c fused_read_config::reactors_per_device lane reactors
 * ahead of its synchronous ones, and @c host_batch_config::reactors batch reactors shared
 * by every GPU; requests are routed by role: a request with a lane read goes to the lane
 * reactors of its GPU, a host-only read to the batch reactors (else every reactor without
 * lanes), anything else (writes, flushes, other device reads) to the synchronous reactors
 * of its GPU.  Routing is a preference: every reactor can serve every slice through its
 * synchronous paths.  A flat pool gives every reactor every enabled capability and routes
 * as without them.
 *
 * A device write is prepared the same way (its GPU from the source pointer, a
 * gate on its stream), so it reaches its source GPU's group and is staged
 * through that GPU's slots; it is split by @c config::write_split_bytes.
 *
 * Vectored reads (@c device_readv_async_io, @c host_device_readv_async_io and
 * the datasource's @c device_read_ranges_async on top of them) keep the generic
 * contract: the returned future settles only when every slice has stopped
 * touching its destination, failures and cancellations included, and a
 * successful device read is complete without further stream synchronization
 * (see @c cufile_reactor).
 */
class cufile_ioctx : public templated_ioctx<cufile_reactor> {
 public:
  /// Build one flat pool of @p n_reactors reactors serving every GPU, all
  /// sharing @p ctx.  The ioctx config is sourced from the reactors themselves --
  /// see @c templated_ioctx.
  cufile_ioctx(std::size_t n_reactors, std::shared_ptr<cufile_reactor::reactor_context> ctx);

  /// Build the pool @p plan describes: a pinned group of reactors per GPU (or a
  /// flat pool when it has no groups), all sharing @p ctx.  @ref start reserves
  /// the staging the plan budgets for each GPU before starting the reactors.
  cufile_ioctx(pool_plan plan, std::shared_ptr<cufile_reactor::reactor_context> ctx);

  [[nodiscard]] io_context_type type() const noexcept override { return io_context_type::cufile; }

  /// Reserve each GPU's planned staging (see @c reserve_staging), then start the
  /// reactors.  Idempotent.
  void start() override;

  /**
   * @brief Reactor selection tuned to synchronous reactors.
   *
   * A cuFile reactor serves one transfer at a time (synchronous cuFile and POSIX
   * calls), so a request's queue depth is the number of reactors it is spread
   * over.  The candidates are, by role (see the class comment): for a host-only read
   * the batch reactors (else every reactor without lanes); for anything else the
   * synchronous reactors of @p device_id's group (its whole group when it has none;
   * every synchronous reactor when the pool is flat or the GPU has no group).  Requests
   * with a lane read are routed by @ref select_reactors to the lane reactors.  A vectored read
   * with more slices than the base's two gets every candidate, least backlogged
   * first, a host-only vectored write the least backlogged one (its page-cache
   * writes with early writeback keep the device busy, and interleaving writers
   * of one file only contends on its inode), any other operation the two least
   * backlogged.
   */
  std::vector<cufile_reactor*> next_reactor(cufile_io_object const& object,
                                            std::size_t n_slices,
                                            io_op_type type,
                                            int device_id = -1) override;

  /// Prepare the device slices on the calling thread (their GPU, stream gate and
  /// split; see the class comment), then dispatch as @c templated_ioctx does.
  exec::semi_future<std::size_t> mixed_readv_async_io(
    io_object const& object, std::vector<prepared_io_slice>&& input_slices) noexcept override;

  /// Prepare the device write slices on the calling thread (their GPU from the
  /// source pointer, and their stream gate, as for reads), then dispatch as
  /// @c templated_ioctx does: a device write goes to its source GPU's group.
  exec::semi_future<std::size_t> mixed_writev_async_io(
    io_object const& object, std::vector<prepared_io_slice>&& input_slices) noexcept override;

  /**
   * @brief What GPUDirect Storage can do for reads of @p object, an object this
   *        ioctx opened, under this ioctx's configuration (see @ref probe_object).
   *
   * Advisory: computed on each call from the object's own descriptors and
   * registration (passive storage discovery is not cached), and logged once
   * per device and verdict.  Reads never depend on it.
   *
   * @throws std::invalid_argument when @p object was not opened by a cuFile ioctx.
   */
  [[nodiscard]] path_info path_capability(io_object const& object) const;

  /// How the pool has served its reads so far: every reactor's counters summed.
  [[nodiscard]] reactor_stats stats() const noexcept;

  /// As @ref stats, for the reactors of @p device's group only (zero when the
  /// pool has no group for it, e.g. a flat pool).
  [[nodiscard]] reactor_stats stats(int device) const noexcept;

  /// The plan the pool was built from (one flat group when built from a count).
  [[nodiscard]] pool_plan const& plan() const noexcept { return _plan; }

  /// Reactors in the pool, in build order: per GPU its lane reactors then its synchronous
  /// ones, then the host-batch reactors.
  [[nodiscard]] std::size_t reactor_count() const noexcept { return _reactors.size(); }

  /// Reactor @p index (see @ref reactor_count), for per-reactor statistics, placement and
  /// backlog.  @throws std::out_of_range for an index past the pool.
  [[nodiscard]] cufile_reactor const& reactor(std::size_t index) const
  {
    return *_reactors.at(index);
  }

  std::size_t host_write_io(io_object const& object,
                            std::size_t offset,
                            std::size_t size,
                            std::uint8_t const* source) override;
  exec::semi_future<void> flush_async_io(io_object const& object,
                                         io_priority priority) noexcept override;

 protected:
  std::vector<cufile_reactor*> select_reactors(cufile_io_object const& object,
                                               std::span<prepared_io_slice const> slices,
                                               io_op_type operation,
                                               int device_id) override;

  /// 1 for a lane read a lane reactor can take now, else 0: a request's lane reads go to
  /// lane reactors and its other slices to the reactors their own role picks, under one
  /// coordinator (see @c templated_ioctx::slice_class).
  [[nodiscard]] std::size_t slice_class(cufile_io_object const& object,
                                        prepared_io_slice const& slice) const override;

  /// Opens @p path with this ioctx's @c use_odirect / @c compat_mode, which the
  /// static @c cufile_reactor::create_io_object(path) cannot see.
  std::shared_ptr<io_object> create_io_object(std::string path) override;

  /// Opens @p path for writing with this ioctx's @c use_odirect / @c compat_mode.
  std::shared_ptr<io_object> create_io_object_for_write(std::string path,
                                                        write_open_options options) override;

 private:
  /// The reactors of one GPU: @c _reactors[first, first + count), of which the first
  /// @c lanes are its lane reactors and the rest its synchronous ones.
  struct group_range {
    int device{-1};
    std::size_t first{0};
    std::size_t count{0};
    std::size_t lanes{0};
  };

  /// What a request needs of the reactors serving it (see the class comment).
  enum class reactor_role : std::uint8_t { lane, sync, host };

  /// The reactors (indices into @c _reactors) of @p role for @p device_id, by the rules of
  /// the class comment; never empty when the pool has reactors.
  [[nodiscard]] std::vector<std::size_t> candidates(reactor_role role, int device_id) const;

  /// The reactors of @p candidates a request of @p n_slices slices of @p type gets: every
  /// candidate for a vectored read of more than two slices, else the two least backlogged
  /// (one for a host-only vectored write), least backlogged first.
  [[nodiscard]] std::vector<cufile_reactor*> rank(std::vector<std::size_t> const& candidates,
                                                  std::size_t n_slices,
                                                  io_op_type type,
                                                  int device_id);

  /// Whether a request with @p slices of @p object has a lane read for the fused lanes.
  [[nodiscard]] bool has_lane_read(cufile_io_object const& object,
                                   std::span<prepared_io_slice const> slices) const noexcept;

  /// Whether fused reads are enabled and the loaded cuFile offers @c cuFileReadAsync.
  [[nodiscard]] bool fused_reads_usable() const noexcept;

  /// Visit the indices of the reactors that may own lanes for @p device_id (its group's
  /// lane reactors; every reactor of a flat pool) until @p visit returns false.
  template <typename Visit>
  void for_each_lane_reactor(int device_id, Visit&& visit) const;

  /// Whether some reactor can serve lane reads into @p device_id now (see
  /// @c cufile_reactor::lanes_usable).
  [[nodiscard]] bool lanes_routable(int device_id) const noexcept;

  /// Whether @p slice is a lane read a lane reactor can take now.
  [[nodiscard]] bool is_routable_lane_read(cufile_io_object const& object,
                                           prepared_io_slice const& slice) const noexcept;

  /// Resolve the GPU, stream gate and split of every device slice (see the class
  /// comment).  @throws on a capturing stream or a CUDA failure.
  void prepare_device_slices(io_object const& object, std::vector<prepared_io_slice>& slices);

  /// Resolve the GPU (from the source pointer) and stream gate of every device
  /// write slice.  @throws on a capturing stream or a CUDA failure.
  void prepare_device_writes(std::vector<prepared_io_slice>& slices);

  /// Replace runs of neighbouring small device slices by fused slices (see
  /// @c config::fuse_gap_bytes).  Slices outside the file, with a completion
  /// callback or a host destination are left alone.  May throw @c std::bad_alloc,
  /// before it changes @p slices.
  void fuse_device_slices(std::vector<prepared_io_slice>& slices, std::size_t file_size) const;

  pool_plan _plan;
  std::vector<group_range> _groups;
  /// The host-batch reactors of a planned pool: @c _reactors[_batch_first, + _batch_count).
  std::size_t _batch_first{0};
  std::size_t _batch_count{0};
};

/**
 * @brief Build the cuFile ioctx the datasource factory builds, throwing on failure.
 *
 * With @p n_reactors 0 (the @c io_config::cufile_n_reactors default), discovers
 * the topology (NVML and sysfs; RDMA NICs kept when their port is active, IP or
 * not), plans a pool per GPU with @ref plan_cufile_pool, logs the plan at INFO and
 * each of its warnings at WARN, applies the plan's staging piece size and
 * per-device staging caps, and builds the pool from the plan.  With a positive
 * @p n_reactors, builds one flat pool of that many reactors from @p cfg as is (every
 * reactor then has every enabled capability: fused lanes, host batch).
 * The returned ioctx is not started: call @c start() before submitting I/O.
 *
 * @throws std::invalid_argument / std::runtime_error when @p cfg is malformed or
 *         a reactor cannot be built (the registry's factory reports the same as a
 *         null ioctx).
 */
[[nodiscard]] std::shared_ptr<cufile_ioctx> make_cufile_ioctx(std::size_t n_reactors,
                                                              config const& cfg);

}  // namespace cucascade::io::cufile
