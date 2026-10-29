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
#include <cucascade/io/cufile/platform.hpp>
#include <cucascade/io/templated_ioctx.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace cucascade::io::cufile {

// ---------------------------------------------------------------------------
// cufile_ioctx
// ---------------------------------------------------------------------------

/**
 * @brief cuFile / GPUDirect Storage local-disk ioctx.  Thin specialisation of
 *        @c templated_ioctx<cufile_reactor>.
 *
 * A pool of @c cufile_reactor s serves device reads with synchronous
 * @c cuFileRead (or, when cuFile cannot or should not serve them, POSIX reads
 * staged through each reactor's pinned scratch) and host reads with POSIX
 * @c pread; the pool shares one @c reactor_context (config only).  Writes
 * (@c open_io_object_for_write and the write API) use @c cuFileWrite for large
 * aligned device bodies on a real GDS platform and staged / direct @c pwrite
 * otherwise (see @c cufile_reactor).
 * Selected by @c io_backend::cufile and registered from the datasource factory
 * like the uring / rest backends.
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
  /// Build a pool of @p n_reactors reactors, all sharing @p ctx.  The ioctx
  /// config is sourced from the reactors themselves -- see @c templated_ioctx.
  cufile_ioctx(std::size_t n_reactors, std::shared_ptr<cufile_reactor::reactor_context> ctx);

  /// Build one reactor per placement of @p layout, all sharing @p ctx; each
  /// worker is pinned to its placement's CPUs, and device requests prefer the
  /// reactors of their GPU's NUMA node (see @c templated_ioctx::next_reactor).
  cufile_ioctx(reactor_layout const& layout, std::shared_ptr<cufile_reactor::reactor_context> ctx);

  [[nodiscard]] io_context_type type() const noexcept override { return io_context_type::cufile; }

  /**
   * @brief Reactor selection tuned to synchronous reactors.
   *
   * A cuFile reactor serves one transfer at a time (synchronous cuFile and POSIX
   * calls), so a request's queue depth is the number of reactors it is spread
   * over.  A vectored read with more slices than the base's two gets every
   * reactor it may use (its GPU's NUMA-local reactors unless they spill, see
   * @c templated_ioctx::routable_reactors), least backlogged first.  A
   * host-only vectored write goes to one reactor: its page-cache writes with
   * early writeback keep the device busy, and interleaving writers of one file
   * only contends on its inode (docs/io/gpuio-gds-practices.md section 8).
   * Other operations keep the base selection.
   */
  std::vector<cufile_reactor*> next_reactor(cufile_io_object const& object,
                                            std::size_t n_slices,
                                            io_op_type type,
                                            int device_id = -1) override;

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

 protected:
  /// Opens @p path with this ioctx's @c use_odirect / @c compat_mode, which the
  /// static @c cufile_reactor::create_io_object(path) cannot see.
  std::shared_ptr<io_object> create_io_object(std::string path) override;

  /// Opens @p path for writing with this ioctx's @c use_odirect / @c compat_mode.
  std::shared_ptr<io_object> create_io_object_for_write(std::string path,
                                                        write_open_options options) override;
};

}  // namespace cucascade::io::cufile
