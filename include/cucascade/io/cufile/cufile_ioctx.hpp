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
 * A pool of @c cufile_reactor s serves device reads and writes with synchronous
 * @c cuFileRead / @c cuFileWrite (or, when cuFile cannot or should not serve
 * them, POSIX I/O staged through each reactor's pinned scratch) and host I/O with
 * POSIX @c pread / @c pwrite; the pool shares one @c reactor_context (config
 * only).  Selected by
 * @c io_backend::cufile and registered from the datasource factory like the
 * uring / kvikio / rest backends.
 */
class cufile_ioctx : public templated_ioctx<cufile_reactor> {
 public:
  /// Build a pool of @p n_reactors reactors, all sharing @p ctx.  The ioctx
  /// config is sourced from the reactors themselves -- see @c templated_ioctx.
  cufile_ioctx(std::size_t n_reactors, std::shared_ptr<cufile_reactor::reactor_context> ctx);

  [[nodiscard]] io_context_type type() const noexcept override { return io_context_type::cufile; }

  /**
   * @brief Reactor selection tuned to synchronous reactors.
   *
   * A cuFile reactor serves one transfer at a time (synchronous cuFile and POSIX
   * calls), so a request's queue depth is the number of reactors it is spread
   * over.  A vectored read with more slices than the base's two gets every
   * reactor, least backlogged first.  A host-only vectored write goes to one
   * reactor: its page-cache writes with early writeback keep the device busy,
   * and interleaving writers of one file only contends on its inode
   * (docs/io/gpuio-gds-practices.md section 8).
   * Other operations keep the base selection.
   */
  std::vector<cufile_reactor*> next_reactor(cufile_io_object const& object,
                                            std::size_t n_slices,
                                            io_op_type type,
                                            int device_id = -1) override;

 protected:
  /// Opens @p path with this ioctx's @c use_odirect / @c compat_mode, which the
  /// static @c cufile_reactor::create_io_object(path) cannot see.
  std::shared_ptr<io_object> create_io_object(std::string path) override;

  /// Opens @p path for writing with this ioctx's @c use_odirect / @c compat_mode.
  std::shared_ptr<io_object> create_io_object_for_write(std::string path,
                                                        write_open_options options) override;
};

}  // namespace cucascade::io::cufile
