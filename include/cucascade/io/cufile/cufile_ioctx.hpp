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
#include <string_view>
#include <vector>

namespace cucascade::io::cufile {

/**
 * @brief Whether a real GPUDirect Storage path is usable on this host.
 *
 * True iff libcufile loaded, its driver opened and reported its properties, at
 * least one storage class reports a GDS mode and the environment does not force
 * compatibility mode.  A specific file can still end up in compatibility mode
 * (an unsupported filesystem such as tmpfs).  The first call opens the cuFile
 * driver; later calls return its result.  Thread-safe.
 */
[[nodiscard]] bool gds_available() noexcept;

/**
 * @brief The cuFile driver's GDS status flags: bit `CU_FILE_<X>_SUPPORTED` is set
 *        when storage class X has a GDS mode.
 *
 * 0 exactly when @c gds_available() is false.  Opens the cuFile driver like
 * @c gds_available() does.  Thread-safe.
 */
[[nodiscard]] unsigned int gds_status_flags() noexcept;

/**
 * @brief Whether a driver reporting @p status_flags has a GDS mode for files on
 *        a @p filesystem_type filesystem (the type @c /proc/self/mountinfo names).
 *
 * ext4 and xfs count through the NVMe class and only when @p on_nvme (every
 * drive under the filesystem is NVMe); gpfs, lustre, wekafs, beegfs and nfs /
 * nfs4 through their own class.  Any other filesystem (tmpfs, overlay, ...) is
 * served in compatibility mode.  A prediction: cuFile does not say which mode a
 * file gets, and can still fall back to compatibility mode on a supported
 * filesystem (cufile.json denylists, mount options, RDMA setup).
 */
[[nodiscard]] bool gds_supports_filesystem(unsigned int status_flags,
                                           std::string_view filesystem_type,
                                           bool on_nvme) noexcept;

/**
 * @brief cuFile / GPUDirect Storage local-disk ioctx: one @c cufile_reactor per
 *        GPU.
 *
 * A device read or write goes to the reactor of its GPU -- the GPU the caller
 * names, else the submitting thread's current device -- and fails when the
 * context does not serve that GPU.  Host I/O, which has no GPU, goes to the
 * least-backlogged reactors.  Selected by @c io_backend::cufile and registered
 * from the datasource factory like the uring / rest backends.
 */
class cufile_ioctx final : public templated_ioctx<cufile_reactor> {
 public:
  /// A reactor for every GPU of @c cfg.devices (every visible GPU when empty),
  /// its threads pinned to the CPUs of the GPU's NUMA node when that is known.
  /// @throws std::invalid_argument for no GPUs, or a @p cfg the reactor rejects.
  explicit cufile_ioctx(config const& cfg);

  [[nodiscard]] io_context_type type() const noexcept override { return io_context_type::cufile; }

  /// One gauge snapshot per reactor (so per GPU), in reactor order.
  [[nodiscard]] std::vector<cufile_reactor::gauges> reactor_gauges() const;

  std::vector<cufile_reactor*> next_reactor(io_object_type const& object,
                                            std::size_t n_slices,
                                            io_op_type type,
                                            int device_id = -1) override;

 protected:
  /// The slices of one request are routed per GPU, host slices apart from
  /// device ones.
  [[nodiscard]] std::size_t slice_class(io_object_type const& object,
                                        prepared_io_slice const& slice) const override;

 private:
  /// The GPU of each reactor, in reactor order.
  std::vector<int> _devices;
};

}  // namespace cucascade::io::cufile
