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

#include <cstddef>

namespace cucascade::io::cufile {

/**
 * @brief What the process's cuFile library reports about this host.
 *
 * cuFile falls back to a POSIX "compatibility mode" silently whenever no real
 * GPUDirect Storage path (nvidia-fs, PCIe P2PDMA, C2C, an RDMA filesystem client)
 * is active.  Reads and writes stay correct there, but they bounce through host
 * memory inside cuFile, so a caller choosing between the cuFile and io_uring
 * backends needs to know which mode it would get.
 */
struct platform_info {
  bool library_available{false};  ///< libcufile could be loaded.
  bool driver_opened{false};      ///< @c cuFileDriverOpen succeeded.
  /// No storage class has a real GDS mode: @c dstatusflags is zero.  This is the
  /// rule gpuio validated (it never calls @c cuFileDriverGetP2PFlags, which resets
  /// cuFile's running parameters in libcufile 1.19).
  bool compat_only{true};
  /// Compatibility mode is forced by the environment (@c KVIKIO_COMPAT_MODE=ON or
  /// @c CUFILE_FORCE_COMPAT_MODE=true) whatever the driver reports.
  bool forced_compat{false};
  int version{0};                        ///< @c cuFileGetVersion, e.g. 1170; 0 when unknown.
  unsigned dstatusflags{0};              ///< Bit per storage class with a real GDS mode.
  std::size_t max_direct_io_size_kb{0};  ///< cuFile's largest single DMA, in KiB.
};

/**
 * @brief Probe the cuFile library once per process and return the result.
 *
 * The first call loads libcufile, opens the cuFile driver (the same process-wide
 * open the cuFile reactor performs; it costs a few hundred milliseconds and lets
 * cuFile's thread pool retain a CUDA context on each visible GPU) and reads
 * @c cuFileDriverGetProperties; the effective mode is logged once.  Later calls
 * return the cached result.  Thread-safe; never throws.
 */
[[nodiscard]] platform_info const& probe_platform() noexcept;

/**
 * @brief Whether a real GPUDirect Storage path is usable on this host.
 *
 * True iff libcufile loaded, the driver opened, at least one storage class reports
 * a real GDS mode and the environment does not force compatibility mode.  A
 * specific file can still end up in compatibility mode (an unsupported filesystem
 * such as tmpfs or btrfs); @c cufile_io_object::cufile_registered() only says the
 * handle registered, not which path cuFile takes for it.
 */
[[nodiscard]] bool gds_available() noexcept;

}  // namespace cucascade::io::cufile
