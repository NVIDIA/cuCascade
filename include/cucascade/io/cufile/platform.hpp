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
#include <cucascade/memory/topology_discovery.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

namespace cucascade::io::cufile {

class cufile_io_object;

/**
 * @brief What the process's cuFile library reports about this host.
 *
 * cuFile falls back to a POSIX "compatibility mode" silently whenever no real
 * GPUDirect Storage path (nvidia-fs, PCIe P2PDMA, C2C, an RDMA filesystem client)
 * is active.  Reads stay correct there, but they bounce through host memory
 * inside cuFile, so a caller choosing between the cuFile and io_uring backends
 * needs to know which mode it would get.
 *
 * This is a process-wide snapshot of the library and driver; it says nothing
 * about a particular file (see @ref probe_path for that).
 */
struct platform_info {
  bool library_available{false};  ///< libcufile could be loaded.
  bool driver_opened{false};      ///< @c cuFileDriverOpen succeeded.
  /// @c cuFileDriverGetProperties succeeded, so @c dstatusflags, @c compat_only
  /// and @c max_direct_io_size_kb are meaningful.  False with an open driver
  /// means the mode is unknown, not that it is compatibility-only.
  bool properties_available{false};
  /// No storage class has a real GDS mode: @c dstatusflags is zero (true too
  /// when the properties are unavailable).  This is the rule gpuio validated (it
  /// never calls @c cuFileDriverGetP2PFlags, which resets cuFile's running
  /// parameters in libcufile 1.19).
  bool compat_only{true};
  /// Compatibility mode is forced by @c CUFILE_FORCE_COMPAT_MODE: a value
  /// beginning with true (any case), as cuFile reads it, whatever the driver reports.
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
 * True iff libcufile loaded, the driver opened and reported its properties, at
 * least one storage class reports a real GDS mode and the environment does not
 * force compatibility mode.  A specific file can still end up in compatibility
 * mode (an unsupported filesystem such as tmpfs or btrfs);
 * @c cufile_io_object::cufile_registered() only says the handle registered, not
 * which path cuFile takes for it -- see @ref probe_path.
 */
[[nodiscard]] bool gds_available() noexcept;

/**
 * @brief What GPUDirect Storage can do for reads of one file.
 *
 * Never a guarantee: the transfer mode cuFile picks also depends on the GPU, the
 * buffer and its alignment, cuFile's own configuration (@c cufile.json) and
 * driver state, none of which is observable before a read.
 */
enum class path_capability : std::uint8_t {
  /// cuFile cannot serve the file at all: libcufile or its driver is missing,
  /// the path does not name a readable regular file, or @c compat::off would
  /// refuse to open it.  (With @c compat::automatic reads still work, via POSIX.)
  unavailable,
  /// cuFile can only bounce through host memory, or is not used, for this file:
  /// policy (@c compat::on, @c use_odirect=false, @c CUFILE_FORCE_COMPAT_MODE),
  /// no storage class with a real GDS mode, an unsupported filesystem or storage
  /// class, or a file that could not be opened @c O_DIRECT / registered.
  compatibility_only,
  /// Every gate cuCascade can check passed: the driver reports a real GDS mode
  /// for the file's storage class, the filesystem is one GDS supports on it,
  /// and the file is open @c O_DIRECT and registered with cuFile.  cuFile may
  /// DMA directly, but may still fall back per transfer.
  direct_candidate,
  /// Not determined: the driver properties or the file's storage stack could
  /// not be read, or the stack is mixed / layered.  Not a negative answer.
  unknown,
};

/// The storage class whose @c dstatusflags bit decides a file's GDS mode.
enum class storage_class : std::uint8_t {
  none,      ///< No class applies (memory / pseudo filesystem, unknown stack).
  nvme,      ///< Local PCIe NVMe (@c CU_FILE_NVME_SUPPORTED).
  nvmeof,    ///< NVMe over fabrics (@c CU_FILE_NVMEOF_SUPPORTED).
  nfs,       ///< NFS (@c CU_FILE_NFS_SUPPORTED).
  lustre,    ///< Lustre (@c CU_FILE_LUSTRE_SUPPORTED).
  wekafs,    ///< WekaFS (@c CU_FILE_WEKAFS_SUPPORTED).
  gpfs,      ///< IBM Storage Scale / GPFS (@c CU_FILE_GPFS_SUPPORTED).
  beegfs,    ///< BeeGFS (@c CU_FILE_BEEGFS_SUPPORTED).
  virtiofs,  ///< VirtioFS (@c CU_FILE_VIRTIOFS_SUPPORTED).
};

/**
 * @brief Evidence and verdict of @ref probe_path for one file.
 *
 * A by-value snapshot: the passive storage discovery is not cached, and the
 * platform part is the process-wide @ref probe_platform snapshot.
 */
struct path_info {
  path_capability capability = path_capability::unknown;  ///< The verdict.
  std::string reason;                                     ///< Why: the deciding gate, in words.

  // -- policy ---------------------------------------------------------------
  compat compat_mode = compat::automatic;  ///< Effective configured policy.
  bool use_odirect{true};                  ///< Configured @c use_odirect.

  // -- platform -------------------------------------------------------------
  platform_info platform;  ///< The process-wide cuFile snapshot.

  // -- storage --------------------------------------------------------------
  memory::path_storage_info storage;                  ///< Passive mount / device discovery.
  storage_class matched_class = storage_class::none;  ///< Class the verdict checked.
  /// Whether the driver reports a real GDS mode for @c matched_class (its
  /// @c dstatusflags bit is set).  Never inferred from any other bit.
  bool class_supported{false};
  /// Whether the filesystem is one GDS serves directly on @c matched_class
  /// (ext4 / xfs on local NVMe; the network filesystem itself otherwise).
  bool filesystem_supported{false};

  // -- the opened file ------------------------------------------------------
  bool opened{false};       ///< The file was opened read-only.
  bool direct_open{false};  ///< An @c O_DIRECT descriptor is open.
  bool registered{false};   ///< @c cuFileHandleRegister succeeded for it.
  std::string open_error;   ///< Why opening failed, when it did.
};

/**
 * @brief Report what GPUDirect Storage can do for reads of @p path under @p cfg.
 *
 * Active, unlike passive storage discovery: it calls @ref probe_platform (loads
 * libcufile and opens the cuFile driver once per process, which can retain a
 * CUDA context per visible GPU and create @c cufile.log), and opens @p path
 * read-only the way the reactor would -- buffered and, policy permitting,
 * @c O_DIRECT, registered with cuFile -- then closes it.  It reads no data and
 * never writes.  Best effort and never throws: the file can change right after.
 *
 * Do not call it while holding a lock other threads route reads under; it can
 * take hundreds of milliseconds on first use.
 */
[[nodiscard]] path_info probe_path(std::string const& path, config const& cfg = {}) noexcept;

/**
 * @brief As @ref probe_path, for a file the reactor already opened with policy
 *        @p cfg -- so the verdict describes exactly the file that is read.
 *        Never throws.
 */
[[nodiscard]] path_info probe_object(cufile_io_object const& object, config const& cfg) noexcept;

}  // namespace cucascade::io::cufile
