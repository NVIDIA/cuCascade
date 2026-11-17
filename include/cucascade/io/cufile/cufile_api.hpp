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

#include <cufile.h>

#include <cstddef>
#include <cstdint>

namespace cucascade::io::cufile {

/**
 * @brief cuFile's size_t configuration parameters (@c CUFileSizeTConfigParameter_t).
 *
 * Declared here rather than taken from @c cufile.h: the parameter API first shipped in a later
 * libcufile than the oldest header cuCascade builds against, and the library is loaded at run time
 * anyway.  The values are the header's enumerators, identical in libcufile 1.16 through 1.20.
 */
enum class size_parameter : int {
  profile_stats                     = 0,
  execution_max_io_queue_depth      = 1,
  execution_max_io_threads          = 2,
  execution_min_io_threshold_kb     = 3,
  execution_max_request_parallelism = 4,
  max_direct_io_size_kb             = 5,
  max_device_cache_size_kb          = 6,
  per_buffer_cache_size_kb          = 7,
  max_device_pinned_mem_size_kb     = 8,
  io_batch_size                     = 9,
  poll_threshold_size_kb            = 10,
  batch_io_timeout_ms               = 11,
};

/// cuFile's boolean configuration parameters (@c CUFileBoolConfigParameter_t); see
/// @ref size_parameter for why they are declared here.
enum class bool_parameter : int {
  use_poll_mode           = 0,
  allow_compat_mode       = 1,
  force_compat_mode       = 2,
  api_check_aggressive    = 3,
  execution_parallel_io   = 4,
  allow_system_memory     = 6,
  use_pci_p2pdma          = 7,
  prefer_io_uring         = 8,
  force_odirect_mode      = 9,
  skip_topology_detection = 10,
  stream_memops_bypass    = 11,
  posix_io_mode           = 12,
  gds_fallback_io         = 13,
};

/// cuFile's string configuration parameters (@c CUFileStringConfigParameter_t); see
/// @ref size_parameter for why they are declared here.
enum class string_parameter : int {
  logging_level    = 0,
  env_logfile_path = 1,
  log_dir          = 2,
  rdma_transport   = 3,
};

/**
 * @brief The cuFile entry points cuCascade calls, resolved at run time.
 *
 * libcufile.so.0 is loaded with `dlopen(RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE)` on first use and
 * never linked, so binaries run on hosts without it: @ref get() then returns null and every cuFile
 * path falls back to POSIX I/O.  `cufile.h` is a build-time dependency only.  `RTLD_NODELETE`
 * keeps the library mapped for the life of the process (it can crash at `dlclose` with statistics
 * enabled).
 *
 * The read entry points are required; a library that lacks one is treated as absent.  The
 * optional members are null when the loaded library predates them, and every caller checks:
 * without @c buf_register / @c buf_deregister the reactor reads into unregistered memory, and
 * without the parameter getters the platform probe reports cuFile's driver properties only.  The
 * members are plain C function pointers: no state, no virtual call.
 *
 * This is a private detail of the cuFile backend; nothing outside the backend and its tests
 * should include this header.
 */
struct cufile_api {
  // -- required -------------------------------------------------------------
  decltype(&cuFileDriverOpen) driver_open{nullptr};
  decltype(&cuFileDriverGetProperties) driver_get_properties{nullptr};
  decltype(&cuFileHandleRegister) handle_register{nullptr};
  decltype(&cuFileHandleDeregister) handle_deregister{nullptr};
  decltype(&cuFileRead) read{nullptr};

  // -- optional -------------------------------------------------------------
  /// @c cuFileBufRegister: pins device memory for direct DMA (every libcufile has it, but a
  /// library without it still serves reads).
  decltype(&cuFileBufRegister) buf_register{nullptr};
  /// @c cuFileBufDeregister; resolved together with @c buf_register (both or neither).
  decltype(&cuFileBufDeregister) buf_deregister{nullptr};

  using get_size_parameter_fn   = CUfileError_t (*)(size_parameter, std::size_t*);
  using get_bool_parameter_fn   = CUfileError_t (*)(bool_parameter, bool*);
  using get_string_parameter_fn = CUfileError_t (*)(string_parameter, char*, int);
  /// @c cuFileGetParameterSizeT: the effective value once the driver is open.
  get_size_parameter_fn get_size_parameter{nullptr};
  /// @c cuFileGetParameterBool.
  get_bool_parameter_fn get_bool_parameter{nullptr};
  /// @c cuFileGetParameterString.
  get_string_parameter_fn get_string_parameter{nullptr};

  /**
   * @brief The table every cuFile call goes through: libcufile's, loaded on first
   *        call, unless a test substituted its own (@c detail::set_cufile_api_for_testing).
   *
   * Thread-safe; never throws.
   *
   * @return The table, or null when libcufile cannot be loaded or lacks a required symbol (the
   *         reason is logged once).
   */
  [[nodiscard]] static cufile_api const* get() noexcept;

  /// libcufile's own table, never a test substitute: what the process-wide
  /// platform probe describes.  Null when the library is unavailable.
  [[nodiscard]] static cufile_api const* loaded() noexcept;

  /// Which table @ref get returns: 0 for libcufile's, a fresh number for every
  /// test substitute installed, so state kept per table (e.g. a failed
  /// registration) never carries over to another one at the same address.
  [[nodiscard]] static std::uint64_t generation() noexcept;

  /**
   * @brief Open the cuFile driver once per process (the test substitute's
   *        @c driver_open, every call, while one is installed).
   *
   * Later calls return the first call's outcome.  The driver is never closed: cuFile releases it
   * at process exit.  Thread-safe; never throws.
   *
   * @return Whether `cuFileDriverOpen` succeeded (false when the library is unavailable).
   */
  [[nodiscard]] static bool open_driver() noexcept;

  /// As @ref open_driver, always for libcufile itself.
  [[nodiscard]] static bool open_loaded_driver() noexcept;

  /**
   * @brief `cuFileGetVersion`, e.g. 1170 for libcufile 1.17.
   *
   * @return The version, or 0 when the library is unavailable or does not report one.
   */
  [[nodiscard]] static int version() noexcept;
};

}  // namespace cucascade::io::cufile
