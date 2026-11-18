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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>

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
 * The eight legacy entry points are required. Async reads, async writes, stream registration,
 * batch I/O and parameters are independent optional groups. Check the corresponding capability
 * before calling an optional member. Loading the table does not open the driver or call setters.
 *
 * This is a private detail of the cuFile backend; nothing outside the backend and its tests
 * should include this header.
 */
struct cufile_api {
  decltype(&cuFileDriverOpen) driver_open{nullptr};
  decltype(&cuFileDriverGetProperties) driver_get_properties{nullptr};
  decltype(&cuFileHandleRegister) handle_register{nullptr};
  decltype(&cuFileHandleDeregister) handle_deregister{nullptr};
  decltype(&cuFileBufRegister) buf_register{nullptr};
  decltype(&cuFileBufDeregister) buf_deregister{nullptr};
  decltype(&cuFileRead) read{nullptr};
  decltype(&cuFileWrite) write{nullptr};

  using get_size_parameter_fn   = CUfileError_t (*)(size_parameter, std::size_t*);
  using get_bool_parameter_fn   = CUfileError_t (*)(bool_parameter, bool*);
  using get_string_parameter_fn = CUfileError_t (*)(string_parameter, char*, int);
  /// @c cuFileGetParameterSizeT: the effective value once the driver is open.
  get_size_parameter_fn get_size_parameter{nullptr};
  /// @c cuFileGetParameterBool.
  get_bool_parameter_fn get_bool_parameter{nullptr};
  /// @c cuFileGetParameterString.
  get_string_parameter_fn get_string_parameter{nullptr};

  decltype(&cuFileReadAsync) read_async{nullptr};
  decltype(&cuFileWriteAsync) write_async{nullptr};
  decltype(&cuFileStreamRegister) stream_register{nullptr};
  decltype(&cuFileStreamDeregister) stream_deregister{nullptr};
  decltype(&cuFileBatchIOSetUp) batch_setup{nullptr};
  decltype(&cuFileBatchIOSubmit) batch_submit{nullptr};
  decltype(&cuFileBatchIOGetStatus) batch_get_status{nullptr};
  decltype(&cuFileBatchIOCancel) batch_cancel{nullptr};
  decltype(&cuFileBatchIODestroy) batch_destroy{nullptr};
  bool has_async_read{false};
  bool has_async_write{false};
  bool has_stream_register{false};
  bool has_batch{false};
  bool has_parameters{false};

  /// Resolver test seam: the context belongs to the caller; no global loader state is changed.
  using symbol_resolver = void* (*)(void* context, char const* name) noexcept;

  /**
   * @brief Resolve a table without opening a library or calling any resolved function.
   * @return nullopt for a null resolver or any missing required symbol. Optional groups are
   *         independent; parameter pointers remain individually usable when their group is partial.
   */
  [[nodiscard]] static std::optional<cufile_api> resolve_symbols(symbol_resolver resolver,
                                                                 void* context) noexcept;

  /**
   * @brief The process-wide table, loaded on first call.
   *
   * Thread-safe; never throws.
   *
   * @return The table, or null when libcufile cannot be loaded or lacks a required symbol (the
   *         reason is logged once).
   */
  [[nodiscard]] static cufile_api const* get() noexcept;
  [[nodiscard]] static cufile_api const* loaded() noexcept;
  [[nodiscard]] static std::uint64_t generation() noexcept;
  [[nodiscard]] static bool open_loaded_driver() noexcept;

  /**
   * @brief Open the cuFile driver once per process.
   *
   * Later calls return the first call's outcome.  The driver is never closed: cuFile releases it
   * at process exit.  Thread-safe; never throws.
   *
   * @return Whether the driver is open, including an external already-open driver (false when
   * the library is unavailable). An external driver's parameters remain unverified.
   */
  [[nodiscard]] static bool open_driver() noexcept;

  /**
   * @brief `cuFileGetVersion`, e.g. 1170 for libcufile 1.17.
   *
   * @return The version, or 0 when the library is unavailable or does not report one.
   */
  [[nodiscard]] static int version() noexcept;

  // ABI slots are unconditional, even with no optional declarations/detection macros.
  // Only implementation adapters cast these back to a header-validated exact signature.
  using opaque_function = void (*)();
  opaque_function set_parameter_size_t{nullptr};
  opaque_function set_parameter_bool{nullptr};
  opaque_function set_parameter_string{nullptr};
  opaque_function get_parameter_size_t{nullptr};
  opaque_function get_parameter_bool{nullptr};
  opaque_function get_parameter_string{nullptr};
  opaque_function get_parameter_min_max_value{nullptr};
  opaque_function set_parameter_posix_pool_slab_array{nullptr};
  opaque_function get_parameter_posix_pool_slab_array{nullptr};
  opaque_function set_parameter_gpu_bounce_buffer_slab_array{nullptr};
  opaque_function get_parameter_gpu_bounce_buffer_slab_array{nullptr};

  /// Layout fingerprint from the compiled library, for cross-detection consumer tests.
  [[nodiscard]] static std::size_t abi_size() noexcept;
};

/**
 * @brief Shared, single-attempt driver-open coordinator.
 *
 * Both legacy open_driver() and runtime initialization acquire this transaction. Keeping it
 * alive across setters, open, properties and readback prevents legacy registration from racing
 * pre-open configuration. Independent instances isolate injected runtimes in tests. Calls to
 * cuFileDriverOpen outside cuCascade cannot be serialized here; its already-open status is
 * retained, never treated as permission to close/reopen the driver.
 */
class driver_open_state {
 public:
  struct result {
    bool attempted{false};
    bool opened{false};
    std::optional<CUfileError_t> status;
  };

  class transaction {
   public:
    ~transaction() noexcept;
    [[nodiscard]] bool attempted() const noexcept { return _state._result.attempted; }
    [[nodiscard]] result open(cufile_api const* api) noexcept;

   private:
    friend class driver_open_state;
    explicit transaction(driver_open_state& state) : _state(state), _lock(state._mutex) {}
    driver_open_state& _state;
    std::unique_lock<std::mutex> _lock;
  };

  [[nodiscard]] transaction acquire() { return transaction{*this}; }
  /// Lock-free after the first complete initialization transaction, including failed opens.
  [[nodiscard]] bool open(cufile_api const* api) noexcept;
  [[nodiscard]] static driver_open_state& process() noexcept;

 private:
  std::mutex _mutex;
  result _result;
  // Published only when the opening transaction ends, after properties and readback.
  std::atomic<int> _outcome{-1};
};

}  // namespace cucascade::io::cufile
