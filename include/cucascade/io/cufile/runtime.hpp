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
#include <cucascade/io/cufile/cufile_api.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <variant>
#include <vector>

namespace cucascade::io::cufile {

enum class process_parameter : std::uint8_t {
  cufile_json_path,
  allow_compat_mode,
  force_compat_mode,
  max_direct_io_size_kib,
  max_device_pinned_mem_size_kib,
  max_device_cache_size_kib,
  rdma_dev_addr_list,
  rdma_load_balancing_policy,
  gpu_bounce_buffer_slabs,
  stats_level,
  logging_level
};

using parameter_value =
  std::variant<bool, std::size_t, std::string, std::vector<std::string>, gpu_bounce_slabs>;

enum class parameter_status : std::uint8_t { applied, unapplied, unsupported, unverified, differs };

struct parameter_report {
  process_parameter parameter;
  parameter_value requested;
  parameter_status status{parameter_status::unapplied};
  std::string_view reason;
};

struct parameter_adapter;
/** @brief Header/ID-checked production adapters; also usable with an injected fake API. */
[[nodiscard]] parameter_adapter production_parameter_adapter() noexcept;

struct optional_group_report {
  bool async_read{false};
  bool async_write{false};
  bool stream_register{false};
  bool batch{false};
  bool parameters{false};
};

struct init_report {
  bool library_available{false};
  bool driver_attempted_before_init{false};
  bool driver_previously_opened{false};
  bool driver_opened{false};
  bool properties_available{false};
  bool forced_compat{false};
  bool parameters_differ{false};
  std::uint64_t generation{0};
  std::optional<CUfileError_t> open_status;
  std::optional<CUfileError_t> properties_status;
  CUfileDrvProps_t properties{};
  optional_group_report optional_groups;
  process_cufile_parameters frozen_parameters;
  /// Always describes the frozen first request; differing later requests are never applied.
  std::vector<parameter_report> parameters;

  [[nodiscard]] bool has_unapplied_parameters() const noexcept;
};

class configuration_error : public std::runtime_error {
 public:
  explicit configuration_error(init_report report);
  [[nodiscard]] init_report const& report() const noexcept { return _report; }

 private:
  init_report _report;
};

/** @brief Typed adapter seam; injected callbacks must be noexcept and must not reenter runtime. */
struct parameter_adapter {
  enum class set_result : std::uint8_t { unsupported, rejected, accepted };
  enum class verification : std::uint8_t { unavailable, matches, differs };

  void* context{nullptr};
  set_result (*set)(void*, cufile_api const&, process_parameter, parameter_value const&) noexcept {
    nullptr};
  verification (*verify)(void*,
                         cufile_api const&,
                         process_parameter,
                         parameter_value const&) noexcept {nullptr};
};

/**
 * @brief Process-lifetime initialization, independent of reactor/context creation.
 *
 * Initialization loads/resolves, freezes requests, applies supported setters under the shared
 * driver transaction, opens once, then queries properties/readback before releasing that lock.
 * It never closes the driver and never calls cuFileDriverGetP2PFlags. The legacy entry point
 * joins the same transaction, freezing defaults if called first, without enabling an engine.
 * cufile_json_path uses process-wide setenv here on the control/bootstrap path. Initialize
 * before starting threads that read the environment; our driver lock cannot protect them.
 */
class runtime {
 public:
  [[nodiscard]] static runtime& instance();

  /// Provider seam models library loading and optional symbol resolution without global state.
  using api_provider = cufile_api const* (*)(void* context) noexcept;
  runtime(api_provider provider,
          void* context,
          driver_open_state& driver,
          parameter_adapter adapter = {},
          bool forced_compat        = false);

  /**
   * @brief Independent injected runtime; dependencies must outlive this object.
   * @note A null API models a missing library. This never changes the production singleton.
   */
  runtime(cufile_api const* api,
          driver_open_state& driver,
          parameter_adapter adapter = {},
          bool forced_compat        = false);

  /**
   * @brief Initialize once; report every requested setting and retain the first result forever.
   * @throws configuration_error In strict mode, for unapplied settings or ANY differing request.
   * Missing/failed getters leave accepted setters unverified. Contradicting readback is differs.
   * Strict mode permits unverified, but rejects differs/rejected/unsupported settings. Strict
   * failure does not undo initialization, reopen the driver, or unfreeze the first request.
   */
  [[nodiscard]] init_report ensure_initialized(process_cufile_parameters const& parameters,
                                               bool strict = false);

  /**
   * @brief Join the frozen runtime, initializing defaults only if no request was frozen yet.
   * Backend resource creation uses this overload, not an explicit empty parameter request.
   * Once initialized, the immutable report is available without taking the process mutex.
   */
  [[nodiscard]] init_report ensure_initialized();

  /**
   * @brief Capability gate only; default legacy mode always returns false. Never initializes.
   * async_auto additionally requires direct-capable properties and no forced compat mode.
   * With writes=true the independent async-write group is checked instead of async-read.
   * async_simulated_for_testing requires only an open driver and the selected symbols, permitting
   * the explicit compat/simulator tests in AM-4/AM-7. Neither mode qualifies a specific mount.
   */
  [[nodiscard]] bool engine_possible(engine_mode mode = engine_mode::legacy,
                                     bool writes      = false) const noexcept;
  [[nodiscard]] std::uint64_t generation() const noexcept;
  /// Null before initialization or when the library is absent.
  [[nodiscard]] cufile_api const* api() const noexcept;

 private:
  runtime();
  void initialize(process_cufile_parameters const& parameters);

  bool const _process;
  cufile_api const* const _injected_api;
  api_provider const _provider{nullptr};
  void* const _provider_context{nullptr};
  driver_open_state& _driver;
  parameter_adapter const _adapter;
  bool const _forced_compat;
  std::mutex _mutex;
  bool _initialized{false};
  init_report _report;
  std::atomic<cufile_api const*> _api{nullptr};
  std::atomic<bool> _engine_capable{false};
  std::atomic<bool> _async_capable{false};
  std::atomic<bool> _write_capable{false}, _direct_capable{false};
  std::atomic<std::uint64_t> _generation{0};
};

}  // namespace cucascade::io::cufile
