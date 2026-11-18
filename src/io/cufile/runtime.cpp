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

#include <cucascade/io/config.hpp>
#include <cucascade/io/cufile/runtime.hpp>
#include <cucascade/log/logging.hpp>

#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <utility>

namespace cucascade::io::cufile {

std::array<std::size_t, 6> cufile_config_library_abi_reference() noexcept
{
  return {sizeof(config),
          alignof(config),
          sizeof(async_engine_config),
          alignof(async_engine_config),
          sizeof(cucascade::io::io_config),
          alignof(cucascade::io::io_config)};
}

namespace {

bool forced_compat_environment() noexcept
{
  auto const* value = std::getenv("CUFILE_FORCE_COMPAT_MODE");
  if (value == nullptr) { return false; }
  constexpr char upper[] = "TRUE";
  for (std::size_t index = 0; index < 4; ++index) {
    if (value[index] != upper[index] && value[index] != upper[index] + ('a' - 'A')) {
      return false;
    }
  }
  return true;
}

std::string_view parameter_name(process_parameter parameter) noexcept
{
  switch (parameter) {
    case process_parameter::cufile_json_path: return "cufile_json_path";
    case process_parameter::allow_compat_mode: return "allow_compat_mode";
    case process_parameter::force_compat_mode: return "force_compat_mode";
    case process_parameter::max_direct_io_size_kib: return "max_direct_io_size_kib";
    case process_parameter::max_device_pinned_mem_size_kib: return "max_device_pinned_mem_size_kib";
    case process_parameter::max_device_cache_size_kib: return "max_device_cache_size_kib";
    case process_parameter::rdma_dev_addr_list: return "rdma_dev_addr_list";
    case process_parameter::rdma_load_balancing_policy: return "rdma_load_balancing_policy";
    case process_parameter::gpu_bounce_buffer_slabs: return "gpu_bounce_buffer_slabs";
    case process_parameter::stats_level: return "stats_level";
    case process_parameter::logging_level: return "logging_level";
  }
  return "unknown";
}

std::vector<parameter_report> requested_parameters(process_cufile_parameters const& parameters)
{
  std::vector<parameter_report> result;
  result.reserve(11);
  auto const add = [&](process_parameter id, auto const& value) {
    if (value.has_value()) {
      result.push_back({id, parameter_value{*value}, parameter_status::unapplied, {}});
    }
  };
  add(process_parameter::cufile_json_path, parameters.cufile_json_path);
  add(process_parameter::allow_compat_mode, parameters.allow_compat_mode);
  add(process_parameter::force_compat_mode, parameters.force_compat_mode);
  add(process_parameter::max_direct_io_size_kib, parameters.max_direct_io_size_kib);
  add(process_parameter::max_device_pinned_mem_size_kib, parameters.max_device_pinned_mem_size_kib);
  add(process_parameter::max_device_cache_size_kib, parameters.max_device_cache_size_kib);
  add(process_parameter::rdma_dev_addr_list, parameters.rdma_dev_addr_list);
  add(process_parameter::rdma_load_balancing_policy, parameters.rdma_load_balancing_policy);
  add(process_parameter::gpu_bounce_buffer_slabs, parameters.gpu_bounce_buffer_slabs);
  add(process_parameter::stats_level, parameters.stats_level);
  add(process_parameter::logging_level, parameters.logging_level);
  return result;
}

}  // namespace

bool init_report::has_unapplied_parameters() const noexcept
{
  return std::any_of(parameters.begin(), parameters.end(), [](parameter_report const& entry) {
    return entry.status == parameter_status::unapplied ||
           entry.status == parameter_status::unsupported ||
           entry.status == parameter_status::differs;
  });
}

configuration_error::configuration_error(init_report report)
  : std::runtime_error(
      report.parameters_differ
        ? "cuFile process parameters differ from the frozen initialization request"
        : "cuFile process parameters could not be applied; inspect the init report"),
    _report(std::move(report))
{
}

runtime::runtime()
  : _process(true),
    _injected_api(nullptr),
    _driver(driver_open_state::process()),
    _adapter(production_parameter_adapter()),
    _forced_compat(false)
{
}

runtime::runtime(cufile_api const* api,
                 driver_open_state& driver,
                 parameter_adapter adapter,
                 bool forced_compat)
  : _process(false),
    _injected_api(api),
    _driver(driver),
    _adapter(adapter),
    _forced_compat(forced_compat)
{
}

runtime::runtime(api_provider provider,
                 void* context,
                 driver_open_state& driver,
                 parameter_adapter adapter,
                 bool forced_compat)
  : _process(false),
    _injected_api(nullptr),
    _provider(provider),
    _provider_context(context),
    _driver(driver),
    _adapter(adapter),
    _forced_compat(forced_compat)
{
}

runtime& runtime::instance()
{
  static auto* const process_runtime = new runtime;
  return *process_runtime;
}

void runtime::initialize(process_cufile_parameters const& parameters)
{
  parameters.validate();
  // Resolve all optional groups before freezing. Allocate reports before the first setter so
  // an allocation failure cannot partially apply the request and then retry initialization.
  auto const* api           = _provider != nullptr ? _provider(_provider_context)
                                                   : (_process ? cufile_api::get() : _injected_api);
  auto entries              = requested_parameters(parameters);
  _report.frozen_parameters = parameters;
  _report.parameters        = std::move(entries);
  _report.library_available = api != nullptr;
  _report.forced_compat     = _process ? forced_compat_environment() : _forced_compat;
  if (api != nullptr) {
    _report.optional_groups = {api->has_async_read,
                               api->has_async_write,
                               api->has_stream_register,
                               api->has_batch,
                               api->has_parameters};
  }

  auto transaction                     = _driver.acquire();
  _report.driver_attempted_before_init = transaction.attempted();
  bool json_environment_created        = false;
  for (auto& entry : _report.parameters) {
    if (entry.parameter == process_parameter::cufile_json_path) {
      // Process-wide mutation: bootstrap must precede unrelated environment readers.
      // Never overwrite a caller-selected JSON file, even in non-strict mode.
      auto const& path    = std::get<std::string>(entry.requested);
      auto const* current = std::getenv("CUFILE_ENV_PATH_JSON");
      if (api == nullptr) {
        entry.status = parameter_status::unsupported;
        entry.reason = "library unavailable";
      } else if (_report.driver_attempted_before_init) {
        entry.reason = "driver already open";
      } else if (path.front() != '/' || ::access(path.c_str(), R_OK) != 0) {
        entry.reason = "json not readable";
      } else if (current && path != current) {
        entry.reason = "conflicting environment";
      } else if (!current && ::setenv("CUFILE_ENV_PATH_JSON", path.c_str(), 0) != 0) {
        entry.reason = "setenv failed";
      } else {
        auto const* selected = std::getenv("CUFILE_ENV_PATH_JSON");
        if (!selected || path != selected) {
          entry.reason = "conflicting environment";
        } else {
          json_environment_created = current == nullptr;
          entry.status             = parameter_status::unverified;
          entry.reason =
            "CUFILE_ENV_PATH_JSON selected before driver open; library semantics unverified";
        }
      }
    } else if (api == nullptr) {
      entry.status = parameter_status::unsupported;
      entry.reason = "library unavailable";
    } else if (_report.driver_attempted_before_init) {
      entry.reason = "driver opening already attempted; pre-open setting not applied";
    } else if (_adapter.set == nullptr) {
      entry.status = parameter_status::unsupported;
      entry.reason = "no validated pre-open adapter compiled for this parameter";
    } else {
      switch (_adapter.set(_adapter.context, *api, entry.parameter, entry.requested)) {
        case parameter_adapter::set_result::unsupported:
          entry.status = parameter_status::unsupported;
          entry.reason = "parameter setter unsupported";
          break;
        case parameter_adapter::set_result::rejected:
          entry.reason = "parameter setter rejected the request";
          break;
        case parameter_adapter::set_result::accepted:
          entry.status = parameter_status::unverified;
          entry.reason = "setter accepted; readback unavailable";
          break;
      }
    }
  }

  auto const opened                = transaction.open(api);
  _report.driver_opened            = opened.opened;
  _report.open_status              = opened.status;
  _report.driver_previously_opened = _report.driver_attempted_before_init && opened.opened;
  bool const externally_opened =
    opened.status.has_value() && opened.status->err == CU_FILE_DRIVER_ALREADY_OPEN;
  if (externally_opened) {
    if (json_environment_created) {
      auto const* selected = std::getenv("CUFILE_ENV_PATH_JSON");
      if (selected && parameters.cufile_json_path == selected) {
        ::unsetenv("CUFILE_ENV_PATH_JSON");
      }
    }
    _report.driver_previously_opened = true;
    for (auto& entry : _report.parameters) {
      entry.status = parameter_status::unapplied;
      entry.reason = entry.parameter == process_parameter::cufile_json_path
                       ? "driver already open"
                       : "driver was already open outside cuCascade; pre-open setting not applied";
    }
  }
  if (opened.opened && api != nullptr && api->driver_get_properties != nullptr) {
    auto const status            = api->driver_get_properties(&_report.properties);
    _report.properties_status    = status;
    _report.properties_available = status.err == CU_FILE_SUCCESS;
  }
  for (auto& entry : _report.parameters) {
    if (entry.status != parameter_status::unverified ||
        entry.parameter == process_parameter::cufile_json_path) {
      continue;
    }
    if (!opened.opened) {
      entry.reason = "setter accepted but driver did not open; effective value unverified";
    } else if (_adapter.verify != nullptr && api != nullptr) {
      switch (_adapter.verify(_adapter.context, *api, entry.parameter, entry.requested)) {
        case parameter_adapter::verification::unavailable: break;
        case parameter_adapter::verification::differs:
          entry.status = parameter_status::differs;
          entry.reason = "setter accepted but readback differs from the requested value";
          break;
        case parameter_adapter::verification::matches:
          entry.status = parameter_status::applied;
          entry.reason = "setter accepted and readback matches";
          break;
      }
    }
  }

  // The whole snapshot is complete before publishing readiness. A generation identifies even
  // an unavailable/failing frozen runtime; initialization never retries or bumps it again.
  _api.store(api, std::memory_order_release);
  _report.generation = 1;
  _async_capable.store(
    opened.opened && api != nullptr && api->has_async_read && api->read_async != nullptr,
    std::memory_order_release);
  _engine_capable.store(opened.opened && _report.properties_available && api != nullptr &&
                          api->has_async_read && api->read_async != nullptr &&
                          _report.properties.nvfs.dstatusflags != 0 && !_report.forced_compat &&
                          !_report.frozen_parameters.force_compat_mode.value_or(false),
                        std::memory_order_release);
  _write_capable.store(
    opened.opened && api != nullptr && api->has_async_write && api->write_async != nullptr,
    std::memory_order_release);
  _direct_capable.store(opened.opened && _report.properties_available &&
                          _report.properties.nvfs.dstatusflags != 0 && !_report.forced_compat &&
                          !_report.frozen_parameters.force_compat_mode.value_or(false),
                        std::memory_order_release);
  _initialized = true;
  _generation.store(_report.generation, std::memory_order_release);
  for (auto const& entry : _report.parameters) {
    if (entry.status != parameter_status::applied) {
      CUCASCADE_LOG_WARN(
        "cufile: process parameter {}: {}", parameter_name(entry.parameter), entry.reason);
    }
  }
}

init_report runtime::ensure_initialized()
{
  if (_generation.load(std::memory_order_acquire) == 0) {
    std::lock_guard lock(_mutex);
    if (!_initialized) { initialize({}); }
  }
  return _report;
}

init_report runtime::ensure_initialized(process_cufile_parameters const& parameters, bool strict)
{
  std::lock_guard lock(_mutex);
  if (!_initialized) { initialize(parameters); }
  auto report              = _report;
  report.parameters_differ = parameters != _report.frozen_parameters;
  if (report.parameters_differ) {
    CUCASCADE_LOG_WARN(
      "cufile: process parameters frozen by earlier initialization; differing request ignored");
  }
  if (strict && (report.parameters_differ || report.has_unapplied_parameters())) {
    throw configuration_error(std::move(report));
  }
  return report;
}

bool runtime::engine_possible(engine_mode mode, bool writes) const noexcept
{
  if (_generation.load(std::memory_order_acquire) == 0) { return false; }
  if (writes)
    return _write_capable.load(std::memory_order_acquire) &&
           (mode == engine_mode::async_simulated_for_testing ||
            (mode == engine_mode::async_auto && _direct_capable.load(std::memory_order_acquire)));
  if (mode == engine_mode::async_simulated_for_testing) {
    return _async_capable.load(std::memory_order_acquire);
  }
  return mode == engine_mode::async_auto && _engine_capable.load(std::memory_order_acquire);
}

std::uint64_t runtime::generation() const noexcept
{
  return _generation.load(std::memory_order_acquire);
}

cufile_api const* runtime::api() const noexcept
{
  if (_generation.load(std::memory_order_acquire) == 0) { return nullptr; }
  return _api.load(std::memory_order_acquire);
}

}  // namespace cucascade::io::cufile
