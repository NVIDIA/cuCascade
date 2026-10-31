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

#include <cucascade/io/cufile/cufile_api.hpp>
#include <cucascade/io/cufile/platform.hpp>
#include <cucascade/log/logging.hpp>

#include <cufile.h>

#include <cstdlib>
#include <exception>
#include <string_view>

namespace cucascade::io::cufile {

namespace {

/// Whether @p name is set to a value cuFile reads as "on".
[[nodiscard]] bool env_flag_on(char const* name) noexcept
{
  char const* value = std::getenv(name);
  if (value == nullptr) return false;
  std::string_view const text{value};
  return text == "1" || text == "ON" || text == "on" || text == "On" || text == "TRUE" ||
         text == "true" || text == "True" || text == "YES" || text == "yes";
}

[[nodiscard]] platform_info compute_platform() noexcept
{
  platform_info info{};
  info.forced_compat = env_flag_on("CUFILE_FORCE_COMPAT_MODE");
  try {
    auto const* api        = cufile_api::get();
    info.library_available = api != nullptr;
    if (!info.library_available) return info;
    info.version = cufile_api::version();
    if (!cufile_api::open_driver()) return info;
    CUfileDrvProps_t props{};
    if (api->driver_get_properties(&props).err != CU_FILE_SUCCESS) return info;
    info.driver_opened         = true;
    info.dstatusflags          = props.nvfs.dstatusflags;
    info.compat_only           = info.dstatusflags == 0;
    info.max_direct_io_size_kb = props.nvfs.max_direct_io_size;
  } catch (std::exception const& e) {
    CUCASCADE_LOG_WARN("cufile: probing the cuFile library failed: {}", e.what());
  } catch (...) {
    CUCASCADE_LOG_WARN("cufile: probing the cuFile library failed with an unknown error");
  }
  return info;
}

}  // namespace

platform_info const& probe_platform() noexcept
{
  static platform_info const info = [] {
    auto result = compute_platform();
    if (!result.driver_opened) {
      CUCASCADE_LOG_INFO("cufile: cuFile unavailable (library {}, driver not opened)",
                         result.library_available ? "loaded" : "missing");
    } else if (result.compat_only || result.forced_compat) {
      CUCASCADE_LOG_INFO(
        "cufile: cuFile {} runs in COMPATIBILITY mode ({}): reads bounce through host memory "
        "inside cuFile",
        result.version,
        result.forced_compat ? "forced by the environment" : "no storage class reports GDS");
    } else {
      CUCASCADE_LOG_INFO("cufile: cuFile {} reports GPUDirect Storage (dstatusflags {:#x})",
                         result.version,
                         result.dstatusflags);
    }
    return result;
  }();
  return info;
}

bool gds_available() noexcept
{
  auto const& info = probe_platform();
  return info.driver_opened && !info.compat_only && !info.forced_compat;
}

}  // namespace cucascade::io::cufile
