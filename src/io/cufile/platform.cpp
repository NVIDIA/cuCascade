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
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/detail.hpp>
#include <cucascade/io/cufile/platform.hpp>
#include <cucascade/log/logging.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <cufile.h>

#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

namespace cucascade::io::cufile {

namespace detail {

bool force_compat_env_on(char const* value) noexcept
{
  if (value == nullptr) { return false; }
  constexpr char upper[] = "TRUE";
  for (int i = 0; i < 4; ++i) {
    // A null terminator cannot match a letter, so short values stop here.
    if (value[i] != upper[i] && value[i] != upper[i] + ('a' - 'A')) { return false; }
  }
  return true;
}

}  // namespace detail

namespace {

[[nodiscard]] platform_info compute_platform() noexcept
{
  platform_info info{};
  info.forced_compat = detail::force_compat_env_on(std::getenv("CUFILE_FORCE_COMPAT_MODE"));
  try {
    auto const* api        = cufile_api::get();
    info.library_available = api != nullptr;
    if (!info.library_available) return info;
    info.version = cufile_api::version();
    if (!cufile_api::open_driver()) return info;
    info.driver_opened = true;
    CUfileDrvProps_t props{};
    if (api->driver_get_properties(&props).err != CU_FILE_SUCCESS) {
      CUCASCADE_LOG_WARN("cufile: cuFileDriverGetProperties failed; the GDS mode is unknown");
      return info;
    }
    info.properties_available  = true;
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
    } else if (!result.properties_available) {
      CUCASCADE_LOG_INFO("cufile: cuFile {} driver opened, but its mode is unknown",
                         result.version);
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
  return info.driver_opened && info.properties_available && !info.compat_only &&
         !info.forced_compat;
}

// ---------------------------------------------------------------------------
// per-path capability
// ---------------------------------------------------------------------------

namespace {

using memory::path_storage_status;
using memory::storage_kind;

/// The @c CUfileDriverStatusFlags bit of @p cls, or -1 for @c storage_class::none.
[[nodiscard]] int status_bit(storage_class cls) noexcept
{
  switch (cls) {
    case storage_class::nvme: return CU_FILE_NVME_SUPPORTED;
    case storage_class::nvmeof: return CU_FILE_NVMEOF_SUPPORTED;
    case storage_class::nfs: return CU_FILE_NFS_SUPPORTED;
    case storage_class::lustre: return CU_FILE_LUSTRE_SUPPORTED;
    case storage_class::wekafs: return CU_FILE_WEKAFS_SUPPORTED;
    case storage_class::gpfs: return CU_FILE_GPFS_SUPPORTED;
    case storage_class::beegfs: return CU_FILE_BEEGFS_SUPPORTED;
    case storage_class::virtiofs: return CU_FILE_VIRTIOFS_SUPPORTED;
    case storage_class::none: break;
  }
  return -1;
}

[[nodiscard]] char const* class_name(storage_class cls) noexcept
{
  switch (cls) {
    case storage_class::nvme: return "NVMe";
    case storage_class::nvmeof: return "NVMe-oF";
    case storage_class::nfs: return "NFS";
    case storage_class::lustre: return "Lustre";
    case storage_class::wekafs: return "WekaFS";
    case storage_class::gpfs: return "GPFS";
    case storage_class::beegfs: return "BeeGFS";
    case storage_class::virtiofs: return "VirtioFS";
    case storage_class::none: break;
  }
  return "none";
}

/// The storage class of a network / shared filesystem type, or none.
[[nodiscard]] storage_class network_class(std::string_view fs_type) noexcept
{
  if (fs_type == "nfs" || fs_type == "nfs4" || fs_type == "vastnfs") return storage_class::nfs;
  if (fs_type == "lustre") return storage_class::lustre;
  if (fs_type == "wekafs") return storage_class::wekafs;
  if (fs_type == "gpfs") return storage_class::gpfs;
  if (fs_type == "beegfs") return storage_class::beegfs;
  if (fs_type == "virtiofs") return storage_class::virtiofs;
  return storage_class::none;
}

[[nodiscard]] char const* capability_name(path_capability capability) noexcept
{
  switch (capability) {
    case path_capability::unavailable: return "unavailable";
    case path_capability::compatibility_only: return "compatibility-only";
    case path_capability::direct_candidate: return "direct candidate";
    case path_capability::unknown: break;
  }
  return "unknown";
}

void set_verdict(path_info& info, path_capability capability, std::string reason)
{
  info.capability = capability;
  info.reason     = std::move(reason);
}

/// Log a verdict once per (device, verdict) pair: per mount, not per path.
void log_once(path_info const& info) noexcept
{
  try {
    static std::mutex mutex;
    static std::set<std::tuple<unsigned, unsigned, int>> logged;
    {
      std::lock_guard lock(mutex);
      if (!logged
             .emplace(info.storage.device_major,
                      info.storage.device_minor,
                      static_cast<int>(info.capability))
             .second) {
        return;
      }
    }
    CUCASCADE_LOG_INFO(
      "cufile: '{}' ({} on {}): GDS read capability {}: {}",
      info.storage.path,
      info.storage.filesystem_type.empty() ? "unknown fs" : info.storage.filesystem_type,
      info.storage.mount_point.empty() ? "unknown mount" : info.storage.mount_point,
      capability_name(info.capability),
      info.reason);
  } catch (...) {
    // Logging is best effort.
  }
}

void fill_policy(path_info& info, config const& cfg)
{
  info.compat_mode = cfg.compat_mode.value_or(compat::automatic);
  info.use_odirect = cfg.use_odirect;
  info.platform    = probe_platform();
}

void fill_from_object(path_info& info, cufile_io_object const& object)
{
  info.storage = memory::topology_discovery::discover_storage_for_descriptor(object.buffered_fd(),
                                                                             object.object_path());
  info.opened  = true;
  info.direct_open = object.direct_fd() >= 0;
  info.registered  = object.cufile_registered();
}

}  // namespace

namespace detail {

void decide_path_capability(path_info& info) noexcept
{
  try {
    auto const& storage       = info.storage;
    auto const& platform      = info.platform;
    info.matched_class        = storage_class::none;
    info.class_supported      = false;
    info.filesystem_supported = false;

    // The file itself.
    if (storage.status == path_storage_status::not_found ||
        storage.status == path_storage_status::not_regular_file ||
        storage.status == path_storage_status::inaccessible) {
      set_verdict(
        info, path_capability::unavailable, "not a readable regular file: " + storage.reason);
      return;
    }
    if (!info.opened) {
      set_verdict(info,
                  path_capability::unavailable,
                  "the cuFile reactor would not open the file: " + info.open_error);
      return;
    }

    // Policy.
    if (info.compat_mode == compat::on) {
      set_verdict(info,
                  path_capability::compatibility_only,
                  "compat_mode=on: cuFile is not used, every read goes through POSIX");
      return;
    }
    if (!info.use_odirect) {
      set_verdict(info,
                  path_capability::compatibility_only,
                  "use_odirect=false: the file is never registered with cuFile");
      return;
    }

    // Platform.
    if (!platform.library_available) {
      set_verdict(info, path_capability::unavailable, "libcufile cannot be loaded");
      return;
    }
    if (!platform.driver_opened) {
      set_verdict(info, path_capability::unavailable, "the cuFile driver did not open");
      return;
    }
    if (platform.forced_compat) {
      set_verdict(info,
                  path_capability::compatibility_only,
                  "CUFILE_FORCE_COMPAT_MODE forces cuFile compatibility mode");
      return;
    }
    if (!platform.properties_available) {
      set_verdict(info,
                  path_capability::unknown,
                  "cuFileDriverGetProperties failed: the driver's GDS modes are unknown");
      return;
    }
    if (platform.dstatusflags == 0) {
      set_verdict(info,
                  path_capability::compatibility_only,
                  "no storage class reports a real GDS mode (no nvidia-fs, P2PDMA or GDS-enabled "
                  "filesystem client): cuFile runs in compatibility mode");
      return;
    }

    // Storage.
    switch (storage.kind) {
      case storage_kind::local_block: {
        if (!storage.all_devices_nvme()) {
          set_verdict(info,
                      storage.mixed_devices() ? path_capability::unknown
                                              : path_capability::compatibility_only,
                      storage.mixed_devices()
                        ? "mixed device stack under " + storage.mount_point
                        : "the file is not on NVMe: no GDS storage class checked for it");
          return;
        }
        // NVMe namespaces reached without a PCI controller are fabrics-attached.
        bool pci = true;
        for (auto const& device : storage.devices) {
          if (device.pci_bus_id.empty()) pci = false;
        }
        info.matched_class = pci ? storage_class::nvme : storage_class::nvmeof;
        info.filesystem_supported =
          (storage.filesystem_type == "ext4" &&
           storage.super_options.find("data=journal") == std::string::npos) ||
          storage.filesystem_type == "xfs";
        break;
      }
      case storage_kind::network:
        info.matched_class        = network_class(storage.filesystem_type);
        info.filesystem_supported = info.matched_class != storage_class::none;
        if (info.matched_class == storage_class::none) {
          set_verdict(
            info,
            path_capability::compatibility_only,
            "network filesystem " + storage.filesystem_type + " has no GDS storage class");
          return;
        }
        break;
      case storage_kind::memory:
        set_verdict(info,
                    path_capability::compatibility_only,
                    "memory filesystem " + storage.filesystem_type + " has no GDS path");
        return;
      case storage_kind::unknown:
        info.matched_class        = network_class(storage.filesystem_type);
        info.filesystem_supported = info.matched_class != storage_class::none;
        if (info.matched_class == storage_class::none) {
          set_verdict(info,
                      path_capability::unknown,
                      "the file's storage is unknown: " +
                        (storage.reason.empty() ? std::string{"no detail"} : storage.reason));
          return;
        }
        break;
    }

    auto const bit = status_bit(info.matched_class);
    info.class_supported =
      bit >= 0 && (platform.dstatusflags & (1u << static_cast<unsigned>(bit))) != 0;
    std::string const cls = class_name(info.matched_class);
    if (!info.class_supported) {
      set_verdict(info,
                  path_capability::compatibility_only,
                  "the cuFile driver reports no GDS mode for " + cls);
      return;
    }
    if (!info.filesystem_supported) {
      set_verdict(info,
                  path_capability::compatibility_only,
                  "filesystem " + storage.filesystem_type + " is not served by GDS on " + cls);
      return;
    }
    if (!info.direct_open) {
      set_verdict(info,
                  path_capability::compatibility_only,
                  "the file has no O_DIRECT descriptor (refused by the filesystem)");
      return;
    }
    if (!info.registered) {
      set_verdict(info,
                  path_capability::compatibility_only,
                  "cuFileHandleRegister did not register the file");
      return;
    }
    set_verdict(info,
                path_capability::direct_candidate,
                "the cuFile driver reports GDS for " + cls + ", " + storage.filesystem_type +
                  " is supported there, and the file is registered O_DIRECT");
  } catch (...) {
    info.capability = path_capability::unknown;
  }
}

}  // namespace detail

path_info probe_object(cufile_io_object const& object, config const& cfg) noexcept
{
  path_info info;
  try {
    fill_policy(info, cfg);
    fill_from_object(info, object);
    detail::decide_path_capability(info);
    log_once(info);
  } catch (...) {
    info.capability = path_capability::unknown;
    info.reason     = "probing the file failed";
  }
  return info;
}

path_info probe_path(std::string const& path, config const& cfg) noexcept
{
  path_info info;
  try {
    fill_policy(info, cfg);
    std::unique_ptr<cufile_io_object> object;
    try {
      object = cufile_reactor::create_io_object(path, cfg.use_odirect, cfg.compat_mode);
    } catch (std::exception const& e) {
      info.open_error = e.what();
    } catch (...) {
      info.open_error = "unknown error";
    }
    if (object != nullptr) {
      fill_from_object(info, *object);
    } else {
      info.storage = memory::topology_discovery::discover_storage_for_path(path);
    }
    detail::decide_path_capability(info);
    log_once(info);
  } catch (...) {
    info.capability = path_capability::unknown;
    info.reason     = "probing the file failed";
  }
  return info;
}

}  // namespace cucascade::io::cufile
