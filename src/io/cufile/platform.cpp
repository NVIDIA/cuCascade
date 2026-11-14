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

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

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

/// Fill @p out from cuFile's parameter getters, when the library has them.
void read_parameters(cufile_api const& api, platform_info::parameters_info& out) noexcept
{
  out.available = api.get_size_parameter != nullptr && api.get_bool_parameter != nullptr;
  if (!out.available) return;
  auto size = [&api](size_parameter param) -> std::optional<std::size_t> {
    std::size_t value = 0;
    if (api.get_size_parameter(param, &value).err != CU_FILE_SUCCESS) return std::nullopt;
    return value;
  };
  auto flag = [&api](bool_parameter param) -> std::optional<bool> {
    bool value = false;
    if (api.get_bool_parameter(param, &value).err != CU_FILE_SUCCESS) return std::nullopt;
    return value;
  };
  auto text = [&api](string_parameter param) -> std::optional<std::string> {
    if (api.get_string_parameter == nullptr) return std::nullopt;
    std::array<char, 256> buffer{};
    if (api.get_string_parameter(param, buffer.data(), static_cast<int>(buffer.size())).err !=
        CU_FILE_SUCCESS) {
      return std::nullopt;
    }
    buffer.back() = '\0';
    return std::string{buffer.data()};
  };
  try {
    out.parallel_io              = flag(bool_parameter::execution_parallel_io);
    out.max_io_threads           = size(size_parameter::execution_max_io_threads);
    out.max_io_queue_depth       = size(size_parameter::execution_max_io_queue_depth);
    out.min_io_threshold_size_kb = size(size_parameter::execution_min_io_threshold_kb);
    out.max_request_parallelism  = size(size_parameter::execution_max_request_parallelism);
    out.use_poll_mode            = flag(bool_parameter::use_poll_mode);
    out.allow_compat_mode        = flag(bool_parameter::allow_compat_mode);
    out.force_compat_mode        = flag(bool_parameter::force_compat_mode);
    out.use_pci_p2pdma           = flag(bool_parameter::use_pci_p2pdma);
    out.io_batch_size            = size(size_parameter::io_batch_size);
    out.profile_stats            = size(size_parameter::profile_stats);
    out.rdma_transport           = text(string_parameter::rdma_transport);
    out.logging_level            = text(string_parameter::logging_level);
  } catch (...) {
    // Best effort: keep what was read.
  }
}

[[nodiscard]] platform_info compute_platform() noexcept
{
  platform_info info{};
  info.forced_compat = detail::force_compat_env_on(std::getenv("CUFILE_FORCE_COMPAT_MODE"));
  try {
    auto const* api        = cufile_api::loaded();
    info.library_available = api != nullptr;
    if (!info.library_available) return info;
    info.version = cufile_api::version();
    if (!cufile_api::open_loaded_driver()) return info;
    info.driver_opened = true;
    CUfileDrvProps_t props{};
    if (api->driver_get_properties(&props).err != CU_FILE_SUCCESS) {
      CUCASCADE_LOG_WARN("cufile: cuFileDriverGetProperties failed; the GDS mode is unknown");
      return info;
    }
    info.properties_available          = true;
    info.dstatusflags                  = props.nvfs.dstatusflags;
    info.compat_only                   = info.dstatusflags == 0;
    info.max_direct_io_size_kb         = props.nvfs.max_direct_io_size;
    info.nvfs_major_version            = props.nvfs.major_version;
    info.nvfs_minor_version            = props.nvfs.minor_version;
    info.poll_threshold_size_kb        = props.nvfs.poll_thresh_size;
    info.dcontrolflags                 = props.nvfs.dcontrolflags;
    info.fflags                        = props.fflags;
    info.max_device_cache_size_kb      = props.max_device_cache_size;
    info.per_buffer_cache_size_kb      = props.per_buffer_cache_size;
    info.max_device_pinned_mem_size_kb = props.max_device_pinned_mem_size;
    info.max_batch_io_size             = props.max_batch_io_size;
    info.max_batch_io_timeout_msecs    = props.max_batch_io_timeout_msecs;
    read_parameters(*api, info.parameters);
    if (info.parameters.force_compat_mode.value_or(false)) info.forced_compat = true;
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
    if (result.properties_available) {
      auto const& p = result.parameters;
      CUCASCADE_LOG_INFO(
        "cufile: nvidia-fs {}.{}, max direct I/O {} KiB, bounce cache {} KiB/GPU in {} KiB "
        "buffers, pinned limit {} KiB/GPU, poll mode {}, parallel_io {}, max_io_threads {}, "
        "max_request_parallelism {}",
        result.nvfs_major_version,
        result.nvfs_minor_version,
        result.max_direct_io_size_kb,
        result.max_device_cache_size_kb,
        result.per_buffer_cache_size_kb,
        result.max_device_pinned_mem_size_kb,
        (result.dcontrolflags & 1U) != 0 ? "on" : "off",
        p.parallel_io ? (*p.parallel_io ? "on" : "off") : "unknown",
        p.max_io_threads ? std::to_string(*p.max_io_threads) : std::string{"unknown"},
        p.max_request_parallelism ? std::to_string(*p.max_request_parallelism)
                                  : std::string{"unknown"});
    }
    return result;
  }();
  return info;
}

namespace {
/// -1: no override; 0 / 1: what gds_available() reports (tests only).
std::atomic<int> gds_available_override{-1};
}  // namespace

namespace detail {
void set_gds_available_for_testing(std::optional<bool> value) noexcept
{
  gds_available_override.store(value ? (*value ? 1 : 0) : -1, std::memory_order_release);
}
}  // namespace detail

bool gds_available() noexcept
{
  if (auto const forced = gds_available_override.load(std::memory_order_acquire); forced >= 0) {
    return forced == 1;
  }
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

/// @c CU_FILE_VIRTIOFS_SUPPORTED, which CUDA 12.9's cufile.h (cuFile 1.14) does
/// not declare; the driver reports the bit whatever header this was built with.
constexpr int virtiofs_supported_bit = 13;

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
    case storage_class::virtiofs: return virtiofs_supported_bit;
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
  info.gds         = memory::discover_gds_setup();
}

/// Whether @p options (a comma-separated mount option list) holds @p option.
[[nodiscard]] bool has_option(std::string const& options, std::string_view option) noexcept
{
  std::size_t start = 0;
  while (start <= options.size()) {
    auto const comma = std::min(options.find(',', start), options.size());
    if (std::string_view{options}.substr(start, comma - start) == option) return true;
    start = comma + 1;
  }
  return false;
}

[[nodiscard]] bool listed(std::vector<std::string> const& list, std::string const& value) noexcept
{
  return !value.empty() && std::find(list.begin(), list.end(), value) != list.end();
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
        // GDS serves NFS only over RDMA.
        if (info.matched_class == storage_class::nfs &&
            !has_option(storage.super_options, "proto=rdma") &&
            !has_option(storage.super_options, "proto=rdma6") &&
            !has_option(storage.mount_options, "proto=rdma") &&
            !has_option(storage.mount_options, "proto=rdma6")) {
          info.filesystem_supported = false;
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
                  info.matched_class == storage_class::nfs
                    ? "NFS mount " + storage.mount_point +
                        " is not over RDMA (no proto=rdma): GDS serves NFS over RDMA only"
                    : "filesystem " + storage.filesystem_type + " is not served by GDS on " + cls);
      return;
    }
    // Without any cuFile configuration file, cuFile runs on its built-in defaults,
    // whose RDMA address list is empty.
    if (auto const& json = info.gds.cufile_json;
        info.matched_class == storage_class::gpfs && json.examined && !json.found) {
      set_verdict(info,
                  path_capability::compatibility_only,
                  "no cuFile configuration file (" + json.path +
                    "): Storage Scale GDS reads need rdma_dev_addr_list and fall back to "
                    "compatibility mode without it");
      return;
    }
    if (auto const& json = info.gds.cufile_json; json.parsed) {
      if (listed(json.settings.denied_filesystems, storage.filesystem_type) ||
          listed(json.settings.denied_mounts, storage.mount_point)) {
        set_verdict(info,
                    path_capability::compatibility_only,
                    json.path + " denylists " +
                      (listed(json.settings.denied_mounts, storage.mount_point)
                         ? "mount " + storage.mount_point
                         : "filesystem " + storage.filesystem_type) +
                      ": cuFile does not use GDS for it");
        return;
      }
      if (info.matched_class == storage_class::gpfs && json.settings.rdma_dev_addr_list.empty()) {
        set_verdict(info,
                    path_capability::compatibility_only,
                    json.path +
                      " lists no rdma_dev_addr_list: Storage Scale GDS reads need the client's "
                      "RDMA addresses and fall back to compatibility mode without them");
        return;
      }
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
