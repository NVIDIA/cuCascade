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

#include <cucascade/memory/net_topology.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/**
 * @file gds_topology.hpp
 * @brief How GPUDirect Storage is set up on this host, as the kernel and cuFile's
 *        configuration file expose it.
 *
 * GDS throughput is decided as much by the host's setup as by the code issuing
 * the reads: whether the nvidia-fs driver is loaded (without it cuFile only runs
 * in compatibility mode), whether GPUDirect RDMA peer memory is available for a
 * network filesystem client (IBM Storage Scale, NFS over RDMA, Lustre, ...),
 * and how @c cufile.json sizes cuFile's direct I/O, bounce buffers and pinned
 * memory.  This discovery is passive -- procfs, sysfs and the configuration file
 * only; it never loads libcufile or initializes CUDA -- so it belongs to the
 * topology library.  What libcufile itself reports once loaded (the effective
 * parameters, which also honour environment overrides) is
 * @c cucascade::io::cufile::platform_info.
 */

namespace cucascade::memory {

/**
 * @brief The settings of cuFile's configuration file (@c cufile.json) that decide
 *        GDS read throughput.
 *
 * Each field is unset when the file does not set it: cuFile then uses its
 * built-in default (for libcufile 1.x: @c max_direct_io_size_kb 16384,
 * @c max_device_cache_size_kb 131072, @c per_buffer_cache_size_kb 1024,
 * @c max_device_pinned_mem_size_kb 33554432).  Sizes are in KiB, as in the file.
 */
struct cufile_json_settings {
  // -- "properties" ---------------------------------------------------------
  std::optional<std::size_t> max_direct_io_size_kb;          ///< Largest single direct I/O.
  std::optional<std::size_t> max_device_cache_size_kb;       ///< GPU bounce-buffer pool per GPU.
  std::optional<std::size_t> per_buffer_cache_size_kb;       ///< One bounce buffer.
  std::optional<std::size_t> max_device_pinned_mem_size_kb;  ///< cuFileBufRegister cap per GPU.
  std::optional<bool> use_poll_mode;                         ///< Poll small reads to completion.
  std::optional<std::size_t> poll_mode_max_size_kb;          ///< Largest polled I/O.
  std::optional<bool> allow_compat_mode;                     ///< POSIX fallback allowed.
  std::optional<bool> use_pci_p2pdma;                        ///< Kernel PCI P2PDMA path.
  std::vector<std::string> rdma_dev_addr_list;               ///< RDMA NIC addresses cuFile uses.
  std::optional<std::string> rdma_load_balancing_policy;     ///< e.g. "RoundRobin".
  std::optional<bool> rdma_dynamic_routing;                  ///< Route through other GPUs/NICs.

  // -- "execution" (cuFile's internal I/O thread pool) ----------------------
  std::optional<bool> parallel_io;                      ///< Split large I/O across threads.
  std::optional<std::size_t> max_io_threads;            ///< Threads in that pool.
  std::optional<std::size_t> max_io_queue_depth;        ///< Its queue depth.
  std::optional<std::size_t> min_io_threshold_size_kb;  ///< Smallest I/O it splits.
  std::optional<std::size_t> max_request_parallelism;   ///< Pieces one request is split into.

  // -- "fs" -----------------------------------------------------------------
  std::optional<bool> gpfs_gds_write_support;  ///< fs.gpfs.gds_write_support.
  std::optional<bool> gpfs_gds_async_support;  ///< fs.gpfs.gds_async_support.
  /// fs.lustre.posix_gds_min_kb: Lustre I/O below this goes through POSIX.
  std::optional<std::size_t> lustre_posix_gds_min_kb;
  /// fs.beegfs.posix_gds_min_kb: BeeGFS I/O below this goes through POSIX.
  std::optional<std::size_t> beegfs_posix_gds_min_kb;

  // -- "denylist": GDS is disabled for whatever is listed --------------------
  std::vector<std::string> denied_drivers;      ///< denylist.drivers
  std::vector<std::string> denied_devices;      ///< denylist.devices
  std::vector<std::string> denied_mounts;       ///< denylist.mounts
  std::vector<std::string> denied_filesystems;  ///< denylist.filesystems

  // -- "profile" / "logging" ------------------------------------------------
  std::optional<std::size_t> cufile_stats;   ///< profile.cufile_stats level (0-3).
  std::optional<std::string> logging_level;  ///< logging.level.
};

/**
 * @brief Where cuFile's configuration comes from and what it says.
 *
 * cuFile starts from its built-in defaults, applies @c /etc/cufile.json, then
 * the file @c $CUFILE_ENV_PATH_JSON names (read with @c secure_getenv), whose
 * keys win one by one; a missing file is skipped.  The files are JSON with
 * @c // and @c /&lowast; &lowast;/ comments.  @c settings is the merged result.
 */
struct cufile_config_file {
  std::string path;           ///< The system file (@c /etc/cufile.json).
  std::string override_path;  ///< The file @c CUFILE_ENV_PATH_JSON names; empty if none.
  /// @c override_path came from @c CUFILE_ENV_PATH_JSON (not from a caller).
  bool from_environment{false};
  bool examined{false};  ///< Discovery looked for the files (false: never looked).
  bool found{false};     ///< At least one of the files exists and could be read.
  /// Every file found was parsed, so @c settings is meaningful (also when no file
  /// exists: then it is all unset, i.e. cuFile's defaults).
  bool parsed{false};
  std::string error;              ///< Why a file was not found or parsed; empty otherwise.
  cufile_json_settings settings;  ///< The merged settings (unset = cuFile's default).
};

/**
 * @brief One GPU's line in @c /proc/driver/nvidia-fs/stats.
 */
struct nvidia_fs_gpu_stats {
  std::string pci_bus_id;              ///< e.g. "0000:be:00.0".
  std::size_t registered_mib{0};       ///< Memory registered with cuFileBufRegister.
  std::size_t cache_mib{0};            ///< nvidia-fs bounce cache in use.
  std::size_t max_pinned_mib{0};       ///< Peak pinned (BAR1-mapped) memory.
  std::size_t cross_root_port_pct{0};  ///< Share of I/O crossing a PCIe root port.
};

/**
 * @brief The nvidia-fs kernel driver, which gives cuFile its direct DMA path
 *        for NVMe and for RDMA filesystem clients.
 */
struct nvidia_fs_info {
  bool loaded{false};       ///< The @c nvidia_fs module is live (@c /proc/modules).
  std::string version;      ///< Driver version, e.g. "2.20.5"; empty if unknown.
  std::string gds_version;  ///< "GDS Version" in the stats file; empty if unknown.
  /// "Mellanox PeerDirect Supported" in the stats file: whether RDMA filesystem
  /// clients (Storage Scale, NFSoRDMA, Lustre, ...) can DMA into GPU memory.
  /// Unset when the stats file is unreadable or does not report it (recent
  /// nvidia-fs releases no longer print the line; see @c peer_memory_loaded).
  std::optional<bool> peer_direct_supported;
  std::optional<bool> io_stats_enabled;   ///< Per-I/O statistics are being collected.
  std::vector<nvidia_fs_gpu_stats> gpus;  ///< Per-GPU registration and cache usage.
};

/**
 * @brief How GPUDirect Storage is set up on this host (passive discovery).
 */
struct gds_setup_info {
  nvidia_fs_info nvidia_fs;  ///< The nvidia-fs driver.
  /// GPUDirect RDMA peer memory is available: @c nvidia_peermem (or the legacy
  /// @c nv_peer_mem) is live.  RDMA filesystem clients need it, or nvidia-fs's
  /// PeerDirect support, to DMA into GPU memory.
  bool peer_memory_loaded{false};
  bool rdma_core_loaded{false};      ///< @c ib_core is live (an RDMA stack is loaded).
  bool gpfs_client_loaded{false};    ///< IBM Storage Scale (@c mmfs26 / @c mmfslinux) is live.
  bool lustre_client_loaded{false};  ///< The Lustre client (@c lustre) is live.
  bool beegfs_client_loaded{false};  ///< The BeeGFS client (@c beegfs) is live.
  bool wekafs_client_loaded{false};  ///< The WekaFS client (@c wekafsio) is live.
  bool nfs_rdma_loaded{false};       ///< NFS over RDMA (@c rpcrdma) is live.
  /// @c CUFILE_FORCE_COMPAT_MODE begins with "true" (any case), as cuFile reads
  /// it: every cuFile I/O then bounces through host memory.
  bool force_compat_env{false};
  cufile_config_file cufile_json;  ///< cuFile's configuration file.

  /// Whether the kernel side of a direct GDS path exists: nvidia-fs is loaded.
  /// (cuFile can also DMA through the kernel's PCI P2PDMA without it, on NVMe
  /// only, when @c use_pci_p2pdma is set.)  Not a guarantee for any file.
  [[nodiscard]] bool kernel_path_present() const noexcept { return nvidia_fs.loaded; }
};

/**
 * @brief Discover how GPUDirect Storage is set up.
 *
 * Reads @c /proc/modules, @c /sys/module/nvidia_fs/version,
 * @c /proc/driver/nvidia-fs/{version,stats} and cuFile's configuration file.
 * Passive: never loads libcufile or the CUDA driver.  Never throws; whatever
 * cannot be read is left at its default.
 *
 * @param roots Where procfs / sysfs are mounted; tests point them at a fake tree.
 * @param cufile_json_path The system configuration file; empty means
 *        @c /etc/cufile.json.
 * @param override_json_path The file merged on top; unset reads
 *        @c CUFILE_ENV_PATH_JSON the way cuFile does, empty means none.
 */
[[nodiscard]] gds_setup_info discover_gds_setup(
  discovery_roots const& roots                         = {},
  std::string_view cufile_json_path                    = {},
  std::optional<std::string> const& override_json_path = std::nullopt);

/**
 * @brief Read cuFile's configuration the way cuFile does: @p path, then
 *        @p override_path (if not empty) merged on top key by key.  A missing
 *        file is skipped.  Never throws.
 */
[[nodiscard]] cufile_config_file read_cufile_config(std::string const& path,
                                                    std::string const& override_path = {});

}  // namespace cucascade::memory
