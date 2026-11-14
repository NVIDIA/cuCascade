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
#include <cucascade/io/cufile/platform.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace cucascade::io::cufile {

/// Reactors per GPU when the pool reads mostly from a network filesystem
/// (Storage Scale, NFS, Lustre, WekaFS, BeeGFS over RDMA).  Little's law: 50 GB/s
/// per GPU at 2-5 ms per read needs 100-250 MB in flight, 12-31 reads of 8 MB or
/// 25-62 of 4 MB; one reactor keeps one read in flight.
inline constexpr std::size_t network_reactors_per_device = 32;
/// Reactors per GPU when the pool reads mostly from local block storage (NVMe at
/// ~0.1 ms needs a few reads in flight per drive).
inline constexpr std::size_t local_reactors_per_device = 8;
/// Reactors per GPU when the storage is not known.
inline constexpr std::size_t default_reactors_per_device = 16;

/**
 * @brief One GPU's share of a planned cuFile pool.
 */
struct device_group_plan {
  int device{-1};                ///< CUDA ordinal.
  std::string pci_bus_id;        ///< As CUDA reports it, e.g. "0000:1B:00.0".
  int numa_node{-1};             ///< The GPU's NUMA node (-1 if unknown).
  std::size_t n_reactors{0};     ///< Reactors serving it: its queue depth.
  std::vector<int> cpus;         ///< CPUs those reactors are pinned to; empty = unpinned.
  std::size_t staging_slots{0};  ///< Registered staging slots reserved for it at start.
  /// Cap on registered staging for this device, in bytes (0 = none): the config's
  /// cap, half of the BAR1 free at planning time, and cuFile's pinned-memory limit.
  std::size_t staging_cap_bytes{0};
  std::string reason;  ///< How the numbers were chosen, in words.
};

/**
 * @brief A cuFile reactor pool: one group per GPU, or one flat group.
 */
struct pool_plan {
  /// One entry per GPU served; empty for a flat pool (no GPU, or a configured
  /// total via @c io_config::cufile_n_reactors).
  std::vector<device_group_plan> groups;
  std::size_t flat_reactors{0};  ///< Reactors of a flat pool (when @c groups is empty).
  /// The staging piece size the pool should use: the config's, capped at cuFile's
  /// @c max_direct_io_size (cuFile issues a larger call as sequential chunks of
  /// that size, so smaller pieces spread over more reactors read faster).
  std::size_t staging_piece_bytes{0};
  std::string reason;  ///< Overall summary, in words.
  /// Setup problems found while planning that cost GDS throughput, each a
  /// sentence an operator can act on (e.g. an empty @c rdma_dev_addr_list with a
  /// GDS-enabled Storage Scale client).  Logged as warnings by the factory.
  std::vector<std::string> warnings;

  /// Reactors the pool runs.
  [[nodiscard]] std::size_t total_reactors() const noexcept
  {
    if (groups.empty()) return flat_reactors;
    std::size_t total = 0;
    for (auto const& group : groups) {
      total += group.n_reactors;
    }
    return total;
  }

  /// The staging cap of @p device (see @ref device_group_plan::staging_cap_bytes),
  /// or @p fallback when the plan has no group for it.
  [[nodiscard]] std::size_t staging_cap_of(int device, std::size_t fallback) const noexcept
  {
    for (auto const& group : groups) {
      if (group.device == device) return group.staging_cap_bytes;
    }
    return fallback;
  }
};

/**
 * @brief Plan a cuFile pool from the discovered topology: a group of reactors per
 *        visible GPU, pinned to the CPUs local to it and sized by Little's law,
 *        with a registered staging budget the device can hold.
 *
 * - Reactors per GPU: @c cfg.reactors_per_device, or, when 0,
 *   @ref network_reactors_per_device / @ref local_reactors_per_device /
 *   @ref default_reactors_per_device by what @c cfg.storage_hint_path is stored on.
 * - CPUs: the GPU's CPU affinity from @p topology (matched by PCI bus ID, not by
 *   index), intersected with the calling thread's affinity; unpinned when
 *   @c cfg.pin_to_device_cpus is off, the topology is unknown or the
 *   intersection is empty.
 * - Staging (only when @c cfg.staging_mode is @c always, or @c automatic and
 *   @p platform reports GDS): reactors x @c staging_slots_per_reactor slots,
 *   bounded by @c cfg.max_staging_bytes_per_device, half the BAR1 free at
 *   discovery and cuFile's @c max_device_pinned_mem_size; pieces no larger than
 *   cuFile's @c max_direct_io_size.
 * - Warnings: GDS setup problems visible from here (see @ref pool_plan::warnings),
 *   e.g. Storage Scale storage without GDS, or GDS without RDMA addresses.
 *
 * Groups are planned for @c cfg.devices, or for every visible device when that
 * is empty (unknown devices are reported in @ref pool_plan::warnings).
 *
 * Calls the CUDA runtime (device count and PCI bus IDs) but creates nothing.
 * Never throws: whatever cannot be learned is reported in the reasons.
 *
 * @param cfg The cuFile backend's configuration.
 * @param topology The discovered topology, or null.
 * @param platform The cuFile platform probe (see @ref probe_platform).
 */
[[nodiscard]] pool_plan plan_cufile_pool(config const& cfg,
                                         memory::system_topology_info const* topology,
                                         platform_info const& platform) noexcept;

/// Render @p plan for a log line.
[[nodiscard]] std::string describe(pool_plan const& plan);

}  // namespace cucascade::io::cufile
