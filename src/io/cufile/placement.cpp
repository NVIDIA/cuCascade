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

#include <cucascade/exec/thread_util.hpp>
#include <cucascade/io/cufile/placement.hpp>
#include <cucascade/memory/net_topology.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <cuda_runtime.h>

#include <cufile.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <exception>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <vector>

namespace cucascade::io::cufile {

namespace {

/// A PCI address, comparable whatever its spelling: NVML writes the domain with
/// eight hex digits ("00000000:1B:00.0"), CUDA and sysfs with four.
struct pci_address {
  unsigned domain{0};
  unsigned bus{0};
  unsigned device{0};
  unsigned function{0};

  bool operator==(pci_address const&) const = default;
};

[[nodiscard]] std::optional<pci_address> parse_pci(std::string_view text) noexcept
{
  // domain ':' bus ':' device '.' function, all hex.
  pci_address out;
  std::array<unsigned*, 4> const fields{&out.domain, &out.bus, &out.device, &out.function};
  std::array<char, 3> const separators{':', ':', '.'};
  auto const* cursor = text.data();
  auto const* end    = text.data() + text.size();
  for (std::size_t i = 0; i < fields.size(); ++i) {
    auto const [next, ec] = std::from_chars(cursor, end, *fields[i], 16);
    if (ec != std::errc{} || next == cursor) return std::nullopt;
    cursor = next;
    if (i < separators.size()) {
      if (cursor == end || *cursor != separators[i]) return std::nullopt;
      ++cursor;
    }
  }
  return out;
}

[[nodiscard]] memory::gpu_topology_info const* find_gpu(
  memory::system_topology_info const* topology, std::string const& pci_bus_id) noexcept
{
  if (topology == nullptr) return nullptr;
  auto const wanted = parse_pci(pci_bus_id);
  if (!wanted) return nullptr;
  for (auto const& gpu : topology->gpus) {
    if (parse_pci(gpu.pci_bus_id) == wanted) return &gpu;
  }
  return nullptr;
}

/// Whether @p platform reports a real GDS mode for storage class bit @p bit.
[[nodiscard]] bool class_bit_set(platform_info const& platform, int bit) noexcept
{
  return platform.properties_available &&
         (platform.dstatusflags & (1U << static_cast<unsigned>(bit))) != 0;
}

/// Setup problems that cost GDS throughput, as far as the topology and the
/// storage hint show them.
void collect_warnings(memory::system_topology_info const* topology,
                      platform_info const& platform,
                      memory::path_storage_info const* hint,
                      std::vector<std::string>& out)
{
  bool const on_gpfs  = hint != nullptr && hint->filesystem_type == "gpfs";
  bool const gpfs_gds = class_bit_set(platform, CU_FILE_GPFS_SUPPORTED);
  if (on_gpfs && platform.driver_opened && !gpfs_gds) {
    out.push_back(
      "storage_hint_path is on IBM Storage Scale but cuFile reports no GDS mode for it: every "
      "read goes through cuFile's compatibility mode (host bounce).  Check `gdscheck -p`, "
      "verbsGPUDirectStorage=yes / verbsRdma=enable on the cluster, nvidia_peermem or DMA-BUF, "
      "and that GDS clients are supported on this CPU architecture (IBM documents x86 only)");
  }
  if (topology != nullptr) {
    auto const& gds = topology->gds;
    if (gpfs_gds && gds.cufile_json.parsed && gds.cufile_json.settings.rdma_dev_addr_list.empty()) {
      out.push_back("cuFile reports GDS for Storage Scale but " + gds.cufile_json.path +
                    " lists no rdma_dev_addr_list: Storage Scale GDS reads fall back to "
                    "compatibility mode without RDMA client addresses (set them to the IPs of the "
                    "RDMA ports in verbsPorts)");
    }
    if (gpfs_gds && !gds.peer_memory_loaded) {
      out.push_back(
        "cuFile reports GDS for Storage Scale but nvidia_peermem is not loaded: GPU memory must "
        "be registered with the HCA through DMA-BUF instead (check `gdscheck -p`)");
    }
    if (on_gpfs && gds.cufile_json.parsed) {
      auto const& denied = gds.cufile_json.settings.denied_filesystems;
      if (std::find(denied.begin(), denied.end(), "gpfs") != denied.end()) {
        out.push_back(gds.cufile_json.path +
                      " denylists the gpfs filesystem: cuFile never uses GDS for it");
      }
    }
    if (gds.force_compat_env) {
      out.push_back(
        "CUFILE_FORCE_COMPAT_MODE is set: every cuFile read bounces through host memory");
    }
  }
  // cuFile moves one call in sequential chunks of max_direct_io_size, so that is
  // all one reactor keeps in flight per call on a GDS path.
  bool const gds = platform.driver_opened && platform.properties_available &&
                   !platform.compat_only && !platform.forced_compat;
  if (gds && platform.max_direct_io_size_kb != 0 && platform.max_direct_io_size_kb < 4096) {
    out.push_back("cuFile's max_direct_io_size is " +
                  std::to_string(platform.max_direct_io_size_kb) +
                  " KiB: each call keeps at most that in flight, so a deep pool is needed; raise "
                  "properties.max_direct_io_size_kb (up to 16384) in cufile.json");
  }
  if (platform.properties_available && (platform.dcontrolflags & 1U) != 0 && on_gpfs) {
    out.push_back(
      "cuFile poll mode is on: Storage Scale does not support it (requests are converted to "
      "synchronous GDS I/O); set properties.use_poll_mode false");
  }
}

/// Reactors per GPU and why.
[[nodiscard]] std::size_t reactors_per_device(config const& cfg,
                                              memory::path_storage_info const* hint,
                                              std::string& why)
{
  if (cfg.reactors_per_device != 0) {
    why = std::to_string(cfg.reactors_per_device) + " reactors (configured)";
    return cfg.reactors_per_device;
  }
  if (hint == nullptr) {
    why = std::to_string(default_reactors_per_device) +
          " reactors (storage unknown: no storage_hint_path)";
    return default_reactors_per_device;
  }
  auto const& storage = *hint;
  switch (storage.kind) {
    case memory::storage_kind::network:
      why = std::to_string(network_reactors_per_device) + " reactors (" + storage.filesystem_type +
            " on " + storage.mount_point +
            " is a network filesystem: deep queues hide its latency)";
      return network_reactors_per_device;
    case memory::storage_kind::local_block:
      why = std::to_string(local_reactors_per_device) + " reactors (" + storage.filesystem_type +
            " on " + storage.mount_point + " is local block storage)";
      return local_reactors_per_device;
    default: break;
  }
  why = std::to_string(default_reactors_per_device) + " reactors (storage of '" +
        cfg.storage_hint_path + "' unknown" +
        (storage.reason.empty() ? std::string{} : ": " + storage.reason) + ")";
  return default_reactors_per_device;
}

/// Whether the optional feature block @p block is enabled and valid.  An invalid one is
/// reported in @p plan's warnings and planned as disabled (the reactor rejects it later).
template <typename Block>
[[nodiscard]] bool feature_valid(Block const& block,
                                 char const* name,
                                 char const* what,
                                 pool_plan& plan)
{
  if (!block.enabled) return false;
  try {
    block.validate();
    return true;
  } catch (std::exception const& e) {
    plan.warnings.push_back(std::string{name} + " is invalid (" + e.what() +
                            "): the pool is planned without " + what);
    return false;
  }
}

[[nodiscard]] std::vector<int> intersect_in_order(std::vector<int> const& cpus,
                                                  std::vector<int> const& allowed)
{
  if (allowed.empty()) return cpus;  // unknown inherited affinity: trust the GPU's
  std::vector<int> out;
  for (int cpu : cpus) {
    if (std::find(allowed.begin(), allowed.end(), cpu) != allowed.end()) out.push_back(cpu);
  }
  return out;
}

}  // namespace

pool_plan plan_cufile_pool(config const& cfg,
                           memory::system_topology_info const* topology,
                           platform_info const& platform) noexcept
{
  pool_plan plan;
  try {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
      std::ignore = cudaGetLastError();
      plan.flat_reactors =
        cfg.reactors_per_device != 0 ? cfg.reactors_per_device : default_reactors_per_device;
      plan.reason = "no CUDA device visible: one flat pool of " +
                    std::to_string(plan.flat_reactors) + " reactors";
      return plan;
    }

    std::optional<memory::path_storage_info> hint;
    if (!cfg.storage_hint_path.empty()) {
      hint = memory::topology_discovery::discover_storage_for_path(cfg.storage_hint_path);
    }
    // An invalid feature block is planned as disabled, like any other planning problem.
    bool const fused = feature_valid(cfg.fused_reads, "fused_reads", "fused-read lanes", plan);
    bool const batch = feature_valid(cfg.host_batch, "host_batch", "host batch reactors", plan);

    std::string sizing;
    auto per_device = reactors_per_device(cfg, hint ? &*hint : nullptr, sizing);
    if (fused && cfg.reactors_per_device == 0) {
      per_device = local_reactors_per_device;
      sizing     = std::to_string(per_device) +
               " synchronous reactors (fused reads on: the lanes carry the device reads, the "
               "synchronous reactors serve writes and fallbacks)";
    }
    bool const gds = platform.driver_opened && platform.properties_available &&
                     !platform.compat_only && !platform.forced_compat;
    bool const stage =
      cfg.staging_mode == staging::always || (cfg.staging_mode == staging::automatic && gds);
    collect_warnings(topology, platform, hint ? &*hint : nullptr, plan.warnings);

    // cuFile moves a call in sequential chunks of max_direct_io_size: pieces no
    // larger than that spread over more reactors instead -- but never below a
    // size cuFile still serves (gds_threshold) or 1 MiB.
    auto effective = cfg;
    if (stage && platform.max_direct_io_size_kb != 0) {
      auto const floor = std::max<std::size_t>(
        1UL << 20, (cfg.gds_threshold + IO_BLOCK_SIZE - 1) & ~(IO_BLOCK_SIZE - 1));
      auto const chunk =
        std::max(floor, (platform.max_direct_io_size_kb * 1024) & ~(IO_BLOCK_SIZE - 1));
      if (chunk < effective.staging_piece_bytes) effective.staging_piece_bytes = chunk;
    }
    plan.staging_piece_bytes  = effective.staging_piece_bytes;
    auto const allowed        = exec::thread_util::get_current_thread_affinity();
    auto const slot_footprint = effective.staging_slot_bytes() + (64UL << 10);
    auto const lane_footprint = cfg.fused_reads.slot_bytes + (64UL << 10);
    if (fused) plan.lane_slot_bytes = cfg.fused_reads.slot_bytes;
    if (batch) plan.host_batch_reactors = cfg.host_batch.reactors;

    std::vector<int> devices;
    for (int device : cfg.devices) {
      if (device >= 0 && device < count &&
          std::find(devices.begin(), devices.end(), device) == devices.end()) {
        devices.push_back(device);
      } else {
        plan.warnings.push_back("devices lists " + std::to_string(device) +
                                ", which is not a visible CUDA device (or repeats): ignored");
      }
    }
    if (devices.empty()) {
      for (int device = 0; device < count; ++device) {
        devices.push_back(device);
      }
    }
    for (int const device : devices) {
      device_group_plan group;
      group.device     = device;
      group.n_reactors = per_device;
      std::array<char, 32> pci{};
      if (cudaDeviceGetPCIBusId(pci.data(), static_cast<int>(pci.size()), device) == cudaSuccess) {
        group.pci_bus_id = pci.data();
      } else {
        std::ignore = cudaGetLastError();
      }
      auto const* gpu = find_gpu(topology, group.pci_bus_id);
      if (gpu != nullptr) group.numa_node = gpu->numa_node;

      std::string placement;
      if (!cfg.pin_to_device_cpus) {
        placement = "unpinned (pin_to_device_cpus off)";
      } else if (gpu == nullptr) {
        placement = topology == nullptr ? "unpinned (no topology)"
                                        : "unpinned (GPU not found in the topology)";
      } else if (gpu->cpu_cores.empty()) {
        placement = "unpinned (the GPU reports no local CPUs)";
      } else {
        group.cpus = intersect_in_order(gpu->cpu_cores, allowed);
        placement  = group.cpus.empty()
                       ? "unpinned (the GPU's CPUs " + memory::format_cpu_list(gpu->cpu_cores) +
                          " are outside the inherited affinity)"
                       : "pinned to CPUs " + memory::format_cpu_list(group.cpus) + " (NUMA node " +
                          std::to_string(gpu->numa_node) + ")";
      }

      // The registered-memory cap: lanes and staging slots lease from the same pool.
      std::size_t cap = 0;
      if (stage || fused) {
        cap          = cfg.max_staging_bytes_per_device;
        auto tighten = [&cap](std::size_t limit) {
          if (limit != 0 && (cap == 0 || limit < cap)) cap = limit;
        };
        if (gpu != nullptr) tighten(gpu->bar1_free_bytes / 2);
        // cuFile counts its own bounce buffers against the pinned limit too.
        if (platform.max_device_pinned_mem_size_kb > platform.max_device_cache_size_kb) {
          tighten((platform.max_device_pinned_mem_size_kb - platform.max_device_cache_size_kb) *
                  1024);
        }
        group.staging_cap_bytes = cap;
      }
      auto const sync_slots_wanted = per_device * cfg.staging_slots_per_reactor;

      // Lanes are budgeted first: they carry the device reads.
      std::string lanes_why;
      std::size_t lane_bytes = 0;
      if (fused) {
        group.lane_reactors     = cfg.fused_reads.reactors_per_device;
        group.lanes_per_reactor = cfg.fused_reads.lanes_per_reactor;
        auto const lanes_wanted = std::min(group.lane_reactors * group.lanes_per_reactor,
                                           cfg.fused_reads.max_lanes_per_device);
        group.lane_slots =
          cap != 0 ? std::min(lanes_wanted, cap / lane_footprint) : lanes_wanted;
        lane_bytes = group.lane_slots * lane_footprint;
        lanes_why  = std::to_string(group.lane_reactors) + " lane reactors x " +
                    std::to_string(group.lanes_per_reactor) + " lanes, " +
                    std::to_string(group.lane_slots) + " lane slots of " +
                    std::to_string(cfg.fused_reads.slot_bytes >> 10) + " KiB" +
                    (cap != 0 ? " (cap " + std::to_string(cap >> 20) + " MiB)" : "");
        if (group.lane_slots < lanes_wanted) {
          auto const needed =
            lanes_wanted * lane_footprint + (stage ? sync_slots_wanted * slot_footprint : 0);
          auto const needed_mib = (needed + (1UL << 20) - 1) >> 20;
          bool const config_binds =
            cfg.max_staging_bytes_per_device != 0 && cap == cfg.max_staging_bytes_per_device;
          plan.warnings.push_back(
            "GPU " + std::to_string(device) + ": the registered-memory cap of " +
            std::to_string(cap >> 20) + " MiB holds only " + std::to_string(group.lane_slots) +
            " of " + std::to_string(lanes_wanted) + " fused-read lanes of " +
            std::to_string(cfg.fused_reads.slot_bytes >> 10) +
            " KiB, so fewer reads are in flight; set max_staging_bytes_per_device to at least " +
            std::to_string(needed_mib) + " MiB to fit every lane" +
            (stage ? " plus the synchronous staging" : "") +
            (config_binds ? ""
                          : " (the GPU's free BAR1 or cuFile's pinned-memory limit binds here: "
                            "raise that, or lower fused_reads.lanes_per_reactor / slot_bytes)"));
        }
      }

      std::string staging_why;
      if (!stage) {
        staging_why = cfg.staging_mode == staging::never
                        ? "no staging (staging_mode never)"
                        : "no staging (cuFile reports no GPUDirect Storage mode)";
      } else {
        auto wanted = sync_slots_wanted;
        if (cap != 0) wanted = std::min(wanted, (cap - lane_bytes) / slot_footprint);
        group.staging_slots = wanted;
        staging_why         = std::to_string(wanted) + " staging slots of " +
                      std::to_string(effective.staging_slot_bytes() >> 10) + " KiB" +
                      (cap != 0 ? " (cap " + std::to_string(cap >> 20) + " MiB)" : "");
        if (wanted < per_device * cfg.staging_slots_per_reactor) {
          staging_why +=
            "; fewer than reactors x slots: the cap, BAR1 or cuFile's pinned limit "
            "binds, and reactors without a slot read directly";
          if (lane_bytes != 0) {
            auto const needed     = lane_bytes + sync_slots_wanted * slot_footprint;
            auto const needed_mib = (needed + (1UL << 20) - 1) >> 20;
            plan.warnings.push_back(
              "GPU " + std::to_string(device) + ": the " + std::to_string(group.lane_slots) +
              " fused-read lane slots leave its " + std::to_string(per_device) +
              " synchronous reactors " + std::to_string(wanted) + " of " +
              std::to_string(sync_slots_wanted) +
              " staging slots, and a reactor without a slot reads directly; set "
              "max_staging_bytes_per_device to at least " +
              std::to_string(needed_mib) + " MiB, or lower fused_reads.lanes_per_reactor");
          }
        }
      }
      group.reason = sizing + ", " + placement + ", " + staging_why;
      if (fused) group.reason += ", " + lanes_why;
      plan.groups.push_back(std::move(group));
    }
    plan.reason = std::to_string(plan.groups.size()) + " GPU group(s), " +
                  std::to_string(plan.total_reactors()) + " reactors";
    if (fused || batch) {
      std::size_t lane_total = 0;
      std::size_t sync_total = 0;
      for (auto const& group : plan.groups) {
        lane_total += group.lane_reactors;
        sync_total += group.n_reactors;
      }
      plan.reason += " (" + std::to_string(lane_total) + " lane, " + std::to_string(sync_total) +
                     " synchronous, " + std::to_string(plan.host_batch_reactors) +
                     " host-batch)";
    }
  } catch (std::exception const& e) {
    plan.groups.clear();
    plan.host_batch_reactors = 0;
    plan.flat_reactors       = default_reactors_per_device;
    plan.reason        = std::string{"planning failed ("} + e.what() + "): one flat pool";
  } catch (...) {
    plan.groups.clear();
    plan.host_batch_reactors = 0;
    plan.flat_reactors       = default_reactors_per_device;
    plan.reason        = "planning failed: one flat pool";
  }
  return plan;
}

std::string describe(pool_plan const& plan)
{
  std::string out = "cufile pool: " + plan.reason;
  if (plan.staging_piece_bytes != 0) {
    out += ", staged pieces of " + std::to_string(plan.staging_piece_bytes >> 10) + " KiB";
  }
  if (plan.lane_slot_bytes != 0) {
    out += ", lane slots of " + std::to_string(plan.lane_slot_bytes >> 10) + " KiB";
  }
  if (plan.host_batch_reactors != 0) {
    out += ", " + std::to_string(plan.host_batch_reactors) +
           " host-batch reactors serving every GPU's host reads";
  }
  for (auto const& group : plan.groups) {
    out += "; GPU " + std::to_string(group.device) + " (" + group.pci_bus_id + "): " + group.reason;
  }
  return out;
}

}  // namespace cucascade::io::cufile
