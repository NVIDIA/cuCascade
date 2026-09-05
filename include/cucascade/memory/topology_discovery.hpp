/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace cucascade::memory {

/**
 * @brief GPU information.
 */
struct gpu_topology_info {
  unsigned int id{0};                        ///< GPU device ID.
  std::string name;                          ///< GPU device name.
  std::string pci_bus_id;                    ///< PCI bus ID.
  std::string uuid;                          ///< GPU UUID.
  int numa_node{-1};                         ///< NUMA node ID (-1 if unknown).
  std::string cpu_affinity_list;             ///< CPU affinity list.
  std::vector<int> cpu_cores;                ///< List of CPU core IDs.
  std::vector<int> memory_binding;           ///< NUMA nodes for memory binding.
  std::vector<std::string> network_devices;  ///< Network devices (NICs) optimal for this GPU.
  bool hw_decompression_available{false};    ///< Hardware-accelerated decompression engine present.
};

/**
 * @brief Network device information.
 */
struct network_device_info {
  std::string name;        ///< Device name (e.g., "mlx5_0").
  int numa_node;           ///< NUMA node ID (-1 if unknown).
  std::string pci_bus_id;  ///< PCI bus ID.
};

enum class StorageDriveType {
  NVME,      // NVMe SSD
  SATA_SSD,  // SATA Solid State Drive
  SATA_HDD,  // SATA Hard Disk Drive
  UNKNOWN
};

struct storage_device_info {
  StorageDriveType type = StorageDriveType::UNKNOWN;  ///< Type of storage drive.
  std::string name;                                   ///< Device name (e.g., "nvme0n1").
  int numa_node{-1};                                  ///< NUMA node ID (-1 if unknown).
  std::string pci_bus_id;                             ///< PCI bus ID.
};

/**
 * @brief NUMA node memory information.
 *
 * Capacities are reported in bytes and are read from
 * `/sys/devices/system/node/node<id>/meminfo`. They are 0 when the kernel does not
 * expose the corresponding entry.
 *
 * @note Not every NUMA node backs host memory. Systems such as DGX Station and
 *       Grace-Hopper expose device memory (GPU HBM) as its own CPU-less NUMA node, and
 *       CXL memory expanders do the same. Those nodes are flagged with
 *       `is_device_memory` and must not be used to size a host memory space.
 */
struct numa_topology_info {
  int id{-1};                      ///< NUMA node ID.
  std::size_t memory_capacity{0};  ///< Total memory of the node in bytes (0 if unknown).
  std::size_t free_memory{0};      ///< Currently free memory of the node in bytes (0 if unknown).
  bool has_cpus{false};            ///< Whether any CPU is assigned to this node.
  bool is_device_memory{false};    ///< Whether this node is device memory rather than host memory.
};

/**
 * @brief System topology information.
 */
struct system_topology_info {
  std::string hostname;                              ///< System hostname.
  unsigned int num_gpus;                             ///< Total number of GPUs.
  int num_numa_nodes;                                ///< Total number of NUMA nodes.
  int num_network_devices;                           ///< Total number of network devices.
  std::vector<gpu_topology_info> gpus;               ///< GPU topology information.
  std::vector<network_device_info> network_devices;  ///< Network device information.
  std::vector<storage_device_info> storage_devices;  ///< Storage device information.
  std::vector<numa_topology_info> numa_nodes;        ///< NUMA node information, sorted by id.

  /**
   * @brief Find a NUMA node by ID.
   *
   * @param numa_id NUMA node ID to look up.
   * @return Pointer to the node, or `nullptr` if no such node was discovered.
   */
  [[nodiscard]] numa_topology_info const* find_numa_node(int numa_id) const
  {
    for (auto const& node : numa_nodes) {
      if (node.id == numa_id) { return &node; }
    }
    return nullptr;
  }

  /**
   * @brief Get the memory capacity of a NUMA node.
   *
   * @param numa_id NUMA node ID to look up.
   * @return Capacity of the node in bytes, or `std::nullopt` if the node was not
   *         discovered or the kernel did not report its capacity. The two cases are
   *         distinguishable via `find_numa_node()`.
   */
  [[nodiscard]] std::optional<std::size_t> get_numa_memory_capacity(int numa_id) const
  {
    auto const* node = find_numa_node(numa_id);
    if (node == nullptr || node->memory_capacity == 0) { return std::nullopt; }
    return node->memory_capacity;
  }

  /**
   * @brief Get the free memory of a NUMA node.
   *
   * @param numa_id NUMA node ID to look up.
   * @return Free memory of the node in bytes, or `std::nullopt` if the node was not
   *         discovered or the kernel did not report its free memory.
   */
  [[nodiscard]] std::optional<std::size_t> get_numa_free_memory(int numa_id) const
  {
    auto const* node = find_numa_node(numa_id);
    if (node == nullptr || node->free_memory == 0) { return std::nullopt; }
    return node->free_memory;
  }

  /**
   * @brief Get the summed memory capacity of all host-backing NUMA nodes.
   *
   * Nodes flagged as device memory are excluded, so the result is usable host memory.
   *
   * @return Total host memory capacity in bytes.
   */
  [[nodiscard]] std::size_t get_total_numa_memory_capacity() const
  {
    std::size_t total = 0;
    for (auto const& node : numa_nodes) {
      if (node.is_device_memory) { continue; }
      total += node.memory_capacity;
    }
    return total;
  }
};

/**
 * @brief Verification level for network device discovery.
 *
 * Controls how strictly network devices are validated before being included
 * in the discovered topology.
 */
enum class NetworkDeviceVerification {
  EXISTS_ACTIVE_IP = 0,  ///< Device exists, port is active, uverbs accessible, and has an IP
                         ///< address (default).
  EXISTS_ACTIVE = 1,     ///< Device exists, port is active, and uverbs device node is accessible.
  EXISTS        = 2      ///< Device exists only (no port, uverbs, or IP checks).
};

/**
 * @brief PCIe topology path types.
 */
enum class PciePathType {
  PIX  = 0,  ///< Connection traversing at most a single PCIe bridge (best).
  PXB  = 1,  ///< Connection traversing multiple PCIe bridges.
  PHB  = 2,  ///< Connection traversing PCIe Host Bridge.
  NODE = 3,  ///< Connection traversing PCIe and interconnect within NUMA node.
  SYS  = 4   ///< Connection traversing NUMA interconnect (worst).
};

/**
 * @brief Discover system topology including GPUs, NUMA nodes, and network devices.
 *
 * This class provides methods to discover system topology information using NVML
 * and /sys filesystem queries. It dynamically identifies GPU-to-NUMA-to-NIC mappings
 * based on PCIe topology.
 *
 * Example usage:
 * @code
 * cucascade::memory:topology_discovery discovery;
 * if (discovery.discover()) {
 *     auto topology = discovery.get_topology();
 * }
 * @endcode
 */
class topology_discovery {
 public:
  /**
   * @brief Discover system topology.
   *
   * This method performs the actual discovery of GPUs, NUMA nodes, CPU affinity,
   * and network devices. It must be called before `get_topology()`.
   *
   * @param net_verification Controls how strictly network devices are validated.
   * @return true if discovery was successful, false otherwise.
   */
  [[nodiscard]] bool discover(
    NetworkDeviceVerification net_verification = NetworkDeviceVerification::EXISTS_ACTIVE_IP);

  /**
   * @brief Get the discovered topology information.
   *
   * @return system_topology_info structure containing all topology data.
   * @note `discover()` must be called first.
   */
  [[nodiscard]] system_topology_info const& get_topology() const { return _topology.value(); }

  /**
   * @brief Check if topology has been discovered.
   *
   * @return true if `discover()` has been called successfully.
   */
  [[nodiscard]] bool is_discovered() const { return _topology.has_value(); }

 private:
  std::optional<system_topology_info> _topology;  ///< Discovered topology information.
};

}  // namespace cucascade::memory
