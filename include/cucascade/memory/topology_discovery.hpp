/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cucascade/memory/net_topology.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cucascade::memory {

/**
 * @brief GPU runtime attributes.
 *
 * Attributes whose discovery requires initializing a CUDA context (via the CUDA
 * driver API). Populated only when `discover()` / `discover_runtime_attributes()`
 * are explicitly asked to. Kept separate so passive topology discovery via NVML
 * and sysfs never has to spin up a CUDA context.
 */
struct gpu_runtime_attributes {
  bool hw_decomp{false};  ///< Hardware-accelerated decompression engine present.
};

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
  std::optional<gpu_runtime_attributes>
    runtime_attributes;  ///< Runtime attributes (populated only when explicitly requested; empty
                         ///< means "not queried", not "unsupported").
};

/**
 * @brief Network device information.
 *
 * `net_interface` and `mtu` describe the kernel network interface backing the
 * RDMA device (`/sys/class/infiniband/<name>/device/net/<interface>`): the
 * Ethernet netdev of a RoCE NIC or the IPoIB interface of an InfiniBand HCA. A
 * device can back several interfaces (e.g. a multi-port HCA, or IPoIB P_Key
 * child interfaces); the one reported is the first, by name, that has an IP
 * address, or the first by name when none has one. A device without a network
 * interface (InfiniBand without IPoIB) reports an empty `net_interface` and an
 * `mtu` of 0.
 *
 * @note `mtu` is the network-interface (L3) MTU, i.e. the payload size of an
 *       IP packet on the link. It is not the RDMA path MTU (`ibv_port_attr::active_mtu`),
 *       which is negotiated per port and only exposed through libibverbs.
 */
struct network_device_info {
  std::string name;           ///< Device name (e.g., "mlx5_0").
  int numa_node;              ///< NUMA node ID (-1 if unknown).
  std::string pci_bus_id;     ///< PCI bus ID.
  std::string net_interface;  ///< Backing network interface (e.g., "ib0"); empty if none.
  int mtu{0};                 ///< MTU of `net_interface` in bytes (0 if unknown or none).
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
 * @brief Outcome of a per-path storage discovery (see
 *        @ref topology_discovery::discover_storage_for_path).
 *
 * @c unknown is not a negative answer: it means the mount or device stack
 * could not be determined (sysfs or procfs unreadable, an overlay / container
 * mount, a filesystem without a block-device mapping, ...).
 */
enum class path_storage_status {
  ok,                ///< The file, its mount and its backing were resolved.
  not_found,         ///< The path does not exist.
  not_regular_file,  ///< The path exists but is not a regular file.
  inaccessible,      ///< The path could not be opened or stat'ed (e.g. EACCES).
  unknown,           ///< The file was opened, but its mount or backing is unknown.
};

/**
 * @brief What backs the filesystem holding a file.
 */
enum class storage_kind {
  local_block,  ///< A local block-device stack (see @c path_storage_info::devices).
  network,      ///< A network / distributed filesystem (NFS, Lustre, GPFS, ...).
  memory,       ///< A memory filesystem (tmpfs, ramfs, ...).
  unknown,      ///< Not determined (overlay, FUSE, pseudo or unresolved filesystems).
};

/**
 * @brief A leaf block device under a file's filesystem.
 *
 * Leaves are whole disks reached from the filesystem's device through partition
 * parents and device-mapper / md @c slaves; a stack spanning several disks has
 * several leaves.
 */
struct block_device_info {
  std::string name;                                   ///< Kernel name, e.g. "nvme0n1".
  unsigned int major{0};                              ///< Device major number.
  unsigned int minor{0};                              ///< Device minor number.
  StorageDriveType type = StorageDriveType::UNKNOWN;  ///< Drive class, when recognized.
  int numa_node{-1};                                  ///< NUMA node (-1 if unknown).
  std::string pci_bus_id;  ///< PCI bus ID of the controller; empty when not on PCI.
};

/**
 * @brief Passive snapshot of where a file is stored: its mount, filesystem and
 *        backing block devices.
 *
 * Produced from @c fstat / @c statx, @c /proc/self/mountinfo and
 * @c /sys/dev/block only: discovery never initializes CUDA, loads cuFile, creates
 * files or changes mounts.  It is a snapshot -- the answer is not cached, and a
 * file can move, or a mount change, right after it was taken.
 */
struct path_storage_info {
  path_storage_status status = path_storage_status::unknown;  ///< Overall outcome.
  storage_kind kind          = storage_kind::unknown;         ///< What backs the filesystem.
  /// Human-readable notes: why the status is not @c ok, or caveats of a
  /// resolved answer (fallback mount match, mixed device stack, ...).
  std::string reason;
  std::string path;              ///< The path as given.
  std::string resolved_path;     ///< Absolute path of the opened file, symlinks resolved.
  unsigned int device_major{0};  ///< Major number of the file's @c st_dev.
  unsigned int device_minor{0};  ///< Minor number of the file's @c st_dev.
  std::optional<int> mount_id;   ///< Mount ID, when the kernel reports it (@c statx).
  std::string mount_point;       ///< Where the filesystem is mounted.
  std::string mount_root;        ///< Root of the mount within its filesystem (bind mounts).
  std::string filesystem_type;   ///< e.g. "ext4", "xfs", "nfs4", "tmpfs".
  std::string mount_source;      ///< e.g. "/dev/nvme0n1p2", "server:/export".
  std::string mount_options;     ///< Per-mount options.
  std::string super_options;     ///< Per-superblock options.
  std::vector<block_device_info> devices;  ///< Leaf block devices (local_block only).

  /// Whether at least one leaf device was found and every leaf is NVMe.
  [[nodiscard]] bool all_devices_nvme() const noexcept
  {
    if (devices.empty()) { return false; }
    for (auto const& device : devices) {
      if (device.type != StorageDriveType::NVME) { return false; }
    }
    return true;
  }

  /// Whether the leaf devices are of more than one drive class.
  [[nodiscard]] bool mixed_devices() const noexcept
  {
    for (auto const& device : devices) {
      if (device.type != devices.front().type) { return true; }
    }
    return false;
  }
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
 * @brief Kernel TLS (kTLS) state of the running kernel.
 *
 * kTLS is provided by the kernel's `tls` module, which registers the "tls" TCP
 * upper-layer protocol (ULP). Once registered, `setsockopt(TCP_ULP, "tls")` succeeds
 * and record encryption/decryption can be offloaded to the kernel.
 *
 * Detection is passive: it reads `/proc/modules` and
 * `/proc/sys/net/ipv4/tcp_available_ulp` and never loads the module.
 *
 * @note `loaded == false` does not mean kTLS can never be used: the kernel loads the
 *       module on demand when a process holding `CAP_NET_ADMIN` sets the ULP. An
 *       unprivileged process, however, cannot trigger that load.
 */
struct kernel_tls_info {
  /// The `tls` ULP is registered, i.e. kTLS is usable now: either the `tls` module is
  /// loaded (and live) or kTLS is built into the kernel (see `built_in`).
  bool loaded{false};
  /// kTLS is built into the kernel (`CONFIG_TLS=y`) rather than provided by a loadable
  /// module. Only reported when `/proc/modules` is readable; false when it is not.
  bool built_in{false};
};

/**
 * @brief Discover the kernel TLS (kTLS) state of the running kernel.
 *
 * Reads `/proc/modules` (a `tls` row in the `Live` state means the module is loaded)
 * and `/proc/sys/net/ipv4/tcp_available_ulp` (a `tls` token means the ULP is
 * registered). The ULP list is authoritative; the module row distinguishes a loadable
 * module from a built-in one and stands in for the ULP list when that file is
 * unreadable.
 *
 * @param roots Where procfs is mounted; tests point it at a fake tree.
 * @return The kTLS state; all fields false when neither file reports kTLS.
 */
[[nodiscard]] kernel_tls_info discover_kernel_tls(discovery_roots const& roots = {});

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
  kernel_tls_info kernel_tls;                        ///< Kernel TLS (kTLS) state.

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
   * network devices (including their MTU), storage devices, and the kernel TLS
   * state. It must be called before `get_topology()`.
   *
   * By default this call uses only NVML and Linux sysfs and therefore does not
   * touch the CUDA driver. Set @p with_runtime_attributes to true to also
   * populate per-hardware runtime attributes (e.g. `gpu_runtime_attributes`),
   * which queries CUDA driver device attributes.
   *
   * @param net_verification Controls how strictly network devices are validated.
   * @param with_runtime_attributes If true, also discover runtime attributes for
   * each hardware class (see `discover_runtime_attributes`). Defaults to false so
   * that discovery does not touch the CUDA driver. When true, the caller must
   * have already initialized the CUDA driver API (see
   * `discover_runtime_attributes` for the exact precondition).
   * @return true if discovery was successful, false otherwise.
   */
  [[nodiscard]] bool discover(
    NetworkDeviceVerification net_verification = NetworkDeviceVerification::EXISTS_ACTIVE_IP,
    bool with_runtime_attributes               = false);

  /**
   * @brief Discover runtime attributes for each hardware class in @p topology.
   *
   * Populates the `runtime_attributes` field of each entry in `topology.gpus`
   * (and, in the future, other hardware classes). This is the only path in this
   * component that issues CUDA driver calls — every other discovery step relies
   * solely on NVML and sysfs. The CUDA driver library is loaded dynamically,
   * therefore, systems without it remain able to use passive topology discovery.
   *
   * Safe to call multiple times; existing runtime attribute values are
   * overwritten.
   *
   * @pre The CUDA driver API has already been initialized by the caller —
   * either via an explicit `cuInit(0)` or via any prior CUDA runtime call that
   * transitively initializes the driver. This function does not call `cuInit`
   * and does not create a CUDA context; per-GPU queries that fail (e.g.
   * because the driver library is unavailable or uninitialized) leave
   * `hw_decomp` as `false`.
   *
   * @param topology Topology to enrich in place.
   */
  static void discover_runtime_attributes(system_topology_info& topology);

  /**
   * @brief Discover where the file at @p path is stored.
   *
   * Opens @p path read-only (following symlinks) and resolves the opened file's
   * mount from its @c statx mount ID when the kernel reports one, otherwise
   * by the longest covering mount point of the file's device. A mount ID that
   * mountinfo no longer lists leaves the storage unknown unless such a
   * same-device mount covers the path. Only when no mount ID is known may a
   * path-only match be used. For a block filesystem the
   * device is followed through @c /sys/dev/block, partition parents and
   * device-mapper / md @c slaves (with cycle protection) down to its leaf
   * disks, recording each leaf's drive class, PCI bus ID and NUMA node.
   *
   * Passive: uses procfs and sysfs only, no CUDA, no cuFile, no writes.  Never
   * throws; failures are reported through @c path_storage_info::status and
   * @c reason.  The result is not cached.
   *
   * @param path Path of a regular file.
   * @return The storage snapshot.
   */
  [[nodiscard]] static path_storage_info discover_storage_for_path(std::string_view path);

  /**
   * @brief As @ref discover_storage_for_path, for an already-open descriptor.
   *
   * Describes the file @p fd refers to, so a caller that opened the file can
   * learn about exactly the file it will read, without a second, race-prone
   * path lookup.  @p path is only reported back (and used for the resolved
   * path when procfs cannot name the descriptor).
   *
   * @param fd   Open descriptor of a regular file; not closed.
   * @param path The path it was opened from, for reporting.
   */
  [[nodiscard]] static path_storage_info discover_storage_for_descriptor(
    int fd, std::string_view path = {});

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
