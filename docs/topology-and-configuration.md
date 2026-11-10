# Topology & Configuration

Hardware topology discovery, NUMA-aware memory placement, and system configuration.

## Table of Contents

- [Overview](#overview)
- [Hardware Topology Discovery](#hardware-topology-discovery)
  - [What Gets Detected](#what-gets-detected)
  - [How Detection Works](#how-detection-works)
  - [CUDA Context Independence](#cuda-context-independence)
  - [GPU-to-NUMA Affinity](#gpu-to-numa-affinity)
  - [PCIe Path Types](#pcie-path-types)
- [Configuring cuCascade](#configuring-cucascade)
  - [Builder Pattern](#builder-pattern)
  - [GPU Configuration](#gpu-configuration)
  - [Host Configuration](#host-configuration)
  - [Disk Configuration](#disk-configuration)
  - [Building from Topology](#building-from-topology)
- [Configuration Examples](#configuration-examples)
  - [Single GPU Setup](#single-gpu-setup)
  - [Multi-GPU with NUMA Awareness](#multi-gpu-with-numa-awareness)
  - [Custom Memory Resources](#custom-memory-resources)
  - [Full Production Setup](#full-production-setup)
- [Default Values Reference](#default-values-reference)
- [Key Source Files](#key-source-files)

---

## Overview

Optimal memory performance in multi-GPU systems requires understanding the hardware topology. A GPU accesses memory fastest from its local NUMA node -- allocating pinned host memory on a remote NUMA node can halve PCIe transfer bandwidth.

cuCascade automatically discovers the hardware topology and uses it to:
- Bind host memory spaces to the correct NUMA nodes for each GPU
- Identify optimal network devices for GPU-direct communication
- Report NVMe storage devices with their NUMA affinity (disk spaces themselves are configured explicitly)

## Hardware Topology Discovery

**File**: `include/cucascade/memory/topology_discovery.hpp`

### What Gets Detected

```cpp
topology_discovery discovery;
if (discovery.discover()) {
    auto const& topo = discovery.get_topology();

    topo.hostname;           // System hostname
    topo.num_gpus;           // Number of entries in topo.gpus (one per MIG instance)
    topo.num_numa_nodes;     // Total NUMA node count
    topo.numa_nodes;         // Per-NUMA-node capacity and memory kind
    topo.gpus;               // Per-GPU topology info
    topo.network_devices;    // NICs with NUMA affinity
    topo.storage_devices;    // NVMe drives with NUMA affinity

    // Lookups return nullopt for an unknown node rather than a silent 0.
    topo.get_numa_memory_capacity(0);      // std::optional<std::size_t>
    topo.get_numa_free_memory(0);          // std::optional<std::size_t>
    topo.get_total_numa_memory_capacity(); // host memory only
    topo.find_numa_node(0);                // numa_topology_info const*
}
```

`discover()` currently always returns `true`: when NVML is unavailable it succeeds with an empty
`gpus` list, so check `topo.gpus.empty()` to detect a GPU-less discovery. It throws
`std::invalid_argument` for an out-of-range numeric `CUDA_VISIBLE_DEVICES` entry.

**Per-NUMA-node information** (`numa_topology_info`), read from
`/sys/devices/system/node/node<id>/`:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `int` | NUMA node ID |
| `memory_capacity` | `std::size_t` | `MemTotal` in bytes (0 if unknown) |
| `free_memory` | `std::size_t` | `MemFree` in bytes (0 if unknown) |
| `has_cpus` | `bool` | Whether `cpulist` names any CPU |
| `is_device_memory` | `bool` | Node has memory but no CPUs -- device memory, not host memory |

A node with memory and no CPUs is not host memory. DGX Station and Grace-Hopper expose GPU
HBM as its own CPU-less node, and CXL memory expanders do the same. Such nodes are excluded
from `get_total_numa_memory_capacity()`, and when `set_usage_limit_ratio_per_numa_region()` is
used, `build()` throws rather than sizing a host space from device memory.

**Per-GPU information** (`gpu_topology_info`):

| Field | Type | Description |
|-------|------|-------------|
| `id` | `unsigned int` | Position in the visible GPU list: NVML (PCI bus) order after `CUDA_VISIBLE_DEVICES` filtering and MIG expansion. Matches the CUDA device ordinal only when CUDA enumerates in the same order (e.g. `CUDA_DEVICE_ORDER=PCI_BUS_ID`) |
| `name` | `string` | Device name (e.g., "NVIDIA A100") |
| `pci_bus_id` | `string` | PCIe bus address |
| `uuid` | `string` | Unique GPU identifier |
| `numa_node` | `int` | NUMA node this GPU is closest to (-1 if unknown) |
| `cpu_affinity_list` | `string` | CPU core affinity string |
| `cpu_cores` | `vector<int>` | List of optimal CPU core IDs |
| `memory_binding` | `vector<int>` | NUMA nodes for memory binding |
| `network_devices` | `vector<string>` | NICs optimal for this GPU |
| `runtime_attributes` | `optional<gpu_runtime_attributes>` | Runtime attributes such as `hw_decomp`; empty unless requested (see [CUDA Context Independence](#cuda-context-independence)) |

**Network devices** (`network_device_info`): name, NUMA node, PCIe bus ID

**Storage devices** (`storage_device_info`): type (`StorageDriveType`: NVME/SATA_SSD/SATA_HDD/UNKNOWN), name, NUMA node, PCIe bus ID. Only NVMe controllers are currently detected.

### How Detection Works

1. **NVML initialization** -- the CUDA toolkit's static NVML loader (`CUDA::nvml_static`) is linked in and loads the driver's `libnvidia-ml.so.1` at runtime; NVML is initialized once per process (again in a `fork()`ed child), and if initialization fails, discovery continues without GPUs
2. **GPU enumeration** -- NVML queries device count, names, UUIDs, PCIe bus IDs; a MIG-enabled GPU contributes one entry per MIG instance, and `CUDA_VISIBLE_DEVICES` filters and orders the list
3. **NUMA mapping** -- `nvmlDeviceGetMemoryAffinity()` gives the NUMA node of each GPU
4. **CPU affinity** -- reads `/sys/bus/pci/devices/<bus_id>/local_cpulist`
5. **NUMA nodes** -- reads `meminfo` and `cpulist` under `/sys/devices/system/node/node<id>/`
6. **Network devices** -- scans `/sys/class/infiniband/` for NICs with NUMA info, with configurable verification (see [Network Device Verification](#network-device-verification))
7. **Storage devices** -- scans `/sys/class/nvme/` for NVMe controllers with NUMA info

### CUDA Context Independence

By default, `discover()` uses only NVML and Linux sysfs. It may be called before or after CUDA
initialization, but it does not initialize the CUDA driver or runtime, create or require a CUDA
context, or otherwise alter CUDA process state.

Runtime attributes are explicitly opt-in. Passing `true` for `with_runtime_attributes`, or calling
`discover_runtime_attributes()` directly, queries the CUDA driver and requires the caller to have
initialized it first. The driver library is loaded dynamically only when runtime attributes are
requested, so passive topology discovery remains available when `libcuda.so.1` is absent. These
queries do not initialize the driver or create a CUDA context, therefore, unavailable drivers and
failed queries leave the corresponding runtime attributes unsupported.

Topology discovery is compiled into its own library, `cuCascade::cucascade_topology_discovery`,
which links only NVML, NVTX and `libdl` (no CUDA runtime, RMM or cudf). Configuring with
`-DCUCASCADE_TOPOLOGY_ONLY=ON` builds only that library (plus its tests when
`CUCASCADE_BUILD_TESTS` is ON), without RMM or cudf.

### Network Device Verification

The `discover()` method accepts a `NetworkDeviceVerification` parameter that controls how strictly network devices are validated before inclusion. This prevents assigning non-functional devices to `UCX_NET_DEVICES` (which would cause connection failures).

| Level | Enum Value | Checks Performed |
|-------|------------|------------------|
| **Strictest (default)** | `EXISTS_ACTIVE_IP` | Device exists, uverbs device node accessible, at least one port is ACTIVE, and the associated net interface has an IP address |
| **Medium** | `EXISTS_ACTIVE` | Device exists, uverbs device node accessible, and at least one port is ACTIVE |
| **Most permissive** | `EXISTS` | Device exists in `/sys/class/infiniband/` |

```cpp
topology_discovery discovery;

// Default: strictest (requires active port + IP address)
[[maybe_unused]] bool discovered = discovery.discover();

// Only require active port (skip IP check)
discovered = discovery.discover(NetworkDeviceVerification::EXISTS_ACTIVE);

// Include all devices regardless of state
discovered = discovery.discover(NetworkDeviceVerification::EXISTS);
```

Port state is read from `/sys/class/infiniband/<device>/ports/<N>/state`. The uverbs device node check verifies that `/dev/infiniband/uverbsN` exists for the device (in containerized environments sysfs may be mounted from the host while the device nodes are not passed through). IP address presence is checked via `getifaddrs()` on the net interface found under `/sys/class/infiniband/<device>/device/net/`.

### GPU-to-NUMA Affinity

In a typical multi-GPU server:

```
NUMA Node 0                    NUMA Node 1
├── CPU cores 0-31             ├── CPU cores 32-63
├── 256 GB DDR5                ├── 256 GB DDR5
├── GPU 0 (PCIe)               ├── GPU 2 (PCIe)
├── GPU 1 (PCIe)               ├── GPU 3 (PCIe)
├── mlx5_0 (NIC)               ├── mlx5_1 (NIC)
└── nvme0n1 (NVMe)             └── nvme1n1 (NVMe)
```

cuCascade uses this mapping to ensure that when GPU 0 spills data to host memory, it goes to NUMA node 0's pinned memory -- not NUMA node 1, which would require cross-NUMA traffic.

### PCIe Path Types

The `PciePathType` enum describes the connection quality between devices. Discovery estimates
the path type between each GPU and NIC from their NUMA nodes and PCI bus numbers, and assigns
each GPU the NICs with the best path type (`gpu_topology_info::network_devices`). This is a
heuristic: NICs on a different NUMA node are `SYS`; otherwise a PCI bus-number distance of at
most 2 is `PIX`, at most 10 (or an unparsable bus) is `PHB`, and anything larger is `NODE`. It
never yields `PXB`, and the path type itself is not stored in the topology structs.

| Type | Value | Meaning | Relative Performance |
|------|-------|---------|---------------------|
| `PIX` | 0 | Single PCIe bridge | Best |
| `PXB` | 1 | Multiple PCIe bridges | Good |
| `PHB` | 2 | PCIe Host Bridge | Moderate |
| `NODE` | 3 | PCIe + interconnect within NUMA | Below average |
| `SYS` | 4 | Cross-NUMA interconnect | Worst |

---

## Configuring cuCascade

### Builder Pattern

**File**: `include/cucascade/memory/reservation_manager_configurator.hpp`

The `reservation_manager_configurator` provides a fluent builder API:

```cpp
reservation_manager_configurator configurator;
configurator
    .set_number_of_gpus(2)
    .set_gpu_usage_limit(4ULL << 30)           // 4 GB per GPU
    .set_reservation_fraction_per_gpu(0.85)    // 85% reservable
    .set_per_numa_region_capacity(16ULL << 30)        // 16 GB per host space
    .use_numa_id_as_host_id();                      // One host space per NUMA node

// With topology
topology_discovery discovery;
[[maybe_unused]] bool discovered = discovery.discover();
auto configs = configurator.build(discovery.get_topology());

// Or let build() run topology_discovery::discover() itself:
// auto configs = configurator.build();

// Create the manager
memory_reservation_manager manager(std::move(configs));
```

### GPU Configuration

| Method | Description | Default |
|--------|-------------|---------|
| `set_number_of_gpus(n)` | Number of GPUs to use | 1 |
| `set_gpu_ids({0, 2})` | Explicit GPU device IDs | `{0}` |
| `set_gpu_usage_limit(bytes)` | Absolute memory capacity per GPU | 1 GB |
| `set_usage_limit_ratio_per_gpu(0.8)` | Fraction of total GPU memory to use | N/A |
| `set_reservation_fraction_per_gpu(0.85)` | Fraction of capacity that can be reserved | 0.85 |
| `set_reservation_limit_per_gpu(bytes)` | Absolute reservation limit | N/A |
| `set_downgrade_fractions_per_gpu(0.85, 0.65)` | (trigger, stop) fractions for downgrade | (0.85, 0.65) |
| `track_reservation_per_stream(true)` | Enable per-stream reservation tracking | `true` |
| `set_gpu_memory_resource_factory(fn)` | Custom GPU allocator factory | RMM async pool |

Note: `set_gpu_usage_limit()` and `set_usage_limit_ratio_per_gpu()` are mutually exclusive -- use one or the other; the last call wins. The `fraction_or_size` type handles this internally.

### Host Configuration

| Method | Description | Default |
|--------|-------------|---------|
| `use_gpu_id_as_host_id()` | One host space per GPU (testing) | N/A |
| `use_numa_id_as_host_id()` | One host space per NUMA node (production) | `use_numa_id_as_host_id()` |
| `set_total_host_capacity(bytes)` | Total host memory across all spaces | N/A |
| `set_per_numa_region_capacity(bytes)` | Memory per host space | 4 GB per space |
| `set_usage_limit_ratio_per_numa_region(0.25)` | Memory per host space as a fraction of its NUMA region's capacity | N/A |
| `set_reservation_fraction_per_numa_region(0.85)` | Fraction reservable | 0.85 |
| `set_reservation_limit_per_numa_region(bytes)` | Absolute reservation limit | N/A |
| `set_downgrade_fractions_per_numa_region(0.85, 0.65)` | (trigger, stop) fractions | (0.85, 0.65) |
| `set_host_pool_features(chunk, block, count)` | Block allocator settings: block size in bytes, blocks per pool, initial pools | (1MB, 128, 4) |
| `set_host_portability(bool)` | Whether default pinned host allocations are portable across GPUs | `true` with more than one GPU |
| `set_host_memory_resource_factory(fn)` | Custom host allocator factory | NUMA-pinned |

Host creation policies:
- **`use_gpu_id_as_host_id()`** -- creates one host space per GPU and identifies it by the GPU id. The space is still bound to that GPU's NUMA node for allocation; only the space id differs. Useful for testing.
- **`use_numa_id_as_host_id()`** -- creates one host space per NUMA node, identified by the NUMA node id and shared by GPUs on that node. Default, and optimal for production.

`set_total_host_capacity()` is the only host-wide setting: it is divided evenly across the host spaces. Every other capacity setting applies per space, i.e. per NUMA region.

### Disk Configuration

```cpp
configurator.set_disk_mounting_point(
    0,                    // disk UUID
    1ULL << 40,          // 1 TB capacity
    "/mnt/nvme0"         // mount path
);
```

### Building from Topology

`build(topology)` uses the topology passed in. `build()` runs `topology_discovery::discover()`
with default settings and then does the same. In both cases:
- The configured GPU IDs (or GPUs `0..n-1` for `set_number_of_gpus(n)`) must be present in the
  topology, otherwise `build()` throws
- The total memory of every GPU in the topology (not only the configured ones) is queried through
  the CUDA runtime (`rmm::available_device_memory()`), using the topology `id` as the CUDA device
  ordinal. Unlike `discover()`, this initializes CUDA and creates a context on each visible GPU.
  `set_usage_limit_ratio_per_gpu()` is applied to that total
- Each GPU's NUMA node comes from the topology, and host spaces are created for those NUMA nodes
- `set_usage_limit_ratio_per_numa_region()` sizes each host space from its NUMA node's capacity in
  the topology

---

## Configuration Examples

### Single GPU Setup

```cpp
reservation_manager_configurator configurator;
configurator
    .set_gpu_usage_limit(4ULL << 30)       // 4 GB GPU
    .set_reservation_fraction_per_gpu(0.8) // 80% reservable
    .set_per_numa_region_capacity(8ULL << 30)     // 8 GB host
    .use_gpu_id_as_host_id();

auto configs = configurator.build();
memory_reservation_manager manager(std::move(configs));
```

### Multi-GPU with NUMA Awareness

```cpp
topology_discovery discovery;
[[maybe_unused]] bool discovered = discovery.discover();

reservation_manager_configurator configurator;
configurator
    .set_gpu_ids({0, 1, 2, 3})
    .set_usage_limit_ratio_per_gpu(0.8)          // 80% of each GPU
    .set_reservation_fraction_per_gpu(0.85)
    .set_downgrade_fractions_per_gpu(0.85, 0.65)
    .set_per_numa_region_capacity(32ULL << 30)           // 32 GB per NUMA node
    .use_numa_id_as_host_id()
    .set_host_pool_features(
        2ULL << 20,  // 2 MB blocks
        64,          // 64 blocks per pool
        8            // 8 initial pools = 1 GB
    );

auto configs = configurator.build(discovery.get_topology());
memory_reservation_manager manager(std::move(configs));
```

### Custom Memory Resources

```cpp
// Custom GPU allocator factory (a DeviceMemoryResourceFactoryFn)
auto custom_gpu_factory = [](int device_id, std::size_t capacity)
    -> ::cuda::mr::any_resource<::cuda::mr::device_accessible> {
    // Use your own pool or allocator
    return my_custom_gpu_allocator{device_id, capacity};
};

reservation_manager_configurator configurator;
configurator
    .set_gpu_memory_resource_factory(custom_gpu_factory)
    .set_gpu_usage_limit(8ULL << 30);

auto configs = configurator.build();
```

### Full Production Setup

```cpp
topology_discovery discovery;
[[maybe_unused]] bool discovered = discovery.discover();
if (discovery.get_topology().gpus.empty()) {
    throw std::runtime_error("No GPUs discovered");
}

reservation_manager_configurator configurator;

// GPU: use 80% of each GPU, 85% reservable
configurator
    .set_gpu_ids({0, 1, 2, 3})
    .set_usage_limit_ratio_per_gpu(0.8)
    .set_reservation_fraction_per_gpu(0.85)
    .set_downgrade_fractions_per_gpu(0.85, 0.65)
    .track_reservation_per_stream(true);

// Host: NUMA-aware, 64 GB per node, 2 MB blocks
configurator
    .use_numa_id_as_host_id()
    .set_per_numa_region_capacity(64ULL << 30)
    .set_reservation_fraction_per_numa_region(0.85)
    .set_downgrade_fractions_per_numa_region(0.85, 0.65)
    .set_host_pool_features(2ULL << 20, 128, 8);

// Disk: 1 TB NVMe
configurator.set_disk_mounting_point(0, 1ULL << 40, "/mnt/nvme0");

auto configs = configurator.build(discovery.get_topology());
memory_reservation_manager manager(std::move(configs));
```

---

## Default Values Reference

| Setting | Default Value |
|---------|---------------|
| Number of GPUs | 1 |
| GPU capacity | 1 GB |
| GPU reservation fraction | 0.85 (85%) |
| GPU downgrade trigger | 0.85 (85%) |
| GPU downgrade stop | 0.65 (65%) |
| Per-stream tracking | `true` |
| Host capacity | 4 GB per host space |
| Host reservation fraction | 0.85 (85%) |
| Host downgrade trigger | 0.85 (85%) |
| Host downgrade stop | 0.65 (65%) |
| Host block size | 1 MB (`1 << 20`) |
| Host pool size | 128 blocks per pool |
| Host initial pools | 4 |
| Host creation policy | `use_numa_id_as_host_id()` |
| Host portability | Portable when more than one GPU is configured |
| GPU allocator | `rmm::mr::cuda_async_memory_resource` |
| Host allocator | `numa_region_pinned_host_memory_resource` |
| Stream pool size | 16 streams |

---

## Key Source Files

| File | Purpose |
|------|---------|
| `include/cucascade/memory/topology_discovery.hpp` | `topology_discovery`, `system_topology_info`, `gpu_topology_info` |
| `include/cucascade/memory/reservation_manager_configurator.hpp` | `reservation_manager_configurator` builder |
| `include/cucascade/memory/config.hpp` | `gpu_memory_space_config`, `host_memory_space_config`, `disk_memory_space_config` |
| `include/cucascade/memory/common.hpp` | `Tier` enum, `memory_space_id`, `DeviceMemoryResourceFactoryFn` |
| `include/cucascade/memory/numa_region_pinned_host_allocator.hpp` | NUMA-aware pinned host allocation |
| `src/memory/topology_discovery.cpp` | NVML and `/sys` queries, lazily loaded CUDA driver for runtime attributes |
| `src/memory/reservation_manager_configurator.cpp` | Builder implementation |
