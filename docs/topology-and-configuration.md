# Topology & Configuration

Hardware topology discovery, NUMA-aware memory placement, and system configuration.

## Table of Contents

- [Overview](#overview)
- [Hardware Topology Discovery](#hardware-topology-discovery)
  - [What Gets Detected](#what-gets-detected)
  - [How Detection Works](#how-detection-works)
  - [GPU-to-NUMA Affinity](#gpu-to-numa-affinity)
  - [PCIe Path Types](#pcie-path-types)
- [Configuring cuCascade](#configuring-cucascade)
  - [Builder Pattern](#builder-pattern)
  - [GPU Configuration](#gpu-configuration)
  - [Host Configuration](#host-configuration)
  - [Disk Configuration](#disk-configuration)
  - [Topology-Aware vs Manual Configuration](#topology-aware-vs-manual-configuration)
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
- Map storage devices to NUMA nodes for disk-tier placement

## Hardware Topology Discovery

**File**: `include/cucascade/memory/topology_discovery.hpp`

### What Gets Detected

```cpp
topology_discovery discovery;
if (discovery.discover()) {
    auto const& topo = discovery.get_topology();

    topo.hostname;           // System hostname
    topo.num_gpus;           // Total GPU count
    topo.num_numa_nodes;     // Total NUMA node count
    topo.gpus;               // Per-GPU topology info
    topo.network_devices;    // NICs with NUMA affinity
    topo.storage_devices;    // NVMe/SATA drives with NUMA affinity
}
```

**Per-GPU information** (`gpu_topology_info`):

| Field | Type | Description |
|-------|------|-------------|
| `id` | `unsigned int` | GPU device ID |
| `name` | `string` | Device name (e.g., "NVIDIA A100") |
| `pci_bus_id` | `string` | PCIe bus address |
| `uuid` | `string` | Unique GPU identifier |
| `numa_node` | `int` | NUMA node this GPU is closest to (-1 if unknown) |
| `cpu_affinity_list` | `string` | CPU core affinity string |
| `cpu_cores` | `vector<int>` | List of optimal CPU core IDs |
| `memory_binding` | `vector<int>` | NUMA nodes for memory binding |
| `network_devices` | `vector<string>` | NICs optimal for this GPU |

**Network devices** (`network_device_info`): name, NUMA node, PCIe bus ID

**Storage devices** (`storage_device_info`): type (NVMe/SATA_SSD/SATA_HDD/UNKNOWN), name, NUMA node, PCIe bus ID

### How Detection Works

1. **NVML loading** -- `libnvidia-ml.so` is loaded at runtime via `dlopen()` (no build-time dependency)
2. **GPU enumeration** -- NVML queries device count, names, UUIDs, PCIe bus IDs
3. **NUMA mapping** -- reads `/sys/bus/pci/devices/<bus_id>/numa_node` for each GPU
4. **CPU affinity** -- reads `/sys/bus/pci/devices/<bus_id>/local_cpulist`
5. **Network devices** -- scans `/sys/class/infiniband/` and `/sys/class/net/` for NICs with NUMA info
6. **Storage devices** -- scans `/sys/block/` for NVMe and SATA devices with NUMA info

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

The `PciePathType` enum describes the connection quality between devices:

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
    .set_per_host_capacity(16ULL << 30)        // 16 GB per host space
    .use_host_per_numa();                      // One host space per NUMA node

// With topology
topology_discovery discovery;
discovery.discover();
auto configs = configurator.build(discovery.get_topology());

// Without topology (uses defaults)
auto configs = configurator.build();

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

Note: `set_gpu_usage_limit()` and `set_usage_limit_ratio_per_gpu()` are mutually exclusive -- use one or the other. The `fraction_or_size` type handles this internally.

### Host Configuration

| Method | Description | Default |
|--------|-------------|---------|
| `use_host_per_gpu()` | One host space per GPU (testing) | N/A |
| `use_host_per_numa()` | One host space per NUMA node (production) | `use_host_per_numa()` |
| `set_total_host_capacity(bytes)` | Total host memory across all spaces | 4 GB |
| `set_per_host_capacity(bytes)` | Memory per host space | 4 GB |
| `set_reservation_fraction_per_host(0.85)` | Fraction reservable | 0.85 |
| `set_reservation_limit_per_host(bytes)` | Absolute reservation limit | N/A |
| `set_downgrade_fractions_per_host(0.85, 0.65)` | (trigger, stop) fractions | (0.85, 0.65) |
| `set_host_pool_features(chunk, block, count)` | Block allocator settings | (1MB, 128, 4) |
| `set_host_memory_resource_factory(fn)` | Custom host allocator factory | NUMA-pinned |

Host creation policies:
- **`use_host_per_gpu()`** -- creates one host space per GPU, regardless of NUMA topology. Useful for testing.
- **`use_host_per_numa()`** -- creates one host space per NUMA node, shared by GPUs on that node. Optimal for production.

### Disk Configuration

```cpp
configurator.set_disk_mounting_point(
    0,                    // disk UUID
    1ULL << 40,          // 1 TB capacity
    "/mnt/nvme0"         // mount path
);
```

### Topology-Aware vs Manual Configuration

**With topology** (`build(topology)`):
- GPU capacities are queried from the hardware
- NUMA nodes are automatically mapped to GPUs
- Host spaces are created for the correct NUMA nodes

**Without topology** (`build()`):
- Uses the configured GPU count/IDs
- GPU capacity must be set explicitly via `set_gpu_usage_limit()`
- Host spaces are created with sequential IDs (0, 1, ...)

---

## Configuration Examples

### Single GPU Setup

```cpp
reservation_manager_configurator configurator;
configurator
    .set_gpu_usage_limit(4ULL << 30)       // 4 GB GPU
    .set_reservation_fraction_per_gpu(0.8) // 80% reservable
    .set_per_host_capacity(8ULL << 30)     // 8 GB host
    .use_host_per_gpu();

auto configs = configurator.build();
memory_reservation_manager manager(std::move(configs));
```

### Multi-GPU with NUMA Awareness

```cpp
topology_discovery discovery;
discovery.discover();

reservation_manager_configurator configurator;
configurator
    .set_gpu_ids({0, 1, 2, 3})
    .set_usage_limit_ratio_per_gpu(0.8)          // 80% of each GPU
    .set_reservation_fraction_per_gpu(0.85)
    .set_downgrade_fractions_per_gpu(0.85, 0.65)
    .set_per_host_capacity(32ULL << 30)           // 32 GB per NUMA node
    .use_host_per_numa()
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
// Custom GPU allocator factory
auto custom_gpu_factory = [](int device_id, size_t capacity)
    -> std::unique_ptr<rmm::mr::device_memory_resource> {
    // Use your own pool or allocator
    return std::make_unique<my_custom_gpu_allocator>(device_id, capacity);
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
if (!discovery.discover()) {
    throw std::runtime_error("Failed to discover hardware topology");
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
    .use_host_per_numa()
    .set_per_host_capacity(64ULL << 30)
    .set_reservation_fraction_per_host(0.85)
    .set_downgrade_fractions_per_host(0.85, 0.65)
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
| Host capacity | 4 GB |
| Host reservation fraction | 0.85 (85%) |
| Host downgrade trigger | 0.85 (85%) |
| Host downgrade stop | 0.65 (65%) |
| Host block size | 1 MB (`1 << 20`) |
| Host pool size | 128 blocks per pool |
| Host initial pools | 4 |
| Host creation policy | `use_host_per_numa()` |
| GPU allocator | `rmm::cuda_async_memory_resource` |
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
| `src/memory/topology_discovery.cpp` | NVML loading and `/sys` queries |
| `src/memory/reservation_manager_configurator.cpp` | Builder implementation |
