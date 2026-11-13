# cuCascade Documentation

Comprehensive documentation for the cuCascade GPU memory management library.

## Quick Navigation

| Document | Description |
|----------|-------------|
| [Architecture Overview](ARCHITECTURE.md) | High-level system design, component relationships, and data flow |
| [Memory Management](memory-management.md) | Deep dive into the three-tier memory system, reservations, and allocators |
| [Data Management](data-management.md) | Data batch lifecycle, repositories, and representation conversion |
| [Topology & Configuration](topology-and-configuration.md) | Hardware discovery, NUMA awareness, and system configuration |
| [Development Guide](development-guide.md) | Building, testing, benchmarking, and contributing |
| [Data Batch State Transitions](data_batch_state_transitions.md) | State machine reference for `data_batch` |
| [Bandwidth Profiler](bandwidth-profiler.md) | Design of `measure_bandwidth`, which measures pairwise transfer throughput between memory spaces |

## Reading Paths

**"I want to understand what cuCascade does"**
1. [Architecture Overview](ARCHITECTURE.md) -- start here for the big picture
2. [Memory Management](memory-management.md) -- understand the core memory tier system
3. [Data Management](data-management.md) -- understand how data moves between tiers

**"I want to use cuCascade in my project"**
1. [Architecture Overview](ARCHITECTURE.md) -- understand the design
2. [Topology & Configuration](topology-and-configuration.md) -- learn how to configure the system
3. [Development Guide](development-guide.md) -- build and integration instructions

**"I want to contribute to cuCascade"**
1. [Development Guide](development-guide.md) -- build, test, and code quality setup
2. [Architecture Overview](ARCHITECTURE.md) -- understand the component relationships
3. Deep dive into [Memory Management](memory-management.md) or [Data Management](data-management.md) as needed

## Module Overview

cuCascade is built as layered libraries. At the bottom, `cucascade_topology_discovery` provides NVML/sysfs hardware topology discovery (`src/memory/topology_discovery.cpp`; the only library built with `CUCASCADE_TOPOLOGY_ONLY`). On top of it, the cudf-free core library (`cucascade`) contains two main modules, plus CUDA event and batched device-copy helpers (`include/cucascade/cuda/`, `src/cuda/`):

- **Memory Module** (`include/cucascade/memory/`, `src/memory/`): Manages a three-tier memory hierarchy (GPU, HOST, DISK) with reservation-based allocation, per-stream tracking, NUMA-aware host allocation, and pluggable OOM/limit policies.

- **Data Module** (`include/cucascade/data/`, `src/data/`): Manages data lifecycle across memory tiers with reader/writer-locked batches (`idle`, `read_only` and `mutable_locked` states, accessed through the RAII `read_only_data_batch`/`mutable_data_batch` handles), type-indexed representation converters, and partitioned (optionally position-ordered) repository storage for multi-pipeline coordination.

Optional layers (built by default; disable with the options below) build on the core:

- **cuDF Layer** (`include/cucascade/cudf/`, `src/cudf/`; library `cucascade_cudf`, option `CUCASCADE_BUILD_CUDF`): cuDF-backed GPU and host representations, the built-in GPU/host/disk converters, the bandwidth profiler, and (when `CUCASCADE_BUILD_IO` is on) the `cudf::io::datasource` bridge to the I/O layer.

- **I/O and Execution Layer** (`include/cucascade/io/`, `include/cucascade/exec/`, `src/io/`, `src/exec/`; library `cucascade_io`, option `CUCASCADE_BUILD_IO`): Async I/O contexts (`ioctx`: io_uring for local files, REST/S3 and S3-over-RDMA for object stores) selected per path by `io_context_registry`, a prefetching cache, and the thread pool, futures and admission control they run on. The `cudf::io::datasource` built on these lives in the cuDF layer. See the [REST transport design](io/rest-transport.md) and the I/O headers.

## Related Resources

- [Main README](../README.md) -- project overview, quick start, and usage examples
- [RAPIDS cuDF](https://github.com/rapidsai/cudf) -- GPU DataFrame library (required only by the cuDF layer)
- [RMM](https://github.com/rapidsai/rmm) -- RAPIDS Memory Manager (memory allocation backend)
- [Pixi](https://pixi.sh/) -- Package management tool used for builds
