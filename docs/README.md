# cuCascade Documentation

Comprehensive documentation for the cuCascade GPU memory management library.

## Quick Navigation

| Document | Description |
|----------|-------------|
| [Architecture Overview](ARCHITECTURE.md) | High-level system design, component relationships, and data flow |
| [Memory Management](memory-management.md) | Deep dive into the tiered memory system (GPU, HOST, DISK, REMOTE), reservations, and allocators |
| [Data Management](data-management.md) | Data batch lifecycle, repositories, and representation conversion |
| [Topology & Configuration](topology-and-configuration.md) | Hardware discovery, NUMA awareness, and system configuration |
| [Development Guide](development-guide.md) | Building, testing, benchmarking, and contributing |
| [Data Batch State Transitions](data_batch_state_transitions.md) | State machine reference for `data_batch` |
| [GDS reads](io/gpuio-gds-practices.md) | The cuFile / GPUDirect Storage read backend: configuration, fallbacks, capability probe, measurements |
| [REST over io_uring](io/rest-uring-design.md) | The S3 / object-store backend: io_uring HTTP/1.1 transport, kTLS and user-space TLS, S3 writes |
| [REMOTE tier](io/remote-tier-design.md) | Spilling batches to an S3 object store: configuration, object naming, failure handling, lifecycle-rule requirement |

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

cuCascade is organized into two core modules:

- **Memory Module** (`include/cucascade/memory/`, `src/memory/`): Manages a tiered memory hierarchy (GPU, HOST, DISK, REMOTE object store) with reservation-based allocation, per-stream tracking, NUMA-aware host allocation, and pluggable OOM/limit policies.

- **Data Module** (`include/cucascade/data/`, `src/data/`): Manages data lifecycle across memory tiers with a state machine for batch processing, type-indexed representation converters, and partitioned repository storage for multi-pipeline coordination.

## Related Resources

- [Main README](../README.md) -- project overview, quick start, and usage examples
- [RAPIDS cuDF](https://github.com/rapidsai/cudf) -- GPU DataFrame library (core dependency)
- [RMM](https://github.com/rapidsai/rmm) -- RAPIDS Memory Manager (memory allocation backend)
- [Pixi](https://pixi.sh/) -- Package management tool used for builds
