# cuCascade

A high-performance GPU memory management library for data-intensive applications requiring intelligent tiered memory allocation across GPU, host, and disk storage.

# Overview



**Key Features:**
- **Tiered Memory Management**: Seamlessly manage GPU (fastest), pinned host (medium), and disk (largest capacity) memory tiers, provides numa aware allocators
- **Memory Reservation System**: Avoid oversubscribing your GPU by making reservations and using allocators that respect reservations
- **Hardware Topology Discovery**: Automatic detection of NUMA regions and GPU-CPU affinity for optimal memory placement
- **Stream-Aware Tracking**: Per-stream memory usage tracking and reservation enforcement
- **cuDF Integration**: Optional `cucascade-cudf` library with cuDF table representations and converters for spilling batches to Host or Disk
- **Data Source I/O**: Optional `cucascade-io` library with io_uring and REST/S3 readers and a prefetching cache
- **Pluggable Policies**: Control what happens when you OOM or try to allocate more than a reservation by creating policies that plug into the system.

# Getting Started

```bash
# Option A: Using Pixi (recommended)
curl -fsSL https://pixi.sh/install.sh | bash
git clone https://github.com/nvidia/cuCascade.git
cd cuCascade
pixi install
pixi run build

# Build in debug mode
pixi run build-debug

# Option B: Using CMake directly
git clone https://github.com/nvidia/cuCascade.git
cd cuCascade
cmake --preset release
cmake --build build/release

# Run tests
pixi run test

# Run benchmarks (optional)
pixi run benchmarks
```

# Requirements

- **OS/Arch**: Linux (x86_64, aarch64)
- **Compiler**: C++20 compatible compiler
- **Build Tools**: CMake 4.0+, Ninja, sccache (used by the CMake presets)
- **GPU/Drivers**: CUDA 12.9+ or 13+, compatible NVIDIA driver
- **Dependencies**: RMM and libnuma (core); libcudf (only for the optional `cucascade-cudf` library); liburing, libcurl and OpenSSL (only for the optional `cucascade-io` library, which also needs kvikIO when built with `CUCASCADE_BUILD_CUDF=ON`)

> **Library layout:** the core `cuCascade::cucascade` target is cudf-free and depends only on
> RMM, CUDA, libnuma and Threads. The cudf-backed representations, converters, and bandwidth profiler
> live in a separate `cuCascade::cucascade_cudf` target (headers under `include/cucascade/cudf/`).
> The io layer (headers under `include/cucascade/io/` and `include/cucascade/exec/`) lives in the
> cudf-free `cuCascade::cucascade_io` target. These libraries can be turned off at configure time
> with `-DCUCASCADE_BUILD_CUDF=OFF` and `-DCUCASCADE_BUILD_IO=OFF`.
>
> cudf and io are exposed as package *components*, so core consumers never need cudf or the io
> dependencies installed:
>
> ```cmake
> find_package(cuCascade REQUIRED)                  # cudf-free core
> target_link_libraries(my_app PRIVATE cuCascade::cucascade)
>
> find_package(cuCascade REQUIRED COMPONENTS cudf)  # also resolves cudf (and the io dependencies if io was built)
> target_link_libraries(my_app PRIVATE cuCascade::cucascade_cudf)
>
> find_package(cuCascade REQUIRED COMPONENTS io)    # also resolves liburing, libcurl, OpenSSL
> target_link_libraries(my_app PRIVATE cuCascade::cucascade_io)
> ```
>
> See [Using cuCascade as a Dependency](docs/development-guide.md#using-cucascade-as-a-dependency)
> for dependencies the io layer's consumers must provide themselves.

# Usage

```cpp
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <cucascade/memory/topology_discovery.hpp>

using namespace cucascade::memory;

// 1. Discover hardware topology
topology_discovery discovery;
[[maybe_unused]] bool discovered = discovery.discover();
auto const& topology = discovery.get_topology();
if (topology.gpus.empty()) {
    throw std::runtime_error("No GPUs discovered");
}

// 2. Configure the memory reservation manager
reservation_manager_configurator configurator;
configurator.set_gpu_usage_limit(4ULL << 30)                // 4GB per GPU
            .set_reservation_fraction_per_gpu(0.8)          // Reserve up to 80%
            .set_per_numa_region_capacity(16ULL << 30)      // 16GB per NUMA region
            .use_numa_id_as_host_id();                      // One host space per NUMA node

// 3. Create the manager with the discovered topology
auto configs = configurator.build(topology);
memory_reservation_manager manager(std::move(configs));

// 4. Request reservations using strategies
// Request 1GB on any available GPU
auto gpu_res = manager.request_reservation(any_memory_space_in_tier(Tier::GPU), 1ULL << 30);

// Request 2GB on any host memory space
auto host_res = manager.request_reservation(any_memory_space_in_tier(Tier::HOST), 2ULL << 30);
```

- More examples: See `test/` directory for comprehensive usage examples

# Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Architecture Overview](docs/ARCHITECTURE.md)**: High-level description of the library's design, core components, and intended usage flows with Mermaid diagrams.
- **[Documentation Index](docs/README.md)**: All design documents (memory and data management, topology and configuration, development guide) with suggested reading paths.
- **API Reference** (`docs/API_REFERENCE.md`): Detailed API documentation and class hierarchies automatically generated from code comments. It is not checked in.

To generate the API reference:
```bash
pixi run docs
```

# Contribution Guidelines

- Start here: [Development Guide](docs/development-guide.md)
- Development quickstart:

```bash
git clone https://github.com/nvidia/cuCascade.git
cd cuCascade
pixi install
pixi run build
pixi run build-debug  # Debug mode
pixi run test

# Run benchmarks
pixi run benchmarks
```

## Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) for code quality checks including C++/CUDA formatting (clang-format), CMake linting, spell checking, and more.

```bash
# Run all checks manually
pixi run lint

# Install hooks to run automatically on every commit
pixi run lint-install

# Update hook versions
pixi run pre-commit autoupdate
```

## Security

- Vulnerability disclosure: `SECURITY.md`
- Do not file public issues for security reports.
- Report vulnerabilities via: https://www.nvidia.com/object/submit-security-vulnerability.html

## Support

- Level: Experimental
- How to get help: GitHub Issues
- For NVIDIA product security concerns: https://www.nvidia.com/en-us/security

# Project Structure

```
cuCascade/
├── include/cucascade/
│   ├── error.hpp, nvtx.hpp        # Error macros and NVTX ranges (core)
│   ├── data/                      # Data batches, repositories, disk tier (core; key headers)
│   │   ├── common.hpp             # idata_representation base class
│   │   ├── data_batch.hpp         # Batch processing for data
│   │   ├── data_repository.hpp    # Data storage abstraction
│   │   ├── data_repository_manager.hpp
│   │   ├── disk_data_representation.hpp  # Disk-tier representation
│   │   ├── disk_io_backend.hpp    # Pluggable disk I/O backend interface
│   │   └── representation_converter.hpp  # Converter registry
│   ├── memory/                    # Memory management headers (core; key headers)
│   │   ├── common.hpp             # Tier enum, memory_space_id, utilities
│   │   ├── memory_reservation_manager.hpp  # Central reservation coordinator
│   │   ├── memory_reservation.hpp # Reservation types and policies
│   │   ├── memory_space.hpp       # Memory space abstraction
│   │   ├── reservation_aware_resource_adaptor.hpp  # GPU memory resource
│   │   ├── fixed_size_host_memory_resource.hpp     # Host memory resource
│   │   ├── disk_access_limiter.hpp                 # Disk tier limiter
│   │   ├── reservation_manager_configurator.hpp    # Builder for config
│   │   ├── topology_discovery.hpp # Hardware topology detection
│   │   ├── numa_region_pinned_host_allocator.hpp   # NUMA-aware allocator
│   │   ├── notification_channel.hpp   # Cross-reservation signaling
│   │   ├── stream_pool.hpp        # CUDA stream management
│   │   └── oom_handling_policy.hpp    # OOM handling strategies
│   ├── cuda/                      # CUDA stream, event and copy helpers (core)
│   ├── utils/                     # Atomics, cleanup and exception-logging helpers (core)
│   ├── log/                       # Logging macros (core)
│   ├── cudf/                      # cudf representations and converters (cucascade-cudf)
│   ├── io/                        # Data sources and cache (cucascade-io)
│   └── exec/                      # Thread pool and futures (cucascade-io)
├── src/
│   ├── cuda/, data/, memory/      # Core library implementation
│   ├── cudf/                      # cucascade-cudf implementation
│   └── exec/, io/                 # cucascade-io implementation
├── test/
│   ├── data/                      # Data tests (core and cudf-layer converters)
│   ├── memory/                    # Memory module tests
│   ├── cudf/                      # cudf converter and cudf datasource/cache tests
│   ├── io/                        # io layer tests
│   └── utils/                     # Test utilities (cuDF helpers)
├── benchmark/                     # Performance benchmarks
│   ├── benchmark_*.cpp            # Converter benchmarks (cucascade_benchmarks)
│   ├── *parquet*_benchmark.cpp    # Parquet/io datasource benchmarks (cucascade-io)
│   └── README.md                  # Benchmark documentation
├── cmake/                         # CMake configuration modules
├── CMakeLists.txt                 # Main CMake configuration
├── CMakePresets.json              # CMake presets for build configurations
└── pixi.toml                      # Pixi dependency management
```

# References

- [RAPIDS cuDF](https://github.com/rapidsai/cudf) - GPU DataFrame library
- [Pixi](https://pixi.sh/) - Package management tool

# License

This project is licensed under the Apache License 2.0 - see the [LICENSE.md](LICENSE.md) file for details.
