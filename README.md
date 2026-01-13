# cuCascade

A high-performance GPU memory management library for data-intensive applications requiring intelligent tiered memory allocation across GPU, host, and disk storage.

# Overview



**Key Features:**
- **Tiered Memory Management**: Seamlessly manage GPU (fastest), pinned host (medium), and disk (largest capacity) memory tiers, provides numa aware allocators
- **Memory Reservation System**: Avoid oversubscribing your GPU by making reservations and using allocators that respect reservations
- **Hardware Topology Discovery**: Automatic detection of NUMA regions and GPU-CPU affinity for optimal memory placement
- **Stream-Aware Tracking**: Per-stream memory usage tracking and reservation enforcement
- **cuDF Integration**: Native support for GPU DataFrames with batch processing capabilities and spilling to Host or Disk
- **Pluggable Policies**: Control what happens when you OOM, try to allocate more than a reservation, how you pick what data to spill, by creating policies that plug into the system.

# Getting Started

```bash
# Option A: Using Pixi (recommended)
curl -fsSL https://pixi.sh/install.sh | bash
git clone https://github.com/nvidia/cuCascade.git
cd cuCascade
pixi install
pixi run build

# Option B: Using CMake directly
git clone https://github.com/nvidia/cuCascade.git
cd cuCascade
cmake --preset release
cmake --build build/release

# Verify installation
cd build/release && ctest --output-on-failure
```

# Requirements

- **OS/Arch**: Linux (x86_64, aarch64)
- **Compiler**: C++20 compatible compiler
- **Build Tools**: CMake 4.1+, Ninja
- **GPU/Drivers**: CUDA 13+, compatible NVIDIA driver
- **Dependencies**: libcudf 25.10+

# Usage

```cpp
#include <memory/memory_reservation_manager.hpp>
#include <memory/reservation_manager_configurator.hpp>
#include <memory/topology_discovery.hpp>

using namespace cucascade::memory;

// 1. Discover hardware topology
topology_discovery discovery;
if (!discovery.discover()) {
    // Handle discovery failure
}
auto const& topology = discovery.get_topology();

// 2. Configure the memory reservation manager
reservation_manager_configurator configurator;
configurator.set_gpu_usage_limit(4ULL << 30)                // 4GB per GPU
            .set_reservation_limit_ratio_per_gpu(0.8)       // Reserve up to 80%
            .set_capacity_per_numa_node(16ULL << 30)        // 16GB per NUMA node
            .bind_cpu_tier_to_gpus();                       // Bind CPU tiers to GPUs

// 3. Create the manager with the discovered topology
auto configs = configurator.build(topology);
memory_reservation_manager manager(std::move(configs));

// 4. Request reservations using strategies
// Request 1GB on any available GPU
auto gpu_res = manager.request_reservation(any_memory_space_in_tier(Tier::GPU), 1ULL << 30);

// Request 2GB on a specific host NUMA node
auto host_res = manager.request_reservation(specific_memory_space(Tier::HOST, 0), 2ULL << 30);
```

- More examples: See `test/` directory for comprehensive usage examples

# Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Architecture Overview](docs/ARCHITECTURE.md)**: High-level description of the library's design, core components, and intended usage flows with Mermaid diagrams.
- **[API Reference](docs/API_REFERENCE.md)**: Detailed API documentation and class hierarchies automatically generated from code comments.

To regenerate the documentation:
```bash
pixi run docs
```

# Contribution Guidelines

- Start here: `CONTRIBUTING.md`
- Code of Conduct: `CODE_OF_CONDUCT.md`
- Development quickstart:

```bash
git clone https://github.com/nvidia/cuCascade.git
cd cuCascade
pixi install
pixi run build
pixi run test
```

## Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) for code quality checks including C++/CUDA formatting (clang-format), CMake linting, spell checking, and more.

```bash
# Run all checks manually
pixi run lint

# Install hooks to run automatically on every commit
pixi run lint-install

# Update hook versions
pre-commit autoupdate
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
├── include/
│   ├── data/                      # Data representation headers
│   │   ├── common.hpp             # Common data utilities
│   │   ├── data_batch.hpp         # Batch processing for data
│   │   ├── data_repository.hpp    # Data storage abstraction
│   │   ├── data_repository_manager.hpp
│   │   ├── cpu_data_representation.hpp
│   │   └── gpu_data_representation.hpp
│   └── memory/                    # Memory management headers
│       ├── common.hpp             # Tier enum, memory_space_id, utilities
│       ├── memory_reservation_manager.hpp  # Central reservation coordinator
│       ├── memory_reservation.hpp # Reservation types and policies
│       ├── memory_space.hpp       # Memory space abstraction
│       ├── reservation_aware_resource_adaptor.hpp  # GPU memory resource
│       ├── fixed_size_host_memory_resource.hpp     # Host memory resource
│       ├── disk_access_limiter.hpp                 # Disk tier limiter
│       ├── reservation_manager_configurator.hpp    # Builder for config
│       ├── topology_discovery.hpp # Hardware topology detection
│       ├── numa_region_pinned_host_allocator.hpp   # NUMA-aware allocator
│       ├── notification_channel.hpp   # Cross-reservation signaling
│       ├── stream_pool.hpp        # CUDA stream management
│       └── oom_handling_policy.hpp    # OOM handling strategies
├── src/
│   ├── data/                      # Data representation implementation
│   └── memory/                    # Memory management implementation
├── test/
│   ├── data/                      # Data module tests
│   ├── memory/                    # Memory module tests
│   └── utils/                     # Test utilities (cuDF helpers)
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
