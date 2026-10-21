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
- **Build Tools**: CMake 4.1+, Ninja
- **GPU/Drivers**: CUDA 13+, compatible NVIDIA driver
- **Dependencies**: RMM (core); libcudf 25.10+ (only for the optional `cucascade-cudf` library)

> **Library layout:** the core `cuCascade::cucascade` target is cudf-free and depends only on
> RMM + CUDA (plus libnuma and kvikIO/cuFile for the disk tier). The cudf-backed representations,
> converters, and bandwidth profiler live in a separate `cuCascade::cucascade_cudf` target (headers
> under `include/cucascade/cudf/`). The cudf library can be turned off at configure time with
> `-DCUCASCADE_BUILD_CUDF=OFF`.
>
> cudf is exposed as a package *component*, so core consumers never need cudf installed:
>
> ```cmake
> find_package(cuCascade REQUIRED)                  # cudf-free core
> target_link_libraries(my_app PRIVATE cuCascade::cucascade)
>
> find_package(cuCascade REQUIRED COMPONENTS cudf)  # also resolves cudf
> target_link_libraries(my_app PRIVATE cuCascade::cucascade_cudf)
> ```

# Usage

## Relocating encoded batches

`encoded_data_representation` owns an encoded payload in DISK, HOST, or GPU memory. Its encoding
identifier, version, logical uncompressed size, and decoder metadata travel with the payload.
`mutable_data_batch::relocate_to()` copies the encoded bytes to a destination memory space and
releases the old storage after the transfer completes. It preserves batch identity and leaves the
original representation installed if allocation, a copy, or I/O fails.
The class is part of the `cucascade_cudf` library (`cuCascade::cucascade_cudf`), although its
interface has no cuDF dependency.

```cpp
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/encoded_data_representation.hpp>

// disk_space, host_space, and gpu_space are memory::memory_space instances.
// decoder_metadata contains the schema/chunk information required by your codec.
cucascade::encoded_data_metadata metadata{
  .encoding          = "my_codec",
  .version           = 1,
  .uncompressed_size = original_size,
  .metadata          = decoder_metadata,
};

auto encoded = cucascade::encoded_data_representation::from_disk(
  file_path, file_offset, compressed_size, std::move(metadata), disk_space,
  cucascade::encoded_file_ownership::borrowed);
auto batch = cucascade::data_batch::make(batch_id, std::move(encoded));
{
  auto writable = batch->to_mutable();
  writable.relocate_to(&host_space, stream);  // Still encoded.
  writable.relocate_to(&gpu_space, stream);   // Still encoded.
  writable.relocate_to(&disk_space, stream);  // Still encoded, in a new file.

  // Decode explicitly, after registering your encoding's converter:
  // writable.convert_to<gpu_table_representation>(converters, &gpu_space, stream);
}
```

Use `from_host()` to copy encoded bytes into host blocks or `from_device()` to copy a device
buffer into the GPU memory space. `from_disk()` adopts a file offset and length without reading
it; borrowed files remain owned by the caller, while `encoded_file_ownership::owned` transfers
file deletion to the representation after successful construction. Relocated disk copies are new
files under the DISK space's mount path, hold the payload at offset 0, and own their files. DISK
transfers use the space's io context (`memory_space::set_io_context()`, or the process-shared
default); a GPU payload is written from device memory when that context can write device memory
and through bounded host staging otherwise. Memory spaces must outlive their representations, as
with the existing table representations.

`get_payload_size()` reports the encoded byte count; `get_size_in_bytes()` reports the storage
footprint, including host block rounding. `get_uncompressed_data_size_in_bytes()` reports the
logical size supplied in the metadata. Reserve destination capacity for the physical footprint:
`encoded_data_representation::relocation_reservation_bytes(space, payload_size)` returns the
bytes a destination reservation must cover. The HOST/DISK reservation overload is
`writable.relocate_to(*reservation, stream)`; a DISK reservation is validated but not consumed,
and must outlive the copy's file. For GPU reservations, first transfer the
reservation to the GPU allocator's stream tracker using `attach_reservation_to_tracker()`, then
use `relocate_to(&gpu_space, stream)`. Passing a caller-owned GPU reservation is rejected because
attaching it consumes its ownership. The participating GPU owns `stream`; for cross-GPU
relocation, use a stream on the destination GPU. Relocation is synchronous, including disk I/O.

Codec implementations register explicit converters through `representation_converter_registry`.
Relocation moves already encoded payloads and never decodes them. Existing representations can
opt in by overriding `idata_representation::relocate()`; their default implementation reports
unsupported relocation. Moving a batch to its current memory space is a no-op. Applications must
rebuild against the updated representation interface.

The `cucascade_relocation_benchmarks` executable compares all six transfer directions with raw
copies through the same io context. DISK cases create their files in `$CUCASCADE_BENCH_DISK_DIR`
(default `/tmp`); writes measure request completion and exclude disk writeback. Throughput
counters count encoded payload bytes.

## Configuring memory spaces

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
├── benchmark/                     # Performance benchmarks
│   ├── benchmark_representation_converter.cpp  # Converter benchmarks
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
