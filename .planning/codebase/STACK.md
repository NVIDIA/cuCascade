# Technology Stack

**Analysis Date:** 2026-04-13

## Languages

**Primary:**
- C++20 - All library source, headers, tests, and benchmarks
- CUDA 20 - GPU kernel code in `src/` (CUDA standard applied privately to object library)

**Secondary:**
- C - Listed in CMake LANGUAGES (C CXX CUDA) for compatibility
- Python - Documentation generation scripts in `scripts/` (Jinja2-based API doc generator)

## Runtime

**Environment:**
- Linux (linux-64 or linux-aarch64 only; explicitly declared in `pixi.toml`)
- NVIDIA GPU required at runtime (Turing/SM75 minimum, up to SM120 supported)

**Package Manager:**
- Pixi >= 0.59 (conda-based, defined in `pixi.toml`)
- No lockfile committed (cache disabled in CI)

## Frameworks

**Core:**
- libcudf 26.06 (nightly) / 26.02 (stable) - GPU DataFrame library; provides `cudf::table`, pack/unpack serialization, and column memory management
- RMM (RAPIDS Memory Manager) - Device memory resource abstraction; base class `rmm::mr::device_memory_resource` is subclassed throughout `include/cucascade/memory/`
- CUDA Toolkit (12.9 or 13.x) - `CUDA::cudart` for stream management, `CUDA::nvml` (dev headers) for topology discovery

**Testing:**
- Catch2 v2.13.10 - Fetched via CMake `FetchContent` from GitHub at build time; single-include header used in `test/`
- CTest - CMake test runner; registered via `add_test(NAME cucascade_tests ...)`

**Benchmarking:**
- Google Benchmark v1.8.3 - Fetched via CMake `FetchContent` from GitHub at build time; used in `benchmark/`

**Build/Dev:**
- CMake 4.x (pixi dep), minimum 3.26.4 (enforced in `CMakeLists.txt`)
- Ninja - Build generator (set in `CMakePresets.json` default preset)
- sccache - Compiler cache; applied to C, C++, and CUDA compilers via `CMAKE_*_COMPILER_LAUNCHER` in `CMakePresets.json`
- Doxygen - API documentation generation from header comments
- pre-commit - Linting/formatting gate (see `.pre-commit-config.yaml`)

## Key Dependencies

**Critical:**
- `rmm::rmm` - Every memory resource in `include/cucascade/memory/` subclasses `rmm::mr::device_memory_resource`; RMM stream types (`rmm::cuda_stream_view`, `rmm::cuda_stream`) used throughout APIs
- `cudf::cudf` - `gpu_table_representation` wraps `cudf::table`; `cudf::pack`/`unpack` used for GPU→HOST serialization in built-in converters; direct cudf API access is the GPU data path
- `CUDA::cudart` - Stream creation/synchronization in `exclusive_stream_pool` (`include/cucascade/memory/stream_pool.hpp`)
- `CUDA::nvml` (headers only) - `nvml.h` included in `src/memory/topology_discovery.cpp`; loaded at runtime via `dlopen("libnvidia-ml.so.1")` to avoid link-time dependency
- `numa` (libnuma) - Linked as `${NUMA_LIB}`; used in `numa_region_pinned_host_memory_resource` (`include/cucascade/memory/numa_region_pinned_host_allocator.hpp`) for NUMA-aware pinned host allocation
- `Threads::Threads` - std::mutex, std::shared_mutex, std::condition_variable used in `memory_reservation_manager`, `exclusive_stream_pool`, and `synchronized_data_batch`
- `cuda::mr` (CCCL) - `cuda::mr::device_accessible` and `cuda::mr::host_accessible` properties declared on `numa_region_pinned_host_memory_resource`

**Infrastructure:**
- `fmt` (pixi dep) - Available but not observed directly in headers; likely used in implementation files
- `libcurand-dev` (pixi dep) - Available for random data generation (used in tests/benchmarks)
- `numactl` (pixi dep) - Provides libnuma at build time

## Configuration

**Environment:**
- `CUDA_VISIBLE_DEVICES` - Respected by `topology_discovery::discover()` to filter visible GPU indices (supports numeric indices, GPU UUIDs, and MIG device UUIDs)
- `CUDAARCHS` - Set per pixi environment feature (e.g., `75-real;80-real;86-real;90a-real;100f-real;120a-real;120` for CUDA 13)
- `CMAKE_PREFIX_PATH` - Passed through from environment for locating RMM/cuDF/CUDAToolkit packages
- `SCCACHE_GHA_ENABLED` - Set to `true` in CI for GitHub Actions sccache integration

**Build:**
- `CMakePresets.json` - Defines `debug`, `release`, `relwithdebinfo` presets; all use Ninja generator and sccache
- Build outputs: `build/debug/`, `build/release/`, `build/relwithdebinfo/`
- CMake options:
  - `CUCASCADE_BUILD_TESTS` (default ON)
  - `CUCASCADE_BUILD_BENCHMARKS` (default ON)
  - `CUCASCADE_BUILD_SHARED_LIBS` (default ON)
  - `CUCASCADE_BUILD_STATIC_LIBS` (default ON)
  - `CUCASCADE_NVTX` (default OFF) - Enables NVTX3 range annotations when ON
  - `CUCASCADE_WARNINGS_AS_ERRORS` (default ON)

## Platform Requirements

**Development:**
- Linux x86_64 or aarch64
- NVIDIA GPU (SM75+ / Turing+)
- CUDA 12.9 or 13.x toolkit installed
- Pixi >= 0.59

**Production:**
- Linux only (topology discovery reads `/sys/bus/pci/devices/`, `/sys/class/infiniband`, `/sys/class/nvme`)
- NVIDIA driver with `libnvidia-ml.so.1` available at runtime (topology discovery degrades gracefully without it)
- Optional: InfiniBand/RoCE NICs (discovered via `/sys/class/infiniband`) and NVMe drives (via `/sys/class/nvme`)
- Library outputs: `libcucascade.so` (versioned 0.1.0) and/or `libcucascade.a`

---

*Stack analysis: 2026-04-13*
