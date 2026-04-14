# External Integrations

**Analysis Date:** 2026-04-13

## APIs & External Services

**NVIDIA RAPIDS (libcudf / RMM):**
- libcudf - GPU DataFrame processing library; provides `cudf::table`, `cudf::column`, pack/unpack serialization for GPU-to-host data movement
  - SDK/Client: `cudf::cudf` CMake target (found via `find_package(cudf REQUIRED CONFIG)`)
  - Used in: `include/cucascade/data/gpu_data_representation.hpp`, `include/cucascade/data/cpu_data_representation.hpp`, `src/data/`
  - Channel: `rapidsai-nightly` (default) or `rapidsai` (stable), via pixi/conda
- RMM (RAPIDS Memory Manager) - Memory resource abstraction layer
  - SDK/Client: `rmm::rmm` CMake target (found via `find_package(rmm REQUIRED CONFIG)` from libcudf installation)
  - Used in: All `include/cucascade/memory/` headers; every custom memory resource subclasses `rmm::mr::device_memory_resource`
  - Key types consumed: `rmm::cuda_stream_view`, `rmm::cuda_stream`, `rmm::cuda_device_id`, `rmm::mr::device_memory_resource`

**NVIDIA CUDA Toolkit:**
- CUDA Runtime - Stream and device management
  - SDK/Client: `CUDA::cudart` CMake target (found via `find_package(CUDAToolkit REQUIRED)`)
  - Used in: `include/cucascade/memory/stream_pool.hpp`, GPU kernel in `test/memory/test_gpu_kernels.cu`
- NVML (NVIDIA Management Library) - GPU/system topology discovery
  - SDK/Client: `CUDA::nvml` for headers (`nvml.h`); runtime-loaded via `dlopen("libnvidia-ml.so.1")`
  - Used in: `src/memory/topology_discovery.cpp` exclusively
  - Note: Loaded dynamically at runtime; NVML unavailability is handled gracefully (fallback to reporting system info without GPU data)
- CCCL (cuda-cpp-core-libraries) - `cuda::mr` property tags (`cuda::mr::device_accessible`, `cuda::mr::host_accessible`) declared on `numa_region_pinned_host_memory_resource`
  - Used in: `include/cucascade/memory/numa_region_pinned_host_allocator.hpp`

## Data Storage

**Databases:**
- Not applicable — cuCascade is itself a tiered storage/caching library, not a consumer of a database

**In-Process Memory Tiers (managed internally):**
- GPU device memory - Managed via `rmm::mr::device_memory_resource` subclasses; tracked in `memory_space` with tier `Tier::GPU`
  - Config: `gpu_memory_space_config` in `include/cucascade/memory/config.hpp`
- Host (NUMA-pinned) memory - Managed via `small_pinned_host_memory_resource` and `fixed_size_host_memory_resource`; tracked with tier `Tier::HOST`
  - Config: `host_memory_space_config` in `include/cucascade/memory/config.hpp`
  - NUMA binding: `numa_region_pinned_host_memory_resource` uses `libnuma` for NUMA-local pinned allocation
- Disk/NVMe - Tracked and capacity-limited via `disk_access_limiter` (`include/cucascade/memory/disk_access_limiter.hpp`); uses filesystem paths
  - Config: `disk_memory_space_config` with `mount_paths` field in `include/cucascade/memory/config.hpp`
  - Mount path configured at `disk_memory_space_config::mount_paths`; no env var, must be passed programmatically

**File Storage:**
- Local filesystem (disk tier) - Data written to/read from paths under `disk_memory_space_config::mount_paths`; files identified by base filename in `disk_reserved_arena::_base_name`

**Caching:**
- cuCascade itself is the cache/tiered storage system; no external cache service consumed

## Authentication & Identity

**Auth Provider:**
- None — this is a C++ system library with no network API endpoints, authentication, or identity concepts

## Monitoring & Observability

**Error Tracking:**
- None (no external error tracking service integrated)

**Profiling (optional):**
- NVTX3 - NVIDIA Tools Extension for GPU profiling; opt-in via CMake option `CUCASCADE_NVTX=ON`
  - CMake target: `nvtx3::nvtx3-cpp` (found via `find_package(nvtx3 REQUIRED)`)
  - Controlled by compile definition: `CUCASCADE_NVTX`
  - Default: OFF

**Logs:**
- `std::cerr` only - Used in `src/memory/topology_discovery.cpp` for NVML warnings (e.g., "NVML not available", "Failed to get device count")
- No structured logging framework

## CI/CD & Deployment

**Hosting:**
- NVIDIA internal GitHub Actions runners: `linux-amd64-cpu4` (build), `linux-amd64-gpu-t4-latest-1` (test/benchmark)

**CI Pipeline:**
- GitHub Actions (`.github/workflows/ci.yml`)
  - Triggers: push to `main` or `pull-request/[0-9]+` branches
  - Matrix: CUDA 12 x RAPIDS nightly/stable, CUDA 13 x RAPIDS nightly/stable (4 configurations for build+test; benchmarks only on CUDA 13 nightly)
  - Steps: checkout → setup-pixi → sccache → `pixi run build` → artifact upload → `pixi run test`
  - Benchmark comparison uses `benchmark-action/github-action-benchmark` with 120% regression threshold; comments on PRs when regression detected
  - Baseline benchmark data cached in GitHub Actions cache under key `benchmark_result-linux-amd64-gpu-t4-latest-1-v1`

**Pre-commit Hooks (`.pre-commit-config.yaml`):**
- `clang-format` v20.1.4 - C/C++/CUDA formatting via `.clang-format` (100-column limit, WebKit brace style, C++20 standard)
- `cmake-format` + `cmake-lint` v0.6.13 - CMake file formatting and linting
- `black` 25.1.0 - Python script formatting
- `codespell` v2.4.1 - Spell checking with custom word list at `.codespell_words`
- Standard pre-commit hooks: large file check, merge conflict detection, JSON/YAML/TOML validation, trailing whitespace

## External Fetch Dependencies (CMake FetchContent)

**Catch2 v2.13.10:**
- Source: `https://github.com/catchorg/Catch2.git`
- Fetched at: test build time (`test/CMakeLists.txt`)
- Not vendored; requires internet access or CMake cache during build

**Google Benchmark v1.8.3:**
- Source: `https://github.com/google/benchmark.git`
- Fetched at: benchmark build time (`benchmark/CMakeLists.txt`)
- Not vendored; requires internet access or CMake cache during build

## Webhooks & Callbacks

**Incoming:**
- None — library has no web server or API endpoints

**Outgoing:**
- None — no HTTP/network calls at runtime

## System-Level Interfaces

**Linux /sys filesystem (read-only at runtime):**
- `/sys/bus/pci/devices/<pci>/numa_node` - NUMA node for GPU and NIC devices
- `/sys/bus/pci/devices/<pci>/local_cpulist` - CPU affinity for GPU devices
- `/sys/class/infiniband/` - InfiniBand/RoCE NIC discovery
- `/sys/class/nvme/` - NVMe storage device discovery
- `/sys/devices/system/node/node*` - NUMA node count
- All reads in `src/memory/topology_discovery.cpp`

**Environment Variables Consumed:**
- `CUDA_VISIBLE_DEVICES` - Standard NVIDIA env var for GPU visibility filtering; parsed by `topology_discovery::discover()` to remap device indices
- `CUDAARCHS` - Set by pixi feature activation; consumed by CUDA compiler for target architecture selection
- `CMAKE_PREFIX_PATH` - Required at CMake configure time to locate conda-installed packages (RMM, cuDF, CUDAToolkit)
- `SCCACHE_GHA_ENABLED` - Set in CI to activate GitHub Actions sccache backend

---

*Integration audit: 2026-04-13*
