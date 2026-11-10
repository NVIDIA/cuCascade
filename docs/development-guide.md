# Development Guide

Building, testing, benchmarking, and contributing to cuCascade.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Build System](#build-system)
  - [CMake Structure](#cmake-structure)
  - [Build Options](#build-options)
  - [Build Presets](#build-presets)
  - [Pixi Environments](#pixi-environments)
  - [Compilation Caching](#compilation-caching)
  - [Adding New Source Files](#adding-new-source-files)
- [Testing](#testing)
  - [Test Framework](#test-framework)
  - [Test Organization](#test-organization)
  - [Test Utilities](#test-utilities)
  - [GPU Pool Configuration](#gpu-pool-configuration)
  - [Running Specific Tests](#running-specific-tests)
- [Benchmarking](#benchmarking)
  - [Available Benchmarks](#available-benchmarks)
  - [Running Benchmarks](#running-benchmarks)
  - [Regression Detection](#regression-detection)
- [Code Quality](#code-quality)
  - [Pre-commit Hooks](#pre-commit-hooks)
  - [Formatting](#formatting)
- [Documentation](#documentation)
- [CI Pipeline](#ci-pipeline)
- [Using cuCascade as a Dependency](#using-cucascade-as-a-dependency)
- [Key Commands Reference](#key-commands-reference)

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| OS | Linux (x86_64 or aarch64) |
| C++ Compiler | C++20 compatible |
| CMake | 4.0+ |
| Ninja | Any |
| sccache | Any (the CMake presets use it as compiler launcher) |
| CUDA Toolkit | 12.9+ (CUDA 12 and 13 are both built in CI) |
| NVIDIA GPU | Compute capability 7.5+ |
| NVIDIA Driver | Compatible with the CUDA Toolkit |
| RMM | Required by the core library (installed with libcudf) |
| libcudf | Nightly or stable RAPIDS release (see `pixi.toml`); only for `CUCASCADE_BUILD_CUDF` |
| numactl | Development headers |
| liburing, libcurl, OpenSSL, pkg-config | Required by `CUCASCADE_BUILD_IO` (ON by default); with `CUCASCADE_BUILD_CUDF` it also needs kvikIO (installed with libcudf) |

The easiest way to get all dependencies is via [Pixi](https://pixi.sh/), which manages the full environment.

## Getting Started

```bash
# Install Pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Clone the repository
git clone https://github.com/nvidia/cuCascade.git
cd cuCascade

# Install dependencies
pixi install

# Build (release mode)
pixi run build

# Build (debug mode)
pixi run build-debug

# Run tests
pixi run test

# Run benchmarks
pixi run benchmarks
```

Alternatively, build directly with CMake. The presets read dependencies from `CMAKE_PREFIX_PATH` and use `sccache` as compiler launcher, so run them inside the Pixi environment (`pixi shell`) or an equivalent one:

```bash
cmake --preset release
cmake --build build/release

# Run tests
cd build/release && ctest --output-on-failure
```

`pixi run test` runs `ctest` in `build/release`; it does not build first.

## Project Structure

```
cuCascade/
├── include/cucascade/
│   ├── memory/                  # Memory module headers
│   │   ├── common.hpp           # Tier enum, memory_space_id
│   │   ├── config.hpp           # Tier-specific config structs
│   │   ├── memory_reservation_manager.hpp
│   │   ├── memory_space.hpp
│   │   ├── memory_reservation.hpp
│   │   ├── reservation_aware_resource_adaptor.hpp
│   │   ├── fixed_size_host_memory_resource.hpp
│   │   ├── small_pinned_host_memory_resource.hpp
│   │   ├── disk_access_limiter.hpp
│   │   ├── disk_table.hpp
│   │   ├── topology_discovery.hpp
│   │   ├── reservation_manager_configurator.hpp
│   │   ├── notification_channel.hpp
│   │   ├── stream_pool.hpp
│   │   ├── oom_handling_policy.hpp
│   │   └── ...
│   ├── data/                    # Data module headers
│   │   ├── common.hpp           # idata_representation interface
│   │   ├── data_batch.hpp       # Batch lifecycle + RAII accessors
│   │   ├── data_repository.hpp  # Partitioned batch storage
│   │   ├── data_repository_manager.hpp
│   │   ├── representation_converter.hpp
│   │   ├── disk_data_representation.hpp
│   │   ├── disk_file_format.hpp
│   │   ├── disk_io_backend.hpp
│   │   └── io_backend_registry.hpp
│   ├── cuda/                    # CUDA stream/event wrappers, batched device copies
│   ├── utils/                   # atomics, overloaded, cleanup and error helpers
│   ├── cudf/                    # cudf-coupled layer: GPU/host representations, built-in converters,
│   │                            # bandwidth profiler, cudf datasource bridge
│   ├── io/                      # I/O layer: io_uring, REST/S3 and kvikIO contexts, prefetching cache
│   ├── exec/                    # Thread pool, semi-futures, admission control (part of the I/O layer)
│   ├── log/                     # Logging macros
│   ├── error.hpp                # CUCASCADE_CUDA_TRY, CUCASCADE_FAIL, CUCASCADE_FUNC_RANGE
│   └── nvtx.hpp                 # libcucascade NVTX domain
├── src/
│   ├── memory/                  # Memory module implementation (incl. topology discovery)
│   ├── data/                    # Data module implementation
│   ├── cuda/                    # CUDA helpers
│   ├── cudf/                    # cudf-coupled layer
│   ├── io/                      # I/O layer
│   └── exec/                    # Execution utilities for the I/O layer
├── test/
│   ├── unittest.cpp             # Test runner with GPU pool setup
│   ├── unittest_topology_discovery.cpp  # Test runner for topology discovery tests
│   ├── memory/                  # Memory module tests (+ GPU kernels)
│   ├── data/                    # Data module tests
│   ├── cudf/                    # cudf-coupled layer tests
│   ├── io/                      # I/O layer tests
│   └── utils/                   # Test utilities (mocks, cuDF helpers, allocation counter)
├── benchmark/                   # Google Benchmark suite + standalone parquet benchmarks
├── docs/                        # Documentation (you are here)
├── cmake/                       # CMake package config templates + Find modules
├── scripts/                     # Tooling scripts
│   ├── generate_api_docs.py     # Doxygen XML -> Markdown converter
│   ├── compare_benchmarks.py    # Benchmark regression detector
│   └── verify_gds_environment.sh
├── CMakeLists.txt               # Root build configuration
├── CMakePresets.json            # Build presets (debug, release, relwithdebinfo, s3-bench)
├── pixi.toml                    # Pixi environment + task definitions
├── Doxyfile                     # Doxygen configuration
└── .pre-commit-config.yaml      # Code quality hooks
```

## Build System

### CMake Structure

cuCascade uses an **object library pattern** for efficient compilation. Each library is compiled once into an OBJECT library, which is then linked into a STATIC and a SHARED library:

```
cucascade_topology_discovery_objects    src/memory/topology_discovery.cpp
    -> cucascade_topology_discovery_{static,shared}
         Links: NVML (static), NVTX, dl (no link-time libcuda dependency)

cucascade_objects (core)                src/{cuda,memory,data}/*.cpp (except topology discovery)
    -> cucascade_{static,shared}
         Links: RMM, CUDA runtime, NVTX, pthreads, numa, topology discovery

cucascade_io_objects                    src/{exec,io}/*.cpp            (CUCASCADE_BUILD_IO)
    -> cucascade_io_{static,shared}
         Links: core, liburing, libcurl, OpenSSL, kvikIO (only with CUCASCADE_BUILD_CUDF)

cucascade_cudf_objects                  src/cudf/*.cpp                 (CUCASCADE_BUILD_CUDF)
    -> cucascade_cudf_{static,shared}
         Links: core, cuDF, cucascade_io (if CUCASCADE_BUILD_IO)
```

The core library is cudf-free; only `cucascade_cudf` links libcudf.

Both library types are built by default. At least one must be enabled via `CUCASCADE_BUILD_STATIC_LIBS` and/or `CUCASCADE_BUILD_SHARED_LIBS`.

**Compiler warnings** are strict by default for the library sources and benchmarks (test sources are compiled without them):
```
-Wall -Wextra -Wpedantic -Wcast-align -Wunused -Wconversion
-Wsign-conversion -Wdouble-promotion -Wformat=2
-Wimplicit-fallthrough -Werror
```

Warnings-as-errors can be disabled with `-DCUCASCADE_WARNINGS_AS_ERRORS=OFF`.

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CUCASCADE_BUILD_TESTS` | `ON` | Build the test binaries |
| `CUCASCADE_BUILD_BENCHMARKS` | `ON` | Build the benchmarks (skipped unless `CUCASCADE_BUILD_CUDF=ON`) |
| `CUCASCADE_BUILD_SHARED_LIBS` | `ON` | Build the shared libraries |
| `CUCASCADE_BUILD_STATIC_LIBS` | `ON` | Build the static libraries |
| `CUCASCADE_BUILD_CUDF` | `ON` | Build the cudf-coupled `cucascade_cudf` library; `OFF` gives a cudf-free build |
| `CUCASCADE_BUILD_IO` | `ON` | Build the `cucascade_io` library (datasources, io_uring/REST reactors, prefetching cache) |
| `CUCASCADE_TOPOLOGY_ONLY` | `OFF` | Build only `cucascade_topology_discovery` (no RMM/cuDF); forces I/O and benchmarks off |
| `CUCASCADE_BUILD_S3_BENCHMARK` | `OFF` | Build the S3 parquet benchmarks; needs `aws-sdk-cpp` and the I/O, cudf and benchmark options |
| `CUCASCADE_WARNINGS_AS_ERRORS` | `ON` | Add `-Werror` |

`CMakeLists.txt` also has options for embedding hosts to supply their own moodycamel queue and abseil `AnyInvocable` (`CUCASCADE_MOODYCAMEL_*`, `CUCASCADE_USE_ABSEIL_INVOCABLE`, `CUCASCADE_ABSEIL_INVOCABLE_TARGET`).

### Build Presets

Defined in `CMakePresets.json`:

| Preset | Build Dir | Mode | Description |
|--------|-----------|------|-------------|
| `debug` | `build/debug` | Debug | Full debug symbols, no optimization |
| `release` | `build/release` | Release | Full optimization |
| `relwithdebinfo` | `build/relwithdebinfo` | RelWithDebInfo | Optimization + debug symbols |
| `s3-bench` | `build/s3-bench` | Release | Enables `CUCASCADE_BUILD_S3_BENCHMARK`, disables tests |

All presets use Ninja and export compile commands for IDE support.

The Pixi environments set `CUDAARCHS` per CUDA version (see `pixi.toml`). Outside Pixi, set `CUDAARCHS` or pass `-DCMAKE_CUDA_ARCHITECTURES=...`; otherwise CMake uses the CUDA compiler's default architecture, because it sets `CMAKE_CUDA_ARCHITECTURES` when enabling CUDA and the `75 80 86 90` fallback in `CMakeLists.txt` never applies.

### Pixi Environments

Defined in `pixi.toml`; select one with `pixi run -e <env> <task>`:

| Environment | CUDA | libcudf |
|-------------|------|---------|
| `default`, `cuda-13-nightly` | 13 | Nightly (`rapidsai-nightly` channel) |
| `cuda-12-nightly` | 12.9 | Nightly |
| `cuda-13-stable` | 13 | Stable (`rapidsai` channel) |
| `cuda-12-stable` | 12.9 | Stable |
| `s3-bench` | 13 | Nightly, plus `aws-sdk-cpp` for the S3 benchmarks |

The `build` and `test` tasks always use `build/release`. CMake caches the compilers and package locations of the environment that first configured a build directory, so run `pixi run clean` (or use a separate build directory) when switching environments.

### Compilation Caching

[sccache](https://github.com/mozilla/sccache) is configured for C++ and CUDA compilation caching. It's used both locally and in CI (with GitHub Actions cache backend).

### Adding New Source Files

1. Create the `.hpp` header in the module's directory under `include/cucascade/` (e.g. `memory/`, `data/`, `io/`)
2. Create the `.cpp` implementation in the matching `src/` directory
3. Add the `.cpp` file to that directory's `CMakeLists.txt` (sources are listed explicitly, no globs), on the object library it belongs to: `cucascade_objects` (`src/cuda`, `src/memory`, `src/data`), `cucascade_io_objects` (`src/io`, `src/exec`) or `cucascade_cudf_objects` (`src/cudf`):

```cmake
target_sources(
  cucascade_objects
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/your_new_file.cpp)
```

Test and benchmark sources are likewise listed explicitly in `test/CMakeLists.txt` and `benchmark/CMakeLists.txt`.

---

## Testing

### Test Framework

cuCascade uses [Catch2](https://github.com/catchorg/Catch2) v3 (fetched from GitHub at build time; the tag is pinned in `test/CMakeLists.txt`). Test files include `<catch2/catch_all.hpp>`.

Tests are tagged `TEST_CASE`s with `REQUIRE` assertions:
```cpp
TEST_CASE("data batch acquires read-only access", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  REQUIRE(batch->get_state() == batch_state::idle);

  auto read_only = batch->to_read_only();
  REQUIRE(batch->get_state() == batch_state::read_only);
}
```

### Test Organization

Tests are split into executables by the library they link, which decides where a new test goes (not its directory):

| Executable | Links | Built when | Covers |
|------------|-------|------------|--------|
| `cucascade_tests` | Core (`cucascade`) | Not `CUCASCADE_TOPOLOGY_ONLY` | Memory resources, reservations, peer access, data repositories, disk I/O backends, GPU kernels (`test/memory/`, `test/data/`) |
| `cucascade_io_tests` | `cucascade_io` | `CUCASCADE_BUILD_IO` | URI parsing, io_uring, REST/S3, kvikIO, cache (`test/io/`) |
| `cucascade_cudf_tests` | `cucascade_cudf` | `CUCASCADE_BUILD_CUDF` | Data batches, representations, converters, bandwidth profiler, datasource cache (`test/data/`, `test/cudf/`) |
| `cucascade_topology_discovery_tests` | `cucascade_topology_discovery` | Always | Hardware topology discovery |

CTest additionally runs `cucascade_cuda_driver_loading_test` in several modes and checks that the topology discovery libraries have no link-time dependency on the CUDA driver (`libcuda`).

### Test Utilities

**`test/utils/mock_test_utils.hpp`**:
- `make_mock_memory_space()` -- standalone `memory_space` for one tier, without a reservation manager (plain `cuda_memory_resource` on GPU, optional DISK I/O backend override)
- `mock_data_representation` -- implements `idata_representation` for testing
- `create_conversion_test_configs()` -- memory manager configs (1 GPU + 1 HOST)

**`test/utils/cudf_test_utils.hpp`** (cudf tests only):
- `create_simple_cudf_table()` -- factory for test cuDF tables (INT32/INT64 columns)
- `cudf_tables_have_equal_contents_on_stream()` -- stream-aware table comparison
- `expect_cudf_tables_equal_on_stream()` -- assertion wrapper with detailed error reporting

**`test/utils/test_memory_resources.hpp`**:
- `shared_device_resource` -- shared allocator wrapper
- `make_shared_current_device_resource()` -- wraps the current RMM device resource for sharing

**`test/utils/allocation_counter.hpp`**:
- `scoped_allocation_counter` -- counts (or fails) the calling thread's `operator new` calls, to test allocation-free paths

**`test/utils/child_test_runner.hpp`**:
- Runs a test in a child process, for tests that change environment variables or need fresh CUDA/NVML state

### GPU Pool Configuration

The test runner (`test/unittest.cpp`, used by `cucascade_tests`, `cucascade_io_tests` and `cucascade_cudf_tests`) installs an RMM `cuda_async_memory_resource` pool as the current device resource:
- Default: 2 GiB initial, 10 GiB max (capped at 90% of device memory)
- Lower the maximum with an environment variable, e.g. `CUCASCADE_TEST_GPU_POOL_BYTES=4294967296` (4 GiB); values above the default are capped
- Skips the pool if no GPU is detected

A `device_sync_listener` synchronizes the GPU after each test case to catch async errors.

### Running Specific Tests

CTest registers each test executable as a single test, so `ctest -R` selects whole executables. To run a subset, pass a Catch2 test spec (tags, test names, `*` wildcards) to the executable:

```bash
# Run all tests
pixi run test

# Run one executable through CTest
ctest --test-dir build/release -R cucascade_io_tests --output-on-failure

# Run with Catch2 filters
./build/release/test/cucascade_cudf_tests "[data_batch]"
./build/release/test/cucascade_tests "*reservation*"
./build/release/test/cucascade_cudf_tests "data_batch to_read_only acquires shared access"

# List tags and test names
./build/release/test/cucascade_tests --list-tags
./build/release/test/cucascade_tests --list-tests
```

Tags starting with `.` are hidden and only run when named explicitly. For example, most multi-GPU tests are tagged `[.multi-device]`, so neither `pixi run test` nor CI runs them (a few cudf tests use the non-hidden `[multi-device]` tag and skip themselves on a single GPU); on a machine with two or more GPUs, pass `"[multi-device]"` to both `cucascade_tests` and `cucascade_cudf_tests`.

---

## Benchmarking

### Available Benchmarks

`cucascade_benchmarks` is built with [Google Benchmark](https://github.com/google/benchmark) v1.8.3 and requires `CUCASCADE_BUILD_CUDF=ON`:

| Source | Benchmarks | Description |
|--------|------------|-------------|
| `benchmark_representation_converter.cpp` | `BM_ConvertGpuToHost`, `BM_ConvertHostToGpu` | GPU <-> HOST conversion via `host_data_packed_representation`; Size: 64KiB-512MiB, Cols: 2-8, Threads: 1-4 |
| | `BM_ConvertGpuToHostFast`, `BM_ConvertHostFastToGpu` | GPU <-> HOST conversion via `host_data_representation` (no `cudf::pack`); same parameters |
| | `BM_GpuToHostThroughput`, `BM_HostToGpuThroughput` | Raw copy bandwidth; Size: 64KiB-512MiB, Threads: 1-4 |
| `benchmark_disk_converter.cpp` | `BM_Convert{GpuToDisk,DiskToGpu,HostToDisk,DiskToHost}`, `BM_ConvertGpuToDisk{String,List,Struct}Columns`, `*Pipeline` | GPU/HOST <-> DISK conversion and the pipeline I/O backend; files are written under `/tmp` |
| `benchmark_gpu_to_gpu_converter.cpp` | `BM_ConvertGpuToGpu`, `BM_RawPeerCopy` | Cross-GPU conversion vs raw peer copy; skipped without two peer-capable GPUs |

With `CUCASCADE_BUILD_IO=ON`, the standalone (non-Google Benchmark) executables `cucascade_parquet_benchmark` and `cucascade_parquet_io_benchmark` compare parquet reads through cudf's default datasource and the `cucascade::io` datasources. The S3 variants (`cucascade_s3_parquet_benchmark`, `cucascade_s3_parquet_io_benchmark`) need `CUCASCADE_BUILD_S3_BENCHMARK=ON`; `pixi run -e s3-bench build-s3-bench` builds `cucascade_s3_parquet_benchmark`.

### Running Benchmarks

```bash
# All benchmarks
pixi run benchmarks

# Or directly:
cd build/release && ./benchmark/cucascade_benchmarks

# Filter to specific benchmarks
./benchmark/cucascade_benchmarks --benchmark_filter=Convert
./benchmark/cucascade_benchmarks --benchmark_filter=Throughput

# JSON output for comparison
./benchmark/cucascade_benchmarks \
    --benchmark_out_format=json \
    --benchmark_out=results.json \
    --benchmark_min_time=2.0s
```

### Regression Detection

The `scripts/compare_benchmarks.py` script compares two benchmark runs:

```bash
python scripts/compare_benchmarks.py --baseline baseline.json --current current.json --max-regression-pct 10
```

- Compares the per-benchmark `real_time` of two `--benchmark_out_format=json` files
- Reports regressions, new benchmarks, and benchmarks missing from the current run
- Default threshold: 10% (`--max-regression-pct`)
- Exit codes: 0 (pass), 1 (regression detected), 2 (baseline has no results)

---

## Code Quality

### Pre-commit Hooks

Install and run:

```bash
# Install hooks (runs automatically on every commit)
pixi run lint-install

# Run all checks manually
pixi run lint

# Update hook versions
pixi run pre-commit autoupdate
```

Configured hooks (`.pre-commit-config.yaml`):

| Hook | Purpose |
|------|---------|
| Large file detection | Prevents accidental large binary commits |
| Case conflict check | Catches filename case issues |
| JSON/TOML/YAML validation | Validates config file syntax (and pretty-formats JSON) |
| Merge conflict markers | Catches unresolved merge conflicts |
| Private key detection | Prevents accidental credential commits |
| End-of-file / trailing whitespace / mixed line endings | Consistent file endings |
| **clang-format** (v20.1.4) | C/C++/CUDA code formatting |
| **cmake-format** (v0.6.13) | CMake file formatting + linting |
| **Black** (25.1.0) | Python code formatting |
| **codespell** (v2.4.1) | Spell checking (with `.codespell_words` exceptions) |
| **texthooks** (0.7.1) | Smart quote fixing |

### Formatting

C/C++/CUDA formatting uses clang-format with a Google-based style (`.clang-format`):
- Column limit: 100
- Indent: 2 spaces (no tabs)
- Pointer alignment: left (`int* ptr`)
- Include ordering: priority groups (local -> benchmark/test -> cuDF test -> cuCascade -> cuDF -> other RAPIDS -> RMM -> CCCL/CUDA -> other system -> STL)

---

## Documentation

Generate API documentation from source code comments:

```bash
pixi run docs
```

This runs:
1. **Doxygen** -- parses all headers in `include/`, generates XML
2. **`scripts/generate_api_docs.py`** -- converts Doxygen XML to Markdown with Mermaid class hierarchy diagrams

Output:
- HTML: `build/docs/html/` (browsable API docs)
- XML: `build/docs/xml/` (intermediate)
- Markdown: `docs/API_REFERENCE.md` (generated)

---

## CI Pipeline

GitHub Actions (`.github/workflows/ci.yml`) runs on pushes to `main` and to the `pull-request/<N>` branches that copy-pr-bot creates for pull requests. Jobs run in order lint -> build -> test:

| Job | Runner | What It Does |
|-----|--------|-------------|
| **lint** | `linux-amd64-cpu4` (CPU only) | Runs the pre-commit hooks |
| **build** | `linux-amd64-cpu4` (CPU only) | For each `cuda-{12,13}-{nightly,stable}` Pixi environment: `pixi run -e <env> build` with sccache, uploads `build/release` |
| **test** | `linux-amd64-gpu-t4-latest-1` (T4 GPU) | For each environment: downloads its build, runs `pixi run -e <env> test` |

Benchmarks are not run in CI.

---

## Using cuCascade as a Dependency

cuCascade installs CMake config files for `find_package`:

```cmake
find_package(cuCascade CONFIG REQUIRED)

target_link_libraries(my_app PRIVATE cuCascade::cucascade)

# cudf-coupled layer and/or I/O layer
find_package(cuCascade CONFIG REQUIRED COMPONENTS cudf io)
find_package(kvikio CONFIG REQUIRED)  # if built with CUCASCADE_BUILD_CUDF=ON and not found via cudf

target_link_libraries(my_app PRIVATE cuCascade::cucascade_cudf cuCascade::cucascade_io)
```

Available targets (each name prefers the shared library if available; append `_shared` or `_static` to pick one explicitly):
- `cuCascade::cucascade` -- cudf-free core library
- `cuCascade::cucascade_cudf` -- cudf-coupled layer (`cudf` component; installed with `CUCASCADE_BUILD_CUDF`)
- `cuCascade::cucascade_io` -- I/O layer (`io` component; installed with `CUCASCADE_BUILD_IO`)
- `cuCascade::cucascade_topology_discovery` -- topology discovery only (also the only library in a `CUCASCADE_TOPOLOGY_ONLY` install)

Transitive dependencies of the core (CUDA Toolkit, RMM, Threads, numa) are found automatically. cuDF is only looked up when the `cudf` component is requested, and liburing, libcurl and OpenSSL when `io` or `cudf` is requested. In an install built with `CUCASCADE_BUILD_CUDF=ON`, `cucascade_io` (and through it `cucascade_cudf`) also links `kvikio::kvikio`, which the cuCascade package config does not look up; if `kvikio::kvikio` is not already defined (e.g. when requesting only `io`), call `find_package(kvikio CONFIG REQUIRED)` yourself. The moodycamel `concurrentqueue` headers used by the I/O headers are not installed; consumers of the I/O layer must provide them.

---

## Key Commands Reference

| Command | Description |
|---------|-------------|
| `pixi run build` | Build release mode |
| `pixi run build-debug` | Build debug mode |
| `pixi run test` | Run test suite (`ctest` in `build/release`) |
| `pixi run benchmarks` | Run benchmarks |
| `pixi run lint` | Run all code quality checks |
| `pixi run lint-install` | Install pre-commit hooks |
| `pixi run docs` | Generate API documentation |
| `pixi run clean` | Remove the `build` directory |
| `pixi run -e <env> <task>` | Run a task in another Pixi environment |
| `cmake --preset release` | Configure release build |
| `cmake --build build/release` | Build after configure |
| `ctest --output-on-failure` | Run tests, printing the output of failing ones |
