# Development Guide

Building, testing, benchmarking, and contributing to cuCascade.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Build System](#build-system)
  - [CMake Structure](#cmake-structure)
  - [Build Presets](#build-presets)
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
- [Logging](#logging)
  - [Installing a Sink](#installing-a-sink)
  - [Standalone Use](#standalone-use)
  - [Catching Exceptions](#catching-exceptions)
  - [Compile-Time Floor](#compile-time-floor)
- [Key Commands Reference](#key-commands-reference)

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| OS | Linux (x86_64 or aarch64) |
| C++ Compiler | C++20 compatible |
| CMake | 4.1+ |
| Ninja | Any |
| CUDA Toolkit | 13+ |
| NVIDIA Driver | Compatible with CUDA 13 |
| libcudf | 25.10+ |
| RMM | Via cuDF dependency |

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

Alternatively, build directly with CMake:

```bash
cmake --preset release
cmake --build build/release

# Run tests
cd build/release && ctest --output-on-failure
```

## Project Structure

```
cuCascade/
├── include/cucascade/
│   ├── data/                    # Data module headers
│   │   ├── common.hpp           # idata_representation interface
│   │   ├── data_batch.hpp       # Batch lifecycle + state machine
│   │   ├── data_repository.hpp  # Partitioned batch storage
│   │   ├── data_repository_manager.hpp
│   │   ├── representation_converter.hpp
│   │   ├── cpu_data_representation.hpp
│   │   └── gpu_data_representation.hpp
│   ├── memory/                  # Memory module headers (17 files)
│   │   ├── common.hpp           # Tier enum, memory_space_id
│   │   ├── config.hpp           # Tier-specific config structs
│   │   ├── memory_reservation_manager.hpp
│   │   ├── memory_space.hpp
│   │   ├── memory_reservation.hpp
│   │   ├── reservation_aware_resource_adaptor.hpp
│   │   ├── fixed_size_host_memory_resource.hpp
│   │   ├── disk_access_limiter.hpp
│   │   ├── topology_discovery.hpp
│   │   ├── reservation_manager_configurator.hpp
│   │   ├── notification_channel.hpp
│   │   ├── stream_pool.hpp
│   │   ├── oom_handling_policy.hpp
│   │   ├── error.hpp
│   │   ├── numa_region_pinned_host_allocator.hpp
│   │   ├── host_table.hpp
│   │   ├── host_table_packed.hpp
│   │   └── null_device_memory_resource.hpp
│   └── utils/                   # Utility headers
│       ├── atomics.hpp          # atomic_peak_tracker, atomic_bounded_counter
│       └── overloaded.hpp       # Variant visitor helper
├── src/
│   ├── data/                    # Data module implementation (6 .cpp files)
│   └── memory/                  # Memory module implementation (15 .cpp files)
├── test/
│   ├── unittest.cpp             # Test runner with GPU pool setup
│   ├── data/                    # Data module tests (5 files)
│   ├── memory/                  # Memory module tests (3 files + GPU kernels)
│   └── utils/                   # Test utilities (mocks, cuDF helpers)
├── benchmark/                   # Google Benchmark suite
├── docs/                        # Documentation (you are here)
├── cmake/                       # CMake package config template
├── scripts/                     # Tooling scripts
│   ├── generate_api_docs.py     # Doxygen XML -> Markdown converter
│   └── compare_benchmarks.py    # Benchmark regression detector
├── CMakeLists.txt               # Root build configuration
├── CMakePresets.json            # Build presets (debug, release, relwithdebinfo)
├── pixi.toml                    # Pixi environment + task definitions
├── Doxyfile                     # Doxygen configuration
└── .pre-commit-config.yaml      # Code quality hooks
```

## Build System

### CMake Structure

cuCascade uses an **object library pattern** for efficient compilation:

```
cucascade_objects (OBJECT library)
    ├── src/data/*.cpp
    └── src/memory/*.cpp
         │
         ├── cucascade_static (STATIC library)
         │     Links: RMM, cuDF, CUDA runtime, CUDA NVML, pthreads
         │
         └── cucascade_shared (SHARED library)
               Same links, SOVERSION 0
```

Source files are compiled once into `cucascade_objects`, then linked into both static and shared libraries. This halves build time compared to compiling sources twice.

Both library types are built by default. At least one must be enabled via `BUILD_STATIC_LIBS` and/or `BUILD_SHARED_LIBS`.

**Compiler warnings** are strict by default:
```
-Wall -Wextra -Wpedantic -Wcast-align -Wunused -Wconversion
-Wsign-conversion -Wnull-dereference -Wdouble-promotion
-Wformat=2 -Wimplicit-fallthrough -Werror
```

Warnings-as-errors can be disabled with `-DWARNINGS_AS_ERRORS=OFF`.

### Build Presets

Defined in `CMakePresets.json`:

| Preset | Build Dir | Mode | Description |
|--------|-----------|------|-------------|
| `debug` | `build/debug` | Debug | Full debug symbols, no optimization |
| `release` | `build/release` | Release | Full optimization |
| `relwithdebinfo` | `build/relwithdebinfo` | RelWithDebInfo | Optimization + debug symbols |

All presets use Ninja and export compile commands for IDE support.

Default CUDA architectures: 75, 80, 86, 90 (override with `CMAKE_CUDA_ARCHITECTURES`).

### Compilation Caching

[sccache](https://github.com/mozilla/sccache) is configured for C, C++, and CUDA compilation caching. It's used both locally and in CI (with GitHub Actions cache backend).

### Adding New Source Files

1. Create the `.hpp` header in `include/cucascade/data/` or `include/cucascade/memory/`
2. Create the `.cpp` implementation in `src/data/` or `src/memory/`
3. Add the `.cpp` file to the corresponding `src/data/CMakeLists.txt` or `src/memory/CMakeLists.txt`:

```cmake
target_sources(
  cucascade_objects
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/your_new_file.cpp)
```

---

## Testing

### Test Framework

cuCascade uses [Catch2](https://github.com/catchorg/Catch2) v2.13.10 (fetched from GitHub at build time).

Tests use BDD-style assertions:
```cpp
TEST_CASE("data batch acquires read-only access", "[data_batch]") {
    auto batch = data_batch::make(1, make_gpu_representation());

    REQUIRE(batch->get_state() == batch_state::idle);

    auto read_only = batch->to_read_only();
    REQUIRE(batch->get_state() == batch_state::read_only);
}
```

### Test Organization

All tests compile into a single executable `cucascade_tests`:

| Directory | Tests | Coverage |
|-----------|-------|----------|
| `test/data/test_data_batch.cpp` | 47+ cases | State transitions, processing handles, cloning |
| `test/data/test_data_repository.cpp` | 140+ cases | Add/pop, partitioning, blocking, threading |
| `test/data/test_data_repository_manager.cpp` | 100+ cases | Multi-operator, batch IDs, concurrent access |
| `test/data/test_data_representation.cpp` | Representation interface | Size, tier, clone operations |
| `test/data/test_representation_converter.cpp` | Converter registry | Registration, lookup, conversion |
| `test/memory/test_memory_reservation_manager.cpp` | Reservation system | Strategies, limits, multi-space |
| `test/memory/test_topology_discovery.cpp` | Hardware detection | NVML integration |
| `test/memory/test_gpu_kernels.cu` | GPU kernel tests | Device-side operations |

### Test Utilities

**`test/utils/mock_test_utils.hpp`**:
- `make_mock_memory_space()` -- lightweight memory spaces without real allocators
- `mock_data_representation` -- implements `idata_representation` for testing
- `create_conversion_test_configs()` -- memory manager configs (1 GPU + 1 HOST)
- `create_simple_cudf_table()` -- factory for test cuDF tables (INT32/INT64 columns)

**`test/utils/cudf_test_utils.hpp`**:
- `cudf_tables_have_equal_contents_on_stream()` -- stream-aware table comparison
- `expect_cudf_tables_equal_on_stream()` -- assertion wrapper with detailed error reporting
- `logging_device_resource` -- RMM allocation tracing
- `shared_device_resource` -- shared allocator wrapper

**`test/utils/test_memory_resources.hpp`**:
- `make_shared_current_device_resource()` -- wraps the current RMM device resource for sharing

### GPU Pool Configuration

The test runner (`test/unittest.cpp`) creates a global GPU memory pool:
- Default: 2 GB initial, 10 GB max (capped at 90% of device memory)
- Override with environment variable: `CUCASCADE_TEST_GPU_POOL_BYTES=4294967296` (4 GB)
- Gracefully degrades if no GPU is detected

A `device_sync_listener` synchronizes the GPU after each test case to catch async errors.

### Running Specific Tests

```bash
# Run all tests
pixi run test

# Run with Catch2 filters
cd build/release && ./test/cucascade_tests "[data_batch]"
cd build/release && ./test/cucascade_tests "reservation"
cd build/release && ./test/cucascade_tests -c "specific test name"
```

---

## Benchmarking

### Available Benchmarks

Built with [Google Benchmark](https://github.com/google/benchmark) v1.8.3:

| Benchmark | Description | Parameters |
|-----------|-------------|------------|
| `BM_ConvertGpuToHost` | GPU -> HOST data conversion | Size: 64KiB-512MiB, Cols: 2-8, Threads: 1-4 |
| `BM_ConvertHostToGpu` | HOST -> GPU data conversion | Same as above |
| `BM_GpuToHostThroughput` | Raw GPU -> HOST bandwidth | Size: 64KiB-512MiB, Threads: 1-4 |
| `BM_HostToGpuThroughput` | Raw HOST -> GPU bandwidth | Same as above |

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
python scripts/compare_benchmarks.py baseline.json current.json --threshold 10
```

- Reports regressions, improvements, new tests, and missing tests
- Default threshold: 10% (configurable)
- Exit codes: 0 (pass), 1 (regression detected), 2 (no baseline)

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
pre-commit autoupdate
```

Configured hooks (`.pre-commit-config.yaml`):

| Hook | Purpose |
|------|---------|
| Large file detection | Prevents accidental large binary commits |
| Case conflict check | Catches filename case issues |
| JSON/TOML/YAML validation | Validates config file syntax |
| Merge conflict markers | Catches unresolved merge conflicts |
| Private key detection | Prevents accidental credential commits |
| End-of-file / trailing whitespace | Consistent file endings |
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
- Include ordering: 10 priority groups (local -> benchmark -> cuCascade -> RAPIDS -> RMM -> CCCL -> CUDA -> system -> STL)

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

GitHub Actions (`.github/workflows/ci.yml`) runs on pushes to `main` and pull requests:

| Stage | Runner | What It Does |
|-------|--------|-------------|
| **Build** | `linux-amd64-cpu4` (CPU only) | `pixi run build` with sccache, uploads artifacts |
| **Test** | `linux-amd64-gpu-t4-latest-1` (T4 GPU) | Downloads build, runs `pixi run test` |

---

## Using cuCascade as a Dependency

cuCascade installs CMake config files for `find_package`:

```cmake
find_package(cuCascade CONFIG REQUIRED)

target_link_libraries(my_app PRIVATE cuCascade::cucascade)
```

Available targets:
- `cuCascade::cucascade` -- prefers shared library if available
- `cuCascade::cucascade_shared` -- shared library explicitly
- `cuCascade::cucascade_static` -- static library explicitly

Transitive dependencies (cuDF, RMM, CUDA Toolkit, Threads) are automatically found.

---

## Logging

cuCascade ships no logging backend. The `CUCASCADE_LOG_TRACE` through `CUCASCADE_LOG_FATAL` macros
(from `<cucascade/log/logging.hpp>`) format their arguments with `std::format` and deliver each
record to a sink that the embedding host installs.

```cpp
CUCASCADE_LOG_WARN("pinned pool at {}% of capacity", percent);
```

Until a host installs a sink, warnings and errors go to stderr (the built-in sink from
`make_stderr_sink()`, at `level::warn`) and everything below is dropped. A call site below the
threshold costs one relaxed atomic load and its arguments are never evaluated; `clear_sink()`
silences everything.

### Installing a Sink

Subclass `cucascade::log::sink`. Its `write` is called concurrently from any logging thread, so it
must be thread-safe. It may also run while cuCascade holds an internal lock (a failed CUDA call in
a deallocation or destructor is reported from there), so it must not call back into cuCascade, and
any lock it takes must not be held by other threads while they call into cuCascade. Forwarding to a
logger that locks only for the duration of the write, as spdlog does, is fine:

```cpp
class host_sink final : public cucascade::log::sink {
 public:
  void write(const cucascade::log::record& rec) override
  {
    my_logger.log(rec.lvl, rec.location.file_name(), rec.location.line(), rec.message);
  }
};
cucascade::log::set_sink(std::make_shared<host_sink>(), cucascade::log::level::info);
```

Or wrap a lambda with `make_sink`. Captures must own what they use:

```cpp
auto logger = std::make_shared<my_logger>();
cucascade::log::set_sink(cucascade::log::make_sink([logger](const cucascade::log::record& rec) {
  logger->log(rec.lvl, rec.message);
}));
```

The second argument to `set_sink` sets the threshold. Omit it to keep the current one, which starts
at `level::warn`. Mirror the host logger's own level there, or call `set_min_level()` later.
Filtering happens in `is_enabled()` before formatting, so records the host would drop are never built.

`set_sink` and `clear_sink` return the replaced sink. Every in-flight `write` shares ownership of
it, so it is destroyed, on whichever thread held the last reference, only after the last delivered
record. Give the sink shared ownership of everything it touches (capture a `shared_ptr`, as above);
that release orders the teardown. Do not poll `use_count()`, which is a relaxed load and
synchronizes nothing.

### Standalone Use

`cucascade::log::use_stderr_sink(level)` installs a built-in sink (`info` by default) that writes
`<UTC timestamp> [LEVEL] <tid> file:line: message` lines to stderr.

### Catching Exceptions

`CUCASCADE_TRY_AND_LOG_EXCEPTION` (`<cucascade/utils/error_utils.hpp>`) runs an expression,
swallows any exception, and logs it at error level with `what()` appended. Optional format
arguments prefix the message: `CUCASCADE_TRY_AND_LOG_EXCEPTION(close(), "fd {}", fd)`.

### Compile-Time Floor

`CUCASCADE_MIN_LOG_LEVEL` (0 = trace through 6 = off) removes call sites below it during
preprocessing; their format strings and arguments are still checked, but never evaluated. Set it
through the CMake cache variable of the same name. Every cuCascade target propagates it as a
PUBLIC compile definition, so cuCascade's own sources and CMake consumers compile the macros with
the same floor. It must be one value for the whole program: the macros expand inside inline and
template functions of installed headers, so two translation units with different floors would
violate the one-definition rule. Including the headers without the CMake targets yields 0.

---

## Key Commands Reference

| Command | Description |
|---------|-------------|
| `pixi run build` | Build release mode |
| `pixi run build-debug` | Build debug mode |
| `pixi run test` | Run test suite |
| `pixi run benchmarks` | Run benchmarks |
| `pixi run lint` | Run all code quality checks |
| `pixi run lint-install` | Install pre-commit hooks |
| `pixi run docs` | Generate API documentation |
| `cmake --preset release` | Configure release build |
| `cmake --build build/release` | Build after configure |
| `ctest --output-on-failure` | Run tests with verbose errors |
