<!-- GSD:project-start source:PROJECT.md -->
## Project

**cuCascade Disk Data Representation**

A disk-tier data representation for cuCascade that enables persisting data batches to disk and reading them back. This adds `disk_data_representation` alongside the existing GPU and CPU representations, with converters for GPU<->disk and CPU(host_data)<->disk transitions. Two I/O backends are supported at runtime: NVIDIA GDS (GPUDirect Storage) for direct GPU-disk transfers, and libkvikIO for portable async I/O.

**Core Value:** Reliable round-trip persistence of data batches to disk with correct handling of all libcudf column types, selectable between GDS and kvikIO at runtime.

### Constraints

- **Tech stack**: Must use C++20, follow existing cucascade coding conventions (namespaces, RAII, error macros)
- **Dependencies**: GDS and kvikIO must be added as dependencies (via pixi/CMake)
- **Compatibility**: Must work with all cudf column types including nested types (list, struct) and string columns
- **Thread safety**: Disk I/O operations must be safe for concurrent use (multiple batches writing/reading simultaneously)
- **RAII**: File handles and disk resources must follow existing RAII ownership patterns
- **Buffer registration**: Cannot assume RMM-allocated GPU memory is registered with cuFile -- use preallocated staging buffers instead of registering per-transfer
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- C++20 - All library source code (`src/memory/`, `src/data/`) and public headers (`include/cucascade/`)
- CUDA C++20 - GPU kernel code (`test/memory/test_gpu_kernels.cu`) and CUDA runtime integration
- Python - Build scripts (`scripts/generate_api_docs.py`, `scripts/compare_benchmarks.py`)
- CMake 3.26.4+ - Build system (`CMakeLists.txt`, `cmake/`, `CMakePresets.json`)
## Runtime
- NVIDIA GPU required (CUDA architectures: 75, 80, 86, 90 default; extended via pixi to 100f, 120a, 120)
- CUDA Toolkit 12.9+ or 13+ (two supported tracks)
- Linux only: `linux-64` and `linux-aarch64` platforms (`pixi.toml` line 4)
- NUMA-aware: requires `libnuma` (`numactl` package)
- CUDA 12 environments: `75-real;80-real;86-real;90a-real;100f-real`
- CUDA 13 environments: `75-real;80-real;86-real;90a-real;100f-real;120a-real;120`
- CMake default fallback: `75 80 86 90`
## Build System
- Minimum version: 3.26.4
- Generator: Ninja (configured via `CMakePresets.json`)
- Compiler cache: sccache (configured for C, CXX, and CUDA compilers)
- `debug` - Debug build, output to `build/debug/`
- `release` - Release build, output to `build/release/`
- `relwithdebinfo` - RelWithDebInfo build, output to `build/relwithdebinfo/`
| Option | Default | Purpose |
|--------|---------|---------|
| `CUCASCADE_BUILD_TESTS` | ON | Build the test suite |
| `CUCASCADE_BUILD_BENCHMARKS` | ON | Build the benchmark suite |
| `CUCASCADE_BUILD_SHARED_LIBS` | ON | Build shared library |
| `CUCASCADE_BUILD_STATIC_LIBS` | ON | Build static library |
| `CUCASCADE_NVTX` | OFF | Enable NVTX range annotations |
| `CUCASCADE_WARNINGS_AS_ERRORS` | ON | Treat compiler warnings as errors |
## Package Manager
- Config: `pixi.toml`
- Lockfile: `pixi.lock` (committed)
- Channels: `rapidsai-nightly`, `conda-forge` (default); `rapidsai`, `conda-forge` (stable)
| Environment | CUDA | cuDF |
|-------------|------|------|
| `default` | 13 | nightly (26.06) |
| `cuda-13-nightly` | 13 | nightly (26.06) |
| `cuda-12-nightly` | 12.9 | nightly (26.06) |
| `cuda-13-stable` | 13 | stable (26.02) |
| `cuda-12-stable` | 12.9 | stable (26.02) |
## Key Dependencies
- **libcudf** 26.06 (nightly) / 26.02 (stable) - RAPIDS cuDF for columnar data representation
- **RMM** (via cudf) - RAPIDS Memory Manager for GPU memory allocation
- **CUDA Runtime** (`CUDA::cudart`) - GPU kernel launching and memory operations
- **NVIDIA NVML** (`CUDA::nvml`) - GPU management and topology discovery
- **libnuma** - NUMA-aware host memory allocation
- **pthreads** (`Threads::Threads`) - Thread support
- **NVTX3** - NVIDIA Tools Extension for profiling annotations (when `CUCASCADE_NVTX=ON`)
- **Catch2** v2.13.10 - Unit test framework
- **Google Benchmark** v1.8.3 - Microbenchmark framework
- `cmake` 4.x
- `cuda-nvcc` - NVIDIA CUDA compiler
- `cxx-compiler` - Host C++ compiler
- `ninja` - Build system generator
- `sccache` - Compiler cache
- `fmt` - Format library
- `doxygen` - API documentation generation
- `pre-commit` - Git hooks for linting
## Compiler Requirements
## Library Outputs
- `cucascade_shared` - Shared library (`libcucascade.so`, versioned)
- `cucascade_static` - Static library (`libcucascade.a`)
- `cuCascade::cucascade` - Default alias (prefers shared)
- `cucascade_tests` - Test executable
- `cucascade_benchmarks` - Benchmark executable
- Headers installed to `include/`
- CMake package config generated from `cmake/cuCascadeConfig.cmake.in`
- Consumers link via `find_package(cuCascade)` then `cuCascade::cucascade`
## Configuration
- `CUDAARCHS` - Set by pixi environment activation to control CUDA architecture targets
- `CMAKE_PREFIX_PATH` - Passed through from environment for dependency resolution
- `CMakePresets.json` - Primary build configuration
- `CMakeLists.txt` - Root build definition
- `cmake/cuCascadeConfig.cmake.in` - Package config template for consumers
## Platform Requirements
- Linux (x86_64 or aarch64)
- NVIDIA GPU with compute capability >= 7.5 (Turing or newer)
- Pixi >= 0.59
- CUDA Toolkit 12.9+ or 13+
- Build: `linux-amd64-cpu4` runner
- Test/Benchmark: `linux-amd64-gpu-t4-latest-1` runner (NVIDIA T4 GPU)
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Headers: `snake_case.hpp` (always `.hpp`, never `.h` for project headers)
- Source: `snake_case.cpp` for C++, `snake_case.cu` for CUDA kernels
- CUDA headers: `snake_case.cuh` for CUDA-specific headers (e.g., `test/memory/test_gpu_kernels.cuh`)
- Test files: `test_<module_name>.cpp` (prefixed with `test_`)
- Test CUDA files: `test_<module_name>.cu` / `test_<module_name>.cuh`
- Benchmark files: `benchmark_<module_name>.cpp`
- Use `snake_case` for all class and struct names: `memory_space`, `data_batch`, `reservation_aware_resource_adaptor`, `exclusive_stream_pool`, `borrowed_stream`
- Interface classes prefixed with `i`: `idata_representation` (in `include/cucascade/data/common.hpp`)
- Config structs suffixed with `_config`: `gpu_memory_space_config`, `host_memory_space_config` (in `include/cucascade/memory/config.hpp`)
- Hash structs suffixed with `_hash`: `memory_space_hash`, `converter_key_hash`
- RAII handles include `_handle` suffix: `data_batch_processing_handle`
- Error category structs suffixed with `_category`: `memory_error_category` (in `include/cucascade/memory/error.hpp`)
- Enum classes use `snake_case` for names and values: `batch_state::idle`, `batch_state::in_transit`
- Exception: `Tier` enum uses PascalCase for the enum name and UPPER_CASE for values: `Tier::GPU`, `Tier::HOST`, `Tier::DISK` (in `include/cucascade/memory/common.hpp`)
- Exception: `MemoryError` uses PascalCase name and UPPER_CASE values: `MemoryError::ALLOCATION_FAILED` (in `include/cucascade/memory/error.hpp`)
- Nested enum classes in classes follow same patterns: `exclusive_stream_pool::stream_acquire_policy::GROW` (in `include/cucascade/memory/stream_pool.hpp`)
- Use `snake_case`: `get_available_memory()`, `make_reservation_or_null()`, `acquire_stream()`
- Getters prefixed with `get_`: `get_tier()`, `get_device_id()`, `get_batch_id()`
- Boolean queries prefixed with `should_` or `has_` or `is_`: `should_downgrade_memory()`
- Factory functions prefixed with `make_` or `create_`: `make_reservation()`, `make_mock_memory_space()`
- Try-pattern methods prefixed with `try_to_`: `try_to_create_task()`, `try_to_lock_for_processing()`
- Blocking wait methods prefixed with `wait_to_`: `wait_to_create_task()`, `wait_to_lock_for_in_transit()`
- Member variables prefixed with underscore: `_id`, `_capacity`, `_mutex`, `_data`, `_streams`
- Local variables use `snake_case`: `gpu_device_0`, `reservation_size`
- Constants use `snake_case`: `expected_gpu_capacity`, `default_block_size`
- Compile-time constants use `constexpr`: `static constexpr std::size_t default_size{16};`
- Size literal constants use `ull` suffix and bit shifts: `2ull << 30` for 2GB, `1UL << 20` for 1MB
- Use PascalCase for function type aliases: `DeviceMemoryResourceFactoryFn` (in `include/cucascade/memory/common.hpp`)
- Use `snake_case` for simpler aliases: `reserving_adaptor_type`
- All caps with `CUCASCADE_` prefix: `CUCASCADE_CUDA_TRY`, `CUCASCADE_FAIL`, `CUCASCADE_FUNC_RANGE`
- Use `snake_case`: `cucascade`, `cucascade::memory`, `cucascade::test`, `cucascade::utils`
## Code Style
- Tool: clang-format v20.1.4 (enforced via pre-commit)
- Config: `.clang-format` at project root
- Key settings:
- cmake-format and cmake-lint for CMake files (line width 220, disabled code C0307)
- codespell for spell checking (ignore words in `.codespell_words`)
- black for any Python files
## Include Organization
- Use `#pragma once` (not traditional `#ifndef` guards)
- Every header file starts with the SPDX license block followed by `#pragma once`
- `IncludeBlocks: Regroup` means includes are automatically sorted into groups separated by blank lines
- `SortIncludes: true` ensures alphabetical ordering within groups
#include "utils/cudf_test_utils.hpp"    // local quoted
#include "utils/mock_test_utils.hpp"
#include <cucascade/data/data_batch.hpp> // cucascade
#include <rmm/cuda_stream.hpp>           // RMM
#include <catch2/catch.hpp>              // system with dot
#include <atomic>                        // STL
#include <memory>
## Namespace Usage
- Top-level namespace: `cucascade`
- Subnamespaces: `cucascade::memory`, `cucascade::utils`
- Test namespace: `cucascade::test`
- No namespace indentation (configured in `.clang-format`)
- Close with comment: `}  // namespace cucascade`
- `using namespace cucascade;` is acceptable at file scope in test files
- `using cucascade::test::create_simple_cudf_table;` for specific test utilities
- Nested namespaces use the traditional (non-C++17) form: `namespace cucascade { namespace test {` rather than `namespace cucascade::test {`
- Use anonymous `namespace { }` for file-local helpers in `.cpp` and test files (e.g., test fixtures, benchmark configs)
- Close with `}  // namespace`
## Error Handling
- `CUCASCADE_CUDA_TRY(call)` - wraps CUDA runtime calls; throws `cucascade::cuda_error` on failure (defined in `include/cucascade/error.hpp`)
- `CUCASCADE_CUDA_TRY(call, exception_type)` - two-arg form throws custom exception type
- `CUCASCADE_CUDA_TRY_ALLOC(call)` - throws `rmm::out_of_memory` for OOM, `rmm::bad_alloc` otherwise
- `CUCASCADE_CUDA_TRY_ALLOC(call, num_bytes)` - two-arg form includes requested bytes in message
- `CUCASCADE_ASSERT_CUDA_SUCCESS(call)` - assert-based check for noexcept/destructor contexts; in release builds the call executes but error is discarded
- `CUCASCADE_FAIL(message)` - throws `cucascade::logic_error` with file/line context
- `CUCASCADE_FAIL(message, exception_type)` - throws custom exception type
- `cucascade::cuda_error` - inherits `std::runtime_error`
- `cucascade::logic_error` - inherits `std::logic_error`
- Also uses `rmm::out_of_memory`, `rmm::bad_alloc` for allocation failures
- `cucascade::memory::MemoryError` enum: `SUCCESS`, `ALLOCATION_FAILED`, `LIMIT_EXCEEDED`, `POOL_EXHAUSTED`
- `cucascade::memory::cucascade_out_of_memory` - inherits `rmm::out_of_memory`, carries `error_kind`, `requested_bytes`, `global_usage`, `pool_handle`
- Uses `std::error_category` pattern for `MemoryError`
- Use exceptions for error handling, not error codes (except for the `MemoryError` category which bridges to `std::error_code`)
- Use `std::runtime_error` or `std::invalid_argument` for general errors in constructors/methods
- Use `std::logic_error` for programming errors (e.g., zero pool size in `exclusive_stream_pool`)
- Use `[[nodiscard]]` on getters and methods returning important values
- Disable copy/move explicitly with `= delete` when objects should not be copied
## Documentation Style
- Use `/** ... */` style (Javadoc-style) for all public API documentation
- `@brief` for one-line summaries
- `@param` for parameters
- `@return` for return values
- `@throws` for exception specifications
- `@tparam` for template parameters
- `@note` for important caveats
- `@copydoc` to reference other documentation
- Multi-line descriptions after `@brief` are separated by a blank comment line
- Doxygen config in `Doxyfile` - generates HTML and XML from `include/` headers only
- Public API is extracted; private members are excluded
- Use `//` for inline comments
- Use `///` with `<` for trailing member documentation: `///< description`
- Section separators use: `//===----------------------------------------------------------------------===//`
- `// clang-format off` / `// clang-format on` around macro blocks that need custom formatting
## Pre-commit Hooks and Linting
## Compiler Warnings
- `-Wall -Wextra -Wpedantic`
- `-Wcast-align -Wunused -Wconversion -Wsign-conversion`
- `-Wnull-dereference -Wdouble-promotion -Wformat=2 -Wimplicit-fallthrough`
- Optional `-Werror` via `CUCASCADE_WARNINGS_AS_ERRORS` option (default ON)
## C++ Standard and Features
- `std::derived_from` concept (in `include/cucascade/data/common.hpp`, `include/cucascade/memory/memory_space.hpp`)
- `requires` clauses on templates
- Spaceship operator `<=>` (in `include/cucascade/memory/common.hpp`)
- Structured bindings (e.g., `auto [free_bytes, total_bytes] = rmm::available_device_memory()`)
- `[[nodiscard]]` attribute (extensively used on getters)
- `[[maybe_unused]]` attribute (used on interface default parameters)
- `std::bind_front` (in `src/memory/stream_pool.cpp`)
## RAII and Ownership Patterns
- `std::unique_ptr` for exclusive ownership (allocators, data representations, reservations)
- `std::shared_ptr` for shared ownership (data batches, memory spaces in tests)
- `std::weak_ptr` for non-owning references (processing handles to batches)
- `borrowed_stream` (in `include/cucascade/memory/stream_pool.hpp`) - RAII handle that returns a CUDA stream to the pool on destruction
- Uses a `std::function<void(rmm::cuda_stream&&)>` release callback
- Move-only; copy disabled
- Explicitly delete copy/move when objects should be pinned: `= delete` (e.g., `memory_space`, `exclusive_stream_pool`)
- Explicitly default move when move-only semantics are desired: `= default`
- Document move semantics in Doxygen comments
- `mutable std::mutex _mutex` for internal locking
- `std::lock_guard<std::mutex>` for scoped locking (simple cases)
- `std::unique_lock<std::mutex>` for condition variable waits
- `std::condition_variable` for blocking waits (data_batch state transitions, stream pool acquire)
- CUDA device guard: `rmm::cuda_set_device_raii` for multi-GPU safety
## NVTX Profiling
- Enabled via `CUCASCADE_NVTX` CMake option (default OFF)
- Adds `CUCASCADE_NVTX` compile definition when enabled
- Links `nvtx3::nvtx3-cpp`
- `CUCASCADE_FUNC_RANGE()` macro at function entry points for profiling
- Custom domain: `cucascade::libcucascade_domain` (in `include/cucascade/error.hpp`)
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Three-tier memory hierarchy: GPU (device) -> HOST (pinned) -> DISK (filesystem)
- Reservation-based memory management with per-stream/per-thread allocation tracking
- Abstract data representation layer with type-erased converter registry
- Thread-safe state machine for data batch lifecycle (idle -> task_created -> processing -> in_transit)
- RAII ownership throughout (reservations, allocations, processing handles, streams)
- C++20 with concepts, `std::variant`, `std::span`, three-way comparison
## Two Major Subsystems
### 1. Memory Subsystem (`cucascade::memory`)
```
```
### 2. Data Subsystem (`cucascade`)
```
```
## Layers
- Purpose: Build memory space configs from system topology or explicit parameters
- Location: `include/cucascade/memory/reservation_manager_configurator.hpp`, `include/cucascade/memory/config.hpp`
- Contains: `reservation_manager_configurator` (fluent builder), `gpu_memory_space_config`, `host_memory_space_config`, `disk_memory_space_config`
- Depends on: `topology_discovery`
- Used by: Application bootstrap code
- Purpose: Discover GPU-NUMA-NIC mappings via NVML and /sys filesystem
- Location: `include/cucascade/memory/topology_discovery.hpp`, `src/memory/topology_discovery.cpp`
- Contains: `topology_discovery`, `system_topology_info`, `gpu_topology_info`, `network_device_info`, `storage_device_info`
- Depends on: NVML, Linux sysfs
- Used by: `reservation_manager_configurator`
- Purpose: Own and coordinate memory spaces, process reservation requests via strategy pattern
- Location: `include/cucascade/memory/memory_reservation_manager.hpp`, `include/cucascade/memory/memory_space.hpp`
- Contains: `memory_reservation_manager`, `memory_space`, reservation request strategies
- Depends on: Memory resource layer, notification_channel
- Used by: Data layer, application code
- Purpose: Actual allocation and deallocation with per-stream tracking and reservation enforcement
- Location: `include/cucascade/memory/reservation_aware_resource_adaptor.hpp`, `include/cucascade/memory/fixed_size_host_memory_resource.hpp`, `include/cucascade/memory/disk_access_limiter.hpp`
- Contains: Tier-specific allocators wrapping RMM upstream resources
- Depends on: RMM, CUDA runtime, `notification_channel`, `atomics` utilities
- Used by: `memory_space`
- Purpose: Abstract data storage with tier-specific implementations
- Location: `include/cucascade/data/common.hpp`, `include/cucascade/data/gpu_data_representation.hpp`, `include/cucascade/data/cpu_data_representation.hpp`
- Contains: `idata_representation` (abstract), `gpu_table_representation`, `host_data_representation`, `host_data_packed_representation`
- Depends on: `memory_space`, cuDF (`cudf::table`)
- Used by: `data_batch`, converter registry
- Purpose: Lifecycle management of data units with state machine and processing counting
- Location: `include/cucascade/data/data_batch.hpp`
- Contains: `data_batch`, `batch_state` enum, `data_batch_processing_handle` (RAII), `idata_batch_probe`
- Depends on: `idata_representation`, `representation_converter_registry`
- Used by: `idata_repository`, application code
- Purpose: Partitioned collections of data batches with blocking pop and state-aware retrieval
- Location: `include/cucascade/data/data_repository.hpp`, `include/cucascade/data/data_repository_manager.hpp`
- Contains: `idata_repository<PtrType>`, `data_repository_manager<PtrType>`
- Depends on: `data_batch`
- Used by: Application pipeline code
## Data Flow
- `data_batch` uses mutex-protected state machine with 4 states: `idle`, `task_created`, `processing`, `in_transit`
- Processing uses reference counting (`_processing_count`) to allow concurrent readers
- `task_created` count tracks pending tasks before processing begins
- `data_batch_processing_handle` uses `weak_ptr` to avoid preventing batch destruction
- `idata_batch_probe` interface enables external observation of state transitions
## Key Abstractions
- Purpose: Represents a single memory location (tier + device_id) with capacity limits
- Examples: `include/cucascade/memory/memory_space.hpp`
- Pattern: Variant-based dispatch for tier-specific allocator (`_reservation_allocator` is `std::variant`)
- Owns: upstream allocator, reservation-aware adaptor, CUDA stream pool, notification channel
- Purpose: RAII handle for reserved memory bytes in a specific memory_space
- Examples: `include/cucascade/memory/memory_reservation.hpp`
- Pattern: Factory method (`reservation::create()`), non-copyable/non-movable, destructor releases bytes
- Contains: `reserved_arena` (tier-specific subclass with grow/shrink capabilities)
- Purpose: Strategy pattern for selecting candidate memory spaces for a reservation
- Examples: `any_memory_space_in_tier`, `specific_memory_space`, `any_memory_space_to_downgrade`, `any_memory_space_to_upgrade`, `any_memory_space_in_tier_with_preference`
- Pattern: Polymorphic strategy with `get_candidates()` virtual method
- Distinguishes strong ordering (deterministic selection) vs weak ordering (any available)
- Purpose: Abstract interface for tier-specific data storage formats
- Examples: `include/cucascade/data/common.hpp`
- Pattern: Polymorphic with `get_size_in_bytes()` and `clone()` pure virtuals
- Concrete types: `gpu_table_representation` (wraps `cudf::table`), `host_data_representation` (direct buffer copies), `host_data_packed_representation` (cudf::pack format)
- Template `cast<T>()` method with C++20 `requires` clause for safe downcasting
- Purpose: Type-pair dispatch table for converting between representation types
- Examples: `include/cucascade/data/representation_converter.hpp`
- Pattern: `std::unordered_map<converter_key, converter_fn>` keyed by `{source_type_index, target_type_index}`
- Registration: `register_converter<SourceType, TargetType>(fn)` with compile-time type constraints
- Lookup: `convert<TargetType>(source, memory_space, stream)` uses runtime `typeid(source)`
- Purpose: Waitable notification mechanism for cross-component signaling (e.g., reservation release)
- Examples: `include/cucascade/memory/notification_channel.hpp`
- Pattern: Shared ownership via `shared_ptr`, `event_notifier` instances post notifications, `wait()` blocks until notified or shutdown
- Purpose: Thread-safe pool of CUDA streams with exclusive borrow semantics
- Examples: `include/cucascade/memory/stream_pool.hpp`
- Pattern: Borrow/return with `borrowed_stream` RAII handle; supports BLOCK (wait) or GROW (create new) acquire policies
## Entry Points
- Location: `include/cucascade/memory/reservation_manager_configurator.hpp`
- Triggers: Application initialization
- Responsibilities: Build config -> create `memory_reservation_manager` -> use reservations
- Location: `include/cucascade/memory/memory_reservation_manager.hpp`
- Triggers: `request_reservation(strategy, size)`
- Responsibilities: Select memory space via strategy, attempt reservation, block if no space available
- Location: `include/cucascade/data/data_repository_manager.hpp`
- Triggers: Pipeline operator registration, batch insertion/retrieval
- Responsibilities: Route batches to operator-port repositories, generate unique batch IDs
## Error Handling
- `CUCASCADE_CUDA_TRY(call)` wraps CUDA runtime calls, throws `cucascade::cuda_error` on failure
- `CUCASCADE_CUDA_TRY_ALLOC(call, bytes)` throws `rmm::out_of_memory` for allocation failures
- `CUCASCADE_ASSERT_CUDA_SUCCESS(call)` for noexcept/destructor contexts (assert in debug, no-op in release)
- `CUCASCADE_FAIL(msg)` throws `cucascade::logic_error` with file/line context
- `cucascade::memory::cucascade_out_of_memory` extends `rmm::out_of_memory` with detailed diagnostics (error kind, requested bytes, global usage, pool handle)
- `MemoryError` enum with `std::error_code` integration for structured memory errors
- `oom_handling_policy` interface allows pluggable OOM recovery (default: `throw_on_oom_policy`)
- `reservation_limit_policy` handles over-reservation: `ignore`, `fail`, or `increase` strategies
## Cross-Cutting Concerns
- All public APIs on `memory_space`, `data_batch`, `idata_repository`, `data_repository_manager` are mutex-protected
- Atomic counters (`atomic_bounded_counter`, `atomic_peak_tracker`) for lock-free allocation tracking in hot paths
- Condition variables for blocking waits in reservation manager, repository pop, and batch state transitions
- `notification_channel` provides cross-component async signaling with shutdown support
- `memory_reservation_manager` owns all `memory_space` instances via `unique_ptr`
- `memory_space` owns its allocator and reservation adaptor
- `reservation` releases bytes back to its `memory_space` on destruction
- `data_batch` owns its `idata_representation` via `unique_ptr`
- `data_batch_processing_handle` holds `weak_ptr` to avoid preventing batch destruction
- `fixed_multiple_blocks_allocation` returns blocks to pool on destruction
- Optional NVTX integration via `CUCASCADE_NVTX` compile-time flag
- `CUCASCADE_FUNC_RANGE()` macro for function-level profiling ranges in `libcucascade` domain
- `numa_region_pinned_host_memory_resource` allocates NUMA-local pinned memory
- `topology_discovery` maps GPUs to NUMA nodes for optimal memory placement
- `reservation_manager_configurator` supports per-NUMA or per-GPU host tier binding
## Template Patterns
- `data_repository_manager<PtrType>` and `idata_repository<PtrType>` are templated on `shared_ptr<data_batch>` or `unique_ptr<data_batch>`
- SFINAE (`std::enable_if`) selects copy vs move semantics in `add_data_batch_impl`
- Type aliases: `shared_data_repository_manager`, `unique_data_repository_manager`
- `tier_memory_resource_trait<Tier>` maps `Tier` enum to concrete allocator types at compile time
- `memory_space::get_memory_resource_of<Tier::GPU>()` returns correctly-typed allocator pointer
- `std::derived_from<T, rmm::mr::device_memory_resource>` constrains `get_memory_resource_as<T>()`
- `std::derived_from<TargetType, idata_representation>` constrains `cast<TargetType>()`
- `std::integral<T>` constrains `atomic_peak_tracker` and `atomic_bounded_counter`
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
