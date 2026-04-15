<!-- GSD:project-start source:PROJECT.md -->
## Project

**cuCascade Disk I/O Performance Optimization**

Performance optimization of cuCascade's disk I/O backends (GDS and kvikIO) to approach raw hardware throughput on NVMe storage. The disk tier enables persisting GPU data batches to disk and reading them back, but current throughput is 5-11x below what the hardware can deliver. This project closes that gap.

**Core Value:** Both GDS and kvikIO disk I/O backends achieve within 80% of raw hardware throughput (gdsio/dd baselines) for read and write paths.

### Constraints

- **Buffer registration**: Cannot assume RMM-allocated GPU memory is pre-registered with cuFile — but CAN register temporarily per-transfer for large I/O
- **Thread safety**: Disk I/O operations must remain safe for concurrent use
- **RAII**: All file handles and disk resources must follow existing RAII patterns
- **API compatibility**: idisk_io_backend interface must not change (backends are swappable)
- **Benchmark path**: Benchmarks should use /mnt/disk_2 for NVMe-fair comparison with baselines
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- C++20 - All library source code (`src/memory/`, `src/data/`) and public headers (`include/cucascade/`)
- CUDA C++20 - GPU kernel code (`test/memory/test_gpu_kernels.cu`) and direct CUDA runtime calls throughout `src/`
- CMake 3.26.4+ - Build system (`CMakeLists.txt`, `cmake/`, `CMakePresets.json`)
- Python - Utility scripts (`scripts/generate_api_docs.py`, `scripts/compare_benchmarks.py`)
## Runtime
- Linux only: `linux-64` and `linux-aarch64` (enforced via `pixi.toml` line 4)
- NVIDIA GPU required; compute capability >= 7.5 (Turing or newer)
- CUDA Toolkit 12.9+ (cuda-12 track) or 13+ (cuda-13 track)
- NUMA-aware host: requires `libnuma` (installed via `numactl` pixi dependency)
- Pixi >= 0.59
- Config: `pixi.toml`
- Lockfile: `pixi.lock` (committed)
- Channels: `rapidsai-nightly`, `conda-forge` (default); `rapidsai`, `conda-forge` (cudf-stable feature)
## Frameworks
- RMM (RAPIDS Memory Manager) - GPU/host memory resource abstraction; provides `rmm::mr::device_memory_resource`, `rmm::cuda_stream_view`, `rmm::out_of_memory`, `rmm::bad_alloc`; pulled in via `find_package(rmm REQUIRED CONFIG)` from libcudf installation
- libcudf 26.06 (nightly) / 26.02 (stable) - Columnar data representation; provides `cudf::table`, `cudf::column`, `cudf::type_id`, `cudf::pack`/`unpack`; pulled in via `find_package(cudf REQUIRED CONFIG)`
- Catch2 v2.13.10 - Unit test framework; fetched via `FetchContent` in `test/CMakeLists.txt`; test executable: `cucascade_tests`
- Google Benchmark v1.8.3 - Microbenchmark framework; fetched via `FetchContent` in `benchmark/CMakeLists.txt`; benchmark executable: `cucascade_benchmarks`
- Ninja - Build generator (configured in `CMakePresets.json`)
- sccache - Compiler cache for C, CXX, and CUDA compilers (`CMAKE_C_COMPILER_LAUNCHER`, `CMAKE_CXX_COMPILER_LAUNCHER`, `CMAKE_CUDA_COMPILER_LAUNCHER` in `CMakePresets.json`)
- clang-format v20.1.4 - Code formatting enforced via pre-commit (`.clang-format` at repo root)
- cmake-format / cmake-lint v0.6.13 - CMake file linting via pre-commit
- black v25.1.0 - Python formatting via pre-commit
- codespell v2.4.1 - Spell checking via pre-commit (ignore list: `.codespell_words`)
- Doxygen - API documentation generation; config: `Doxyfile`; output parsed by `scripts/generate_api_docs.py`
## Key Dependencies
- `libcudf` 26.06 / 26.02 - Core data representation; `cudf::table` is the GPU-tier data container; all column type handling (LIST, STRUCT, STRING, DICTIONARY32, etc.) delegates to cudf
- `RMM` (via cudf) - `rmm::mr::device_memory_resource` is the base class for all custom allocators; `rmm::cuda_stream_view` is used throughout for CUDA stream propagation
- `CUDA::cudart` - Direct CUDA runtime API calls (`cudaMalloc`, `cudaMemcpyAsync`, `cudaStreamSynchronize`, `cudaFree`, `cudaMallocHost`, `cudaFreeHost`)
- `CUDA::nvml` - GPU topology discovery via NVML in `src/memory/topology_discovery.cpp`
- `kvikio` 26.06 / 26.02 - Async disk I/O with automatic GDS/POSIX fallback; used in `src/data/kvikio_io_backend.cpp` via `kvikio::FileHandle`; linked PRIVATE via `kvikio::kvikio`
- `libcufile` (cuFile / GDS) - NVIDIA GPUDirect Storage for direct GPU↔NVMe transfers; `<cufile.h>` used in `src/data/gds_io_backend.cpp`; found via `find_library(CUFILE_LIB cufile ...)` — optional at configure time, required at runtime for GDS backend
- `libnuma` - NUMA-aware pinned host memory allocation in `src/memory/numa_region_pinned_host_allocator.cpp`; found via `find_library(NUMA_LIB numa REQUIRED)`
- `Threads::Threads` (pthreads) - Thread support; `std::mutex`, `std::condition_variable`, `std::async` throughout
- `fmt` - Format library (pixi dependency; available in environment)
- `nvtx3::nvtx3-cpp` - NVIDIA NVTX profiling annotations; only linked when `CUCASCADE_NVTX=ON`; used via `CUCASCADE_FUNC_RANGE()` macro in `include/cucascade/error.hpp`
## Configuration
- `CUDAARCHS` - Set by pixi environment activation to select CUDA architecture targets
- `CMAKE_PREFIX_PATH` - Passed through from pixi environment for dependency resolution
- `CUCASCADE_BUILD_TESTS` (default ON) - Adds `test/` subdirectory
- `CUCASCADE_BUILD_BENCHMARKS` (default ON) - Adds `benchmark/` subdirectory
- `CUCASCADE_BUILD_SHARED_LIBS` (default ON) - Builds `libcucascade.so`
- `CUCASCADE_BUILD_STATIC_LIBS` (default ON) - Builds `libcucascade.a`
- `CUCASCADE_NVTX` (default OFF) - Enables NVTX profiling ranges
- `CUCASCADE_WARNINGS_AS_ERRORS` (default ON) - Treats all compiler warnings as errors
- `debug` → `build/debug/`
- `release` → `build/release/`
- `relwithdebinfo` → `build/relwithdebinfo/`
## Library Outputs
- `cucascade_shared` (`libcucascade.so`, versioned) - alias `cuCascade::cucascade_shared`
- `cucascade_static` (`libcucascade.a`) - alias `cuCascade::cucascade_static`
- `cuCascade::cucascade` - Default alias pointing to shared if available, else static
- `cucascade_tests` - Test executable linked against Catch2
- `cucascade_benchmarks` - Benchmark executable linked against Google Benchmark
- Headers installed to `include/`; CMake package config at `cmake/cuCascadeConfig.cmake.in`
- Consumers: `find_package(cuCascade)` then `target_link_libraries(... cuCascade::cucascade)`
## Platform Requirements
- Linux x86_64 or aarch64
- NVIDIA GPU with compute capability >= 7.5
- Pixi >= 0.59
- CUDA Toolkit 12.9+ or 13+
- Build runner: `linux-amd64-cpu4`
- Test/Benchmark runner: `linux-amd64-gpu-t4-latest-1` (NVIDIA T4 GPU)
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Headers: `snake_case.hpp` — never `.h` for project headers
- CUDA headers: `snake_case.cuh` (e.g., `test/memory/test_gpu_kernels.cuh`)
- Source: `snake_case.cpp` for C++, `snake_case.cu` for CUDA kernels
- Test files: `test_<module_name>.cpp` (e.g., `test/data/test_disk_io_backend.cpp`)
- Benchmark files: `benchmark_<module_name>.cpp` (e.g., `benchmark/benchmark_disk_converter.cpp`)
- `snake_case` for all: `memory_space`, `data_batch`, `disk_data_representation`
- Interface classes prefixed with `i`: `idata_representation`, `idisk_io_backend`
- Config structs suffixed with `_config`: `gpu_memory_space_config`, `disk_memory_space_config`
- Hash structs suffixed with `_hash`: `converter_key_hash`
- RAII handles suffixed with `_handle`: `data_batch_processing_handle`
- Error category structs suffixed with `_category`: `memory_error_category`
- Exception: `Tier` enum uses PascalCase name, UPPER_CASE values: `Tier::GPU`, `Tier::HOST`, `Tier::DISK`
- Exception: `MemoryError` uses PascalCase name, UPPER_CASE values: `MemoryError::ALLOCATION_FAILED`
- `snake_case`: `get_available_memory()`, `make_reservation_or_null()`
- Getters prefixed with `get_`: `get_tier()`, `get_device_id()`, `get_batch_id()`
- Boolean queries prefixed with `should_`, `has_`, or `is_`: `should_downgrade_memory()`
- Factory functions prefixed with `make_` or `create_`: `make_mock_memory_space()`, `create_simple_cudf_table()`
- Try-pattern methods prefixed with `try_to_`: `try_to_create_task()`, `try_to_lock_for_processing()`
- Blocking wait methods prefixed with `wait_to_`: `wait_to_create_task()`
- Member variables prefixed with underscore: `_id`, `_capacity`, `_mutex`, `_disk_table`
- Local variables: `snake_case` — `gpu_device_0`, `reservation_size`
- Constants: `snake_case` — `expected_gpu_capacity`, `default_block_size`
- Compile-time constants: `constexpr` — `static constexpr std::size_t default_size{16};`
- Size literal constants use `ull` suffix and bit shifts: `2ull << 30` for 2 GB, `1UL << 20` for 1 MB
- Benchmark-local byte-unit constants: `constexpr uint64_t KiB = 1024ULL;`
- Function type aliases: PascalCase — `DeviceMemoryResourceFactoryFn`, `representation_converter_fn` (simpler aliases are `snake_case`)
- Enum classes: `snake_case` for names and values — `batch_state::idle`, `batch_state::in_transit`
- All caps with `CUCASCADE_` prefix: `CUCASCADE_CUDA_TRY`, `CUCASCADE_FAIL`, `CUCASCADE_FUNC_RANGE`
## Code Style
- Tool: clang-format v20.1.4 (enforced by pre-commit hook via `.clang-format`)
- Column limit: 100 characters
- Indent width: 2 spaces (tabs never used)
- Standard: `c++20`
- Brace style: WebKit (`BreakBeforeBraces: WebKit`) — opening braces on same line for most constructs
- Pointer alignment: left — `void* ptr`, not `void *ptr`
- `AlignConsecutiveAssignments: true` — align consecutive assignment operators
- `AlignConsecutiveMacros: true` — align consecutive macro definitions
- `BinPackArguments: false` — all arguments on one line or each on its own line
- No trailing whitespace (enforced by pre-commit `trailing-whitespace` hook)
- Files must end with a newline (enforced by `end-of-file-fixer`)
- cmake-format and cmake-lint for CMake files (line width 220, disabled code C0307)
- codespell for spell checking (ignore-words in `.codespell_words`)
- black for Python files
- clang-format for all C++/CUDA source
- `CUCASCADE_WARNINGS_AS_ERRORS=ON` by default — all warnings are errors
- `-Wall -Wextra -Wpedantic`
- `-Wcast-align -Wunused -Wconversion -Wsign-conversion`
- `-Wnull-dereference -Wdouble-promotion -Wformat=2 -Wimplicit-fallthrough`
## License Header
## Include Organization
#include "utils/cudf_test_utils.hpp"     // quoted local
#include "utils/mock_test_utils.hpp"
#include <cucascade/data/disk_data_representation.hpp>  // cucascade
#include <cucascade/data/representation_converter.hpp>
#include <cudf/column/column_factories.hpp>  // cuDF
#include <rmm/cuda_stream.hpp>  // RMM
#include <catch2/catch.hpp>  // system with dot
#include <memory>   // STL
#include <vector>
## Namespace Usage
- Top-level: `cucascade`
- Subnamespaces: `cucascade::memory`, `cucascade::utils`
- Test namespace: `cucascade::test`
- No namespace indentation
- Close with comment: `}  // namespace cucascade` or `}  // namespace`
- Nested namespaces use traditional form: `namespace cucascade { namespace test {` (not C++17 `::`)
- Use anonymous `namespace { }` for file-local helpers in `.cpp` and test files
- `using namespace cucascade;` at file scope in test files is acceptable
- Specific test utilities imported explicitly: `using cucascade::test::create_simple_cudf_table;`
- C++17 nested namespace shorthand used in `test_memory_resources.hpp`: `namespace cucascade::test {`
## Error Handling
- `CUCASCADE_CUDA_TRY(call)` — wraps CUDA runtime calls; throws `cucascade::cuda_error` on failure
- `CUCASCADE_CUDA_TRY(call, exception_type)` — two-arg form throws custom exception type
- `CUCASCADE_CUDA_TRY_ALLOC(call)` — throws `rmm::out_of_memory` for OOM, `rmm::bad_alloc` otherwise
- `CUCASCADE_CUDA_TRY_ALLOC(call, num_bytes)` — two-arg form includes requested bytes in message
- `CUCASCADE_ASSERT_CUDA_SUCCESS(call)` — assert-based check for noexcept/destructor contexts; in release builds the call executes but the error is discarded
- `CUCASCADE_FAIL(message)` — throws `cucascade::logic_error` with file/line context
- `CUCASCADE_FAIL(message, exception_type)` — throws custom exception type
- `cucascade::cuda_error` — inherits `std::runtime_error`; for CUDA runtime failures
- `cucascade::logic_error` — inherits `std::logic_error`; for programming errors
- `rmm::out_of_memory`, `rmm::bad_alloc` — for allocation failures
- `cucascade::memory::cucascade_out_of_memory` — extends `rmm::out_of_memory` with diagnostics
- Use exceptions for errors; not error codes (except `MemoryError` which bridges to `std::error_code`)
- Use `std::runtime_error` or `std::invalid_argument` for general errors in constructors
- Destructors must be exception-safe: use `CUCASCADE_ASSERT_CUDA_SUCCESS` (not `CUCASCADE_CUDA_TRY`) and wrap filesystem operations in try/catch with silent discard (see `disk_data_representation::~disk_data_representation()` in `src/data/disk_data_representation.cpp`)
- Use `[[nodiscard]]` on getters and methods returning important values
## Documentation Style
- `@brief` — one-line summary
- `@param` — parameters
- `@return` — return values
- `@throws` — exception specifications
- `@tparam` — template parameters
- `@note` — important caveats
- `@code` / `@endcode` — inline code examples (used in `include/cucascade/data/representation_converter.hpp`)
- `@example` — usage examples
- Multi-line descriptions separated by a blank `*` line after `@brief`
- Trailing member docs use `///< description`
- Section separators: `//===----------------------------------------------------------------------===//`
- Inline comments use `//`
- `// clang-format off` / `// clang-format on` around macro blocks needing custom formatting
## RAII and Ownership Patterns
- `std::unique_ptr` for exclusive ownership (allocators, data representations, reservations, tables)
- `std::shared_ptr` for shared ownership (data batches, memory spaces in tests)
- `std::weak_ptr` for non-owning references (processing handles to batches)
- Explicitly delete copy/move when objects must be pinned: `= delete` (e.g., `representation_converter_registry`)
- Explicitly default move when move-only semantics are desired: `= default`
- `mutable std::mutex _mutex` for internal locking in thread-safe classes
## C++20 Features in Use
- `std::derived_from` concept in `requires` clauses: `requires std::derived_from<TargetType, idata_representation>`
- `static_assert` with `std::is_base_of_v` at template registration sites
- `[[nodiscard]]` attribute on getters
- `[[maybe_unused]]` on interface default parameters (e.g., `clone([[maybe_unused]] rmm::cuda_stream_view stream)`)
- Structured bindings: `auto [free_bytes, total_bytes] = rmm::available_device_memory();`
- `std::span` (in memory layer)
- Three-way comparison `<=>` (in `include/cucascade/memory/common.hpp`)
## NVTX Profiling
- Enabled via `CUCASCADE_NVTX` CMake option (default OFF)
- `CUCASCADE_FUNC_RANGE()` macro at function entry points for profiling
- Custom domain: `cucascade::libcucascade_domain` (in `include/cucascade/error.hpp`)
- Links `nvtx3::nvtx3-cpp` when enabled
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Three memory tiers represented by `Tier::GPU`, `Tier::HOST`, `Tier::DISK` enum values (`include/cucascade/memory/common.hpp`)
- Strategy pattern for reservation requests; converter registry pattern for data tier transitions
- RAII-first ownership: reservations, processing handles, streams, and disk files all release automatically on destruction
- Thread-safe state machine on every `data_batch` using mutex + condition variable
- C++20 concepts constrain template parameters at compile time; `std::variant` dispatches tier-specific allocator types at runtime
## Two Major Subsystems
### 1. Memory Subsystem (`cucascade::memory`)
```
```
### 2. Data Subsystem (`cucascade`)
```
```
## Layers
- Purpose: Build memory space configs, optionally from NUMA/GPU topology
- Location: `include/cucascade/memory/reservation_manager_configurator.hpp`, `include/cucascade/memory/config.hpp`
- Contains: `reservation_manager_configurator` (fluent builder), `gpu_memory_space_config`, `host_memory_space_config`, `disk_memory_space_config`, `memory_space_config` (variant)
- Depends on: `topology_discovery`
- Used by: Application bootstrap code
- Purpose: Query NVML and Linux sysfs for GPU-NUMA-NIC-storage topology
- Location: `include/cucascade/memory/topology_discovery.hpp`, `src/memory/topology_discovery.cpp`
- Contains: `topology_discovery`, `system_topology_info`, `gpu_topology_info`, `network_device_info`, `storage_device_info`
- Depends on: NVML, Linux sysfs
- Used by: `reservation_manager_configurator`
- Purpose: Coordinate reservation requests across all memory spaces via strategy pattern
- Location: `include/cucascade/memory/memory_reservation_manager.hpp`, `src/memory/memory_reservation_manager.cpp`
- Contains: `memory_reservation_manager`, strategy structs (`any_memory_space_in_tier`, `specific_memory_space`, `any_memory_space_to_downgrade`, `any_memory_space_to_upgrade`, `any_memory_space_in_tier_with_preference`, `any_memory_space_in_tiers`)
- Depends on: Memory resource layer, `notification_channel`
- Used by: Data layer, application code
- Purpose: Tier-specific allocation and deallocation with reservation enforcement
- Location: `include/cucascade/memory/reservation_aware_resource_adaptor.hpp`, `include/cucascade/memory/fixed_size_host_memory_resource.hpp`, `include/cucascade/memory/disk_access_limiter.hpp`
- Contains: Per-tier allocators wrapping RMM upstream resources; per-stream/per-thread tracking via atomic counters
- Depends on: RMM, CUDA runtime, `notification_channel`, `atomics` utilities
- Used by: `memory_space`
- Purpose: Represent a single tier+device location; own its allocator; provide streams
- Location: `include/cucascade/memory/memory_space.hpp`, `src/memory/memory_space.cpp`
- Contains: `memory_space` (non-copyable, non-movable), `memory_space_id`, `memory_space_hash`
- `_reservation_allocator` is a `std::variant` selecting `reservation_aware_resource_adaptor` (GPU), `fixed_size_host_memory_resource` (HOST), or `disk_access_limiter` (DISK) at construction time
- Depends on: Config layer, memory resource layer
- Used by: `memory_reservation_manager`, `idata_representation`
- Purpose: Tier-specific data storage format; all derive from `idata_representation`
- Location: `include/cucascade/data/common.hpp`, `include/cucascade/data/gpu_data_representation.hpp`, `include/cucascade/data/cpu_data_representation.hpp`, `include/cucascade/data/disk_data_representation.hpp`
- Contains: `idata_representation` (abstract: `get_size_in_bytes()`, `get_uncompressed_data_size_in_bytes()`, `clone()`, templated `cast<T>()`), four concrete types
- `disk_data_representation` owns a `disk_table_allocation` (file path + `column_metadata` vector); destructor deletes the file (RAII)
- Depends on: `memory_space`, cuDF (`cudf::table`)
- Used by: `data_batch`, converter registry
- Purpose: Type-pair dispatch table for converting between representation types
- Location: `include/cucascade/data/representation_converter.hpp`, `src/data/representation_converter.cpp`
- Contains: `representation_converter_registry`, `converter_key` (`{source_type_index, target_type_index}`), `representation_converter_fn`
- Registration: `register_converter<SourceType, TargetType>(fn)` with static_assert constraints
- Lookup: `convert<TargetType>(source, memory_space, stream)` uses `typeid(source)` at runtime
- `register_builtin_converters()` registers GPU↔HOST and GPU↔DISK and HOST↔DISK converters; overload accepts `shared_ptr<idisk_io_backend>` to select I/O backend
- Depends on: `idata_representation`, `idisk_io_backend`
- Used by: `data_batch::convert_to()`, `data_batch::clone_to()`
- Purpose: Abstract disk I/O; concrete backends selectable at runtime
- Location: `include/cucascade/data/disk_io_backend.hpp`, `src/data/gds_io_backend.cpp`, `src/data/kvikio_io_backend.cpp`, `src/data/pipeline_io_backend.cpp`, `src/data/io_backend_internal.hpp`
- Contains: `idisk_io_backend` (abstract with `write_device`, `read_device`, `write_host`, `read_host`, `write_device_batch`, `read_device_batch`), `io_backend_type` enum (`KVIKIO`, `GDS`, `PIPELINE`), `make_io_backend(type)` factory
- `GDS` uses raw cuFile batch API; `KVIKIO` uses kvikIO with automatic GDS/POSIX fallback; `PIPELINE` uses double-buffered pinned host transfer for D2H overlap with disk writes
- Depends on: kvikIO, cuFile (GDS)
- Used by: built-in disk converters registered via `register_builtin_converters()`
- Purpose: Lifecycle management; state machine; processing-count reference counting
- Location: `include/cucascade/data/data_batch.hpp`, `src/data/data_batch.cpp`
- Contains: `data_batch` (owns `unique_ptr<idata_representation>`), `batch_state` enum, `data_batch_processing_handle` (RAII, holds `weak_ptr<data_batch>`), `idata_batch_probe` interface, `lock_for_processing_result`
- Allowed state transitions: `idle → in_transit | task_created`, `task_created → processing | idle`, `processing → idle`, `in_transit → idle`
- Depends on: `idata_representation`, `representation_converter_registry`
- Used by: `idata_repository`, application code
- Purpose: Partitioned, thread-safe collections of batches; blocking pop with state transition
- Location: `include/cucascade/data/data_repository.hpp`, `src/data/data_repository.cpp`
- Contains: `idata_repository<PtrType>` (template: `shared_ptr<data_batch>` or `unique_ptr<data_batch>`), `shared_data_repository`, `unique_data_repository`
- `pop_data_batch(target_state)` blocks on condition variable until a batch can transition
- `pop_data_batch_by_id()` / `get_data_batch_by_id()` for directed retrieval
- Depends on: `data_batch`
- Used by: `data_repository_manager`
- Purpose: Top-level coordinator; operator-port keyed repository map; unique batch ID generation
- Location: `include/cucascade/data/data_repository_manager.hpp`, `src/data/data_repository_manager.cpp`
- Contains: `data_repository_manager<PtrType>`, `operator_port_key`, `shared_data_repository_manager`, `unique_data_repository_manager`
- Batch IDs generated atomically via `_next_data_batch_id` (`std::atomic<uint64_t>`)
- SFINAE selects copy vs. move in `add_data_batch_impl` based on `PtrType`
- Depends on: `idata_repository`
- Used by: Application pipeline code
## Data Flow
- `data_batch` mutex protects all state transitions and `_processing_count`
- Blocking `wait_to_*` methods use `_internal_cv` on the batch mutex
- `idata_repository` propagates state change notifications via `_state_change_cv` pointer set on each batch
## Key Abstractions
- Purpose: Uniform interface for tier-specific storage formats
- Location: `include/cucascade/data/common.hpp`
- Pattern: Abstract base with `get_size_in_bytes()`, `get_uncompressed_data_size_in_bytes()`, `clone()`, and templated `cast<T>()` (requires `std::derived_from<T, idata_representation>`)
- Concrete types: `gpu_table_representation` (`include/cucascade/data/gpu_data_representation.hpp`), `host_data_representation`, `host_data_packed_representation` (`include/cucascade/data/cpu_data_representation.hpp`), `disk_data_representation` (`include/cucascade/data/disk_data_representation.hpp`)
- Purpose: Owns a single tier+device memory budget and its allocator
- Location: `include/cucascade/memory/memory_space.hpp`
- Pattern: Non-copyable/non-movable; variant-based allocator dispatch; exposes `make_reservation_or_null()`, `should_downgrade_memory()`, `get_disk_mount_path()`
- Template helpers: `get_memory_resource_of<Tier::GPU>()` returns correctly typed allocator pointer
- Purpose: Type-pair dispatch for tier conversions
- Location: `include/cucascade/data/representation_converter.hpp`
- Pattern: `unordered_map<converter_key, representation_converter_fn>` keyed by `{typeid(Source), typeid(Target)}`; thread-safe with internal mutex
- Purpose: Unit of data movement; state machine + reference counting
- Location: `include/cucascade/data/data_batch.hpp`
- Pattern: Owns `unique_ptr<idata_representation>`; `data_batch_processing_handle` holds `weak_ptr` so handle doesn't keep batch alive; `idata_batch_probe` for external state observation callbacks
- Purpose: On-disk file descriptor and binary format for a serialized cuDF table
- Location: `include/cucascade/memory/disk_table.hpp`, `include/cucascade/data/disk_file_format.hpp`
- Pattern: File starts with 32-byte `disk_file_header` (magic `0x43554353`, version, num_columns, metadata_size, data_offset); column metadata serialized depth-first; column data aligned to 4096-byte boundaries for GDS DMA
- Purpose: Abstraction over GDS, kvikIO, and pipeline I/O strategies
- Location: `include/cucascade/data/disk_io_backend.hpp`
- Pattern: Interface with `write_device`/`read_device`/`write_host`/`read_host` and batch variants; `make_io_backend(io_backend_type)` factory
- Purpose: Cross-component signaling when reservations are released
- Location: `include/cucascade/memory/notification_channel.hpp`
- Pattern: Shared ownership via `shared_ptr`; `event_notifier` instances post notifications; `wait()` blocks until notified or shutdown
- Purpose: RAII borrow of a CUDA stream from `exclusive_stream_pool`
- Location: `include/cucascade/memory/stream_pool.hpp`
- Pattern: Move-only; destructor calls `release_fn` returning stream to pool; acquire policies: `GROW` (create new) or `BLOCK` (wait)
- Purpose: Strategy pattern determining candidate `memory_space` objects for a reservation
- Location: `include/cucascade/memory/memory_reservation_manager.hpp`
- Pattern: Abstract base `get_candidates(manager)`, concrete strategies: `any_memory_space_in_tier`, `specific_memory_space`, `any_memory_space_to_downgrade`, `any_memory_space_to_upgrade`, `any_memory_space_in_tier_with_preference`, `any_memory_space_in_tiers`
## Entry Points
- Location: `include/cucascade/memory/reservation_manager_configurator.hpp`
- Triggers: Application initialization
- Responsibilities: Fluent builder collects GPU/HOST/DISK settings → `build()` emits `vector<memory_space_config>` → caller passes to `memory_reservation_manager` constructor
- Location: `include/cucascade/memory/memory_reservation_manager.hpp`
- Triggers: `request_reservation(strategy, size)`
- Responsibilities: Evaluate strategy → iterate candidate `memory_space` objects → call `make_reservation_or_null()` → block and wait on `_wait_cv` if none available
- Location: `include/cucascade/data/data_repository_manager.hpp`
- Triggers: Pipeline operator producing a batch
- Responsibilities: Assign unique batch ID via `get_next_data_batch_id()` → `add_data_batch()` routes to the operator-port's repository → `add_data_batch` notifies blocked consumers via `_cv`
- Location: `include/cucascade/data/data_batch.hpp` (`convert_to<T>()`)
- Triggers: Memory pressure downgrade or upgrade request
- Responsibilities: Lock batch mutex → check `_processing_count == 0` → call `registry.convert<T>()` → replace `_data` with new representation
## Error Handling
- `CUCASCADE_CUDA_TRY(call)` — throws `cucascade::cuda_error` on CUDA runtime failure (defined in `include/cucascade/error.hpp`)
- `CUCASCADE_CUDA_TRY_ALLOC(call, bytes)` — throws `rmm::out_of_memory` for `cudaErrorMemoryAllocation`, `rmm::bad_alloc` otherwise
- `CUCASCADE_ASSERT_CUDA_SUCCESS(call)` — assert in debug builds; no-op in release; used in destructors and `noexcept` paths
- `CUCASCADE_FAIL(msg)` / `CUCASCADE_FAIL(msg, exception_type)` — throws `cucascade::logic_error` or custom type with file/line context
- `cucascade::memory::cucascade_out_of_memory` extends `rmm::out_of_memory` with `error_kind`, `requested_bytes`, `global_usage`, `pool_handle`
- `oom_handling_policy` interface (`include/cucascade/memory/oom_handling_policy.hpp`) allows pluggable OOM recovery; default `throw_on_oom_policy`
- `reservation_limit_policy` handles over-reservation: `ignore`, `fail`, or `increase` strategies
## Cross-Cutting Concerns
- All public methods on `memory_space`, `data_batch`, `idata_repository`, `data_repository_manager` are mutex-protected
- `representation_converter_registry` uses internal mutex for concurrent register/lookup
- Atomic counters (`atomic_bounded_counter`, `atomic_peak_tracker` in `include/cucascade/utils/atomics.hpp`) for lock-free allocation tracking in hot paths
- `notification_channel` provides cross-component async signaling with `shutdown()` support
- `memory_reservation_manager` owns all `memory_space` instances via `vector<unique_ptr<memory_space>>`
- `memory_space` owns its allocator and reservation adaptor
- `reservation` releases bytes back to its `memory_space` on destruction
- `data_batch` owns its `idata_representation` via `unique_ptr`
- `disk_data_representation` owns and deletes its backing file on destruction
- `data_batch_processing_handle` holds `weak_ptr<data_batch>` to avoid preventing batch destruction
- `data_batch::convert_to()` and `clone_to()` assert `_processing_count == 0` before allowing representation swap
- `pop_data_batch(batch_state::processing)` throws immediately — callers must use `task_created` + `try_to_lock_for_processing()`
- `disk_data_representation::clone()` always throws `cucascade::logic_error` — disk representations must be materialized to another tier via converter
- `CUCASCADE_FUNC_RANGE()` macro emits NVTX range when `CUCASCADE_NVTX` compile definition is present
- Custom domain: `cucascade::libcucascade_domain` (defined in `include/cucascade/error.hpp`)
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
