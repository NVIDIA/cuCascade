# Codebase Structure

**Analysis Date:** 2026-04-13

## Directory Layout

```
cuCascade/
├── include/
│   └── cucascade/          # All public headers — installed alongside the library
│       ├── cuda_utils.hpp  # CUDA error-checking macros, NVTX helpers
│       ├── data/           # Data management public API
│       │   ├── common.hpp                  # idata_representation interface
│       │   ├── data_batch.hpp              # synchronized_data_batch + accessor types
│       │   ├── data_repository.hpp         # idata_repository<PtrType> template
│       │   ├── data_repository_manager.hpp # data_repository_manager<PtrType> template
│       │   ├── representation_converter.hpp# representation_converter_registry
│       │   ├── gpu_data_representation.hpp # gpu_table_representation
│       │   └── cpu_data_representation.hpp # host_data_representation, host_data_packed_representation
│       ├── memory/         # Memory management public API
│       │   ├── common.hpp                         # Tier enum, memory_space_id, factory fns
│       │   ├── config.hpp                         # gpu/host/disk_memory_space_config structs
│       │   ├── memory_space.hpp                   # memory_space class
│       │   ├── memory_reservation.hpp             # reservation, reserved_arena, limit policies
│       │   ├── memory_reservation_manager.hpp     # memory_reservation_manager + strategies
│       │   ├── reservation_manager_configurator.hpp # fluent builder for manager config
│       │   ├── reservation_aware_resource_adaptor.hpp # RMM adaptor for GPU per-stream tracking
│       │   ├── fixed_size_host_memory_resource.hpp   # block-pool host allocator
│       │   ├── disk_access_limiter.hpp            # disk reservation tracker
│       │   ├── host_table.hpp                     # host_table_allocation + column_metadata
│       │   ├── host_table_packed.hpp              # host_table_packed_allocation (cudf::pack)
│       │   ├── notification_channel.hpp           # condition-variable notification primitive
│       │   ├── oom_handling_policy.hpp            # OOM policy interface
│       │   ├── stream_pool.hpp                    # CUDA stream pool wrapper
│       │   ├── topology_discovery.hpp             # NVML-based GPU/NUMA/NIC discovery
│       │   ├── numa_region_pinned_host_allocator.hpp
│       │   ├── null_device_memory_resource.hpp
│       │   ├── small_pinned_host_memory_resource.hpp
│       │   ├── memory_space.hpp
│       │   └── error.hpp
│       └── utils/
│           ├── atomics.hpp    # atomic_bounded_counter, atomic_peak_tracker
│           └── overloaded.hpp # std::visit helper
├── src/
│   ├── data/               # Data layer implementations + CMakeLists.txt
│   │   ├── CMakeLists.txt
│   │   ├── cpu_data_representation.cpp
│   │   ├── data_batch.cpp
│   │   ├── data_repository.cpp         # Template explicit instantiations + shared_ptr specialization
│   │   ├── data_repository_manager.cpp
│   │   ├── gpu_data_representation.cpp
│   │   └── representation_converter.cpp # Converter registry impl + register_builtin_converters
│   └── memory/             # Memory layer implementations + CMakeLists.txt
│       ├── CMakeLists.txt
│       ├── common.cpp
│       ├── disk_access_limiter.cpp
│       ├── error.cpp
│       ├── fixed_size_host_memory_resource.cpp
│       ├── memory_reservation.cpp
│       ├── memory_reservation_manager.cpp
│       ├── memory_space.cpp
│       ├── notification_channel.cpp
│       ├── null_device_memory_resource.cpp
│       ├── numa_region_pinned_host_allocator.cpp
│       ├── oom_handling_policy.cpp
│       ├── reservation_aware_resource_adaptor.cpp
│       ├── reservation_manager_configurator.cpp
│       ├── small_pinned_host_memory_resource.cpp
│       ├── stream_pool.cpp
│       └── topology_discovery.cpp
├── test/
│   ├── unittest.cpp            # Catch2 main() + GPU pool fixture (sets RMM per-device resource)
│   ├── CMakeLists.txt
│   ├── data/                   # Data layer tests
│   │   ├── CMakeLists.txt
│   │   ├── test_data_batch.cpp
│   │   ├── test_data_repository.cpp
│   │   ├── test_data_repository_manager.cpp
│   │   ├── test_data_representation.cpp
│   │   └── test_representation_converter.cpp
│   ├── memory/                 # Memory layer tests
│   │   ├── CMakeLists.txt
│   │   ├── test_memory_reservation_manager.cpp
│   │   ├── test_small_pinned_host_memory_resource.cpp
│   │   └── test_topology_discovery.cpp
│   └── utils/
│       ├── CMakeLists.txt
│       └── cudf_test_utils.cpp
├── benchmark/
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── benchmark_representation_converter.cpp
│   └── visualize_results.ipynb
├── cmake/
│   └── cuCascadeConfig.cmake.in  # CMake package config template for consumers
├── docs/                         # Project documentation
├── scripts/                      # Helper scripts
├── CMakeLists.txt                # Root build — defines cucascade_objects, static/shared libs
└── pixi.toml                     # Pixi environment + task definitions (build system)
```

## Directory Purposes

**`include/cucascade/`:**
- Purpose: Entire public API. All headers installed by `cmake --install`. No private headers here.
- Contains: Abstract interfaces, concrete types, template definitions, macros
- Key files: `data/data_batch.hpp`, `memory/memory_reservation_manager.hpp`, `memory/config.hpp`

**`include/cucascade/data/`:**
- Purpose: Pipeline data management — batch containers, repositories, representation converters
- Key types defined here: `idata_representation`, `synchronized_data_batch`, `idata_repository`, `data_repository_manager`, `representation_converter_registry`

**`include/cucascade/memory/`:**
- Purpose: Multi-tier memory management — GPU, host, disk spaces with reservation tracking
- Key types defined here: `Tier`, `memory_space_id`, `memory_space`, `memory_reservation_manager`, `reservation`, `reservation_aware_resource_adaptor`, `fixed_size_host_memory_resource`, `disk_access_limiter`

**`src/data/` and `src/memory/`:**
- Purpose: Implementation files corresponding 1:1 to headers; each subdirectory has its own `CMakeLists.txt` that appends sources to the `cucascade_objects` OBJECT library target
- Key note: Template class bodies live in `.hpp` files; `.cpp` files contain only explicit instantiations and non-template implementations

**`test/`:**
- Purpose: Catch2 unit tests. `unittest.cpp` is the single test main; all test `.cpp` files are compiled into one executable
- Key files: `test/unittest.cpp` sets up a global `rmm::mr::cuda_async_memory_resource` pool before tests run, and registers a `device_sync_listener` that calls `cudaDeviceSynchronize()` after each test case

**`benchmark/`:**
- Purpose: Standalone Google Benchmark (or custom) benchmarks for the representation converter; optional build via `CUCASCADE_BUILD_BENCHMARKS=ON`

**`cmake/`:**
- Purpose: CMake package config template for downstream consumers linking via `find_package(cuCascade)`

## Key File Locations

**Entry Points (library initialization):**
- `include/cucascade/memory/reservation_manager_configurator.hpp`: fluent builder — start here to configure memory spaces
- `include/cucascade/data/representation_converter.hpp`: `register_builtin_converters()` — call once at startup

**Configuration:**
- `CMakeLists.txt`: top-level build; options `CUCASCADE_BUILD_TESTS`, `CUCASCADE_BUILD_BENCHMARKS`, `CUCASCADE_NVTX`
- `pixi.toml`: environment and task definitions
- `include/cucascade/memory/config.hpp`: `gpu_memory_space_config`, `host_memory_space_config`, `disk_memory_space_config` structs

**Core Data Logic:**
- `include/cucascade/data/data_batch.hpp`: `synchronized_data_batch` — central concurrency-safe data container
- `include/cucascade/data/data_repository.hpp`: `idata_repository<PtrType>` — batch storage per pipeline stage
- `include/cucascade/data/data_repository_manager.hpp`: `data_repository_manager<PtrType>` — cross-pipeline coordinator
- `src/data/representation_converter.cpp`: built-in GPU↔host converters, including cudf::pack path and direct buffer-copy path

**Core Memory Logic:**
- `include/cucascade/memory/memory_space.hpp`: `memory_space` — owns allocator + reservation adaptor + stream pool
- `include/cucascade/memory/memory_reservation_manager.hpp`: `memory_reservation_manager` — central coordinator + reservation strategies
- `include/cucascade/memory/reservation_aware_resource_adaptor.hpp`: RMM adaptor tracking per-stream allocations
- `include/cucascade/memory/fixed_size_host_memory_resource.hpp`: block-pool pinned host allocator

**Testing:**
- `test/unittest.cpp`: test runner main, GPU pool setup, device sync listener
- `test/data/test_data_batch.cpp`: tests for `synchronized_data_batch` accessor semantics
- `test/memory/test_memory_reservation_manager.cpp`: tests for reservation strategies and tier management

## Naming Conventions

**Files:**
- Headers: `snake_case.hpp` (e.g., `data_batch.hpp`, `memory_reservation_manager.hpp`)
- Sources: `snake_case.cpp` matching the header they implement (e.g., `data_batch.cpp`)
- Tests: `test_<module>.cpp` (e.g., `test_data_batch.cpp`, `test_memory_reservation_manager.cpp`)

**Directories:**
- `snake_case` everywhere (e.g., `include/cucascade/data/`, `src/memory/`)

**Types:**
- Classes/structs: `snake_case` (e.g., `synchronized_data_batch`, `memory_space`, `reservation_aware_resource_adaptor`)
- Enums: `PascalCase` for the enum type (`Tier`, `StorageDriveType`), `UPPER_CASE` for enumerators (`GPU`, `HOST`, `DISK`)
- Type aliases: `snake_case` with trailing `_type` or descriptive suffix (e.g., `shared_data_repository`, `unique_data_repository_manager`)
- Template parameters: `PascalCase` (e.g., `PtrType`, `TargetRepresentation`, `TargetType`)

**Members:**
- Private data members: `_snake_case` with leading underscore (e.g., `_batch_id`, `_rw_mutex`, `_data_batches`)
- Public members in structs: `snake_case` without prefix (e.g., `operator_id`, `port_id`, `tier`, `device_id`)

**Functions:**
- `snake_case` for all free functions and methods (e.g., `get_batch_id()`, `pop_data_batch()`, `register_builtin_converters()`)
- Getter pattern: `get_<noun>()` (e.g., `get_available_memory()`, `get_tier()`)
- Factory pattern: `make_<noun>()` free functions (e.g., `make_default_gpu_memory_resource()`, `make_default_oom_policy()`)
- Boolean queries: `is_<state>()`, `has_<noun>()`, `should_<action>()` (e.g., `is_stream_tracked()`, `has_converter()`, `should_downgrade_memory()`)

**Macros:**
- `CUCASCADE_<UPPER_SNAKE_CASE>` (e.g., `CUCASCADE_CUDA_TRY`, `CUCASCADE_FUNC_RANGE`)

## Where to Add New Code

**New data representation type (e.g., disk-resident compressed format):**
- Implement class deriving from `idata_representation` in `include/cucascade/data/` (header) and `src/data/` (source)
- Register converters in `src/data/representation_converter.cpp` in `register_builtin_converters()`, or let callers register them at runtime
- Add tests in `test/data/test_data_representation.cpp` or a new `test_data_<name>.cpp`

**New memory resource or tier variant:**
- Add a new `_memory_space_config` struct in `include/cucascade/memory/config.hpp`
- Add a constructor overload in `memory_space` (`include/cucascade/memory/memory_space.hpp`, `src/memory/memory_space.cpp`)
- Implement the resource adaptor in `include/cucascade/memory/` and `src/memory/`
- Update `reservation_manager_configurator` to support the new tier

**New reservoir strategy:**
- Derive from `reservation_request_strategy` in `include/cucascade/memory/memory_reservation_manager.hpp`
- Implement `get_candidates(memory_reservation_manager&)` in `src/memory/memory_reservation_manager.cpp`

**New repository eviction policy (e.g., LRU):**
- Derive from `idata_repository<PtrType>` in a new header under `include/cucascade/data/`
- Override `add_data_batch()` and/or `pop_data_batch()` with policy-specific logic
- Add implementation in `src/data/`, add source to `src/data/CMakeLists.txt`

**New utility primitive:**
- Place in `include/cucascade/utils/` (header-only is preferred for small templates)

**New test:**
- Add `.cpp` file to the appropriate `test/data/` or `test/memory/` directory
- Register it in the corresponding `test/data/CMakeLists.txt` or `test/memory/CMakeLists.txt`

## Special Directories

**`.planning/`:**
- Purpose: GSD planning artifacts (codebase analysis, phase plans)
- Generated: No (manually maintained)
- Committed: Yes

**`.pixi/`:**
- Purpose: Pixi-managed environment (compiled packages, toolchain)
- Generated: Yes (by `pixi install`)
- Committed: No

**`cmake/`:**
- Purpose: CMake package config template (`cuCascadeConfig.cmake.in`)
- Generated: No
- Committed: Yes

**`build/` (if present after cmake configure):**
- Purpose: CMake out-of-source build artifacts
- Generated: Yes
- Committed: No

---

*Structure analysis: 2026-04-13*
