<!-- GSD:project-start source:PROJECT.md -->
## Project

**cuCascade data_batch Refactor (PR #99 v2)**

A redesign of cuCascade's `data_batch` concurrency model, addressing review feedback from @dhruv9vats on PR #99. Replaces the current `synchronized_data_batch` nested-class wrapper with a simpler 3-class design where `data_batch` is the "idle" state and all data access requires acquiring a lock through RAII accessor types. Targets cuCascade's data layer (`include/cucascade/data/`, `src/data/`).

**Core Value:** Compile-time enforced data access safety: it must be impossible to read or mutate batch data without holding the appropriate lock, and move semantics must make stale references a compile error.

### Constraints

- **C++ standard**: C++20 — must compile with CUDA 12.9/13.x toolchains
- **API boundary**: Changes are breaking — `data_batch` replaces `synchronized_data_batch` throughout
- **Thread safety**: All public API must be safe for concurrent access from multiple threads
- **Build**: Must pass `pixi run build` (all 61 targets) and `pixi run test` (all tests)
- **Backward compatibility**: None required — this is a breaking API change (PR already tagged `feat!`)
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- C++20 - All library source, headers, tests, and benchmarks
- CUDA 20 - GPU kernel code in `src/` (CUDA standard applied privately to object library)
- C - Listed in CMake LANGUAGES (C CXX CUDA) for compatibility
- Python - Documentation generation scripts in `scripts/` (Jinja2-based API doc generator)
## Runtime
- Linux (linux-64 or linux-aarch64 only; explicitly declared in `pixi.toml`)
- NVIDIA GPU required at runtime (Turing/SM75 minimum, up to SM120 supported)
- Pixi >= 0.59 (conda-based, defined in `pixi.toml`)
- No lockfile committed (cache disabled in CI)
## Frameworks
- libcudf 26.06 (nightly) / 26.02 (stable) - GPU DataFrame library; provides `cudf::table`, pack/unpack serialization, and column memory management
- RMM (RAPIDS Memory Manager) - Device memory resource abstraction; base class `rmm::mr::device_memory_resource` is subclassed throughout `include/cucascade/memory/`
- CUDA Toolkit (12.9 or 13.x) - `CUDA::cudart` for stream management, `CUDA::nvml` (dev headers) for topology discovery
- Catch2 v2.13.10 - Fetched via CMake `FetchContent` from GitHub at build time; single-include header used in `test/`
- CTest - CMake test runner; registered via `add_test(NAME cucascade_tests ...)`
- Google Benchmark v1.8.3 - Fetched via CMake `FetchContent` from GitHub at build time; used in `benchmark/`
- CMake 4.x (pixi dep), minimum 3.26.4 (enforced in `CMakeLists.txt`)
- Ninja - Build generator (set in `CMakePresets.json` default preset)
- sccache - Compiler cache; applied to C, C++, and CUDA compilers via `CMAKE_*_COMPILER_LAUNCHER` in `CMakePresets.json`
- Doxygen - API documentation generation from header comments
- pre-commit - Linting/formatting gate (see `.pre-commit-config.yaml`)
## Key Dependencies
- `rmm::rmm` - Every memory resource in `include/cucascade/memory/` subclasses `rmm::mr::device_memory_resource`; RMM stream types (`rmm::cuda_stream_view`, `rmm::cuda_stream`) used throughout APIs
- `cudf::cudf` - `gpu_table_representation` wraps `cudf::table`; `cudf::pack`/`unpack` used for GPU→HOST serialization in built-in converters; direct cudf API access is the GPU data path
- `CUDA::cudart` - Stream creation/synchronization in `exclusive_stream_pool` (`include/cucascade/memory/stream_pool.hpp`)
- `CUDA::nvml` (headers only) - `nvml.h` included in `src/memory/topology_discovery.cpp`; loaded at runtime via `dlopen("libnvidia-ml.so.1")` to avoid link-time dependency
- `numa` (libnuma) - Linked as `${NUMA_LIB}`; used in `numa_region_pinned_host_memory_resource` (`include/cucascade/memory/numa_region_pinned_host_allocator.hpp`) for NUMA-aware pinned host allocation
- `Threads::Threads` - std::mutex, std::shared_mutex, std::condition_variable used in `memory_reservation_manager`, `exclusive_stream_pool`, and `synchronized_data_batch`
- `cuda::mr` (CCCL) - `cuda::mr::device_accessible` and `cuda::mr::host_accessible` properties declared on `numa_region_pinned_host_memory_resource`
- `fmt` (pixi dep) - Available but not observed directly in headers; likely used in implementation files
- `libcurand-dev` (pixi dep) - Available for random data generation (used in tests/benchmarks)
- `numactl` (pixi dep) - Provides libnuma at build time
## Configuration
- `CUDA_VISIBLE_DEVICES` - Respected by `topology_discovery::discover()` to filter visible GPU indices (supports numeric indices, GPU UUIDs, and MIG device UUIDs)
- `CUDAARCHS` - Set per pixi environment feature (e.g., `75-real;80-real;86-real;90a-real;100f-real;120a-real;120` for CUDA 13)
- `CMAKE_PREFIX_PATH` - Passed through from environment for locating RMM/cuDF/CUDAToolkit packages
- `SCCACHE_GHA_ENABLED` - Set to `true` in CI for GitHub Actions sccache integration
- `CMakePresets.json` - Defines `debug`, `release`, `relwithdebinfo` presets; all use Ninja generator and sccache
- Build outputs: `build/debug/`, `build/release/`, `build/relwithdebinfo/`
- CMake options:
## Platform Requirements
- Linux x86_64 or aarch64
- NVIDIA GPU (SM75+ / Turing+)
- CUDA 12.9 or 13.x toolkit installed
- Pixi >= 0.59
- Linux only (topology discovery reads `/sys/bus/pci/devices/`, `/sys/class/infiniband`, `/sys/class/nvme`)
- NVIDIA driver with `libnvidia-ml.so.1` available at runtime (topology discovery degrades gracefully without it)
- Optional: InfiniBand/RoCE NICs (discovered via `/sys/class/infiniband`) and NVMe drives (via `/sys/class/nvme`)
- Library outputs: `libcucascade.so` (versioned 0.1.0) and/or `libcucascade.a`
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Source files: `snake_case.cpp` — e.g., `data_batch.cpp`, `memory_reservation.cpp`
- Header files: `snake_case.hpp` — e.g., `data_repository.hpp`, `cuda_utils.hpp`
- CUDA kernel files: `snake_case.cu` / `snake_case.cuh`
- Test files: `test_<subject>.cpp` — e.g., `test_data_batch.cpp`
- Concrete classes: `snake_case` — e.g., `synchronized_data_batch`, `memory_reservation_manager`
- Abstract interfaces: prefix `i` + `snake_case` — e.g., `idata_representation`, `idata_repository`
- Enumerations: `PascalCase` for name, `UPPER_CASE` for enumerators — e.g., `enum class Tier { GPU, HOST, DISK, SIZE }`
- Template traits: `snake_case_trait` — e.g., `tier_memory_resource_trait<Tier::GPU>`
- All private/protected: underscore prefix + snake_case — `_batch_id`, `_rw_mutex`, `_subscriber_count`
- No `m_` Hungarian prefix; use single `_` prefix exclusively
- All functions: `snake_case` — `get_batch_id()`, `set_data()`, `try_get_read_only()`
- Factory functions: `make_<name>` — e.g., `make_mock_memory_space()`, `make_default_gpu_memory_resource()`
- Predicate methods: `has_<name>`, `is_<name>` — e.g., `has_converter()`, `all_empty()`
- Boolean-returning subscriptions: `subscribe()` / `unsubscribe()`
- All caps with project prefix: `CUCASCADE_CUDA_TRY`, `CUCASCADE_CUDA_TRY_ALLOC`, `CUCASCADE_FUNC_RANGE`, `CUCASCADE_STRINGIFY`
- Root: `cucascade`
- Sub-namespaces by domain: `cucascade::memory`, `cucascade::utils`
- Test-only utilities: `cucascade::test`
- Anonymous namespaces: used in `.cpp` files for file-local helpers
- `PascalCase` — e.g., `TargetRepresentation`, `TargetType`, `TIER` (compile-time value parameter uses UPPER_CASE)
## Code Style
- Tool: `clang-format` v20.1.4 via pre-commit hook
- Config: `.clang-format` (root, BasedOnStyle: Google with overrides)
- Column limit: 100 characters
- Indent width: 2 spaces (no tabs)
- Brace style: WebKit — opening brace on same line for all constructs
- Pointer alignment: Left — `int* ptr` not `int *ptr`
- Trailing comments: 2 spaces before `//`
- `AlignConsecutiveAssignments: true` — align `=` in consecutive declarations
- `AlignTrailingComments: true`
- `BinPackParameters: false` — each parameter on its own line if wrapping
- `BreakConstructorInitializers: BeforeColon` — `:` on new line for init lists
- `ConstructorInitializerIndentWidth: 2` / `ContinuationIndentWidth: 2`
- `MaxEmptyLinesToKeep: 1`
- Tool: `cmake-lint` and `cmake-format` (via pre-commit)
- C++ linting: compiler warnings treated as errors (`-Wall -Wextra -Wpedantic -Wcast-align -Wunused -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion -Wformat=2 -Wimplicit-fallthrough`)
- Spell checking: `codespell` (with `.codespell_words` ignore list)
## Import Organization
## Error Handling
- CUDA runtime errors: use `CUCASCADE_CUDA_TRY(call)` macro — throws `rmm::cuda_error` with file/line context. Defined in `include/cucascade/cuda_utils.hpp`
- CUDA allocation failures: use `CUCASCADE_CUDA_TRY_ALLOC(call)` or `CUCASCADE_CUDA_TRY_ALLOC(call, num_bytes)` — throws `rmm::out_of_memory` or `rmm::bad_alloc`
- Debug CUDA checks: `CUCASCADE_ASSERT_CUDA_SUCCESS(call)` — asserts in debug, no-op in release
- Memory allocation errors: throw `rmm::out_of_memory`, `rmm::bad_alloc`, or custom `cucascade_out_of_memory` (subclasses `rmm::out_of_memory`)
- Logic/precondition failures: throw `std::runtime_error`, `std::invalid_argument`, `std::logic_error`
- Custom error system: `MemoryError` enum + `std::error_code` integration in `src/memory/error.cpp`
- Destructors: `noexcept` — all cleanup in destructors must not throw
## Comments
- All public API methods and classes documented with Doxygen `/** */` blocks
- Tags used: `@brief`, `@param`, `@return`, `@throws`, `@note`, `@tparam`, `@code`/`@endcode`
- Member fields: `///< inline comment` after declaration — e.g., `int64_t _size; ///< doc`
- Interface classes (`idata_representation`, `idata_repository`) are heavily documented with intent and invariants
- Explain non-obvious behavior — e.g., why a lock is released before re-acquired
- Mark disabled tests: `// Disabled: reason` comment before `TEST_CASE`
- Rarely used for obvious code
## Function Design
- Stream parameters: always `rmm::cuda_stream_view stream` (not `cudaStream_t` directly)
- Output via return values, not output parameters
- Pass `std::unique_ptr` by value to transfer ownership; pass `const T&` or `T*` for non-owning access
- `[[maybe_unused]]` attribute on unused parameters when required by interface — e.g., `[[maybe_unused]] rmm::cuda_stream_view stream`
- Nullable results: return `nullptr` (for pointer types) or `std::nullopt` (for `std::optional`)
- Non-blocking acquire: `std::optional<accessor_type>` — e.g., `try_get_read_only()`, `try_get_mutable()`
- Factory functions return `std::unique_ptr` or `std::shared_ptr`
## Module Design
- Public API headers in `include/cucascade/` only — never expose internal headers
- Implementation details in `src/` (not installed)
- Template implementations that must be in headers live in the `.hpp` file below the class definition
- `requires std::derived_from<T, BaseType>` used for template type safety
- `static_assert(std::is_base_of_v<Base, Derived>, "message")` in template implementations
- `std::integral T` constraint on atomic utility templates
## Namespace Closure Comments
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Header-only interfaces in `include/cucascade/` with implementations in `src/`
- All public API lives under the `cucascade` namespace; memory internals under `cucascade::memory`
- Template-parameterized by `PtrType` (shared_ptr vs. unique_ptr) so callers control ownership semantics
- Policy-based design for memory reservation limits, OOM handling, and downgrade strategies
- Thread-safety enforced through RAII accessor types, not raw lock calls at the call site
## Layers
- Purpose: Represent, store, and route cuDF table batches through a processing pipeline
- Location: `include/cucascade/data/`, `src/data/`
- Contains: `idata_representation` hierarchy, `synchronized_data_batch`, `idata_repository`, `data_repository_manager`, `representation_converter_registry`
- Depends on: Memory layer (for `memory_space`, tier information), libcudf (for `cudf::table`)
- Used by: External pipeline executors / query engines
- Purpose: Abstract GPU, host (NUMA-pinned), and disk storage into a unified reservation-based interface
- Location: `include/cucascade/memory/`, `src/memory/`
- Contains: `memory_space`, `memory_reservation_manager`, `reservation`, `reservation_aware_resource_adaptor`, `fixed_size_host_memory_resource`, `disk_access_limiter`, `notification_channel`, `reservation_manager_configurator`, `topology_discovery`
- Depends on: RMM, CUDA runtime, libnuma, NVML
- Used by: Data layer, external callers that need to make/release reservations
- Purpose: Shared low-level primitives used across both subsystems
- Location: `include/cucascade/utils/`, `include/cucascade/cuda_utils.hpp`
- Contains: `atomic_bounded_counter`, `atomic_peak_tracker`, `overloaded` (std::visit helper), CUDA error-checking macros (`CUCASCADE_CUDA_TRY`, `CUCASCADE_CUDA_TRY_ALLOC`)
- Depends on: Nothing outside C++20 standard library and CUDA headers
- Used by: Both data and memory layers
## Data Flow
- Subscriber interest is tracked by `synchronized_data_batch::_subscriber_count` (atomic, no lock needed)
- Batch IDs are monotonically increasing atomics in `data_repository_manager::_next_data_batch_id`
- Memory pressure state lives in `memory_space` (`_start_downgrading_memory_threshold`, `_stop_downgrading_memory_threshold`)
## Key Abstractions
- Purpose: Uniform handle for data regardless of which memory tier it lives in
- Examples: `include/cucascade/data/gpu_data_representation.hpp` (`gpu_table_representation`), `include/cucascade/data/cpu_data_representation.hpp` (`host_data_representation`, `host_data_packed_representation`)
- Pattern: Virtual interface with `get_size_in_bytes()`, `get_uncompressed_data_size_in_bytes()`, `clone()`, plus a compile-time-checked `cast<TargetType>()` template
- Purpose: Owns an inner `data_batch` (pure data) plus a `shared_mutex`; exposes data only through RAII accessor types that bundle lock lifetime
- Examples: `include/cucascade/data/data_batch.hpp`
- Pattern: Pimpl-style separation — the inner `data_batch` class is private and only accessible via `read_only_data_batch` or `mutable_data_batch`
- Purpose: Partitioned, thread-safe collection of `synchronized_data_batch` pointers with blocking/non-blocking pop semantics
- Examples: `include/cucascade/data/data_repository.hpp`; instantiated as `shared_data_repository` and `unique_data_repository`
- Pattern: Template with `shared_ptr` and `unique_ptr` specializations; `get_data_batch_by_id` only supported for `shared_ptr` (explicit template specializations in `src/data/data_repository.cpp`)
- Purpose: Registers and dispatches conversion functions between `idata_representation` subclasses, keyed on `(typeid(source), typeid(target))`
- Examples: `include/cucascade/data/representation_converter.hpp`
- Pattern: `std::unordered_map<converter_key, representation_converter_fn>` with mutex; extensible at runtime; built-in converters registered via `register_builtin_converters()`
- Purpose: Represents one physical memory region (one GPU device, one NUMA node, or one disk mount) with fixed capacity, reservation tracking, and threshold-based downgrade signals
- Examples: `include/cucascade/memory/memory_space.hpp`, constructed from `gpu_memory_space_config`, `host_memory_space_config`, or `disk_memory_space_config`
- Pattern: Non-copyable, non-movable (addresses must be stable for `reservation` back-pointers); owns the RMM allocator and stream pool
- Purpose: Pluggable selection of which `memory_space` candidates to try when requesting a reservation
- Examples: `include/cucascade/memory/memory_reservation_manager.hpp` — concrete types: `any_memory_space_in_tier`, `any_memory_space_in_tier_with_preference`, `any_memory_space_in_tiers`, `specific_memory_space`, `any_memory_space_to_downgrade`, `any_memory_space_to_upgrade`
- Pattern: Abstract base with `get_candidates(memory_reservation_manager&)` returning ordered candidate list; manager tries each candidate in order
## Entry Points
- Location: `include/cucascade/memory/memory_reservation_manager.hpp`
- Triggers: External caller provides `std::vector<memory_space_config>` (built using `reservation_manager_configurator`)
- Responsibilities: Owns all `memory_space` instances, provides the `request_reservation(strategy, size)` entry point, coordinates cross-space waiting via condition variable
- Location: `include/cucascade/memory/reservation_manager_configurator.hpp`
- Triggers: Called after fluent builder configuration; optionally accepts `system_topology_info` from `topology_discovery::discover()`
- Responsibilities: Produces the `std::vector<memory_space_config>` consumed by `memory_reservation_manager`
- Location: `include/cucascade/data/data_repository_manager.hpp`
- Triggers: Pipeline stage pushes a finished batch for downstream consumption
- Responsibilities: Routes batch to one or more operator-port repositories; generates unique IDs via `get_next_data_batch_id()`
- Location: `include/cucascade/data/representation_converter.hpp`, implemented in `src/data/representation_converter.cpp`
- Triggers: Called once during system initialization before any tier migration is attempted
- Responsibilities: Registers the GPU-to-host and host-to-GPU converter functions
## Error Handling
- CUDA API errors: `CUCASCADE_CUDA_TRY(call)` throws `rmm::cuda_error`; `CUCASCADE_CUDA_TRY_ALLOC(call)` throws `rmm::out_of_memory` or `rmm::bad_alloc`
- Out-of-memory during allocation: `oom_handling_policy` interface — `throw_on_oom_policy` (default) rethrows; alternative policies can retry
- Reservation overflow: `reservation_limit_policy` interface — `fail_reservation_limit_policy` throws `rmm::out_of_memory`; `ignore_reservation_limit_policy` allows overflow; `increase_reservation_limit_policy` attempts to grow the reservation
- Repository access errors: `std::out_of_range` for bad partition indices, `std::runtime_error` for unsupported operations (e.g., `get_data_batch_by_id` on `unique_ptr` repository)
- Converter not found: `std::runtime_error` from `representation_converter_registry::convert_impl`
## Cross-Cutting Concerns
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, or `.github/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
