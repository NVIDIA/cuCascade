# Architecture

**Analysis Date:** 2026-04-13

## Pattern Overview

**Overall:** Two-subsystem library — a Data Management subsystem and a Memory Management subsystem — with clean interface boundaries between them.

**Key Characteristics:**
- Header-only interfaces in `include/cucascade/` with implementations in `src/`
- All public API lives under the `cucascade` namespace; memory internals under `cucascade::memory`
- Template-parameterized by `PtrType` (shared_ptr vs. unique_ptr) so callers control ownership semantics
- Policy-based design for memory reservation limits, OOM handling, and downgrade strategies
- Thread-safety enforced through RAII accessor types, not raw lock calls at the call site

## Layers

**Data Layer (pipeline-facing):**
- Purpose: Represent, store, and route cuDF table batches through a processing pipeline
- Location: `include/cucascade/data/`, `src/data/`
- Contains: `idata_representation` hierarchy, `synchronized_data_batch`, `idata_repository`, `data_repository_manager`, `representation_converter_registry`
- Depends on: Memory layer (for `memory_space`, tier information), libcudf (for `cudf::table`)
- Used by: External pipeline executors / query engines

**Memory Layer (resource management):**
- Purpose: Abstract GPU, host (NUMA-pinned), and disk storage into a unified reservation-based interface
- Location: `include/cucascade/memory/`, `src/memory/`
- Contains: `memory_space`, `memory_reservation_manager`, `reservation`, `reservation_aware_resource_adaptor`, `fixed_size_host_memory_resource`, `disk_access_limiter`, `notification_channel`, `reservation_manager_configurator`, `topology_discovery`
- Depends on: RMM, CUDA runtime, libnuma, NVML
- Used by: Data layer, external callers that need to make/release reservations

**Utility Layer:**
- Purpose: Shared low-level primitives used across both subsystems
- Location: `include/cucascade/utils/`, `include/cucascade/cuda_utils.hpp`
- Contains: `atomic_bounded_counter`, `atomic_peak_tracker`, `overloaded` (std::visit helper), CUDA error-checking macros (`CUCASCADE_CUDA_TRY`, `CUCASCADE_CUDA_TRY_ALLOC`)
- Depends on: Nothing outside C++20 standard library and CUDA headers
- Used by: Both data and memory layers

## Data Flow

**Batch ingestion and routing:**

1. Caller creates a `synchronized_data_batch` with a `uint64_t` ID and an `idata_representation` (e.g., `gpu_table_representation` wrapping a `cudf::table`)
2. Caller invokes `data_repository_manager::add_data_batch(batch, operator_ports)` — for `shared_ptr` repositories the batch is copied to each destination; for `unique_ptr` only a single destination is allowed
3. Each `idata_repository<PtrType>` stores the batch in its internal `_data_batches[partition_idx]` vector and notifies waiting threads via `_cv`
4. Consumer calls `pop_data_batch()` or `pop_data_batch_by_id()` to retrieve and take ownership; `get_data_batch_by_id()` is available only for `shared_ptr` repositories (non-destructive copy)

**Memory tier downgrade (GPU → HOST → DISK):**

1. `memory_space::should_downgrade_memory()` returns true when `get_total_reserved_memory() > _start_downgrading_memory_threshold`
2. External orchestrator identifies downgrade candidates via `data_repository_manager::get_data_batches_for_downgrade()`
3. On a target batch: acquire `mutable_data_batch` accessor, call `data_batch::convert_to<TargetRepresentation>(registry, target_memory_space, stream)` — this invokes `representation_converter_registry::convert<>()` which dispatches to the registered converter function via `(source_type, target_type)` key
4. Built-in converters (registered via `register_builtin_converters()`): `gpu_table_representation` → `host_data_packed_representation` (uses `cudf::pack`) and `gpu_table_representation` → `host_data_representation` (direct per-column buffer copy, cheaper)
5. Reservation on the old tier is released when the old `idata_representation` unique_ptr is destroyed; a new reservation on the target tier is held by the new representation

**Lock-safe batch access:**

1. Caller calls `synchronized_data_batch::get_read_only()` — acquires `std::shared_lock`, returns `read_only_data_batch` RAII handle
2. Multiple concurrent readers can hold `read_only_data_batch` simultaneously; `operator->` exposes `const data_batch*`
3. Writer calls `synchronized_data_batch::get_mutable()` — acquires `std::unique_lock`, returns `mutable_data_batch` handle; blocks until all readers release
4. Upgrade/downgrade: `mutable_data_batch::from_read_only(ro)` releases shared lock then blocks for unique lock; `read_only_data_batch::from_mutable(rw)` releases unique lock then re-acquires shared lock

**State Management:**
- Subscriber interest is tracked by `synchronized_data_batch::_subscriber_count` (atomic, no lock needed)
- Batch IDs are monotonically increasing atomics in `data_repository_manager::_next_data_batch_id`
- Memory pressure state lives in `memory_space` (`_start_downgrading_memory_threshold`, `_stop_downgrading_memory_threshold`)

## Key Abstractions

**`idata_representation` (abstract base):**
- Purpose: Uniform handle for data regardless of which memory tier it lives in
- Examples: `include/cucascade/data/gpu_data_representation.hpp` (`gpu_table_representation`), `include/cucascade/data/cpu_data_representation.hpp` (`host_data_representation`, `host_data_packed_representation`)
- Pattern: Virtual interface with `get_size_in_bytes()`, `get_uncompressed_data_size_in_bytes()`, `clone()`, plus a compile-time-checked `cast<TargetType>()` template

**`synchronized_data_batch` (concurrency wrapper):**
- Purpose: Owns an inner `data_batch` (pure data) plus a `shared_mutex`; exposes data only through RAII accessor types that bundle lock lifetime
- Examples: `include/cucascade/data/data_batch.hpp`
- Pattern: Pimpl-style separation — the inner `data_batch` class is private and only accessible via `read_only_data_batch` or `mutable_data_batch`

**`idata_repository<PtrType>` (template storage):**
- Purpose: Partitioned, thread-safe collection of `synchronized_data_batch` pointers with blocking/non-blocking pop semantics
- Examples: `include/cucascade/data/data_repository.hpp`; instantiated as `shared_data_repository` and `unique_data_repository`
- Pattern: Template with `shared_ptr` and `unique_ptr` specializations; `get_data_batch_by_id` only supported for `shared_ptr` (explicit template specializations in `src/data/data_repository.cpp`)

**`representation_converter_registry` (type-keyed dispatch):**
- Purpose: Registers and dispatches conversion functions between `idata_representation` subclasses, keyed on `(typeid(source), typeid(target))`
- Examples: `include/cucascade/data/representation_converter.hpp`
- Pattern: `std::unordered_map<converter_key, representation_converter_fn>` with mutex; extensible at runtime; built-in converters registered via `register_builtin_converters()`

**`memory_space` (tier abstraction):**
- Purpose: Represents one physical memory region (one GPU device, one NUMA node, or one disk mount) with fixed capacity, reservation tracking, and threshold-based downgrade signals
- Examples: `include/cucascade/memory/memory_space.hpp`, constructed from `gpu_memory_space_config`, `host_memory_space_config`, or `disk_memory_space_config`
- Pattern: Non-copyable, non-movable (addresses must be stable for `reservation` back-pointers); owns the RMM allocator and stream pool

**`reservation_request_strategy` (strategy pattern):**
- Purpose: Pluggable selection of which `memory_space` candidates to try when requesting a reservation
- Examples: `include/cucascade/memory/memory_reservation_manager.hpp` — concrete types: `any_memory_space_in_tier`, `any_memory_space_in_tier_with_preference`, `any_memory_space_in_tiers`, `specific_memory_space`, `any_memory_space_to_downgrade`, `any_memory_space_to_upgrade`
- Pattern: Abstract base with `get_candidates(memory_reservation_manager&)` returning ordered candidate list; manager tries each candidate in order

## Entry Points

**`memory_reservation_manager` construction:**
- Location: `include/cucascade/memory/memory_reservation_manager.hpp`
- Triggers: External caller provides `std::vector<memory_space_config>` (built using `reservation_manager_configurator`)
- Responsibilities: Owns all `memory_space` instances, provides the `request_reservation(strategy, size)` entry point, coordinates cross-space waiting via condition variable

**`reservation_manager_configurator::build()`:**
- Location: `include/cucascade/memory/reservation_manager_configurator.hpp`
- Triggers: Called after fluent builder configuration; optionally accepts `system_topology_info` from `topology_discovery::discover()`
- Responsibilities: Produces the `std::vector<memory_space_config>` consumed by `memory_reservation_manager`

**`data_repository_manager::add_data_batch()`:**
- Location: `include/cucascade/data/data_repository_manager.hpp`
- Triggers: Pipeline stage pushes a finished batch for downstream consumption
- Responsibilities: Routes batch to one or more operator-port repositories; generates unique IDs via `get_next_data_batch_id()`

**`register_builtin_converters(registry)`:**
- Location: `include/cucascade/data/representation_converter.hpp`, implemented in `src/data/representation_converter.cpp`
- Triggers: Called once during system initialization before any tier migration is attempted
- Responsibilities: Registers the GPU-to-host and host-to-GPU converter functions

## Error Handling

**Strategy:** Exceptions for all error paths. No error codes.

**Patterns:**
- CUDA API errors: `CUCASCADE_CUDA_TRY(call)` throws `rmm::cuda_error`; `CUCASCADE_CUDA_TRY_ALLOC(call)` throws `rmm::out_of_memory` or `rmm::bad_alloc`
- Out-of-memory during allocation: `oom_handling_policy` interface — `throw_on_oom_policy` (default) rethrows; alternative policies can retry
- Reservation overflow: `reservation_limit_policy` interface — `fail_reservation_limit_policy` throws `rmm::out_of_memory`; `ignore_reservation_limit_policy` allows overflow; `increase_reservation_limit_policy` attempts to grow the reservation
- Repository access errors: `std::out_of_range` for bad partition indices, `std::runtime_error` for unsupported operations (e.g., `get_data_batch_by_id` on `unique_ptr` repository)
- Converter not found: `std::runtime_error` from `representation_converter_registry::convert_impl`

## Cross-Cutting Concerns

**Logging:** No internal logging framework. NVTX range annotations are conditional on `CUCASCADE_NVTX` compile flag (macro: `CUCASCADE_FUNC_RANGE()`). `memory_space::to_string()` provides human-readable state for diagnostics.

**Validation:** Constructor-time via exceptions (e.g., `memory_reservation_manager` throws `std::invalid_argument` on empty config). Compile-time via `static_assert` in `representation_converter_registry` template methods and C++20 `requires` constraints on `idata_representation::cast<T>()`.

**Concurrency:** Every public container (`idata_repository`, `data_repository_manager`, `representation_converter_registry`, `reservation_aware_resource_adaptor`) is internally mutex-protected. `synchronized_data_batch` uses `shared_mutex` for read/write separation. Atomic counters (`_subscriber_count`, `_next_data_batch_id`, `_total_allocated_bytes`) avoid locks on hot paths. `notification_channel` provides condition-variable-based cross-thread signaling when reservations are released.

---

*Architecture analysis: 2026-04-13*
