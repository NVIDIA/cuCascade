# Codebase Concerns

**Analysis Date:** 2026-04-13

## Tech Debt

**GPU Memory Resource Placeholder:**
- Issue: `gpu_table_representation` documents that it will eventually use a custom GPU memory resource, but currently the allocation type is `IAllocatedMemory` (raw RMM). The owning allocation object is never stored — only the `cudf::table` is kept. This means GPU memory is not tracked or bounded by the reservation system for GPU data.
- Files: `include/cucascade/data/gpu_data_representation.hpp:38-40`, `src/data/gpu_data_representation.cpp`
- Impact: GPU memory usage bypasses the `reservation_aware_resource_adaptor` reservation enforcement after data is promoted to GPU. OOM pressure on the GPU side is not surfaced to the tiering system until the allocator fails.
- Fix approach: Introduce a GPU-side allocation wrapper similar to `fixed_multiple_blocks_allocation` and store it alongside the `cudf::table` in `gpu_table_representation`.

**NUMA-Aware Pinned Memory Not Implemented:**
- Issue: `fixed_size_host_memory_resource` explicitly documents that it does not yet handle multiple NUMA domains. The comment says "We need to make our own pinned allocator that allocates numa local memory and then registers it with cuda."
- Files: `include/cucascade/memory/fixed_size_host_memory_resource.hpp:47-48`
- Impact: In multi-socket systems, host memory may be allocated on a remote NUMA node relative to the GPU, degrading H2D/D2H bandwidth by 30–50%. NUMA-local placement is critical for NVLink bandwidth.
- Fix approach: Implement a custom pinned allocator using `numa_alloc_onnode()` + `cudaHostRegister()`. The `host_memory_space_config.numa_id` field already exists for this purpose.

**Disk Tier is a Capacity Limiter Only:**
- Issue: `disk_access_limiter` tracks bytes reserved against a capacity limit, but performs no actual disk I/O. The `disk_reserved_arena::base_name()` returns a file path string, but no file creation, write, or read logic exists anywhere. The DISK tier is wired into the reservation and tiering infrastructure but is non-functional as storage.
- Files: `src/memory/disk_access_limiter.cpp`, `include/cucascade/memory/disk_access_limiter.hpp`
- Impact: Any code path that attempts to spill data to the DISK tier will silently succeed in acquiring a reservation but the data never actually moves to disk. There is no converter registered for disk-resident representations in `register_builtin_converters`.
- Fix approach: Implement disk serialization (e.g., Apache Arrow IPC or Parquet) in a `disk_data_representation` class and register converters for GPU→DISK and HOST→DISK paths.

**`get_data_batches_for_downgrade` is a Stub:**
- Issue: `data_repository_manager::get_data_batches_for_downgrade()` is a completely empty placeholder that returns an empty vector. Downgrade logic that calls this will silently do nothing.
- Files: `include/cucascade/data/data_repository_manager.hpp:213-221`
- Impact: Any eviction/tiering logic relying on this method to identify downgrade candidates is broken. Memory pressure on the GPU tier will not trigger actual data movement.
- Fix approach: Implement eviction candidate selection by iterating `_repositories`, locking batches that are in GPU tier and not currently in use, and returning them up to `amount_to_downgrade` bytes.

**`do_reserve_upto` in `disk_access_limiter` Returns Wrong Value:**
- Issue: `disk_access_limiter::do_reserve_upto()` calls `_total_allocated_bytes.add_bounded(size_bytes, limit_bytes)` which can modify `size_bytes` via a reference parameter to the actual reserved amount. However, the method then returns the original `size_bytes` parameter value, not the actual bounded amount. The RAII arena created in `reserve_upto()` is given `reserved_bytes` which is the return value of `do_reserve_upto` — so the arena size is wrong when the limit is hit mid-reservation.
- Files: `src/memory/disk_access_limiter.cpp:112-117`, `include/cucascade/utils/atomics.hpp:87-97`
- Impact: `disk_reserved_arena._size` may claim to hold more bytes than were actually reserved. Any capacity accounting downstream using `arena.size()` will be incorrect.
- Fix approach: `do_reserve_upto` should return the post-bounded `diff` value, not the original `size_bytes`.

**`reservation_manager_configurator` Uses `assert()` for Input Validation:**
- Issue: Throughout `src/memory/reservation_manager_configurator.cpp`, user-facing configuration errors (zero GPU count, invalid fractions, empty mount paths) are guarded with `assert()`, which is disabled in release builds. These assertions encode API contracts that must hold in production, not just debug.
- Files: `src/memory/reservation_manager_configurator.cpp:43,50,63,77,85,86,115,123,132,133,141,155,156,157,167,175,183`
- Impact: Invalid configurations silently succeed in release builds, potentially constructing an unusable `memory_reservation_manager`.
- Fix approach: Replace `assert()` with explicit `throw std::invalid_argument(...)` for all user-facing precondition checks, or use a dedicated validation step that runs unconditionally.

**Duplicate Data Structures for Two "Fast" Converter Paths:**
- Issue: There are two parallel GPU↔HOST converter families: a "packed" path (`host_data_packed_representation` / `cudf::pack`) and a "fast" path (`host_data_representation` / direct column-buffer copy). Both are registered in `register_builtin_converters`. Callers must select the correct target type at compile time. This doubles the surface area of converter combinations (GPU→HOST, GPU→HOST_FAST, HOST_FAST→GPU, HOST_FAST→HOST_FAST) without a unified decision point.
- Files: `src/data/representation_converter.cpp:818-848`, `include/cucascade/data/cpu_data_representation.hpp`
- Impact: New code must know which path to use; wrong selection compiles but produces a runtime "no converter registered" error. Integration burden is high when adding new representation types.
- Fix approach: Deprecate the packed path or make the "fast" path the default and hide the selection behind a single `convert_gpu_to_host` / `convert_host_to_gpu` entry point that always picks the fast path.

## Known Bugs

**`request_reservation` in `memory_reservation_manager` Has No Wake Mechanism:**
- Symptoms: `request_reservation()` blocks on `_wait_cv.wait(lock)` when no space is immediately available. The condition variable is never notified — there is no `_wait_cv.notify_all()` call anywhere in the codebase. Threads that block here will wait indefinitely.
- Files: `src/memory/memory_reservation_manager.cpp:155-164`, `include/cucascade/memory/memory_reservation_manager.hpp:278-280`
- Trigger: Any call to `request_reservation()` when all memory spaces are currently at capacity.
- Workaround: The `per-space` `notification_channel` does wake `memory_space::make_reservation()` correctly (single-space blocking), but `request_reservation()` in the manager uses a completely separate mutex/CV pair that is never signaled.
- Fix: Add `_wait_cv.notify_all()` wherever a reservation is released (at minimum in `reservation`'s destructor callback path, mirroring how `notification_channel` works).

**Move Constructor/Assignment of `synchronized_data_batch` Does Not Lock:**
- Symptoms: `synchronized_data_batch::synchronized_data_batch(synchronized_data_batch&& other)` and `operator=` move the inner `data_batch` without acquiring the mutex on `other`. If another thread holds a `read_only_data_batch` or `mutable_data_batch` referencing `other`, the move violates the locking contract because `other._batch` is modified while a live accessor may be dereferencing it via `parent_->_batch`.
- Files: `src/data/data_batch.cpp:91-99`, `include/cucascade/data/data_batch.hpp:167-171`
- Trigger: Moving a `synchronized_data_batch` instance while a live accessor (from another thread) is holding a lock on it. Practically this is unlikely since callers own the instance, but the design allows it.
- Workaround: Do not move a `synchronized_data_batch` while any accessor is live. There is no enforcement of this invariant.
- Fix: Either delete the move constructor and assignment entirely (since the class is non-copyable anyway and owning containers use pointers), or acquire `_rw_mutex` as a unique lock before moving `_batch`.

**`from_read_only` Upgrade Has an Unlocked Window:**
- Symptoms: `mutable_data_batch::from_read_only()` releases the shared lock, then blocks waiting for the unique lock. During this window the `parent_` pointer is not protected and the `synchronized_data_batch` could theoretically be destroyed. Similarly `from_mutable()` has the same unlock-then-lock gap.
- Files: `src/data/data_batch.cpp:58-78`, `include/cucascade/data/data_batch.hpp:107,143`
- Trigger: Caller destroys the `synchronized_data_batch` while a conversion is in progress between the unlock and the re-lock.
- Workaround: The accessor holds a raw pointer to the parent; callers must guarantee the parent outlives the accessor.
- Fix: Document this lifetime constraint explicitly in the API. Consider wrapping the accessor factory in a function that takes a `shared_ptr<synchronized_data_batch>` to prevent use-after-free.

## Security Considerations

**`const_cast` on `memory_space*` Throughout Converter Functions:**
- Risk: Every built-in converter in `representation_converter.cpp` casts the `const memory::memory_space*` parameter to non-const via `const_cast`. This is needed because `idata_representation` takes `memory_space&` (non-const). This breaks const-correctness guarantees and could mask aliasing bugs.
- Files: `src/data/representation_converter.cpp:179,226,279,334,602,760,813`
- Current mitigation: None.
- Recommendations: Either change `idata_representation` to accept `const memory_space&` (preferred), or change the converter API to pass `memory_space*` directly instead of `const memory_space*`.

**`const_cast` in `memory_reservation_manager` Span Returns:**
- Risk: `get_memory_spaces_for_tier()` and `get_all_memory_spaces()` return `span<const memory_space*>` by casting away the const of their internal mutable containers. The lifetime of the returned span is tied to the manager; callers holding the span after manager destruction get dangling pointers.
- Files: `src/memory/memory_reservation_manager.cpp:184,191`
- Current mitigation: None.
- Recommendations: Return by value (copy the vector of pointers) or introduce a `const_span` type.

## Performance Bottlenecks

**`pop_data_batch` Uses `vector::erase` at Front — O(N) Per Pop:**
- Problem: `idata_repository::pop_data_batch()` removes from the front of `std::vector` with `erase(begin())`, which shifts all remaining elements left. For large batches-in-flight this is O(N) per pop.
- Files: `include/cucascade/data/data_repository.hpp:119`
- Cause: `std::vector` used as a queue.
- Improvement path: Replace `std::vector<PtrType>` with `std::deque<PtrType>` for O(1) front-pop, or with a `std::list` when mid-vector removal by ID (`pop_data_batch_by_id`) is frequent.

**GPU-to-GPU Copy Always Uses `cudf::pack` Intermediate:**
- Problem: `convert_gpu_to_gpu` packs the entire table into a single contiguous GPU buffer via `cudf::pack`, then copies it to the target device, then unpacks it. This doubles peak GPU memory usage on the source device and is entirely synchronous.
- Files: `src/data/representation_converter.cpp:130-179`
- Cause: No direct peer-copy path for non-contiguous table buffers.
- Improvement path: Use the same column-by-column `cudaMemcpyPeerAsync` approach used in the "fast" H2D/D2H path to avoid the intermediate allocation.

**Multiple `stream.synchronize()` Calls in Hot Paths:**
- Problem: Both `convert_gpu_to_host` and `convert_host_to_gpu` (packed paths) call `stream.synchronize()` multiple times (once before pack, once after copy). The fast converters also call `stream.synchronize()` at the end. Each synchronize stalls the CPU until the GPU drains that stream.
- Files: `src/data/representation_converter.cpp:137,153,164,175,192,221,271,275,596,756`
- Cause: Sequential synchronous design for correctness; no attempt to pipeline or overlap GPU work with CPU-side setup.
- Improvement path: Use CUDA events to synchronize only the specific operation rather than draining the full stream. Overlap CPU metadata setup with GPU data copies.

**`fixed_size_host_memory_resource` Lock Granularity:**
- Problem: The entire allocator is protected by a single `_mutex`. Both `allocate_multiple_blocks_internal` and `expand_pool` hold this lock for the duration of upstream allocation, which can involve `cudaMallocHost`. This serializes all concurrent host block allocations.
- Files: `include/cucascade/memory/fixed_size_host_memory_resource.hpp:422`, `src/memory/fixed_size_host_memory_resource.cpp`
- Cause: Simple global mutex design.
- Improvement path: Use a lock-free free-list for the hot path (`allocate`/`deallocate`), reserving the global mutex only for `expand_pool`.

## Fragile Areas

**`reservoir_manager_configurator::build()` Calls `topology_discovery::discover()` and Ignores Failure:**
- Files: `src/memory/reservation_manager_configurator.cpp:191-192`
- Why fragile: `discover()` returns `bool` indicating success, but the return value is marked `[[maybe_unused]]` and only checked via `assert(status)`. In release builds, a failed discovery silently produces an empty topology. Subsequent GPU/NUMA lookups will throw or produce incorrect configurations.
- Safe modification: Always check `discover()` and throw `std::runtime_error` on failure, unconditionally.
- Test coverage: No tests for topology discovery failure paths.

**`synchronized_data_batch` Move is Accessible but Unsafe Under Concurrent Use:**
- Files: `include/cucascade/data/data_batch.hpp:167-171`, `src/data/data_batch.cpp:91-99`
- Why fragile: Move constructor is public and takes no locks. The class is stored in `std::vector` in one place (via `shared_ptr`), but if any code moves the value type directly while live accessors exist, it produces UB.
- Safe modification: If a `synchronized_data_batch` must be moveable, document that it can only be moved before any accessor is created. Consider marking move as `= delete` and always using heap allocation (the library already defaults to `shared_ptr`/`unique_ptr` ownership).
- Test coverage: Move tests exist but only check single-threaded behavior.

**`idata_repository::add_data_batch` Can Silently Resize Partition Vector:**
- Files: `include/cucascade/data/data_repository.hpp:78-88`
- Why fragile: `add_data_batch()` auto-resizes `_data_batches` to accommodate any `partition_idx`. `pop_data_batch` and friends throw `std::out_of_range` if `partition_idx` is out of range at call time, but the partition vector may have grown in a different call. There is no way to query which partition indices are valid after multi-threaded adds.
- Safe modification: Require partition count to be fixed at construction time (pass it to the constructor), or validate that `partition_idx` is within a pre-declared range.
- Test coverage: Partition resize behavior is tested, but concurrency with mixed add/pop across partitions is not.

## Scaling Limits

**`operator_port_key` Hash via `std::map` (O(log N) Lookup):**
- Current capacity: Acceptable for tens of operators.
- Limit: `data_repository_manager` uses `std::map<operator_port_key, ...>` which provides O(log N) lookup. At thousands of operators, repository lookup per batch becomes a bottleneck under the global `_mutex`.
- Scaling path: Replace `std::map` with `std::unordered_map` using a hash on `(operator_id, port_id)`.

**`converter_key_hash` Uses XOR — Prone to Collision:**
- Current capacity: Works for the small set of built-in converters.
- Limit: `h1 ^ (h2 << 1)` is a known-weak hash combination. With many custom converter registrations, collision rate degrades lookup performance.
- Scaling path: Use a better hash combiner (e.g., boost::hash_combine style: `h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2))`).

## Dependencies at Risk

**`cudf::pack` / `cudf::unpack` Used in Non-Fast Converters:**
- Risk: `cudf::pack` allocates a contiguous GPU buffer equal to the full table size. For large tables this can OOM the source GPU. The library documents this as a known issue in the GPU-to-GPU and GPU-to-HOST (non-fast) paths.
- Impact: GPU OOM on the source device when spilling large batches.
- Migration plan: The "fast" converters (`convert_gpu_to_host_fast`, `convert_host_fast_to_gpu`) bypass `cudf::pack`. The packed converters should be removed or restricted to small data only.

**`cudaMemcpyBatchAsync` Required for CUDA 12.8+:**
- Risk: The fast converter uses `cudaMemcpyBatchAsync` which requires CUDA 12.8+. A compile-time fallback exists for CUDA < 12.8, but the CUDA 13 vs CUDA 12.x API difference (presence of `failIdx` parameter) is handled via preprocessor conditionals that will require maintenance as CUDA versions advance.
- Files: `src/data/representation_converter.cpp:402-424`
- Impact: API breakage if CUDA removes or changes this call again.
- Migration plan: Centralize the CUDA version dispatch into a single helper function so future CUDA version changes require one edit.

## Missing Critical Features

**No Downgrade Execution Logic:**
- Problem: The full tiering pipeline (detect pressure → select candidates → convert to lower tier → update repository) has infrastructure (detection via `should_downgrade_memory()`, reservation strategies for downgrade) but no code that executes the actual downgrade. `get_data_batches_for_downgrade()` is a stub.
- Blocks: The library cannot operate as a tiered cache under memory pressure without this.

**No Disk-Tier Read/Write Implementation:**
- Problem: The DISK tier is configured, reservation is tracked, but data cannot actually be written to or read from disk.
- Blocks: Spilling beyond host memory capacity.

**No Batch ID Generation Coordination Across Repository Managers:**
- Problem: Each `data_repository_manager` has its own `_next_data_batch_id` counter starting at 0. If multiple managers exist in a system (one per query, for example), batch IDs will collide across managers.
- Blocks: Any cross-manager batch tracking or deduplication.

## Test Coverage Gaps

**No Tests for Disk Tier:**
- What's not tested: Disk reservation, `disk_access_limiter` under concurrent reservation, and any disk read/write path.
- Files: `src/memory/disk_access_limiter.cpp`, `include/cucascade/memory/disk_access_limiter.hpp`
- Risk: Disk tier bugs (including the `do_reserve_upto` accounting bug) will not be caught.
- Priority: High (once disk I/O is implemented).

**No Tests for `request_reservation` Blocking Path:**
- What's not tested: The code path in `memory_reservation_manager::request_reservation()` that blocks on `_wait_cv.wait(lock)`. The bug (never notified CV) cannot be caught without a test that exhausts all memory and then releases some.
- Files: `src/memory/memory_reservation_manager.cpp:155-164`
- Risk: Deadlock in production under memory pressure.
- Priority: High.

**No Tests for GPU-to-GPU Cross-Device Conversion:**
- What's not tested: `convert_gpu_to_gpu()` in `representation_converter.cpp`. Requires two physical GPUs.
- Files: `src/data/representation_converter.cpp:130-179`
- Risk: Peer copy configuration issues (e.g., peer access not enabled) will only be found at runtime.
- Priority: Medium (hidden test tag `[.multi-device]` exists in test_memory_reservation_manager.cpp as a pattern; similar tagging should be applied here).

**No Tests for `synchronized_data_batch` Move Under Concurrent Accessors:**
- What's not tested: Moving a `synchronized_data_batch` while another thread holds a `read_only_data_batch` on the same instance.
- Files: `src/data/data_batch.cpp:91-99`
- Risk: UB; difficult to catch without a thread sanitizer test.
- Priority: Medium.

**No Tests for Column Types Beyond Fixed-Width in Fast Converter:**
- What's not tested: The `plan_column_copy` / `collect_column_d2h_ops` / `reconstruct_column` functions with STRUCT, LIST, nested STRING columns, DICTIONARY32.
- Files: `src/data/representation_converter.cpp:434-717`
- Risk: Off-by-one in child column recursion or alignment for complex nested types would corrupt data silently.
- Priority: High.

---

*Concerns audit: 2026-04-13*
