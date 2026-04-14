# Phase 3: Test Migration and Build Validation - Research

**Researched:** 2026-04-14
**Domain:** C++20 test migration for RAII lock-accessor refactor (Catch2 v2, shared_mutex, move semantics)
**Confidence:** HIGH

## Summary

Phase 3 migrates three test files (`test_data_batch.cpp`, `test_data_repository.cpp`, `test_data_repository_manager.cpp`) from the old `synchronized_data_batch` API to the new 3-class `data_batch` / `read_only_data_batch<PtrType>` / `mutable_data_batch<PtrType>` system implemented in Phases 1 and 2. The old type `synchronized_data_batch` has been completely removed from headers and source files -- it exists only in these 3 test files (130 total references across the files). After this phase, `pixi run build` must compile all 61 targets and `pixi run test` must pass with zero failures.

The migration is fundamentally an API translation exercise, but `test_data_batch.cpp` requires a deep rewrite because the old API (instance methods like `batch.get_read_only()`, nested types like `synchronized_data_batch::read_only_data_batch`) has been replaced with static methods (`data_batch::to_read_only(std::move(ptr))`), free-standing template types, and move-ownership semantics. The repository test files (`test_data_repository.cpp`, `test_data_repository_manager.cpp`) are simpler -- they require only type name replacement (`synchronized_data_batch` to `data_batch`) since the repository layer stores idle (unlocked) batches and the batch creation API (`make_shared<data_batch>(id, data)` / `make_unique<data_batch>(id, data)`) is unchanged in shape.

**Primary recommendation:** Split the work into three parts: (1) full rewrite of `test_data_batch.cpp` exercising all new API paths including new tests for destruction order and try-variant semantics, (2) mechanical type replacement in the two repository test files, (3) full build + test validation. The `test_data_batch.cpp` rewrite is the critical path -- the other two files are straightforward find-and-replace.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TEST-01 | Rewrite `test_data_batch.cpp` for new 3-class API and static transition methods | Complete API mapping documented in Architecture Patterns; old-to-new translation table provided |
| TEST-02 | Test member destruction order (lock released before parent dropped) -- TSan/ASan safe | Pitfall 1/10 from PITFALLS.md; specific test pattern documented in Code Examples |
| TEST-03 | Test move semantics invalidation (compile-time safety of moved-from objects) | Cannot test at runtime (moved-from shared_ptr is null by spec); verify via construction patterns |
| TEST-04 | Test try variants (success nullifies source, failure leaves unchanged) | New API `try_to_read_only(PtrType&)` / `try_to_mutable(PtrType&)` patterns documented |
| TEST-05 | Test clone via read_only_data_batch (no recursive lock) | Clone is now a method on `read_only_data_batch<PtrType>`, not on `data_batch` |
| TEST-06 | Update `test_data_repository.cpp` for new data_batch type | 28 `synchronized_data_batch` references to replace + helper function signature |
| TEST-07 | Update `test_data_repository_manager.cpp` for new data_batch type | 21 `synchronized_data_batch` references to replace |
| TEST-08 | Test concurrent readers (shared_ptr path) and serialized writers | Existing tests adapt with new static method API; patterns documented |
| TEST-09 | Build passes cleanly (`pixi run build`, all targets) | Build command: `pixi run build`; validates all 61 targets |
| TEST-10 | All tests pass (`pixi run test`) | Test command: `pixi run test`; validates CTest runner |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **C++ standard**: C++20, must compile with CUDA 12.9/13.x toolchains
- **Build validation**: Must pass `pixi run build` (all 61 targets) and `pixi run test`
- **Code style**: clang-format v20.1.4 (BasedOnStyle: Google with overrides), 100-char column limit, 2-space indent
- **Pre-commit**: `.pre-commit-config.yaml` enforces clang-format, cmake-lint, codespell
- **Test framework**: Catch2 v2.13.10, fetched via CMake FetchContent
- **Test naming**: `test_<subject>.cpp`
- **Verification loop**: Build and run tests after implementation
- **Naming**: snake_case for classes, functions; `_` prefix for private members
- **Error handling**: Destructors noexcept; throw `std::runtime_error` for precondition failures
- **Documentation**: Doxygen `/** */` for public API; `///< inline` for members
- **GSD workflow**: Use GSD commands for edits

## Architecture Patterns

### Old API vs New API Translation Table

This is the core reference for the test migration. Every test in `test_data_batch.cpp` uses old API patterns that must be mechanically translated to the new static-method API.

| Old Pattern (synchronized_data_batch) | New Pattern (data_batch) | Notes |
|---------------------------------------|--------------------------|-------|
| `synchronized_data_batch batch(id, data)` | `auto batch = std::make_shared<data_batch>(id, std::move(data))` | Must use smart pointer -- `data_batch` is non-copyable, non-movable |
| `batch.get_read_only()` | `data_batch::to_read_only(std::move(batch))` | Static method, moves ownership |
| `batch.get_mutable()` | `data_batch::to_mutable(std::move(batch))` | Static method, moves ownership |
| `batch.try_get_read_only()` | `data_batch::try_to_read_only(batch)` | Takes lvalue ref, nullifies on success |
| `batch.try_get_mutable()` | `data_batch::try_to_mutable(batch)` | Takes lvalue ref, nullifies on success |
| `ro->get_batch_id()` | `ro.get_batch_id()` | Accessor is value type, not pointer |
| `ro->get_current_tier()` | `ro.get_current_tier()` | Direct member access |
| `ro->get_data()` | `ro.get_data()` | Direct member access |
| `rw->set_data(...)` | `rw.set_data(...)` | Direct member access |
| `synchronized_data_batch::read_only_data_batch::from_mutable(rw)` | `data_batch::to_idle(std::move(rw))` then `data_batch::to_read_only(std::move(idle))` | No direct locked-to-locked; goes through idle |
| `synchronized_data_batch::mutable_data_batch::from_read_only(ro)` | `data_batch::to_idle(std::move(ro))` then `data_batch::to_mutable(std::move(idle))` | No direct locked-to-locked; goes through idle |
| `batch->clone(new_id, stream)` | `ro.clone(new_id, stream)` | Clone is now on `read_only_data_batch`, not `data_batch` |
| `batch.get_batch_id()` | `batch->get_batch_id()` | Through smart pointer now |
| `batch.subscribe()` | `batch->subscribe()` | Through smart pointer now |
| Scope-based lock release (let accessor go out of scope) | `data_batch::to_idle(std::move(accessor))` or let accessor destruct | Explicit `to_idle` returns the PtrType back |
| Move constructor `synchronized_data_batch(std::move(other))` | DELETED -- `data_batch` is non-movable | Old tests for move constructor/assignment must be REMOVED |

### Key Semantic Differences

1. **Ownership model change**: Old API used stack-allocated `synchronized_data_batch` objects. New API requires smart-pointer-wrapped `data_batch` objects. Every test that creates a `synchronized_data_batch` on the stack must be converted to `make_shared<data_batch>` or `make_unique<data_batch>`.

2. **Accessor types are now templates**: Old `synchronized_data_batch::read_only_data_batch` is now `read_only_data_batch<std::shared_ptr<data_batch>>` (or `unique_ptr` variant). In tests, `auto` hides this but the type must be correct for explicit type annotations.

3. **Static methods consume ownership**: `to_read_only(std::move(batch))` moves the smart pointer INTO the accessor. The original variable is null after the call. To get the batch back, you must call `to_idle(std::move(accessor))`. This is fundamentally different from the old API where the batch remained accessible.

4. **Multiple simultaneous read-only accessors**: In the old API, `batch.get_read_only()` could be called multiple times on the same batch. In the new API with `shared_ptr`, `data_batch::to_read_only(std::move(batch))` consumes the pointer. To get multiple readers, you need multiple `shared_ptr` copies: `auto batch2 = batch; auto ro1 = data_batch::to_read_only(std::move(batch)); auto ro2 = data_batch::to_read_only(std::move(batch2));`. Or: first clone the `shared_ptr`, then move each copy.

5. **No direct locked-to-locked conversion**: The old `from_mutable(rw)` / `from_read_only(ro)` methods are gone. All conversions go through idle. Tests that tested direct conversion must be rewritten as two-step operations.

### Test File Categorization

| File | Lines | `synchronized_data_batch` refs | Migration Type | Effort |
|------|-------|-------------------------------|----------------|--------|
| `test_data_batch.cpp` | 726 | 81 | **Full rewrite** -- API semantics changed | HIGH |
| `test_data_repository.cpp` | 928 | 28 | **Type replacement** -- `synchronized_data_batch` -> `data_batch` in constructor calls + helper function | LOW |
| `test_data_repository_manager.cpp` | ~1140 | 21 | **Type replacement** -- same as above | LOW |

### Tests to DELETE (no new API equivalent)

These old tests must be removed because the new `data_batch` is non-copyable and non-movable:

- `synchronized_data_batch Move Constructor` (line 53) -- `data_batch` deletes move ops
- `synchronized_data_batch Move Assignment` (line 63) -- `data_batch` deletes move ops

### Tests to REWRITE (API semantics changed)

Every remaining test in `test_data_batch.cpp` needs rewriting because:
- Constructor changes from stack allocation to smart pointer
- Accessor acquisition changes from instance method to static method with move semantics
- Accessor access changes from `->` (pointer) to `.` (value type)
- Clone changes from `data_batch` method to `read_only_data_batch` method
- Lock conversion changes from direct to two-step through idle

### Tests to ADD (new behaviors to verify)

| New Test | Requirement | What It Verifies |
|----------|-------------|------------------|
| Construction via shared_ptr and unique_ptr | TEST-01 | Both PtrType variants work |
| All 6 blocking transitions | TEST-01 | `to_read_only`, `to_mutable`, 2x `to_idle`, `to_idle(ro) -> to_mutable`, `to_idle(rw) -> to_read_only` |
| `try_to_read_only` success nullifies source | TEST-04 | `batch` is null after success |
| `try_to_read_only` failure leaves source unchanged | TEST-04 | `batch` is non-null after failure |
| `try_to_mutable` success nullifies source | TEST-04 | `batch` is null after success |
| `try_to_mutable` failure leaves source unchanged | TEST-04 | `batch` is non-null after failure |
| Destruction order safety (last shared_ptr in accessor) | TEST-02 | Create accessor, drop all other refs, destroy accessor -- no crash |
| Clone via read_only_data_batch (not data_batch) | TEST-05 | `ro.clone(id, stream)` creates independent copy |
| Concurrent readers (shared_ptr path with copies) | TEST-08 | Multiple `read_only_data_batch` from different `shared_ptr` copies |
| Serialized exclusive writers | TEST-08 | `atomic<int>` concurrency counter proves no overlap |

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Test mocks for data representations | Custom mock | `cucascade::test::mock_data_representation` | Already exists in `test/utils/mock_test_utils.hpp`, covers GPU/HOST/DISK tiers |
| Memory space construction for tests | Manual memory_space setup | `cucascade::test::make_mock_memory_space()` | Already exists, handles all three tier configs |
| cuDF table creation for GPU tests | Manual cuDF column construction | `cucascade::test::create_simple_cudf_table()` | Already exists with overloads for row count, column count, allocator, stream |
| cuDF table comparison | Byte-by-byte GPU comparison | `cucascade::test::expect_cudf_tables_equal_on_stream()` | Already exists, handles column-wise comparison with hex diff output |
| Thread synchronization in tests | Custom barriers | `std::thread` + `join()` + `std::atomic` | Established pattern throughout existing tests |

**Key insight:** All test utilities are already available. The migration is about API translation, not creating new test infrastructure.

## Common Pitfalls

### Pitfall 1: Multiple Readers Require Multiple shared_ptr Copies

**What goes wrong:** The old API allowed `batch.get_read_only()` multiple times on the same object. The new API moves ownership: `data_batch::to_read_only(std::move(batch))` nullifies `batch`. Attempting to create a second reader from the now-null `batch` causes a null dereference.

**Why it happens:** Static methods take `PtrType&&` (rvalue reference) for blocking variants, consuming the source.

**How to avoid:** For tests that need multiple simultaneous readers, copy the `shared_ptr` first:
```cpp
auto batch = std::make_shared<data_batch>(1, std::move(data));
auto batch2 = batch;  // copy the shared_ptr
auto batch3 = batch;  // another copy
auto ro1 = data_batch::to_read_only(std::move(batch));
auto ro2 = data_batch::to_read_only(std::move(batch2));
auto ro3 = data_batch::to_read_only(std::move(batch3));
```

**Warning signs:** Segfault or null dereference in tests that had multiple `get_read_only()` calls.

### Pitfall 2: Accessor Access Syntax Changed from Pointer to Value

**What goes wrong:** The old accessors were returned by value but accessed via `->` (they had `operator->`). The new accessors are accessed via `.` (direct member functions). Using `->` on the new types will fail to compile.

**Why it happens:** The new `read_only_data_batch<PtrType>` is a value type with named methods, not a smart-pointer-like wrapper.

**How to avoid:** Systematic replacement: `ro->get_batch_id()` becomes `ro.get_batch_id()`, `rw->set_data(...)` becomes `rw.set_data(...)`.

**Warning signs:** Compile errors about "no operator-> defined."

### Pitfall 3: Locked-to-Locked Conversion Now Requires Two Steps

**What goes wrong:** The old API had `read_only_data_batch::from_mutable()` and `mutable_data_batch::from_read_only()`. These are gone. Tests that use these must be rewritten as two-step operations.

**Why it happens:** The new design routes all conversions through idle to avoid the TOCTOU issue being hidden in the API.

**How to avoid:** Replace `synchronized_data_batch::read_only_data_batch::from_mutable(std::move(rw))` with:
```cpp
auto idle = data_batch::to_idle(std::move(rw));
auto ro = data_batch::to_read_only(std::move(idle));
```

**Warning signs:** Compile errors about missing `from_mutable` / `from_read_only` methods.

### Pitfall 4: try Variants Use Lvalue Reference, Not Rvalue

**What goes wrong:** Blocking variants take `PtrType&&` (rvalue, moves). Try variants take `PtrType&` (lvalue, conditionally nullifies). Mixing up the signatures causes compile errors.

**Why it happens:** Try variants must leave the source unchanged on failure, which requires lvalue semantics.

**How to avoid:** Use `data_batch::try_to_read_only(batch)` (no `std::move`), not `data_batch::try_to_read_only(std::move(batch))`.

**Warning signs:** Compile error about rvalue to lvalue reference binding.

### Pitfall 5: Clone Tests Must Acquire read_only First

**What goes wrong:** The old API had `batch->clone(new_id, stream)` directly on `synchronized_data_batch`. The new API has `clone()` only on `read_only_data_batch`. Tests that call clone on a bare `data_batch` pointer will fail to compile.

**Why it happens:** Clone was moved to `read_only_data_batch` to avoid the recursive shared_mutex deadlock (Pitfall 5 from PITFALLS.md).

**How to avoid:** First acquire a read-only accessor, then clone:
```cpp
auto ro = data_batch::to_read_only(std::move(batch));
auto cloned = ro.clone(new_id, stream);
```

**Warning signs:** Compile error about `clone` not being a member of `data_batch`.

### Pitfall 6: Thread Tests Need shared_ptr Copies for Concurrent Access

**What goes wrong:** Multi-threaded tests that pass a reference to `batch` into thread lambdas and call `batch.get_read_only()` won't work because: (a) `data_batch` is accessed through a smart pointer not directly, and (b) `to_read_only` moves ownership.

**Why it happens:** Each thread needs its own `shared_ptr` copy to independently acquire locks.

**How to avoid:** For concurrent reader tests, give each thread its own `shared_ptr` copy. For tests that pass the raw `data_batch*`, use `batch.get()` for the pointer but `shared_ptr` copies for lock acquisition.

## Code Examples

### Example 1: Basic Construction and Read-Only Access

```cpp
// Old:
auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
synchronized_data_batch batch(1, std::move(data));
auto ro = batch.get_read_only();
REQUIRE(ro->get_batch_id() == 1);

// New:
auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
auto batch = std::make_shared<data_batch>(1, std::move(data));
auto ro = data_batch::to_read_only(std::move(batch));
REQUIRE(ro.get_batch_id() == 1);
```
[VERIFIED: read from include/cucascade/data/data_batch.hpp lines 135-137, 273]

### Example 2: Mutable Access with set_data

```cpp
// Old:
auto rw = batch.get_mutable();
rw->set_data(std::make_unique<mock_data_representation>(memory::Tier::HOST, 2048));
// rw goes out of scope, releases lock

// New:
auto rw = data_batch::to_mutable(std::move(batch));
rw.set_data(std::make_unique<mock_data_representation>(memory::Tier::HOST, 2048));
auto batch = data_batch::to_idle(std::move(rw));  // explicitly get batch back
```
[VERIFIED: read from include/cucascade/data/data_batch.hpp lines 148-149, 377]

### Example 3: Multiple Concurrent Readers (shared_ptr path)

```cpp
// Old:
auto ro1 = batch.get_read_only();
auto ro2 = batch.get_read_only();
auto ro3 = batch.get_read_only();

// New (need shared_ptr copies):
auto batch = std::make_shared<data_batch>(1, std::move(data));
auto b2 = batch;
auto b3 = batch;
auto ro1 = data_batch::to_read_only(std::move(batch));
auto ro2 = data_batch::to_read_only(std::move(b2));
auto ro3 = data_batch::to_read_only(std::move(b3));
REQUIRE(ro1.get_batch_id() == 1);
REQUIRE(ro2.get_batch_id() == 1);
REQUIRE(ro3.get_batch_id() == 1);
```
[VERIFIED: read from include/cucascade/data/data_batch.hpp lines 437-443]

### Example 4: Try Variant (success path)

```cpp
auto batch = std::make_shared<data_batch>(1, std::move(data));
auto result = data_batch::try_to_read_only(batch);
REQUIRE(result.has_value());
REQUIRE(batch == nullptr);           // source nullified on success
REQUIRE(result->get_batch_id() == 1);
```
[VERIFIED: read from include/cucascade/data/data_batch.hpp lines 478-484]

### Example 5: Try Variant (failure path -- mutable lock held by another thread)

```cpp
auto batch = std::make_shared<data_batch>(1, std::move(data));
auto batch_copy = batch;
auto rw = data_batch::to_mutable(std::move(batch));

std::atomic<bool> got_lock{false};
std::thread t([&batch_copy, &got_lock]() {
  auto result = data_batch::try_to_read_only(batch_copy);
  got_lock.store(result.has_value());
  // batch_copy unchanged on failure
});
t.join();
REQUIRE(got_lock.load() == false);
```
[VERIFIED: read from include/cucascade/data/data_batch.hpp lines 478-484]

### Example 6: Destruction Order Safety Test (TEST-02)

```cpp
// Verifies member declaration order: PtrType before lock guard.
// If wrong, TSan/ASan will flag use-after-destroy on the mutex.
TEST_CASE("data_batch destruction order safety", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  // Create accessor -- this is now the ONLY shared_ptr holding the batch alive.
  auto ro = data_batch::to_read_only(std::move(batch));
  // batch is null now. The only reference to the data_batch is inside ro._batch.

  // When ro is destroyed, destruction order is:
  //   1. _lock (shared_lock) releases the shared lock
  //   2. _batch (shared_ptr) drops the last reference, destroys data_batch + mutex
  // If the order were reversed, the mutex would be destroyed before the lock releases.

  // This should NOT crash under TSan/ASan:
  // (ro goes out of scope here)
}
```
[VERIFIED: read from include/cucascade/data/data_batch.hpp lines 335-339 -- member order is PtrType first, lock second]

### Example 7: Locked-to-Locked Conversion (through idle)

```cpp
// Old:
auto rw = batch.get_mutable();
auto ro = synchronized_data_batch::read_only_data_batch::from_mutable(std::move(rw));

// New (two-step through idle):
auto rw = data_batch::to_mutable(std::move(batch));
auto idle = data_batch::to_idle(std::move(rw));     // release exclusive lock
auto ro = data_batch::to_read_only(std::move(idle)); // acquire shared lock
```
[VERIFIED: read from include/cucascade/data/data_batch.hpp lines 458-473]

### Example 8: Clone via read_only_data_batch

```cpp
// Old:
auto batch = std::make_shared<synchronized_data_batch>(1, std::move(data));
auto cloned = batch->clone(100, rmm::cuda_stream_view{});

// New:
auto batch = std::make_shared<data_batch>(1, std::move(data));
auto ro = data_batch::to_read_only(std::move(batch));
auto cloned = ro.clone(100, rmm::cuda_stream_view{});
// cloned is std::shared_ptr<data_batch>
```
[VERIFIED: read from include/cucascade/data/data_batch.hpp lines 498-510]

### Example 9: Repository Test Type Replacement

```cpp
// Old:
auto batch = std::make_shared<synchronized_data_batch>(1, std::move(data));
repository.add_data_batch(batch);

// New:
auto batch = std::make_shared<data_batch>(1, std::move(data));
repository.add_data_batch(batch);

// Old (unique_ptr):
auto batch = std::make_unique<synchronized_data_batch>(1, std::move(data));
repository.add_data_batch(std::move(batch));

// New (unique_ptr):
auto batch = std::make_unique<data_batch>(1, std::move(data));
repository.add_data_batch(std::move(batch));
```
[VERIFIED: read from include/cucascade/data/data_repository.hpp lines 282-283 -- type aliases now use `data_batch`]

### Example 10: Helper Function in test_data_repository.cpp

```cpp
// Old (line 773-781):
std::vector<std::shared_ptr<synchronized_data_batch>> create_test_batches(
    std::vector<uint64_t> batch_ids)
{
  std::vector<std::shared_ptr<synchronized_data_batch>> batches;
  for (auto batch_id : batch_ids) {
    auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    batches.emplace_back(std::make_shared<synchronized_data_batch>(batch_id, std::move(data)));
  }
  return batches;
}

// New:
std::vector<std::shared_ptr<data_batch>> create_test_batches(
    std::vector<uint64_t> batch_ids)
{
  std::vector<std::shared_ptr<data_batch>> batches;
  for (auto batch_id : batch_ids) {
    auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    batches.emplace_back(std::make_shared<data_batch>(batch_id, std::move(data)));
  }
  return batches;
}
```
[VERIFIED: read from test/data/test_data_repository.cpp lines 773-781]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `synchronized_data_batch` with nested accessor types | 3 flat classes: `data_batch`, `read_only_data_batch<PtrType>`, `mutable_data_batch<PtrType>` | Phase 1 (this PR) | All tests must use new types |
| Instance methods for lock acquisition (`batch.get_read_only()`) | Static methods with move semantics (`data_batch::to_read_only(std::move(ptr))`) | Phase 1 | Fundamentally different calling convention |
| Clone on `synchronized_data_batch` | Clone on `read_only_data_batch` | Phase 1 | Avoids recursive shared_mutex deadlock |
| `enable_shared_from_this` + factory | Public constructor, no `enable_shared_from_this` | Phase 1 | Direct `make_shared<data_batch>(...)` construction |
| Direct locked-to-locked conversion (`from_mutable`, `from_read_only`) | Two-step through idle (`to_idle` then re-acquire) | Phase 1 | Makes TOCTOU window explicit |
| SFINAE dispatch in repository manager | `if constexpr` dispatch | Phase 2 | No test impact |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `pixi run build` and `pixi run test` are the correct build/test commands | Phase Requirements | Build/test validation would use wrong commands |
| A2 | TSan/ASan can be run via the standard test executable (no special build flags needed beyond what pixi provides) | Common Pitfalls | Destruction order test may not detect issues without sanitizers |

**Note on A2:** The phase success criteria say "TSan/ASan clean" but the standard `pixi run test` may not run with sanitizers enabled. The tests should be *correct* under sanitizers, meaning the code has no UB that sanitizers would flag. Whether sanitizers are actually run is a CI/build configuration concern outside this phase's scope.

## Open Questions

1. **TSan/ASan build configuration**
   - What we know: Tests must be "TSan/ASan clean" per TEST-02. The code must be correct (no UB).
   - What's unclear: Whether the standard `pixi run test` runs with sanitizers, or if a separate build configuration is needed.
   - Recommendation: Write the tests to be correct under sanitizers. If the standard build does not run sanitizers, note this as a follow-up. The destruction order test (Example 6) will catch the bug with or without sanitizers if the member order is wrong.

2. **TEST-03 compile-time safety verification**
   - What we know: Move semantics make moved-from `shared_ptr` null at runtime. The "compile-time safety" is that you cannot accidentally use the old variable because it has been moved.
   - What's unclear: Whether the requirement expects `static_assert` or compile-error tests, or just runtime verification that moved-from pointers are null.
   - Recommendation: Test at runtime that `batch == nullptr` after `to_read_only(std::move(batch))`. Compile-time enforcement is inherent in the API design (you cannot pass a moved-from pointer to another function because it is null). Add a comment documenting this.

## Sources

### Primary (HIGH confidence)
- `include/cucascade/data/data_batch.hpp` -- Complete new API (547 lines, all transition methods, accessor types, clone operations)
- `src/data/data_batch.cpp` -- Non-template implementations + explicit template instantiations
- `include/cucascade/data/data_repository.hpp` -- Updated type aliases (line 282-283: `shared_data_repository`, `unique_data_repository`)
- `include/cucascade/data/data_repository_manager.hpp` -- Updated type aliases (line 272-273)
- `test/data/test_data_batch.cpp` -- Current test file (726 lines, 81 `synchronized_data_batch` references)
- `test/data/test_data_repository.cpp` -- Current test file (928 lines, 28 references)
- `test/data/test_data_repository_manager.cpp` -- Current test file (~1140 lines, 21 references)
- `test/utils/mock_test_utils.hpp` -- Test utilities (`mock_data_representation`, `make_mock_memory_space`, `create_simple_cudf_table`)
- `.planning/research/PITFALLS.md` -- Domain pitfalls (15 pitfalls, Pitfalls 1/5/10/11 directly relevant to testing)
- `.planning/phases/01-core-type-system/01-CONTEXT.md` -- Phase 1 design decisions (D-01 through D-25)
- `.planning/phases/02-repository-integration/02-CONTEXT.md` -- Phase 2 design decisions (D-01 through D-04)

### Secondary (MEDIUM confidence)
- `test/CMakeLists.txt` -- Build configuration for test executable
- `.planning/REQUIREMENTS.md` -- Full requirements list (TEST-01 through TEST-10)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- Catch2 v2.13.10 is already in use, no new dependencies
- Architecture: HIGH -- Complete API mapping derived from reading actual implementation
- Pitfalls: HIGH -- Derived from actual API differences confirmed by reading header
- Code examples: HIGH -- All examples verified against implementation source

**Research date:** 2026-04-14
**Valid until:** Indefinite (codebase-specific, tied to this PR)
