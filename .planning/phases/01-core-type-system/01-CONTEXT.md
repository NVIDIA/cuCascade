# Phase 1: Core Type System - Context

**Gathered:** 2026-04-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement the complete data_batch type system as one atomic compilation unit: `data_batch`, `read_only_data_batch<PtrType>`, `mutable_data_batch<PtrType>`, all 8 state transition static methods, clone operations, and mutual friend relationships. This phase replaces the existing `synchronized_data_batch` nested-class design in `data_batch.hpp` and `data_batch.cpp`.

</domain>

<decisions>
## Implementation Decisions

### Class Design
- **D-01:** 3 flat classes, no nesting. `data_batch` is the "idle" state — owns data + mutex but exposes almost nothing publicly.
- **D-02:** No `enable_shared_from_this` — static methods receive PtrType as parameter, no need to obtain shared_ptr from inside the object.
- **D-03:** Public constructor on `data_batch` — no passkey idiom needed.
- **D-04:** Deleted move and copy operations on `data_batch` — fixes the known move-without-lock bug from PR #99 v1. The object is never moved; only the PtrType to it is moved.
- **D-05:** `const uint64_t _batch_id` — immutable after construction.

### Template Structure
- **D-06:** `read_only_data_batch<PtrType>` and `mutable_data_batch<PtrType>` are full class templates. Static methods on `data_batch` are also templated on PtrType.
- **D-07:** Template definitions live in the header (`data_batch.hpp`). Explicit template instantiations for `shared_ptr<data_batch>` and `unique_ptr<data_batch>` in `data_batch.cpp` — matches existing `idata_repository` pattern.
- **D-08:** All 3 classes in a single header (`data_batch.hpp`) due to mutual friend relationships.

### Accessor Methods
- **D-09:** Return types match current codebase: `get_data()` returns `idata_representation*`, `get_memory_space()` returns `memory_space*`, `get_current_tier()` returns `Tier` (value type). Raw pointers, not references.
- **D-10:** `mutable_data_batch` exposes full read+write: all methods from read_only plus `set_data(unique_ptr<idata_representation>)` and `convert_to<T>(registry, target_memory_space, stream)`.
- **D-11:** Accessor types store PtrType parent (first member) before lock guard (second member) — destruction order is load-bearing (lock releases before parent drops).
- **D-12:** Move-only semantics (deleted copy), `noexcept` on move operations.

### State Transitions
- **D-13:** All 6 blocking transitions + 2 try variants are static methods on `data_batch`, templated on PtrType.
- **D-14:** Blocking variants take PtrType by rvalue reference (`&&`) — move semantics invalidate source.
- **D-15:** Try variants take PtrType by mutable lvalue reference (`&`) — nullify on success, unchanged on failure.
- **D-16:** No direct locked-to-locked conversion. `to_mutable(read_only_data_batch&&)` and `to_read_only(mutable_data_batch&&)` go through idle internally (release lock, reacquire through parent).
- **D-17:** `[[nodiscard]]` on all transition methods.

### Clone Operations
- **D-18:** `clone()` and `clone_to<T>()` are methods on `read_only_data_batch` — NOT on `data_batch`. Avoids recursive shared_mutex deadlock since the caller already holds the shared lock.
- **D-19:** Both return PtrType (match the template parameter).
- **D-20:** `clone()` = deep copy, same representation type. `clone_to<T>()` = deep copy + representation conversion via converter registry.

### Public API on data_batch
- **D-21:** `get_batch_id()` — lock-free, public.
- **D-22:** `subscribe()`, `unsubscribe()`, `get_subscriber_count()` — atomic, public.
- **D-23:** No public access to data, tier, or memory space. These require acquiring a lock through an accessor.

### Friend Relationships
- **D-24:** Mutual friends: `data_batch` <-> `read_only_data_batch<PtrType>`, `data_batch` <-> `mutable_data_batch<PtrType>`.
- **D-25:** `read_only_data_batch` and `mutable_data_batch` do NOT need to know about each other — all conversions go through `data_batch`.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Current Implementation (to be replaced)
- `include/cucascade/data/data_batch.hpp` -- Current synchronized_data_batch design with nested inner class
- `src/data/data_batch.cpp` -- Current implementation (constructor, lock acquisition, clone, subscribers)

### Data Representation Interface
- `include/cucascade/data/common.hpp` -- `idata_representation` base class with `clone()` virtual, `get_size_in_bytes()`, `get_current_tier()`
- `include/cucascade/data/gpu_data_representation.hpp` -- `gpu_table_representation` concrete type
- `include/cucascade/data/cpu_data_representation.hpp` -- `host_data_representation`, `host_data_packed_representation`

### Converter Registry
- `include/cucascade/data/representation_converter.hpp` -- `representation_converter_registry` used by `convert_to<T>()` and `clone_to<T>()`

### Repository (consumers of data_batch type)
- `include/cucascade/data/data_repository.hpp` -- `idata_repository<PtrType>` template with explicit instantiations
- `src/data/data_repository.cpp` -- Explicit template instantiations for shared_ptr and unique_ptr
- `include/cucascade/data/data_repository_manager.hpp` -- Routes batches to repositories

### Memory Layer (referenced by accessor methods)
- `include/cucascade/memory/common.hpp` -- `Tier` enum, `memory_space` forward declaration

### Tests (will be rewritten in Phase 3, but show expected behavior)
- `test/data/test_data_batch.cpp` -- Current test patterns for synchronized_data_batch
- `test/utils/mock_test_utils.hpp` -- `mock_data_representation` used in tests

### Research
- `.planning/research/PITFALLS.md` -- Critical: member declaration order, clone deadlock, lock upgrade TOCTOU
- `.planning/research/FEATURES.md` -- Table stakes and anti-features list
- `.planning/research/ARCHITECTURE.md` -- Build order and component boundaries

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `mock_data_representation` in `test/utils/mock_test_utils.hpp` — lightweight test mock that takes Tier and size, used by all data_batch tests
- `make_mock_memory_space()` in same file — creates test memory_space instances
- `idata_representation::clone()` virtual — already implemented on all concrete types (deep copy)

### Established Patterns
- Explicit template instantiation: `idata_repository` uses `template class idata_repository<shared_ptr<...>>` in `.cpp` — follow same pattern for accessor templates
- RAII lock wrappers: `shared_lock` and `unique_lock` already used throughout codebase
- Deleted copy with defaulted move: common pattern in memory layer classes
- `[[maybe_unused]]` on stream parameters where not used

### Integration Points
- `data_batch.hpp` is included by `data_repository.hpp`, `data_repository_manager.hpp`, and all test files
- Type aliases `shared_data_repository` and `unique_data_repository` reference the batch type directly
- `representation_converter_registry` is passed to `convert_to<T>()` and `clone_to<T>()` — interface unchanged

</code_context>

<specifics>
## Specific Ideas

- The user explicitly wants the "state machine through move semantics" pattern where using a moved-from object is a compile error
- Lock conversion always goes through idle: `ro -> to_idle -> to_mutable`, never direct `ro -> mutable`
- The user considered `enable_shared_from_this` but rejected it after realizing static methods already receive PtrType as parameter
- The user wants PtrType agnosticism preserved even though unique_ptr limits to single reader — other engines may use it

</specifics>

<deferred>
## Deferred Ideas

- Repository `try_pop` API (iterates batches, tries `try_to_mutable` on each) — dhruv's suggestion, v2
- Probing/observability interface (`idata_batch_probe` replacement) — dhruv flagged as future need
- `get_data_as<T>()` convenience template on accessors — evaluate after refactor
- `withLock` callback pattern (folly-style) — evaluate based on call site patterns

</deferred>

---

*Phase: 01-core-type-system*
*Context gathered: 2026-04-14*
