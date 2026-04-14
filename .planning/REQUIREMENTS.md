# Requirements: cuCascade data_batch Refactor

**Defined:** 2026-04-14
**Core Value:** Compile-time enforced data access safety — impossible to read or mutate batch data without holding the appropriate lock, with move semantics making stale references a compile error.

## v1 Requirements

### Core API

- [ ] **CORE-01**: 3 flat classes: `data_batch`, `read_only_data_batch`, `mutable_data_batch` — no `synchronized_data_batch` wrapper
- [ ] **CORE-02**: `data_batch` has no public access to data, tier, or memory space — only way to access data is through an accessor type
- [ ] **CORE-03**: Public constructor on `data_batch` (no enable_shared_from_this, no passkey idiom needed)
- [ ] **CORE-04**: `const uint64_t _batch_id` — immutable after construction
- [ ] **CORE-05**: Lock-free `get_batch_id()` public on `data_batch`
- [ ] **CORE-06**: Atomic subscriber count on `data_batch` — `subscribe()`, `unsubscribe()`, `get_subscriber_count()`
- [ ] **CORE-07**: Deleted move and copy operations on `data_batch` (fixes known move-without-lock bug)

### Accessor Types

- [ ] **ACC-01**: `read_only_data_batch` with named methods: `get_batch_id()`, `get_current_tier()`, `get_data()`, `get_memory_space()`
- [ ] **ACC-02**: `mutable_data_batch` with all read methods + `set_data()`, `convert_to<T>()`
- [ ] **ACC-03**: PtrType-agnostic accessors — templated on PtrType (`shared_ptr` or `unique_ptr`)
- [ ] **ACC-04**: Accessors store PtrType parent for lifetime management
- [ ] **ACC-05**: Move-only semantics on both accessor types (deleted copy)
- [ ] **ACC-06**: `noexcept` on move operations
- [ ] **ACC-07**: Correct member declaration order: PtrType parent before lock guard (lock releases before parent drops)
- [ ] **ACC-08**: `[[nodiscard]]` on all accessor-returning methods

### State Transitions

- [ ] **TRANS-01**: `data_batch::to_read_only(PtrType&&)` — idle to shared lock (blocking)
- [ ] **TRANS-02**: `data_batch::to_mutable(PtrType&&)` — idle to exclusive lock (blocking)
- [ ] **TRANS-03**: `data_batch::to_idle(read_only_data_batch&&)` — release shared lock, return PtrType
- [ ] **TRANS-04**: `data_batch::to_idle(mutable_data_batch&&)` — release exclusive lock, return PtrType
- [ ] **TRANS-05**: `data_batch::to_mutable(read_only_data_batch&&)` — release shared, acquire exclusive (through idle internally)
- [ ] **TRANS-06**: `data_batch::to_read_only(mutable_data_batch&&)` — release exclusive, acquire shared (through idle internally)
- [ ] **TRANS-07**: `data_batch::try_to_read_only(PtrType&)` — non-blocking, nullifies source on success, unchanged on failure
- [ ] **TRANS-08**: `data_batch::try_to_mutable(PtrType&)` — non-blocking, nullifies source on success, unchanged on failure
- [ ] **TRANS-09**: All transitions use move semantics — moved-from objects are compile-time invalid
- [ ] **TRANS-10**: `[[nodiscard]]` on all transition methods

### Clone Operations

- [ ] **CLONE-01**: `clone()` as method on `read_only_data_batch` (not internally-locking on data_batch — avoids shared_mutex deadlock)
- [ ] **CLONE-02**: `clone_to<T>()` as method on `read_only_data_batch`

### Repository Integration

- [ ] **REPO-01**: `idata_repository<PtrType>` stays templated — supports both `shared_ptr<data_batch>` and `unique_ptr<data_batch>`
- [ ] **REPO-02**: Update type aliases: `shared_data_repository` and `unique_data_repository` use new `data_batch` (not `synchronized_data_batch`)
- [ ] **REPO-03**: Update `data_repository_manager` to use new `data_batch`
- [ ] **REPO-04**: Mutual friend relationships: `data_batch` <-> `read_only_data_batch`, `data_batch` <-> `mutable_data_batch`

### Tests

- [ ] **TEST-01**: Rewrite `test_data_batch.cpp` for new 3-class API and static transition methods
- [ ] **TEST-02**: Test member destruction order (lock released before parent dropped) — TSan/ASan safe
- [ ] **TEST-03**: Test move semantics invalidation (compile-time safety of moved-from objects)
- [ ] **TEST-04**: Test try variants (success nullifies source, failure leaves unchanged)
- [ ] **TEST-05**: Test clone via read_only_data_batch (no recursive lock)
- [ ] **TEST-06**: Update `test_data_repository.cpp` for new data_batch type
- [ ] **TEST-07**: Update `test_data_repository_manager.cpp` for new data_batch type
- [ ] **TEST-08**: Test concurrent readers (shared_ptr path) and serialized writers
- [ ] **TEST-09**: Build passes cleanly (`pixi run build`, all targets)
- [ ] **TEST-10**: All tests pass (`pixi run test`)

## v2 Requirements

### Repository Enhancements

- **REPO-V2-01**: `try_pop` on repository — iterates batches, tries `try_to_mutable` on each, returns first success
- **REPO-V2-02**: Predicate-based pop — `pop_if(predicate)` for filtered retrieval

### Observability

- **OBS-V2-01**: Probing interface for data_batch state inspection (replacement for removed `idata_batch_probe`)
- **OBS-V2-02**: Lock contention metrics (optional diagnostic counters)

### Convenience API

- **CONV-V2-01**: `get_data_as<T>()` convenience template on accessors for downcasting data representations

## Out of Scope

| Feature | Reason |
|---------|--------|
| `enable_shared_from_this` | Static methods receive PtrType as parameter — no need to obtain shared_ptr from inside the object |
| Passkey idiom / private constructor | Not needed without `enable_shared_from_this` — public constructor is safe |
| Try variants for locked-to-locked conversions | Complex return semantics (variant/result type), not needed yet |
| Direct read_only <-> mutable conversion | All conversions route through `data_batch` statics via idle state |
| Timed lock variants (try_for, try_until) | Unnecessary complexity for this use case |
| operator-> / operator* on accessors | data_batch has no public data API — named methods on accessors are the interface |
| Recursive locking support | `std::shared_mutex` does not support it; anti-pattern |
| Default-constructible accessors | Meaningless without a lock — prevents accidental misuse |
| Memory layer changes | Scoped to data layer only |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| | | |

**Coverage:**
- v1 requirements: 33 total
- Mapped to phases: 0
- Unmapped: 33 (pending roadmap creation)

---
*Requirements defined: 2026-04-14*
*Last updated: 2026-04-14 after initial definition*
