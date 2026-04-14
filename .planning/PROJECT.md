# cuCascade data_batch Refactor (PR #99 v2)

## What This Is

A redesign of cuCascade's `data_batch` concurrency model, addressing review feedback from @dhruv9vats on PR #99. Replaces the current `synchronized_data_batch` nested-class wrapper with a simpler 3-class design where `data_batch` is the "idle" state and all data access requires acquiring a lock through RAII accessor types. Targets cuCascade's data layer (`include/cucascade/data/`, `src/data/`).

## Core Value

Compile-time enforced data access safety: it must be impossible to read or mutate batch data without holding the appropriate lock, and move semantics must make stale references a compile error.

## Requirements

### Validated

- data batch representation system (idata_representation hierarchy) -- existing
- data batch repository with partitioned, thread-safe storage -- existing
- representation converter registry with type-keyed dispatch -- existing
- data repository manager for multi-port batch routing -- existing
- clone/clone_to operations for batch duplication -- existing
- subscriber count tracking for interest management -- existing
- batch ID system for repository lookups -- existing

### Active

- [ ] **DB-01**: 3 flat classes: `data_batch`, `read_only_data_batch`, `mutable_data_batch` (no `synchronized_data_batch` wrapper)
- [ ] **DB-02**: `data_batch` has minimal public API — no public access to data, tier, or memory space. Only way to access data is through an accessor type
- [ ] **DB-03**: PtrType-agnostic accessors — templated on PtrType (shared_ptr or unique_ptr), store PtrType parent for lifetime management
- [ ] **DB-04**: All state transitions through `data_batch` static methods with move semantics — moved-from objects are compile-time invalid
- [ ] **DB-05**: 6 blocking static conversion methods + 2 try variants on `data_batch`:
  - `to_read_only(PtrType&&)` — idle to shared lock
  - `to_mutable(PtrType&&)` — idle to exclusive lock
  - `to_idle(read_only_data_batch&&)` — release shared lock, return PtrType
  - `to_idle(mutable_data_batch&&)` — release exclusive lock, return PtrType
  - `to_mutable(read_only_data_batch&&)` — release shared, acquire exclusive (through idle)
  - `to_read_only(mutable_data_batch&&)` — release exclusive, acquire shared (through idle)
  - `try_to_read_only(PtrType&)` — non-blocking, nullifies on success, unchanged on failure
  - `try_to_mutable(PtrType&)` — non-blocking, nullifies on success, unchanged on failure
- [ ] **DB-06**: Public constructor on `data_batch` — no enable_shared_from_this needed (static methods receive PtrType as parameter)
- [ ] **DB-07**: `get_batch_id()` public and lock-free on `data_batch` — needed for repository lookups
- [ ] **DB-08**: `const uint64_t _batch_id` — immutable after construction
- [ ] **DB-09**: Atomic subscriber count (`subscribe()`, `unsubscribe()`, `get_subscriber_count()`) on `data_batch` — independent of lock state
- [ ] **DB-10**: Deleted move and copy operations on `data_batch` (fixes known move-without-lock bug)
- [ ] **DB-11**: `read_only_data_batch` exposes: `get_batch_id()`, `get_current_tier()`, `get_data()`, `get_memory_space()`
- [ ] **DB-12**: `mutable_data_batch` exposes: everything read_only has + `set_data()`, `convert_to<T>()`
- [ ] **DB-13**: `clone()` and `clone_to<T>()` on `read_only_data_batch` (not internally-locking on data_batch — avoids shared_mutex deadlock)
- [ ] **DB-14**: `idata_repository<PtrType>` stays templated — supports both shared_ptr and unique_ptr to data_batch
- [ ] **DB-15**: Update all tests to exercise new 3-class API, state transitions, try variants, and compile-time safety
- [ ] **DB-16**: Mutual friend relationships: `data_batch` <-> `read_only_data_batch`, `data_batch` <-> `mutable_data_batch`
- [ ] **DB-17**: `[[nodiscard]]` on all transition and accessor-returning methods

### Out of Scope

- `enable_shared_from_this` — static methods receive PtrType as parameter, no need to obtain shared_ptr from inside the object
- Probing/observability interface (`idata_batch_probe`) — dhruv flagged as future need, not blocking this refactor
- Try variants for locked-to-locked conversions — complex return semantics (need variant or result type), not needed yet
- Direct read_only <-> mutable conversion without going through idle — all conversions route through `data_batch` statics
- Repository `try_pop` API — dhruv suggested this, can be added as follow-up using `try_to_mutable`
- Memory layer changes — this refactor is scoped to data layer only

## Context

**PR #99 review feedback from @dhruv9vats:**
1. Wants `enable_shared_from_this` so accessors extend parent lifetime (not needed — static methods receive PtrType as parameter)
2. Wants `shared_ptr` only, not PtrType-agnostic (kept PtrType agnostic — unique_ptr gets single reader, shared_ptr gets concurrent readers)
3. Wants non-blocking try variants for lock acquisition (adopted for idle->locked)
4. Wants `const` batch_id (adopted)
5. Wants docstring for subscriber_count use case (will address)
6. Notes `idata_batch_probe` needed for future observability (deferred)
7. Suggests repository `try_pop` pattern (deferred as follow-up)

**Known bug in current PR:** Move constructor/assignment of `synchronized_data_batch` doesn't lock the mutex on the source. New design deletes move/copy on `data_batch` entirely.

**Existing codebase state:** cuCascade is a C++20 GPU-accelerated data caching library using libcudf, RMM, CUDA. Build system is CMake + pixi. Tests use Catch2 v2.

## Constraints

- **C++ standard**: C++20 — must compile with CUDA 12.9/13.x toolchains
- **API boundary**: Changes are breaking — `data_batch` replaces `synchronized_data_batch` throughout
- **Thread safety**: All public API must be safe for concurrent access from multiple threads
- **Build**: Must pass `pixi run build` (all 61 targets) and `pixi run test` (all tests)
- **Backward compatibility**: None required — this is a breaking API change (PR already tagged `feat!`)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 3 flat classes instead of nested | Simpler mental model, no unnecessary indirection | -- Pending |
| No enable_shared_from_this | Static methods receive PtrType as param — no need to get shared_ptr from inside object | -- Pending |
| PtrType-agnostic accessors | Keep unique_ptr support (single reader) alongside shared_ptr (concurrent readers) | -- Pending |
| All conversions through data_batch statics | Centralizes lock management, state machine is explicit | -- Pending |
| No direct locked-to-locked conversion | Forces explicit "back through idle" pattern, simpler API | -- Pending |
| try variants use `&` not `&&` | Conditional move — nullifies on success, unchanged on failure | -- Pending |
| Public constructor, no passkey | Not needed without enable_shared_from_this — static methods enforce PtrType at lock acquisition | -- Pending |
| Delete move/copy on data_batch | Fixes known move-without-lock bug; object is never moved, only PtrType to it is | -- Pending |
| clone() on read_only_data_batch | Avoids recursive shared_mutex deadlock if clone() internally locked on data_batch | -- Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check -- still the right priority?
3. Audit Out of Scope -- reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-14 after design refinement (dropped enable_shared_from_this, kept PtrType agnostic)*
