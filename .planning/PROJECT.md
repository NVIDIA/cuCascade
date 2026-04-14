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
- [x] **DB-01**: 3 flat classes: `data_batch`, `read_only_data_batch`, `mutable_data_batch` -- Validated in Phase 01
- [x] **DB-02**: `data_batch` minimal public API, data access only through accessor types -- Validated in Phase 01
- [x] **DB-03**: PtrType-agnostic accessors templated on shared_ptr/unique_ptr -- Validated in Phase 01
- [x] **DB-04**: All state transitions through `data_batch` static methods with move semantics -- Validated in Phase 01
- [x] **DB-05**: 6 blocking static conversion methods + 2 try variants -- Validated in Phase 01, tested in Phase 03
- [x] **DB-06**: Public constructor, no enable_shared_from_this -- Validated in Phase 01
- [x] **DB-07**: `get_batch_id()` public and lock-free -- Validated in Phase 01
- [x] **DB-08**: `const uint64_t _batch_id` immutable after construction -- Validated in Phase 01
- [x] **DB-09**: Atomic subscriber count operations -- Validated in Phase 01, tested in Phase 03
- [x] **DB-10**: Deleted move/copy on `data_batch` -- Validated in Phase 01, static_assert in Phase 03
- [x] **DB-11**: `read_only_data_batch` exposes read-only accessors -- Validated in Phase 01
- [x] **DB-12**: `mutable_data_batch` exposes read+write accessors -- Validated in Phase 01
- [x] **DB-13**: `clone()` on `read_only_data_batch` (avoids shared_mutex deadlock) -- Validated in Phase 01, tested in Phase 03
- [x] **DB-14**: `idata_repository<PtrType>` stays templated -- Validated in Phase 02
- [x] **DB-15**: All tests exercise new 3-class API -- Validated in Phase 03 (full rewrite + build + test pass)
- [x] **DB-16**: Mutual friend relationships between all 3 classes -- Validated in Phase 01
- [x] **DB-17**: `[[nodiscard]]` on all transition/accessor methods -- Validated in Phase 01

### Active

(none -- all requirements validated)

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
| 3 flat classes instead of nested | Simpler mental model, no unnecessary indirection | Confirmed (Phase 01) |
| No enable_shared_from_this | Static methods receive PtrType as param — no need to get shared_ptr from inside object | Confirmed (Phase 01) |
| PtrType-agnostic accessors | Keep unique_ptr support (single reader) alongside shared_ptr (concurrent readers) | Confirmed (Phase 01) |
| All conversions through data_batch statics | Centralizes lock management, state machine is explicit | Confirmed (Phase 01) |
| No direct locked-to-locked conversion | Forces explicit "back through idle" pattern, simpler API | Confirmed (Phase 01) |
| try variants use `&` not `&&` | Conditional move — nullifies on success, unchanged on failure | Confirmed (Phase 01) |
| Public constructor, no passkey | Not needed without enable_shared_from_this — static methods enforce PtrType at lock acquisition | Confirmed (Phase 01) |
| Delete move/copy on data_batch | Fixes known move-without-lock bug; object is never moved, only PtrType to it is | Confirmed (Phase 01) |
| clone() on read_only_data_batch | Avoids recursive shared_mutex deadlock if clone() internally locked on data_batch | Confirmed (Phase 01) |

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
*Last updated: 2026-04-14 after Phase 03 completion — all requirements validated, all decisions confirmed, build+test passing*
