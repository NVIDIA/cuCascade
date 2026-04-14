# Roadmap: cuCascade data_batch Refactor

## Overview

Replace `synchronized_data_batch` with a flat 3-class design (`data_batch`, `read_only_data_batch`, `mutable_data_batch`) where lock state is encoded in the C++ type system. Phase 1 delivers the entire new API atomically in a single header (mutual friend relationships and accessor-transition coupling make intermediate states uncompilable). Phase 2 ripples the type change through the repository layer. Phase 3 rewrites all tests and validates the full build.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Core Type System** - Implement data_batch, both accessor types, all state transitions, clone operations, and friend relationships as one atomic compilation unit
- [ ] **Phase 2: Repository Integration** - Update idata_repository, type aliases, and data_repository_manager to use new data_batch
- [ ] **Phase 3: Test Migration and Build Validation** - Rewrite all tests for 3-class API, verify full build and test pass cleanly

## Phase Details

### Phase 1: Core Type System
**Goal**: The complete data_batch type system compiles and provides compile-time enforced data access safety through RAII accessor types and move-semantic state transitions
**Depends on**: Nothing (first phase)
**Requirements**: CORE-01, CORE-02, CORE-03, CORE-04, CORE-05, CORE-06, CORE-07, ACC-01, ACC-02, ACC-03, ACC-04, ACC-05, ACC-06, ACC-07, ACC-08, TRANS-01, TRANS-02, TRANS-03, TRANS-04, TRANS-05, TRANS-06, TRANS-07, TRANS-08, CLONE-01, CLONE-02, REPO-04
**Success Criteria** (what must be TRUE):
  1. `data_batch` exposes only `get_batch_id()`, subscriber count operations, and deleted copy/move -- no public access to data, tier, or memory space
  2. `read_only_data_batch` and `mutable_data_batch` are move-only, PtrType-templated accessor types constructed exclusively through `data_batch` static methods that consume ownership via move semantics, with correct member declaration order (PtrType parent before lock guard)
  3. All 4 blocking transition methods (to_read_only, to_mutable, to_idle x2) and 2 try variants compile, are marked `[[nodiscard]]`, and enforce correct lock semantics (shared for read_only, exclusive for mutable) -- no locked-to-locked shortcuts
  4. `clone()` and `clone_to<T>()` work from a `read_only_data_batch` without deadlocking (no internal lock acquisition on data_batch)
  5. `data_batch.hpp` compiles cleanly with the CUDA C++20 toolchain
**Plans:** 2 plans

Plans:
- [ ] 01-01-PLAN.md -- Write complete data_batch.hpp header (3 classes, transitions, clone, friends)
- [ ] 01-02-PLAN.md -- Write data_batch.cpp (method bodies, template instantiations) and validate compilation

### Phase 2: Repository Integration
**Goal**: The repository layer compiles and works with the new data_batch type, maintaining PtrType-agnostic template support
**Depends on**: Phase 1
**Requirements**: REPO-01, REPO-02, REPO-03
**Success Criteria** (what must be TRUE):
  1. `idata_repository<PtrType>` compiles with both `shared_ptr<data_batch>` and `unique_ptr<data_batch>` -- no `synchronized_data_batch` references remain
  2. `shared_data_repository` and `unique_data_repository` type aliases resolve to the new `data_batch` type
  3. `data_repository_manager` compiles and routes batches through the new type system
**Plans**: TBD

### Phase 3: Test Migration and Build Validation
**Goal**: All tests exercise the new 3-class API and the full build passes cleanly, proving the refactor is correct end-to-end
**Depends on**: Phase 2
**Requirements**: TEST-01, TEST-02, TEST-03, TEST-04, TEST-05, TEST-06, TEST-07, TEST-08, TEST-09, TEST-10
**Success Criteria** (what must be TRUE):
  1. `test_data_batch.cpp` exercises the 3-class API: construction, all 6 blocking transitions, both try variants, clone operations, and subscriber count
  2. Tests verify destruction order safety (lock released before parent dropped) and are TSan/ASan clean
  3. Tests verify concurrent readers (shared_ptr path with multiple simultaneous `read_only_data_batch` instances) and serialized exclusive writers
  4. `pixi run build` compiles all 61 targets and `pixi run test` passes all tests with zero failures
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Core Type System | 0/2 | Planning complete | - |
| 2. Repository Integration | 0/0 | Not started | - |
| 3. Test Migration and Build Validation | 0/0 | Not started | - |
