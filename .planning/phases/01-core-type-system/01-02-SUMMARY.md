---
phase: 01-core-type-system
plan: 02
subsystem: data
tags: [c++20, data_batch, explicit-template-instantiation, raii, shared_mutex]

# Dependency graph
requires:
  - phase: 01-01
    provides: "data_batch.hpp header with 3-class type system declarations"
provides:
  - "data_batch.cpp with non-template method bodies for data_batch class"
  - "Explicit template instantiations for read_only_data_batch and mutable_data_batch (shared_ptr + unique_ptr)"
  - "clang-format v20.1.4 compliance on both header and implementation"
affects: [02-repository-integration, 03-test-migration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Explicit template instantiation in .cpp for PtrType-parameterized accessor classes"
    - "Lock-free atomic subscriber count with underflow guard (fetch_sub + check prev==0)"
    - "Null-safe get_memory_space() pattern (check _data before dereference)"

key-files:
  created: []
  modified:
    - src/data/data_batch.cpp
    - include/cucascade/data/data_batch.hpp

key-decisions:
  - "Preserved subscriber underflow pattern: subtract first, check prev==0, restore on underflow -- matches existing behavior per Pitfall 6"
  - "Removed gpu_data_representation.hpp include -- clone/convert methods now use polymorphic idata_representation::clone() virtual via templates in header"

patterns-established:
  - "Explicit instantiation pattern: template class accessor<shared_ptr<data_batch>>; template class accessor<unique_ptr<data_batch>>; -- mirroring data_repository.cpp"

requirements-completed: [CORE-03, CORE-04, CORE-05, CORE-06, ACC-03]

# Metrics
duration: 54min
completed: 2026-04-14
---

# Phase 01 Plan 02: data_batch.cpp Implementation Summary

**Non-template method bodies for data_batch class with explicit template instantiations for both PtrType specializations, passing clang-format v20.1.4**

## Performance

- **Duration:** 54 min
- **Started:** 2026-04-14T19:42:03Z
- **Completed:** 2026-04-14T20:36:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Replaced entire synchronized_data_batch implementation in data_batch.cpp with flat data_batch class methods
- Implemented all 9 non-template method bodies matching header declarations exactly (constructor, 4 public API, 4 private accessors)
- Added 4 explicit template instantiations (read_only_data_batch and mutable_data_batch x shared_ptr and unique_ptr)
- Applied clang-format v20.1.4 to both header and implementation files, resolving brace placement and alignment issues

## Task Commits

Each task was committed atomically:

1. **Task 1: Write data_batch.cpp with method bodies and explicit template instantiations** - `c8bc46f` (feat)
2. **Task 2: Validate compilation and apply clang-format** - `e24067f` (chore)

## Files Created/Modified
- `src/data/data_batch.cpp` - Complete rewrite: non-template method bodies (constructor, get_batch_id, subscribe/unsubscribe, get_subscriber_count, private data accessors) + explicit template instantiations for 4 accessor/PtrType combinations
- `include/cucascade/data/data_batch.hpp` - clang-format fixes only (brace placement, alignment, no semantic changes)

## Decisions Made
- Preserved the existing subscriber underflow pattern (subtract-then-check) from the old synchronized_data_batch implementation, per Pitfall 6 in RESEARCH.md
- Removed gpu_data_representation.hpp include since clone/convert methods now use polymorphic dispatch through idata_representation::clone() virtual in the header templates

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed clang-format violations in both files**
- **Found during:** Task 2 (compilation validation)
- **Issue:** Both data_batch.hpp (from Plan 01-01) and data_batch.cpp had clang-format v20.1.4 violations: brace placement for multi-line function definitions, trailing comment alignment, and single-statement body formatting
- **Fix:** Applied clang-format -i to both files
- **Files modified:** src/data/data_batch.cpp, include/cucascade/data/data_batch.hpp
- **Verification:** clang-format --dry-run --Werror passes with exit code 0 for both files
- **Committed in:** e24067f

---

**Total deviations:** 1 auto-fixed (1 formatting bug)
**Impact on plan:** Formatting fix was necessary for code quality compliance. No scope creep.

## Issues Encountered

- **pixi run build failed due to environment resolution:** All pixi environments failed to solve because RAPIDS stable packages (libcudf 26.2.x) are unavailable for CUDA 13 on linux-aarch64. This is a pre-existing environment issue (the stable environments reference CUDA 12 packages that don't exist for the CUDA 13 + aarch64 platform). The build failure is NOT related to our code changes -- it occurs before any compilation begins, during conda dependency resolution.
- **Verification approach:** Since full build was blocked by environment issues, verification was done through: (a) structural validation that all 9 cpp method signatures match header declarations exactly, (b) clang-format compliance check with the project's v20.1.4 formatter, (c) acceptance criteria checks (no synchronized_data_batch, correct template instantiations, correct include structure).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- data_batch.hpp + data_batch.cpp pair is complete and internally consistent
- Ready for Phase 2 (repository integration): data_repository.hpp/cpp, data_repository_manager.hpp/cpp need updating to use data_batch instead of synchronized_data_batch
- The pixi build environment issue needs resolution before full compilation validation can occur (affects all phases, not specific to this plan)

---
*Phase: 01-core-type-system*
*Completed: 2026-04-14*
