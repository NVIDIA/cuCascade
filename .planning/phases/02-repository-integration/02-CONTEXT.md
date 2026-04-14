# Phase 2: Repository Integration - Context

**Gathered:** 2026-04-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Update the repository layer to use the new `data_batch` type from Phase 1. Replace all `synchronized_data_batch` references in `idata_repository`, `data_repository_manager`, type aliases, and explicit template instantiations. Modernize SFINAE dispatch to `if constexpr`. This phase targets 4 files (2 headers, 2 `.cpp` files) — test file updates are Phase 3 scope.

</domain>

<decisions>
## Implementation Decisions

### SFINAE Modernization
- **D-01:** Replace the two `std::enable_if` overloads of `add_data_batch_impl` in `data_repository_manager.hpp` with a single function using `if constexpr (std::is_copy_constructible_v<T>)`. This collapses shared/unique dispatch into one function body, avoids coupling the condition to a specific type name, and is idiomatic C++20.

### Type Replacement
- **D-02:** All `synchronized_data_batch` references in the 4 repository files become `data_batch`. This is a mechanical find-and-replace for type aliases, explicit template instantiations, and SFINAE conditions.
- **D-03:** `get_data_batch_by_id` specializations in `data_repository.cpp` update their template parameters from `synchronized_data_batch` to `data_batch` — no behavioral change (shared_ptr copies pointer, unique_ptr throws).

### Scope
- **D-04:** Phase 2 modifies only the 4 repository layer files. Test files (~850 `synchronized_data_batch` references across 3 test files) are Phase 3 scope. The full build will not pass until Phase 3 — Phase 2's success criterion is that the repository layer files compile cleanly with the new type.

### Claude's Discretion
- Include/forward-declaration cleanup as needed to make the repository layer compile
- Whether `get_data_batch_by_id` specializations should also be modernized to `if constexpr` (same pattern as manager dispatch)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### New Type System (Phase 1 output)
- `include/cucascade/data/data_batch.hpp` -- New 3-class type system (data_batch, read_only_data_batch, mutable_data_batch)
- `src/data/data_batch.cpp` -- Non-template method bodies + explicit accessor instantiations

### Repository Layer (files to modify)
- `include/cucascade/data/data_repository.hpp` -- `idata_repository<PtrType>` template + type aliases (lines 282-283)
- `src/data/data_repository.cpp` -- Explicit instantiations (lines 23-24) + get_data_batch_by_id specializations (lines 27-57)
- `include/cucascade/data/data_repository_manager.hpp` -- `data_repository_manager<PtrType>` + SFINAE dispatch (lines 242-269) + type aliases (lines 279-282)
- `src/data/data_repository_manager.cpp` -- Explicit instantiations (lines 27-28)

### Phase 1 Context
- `.planning/phases/01-core-type-system/01-CONTEXT.md` -- All Phase 1 design decisions (D-01 through D-25)

### Research
- `.planning/research/ARCHITECTURE.md` -- Build order and component boundaries

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 1's `data_batch.hpp` already defines the complete type system — repository layer just needs to reference it
- `idata_repository` template is already fully generic on PtrType — no template logic changes needed, only type alias updates

### Established Patterns
- Explicit template instantiation in `.cpp` files for both `shared_ptr<T>` and `unique_ptr<T>` — follow existing pattern
- `data_batch::get_batch_id()` is public and lock-free — repository lookup by ID works unchanged
- Repository stores/routes idle (unlocked) batches — no lock interaction at the repository level

### Integration Points
- Type aliases `shared_data_repository` / `unique_data_repository` in `data_repository.hpp` — downstream code depends on these
- Type aliases `shared_data_repository_manager` / `unique_data_repository_manager` in `data_repository_manager.hpp`
- Manager's `add_data_batch_impl` SFINAE dispatch — being modernized to `if constexpr`

</code_context>

<specifics>
## Specific Ideas

- User chose `if constexpr` over `requires` clause or keeping `std::enable_if` — values the single-function-body clarity and decoupling the condition from a specific type name
- The `std::is_copy_constructible_v<T>` trait is preferred over `std::is_same_v<T, shared_ptr<data_batch>>` because it's type-agnostic and would work even if PtrType changes in the future

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 02-repository-integration*
*Context gathered: 2026-04-14*
