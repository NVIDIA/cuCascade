# Phase 02: Repository Integration - Research

**Researched:** 2026-04-14
**Domain:** C++20 template class type replacement and SFINAE-to-constexpr modernization
**Confidence:** HIGH

## Summary

Phase 2 is a focused, mechanical refactoring of 4 repository-layer files to replace all references to the deleted `synchronized_data_batch` class with the new `data_batch` class from Phase 1. The old class no longer exists in `data_batch.hpp` -- it was replaced by the 3-class type system (`data_batch`, `read_only_data_batch<PtrType>`, `mutable_data_batch<PtrType>`). The 4 repository files still reference the old name in type aliases, explicit template instantiations, SFINAE conditions, and function specializations, making them uncompilable.

The only non-mechanical change is modernizing the `add_data_batch_impl` dispatch in `data_repository_manager.hpp`: two `std::enable_if` overloads (one for `shared_ptr<synchronized_data_batch>`, one for `unique_ptr<synchronized_data_batch>`) are replaced by a single function using `if constexpr (std::is_copy_constructible_v<PtrType>)`. This is cleaner, decoupled from the specific type name, and idiomatic C++20. The trait works correctly because `shared_ptr<data_batch>` is copy-constructible while `unique_ptr<data_batch>` is not.

**Primary recommendation:** Treat this as a single atomic change across all 4 files. Every change is a direct textual substitution or the SFINAE-to-constexpr rewrite. No behavioral logic changes, no new functions, no new tests.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Replace the two `std::enable_if` overloads of `add_data_batch_impl` in `data_repository_manager.hpp` with a single function using `if constexpr (std::is_copy_constructible_v<T>)`. This collapses shared/unique dispatch into one function body, avoids coupling the condition to a specific type name, and is idiomatic C++20.
- **D-02:** All `synchronized_data_batch` references in the 4 repository files become `data_batch`. This is a mechanical find-and-replace for type aliases, explicit template instantiations, and SFINAE conditions.
- **D-03:** `get_data_batch_by_id` specializations in `data_repository.cpp` update their template parameters from `synchronized_data_batch` to `data_batch` -- no behavioral change (shared_ptr copies pointer, unique_ptr throws).
- **D-04:** Phase 2 modifies only the 4 repository layer files. Test files (~850 `synchronized_data_batch` references across 3 test files) are Phase 3 scope. The full build will not pass until Phase 3 -- Phase 2's success criterion is that the repository layer files compile cleanly with the new type.

### Claude's Discretion
- Include/forward-declaration cleanup as needed to make the repository layer compile
- Whether `get_data_batch_by_id` specializations should also be modernized to `if constexpr` (same pattern as manager dispatch)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REPO-01 | `idata_repository<PtrType>` stays templated -- supports both `shared_ptr<data_batch>` and `unique_ptr<data_batch>` | The template class `idata_repository<PtrType>` is already generic on PtrType; only the type aliases and explicit instantiations reference `synchronized_data_batch`. Changing those to `data_batch` preserves the template. No structural change needed to the class itself. |
| REPO-02 | Update type aliases: `shared_data_repository` and `unique_data_repository` use new `data_batch` (not `synchronized_data_batch`) | Lines 282-283 of `data_repository.hpp` and lines 279-282 of `data_repository_manager.hpp` contain the 4 type aliases. Direct textual replacement. |
| REPO-03 | Update `data_repository_manager` to use new `data_batch` | The manager references `synchronized_data_batch` in: (1) two `enable_if` SFINAE conditions (lines 243, 254), (2) two type aliases (lines 279-282), and its `.cpp` file has two explicit instantiations (lines 27-28). The SFINAE conditions are modernized to `if constexpr` per D-01; everything else is textual replacement. |
</phase_requirements>

## Architecture Patterns

### Files and Their Changes

```
include/cucascade/data/
  data_repository.hpp          # Lines 282-283: type aliases
  data_repository_manager.hpp  # Lines 242-269: SFINAE -> if constexpr
                               # Lines 279-282: type aliases
src/data/
  data_repository.cpp          # Lines 23-24: explicit instantiations
                               # Lines 27-57: get_data_batch_by_id specializations
  data_repository_manager.cpp  # Lines 27-28: explicit instantiations
```

### Pattern 1: Type Alias Replacement (Mechanical)

**What:** Replace `synchronized_data_batch` with `data_batch` in all `using` declarations.
**When:** All 4 type aliases across both headers.

**Before (data_repository.hpp:282-283):**
```cpp
using shared_data_repository = idata_repository<std::shared_ptr<synchronized_data_batch>>;
using unique_data_repository = idata_repository<std::unique_ptr<synchronized_data_batch>>;
```

**After:**
```cpp
using shared_data_repository = idata_repository<std::shared_ptr<data_batch>>;
using unique_data_repository = idata_repository<std::unique_ptr<data_batch>>;
```
[VERIFIED: include/cucascade/data/data_repository.hpp lines 282-283]

### Pattern 2: Explicit Template Instantiation Replacement (Mechanical)

**What:** Replace `synchronized_data_batch` with `data_batch` in all `template class` declarations in `.cpp` files.
**When:** Both `.cpp` files.

**Before (data_repository.cpp:23-24):**
```cpp
template class idata_repository<std::shared_ptr<synchronized_data_batch>>;
template class idata_repository<std::unique_ptr<synchronized_data_batch>>;
```

**After:**
```cpp
template class idata_repository<std::shared_ptr<data_batch>>;
template class idata_repository<std::unique_ptr<data_batch>>;
```
[VERIFIED: src/data/data_repository.cpp lines 23-24]

### Pattern 3: SFINAE to if-constexpr Modernization (D-01)

**What:** Collapse two `enable_if`-dispatched overloads of `add_data_batch_impl` into one function body using `if constexpr`.
**When:** `data_repository_manager.hpp` lines 241-269.

**Before (two overloads):**
```cpp
// shared_ptr overload
template <typename T = PtrType>
typename std::enable_if<std::is_same<T, std::shared_ptr<synchronized_data_batch>>::value>::type
add_data_batch_impl(T batch, std::vector<std::pair<size_t, std::string_view>>& ops)
{
  std::lock_guard<std::mutex> lock(_mutex);
  for (auto& op : ops) {
    _repositories[{op.first, std::string(op.second)}]->add_data_batch(batch);
  }
}

// unique_ptr overload
template <typename T = PtrType>
typename std::enable_if<std::is_same<T, std::unique_ptr<synchronized_data_batch>>::value>::type
add_data_batch_impl(T batch, std::vector<std::pair<size_t, std::string_view>>& ops)
{
  std::lock_guard<std::mutex> lock(_mutex);
  if (ops.size() > 1) {
    throw std::runtime_error(
      "unique_ptr data_batch can only be added to one repository. "
      "Use shared_ptr for multiple destinations.");
  }
  if (!ops.empty()) {
    auto& op = ops[0];
    _repositories[{op.first, std::string(op.second)}]->add_data_batch(std::move(batch));
  } else {
    throw std::runtime_error("No operator ports provided");
  }
}
```

**After (single function with if constexpr):**
```cpp
void add_data_batch_impl(PtrType batch, std::vector<std::pair<size_t, std::string_view>>& ops)
{
  std::lock_guard<std::mutex> lock(_mutex);
  if constexpr (std::is_copy_constructible_v<PtrType>) {
    // shared_ptr path: copy to each repository
    for (auto& op : ops) {
      _repositories[{op.first, std::string(op.second)}]->add_data_batch(batch);
    }
  } else {
    // unique_ptr path: move to exactly one repository
    if (ops.size() > 1) {
      throw std::runtime_error(
        "unique_ptr data_batch can only be added to one repository. "
        "Use shared_ptr for multiple destinations.");
    }
    if (!ops.empty()) {
      auto& op = ops[0];
      _repositories[{op.first, std::string(op.second)}]->add_data_batch(std::move(batch));
    } else {
      throw std::runtime_error("No operator ports provided");
    }
  }
}
```
[VERIFIED: include/cucascade/data/data_repository_manager.hpp lines 240-269]

**Why `std::is_copy_constructible_v<PtrType>` is correct:**
- `std::shared_ptr<data_batch>` is copy-constructible -- the `if constexpr` true branch compiles (copies `batch` to each repository). [VERIFIED: C++ standard -- shared_ptr is always copy-constructible]
- `std::unique_ptr<data_batch>` is NOT copy-constructible -- the `if constexpr` false branch compiles (requires exactly one destination, moves `batch`). [VERIFIED: C++ standard -- unique_ptr is move-only]
- This trait is type-agnostic: it does not hardcode `data_batch` in the condition, so it remains valid if the pointed-to type ever changes.

### Pattern 4: Template Specialization Replacement (data_repository.cpp)

**What:** Update the `get_data_batch_by_id` explicit specializations to use `data_batch` instead of `synchronized_data_batch`.
**When:** `data_repository.cpp` lines 26-57.

**Before:**
```cpp
template <>
std::shared_ptr<synchronized_data_batch>
idata_repository<std::shared_ptr<synchronized_data_batch>>::get_data_batch_by_id(
  uint64_t batch_id, size_t partition_idx)
{ /* ... */ }

template <>
std::unique_ptr<synchronized_data_batch>
idata_repository<std::unique_ptr<synchronized_data_batch>>::get_data_batch_by_id(
  uint64_t /*batch_id*/, size_t /*partition_idx*/)
{ /* ... */ }
```

**After:**
```cpp
template <>
std::shared_ptr<data_batch>
idata_repository<std::shared_ptr<data_batch>>::get_data_batch_by_id(
  uint64_t batch_id, size_t partition_idx)
{ /* body unchanged */ }

template <>
std::unique_ptr<data_batch>
idata_repository<std::unique_ptr<data_batch>>::get_data_batch_by_id(
  uint64_t /*batch_id*/, size_t /*partition_idx*/)
{ /* body unchanged */ }
```
[VERIFIED: src/data/data_repository.cpp lines 26-57]

**Note on the specialization bodies:** The shared_ptr specialization calls `(*it)->get_batch_id()`, which works because `data_batch::get_batch_id()` is public and lock-free. The unique_ptr specialization throws -- no change to behavior.

### Anti-Patterns to Avoid

- **Removing the `unique_ptr` path:** The CONTEXT.md does not authorize de-templating. REPO-01 explicitly requires keeping `idata_repository<PtrType>` templated with both `shared_ptr<data_batch>` and `unique_ptr<data_batch>`. The earlier ARCHITECTURE.md research recommended de-templating, but the user rejected that approach -- the template stays.
- **Touching test files:** D-04 explicitly scopes test file updates to Phase 3. Do not modify any `test/` files.
- **Changing function signatures or behavior:** This phase is purely a type-name substitution plus the SFINAE modernization. No new methods, no deleted methods, no changed semantics.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| shared_ptr vs unique_ptr dispatch | Custom type traits or tag dispatch | `std::is_copy_constructible_v<PtrType>` with `if constexpr` | Standard trait, zero maintenance, correct by construction |

## Common Pitfalls

### Pitfall 1: Forgetting an explicit instantiation site

**What goes wrong:** The `.cpp` file has explicit template instantiations. If one is missed, the linker will fail with undefined symbol errors for whichever specialization was not instantiated.
**Why it happens:** There are 4 instantiation sites across 2 `.cpp` files (2 in `data_repository.cpp`, 2 in `data_repository_manager.cpp`), plus 2 explicit specializations of `get_data_batch_by_id`.
**How to avoid:** Grep for `synchronized_data_batch` across all 4 files after editing. Count should be 0.
**Warning signs:** `undefined reference to cucascade::idata_repository<...>` linker errors.

### Pitfall 2: include/forward-declaration mismatch

**What goes wrong:** The repository headers include `data_batch.hpp` which now defines `data_batch` (not `synchronized_data_batch`). If any forward declaration of `synchronized_data_batch` exists elsewhere, it would conflict.
**Why it happens:** Sometimes old forward declarations linger in headers.
**How to avoid:** Verify `synchronized_data_batch` is not forward-declared anywhere in `include/`. Already verified: `grep -r "class synchronized_data_batch" include/` returns no results.
**Warning signs:** Compilation error about incomplete type or redefinition.

### Pitfall 3: if constexpr discarded branch must still be syntactically valid

**What goes wrong:** Both branches of `if constexpr` must be well-formed (syntactically valid, names must be findable) even though only one is instantiated.
**Why it happens:** Unlike `#if`/`#else`, `if constexpr` does not skip parsing of the discarded branch -- it only skips template-dependent name lookup.
**How to avoid:** Both branches in the `add_data_batch_impl` rewrite reference `_repositories`, `std::move`, and `add_data_batch` -- all of which are template-dependent (through `PtrType` or `repository_type`), so the discarded branch is safe. The `throw` statements use string literals which are always valid. No issue expected.
**Warning signs:** Compilation error in the discarded branch, typically about calling a deleted function.

### Pitfall 4: Doxygen comments still mentioning synchronized_data_batch

**What goes wrong:** Comments in the modified files still reference `synchronized_data_batch`, creating stale documentation.
**Why it happens:** Doxygen `@brief` and `@tparam` descriptions may mention the old type name.
**How to avoid:** After making the type changes, scan all 4 files for any remaining `synchronized` string in comments and update to `data_batch`.
**Warning signs:** None at compile time -- this is a documentation quality issue.

## Discretion Recommendation: get_data_batch_by_id modernization

The CONTEXT.md grants discretion on whether to modernize the `get_data_batch_by_id` specializations to `if constexpr` (matching the manager dispatch pattern).

**Recommendation: Do NOT modernize.** Keep the explicit template specializations as-is (just with the type name updated).

**Rationale:**
1. The specializations are in the `.cpp` file -- they are implementation details, not visible in the header API.
2. To use `if constexpr`, the function body would need to be in the header (template definition must be visible at instantiation point), which would change the header/source split.
3. The current pattern (explicit specializations in `.cpp`) is already used throughout the codebase and is the established convention per Phase 1 CONTEXT.md.
4. The change would be gratuitous complexity for zero user-visible benefit -- the function behavior is identical either way.

## Code Examples

### Complete data_repository.hpp type alias change
```cpp
// Source: include/cucascade/data/data_repository.hpp (current lines 282-283)
// BEFORE:
using shared_data_repository = idata_repository<std::shared_ptr<synchronized_data_batch>>;
using unique_data_repository = idata_repository<std::unique_ptr<synchronized_data_batch>>;

// AFTER:
using shared_data_repository = idata_repository<std::shared_ptr<data_batch>>;
using unique_data_repository = idata_repository<std::unique_ptr<data_batch>>;
```

### Complete data_repository_manager.hpp type alias change
```cpp
// Source: include/cucascade/data/data_repository_manager.hpp (current lines 279-282)
// BEFORE:
using shared_data_repository_manager =
  data_repository_manager<std::shared_ptr<synchronized_data_batch>>;
using unique_data_repository_manager =
  data_repository_manager<std::unique_ptr<synchronized_data_batch>>;

// AFTER:
using shared_data_repository_manager =
  data_repository_manager<std::shared_ptr<data_batch>>;
using unique_data_repository_manager =
  data_repository_manager<std::unique_ptr<data_batch>>;
```

### Complete add_data_batch_impl replacement
```cpp
// Source: include/cucascade/data/data_repository_manager.hpp (current lines 240-269)
// Single function replaces two enable_if overloads:
void add_data_batch_impl(PtrType batch, std::vector<std::pair<size_t, std::string_view>>& ops)
{
  std::lock_guard<std::mutex> lock(_mutex);
  if constexpr (std::is_copy_constructible_v<PtrType>) {
    for (auto& op : ops) {
      _repositories[{op.first, std::string(op.second)}]->add_data_batch(batch);
    }
  } else {
    if (ops.size() > 1) {
      throw std::runtime_error(
        "unique_ptr data_batch can only be added to one repository. "
        "Use shared_ptr for multiple destinations.");
    }
    if (!ops.empty()) {
      auto& op = ops[0];
      _repositories[{op.first, std::string(op.second)}]->add_data_batch(std::move(batch));
    } else {
      throw std::runtime_error("No operator ports provided");
    }
  }
}
```

### Include chain verification
```
data_repository.hpp
  #include <cucascade/data/data_batch.hpp>  -- defines class data_batch (Phase 1 output)
  -- No include of old synchronized_data_batch anywhere

data_repository_manager.hpp
  #include <cucascade/data/data_batch.hpp>   -- defines class data_batch
  #include <cucascade/data/data_repository.hpp> -- defines idata_repository<PtrType>
  -- Already includes <type_traits> (needed for std::is_copy_constructible_v)
```
[VERIFIED: data_repository.hpp line 20, data_repository_manager.hpp lines 20-21]

**Note on `<type_traits>` include:** `data_repository_manager.hpp` does NOT currently include `<type_traits>`. It relies on `data_batch.hpp` pulling it in transitively (data_batch.hpp line 31: `#include <type_traits>`). For the `if constexpr` rewrite using `std::is_copy_constructible_v`, this header needs `<type_traits>`. The safest approach is to add `#include <type_traits>` directly to `data_repository_manager.hpp` rather than relying on transitive includes. [VERIFIED: data_repository_manager.hpp includes list at lines 18-30 -- no `<type_traits>` present]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `if constexpr` discarded branches for template-dependent expressions in a class template member are safe from ODR and compilation errors | Pitfall 3 | Would cause compile error in the non-instantiated branch -- but this is well-established C++17/20 behavior and the expressions are all template-dependent |

## Open Questions

1. **Comment cleanup scope**
   - What we know: The 4 files have Doxygen comments referencing `synchronized_data_batch` (e.g., `data_repository.hpp` header comment block, `data_repository_manager.hpp` class docs).
   - What's unclear: Whether the Doxygen comments in these files reference `synchronized_data_batch` by name or only generically as "data_batch".
   - Recommendation: Scan all 4 files for the string `synchronized` in comments and update any found. Low effort, high documentation quality.

## Sources

### Primary (HIGH confidence)
- `include/cucascade/data/data_batch.hpp` -- Phase 1 output, verified complete 3-class system with 546 lines
- `include/cucascade/data/data_repository.hpp` -- Full file read, 285 lines
- `src/data/data_repository.cpp` -- Full file read, 59 lines
- `include/cucascade/data/data_repository_manager.hpp` -- Full file read, 284 lines
- `src/data/data_repository_manager.cpp` -- Full file read, 30 lines
- `src/data/data_batch.cpp` -- Phase 1 output, verified 72 lines with explicit instantiations
- `.planning/phases/02-repository-integration/02-CONTEXT.md` -- User decisions D-01 through D-04
- `.planning/phases/01-core-type-system/01-VERIFICATION.md` -- Phase 1 verification confirming all 26 requirements satisfied

### Secondary (MEDIUM confidence)
- `.planning/research/ARCHITECTURE.md` -- Build order and component boundary analysis

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- this is a C++20 refactoring of existing code; no new libraries needed
- Architecture: HIGH -- all 4 files fully read, every line requiring change identified with exact line numbers
- Pitfalls: HIGH -- all pitfalls are C++ compilation mechanics with well-known behavior

**Research date:** 2026-04-14
**Valid until:** indefinite (code-level analysis of specific files, no external dependencies)
