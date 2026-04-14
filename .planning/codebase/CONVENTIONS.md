# Coding Conventions

**Analysis Date:** 2026-04-13

## Naming Patterns

**Files:**
- Source files: `snake_case.cpp` — e.g., `data_batch.cpp`, `memory_reservation.cpp`
- Header files: `snake_case.hpp` — e.g., `data_repository.hpp`, `cuda_utils.hpp`
- CUDA kernel files: `snake_case.cu` / `snake_case.cuh`
- Test files: `test_<subject>.cpp` — e.g., `test_data_batch.cpp`

**Types / Classes:**
- Concrete classes: `snake_case` — e.g., `synchronized_data_batch`, `memory_reservation_manager`
- Abstract interfaces: prefix `i` + `snake_case` — e.g., `idata_representation`, `idata_repository`
- Enumerations: `PascalCase` for name, `UPPER_CASE` for enumerators — e.g., `enum class Tier { GPU, HOST, DISK, SIZE }`
- Template traits: `snake_case_trait` — e.g., `tier_memory_resource_trait<Tier::GPU>`

**Member Variables:**
- All private/protected: underscore prefix + snake_case — `_batch_id`, `_rw_mutex`, `_subscriber_count`
- No `m_` Hungarian prefix; use single `_` prefix exclusively

**Functions / Methods:**
- All functions: `snake_case` — `get_batch_id()`, `set_data()`, `try_get_read_only()`
- Factory functions: `make_<name>` — e.g., `make_mock_memory_space()`, `make_default_gpu_memory_resource()`
- Predicate methods: `has_<name>`, `is_<name>` — e.g., `has_converter()`, `all_empty()`
- Boolean-returning subscriptions: `subscribe()` / `unsubscribe()`

**Macros:**
- All caps with project prefix: `CUCASCADE_CUDA_TRY`, `CUCASCADE_CUDA_TRY_ALLOC`, `CUCASCADE_FUNC_RANGE`, `CUCASCADE_STRINGIFY`

**Namespaces:**
- Root: `cucascade`
- Sub-namespaces by domain: `cucascade::memory`, `cucascade::utils`
- Test-only utilities: `cucascade::test`
- Anonymous namespaces: used in `.cpp` files for file-local helpers

**Template Parameters:**
- `PascalCase` — e.g., `TargetRepresentation`, `TargetType`, `TIER` (compile-time value parameter uses UPPER_CASE)

## Code Style

**Formatting:**
- Tool: `clang-format` v20.1.4 via pre-commit hook
- Config: `.clang-format` (root, BasedOnStyle: Google with overrides)
- Column limit: 100 characters
- Indent width: 2 spaces (no tabs)
- Brace style: WebKit — opening brace on same line for all constructs
- Pointer alignment: Left — `int* ptr` not `int *ptr`
- Trailing comments: 2 spaces before `//`

**Key formatting rules enforced:**
- `AlignConsecutiveAssignments: true` — align `=` in consecutive declarations
- `AlignTrailingComments: true`
- `BinPackParameters: false` — each parameter on its own line if wrapping
- `BreakConstructorInitializers: BeforeColon` — `:` on new line for init lists
- `ConstructorInitializerIndentWidth: 2` / `ContinuationIndentWidth: 2`
- `MaxEmptyLinesToKeep: 1`

**Linting:**
- Tool: `cmake-lint` and `cmake-format` (via pre-commit)
- C++ linting: compiler warnings treated as errors (`-Wall -Wextra -Wpedantic -Wcast-align -Wunused -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion -Wformat=2 -Wimplicit-fallthrough`)
- Spell checking: `codespell` (with `.codespell_words` ignore list)

## Import Organization

**Order (enforced by clang-format `IncludeBlocks: Regroup`):**
1. Quoted local includes: `"utils/mock_test_utils.hpp"`
2. Test/benchmark includes: `<test/...>`, `<benchmarks/...>`
3. cuDF test includes: `<cudf_test/...>`
4. cuCascade includes: `<cucascade/...>`
5. cuDF includes: `<cudf/...>`, `<nvtext/...>`
6. Other RAPIDS: `<raft/...>`, `<kvikio/...>`
7. RMM includes: `<rmm/...>`
8. CCCL includes: `<thrust/...>`, `<cub/...>`, `<cuda/...>`
9. CUDA runtime: `<cuda_runtime_api.h>`, `<cooperative_groups>`, `<nvtx3>`
10. System includes with `.`: `<system_error>`, `<cassert>`
11. STL includes without `.`: `<memory>`, `<atomic>`, `<mutex>`, `<thread>`, `<vector>`

**Path Aliases:** None — all includes use full paths relative to include roots.

## Error Handling

**Patterns:**
- CUDA runtime errors: use `CUCASCADE_CUDA_TRY(call)` macro — throws `rmm::cuda_error` with file/line context. Defined in `include/cucascade/cuda_utils.hpp`
- CUDA allocation failures: use `CUCASCADE_CUDA_TRY_ALLOC(call)` or `CUCASCADE_CUDA_TRY_ALLOC(call, num_bytes)` — throws `rmm::out_of_memory` or `rmm::bad_alloc`
- Debug CUDA checks: `CUCASCADE_ASSERT_CUDA_SUCCESS(call)` — asserts in debug, no-op in release
- Memory allocation errors: throw `rmm::out_of_memory`, `rmm::bad_alloc`, or custom `cucascade_out_of_memory` (subclasses `rmm::out_of_memory`)
- Logic/precondition failures: throw `std::runtime_error`, `std::invalid_argument`, `std::logic_error`
- Custom error system: `MemoryError` enum + `std::error_code` integration in `src/memory/error.cpp`
- Destructors: `noexcept` — all cleanup in destructors must not throw

**Exception hierarchy:**
```
std::exception
└── rmm::out_of_memory
    └── cucascade_out_of_memory   (src/memory/error.cpp)
```

## Comments

**File License Header:**
Every source file begins with a standard Apache 2.0 SPDX block:
```cpp
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * ...
 */
```

**Doxygen:**
- All public API methods and classes documented with Doxygen `/** */` blocks
- Tags used: `@brief`, `@param`, `@return`, `@throws`, `@note`, `@tparam`, `@code`/`@endcode`
- Member fields: `///< inline comment` after declaration — e.g., `int64_t _size; ///< doc`
- Interface classes (`idata_representation`, `idata_repository`) are heavily documented with intent and invariants

**Section separators in `.cpp` files:**
```cpp
// =============================================================================
// Section name
// =============================================================================
```
Used to organize large `.cpp` files into logical sections (e.g., `data_batch.cpp` separates inner class, read-only accessor, mutable accessor, and wrapper implementations).

**Inline comments:**
- Explain non-obvious behavior — e.g., why a lock is released before re-acquired
- Mark disabled tests: `// Disabled: reason` comment before `TEST_CASE`
- Rarely used for obvious code

## Function Design

**Size:** Functions tend to be short and focused. Complex logic is broken into clearly named private helpers or delegated to separate impl functions (e.g., `register_converter` delegates to `register_converter_impl`).

**Parameters:**
- Stream parameters: always `rmm::cuda_stream_view stream` (not `cudaStream_t` directly)
- Output via return values, not output parameters
- Pass `std::unique_ptr` by value to transfer ownership; pass `const T&` or `T*` for non-owning access
- `[[maybe_unused]]` attribute on unused parameters when required by interface — e.g., `[[maybe_unused]] rmm::cuda_stream_view stream`

**Return Values:**
- Nullable results: return `nullptr` (for pointer types) or `std::nullopt` (for `std::optional`)
- Non-blocking acquire: `std::optional<accessor_type>` — e.g., `try_get_read_only()`, `try_get_mutable()`
- Factory functions return `std::unique_ptr` or `std::shared_ptr`

## Module Design

**Exports:**
- Public API headers in `include/cucascade/` only — never expose internal headers
- Implementation details in `src/` (not installed)
- Template implementations that must be in headers live in the `.hpp` file below the class definition

**Deleted special members:** Always explicitly `= delete` for copy operations on non-copyable types. Always explicitly `= default` or `= delete` for move operations:
```cpp
MyClass(const MyClass&)            = delete;
MyClass& operator=(const MyClass&) = delete;
MyClass(MyClass&&) noexcept        = default;
MyClass& operator=(MyClass&&) noexcept = default;
```

**Concepts / C++20 constraints:**
- `requires std::derived_from<T, BaseType>` used for template type safety
- `static_assert(std::is_base_of_v<Base, Derived>, "message")` in template implementations
- `std::integral T` constraint on atomic utility templates

**RAII everywhere:** All resource ownership is expressed through smart pointers or scope-based objects. No bare `new`/`delete`. Lock lifetimes are always tied to RAII accessor objects (`std::unique_lock`, `std::shared_lock`).

**`[[nodiscard]]` usage:** Applied to methods whose return value must not be silently discarded — e.g., `size()`, `tier()`, `device_id()`, `peak()`, `try_add()`, `try_sub()`.

**`[[gnu::always_inline]]`:** Used selectively in hot-path memory resource methods (e.g., `reserved_arena::size()`).

## Namespace Closure Comments

All namespaces are closed with a comment:
```cpp
}  // namespace memory
}  // namespace cucascade
```

---

*Convention analysis: 2026-04-13*
