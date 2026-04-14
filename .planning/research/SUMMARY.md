# Research Summary

## Executive Summary

The cuCascade data_batch refactor replaces a nested `synchronized_data_batch` wrapper with a flat 3-class design (`data_batch`, `read_only_data_batch`, `mutable_data_batch`) where state -- idle, read-locked, or write-locked -- is encoded in the C++ type system. Accessors hold `shared_ptr<data_batch>` (via `enable_shared_from_this`) rather than raw pointer borrows, eliminating the dangling-pointer risk flagged in PR #99 review. All transitions are static methods on `data_batch` with move semantics, centralizing lock management and making stale references a compile error.

## Stack

100% C++ standard library -- no new dependencies needed:
- `std::shared_mutex` + `std::shared_lock`/`std::unique_lock` for reader-writer locking
- `std::enable_shared_from_this` for accessor lifetime extension
- Passkey idiom for `make_shared` with private constructor
- `std::optional` for try-variant return types
- `std::atomic<size_t>` with `memory_order_relaxed` for subscriber count
- `[[nodiscard]]` on all transition methods (zero-cost, prevents unnamed-temporary bug)

No CUDA toolchain risk -- data layer is pure host C++ (no `.cu` files).

## Table Stakes Features

- Move-only semantics on accessors (deleted copy)
- Const-correct data access (read_only exposes const subset)
- Blocking `to_read_only` / `to_mutable` from idle
- `to_idle` RAII release returning `shared_ptr<data_batch>`
- Lock-free `get_batch_id()` and subscriber count on data_batch
- `data_batch::create()` factory with private constructor (passkey idiom)
- Mutual friend relationships between data_batch and both accessor types
- Named methods on accessors (not `operator->`)
- Deleted move ctor/assign on `data_batch` itself
- `[[nodiscard]]` on all acquisition methods
- Non-blocking try variants (`try_to_read_only`, `try_to_mutable`)

## Anti-Features (Do NOT Build)

- `operator->` on accessors -- data_batch has no public data API to dereference to
- Direct shared-to-exclusive lock upgrade -- deadlock risk, explicitly rejected by WG21
- Default-constructible accessors -- meaningless without a lock
- Copy semantics on accessors -- locks are not copyable
- Timed lock variants -- unnecessary complexity for this use case
- Recursive locking -- `std::shared_mutex` does not support it
- Implicit conversions between accessor types

## Architecture

- All 3 classes in a single header (`data_batch.hpp`) -- mutual friend relationships require it
- Repository layer de-templates to `shared_ptr<data_batch>` directly (drop `unique_data_repository`)
- Converter infrastructure (`representation_converter_registry`) unchanged
- Build order: data_batch core -> accessor types -> transitions -> clone -> repository -> tests

## Critical Pitfalls

1. **Member declaration order is load-bearing**: `shared_ptr<data_batch>` MUST be declared before the lock guard in accessors. Wrong order = mutex destroyed while locked = UB. Test with TSan/ASan.

2. **clone() recursive lock = deadlock**: If `clone()` acquires a lock internally and caller already holds a `read_only_data_batch`, it deadlocks (shared_mutex is non-reentrant). Fix: make `clone()` a method on `read_only_data_batch` (or static taking `const read_only_data_batch&`), not an internally-locking method on idle `data_batch`. **This overrides DB-13.**

3. **`shared_from_this()` before shared_ptr manages object throws `bad_weak_ptr`**: Prevented by private constructor + passkey factory. Passkey struct needs explicit private constructor (not `= default`) to block aggregate initialization bypass.

4. **ABBA deadlock: repository mutex vs data_batch shared_mutex**: Static-method + move-semantics design naturally enforces correct order (batch lock released by `to_idle` before repo lock acquired), but must be documented.

5. **Non-atomic upgrade creates TOCTOU window**: `to_mutable(read_only_data_batch&&)` releases shared lock then acquires exclusive -- another writer can interpose. API makes this explicit; callers must not assume state is unchanged after upgrade.

## Roadmap Implications

Phases 1-4 are within `data_batch.hpp`/`data_batch.cpp` -- treat as one atomic PR. Phase 5 ripples through repository layer. Phase 6 is the test gate.

1. Core data_batch class (factory, enable_shared_from_this, passkey, batch_id, subscribers)
2. Accessor types (correct member order, move-only, named methods)
3. Static transition methods (all 6 blocking + 2 try variants, `[[nodiscard]]`)
4. Clone operations (method on read_only_data_batch, not internally-locking)
5. Repository de-templating (drop unique_data_repository, de-template idata_repository)
6. Test migration (rewrite all tests for new API, add TSan/ASan tests)

## Open Questions

- Read/write ratio in Sirius workloads -- determines if `shared_mutex` overhead vs plain `mutex` is justified. Flag for post-MVP benchmarking, not a blocker.
- Subscriber count memory ordering (`relaxed` vs `acquire/release`) -- depends on downstream usage patterns.
- Whether to keep `idata_repository` as a template with single instantiation or fully de-template.

---

*Research synthesis: 2026-04-14*
