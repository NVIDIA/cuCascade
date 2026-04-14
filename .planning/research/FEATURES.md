# Feature Landscape: RAII Lock Accessor Types for data_batch

**Domain:** C++ concurrency wrapper with RAII accessor types (reader-writer lock pattern)
**Researched:** 2026-04-13
**Overall confidence:** HIGH (patterns drawn from folly::Synchronized, boost::synchronized_value, safe library, std::unique_lock/shared_lock, and existing cuCascade codebase)

## Table Stakes

Features users expect. Missing = correctness bugs or unusable API.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Move-only semantics (delete copy) | Prevents double-unlock. Two objects holding the same lock is undefined behavior. Every RAII lock type in the standard library does this. | Low | Already present in current codebase. `= delete` copy ctor/assign, `= default` move ctor/assign. |
| `noexcept` move operations | Move must never throw -- a throwing move inside `std::vector::push_back` or `std::optional` assignment would leave the mutex in an inconsistent state. `std::unique_lock` and `std::shared_lock` both guarantee `noexcept` moves. | Low | Already present. Must verify `static_assert(std::is_nothrow_move_constructible_v<...>)` in tests. |
| Const-correct data access | `read_only_data_batch` must expose only `const` access to the underlying data. `mutable_data_batch` exposes non-const. This is the whole point -- compile-time enforcement of read/write separation. | Low | Current design uses `const data_batch*` vs `data_batch*` through `operator->`. New flat-class design should expose const getters on `read_only_data_batch` and non-const setters only on `mutable_data_batch`. |
| Blocking lock acquisition | `data_batch::to_read_only(shared_ptr<data_batch>&&)` and `data_batch::to_mutable(shared_ptr<data_batch>&&)` block until the lock is available. This is the primary acquisition path. | Low | Wraps `std::shared_lock`/`std::unique_lock` construction. Straightforward. |
| Non-blocking try variants | `try_to_read_only(shared_ptr<data_batch>&)` and `try_to_mutable(shared_ptr<data_batch>&)` attempt lock without blocking. Required by downgrade orchestrator (must not deadlock when probing batches for eviction candidates). | Medium | Uses `std::try_to_lock` tag. The mutable-reference-with-conditional-nullify semantics (DB-06) require careful implementation -- success nullifies source, failure leaves it untouched. |
| Lock-free batch ID access | `get_batch_id()` must be callable on `data_batch` (idle state) without holding any lock. Required for repository lookups and downgrade candidate selection. | Low | Already designed as `const uint64_t _batch_id` (DB-09). Immutable after construction, no synchronization needed. |
| Lock-free subscriber count | `subscribe()`, `unsubscribe()`, `get_subscriber_count()` must work without acquiring the read/write lock. These are independent of data access and used for interest tracking. | Low | Atomic operations on `data_batch` directly (DB-10). Already proven in current codebase. |
| RAII lock release on destruction | When `read_only_data_batch` or `mutable_data_batch` is destroyed, the lock must be released and the `shared_ptr<data_batch>` must be returned to idle state. This is the fundamental RAII guarantee. | Medium | In the new design, the accessor holds `shared_ptr<data_batch>` (extending lifetime). Destructor must release the shared/unique lock on the internal mutex. |
| Factory function with private constructor | `data_batch::create(...)` returns `shared_ptr<data_batch>`. Private constructor prevents stack/unique_ptr allocation that would break `enable_shared_from_this`. | Low | Use passkey idiom or nested private struct to allow `make_shared` from within the static factory while keeping constructor inaccessible to callers. |
| `enable_shared_from_this` integration | Accessors must hold `shared_ptr<data_batch>` obtained via `shared_from_this()`, guaranteeing the parent outlives the accessor. This is the core design change from the current borrow-pointer approach. | Medium | Must verify that `shared_from_this()` is only called after the `data_batch` is managed by a `shared_ptr`. The private constructor + factory pattern (DB-07) enforces this. |
| Static conversion methods on `data_batch` | All 6 transitions (DB-05) as static methods. Centralizes state machine logic in one place rather than spreading it across accessor classes. | Medium | `to_read_only(shared_ptr<data_batch>&&)`, `to_mutable(shared_ptr<data_batch>&&)`, `to_idle(read_only_data_batch&&)`, `to_idle(mutable_data_batch&&)`, plus the locked-to-locked shortcuts that go through idle internally. |
| Mutual friend relationships | `data_batch` must be friends with both accessor types and vice versa (DB-16). Accessors need to access the internal mutex and data; `data_batch` statics need to construct accessors. | Low | Standard C++ friend declarations. Three-way friendship between the three classes. |
| Accessor exposes batch metadata | `read_only_data_batch` must expose: `get_batch_id()`, `get_current_tier()`, `get_data()`, `get_memory_space()` (DB-11). `mutable_data_batch` adds `set_data()`, `convert_to<T>()` (DB-12). | Low | Direct delegation methods. Not `operator->` on an inner type -- explicit named methods on the accessor itself. |
| Clone operations on idle `data_batch` | `clone()` and `clone_to<T>()` on `data_batch` acquire read lock internally (DB-13). Must work without requiring the caller to hold a lock first. | Medium | Internally does `to_read_only(shared_from_this())`, reads data, releases, returns new `shared_ptr<data_batch>`. Must handle the case where another thread holds a mutable lock (blocks until available). |

## Differentiators

Features that improve the API beyond correctness. Not expected, but valued.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| `[[nodiscard]]` on all lock acquisition methods | Prevents the classic bug where `data_batch::to_read_only(std::move(ptr));` acquires and immediately releases the lock because the return value was discarded. This is the single most common RAII lock bug in C++. folly::Synchronized marks all lock acquisition as `[[nodiscard]]`. | Low | Add `[[nodiscard]]` attribute to all 6 static conversion methods and both try variants. Zero cost, high safety value. |
| `[[nodiscard]]` on factory `create()` | Prevents discarding the only `shared_ptr<data_batch>` returned from the factory. Discarding it would immediately destroy the batch. | Low | Same pattern as above. |
| `static_assert` nothrow move guarantees | Compile-time verification that accessor types are nothrow-moveable. Catches regressions if someone adds a throwing member. `static_assert(std::is_nothrow_move_constructible_v<read_only_data_batch>)` in the header or test file. | Low | Documents and enforces the noexcept contract. Standard practice in production concurrency code. |
| Explicit `get_size_in_bytes()` / `get_uncompressed_data_size_in_bytes()` on accessors | Convenience methods that delegate to `get_data()->get_size_in_bytes()`. Avoids callers needing to chain through the representation pointer. Useful for downgrade decision-making code that frequently queries sizes. | Low | Pure convenience. Only add if call sites consistently need this. Evaluate during implementation -- if 3+ call sites chain through `get_data()`, add the delegation. |
| Scoped accessor with callback (withLock pattern) | `data_batch::with_read_only(shared_ptr<data_batch>, Fn&&)` and `data_batch::with_mutable(shared_ptr<data_batch>, Fn&&)` -- acquire lock, call lambda, release. Prevents accidentally holding locks longer than needed. folly::Synchronized's `withLock`/`withRLock`/`withWLock` is the gold standard here. | Medium | Not in the current requirements. Evaluate as a follow-up if callers consistently acquire-use-release in tight scopes. The static method + move pattern already prevents dangling. |
| Debug-mode lock ownership assertions | In debug builds, assert that the accessor actually owns the lock before allowing data access. Catches bugs where someone `std::move`s an accessor and then tries to use the moved-from object. | Medium | `assert(lock_.owns_lock())` or similar in `operator->` / getter methods. Zero cost in release builds. Catches subtle use-after-move bugs. |
| `get_data_as<T>()` convenience template on accessor | `auto& repr = ro.get_data_as<gpu_table_representation>();` instead of `dynamic_cast<gpu_table_representation*>(ro.get_data())`. Delegates to `idata_representation::cast<T>()` which already exists. | Low | Pure ergonomic sugar. Reduces boilerplate at call sites. Already have the `cast<T>()` infrastructure on `idata_representation`. |
| Accessor `swap()` support | `std::swap(accessor_a, accessor_b)` via ADL-findable `swap()` or implicit from move operations. Enables idiomatic C++ patterns. | Low | Free if move ctor/assign are correctly implemented. Standard library's `std::swap` uses moves. No custom `swap()` needed unless profiling shows benefit. |

## Anti-Features

Features to explicitly NOT build. Each of these would introduce correctness risks, complexity, or API confusion.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Implicit conversion between accessor types | An implicit `read_only_data_batch` to `mutable_data_batch` conversion would silently upgrade a shared lock to exclusive, potentially deadlocking if other readers exist. folly::Synchronized explicitly rejects atomic shared-to-exclusive transitions for this reason: "these transitions can cause deadlocks." | All transitions go through `data_batch` static methods with explicit move semantics. The `to_mutable(read_only_data_batch&&)` releases the shared lock, goes through idle, then acquires exclusive. The gap is intentional -- it prevents deadlock. |
| Recursive locking / reentrant mutex | `std::shared_mutex` does not support recursive locking. Even if it did, recursive locks mask design problems (caller doesn't know if they already hold the lock) and make reasoning about lock ordering impossible. Raymond Chen's "anti-lock pattern" analysis shows recursive locking creates subtle bugs when temporarily dropping locks. | If a function needs data access and might be called from a context that already holds a lock, pass the accessor by reference rather than re-acquiring. Design the call graph so lock acquisition happens at well-defined boundaries. |
| `operator->` / `operator*` on accessors (pointer-like semantics) | The current codebase uses `operator->` returning `const data_batch*` / `data_batch*`. This is problematic for the new design because: (1) `data_batch` in the new design has a minimal public API (no data access methods), so there's nothing useful to dereference to; (2) pointer-like semantics suggest the accessor IS a pointer to something, which is misleading -- it IS the access interface; (3) it prevents adding accessor-specific methods cleanly. | Expose named methods directly on the accessor: `ro.get_data()`, `ro.get_current_tier()`, `rw.set_data(...)`. The accessor IS the API surface, not a proxy to something else. |
| Copy semantics on accessors | Copying a lock accessor would mean two objects sharing the same lock, or silently acquiring a second lock. Either way is a correctness bug. | Move-only. Delete copy ctor and copy assignment. This is already in the design. |
| `release()` / manual lock management | Exposing `release()` on the accessor (returning the raw `shared_ptr<data_batch>` and releasing the lock) breaks the RAII guarantee. Callers could hold a dangling reference to data after the lock is released. | Use `data_batch::to_idle(accessor&&)` which is the only way to release. Move semantics ensure the accessor is invalidated. |
| Direct locked-to-locked conversion | `to_mutable(read_only_data_batch&&)` might seem like it should atomically upgrade the lock. But `shared_mutex` does not support atomic upgrade, and attempting it causes deadlock when multiple readers try to upgrade simultaneously. | Route through idle: release shared lock, return to idle, acquire exclusive. The caller explicitly acknowledges the gap where another writer could interleave. This is documented in PROJECT.md as a deliberate design decision. |
| Default-constructible accessors | A default-constructed accessor would hold no lock and point to no data. This creates a "zombie" state that every method would need to check for, or callers could accidentally use. `std::lock_guard` is not default-constructible for the same reason. | Private constructors only. Accessors are created exclusively through `data_batch` static methods. |
| `operator bool()` on accessors | Tempting to add `operator bool()` to check if the accessor is in a valid (non-moved-from) state. But this encourages a pattern where callers test validity instead of relying on the type system. The move semantics already make moved-from objects unusable at compile time (the original variable's type doesn't change, but it holds a moved-from `shared_ptr` which is null). | Rely on move semantics to prevent use-after-move. If debug-mode assertions are needed, use `assert` in method implementations rather than a public validity check. `operator bool()` would be appropriate for `std::optional` results from try variants, which is already the design for try_to_read_only / try_to_mutable. |
| Timed lock acquisition (`try_for`, `try_until`) | `std::shared_mutex` does not support timed locking (it has no `try_lock_for` / `try_lock_until`). Even `std::shared_timed_mutex` has it, but switching mutex types adds overhead and the use case (timeout-based lock acquisition) is not needed in cuCascade's data flow. | Use the non-blocking try variants. If the lock isn't available, the caller retries or moves on. The downgrade orchestrator already uses this pattern. |
| Exposing the internal mutex | Leaking `std::shared_mutex&` from the accessor or `data_batch` would let callers bypass the RAII pattern entirely. | Keep the mutex as a private member of `data_batch`. Only the static methods and friend accessor classes interact with it. |
| Thread-safe move of `data_batch` while locked | The current codebase has a known bug: move constructor/assignment of `synchronized_data_batch` doesn't lock the mutex on the source. Trying to "fix" this by locking during move is the wrong approach -- moving a locked object is inherently unsafe. | Delete move constructor and move assignment on `data_batch`. Since `data_batch` always lives behind `shared_ptr` (enforced by factory), there is no need for move. `shared_ptr` handles ownership transfer. |

## Feature Dependencies

```
Factory (DB-07) --> enable_shared_from_this (DB-03) --> Accessor shared_ptr holding (DB-03)
  (Factory ensures shared_ptr management, which enables shared_from_this, which accessors use)

Private constructor --> Factory required
  (If constructor is private, only factory can create instances)

Static conversion methods (DB-05) --> Mutual friend relationships (DB-16)
  (Statics on data_batch construct accessors, so data_batch must be friend of accessor private ctors)

Const-correct data access --> Accessor method design (DB-11, DB-12)
  (read_only exposes const-only subset; mutable exposes full API)

Try variants (DB-06) --> Blocking variants (DB-05)
  (Try variants are the non-blocking counterpart; implement blocking first, then add try_to_lock tag)

Lock-free batch ID (DB-08) --> const uint64_t _batch_id (DB-09)
  (Immutability is what makes lock-free access safe)

Clone operations (DB-13) --> Blocking lock acquisition + enable_shared_from_this
  (Clone internally acquires read lock using shared_from_this)

[[nodiscard]] --> All acquisition methods
  (Applied after methods exist; no dependency on implementation)

static_assert noexcept --> Move operations defined
  (Assert after move ctor/assign are implemented)
```

## MVP Recommendation

Prioritize (Phase 1 -- must ship together for correctness):
1. **Factory + private constructor + enable_shared_from_this** -- Foundation for everything else
2. **data_batch class with minimal API** -- batch_id, subscriber count, mutex (all lock-free operations)
3. **read_only_data_batch with const access** -- Blocking acquisition via `to_read_only`
4. **mutable_data_batch with mutable access** -- Blocking acquisition via `to_mutable`
5. **to_idle conversions** -- Release accessors back to idle `shared_ptr<data_batch>`
6. **Move-only semantics, noexcept moves, deleted copies** -- Correctness invariants

Prioritize (Phase 1 -- should ship with MVP if possible):
7. **`[[nodiscard]]` on all acquisition methods** -- Zero-cost safety
8. **Delete move ctor/assign on data_batch itself** -- Prevents the known bug from PR #99
9. **static_assert nothrow moves** -- Compile-time regression guard

Defer to Phase 2:
- **Try variants** (DB-06) -- Important but more complex; blocking variants cover the critical path
- **Locked-to-locked via idle shortcuts** (`to_mutable(read_only_data_batch&&)`, `to_read_only(mutable_data_batch&&)`)
- **Clone operations** (DB-13)
- **`get_data_as<T>()` convenience template**

Defer to follow-up:
- **withLock callback pattern** -- Evaluate after seeing real call site patterns
- **Debug-mode assertions** -- Valuable but not blocking

## Sources

- [folly::Synchronized documentation](https://github.com/facebook/folly/blob/main/folly/docs/Synchronized.md) -- LockedPtr design, withLock pattern, explicit rejection of atomic shared-to-exclusive upgrades. HIGH confidence.
- [boost::synchronized_value docs](https://www.boost.org/doc/libs/1_78_0/doc/html/thread/sds.html) -- operator->, synchronize() scoped accessor, strict_lock_ptr. HIGH confidence.
- [LouisCharlesC/safe library](https://github.com/LouisCharlesC/safe) -- Header-only read/write wrapper combining mutexes with locks, read-only enforcement via shared locking. MEDIUM confidence.
- [dragazo/rustex](https://github.com/dragazo/rustex) -- Rust-style mutex wrapping data inside the mutex object. MEDIUM confidence.
- [mguludag/synchronized_value](https://github.com/mguludag/synchronized_value) -- Modern C++ thread-safe value wrapper with flexible locking strategies. MEDIUM confidence.
- [std::shared_mutex cppreference](https://en.cppreference.com/w/cpp/thread/shared_mutex.html) -- No recursive locking, no timed locking. HIGH confidence.
- [std::lock_guard cppreference](https://en.cppreference.com/w/cpp/thread/lock_guard.html) -- Not default-constructible, non-copyable, non-movable. HIGH confidence.
- [Raymond Chen: The anti-lock pattern](https://devblogs.microsoft.com/oldnewthing/20240814-00/?p=110129) -- Recursive locking pitfalls. HIGH confidence.
- [cppreference: [[nodiscard]]](https://en.cppreference.com/w/cpp/language/attributes/nodiscard.html) -- Preventing discarded lock acquisitions. HIGH confidence.
- [RAII, locks and clang-tidy](https://thinkingeek.com/2021/03/01/raii-locks-clang-tidy/) -- Unnamed temporary lock object bug. HIGH confidence.
- [Abseil Tip #134: make_unique and private constructors](https://abseil.io/tips/134) -- Passkey idiom for private constructor + make_shared. HIGH confidence.
- [Foonathan: Move Safety](https://www.foonathan.net/2016/07/move-safety/) -- Defining valid moved-from states. HIGH confidence.
- Existing cuCascade codebase: `include/cucascade/data/data_batch.hpp`, `src/data/data_batch.cpp`, `test/data/test_data_batch.cpp`. HIGH confidence (primary source).
