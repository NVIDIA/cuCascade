# Technology Stack: C++20 Concurrency Primitives for RAII Lock Accessors

**Project:** cuCascade data_batch Refactor (DB-01 through DB-16)
**Researched:** 2026-04-13
**Overall confidence:** HIGH -- all recommendations use C++ standard library features already proven in this codebase

## Executive Summary

The cuCascade data_batch refactor requires a specific set of C++20/C++17 concurrency primitives to implement the 3-class RAII accessor design (`data_batch`, `read_only_data_batch`, `mutable_data_batch`). The existing codebase already uses `std::shared_mutex`, `std::shared_lock`, `std::unique_lock`, and `std::enable_shared_from_this` -- the refactor extends these patterns, not introduces new ones.

The recommended stack is **100% C++ standard library**. No third-party concurrency libraries (folly, abseil, Boost) are needed or appropriate for this scope. The refactor is a structural redesign of class relationships and API surfaces, not a concurrency primitive upgrade.

## Recommended Stack

### Core Concurrency Primitives

| Technology | Header | Purpose | Why | Confidence |
|------------|--------|---------|-----|------------|
| `std::shared_mutex` | `<shared_mutex>` | Reader-writer lock protecting batch data | Already in use. Allows multiple concurrent readers (`read_only_data_batch`) with exclusive writer access (`mutable_data_batch`). Exactly matches the access pattern. | HIGH |
| `std::shared_lock<std::shared_mutex>` | `<shared_mutex>` | RAII shared (read) lock holder | Stored inside `read_only_data_batch`. Move-only. Supports `try_to_lock` constructor tag for non-blocking acquisition (DB-06). `mutex()` accessor enables lock upgrade/downgrade by extracting the mutex pointer. | HIGH |
| `std::unique_lock<std::shared_mutex>` | `<mutex>` | RAII exclusive (write) lock holder | Stored inside `mutable_data_batch`. Move-only. Supports `try_to_lock` constructor tag for non-blocking acquisition (DB-06). `mutex()` accessor enables lock downgrade by extracting the mutex pointer. | HIGH |
| `std::enable_shared_from_this<data_batch>` | `<memory>` | Allows `data_batch` to produce `shared_ptr<data_batch>` from within its own methods | Required by DB-03. Accessors hold `shared_ptr<data_batch>` obtained via `shared_from_this()`, which automatically extends parent lifetime while accessor exists. | HIGH |
| `std::atomic<size_t>` | `<atomic>` | Lock-free subscriber count | Already in use (DB-10). Independent of mutex state. `memory_order_relaxed` is correct for a simple counter with no ordering dependencies on other data. | HIGH |

### Lock Tag Types

| Tag | Header | Purpose | Why |
|-----|--------|---------|-----|
| `std::try_to_lock` | `<mutex>` | Non-blocking lock attempt for `try_to_read_only` and `try_to_mutable` (DB-06) | Passed to `shared_lock`/`unique_lock` constructor. Returns immediately, caller checks `owns_lock()`. Already used in current codebase. |
| `std::defer_lock` | `<mutex>` | Create lock wrapper without acquiring lock | **Not needed.** All our constructors either lock immediately or use `try_to_lock`. Listed for completeness only. |
| `std::adopt_lock` | `<mutex>` | Wrap an already-held lock | **Not needed.** We never acquire a raw lock and then wrap it. Listed for completeness only. |

### Smart Pointer Infrastructure

| Technology | Header | Purpose | Why | Confidence |
|------------|--------|---------|-----|------------|
| `std::shared_ptr<data_batch>` | `<memory>` | Ownership of `data_batch` instances throughout the system | Required by DB-03, DB-07, DB-14. `enable_shared_from_this` mandates shared_ptr management. Move semantics on `shared_ptr&&` enforce the state machine transitions (DB-04, DB-05). | HIGH |
| `std::make_shared<data_batch>` | `<memory>` | **Cannot use directly** -- private constructor | See Passkey Idiom below. `data_batch::create()` factory (DB-07) must use a workaround since `make_shared` cannot call private constructors. | HIGH |
| `std::optional<T>` | `<optional>` | Return type for try-variant methods (DB-06) | `try_to_read_only` returns `std::optional<read_only_data_batch>`, `try_to_mutable` returns `std::optional<mutable_data_batch>`. `std::nullopt` signals lock acquisition failure. Already used in current codebase. | HIGH |

### C++20 Features Relevant to This Design

| Feature | Available | Use in This Refactor | Confidence |
|---------|-----------|---------------------|------------|
| Concepts | Yes (C++20) | **Optional.** Could constrain template parameters on clone_to, convert_to. Not critical path. | MEDIUM |
| `std::atomic<std::shared_ptr<T>>` | Yes (C++20) | **Do not use.** The shared_ptr to data_batch is not concurrently modified -- ownership transfers happen through move semantics in static methods, not through atomic CAS on a shared pointer. | HIGH |
| `[[nodiscard]]` | Yes (C++17) | **Use on all factory methods and try variants.** Discarding a `shared_ptr<data_batch>` from `create()`, or an `optional` from `try_to_read_only()`, is always a bug. | HIGH |
| `[[maybe_unused]]` | Yes (C++17) | Not needed for this design. | N/A |
| Designated initializers | Yes (C++20) | Not applicable (no aggregate types in this design). | N/A |
| Three-way comparison | Yes (C++20) | Not needed. `data_batch` identity is by `batch_id`, not by value comparison. | N/A |

## Critical Implementation Patterns

### 1. The Passkey Idiom for Private Constructor + make_shared (DB-07)

`std::make_shared` cannot call private constructors. Since `enable_shared_from_this` requires `shared_ptr` management, we need a factory function. The standard solution is the **passkey idiom**:

```cpp
class data_batch : public std::enable_shared_from_this<data_batch> {
 public:
  // Passkey: public type, private constructor
  struct create_token {
   private:
    friend class data_batch;
    create_token() = default;
  };

  // Constructor is technically public but uncallable outside data_batch
  // because only data_batch can construct create_token
  explicit data_batch(create_token, uint64_t batch_id,
                      std::unique_ptr<idata_representation> data);

  // Factory: the only public way to create a data_batch
  [[nodiscard]] static std::shared_ptr<data_batch> create(
    uint64_t batch_id, std::unique_ptr<idata_representation> data);
};
```

**Why passkey over alternatives:**
- `std::shared_ptr<data_batch>(new data_batch(...))` works but requires two allocations (one for object, one for control block). `make_shared` does a single allocation.
- Derived-class trick (`struct DerivedHack : data_batch { ... }`) is fragile, breaks if data_batch becomes final, and doesn't work with `enable_shared_from_this` correctly.
- Passkey is the cleanest: single allocation via `make_shared`, no derived class hacks, constructor is effectively private.

**Confidence:** HIGH -- this is a well-known C++ pattern. The existing `notification_channel` in this codebase uses `enable_shared_from_this` but with a public constructor; the passkey idiom adds enforcement.

### 2. Move-Semantic State Transitions (DB-04, DB-05)

The state machine uses `&&` (rvalue reference) parameters to enforce single-use transitions:

```cpp
// Idle -> Read: consumes the shared_ptr, caller cannot use it after
[[nodiscard]] static read_only_data_batch to_read_only(
  std::shared_ptr<data_batch>&& idle_batch);

// Read -> Idle: consumes the accessor, returns the batch
[[nodiscard]] static std::shared_ptr<data_batch> to_idle(
  read_only_data_batch&& reader);
```

After `std::move(batch)` into `to_read_only`, the caller's `shared_ptr` is null. This is the C++ equivalent of Rust's ownership transfer -- the compiler cannot enforce "don't use after move" (that's a runtime null), but combined with `[[nodiscard]]` and the type system (you can't call read methods on a `shared_ptr<data_batch>`), it achieves practical compile-time safety.

**Key insight:** The moved-from `shared_ptr` is guaranteed to be null (not just unspecified) per the C++ standard for `shared_ptr`'s move constructor. This is stronger than the general "valid but unspecified" moved-from guarantee.

**Confidence:** HIGH -- `shared_ptr` move-from-null guarantee is specified in [util.smartptr.shared.const].

### 3. Try Variants with Mutable Reference (DB-06)

The try variants use `&` (lvalue reference) instead of `&&` because the transition is conditional:

```cpp
// On success: *idle_batch is nullified (moved-from), returns accessor
// On failure: *idle_batch is unchanged, returns nullopt
[[nodiscard]] static std::optional<read_only_data_batch> try_to_read_only(
  std::shared_ptr<data_batch>& idle_batch);
```

Implementation uses `std::shared_lock<std::shared_mutex>(mtx, std::try_to_lock)` then checks `owns_lock()`. On success, move out of `idle_batch`; on failure, return `std::nullopt` with `idle_batch` untouched.

**Confidence:** HIGH -- this pattern is already implemented in the current codebase's `try_get_read_only`/`try_get_mutable`.

### 4. Lock Upgrade/Downgrade Through Idle (DB-05)

The design explicitly does NOT support direct read-to-write upgrade. All transitions go through the idle state:

```cpp
// read_only -> mutable: release shared lock, return to idle, acquire exclusive
static mutable_data_batch to_mutable(read_only_data_batch&& reader) {
  auto batch = to_idle(std::move(reader));  // releases shared lock
  return to_mutable(std::move(batch));       // acquires exclusive lock
}
```

**Why no direct upgrade:** `std::shared_mutex` does not support atomic upgrade from shared to exclusive. Attempting `unlock_shared()` then `lock()` creates a window where another writer can interpose. The current codebase's `from_read_only` and `from_mutable` already do exactly this (unlock, re-lock), but the new design makes the intermediate idle state explicit through the type system rather than hiding it inside the accessor.

**Confidence:** HIGH -- verified from current `data_batch.cpp` lines 72-78 where `from_read_only` does `ro.lock_.unlock()` then `std::unique_lock<std::shared_mutex> lock(*mtx)`.

### 5. Accessor Lifetime Extension via shared_from_this (DB-03)

```cpp
class read_only_data_batch {
 private:
  std::shared_ptr<data_batch> batch_;      // extends parent lifetime
  std::shared_lock<std::shared_mutex> lock_;  // holds the read lock
};
```

The accessor stores a `shared_ptr<data_batch>` obtained from `batch->shared_from_this()` inside the static `to_read_only` method. This is the key improvement over the current design where accessors hold a raw pointer (`synchronized_data_batch* parent_`) that can dangle.

**Precondition for shared_from_this():** The object must already be managed by a `shared_ptr` when `shared_from_this()` is called. Since all `data_batch` objects are created via `data_batch::create()` which returns `shared_ptr<data_batch>`, and the static methods take `shared_ptr<data_batch>` parameters, this precondition is always satisfied.

**Confidence:** HIGH -- `enable_shared_from_this` is well-specified and the factory pattern (DB-07) guarantees the precondition.

## What NOT to Use (and Why)

### Primitives to Avoid

| Primitive | Why NOT | What to Use Instead |
|-----------|---------|-------------------|
| `std::recursive_mutex` | Hides poor design. The RAII accessor pattern makes re-entrancy unnecessary -- if you hold a `mutable_data_batch`, you already have exclusive access. Recursive mutex adds overhead and masks bugs where code accidentally re-locks. | `std::shared_mutex` with RAII accessors that make lock state visible in the type system. |
| `std::mutex` (for data_batch) | Does not support concurrent readers. The caching use case is read-heavy (multiple consumers reading batch data simultaneously). A plain mutex would serialize all access unnecessarily. | `std::shared_mutex` for the read/write distinction. `std::mutex` remains appropriate for the repository and converter registry where operations are short and exclusive. |
| `std::atomic<std::shared_ptr<T>>` | Overkill. The `shared_ptr<data_batch>` is not concurrently modified by multiple threads through the same variable. Ownership transfers happen through move semantics in single-threaded contexts (the caller moves their ptr into a static method). The mutex protects the data, not the pointer. | `std::shared_ptr<data_batch>` with move semantics. |
| `std::lock_guard` (for data_batch accessors) | Cannot be moved. Accessors need to be returned from factory functions, which requires move semantics. `lock_guard` is non-movable by design. | `std::shared_lock` (movable) and `std::unique_lock` (movable). `lock_guard` remains appropriate for simple scoped locking in repository/converter code. |
| `std::scoped_lock` | Designed for locking multiple mutexes simultaneously (deadlock avoidance). This design uses a single `shared_mutex` per `data_batch`. Also non-movable. | `std::shared_lock` / `std::unique_lock`. |
| `std::condition_variable_any` | No wait/notify pattern needed for lock acquisition. The try variants return immediately; blocking variants simply block on mutex acquisition. | `std::try_to_lock` tag for non-blocking attempts. |
| Lock-free data structures | The data being protected (a `unique_ptr<idata_representation>` pointing to GPU table data) is inherently non-atomic. Lock-free techniques apply to integers, pointers, and small structs, not to complex data ownership transfers. | `std::shared_mutex` with RAII accessors. The subscriber count (a single integer) correctly uses `std::atomic<size_t>`. |

### Third-Party Libraries to Avoid

| Library | Why NOT for This Project |
|---------|--------------------------|
| `folly::Synchronized<T>` | Excellent library, but adds a large dependency (Facebook Folly) for one feature. The cuCascade data_batch design is more specialized than Synchronized's generic callback-based access -- we need typed accessor classes with specific API surfaces (DB-11, DB-12), not generic `withRLock`/`withWLock` callbacks. Also, Folly's CMake integration with CUDA toolchains is non-trivial. |
| `absl::Mutex` | Provides `ReaderMutexLock`/`WriterMutexLock` RAII types and deadlock detection, but the locking primitives aren't the bottleneck -- the class structure redesign is. Adding Abseil as a dependency for mutex types alone is unjustified when `std::shared_mutex` already works in this codebase. Abseil's `Mutex` also lacks the `mutex()` accessor on its lock guards, which we need for the upgrade/downgrade pattern. |
| `LouisCharlesC/safe` | Header-only read/write mutex wrapper. Conceptually similar to what we're building, but generic -- we need domain-specific accessor types that expose specific methods (get_data, set_data, convert_to), not generic pointer-like access. The safe library would add indirection without benefit. |
| `dragazo/rustex` | Rust-style mutex wrapper for C++. Interesting pattern but the cuCascade design already achieves the same "data-protecting mutex" goal through the 3-class design. Adding a dependency for style alone is wrong. |
| `boost::shared_mutex` / `boost::upgrade_mutex` | Boost's `upgrade_mutex` supports atomic shared-to-exclusive upgrade, which we explicitly chose not to use (all transitions go through idle). Adding Boost for an unused feature is wrong. The project has zero Boost dependencies today. |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Mutex type | `std::shared_mutex` | `std::mutex` | No concurrent reader support |
| Mutex type | `std::shared_mutex` | `boost::upgrade_mutex` | Atomic upgrade not needed; no Boost dependency today |
| Lock guard (read) | `std::shared_lock` | `std::lock_guard` | Not movable; can't return from factory |
| Lock guard (write) | `std::unique_lock` | `std::lock_guard` | Not movable; can't return from factory |
| Lifetime extension | `enable_shared_from_this` | Raw pointer borrowing | Dangling pointer risk (current design's weakness) |
| Factory enforcement | Passkey idiom + `make_shared` | `shared_ptr(new T)` | Two allocations instead of one |
| Factory enforcement | Passkey idiom + `make_shared` | Derived class trick | Fragile; breaks with `final`; `enable_shared_from_this` mismatch |
| Non-blocking API | `std::optional<accessor>` | Error codes / exceptions | Optional is idiomatic for "might not have a value" |
| Concurrency library | C++ stdlib only | folly::Synchronized | Unnecessary large dependency for this scope |
| Subscriber count | `std::atomic<size_t>` | Mutex-protected `size_t` | Atomic is lock-free and independent of data access lock |

## CUDA Toolchain Compatibility

The data layer (`include/cucascade/data/`, `src/data/`) is **pure host C++ code** -- no `.cu` files, no `__device__` annotations, no CUDA kernels. All source files are `.cpp` and compiled by the host compiler (GCC/Clang) through CMake, not by nvcc's device compiler.

All recommended primitives (`std::shared_mutex`, `std::shared_lock`, `std::unique_lock`, `std::enable_shared_from_this`, `std::atomic`) are C++17 standard library features compiled by the host compiler. They have zero interaction with the CUDA toolchain. The cuCascade build already uses these features successfully (verified: `shared_mutex` in current `data_batch.hpp`, `enable_shared_from_this` in `notification_channel.hpp`, `std::atomic` in current `synchronized_data_batch`).

**No CUDA compatibility risk exists for this refactor.**

**Confidence:** HIGH -- verified by examining the build system (no `.cu` files in `src/data/`), source code (no device annotations), and current successful usage of all required primitives.

## Memory Ordering for Atomic Subscriber Count

The existing code uses `std::memory_order_relaxed` for all subscriber count operations. This is correct because:

1. The subscriber count has no ordering relationship with the data protected by the `shared_mutex`
2. No code reads the subscriber count and then makes a decision that requires seeing specific data state
3. The count is a standalone metric for interest tracking

If future code needs to make decisions like "unsubscribe and if count reaches zero, destroy data," the ordering would need to be upgraded to `memory_order_acq_rel`. But per DB-10 and the current design, `relaxed` is correct.

**Confidence:** HIGH -- the semantics haven't changed from the current implementation.

## std::shared_mutex Performance Characteristics

`std::shared_mutex` implementations on Linux (glibc/libstdc++) have **reader-preference** behavior: if readers continuously hold shared locks, writers can starve. This is relevant to cuCascade because the caching use case could have many readers with infrequent writers.

**For this project, writer starvation is acceptable because:**
1. Write operations (set_data, convert_to) are initiated by the caching system, not by external user requests
2. The caching system can retry or queue writes
3. Lock hold times are short (metadata access, pointer swaps) -- long operations like GPU data conversion happen before acquiring the lock
4. The try variants (DB-06) provide a non-blocking escape hatch

**If writer starvation becomes a problem in production,** the mitigation is a fair reader-writer lock (e.g., `yamc::fair::shared_mutex` or a custom turnstile). This would be a drop-in replacement since the accessor types only depend on the `SharedMutex` concept interface, not on `std::shared_mutex` specifically. Templating the mutex type is a future option but should NOT be done now (YAGNI).

**Confidence:** MEDIUM -- performance characteristics depend on workload; the analysis is based on typical caching patterns.

## Installation / Dependencies

No new dependencies are required. All recommended technologies are part of the C++ standard library available in C++20 mode.

```bash
# No new packages needed. Existing build command works:
pixi run build

# Required headers (all already included in current codebase):
# <shared_mutex>   - std::shared_mutex, std::shared_lock
# <mutex>          - std::unique_lock, std::lock_guard, std::try_to_lock
# <memory>         - std::shared_ptr, std::make_shared, std::enable_shared_from_this
# <atomic>         - std::atomic
# <optional>       - std::optional
```

## Sources

- [cppreference: std::shared_mutex](https://en.cppreference.com/w/cpp/thread/shared_mutex.html) -- Official C++ reference for shared_mutex API
- [cppreference: std::shared_lock](https://en.cppreference.com/w/cpp/thread/shared_lock.html) -- shared_lock constructors and move semantics
- [cppreference: std::enable_shared_from_this](https://en.cppreference.com/w/cpp/memory/enable_shared_from_this.html) -- Preconditions and pitfalls
- [cppreference: Lock tags (defer_lock, try_to_lock, adopt_lock)](https://en.cppreference.com/w/cpp/thread/lock_tag_t.html) -- Lock strategy tags
- [ACCU: Thread-Safe Access Guards](https://accu.org/journals/overload/19/104/reese_1967/) -- Accessor guard pattern theory
- [MutexProtected: A C++ Pattern for Easier Concurrency](https://awesomekling.github.io/MutexProtected-A-C++-Pattern-for-Easier-Concurrency/) -- Data-protecting mutex pattern
- [Mutexes in Rust and C++: Protecting Data versus Protecting Code](https://geo-ant.github.io/blog/2020/mutexes-rust-vs-cpp/) -- Rust vs C++ mutex philosophy comparison
- [folly::Synchronized documentation](https://github.com/facebook/folly/blob/main/folly/docs/Synchronized.md) -- Facebook's synchronized wrapper
- [abseil synchronization guide](https://abseil.io/docs/cpp/guides/synchronization) -- Google's mutex documentation
- [abseil Mutex design notes](https://abseil.io/about/design/mutex) -- absl::Mutex design rationale
- [LouisCharlesC/safe library](https://github.com/LouisCharlesC/safe) -- Header-only mutex wrapper
- [Embedded Artistry: shared_ptr and shared_from_this](https://embeddedartistry.com/blog/2017/01/11/stdshared_ptr-and-shared_from_this/) -- enable_shared_from_this pitfalls
- [Abseil Tip #134: make_unique and private constructors](https://abseil.io/tips/134) -- Passkey idiom for factory functions
- [CUDA C++ Programming Guide: C++ Language Support](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html) -- nvcc C++20 support
- [Benchmarking Reader-Writer Lock Performance](https://turingcompl33t.github.io/RWLock-Benchmark/) -- shared_mutex performance data
- [Readers-writer lock (Wikipedia)](https://en.wikipedia.org/wiki/Readers%E2%80%93writer_lock) -- Writer starvation theory

---

*Stack research: 2026-04-13*
