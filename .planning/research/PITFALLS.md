# Domain Pitfalls

**Domain:** C++ concurrent data accessor with shared_mutex + enable_shared_from_this + RAII lock guards
**Project:** cuCascade data_batch refactor (PR #99 v2)
**Researched:** 2026-04-13

## Critical Pitfalls

Mistakes that cause undefined behavior, data corruption, or deadlocks.

### Pitfall 1: Destroying the Mutex While Accessors Hold Locks

**What goes wrong:** The `data_batch` owns the `shared_mutex`. If the last `shared_ptr<data_batch>` is destroyed while a `read_only_data_batch` or `mutable_data_batch` still holds a `shared_lock`/`unique_lock` on that mutex, the mutex is destroyed while locked. This is undefined behavior per the C++ standard ([CON50-CPP](https://wiki.sei.cmu.edu/confluence/display/cplusplus/CON50-CPP.+Do+not+destroy+a+mutex+while+it+is+locked)).

**Why it happens:** The new design has accessors hold `shared_ptr<data_batch>` to extend parent lifetime (DB-03). But if the implementation stores the `shared_ptr` in the accessor but the lock references a mutex *inside* the `data_batch`, the reference count keeps the `data_batch` alive -- this is correct. The pitfall arises if someone bypasses the `shared_ptr` mechanism. For example: storing a raw `data_batch*` in the accessor (the current bug), or if the accessor's `shared_ptr` is accidentally reset/moved before the lock guard's destructor runs.

**Consequences:** Undefined behavior. On Linux (glibc/pthreads), destroying a locked `pthread_rwlock_t` can corrupt futex state, causing hangs or crashes in unrelated threads. On MSVC (SRWLOCK), behavior is similarly undefined.

**Prevention:**
1. Accessors MUST store `shared_ptr<data_batch>` (not raw pointer) as their first member. C++ destroys members in reverse declaration order, so the lock guard (declared after the `shared_ptr`) is destroyed first, releasing the lock, then the `shared_ptr` releases its reference.
2. Member declaration order in accessor classes must be: `shared_ptr<data_batch>` first, then `shared_lock`/`unique_lock` second. This is load-bearing and must be documented with a comment.
3. Delete or `= delete` the accessor's copy/move assignment operators that could accidentally reset the `shared_ptr` while the lock is still held. Move construction is fine (both move together), but assignment to a live accessor is dangerous.

**Detection:** Thread sanitizer (TSan) will flag use-after-destroy. Code review must verify member declaration order. A static analysis rule checking that the `shared_ptr` member precedes the lock member would catch ordering mistakes.

**Phase:** Core implementation phase (DB-03). This is the most fundamental invariant of the new design.

**Confidence:** HIGH -- well-documented UB in C++ standard and CERT guidelines.

---

### Pitfall 2: Non-Atomic Lock Upgrade (Shared-to-Exclusive) Creates an Unlocked Window

**What goes wrong:** `std::shared_mutex` does not support atomic lock upgrade from shared to exclusive. The `to_mutable(read_only_data_batch&&)` conversion must release the shared lock, then acquire an exclusive lock. During this window (current code: `data_batch.cpp:72-78`), the data can be mutated by another thread, or the batch can be moved/destroyed.

**Why it happens:** This is a fundamental limitation of `std::shared_mutex` -- if two threads both try to atomically upgrade shared-to-exclusive, deadlock is guaranteed (both hold shared, both wait for exclusive, neither can proceed). The C++ standard deliberately omits upgrade semantics. Boost and Facebook's Folly provide `upgrade_mutex` with a third ownership mode, but `std::shared_mutex` does not.

**Consequences:** The current code already has this bug (documented in CONCERNS.md). In the new design, routing through idle (`to_idle(read_only&&)` then `to_mutable(shared_ptr&&)`) makes the window explicit in the API. But callers who assume the data hasn't changed between "was reading" and "now writing" will have correctness bugs. This is the "time-of-check to time-of-use" (TOCTOU) problem.

**Prevention:**
1. The new API correctly forces this through idle (DB-05: `to_mutable(read_only_data_batch&&)` releases shared, acquires exclusive "through idle"). This makes the unlocked window visible in the type system -- the caller gets back a `shared_ptr<data_batch>` (idle) and must explicitly re-acquire mutable.
2. Document clearly that data may have changed after the round-trip. Callers must re-validate any state they observed during the read-only phase.
3. Do NOT provide a "direct upgrade" API. The temptation is strong but the semantics are impossible to get right with `std::shared_mutex`.
4. If truly atomic upgrade is needed in the future, consider Boost's `upgrade_mutex` or `folly::SharedMutex` which provide a third "upgrade" ownership level.

**Detection:** Code review for any caller pattern that reads state under shared lock, releases, acquires exclusive, and assumes state is unchanged. This is a logic bug, not a crash, so TSan won't help.

**Phase:** API design phase (DB-05). The conversion method signatures must encode this constraint.

**Confidence:** HIGH -- fundamental `std::shared_mutex` limitation, documented in WG21 papers (N3427, N3568).

---

### Pitfall 3: Calling shared_from_this() Before shared_ptr Manages the Object

**What goes wrong:** `enable_shared_from_this::shared_from_this()` throws `std::bad_weak_ptr` if called when the object is not yet managed by any `shared_ptr`. The internal `weak_ptr` is initialized by the `shared_ptr` constructor, not by the object's own constructor. This means `shared_from_this()` is illegal inside the `data_batch` constructor body, in any method called from the constructor, or if someone constructs a `data_batch` on the stack or via `unique_ptr`.

**Why it happens:** `enable_shared_from_this` stores a `weak_ptr<T>` that is set by `shared_ptr<T>`'s constructor via a special friendship. Until a `shared_ptr` is constructed from the raw `T*`, the `weak_ptr` is empty. Calling `shared_from_this()` on an empty `weak_ptr` throws.

**Consequences:** Runtime crash (`std::bad_weak_ptr` exception). Particularly insidious because it compiles fine -- the error is purely runtime.

**Prevention:**
1. Private constructor + static `data_batch::create(...)` factory returning `shared_ptr<data_batch>` (DB-07). This is already in the design and is the correct approach.
2. Use the passkey idiom to make the constructor technically public (for `make_shared` compatibility) but unconstructable without a private token. This allows `make_shared` (single allocation) instead of `shared_ptr(new T(...))` (two allocations).
3. `= delete` any constructor that takes `unique_ptr<data_batch>` or raw `data_batch*` on consuming APIs. The repository should only accept `shared_ptr<data_batch>`.
4. Never call `shared_from_this()` in the constructor body. If initialization needs a `shared_ptr` to `this`, use a two-phase init or a post-construction callback from the factory.

**Detection:** Runtime exception at point of misuse. Unit tests that construct `data_batch` via the factory and immediately call accessor methods will catch regressions. A test that tries to construct `data_batch` directly (if possible) should fail to compile.

**Phase:** Core implementation phase (DB-07, DB-03). Factory must exist before accessors can call `shared_from_this()`.

**Confidence:** HIGH -- documented on cppreference, Raymond Chen's blog, and every `enable_shared_from_this` tutorial.

---

### Pitfall 4: enable_shared_from_this operator= Is a No-Op -- Move/Copy Assignment Corrupts Internal weak_ptr

**What goes wrong:** `enable_shared_from_this::operator=` deliberately does nothing -- it returns `*this` without modifying the internal `weak_ptr`. This means if you write a move assignment operator for `data_batch` that moves the base class (`enable_shared_from_this` subobject), the target's internal `weak_ptr` still points to the *target* object (correct), but the source's `weak_ptr` is unchanged (still points to source, but source is now in a moved-from state). This is safe as long as you never call `shared_from_this()` on the moved-from source. However, if you write `data_batch& operator=(data_batch&& other)` and the compiler-generated version calls `enable_shared_from_this::operator=(std::move(other))`, it does nothing -- which is the *correct* behavior. The pitfall is if someone manually implements assignment and tries to "fix" the `weak_ptr`, or if they don't understand why it's a no-op.

**Why it happens:** `enable_shared_from_this::operator=` is a no-op by design. The `weak_ptr` is not a data member you control -- it's managed by `shared_ptr`. If assignment copied it, the target would claim to be the source, which would be wrong.

**Consequences:** If someone manually moves the `enable_shared_from_this` subobject (e.g., via placement new or memcpy), `shared_from_this()` on the target will return a `shared_ptr` to the wrong object or throw `bad_weak_ptr`.

**Prevention:**
1. `= delete` move constructor and move assignment on `data_batch`. Since `data_batch` is always managed by `shared_ptr` and state transitions are done via static methods that take `shared_ptr<data_batch>&&` (moving the pointer, not the object), there is no need to move the object itself.
2. If move must exist for some reason, do NOT manually implement it. Let the compiler generate it (it will call `enable_shared_from_this::operator=` which correctly does nothing to the `weak_ptr`).
3. Document that the `data_batch` object itself is immovable. Ownership transfer is done by moving `shared_ptr<data_batch>`, not by moving `data_batch`.

**Detection:** Code review. If `data_batch` has `= delete` on move ops, the compiler catches attempts. If move ops exist, test that `shared_from_this()` works correctly after move.

**Phase:** Core implementation phase (DB-04, DB-07). Must be decided when defining `data_batch` class.

**Confidence:** HIGH -- cppreference documents this explicitly.

---

### Pitfall 5: Self-Deadlock from Recursive Locking (Clone/Convert While Holding Lock)

**What goes wrong:** `std::shared_mutex` is not recursive. If a thread holds a shared lock (via `read_only_data_batch`) and then calls a method that internally acquires a shared lock (e.g., `clone()` at line 165 of current code), behavior is implementation-defined: on some platforms it succeeds (shared locks are re-entrant), on others it deadlocks. If a thread holds a shared lock and calls something that acquires an exclusive lock, it always deadlocks. If a thread holds an exclusive lock and calls anything that acquires any lock on the same mutex, it always deadlocks.

**Why it happens:** The new design has `clone()` (DB-13) acquiring a read lock internally. If a caller already holds a `read_only_data_batch` on the same `data_batch` and then calls `clone()`, the thread attempts to acquire a second shared lock on the same mutex. On glibc's `pthread_rwlock_t`, this may succeed (implementation detail, not guaranteed). On MSVC's SRWLOCK, this deadlocks. The C++ standard says behavior is undefined if a thread that already owns a lock calls `lock()` or `lock_shared()` again.

**Consequences:** Deadlock on some platforms, apparently works on others. Cross-platform portability bug.

**Prevention:**
1. `clone()` and `clone_to()` should NOT acquire locks internally in the new design. Instead, they should be methods on `read_only_data_batch` (which already holds the shared lock) or take the accessor as a parameter.
2. Alternatively, make `clone()` a static method on `data_batch` that takes a `read_only_data_batch&` -- the lock is already held by the accessor.
3. Never have a public API that acquires a lock internally if an accessor (which also holds a lock) might call it. Audit all methods for hidden re-locking.
4. Document thread-safety guarantees per-method: "This method acquires [no lock / shared lock / exclusive lock] internally."

**Detection:** TSan can detect some recursive lock attempts. Testing on MSVC (or Windows) will catch deadlocks that succeed on Linux. A code audit rule: "any method that acquires `_rw_mutex` must not be callable from code paths that already hold a lock on the same mutex."

**Phase:** API design phase (DB-13, DB-11, DB-12). Must decide where clone lives before implementing accessors.

**Confidence:** HIGH -- `std::shared_mutex` non-recursive guarantee is in the standard; MSVC/glibc behavioral difference is documented.

---

### Pitfall 6: try_to_* Variants Have a Data Race on the shared_ptr If Called Concurrently

**What goes wrong:** The proposed `try_to_read_only(shared_ptr<data_batch>&)` and `try_to_mutable(shared_ptr<data_batch>&)` (DB-06) take a mutable reference to a `shared_ptr` and nullify it on success. If two threads call `try_to_read_only()` on the same `shared_ptr<data_batch>` simultaneously, both threads read and potentially write the same `shared_ptr` object without synchronization. This is a data race on the `shared_ptr` control block.

**Why it happens:** `shared_ptr` operations on the *same instance* are not thread-safe. The reference count is atomic, but the stored pointer and the `shared_ptr` object itself are not. Two threads doing `std::move(ptr)` on the same `shared_ptr` is undefined behavior.

**Consequences:** Use-after-free, double-free, or corrupted reference count. Crashes under load.

**Prevention:**
1. Document that `try_to_*` methods are NOT thread-safe on the `shared_ptr` argument. The caller must ensure exclusive access to the `shared_ptr` they pass in (e.g., it's a local variable, or protected by the repository's mutex).
2. In practice, the repository's `pop_data_batch()` already returns a `shared_ptr` under a lock. The caller should receive that `shared_ptr` exclusively and then call `try_to_mutable()` on their local copy. This is safe because each thread has its own `shared_ptr` instance.
3. Consider `atomic<shared_ptr<data_batch>>` if truly concurrent access to the same `shared_ptr` is needed. But this adds overhead and is usually the wrong design.
4. An alternative API: `try_to_read_only(shared_ptr<data_batch>&&)` (rvalue reference, forces caller to move in) makes the ownership transfer explicit and prevents accidental concurrent use of the same `shared_ptr` lvalue.

**Detection:** TSan will flag the data race. Test with multiple threads calling `try_to_read_only` on the same `shared_ptr` instance.

**Phase:** API design phase (DB-06). Must decide the reference semantics before implementing.

**Confidence:** HIGH -- `shared_ptr` thread-safety rules are explicit in the standard: same instance, non-const operations, no synchronization = data race.

---

## Moderate Pitfalls

### Pitfall 7: Writer Starvation Under Continuous Read Load

**What goes wrong:** If multiple threads continuously hold `read_only_data_batch` accessors, a thread trying to acquire `mutable_data_batch` (exclusive lock) may wait indefinitely. As long as at least one shared lock is held, the exclusive lock cannot be acquired. If new readers keep arriving before all existing readers release, the writer starves.

**Why it happens:** The C++ standard does not specify fairness for `std::shared_mutex`. On Linux (glibc), `pthread_rwlock_t` is typically writer-preferring (new readers block if a writer is waiting), which mitigates this. On MSVC (SRWLOCK), behavior is similar but with a known bug (microsoft/STL#4448) where shared ownership can be granted when exclusive was requested.

**Prevention:**
1. Design the system so read-only accessors are short-lived. Do not hold `read_only_data_batch` across CUDA stream synchronization, network I/O, or other blocking operations.
2. For the tiering/conversion use case, acquire mutable, do the conversion, release. Don't hold a read lock "in case you might need to convert later."
3. Consider a timeout on exclusive lock acquisition using `std::shared_timed_mutex` if writer starvation becomes a production issue. The API would return `optional<mutable_data_batch>` with a timeout.
4. Monitor lock wait times in production. Add metrics for time spent waiting in `to_mutable()`.

**Detection:** Performance testing under mixed read/write load. If mutable acquisition latency grows with reader count, writer starvation is occurring.

**Phase:** Performance testing phase (after core implementation). Not a correctness bug but a liveness issue.

**Confidence:** MEDIUM -- platform-dependent; Linux's writer-preferring default mitigates this, but MSVC behavior is less predictable.

---

### Pitfall 8: shared_mutex Performance Overhead for Write-Heavy Workloads

**What goes wrong:** `std::shared_mutex` has higher overhead than `std::mutex` due to managing shared/exclusive state. For workloads that are primarily writes (data mutations, tier conversions), using `shared_mutex` is strictly worse than `std::mutex` -- you pay the overhead of reader/writer discrimination but never benefit from concurrent reads.

**Why it happens:** `shared_mutex` must track the number of shared lock holders (typically via an atomic counter or futex word), perform memory barriers on shared lock acquire/release, and handle the shared-to-exclusive transition. A plain `mutex` does none of this.

**Consequences:** 2-5x slower lock acquisition in write-heavy scenarios compared to `std::mutex`. Benchmarks from C++ Stories (2026) and TuringCompl33t confirm this: with few readers, `shared_mutex` performs worse than `mutex`.

**Prevention:**
1. Measure the actual read/write ratio in cuCascade's usage patterns. If most access is through `mutable_data_batch` (tier conversions, data updates), `shared_mutex` may not be the right primitive.
2. Profile lock acquisition latency in realistic workloads (not micro-benchmarks).
3. If the pattern is "one writer, zero or one reader," switch to `std::mutex` with `unique_lock` only. The accessor pattern still works -- `read_only_data_batch` just holds a `unique_lock` instead of `shared_lock` and provides a const view.
4. The current design is correct for the documented use case (multiple concurrent readers, infrequent writes). Validate this assumption.

**Detection:** Benchmark `shared_mutex` vs `mutex` with cuCascade's actual contention patterns.

**Phase:** Performance optimization phase (after correctness is verified).

**Confidence:** MEDIUM -- depends on actual workload. The shared_mutex choice is architecturally sound if reads dominate.

---

### Pitfall 9: Accessor Escaping Its Scope via shared_ptr Prevents Lock Release

**What goes wrong:** Since accessors hold `shared_ptr<data_batch>`, they keep the batch alive. But they also hold a lock. If an accessor is stored in a long-lived data structure (e.g., captured in a lambda, stored in a container, passed to another thread), the lock is held for the entire lifetime of that container or lambda. This silently turns a "scoped lock" into a "held-forever lock."

**Why it happens:** The RAII pattern means the lock lives as long as the accessor object. With `shared_ptr` lifetime extension, there's no "owner" that forces the accessor to be destroyed promptly. A `read_only_data_batch` stored in a `std::vector` keeps its shared lock until the vector is cleared.

**Consequences:** Other threads cannot acquire exclusive locks. System appears deadlocked but is actually just holding locks too long. Extremely difficult to debug because the lock holder is not at a blocking call -- it's just a live object somewhere.

**Prevention:**
1. Make accessors non-copyable (already in design -- move-only). This prevents accidental copies in containers.
2. Consider making accessors `[[nodiscard]]` to warn if they're created and immediately discarded (accidental unlock).
3. Document the expected usage pattern: create accessor, use it, let it go out of scope. Never store accessors in member variables or long-lived containers.
4. In code review, flag any `read_only_data_batch` or `mutable_data_batch` that appears as a class member or is stored in a container.
5. Consider adding a `release()` method that explicitly releases the lock and invalidates the accessor, for cases where scope-based lifetime doesn't match usage.

**Detection:** Code review for accessor storage patterns. Long lock-hold times in profiling. TSan may report lock held across suspension points (if using coroutines).

**Phase:** API documentation and code review guidelines (ongoing).

**Confidence:** HIGH -- this is a well-known antipattern with RAII lock guards.

---

### Pitfall 10: Member Declaration Order in Accessors Determines Destruction Order

**What goes wrong:** C++ destroys class members in reverse declaration order. If an accessor declares the lock *before* the `shared_ptr`, the `shared_ptr` is destroyed first (releasing the reference), potentially destroying the `data_batch` and its mutex, and then the lock guard tries to unlock a destroyed mutex.

**Why it happens:** This is a subtle C++ rule that is easy to get wrong, especially during refactoring. The correct order is:
```cpp
class read_only_data_batch {
    std::shared_ptr<data_batch> parent_;  // destroyed SECOND (releases ref)
    std::shared_lock<std::shared_mutex> lock_;  // destroyed FIRST (unlocks)
};
```
If someone swaps these during refactoring, the `shared_ptr` is destroyed first.

**Consequences:** Same as Pitfall 1 -- mutex destroyed while locked, undefined behavior.

**Prevention:**
1. Document the ordering requirement with a comment: `// INVARIANT: parent_ must be declared before lock_ so lock is released before shared_ptr reference`
2. Write a unit test that creates an accessor, drops all other `shared_ptr` references to the `data_batch`, and then destroys the accessor. If the order is wrong, TSan or ASan will flag the use-after-free.
3. Consider wrapping the two members in a helper struct that enforces the ordering.
4. Add a static_assert or compile-time check if possible (e.g., via `offsetof` if the types are standard-layout, which they aren't, so this is more of a documentation approach).

**Detection:** ASan/TSan in tests. Code review. The "last reference" test described above.

**Phase:** Core implementation phase (DB-03). Must be correct from the first implementation.

**Confidence:** HIGH -- this is a well-known C++ destruction order pitfall.

---

### Pitfall 11: Repository's Internal Mutex vs. data_batch's shared_mutex -- Lock Ordering

**What goes wrong:** The `idata_repository` has its own `std::mutex` (`_mutex`) for thread-safe access to its internal vector. The `data_batch` has a `shared_mutex` for data access. If one code path acquires the repository lock first then the batch lock, and another code path acquires the batch lock first then the repository lock, classic ABBA deadlock occurs.

**Why it happens:** The repository lock protects the container (which batch is where). The batch lock protects the data (what's inside the batch). Natural usage patterns can invert the order:
- Path A: `repo.pop_data_batch()` (acquires repo lock) -> `data_batch::to_mutable(ptr)` (acquires batch lock)
- Path B: caller holds `mutable_data_batch` (holds batch lock) -> calls `repo.add_data_batch(to_idle(accessor))` (acquires repo lock)

**Consequences:** Deadlock. Two threads permanently blocked.

**Prevention:**
1. Establish a global lock ordering: repository lock is ALWAYS acquired before batch lock. Document this.
2. Path B above is the dangerous one: a caller should release their accessor (releasing the batch lock) *before* adding the idle batch back to the repository. The move-semantics design (DB-05) naturally enforces this -- `to_idle(mutable_data_batch&&)` consumes the accessor, releasing the lock, and returns a `shared_ptr<data_batch>`. The caller then passes this `shared_ptr` to `repo.add_data_batch()`. The lock is already released when the repository lock is acquired.
3. Audit all code paths where both locks might be held. The static method design centralizes this, making audit easier.

**Detection:** TSan with lock-order inversion detection. Stress testing with concurrent add/pop/lock operations.

**Phase:** Integration phase (DB-14, when updating `idata_repository`).

**Confidence:** HIGH -- ABBA deadlock is a classic concurrency bug; the two-mutex topology here is real.

---

## Minor Pitfalls

### Pitfall 12: Passkey Idiom Must Have Explicitly Defined (Not Defaulted) Constructor

**What goes wrong:** If the passkey class has no user-defined constructor, aggregate initialization can bypass the access control. Code like `data_batch(data_batch::private_token{}, ...)` compiles even if `private_token` is declared private, because aggregate initialization doesn't call a constructor.

**Why it happens:** C++ aggregate initialization rules. A class with no user-declared constructors, no private/protected non-static data members, and no virtual functions is an aggregate and can be initialized with `{}` regardless of where the `{}` appears.

**Prevention:** Give the passkey class an explicitly defined private constructor: `struct private_token { private: private_token() {} friend class data_batch; };`. Do not use `= default`.

**Detection:** Try to construct `data_batch` outside the factory in a unit test. If it compiles, the passkey is broken.

**Phase:** Core implementation phase (DB-07).

**Confidence:** HIGH -- well-known C++ aggregate initialization gotcha.

---

### Pitfall 13: subscriber_count Memory Ordering (relaxed) May Be Insufficient

**What goes wrong:** The current code uses `memory_order_relaxed` for `subscribe()`/`unsubscribe()`/`get_subscriber_count()`. If the subscriber count is used to make decisions that must be sequenced with other operations (e.g., "if subscriber count is 0, destroy the batch"), relaxed ordering may allow a thread to see a stale count and make an incorrect decision.

**Why it happens:** `memory_order_relaxed` only guarantees atomicity (no torn reads/writes). It does NOT guarantee that other memory operations (like "batch data is fully written") are visible to a thread that observes the count change.

**Prevention:**
1. If subscriber count is purely informational (diagnostics, logging), relaxed is fine.
2. If subscriber count guards a "safe to destroy" or "safe to evict" decision, use at least `memory_order_acquire` on `load()` and `memory_order_release` on `store()`/`fetch_sub()`, so that the count change is ordered with the data access.
3. Document the intended memory ordering semantics: "Subscriber count is eventually consistent. Do not use it for synchronization decisions."
4. The current `unsubscribe()` has a separate bug: it does `fetch_sub(1)` and then checks if `prev == 0`. But `fetch_sub` returns the value *before* the subtraction, so `prev == 0` means the count was already 0 and the subtraction just wrapped to `SIZE_MAX`. The check is correct in intent but the relaxed ordering means another thread's `subscribe()` might not be visible yet.

**Detection:** Memory ordering bugs are extremely difficult to test. Use TSan. Review with a concurrency expert. Consider using `seq_cst` initially and relaxing only after proving correctness.

**Phase:** Core implementation phase (DB-10). Decide memory ordering semantics early.

**Confidence:** MEDIUM -- depends on how subscriber count is used downstream. The current relaxed usage is fine if count is informational.

---

### Pitfall 14: CUDA Stream Synchronization While Holding Locks

**What goes wrong:** Methods like `clone()` (DB-13) and `convert_to()` (DB-12) invoke CUDA operations (`cudf::table` copy, `cudaMemcpyAsync`) that may require stream synchronization. If a `mutable_data_batch` holds an exclusive lock and calls `convert_to()`, which internally calls `stream.synchronize()`, the lock is held for the entire duration of the GPU operation (potentially milliseconds). During this time, no other thread can acquire even a shared lock.

**Why it happens:** CUDA `stream.synchronize()` is a blocking CPU call that waits for all previously enqueued work on that stream to complete. The lock is held across this blocking call because the accessor's RAII pattern keeps the lock alive.

**Consequences:** High lock contention. Other threads that need to read batch metadata (e.g., `get_batch_id()` -- wait, that's lock-free in the new design, good) or read batch data will block. If the GPU operation takes 10ms, every reader waits 10ms.

**Prevention:**
1. `get_batch_id()` is already lock-free (DB-08, DB-09). Good.
2. For `clone()`: acquire the shared lock, memcpy the metadata, launch the async GPU copy, release the lock, THEN synchronize the stream. The lock only needs to be held while reading the source data pointers, not while waiting for the copy to complete.
3. For `convert_to()`: this genuinely needs exclusive access (it replaces `_data`). Minimize the time under lock by pre-allocating the target buffer and launching the copy before acquiring the lock, then holding the lock only for the pointer swap.
4. Document: "Do not call `stream.synchronize()` while holding an accessor. Launch async work, release the accessor, then synchronize."

**Detection:** Latency profiling on lock acquisition. If p99 lock acquisition time is in the milliseconds range, someone is holding a lock across a GPU sync.

**Phase:** Performance optimization phase. Correctness first, then optimize lock hold times.

**Confidence:** MEDIUM -- specific to CUDA workloads. The impact depends on how often conversions happen under load.

---

### Pitfall 15: Moved-From shared_ptr Not Truly Null in Optimized Builds

**What goes wrong:** After `shared_ptr<data_batch> ptr = ...; auto accessor = data_batch::to_mutable(std::move(ptr));`, the moved-from `ptr` is guaranteed to be empty (null) by the standard. This is fine. But some developers may write `if (ptr)` checks that they expect to fail, and then a subsequent code path accidentally uses the now-null `ptr`. The "moved-from is null" guarantee for `shared_ptr` is reliable, but the try variants (DB-06) using `&` instead of `&&` introduce a subtlety: on failure, the `shared_ptr` is unchanged, on success it's nullified. Callers must check the return value, not the `shared_ptr`, to know what happened.

**Why it happens:** The try variant API is `optional<read_only_data_batch> try_to_read_only(shared_ptr<data_batch>&)`. On success, `ptr` becomes null and the return is engaged. On failure, `ptr` is unchanged and the return is `nullopt`. If callers check `ptr` instead of the return value, they get confused: a non-null `ptr` could mean "lock failed, ptr still valid" or "never called the function."

**Prevention:**
1. Make the API self-documenting. Return a result type that forces the caller to inspect it. `[[nodiscard]]` on the return type.
2. Consider returning a `variant<read_only_data_batch, shared_ptr<data_batch>>` -- on success you get the accessor, on failure you get your pointer back. But this adds complexity.
3. The simpler `optional` return is fine with `[[nodiscard]]`. Document: "On success, the source shared_ptr is null and the returned optional is engaged. On failure, the source shared_ptr is unchanged and the returned optional is empty."
4. Test both paths explicitly in unit tests.

**Detection:** Compiler warning from `[[nodiscard]]`. Code review for callers that ignore the return value.

**Phase:** API design phase (DB-06).

**Confidence:** HIGH -- the API is novel enough that misuse is likely without clear documentation.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| DB-03: Accessors hold shared_ptr | Pitfall 1, 10: Mutex destroyed while locked due to wrong member order | Enforce declaration order: shared_ptr before lock. Test with "last reference" scenario. |
| DB-05: Static conversion methods | Pitfall 2: Non-atomic upgrade creates unlocked window | Route through idle explicitly. Document TOCTOU risk. |
| DB-06: try variants with `&` | Pitfall 6: Data race on shared_ptr if called concurrently from same ptr | Document non-thread-safe on the shared_ptr argument. Callers use local copies. |
| DB-07: Factory + private constructor | Pitfall 3: shared_from_this before shared_ptr manages object | Use passkey idiom (Pitfall 12). Never call shared_from_this in constructor. |
| DB-07: enable_shared_from_this | Pitfall 4: operator= no-op corrupts nothing but confuses maintainers | Delete move ops on data_batch. Document the no-op. |
| DB-13: clone() acquires read lock | Pitfall 5: Self-deadlock from recursive lock acquisition | Make clone a method on read_only_data_batch, not on data_batch. |
| DB-14: Repository integration | Pitfall 11: ABBA deadlock between repo mutex and batch shared_mutex | Establish lock ordering: repo lock before batch lock. Move semantics enforce this. |
| Performance testing | Pitfall 7, 8: Writer starvation, shared_mutex overhead | Profile under realistic load. Keep accessors short-lived. |
| CUDA integration | Pitfall 14: Holding lock across GPU sync | Minimize lock scope. Async copy then sync after release. |

## Sources

- [CON50-CPP: Do not destroy a mutex while it is locked -- CERT](https://wiki.sei.cmu.edu/confluence/display/cplusplus/CON50-CPP.+Do+not+destroy+a+mutex+while+it+is+locked)
- [CON53-CPP: Avoid deadlock by locking in a predefined order -- CERT](https://wiki.sei.cmu.edu/confluence/display/cplusplus/CON53-CPP.+Avoid+deadlock+by+locking+in+a+predefined+order)
- [enable_shared_from_this overview and pitfalls -- nextptr](https://www.nextptr.com/tutorial/ta1414193955/enable_shared_from_this-overview-examples-and-internals)
- [enable_shared_from_this::operator= -- cppreference](https://en.cppreference.com/w/cpp/memory/enable_shared_from_this/operator=.html)
- [shared_from_this common pitfalls -- runebook](https://runebook.dev/en/docs/cpp/memory/enable_shared_from_this/shared_from_this)
- [My class derives from enable_shared_from_this but shared_from_this doesn't work -- Raymond Chen](https://devblogs.microsoft.com/oldnewthing/20220720-00/?p=106877)
- [Shared locking in C++ (N3427) -- WG21](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3427.html)
- [Shared Locking Revision 1 (N3568) -- WG21](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3568.html)
- [shared_mutex pitfalls, deadlocks, and alternatives -- runebook](https://runebook.dev/en/docs/cpp/thread/shared_mutex)
- [When shared_mutex outperforms mutex: benchmark -- Tech For Talk](https://techfortalk.co.uk/2026/01/03/when-stdshared_mutex-outperforms-stdmutex-a-google-benchmark-study/)
- [Understanding shared_mutex from C++17 -- C++ Stories](https://www.cppstories.com/2026/shared_mutex/)
- [Benchmarking Reader-Writer Lock Performance -- TuringCompl33t](https://turingcompl33t.github.io/RWLock-Benchmark/)
- [Passkey idiom for private constructors with make_shared -- GitHub Gist](https://gist.github.com/RklAlx/6727537)
- [Passkey idiom -- Simplify C++](https://arne-mertz.de/2016/10/passkey-idiom/)
- [Abseil Tip 134: make_unique and private constructors](https://abseil.io/tips/134)
- [MSVC STL SRWLOCK bug -- microsoft/STL#4448](https://github.com/microsoft/STL/issues/4448)
- [shared_ptr thread safety -- Lei Mao](https://leimao.github.io/blog/CPP-Shared-Ptr-Thread-Safety/)
