# Phase 1: Core Type System - Research

**Researched:** 2026-04-14
**Domain:** C++20 RAII lock accessor types with PtrType-agnostic templates, shared_mutex state machine via move semantics
**Confidence:** HIGH

## Summary

Phase 1 implements the complete data_batch type system as one atomic compilation unit: `data_batch`, `read_only_data_batch<PtrType>`, `mutable_data_batch<PtrType>`, all 8 state transition static methods, clone operations, and mutual friend relationships. This replaces the existing `synchronized_data_batch` nested-class design.

The design is fundamentally different from the earlier research summary in critical ways locked by user decisions: **no `enable_shared_from_this`** (static methods receive PtrType as parameter), **public constructor** (no passkey idiom), and **PtrType-agnostic templates** (supporting both `shared_ptr<data_batch>` and `unique_ptr<data_batch>`). These decisions simplify the implementation significantly -- no factory function needed, no `bad_weak_ptr` pitfall, no stack-allocation concern -- but require careful template mechanics for friend relationships and explicit instantiations.

The implementation touches exactly two files (`include/cucascade/data/data_batch.hpp` and `src/data/data_batch.cpp`). All three classes must live in the same header due to mutual friend relationships. The `.cpp` file provides explicit template instantiations for both `shared_ptr<data_batch>` and `unique_ptr<data_batch>`, following the established `idata_repository` pattern in `src/data/data_repository.cpp`.

**Primary recommendation:** Implement all three classes and all transitions in a single pass within `data_batch.hpp`/`data_batch.cpp`. The mutual dependencies make incremental compilation impossible -- this is inherently atomic work.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** 3 flat classes, no nesting. `data_batch` is the "idle" state -- owns data + mutex but exposes almost nothing publicly.
- **D-02:** No `enable_shared_from_this` -- static methods receive PtrType as parameter, no need to obtain shared_ptr from inside the object.
- **D-03:** Public constructor on `data_batch` -- no passkey idiom needed.
- **D-04:** Deleted move and copy operations on `data_batch` -- fixes the known move-without-lock bug from PR #99 v1. The object is never moved; only the PtrType to it is moved.
- **D-05:** `const uint64_t _batch_id` -- immutable after construction.
- **D-06:** `read_only_data_batch<PtrType>` and `mutable_data_batch<PtrType>` are full class templates. Static methods on `data_batch` are also templated on PtrType.
- **D-07:** Template definitions live in the header (`data_batch.hpp`). Explicit template instantiations for `shared_ptr<data_batch>` and `unique_ptr<data_batch>` in `data_batch.cpp` -- matches existing `idata_repository` pattern.
- **D-08:** All 3 classes in a single header (`data_batch.hpp`) due to mutual friend relationships.
- **D-09:** Return types match current codebase: `get_data()` returns `idata_representation*`, `get_memory_space()` returns `memory_space*`, `get_current_tier()` returns `Tier` (value type). Raw pointers, not references.
- **D-10:** `mutable_data_batch` exposes full read+write: all methods from read_only plus `set_data(unique_ptr<idata_representation>)` and `convert_to<T>(registry, target_memory_space, stream)`.
- **D-11:** Accessor types store PtrType parent (first member) before lock guard (second member) -- destruction order is load-bearing (lock releases before parent drops).
- **D-12:** Move-only semantics (deleted copy), `noexcept` on move operations.
- **D-13:** All 6 blocking transitions + 2 try variants are static methods on `data_batch`, templated on PtrType.
- **D-14:** Blocking variants take PtrType by rvalue reference (`&&`) -- move semantics invalidate source.
- **D-15:** Try variants take PtrType by mutable lvalue reference (`&`) -- nullify on success, unchanged on failure.
- **D-16:** No direct locked-to-locked conversion. `to_mutable(read_only_data_batch&&)` and `to_read_only(mutable_data_batch&&)` go through idle internally (release lock, reacquire through parent).
- **D-17:** `[[nodiscard]]` on all transition methods.
- **D-18:** `clone()` and `clone_to<T>()` are methods on `read_only_data_batch` -- NOT on `data_batch`. Avoids recursive shared_mutex deadlock since the caller already holds the shared lock.
- **D-19:** Both return PtrType (match the template parameter).
- **D-20:** `clone()` = deep copy, same representation type. `clone_to<T>()` = deep copy + representation conversion via converter registry.
- **D-21:** `get_batch_id()` -- lock-free, public.
- **D-22:** `subscribe()`, `unsubscribe()`, `get_subscriber_count()` -- atomic, public.
- **D-23:** No public access to data, tier, or memory space. These require acquiring a lock through an accessor.
- **D-24:** Mutual friends: `data_batch` <-> `read_only_data_batch<PtrType>`, `data_batch` <-> `mutable_data_batch<PtrType>`.
- **D-25:** `read_only_data_batch` and `mutable_data_batch` do NOT need to know about each other -- all conversions go through `data_batch`.

### Claude's Discretion

No areas explicitly marked for discretion. All design decisions are locked. Implementation details (ordering within methods, helper functions, comment style) follow codebase conventions.

### Deferred Ideas (OUT OF SCOPE)

- Repository `try_pop` API (iterates batches, tries `try_to_mutable` on each) -- dhruv's suggestion, v2
- Probing/observability interface (`idata_batch_probe` replacement) -- dhruv flagged as future need
- `get_data_as<T>()` convenience template on accessors -- evaluate after refactor
- `withLock` callback pattern (folly-style) -- evaluate based on call site patterns

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CORE-01 | 3 flat classes: data_batch, read_only_data_batch, mutable_data_batch | Architecture pattern: flat peer classes with mutual friend (D-01, D-08) |
| CORE-02 | data_batch has no public access to data/tier/memory_space | Private data accessors behind friend wall pattern (D-23) |
| CORE-03 | Public constructor on data_batch | Simplified by D-02 (no enable_shared_from_this), D-03 |
| CORE-04 | const uint64_t _batch_id -- immutable after construction | D-05, constructor initializer list sets once |
| CORE-05 | Lock-free get_batch_id() public on data_batch | Const member, no synchronization needed (D-21) |
| CORE-06 | Atomic subscriber count on data_batch | Existing pattern in current code, memory_order_relaxed (D-22) |
| CORE-07 | Deleted move and copy operations on data_batch | D-04, fixes known move-without-lock bug |
| ACC-01 | read_only_data_batch with named methods | Delegation to data_batch private methods via friend (D-09) |
| ACC-02 | mutable_data_batch with all read + write methods | D-10, delegation pattern with set_data and convert_to |
| ACC-03 | PtrType-agnostic accessors | Full class templates (D-06), explicit instantiation (D-07) |
| ACC-04 | Accessors store PtrType parent for lifetime management | D-11, PtrType as first member |
| ACC-05 | Move-only semantics on both accessor types | D-12, deleted copy |
| ACC-06 | noexcept on move operations | D-12, standard RAII lock guarantee |
| ACC-07 | Correct member declaration order: PtrType before lock guard | D-11, destruction order is load-bearing |
| ACC-08 | [[nodiscard]] on all accessor-returning methods | D-17 |
| TRANS-01 | to_read_only(PtrType&&) -- idle to shared lock | Static method on data_batch, D-13/D-14 |
| TRANS-02 | to_mutable(PtrType&&) -- idle to exclusive lock | Static method on data_batch, D-13/D-14 |
| TRANS-03 | to_idle(read_only_data_batch&&) -- release shared lock | Returns PtrType, D-13 |
| TRANS-04 | to_idle(mutable_data_batch&&) -- release exclusive lock | Returns PtrType, D-13 |
| TRANS-05 | to_mutable(read_only_data_batch&&) -- through idle | D-16, release then reacquire |
| TRANS-06 | to_read_only(mutable_data_batch&&) -- through idle | D-16, release then reacquire |
| TRANS-07 | try_to_read_only(PtrType&) -- non-blocking | D-15, nullifies on success, unchanged on failure |
| TRANS-08 | try_to_mutable(PtrType&) -- non-blocking | D-15, nullifies on success, unchanged on failure |
| TRANS-09 | Move semantics invalidation | D-14, rvalue ref for blocking, lvalue ref for try |
| TRANS-10 | [[nodiscard]] on all transition methods | D-17 |
| CLONE-01 | clone() on read_only_data_batch | D-18, avoids recursive shared_mutex deadlock |
| CLONE-02 | clone_to\<T\>() on read_only_data_batch | D-18/D-20, deep copy + conversion |
| REPO-04 | Mutual friend relationships | D-24/D-25, bidirectional but not accessor-to-accessor |

</phase_requirements>

## Standard Stack

### Core

No new dependencies. 100% C++ standard library primitives.

| Primitive | Header | Purpose | Why Standard |
|-----------|--------|---------|--------------|
| `std::shared_mutex` | `<shared_mutex>` | Reader-writer lock for data_batch | Already used in current codebase; concurrent readers, exclusive writers [VERIFIED: data_batch.hpp line 31] |
| `std::shared_lock<std::shared_mutex>` | `<shared_mutex>` | RAII shared lock guard for read_only_data_batch | Standard RAII pattern; supports `try_to_lock` tag [VERIFIED: data_batch.hpp line 119] |
| `std::unique_lock<std::shared_mutex>` | `<shared_mutex>` | RAII exclusive lock guard for mutable_data_batch | Standard RAII pattern; supports `try_to_lock` tag [VERIFIED: data_batch.hpp line 154] |
| `std::atomic<size_t>` | `<atomic>` | Subscriber count | Lock-free interest tracking [VERIFIED: data_batch.hpp line 208] |
| `std::optional` | `<optional>` | Try variant return types | Standard nullable value type [VERIFIED: data_batch.hpp line 30] |
| `std::unique_ptr<idata_representation>` | `<memory>` | Data ownership | Existing pattern throughout codebase [VERIFIED: data_batch.hpp line 76] |

### Supporting (already in project)

| Dependency | Purpose | How Used |
|------------|---------|----------|
| `idata_representation` | Base class for data held by batch | `data_batch._data` member, `clone()`, `get_current_tier()`, `get_size_in_bytes()` [VERIFIED: common.hpp] |
| `representation_converter_registry` | Type-keyed conversion dispatch | Called by `convert_to<T>()` and `clone_to<T>()` [VERIFIED: representation_converter.hpp] |
| `memory::Tier` | Memory tier enum | Returned by `get_current_tier()` [VERIFIED: memory/common.hpp line 34] |
| `memory::memory_space` | Memory space reference | Returned by `get_memory_space()` [VERIFIED: common.hpp line 76-83] |
| `rmm::cuda_stream_view` | CUDA stream parameter | Used by clone and convert operations [VERIFIED: data_batch.hpp line 81] |

### Alternatives Considered

None. All decisions are locked per CONTEXT.md. The stack is pure C++ standard library.

## Architecture Patterns

### Recommended File Structure

```
include/cucascade/data/
  data_batch.hpp        # All 3 classes (data_batch, read_only_data_batch<PtrType>, mutable_data_batch<PtrType>)
src/data/
  data_batch.cpp        # Explicit template instantiations + non-template method bodies
```

No new files. Same two files as current implementation, rewritten in place.

### Header Layout

[VERIFIED: patterns from current data_batch.hpp and data_repository.hpp]

```
data_batch.hpp
  |
  +-- #pragma once
  +-- #includes (same set as current, minus <stdexcept> if unused)
  +-- Forward declarations: memory::memory_space, representation_converter_registry
  |
  +-- namespace cucascade {
  |
  +-- Forward declarations of templates:
  |     template <typename PtrType> class read_only_data_batch;
  |     template <typename PtrType> class mutable_data_batch;
  |
  +-- class data_batch {
  |     public:
  |       // Construction
  |       data_batch(uint64_t batch_id, std::unique_ptr<idata_representation> data);
  |       ~data_batch() = default;
  |
  |       // Deleted move/copy (D-04, D-07)
  |       data_batch(data_batch&&) = delete;
  |       data_batch& operator=(data_batch&&) = delete;
  |       data_batch(const data_batch&) = delete;
  |       data_batch& operator=(const data_batch&) = delete;
  |
  |       // Lock-free public API (D-21, D-22)
  |       uint64_t get_batch_id() const;
  |       bool subscribe();
  |       void unsubscribe();
  |       size_t get_subscriber_count() const;
  |
  |       // Static transition methods -- templated on PtrType (D-13)
  |       template <typename PtrType>
  |       [[nodiscard]] static read_only_data_batch<PtrType> to_read_only(PtrType&& batch);
  |       template <typename PtrType>
  |       [[nodiscard]] static mutable_data_batch<PtrType> to_mutable(PtrType&& batch);
  |       template <typename PtrType>
  |       [[nodiscard]] static PtrType to_idle(read_only_data_batch<PtrType>&& accessor);
  |       template <typename PtrType>
  |       [[nodiscard]] static PtrType to_idle(mutable_data_batch<PtrType>&& accessor);
  |       template <typename PtrType>
  |       [[nodiscard]] static mutable_data_batch<PtrType> to_mutable(read_only_data_batch<PtrType>&& accessor);
  |       template <typename PtrType>
  |       [[nodiscard]] static read_only_data_batch<PtrType> to_read_only(mutable_data_batch<PtrType>&& accessor);
  |       template <typename PtrType>
  |       [[nodiscard]] static std::optional<read_only_data_batch<PtrType>> try_to_read_only(PtrType& batch);
  |       template <typename PtrType>
  |       [[nodiscard]] static std::optional<mutable_data_batch<PtrType>> try_to_mutable(PtrType& batch);
  |
  |     private:
  |       // Friend declarations (D-24)
  |       template <typename PtrType> friend class read_only_data_batch;
  |       template <typename PtrType> friend class mutable_data_batch;
  |
  |       // Private data accessors (D-23) -- only friends can call
  |       memory::Tier get_current_tier() const;
  |       idata_representation* get_data() const;
  |       memory::memory_space* get_memory_space() const;
  |       void set_data(std::unique_ptr<idata_representation> data);
  |       template <typename TargetRepresentation>
  |       void convert_to(representation_converter_registry& registry,
  |                        const memory::memory_space* target_memory_space,
  |                        rmm::cuda_stream_view stream);
  |
  |       // Members
  |       const uint64_t _batch_id;                          // D-05
  |       std::unique_ptr<idata_representation> _data;
  |       mutable std::shared_mutex _rw_mutex;
  |       std::atomic<size_t> _subscriber_count{0};
  |   };
  |
  +-- template <typename PtrType>
  |   class read_only_data_batch {
  |     public:
  |       // Named accessor methods (D-09)
  |       uint64_t get_batch_id() const;
  |       memory::Tier get_current_tier() const;
  |       idata_representation* get_data() const;
  |       memory::memory_space* get_memory_space() const;
  |
  |       // Clone operations (D-18, D-19, D-20)
  |       [[nodiscard]] PtrType clone(uint64_t new_batch_id, rmm::cuda_stream_view stream) const;
  |       template <typename TargetRepresentation>
  |       [[nodiscard]] PtrType clone_to(representation_converter_registry& registry,
  |                                       uint64_t new_batch_id,
  |                                       const memory::memory_space* target_memory_space,
  |                                       rmm::cuda_stream_view stream) const;
  |
  |       // Move-only (D-12)
  |       read_only_data_batch(read_only_data_batch&&) noexcept = default;
  |       read_only_data_batch& operator=(read_only_data_batch&&) noexcept = default;
  |       read_only_data_batch(const read_only_data_batch&) = delete;
  |       read_only_data_batch& operator=(const read_only_data_batch&) = delete;
  |
  |     private:
  |       friend class data_batch;  // D-24
  |       read_only_data_batch(PtrType parent, std::shared_lock<std::shared_mutex> lock);
  |
  |       PtrType _batch;                                    // FIRST -- destroyed second (D-11)
  |       std::shared_lock<std::shared_mutex> _lock;         // SECOND -- destroyed first (D-11)
  |   };
  |
  +-- template <typename PtrType>
  |   class mutable_data_batch {
  |     public:
  |       // Read methods (same as read_only)
  |       uint64_t get_batch_id() const;
  |       memory::Tier get_current_tier() const;
  |       idata_representation* get_data() const;
  |       memory::memory_space* get_memory_space() const;
  |
  |       // Write methods (D-10)
  |       void set_data(std::unique_ptr<idata_representation> data);
  |       template <typename TargetRepresentation>
  |       void convert_to(representation_converter_registry& registry,
  |                        const memory::memory_space* target_memory_space,
  |                        rmm::cuda_stream_view stream);
  |
  |       // Move-only (D-12)
  |       mutable_data_batch(mutable_data_batch&&) noexcept = default;
  |       mutable_data_batch& operator=(mutable_data_batch&&) noexcept = default;
  |       mutable_data_batch(const mutable_data_batch&) = delete;
  |       mutable_data_batch& operator=(const mutable_data_batch&) = delete;
  |
  |     private:
  |       friend class data_batch;  // D-24
  |       mutable_data_batch(PtrType parent, std::unique_lock<std::shared_mutex> lock);
  |
  |       PtrType _batch;                                    // FIRST -- destroyed second (D-11)
  |       std::unique_lock<std::shared_mutex> _lock;         // SECOND -- destroyed first (D-11)
  |   };
  |
  +-- // Template implementations (all inline in header per D-07)
  +-- } // namespace cucascade
```

### Pattern 1: Static Transition Methods with PtrType Templates

**What:** All state transitions are `template <typename PtrType> static` methods on `data_batch`. The PtrType is the smart pointer type wrapping `data_batch*`.

**When to use:** Every state transition.

**Why this works without `enable_shared_from_this`:** The caller passes the PtrType (e.g., `shared_ptr<data_batch>`) as a parameter. The static method moves it into the accessor's `_batch` member. No need for the object to know its own pointer type. [VERIFIED: D-02 explicitly rejects enable_shared_from_this for this reason]

**Example (idle to read-locked):**
```cpp
// Source: D-13, D-14
template <typename PtrType>
[[nodiscard]] read_only_data_batch<PtrType>
data_batch::to_read_only(PtrType&& batch)
{
  auto ptr = std::move(batch);  // batch is now nullptr at caller
  std::shared_lock<std::shared_mutex> lock(ptr->_rw_mutex);
  return read_only_data_batch<PtrType>(std::move(ptr), std::move(lock));
}
```

**Example (try variant, non-blocking):**
```cpp
// Source: D-15
template <typename PtrType>
[[nodiscard]] std::optional<read_only_data_batch<PtrType>>
data_batch::try_to_read_only(PtrType& batch)
{
  std::shared_lock<std::shared_mutex> lock(batch->_rw_mutex, std::try_to_lock);
  if (!lock.owns_lock()) { return std::nullopt; }
  auto ptr = std::move(batch);  // nullify source only on success
  return read_only_data_batch<PtrType>(std::move(ptr), std::move(lock));
}
```

### Pattern 2: Locked-to-Locked Through Idle

**What:** `to_mutable(read_only_data_batch<PtrType>&&)` releases the shared lock, extracts the PtrType, then reacquires as exclusive. The gap is intentional and documented.

**When to use:** TRANS-05, TRANS-06.

**Why through idle:** `std::shared_mutex` does not support atomic lock upgrade. Attempting it causes deadlock when multiple readers try to upgrade simultaneously. [VERIFIED: pitfall documented in PITFALLS.md Pitfall 2, confirmed by WG21 N3427/N3568]

**Example:**
```cpp
// Source: D-16
template <typename PtrType>
[[nodiscard]] mutable_data_batch<PtrType>
data_batch::to_mutable(read_only_data_batch<PtrType>&& accessor)
{
  // Extract parent pointer and release shared lock
  auto ptr = std::move(accessor._batch);
  accessor._lock.unlock();
  // Now in idle state -- reacquire as exclusive
  std::unique_lock<std::shared_mutex> lock(ptr->_rw_mutex);
  return mutable_data_batch<PtrType>(std::move(ptr), std::move(lock));
}
```

### Pattern 3: Private Data Accessors Behind Friend Wall

**What:** All methods that read or write data on `data_batch` are `private`. Only the template friend classes can call them. The public API on `data_batch` is limited to `get_batch_id()` and subscriber count operations.

**When to use:** Always. This is the compile-time enforcement mechanism (CORE-02).

**Why:** Prevents callers from accessing data on an idle `data_batch` without holding a lock. [VERIFIED: D-23]

### Pattern 4: Accessor Delegates to data_batch Private Methods

**What:** Named methods on accessors delegate to `data_batch`'s private methods through the friend relationship. No `operator->`.

**When to use:** All accessor methods (ACC-01, ACC-02).

**Example:**
```cpp
// Source: D-09, existing pattern in data_batch.cpp lines 33-47
template <typename PtrType>
idata_representation* read_only_data_batch<PtrType>::get_data() const
{
  return _batch->get_data();
}
```

### Pattern 5: Clone on read_only_data_batch (Not data_batch)

**What:** `clone()` and `clone_to<T>()` are methods on `read_only_data_batch`, not on `data_batch`. The caller already holds the shared lock, so clone does NOT need to acquire its own lock.

**When to use:** CLONE-01, CLONE-02.

**Why:** Avoids recursive `shared_mutex` deadlock. `std::shared_mutex` is NOT reentrant -- calling `lock_shared()` when the thread already holds a shared lock is undefined behavior. [VERIFIED: PITFALLS.md Pitfall 5]

**Example:**
```cpp
// Source: D-18, D-19
template <typename PtrType>
PtrType read_only_data_batch<PtrType>::clone(
  uint64_t new_batch_id, rmm::cuda_stream_view stream) const
{
  if (_batch->_data == nullptr) {
    throw std::runtime_error("Cannot clone: data is null");
  }
  auto cloned_data = _batch->_data->clone(stream);
  // Construct new data_batch, wrap in PtrType
  // For shared_ptr: std::make_shared<data_batch>(new_batch_id, std::move(cloned_data))
  // For unique_ptr: std::make_unique<data_batch>(new_batch_id, std::move(cloned_data))
  return PtrType(new data_batch(new_batch_id, std::move(cloned_data)));
}
```

**Important subtlety for PtrType return:** `clone()` returns `PtrType`. For `shared_ptr`, use `std::make_shared<data_batch>(...)`. For `unique_ptr`, use `std::make_unique<data_batch>(...)`. Since this is a template, the implementation needs to work for both. Using `PtrType(new data_batch(...))` works generically but misses the `make_shared` optimization (single allocation for object + control block). Consider a helper or `if constexpr` to dispatch. See "Common Pitfalls" section.

### Pattern 6: Explicit Template Instantiation

**What:** Template definitions are in the header. The `.cpp` file provides explicit instantiations for the two supported PtrType values.

**When to use:** D-07. Follow the existing pattern in `src/data/data_repository.cpp`.

**Example (data_batch.cpp):**
```cpp
// Source: existing pattern from data_repository.cpp lines 23-24
namespace cucascade {

// Explicit template instantiations for accessor types
template class read_only_data_batch<std::shared_ptr<data_batch>>;
template class read_only_data_batch<std::unique_ptr<data_batch>>;
template class mutable_data_batch<std::shared_ptr<data_batch>>;
template class mutable_data_batch<std::unique_ptr<data_batch>>;

}  // namespace cucascade
```

**Note:** The static template methods on `data_batch` are defined in the header and instantiated implicitly when used. The explicit instantiations cover the accessor classes whose non-inline methods (if any) need to be emitted in the `.cpp` translation unit.

### Anti-Patterns to Avoid

- **`operator->` on accessors:** Current design uses it (line 100, 136 of data_batch.hpp). New design replaces with named methods. `operator->` leaks raw pointers and prevents adding accessor-specific methods. [VERIFIED: D-09]
- **Inheritance between accessor types:** `mutable_data_batch : public read_only_data_batch` would cause slicing and wrong lock type in base. Use delegation instead. [VERIFIED: ARCHITECTURE.md Anti-Pattern 5]
- **Nested classes:** Current `synchronized_data_batch` nests everything inside. New design uses flat peer classes. [VERIFIED: D-01]
- **Moving data_batch objects:** Never move the `data_batch` itself -- only move the PtrType to it. [VERIFIED: D-04]
- **Clone on idle data_batch:** Would need to acquire lock internally, risking recursive deadlock if caller already holds one. [VERIFIED: D-18, PITFALLS.md Pitfall 5]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Reader-writer locking | Custom lock implementation | `std::shared_mutex` + `std::shared_lock`/`std::unique_lock` | Proven, standard, debuggable with TSan [VERIFIED: current codebase uses this] |
| Non-blocking lock try | Custom spin-lock or polling | `std::try_to_lock` tag with standard lock guards | Correct atomicity, well-tested [VERIFIED: data_batch.cpp lines 121-132] |
| Atomic subscriber count | Custom lock around counter | `std::atomic<size_t>` with `memory_order_relaxed` | Lock-free, no contention with data lock [VERIFIED: data_batch.hpp line 208] |
| Deep copy of data | Manual memcpy or cudf copy | `idata_representation::clone(stream)` virtual | Polymorphic, handles all representation types [VERIFIED: common.hpp line 112] |
| Type-keyed conversion | Switch/if on type | `representation_converter_registry` | Extensible, already implemented [VERIFIED: representation_converter.hpp] |
| PtrType construction dispatch | Template specializations | `PtrType(new data_batch(...))` or `if constexpr` | Handles both shared_ptr and unique_ptr generically |

## Common Pitfalls

### Pitfall 1: Member Declaration Order Is Load-Bearing

**What goes wrong:** If the lock guard member is declared before the PtrType member in accessor classes, C++ destroys the PtrType first (reverse declaration order). If the PtrType holds the last reference (e.g., `unique_ptr`), the `data_batch` object and its mutex are destroyed while the lock guard still references that mutex. Unlocking a destroyed mutex is undefined behavior. [VERIFIED: PITFALLS.md Pitfall 1, Pitfall 10; CERT CON50-CPP]

**Why it happens:** C++ destroys class members in reverse declaration order. This is a subtle rule easily broken during refactoring.

**How to avoid:**
1. `PtrType _batch;` MUST be declared before `lock_type _lock;` in both accessor classes (D-11).
2. Add a comment: `// INVARIANT: _batch must be declared before _lock -- destruction order is load-bearing`
3. With `unique_ptr<data_batch>`, the accessor IS the sole owner. When the accessor is destroyed, `_lock` destructor runs first (unlocks), then `_batch` destructor runs (deletes the data_batch). Correct.

**Warning signs:** ASan/TSan reports use-after-free on mutex operations.

### Pitfall 2: PtrType-Generic clone() Return

**What goes wrong:** `clone()` returns `PtrType`. For `shared_ptr<data_batch>`, you want `std::make_shared<data_batch>(...)` (single allocation for object + control block). For `unique_ptr<data_batch>`, you want `std::make_unique<data_batch>(...)`. A generic `PtrType(new data_batch(...))` works but misses the `make_shared` optimization and triggers `-Wnew-delete-type-mismatch` warnings in some compilers.

**How to avoid:** Use `if constexpr` to dispatch:
```cpp
if constexpr (std::is_same_v<PtrType, std::shared_ptr<data_batch>>) {
  return std::make_shared<data_batch>(new_batch_id, std::move(cloned_data));
} else {
  return std::make_unique<data_batch>(new_batch_id, std::move(cloned_data));
}
```
Or use a private static helper `data_batch::make(uint64_t, unique_ptr<idata_representation>)` that returns PtrType. [ASSUMED -- compiler behavior on generic `PtrType(new T)` may vary]

### Pitfall 3: Non-Atomic Upgrade Creates TOCTOU Window

**What goes wrong:** `to_mutable(read_only_data_batch<PtrType>&&)` releases the shared lock then acquires exclusive. During the gap, another thread can acquire exclusive, mutate data, and release. The caller's state observations from the read phase may be stale. [VERIFIED: PITFALLS.md Pitfall 2, WG21 N3427/N3568]

**How to avoid:** Document the TOCTOU window in the method's Doxygen comment. Callers must re-validate any state they observed during the read phase. The API design makes this explicit -- the gap is not hidden.

### Pitfall 4: try_to_* Data Race on Same PtrType Instance

**What goes wrong:** `try_to_read_only(PtrType&)` reads and potentially moves the PtrType argument. If two threads call this on the same `shared_ptr<data_batch>` lvalue concurrently, both threads read/write the same `shared_ptr` object without synchronization -- data race. [VERIFIED: PITFALLS.md Pitfall 6]

**How to avoid:** Document that `try_to_*` methods are NOT thread-safe on the PtrType argument. Callers must ensure exclusive access to the PtrType they pass in (typically a thread-local copy obtained under the repository's mutex). This is the natural usage pattern -- repositories store batches under their own mutex, callers pop to a local variable first.

### Pitfall 5: Template Friend Declarations Syntax

**What goes wrong:** Friend template declarations in C++ have specific syntax requirements. A common mistake is writing `friend class read_only_data_batch;` (without template parameter) inside `data_batch`, which befriends a non-template class that doesn't exist. [ASSUMED -- common C++ template friend pitfall]

**How to avoid:** Use the correct syntax:
```cpp
class data_batch {
  template <typename PtrType> friend class read_only_data_batch;
  template <typename PtrType> friend class mutable_data_batch;
};
```
And in the accessor:
```cpp
template <typename PtrType>
class read_only_data_batch {
  friend class data_batch;  // non-template class befriending is fine
};
```

### Pitfall 6: Subscriber Count Underflow

**What goes wrong:** The current code uses `fetch_sub(1)` and checks if the previous value was 0 AFTER the subtraction. But `fetch_sub` on an unsigned type wraps to `SIZE_MAX` when the value is 0. The check catches this and re-adds 1, but with `memory_order_relaxed`, another thread's `subscribe()` might not be visible yet. [VERIFIED: data_batch.cpp lines 148-153, PITFALLS.md Pitfall 13]

**How to avoid:** Keep the existing pattern (check-after-subtract + throw) as it's proven to work for the current use case. The relaxed ordering is acceptable because subscriber count is informational, not used for synchronization decisions (D-22). Document this explicitly.

### Pitfall 7: Accessor Move Assignment to Self

**What goes wrong:** `accessor_a = std::move(accessor_a)` -- if the defaulted move assignment doesn't handle self-assignment, the lock could be released and then the object left in a broken state. [ASSUMED -- standard library lock guards handle this, but worth verifying]

**How to avoid:** `= default` move operations on accessors are safe because `shared_lock` and `unique_lock` have well-defined move semantics that handle moved-from state. The defaulted operations will work correctly. Verify with a test.

## Code Examples

### Complete data_batch Class Definition

```cpp
// Source: synthesized from D-01 through D-25, matching codebase conventions
// from data_batch.hpp and data_repository.hpp

class data_batch {
 public:
  /**
   * @brief Construct a new data_batch in idle state.
   *
   * @param batch_id Unique identifier for this batch (immutable after construction)
   * @param data The data representation this batch owns
   */
  data_batch(uint64_t batch_id, std::unique_ptr<idata_representation> data);
  ~data_batch() = default;

  // Non-copyable, non-movable (D-04, D-07)
  data_batch(data_batch&&)            = delete;
  data_batch& operator=(data_batch&&) = delete;
  data_batch(const data_batch&)       = delete;
  data_batch& operator=(const data_batch&) = delete;

  // -- Lock-free public API (D-21, D-22, D-23) --

  /** @brief Get the unique batch identifier. Lock-free. */
  uint64_t get_batch_id() const;

  /** @brief Register interest in this batch. Atomic, lock-free. */
  bool subscribe();

  /** @brief Deregister interest. Throws if count already zero. */
  void unsubscribe();

  /** @brief Current subscriber count. Atomic, lock-free. */
  size_t get_subscriber_count() const;

  // -- Static transition methods (D-13, D-14, D-15, D-17) --

  /** @brief Idle -> shared lock (blocking). Caller gives up PtrType ownership. */
  template <typename PtrType>
  [[nodiscard]] static read_only_data_batch<PtrType> to_read_only(PtrType&& batch);

  /** @brief Idle -> exclusive lock (blocking). Caller gives up PtrType ownership. */
  template <typename PtrType>
  [[nodiscard]] static mutable_data_batch<PtrType> to_mutable(PtrType&& batch);

  /** @brief Shared lock -> idle. Returns PtrType to caller. */
  template <typename PtrType>
  [[nodiscard]] static PtrType to_idle(read_only_data_batch<PtrType>&& accessor);

  /** @brief Exclusive lock -> idle. Returns PtrType to caller. */
  template <typename PtrType>
  [[nodiscard]] static PtrType to_idle(mutable_data_batch<PtrType>&& accessor);

  /**
   * @brief Shared lock -> exclusive lock (through idle internally).
   *
   * @note NON-ATOMIC: releases shared lock, then acquires exclusive.
   * Data may have changed between the release and reacquisition.
   * Callers must re-validate any state observed during read-only access.
   */
  template <typename PtrType>
  [[nodiscard]] static mutable_data_batch<PtrType> to_mutable(
    read_only_data_batch<PtrType>&& accessor);

  /** @brief Exclusive lock -> shared lock (through idle internally). */
  template <typename PtrType>
  [[nodiscard]] static read_only_data_batch<PtrType> to_read_only(
    mutable_data_batch<PtrType>&& accessor);

  /**
   * @brief Non-blocking idle -> shared lock.
   *
   * On success: batch is nullified, returned optional is engaged.
   * On failure: batch is unchanged, returned optional is empty.
   *
   * @note NOT thread-safe on the PtrType argument itself. Caller must
   * ensure exclusive access to the PtrType they pass in.
   */
  template <typename PtrType>
  [[nodiscard]] static std::optional<read_only_data_batch<PtrType>> try_to_read_only(
    PtrType& batch);

  /** @brief Non-blocking idle -> exclusive lock. Same semantics as try_to_read_only. */
  template <typename PtrType>
  [[nodiscard]] static std::optional<mutable_data_batch<PtrType>> try_to_mutable(PtrType& batch);

 private:
  // Friends (D-24)
  template <typename PtrType>
  friend class read_only_data_batch;
  template <typename PtrType>
  friend class mutable_data_batch;

  // Private data accessors -- only friends can call (D-23)
  memory::Tier get_current_tier() const;
  idata_representation* get_data() const;
  memory::memory_space* get_memory_space() const;
  void set_data(std::unique_ptr<idata_representation> data);

  template <typename TargetRepresentation>
  void convert_to(representation_converter_registry& registry,
                  const memory::memory_space* target_memory_space,
                  rmm::cuda_stream_view stream);

  // Members
  const uint64_t _batch_id;                       ///< Immutable batch identifier (D-05)
  std::unique_ptr<idata_representation> _data;    ///< Owned data representation
  mutable std::shared_mutex _rw_mutex;            ///< Reader-writer lock
  std::atomic<size_t> _subscriber_count{0};       ///< Interest counter
};
```

### Complete read_only_data_batch Template

```cpp
// Source: synthesized from D-06, D-09, D-11, D-12, D-18
template <typename PtrType>
class read_only_data_batch {
 public:
  // Named accessor methods (D-09) -- delegates to data_batch privates
  uint64_t get_batch_id() const { return _batch->get_batch_id(); }
  memory::Tier get_current_tier() const { return _batch->get_current_tier(); }
  idata_representation* get_data() const { return _batch->get_data(); }
  memory::memory_space* get_memory_space() const { return _batch->get_memory_space(); }

  // Clone operations (D-18, D-19, D-20) -- lock already held
  [[nodiscard]] PtrType clone(uint64_t new_batch_id, rmm::cuda_stream_view stream) const;

  template <typename TargetRepresentation>
  [[nodiscard]] PtrType clone_to(representation_converter_registry& registry,
                                  uint64_t new_batch_id,
                                  const memory::memory_space* target_memory_space,
                                  rmm::cuda_stream_view stream) const;

  // Move-only (D-12)
  read_only_data_batch(read_only_data_batch&&) noexcept            = default;
  read_only_data_batch& operator=(read_only_data_batch&&) noexcept = default;
  read_only_data_batch(const read_only_data_batch&)                = delete;
  read_only_data_batch& operator=(const read_only_data_batch&)     = delete;

 private:
  friend class data_batch;  // D-24

  read_only_data_batch(PtrType parent, std::shared_lock<std::shared_mutex> lock)
    : _batch(std::move(parent)), _lock(std::move(lock))
  {
  }

  // INVARIANT: _batch must be declared before _lock -- destruction order is load-bearing (D-11)
  PtrType _batch;                                 ///< Parent lifetime (destroyed second)
  std::shared_lock<std::shared_mutex> _lock;      ///< Shared lock (destroyed first)
};
```

### Transition Method Implementation (to_idle)

```cpp
// Source: D-13, returns PtrType to caller
template <typename PtrType>
PtrType data_batch::to_idle(read_only_data_batch<PtrType>&& accessor)
{
  auto ptr = std::move(accessor._batch);
  accessor._lock.unlock();
  return ptr;
}
```

## State of the Art

| Old Approach (current) | New Approach (this phase) | Why Changed |
|------------------------|--------------------------|-------------|
| Nested `synchronized_data_batch` wrapping inner `data_batch` | 3 flat peer classes | Simpler mental model, no inner class indirection (D-01) |
| `operator->` returning raw pointer | Named delegation methods | Prevents raw pointer escape, enables fine-grained const enforcement (D-09) |
| Accessor holds raw `synchronized_data_batch*` (borrow) | Accessor holds `PtrType` (ownership) | Extends parent lifetime, fixes dangling pointer bug flagged in PR #99 review (D-11) |
| Instance methods for lock acquisition (`batch.get_read_only()`) | Static methods on `data_batch` (`data_batch::to_read_only(std::move(ptr))`) | Centralizes state machine, makes ownership transfer explicit (D-13) |
| Move ctor/assign on wrapper class | Deleted move/copy on `data_batch` | Fixes known move-without-lock bug (D-04) |
| `clone()` on wrapper with internal lock | `clone()` on `read_only_data_batch` (lock already held) | Avoids recursive shared_mutex deadlock (D-18) |
| Non-template accessors | `read_only_data_batch<PtrType>`, `mutable_data_batch<PtrType>` | PtrType agnostic -- supports both shared_ptr and unique_ptr (D-06) |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `PtrType(new data_batch(...))` compiles for both `shared_ptr` and `unique_ptr` without warnings | Pitfall 2 / clone pattern | May need `if constexpr` dispatch or helper; minor implementation detail |
| A2 | `= default` move operations on accessor classes produce `noexcept` moves when both members (`PtrType` and lock guard) have `noexcept` move | ACC-06 | If not noexcept, `std::optional` and `std::vector` operations may copy instead of move; verify with `static_assert` |
| A3 | Explicit template instantiation of accessor classes in `.cpp` is sufficient to avoid linker errors for methods defined in the header | Pattern 6 | If methods are all inline in header, explicit instantiation is a no-op but harmless |
| A4 | Self-move-assignment on defaulted accessor move is safe | Pitfall 7 | Could cause double-unlock in pathological cases; verify with test |
| A5 | `static_assert(std::is_nothrow_move_constructible_v<read_only_data_batch<PtrType>>)` compiles with `= default` move | Code Examples | Depends on PtrType's move being noexcept (true for both `shared_ptr` and `unique_ptr`) |

## Open Questions

1. **Memory ordering for subscriber count**
   - What we know: Current code uses `memory_order_relaxed` everywhere (data_batch.cpp lines 142-158). PITFALLS.md Pitfall 13 notes this is fine if subscriber count is informational.
   - What's unclear: Whether downstream code uses subscriber count for synchronization decisions (e.g., "if count == 0, safe to evict").
   - Recommendation: Keep `relaxed` to match current behavior. Document that subscriber count is eventually consistent. If synchronization semantics are needed later, upgrade to `acquire`/`release`.

2. **clone() return type optimization for shared_ptr**
   - What we know: `PtrType(new data_batch(...))` works generically but `make_shared` is preferred for `shared_ptr` (single allocation).
   - What's unclear: Whether the performance difference matters at the batch creation rate.
   - Recommendation: Use `if constexpr` dispatch for correctness and to avoid compiler warnings. Zero runtime cost, cleaner code.

3. **Whether accessor methods should be inline or outlined**
   - What we know: Simple delegation methods (1-line getters) are natural candidates for inline. Current accessor methods (`operator->`) are inline in the header (data_batch.hpp lines 100-101).
   - What's unclear: Whether clang-format / code style prefers all methods inline for template classes, or whether longer methods (clone, convert_to) should be defined after the class.
   - Recommendation: Short delegation methods inline in the class definition. Longer methods (clone, clone_to, convert_to) defined below the class definitions but still in the header (required for templates). This matches the current `clone_to` template at data_batch.hpp lines 224-236.

## Project Constraints (from CLAUDE.md)

- **Sandbox mode:** All commands run through sandbox
- **Built-in tools preferred:** Read/Write/Glob/Grep over shell equivalents
- **Agent parallelism:** Use parallel agents where possible
- **Maximum reasoning effort:** ultrathink for all tasks
- **Verification loop:** Build and test after non-trivial changes
- **Code quality bar:** Principal engineer level -- question approach before diving in
- **Code style:** clang-format v20.1.4, column limit 100, indent 2, WebKit braces, pointer left-aligned [VERIFIED: .clang-format]
- **Naming:** snake_case classes, `_prefix` for private members, `PascalCase` enums, `UPPER_CASE` enumerators [VERIFIED: CLAUDE.md conventions]
- **Doxygen:** All public API methods documented with `@brief`, `@param`, `@return`, `@throws` [VERIFIED: existing headers]
- **Warning flags:** `-Wall -Wextra -Wpedantic -Wcast-align -Wunused -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion -Wformat=2 -Wimplicit-fallthrough` [VERIFIED: CLAUDE.md]
- **Build verification:** `pixi run build` (all 61 targets) and `pixi run test` [VERIFIED: PROJECT.md constraints]
- **C++ standard:** C++20 -- must compile with CUDA 12.9/13.x toolchains [VERIFIED: PROJECT.md]
- **License header:** Apache-2.0 with NVIDIA copyright [VERIFIED: all existing source files]

## Sources

### Primary (HIGH confidence)
- Current `data_batch.hpp` (lines 1-238) -- existing implementation being replaced
- Current `data_batch.cpp` (lines 1-175) -- existing implementation patterns
- `data_repository.hpp` (lines 1-285) -- template instantiation pattern, PtrType usage
- `data_repository.cpp` (lines 1-59) -- explicit template instantiation pattern
- `common.hpp` (lines 1-148) -- `idata_representation` interface, `clone()` virtual
- `representation_converter.hpp` (lines 1-257) -- converter registry interface
- `memory/common.hpp` (lines 1-85) -- `Tier` enum, `memory_space_id`
- `mock_test_utils.hpp` (lines 1-207) -- test utilities for data representation
- CONTEXT.md (D-01 through D-25) -- locked design decisions
- PITFALLS.md (Pitfalls 1-15) -- verified concurrency hazards

### Secondary (MEDIUM confidence)
- `.planning/research/ARCHITECTURE.md` -- build order and component boundaries
- `.planning/research/FEATURES.md` -- table stakes and anti-features
- `.planning/research/SUMMARY.md` -- prior research synthesis (note: some recommendations overridden by CONTEXT.md decisions)

### Tertiary (references, not verified in this session)
- CERT CON50-CPP (mutex destruction while locked) -- cited in PITFALLS.md
- WG21 N3427/N3568 (shared locking / no atomic upgrade) -- cited in PITFALLS.md
- cppreference std::shared_mutex, std::shared_lock, std::unique_lock

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- 100% C++ standard library, all primitives verified in current codebase
- Architecture: HIGH -- all design decisions locked in CONTEXT.md, patterns verified in existing code
- Pitfalls: HIGH -- comprehensive pitfall analysis from prior research phase, all verified against codebase

**Research date:** 2026-04-14
**Valid until:** 2026-05-14 (stable -- C++ standard library, no external dependency drift)
