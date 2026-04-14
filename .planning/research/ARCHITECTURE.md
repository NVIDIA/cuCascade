# Architecture Patterns

**Domain:** State-machine-via-move-semantics for data_batch concurrency
**Researched:** 2026-04-13

## Recommended Architecture

The refactored design replaces the nested `synchronized_data_batch` wrapper with 3 flat, peer-level classes in the `cucascade` namespace. The state machine is encoded entirely in the C++ type system: you cannot read data without a `read_only_data_batch`, you cannot mutate data without a `mutable_data_batch`, and you cannot hold a stale reference after a state transition because the moved-from object is compile-time dead.

### State Machine

```
                    to_read_only(shared_ptr<data_batch>&&)
              +--------------------------------------------+
              |                                            |
              v                                            |
    +-------------------+                        +---------+---------+
    | read_only_data_   |  to_idle(ro&&)         |   shared_ptr<     |
    | batch             |----------------------->|   data_batch>     |
    |                   |                        |   (idle state)    |
    | [shared lock]     |                        |                   |
    +-------------------+                        +---------+---------+
              |                                            |
              | to_mutable(ro&&)                           |
              | (releases shared,                          | to_mutable(shared_ptr<data_batch>&&)
              |  acquires exclusive                        |
              |  -- goes through idle)                     |
              v                                            v
    +-------------------+                        +-------------------+
    | mutable_data_     |  to_idle(rw&&)         |   mutable_data_   |
    | batch             |----------------------->|   batch           |
    |                   |                        |   (created from   |
    | [exclusive lock]  |                        |    idle)          |
    +-------------------+                        +-------------------+
              |
              | to_read_only(rw&&)
              | (releases exclusive,
              |  acquires shared
              |  -- goes through idle)
              v
    +-------------------+
    | read_only_data_   |
    | batch             |
    +-------------------+
```

### Component Boundaries

| Component | Responsibility | Communicates With | Header |
|-----------|---------------|-------------------|--------|
| `data_batch` | Owns data + mutex + subscriber count. Factory creates `shared_ptr`. All 6 static transition methods live here. `enable_shared_from_this` base. | Accessors (friend), `idata_repository`, `data_repository_manager` | `data_batch.hpp` |
| `read_only_data_batch` | RAII shared-lock holder. Exposes const-only data access (`get_batch_id`, `get_current_tier`, `get_data`, `get_memory_space`). Holds `shared_ptr<data_batch>` to extend parent lifetime. | `data_batch` (friend, calls private accessors) | `data_batch.hpp` (same file) |
| `mutable_data_batch` | RAII exclusive-lock holder. Exposes all read methods plus `set_data`, `convert_to<T>`. Holds `shared_ptr<data_batch>` to extend parent lifetime. | `data_batch` (friend, calls private accessors) | `data_batch.hpp` (same file) |
| `idata_repository<PtrType>` | Partitioned thread-safe storage of `shared_ptr<data_batch>`. Pop/get operations return idle-state batches. | `data_batch` via `get_batch_id()` (public, no lock needed) | `data_repository.hpp` |
| `data_repository_manager<PtrType>` | Multi-port routing, batch ID generation, downgrade candidate selection. | `idata_repository`, `data_batch` | `data_repository_manager.hpp` |
| `representation_converter_registry` | Type-keyed conversion dispatch. Called by `mutable_data_batch::convert_to<T>()`. | `idata_representation` hierarchy | `representation_converter.hpp` |

### Data Flow

**Batch creation (idle):**
```
data_batch::create(batch_id, representation) -> shared_ptr<data_batch>
```
Factory function enforces `shared_ptr` management required by `enable_shared_from_this`. The returned pointer is in idle state -- no lock held, data inaccessible except through transition.

**Read path:**
```
shared_ptr<data_batch> idle = ...;
read_only_data_batch ro = data_batch::to_read_only(std::move(idle));
// idle is now nullptr -- compile error if used
auto tier = ro.get_current_tier();  // const access
auto* repr = ro.get_data();         // const idata_representation*
shared_ptr<data_batch> idle2 = data_batch::to_idle(std::move(ro));
// ro is now moved-from
```

**Mutation path:**
```
shared_ptr<data_batch> idle = ...;
mutable_data_batch rw = data_batch::to_mutable(std::move(idle));
rw.set_data(new_repr);
rw.convert_to<host_data_packed_representation>(registry, space, stream);
shared_ptr<data_batch> idle2 = data_batch::to_idle(std::move(rw));
```

**Non-blocking try path:**
```
shared_ptr<data_batch> idle = ...;
auto maybe_ro = data_batch::try_to_read_only(idle);
// idle is CONDITIONALLY nullified: null on success, unchanged on failure
if (maybe_ro) {
    // idle is nullptr, maybe_ro holds the accessor
} else {
    // idle is unchanged, try again later
}
```

**Locked-to-locked transition (through idle internally):**
```
read_only_data_batch ro = ...;
mutable_data_batch rw = data_batch::to_mutable(std::move(ro));
// Internally: release shared lock, acquire exclusive lock
// ro is moved-from
```

**Repository integration:**
```
// Repositories store shared_ptr<data_batch> in idle state
auto repo = make_shared_data_repository();
auto batch = data_batch::create(42, std::move(repr));
repo->add_data_batch(batch);  // shared_ptr copy, idle state

// Consumer pops idle batch, transitions to access data
auto idle = repo->pop_data_batch();
auto ro = data_batch::to_read_only(std::move(idle));
```

## Header Organization

All three classes live in a single header (`data_batch.hpp`) because they form a tightly-coupled unit with mutual friend relationships. Separating them would create circular include problems or require a forward-declaration header that adds complexity with no benefit.

### Header Structure

```
data_batch.hpp
  |
  +-- Forward declarations: memory::memory_space
  |
  +-- class data_batch : public enable_shared_from_this<data_batch>
  |     |-- public: create() factory, get_batch_id(), subscribe/unsubscribe,
  |     |           clone(), clone_to<T>()
  |     |-- public: 6 static to_* transition methods + 2 try_* variants
  |     |-- private: constructor, _batch_id (const), _data, _rw_mutex, _subscriber_count
  |     |-- friend: read_only_data_batch, mutable_data_batch
  |     +-- private: data accessor methods only friends can call
  |
  +-- class read_only_data_batch
  |     |-- public: get_batch_id(), get_current_tier(), get_data(), get_memory_space()
  |     |-- public: move-only (delete copy)
  |     |-- private: constructor(shared_ptr<data_batch>, shared_lock)
  |     |-- private: shared_ptr<data_batch> _batch, shared_lock _lock
  |     +-- friend: data_batch
  |
  +-- class mutable_data_batch
        |-- public: all read_only methods + set_data(), convert_to<T>()
        |-- public: move-only (delete copy)
        |-- private: constructor(shared_ptr<data_batch>, unique_lock)
        |-- private: shared_ptr<data_batch> _batch, unique_lock _lock
        +-- friend: data_batch
```

### Friend Relationship Rationale

The friend relationships are bidirectional and minimal:

- **`data_batch` friends `read_only_data_batch` and `mutable_data_batch`**: So `data_batch` static methods can construct accessors (calling their private constructors).
- **`read_only_data_batch` and `mutable_data_batch` friend `data_batch`**: So the accessors can call private data-access methods on `data_batch` (the internal getters/setters that are not part of the public API).

This is the same mutual-friend pattern as the existing code -- just flattened from nested classes to peer classes.

### Key Difference from Current Design: Accessor Holds shared_ptr, Not Raw Pointer

The current accessors hold `synchronized_data_batch*` (raw pointer borrow -- does NOT extend lifetime). The new design has accessors hold `shared_ptr<data_batch>` (extends lifetime via `enable_shared_from_this`). This is the critical design change that addresses dhruv's review feedback.

**Consequence for data flow:** An accessor obtained from a `data_batch` keeps the batch alive even if all other `shared_ptr` copies are destroyed. This is safe because the accessor also holds the lock, so the batch cannot be in an inconsistent state.

**Consequence for construction:** `data_batch` MUST always be created via `data_batch::create()` which returns `shared_ptr`. Stack allocation and `unique_ptr` management are blocked by making the constructor private. This is the same pattern `notification_channel` already uses in this codebase (see `include/cucascade/memory/notification_channel.hpp`).

## Patterns to Follow

### Pattern 1: Static Transition Methods (Centralized State Machine)

**What:** All state transitions are static methods on `data_batch`, not methods on the accessor types.

**When:** Always. This is the core architectural decision.

**Why:** Centralizes the lock management logic in one place. The state machine is explicit in `data_batch`'s public API. Callers see the full transition graph without looking at accessor internals.

**Example:**
```cpp
class data_batch : public std::enable_shared_from_this<data_batch> {
public:
    // Idle -> locked (blocking)
    static read_only_data_batch to_read_only(std::shared_ptr<data_batch>&& batch);
    static mutable_data_batch to_mutable(std::shared_ptr<data_batch>&& batch);

    // Locked -> idle
    static std::shared_ptr<data_batch> to_idle(read_only_data_batch&& accessor);
    static std::shared_ptr<data_batch> to_idle(mutable_data_batch&& accessor);

    // Locked -> locked (through idle internally)
    static mutable_data_batch to_mutable(read_only_data_batch&& accessor);
    static read_only_data_batch to_read_only(mutable_data_batch&& accessor);

    // Idle -> locked (non-blocking, conditional move)
    static std::optional<read_only_data_batch> try_to_read_only(std::shared_ptr<data_batch>& batch);
    static std::optional<mutable_data_batch> try_to_mutable(std::shared_ptr<data_batch>& batch);
};
```

### Pattern 2: Rvalue Reference for Guaranteed Transitions, Lvalue Reference for Try Variants

**What:** Blocking transitions take `&&` (rvalue ref) to force the caller to `std::move()`. Non-blocking try variants take `&` (lvalue ref) because the caller keeps the pointer on failure.

**When:** `&&` for `to_read_only`, `to_mutable`, `to_idle`. `&` for `try_to_read_only`, `try_to_mutable`.

**Why:** Rvalue reference makes moved-from state visible at the call site. `std::move(idle)` is a deliberate act. After the call, `idle` is guaranteed nullptr. For try variants, the caller needs to keep the pointer if the lock fails -- hence lvalue ref with conditional nullification on success.

**Example:**
```cpp
// Blocking: caller gives up ownership, guaranteed to succeed (may block)
static read_only_data_batch to_read_only(std::shared_ptr<data_batch>&& batch) {
    auto ptr = std::move(batch);  // batch is now nullptr at caller
    std::shared_lock lock(ptr->_rw_mutex);
    return read_only_data_batch(std::move(ptr), std::move(lock));
}

// Non-blocking: caller keeps ownership on failure
static std::optional<read_only_data_batch> try_to_read_only(std::shared_ptr<data_batch>& batch) {
    std::shared_lock lock(batch->_rw_mutex, std::try_to_lock);
    if (!lock.owns_lock()) return std::nullopt;
    auto ptr = std::move(batch);  // batch is now nullptr only on success
    return read_only_data_batch(std::move(ptr), std::move(lock));
}
```

### Pattern 3: Private Data Accessors Behind Friend Wall

**What:** All methods that read or write data (`get_data()`, `set_data()`, `get_current_tier()`, etc.) are private on `data_batch`. Only friend accessor classes can call them.

**When:** Always. This is the compile-time enforcement mechanism.

**Why:** Prevents callers from accessing data on an idle `data_batch` without holding a lock. If you have a `shared_ptr<data_batch>`, the only useful things you can do are: transition it, get its batch ID, or manage subscribers. You MUST transition to an accessor to see the data.

**Example:**
```cpp
class data_batch : public std::enable_shared_from_this<data_batch> {
public:
    uint64_t get_batch_id() const { return _batch_id; }  // Lock-free, public
    bool subscribe();
    void unsubscribe();

private:
    friend class read_only_data_batch;
    friend class mutable_data_batch;

    // Only accessible through accessors
    memory::Tier get_current_tier() const;
    const idata_representation* get_data() const;
    const memory::memory_space* get_memory_space() const;
    void set_data(std::unique_ptr<idata_representation> data);

    template <typename T>
    void convert_to(representation_converter_registry& registry,
                    const memory::memory_space* target_memory_space,
                    rmm::cuda_stream_view stream);
};
```

### Pattern 4: Accessor Delegates to data_batch Private Methods

**What:** `read_only_data_batch` and `mutable_data_batch` expose public methods that delegate to `data_batch`'s private methods through the friend relationship.

**When:** Always. Accessors are thin wrappers, not data holders.

**Why:** Single source of truth for data access logic. Accessors just enforce const-correctness (read_only returns `const` results; mutable allows both read and write).

**Example:**
```cpp
class read_only_data_batch {
public:
    uint64_t get_batch_id() const { return _batch->get_batch_id(); }
    memory::Tier get_current_tier() const { return _batch->get_current_tier(); }
    const idata_representation* get_data() const { return _batch->get_data(); }
    const memory::memory_space* get_memory_space() const { return _batch->get_memory_space(); }
private:
    std::shared_ptr<data_batch> _batch;
    std::shared_lock<std::shared_mutex> _lock;
};

class mutable_data_batch {
public:
    // All read_only methods (same delegation)
    uint64_t get_batch_id() const { return _batch->get_batch_id(); }
    memory::Tier get_current_tier() const { return _batch->get_current_tier(); }
    const idata_representation* get_data() const { return _batch->get_data(); }
    const memory::memory_space* get_memory_space() const { return _batch->get_memory_space(); }

    // Mutable-only methods
    void set_data(std::unique_ptr<idata_representation> data) { _batch->set_data(std::move(data)); }

    template <typename T>
    void convert_to(representation_converter_registry& registry,
                    const memory::memory_space* target_memory_space,
                    rmm::cuda_stream_view stream) {
        _batch->convert_to<T>(registry, target_memory_space, stream);
    }
private:
    std::shared_ptr<data_batch> _batch;
    std::unique_lock<std::shared_mutex> _lock;
};
```

**Note: No operator-> or operator\*.** Unlike the current design which uses `operator->` to expose the inner `data_batch`, the new design uses explicit named methods. This is better because: (a) it lets `read_only_data_batch` expose a strict subset of methods without needing a `const data_batch*` cast, (b) it prevents callers from caching the raw pointer and outliving the lock.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Accessor Holding Raw Pointer to Parent

**What:** `read_only_data_batch` stores `synchronized_data_batch*` (raw pointer).

**Why bad:** If the `synchronized_data_batch` is destroyed while an accessor is alive, the accessor has a dangling pointer. This is the exact bug dhruv flagged in PR #99 review.

**Instead:** Store `shared_ptr<data_batch>` obtained via `shared_from_this()`. The accessor extends the parent's lifetime.

### Anti-Pattern 2: Exposing operator-> on Accessor Types

**What:** `read_only_data_batch::operator->()` returns `const data_batch*`.

**Why bad:** Callers can cache the raw pointer (`auto* ptr = ro.operator->()`), outliving the lock. Also, `const data_batch*` still exposes private methods to anyone with the pointer (UB if they call non-const through const_cast, but the interface tempts it).

**Instead:** Explicit named methods on accessors that delegate to `data_batch` private methods. No raw pointer escapes.

### Anti-Pattern 3: Letting data_batch be Stack-Allocatable

**What:** Public constructor on `data_batch` allows `data_batch batch(42, repr);`.

**Why bad:** `enable_shared_from_this` requires the object to be managed by `shared_ptr`. Stack-allocated objects that call `shared_from_this()` trigger undefined behavior (typically throws `bad_weak_ptr`).

**Instead:** Private constructor + public `static create()` factory returning `shared_ptr<data_batch>`. This is the same pattern `notification_channel` uses in this codebase.

### Anti-Pattern 4: Moving shared_ptr Inside try Variant on Failure

**What:** `try_to_read_only(shared_ptr<data_batch>&& batch)` takes rvalue ref, attempts lock, moves batch regardless of success.

**Why bad:** On lock failure, the caller's batch is gone (moved into the failed function). They can never retry.

**Instead:** `try_to_read_only(shared_ptr<data_batch>& batch)` takes lvalue ref. Only moves the pointer on success. On failure, the caller still owns the batch.

### Anti-Pattern 5: Inheritance Between Accessor Types

**What:** `mutable_data_batch : public read_only_data_batch` to reuse read methods.

**Why bad:** Implicit slicing risk. A function taking `read_only_data_batch` could receive a sliced mutable. Also, the lock types are different (`shared_lock` vs `unique_lock`) so the base class's lock member would be wrong.

**Instead:** Composition via delegation. Both accessor types independently call `data_batch` private methods. Code duplication is minimal (5 one-liner forwarding methods) and the type safety is worth it.

## Integration Points with Existing Code

### Repository Layer (`idata_repository`, `data_repository_manager`)

**Current state:** Repositories are templated on `PtrType` with two instantiations: `shared_ptr<synchronized_data_batch>` and `unique_ptr<synchronized_data_batch>`.

**Required changes:**

1. **Drop `unique_data_repository` entirely.** The new `data_batch` uses `enable_shared_from_this` which is incompatible with `unique_ptr` management. PROJECT.md confirms this is accepted scope.

2. **De-template the repository.** `idata_repository` becomes `idata_repository` (no template) storing `shared_ptr<data_batch>`. Alternatively keep the template but only instantiate for `shared_ptr<data_batch>` -- this depends on whether the template is useful for future extension. Recommendation: de-template. The `unique_ptr` path is dead, and a template with one instantiation is complexity without benefit.

3. **Update `data_repository_manager`.** Drop the `unique_ptr` SFINAE overload of `add_data_batch_impl`. The manager becomes `data_repository_manager` (no template).

4. **`pop_data_batch_by_id`** -- The existing implementation calls `it->get()->get_batch_id()`. In the new design, `shared_ptr<data_batch>::get()` returns `data_batch*`, and `get_batch_id()` is public on `data_batch`. So `(*it)->get_batch_id()` works directly. No lock needed for batch ID lookup.

5. **`get_data_batch_by_id`** -- Returns a copy of `shared_ptr<data_batch>`. Caller gets an idle-state batch. This is the natural integration: repositories store idle batches, callers transition them.

6. **Type aliases** -- Replace:
   ```cpp
   using shared_data_repository = idata_repository<std::shared_ptr<synchronized_data_batch>>;
   using unique_data_repository = idata_repository<std::unique_ptr<synchronized_data_batch>>;
   ```
   With:
   ```cpp
   using shared_data_repository = idata_repository;  // or just use idata_repository directly
   ```

### Converter Integration

**Current state:** `convert_to<T>()` is a method on the inner `data_batch` class, called through `mutable_data_batch::operator->()`.

**New design:** `convert_to<T>()` is a private method on `data_batch`, exposed through `mutable_data_batch::convert_to<T>()` (public forwarding method).

**No changes needed to `representation_converter_registry` itself.** The converter infrastructure is orthogonal to the state machine. The only change is how `convert_to` is invoked (through accessor method instead of operator->).

### Clone Integration

**Current state:** `clone()` and `clone_to<T>()` are methods on `synchronized_data_batch` that internally acquire a read lock.

**New design:** `clone()` and `clone_to<T>()` are public methods on `data_batch` that internally acquire a read lock. They return `shared_ptr<data_batch>` (new idle batch).

**Why they stay on `data_batch` rather than on `read_only_data_batch`:** Clone needs to acquire its own lock internally. If clone were on `read_only_data_batch`, the caller would already hold a shared lock, and the clone would try to acquire another shared lock on the same mutex -- which works with `shared_mutex` (re-entrant for shared) but is fragile and couples clone semantics to the caller's lock state. Better to have clone manage its own locking.

### Test Infrastructure

**Current test patterns to preserve:**
- `mock_data_representation` -- unchanged, orthogonal to state machine
- `make_mock_memory_space()` -- unchanged
- `create_simple_cudf_table()` -- unchanged
- Catch2 v2 test framework -- unchanged

**Test patterns that change:**
- Construction: `synchronized_data_batch batch(1, std::move(data))` becomes `auto batch = data_batch::create(1, std::move(data))`
- Lock acquisition: `batch.get_read_only()` becomes `data_batch::to_read_only(std::move(batch))`
- Try variants: `batch.try_get_read_only()` becomes `data_batch::try_to_read_only(batch)`
- Conversions: `read_only_data_batch::from_mutable(std::move(rw))` becomes `data_batch::to_read_only(std::move(rw))`
- Accessor access: `ro->get_batch_id()` becomes `ro.get_batch_id()` (no operator->, direct method)

## Build Order (Implementation Sequence)

### Phase 1: Core data_batch Class

**Files:** `include/cucascade/data/data_batch.hpp`, `src/data/data_batch.cpp`

**What to build:**
1. `data_batch` class with `enable_shared_from_this`, private constructor, `create()` factory
2. `const uint64_t _batch_id`, `_data`, `_rw_mutex`, `_subscriber_count`
3. Public: `get_batch_id()`, `subscribe()`, `unsubscribe()`, `get_subscriber_count()`
4. Private: `get_current_tier()`, `get_data()`, `get_memory_space()`, `set_data()`, `convert_to<T>()`
5. Friend declarations for accessor classes (forward-declared)

**Why first:** This is the foundation. Everything else depends on `data_batch` existing. You can compile and test the factory, batch ID, and subscriber count without any accessors.

**Testable independently:** Yes. `data_batch::create()`, `get_batch_id()`, subscriber count operations.

### Phase 2: Accessor Types

**Files:** Same header/cpp as Phase 1 (all in `data_batch.hpp` / `data_batch.cpp`)

**What to build:**
1. `read_only_data_batch` with `shared_ptr<data_batch>` + `shared_lock`, named accessors
2. `mutable_data_batch` with `shared_ptr<data_batch>` + `unique_lock`, named accessors + mutators
3. Move-only semantics (delete copy, default move)

**Why second:** Accessors depend on `data_batch`'s private interface from Phase 1. They cannot be built before `data_batch` has its private data-access methods.

**Testable independently:** Not yet -- need the static transition methods from Phase 3 to construct accessors.

### Phase 3: Static Transition Methods

**Files:** Same header/cpp

**What to build:**
1. `to_read_only(shared_ptr<data_batch>&&)` and `to_mutable(shared_ptr<data_batch>&&)` -- idle to locked
2. `to_idle(read_only_data_batch&&)` and `to_idle(mutable_data_batch&&)` -- locked to idle
3. `to_mutable(read_only_data_batch&&)` and `to_read_only(mutable_data_batch&&)` -- locked to locked
4. `try_to_read_only(shared_ptr<data_batch>&)` and `try_to_mutable(shared_ptr<data_batch>&)` -- non-blocking

**Why third:** Transition methods use both `data_batch` internals and accessor constructors. All pieces from Phases 1-2 must exist.

**Testable:** Yes. This is the first point where the full state machine can be exercised end-to-end.

### Phase 4: Clone Operations

**Files:** Same header/cpp

**What to build:**
1. `clone(new_batch_id, stream)` -- acquires read lock internally, returns new `shared_ptr<data_batch>`
2. `clone_to<T>(registry, new_batch_id, target_space, stream)` -- same with conversion

**Why fourth:** Clone depends on `_data` (private) and `_rw_mutex` (private). It also returns `shared_ptr<data_batch>` via `create()`. All of Phase 1 must be solid.

**Testable:** Yes. Clone + verify data independence.

### Phase 5: Repository De-templating

**Files:** `include/cucascade/data/data_repository.hpp`, `src/data/data_repository.cpp`, `include/cucascade/data/data_repository_manager.hpp`, `src/data/data_repository_manager.cpp`

**What to build:**
1. Replace `idata_repository<PtrType>` with non-template `idata_repository` storing `shared_ptr<data_batch>`
2. Remove `unique_data_repository` typedef
3. De-template `data_repository_manager` similarly
4. Remove SFINAE `add_data_batch_impl` overloads
5. Remove `unique_data_repository` explicit template instantiation from `data_repository.cpp`
6. Remove `unique_data_repository_manager` explicit template instantiation from `data_repository_manager.cpp`
7. Update `data/CMakeLists.txt` if source files change (unlikely -- same files, just de-templated)

**Why fifth:** Repositories depend on `data_batch`'s public API (`get_batch_id()`). They do NOT depend on accessors or transition methods. Technically this could be done in parallel with Phases 2-4, but doing it after ensures the new `data_batch` interface is stable before updating consumers.

### Phase 6: Test Migration

**Files:** `test/data/test_data_batch.cpp`, `test/data/test_data_repository.cpp`, `test/data/test_data_repository_manager.cpp`

**What to build:**
1. Rewrite `test_data_batch.cpp` to use new 3-class API
2. Add tests for static transition methods, try variants, compile-time safety
3. Add test for `enable_shared_from_this` (accessor outlives last external shared_ptr)
4. Update repository tests for de-templated API
5. Remove tests for `unique_data_repository` behavior

**Why last:** Tests validate the completed implementation. Run `pixi run test` to confirm all pass.

### Build Dependency Graph

```
Phase 1 (data_batch core)
    |
    v
Phase 2 (accessor types)
    |
    v
Phase 3 (static transitions)  <-- first fully testable checkpoint
    |
    v
Phase 4 (clone)                <-- second testable checkpoint
    |
    v
Phase 5 (repository de-template)
    |
    v
Phase 6 (test migration)      <-- final green build
```

Phases 1-4 are all within `data_batch.hpp` / `data_batch.cpp`. They can be developed as a single atomic change to those two files, with Phase 5 as a follow-up that ripples through the repository layer.

## Scalability Considerations

| Concern | Impact | Approach |
|---------|--------|----------|
| Lock contention under high reader count | Low -- `shared_mutex` allows concurrent readers | Same as current design, unchanged |
| Accessor lifetime extending batch lifetime | Memory held longer if accessor leaked | Document that accessors should be short-lived; no code fix needed, this is inherent to RAII |
| shared_ptr ref counting overhead | Negligible -- one atomic increment per transition | Transitions are infrequent compared to data processing |
| shared_from_this() cost | One weak_ptr lock per accessor creation | Negligible -- same cost as shared_ptr copy |
| De-templating repository removes unique_ptr path | No performance impact | unique_ptr path was unused by Sirius (the primary consumer) |

## Sources

- Current implementation: `include/cucascade/data/data_batch.hpp` (lines 57-238), `src/data/data_batch.cpp`
- PR #99 review feedback: documented in `.planning/PROJECT.md`
- `enable_shared_from_this` pattern precedent: `include/cucascade/memory/notification_channel.hpp` (line 28)
- Repository integration points: `include/cucascade/data/data_repository.hpp`, `src/data/data_repository.cpp`
- Repository manager: `include/cucascade/data/data_repository_manager.hpp`, `src/data/data_repository_manager.cpp`
- Build structure: `src/data/CMakeLists.txt`, `test/data/CMakeLists.txt`

---

*Architecture analysis: 2026-04-13*
