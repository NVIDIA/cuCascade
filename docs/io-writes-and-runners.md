# io layer: write APIs + runner model (`run` / `run_for` / `run_until`)

Hand-off document for agents and architects who need to understand, extend, or **merge other
branches into** this branch. It describes what changed in the io layer relative to `main`,
why, and where each concern now lives.

| | |
|---|---|
| Branch | `phase-2-runnable_ctx` (phase 1 was branch `runnable_ctx`) |
| Base | `bd4aeed` — *feat(io): sync cucascade io with latest changes in SiriusIO (#204)* |
| Phase 1 commit | `b2e8eee` — *experimental io witer* (single squashed commit; §1–§10) |
| Phase 1 size | 73 files, +20,310 / −2,845 (≈ half of it tests) |
| Phase 2 commits | Scheduling port from sirius (§11), on top of `b2e8eee`: `458109f` config knobs + policy, `aadc902` first-I/O stats + runner gauges, `db813e0` prefetch → `background`, `90372a1` uring engine, `4b1decc` tests, `68f03e3` benchmarks |
| Phase 2 size | 26 files, +2,557 / −118 (library +617; the rest tests and benchmarks) |
| Scope | `include/cucascade/io/**`, `src/io/**`, `test/io/**`, `test/CMakeLists.txt`, `benchmark/**`; phase 2 also `include/cucascade/cudf/datasource.hpp` (two diagnostics accessors) |
| Untouched | everything else outside the io layer (memory, data, the rest of the cudf layer, `idisk_io_backend`, `AGENTS.md`) |

> `AGENTS.md` is **stale** about disk I/O: it describes GDS/kvikIO `idisk_io_backend`
> implementations that do not exist (only the `pipeline` backend exists). This branch did not
> touch that area or update `AGENTS.md`.

---

## 1. TL;DR

1. **Writes.** `ioctx` gained a full write API: sync/async, single-segment and vectored
   (`write_segment` list), host- or device-sourced, plus `flush_async`, `commit_async` and
   `open_io_object_for_write`. Implemented for **io_uring** (local files), **REST/S3**
   (whole-object PUT / multipart upload) and **kvikio** (eager local writes).
2. **Runner model.** Reactors no longer own a background thread. Each thread that calls
   `ioctx::run()` / `run_for()` / `run_until()` builds a **private engine** (its own io_uring
   ring or curl multi handle and pinned staging), pulls work from a **shared, per-class
   request queue**, and destroys the engine on the same thread when it returns. One thread
   is one ring, so `IORING_SETUP_SINGLE_ISSUER | DEFER_TASKRUN | COOP_TASKRUN` are kept.
   `start()` / `shutdown()` still work: `start()` spawns `n_runner_threads` jthreads that
   call `run()`.
3. **Scheduling.** Every request has a class (`latency`, `read`, `write`, `background`) and a
   state with timestamps. A pure `scheduling_policy` decides what a runner may pull or
   dispatch (write budgets, latency reservation, background throttle), so writes cannot
   starve reads. Each runner holds several groups concurrently, which removes head-of-line
   blocking.
4. **Cache coherence.** Writes invalidate overlapping `prefetching_cache` chunks, both before
   submission and on completion. A commit invalidates the whole object. A new STALE chunk
   bit stops stale loads from being published.
5. **Designed for a future manager** (NOT implemented). A component that scales runner threads
   per ioctx up or down and pins them can be added on top of: the pull-based queue, the
   runner registry, per-request timestamps, `stats()`, and retire semantics (stop pulling,
   requeue untaken work, drain in-flight, destroy engine on the same thread).
6. **Phase 2 (§11).** Prefetch reads are `background` class and isolated from demand reads by
   per-runner policy (pass tiers, group sub-limit, always-on share, reserved floor);
   `uring::config` gains `slices_per_pass` and `scheduling`; `stats()` gains first-I/O delay
   and per-runner gauges. 1 MiB demand p99 during a 2 GiB prefetch: 328 ms → 8 ms; no
   regression elsewhere; no default changes.

---

## 2. Public API (exact, as committed)

### 2.1 New value types — `include/cucascade/io/types.hpp`

```cpp
enum class request_class : std::uint8_t { automatic = 0, latency, read, write, background };
inline constexpr std::size_t request_class_count     = 4;          // concrete classes
inline constexpr std::size_t latency_class_max_bytes = 256UL << 10;

enum class io_kind : std::uint8_t { read, write, flush, commit };
enum class request_state : std::uint8_t {
  queued, assigned, in_flight, copying, completed, failed, cancelled };

constexpr std::size_t   request_class_index(request_class) noexcept;   // latency=0 read=1 write=2 background=3
constexpr request_class resolve_request_class(request_class, io_kind, std::size_t total_bytes,
                                              bool background_hint = false) noexcept;
// automatic -> write for non-reads; background if hinted; latency if <= 256 KiB; else read.

struct io_options { request_class cls{request_class::automatic}; };

enum class write_durability : std::uint8_t { none, data_sync };
struct write_options {
  write_durability durability{write_durability::none};
  request_class    cls{request_class::automatic};       // automatic -> write
};

enum class write_mode : std::uint8_t { create_or_truncate, create_or_open, open_existing };
struct write_open_options {
  write_mode    mode{write_mode::create_or_truncate};
  std::uint64_t size_hint{0};      // local: fallocate(KEEP_SIZE); REST: picks PUT vs multipart early
  unsigned      permissions{0644}; // new local files
};

struct host_source   { const std::uint8_t* data{nullptr}; };
struct device_source { const std::uint8_t* data{nullptr};
                       ::cuda::stream_ref stream{cudaStream_t{nullptr}};
                       int device_id{-1}; };   // -1: filled from cudaGetDevice() by the ioctx
struct write_segment { range rng; std::variant<host_source, device_source> src;
                       bool is_device() const noexcept; const std::uint8_t* data() const noexcept;
                       std::size_t size() const noexcept; std::size_t offset() const noexcept; };

std::size_t validate_write_segments(std::span<const write_segment>);  // throws on null/overlap/overflow

struct class_stats { std::size_t queued_requests, queued_bytes;
                     std::chrono::nanoseconds last_queue_wait, max_queue_wait; };
struct queue_stats { std::array<class_stats, request_class_count> per_class;
                     std::size_t active_runners, idle_runners, in_flight_requests; };
```

Phase 2 appended `first_io_*` fields to `class_stats`, a `runners` vector to `queue_stats`, and
`runner_stats` (§11.6).

### 2.2 `ioctx` additions — `include/cucascade/io/io_context.hpp`

Non-virtual public entry points. They validate, classify, run the cache-invalidation
bridge, and forward to the backend hooks:

```cpp
std::shared_ptr<io_object> open_io_object_for_write(std::string path, write_open_options = {});

std::size_t host_write(const io_object&, std::size_t offset, std::size_t size,
                       const std::uint8_t* src, write_options = {});
exec::semi_future<std::size_t> host_write_async  (const io_object&, std::size_t offset, std::size_t size,
                                                  const std::uint8_t* src, write_options = {});
exec::semi_future<std::size_t> device_write_async(const io_object&, std::size_t offset, std::size_t size,
                                                  const std::uint8_t* src, ::cuda::stream_ref,
                                                  write_options = {});
exec::semi_future<std::size_t> writev_async      (const io_object&, std::vector<write_segment>,
                                                  write_options = {});
exec::semi_future<void> flush_async (const io_object&);
exec::semi_future<void> commit_async(const io_object&, write_durability = write_durability::none);

std::size_t run      (std::stop_token);
std::size_t run_for  (std::chrono::steady_clock::duration,   std::stop_token = {});
std::size_t run_until(std::chrono::steady_clock::time_point, std::stop_token = {});

virtual bool        supports_write() const noexcept;          // default false
virtual bool        supports_device_write() const noexcept;   // default false
virtual std::size_t active_runners() const noexcept;          // default 0
virtual queue_stats stats() const noexcept;                   // default {}
virtual void        reset_stats_peaks() noexcept;             // phase 2 (§11.6); default no-op
```

The read API gained a trailing defaulted `io_options opts = {}` on `host_read_async`,
`device_read_async` and the backend hook `mixed_readv_async_io(obj, slices, opts)`. Existing
callers compile unchanged. **Overriders of `mixed_readv_async_io` must add the parameter.**
That includes `kvikio_context`, `s3rdma_ioctx`, `templated_ioctx` and test stubs.

Protected backend hooks. The defaults fail with `std::errc::not_supported`, except
`run_impl`, which throws `std::logic_error`:

```cpp
virtual std::shared_ptr<io_object> create_io_object_for_write(std::string path, write_open_options);
virtual std::size_t host_write_io(const io_object&, std::size_t, std::size_t, const std::uint8_t*, write_options);
virtual exec::semi_future<std::size_t> mixed_writev_async_io(const io_object&, std::vector<write_segment>&&,
                                                             write_options) noexcept;   // sole async write hook
virtual exec::semi_future<void> flush_async_io (const io_object&) noexcept;
virtual exec::semi_future<void> commit_async_io(const io_object&, write_durability) noexcept;
virtual std::size_t run_impl(std::stop_token, std::optional<std::chrono::steady_clock::time_point> deadline);
```

The hooks receive already-validated, non-empty, disjoint segments with a resolved class and
resolved `device_id`. **There is no generic "is this object writable" check at the ioctx
level; every backend must reject writes to read-only or committed objects itself**, with
`std::invalid_argument`.

### 2.3 Coordinator finalizer — `include/cucascade/io/io_request.hpp`

```cpp
using finalize_fn = exec::invocable<void(grouped_coordinator&) noexcept>;
void grouped_coordinator::set_finalizer(finalize_fn fn);   // empty -> invalid_argument; twice -> logic_error
```

- **When it runs:** once, on the thread that would settle the last credit, **while still
  holding that credit**. The credit is settled afterwards.
  - That makes `add_tasks()` inside the finalizer safe; it is how follow-up work is chained
    (fdatasync after writes, S3 Complete after parts).
  - To fail the request from inside the finalizer, call `add_tasks(1)` then `report_error`.
- **Precondition:** the caller holds an unsettled credit.

`grouped_io_request` also gained:
- `request_meta`: id, class, kind, state, timestamps (enqueue / assign / complete).
- Write segments and options.
- `create_write(...)`, plus `create_control(...)` for flush and commit.
- `take_front_write_segment`, `take_control`, `control_pending`.
- `cancel_remaining`, which now also settles write segments and the control op.

---

## 3. Semantics (normative)

**Completion**
- A write future resolves once the kernel (local) or object store (REST) has **accepted** the
  bytes; the value is the total number of bytes requested.
- Local writes are visible to subsequent reads at that point.
- `write_durability::data_sync` additionally runs `fdatasync` before resolving (uring and
  kvikio; ignored by REST).

**Ordering and sources**
- **No ordering between requests**, even on the same object. Overlapping concurrent writes
  leave undefined content. Within one request, segments must be disjoint (`invalid_argument`
  otherwise) and may be written in any order.
- **Source buffers** must stay valid until the future resolves.
- **Device sources** are copied device-to-host on the **caller's stream** and ordered after
  the work already enqueued on it, so no caller-side sync is needed.

**Alignment**
- Never padded.
- uring uses O_DIRECT only for physical ops whose offset, length and buffer addresses are all
  4 KiB aligned; unaligned head and tail go through the buffered fd.
- An O_DIRECT `EINVAL` or `EOPNOTSUPP` falls back to buffered I/O.

**Open modes**
- Local backends: `create_or_truncate`, `create_or_open` and `open_existing`.
- REST: only `create_or_truncate`; anything else gives `not_supported`.
- `size_hint`: local `fallocate(KEEP_SIZE)`, which reserves space without changing the
  visible size. REST uses it to choose PUT or multipart early.
- Object size: an atomic high-water mark that grows as writes complete.

**Flush and commit**
- `flush_async`:
  - local: `fdatasync`; allowed on read-only objects (kvikio-compatible);
  - REST: resolves as a no-op.
- `commit_async`:
  - local: optional `fdatasync`, then the object becomes read-only;
  - REST: a single PUT (≤ `multipart_threshold`) or the remaining parts plus Complete; the
    object only becomes visible at commit.
- After a commit, further writes or a second commit fail with `std::invalid_argument`.
- `fdatasync` returning `EINVAL` is treated as success, with a warning logged once.

**Errors**
- `system_error(errno)` for I/O.
- `invalid_argument` for API misuse.
- `not_supported` when the backend lacks the capability.
- `operation_canceled` for requests cancelled by shutdown.

**Threading of completions.** Futures resolve on the **runner thread** that completed the
last operation; continuations installed with `install_callback` run there too.

**`run*()`**
- `run(token)` serves until the token is stopped or `shutdown()` is called. It does *not*
  return just because the queue is empty, which suits a thread pool.
- `run_for` and `run_until` also stop at the deadline. In-flight ops are always drained first,
  so a call can overrun by up to one op's latency.
  - REST write and commit groups are never requeued and are finished by the retiring runner,
    so they can overrun further.
- The return value is the number of grouped requests completed.
- Calling `run*()` re-entrantly on the same thread throws `std::logic_error`.
- **Each `run*()` call builds and destroys an engine.** For uring that is a ring plus 64 MiB
  of pinned staging and `io_uring_register_buffers` (roughly hundreds of µs), so do not call
  `run_for` in a tight loop.

**`start()` / `shutdown()`**
- `start()`:
  - opens admission and spawns `n_runner_threads` jthreads that each call `run()`;
  - waits until every spawned runner has built its engine, and rethrows the first failure
    (for example a pinned allocation failure).
- `shutdown()`:
  - closes admission and cancels queued requests with `operation_canceled`;
  - stops the runners; uring drains in-flight ops, REST cancels in-flight transfers (as
    before);
  - joins the threads and then renews the context stop source, so `start()` → `shutdown()` →
    `start()` works (this was broken at base).
- External `run()` callers count as runners too.
- **Requests submitted before `start()`**, with no runner active, are cancelled with
  `operation_canceled`, as at base.
- **If every runner dies from a fatal engine error,** queued requests fail with that error and
  new submissions fail immediately instead of hanging. A new `start()`, `shutdown()`, or an
  external `run*()` clears that state.

---

## 4. Architecture

### 4.1 Before → after

```
BEFORE (bd4aeed)                                  AFTER (b2e8eee)
templated_ioctx<Reactor>                          templated_ioctx<Reactor>
 ├─ vector<unique_ptr<Reactor>> _reactors          ├─ unique_ptr<Reactor> _reactor   (ONE shared dispatcher)
 ├─ next_reactor(): push to 2 least-backlogged     ├─ start(): n_runner_threads jthreads -> run_impl()
 └─ each Reactor:                                  └─ run_impl() on ANY thread:
     ├─ own MPMC queue + _enqueue_mutex                 slot   = hub.registry().register_runner()
     ├─ own jthread worker_loop()                       engine = reactor.make_engine(slot)   // ring / curl multi built HERE
     └─ ring / curl multi / staging as                  n      = engine->run(stop, deadline) // pull, dispatch, wait
        worker_loop locals                              engine destroyed on this thread; unregister
                                                   Reactor (thread-safe):
                                                     ├─ request_hub  (admission + per-class queue + registry + stats)
                                                     ├─ config / host MR / blocking helpers / host_read
                                                     └─ make_engine(slot) -> Reactor::engine_type
```

The old per-reactor `worker_loop` logic now lives in:
- `src/io/uring/uring_engine.cpp`, class `uring_engine`, pimpl;
- `src/io/rest/rest_engine.cpp`, class `rest_engine`, pimpl.

`uring_reactor.cpp` and `rest_reactor.cpp` shrank to thread-safe dispatchers.

### 4.2 Reactor concept v2 — `include/cucascade/io/templated_ioctx.hpp`

```cpp
template <class E> concept io_engine_c = requires(E& e, std::stop_token s, run_deadline d) {
  { e.run(s, d) } -> std::same_as<std::size_t>; };

template <class R> concept io_reactor_c = /* ... */ {
  typename R::io_object_type; typename R::reactor_config_type; typename R::engine_type;
  { const_reactor.get_config() }; { const_reactor.staging_block_size() } noexcept;
  { reactor.hub() } noexcept -> std::same_as<detail::request_hub&>;
  { reactor.host_read(object, offset, size, dst) } -> std::same_as<std::size_t>;
  { reactor.make_engine(slot) } -> std::same_as<std::unique_ptr<typename R::engine_type>>;
  { R::create_io_object(path) }; { R::supports(path_view) }; { R::align_and_coalesce(...) };
};

template <class R> concept io_writable_reactor_c = io_reactor_c<R> && requires(...) {
  { reactor.host_write(object, offset, size, src, options) } -> std::same_as<std::size_t>;
  { reactor.create_io_object_for_write(path, open_options) } -> std::same_as<std::unique_ptr<...>>;
};
```

Compared with base:
- **Removed from the concept:** `enqueue`, `queued_bytes`, `start`, `shutdown`, `interrupt`.
- **Added:** `engine_type`, `hub()`, `make_engine(slot)`.
- **Writes are opt-in** through `io_writable_reactor_c`; uring and REST both declare
  `supports_write`.
- **Reference minimal engine:** `test/io/stub_reactor.hpp`, which runs completions inline.
  `test/io/test_dispatch_failure_hook.cpp` uses it. Its failure seam is now a protected
  virtual `read_fanout`.

Constructors keep their old shape, `uring_ioctx(n, ctx)` and `rest_ioctx(n, ctx)`, but **`n`
now means the number of runner threads `start()` spawns**, not the number of reactors. It
still comes from `io_config::uring_n_reactors` (default 1) and `rest_n_reactors` (default 2).
`n = 0` means callers drive the context themselves through `run*()`.

### 4.3 Shared layer — `include/cucascade/io/details/`

| File | Role |
|---|---|
| `request_queue.hpp` | One moodycamel MPMC queue per concrete class; per-class counters and queue-wait stats; `pick`/pull support |
| `runner_registry.hpp` / `src/io/details/runner_registry.cpp` | Runner slots: per-slot **eventfd**, idle (parked) flag, wake-one-idle (round-robin), `wake_all`, wait-until-empty for shutdown |
| `request_hub.hpp` / `src/io/details/request_hub.cpp` | Owned by each reactor. Admission (`set_accepting`, `close_admission(reason)`, `rejection_reason`), `enqueue` (publish + wake one), `requeue`, `cancel_queued`, `try_pull(cls, ...)`, `try_park`/`unpark`, **`prepare_wait`**, `finish_group`, `fill_queue_view`, `stats` |
| `scheduling_policy.hpp` | Pure, header-only, unit-tested: `pick(view)` chooses the class to pull; `may_dispatch(cls, need, view)` gates each physical op. Phase 2 adds `background_pressure`, `background_reserve`, `refused_by_background_floor` (§11.5) |
| `event_fd.hpp` | `make_event_fd` (moved from `rest/curl_handle.hpp`, which keeps a forwarding `using`) |

`scheduling_config` defaults (uring: set through `uring::config::scheduling` and validated by
the reactor since phase 2; REST: default-constructed, not configurable):

```cpp
std::size_t max_active_groups{4};          // bulk groups per runner that still EXPAND (have unsubmitted work)
std::size_t max_latency_groups{2};         // latency groups per runner that still expand
double      write_slot_fraction{0.5};      // writes may use <= 50% of staging slots / connections
double      write_ring_fraction{0.5};      // ... and <= 50% of ring entries
double      background_slot_fraction{0.75};    // background share; ALWAYS applied since phase 2
std::chrono::milliseconds write_max_wait{20};  // write starvation guard
std::size_t reserved_latency_slots{2};
std::size_t max_background_groups{2};      // phase 2: background sub-limit inside max_active_groups
std::size_t reserved_background_slots{8};  // phase 2: floor read/write ops leave free for background
```

The group limits apply to `expanding_groups`: groups that still have untaken or undispatched
work. For reads, groups whose ops are all in flight are bounded only by slots, ring entries
or connections. Write, flush and commit groups count for as long as they are held. See §7
for why. Background groups count inside `max_active_groups` and are further limited to
`min(max_background_groups, max_active_groups)`, so a `read` group can always be pulled.

**Engine loop and wait rule.** Each pass: reap completions → poll CUDA events → pull
(`pick` + `try_pull`) → dispatch (`may_dispatch`) → retire finished groups (`finish_group`).
The uring engine (phase 2) dispatches the groups it holds in three tiers — `latency`, then
`read` + `write` in pull order, then `background` — expands at most `slices_per_pass` slices
per group per pass, and publishes its runner gauges once per pass (§11.5).
Before blocking, the engine calls `hub.prepare_wait(slot, …)`, which returns a `wait_action`:

- `pull_now` — work is available and allowed; loop again.
- `park` — nothing to do: park (idle flag), wait on the slot eventfd with the idle timeout,
  then **always** `unpark`.
- `wait_bounded` — queued work is visible but the policy refuses it right now. Do **not**
  park (parking would fail and busy-spin); wait at most about 1 ms on own completions and the
  eventfd.
- `wait_unparked` — at the group limit; block on own completions without parking.

**Retire vs shutdown on exit from `engine.run`.**
- `hub.accepting()` still true, so the runner is retiring (stop token or deadline): requeue
  untaken work, drain in-flight ops, return.
- Admission closed (shutdown): cancel with `operation_canceled`, then drain (uring) or abort
  transfers (REST).

**Wakeups.**
- uring: a **multishot `IORING_OP_POLL_ADD`** on the slot eventfd, with a one-shot fallback.
  It does not use a read SQE, because io_uring fails a read on an `O_NONBLOCK` eventfd with
  `-EAGAIN`.
- REST: the slot eventfd sits in the engine's epoll set.
- The base's fixed 20 ms polling tick (uring) is gone.

### 4.4 Request lifecycle

```
user thread: ioctx::writev_async / host_read_async ...
  -> validate + classify (resolve_request_class) + cache-invalidate (writes, phase 1)
  -> templated_ioctx: build grouped_coordinator + grouped_io_request(s) (meta: id, class, kind, t_enqueue)
  -> reactor.hub().enqueue()  -> per-class queue, wake one parked runner           [state: queued]
runner thread (inside run*):
  -> pick + try_pull                                                              [assigned]
  -> plan physical ops, may_dispatch, submit SQE / curl easy                      [in_flight / copying]
  -> completion -> finish_success / finish_error -> coordinator credit
  -> last credit: finalizer (optional chained work) -> promise fulfilled ON RUNNER [completed|failed|cancelled]
  -> write completion bridge: cache-invalidate (phase 2)
```

---

## 5. Per-backend behaviour

### 5.1 io_uring — `include/cucascade/io/uring/`, `src/io/uring/`

**Engine construction** (on the runner thread):
- Pinned staging of `clamp(64 MiB / block_size, 1, 64)` blocks from the host
  `fixed_size_host_memory_resource`. That is **per engine** now; at base it was per reactor
  at `start()`.
- A ring of depth `2 × slots` with `SINGLE_ISSUER|COOP_TASKRUN|DEFER_TASKRUN`, falling back
  to flags 0.
- `io_uring_register_buffers` over the staging.
- The eventfd poll is armed.
- The staging block size must equal the cache chunk size (the check in `io_context.cpp` is
  unchanged).

**Reads** behave as at base:
- chunking into 256 KiB–16 MiB ops (`dynamic_io_target`);
- direct contiguous host reads (≤ 1 GiB);
- `readv` coalescing of fragmented cache chunks;
- O_DIRECT with buffered fallback;
- short-read resubmit;
- `read_fixed` for single-block staged ops;
- staged device reads with H2D on the user's stream plus an event that is polled.

**Concurrency** (phase 2 changes in §11.5):
- Up to 4 bulk (at most 2 of them `background`) plus 2 latency *expanding* groups per runner.
- Dispatch tiers per pass: `latency` → `read` + `write` in pull order → `background`.
- Each group expands at most `slices_per_pass` (default 8) slices per pass, so the groups a
  runner holds share freed slots; queue depth stays bounded by the staging slots, not the cap.
- Bulk head-of-line fairness keeps large staged ops from being starved: a non-latency op that
  does not fit, or is refused only by the background floor, blocks later non-latency groups
  for the pass. Background ops that fit inside the background reserve are exempt.
- With 64 slots of 1 MiB: background holds at most 48 slots; `read` / `write` ops leave the
  last 8 free while background work is undispatched and below its share.
- Background staged ops (device reads, device-source writes) are sized to fit the background
  reserve: ≤ 8 MiB with 1 MiB blocks, instead of up to 16 MiB.
- An op carries an `engine_ticket`; destroying the op releases its per-class counts.

**Host writes:**
- Zero-copy from caller buffers, with contiguous segments coalesced into `writev`.
- O_DIRECT only for fully aligned middles; buffered head and tail.
- Ops are capped at 16 MiB so reads can interleave.
- Short writes, `EINTR` and `EAGAIN` are resubmitted.

**Device writes:**
1. Ops are sized by `dynamic_io_target` (up to 16 MiB) and take ⌈bytes / block⌉ staging
   blocks.
2. A batched `cudaMemcpyAsync` D2H into those blocks runs on the **caller's stream**,
   followed by `cudaEventRecord`.
3. The engine polls `cudaEventQuery`, backing off from 4 µs to 1 ms, and is never blocked in
   CUDA.
4. When the event fires, the write SQE is prepared (`write_fixed` or `writev`).

Writes may hold 50% of staging (32 MiB), which is about two 16 MiB ops per runner. In
practice that double-buffers D2H against disk writes. O_DIRECT staged writes are allowed only
if the staging blocks are 4 KiB aligned, which is checked at engine construction.

**`data_sync`.** A coordinator finalizer enqueues one `fdatasync` control request through
the **shared queue**, holding an extra credit; it is not runner-local and does not use
`IOSQE_IO_LINK`. The reason: after a retire, the request's last write may finish on a
different runner.

**`flush`** is `IORING_OP_FSYNC(DATASYNC)`. **`commit`** is an optional sync, then
`mark_committed`.

**Sync `host_write`** is a `pwrite` loop on the caller's thread, which works without runners.
The same holds for sync `host_read` (`pread`).

### 5.2 REST / S3 — `include/cucascade/io/rest/`, `src/io/rest/`

**Engine construction** (on the runner thread): curl multi, epoll, 3 timerfds (curl timer,
retry heap, upkeep), a per-engine connection `curl_share`, an easy-handle pool,
`slot_pool(max_connections=64)`, the retry heap, and lazy CUDA events. The global DNS/TLS
share is unchanged.

**Reads** are unchanged:
- ranged GETs with Content-Range / 200 validation;
- the footer stash fast path;
- retries with backoff, jitter and re-authorization;
- stall detection;
- H2D plus event polling.

The blocking helpers (`head_object`, `list_page`, `fetch_footer_suffix`,
`resolve_footer_batch`) still run on caller threads.

**`warmup()`.** The reactor records the bucket plus a generation number and calls
`wake_all()`; each engine primes its own pool when the generation changes. A new engine also
primes from a recent warm-up request (younger than `conn_max_age`). As a result, many short
`run_for` calls will each re-prime.

**Shared curl helpers** (callbacks, retry, backoff, headers) moved into
`src/io/rest/rest_helpers.hpp`, which is private. `sync_request.cpp` uses them too.

**Auth and S3 primitives (new):**
- `authorizer.hpp`:
  - `request_method { GET, HEAD, PUT, POST, DELETE_ }`
  - `struct request_spec { method, object_ref object, canonical_query, payload_sha256_hex = "UNSIGNED-PAYLOAD", extra_headers }`
  - `virtual authorized_request request_authorizer::authorize_request(request_spec const&, std::chrono::seconds)`;
    the default throws `credential_error`. Both SigV4 authorizers implement it.
- `s3::canonicalize_query`: bare keys get `=`; sorted by key, then value.
- `s3/xml_utils.hpp`: escape and unescape, `parse_s3_error` (only matches when the root is
  `<Error>`), `parse_initiate_multipart_upload`, `build_complete_multipart_body`.
  `list_parser.cpp` now uses these.
- `rest/details/sync_request.hpp`: `rest::detail::perform_sync(spec, authorizer, config, body, sync_request_options)`.
  It is a blocking single request with retries, and also retries 403 up to
  `max_auth_retry_attempts`. Options: `accepted_statuses`, `retry_on_error_body`,
  `data_transfer`. `sync_response` has `status`, `body`, `etag`, `attempts`,
  `content_length`.

**Writes** (`rest_upload.hpp/.cpp`, the per-object `upload_session`):
- `create_io_object_for_write` does no network I/O.
- Segments are split at part boundaries and **staged** into pinned blocks from the host MR
  (plain heap if there is none): host via memcpy, device via `cudaMemcpyAsync` on the caller's
  stream plus an event.
- Multipart is chosen by `size_hint` or by written size exceeding the threshold.
  - Initiate is lazy (triggered by the first full part).
  - The write that completes a part uploads it, and its future waits for that upload.
  - A part's staging is freed once its upload succeeds.
- `commit_async`:
  - waits for copies and uploads, then checks for holes (`invalid_argument` if any);
  - small objects: a single PUT;
  - otherwise: the remaining parts, then Complete chained through the coordinator finalizer.
- **Budgets:** data uploads use the write share of connections; Initiate, Complete and Abort
  use the latency budget.
- **Retries:** the GET rules, plus 400 RequestTimeout and Complete returning 200 with an
  `<Error>` body.
- **Lost Complete response:** a retried Complete that gets 404 NoSuchUpload does a blocking
  HEAD on the runner (up to 3 tries). If the object exists with the expected size, the commit
  counts as a success.
- **Abort:**
  - On failure: one async Abort from the engine, no retry.
  - On shutdown: `rest_ioctx::shutdown()` aborts every uncommitted upload **after** runners
    stop, so no part races the Abort; those objects fail with `operation_canceled`.
  - On drop: dropping an uncommitted object records its upload id without blocking, and a
    runner sends an async Abort (or shutdown cleanup does). If the reactor is already gone,
    only a warning is logged.
- **Rejected:** rewriting a range whose part was already uploaded gives `invalid_argument`;
  opening for update gives `not_supported`.
- **Config:** `config::write` (`rest_write_config`): `part_size` 16 MiB (clamped to
  5 MiB–5 GiB), `multipart_threshold` 16 MiB, `max_buffered_parts` 4 (a soft limit; a hard
  limit would deadlock).

### 5.3 kvikio — `include/cucascade/io/kvikio/`, `src/io/kvikio/kvikio_context.cpp`

kvikio still overrides `ioctx` directly; it is not a `templated_ioctx`. **It has no
runners**, and writes are eager: `mixed_writev_async_io` returns an already-resolved future.

- **Opening for write:**
  1. POSIX `open(O_RDWR|O_CLOEXEC[|O_CREAT|O_TRUNC])` first, so errors carry the real errno.
  2. `size_hint` → `fallocate(KEEP_SIZE)`.
  3. Then a kvikio `FileHandle "r+"`. kvikio's own `"a"` mode is O_APPEND and `"w"` always
     truncates, so neither fits.
- **Writes:**
  - Device sources: `cudaStreamSynchronize` once per distinct (device, stream), then
    `pwrite`s run in parallel on kvikio's pool.
  - Host sources: `pwrite`.
  - A short write raises `system_error(io_error)`.
- **Scheme:** `s3://` gives `not_supported`.

### 5.4 Cache coherence — `include/cucascade/io/cache/`, `src/io/cache/prefetching_cache.cpp`, `src/io/io_context.cpp`

```cpp
std::size_t prefetching_cache::invalidate_range(const io_object&, std::size_t offset, std::size_t size) noexcept;
```

**Chunk states:**
- A new `chunk_state` STALE bit (bit 48), with `chunk_state::invalidate()` returning a
  three-value enum.
- Invalidating a cached chunk sends it back to `allocated`; it keeps its buffer and fill
  extent and reloads in place.
- A loading or pinned chunk gets STALE instead:
  - new pins fail;
  - `mark_cached()` refuses it and returns false, so the chunk goes back to `allocated`;
  - the last unpin also sends it back to `allocated`.

**Two-phase invalidation** in `ioctx`, driven by the write entry points:
1. before enqueue;
2. again on write completion.

`commit_async` invalidates every chunk of the object once the commit settles, whether it
succeeded or failed, before the caller's future resolves. That is needed for REST, where
bytes only appear at commit.

**Guarantee:** after `invalidate_range` returns, no read is served bytes that were cached or
loading before the call. A reader racing a write may see old or new bytes.

**Lifetime.** Completion callbacks hold a `shared_ptr<write_invalidation_gate>`, not `this`.
`~prefetching_cache` closes the gate and waits for any invalidation in progress. Re-running
`initialize_cache` while writes are in flight is unsupported.

**Keying.** The cache key is `raw_file_cache_id`, the path. A write that grows a file gets no
cache slots past the size the cache saw (no resize).

---

## 6. Behaviour changes vs base (watch these when merging)

| Area | Base `bd4aeed` | Now |
|---|---|---|
| Reactor threads | 1 jthread per reactor, started by `start()` | Runner threads call `run*()`; `start()` spawns `n` of them |
| Dispatch | `next_reactor` pushes to the 2 least-backlogged reactors; per-reactor queue | Shared per-class queue; runners pull; `next_reactor` **removed** |
| uring wakeup | 20 ms timeouts; enqueue does not wake a CQE wait; `interrupt()` no-op | eventfd poll on the ring; immediate wake |
| uring busy-spin | Possible (pending ops, nothing in flight, copies outstanding) | Fixed (unconditional wait when no progress) |
| Concurrency per reactor or runner | 1 grouped request at a time | Several groups per runner (expanding-group limits) |
| Staging | Allocated at reactor `start()`, freed at reactor destruction | Per engine, allocated and freed on the runner thread |
| Restart | `start()` after `shutdown()` broken (stale stop token) | Works |
| Dead runners | n/a | Queue fails fast with the engine error |
| REST shutdown | Cancels in-flight transfers | Same, plus Abort of uncommitted uploads |
| `make_event_fd` | `rest/curl_handle.hpp` | `io/details/event_fd.hpp` (forwarding `using` kept) |
| Read hook | `mixed_readv_async_io(obj, slices)` | `+ io_options opts` |
| REST loopback test server | Nagle on | `TCP_NODELAY` (≈40 ms → 39 µs per reused connection) |
| 64 KiB host read latency (page cache) | ≈ 6 µs | ≈ 8 µs (eventfd wake cost) |

---

## 7. Performance (measured on this branch)

Phase 2 re-measured these paths against `b2e8eee` (all within ±1–2 %) and added the
demand-vs-prefetch results: §11.8.

Setup: Samsung 990 PRO (rated ≈6.9 GB/s sequential write), dm-crypt + LVM, ext4. **Not**
`/mnt/disk_2`, which was not mounted. Reads use TPC-H sf100 `lineitem.parquet`: 4 columns,
14,652 ranges, 5.05 GB. "Cold" means `posix_fadvise(DONTNEED)` before each run. 2 runners,
medians of interleaved runs (5 cold / 3 warm).

**Reads, GiB/s**

| Path | Base | Now |
|---|---|---|
| cudf host, cold (per-range `host_read_async`) | 3.50 | 3.64 (104%) |
| io_context device (per-range `device_read_async`) | 4.82 | 5.03 (104%) |
| cudf device | 4.86 | 5.04 (104%) |
| io_context **vectored** host, cold | 1.36 | 3.58 |
| warm (cudf host / io_context host) | 12.03 / 11.62 | 11.90 / 11.78 |

- The per-range paths are at parity; +4% is near noise.
- The vectored gain is the old reactor being under-utilized on single large vectored requests
  (one group at a time, slice-by-slice planning). That explanation is inferred from the code
  and the observed sensitivity to queue depth; queue depth was **not directly measured**.
- History: the first runner version regressed per-range reads by 11–18%. Groups counted
  against `max_active_groups` until all their ops *completed*, which capped a runner at about
  6 single-op requests. Fixed by counting only *expanding* groups. Letting write groups stop
  counting too was tried; it queued about 512 MiB of writes and pushed read p99 to 6–8 ms,
  so write, flush and commit groups still count while held.

**Writes, 2 GiB in 16 MiB blocks, GB/s.** `dd` was blocked, so the baseline is a `pwrite`
loop doing the same thing.

| Path | Write | Incl. flush |
|---|---|---|
| `pwrite` O_DIRECT, 1 thread | 5.79–5.98 | 5.78 |
| uring host O_DIRECT | 6.00–6.18 | 6.07 |
| uring device source | 5.57 | 5.51 |
| uring buffered / `pwrite` buffered | 6.95–7.10 / 7.0–7.2 | 3.12 / 3.1–3.2 |
| uring `data_sync` per request | 4.86 | — |
| kvikio host O_DIRECT / device | 4.87 / 3.83 | — |

uring is at drive-limited parity with `pwrite`. After about 10 GB of sustained writes, every
path drops to 1.4–2 GB/s once the SSD's write cache fills.

**4 KiB read latency during 2 GiB writes (p50 / p99)**

| Scenario | p50 | p99 |
|---|---|---|
| uring, idle | 47 µs | ≈ 0.3 ms |
| uring, during host writes | 0.74 ms | 5.1 ms |
| uring, during device writes | 0.6 ms | 2.6–3.6 ms |
| `pread` during `pwrite` | 0.29 ms | 2.6 ms |

Benchmark tool: `benchmark/io_write_benchmark.cpp` → `cucascade_io_write_benchmark`. It
supports write and mixed modes, a built-in `pwrite` baseline, and `pause_ms`; see
`benchmark/README.md`.
- Hidden Catch2 bench `[uring-write-bench]`, configured with `CUCASCADE_WRITE_BENCH_DIR`,
  `_GIB` and `_RUNNERS`.
- **Pitfall:** building one target relinks the shared `libcucascade_io.so`, and stale
  benchmark binaries in the same build dir then crash ("double free"). Rebuild all benchmark
  targets together.

---

## 8. Tests

All io tests are in `cucascade_io_tests` (Catch2 v3, built with `CUCASCADE_BUILD_IO`). At
`b2e8eee`:

| Suite | Result |
|---|---|
| `cucascade_io_tests` | 291 cases, ≈ 8,250 assertions; passed 3× |
| `[rest]` | 84 cases; passed 10× |
| `cucascade_tests` | 101 cases |
| `cucascade_cudf_tests` | 188 cases |
| `cucascade_topology_discovery_tests` | 9 cases |
| pre-commit (clang-format 20.1.4, codespell, cmake-lint) | clean |

At phase 2 (`68f03e3`): `cucascade_io_tests` 310 cases (+19), ≈ 8,535 assertions, passed;
`[scheduling]` passed 6×; the other suites are unchanged in count and passed. The phase-2
benchmark additions (`mode=readmix`, the scheduling knobs, `slices_per_pass` arguments of
the parquet benchmarks) are documented in `benchmark/README.md`.

New test files (phase 1; phase 2 cases are listed in §11.7):

| Area | Files |
|---|---|
| Runner model | `test_templated_ioctx_runner.cpp`, `test_runner_registry.cpp` (incl. a 5,000-request lost-wakeup stress), `test_request_queue.cpp`, `test_scheduling_policy.cpp`, `stub_reactor.hpp` |
| Coordinator | `test_coordinator_finalizer.cpp` |
| uring | `uring/test_uring_runner.cpp`, `uring/test_uring_write.cpp` |
| REST | `rest/test_rest_runner.cpp`, `rest/test_rest_write.cpp`, `rest/test_sync_request.cpp`, `rest/s3/test_sigv4_write.cpp`, `rest/s3/test_xml_utils.cpp`, `rest/loopback_object_store.hpp` (in-memory S3 with fault injection and a query-routing authorizer), `mock_authorizer.hpp::authorize_request` |
| Cache | `test_chunk_state_stale.cpp`, `cache/test_invalidate_range.cpp` (needs `friend struct prefetching_cache_test_access`) |
| kvikio | `kvikio/test_kvikio_write.cpp`, compiled under `CUCASCADE_BUILD_CUDF` |
| Scheduling (phase 2) | `uring/test_uring_scheduling.cpp` (new, `[io][uring][scheduling]`); new cases in `test_scheduling_policy.cpp`, `test_request_queue.cpp`, `test_runner_registry.cpp`, `cache/test_invalidate_range.cpp` |

```bash
pixi run cmake --build build/relwithdebinfo            # everything (benchmarks too)
build/relwithdebinfo/test/cucascade_io_tests           # all io tests
build/relwithdebinfo/test/cucascade_io_tests "[rest]"  # tag subset
# GPUs shared with other processes? shrink the test pool:
CUCASCADE_TEST_GPU_POOL_BYTES=536870912 build/relwithdebinfo/test/cucascade_io_tests
```

Device tests skip cleanly if `cudaMalloc` fails. `test_data_repository` in
`cucascade_tests` aborts with OOM when another process holds the GPUs; that is unrelated to
this branch.

---

## 9. Merge / integration guide

### 9.1 Conflict hotspots

These files were rewritten or heavily restructured. Expect conflicts with any branch that
touched them since `bd4aeed`:

| File | Nature of change |
|---|---|
| `include/cucascade/io/templated_ioctx.hpp` | Full rewrite (+841): concept v2, single reactor, runner lifecycle, `run_impl`, write fan-out |
| `src/io/uring/uring_reactor.cpp` (−1,084) → `src/io/uring/uring_engine.cpp` (new, 2,114) | Old `worker_loop` and its lambdas moved into `uring_engine::impl` member functions; logic kept, then extended (multi-group, writes, wake) |
| `src/io/rest/rest_reactor.cpp` (−1,402) → `src/io/rest/rest_engine.cpp` (new, 2,415) + `rest_helpers.hpp` | Same for REST; blocking helpers stay in `rest_reactor.cpp` |
| `include/cucascade/io/uring/uring_reactor.hpp`, `rest/rest_reactor.hpp` | Now dispatcher classes: hub, `make_engine`, no worker thread or queue |
| `include/cucascade/io/io_context.hpp`, `src/io/io_context.cpp` | Write/run API, cache bridge, `io_options` on reads |
| `include/cucascade/io/io_request.hpp` | `request_meta`, write segments, control ops, finalizer, settle refactor |
| `include/cucascade/io/types.hpp` | New types (§2.1) |
| `include/cucascade/io/cache/{types,prefetching_cache}.hpp`, `src/io/cache/prefetching_cache.cpp` | STALE bit, `invalidate_range`, gate |
| `include/cucascade/io/rest/{authorizer,config,types,rest_ioctx}.hpp`, `src/io/rest/s3/{sigv4_authorizer,list_parser}.cpp` | New methods, write config, shared XML |
| `include/cucascade/io/kvikio/kvikio_context.hpp`, `src/io/kvikio/kvikio_context.cpp` | Write support |
| `test/CMakeLists.txt`, `src/io/CMakeLists.txt` | Many new sources and tests (append-only; usually easy) |

### 9.2 Porting an incoming change

| If the incoming branch… | Port it to… |
|---|---|
| modifies the uring `worker_loop` (planning, CQE handling, short reads, O_DIRECT fallback, H2D copy, drain) | The same-named logic in `uring_engine::impl` (`uring_engine.cpp`): `plan_slice` / `plan_*_write`, `dispatch_one`, `reap_completions`, `resubmit_incomplete`, `start_device_copy`, `poll_copy_completions`, drain. Loop locals are now `impl` members. |
| modifies the REST `worker_loop` (`setup_easy`, `finish`, retries, `physical_ranges`, write callback, upkeep/warmup) | `rest_engine.cpp`; shared free functions are in `src/io/rest/rest_helpers.hpp`. |
| changes REST blocking helpers (`head_object`, `list_page`, footer resolve) | Still in `rest_reactor.cpp`, unchanged in structure. Consider `perform_sync` for new blocking calls. |
| adds a method to the reactor used by `templated_ioctx` | Check concept v2 (§4.2); add it to the reactor (thread-safe) or the engine (runner-thread-only). |
| relies on `next_reactor`, `_reactors`, per-reactor `enqueue` / `queued_bytes` / `start` / `shutdown` / `interrupt` | Gone. Use `reactor()`, `n_runner_threads()`, `hub().enqueue`, `hub().stats()`, ioctx `start` / `shutdown`. |
| adds an `ioctx` subclass or overrides `mixed_readv_async_io` | Add the `io_options opts` parameter. Override the write hooks if it can write. |
| adds ring setup, buffer registration or per-thread resources | Do it in the engine constructor (runner thread) and release in its destructor. Never touch an engine from another thread. |
| changes `grouped_coordinator` / `grouped_io_request` | Keep the finalizer invariant (finalizer runs holding the last credit) and the `request_meta` state transitions. |
| changes cache chunk states | Keep the STALE semantics (§5.4) and the `write_invalidation_gate` lifetime rule. |
| changes `io_config` reactor counts | They now mean runner threads spawned by `start()`. |

### 9.3 Invariants to keep

1. **One engine, one thread.** The ring, curl multi, curl share, easy handles and staging are
   created, used and destroyed only on the runner thread that owns them.
2. Reactors and the hub are thread-safe; engines are not.
3. A runner must unregister from the registry before its thread exits, because thread ids get
   reused. `templated_ioctx` already guarantees this.
4. **Never park when `prepare_wait` says `wait_bounded` or `wait_unparked`**; otherwise the
   runner busy-spins.
5. A local write may only complete once its bytes are visible to later reads (the cache's
   second invalidation phase relies on this).
6. `staging_block_size()` must equal the cache chunk size even when no engine is alive. It is
   derived from the reactor context, not from an engine.
7. Writes from different requests are unordered; do not add implicit ordering assumptions.
8. In-flight ops are never abandoned at a deadline; they are drained.

---

## 10. Known limitations / follow-ups

- **Scaling manager not implemented.** The hooks are in place: `stats()` with per-class queue
  wait, first-I/O delay and per-runner gauges (phase 2), `reset_stats_peaks()`,
  `active_runners()`, retire semantics, and caller-owned runner threads (pinning).
- **Prefetch isolation is per runner** (phase 2, §11.5): N runners admit up to N × 48
  background slots and 2N background groups; there is no global prefetch bound and no
  dedicated prefetch runner. Demand p99 during a prefetch is ≈ 8 ms at 1 runner and ≈ 20–28 ms
  at 2–4 (§11.8); `background_slot_fraction` lowers it at the cost of prefetch bandwidth
  under demand. REST gets the new policy defaults but not the uring-only tiers, per-pass cap
  or background op sizing, and cannot configure them.
- A demand read blocked on an in-flight prefetch (`await_inflight_prefetch`) does not promote
  it; the prefetch keeps only the background floor while demand is busy. The cost is visible
  in `prefetching_handle::demand_wait_ns()`.
- Runner gauges vanish from `stats().runners` when a runner leaves `run*()`, with its
  counters; REST publishes no op gauges (zeros).
- Every `run*()` call rebuilds its engine (64 MiB pinned staging per uring engine), so short
  `run_for` loops are wasteful. A manager should keep runners long-lived.
- Read p99 during heavy uring writes (about 5 ms) is roughly twice that of `pread` + `pwrite`;
  tunable via `write_slot_fraction` and op size.
- The REST lost-Complete HEAD check blocks the runner thread (rare path); it could be made
  async.
- An orphan-upload Abort that is cancelled by shutdown, or that fails, only logs a warning.
- REST `data_sync` is ignored. REST supports only `create_or_truncate`, with no in-place
  updates (S3 has no primitive for that).
- kvikio device writes reach about 66% of `pwrite`, because of the stream sync plus kvikio's
  bounce buffers.
- `parquet_s3_io_benchmark.cpp` was fixed for API drift that was already present at base, but
  is not compile-verified (AWSSDK is absent).
- A TSan race exists only in the loopback test fixture (`_listen_fd`), not in library code.
- `benchmark/README.md` documents the new write benchmark and (phase 2) `mode=readmix` and the
  scheduling knobs. `AGENTS.md` and `docs/ARCHITECTURE.md` were not updated.
- All performance numbers (§7, §11.8) are from a non-target disk; re-measure on `/mnt/disk_2`.

---

## 11. Phase 2: scheduling port from sirius `perf/uring-slices-per-pass-readahead`

Source: sirius commits `3bbd30a83..1703b0dfd` on `26dfc15ca`, whose `src/io` equals sirius
`34f89bab3`, the sync point of `bd4aeed`. Sirius builds cuCascade with `CUCASCADE_BUILD_IO OFF`
and keeps its own `src/io`, so the port is one-way and changes nothing in sirius.

### 11.1 What sirius fixed, and what of it existed here

| Sirius problem (one active request per reactor) | State at `b2e8eee` | Phase 2 |
|---|---|---|
| One slice expanded per loop pass, then a wait for a completion: a many-slice request (a whole-split prefetch) ran at queue depth 1–2 per reactor | **Absent.** `uring_engine::impl::advance_group` expands slices until slots, policy or SQEs refuse, and `run()` loops without waiting while anything progressed: depth is bounded by the staging slots (≤ 64) | `slices_per_pass` kept as a **fairness** knob only (§11.3) |
| FIFO reactor queue + one active request: demand reads queued behind GBs of prefetch | **Mostly absent** (per-class lanes, several groups per runner, `pick` prefers latency / read). Left: (a) prefetch was untagged, so it shared the `latency` / `read` lanes with demand; (b) on one runner the first-pulled bulk group took every freed slot; (c) a ≥ 256 MiB prefetch fans out into 4 groups (`templated_ioctx::read_fanout`) and filled all of `max_active_groups` | Prefetch → `background`; per-runner isolation policy (§11.5) |
| No visibility into queue delay or depth | `stats()`: per-class queued / last / max queue wait | First-I/O delay per class, per-runner gauges, `reset_stats_peaks()` (§11.6) |
| Defaults: 1 reactor; local readahead budget on | Different meaning here | Not ported (§11.2) |

### 11.2 Sirius change → cuCascade outcome

| Sirius (commit) | cuCascade (commit) |
|---|---|
| `uring::config::slices_per_pass{8}`, `max_slices_per_pass = 64`, YAML range check (`88bb31faa`) | Same field, default and bound (`458109f`), applied per group per pass in `advance_group` (`90372a1`); validated by the `uring_reactor` constructor (no YAML layer here). **Different semantics:** a fairness cap, not a depth bound; `slices_per_pass = 1` does not reproduce sirius's legacy QD 1–2 (§11.3) |
| One `submit_prepared()` per pass (`88bb31faa`) | Already present (`_prepared` + `flush_submissions`) |
| `io_class {demand, prefetch}` on `prepared_io_slice`, set in `prefetching_cache::prefetch` (`1703b0dfd`) | `io_options{request_class::background}` at the same call site (`db813e0`). No new enum or slice field: cuCascade classifies per request. Demand loads stay `automatic` (`latency` ≤ 256 KiB, else `read`) |
| `uring.prefetch_reactors`: the last K reactors serve only prefetch (derived 1 when the readahead runs), routed by `templated_ioctx::next_reactor` (`1703b0dfd`) | **Not ported as routing** — `next_reactor` and per-reactor queues no longer exist. Re-expressed as per-runner policy (`458109f`, `90372a1`): 3-tier pass order, `max_background_groups` sub-limit (clamped), always-on `background_slot_fraction`, `reserved_background_slots` floor, background staged ops sized to the reserve, floor-refusal FIFO block with a reserve exemption (§11.5). The global bound sirius got from "K of N reactors" maps to `background_slot_fraction` ≈ K/N per runner. Dedicated prefetch runners: follow-up (§11.10) |
| `default_uring_n_reactors` 1 → 4 (`3bbd30a83`) | **Not ported.** `io_config::uring_n_reactors{1}` (runner threads). Sirius needed more reactors to escape QD 1–2 each; one runner here already runs up to 64 ops over several groups. Each runner costs up to 64 MiB of pinned staging, and `start()` throws if it cannot get it. Measured (P4, §11.8): the default stays 1 |
| `uring::config::n_max_concurrent_scans` → 0, plus the `resolve_readahead` zero-budget opt-out (`3bbd30a83`) | **Not ported.** Advisory here (`ioctx::n_max_concurrent_scans()`, no cuCascade consumer). Sirius's own medians favour readahead *on* once head-of-line blocking is fixed (off 36.7 s, on 34.5 s, on + isolation 33.7 s), so "0" would encode a conclusion drawn from the old reactor |
| Reactor gauges, `take_gauges()`, a 250 ms sampler thread and a DEBUG `[uring_gauges]` line (`88bb31faa`, `1703b0dfd`) | Data, not logs (`aadc902`, `90372a1`): `class_stats::first_io_*`, `queue_stats::runners`, `ioctx::reset_stats_peaks()`. No sampler thread: `CUCASCADE_LOG_*` is compiled out, so the host samples and logs (§11.6) |
| Queue delay = enqueue → first slice expanded, per `io_class`, counted when the first slice is taken | First-I/O delay = enqueue → first physical op submitted, per `request_class`, recorded when the request retires (§11.6) |
| `cache_handle::demand_wait_ns()`, `sirius_datasource::demand_wait_ns()` / `cache_chunk_count()` | Mechanical port (`db813e0`): `prefetching_handle::demand_wait_ns()`, `datasource::demand_wait_ns()`, `datasource::cache_chunk_count()`. No consumer in cuCascade |
| `SIRIUS_PREFETCH_WINDOW_MIB` (temporary env knob) | Added and removed within the sirius branch; nothing to port |
| `test_uring_readv.cpp` cases (depth per spp, queue delay per class, byte-exactness, uncapped shutdown) | Re-expressed for runners in `test_uring_scheduling.cpp` (`4b1decc`, §11.7) |
| Sirius-only, not ported | Scan manager (readahead arming, `[readahead]` / `[split]` timelines, cache-cycle line, REST budget derivation); `sirius_config` YAML keys, derivation and validation; the `SiriusContext` pinned-resource rollback (`2eb20830c`); test YAML pins `uring_n_reactors: 1` (`7fd9b8df3`); `test_templated_ioctx.cpp` routing tests; `test_scan_manager_config.cpp` |

### 11.3 `slices_per_pass`

- `advance_group` makes at most `slices_per_pass` `plan_next` calls (slices, or write segments)
  per group per `process_groups` pass; 0 = no cap (`effective_slices_per_pass` maps it to
  `SIZE_MAX`). Draining never plans, so the cap is inert there.
- A capped group returns with `progressed = true`: `run()` loops again at once and the group
  still counts as expanding. The cap bounds how much of **one pass** a group may claim, not its
  queue depth. Without it the first-pulled group takes every freed slot until it has no
  untaken slices; with it the groups a runner holds share freed slots round-robin.
- The cap is per group: a request fanned out into k groups expands up to k × `slices_per_pass`
  per pass.
- Cost for a lone group: ⌈64 / `slices_per_pass`⌉ passes (one `io_uring_submit` each) to fill
  the slots instead of one; the steady-state refill is unchanged.
- `static_assert(max_slices_per_pass == MAX_NUM_SLOTS)` in `uring_engine.cpp`: a larger cap
  could never take effect.
- Observed (`test_uring_scheduling.cpp`, one runner, a 192-slice group A pulled before an
  8-slice group B, page-cache reads): B's first completion is the 9th overall (index 8) with
  `slices_per_pass = 8`, and comes after all 192 of A's with 0.
- Throughput (P2, §11.8): 0, 1 and 8 are within 1 % on every demand-only parquet path, so the
  default stays 8.

### 11.4 Knobs and validation

`uring::config` (fields appended after `use_odirect`, so designated initialisers keep working):

| Field | Default | Valid | Otherwise |
|---|---|---|---|
| `slices_per_pass` | 8 | 0..`max_slices_per_pass` (64); 0 = no cap | `std::invalid_argument` |
| `scheduling` | `io::detail::scheduling_config{}` | see below | |

`io::detail::scheduling_config` (`include/cucascade/io/details/scheduling_policy.hpp`):

| Field | Default | Valid | Otherwise / notes |
|---|---|---|---|
| `max_active_groups` | 4 | ≥ 1 | 0 → `std::invalid_argument` |
| `max_latency_groups` | 2 | ≥ 1 | 0 → `std::invalid_argument` |
| `max_background_groups` (new) | 2 | ≥ 1 | 0 → `std::invalid_argument` (the config is shared by every runner, so 0 would leave background unserved). Above `max_active_groups` it behaves as `max_active_groups` (clamped, not rejected, so lowering `max_active_groups` alone stays valid) |
| `background_slot_fraction` | 0.75 | [0, 1] | outside or NaN → `std::invalid_argument`. Now applies at all times. Trades demand latency against prefetch throughput; lower it (0.25–0.5) for latency-sensitive hosts (§11.8) |
| `write_slot_fraction`, `write_ring_fraction` | 0.5 | [0, 1] | outside or NaN → `std::invalid_argument` |
| `reserved_background_slots` (new) | 8 | any | not validated; effective value `min(value, total / 4, background share)`; 0 disables the floor and the background op sizing |
| `reserved_latency_slots` | 2 | any | not validated; clamped to half of each axis |
| `write_max_wait` | 20 ms | any | not validated |

- Validation: `validate(config const&)` in `src/io/uring/uring_reactor.cpp`, called by the
  `uring_reactor` constructor, i.e. from `uring_ioctx(n, ctx)` before any runner exists.
- Every share admits one op of a class that has nothing in flight (`share_of` ≥ 1), so a
  fraction of 0 throttles a class to about one op instead of starving it.
- REST: `rest::config` has no `scheduling`; its engine keeps a default-constructed
  `scheduling_policy`, so it gets the new defaults (§11.5) without being configurable.
- Benchmark keys: `slices_per_pass=`, `bg_groups=`, `bg_share=`, `bg_reserve=` of
  `cucascade_io_write_benchmark`; trailing `slices_per_pass` argument of both parquet
  benchmarks (`benchmark/README.md`).

### 11.5 Policy and engine semantics (normative)

**`scheduling_policy::pick(view)`**, the lane to pull, first match wins:
1. `write`, if its oldest request waited ≥ `write_max_wait`, with bulk room and write budget;
2. `latency`, if queued and fewer than `max_latency_groups` latency groups are expanding;
3. nothing, if read + write + background expanding groups ≥ `max_active_groups`;
4. `read`, if queued;
5. `background`, if queued, fewer than `min(max_background_groups, max_active_groups)`
   background groups are expanding, and background holds less than its share on every
   bounded axis;
6. `write`, if queued and within budget.

**`may_dispatch(cls, need, view)`** requires all of:
1. `fits` — physical capacity;
2. latency reservation — non-latency ops leave `min(reserved_latency_slots, total / 2)` free
   while latency work is queued or active;
3. background floor (new) — `read` / `write` ops leave `background_reserve(total)` =
   `min(reserved_background_slots, total / 4, share_of(total, background_slot_fraction))` free
   while `background_pressure(view)`: this runner holds background groups with undispatched
   work **and** background is below its share. `latency` and `background` ops are exempt;
4. class share — `write` within `write_slot_fraction` / `write_ring_fraction`; `background`
   within `background_slot_fraction`, now always (at `b2e8eee` only under latency / read
   pressure).

`refused_by_background_floor(cls, need, view)` is true when the op passes 1, 2 and 4 but
not 3.

**uring engine** (`src/io/uring/uring_engine.cpp`):
- **Tiers.** `process_groups` advances the held groups in three tiers: `latency`, then
  `read` and `write` in pull order, then `background`. A demand group pulled after a prefetch
  group still gets freed slots first. A group pulled in a pass is dispatched from the next
  pass on.
- **Per-pass cap.** §11.3.
- **FIFO capacity block.** A non-latency op that does not fit, **or** that
  `refused_by_background_floor` reports, sets `capacity_blocked`: later non-latency groups
  take no slots this pass, so the blocked demand op dispatches as soon as completions free
  `need + reserve`. Background ops with `slots_in_use + need ≤ background_reserve(total)` are
  exempt (`within_reserve`), so under sustained large demand a prefetch still keeps the
  reserve cycling. A write refused by its share or the latency reservation does not block.
- **Background op sizing.** `staging_free_slots(group)`: staged ops of background groups
  (device-destination reads, device-source writes) are planned against
  `min(free slots, background_reserve(slot count))` when that reserve is > 0, so they always
  fit the slots demand may not take (≤ 8 MiB with 64 × 1 MiB slots, instead of up to 16 MiB).
  Host ops hold one slot whatever their size and are unaffected.
- **Policy from config:** `_policy(_cfg.scheduling)` (was default-constructed).
- **Gauges:** `_slot.publish_gauges(_inflight, _bytes_submitted)` once per pass, after
  `pull_work` and before the wait; `_bytes_submitted` counts each physical op once, at first
  submission.
- **`first_io_at` is stamped once** (`set_group_state`; REST `dispatch_one` / `stage_step`
  likewise), so a group requeued by a retiring runner keeps the time of its first op.

Net effect per uring runner with the defaults (64 slots of 1 MiB): background holds ≤ 48
slots and ≤ 2 expanding groups; `read` / `write` never take the last 8 free slots while
background has undispatched work below its share; while a demand op is blocked for room,
background may add ops only up to 8 slots in total. The REST engine applies the same `pick` /
`may_dispatch` (one connection per op on both axes) but has no tiers, per-pass cap or op
sizing.

Caveats:
- `latency` ops are exempt from the floor (≤ 256 KiB, one slot, ≤ `max_latency_groups`
  expanding), so a burst of small reads can briefly take reserved slots — bounded, not a
  starvation path.
- Isolation is per runner: N runners admit up to N × 48 background slots and 2N background
  groups. Sirius isolated globally (K of N reactors). Measured (§11.8): demand latency during
  a prefetch is drive queueing behind the background bytes in flight, so its floor grows
  with the number of runners holding background groups (p99 ≈ 8 ms at 1 runner, ≈ 20–28 ms
  at 2–4).
- A demand read that touches a loading chunk blocks on the whole in-flight prefetch of its
  handle (`prefetching_cache::await_inflight_prefetch`): prefetch throughput is demand
  latency in disguise. The policy slows background under demand, never stalls it (floor plus
  reserve exemption); nothing promotes a prefetch a reader is waiting on.
  `prefetching_handle::demand_wait_ns()` measures that cost.
- A runner whose background groups are at the sub-limit while only background is queued gets
  `wait_bounded` from `prepare_wait`: it waits at most `blocked_retry_interval` (1 ms), and its
  own completions wake it earlier.

### 11.6 Observability

| API | Content | Semantics |
|---|---|---|
| `class_stats::first_io_count`, `first_io_total`, `first_io_histogram` | Per class: retired grouped requests that submitted ≥ 1 op; sum of (first op submitted − enqueued); log2-µs histogram, `first_io_delay_buckets` = 26 buckets via `first_io_delay_bucket(d)` (bucket 0 < 1 µs, bucket b covers [2^(b−1), 2^b) µs, the last is open-ended from 2^24 µs ≈ 16.8 s) | Monotonic |
| `class_stats::first_io_max`, `max_queue_wait` | Peaks | Cleared by `reset_stats_peaks()` (`max_queue_wait` was a lifetime max at `b2e8eee`) |
| `queue_stats::runners` (`runner_stats`) | Per registered runner, in registration order: `id`, `parked`, `active_groups`, `retired_groups`, `inflight_ops`, `max_inflight_ops`, `bytes_submitted` | Op gauges published by the uring engine once per pass (relaxed; up to one pass stale). REST publishes none (zeros). Empty if the snapshot cannot be allocated (`stats()` stays `noexcept`) |
| `ioctx::reset_stats_peaks()` | New virtual; default no-op (kvikio) | Zeroes `max_queue_wait` and `first_io_max`; sets each runner's `max_inflight_ops` to its current `inflight_ops` |
| `prefetching_handle::demand_wait_ns()`, `datasource::demand_wait_ns()` | ns demand reads spent blocked on the handle's in-flight prefetch; concurrent waiters each add theirs | Monotonic per handle; moves transfer it |
| `datasource::cache_chunk_count()` | Chunks named by the datasource's prefetch request (0 without one) | |

**First-I/O delay.**
- Recorded in `request_hub::finish_group`, once per grouped request: `requeue` records
  nothing and keeps `enqueued_at`, and `first_io_at` is stamped once. Backend-agnostic:
  uring and REST both stamp `first_io_at`.
- Measures queue wait plus the time a runner held the group before any of its ops fitted —
  what sirius's queue-delay gauge targeted.
- Requests that never submitted an op (for example cancelled) are not counted.
- **Counted per grouped request:** a read fanned out by `templated_ioctx::read_fanout`
  (`clamp(bytes / 64 MiB, 1, 4)` groups) counts once per group.

**Host sampling guidance.**
1. Poll `stats()` from a host thread (sirius sampled every 250 ms) and diff the monotonic
   counters: Δ`first_io_total` / Δ`first_io_count` is the window's mean delay,
   Δ`first_io_histogram` gives percentiles, Δ`bytes_submitted` / Δt a runner's submit rate.
2. Read the peaks, then call `reset_stats_peaks()` to open the next window. The two calls are
   not atomic: a peak landing between them is lost.
3. Key per-runner diffs by `runner_stats::id`. A runner disappears from `runners` when it
   leaves `run*()`, taking its counters with it; every `run*()` call registers a new id whose
   counters start at 0. Sums over `runners` are not monotonic across runner churn.
4. After a reset, `max_inflight_ops` restarts from the runner's current depth, not from 0.
5. cuCascade logs nothing. Sirius `[uring_gauges]` fields map to: `inflight` →
   `inflight_ops`; `max_inflight` → `max_inflight_ops`; `MiB_s` → Δ`bytes_submitted`;
   `started` ≈ Δ`first_io_count` (counted at retirement); `queued_requests` / `queued_MiB` →
   `per_class[].queued_requests` / `queued_bytes`; `dq_*` → the `latency` and `read`
   first-I/O fields; `pq_*` → `background`. `pending_ops` and `active_slices` have no
   equivalent.

### 11.7 Tests added

| File | Cases |
|---|---|
| `uring/test_uring_scheduling.cpp` (new, `[io][uring][scheduling]`) | T1 byte-exact ragged slices for `slices_per_pass` 1 / 8 / 0. T2 the cap interleaves the groups a runner holds (§11.3). T3 the cap is per group, not a depth bound. T4 background leaves slots and expansion room for demand: only 2 of 4 background groups are pulled, and the demand read completes after 32 background completions (asserted < 128, the bound without the tiered pass order). T5 first-I/O statistics once per request per class, and peak reset. T6 runner gauges. T7 shutdown settles a 65,536-slice uncapped request. T8 config validation (rejected, accepted, `max_active_groups = 1` alone). T9 `[gpu]` background staged device reads fit the reserve (peak 6 ops of 8 MiB; `read`: 4 of 16 MiB) |
| `test_scheduling_policy.cpp` (`[io][policy]`) | "background never exceeds its share" (renamed from "… held back by its share under foreground pressure"); new: background group cap, clamp above `max_active_groups`, read / write floor while background expands, `refused_by_background_floor` reports only floor refusals, floor clamped to a quarter of the axis and to the share |
| `test_request_queue.cpp` (`[io][queue]`) | log2-µs buckets; per-lane recording; recorded once per request at retirement (a never-started request is not counted; peak reset) |
| `test_runner_registry.cpp` (`[io][runner]`) | gauge snapshot and peak reset |
| `cache/test_invalidate_range.cpp` (`[io][cache]`) | prefetch reads are `background`, demand reads `latency` / `read`; `demand_wait_ns` > 0 after a blocked demand read; moves transfer it |

Determinism: the ordering tests queue requests before an external runner starts and read
page-cached files buffered (completion order = dispatch order); no assertion uses wall-clock
thresholds.

### 11.8 Measured results

**Setup.**
- Hardware: Intel Core Ultra 9 285K (24 cores), 2× RTX 6000 Ada (device 0 used), and the
  same disk as §7: Samsung 990 PRO 2 TB → dm-crypt → LVM → ext4. `/mnt/disk_2` was not
  mounted.
- Binaries: baseline `b2e8eee` vs HEAD `68f03e3`. The baseline was built in a separate tree,
  because the benchmarks are linked with `DT_RPATH`.
- Data: TPC-H sf100 `lineitem.parquet` (4 columns, 14,652 ranges, 5,047 MiB).
- Cold runs evict the file with `posix_fadvise(DONTNEED)` before each run. All values are
  medians. Run-to-run spread is about ±1.5 % cold and ±2 % warm.

**P1, regression gates.** 2 runners, 5 interleaved cold runs (3 warm). All pass, within
±1–2 %.

| Path | `b2e8eee` | `68f03e3` | New / base | Gate |
|---|---|---|---|---|
| cudf host, cold (GiB/s) | 3.65 | 3.66 | 1.003 | PASS |
| io_context device, cold | 5.05 | 5.01 | 0.992 | PASS |
| io_context vectored host, cold | 3.57 | 3.58 | 1.003 | PASS |
| cudf device, cold | 5.01 | 5.00 | 0.998 | PASS |
| warm: cudf host / io_context host | 12.65 / 12.83 | 12.76 / 12.90 | 1.009 / 1.005 | PASS |
| `cucascade_parquet_benchmark` uring, end to end, cold (ms) | 1143 | 1130 | 1.011 (speed) | PASS |
| `mode=mixed`: 4 KiB read p99 during writes (ms) | 5.13 | 5.22 | 1.018 (limit 1.10) | PASS |
| `mode=mixed`: concurrent write GB/s | 6.21 | 6.25 | 1.006 | PASS |
| `mode=write` GB/s (supplementary) | 6.15 | 6.04 | 0.981 | PASS |

**P2, `slices_per_pass`.** GiB/s, cold, 5 runs, HEAD only:

| Path | 0 | 1 | 8 |
|---|---|---|---|
| io_context vectored host | 3.57 | 3.59 | 3.60 |
| cudf host | 3.67 | 3.66 | 3.65 |
| io_context device | 5.08 | 5.07 | 5.03 |

- Every path is within 1 %. The device path was rechecked over 10 interleaved runs: 5.050
  (spp 0) vs 5.045 (spp 8).
- Under the tie rule the default stays 8. The knob has no measurable throughput effect on
  these demand-only paths; what it buys is the fairness of §11.3.

**P4, runner sweep.** GiB/s, cold, 5 runs, HEAD only:

| Path | 1 runner | 2 | 4 |
|---|---|---|---|
| cudf host | 2.69 | 3.67 (+36 %) | 3.53 (+31 %) |
| io_context device | **5.26** | 5.01 (−5 %) | 4.92 (−6 %) |

- The D4 rule needs a gain of at least 10 % on both paths. It is not met, so
  `uring_n_reactors` stays 1.
- Caveat: `cucascade_parquet_io_benchmark`'s `n_threads` sets both the submitting threads and
  the runners. The cudf host gain is therefore at least partly submit parallelism of
  per-range `host_read_async`, not runner count.

**P3, demand vs prefetch.** `mode=readmix` with a 2 GiB background read in 1 MiB slices and
one demand read at a time, pooled over 3 reps. `bg_class=background` is how prefetch is
classified now; `bg_class=read` is the pre-port classification.

| Demand, 1 runner | idle p99 | p99, `background` | p99, `read` | `background` / `read` | Background concurrent / standalone |
|---|---|---|---|---|---|
| 4 KiB host | 0.23 ms | 8.1 ms | 10.5 ms | 0.77× (target ≤ 1.1) | 1.02 |
| 1 MiB host | 0.64 ms | 8.1 ms | 328 ms | 0.025× (target ≤ 0.5) | 1.01 |
| 16 MiB host | 2.9 ms | 10.4 ms | 325 ms | 0.032× | 0.80 (target ≥ 0.8) |
| 16 MiB device (staged) | 5.0 ms | 12.0 ms | 330 ms | 0.036× | 0.82 |

- **With `read`,** a demand read of 1 MiB or more waits for the whole background read: p50 is
  about 327 ms, the background read's full duration.
- **With `background`,** the demand read is served at once. Targets (i)–(iii) are met at the
  defaults; (iii) is borderline for 16 MiB demand.
- **At 2 and 4 runners,** (i) and (ii) still hold: 1 MiB p99 is 20.7 vs 304 ms at 2 runners
  and 18.9 vs 284 ms at 4. The latency floor roughly doubles, though, to a p99 of about
  20–28 ms at every size.
- **Why the floor grows:** the share is per runner, and every runner holding a background
  group reaches 48 ops in flight (`runner_max_inflight_ops` reads 48,48 at 2 runners).
- **With `read` at 2 or more runners,** latency is bimodal: p50 is 5–10 ms on a runner without
  background work, and p99 is about 300 ms behind it.

**Sensitivity.** 1 runner, `bg_class=background`, host demand, `bg_reserve=0`:

| `bg_share` | 4K p50 / p99 (ms) | 1M p50 / p99 | 16M p50 / p99 | Max in-flight ops | Background with 16M demand / standalone |
|---|---|---|---|---|---|
| 0.25 | 2.6 / 2.9 | 2.9 / 3.1 | 4.9 / 5.2 | 16–17 | 0.57 |
| 0.50 | 5.2 / 5.5 | 5.4 / 5.6 | 7.4 / 7.7 | 32–33 | 0.72 |
| 0.75 (default) | 7.8 / 9.3 | 8.0 / 8.7 | 10.1 / 10.3 | 48–49 | 0.80 |

- **Latency is linear in the share and is drive queueing.** It equals the background bytes in
  flight divided by drive bandwidth: share × 64 MiB = 16 / 32 / 48 MiB at ≈ 6.4 GB/s gives
  2.6 / 5.2 / 7.8 ms. At 2 runners, 96 MiB predicts ≈ 15 ms; the measured 1 MiB p50 is
  14.7 ms.
- **The engine adds almost nothing.** Demand first-I/O delay during background reads
  (`first_io_mean_us`) is only 70–175 µs, so the remaining latency is NVMe queueing behind
  background bytes already submitted, not engine scheduling.
- **The share trades prefetch bandwidth under heavy demand.** Standalone background
  throughput is the same at every share (≈ 6.3 GB/s; 16 MiB in flight saturates this drive).
  What changes is how much bandwidth background keeps against heavy demand: 0.57× at 0.25
  with 16 MiB demand.
- **`bg_reserve` 0 vs 8: no measurable effect** (the reserve-8 runs are within noise).
  One-at-a-time demand never needs more than one op of the ≥ 16 free slots, so the floor never
  binds; it is a safeguard for slot-saturating demand.

**Decisions.** No default changes in this phase:
- `slices_per_pass` stays 8 (P2 tie rule).
- `uring_n_reactors` stays 1 (P4 rule not met).
- `background_slot_fraction` stays 0.75. The P3 targets are met at the defaults. A share of
  0.25 cuts demand p99 about 3×, but leaves prefetch at 0.57× of standalone under heavy 16 MiB
  demand. That misses target (iii) and slows prefetches a demand reader may be blocked on.

**Tuning guidance.** `background_slot_fraction` trades demand latency against prefetch
throughput. Demand latency during a prefetch is about share × staging bytes × (runners
holding background) ÷ device bandwidth. Latency-sensitive hosts should lower the share
(0.25–0.5); hosts whose readers often block on prefetches (`await_inflight_prefetch`) should
keep 0.75.

Lane masks and dedicated prefetch runners are not needed against demand-behind-prefetch
head-of-line blocking: target (i) is met by about 40×. They remain an option for isolation
across runners (§11.10).

### 11.9 Behaviour changes vs `b2e8eee` (watch these when merging)

| Area | `b2e8eee` | Phase 2 |
|---|---|---|
| Prefetch class (`prefetching_cache::prefetch`) | automatic: `latency` ≤ 256 KiB, else `read` | `background` (uring and REST) |
| Background share | only under latency / read pressure | always |
| Background groups per runner | up to `max_active_groups` (4) | up to `min(max_background_groups, max_active_groups)` (2) |
| Read / write next to background | no floor | leave `background_reserve` (8) free under background pressure |
| uring pass order | latency, then the rest in pull order | latency, read + write, background |
| uring expansion per group per pass | unbounded | `slices_per_pass` (8) |
| uring FIFO capacity block | physical misfit only | misfit or floor refusal; background within the reserve exempt |
| uring background staged op size | `dynamic_io_target`, up to 16 MiB | ≤ `background_reserve` blocks |
| uring policy | default-constructed | `uring::config::scheduling` |
| Config validation | none | `std::invalid_argument` from `uring_ioctx` construction |
| `request_meta::first_io_at` | re-stamped on every `assigned → in_flight` | stamped once (uring and REST) |
| `class_stats::max_queue_wait` | lifetime max | peak since `reset_stats_peaks()` |
| `ioctx` | — | new virtual `reset_stats_peaks()` (vtable change; overriding is optional) |
| `queue_stats` | plain aggregate | holds `std::vector<runner_stats>`; `stats()` allocates |

Phase-2 conflict hotspots: `scheduling_policy.hpp`; `uring_engine.cpp` (`run`,
`process_groups`, `advance_group`, `staging_free_slots`, `plan_next_slice`, `plan_next_write`,
`dispatch_one`, `begin_staged_write`, `set_group_state`); the queue-observability block of
`types.hpp`; `request_queue.hpp`; `runner_registry.hpp`; `prefetching_cache.{hpp,cpp}`
(`prefetch`, `await_inflight_prefetch`, handle moves). New invariant: a `runner_slot`'s
gauges have one writer, its runner thread (relaxed stores). The only cross-thread write is
`reset_peaks`, which stores the current depth into `max_inflight_ops`; `publish_gauges`
raises the peak with a CAS loop so it cannot overwrite a concurrent reset.

### 11.10 Follow-ups

- **Global background budget:** replace the per-runner slot share with one byte-based
  budget for background bytes in flight across all runners. The demand latency floor
  (background bytes in flight ÷ device bandwidth, §11.8) would then not grow with the runner
  count; a simpler interim is dividing the share by the number of runners holding background
  work. This is the global "K of N reactors" bound sirius had.
- **Benchmark:** decouple submitting threads from runners in `parquet_io_benchmark.cpp`
  (`n_threads` sets both today), for a clean runner sweep (P4 caveat).
- **Dedicated prefetch runners (lane masks)** are *not* needed for demand-behind-prefetch
  head-of-line blocking (P3 target (i) met by about 40×). They remain an option for isolating
  prefetch across runners:
  `runner_slot::lane_mask` over `request_class_index`, set by `templated_ioctx::start()` for
  the last K runners when a new `uring::config::prefetch_runners = K` (0 < K < n, else
  ignored); `request_hub::enqueue` wakes a parked runner whose mask contains the class;
  `try_park` / `prepare_wait` / `fill_queue_view` filter by the runner's mask; per-lane
  live-runner counts, so a lane with no capable runner falls back to all runners (dead-runner
  safety); the engine's `pull_work` / `has_room` use the filtered view. Must keep §9.3-4.
- **REST:** read a `scheduling_config` from `rest::config` instead of `{}`; consider the
  tiered dispatch there.
- **Promote-on-wait:** when a demand read blocks on an in-flight prefetch
  (`await_inflight_prefetch`), move the prefetch's untaken slices to a demand lane (needs
  requeue into another lane).
- **Uncached `handle != nullptr` → `background`:** `ioctx::host_read_async` /
  `device_read_async` (`src/io/io_context.cpp`) classify a read that carries a prefetch handle
  as `background` when the context has no cache — semantically a demand read. The cudf
  datasource never reaches it (its uncached path calls `*_read_async_io` directly). Left as
  is.
- **Defaults:** unchanged in this phase (§11.8). A later runner-count change should re-run P4
  with decoupled submit threads, and document the pinned staging of up to 64 MiB per runner
  (not reservation-accounted; `start()` may throw). It should also audit the test fixtures
  (`test_uring_runner.cpp` pins 320 MiB). `n_max_concurrent_scans` stays until a cuCascade
  consumer exists.
- **`AGENTS.md` is stale:** disk I/O backends (see the note at the top) and the test framework
  (`cucascade_io_tests` uses Catch2 v3.8.1, not v2.13.10).
