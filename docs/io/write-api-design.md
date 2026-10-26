# io_context write API: design (W1)

Status: approved by the lead (decisions recorded in section 11). Scope: `include/cucascade/io/{io_context,templated_ioctx,io_request,types}.hpp`,
`src/io/io_context.cpp`, the uring reactor, the kvikIO context (removed since, section 8), tests in
`test/io/`.
The cuFile reactor's writes (W2, teammate `gds`) build on the reactor contract in section 6.

Guiding rule: the read side's shape, reused. One reactor per thread, configuration static at the
`ioctx` level, one async backend hook, prepared slices and the grouped coordinator unchanged in
spirit. Writes are an optional capability: a reactor that does not implement them compiles and
behaves exactly as today.

What is taken from the reverted attempt (5c7428c): the open-mode vocabulary, `not_supported`
defaults on the base `ioctx`, slice validation (null / overlap / overflow). What is not: the runner
model, request classes, `commit_async`, the cache invalidation gate (replaced by a simpler
`fs_cache::invalidate`, see 3.7).

What is borrowed from gpuio (practice only, not abstractions): split each write at the O_DIRECT
granule into an unaligned head, an aligned body and an unaligned tail; the body goes O_DIRECT, the
head/tail through the page cache; the granule is `max(4 KiB, page size)` so a buffered piece and an
O_DIRECT piece never share a page; a write is never clipped to the size at open; a short write is
resubmitted, a zero-byte write is EIO; durability is an explicit fsync.

## 1. Public API (`ioctx`)

New vocabulary in `types.hpp`:

```cpp
/// How open_io_object_for_write opens a local file (all modes open read+write).
enum class write_mode : std::uint8_t {
  create_or_truncate,  ///< O_CREAT | O_TRUNC (default; also fine on an existing empty file)
  create_or_open,      ///< O_CREAT, keep existing content
  open_existing,       ///< fail with ENOENT if missing; keep content
  create_new,          ///< O_CREAT | O_EXCL, fail with EEXIST if present
};

struct write_open_options {
  write_mode mode{write_mode::create_or_truncate};
  ::mode_t permissions{0644};  ///< for newly created files
  /// Best-effort preallocation hint in bytes (fallocate FALLOC_FL_KEEP_SIZE); 0 = none.
  /// Never changes size(); errors (EOPNOTSUPP, ...) are ignored.
  std::uint64_t size_hint{0};
};

/// One segment of a vectored write: file range [offset, offset + size) and its source.
struct write_slice {
  write_slice() noexcept = default;
  write_slice(std::size_t offset, std::size_t size, std::uint8_t const* src) noexcept;
  range rng;
  std::uint8_t const* src{nullptr};
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] std::size_t offset() const noexcept;
};
```

New members on `ioctx` (`io_context.hpp`):

```cpp
// -- Opening ----------------------------------------------------------------
/// Open @p path for writing (and reading). Leading file: stripped, like open_io_object.
/// @throws std::system_error(std::errc::not_supported) when the backend cannot write;
///         std::system_error(errno) for open failures (ENOENT, EEXIST, EISDIR, EACCES, ...).
[[nodiscard]] std::shared_ptr<io_object> open_io_object_for_write(std::string path,
                                                                  write_open_options options = {});

// -- Capabilities -------------------------------------------------------------
[[nodiscard]] virtual bool supports_write() const noexcept { return false; }
[[nodiscard]] virtual bool supports_device_write() const noexcept { return false; }

// -- Writes (front wrappers: validation + cache guard, then the backend hooks) --
std::size_t host_write(const io_object& obj, std::size_t offset, std::size_t size,
                       const std::uint8_t* src);                              // sync, throws
[[nodiscard]] exec::semi_future<std::size_t> host_write_async(
  const io_object& obj, std::size_t offset, std::size_t size, const std::uint8_t* src,
  io_priority priority = io_priority::automatic) noexcept;
[[nodiscard]] exec::semi_future<std::size_t> device_write_async(
  const io_object& obj, std::size_t offset, std::size_t size, const std::uint8_t* src,
  ::cuda::stream_ref stream, io_priority priority = io_priority::automatic) noexcept;
[[nodiscard]] exec::semi_future<std::size_t> host_writev_async(
  const io_object& obj, std::span<const write_slice> slices,
  io_priority priority = io_priority::automatic) noexcept;
[[nodiscard]] exec::semi_future<std::size_t> device_writev_async(
  const io_object& obj, std::span<const write_slice> slices, ::cuda::stream_ref stream,
  io_priority priority = io_priority::automatic) noexcept;
/// Host and device segments in ONE request (one future, one stream for the device part).
[[nodiscard]] exec::semi_future<std::size_t> mixed_writev_async(
  const io_object& obj, std::span<const write_slice> host_slices,
  std::span<const write_slice> device_slices, ::cuda::stream_ref stream,
  io_priority priority = io_priority::automatic) noexcept;

void flush(const io_object& obj);                                             // sync, throws
[[nodiscard]] exec::semi_future<void> flush_async(
  const io_object& obj, io_priority priority = io_priority::automatic) noexcept;

// -- Backend primitives (defaults: not_supported) ------------------------------
virtual std::size_t host_write_io(const io_object& obj, std::size_t offset, std::size_t size,
                                  const std::uint8_t* src);                   // throws
/// The sole asynchronous write hook; see section 4 for the slice invariants.
virtual exec::semi_future<std::size_t> mixed_writev_async_io(
  const io_object& obj, std::vector<prepared_io_slice>&& slices) noexcept;
virtual void flush_io(const io_object& obj);                                 // throws
virtual exec::semi_future<void> flush_async_io(const io_object& obj,
                                               io_priority priority) noexcept;
protected:
virtual std::shared_ptr<io_object> create_io_object_for_write(std::string path,
                                                              write_open_options options);
```

New on `io_object` (`types.hpp`), both with read-only defaults so every existing object is unchanged:

```cpp
/// True for objects returned by ioctx::open_io_object_for_write.
[[nodiscard]] virtual bool writable() const noexcept { return false; }
/// A physical write ending at @p end completed through this object: a writable object grows
/// size() to cover it (atomic max). Called by io_op_request::finish_success (section 4). No-op default.
virtual void note_written(std::size_t /*end*/) const noexcept {}
```

Usage:

```cpp
auto obj = ctx->open_io_object_for_write("/spill/t42.bin", {.mode = write_mode::open_existing});
std::vector<write_slice> cols{{0, n0, d0}, {4096, n1, d1}};          // device pointers
auto written = ctx->device_writev_async(*obj, cols, stream).get();   // total bytes
auto back    = ctx->open_io_object("/spill/t42.bin");                 // sees the data
```

## 2. Front-wrapper behaviour (`io_context.cpp`)

Checks, in order (async forms return an errored future, never throw; sync forms throw the same):

1. Zero-size scalar write: ready future `0` (no other check, like reads). Vector forms drop
   zero-size segments; all empty -> `0`.
2. `src == nullptr` with `size > 0`: `std::invalid_argument`. `offset + size` overflowing `size_t`
   or exceeding `std::numeric_limits<off_t>::max()`: `std::invalid_argument`.
3. `!supports_write()`, or a device segment with `!supports_device_write()`:
   `std::system_error(std::errc::not_supported)`.
4. `!obj.writable()` (object came from `open_io_object`): `std::system_error(std::errc::bad_file_descriptor)`.
5. Cache invalidation (3.7): `_cache->invalidate(obj.object_path())` when a cache is attached.
6. Build `prepared_io_slice`s (section 3) and call `mixed_writev_async_io` / `host_write_io`;
   invalidate again once the write settled, before the returned future resolves (3.7).

Overlap between segments of one call and slice-level invariants are validated once, in the hook
(shared helper `validate_write_slices`, section 4), so a direct caller of the hook gets the same errors.

`flush` / `flush_async`: checks 3 and 4 (`supports_write`, `writable`), then the hooks.

## 3. Semantics

### 3.1 Return value, partial writes, errors
- A write future resolves with the total number of bytes requested once every byte has been
  written. A write never resolves successfully with fewer bytes: short physical writes are
  resubmitted for the remainder; a 0-byte completion is `EIO`.
- First error wins (`grouped_coordinator`): remaining unstarted slices/ops of the request are
  cancelled, already published ops drain, and only then the future fails with the first error
  (`std::system_error` with the errno, a CUDA error, or `operation_canceled`). So once a write
  future has resolved — success or failure — no op of that request touches the source or file.
- No atomicity and no rollback: after a failure some bytes may already be on disk.
- Errors seen through the future: `bad_file_descriptor`, `not_supported`, `invalid_argument`
  (null, overlap, overflow, foreign object), kernel errors (`ENOSPC`,
  `EIO`, `EFBIG`, `EBADF`, ...), CUDA errors of the D2H staging copy, `operation_canceled`
  (reactor not started, shutting down, or a sibling op failed).

### 3.2 EOF and `io_object::size()`
- Writes are never clamped: a write at or past EOF extends the file; a gap becomes a hole (reads
  return zeros).
- Writable objects keep their size in an `std::atomic<std::size_t>`: `size()` = max(size at open,
  end of every physical write that completed through this object). It is raised before the op's
  coordinator credit is settled, so every read issued after a write future resolved (through the
  same object, same or other thread) sees the extended size and clamps to it.
- Not observed: writes through other io_objects or processes, truncation by others (re-open).
  After a failed write `size()` may under-report bytes that did reach the file.
- A new `io_object` opened after the future resolved (any local-file backend, buffered or
  O_DIRECT, uring, cuFile) sees the data: O_DIRECT reads flush dirty page-cache pages of their range first,
  buffered reads read the page cache. No `flush` needed for visibility; `flush` is for durability.

### 3.3 Alignment and O_DIRECT placement (uring; cuFile may refine for its device path)
Let `G = max(IO_BLOCK_SIZE, sysconf(_SC_PAGESIZE))` (4 KiB on x86, 64 KiB on 64K-page aarch64).
Every write slice `[o, o+n)` is split into at most three pieces: head `[o, min(end, align_up(o,G)))`,
body `[align_up(o,G), align_down(o+n,G))`, tail `[max(body_end, align_down(o+n,G)), o+n)` (empty
pieces dropped). Unaligned writes are never an error and never read-modify-written by us:

| piece | host source | device source |
|---|---|---|
| head / tail (sub-granule) | buffered `pwrite` from the caller buffer | D2H into a staging block, buffered write |
| body, source address `≡ offset (mod 4 KiB)` | O_DIRECT straight from the caller buffer | — |
| body, source not congruent (e.g. 8 B-aligned pinned blocks) | memcpy into staging blocks on the reactor, O_DIRECT | D2H into staging blocks, O_DIRECT |

O_DIRECT is used only when `config::use_odirect` is true and the object has a direct fd; otherwise
the same pieces go buffered. A runtime O_DIRECT rejection (`EINVAL`/`EOPNOTSUPP`) falls back to
the buffered fd for that op, as reads do. The kernel performs the sub-page read-modify-write for
buffered pieces under its page lock, so two writers sharing a block are safe.

Consequence for callers: 4 KiB-aligned offsets and sizes (disk tier: aligned column offsets, only
the column tails are buffered) give fully O_DIRECT writes; nothing else is required.

### 3.4 Stream ordering of device sources
The reactor enqueues the D2H copy of a device segment on the caller's stream (`d_buffer.stream`)
when it starts the op (the read path's H2D copy does the same). Therefore the write observes all
work enqueued on that stream before the call. It may also wait behind work enqueued after the call
(before the reactor reached the op). The source must stay valid and unmodified until the future
resolves — in particular do not release it with a stream-ordered free before that. No buffer
registration is required.

### 3.5 Priority
`io_priority` as for reads: `automatic` resolves to `high`; an explicit `low` queues the request
in the low tier (bounded service, preemption at slice boundaries). Flushes take a priority too.

### 3.6 Thread safety and concurrency
- All write APIs may be called concurrently from any thread, on the same or different objects.
- Concurrent writes to disjoint byte ranges of one object are safe and produce the expected bytes,
  also when they share a 4 KiB block or page: O_DIRECT only ever covers granules entirely inside
  one write, partial granules go through the page cache.
- Overlapping writes in flight at the same time, and reads concurrent with writes of the same
  range, give unspecified content. Segments of one vectored call must not overlap
  (`invalid_argument`).
- Known Linux caveat: when unaligned and aligned writes hit the same file, it sees both buffered and
  O_DIRECT I/O; we never let them share a page, but kernels with multi-page page-cache folios can
  in rare races fail to invalidate a folio next to an O_DIRECT range. Callers that must avoid any
  mixing write granule-aligned ranges (all O_DIRECT) or set `use_odirect = false` (all buffered).

### 3.7 fs_cache
Writes bypass the cache but invalidate it. New `fs_cache::invalidate(std::string_view
object_path) noexcept`: every live generation of the path is retired exactly as a superseding
generation retires it today (marked `retired`, removed from the map, idle chunks reclaimed;
handles, in-flight prefetch/demand terminals and cached-copy retirements that still own the
generation keep it, and its pinned chunks, alive until they let go). Later lookups find no
generation and go to the backend; a later prefetch builds a fresh generation.

The write front wrappers call it twice: before dispatch, and when the write settled (success or
failure), before the returned future resolves, so a prefetch that raced the write and cached
pre-write bytes is dropped too. The second call runs from the completing thread through
`install_callback` and must not touch a cache that `shutdown_cache()` already destroyed: it is a
`fs_cache::pending_invalidation` from `fs_cache::invalidate_later(path)`, which holds one of the
cache's in-flight-IO slots, so the cache destructor waits for it exactly as it waits for
in-flight prefetch IO. Sync `host_write` simply calls `invalidate` before and after. An
invalidated generation carries an `invalidated` flag and is never handed out again, even to a
later request for the same key while an old reader keeps it alive (a superseded generation keeps
its old behaviour). The same two passes drop the path's entry in the ioctx's `metadata_store`
(`metadata_store::invalidate_written(path)`): a parsed footer of the old bytes is as stale as the
chunks. `flush` does not invalidate.

Additions from the review (w1-review):
- `open_io_object_for_write` with `create_or_truncate` or `create_new` invalidates the path too:
  it changes (or replaces) the bytes as much as a write.
- A footer parsed from the old bytes can finish after both passes. `metadata_store` therefore keeps
  a per-path epoch (a fixed 256-slot table indexed by path hash, so it does not grow with spill
  paths): `invalidate_written` advances it, `ioctx` stamps every object it opens with it, and
  `register_metadata` drops a registration from an object stamped before the last invalidation.
- Writes and object identity (ported from Sirius #2132) are two fences on the same store. Entries
  carry the `io_object::identity()` of the open that published them and are reused only by an
  open with an equal identity; a newer identity observed for a path (by open generation) retires
  its entry and fences slower publications of the old one. The write epoch covers what identity
  cannot: an in-place write that leaves the stat identity unchanged (coarse ctime). It is checked
  under the store lock before any identity bookkeeping (an object opened before the write can
  neither observe nor publish), it is part of the `get_or_load` single-flight key (a post-write
  open never joins a pre-write parse), and `invalidate_written` drops the entries of every
  metadata profile of the path (ported from Sirius #2231) and erases its observed identity so its
  history restarts. The epoch is advanced before the lock is taken and `invalidate_written` does
  not allocate (a per-path entry count lets a path without entries skip the scan). The targeted
  `invalidate(path, profile)` retires one entry and leaves the epoch alone, so the open that
  found the entry inconsistent can rebuild and republish it. A writable object always has an open-local
  (`opened_handle`) identity, so nothing parsed through it is reused by another open.
- The fs_cache keys its generations by `io_object::identity_cache_key()`: separate opens of an
  unchanged local file share a generation, and a file replaced by another process (which this
  ioctx's write invalidation never sees) gets a new generation that retires the old one.
- Scope (documented on the write API): only the writing ioctx's cache and metadata store are
  invalidated; paths are matched as given (no normalisation); a reader holding a cache handle
  from before the write keeps its view of the old bytes.
- `shutdown_cache()` / `pre_destroy()` wait for every asynchronous write in flight (each holds a
  pending invalidation); calling them from a continuation on one of the ioctx's reactor threads
  can deadlock.

### 3.8 Lifecycle
Async writes need `start()`; before it or after `shutdown()` they resolve with
`operation_canceled` (as reads). `host_write` / `flush` run on the caller thread and work without
reactors running (as `host_read`). On shutdown: queued slices are cancelled, staging copies in
flight are synchronized before their pinned blocks are released, submitted SQEs drain; every future
resolves. The object (and its fds) is kept alive by the request until then.

## 4. Types and request plumbing (`types.hpp`, `io_request.hpp`)

Reuse the read machinery with a kind flag, no parallel hierarchy:

```cpp
enum class io_op_kind : std::uint8_t { read, write, flush };

struct prepared_io_slice {          // existing; one new field + helpers
  ...
  io_op_kind kind{io_op_kind::read};
  [[nodiscard]] bool is_write() const noexcept;
  [[nodiscard]] bool is_flush() const noexcept;
};
// Factories (the only place that const_casts a write source; reactors never write into it):
prepared_io_slice make_host_write_slice(range, std::uint8_t const* src, io_priority);
prepared_io_slice make_device_write_slice(range, std::uint8_t const* src, ::cuda::stream_ref,
                                          int device_id, io_priority);
prepared_io_slice make_flush_slice(io_priority);   // rng {0,0}, no buffers

/// Validate the write slices of one request; returns total bytes.
/// Throws invalid_argument: kind != write, not exactly one source (contiguous host XOR device),
/// fragmented host buffer, null pointer, size 0, range overflow, overlapping ranges.
std::size_t validate_write_slices(std::span<prepared_io_slice const> slices);
```

`io_op_request` (shared by reactors) gains `io_op_kind kind{io_op_kind::read}`;
`finish_success()` calls `obj->note_written(io_rng.end())` first when `kind == write`
(so reactors cannot forget the size update). `device_cpy_request` gains the mirror of
`copy_async`:

```cpp
/// Copy the device source bytes of @p io_rng (a sub-range of req_rng) into @p host_buf on
/// d_buffer.stream (device_copy_batch), then record @p event on that stream.
[[nodiscard]] cudaError_t copy_to_host_async(range io_rng, std::span<iovec const> host_buf,
                                             cudaEvent_t event = nullptr) const noexcept;
```

`grouped_io_request` and `grouped_coordinator` are unchanged; a flush is a request with one flush
slice and a coordinator of `0` bytes / `1` credit.

## 5. `templated_ioctx` plumbing

Optional write concept, detected with `if constexpr` (non-writing reactors — rest, fakes — compile
unchanged and inherit the `not_supported` defaults):

```cpp
template <class R>
concept io_write_reactor_c =
  io_reactor_c<R> && requires(R reactor, typename R::io_object_type const& object,
                              std::size_t offset, std::size_t size, std::uint8_t const* source,
                              std::string path, write_open_options options) {
    { R::create_io_object_for_write(std::move(path), options) }
      -> std::same_as<std::unique_ptr<typename R::io_object_type>>;
    { reactor.host_write(object, offset, size, source) } -> std::same_as<std::size_t>;
    { R::flush(object) } -> std::same_as<void>;
  };

/// Separate from reactor_traits so its existing full specialization (cufile) stays valid.
template <class R>
struct reactor_write_traits {
  static constexpr bool supports_write = io_write_reactor_c<R>;
  /// Optional `static constexpr bool supports_device_write` on the reactor; default true.
  static constexpr bool supports_device_write = /* supports_write && (R::supports_device_write or true) */;
};
```

Overrides in `templated_ioctx` (each body `if constexpr (supports_write)`, else the base default):
- `supports_write()` / `supports_device_write()` from the traits.
- `create_io_object_for_write` -> `Reactor::create_io_object_for_write` (a derived ioctx with
  config-dependent opening, e.g. `cufile_ioctx`, overrides it as it does `create_io_object`).
- `host_write_io`: `as_typed`, writable check, `next_reactor(..., io_op_type::host_write)`, then
  `reactor->host_write(...)` on the caller thread.
- `flush_io`: `Reactor::flush(typed)`.
- `mixed_writev_async_io`: `as_typed`; writable check; drop empty slices; `validate_write_slices`;
  `not_supported` for device slices without device-write support; resolve device ids with
  `cudaGetDevice` (as reads); **no EOF clamping**; cut slices larger than
  `write_split_bytes()` at granule-aligned boundaries into sub-slices (so one big column spreads
  over the selected reactors and pipelines); then the shared dispatch below.
  `write_split_bytes()` is read from the reactor config's optional `write_split_bytes` member the
  same way `range_batch_slices()` is (absent or 0 = no split); uring defaults to 32 MiB, cuFile
  picks its own (GDS prefers large contiguous submissions).
- `flush_async_io`: one request with one flush slice to `next_reactor(...)[0]`; the coordinator's
  `semi_future<size_t>` is mapped to `semi_future<void>` with `defer_value`.

Shared dispatch: the priority stamping / `next_reactor` / coordinator / byte-balanced partition /
`range_batch_slices` split / enqueue / exception settlement block of `mixed_readv_async_io` is
extracted into one private helper used by both reads and writes (read callbacks are null for
writes, so the settlement code is shared as is). The read path must stay behaviourally identical
(same reactor selection, partitioning, batching, callbacks and errors); all existing io tests stay
green. `io_op_type` gains
`host_write`, `write_vector_async`, `flush_async` for `next_reactor` overrides.

## 6. Reactor write contract (uring now, cuFile in W2) — the stable interface

A reactor opts in by modelling `io_write_reactor_c`. Everything below is what `templated_ioctx`
guarantees to deliver and what the reactor must guarantee back.

**Synchronous members** (caller thread, may block, throw on error):
- `static std::unique_ptr<io_object_type> create_io_object_for_write(std::string path, write_open_options)`:
  open read+write per `mode`/`permissions`, apply `size_hint` best-effort, return an object with
  `writable() == true`, size at open from `fstat`.
- `std::size_t host_write(io_object_type const&, offset, size, std::uint8_t const* src)`: write
  all bytes (retry EINTR/short) or throw `std::system_error`; `EBADF` if not writable; call
  `note_written(offset + size)` on success; return `size`.
- `static void flush(io_object_type const&)`: make every write that completed before the call
  durable (`fdatasync`); throw on error.

**Asynchronous input** — through the existing `enqueue(std::unique_ptr<grouped_io_request>)`:
- A request is homogeneous: all slices `kind == write`, or exactly one `kind == flush` slice. Read
  requests are unchanged.
- Write slice: `rng.size > 0`; `rng` may lie past `size()` (never clamped); exactly one source:
  contiguous host (`std::get<std::uint8_t*>(h_buffer.buffer)`, treat as const) XOR device
  (`d_buffer.data`, `d_buffer.stream`, `d_buffer.device_id >= 0`); never fragmented, never both;
  `on_complete == nullptr`; slices of one request never overlap; `request->priority` resolved.
- The source stays valid and unmodified until the request's coordinator resolves.
- Flush slice: `rng == {0,0}`, no buffers; coordinator bytes 0.

**Execution obligations:**
- Credits: one per slice on entry. Before publishing N physical ops for a slice call
  `coordinator->add_tasks(N - 1)`; every op settles exactly one credit through
  `io_op_request::finish_success` / `finish_error` (or `report_error` before an op exists).
- Set `io_op_request::kind = io_op_kind::write` and `io_rng` to the exact file range written by
  the op (no widening); `finish_success` then raises `size()`. Never report success for fewer
  bytes than `io_rng.size`: retry short writes, fail on 0 bytes (`EIO`) or an error.
- Device sources: read the source only in stream order after the work queued on
  `d_buffer.stream` before the call — enqueue the D2H copy (uring) or `cuFileWriteAsync` (cuFile)
  on `d_buffer.stream`, under `rmm::cuda_set_device_raii(device_id)`; or synchronize that stream
  before a synchronous `cuFileWrite`/`pwrite`. Keep any argument block / staging alive until the
  stream has passed it.
- Do not start further ops of a request once `coordinator->should_continue()` is false; cancel
  them with `operation_canceled` (as for reads). Never resolve a request while one of its ops is
  still able to touch the source or the file.
- Flush slice: `fdatasync` (uring: `IORING_OP_FSYNC` + `IORING_FSYNC_DATASYNC`) on the object;
  settle its credit.
- Shutdown / fatal errors: identical to reads (queued and unstarted -> `operation_canceled` or the
  fatal error; in-flight ops drained or synchronized before their buffers are released).
- Optional: `static constexpr bool supports_device_write = false;` to accept only host writes.

## 7. uring implementation

- `local_io_object`: size becomes `mutable std::atomic<std::size_t>` (`size()` loads, `note_written`
  is an atomic max); new constructor overload taking `io_access::{read_only, read_write}` (a scoped
  enum so a string literal can never bind to it); `writable()`; cache id stays the path.
- `create_io_object_for_write`: buffered fd `open(path, O_RDWR | O_CLOEXEC | mode flags, perms)`,
  direct fd `open(path, O_RDWR | O_DIRECT | O_CLOEXEC)` (optional, warn on failure),
  `fallocate(KEEP_SIZE)` for `size_hint`, `fstat`.
- `host_write` (sync) and `flush` as in section 6, using the 3.3 split (body O_DIRECT only when
  address-congruent; no staging on the caller thread).
- Planner: `plan_slice` dispatches on `slice.kind`. Write: split head/body/tail (pure helper
  `detail::split_write_range(range, granule)`, unit-tested). Host congruent body: ops of at most
  1 GiB straight from the caller buffer. Staged pieces (device sources, non-congruent host
  bodies): ops of `dynamic_io_target(...)` bytes (multiples of the 1 MiB staging block, up to
  16 MiB) cut at granule boundaries, `staging_blocks = ceil(bytes / block)`. Head/tail: one op each.
  Flush: one op, buffered fd, no staging.
- Host bounce: a host body whose address is not congruent with its offset (the disk tier's 8 B
  aligned pinned blocks) is memcpy'd into staging blocks on the reactor and written O_DIRECT,
  rather than written buffered from the caller's buffer.
- Worker: new slot state `staging` (D2H in flight). `dispatch_pending` assigns staging blocks, then
  for a staged write starts `copy_to_host_async` on the slot's event (device) or memcpy's the host
  bytes (host bounce) and moves the slot to `staging` / straight to SQE preparation. A new
  `poll_staging()` moves finished copies to SQE preparation (existing `prepared` / `incomplete`
  path). The worker never blocks on a copy event (they sit on callers' streams, which may be
  stalled by unrelated GPU work; review finding MAJOR-1): while staging or H2D copies are
  pending the CQE wait is capped at 100 us, and with nothing in the kernel it waits 100 us for a
  new request (or backs off when only slot-blocked work remains) and polls the events on the next
  pass. The same applies to the read path's H2D copies. Bytes of staged writes whose D2H is
  pending count in `queued_bytes()`, so `next_reactor` sees that backlog. Only the terminal
  path synchronizes staging events, before their blocks are released.
- `prepare_sqe`: `write_fixed` (single registered staging block) / `write` / `writev` for writes,
  `fsync(DATASYNC)` for flush. Reap: writes complete at `bytes_done == io_rng.size`, 0 -> `EIO`,
  short -> resubmit (an O_DIRECT remainder that is no longer aligned falls back to buffered);
  O_DIRECT and fixed-buffer runtime errors fall back exactly as reads do.
- One new config knob, `write_split_bytes` (default 32 MiB, 0 = no split); `use_odirect` governs
  writes too. Expected pipeline for a 1 GiB device column: 32 sub-slices over 2 reactors, each up to 64 MiB of staging in flight (D2H of op k+1
  overlapping the write of op k).

## 8. kvikIO context (removed)

The kvikIO context below was implemented with W1 and removed on 2026-10-19 together with every
other kvikIO use (the cuFile reactor now loads libcufile itself, `io::cufile::cufile_api`). The
section is kept as a record of what it did.


`kvikio::FileHandle` already writes host and device pointers:
- `create_io_object_for_write`: `open(2)` with the mode flags (create/trunc/excl, perms), a
  best-effort `fallocate(KEEP_SIZE)` for `size_hint`, then `FileHandle{path, "r+", m644,
  compat_mode}` while the first fd is still open; the handle must name the same inode, and a file
  this call created is unlinked when the handle cannot be built. `s3://` -> `not_supported`.
  `kvikio_io_object` gets the atomic size and `writable`.
- `host_write_io`: `pwrite(src, size, offset).get()`, short -> `EIO`, `note_written`.
- `mixed_writev_async_io`: eager like its reads (`make_semi_future_with`); device slices
  `cudaStreamSynchronize(stream)` first, then `pwrite(device_ptr, ...)` (kvikIO picks GDS/bounce).
- `flush_io` / `flush_async_io`: `fdatasync(handle.fd())` (eager).
- `supports_write() == supports_device_write() == true`. Remote objects are not writable.

`rest`: S3 writes since the io_uring REST transport (multipart uploads; bytes visible at `flush`,
the commit): docs/io/rest-uring-design.md section 5. `s3rdma`: no change (defaults:
`not_supported`).

## 9. Tests (Catch2, `cucascade_io_tests`)

New `test/io/uring/test_uring_write.cpp` (`[io][uring][write]`), scratch files under
`std::filesystem::temp_directory_path()` (ext4 here, so O_DIRECT is exercised), each checked
against an in-memory shadow of the file and read back through `open_io_object` (new object) and
through the writable object:
1. Open modes: truncate an existing file, `create_or_open` keeps bytes, `open_existing` on a
   missing file throws ENOENT, `create_new` on an existing file throws EEXIST, `file:` stripped,
   opening a directory fails, `size_hint` keeps `size() == 0`.
2. `host_write` round trip: aligned/unaligned offsets (0, 1, 4095, 4096, 4097), sizes (1, 4095,
   4096, 4097, 3 MiB + 7), aligned and misaligned buffers.
3. `host_write_async`, `device_write_async`, `flush`, `flush_async` round trips.
4. Stream ordering: `cudaMemsetAsync` the device source on a stream, then `device_write_async`
   without synchronizing: file holds the memset bytes.
5. Vector: `host_writev_async` / `device_writev_async` / `mixed_writev_async` with scattered,
   out-of-order, block-sharing segments; the disk-tier layout (4 KiB-aligned starts, unaligned
   sizes, holes, 8 B-aligned pinned host sources, a 64 MiB device segment); zero-size segments
   dropped; all-empty -> 0.
6. Past EOF: hole reads zeros, `size()` grows, reads through the same object see the new size.
7. Zero length: returns 0, file untouched; null source with size 0 accepted, size > 0 rejected.
8. Errors: read-only object -> `bad_file_descriptor` (sync throws, async fails); overlapping
   segments -> `invalid_argument`; foreign object -> `invalid_argument`; `/dev/full` -> `ENOSPC`
   propagated through the coordinator; a writable `local_io_object` with an invalid fd -> `EBADF`.
9. Concurrency: 8 threads x 64 writes of disjoint, block-sharing ranges (sync + async, host +
   device) to one object; final content checked.
10. Shutdown with in-flight writes: many 16 MiB device/host writes, `shutdown()` immediately:
    every future resolves (success or `operation_canceled`), no hang; before `start()` ->
    `operation_canceled`.
11. Low-priority writes complete; `split_write_range` unit cases (granule 4 KiB and 64 KiB).

`test/io/test_templated_ioctx.cpp` (fake reactors): non-writing reactor -> `supports_write()` false
and every write API `not_supported`; a fake writing reactor receives `kind == write` slices,
unclamped past EOF, device ids resolved, large slices split, partitions balanced, flush routed to
one reactor as a single flush slice; `io_op_request::finish_success` raises `size()` for writes only.

`test/io/kvikio/test_kvikio_write.cpp` (cudf build, as the other kvikIO tests): host/device round
trip, past EOF, flush, read-only rejection. Removed with kvikIO.

`test/cudf/test_write_cache_invalidation.cpp` (generations are only built through the cudf
datasource's fadvise, so the controlled_reactor fixtures of `test/io/cache` cannot reach them): a
write (sync and async) retires the path's generations and its metadata entry while an open reader
keeps its retired generation valid; a generation built between the two passes is dropped by the
pending invalidation; an invalidated generation is never handed out again.

All added to `test/CMakeLists.txt` next to their read counterparts.

## 10. Files

Changed: `include/cucascade/io/{types,io_request,io_context,templated_ioctx}.hpp`,
`src/io/io_context.cpp`, `include/cucascade/io/uring/{uring_reactor,types}.hpp`,
`src/io/uring/uring_reactor.cpp`, `include/cucascade/io/kvikio/kvikio_context.hpp` and
`src/io/kvikio/kvikio_context.cpp` (both removed since), tests above, `test/CMakeLists.txt`.
`include/cucascade/io/cache/{fs_cache,metadata_store}.hpp`, `src/io/cache/{fs_cache,metadata_store}.cpp`
(`invalidate`, `invalidate_later`), `test/cudf/test_write_cache_invalidation.cpp`.
Not touched: read semantics, rest, s3rdma, cufile (W2), datasource/cudf layer.

## 11. Measured (this machine, /tmp = root NVMe ext4, 4 GiB per run, 4 reactors)

Hidden test `cucascade_io_tests "[write_bench]"` (4 KiB-aligned 256 MiB segments in one vectored
call): device -> file 5.1 GB/s, pinned host -> file 6.2 GB/s, pinned host + 8 B (host bounce)
5.2 GB/s; `dd oflag=direct bs=16M` reaches 5.0 GB/s on the same file system. A test checks with
`mincore` that aligned writes leave no page in the page cache.

## 12. Decisions (lead review)

- Q1. `flush_async` returns `exec::semi_future<void>`; every write API returns
  `semi_future<size_t>`.
- Q2. `fs_cache::invalidate(object_path)` called before dispatch and again before the write future
  resolves (3.7); no busy guard.
- Q3. The device staging copy is enqueued on the caller's stream when the reactor starts the op
  (3.4, as reads); the doxygen states that the source must stay valid and unmodified until the
  future resolves.
- The write-slice cut is the reactor-config property `write_split_bytes` (uring 32 MiB), not a
  constant in `templated_ioctx`.
- The shared dispatch helper keeps the read path behaviourally identical.
