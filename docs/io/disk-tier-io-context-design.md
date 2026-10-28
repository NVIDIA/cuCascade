# Disk tier on io_context: design

Status: implemented (decisions recorded in section 14). Builds on the io_context write API
(`docs/io/write-api-design.md`) and the cuFile write path (`docs/io/gpuio-gds-practices.md`, section 8). Scope: the DISK memory tier —
`memory_space` (DISK), the GPU/HOST <-> DISK converters, their tests and `benchmark_disk_converter`.

Goal: every disk-tier read and write goes through a `cucascade::io::ioctx` — cuFile/GDS when GDS is
really available, io_uring otherwise. The bespoke `idisk_io_backend` stack is removed. The on-disk
format, the converter registry API (`registry.convert<disk_data_representation>(...)`) and the RAII
lifecycle of spill files stay. Priority: ease of use — a disk space works with no I/O configuration
at all, and any started, write-capable `ioctx` can be plugged in.

## 1. Current state (what is being replaced)

| Piece | Today |
|---|---|
| Interface | `idisk_io_backend` (`include/cucascade/data/disk_io_backend.hpp`): synchronous `write`/`read` (device and host overloads) by *path*, `write_batch`/`read_batch` of `io_batch_entry{ptr,size,file_offset}` |
| Registry | `io_backend_registry` (named factories, default `"pipeline"`); not used by the library itself |
| Only backend | `pipeline_io_backend` (`src/data/pipeline_io_backend.cpp` + `io_worker.hpp`): 2 x 64 MiB pinned buffers, one worker thread, D2H/H2D on a private stream overlapped with `pwrite`/`pread`. Built with `direct_io=false`, i.e. buffered. A mutex serialises **all** device I/O of a space; host I/O opens and closes the file for **every** block segment |
| Owner | `memory_space` (DISK) holds `std::shared_ptr<idisk_io_backend>`; default `make_pipeline_io_backend()`; `get_io_backend()` |
| GPU -> DISK | plan 8 B layout, recompute 4 KiB file offsets, collect all column buffers, one `write_batch` |
| DISK -> GPU | recursive `reconstruct_column_from_disk`; every buffer is allocated and read **synchronously, one at a time** |
| HOST <-> DISK | one `write`/`read` call per (column buffer x host block) segment |

File format (unchanged by this design): no header and no metadata on disk (column metadata stays in
memory in `disk_table_allocation::columns`); every null mask / data buffer starts at a
`DISK_FILE_ALIGNMENT` (4 KiB) aligned offset; sizes are not padded (the gap is a hole); file length =
end of the last buffer. Files are ephemeral: created by `generate_disk_file_path` (`mkstemps`, 0600),
deleted by `~disk_data_representation`, partial files deleted by `disk_file_guard`.

Baseline on this host (old pipeline, `/tmp` on a Samsung 990 PRO NVMe, 4 numeric columns, 512 MiB):
GPU->DISK 2.36 GiB/s (incl. `fdatasync`), HOST->DISK 2.45, DISK->GPU cold 1.92 / warm (page cache)
7.1, DISK->HOST warm 8.3, cold string read 0.3-0.9 GiB/s.

## 2. Layering

```
cucascade (core)    memory_space, disk_data_representation, disk_table_allocation, disk config
   ^                holds std::shared_ptr<io::ioctx> -- forward-declared, never dereferenced
cucascade_io        ioctx, uring / cufile contexts, write API
   ^
cucascade_cudf      disk converters, disk I/O helpers, disk io-context factory + defaults
```

- Core cannot link io (io links core). It does not need to: `std::shared_ptr<T>` can be stored,
  copied and destroyed while `T` is incomplete (the deleter is captured where the pointer is created —
  io, cudf or user code). Core only gains `namespace cucascade::io { class ioctx; }`;
  `libcucascade` references no io symbol.
- All I/O happens in the converters, which already live in the cudf layer, which already links io
  when `CUCASCADE_BUILD_IO=ON`.
- CMake: the disk converters now need io, so `CUCASCADE_BUILD_CUDF=ON` requires
  `CUCASCADE_BUILD_IO=ON` (configure-time `FATAL_ERROR` with a clear message; both default ON, and
  topology-only builds disable both). The existing `if(CUCASCADE_BUILD_IO)` link blocks for
  cucascade_cudf stay and become always-true in a cudf build.

Rejected alternatives:
- Keep `idisk_io_backend` in core and implement it over an ioctx in the io lib: keeps a synchronous,
  by-path, one-buffer-at-a-time interface that hides what the ioctx adds (vectors, futures, reactors)
  and leaves two I/O abstractions.
- Own the context in the converter registry (`register_builtin_converters(registry, ctx)`): loses the
  per-space choice (deliberately moved into `memory_space` earlier) and ties lifetimes to the registry.
- A process-global static context: reactor threads and pinned staging would be torn down during
  static destruction, after the CUDA runtime may be gone.
- Compiling the disk converters out of a cudf build without io: silently loses the DISK tier.
- Two contexts per space (one for host paths, one for device paths): more knobs for a niche gain
  (section 6); one context per space is easier to use.

## 3. Ownership, configuration, lifetime

### 3.1 Core API (`include/cucascade/memory/config.hpp`, `memory_space.hpp`)

```cpp
namespace cucascade::io { class ioctx; }
namespace cucascade::memory {

struct disk_memory_space_config {
  int disk_id{-1};
  std::size_t memory_capacity{0};
  std::string mount_paths;
  /// Context that serves this space's files (uring, cufile, ...); must be started and
  /// support writes. Null: the disk converters adopt the process-shared default on first use.
  std::shared_ptr<io::ioctx> io_context{};
};

class memory_space {
  ...
  /// DISK only (std::logic_error otherwise). Null until configured or adopted.
  [[nodiscard]] std::shared_ptr<io::ioctx> get_io_context() const;
  /// DISK only. Replace the context; conversions in flight keep the one they started with.
  /// @throws std::invalid_argument if @p context is null.
  void set_io_context(std::shared_ptr<io::ioctx> context);
  /// DISK only. The context, or install make_default() first when none is set. Thread-safe;
  /// make_default runs under the space's lock; a null result throws and installs nothing.
  std::shared_ptr<io::ioctx> get_or_init_io_context(
    std::function<std::shared_ptr<io::ioctx>()> const& make_default) const;
};
}
```

Aggregate initialisation `disk_memory_space_config{0, cap, "/tmp"}` keeps compiling. State:
`mutable std::shared_ptr<io::ioctx> _io_context` + `mutable std::mutex _io_context_mutex`
(logically-const lazy init: converters receive `const memory_space*` targets).
`reservation_manager_configurator::set_disk_mounting_point(uuid, capacity, path,
std::shared_ptr<io::ioctx> io_context = nullptr)` forwards it into the config.
`set_io_context` exists because a context built by `io::io_context_registry` needs the
`memory_reservation_manager` that itself creates the disk spaces.

### 3.2 cudf layer (`include/cucascade/cudf/disk_io_context.hpp`, new public header)

```cpp
enum class disk_io_backend : std::uint8_t { automatic, uring, cufile };

struct disk_io_context_options {
  /// automatic: cufile when io::cufile::gds_available(), else uring when io_uring is allowed in
  /// this process, else cufile (its POSIX path).
  disk_io_backend backend{disk_io_backend::automatic};
  /// The io stack's own static configuration: uring_n_reactors / uring, cufile_n_reactors /
  /// cufile. backend and cache are ignored (the disk tier never arms an fs_cache).
  io::io_config io{};
  /// NUMA node of the uring context's private pinned staging (-1: unbound, portable).
  int staging_numa_node{-1};
};

/// Build AND start a context for DISK spaces. It owns everything it needs: for uring a private
/// pinned fixed_size_host_memory_resource (uring_n_reactors x 64 MiB).
[[nodiscard]] std::shared_ptr<io::ioctx> make_disk_io_context(disk_io_context_options const& = {});

/// The process-shared default adopted by DISK spaces configured without a context: the shared
/// cuFile context when io::cufile::gds_available(), otherwise the shared uring context, both built
/// with the io_config defaults on first use.
[[nodiscard]] std::shared_ptr<io::ioctx> default_disk_io_context();
```

- The default serves BOTH host and device paths. On a GDS host HOST <-> DISK therefore goes
  through the cufile reactor's host path, which runs at uring speed (docs/io/gpuio-gds-practices.md,
  section 8).
  Without GDS, automatic picks uring when this process may use io_uring and the cuFile reactor's
  POSIX path otherwise (forced uring fails fast with the errno instead). kvikIO is never
  selected: its context serves reads only (writes report `io::not_implemented_error`). The cuFile
  reactor loads libcufile itself (`io::cufile::cufile_api`) and is built when `cufile.h` was found
  (`CUCASCADE_HAS_CUFILE`); a build without it has no cufile choice, and automatic then fails when
  io_uring is unavailable. The kvikIO read backend and its dependency stay as they are.
- uring staging ownership uses the hook the reactor already provides: a private subclass of
  `uring_reactor::reactor_context` owns the upstream pinned resource and the
  `fixed_size_host_memory_resource` (base-from-member); every reactor's shared `_ctx` keeps it
  alive and each reactor's `_bounce_storage` is released before it. No custom deleter, so the
  context is a plain `make_shared` (its `enable_shared_from_this` stays valid).
- GDS probe:
  `cucascade::io::cufile::gds_available()` in `include/cucascade/io/cufile/platform.hpp` —
  process-wide, computed once, never throws; true iff libcufile loads, the driver opened, it reports
  a real GDS mode (`nvfs.dstatusflags != 0`) and `CUFILE_FORCE_COMPAT_MODE` does not force compat
  (`KVIKIO_COMPAT_MODE` is no longer read). The disk tier asks it through one private
  helper (`detail::gds_is_available()`); the first call opens the cuFile driver (~0.3 s), which only
  happens when an automatic context is built. False on this host (no
  nvidia-fs), so the tests drive both backends through `disk_io_context_options::backend`. A file on a
  filesystem cuFile cannot register (btrfs, tmpfs) still takes the cufile reactor's POSIX path per
  file — correct, just slower; `automatic` does not try to detect that.
- A context built by `io::io_context_registry::make_ioctx(...)` (staging from the HOST memory space)
  works too, through the config or `set_io_context`; the HOST space must then outlive it.
- Applications may share their scan context with the disk tier. The disk tier only uses the
  cache-unaware read primitives (`*_readv_async_io`), so spill files never enter or consult an
  fs_cache; the write API's `fs_cache::invalidate(path)` finds nothing for spill paths.

### 3.3 Lifetime, threads, start / shutdown

- Configuration is static at the context level (reactor count, O_DIRECT, thresholds); the disk tier
  passes no per-request knobs.
- **One reactor set per process by default, not per disk space.** The default is process-shared: a
  function-local `std::weak_ptr` cache (mutex-guarded) hands every space without a context the same
  `ioctx`; the spaces own it, so it is destroyed with the last space that adopted it and rebuilt on a
  later use (e.g. the next test's reservation manager). N disk mounts do not create N reactor pools;
  users who want isolation pass their own contexts.
- **Lazy start.** The default is built — reactor threads spawned, pinned staging allocated — on the
  first conversion that needs it, not when the space is constructed: a process that never spills
  pays nothing. `make_disk_io_context` calls `start()` before returning. User-supplied contexts must
  already be started (documented precondition; an unstarted context fails requests with
  `operation_canceled`), because `templated_ioctx::start()` is not safe to race.
- **Shutdown** is RAII: `~templated_ioctx` runs `pre_destroy()` + `shutdown()` (queued work
  cancelled, in-flight ops drained, threads joined) when the last `shared_ptr` goes — normally when
  the `memory_reservation_manager` destroys its disk spaces; a conversion in flight holds a copy, so
  the context outlives it. Destruction never happens on a reactor thread (no callbacks run there).
  `memory_space::shutdown()` keeps its current meaning and does not touch the context (it may be
  shared).
- Footprint of the default: uring — `uring_n_reactors` (4) threads + 4 x 64 MiB pinned staging (vs 1
  thread + 128 MiB per space today); cuFile — `cufile_n_reactors` (4) threads, no staging.

## 4. Reader / writer path (cudf layer, private `src/cudf/disk_io.{hpp,cpp}`)

```cpp
/// The space's context, adopting default_disk_io_context() if none is set.
std::shared_ptr<io::ioctx> resolve_disk_io_context(memory::memory_space const& space);

/// One file, one request: open_io_object_for_write(path, {open_existing, size_hint = file size}),
/// one device_writev_async / host_writev_async, wait, check the byte count.
void write_disk_file_device(io::ioctx&, std::string const& path,
                            std::span<io::write_slice const>, ::cuda::stream_ref, std::size_t size);
void write_disk_file_host(io::ioctx&, std::string const& path, std::span<io::write_slice const>,
                          std::size_t size);
/// open_io_object, one device_readv_async_io / host_readv_async_io, wait, check the byte count.
void read_disk_file_device(io::ioctx&, std::string const& path, std::span<io::slice const>,
                           ::cuda::stream_ref);
void read_disk_file_host(io::ioctx&, std::string const& path, std::span<io::slice const>);
```

- One `io_object` per conversion (one open/close), one vectored request per table, one future awaited
  by the converting thread — `convert()` stays synchronous for its callers.
- `write_mode::open_existing`: the file was just created by `generate_disk_file_path`; never
  re-create a path someone else removed. `size_hint` preallocates extents (best effort).
- No `flush`: spill files need no durability; write-api-design.md 3.2 guarantees a new (O_DIRECT or buffered)
  `io_object` of any backend sees the data once the write future resolved.
- Reads use the cache-unaware primitives and verify `bytes == sum(slice sizes)` (the ioctx clamps to
  EOF, so a truncated file shows up as a short count -> `std::runtime_error` naming the path).
- Capability checks live in the helpers and throw `std::logic_error`: the write helpers require
  `supports_write()` (host) / `supports_device_write()` (device), the device read helper
  `supports_device_read()` (e.g. a REST context fails there, not in `resolve_disk_io_context`).
- Priority `automatic` (= high) both ways. Later option: spill writes `low` when sharing with scans.
- The helpers build every slice **before** submitting and submit exactly once, so nothing can throw
  between submit and wait; the wait itself is the last statement that touches the request (see 10).

## 5. Converter changes (`src/cudf/representation_converter_builtins.cpp`)

- **GPU -> DISK**: planning unchanged (`wait_for_writer(stream)`, `plan_column_copy`,
  `recompute_file_offsets`); `collect_gpu_column_io_entries` produces `io::write_slice`s; one
  `device_writev_async` under `rmm::cuda_set_device_raii` for the source GPU (the write API binds device slices to
  the current device), on the caller's `stream`, so it observes the table's writer (write-api-design.md 3.4). The
  source representation keeps the table alive until the future resolved.
- **HOST -> DISK**: `write_host_buffer_to_disk` appends one `write_slice` per (buffer x host block)
  segment instead of writing; one `host_writev_async` per table.
- **DISK -> HOST**: `read_disk_buffer_to_host` appends `io::slice`s into the freshly allocated
  blocks; one `host_readv_async_io` per table.
- **DISK -> GPU**: **one batched `device_readv_async_io` per table** instead of one synchronous read
  per buffer. `reconstruct_column_from_disk` splits into three phases:
  1. `allocate_disk_column(meta, stream, mr, slices)` — recursively allocate every null mask
     (`cudf::create_null_mask(num_rows, UNINITIALIZED)`, read `min(size, meta.null_mask_size)`
     bytes) and data buffer (`rmm::device_buffer` on `stream`) into a buffer tree, appending one
     `io::slice` per non-empty buffer;
  2. one `device_readv_async_io(obj, slices, stream)` and wait — the backend writes the
     destinations only after the work enqueued on `stream` (uring copies on `stream`; cuFile waits
     for `stream`, then reads synchronously or copies on its own stream), so after the
     stream-ordered allocations;
  3. `assemble_disk_column(meta, buffers, stream, mr)` — the existing type dispatch (STRING with the
     INT32 -> INT64 offsets cast, LIST keeping INT32 offsets, STRUCT, DICTIONARY32, fixed width /
     fixed point), moving the buffers in.
  Then `make_written_on(table, target, stream)` as today.
- **The trailing `stream.sync()` of DISK -> GPU**: decided by the ioctx read contract, not by a
  test or by one backend's mechanism. (1) The read future resolves only after every byte is in
  device memory, on whichever stream the backend used: uring settles a device op once the event it
  recorded behind its H2D copies on `stream` completes; cuFile settles after its synchronous
  `cuFileRead`, or after the copy events of its POSIX + H2D path on its own copy stream. (2) A
  backend writes the destinations only after the work enqueued on `stream` before the read (uring
  copies on `stream`; cuFile gates on `stream` first); the destinations are stream-ordered
  allocations on `stream`, so this ordering is required, not incidental. The casts of phase 3 are
  enqueued on `stream` afterwards and `make_written_on` records the writer event behind them, which
  every reader waits on. The host-side `stream.sync()` therefore adds nothing and is dropped; the
  code comment states the same contract.

## 6. Backend selection

| Context | Device sources/destinations | Host | Used |
|---|---|---|---|
| uring | staged through the reactor's pinned blocks, D2H/H2D on the caller's stream; aligned bodies O_DIRECT, sub-4 KiB heads/tails buffered (write-api-design.md 3.3) | direct; O_DIRECT when address-congruent, else staged memcpy + O_DIRECT | default without GDS |
| cufile | `cuFileWrite` / `cuFileRead` above `gds_threshold` with real GDS, staged POSIX otherwise | O_DIRECT reads; writes buffered with early writeback unless large and congruent | default when `gds_available()` |

How a user configures it:
1. Nothing: the space adopts `default_disk_io_context()` — cuFile on GDS hosts, uring elsewhere.
2. Per space: `config.io_context = make_disk_io_context({.backend = disk_io_backend::cufile})` (forces
   cuFile even in compatibility mode, e.g. for testing) or `uring`; or
   `space.set_io_context(ctx)` after construction; or any started, write-capable `ioctx` (e.g. one
   shared with the scan path).
3. Process-wide knobs inside the contexts stay where they are (`io::io_config` fields passed through
   `disk_io_context_options::io`, `cufile.json`).

Rationale for "automatic = cuFile only with real GDS": cuFile in compatibility mode bounces through
its own per-call buffers, while uring's pinned staging is the
measured fast path; on this host automatic therefore resolves to uring, and cuFile is exercised
explicitly in tests and the benchmark.

On GDS hosts the automatic default (cuFile) also serves HOST <-> DISK. That path used to be the
cufile reactor's buffered POSIX fallback (one synchronous `pread`/`pwrite` per op); measured against
uring, it was reworked (`perf(cufile): serve the DISK tier's host transfers at uring speed`):
host-only reads go O_DIRECT (through the pinned scratch when unaligned), and host writes that do not
line up with their file offset go through the page cache with writeback started at once
(`sync_file_range`). Measured with
a forced cuFile context (the same code path as on a GDS host), `benchmark_disk_converter`, 4 columns,
single-benchmark processes alternated, GiB/s:

| Path | cuFile before the rework | cuFile after | uring |
|---|---|---|---|
| HOST -> DISK, 512 MiB | 2.57-2.88 | 4.13-4.22 | 4.26-4.40 |
| HOST -> DISK, 64 MiB | 2.20-2.47 | 3.00-3.05 | 3.31-3.39 |
| DISK -> HOST cold, 512 MiB | 1.47-1.82 | 6.35-6.39 | 6.22-6.29 |
| DISK -> HOST cold, 64 MiB | 1.49-1.73 | 5.52-5.67 | 5.22-5.59 |

An earlier figure for cuFile HOST -> DISK (1.40-1.46) was taken with the drive past its SLC cache;
on a recovered drive the pre-fix figure is 2.68-2.81. After the fix the cuFile host path is
0.88-0.96x uring for writes and at least 1.05x for cold reads,
so a single cuFile context per space no longer costs the HOST <-> DISK chain. GPU <-> DISK is
unchanged by the fix. Full table: `docs/io/gpuio-gds-practices.md` section 8.

`cufile.log`: resolving `automatic` opens the cuFile driver once per process (to ask
`gds_available()`), and cuFile writes `cufile.log` into the working directory. There is no clean way
to redirect it from the library: libcufile accepts the log-dir parameter setters but
ignores them, and the only working controls are `CUFILE_LOGFILE_PATH` set before the first cuFile
call or a `cufile.json` — both belong to the user, and a library `setenv` would be process-global
and racy. Resolution: documented on `make_disk_io_context()`; users
set `CUFILE_LOGFILE_PATH` or `cufile.json` to move it.

## 7. Removal — every affected public symbol and test

No deprecation period (decided): the user asked for the reader/writer API change; swappability is
kept because users can pass any `ioctx`.

Removed public headers and symbols (installed under `include/cucascade/data/`):
- `disk_io_backend.hpp`: `cucascade::io_batch_entry`; `cucascade::idisk_io_backend` (`write`/`read`
  device and host overloads, `write_batch`, `read_batch`).
- `io_backend_registry.hpp`: `cucascade::io_backend_factory_fn`; `cucascade::io_backend_registry`
  (`register_backend`, `has_backend`, `create_backend`, `unregister_backend`, `clear`,
  `set_default`, `get_default_name`, `create_default_backend`);
  `cucascade::register_builtin_io_backends`.

Changed public API:
- `memory_space.hpp`: forward declaration `cucascade::idisk_io_backend` removed; constructor
  `memory_space(const disk_memory_space_config&, std::shared_ptr<idisk_io_backend>)` removed (use
  `config.io_context`); `idisk_io_backend& get_io_backend() const` removed (use `get_io_context`);
  `get_io_context` / `set_io_context` / `get_or_init_io_context` added.
- `config.hpp`: `disk_memory_space_config::io_context` added (source compatible).
- `reservation_manager_configurator::set_disk_mounting_point`: one defaulted `io_context` parameter
  (source compatible, ABI change).
- `cudf/builtin_converters.hpp`: documentation only. New header `cudf/disk_io_context.hpp`.

Removed internals (exported from `libcucascade.so` but not in public headers):
`cucascade::make_pipeline_io_backend` (`src/data/io_backend_internal.hpp`, also forward-declared in
`src/memory/memory_space.cpp`), `cucascade::detail::io_worker` (`src/data/io_worker.hpp`),
`src/data/pipeline_io_backend.cpp`, `src/data/io_backend_registry.cpp`.

Kept: `DISK_FILE_ALIGNMENT` (plus `static_assert(DISK_FILE_ALIGNMENT % io::IO_BLOCK_SIZE == 0)`),
`disk_table_allocation`, `generate_disk_file_path`, `disk_data_representation`, `disk_access_limiter`.

Tests:
- `test/data/test_disk_io_backend.cpp` (cucascade_tests, 12 cases) -> `test_disk_memory_space.cpp`:
  removed "io_backend_registry creates pipeline backend", "pipeline backend host write and read
  round-trip", "io_backend_registry default name is pipeline", "... set_default changes default",
  "... set_default throws for unregistered name", "... create_default_backend creates pipeline",
  "... clear resets default name"; rewritten for io contexts: "disk memory_space has default pipeline
  backend", "... accepts custom backend", "... rejects null backend"; kept: "generate_disk_file_path
  produces unique paths", "rmm::align_up rounds to 4KB boundaries".
- `test/data/test_io_worker.cpp` (cucascade_tests, 5 cases, all `[io_worker]`): removed with
  `io_worker`.
- `test/CMakeLists.txt`: the two entries above.
- Unchanged API, now running on contexts: `test_gpu_disk_converters.cpp`,
  `test_disk_host_converters.cpp`, `test/cudf/test_release_table_stream.cpp` (disk sections),
  `test_bandwidth_profiler.cpp` (disk pairs).
- `benchmark/benchmark_disk_converter.cpp`: the `*Pipeline` benchmarks renamed (section 12).
- Docs: `docs/bandwidth-profiler.md` updated.

## 8. File format, alignment, page cache

- Format unchanged (files never outlive the process, so there is no compatibility surface).
- The uring backend writes every column buffer's 4 KiB-aligned body O_DIRECT. Host sources go
  directly when address-congruent, otherwise through staging blocks. Only sub-page heads and
  tails go through the page cache. HOST -> DISK also splits buffers at the 1 MiB host-block
  boundaries, giving a sub-page buffered piece on each side of a boundary.
- The cuFile reactor does the same for device sources: a `cuFileWrite` body with real GDS, and
  staged POSIX (O_DIRECT for granule-aligned pieces) in compatibility mode, where the whole
  device slice is staged. A **host** source slice whose aligned body is below 4 MiB or not
  address-congruent is written whole through the page cache, with writeback started at once
  (`min_direct_host_write`, `include/cucascade/io/cufile/config.hpp` note). This includes every
  HOST -> DISK slice cut from the 1 MiB pinned blocks. On the cuFile backend, HOST -> DISK spills
  therefore transiently occupy page cache. The pages are clean, and so reclaimable, once written
  back.
- O_DIRECT keeps spilled data out of the page cache (uring, and device sources on cuFile), which
  matters because host memory is itself a tier.
- The sub-page tail of each buffer is one extra small op per buffer on both backends (with real
  GDS, the cuFile reactor splits writes into an aligned `cuFileWrite` body + staged POSIX tail;
  in compatibility mode the whole slice is staged). Padding the extents in the file would not
  remove it: the *source* buffers (RMM / host blocks) are not padded, so the tail
  bytes must come from staging anyway. Accepted.
- Follow-ups, not in this change: use `max(4 KiB, page size)` as the file alignment on 64 KiB-page
  aarch64; 4 KiB-align large buffers in `host_table` so HOST <-> DISK needs no staging memcpy.

## 9. Concurrency

- No disk-tier lock remains: contexts are thread-safe and each conversion has its own `io_object`, so
  concurrent conversions on one space run in parallel across reactors (today they serialise on the
  pipeline mutex). One request spreads over at most two reactors (`templated_ioctx` fan-out); the write API cuts
  large slices (`write_split_bytes`, uring 32 MiB) so a single large column still pipelines across
  both.
- `memory_space`'s context accessors are mutex-guarded; conversions hold a `shared_ptr` copy for
  their duration, so a concurrent `set_io_context` never destroys a context in use.
- No callback runs on reactor threads; the converting thread waits on the future.

## 10. Error handling and RAII

- `disk_file_guard` stays: created right after the path is generated, unlinks the file on any
  exception (open failure, `not_supported`, ENOSPC, CUDA error, short read/write).
- **Every converter awaits its future before any owning scope can unwind, including on exception
  paths.** Structure: (1) everything that can throw while preparing — planning, allocations, slice
  vectors, opening the `io_object` — happens before the submit; (2) exactly one submit call
  (`noexcept`, returns a future); (3) the wait. A scope guard (`await_on_unwind`) holding the future
  additionally waits (and swallows the result) if anything between (2) and (3) ever throws, so the
  invariant survives future edits. Partial submits cannot happen: a table is one request, and the write API
  guarantees an errored future resolves only after every published op drained (incl. staging copies
  and cuFile stream work). Hence freeing GPU/host buffers, dropping the source table and unlinking
  the file after the wait are safe.
- Declaration order makes unwinding safe too: buffers and sources are declared before the guard /
  future, so they are destroyed after it waited.
- `~disk_data_representation` unchanged (noexcept `std::filesystem::remove`).
- Error types are preserved (`std::system_error` with errno, CUDA errors, `operation_canceled`).

## 11. Tests

Core (`cucascade_tests`), `test_disk_memory_space.cpp`: `get_io_context()` null by default;
`std::logic_error` on GPU/HOST spaces; `set_io_context(nullptr)` and a null-returning
`get_or_init_io_context` throw; `generate_disk_file_path` and alignment cases kept.

cudf (`cucascade_cudf_tests`, links io):
- The existing round trips in `test_gpu_disk_converters.cpp` / `test_disk_host_converters.cpp` run
  for the default and for each forced backend (`GENERATE(uring, cufile)`, cuFile in compat mode
  here) by building disk spaces with `make_disk_io_context(...)`.
- New: multi-MiB tables (> a staging slot, > `gds_threshold`, > the write split); 8 threads converting
  concurrently on one space; default adoption (two spaces without a context share one default, an
  explicit context is honoured, `set_io_context` swaps); errors leave no file behind (unwritable
  mount; file removed or truncated before DISK -> GPU/HOST throws); `~disk_data_representation`
  removes the file.
- `test_release_table_stream.cpp` and the bandwidth profiler tests run unchanged.

## 12. Benchmark migration (`benchmark/benchmark_disk_converter.cpp`)

- DISK spaces: disk 0 = default (uring here), disk 1 = forced cuFile context; benchmarks take the disk
  id as an argument; `*Pipeline` names become backend-neutral.
- Reads always cold (`drop_os_cache` before each iteration), writes include `fdatasync` (as now).
  Mount directory from `CUCASCADE_BENCH_DISK_DIR` (default `/tmp`; `/mnt/disk_2` is absent here).
- Before/after: the old pipeline at a19cf0a, rebuilt from a `git archive` snapshot in the scratch
  dir, run back to back with the new binary; GPU<->DISK and HOST<->DISK, 64 MiB / 512 MiB, numeric
  and string tables. cuFile numbers labelled "compat".

## 13. Implementation order

CMake (cudf requires io) and core accessors; the cudf factory and `disk_io` helpers; read and
write converters; tests, benchmark and docs. The removal of the old interface, registry, pipeline
backend and `io_worker` lands in the same commit as the converter migration.

## 14. Decisions

- Q1. `CUCASCADE_BUILD_CUDF` requires `CUCASCADE_BUILD_IO` (`FATAL_ERROR` with a clear message).
- Q2. One context per disk space (no host/device roles): `disk_memory_space_config::io_context`,
  `get_io_context` / `set_io_context` / `get_or_init_io_context`, one defaulted configurator
  parameter. Default = shared cuFile context when `io::cufile::gds_available()`, else the shared uring
  context, for both host and device paths. `disk_io_context_options::backend` lets users and tests
  force uring / cufile per space.
- Q3. Process-shared, lazily built defaults with the io_config defaults (weak_ptr cache).
- Q4. Remove the old API outright, no deprecation.
- Automatic stays cuFile on real-GDS hosts; the cufile reactor's host path was brought to uring
  speed instead (section 6). The DISK tier probes io_uring once; forced uring fails fast, automatic
  falls back to the cuFile reactor's POSIX path. `cufile.log`: documented (section 6).
- Q5. Update `docs/bandwidth-profiler.md`.
- DISK -> GPU's trailing `stream.sync()` is decided from the read future's completion guarantee and
  documented in a code comment (section 5).
- Every converter awaits its future before any owning scope can unwind, also on exception paths
  (section 10).

## 15. Results

Measured with `benchmark_disk_converter` on `/tmp` (Samsung 990 PRO 2 TB behind LUKS/dm-crypt,
ext4), 4 columns, 512 MiB tables, page cache dropped before every read on both sides. The old
pipeline backend was rebuilt from a snapshot of the previous tree with the same cold-read change.
Throughput in GiB/s; new = the default context (uring here, since `gds_available()` is false).

| Path | Old pipeline | New (uring) | cuFile (compat mode), before the host-path rework |
|---|---|---|---|
| DISK -> GPU | 1.3-1.9 | 5.1 | 5.0-5.4 |
| DISK -> HOST | 0.6-1.2 | 4.5-4.6 | 1.3-1.4 |
| DISK -> GPU, strings | 1.2 | 4.8-6.3 | 2.8-4.3 |
| GPU -> DISK | 2.3-2.5 | 4.3-4.8 | 1.5-3.7 |
| HOST -> DISK | 2.55-2.62 | 4.1-4.6 | 1.3-1.5 |
| GPU -> DISK, strings | 2.3-2.5 | 4.55-4.9 | 1.3-2.6 |

The cuFile column predates the host-path rework and partly ran with the drive past its SLC
cache; section 6 has the cuFile HOST <-> DISK numbers after the fix (within ~5-12% of uring).

Read rows: two back-to-back runs of the whole sweep in a quiet-disk window. Write rows: old and new
alternated as single-benchmark processes, three rounds. All numbers were taken without nvidia-fs
(cuFile in compatibility mode); they say nothing about real GDS throughput.

Caveat — sustained writes: after ~20 GB of back-to-back writes in one process, every write path
(uring included) drops to ~1.3-1.4 GiB/s. That matches the 990 PRO running out of its SLC cache
(TLC sustained ~1.4-1.6 GB/s); short runs and the old path, which writes about half as fast, rarely
get there. Compare write numbers only from runs that start with the drive's cache drained.

Exit-time behaviour: a release build tolerates DISK spaces (and so their contexts) destroyed after
the CUDA runtime and the cuFile driver were torn down — the context teardown only makes CUDA
runtime calls whose errors are discarded and no cuFile calls (no io_object or registered buffer
outlives a conversion); `test(disk): exit a process with live disk io contexts held in globals`
checks it. Debug builds assert on those post-teardown CUDA errors, as they already do for HOST
memory spaces kept in a global; hence the note on `make_disk_io_context()`.
