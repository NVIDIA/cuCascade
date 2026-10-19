# Disk tier on io_context: design (W3)

Status: approved by the lead (decisions recorded in section 14). Builds on the W1 write API
(`docs/io/write-api-design.md`) and the W2 cuFile write path. Scope: the DISK memory tier —
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
cucascade_io        ioctx, uring / cufile / kvikio contexts, write API (W1, W2)
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
  /// Context that serves this space's files (uring, cufile, kvikio, ...); must be started and
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
enum class disk_io_backend : std::uint8_t { automatic, uring, cufile, kvikio };

struct disk_io_context_options {
  /// automatic: cufile when io::cufile::gds_available(), else uring.
  disk_io_backend backend{disk_io_backend::automatic};
  /// The io stack's own static configuration: uring_n_reactors / uring, cufile_n_reactors /
  /// cufile, kvikio. backend and cache are ignored (the disk tier never arms an fs_cache).
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

- The default serves BOTH host and device paths of a space. On a GDS host, HOST <-> DISK therefore
  goes through the cufile reactor's POSIX path — accepted; follow-up in section 6.
- uring staging ownership uses the hook the reactor already provides: a private subclass of
  `uring_reactor::reactor_context` owns the upstream pinned resource and the
  `fixed_size_host_memory_resource` (base-from-member); every reactor's shared `_ctx` keeps it
  alive and each reactor's `_bounce_storage` is released before it. No custom deleter, so the
  context is a plain `make_shared` (its `enable_shared_from_this` stays valid).
- GDS probe: owned by W2 (agreed with `gds`, landed in ac082ad):
  `cucascade::io::cufile::gds_available()` in `include/cucascade/io/cufile/platform.hpp` —
  process-wide, computed once, never throws; true iff libcufile loads, the driver opened, it reports
  a real GDS mode (`nvfs.dstatusflags != 0`) and `KVIKIO_COMPAT_MODE` / `CUFILE_FORCE_COMPAT_MODE` do
  not force compat. The header does not include kvikIO. The disk tier asks it through one private
  helper (`detail::gds_is_available()`); the first call opens the cuFile driver (~0.3 s), which only
  happens when an automatic context is built. False on this host (no
  nvidia-fs), so the tests drive both backends through `disk_io_context_options::backend`. A file on a
  filesystem cuFile cannot register (btrfs, tmpfs) still takes the cufile reactor's POSIX path per
  file — correct, just slower; `automatic` does not try to detect that.
- A context built by `io::io_context_registry::make_ioctx(...)` (staging from the HOST memory space)
  works too, through the config or `set_io_context`; the HOST space must then outlive it.
- Applications may share their scan context with the disk tier. The disk tier only uses the
  cache-unaware read primitives (`*_readv_async_io`), so spill files never enter or consult an
  fs_cache; W1's write-side `fs_cache::invalidate(path)` finds nothing for spill paths.

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
/// The space's context, adopting default_disk_io_context() if none is set; throws
/// std::logic_error if it does not support writes (e.g. a REST context).
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
- No `flush`: spill files need no durability; W1 3.2 guarantees a new (O_DIRECT or buffered)
  `io_object` of any backend sees the data once the write future resolved.
- Reads use the cache-unaware primitives and verify `bytes == sum(slice sizes)` (the ioctx clamps to
  EOF, so a truncated file shows up as a short count -> `std::runtime_error` naming the path).
- Device paths additionally require `supports_device_write()` (checked, `std::logic_error`).
- Priority `automatic` (= high) both ways. Later option: spill writes `low` when sharing with scans.
- The helpers build every slice **before** submitting and submit exactly once, so nothing can throw
  between submit and wait; the wait itself is the last statement that touches the request (see 10).

## 5. Converter changes (`src/cudf/representation_converter_builtins.cpp`)

- **GPU -> DISK**: planning unchanged (`wait_for_writer(stream)`, `plan_column_copy`,
  `recompute_file_offsets`); `collect_gpu_column_io_entries` produces `io::write_slice`s; one
  `device_writev_async` under `rmm::cuda_set_device_raii` for the source GPU (W1 binds device slices to
  the current device), on the caller's `stream`, so it observes the table's writer (W1 3.4). The
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
  2. one `device_readv_async_io(obj, slices, stream)` and wait — the reactor's H2D copies (uring) or
     `cuFileReadAsync` (cufile) are enqueued on `stream`, after the stream-ordered allocations;
  3. `assemble_disk_column(meta, buffers, stream, mr)` — the existing type dispatch (STRING with the
     INT32 -> INT64 offsets cast, LIST keeping INT32 offsets, STRUCT, DICTIONARY32, fixed width /
     fixed point), moving the buffers in.
  Then `make_written_on(table, target, stream)` as today.
- **The trailing `stream.sync()` of DISK -> GPU**: decided by the completion guarantee of the read
  future, not by a test. uring settles a device op only after the `cudaEventQuery` of the event it
  recorded on `stream` behind the op's H2D copies reports `cudaSuccess`; the cufile reactor settles a
  `cuFileReadAsync` only after the event it recorded on `stream` behind the submission completes, and
  its POSIX fallback after its H2D copy's event. So when the future resolves, every byte is in device
  memory and all stream work up to those events is done; the casts of phase 3 are enqueued on
  `stream` afterwards and `make_written_on` records the writer event behind them, which every reader
  waits on. The host-side `stream.sync()` therefore adds nothing for correctness and is dropped; this
  reasoning goes into a code comment, and a backend that resolved its future before the data landed
  would violate the ioctx read contract, not this converter.

## 6. Backend selection

| Context | Device sources/destinations | Host | Used |
|---|---|---|---|
| uring | staged through the reactor's pinned blocks, D2H/H2D on the caller's stream; aligned bodies O_DIRECT, sub-4 KiB heads/tails buffered (W1 3.3) | direct; O_DIRECT when address-congruent, else staged memcpy + O_DIRECT | default without GDS |
| cufile | `cuFileWrite/ReadAsync` above `gds_threshold` (W2), POSIX below | POSIX | default when `gds_available()` |
| kvikio | `kvikio::FileHandle` (GDS or its own bounce) | pwrite/pread | explicit opt-in only |

How a user configures it:
1. Nothing: the space adopts `default_disk_io_context()` — cuFile on GDS hosts, uring elsewhere.
2. Per space: `config.io_context = make_disk_io_context({.backend = disk_io_backend::cufile})` (forces
   cuFile even in compatibility mode, e.g. for testing), `uring` or `kvikio`; or
   `space.set_io_context(ctx)` after construction; or any started, write-capable `ioctx` (e.g. one
   shared with the scan path).
3. Process-wide knobs inside the contexts stay where they are (`io::io_config` fields passed through
   `disk_io_context_options::io`, `KVIKIO_*` / `cufile.json`).

Rationale for "automatic = cuFile only with real GDS": cuFile in compatibility mode bounces through
its own per-call buffers (gds-reactor-recommendations.md R0), while uring's pinned staging is the
measured fast path; on this host automatic therefore resolves to uring, and cuFile is exercised
explicitly in tests and the benchmark.

Follow-up (not in this change): on GDS hosts HOST <-> DISK goes through the cufile reactor's POSIX
path (one synchronous `pread`/`pwrite` per op on the reactor thread, buffered fd). If that turns out
slow, either improve that path in the reactor (O_DIRECT for congruent host buffers) or let
`make_disk_io_context` build a composite that routes host-only requests to uring.

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
- Docs: `docs/bandwidth-profiler.md` updated; `AGENTS.md` left to the lead.

## 8. File format, alignment, page cache

- Format unchanged (files never outlive the process, so there is no compatibility surface).
- With W1's head/body/tail split every column buffer's body is written O_DIRECT; only its last
  partial 4 KiB page goes through the page cache. HOST -> DISK also splits buffers at the 1 MiB
  host-block boundaries (a sub-page buffered piece each side of a boundary). O_DIRECT keeps spilled
  data out of the page cache, which matters because host memory is itself a tier.
- The sub-page tail of each buffer is one extra small op per buffer on both backends (W2 splits
  cuFile writes into an aligned `cuFileWrite` body + staged POSIX tail). Padding the extents in the
  file would not remove it: the *source* buffers (RMM / host blocks) are not padded, so the tail
  bytes must come from staging anyway. Accepted.
- Follow-ups, not in this change: use `max(4 KiB, page size)` as the file alignment on 64 KiB-page
  aarch64; 4 KiB-align large buffers in `host_table` so HOST <-> DISK needs no staging memcpy.

## 9. Concurrency

- No disk-tier lock remains: contexts are thread-safe and each conversion has its own `io_object`, so
  concurrent conversions on one space run in parallel across reactors (today they serialise on the
  pipeline mutex). One request spreads over at most two reactors (`templated_ioctx` fan-out); W1 cuts
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
  invariant survives future edits. Partial submits cannot happen: a table is one request, and W1
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
  here; kvikio when its writes land) by building disk spaces with `make_disk_io_context(...)`.
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

## 13. Implementation plan (task-9, starts when this design is complete)

1. CMake (cudf requires io) and core: `disk_memory_space_config::io_context`, `memory_space`
   accessors, configurator; remove the old interface, registry, pipeline backend, `io_worker` and
   their tests.
2. cudf: `disk_io_context.{hpp,cpp}` (options, factory with owning uring staging, shared default,
   `detail::gds_is_available()` forwarding to W2's `io::cufile::gds_available()`) and
   `disk_io.{hpp,cpp}`.
3. Read converters first (DISK -> GPU in three phases, DISK -> HOST) — they use the existing read API.
4. Write converters (GPU -> DISK, HOST -> DISK) once io-write completes task-3.
5. Tests (core + cudf), the benchmark, `docs/bandwidth-profiler.md`, header comments.
6. `-Werror` build of all targets; `cucascade_tests`, `cucascade_cudf_tests`, `cucascade_io_tests`
   green; clang-format / codespell clean; before/after benchmark. Commits that each build.

## 14. Decisions (lead review)

- Q1. `CUCASCADE_BUILD_CUDF` requires `CUCASCADE_BUILD_IO` (`FATAL_ERROR` with a clear message).
- Q2. One context per disk space (no host/device roles): `disk_memory_space_config::io_context`,
  `get_io_context` / `set_io_context` / `get_or_init_io_context`, one defaulted configurator
  parameter. Default = shared cuFile context when `io::cufile::gds_available()`, else the shared uring
  context, for both host and device paths. `disk_io_context_options::backend` lets users and tests
  force uring / cufile / kvikio per space.
- Q3. Process-shared, lazily built defaults with the io_config defaults (weak_ptr cache).
- Q4. Remove the old API outright, no deprecation. `AGENTS.md` is left to the lead.
- Q5. Update `docs/bandwidth-profiler.md`.
- DISK -> GPU's trailing `stream.sync()` is decided from the read future's completion guarantee and
  documented in a code comment (section 5).
- Every converter awaits its future before any owning scope can unwind, also on exception paths
  (section 10).
