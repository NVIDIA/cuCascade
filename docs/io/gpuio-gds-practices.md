# gpuio GDS practices for the cuCascade cuFile reactor

Ranked, concrete performance practices borrowed from gpuio's GDS module (`/home/aaramoon/workspace/oss/worktrees/gpuio/gds`, module T29) and applicable to `cucascade::io::cufile` **without redesigning the reactor**: one worker thread per reactor, the tiered queue and its scheduler, `templated_ioctx` dispatch, and configuration fixed at the io_context level all stay. gpuio's architecture (worker pools per device, owned-destination arena, batch engine, per-class routing tables, dozens of knobs) is deliberately **not** copied.

- Written for task W2 (team `gds`), 2026-10-19. Inputs: gpuio `include/gpuio/gds.hpp`, `src/gds/*`, `docs/arch/GDS_DESIGN.md`, `docs/okf/**/*gds*`, `bench/results/**/gds_*`; cuCascade `docs/io/gds-reactor-recommendations.md` (R-items) and `gds-cufile-best-practices.md`; the current reactor at `a19cf0a`.
- **Every number below is compatibility mode** (no `nvidia-fs`, libcufile 1.17.1 from the pixi env, `cuFile running in compatible mode` in `cufile.log`). Host: 24 cores, 2x RTX 6000 Ada, Samsung 990 PRO behind dm-crypt + LVM + ext4. Raw ceiling for reads on that volume: 7.1-7.2 GB/s (gpuio `bench/results/dev_box_fio_baseline.md`; the uring reactor reaches 6.9-7.2 below). Page cache dropped per file with `posix_fadvise(DONTNEED)` before every run. The `/home/aaramoon/workspace` scratch disk is a SATA SSD (~0.55 GB/s, btrfs + zstd) and was not used for throughput.
- Probe: the hidden Catch2 cases `[.cufile-bench]`, `[.cufile-probe-api]`, `[.cufile-probe-submit]`, `[.cufile-probe-write]` in `test/io/cufile/test_cufile_reactor.cpp` (driven by `CUCASCADE_IO_BENCH_*` environment variables).

## 1. Baseline: where the current reactor loses

The current reactor serves a large device-only slice with **one `cuFileReadAsync` on the caller's stream** (`src/io/cufile/cufile_reactor.cpp:589`), serialised by a process-wide per-stream mutex (`:144`, `:588`), one heap argument block and one CUDA event per op, completion polled every 200 µs (`:820`). Everything else is buffered `pread` plus, for device-only slices, a `cudaMallocHost` / `cudaFreeHost` per op (`:642`).

Measured (`[.cufile-bench]`, 4 reactors, 4 GiB file, GB/s):

| scenario | cufile reactor | uring reactor |
|---|---|---|
| 1 device slice of 256 MiB at a time (QD1 per caller) | **2.04** | 6.92 |
| 1 device slice of 1 GiB at a time | **2.43** | 7.16 |
| `device_readv` 4 MiB slices, one stream | **2.40** | 7.19 |
| `device_readv` 16 MiB slices, one stream | **2.41** | 7.17 |
| 1 MiB device reads over 8 streams | 7.07 | 6.94 |
| 8 MiB device reads over 8 streams | 7.09 | 7.00 |
| 256 KiB device reads over 8 streams (POSIX fallback) | **1.48** | 6.80 |
| 4 MiB host reads (buffered `pread` fallback) | 3.76 | 4.48 |

Why (`[.cufile-probe-submit]`, `[.cufile-probe-api]`):

1. **In compat mode `cuFileReadAsync` is synchronous in the submitting thread**: one submit of 4 MiB blocks 2.24 ms, 16 MiB 8.5 ms, 64 MiB 38 ms; the stream wait afterwards is ~1 ms. So each reactor is a QD1 reader that also misses cuFile's internal request split: 1.3-2.4 GB/s per reactor. Lane streams do not help (one thread, 1/4/8/16 lanes: 1.4-2.3 GB/s).
2. **One stream is FIFO** (NVIDIA BP §4.6), so a multi-slice device read on the caller's stream runs one cuFile op at a time even when the slices fan out to two reactors; the per-stream mutex adds lock contention on top.
3. **Stream-API fragility observed here**: with long-lived streams that had carried many ops, `cuFileReadAsync` returned success and later reported `bytes_done = -5` (`cufile.log`: `cuStreamWaitValue32 failed ... ret 2`). cuFile keeps per-stream state keyed by the stream handle, and the reactor submits on *caller* streams it neither registers nor deregisters, so a destroyed-and-reused caller stream handle is a latent failure on top of the documented ~133-outstanding-ops failure (R2) and the same-stream abort the mutex works around.
4. The POSIX fallback allocates pinned memory per op (`cudaMallocHost` is slow and implicitly synchronising) and reads through the page cache.

The synchronous API on the same file (`[.cufile-probe-api]`, unregistered `cudaMalloc` destination, GB/s):

| op size | 1 thread | 4 threads | 8 threads |
|---|---|---|---|
| 256 KiB | 1.36 | 4.04 | 6.55 |
| 1 MiB | 2.59 | 7.20 | 7.17 |
| 4 MiB | 3.74 | 7.14 | 7.21 |
| 16 MiB | 6.00 | 6.99 | 7.13 |
| 64 MiB | **6.83** | 7.18 | 7.16 |

One thread issuing ≥ 16 MiB `cuFileRead` calls already reaches 85-96% of the device, because cuFile splits a call of ≥ `min_io_threshold_size_kb` (8 MiB) over `max_request_parallelism` (4) internal threads.

## 2. Ranked practices

Ranking = expected gain on the device read/write paths x confidence / (risk + code). "Scope" says read (R), write (W) or both.

| # | Practice | Scope | Expected effect (compat, this box) | Risk | Recommend |
|---|---|---|---|---|---|
| G1 | Synchronous `cuFileRead` on the reactor worker, chunked at `max_op_bytes`, instead of `cuFileReadAsync` on the caller's stream | R (W: same for `cuFileWrite`) | single large slice 2.0-2.4 -> ~6-6.8 GB/s; one-stream readv 2.4 -> ~7; removes the global mutex, per-op events / arg blocks / 200 µs polling, and the stream-API failure classes | low-medium (reactor blocks per chunk, as it already does for `pread`) | **apply (task-4)** |
| G2 | Order the read after the caller's stream before writing its memory (`cudaStreamQuery` fast path, else event + synchronize), complete host-observed | R, W (source readiness) | correctness of G1; ~µs per slice | low | **apply with G1** |
| G3 | Chunk boundaries on 4 KiB file offsets; split the EOF tail at `align_down(file_size)`; loop short transfers; `ret == 0` below EOF = error; decode `-1`+`errno` vs cuFile codes | R, W | correctness; direct DMA for the aligned body on a real GDS host | low | **apply with G1** |
| G4 | Device-only small reads: lower `gds_threshold` (1 MiB -> 256 KiB) and replace per-op `cudaMallocHost` with a per-reactor pinned scratch reused across ops (chunked, double-buffered) | R | 256 KiB x 8 streams 1.48 -> ~4 GB/s (sync cuFile 256 KiB x 4 thr = 4.04); unregistered-file device reads stop allocating pinned memory per op | low | **apply (task-4)** |
| G5 | Platform probe: `cuFileDriverGetProperties` once, `compat_only = dstatusflags == 0`, log the effective mode once, expose `cufile::probe_platform()` / `gds_available()` | R, W | no throughput; makes compat visible; needed by the disk tier (W3) | low | **apply (task-4)** |
| G6 | Writes: `cuFileWrite` on an `O_RDWR \| O_DIRECT` registered fd for the 4 KiB-aligned body of a device source, staged POSIX for heads/tails/host sources, extend only to a block boundary, exact tail last, `fdatasync` for flush | W | compat `cuFileWrite` 2.1-2.65 GB/s on a fresh file (flat in threads) vs `dd oflag=direct` 16 MiB 1.5 (fresh) / 3.2 (overwrite); compare with uring in task-5 | medium | **apply (task-5)** |
| G7 | cuFile parameters before the first driver open (`max_direct_io_size_kb = 16384`), only when `cuFileUseCount() == 0` | both | none measured here: a private JSON with parallelism 8 / threshold 1 MiB / 16 threads made 16-64 MiB reads *worse* (6.0/6.83 -> 5.28/5.23) and 4 MiB slightly better (3.74 -> 4.20) | medium (process-wide; kvikIO/cuDF may open first) | defer to a GDS host |
| G8 | Register large device buffers per transfer (`cuFileBufRegister` / `Deregister` around the I/O) | both | compat: +5-15% at <= 4 MiB single-thread, neutral >= 16 MiB; gpuio compat: registered arena 3.69 vs unregistered 6.29 GB/s | medium-high (BAR1, dereg-before-free, NVIDIA advises against one-shot) | **off-by-default knob** `register_min_bytes` (approved; task-5) |
| G9 | Per-reactor counters (bytes by path, fallback reasons, unaligned ops) | both | diagnosis only | low | optional, small |

Not adopted (and why): gpuio's batch engine (2-10% below sync on real GB200 hardware, entry-loss hazards in compat; `decision-gds-engines-and-arena.md` Decision 1), the owned registered device arena (an API change; registration is not a win here, G8), lane streams / stream API (measured above), the two-phase driver probe F5 (the cuFile ioctx is opt-in, so the one-time open cost is accepted), replacing kvikIO's shim with an own `dlopen` table (R6: build-layering work, not performance), per-mount routing tables and NAS defaults.

## 3. Details

### G1. Synchronous `cuFileRead` on the reactor worker, chunked

- **gpuio evidence.** Sync is the default engine everywhere: `GDS_DESIGN.md` §0 item 2 (lines 39-55) and `decision-gds-engines-and-arena.md` Decision 1 ("`auto` = `sync` everywhere"; NIXL on GB200 puts batch 2-10% below sync-MT); the engine is one blocking call per op, `src/gds/gds_engine_sync.cpp:71-93` (`run_read`: unregistered destination = exact pointer + `devPtr_offset` 0; the same function issues `cuFileWrite` for write ops). The stream API is explicitly not implemented: `GDS_DESIGN.md` §13 (line 1241: FIFO per stream, the 133-op failure, the cucascade same-stream abort) and the side-by-side comparison §14 (line 1262). Ops are cut at `max_op_bytes` (16 MiB, `include/gpuio/gds.hpp:69`; `cut_gds_ops`, `src/gds/planner_gds.cpp:175`).
- **cuCascade change sketch** (`cufile_reactor.cpp` only): in `dispatch_slice`, a registered device-only slice >= `gds_threshold` calls a new `read_cufile()` that loops `cuFileRead(handle, dst + done, n, offset + done, 0)` with `n = min(max_op_bytes, remaining)` and finishes the `io_op_request` inline, exactly like the POSIX path does today. Delete `submit_cufile`, `stream_submit_mutex`, `inflight_cufile`, and `cufile_stream_args` (`include/cucascade/io/cufile/types.hpp`); `poll_completions` keeps only the fallback H2D copies. New config field `max_op_bytes` (default **64 MiB**: 6.83 vs 6.00 GB/s at 16 MiB single-threaded here; gpuio's 16 MiB came from batch-entry clipping and NAS, re-sweep on a GDS host). A cuFile failure (negative return) still degrades to the POSIX fallback for that slice, logged once per reactor, so behaviour stays correctness-preserving.
- **Effect.** Per-reactor QD becomes 1 blocking call with cuFile's internal 4-way split, which is what saturates this NVMe from one thread at >= 16 MiB. The pool QD is `cufile_n_reactors` (4), far below the documented async failure edge (R2 resolved structurally, as gpuio notes in §0 item 4).
- **Risk.** The worker blocks for one chunk (~10 ms at 64 MiB); priority preemption stays at slice boundaries as today. Real-GDS behaviour of sync vs stream API is unmeasured here, but it is NVIDIA's own recommended pattern (cuDF, NIXL, DALI).

### G2. Stream ordering for a synchronous read

- **gpuio evidence.** GDS spans complete "host-observed by construction" (`GDS_DESIGN.md` §2.1, lines 281-287); stream-ordered device *sources* are waited on with a bounded, stop-aware event query before cuFile touches them (`src/gds/gds_write.cpp:202`, `gds_wait_write_sources`).
- **cuCascade.** The io_context contract is already host-observed: the uring reactor issues its H2D copy on the caller's stream only after `pread` completes, so a caller must wait on the future before consuming. What the current cuFile path gives in addition is "the read starts after work already queued on the stream" (a stream-ordered allocation from RMM, a prior kernel on the buffer). Keep that: before the first chunk, `cudaStreamQuery(stream)`; if not ready, record a reactor-owned reusable event and `cudaEventSynchronize` it. Errors from the query/record fail the slice with the CUDA error.

### G3. Chunk geometry, EOF tail, short transfers, error decoding

- **gpuio evidence.** EOF rule `split_at_eof`, `src/gds/planner_gds.cpp:72-88` (last op split at `align_down(file_size)` so every op but the tail is 4 KiB-aligned; never read past EOF: the one documented nvidia-fs EOF defect); `classify_gds_ret`, `:457-490` (short below EOF -> resubmit the remainder up to a bound, `ret == 0` below EOF -> `io_error`, more than asked -> error); error decoding `src/gds/cufile_error.cpp:75-83` (`ret == -1` -> `errno`, which may hold a cuFile code > 5000; `ret < -1` -> negative `CUfileOpError`).
- **cuCascade.** Chunk ends land on 4 KiB file boundaries (first chunk may be short to reach one); the last chunk splits at `align_down(file_size)`. A short read below EOF continues; `ret == 0` or > 64 attempts fails with `io_error`. Decode `errno` into `std::system_category` or a cuFile message for the log.

### G4. Small device reads and the fallback staging

- **gpuio evidence.** `gds_min_bytes` 256 KiB (`include/gpuio/gds.hpp:66`, `GDS_DESIGN.md` §3.3 line 388: TPC-H merged requests have nothing between 64 and 256 KiB; kernel >= 6.11 regresses <= 128 KiB I/O in real mode); gpuio never allocates per op ("one allocation per request, 0 per op", `module-gds.md` Invariants) and keeps pinned staging pooled.
- **cuCascade.** (a) Default `gds_threshold` 1 MiB -> 256 KiB: compat sync cuFile at 256 KiB reaches 4.04 GB/s with 4 threads, against 1.48 for today's fallback. (b) The fallback for a device-only slice (small, or a file that did not register: `compat_mode = ON`, no O_DIRECT, tmpfs) reads into a per-reactor pinned scratch allocated once at `start()` (2 x `fallback_chunk` 8 MiB, double-buffered: `pread` chunk k+1 while the H2D of chunk k runs, both on the caller's stream; synchronise the last copy before the slice finishes so the scratch can be reused). Mixed host+device slices keep today's path (they copy from the caller's own host buffer).
- **Risk.** Low; the pinned scratch (16 MiB per reactor) is allocated with `cudaHostAlloc` and freed at shutdown.

### G5. Effective-mode probe

- **gpuio evidence.** `read_platform`, `src/gds/gds_context.cpp:1079-1093`: `compat_only = (dstatusflags == 0)` from `cuFileDriverGetProperties`; `cuFileDriverGetP2PFlags` is deliberately never called (it resets running parameters in 1.19.1; `decision-gds-engines-and-arena.md` Decision 2); one summary line logged at open (`:840-860`).
- **cuCascade.** New `include/cucascade/io/cufile/platform.hpp` (no kvikIO types): `struct platform_info { library_available, driver_opened, compat_only, version, dstatusflags, max_direct_io_size_kb }`, `probe_platform() noexcept` (computed once, shares the reactor's driver-open once-flag) and `gds_available() noexcept` (= driver opened and not compat-only and not forced compat by `KVIKIO_COMPAT_MODE=ON` / `CUFILE_FORCE_COMPAT_MODE`). The reactor logs the mode once at `start()`. The disk tier (W3) uses `gds_available()` to pick cuFile vs uring (agreed with `disk-tier`).

### G6. Write practices (for task-5, on io-write's reactor contract)

- **gpuio evidence.** `GDS_DESIGN.md` §7 (line 923) and `split_write_span`, `src/gds/gds_write.cpp:41-117`: a cuFile piece is a run of device sources whose pointer is congruent to the file offset mod 4 KiB, with both edges on `G = max(4 KiB, O_DIRECT block, page)` and size >= the GDS threshold; heads, tails, the exact-EOF block and host sources go to the staged local path; cuFile and local pieces never share a block; a cuFile write extends the file only to a `G` boundary. Unaligned cuFile writes are read-modify-write through POSIX inside cuFile [DOC]. Durability is an explicit `fsync` (cuFile flushes no metadata). `O_RDWR | O_DIRECT` fds register (GQ15, `GDS_DESIGN.md` line 936). Short writes resubmit the remainder, `ret == 0` is `io_error` (§7 addendum). Source readiness: bounded event wait (G2). The local write path (`flow-write-local.md`) writes exact EOF tails through a buffered fd and, on 64 KiB-page kernels, syncs that range before finishing.
- **cuCascade.** `cufile_io_object` gains a write mode: one `O_RDWR | O_DIRECT` fd registered with cuFile plus a buffered `O_RDWR` fd for tails (no `O_APPEND`/`O_NOATIME`: cuFile rejects them). Per slice: wait for the source stream (G2); aligned body via chunked `cuFileWrite`; unaligned head/tail and host sources via a pinned scratch (D2H for device sources) and `pwrite`; flush = `fdatasync`. Measured compat ceiling so far: `cuFileWrite` 2.1-2.65 GB/s regardless of 1-8 threads and 1-64 MiB ops (fresh file, `[.cufile-probe-write]`); `dd oflag=direct bs=16M` 1.5 GB/s fresh, 3.2 GB/s overwrite. gpuio recorded 0.37-0.42 GB/s sync cuFile writes in compat on its T29f probe and 6.16 GB/s for its aligned device-write gate row (`bench/results/suite/gds_gates_devbox_compat.summary.md:12`); the two disagree, so task-5 measures uring vs cuFile on the same file before choosing defaults.

### G7. cuFile parameters (deferred)

- **gpuio evidence.** `apply_params_locked`, `src/gds/gds_context.cpp:965-1050`, only when `cuFileUseCount() <= 0` (`:786-799`: parameters act only before the first open in the process); default `max_direct_io_size_kb = 16384` (`include/gpuio/gds.hpp:118`); the stdin guard around the open (`:746`).
- **Why deferred.** kvikIO's shim exposes no `cuFileSetParameter*`, so this needs `dlsym(RTLD_NOLOAD)` plumbing; the only measured change on this box made large reads slower; the real-mode value of 16384 is a GDS-host experiment (E7).

### G8. Buffer registration (not by default)

- **gpuio evidence.** Never register caller or pool memory implicitly (`GDS_DESIGN.md` §4.3 line 536 items 2-3: undocumented registrability of `cudaMallocAsync`/RMM memory, nvidia-fs deadlock on free-without-deregister); registration pays only for long-lived reused buffers (the arena). In compat, gpuio's registered arena measured 3.69 GB/s against 6.29 unregistered (`gds_gates_devbox_compat.summary.md:29-30`); on GPFS/RDMA one in-flight I/O per registration (CR H7).
- **Here.** `cuFileBufRegister` of a 1 GiB `cudaMalloc` buffer took 1.3-27 ms; registered sync reads gained 5-15% at <= 4 MiB single-threaded and nothing at >= 16 MiB. The project constraint allows temporary per-transfer registration for large I/O, but nothing measured supports it and NVIDIA's table says "do not register" for a one-shot destination. If kept as an option it must be off by default, only for 4 KiB-aligned slices >= 64 MiB, deregistered before the slice completes.

### G9. Counters (optional)

`GDS_DESIGN.md` §10: bytes by path, fallback reasons, unaligned ops. A small `reactor_stats` accessor would make the compat/fallback split visible in benchmarks; not needed for the perf work.

## 4. Proposed order

task-4 (read): G1 + G2 + G3 (one change), then G4, then G5; re-run `[.cufile-bench]` and the existing `[cufile]` tests after each. task-5 (write): G6 on io-write's contract, plus G8 as an off-by-default knob (`register_min_bytes = 0`; when set, only 4 KiB-aligned contiguous device slices of at least that size are registered before the first chunk and deregistered before the slice settles, on every path).

**Follow-ups for a GDS-capable host** (lead decision, 2026-10-19): G7 (cuFile parameters, starting with `max_direct_io_size_kb = 16384`), G9 (per-reactor counters), re-sweeping `max_op_bytes` and `gds_threshold` in real GDS mode, and the G8 registered-vs-unregistered A/B with 4 KiB-aligned destinations.
