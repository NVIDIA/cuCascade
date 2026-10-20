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
- **cuCascade.** New `include/cucascade/io/cufile/platform.hpp` (no kvikIO types): `struct platform_info { library_available, driver_opened, compat_only, version, dstatusflags, max_direct_io_size_kb }`, `probe_platform() noexcept` (computed once, shares the reactor's driver-open once-flag) and `gds_available() noexcept` (= driver opened and not compat-only and not forced compat by `CUFILE_FORCE_COMPAT_MODE`; `KVIKIO_COMPAT_MODE` was honoured until kvikIO was removed on 2026-10-19). The reactor logs the mode once at `start()`. The disk tier (W3) uses `gds_available()` to pick cuFile vs uring (agreed with `disk-tier`).

### G6. Write practices (for task-5, on io-write's reactor contract)

- **gpuio evidence.** `GDS_DESIGN.md` §7 (line 923) and `split_write_span`, `src/gds/gds_write.cpp:41-117`: a cuFile piece is a run of device sources whose pointer is congruent to the file offset mod 4 KiB, with both edges on `G = max(4 KiB, O_DIRECT block, page)` and size >= the GDS threshold; heads, tails, the exact-EOF block and host sources go to the staged local path; cuFile and local pieces never share a block; a cuFile write extends the file only to a `G` boundary. Unaligned cuFile writes are read-modify-write through POSIX inside cuFile [DOC]. Durability is an explicit `fsync` (cuFile flushes no metadata). `O_RDWR | O_DIRECT` fds register (GQ15, `GDS_DESIGN.md` line 936). Short writes resubmit the remainder, `ret == 0` is `io_error` (§7 addendum). Source readiness: bounded event wait (G2). The local write path (`flow-write-local.md`) writes exact EOF tails through a buffered fd and, on 64 KiB-page kernels, syncs that range before finishing.
- **cuCascade.** `cufile_io_object` gains a write mode: one `O_RDWR | O_DIRECT` fd registered with cuFile plus a buffered `O_RDWR` fd for tails (no `O_APPEND`/`O_NOATIME`: cuFile rejects them). Per slice: wait for the source stream (G2); aligned body via chunked `cuFileWrite`; unaligned head/tail and host sources via a pinned scratch (D2H for device sources) and `pwrite`; flush = `fdatasync`. Measured compat ceiling so far: `cuFileWrite` 2.1-2.65 GB/s regardless of 1-8 threads and 1-64 MiB ops (fresh file, `[.cufile-probe-write]`); `dd oflag=direct bs=16M` 1.5 GB/s fresh, 3.2 GB/s overwrite. gpuio recorded 0.37-0.42 GB/s sync cuFile writes in compat on its T29f probe and 6.16 GB/s for its aligned device-write gate row (`bench/results/suite/gds_gates_devbox_compat.summary.md:12`); the two disagree, so task-5 measures uring vs cuFile on the same file before choosing defaults.

### G7. cuFile parameters (deferred)

- **gpuio evidence.** `apply_params_locked`, `src/gds/gds_context.cpp:965-1050`, only when `cuFileUseCount() <= 0` (`:786-799`: parameters act only before the first open in the process); default `max_direct_io_size_kb = 16384` (`include/gpuio/gds.hpp:118`); the stdin guard around the open (`:746`).
- **Why deferred.** kvikIO's shim exposed no `cuFileSetParameter*` (kvikIO has since been replaced by cuCascade's own loader, `io::cufile::cufile_api`, which can resolve them as optional members); the only measured change on this box made large reads slower; the real-mode value of 16384 is a GDS-host experiment (E7).

### G8. Buffer registration (not by default)

- **gpuio evidence.** Never register caller or pool memory implicitly (`GDS_DESIGN.md` §4.3 line 536 items 2-3: undocumented registrability of `cudaMallocAsync`/RMM memory, nvidia-fs deadlock on free-without-deregister); registration pays only for long-lived reused buffers (the arena). In compat, gpuio's registered arena measured 3.69 GB/s against 6.29 unregistered (`gds_gates_devbox_compat.summary.md:29-30`); on GPFS/RDMA one in-flight I/O per registration (CR H7).
- **Here.** `cuFileBufRegister` of a 1 GiB `cudaMalloc` buffer took 1.3-27 ms; registered sync reads gained 5-15% at <= 4 MiB single-threaded and nothing at >= 16 MiB. The project constraint allows temporary per-transfer registration for large I/O, but nothing measured supports it and NVIDIA's table says "do not register" for a one-shot destination. If kept as an option it must be off by default, only for 4 KiB-aligned slices >= 64 MiB, deregistered before the slice completes.

### G9. Counters (optional)

`GDS_DESIGN.md` §10: bytes by path, fallback reasons, unaligned ops. A small `reactor_stats` accessor would make the compat/fallback split visible in benchmarks; not needed for the perf work.

## 4. Proposed order

task-4 (read): G1 + G2 + G3 (one change), then G4, then G5; re-run `[.cufile-bench]` and the existing `[cufile]` tests after each. task-5 (write): G6 on io-write's contract, plus G8 as an off-by-default knob (`register_min_bytes = 0`; when set, only 4 KiB-aligned contiguous device slices of at least that size are registered before the first chunk and deregistered before the slice settles, on every path).

**Follow-ups for a GDS-capable host** (lead decision, 2026-10-19): G7 (cuFile parameters, starting with `max_direct_io_size_kb = 16384`), G9 (per-reactor counters), re-sweeping `max_op_bytes` and `gds_threshold` in real GDS mode, and the G8 registered-vs-unregistered A/B with 4 KiB-aligned destinations.

## 5. Results of task-4 (read path, compat mode)

Applied: G1 + G2 + G3 (`0819aec`), G4 (`184686d`), G5 (`ac082ad`). `[.cufile-bench]`, 4 reactors, 4 GiB file on the root NVMe, median of 3 rounds per pass, page cache dropped per round; base = `3714b26`; "pass 1 / pass 2" are interleaved A/B runs on a machine shared with other builds, so differences below ~10% are noise.

| scenario | base | after G1 | after G1-G5 |
|---|---|---|---|
| 1 device slice 256 MiB at a time | 1.84 / 1.90 | 6.25 / 6.14 | 6.49 / 6.27 |
| 1 device slice 1 GiB at a time | 1.90 / 1.97 | 6.37 / 6.21 | 6.60 / 6.26 |
| `device_readv` 4 MiB slices, one stream | 2.00 / 1.83 | 5.79 / 5.35 | 5.69 / 5.56 |
| `device_readv` 16 MiB slices, one stream | 1.98 / 2.04 | 6.34 / 6.21 | 6.39 / 6.49 |
| 64 KiB device reads, 8 streams (POSIX via pinned scratch) | n/a | n/a | 1.22 / 1.30 |
| 256 KiB device reads, 8 streams | 1.29 / 1.35 | 2.35 / 2.22 | 3.24 / 4.04 |
| 1 MiB device reads, 8 streams | 6.89 / 6.85 | 6.26 / 6.17 | 5.99 / 5.48 (isolated runs 6.62-6.96) |
| 8 MiB device reads, 8 streams | 6.82 / 6.80 | 6.69 / 6.61 | 6.66 / 6.63 |
| 4 MiB host reads (buffered `pread`) | 2.78 / 2.96 | 2.78 / 2.66 | 2.96 / 2.84 |

Device ceiling 7.1-7.2 GB/s; the uring reactor measured 6.6-7.2 GB/s on the device scenarios in the same session. The 1 MiB x 8 streams row was the one scenario where the old path already saturated the device; its spread after the change (5.5-7.0 across runs, 6.6-7.0 when run alone) is within the noise of this shared host and was not traced to a specific cost.

## 6. Results of task-5 (write path, compat mode)

Applied: G6 (`849d4b3`) and G8 as the off-by-default `register_min_bytes` knob (`b82d802`). `[.cufile-write-bench]` (`test/io/cufile/test_cufile_write.cpp`), 4 reactors, 1 GiB per scenario into a file on the root NVMe, timed including the final `fdatasync`, median of 3 rounds, two interleaved passes with 45 s idle between runs (the 990 PRO's SLC cache otherwise skews later runs), GB/s. "overwrite" rewrites an existing file, "fresh" truncates it every round (every write extends it). `cufile-staged` forces the D2H + `pwrite` path (`gds_threshold` = 1 TiB); `cufile-split` sets `write_split_bytes` = 64 MiB.

| scenario | cufile | cufile-staged | cufile-split | uring |
|---|---|---|---|---|
| overwrite: 1 device slice 256 MiB | 6.17 / 6.24 | 5.55 / 5.50 | 6.43 / - | 5.97 / 5.96 |
| overwrite: 1 device slice 1 GiB | 6.39 / 6.43 | 5.48 / 5.57 | 6.57 / 6.57 | 6.29 / 6.32 |
| overwrite: `device_writev` 4 MiB, one stream | 5.97 / 5.90 | 6.44 / 6.50 | 5.89 / 5.80 | 6.35 / 6.33 |
| overwrite: 8 MiB device writes, 8 streams | 6.56 / 6.55 | 6.51 / 6.48 | 6.58 / 6.54 | 6.25 / 6.05 |
| overwrite: 4 MiB host writes | 6.61 / 6.59 | 6.65 / 6.66 | 6.57 / 6.60 | 6.21 / 6.14 |
| fresh: 1 device slice 256 MiB | 2.63 / 2.24 | 4.62 / 5.28 | 0.62 / 2.24 | 4.23 / 4.78 |
| fresh: 1 device slice 1 GiB | 2.97 / 2.63 | 5.48 / 5.45 | 0.87 / 2.17 | 1.07 / 4.88 |
| fresh: `device_writev` 4 MiB, one stream | 2.42 / 2.23 | 4.35 / 4.64 | 1.05 / 2.44 | 0.72 / 4.94 |
| fresh: 8 MiB device writes, 8 streams | 2.07 / 2.24 | 3.41 / 5.09 | 0.90 / 2.28 | 1.26 / 5.57 |
| fresh: 4 MiB host writes | 4.49 / 4.91 | 3.00 / 4.82 | 0.77 / 4.54 | 1.39 / 5.93 |

Baselines, same session: `dd if=/dev/zero bs=8M oflag=direct` 1 GiB 4.6 GB/s fresh and 4.5 GB/s overwrite (QD1); raw `cuFileWrite` (`[.cufile-probe-write]`, 4 GiB fresh file) 2.1-2.65 GB/s whatever the thread count (1-8) and op size (1-64 MiB).

Reading: overwriting, every path saturates the device (5.5-6.7 GB/s) and the cuFile body path is on par with uring. Extending a fresh file, compat-mode `cuFileWrite` is capped near 2.2-3.0 GB/s: `strace` shows libcufile issuing 1 MiB `O_DIRECT` `pwrite`s from its own bounce pool, each extending the file, where the staged path writes 8 MiB chunks. The fresh-file rows of pass 1 for uring and `cufile-split` were taken while the SSD's write cache was recovering and are not representative; `write_split_bytes` stays 0 (no consistent gain overwriting, worse when extending).

**Policy (lead decision, 2026-10-19).** On a compat-only platform (`cufile::probe_platform().compat_only`) the reactor stages aligned device-write bodies too instead of calling `cuFileWrite`: cuFile has no DMA advantage there and staging measured faster on extending writes. With a real GDS mode the bodies go to `cuFileWrite`. There is no setting; tests cover the `cuFileWrite` path through the test-only `detail::set_cufile_writes_in_compat_for_testing`. Default configuration after the change, same host and method (one pass, GB/s): fresh 1 slice 256 MiB 5.32, 1 GiB 5.53, `device_writev` 4 MiB 4.83, 8 MiB x 8 streams 4.95, host 4.80 (was 2.1-3.0 for the device rows); overwrite 4.44, 4.42, 5.50, 5.90, 6.32. Overwriting with one large slice is the trade-off (6.2-6.4 with `cuFileWrite`, 4.4-5.5 staged, which is one reactor at QD1 like `dd bs=8M`).

## 7. Fixes after the independent review (task-7)

The W2 review (`w2-review`) found that G2 as implemented blocked the worker on the caller's stream (`cudaEventSynchronize`), so a long kernel on one caller's stream stalled every request routed to that reactor, and shutdown could not interrupt it. gpuio's own wait is bounded and checks for stop; the plan's "event + synchronize" was not. Changes:

- **Stream gate, no blocking.** A device-only slice (a read into or a write from device memory) whose stream still has queued work at dispatch is recorded with an event and parked in a deferred list. The worker polls that list with `cudaEventQuery` (like the in-flight H2D copies) and keeps serving other requests; a parked slice is dispatched once the work queued before its dispatch has finished. Shutdown, or a request whose coordinator stopped, settles parked slices at once (nothing was issued for them).
- **Staging copies on a reactor-owned stream.** Once a slice has passed the gate, its staged H2D / D2H copies run on a non-blocking stream per device owned by the reactor, so the double-buffer waits only wait for the reactor's own copies, never for work the caller queues later. At `shutdown()` the copy streams are destroyed and the scratch goes back to a process-wide pool for later reactors: `cudaFreeHost` would synchronize the whole device, so shutdown would wait for other callers' GPU work (review residual R1; a staged-path shutdown test with a held stream covers it).
- **Bounded passes and stop checks.** A worker pass also ends after about `max_op_bytes` of synchronous I/O (not only after `max_inflight` slices), and the cuFile and staged loops check for stop between pieces (settling with `operation_canceled`).
- **G8 registrations are shared, never pulled from under a transfer.** A process-wide table keyed by base address reference-counts registrations: a transfer whose buffer starts at a live registration's base and fits inside it shares it (the last user deregisters); one that overlaps a live registration otherwise takes the staged path; an `ALREADY_REGISTERED` from cuFile (someone else's registration) is treated the same way.
- **Compat write policy keys on `gds_available()`**, so a compat mode forced by `CUFILE_FORCE_COMPAT_MODE` also stages.
- **Tests.** Non-repeating test data (splitmix64 of the offset; the old patterns repeated every 512 KiB / 2 MiB, so misplaced data compared equal); device reads checked on another stream with the destination's stream held; a test-only delay on the staging copies (`detail::set_staging_copy_delay_for_testing`) so a scratch slot reused early, a read settled before its copies landed, or a chunk written before its D2H landed fails deterministically (mutants verified to fail); head-of-line tests (one reactor, stream A held 2 s: a read on idle stream B, a host read and a host write finish within 1.5 s while A's slice waits); shutdown cancels a parked read in well under the hold time; shared-registration and deregistration checks; `next_piece` unit-tested through `cufile/detail.hpp`, which also holds the test seams.
- Ready parked slices are served in portions of about `max_op_bytes` per poll; parked slices count against `max_inflight`, so slices parked behind busy streams can fill the window. The registration table also records unregistered cuFile transfers while `register_min_bytes` is set, so a later registration never covers memory one of them is using. `copy_stream` rejects an unresolved device id.
- Smaller: `got > want` from cuFile is an I/O error; a cause-less cuFile failure is described as such; the cuFile handle is deregistered if wrapping the object throws; read-only fds are `O_CLOEXEC`; stale header docs fixed; the raw probes no longer assert from worker threads.

Re-measured after the fixes (compat, same host and method, one run, GB/s): read 1 slice 1 GiB 6.24, `device_readv` 16 MiB 6.16, 256 KiB x 8 streams 3.52, 64 KiB x 8 streams 1.17; write fresh 1 slice 1 GiB 5.35 (256 MiB 5.24, 8 MiB x 8 streams 5.02), overwrite 1 slice 1 GiB 5.66. No regression against sections 5 and 6.

**Known gap (MINOR-4).** Device reads below `gds_threshold` (and every device read of a file without a cuFile handle) take the staged POSIX path, which is synchronous per reactor: one `pread` of the slice, its H2D copy, then the next slice. With 4 reactors that is a queue depth of 4, so 64 KiB reads reach about 1.2-1.3 GB/s where the uring reactor, which keeps up to 64 operations in flight per reactor, reaches about 6 GB/s. Closing it needs asynchronous POSIX reads inside the cuFile reactor (io_uring-style), which is a redesign; callers with many small device reads should prefer the uring backend, and the DISK tier only routes to cuFile on real-GDS hosts.

## 8. Host transfers of the DISK tier (W3 review MAJOR-2)

On a GDS host the DISK tier's automatic context is the cuFile one, and it also serves HOST <-> DISK: one vectored request per table, with segments of up to 1 MiB (the pinned pool's blocks; column buffers start 8 B-aligned inside them, so file offsets and sources are rarely congruent). The cuFile reactor served those with buffered `pread` / `pwrite`, one segment at a time on each of the two reactors a request reached.

What limits a synchronous writer here (Python probe on the root NVMe, 512 MiB, fresh file preallocated like the DISK tier's, GB/s): O_DIRECT 1 MiB writes 4.34 from one thread but about 3.0 from 2-4 threads (extending writes serialize on the inode); a small buffered write next to each 1 MiB O_DIRECT write drops that to 1.6-2.2; buffered 1 MiB writes alone 3.2-3.3, but 5.3-5.4 with `sync_file_range(SYNC_FILE_RANGE_WRITE)` after each, which starts writeback at once so the kernel's copy of the next segment overlaps the device write of the previous ones.

Changes (cufile reactor and `cufile_ioctx` only):
- **Host-only reads go O_DIRECT**: straight into the destination for a block-aligned range and pointer, else whole blocks through the pinned scratch (8 MiB slots), copied out; a runtime O_DIRECT refusal falls back to the page cache. A vectored read with more than two slices is spread over every reactor (each reactor serves one transfer at a time).
- **Host writes**: a body of at least 4 MiB whose source lines up with the file still goes O_DIRECT from the caller (head and tail buffered); everything else is written through the page cache with its writeback started immediately (`sync_file_range`). A host-only vectored write goes to one reactor. Durability is unchanged (`flush` is `fdatasync`); written pages are clean soon after and reclaimable, but unlike the uring reactor's O_DIRECT writes they do pass through the page cache.

`benchmark_disk_converter`, 4 columns, single-benchmark processes alternated (base = `9e7ad25` via `LD_PRELOAD`, new = this change, uring = disk 0), `CUCASCADE_BENCH_DISK_DIR=/tmp`, GiB/s:

| row | cufile base | cufile new | uring |
|---|---|---|---|
| HOST -> DISK 512 MiB | 2.57-2.88 | 4.13-4.22 | 4.26-4.40 |
| HOST -> DISK 64 MiB | 2.20-2.47 | 3.00-3.05 | 3.31-3.39 |
| DISK -> HOST 512 MiB (cold) | 1.47-1.82 | 6.35-6.39 | 6.22-6.29 |
| DISK -> HOST 64 MiB (cold) | 1.49-1.73 | 5.52-5.67 | 5.22-5.59 |
| GPU -> DISK 512 MiB | 4.70-4.88 | 4.63 | 4.53-4.80 |
| DISK -> GPU 512 MiB | 6.43-6.44 | 6.37-6.47 | 6.32-6.45 |

The review's 1.40-1.46 GiB/s for cuFile HOST -> DISK was taken while the SSD's write cache was exhausted (every writer drops to about 1.4 then); the base figures above are from a recovered drive. cuFile HOST <-> DISK is now within 10 % of uring (HOST -> DISK 64 MiB at 91 %, the rest at 95 % or above); the device rows are unchanged.
