# Recommendations: improving the cuFile/GDS reactor and the TPC-H scan path

Concrete, evidence-linked suggestions for `cucascade::io::cufile` (the reactor ported from the Sirius `gds` work) and for the read path around it. They combine NVIDIA's GDS documentation, other systems' field experience, and the measured TPC-H read patterns.

- Research behind this document: [gds-cufile-best-practices.md](gds-cufile-best-practices.md) (NVIDIA docs), [gds-field-experience.md](gds-field-experience.md) (other systems, papers), [tpch-io-patterns.md](tpch-io-patterns.md) and its two data documents (what the workload reads).
- Written 2026-10-15. **Nothing here has been measured on a working GDS stack.** Both known hosts run cuFile in compatibility mode (no `nvidia-fs`), so every performance statement is either NVIDIA's, someone else's, or an inference. Each recommendation says what it rests on and how to validate it.

Evidence tags: **[DOC]** NVIDIA documentation (key quotes re-verified verbatim, see the addendum of the best-practices document); **[DATA]** our GB300 measurements/traces; **[TEST]** something I ran in this repository; **[FIELD]** other systems' published results; **[INFERENCE]** reasoning, to be validated.

## 0. Where we stand

What exists (branch `gds`): a single-thread-per-reactor cuFile reactor (`include/cucascade/io/cufile/`, `src/io/cufile/`) that serves large contiguous device reads with `cuFileReadAsync` on the caller's CUDA stream and everything else with buffered `pread` (plus an H2D copy for device destinations); a pool/ioctx (`cufile_ioctx`); registry routing (`io_backend::cufile`); 23 test cases in the cuFile and registry groups (compat-mode host; the real `cuFileReadAsync` path is exercised). Two defects were found and fixed while writing this document:

1. **libcufile aborts the process** (`cufile_async.h putStreamOp: Assertion tmp == sinfo`) when two reactor threads call `cuFileReadAsync` on the same stream; one multi-slice device read triggers it. **[TEST]** 12/12 aborts before the fix, 0/12 after serialising submissions per stream. NVIDIA documents only "All APIs are thread-safe"; no source defines concurrent same-stream submission, so treat it as unsupported **[DOC]**. The lock is a workaround; R8 removes the need for it.
2. **The prefetching cache would have been armed for this backend and then failed device reads.** `ioctx::can_use_fs_cache()` is `supports_vector_host_read() || supports_host_to_device_read()` and the reactor advertised the latter, but with the cache on a missing device read carries a fragmented cache-chunk host buffer that the reactor rejects. **[TEST]** failing regression test, fixed by advertising `supports_host_to_device_read = false`.

What the workload needs (from [tpch-io-patterns.md](tpch-io-patterns.md)): requests are mostly 1-8 MiB (87.8% of bytes in requests of at least 1 MiB on SF1000; 30% of requests after merging are still below 1 MiB); nothing is aligned; the suite is I/O-bound at ~88% of the 20.8 GB/s array ceiling already; it reads each unique byte ~5.7 times.

**Success criteria** (proposed): on the SF1000 trace replay, in strict GDS mode, at least 80% of raw array throughput (>= 16.6 GB/s of 20.8 GB/s, the project target) **and** lower CPU per GB than the current staged device path (27.8 ms/GB, versus 12.1 ms/GB for the host path **[DATA]**). Throughput alone cannot be the win: the drives cap every path at ~20.8 GB/s.

## 1. Summary

| ID | Recommendation | Basis | Needs GDS hardware | Effort |
|---|---|---|---|---|
| R0 | Bring GDS up on a host and *prove* it is not compat mode (checklist in section 3) | [DOC] [DATA] | yes (to finish) | admin |
| R1 | Probe the effective mode at startup; add `require_gds`; route to uring when degraded | [DOC] [INFERENCE] | no | small-medium |
| R2 | Cap in-flight work well below cuFile's queue depth (current defaults sit at the documented failure point) | [FIELD] [DATA] | no | trivial |
| R3 | Align destination device pointers (4 KiB, preferably 64 KiB) and widen spans | [DOC] [TEST] | validate yes | small |
| R4 | Replace the fallback path: O_DIRECT + pinned pool, or hand it to uring (hybrid ioctx) | [DATA] [DOC] | no | medium |
| R5 | Configure cuFile explicitly (threads, queue depth, chunk size, slabs) before driver open | [DOC] | validate yes | small |
| R6 | Drop the kvikIO dependency: own `dlopen` wrapper over `cufile.h` | [FIELD] [DOC] | no | medium |
| R7 | Observability: path/alignment counters, gauges, optional request trace | [DATA] | no | small-medium |
| R8 | Try the synchronous thread-pool mode and lane streams, then pick by measurement | [DOC] [FIELD] | yes | medium |
| R9 | Shape requests: threshold per mode, merge policy, split requests above 16 MiB | [DATA] [DOC] | validate yes | small |
| R10 | Do not register buffers per transfer; register only long-lived staging memory if measured necessary | [DOC] | yes | small |
| R11 | Remove duplicate scans inside one query (Q17, Q18, Q21) | [DATA] | no | engine |
| R12 | Evaluate a cross-query cache sized for the ~166 GB working set | [DATA] | no | engine |
| R13 | Prefer larger row groups where files are under our control | [FIELD] [DATA] | no | data prep |

Do R2, R1, R7 first (small, no hardware needed), then R0 in parallel with R3-R6, then the A/B experiments (section 7). R11 and R12 are outside the reactor but, because the suite is I/O-bound, they are worth more than any API choice: removing duplicate reads saves up to ~4.4 s of ~52 s, a full cache up to ~37 s (upper bounds, see [tpch-io-patterns.md](tpch-io-patterns.md) section 7).

## 2. What the evidence says, in short

1. **GDS cannot beat the drives.** When a single device or array is the bottleneck, GDS and the CPU path perform the same (NVIDIA's own P2PDMA test: 3.23 vs 3.19 GiB/s) **[DOC]**; third-party studies agree (KIOXIA, GWDG, fastsafetensors: 26.4 vs 25.8 GB/s on four 7 GB/s drives) **[FIELD]**. GDS helps CPU load, page-cache footprint, and hosts where host memory or PCIe root bandwidth binds. On Grace the NVLink-C2C fabric narrows the gap further (one study: 135 vs 57 GiB/s pinned H2D, GH200 vs H200-PCIe) and **no published GDS measurement exists for GB200/GB300** **[FIELD]**.
2. **The stream API is the least-used path.** NVIDIA's own table lists "Higher execution latency for IO size (<1 MB)" and "Needs multiple streams to submit in parallel"; operations on a stream are FIFO ordered (which implies at most one cuFile IO in flight per stream, an inference from NVIDIA's ordering statement); the work runs on cuFile's internal thread pool, default `execution.max_io_threads = 4` per GPU. NVIDIA's cuDF integration, NIXL, DALI, DeepSpeed and cuDF-Java all use synchronous `cuFileRead` from a thread pool or pooled batches instead **[DOC] [FIELD]**.
3. **Alignment decides direct DMA versus bounce.** All four of file offset, size, device pointer base and device pointer offset must be 4 KiB-aligned; otherwise cuFile silently uses internal GPU bounce buffers plus a copy, even if the buffer is registered **[DOC]**. Our destinations come from RMM (256-byte alignment) and parquet ranges are never aligned **[DATA]**.
4. **Compat mode is silent and common.** `allow_compat_mode` defaults to true; an ext4 mount without an explicit `data=ordered` option, an unloaded `nvidia-fs`, or an unsupported open flag all degrade to POSIX with correct results and no error **[DOC]**. The GB300 `/raid` mount shows exactly that symptom (section 3) **[DATA]**.
5. **Do not register one-shot destinations.** NVIDIA: registration cost "will not be amortized", it can exhaust BAR memory, and it belongs "up front and off the critical path" **[DOC]**. This contradicts the constraint in `AGENTS.md` that per-transfer registration is acceptable for large I/O; consider amending it (R10).
6. **No NVIDIA source gives a GDS-versus-POSIX size threshold for local NVMe.** The 1 MiB default in the reactor was copied from kvikIO and is unvalidated; NVIDIA's numbers point at "64K and above" for the synchronous pool, "<1 MB" being slow for the stream API, and a kernel regression for IO up to 128 KB on kernels 6.11 and later **[DOC]**.
7. **Parquet row-group size dominates GDS suitability.** DuckDB-default ~100 KB column chunks are "suboptimal"; about 4M-row row groups saturate one SSD under GDS **[FIELD]**. The SF1000 files (~1.06M rows) already produce 1-8 MiB merged requests **[DATA]**.

## 3. R0: bring-up and compat-avoidance checklist

Do this on the host where GDS is to be measured. The GB300 `/raid` host status comes from the remote host reports of 2026-10-15 **[DATA]**.

| # | Check | Requirement (source) | GB300 `/raid` host today |
|---|---|---|---|
| 1 | ext4 mounted with `data=ordered` **in the option string** | "must be explicitly part of the mount options so that the library can recognize it"; otherwise `cuFileHandleRegister` fails (TRB 16.7) | options are `rw,relatime,discard,stripe=384`; `cufile.log` says "mount option not found in mount table" and "running in compatible mode". `data=ordered` is ext4's default, so this is about making it visible; unmount and mount again (ext4 generally refuses to change the journaling mode on remount) |
| 2 | `nvidia-fs` loaded (NVIDIA **open** kernel module), or P2PDMA/C2C active | `gdscheck -p` must list `nvfs`, `p2pdma` or `c2c`, not only `compat` (TRB 3.2.2, 18.9) | not loaded; every driver reports `compat` |
| 3 | `cufile.json`: `use_pci_p2pdma` and `block.nvme.use_pci_p2pdma` true (and `block.raid.use_pci_p2pdma` for md) when using P2PDMA/C2C | TRB 16.4-16.6; "GDS P2PDMA for RAID devices is currently supported only on ARM platforms" (TRB 16.6, v1.19.1) | all false |
| 4 | md RAID: udev rules edit, RAID0 only, member drives with PCI affinity to the GPU | TRB 5.15, 17.5 ("only supports RAID 0"), 17.9; the older x86 P2PDMA document says RAID0 is unsupported, so check the version | md RAID0 over 3 NVMe; udev rules and NVMe-to-GPU affinity unknown: run `gdscheck.py -t`, `nvidia-smi topo -mp`, `lspci -tv` |
| 5 | Drives not integrity-capable; device names start with `nvme` | TRB 17.4, 16.12 | not checked |
| 6 | GPU compute mode default (not EXCLUSIVE_PROCESS); persistence mode on (Grace) | NVIDIA forum 295318; REL known issues | not checked |
| 7 | Driver/kernel pairing | nvidia-fs 2.25.7 or newer (2.25.6 regressed); kernels 6.17 and later have an SGL-drive issue in `nvidia-fs` mode; kernel 7.0 needs GDS 1.18 or newer for `nvidia-fs`; REL 10/11 | host kernel 7.0, libcufile **1.17.1** (CUDA 13.2); documentation here is v1.19.1 |
| 8 | ACS/IOMMU settings per the platform guide | BP 3.1; Grace guidance is inconsistent between BP 3.1 and CFG 4.2.1 | not checked |
| 9 | Test environment strictness | `allow_compat_mode=false` and `CUFILE_FORCE_COMPAT_MODE=false` so fallback is an error (TRB cheat sheet) | allow_compat_mode true |

**Prove it, with more than one signal** (none is sufficient alone):

- `gdscheck -p`: the mode line for NVMe.
- `cufile.log` at INFO: no "using cuFile posix read/write with compat mode enabled" at handle register; the `cufile IO mode: POSIX` line appears only at TRACE level (TRB 6.8).
- In-process counters: set the stats level to 3 and read `cuFileGetStatsL3` after a run: `n_posix_reads == 0`, `n_p2p_reads + n_nvfs_reads > 0`, `n_unaligned_reads` near 0 (API 5.6).
- **Do not rely on `/proc/driver/nvidia-fs/stats`** in P2PDMA or C2C mode: "Compatibility mode and P2PDMA I/O statistics are not tracked" (TRB 5.9).
- Black-box: CPU use drops, `nvidia-smi dmon -s putcm` shows `rxpci` at the data rate, throughput differs from the `gdsio -x 2` (CPU path) baseline.

Never compare a compat-mode number with the 80% goal; label such runs "compat".

## 4. Alignment: the rules and how to meet them

**Rules [DOC]** (Best Practices IO Pattern 4, quoted verbatim in the best-practices addendum):

- Direct DMA needs **four** values 4 KiB-aligned: file offset, size, `devPtr_base`, `devPtr_offset` (the pointer actually used is base plus offset). Any miss sends the IO through cuFile's internal GPU bounce buffers (a DMA into a bounce slab, then a device-to-device copy), and registering the buffer does not help.
- GPU BAR mappings are 64 KiB-granular (CUDA GPUDirect RDMA guide), so 64 KiB alignment is the safe target for registered ranges and preferable for destinations; NVIDIA forum reports register failures at 96 KiB and misclassified "mixed" IOs when the device offset is not 64 KiB-aligned (old versions, no NVIDIA reply).
- Unaligned reads still **work**; they are slower. Unaligned **writes** use read-modify-write through POSIX.
- The bounce path draws from a fixed pool of GPU slabs (default 128 x 1 MiB, 8 x 8 MiB, 4 x 16 MiB, 2 x 64 MiB in 1.17+, per GPU). If a 3 MiB read takes an 8 MiB slab, only ~8 such reads can be in flight **[INFERENCE: the slab-selection rule is not documented]**; at ~1.5 ms each that alone would cap throughput near 16 GB/s.
- The kernel side of O_DIRECT has its own alignment: query it with `statx(..., STATX_DIOALIGN)` (Linux 6.1 and later) rather than assuming 512 or 4096. Clamp reads to the file size; a read crossing EOF returns a short count.

**Where each value comes from in this code base:**

| Value | Today | To do |
|---|---|---|
| File offset and size | parquet chunk offsets are unaligned (0.025% aligned) **[DATA]**; `ioctx::align_and_coalesce` / `min_alignment_requirement()` exist and are used inside `fs_cache`, but nothing in the cuFile path calls them; the scan layer (Sirius) is responsible | widen each merged span to 4 KiB; keep the logical window as an offset into the span |
| Destination pointer | `datasource::device_read` (`src/cudf/datasource.cpp:146`) allocates `rmm::device_buffer buf(size, stream)` at the **exact logical size**; RMM aligns to 256 B | allocate **span-sized** buffers (widened length) from an aligned adaptor (`rmm::mr::aligned_resource_adaptor`, 64 KiB), hand out column-chunk pointers inside the span |

**Cheap enforcement in the reactor (R7):** count, per slice, whether offset, size and destination are each 4 KiB-aligned, export the counters, and consider routing fully unaligned slices to the fallback staged path instead of cuFile's bounce, since the data takes a staging hop either way.

## 5. Recommendations

### R0. Bring GDS up and prove it
See section 3. This is the gate for every GDS performance claim. Also decide the cuFile version: both known hosts have 1.17.1; the 1.18/1.19 notes mention Grace multithreading improvements, removal of the batch-size-versus-bounce-buffer limit, and RAID1/RAID10 on Grace, but later releases may need a newer CUDA toolkit.

### R1. Probe the mode; fail loudly or route around
- **What:** at first use, record the effective mode (driver properties, P2P flags from `cuFileDriverGetProperties` / `cuFileDriverGetP2PFlags`), log it once, and add `cufile::config::require_gds` (default false). With it set, an unregisterable file or a compat result is an error. Without it, when compat is detected the registry should send device reads to the uring reactors, which already reach ~20 GB/s with staged pinned memory **[DATA]**, instead of running cuFile's compat path.
- **Why:** compat mode with unregistered buffers "does buffer allocation and deallocation in every `cuFileRead`" and is not performance-representative (TRB 18.12.4); our own tests could not tell the paths apart. A silent fallback also makes benchmark numbers meaningless.
- **How:** the reactor already knows whether `cufile_registered()`; add the driver-property query and an optional canary (register a handle on a file in the target directory, read 1 MiB, check stats counters). Make `create_io_object` throw for compat mode OFF (already implemented) and surface the AUTO-mode degradations as gauges.
- **Validate:** unit tests with a fake cuFile table (R6) for the compat branches; on a GDS host, assert the counters.

### R2. Cap in-flight work
- **What:** lower the defaults (`io_config::cufile_n_reactors` 4 and `cufile::config::max_inflight` 32 give **128 outstanding stream operations per process**) to about 16-32 in total, and enforce a process-wide budget (shared counter in the reactor context) in operations and bytes (~64 MiB).
- **Why:** an NVIDIA GitHub issue reports `cuFileReadAsync` returning success while the result becomes `-5` once ~134 async operations are outstanding ("133 is the golden value to pass, 134 will fail"), close to `execution.max_io_queue_depth` = 128; unresolved **[FIELD]**. 128 is exactly at that edge. Meanwhile the array saturates at modest depth: `fio` random 1 MiB reaches 19.8 of 20.8 GB/s at queue depth 16 and gpuio saturates at 16 per stream **[DATA]**; by Little's law ~16-32 MiB in flight suffices.
- **Validate:** a stress test (16, 32, 64, 128, 256 outstanding) on compat and GDS hosts to find where failures begin; confirm throughput flat above ~16.
- **Note:** several reactors only help the synchronous fallback (it blocks the reactor thread); the submission itself is cheap (~21 us per `cuFileReadAsync` on the host side per an NVIDIA forum trace).

### R3. Align destinations and span-plan the reads
- **What:** (a) allocate device destinations 64 KiB-aligned from an aligned memory-resource adaptor; (b) widen every merged span to 4 KiB and read it once into a span-sized buffer, with chunk pointers into it (cuDF does the same grouping, one device buffer per merged group); (c) pad allocation sizes to 4 KiB.
- **Why / evidence:** section 4. Over-read cost is small (+0.8% bytes on the local layout; +0.05 GB on all of SF1000).
- **Where:** `src/cudf/datasource.cpp:142-150` for the bridge; the same pattern applies in the Sirius scan layer. Add `ioctx::min_destination_alignment()` (new virtual, default 1; cufile returns 4096) so callers do not hard-code it.
- **Validate:** read the same ranges with aligned and unaligned destinations on a GDS host and compare throughput and `n_unaligned_reads`.

### R4. Fix the fallback path (or hand it to uring)
- **What:** today the fallback does a **buffered** `pread` (page cache) and, for device-only slices, a `cudaMallocHost` plus `cudaFreeHost` per operation, plus a per-op CUDA event. On SF1000, 30% of requests (4% of bytes) after gap-0 merging, and 47% of raw requests, fall below the 1 MiB cut-off and use it **[DATA]**.
- **Why it matters:** `cudaMallocHost` is slow and implicitly synchronising (the Theseus authors note large page-locked allocations "are slow to allocate" and use a fixed pinned pool); NVIDIA and the Linux manual both advise against mixing O_DIRECT and buffered I/O on the same file; the uring reactor already has an O_DIRECT, pinned-staged, H2D-copy path that reaches ~20 GB/s **[DATA]**.
- **Options:** (1) *Small:* give the cuFile reactor context the HOST-tier pinned memory resource (as uring has), read with O_DIRECT into aligned pinned blocks, pool the CUDA events. (2) *Better:* a **hybrid ioctx** that owns a uring pool and a cuFile pool and routes per slice: large, aligned device reads to cuFile; everything else (host, small, unaligned, fragmented cache fills) to uring. It also restores `supports_vector_host_read` and the prefetching cache for host-bound traffic, removes the fallback code from the cuFile reactor, and reuses the tested uring path.
- **Validate:** replay the trace with the threshold at 1 MiB and compare the fallback cost per request before and after.

### R5. Configure cuFile explicitly
cuFile reads `/etc/cufile.json`, which does not exist on the development host and whose built-in defaults are unknown. Set parameters before `cuFileDriverOpen` (programmatic `cuFileSetParameter*`, or an application JSON selected with `CUFILE_ENV_PATH_JSON`; parameters set in code take precedence). Starting values to **sweep, not assume** (defaults and meanings from the NVIDIA configuration guide):

| Key | Default | Suggested start | Why |
|---|---|---|---|
| `execution.parallel_io` | true | true | required for the stream APIs |
| `execution.max_io_threads` | 4 per GPU | sweep 4, 8, 16, 32 | all async stream work runs on this pool; with more lanes than threads, operations queue inside cuFile |
| `execution.max_io_queue_depth` | 128 | at least 2x the in-flight cap | see R2 |
| `execution.min_io_threshold_size_kb` / `max_request_parallelism` | 8192 / 4 | keep | a single request below 8 MiB is not split |
| `properties.max_direct_io_size_kb` | 16384 historically; "derived from the slabs" in 1.17+ (a GH200 `gdscheck` shows 1024) | **16384** explicitly | release notes: set it to retain 16 MiB chunks; NVIDIA: larger values "result in a reduced number of calls to the IO stack"; NIXL pins 16384 |
| `properties.gpu_bounce_buffer_slab_config` (1.17+) | sizes 1/8/16/64 MiB, counts 128/8/4/2 | raise the 4-16 MiB slab counts to at least the in-flight cap | bounce concurrency (section 4); no registration needed |
| `properties.max_device_pinned_mem_size_kb` | 32 GiB | keep | cap on cache plus registered buffers |
| `properties.allow_compat_mode` | true | **false in tests/CI**, true in production | make fallback loud |
| `properties.io_priority` | default | try "high"/"med" if cuDF kernels share the GPU | internal copies compete with compute |
| `profile.cufile_stats` | 0 | 3 in tests only | counters for R0/R7; higher levels cost performance |

### R6. Drop the kvikIO dependency
- **What:** replace `kvikio::cuFileAPI` and `kvikio::CompatMode` with a small wrapper we own: lazy `dlopen("libcufile.so.0")`, per-symbol `dlsym` where optional symbols (async, stream, batch, parameter get/set, `cuFileDriverClose_v2`) mean "feature unavailable", not total failure; `cuFileSetParameter*` then one `cuFileDriverOpen`; reference-counted close; handle cache; context discipline. The field research lists what to reproduce and what the shim hides (section 12.11 of [gds-field-experience.md](gds-field-experience.md)).
- **Why:** project guidance is that kvikIO is not a high-quality basis for GDS, and the shim hides exactly the settings NVIDIA says matter (stream registration flags, the parameter API, buffer registration, thresholds). Its all-or-nothing symbol loading and static driver lifecycle have produced field problems, and two parties opening and closing the process-wide driver (cuDF's kvikIO and our reactor) can be a hazard (inference). **It also makes the reactor cudf-free**, so it can live in the core `cucascade_io` build instead of the `CUCASCADE_BUILD_CUDF` layer.
- **Bonus:** a fake cuFile function table enables unit tests of the async path, error decoding and fallback decisions on machines without GDS.
- **Error decoding to build in:** cuFile errors arrive as `-1` with the code in `errno` or as a negative `CUfileOpError`; NVIDIA's own NIXL decodes both (`code = ret == -1 ? errno : -ret; if (IS_CUFILE_ERR(code)) CUFILE_ERRSTR(code) else strerror(code)`).

### R7. Observability
- **Gauges** (like `uring_reactor::take_gauges`): operations in flight, peak, bytes submitted per path (cuFile versus fallback), queued bytes, fallback reasons (below threshold, not registered, submit refused, unaligned, fragmented), queue delay per priority tier, short reads, registration failures.
- **Counters from cuFile:** a `cuFileGetStatsL3` snapshot (posix/p2p/nvfs reads, unaligned reads, registered bytes) next to the reactor's own.
- **Effective mode** logged once at startup (R1).
- **Optional request trace** (environment-enabled): timestamp, reactor, thread, stream, destination kind, offset, size, priority, path, completion latency. This fills the unknowns listed in [tpch-io-patterns.md](tpch-io-patterns.md) section 9 (concurrency, per-stream reuse, latency).

### R8. Submission mode: measure three, then choose
Today every cuFile read is `cuFileReadAsync` on the **caller's** stream.

- **Option A: synchronous pool (NVIDIA's own pattern).** Worker threads (8-32; cuDF's study found 8-32 threads and 1-16 MiB slices best, NIXL defaults to half the hardware threads) call blocking `cuFileRead` on a shared per-file handle, looping on short reads. No stream, no cuFile async thread pool, no `-5`-over-133 issue. Ordering: cuFile is not ordered with the caller's stream, so before reading, make the destination safe (record an event on the caller's stream and synchronise it, normally already complete) so a freshly allocated or cleared buffer or stream-ordered allocation (`cudaMallocAsync`) cannot race with the read.
- **Option B: lane streams.** Keep `cuFileReadAsync` but run reads on reactor-owned **lane streams**, one owner thread per lane, registered once with `cuFileStreamRegister(lane, 0xF)` when offsets and sizes are aligned and known (NVIDIA: "will perform best"). Order against the caller by recording an event on the caller's stream and making the lane wait on it before the read; completion is host-visible through the lane event, as today. Concurrency becomes the number of lanes instead of the number of distinct caller streams, and **no stream is ever shared between threads**, which removes the abort class and the global mutex. Raise `execution.max_io_threads` to at least the lane count (R5).
- **Option C: batch API.** Aimed at many small (<64 KB) non-contiguous IOs; our requests are not that size. Evaluate only if the fallback share stays large.
- **Decision rule:** replay the SF1000 trace in strict GDS mode with each option at equal in-flight bytes; choose the one that meets the 80% target with the lowest CPU ms/GB. NVIDIA's documentation and its own libraries favour A; the stream API's per-op overhead (about 70 us over the synchronous call at 4 KiB in an NVIDIA forum trace) is small next to a 0.5-2.5 ms transfer of 4-16 MiB, so B is not ruled out. Neither has been measured here.
- **Also:** a reusable pool of CUDA events instead of one per operation; optionally `cudaLaunchHostFunc` completion instead of polling every 200 us (polling adds up to 200 us of latency per read, which only matters at low queue depth).

### R9. Shape requests
- **Threshold:** make it a per-mode configuration instead of the kvikIO-derived 1 MiB. Starting points to sweep: synchronous pool 256 KiB (NVIDIA: pool is "better ... 64K and above"; kernels 6.11 and later regress IO up to 128 KB); stream API 1 MiB. Record the request-size distribution so the cut can be chosen from data. On SF1000, a 256 KiB cut would send only 2.8% of requests to the fallback versus 30% at 1 MiB **[DATA]**; on DuckDB-default layouts (91 KiB median chunk) the cut decides almost everything, and Q1's runs of 1.044 MiB sit 4.4% above 1 MiB.
- **Merge:** always merge touching ranges (gap 0: -47% requests, 0% bytes). A gap up to 1 MiB removes ~18% more requests for +4% bytes on SF1000; test 256 KiB-1 MiB on GDS hardware. Never use an 8 MiB gap.
- **Split:** break requests above 16 MiB into <= 16 MiB sub-reads. The trace has single requests up to 28 MiB (orders) and whole-table reads of dimension tables (hundreds of MiB), and the reactor currently issues one cuFile operation per slice whatever its size, occupying slabs and in-flight budget and leaving parallelism on the table.
- **Alignment of splits:** split at 4 KiB-aligned boundaries (R3).

### R10. Registration policy
- **Do not** register per transfer. NVIDIA: for a buffer filled once, "the registration cost will not be amortized", it adds latency and competes for BAR memory; "Read or write the data without buffer registration." Registration takes up to milliseconds and is "expensive"; NVIDIA wants it "up front and off the critical path".
- **Do not** register RMM pool memory: freeing registered memory without `cuFileBufDeregister` can deadlock `nvidia-fs`; BAR1 and the 32 GiB pinned cap limit it; `cudaMallocAsync`-backed memory is not documented as registrable.
- **If** measurement shows the bounce path is the bottleneck (counters from R7 plus an aligned-versus-registered A/B), copy DALI's approach: a pool of long-lived registered staging chunks (for example 16 MiB), aligned reads into them, then a stream-ordered device-to-device copy into the destination.
- **Amend `AGENTS.md`**: the constraint "CAN register temporarily per-transfer for large I/O" conflicts with NVIDIA's guidance; consider rewording it as "per-transfer registration is allowed but not recommended; prefer aligned unregistered reads or a registered staging pool".

### R11. Remove duplicate scans inside one query (engine level)
Q21 reads 98.3 GB of lineitem but 38.8 GB are unique; Q17 71.7 vs 45.0; Q18 12.8 vs 6.4; in total 92.6 GB (11.6% of lineitem, ~4.4 s of raw disk time per suite run). The duplicates come from self-joins that scan the same table more than once. Sharing the scan (or the decoded/raw column chunks) removes them regardless of backend. **[DATA]**

### R12. Evaluate a cross-query cache (engine level)
948 GB requested versus ~166 GB unique over the 22 queries: 82% of bytes were already requested by an earlier query or earlier in the same query. A cache that retained the union would bring the suite's I/O floor from ~45 s to ~8 s (upper bound, in-process back-to-back runs, compute then dominating). cuCascade's `fs_cache` keeps pinned **host** chunks, which a GDS design bypasses on purpose; the natural fit is a GPU- or host-resident tier fed by GDS reads. This competes with working memory (166 GB is ~60% of the GB300's HBM), so size it from the per-query working sets. **[DATA]**

### R13. File layout (when we control the data)
Larger row groups (millions of rows) raise column-chunk size into the 4-16 MiB range all direct cuFile users converge on, and a study of TPC-H lineitem scans found ~4M-row row groups needed for GDS to saturate an SSD; the SF1000 files (~1.06M rows) are already in the useful range. Clustering by the filtered date column would let row-group skipping cut Q6/Q14/Q15 to 2.5-15% of their bytes (estimate). **[FIELD] [DATA]**

## 6. Do not do this

- Per-transfer buffer registration (R10), or registering pool-managed memory.
- Sharing a CUDA stream between threads submitting `cuFileReadAsync` (R8), or deregistering/destroying a stream, handle or buffer while operations are in flight.
- Treating the 1 MiB threshold, kvikIO's defaults (1 thread, 4 MiB tasks, 16 KiB threshold) or kvikIO's benchmarks as evidence about cuFile.
- Comparing compat-mode throughput with the 80% target, or benchmarking without proof of the effective mode (section 3).
- Mixing buffered and O_DIRECT reads of the same file as the long-term fallback design.
- Trusting `gdsio -x 0` numbers that exceed the device (an NVIDIA staff reply confirmed a bug in one report); cross-check with device counters.

## 7. Experiment plan (GDS-capable host)

Order matters; stop and re-plan if an early result contradicts the assumption behind a later one. Record GDS/driver/kernel versions, mount options and topology with every result; run strict mode with proof of mode (section 3); drop the page cache or use O_DIRECT; use `/raid` (or `/mnt/disk_2` where that is the target) and the `fio` raw baseline of 20.8 GB/s.

| # | Experiment | Measure | Decides |
|---|---|---|---|
| E0 | Bring-up and proof (R0) | mode, counters | everything below |
| E1 | Baselines: `fio` raw, `gdsio -x 0/2/5/6` at 1/4/16 MiB x 4/8/16/32 threads, with and without `-b` (unregistered) | GB/s, CPU | reference points; whether cuFile reaches the ceiling at all |
| E2 | Same-stream repro, 2 threads, with/without mutex, with/without `cuFileStreamRegister` | abort or not | whether the lock suffices; file a minimal repro with NVIDIA |
| E3 | SF1000 trace replay: current uring path, cuFile sync pool (R8-A), lanes (R8-B), batch (R8-C) at equal in-flight bytes | GB/s, CPU ms/GB, pinned memory, p50/p99 latency | submission mode |
| E4 | Destination alignment A/B (256 B versus 4 KiB versus 64 KiB), file alignment on/off | GB/s, `n_unaligned_reads`, slab pressure | R3 payoff; slab counts |
| E5 | In-flight sweep (8-256 outstanding); `max_io_threads` 4-32; `max_io_queue_depth` 128/512 | GB/s, failure point | R2, R5 |
| E6 | Threshold sweep 64 KiB-4 MiB on the real request mix, on the production kernel | GB/s, CPU | R9 |
| E7 | `max_direct_io_size_kb` default versus 16384; slab counts | GB/s, syscall count | R5 |
| E8 | Registered staging pool versus unregistered (only if E4/E7 show bounce-limited) | GB/s, registration cost | R10 |
| E9 | Registration support matrix: `cudaMalloc`, `cudaMallocAsync`/RMM async, `cuMemMap`, sub-ranges; BAR1 size on GB300 | errors, size limits | whether pools can ever be registered |
| E10 | GB300-specific: does `gdscheck` list `c2c`; RAID0 over NVMe on Grace; `cuFileGetBARSizeInKB` | mode, limits | whether GDS applies to this host at all |

Open questions that need a GDS host and are not resolved by any source we found: GB300/Grace behaviour end to end, `cudaMallocAsync` registration, per-handle registration cost and handle limits, whether stream flag 0xF changes pointer-lifetime rules, and the exact slab-selection rule. The full list (Q1-Q18) is in [gds-cufile-best-practices.md](gds-cufile-best-practices.md) section 10.

## 8. Known limitations of the current reactor

1. Uses the stream API (queue depth 1 per stream; internal pool of 4 threads by default); not validated against a synchronous pool.
2. Reads are probably "unaligned" by NVIDIA's definition (RMM destinations) and therefore bounce; no counters tell us.
3. The per-stream submit lock is a workaround for a libcufile abort, not a design.
4. The fallback path uses buffered `pread` and `cudaMallocHost` per operation (R4).
5. No prefetching-cache support (the flags are false on purpose); cache-heavy workloads must use uring.
6. Depends on kvikIO and the cudf-coupled build (R6); the cuFile `CompatMode` knob is a kvikIO type.
7. One cuFile operation per slice regardless of size; no splitting above 16 MiB.
8. A cuFile handle per opened object; no process-wide handle cache keyed by (device, inode).
9. The `bytes_done` result slots are ordinary heap memory; NVIDIA's documentation suggests pinned host memory.
10. Defaults (4 reactors x 32 in flight = 128 operations; 1 MiB threshold) are unmeasured and the first sits at a documented failure edge (R2).
11. Only exercised in compat mode so far; no true-GDS result exists.

## 9. Applying this to the disk-tier backend (writes)

The same cuFile rules apply to the spill/disk tier: unaligned writes use read-modify-write through POSIX, so write aligned sizes at aligned offsets with O_DIRECT; `fs.generic.posix_unaligned_writes` forces POSIX for unaligned writes (into the page cache); `gds_rdma_write_support = false` forces compat for writes; durability needs an explicit `fsync` (cuFile does not flush metadata); buffered writes followed by O_DIRECT reads of the same range make the kernel write back dirty pages first, so write spill files with O_DIRECT as well. Measure against `dd oflag=direct` / `gdsio -I 1` baselines on `/mnt/disk_2` per the project constraint.

## 10. Corrections made while writing this document

- Earlier in the session I described NVIDIA's RAID support as "supported in `nvidia-fs` mode but not in P2PDMA mode". The current troubleshooting guide says "GDS P2PDMA for RAID devices is currently supported only on ARM platforms; x86 platforms are not supported" (TRB 16.6) and "Currently, GDS only supports RAID 0" (17.5), while an older x86-oriented P2PDMA document says RAID0 is unsupported. md RAID0 on the GB300 host is therefore plausibly supported, but only with the right version and configuration (section 3, item 4).
- The `repeat_cross_query_bytes` statistic in the Sirius trace summary counts re-reads inside one query too; the corrected figures are in [tpch-sf1000-gb300-trace.md](tpch-sf1000-gb300-trace.md) section 5.
