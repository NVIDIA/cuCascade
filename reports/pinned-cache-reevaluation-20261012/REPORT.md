# cuCascade PR #185 reevaluation — 2026-10-12

**Recommendation: retain the PR.** Current main still performs a pinned host allocation and free for every request larger than 8 KiB. The merged, hardened cache removes that repeated work and reduces the current Sirius hot TPC-H SF50 suite time by **1.60%** on this GB10. The original PR's **13.9%** application improvement was not reproduced under the current configuration.

Original PR: [NVIDIA/cuCascade #185](https://github.com/NVIDIA/cuCascade/pull/185), branch `fix/pinned-large-alloc-cache`, original head `5df1ea0e7ac78dea0713e3a580207cb0286b1740`. Fetched main: `bd4aeedbee1201df68bd042374fd9ab6afb5b46d`. Local merge commit: **`8b2d229`**, with both heads as parents. No remote publication was performed. The user explicitly authorized bypassing the repository's GSD workflow for this task.

## Static value and implementation

Main's recent #189 fixes improve stream ownership and teardown but do not cache large pinned allocations. The optimization therefore remains relevant. Its value depends on repeated size classes; a workload that barely allocates pinned memory should see little application benefit.

The implementation retains power-of-two size classes and a **256 MiB default retained-cache limit per resource**, enabled by default when this resource is used. FIFO eviction selects the oldest cached block across bucket fronts. One map of deques owns cached entries; the redundant eviction index was removed. Requests that cannot be rounded safely or retained within the limit use the direct upstream path. The limit bounds cached large blocks, excluding slabs, live allocations, and detached victims being drained.

Integration preserves main's owning-device event creation, device restoration, stream ordering, and teardown rules. Cleanup addresses the old PR's concrete defects:

- A failed `cudaStreamWaitEvent` propagates an allocation error before consuming the cached candidate.
- Metadata allocation failure cannot escape `noexcept` deallocation. Optional caching falls back to waiting and releasing the block safely.
- Cache-cap arithmetic uses remaining capacity rather than an overflowing addition.
- OOM purge detaches the existing map without allocating replacement metadata, releases retained blocks, then retries once.
- Victim synchronization and freeing occur outside the cache mutex. Slab expansion completes throwing metadata work before publishing pointers.
- Unknown CUDA completion prevents unsafe reuse or freeing; unrecoverable error paths can deliberately retain storage.

Asynchronous allocation establishes GPU stream dependencies. A caller must synchronize that stream before immediate CPU access; the resource's synchronous allocation API supplies this synchronization and cleans up if the wait fails. Installed cuDF host-vector allocation uses the asynchronous API and explicitly synchronizes. Multi-GPU behavior remains untested on this one-GPU machine.

The architect, developer, and reviewer completed their assigned repository and book readings before their work. Their reading lists and assessments are preserved in [ARCHITECTURE.md](ARCHITECTURE.md), [IMPLEMENTATION.md](IMPLEMENTATION.md), and [REVIEW.md](../../REVIEW.md).

## Current Sirius application measurement

| Hot 22-query SF50 suite | Main allocator | Merged cache |
|---|---:|---:|
| Median of four process suite totals | 13.577599 s | 13.360270 s |
| Reduction in elapsed time | | **217.330 ms / 1.60065%** |

The fixed schedule was three baseline noise processes followed by **B, C, C, B, C, B, B, C**. Each process ran every query four times, excluded the first iteration per query, and summed the 22 medians of the remaining three hot iterations. The reported figure is the median of four process totals per arm. Iterations are not treated as independent process repetitions.

Baseline noise totals were 13.454758, 13.511138, and 13.525760 s: range **71.002 ms / 0.52551%**, MAD 14.622 ms. The four adjacent balanced pairs improved by **1.55%, 1.38%, 1.27%, and 2.16%**. Every candidate total was below every paired baseline total. This supports a modest application benefit on this snapshot; it is not an individual-query statistical significance claim or a general speedup guarantee.

All **968 query executions** across the noise and comparison blocks matched CPU reference results; 704 were in the paired comparison, of which 528 hot iterations contributed to the figure and 176 first iterations were excluded. Every query retained one row count and result digest across all blocks. CPU fallback was disabled, and each process's fully flushed combined logs contained all 88 expected GPU starts, completions, and engine query entries without query errors. Result checking occurred after the timer stopped.

The current canonical Python harness rendered SF50 SQL over the existing unclustered 13.45 GB Parquet dataset. Pinning and pin compression were disabled. Configuration was GPU capacity 48 GB, host capacity 24 GB, 8 pipeline threads, 4 task-creation threads, 6 scan threads, 512 MiB scan batches, Sirius LRU cache, prefetch disabled, and Quent telemetry disabled. No cache reset occurred between measured iterations. The canonical initial file-specific `fadvise`/sync behavior was preserved.

The consumer source snapshot is Sirius `45253bf5074eb9462bbe6ef0c1dc081a252adf03`. It uses cuCascade dependency `e1b67fe7d44229597af562d46bb8383828d77981`, whose unrelated reservation-accounting changes were retained identically in both arms. Its allocator and fixed upstream source bytes match fetched main. Only the allocator header/source were overlaid; its translation unit and the constructing SiriusContext owner were rebuilt for each arm. All other 333 consumer archive members and 21 resource archive members remained byte-identical between arms. Current wrapper glue was rebuilt against the same clean DuckDB engine archives. This is a controlled allocator comparison on source-equivalent frozen consumer objects, not a claim of a fresh full consumer/engine rebuild. [Build provenance](sirius-build/README.md) records commands, dependency checks, and hashes.

The original PR omitted its exact historical configuration. This experiment establishes current value under the recorded configuration; it cannot certify a literal reproduction of the old 13.9% claim. Initial correctness smokes and an incomplete noise attempt were excluded. The latter exposed a shared harness date-result parsing bug; only the isolated validation adapter was corrected, then the complete prespecified schedule restarted. No engine or query implementation changed.

See the [consumer contract](sirius-benchmark/CONTRACT.md), [manifest](sirius-benchmark/MANIFEST.json), [results](sirius-benchmark/RESULTS.json), and [raw evidence archive](sirius-benchmark/raw-evidence.tar.gz).

A separate hot-Q18 Nsight run verifies the consumer mechanism: main issued **186 `cudaHostAlloc` and 186 `cudaFreeHost` calls**, while the cache issued **zero of both**, adding 186 reuse waits (901 → 1,087). Both windows contained 1,608 kernels and 3,166,494,775 copied bytes; copy-call counts were 2,389 and 2,383. All observed allocation/free/synchronization calls returned success. Event polling returned normal `cudaErrorNotReady` states. Exports also contain records outside the profiler markers, so attribution explicitly clips to the start/stop interval enclosing exactly one `sirius::query` range. Profiled times are excluded from the result. [Trace summary](sirius-benchmark/PROFILE-SUMMARY.json) and [analyzer](sirius-benchmark/analyze_profiles.py) preserve the boundaries and counts.

The reviewer identified a **preexisting issue in the external Sirius NUMA facade**: its synchronous allocation method forwards to asynchronous allocation without waiting. The core cuCascade synchronous method is correct. A separate observer kept that facade behavior intact and ran all 22 queries four times with CPU validation. It observed **84,466 asynchronous allocations and deallocations, and zero synchronous calls**. The measured path therefore uses cuDF's explicit stream synchronization; the unused facade API still requires correction before callers rely on it for immediate CPU access. This does not expand the cuCascade PR into external Sirius changes. [Observer patch/build](facade-probe/README.md), [runtime manifest](sirius-benchmark/FACADE-PROBE-MANIFEST.json), and [diagnostic evidence](sirius-benchmark/diagnostic-evidence.tar.gz) record this distinction.

## Standalone cuDF and allocator measurements

These separately prespecified measurements preceded the full consumer experiment and have different scopes.

| Warm cuDF Parquet read | Main allocator | Merged cache |
|---|---:|---:|
| Median of four process medians | 48.734650 ms | 48.822875 ms |

The 0.181% increase lies inside the 0.510% baseline noise range: **no demonstrated application gain or regression** for this small fixture. Every read matched 600,572 rows, 16 columns, and full-data digest `16342477538070118188`. Each read used 37 custom pinned allocations, only three above 8 KiB, with 71,199 total requested bytes. The fixture exercises the cache but has little allocation churn to remove.

Separate Nsight traces confirmed that each baseline read issued three host allocations and three frees; the candidate's first read allocated three blocks and subsequent reads issued neither operation. Both traces contained the same 272 GPU kernels and 136 copies across four reads. Runtime, kernel, copy, and NVTX records were present; unsupported Unified Memory tracing was not used. Profiled times were excluded from performance conclusions.

| Warm allocation/free cycle | Main µs/op | Cache µs/op |
|---|---:|---:|
| 16 KiB | 1.883210 | 0.410179 |
| 128 KiB | 1.948040 | 0.417090 |
| 1 MiB | 500.397000 | 0.420434 |
| 16 MiB | 5522.030000 | 0.427122 |

These cycles include allocation-stream synchronization and minimal CPU touches, with separate H2D content checks. They measure avoided allocation/registration work in a warm cache, not transfer throughput, full-buffer initialization, or query speedups. [Standalone results](RESULTS.json), [protocol](CONTRACT.md), and [trace summary](PROFILE-SUMMARY.json) preserve details.

## Correctness and machine

The machine is a single NVIDIA GB10, Linux aarch64, driver 580.126.09, CUDA 13.2.86, and installed cuDF/RMM 26.08. Existing environments were retained without dependency updates. GPU jobs were serialized with a shared lock and an idle gate. Release builds used warnings as errors.

| Validation | Outcome |
|---|---|
| Repository core, cuDF, and I/O suites | **442 cases passed**, 7,429 assertions; 2 multi-GPU cases skipped |
| Compute Sanitizer stream-ordering cases | 2 passed, 12 assertions, zero errors |
| Independent CUDA/host metadata failure injection | 19 failure-path cases passed |
| Destruction with genuinely pending DMA | Passed; storage released after completion |
| Scoped ThreadSanitizer stress | 8 threads × 1,000 large-cache cycles; zero race reports |

Failure injection compiles the exact production allocator translation unit against dynamic cudart because the shipped core's hidden static cudart symbols prevent direct interposition. Fault counters prove interception, including actual Portable/Mapped host allocation flags. Normal suites test the shipped candidate binary. The TSan driver and allocator are instrumented; CUDA/RMM/backing dependencies are not. A process-local ASLR disable was needed after the default TSan launch failed before testing with an address-mapping error. These scopes and the deliberate unknown-completion retention policy are documented in [the review](../../REVIEW.md).

Frozen production source SHA-256: `6c57980ba465f139eea75ed32c0fdb89ec84e7cc3464d797c083cfb87ec38e27`. Tested candidate core library: `3fff46450e43b1e6e6c4f3a2f171a8b634eac8c2aa37ee7eb030ef3e93b75845`. Main/candidate consumer extension hashes are recorded in the consumer manifest. The tests and benchmarks used the same frozen production code.

## Evidence and reproduction

This directory preserves protocols, commands, source-level diagnostic harnesses, JSONL results, test output, build metadata, and compressed raw consumer logs/results/SQL. Scripts record the absolute local dependency/data paths used in this experiment; they require the existing matched environments and frozen donor archives. Start with [INVENTORY.md](INVENTORY.md) and the consumer build README rather than swapping an ABI-incompatible complete library into an old extension.

The [dataset manifest](sirius-benchmark/DATASET-MANIFEST.json) records full-file SHA-256 identities after the experiment; query result identities were checked throughout every block. Canonical Python sources, both timing binaries, configuration, validation adapter, and production source hashes remained unchanged through final verification.

Native Nsight reports and SQLite exports are retained locally in `build/pr-reevaluation/evidence/`. Complete original working artifacts remain in `/tmp/cucascade-pinned-pr-20261012/`. Timing, profiling, and observer artifacts are distinct; profiler/observer elapsed times do not enter the application result.
