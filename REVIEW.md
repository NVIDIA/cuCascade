# Pinned large-allocation cache PR reevaluation

**Verdict:** no unresolved blocking correctness or design findings in the integrated allocator.
The cache remains worth retaining: it reduces the matched current Sirius TPC-H SF50 hot suite time
by **1.60%** on this machine, exceeding its 0.526% baseline noise range with all four balanced comparisons
improving. This is a current-configuration result, not reproduction of the historical 13.9% claim.
Repeated pinned allocation diagnostics also improve substantially. The separate small Parquet-read
workload shows no demonstrated speedup or regression.

The original PR head was `5df1ea0`; the comparison baseline is fetched `origin/main` `bd4aeed`.
Current main still allocates/frees every request above 8192 bytes directly, so the optimization is not
obsolete. Integration preserves main's #189 owning-device, ordering, and teardown fixes and remedies
the old cache implementation's confirmed defects. No publication or remote merge is implied.

## Assigned readings and review scope

Read before allocator review:

- `/home/kkristensen/.codex/agents/reviewer.toml`, repository `CLAUDE.md`, and current-main `AGENTS.md`.
- Broad tables of contents of all three assigned books.
- **C++ Core Guidelines:** complete Resource Management, Performance, Concurrency and Parallelism,
  and Error Handling chapters (local lines 9414–10254, 13843–14284, 14285–15904, 15905–16924).
- **C++ Software Design:** complete Guideline 4, Design for Testability, and Guideline 6, Expected
  Behavior of Abstractions (1888–2248 and 2641–3068).
- **C++20 — The Complete Guide:** complete chapter 19, Concurrency Features, and complete §20.8,
  Utilities to Deal with Bits (22819–24779 and 25418–25647).
- Original and current-main allocator headers/implementations, upstream allocation ownership,
  allocator tests, and installed cuDF host allocator synchronization behavior.
- For the additional consumer audit: frozen Sirius `AGENTS.md`, the isolated build/benchmark
  contracts, allocating owner and NUMA facade/resource view, canonical timer/result parser, and
  installed cuDF 26.08 host-vector/pinned-vector allocation dispatch.

Review covered ownership, asynchronous reuse, CPU access, device selection, metadata exceptions,
OOM retry, teardown, cache cap/rounding, documentation, and test adequacy. GPU runs were executed by
the root agent under its shared GPU lease; the reviewer independently inspected the code and logs,
recomputed the timing results, and supplied the external fault harness.

## Confirmed original defects and verified remedies

Original line references refer to `5df1ea0`. Each remedy was assessed separately from the defect.

| Defect | Required property and cheapest local remedy | Final evidence |
|---|---|---|
| P1: failed reuse waits were asserted/discarded, then returned unordered memory (`cpp:155`, `294`) | Never consume or return a candidate before its wait succeeds; propagate allocation error and leave it tracked | Wait precedes removal in final `cpp:179` and `317`; slab/large injected wait failures both retain the same pointer for retry |
| P1: failed stream/event synchronization still published or freed storage (`cpp:204`, `238`, `347`) | Unknown completion must prevent reuse/free; retain affected storage on unrecoverable CUDA errors | Final fallback and victim paths quarantine instead of releasing; creation/recording plus sync failures and eviction/purge/destructor failures pass injection checks |
| P1: event/cache container allocation could escape `noexcept` deallocation (`cpp:75`, `95`, `191`) | Metadata growth is optional; catch failure, keep accounting transactional, then drain before direct release | Final event recycling and insertion catch failures; three fresh-bucket allocation failures, recycling OOM, and combined record/recycling failures pass |
| P1 integration: old event creation followed current device and old slab teardown did not wait (`cpp:39`, `69`) | Preserve #189's stream owner and backing-storage lifetime guarantees using existing helpers | Final `device_of_stream`/event creation preserve device ownership; recorded work is drained before returning backing blocks; single-GPU ordering/teardown tests pass |
| P2: OOM purge allocated a vector before releasing retained storage (`cpp:330`) | Draining must not require new host metadata allocation; detach the existing container under lock | Final purge swaps the map and resets accounting; recovery still releases storage/retries while event recycling metadata allocation fails |
| P2: cap comparison added sizes before checking the bound (`cpp:182`) | Cap arithmetic must hold for every accepted `size_t` limit; compare against remaining capacity | Final `bucket <= limit - used` test preserves the invariant; bit-ceiling overflow guard and size boundary tests retained |

Allocator helpers and owning entries are private. Slab growth finishes potentially throwing metadata
allocation before publishing pointers. Synchronization and freeing detached victims occur outside
the allocator mutex. Synchronous allocation cleans up its pointer if its completion wait throws.
No additional resource framework or unrelated allocator refactor was needed.

## Correctness evidence

Final Release builds used warnings as errors. Root-run logs establish:

| Validation | Result |
|---|---|
| Focused pinned allocator suite | 992 assertions; 27 cases passed, 1 multi-GPU case skipped |
| Other core tests | 2333 assertions; 89 cases passed |
| cuDF tests | 2564 assertions; 187 cases passed, 1 multi-GPU case skipped |
| I/O tests | 1540 assertions; 139 cases passed |
| Compute Sanitizer memcheck on stream-ordering cases | 12 assertions; 2 cases passed; 0 errors |
| Independent CUDA/host metadata fault harness | 19 cases passed; each injected path checks its fault counter |
| Pending-DMA destruction harness | 1 additional case passed: destructor waits before freeing a blocked DMA source; copied content remains correct |
| Scoped ThreadSanitizer stress test | 8 threads × 1000 large-cache cycles passed; no TSan reports; 512 KiB cap, 114,688 bytes retained |

Failure injection compiles the exact final allocator translation unit into a separate executable
linked against shared cudart, with the candidate library supplying the backing allocator. This is
necessary because the shipped core statically links cudart and hides its CUDA symbols, preventing
`LD_PRELOAD` from intercepting those calls directly. `nm -D` confirms dynamic CUDA imports; fault
counters prove interception. These are source-level failure-path tests, supplementing normal tests
of the actual candidate binary. Interposition observes actual `cudaHostAlloc` flags **3** (Portable
and Mapped), and the repository tests verify host/device accessibility functionally.

The ThreadSanitizer driver and exact production allocator translation unit are instrumented;
CUDA, RMM, and the backing allocator are not. The default launch failed before testing with an
unexpected address-mapping error. A process-local `setarch aarch64 -R` retry succeeded with exit 0
and no race reports (`tsan/no-aslr.log`). This supports the exercised host cache concurrency paths;
it is not a race verdict for uninstrumented dependencies or all application behavior.

The final production source SHA-256 is
`6c57980ba465f139eea75ed32c0fdb89ec84e7cc3464d797c083cfb87ec38e27`.
The tested candidate `libcucascade.so` SHA-256 is
`3fff46450e43b1e6e6c4f3a2f171a8b634eac8c2aa37ee7eb030ef3e93b75845`.

## Independent empirical assessment

Machine: NVIDIA GB10, Linux aarch64, driver 580.126.09, CUDA 13.2, cuDF/RMM 26.08.
The environment was retained. The fixed experiment uses balanced process blocks
**B, C, C, B, C, B, B, C**. Each Parquet process excludes three warmups and measures eight reads.
The input is the existing 19,032,103-byte `lineitem.parquet`, preloaded into host memory; digest
work is excluded from read timing. Default cuDF pinning/copy thresholds remain unchanged.

Four baseline noise processes have medians 48.6271, 48.65245, 48.7284, and 48.8754 ms: range
**0.2483 ms (0.510%)**, MAD 0.05065 ms. Independently recomputing all final JSONL files gives:

| Warm Parquet read | Baseline | Candidate |
|---|---:|---:|
| Median of four process medians | 48.734650 ms | 48.822875 ms |

The candidate is **0.181% slower**, within the predeclared baseline noise range. All processes and
iterations retain digest `16342477538070118188`, 600,572 rows, and 16 columns. Each read makes 37
custom pinned allocations, of which 3 exceed 8192 bytes; total requested bytes are 71,199. The
large path is exercised, but allocation churn occupies too little of this workload to demonstrate
an application benefit. This experiment supports neither a speedup claim nor a regression claim.

Independent inspection of the root's short Nsight Systems traces confirms the intended mechanism:
`cudaHostAlloc` calls fall from 14 to 5 and `cudaFreeHost` calls from 13 to 4, while
`cudaStreamWaitEvent` calls rise from 189 to 198. All of these recorded calls succeed. The cache
avoids repeated host allocation/free calls by adding reuse dependencies. Profiled timings include
startup/instrumentation and are used only for attribution, not the application verdict.

Same-thread CUDA API attribution within the four `read_parquet` NVTX windows sharpens that result:
baseline reads each issue **3 host allocations and 3 host frees**; the candidate's first read issues
3 host allocations and subsequent reads issue **0 host allocations and 0 host frees**. Both traces
contain 272 GPU kernels and 136 GPU copies across their four reads. Unified Memory tracing is
unsupported by this driver configuration; CUDA runtime, kernel, copy, and NVTX records are present.
The per-read counts are recorded in `PROFILE-SUMMARY.json`.

Warm repeated allocation diagnostics show substantial lifecycle savings:

| Request | Baseline µs/op | Candidate µs/op |
|---|---:|---:|
| 16 KiB | 1.883210 | 0.410179 |
| 128 KiB | 1.948040 | 0.417090 |
| 1 MiB | 500.397000 | 0.420434 |
| 16 MiB | 5522.030000 | 0.427122 |

Each diagnostic includes allocation-stream synchronization and minimal CPU touches. Separate H2D
content checks pass. The figures measure avoided allocation/registration work with a warm cache;
they are not transfer throughput, large-buffer initialization throughput, or end-to-end query gains.

## Matched current Sirius consumer assessment

The original PR claimed a hot 22-query SF50 improvement. A separate prespecified full-consumer
experiment addresses that scope directly; the earlier small Parquet workload cannot price it.
The current engine source is Sirius `45253bf`, with cuCascade consumer dependency `e1b67fe` and its
unrelated reservation-accounting changes retained. The comparison substitutes only current-main
versus final-candidate allocator headers/translation units. Both arms freshly compile the allocating
and deleting `SiriusContext` owner for the corresponding class layout. Its source and compiler flags
match between arms; constructor symbols correctly differ by the new cache-limit argument.

The remaining **333 Sirius archive members and 21 core resource archive members** match byte for
byte. The owner depfiles each contain 2229 inputs, differing only by the copied owner source and
overlaid allocator header; no build-generated owner header is substituted. The private NUMA facade
is included only by the rebuilt owner. The exact same DuckDB `069cc9f` archives, common wrapper,
support archives, and linker flags are used. Actual loaded extension hashes match the manifest:
baseline `5e4582186ce64df86d58a570377ee23bbc0aee7e01915ccb8191ba708b5fe6b8` and candidate
`00ed9a0b550907e7f2d49167e24bc127b921006b32cde8156b0c6e31dbc70ec5`.
This is an isolated comparison using matched frozen current consumer objects, not a fresh full
engine rebuild or replacement of its whole dependency with an ABI-incompatible latest library.

The existing unclustered SF50 dataset is 13.45 GB of raw Parquet. Configuration is fixed: one GPU,
48 GB GPU/24 GB host limits, 512 MiB batches, pipeline/creator/scan threads 8/4/6, Sirius LRU cache,
prefetch disabled, no materialized pinning/compression, and Quent telemetry disabled. CPU fallback
is explicitly disabled. Canonical grouped execution excludes the first iteration of each query
and takes the median of its three following iterations. Each process's figure of merit is the sum
of those 22 query medians; the reported arm value is the median of four independent process values.
Prices come from unprofiled `execute(...).fetchall()` wall time, with validation outside the timer.

The fixed protocol uses three baseline-noise processes, then **B, C, C, B, C, B, B, C**.
Independent recomputation of every CSV and validation record gives:

| Current Sirius hot SF50 suite | Baseline | Candidate |
|---|---:|---:|
| Four process totals, ms | 13576.664; 13578.534; 13525.791; 13581.552 | 13366.717; 13391.483; 13353.822; 13287.594 |
| Median total | 13.577599 s | 13.3602695 s |

The reduction is **1.60065%**. Adjacent balanced comparisons improve by 1.546%, 1.378%, 1.271%, and
2.164%. Baseline-noise totals are 13454.758, 13511.138, and 13525.760 ms: range 71.002 ms,
**0.525507%**, MAD 14.622 ms. All 22 per-query point medians are lower, but no individual-query
significance claim is made from the suite-level noise measurement.

All 11 accepted processes contain exactly 22 queries × four iterations and validate every result
against a fresh CPU reference: **968 checked iterations**, including 704 paired iterations, of
which 528 hot iterations contribute to the figure of merit and 176 first iterations are excluded.
Every query has one result digest across all accepted blocks. Each process also records exactly
88 GPU query starts, completions, and engine query entries, with no fallback/error markers.
Float comparison uses the canonical absolute tolerance of `1e-10`; other values compare exactly.

The first incomplete noise block exposed a saved-result parser issue: canonical code bound
`datetime` to the class, while CPU date values serialize as `datetime.date(...)`. An isolated
adapter changes that eval binding to the module, after timing, without changing queries, engines,
or repository source. The TypeError and corrected date round trip were reproduced independently.
The entire aborted block is retained separately and excluded; all three valid noise blocks and the
unchanged fixed paired schedule restart under the corrected adapter hash.

Separate Nsight traces validate all 88 results in each diagnostic process and capture only Q18's
first hot iteration, chosen from baseline contributions. Independent SQLite queries clip to the
profiler-start return and profiler-stop entry, enclosing exactly one `sirius::query` range:

| Q18 hot mechanism, attribution only | Baseline | Candidate |
|---|---:|---:|
| `cudaHostAlloc` / `cudaFreeHost` calls | 186 / 186 | 0 / 0 |
| `cudaStreamWaitEvent` calls | 901 | 1087 |
| GPU kernels | 1608 | 1608 |
| GPU copy bytes | 3,166,494,775 | 3,166,494,775 |

Copy records vary slightly (2389 versus 2383) while bytes are identical. Relevant allocation and
synchronization returns are zero; `cudaEventQuery` code 600 is expected not-ready polling.
Profiled API durations can overlap across threads and are not substituted for unprofiled prices.

## External NUMA synchronous facade contract

The preserved Sirius `numa_small_pinned_mr::allocate_sync` forwards asynchronous allocation on
the per-thread default stream without waiting; its comment incorrectly assumes the pool ignores
stream ordering. The resource view forwards that method unchanged. This is a pre-existing external
contract defect for reused slabs and can also affect cached large buffers if that synchronous entry
point is used. The required property is CPU readiness before returning from synchronous allocation;
the cheapest correction is to synchronize that exact allocation stream, with allocation cleanup if
the wait throws. It does not justify adding blanket synchronization to the core async allocator.

Reachability was checked after all fixed timings with a separate observer extension. Only the
copied facade adds relaxed atomic entry counters and a destruction summary; the owner is rebuilt,
all other core members and the candidate allocator are preserved, and neither timing arm changes.
The observer runs the same **22 queries × four iterations**, validates all 88 results, and records
88 GPU starts/completions. It observes **84,466 asynchronous allocations and 84,466 asynchronous
deallocations, with zero synchronous calls of either kind**. Installed cuDF host allocators use
asynchronous allocation followed by explicit stream synchronization before CPU initialization.
Thus the stale external synchronous entry point is unused in the measured route and does not
invalidate these results. It remains a limitation for other callers, not a resolved Sirius defect.
Observer timings are excluded from all performance comparisons.

## Practical limits

- This is a single-GPU machine. The multi-GPU owning-stream test exists but was skipped; live
  cross-device behavior remains unverified here. Main's proven #189 mechanism is preserved.
- Successful asynchronous allocation waits order GPU work. Callers must synchronize the allocation
  stream before CPU access; `allocate_sync` supplies this. Installed cuDF host allocators do so.
- Unknown CUDA completion intentionally retains storage. This trades error-path leaks for
  preventing unsafe reuse/free; it is not a general CUDA fault recovery service.
- The 256 MiB default bounds retained large-cache capacity per resource, excluding slabs, live
  allocations, and detached victims being drained. Cacheable requests can round up to a power of
  two. Passing a limit below 16 KiB disables the large cache.
- Requested alignment remains inherited/documented behavior: it does not select storage. This
  reevaluation does not extend the resource's over-alignment guarantee.

Detailed initial findings, raw root-run evidence, manifests, benchmark harness, and failure harness
are retained under `/tmp/cucascade-pinned-pr-20261012/`; the root report preserves selected evidence
under `reports/pinned-cache-reevaluation-20261012/`.
