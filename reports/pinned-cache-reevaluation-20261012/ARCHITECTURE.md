> Superseded application inference: the original consumer relink hid RMM's shared registry. Its 1.60% figure and raw-device-allocation attribution do not describe normal consumer resource registration. See [corrected follow-up](followup-20261013/REPORT.md). Core allocator findings remain valid.

# Architecture assessment: pinned large allocation cache

## Reading record

Completed before allocator analysis:

- Architect role configuration: `/home/kkristensen/.codex/agents/architect.toml`.
- Full tables of contents in all three assigned design books.
- *A Philosophy of Software Design*: complete chapters 4 (module depth), 5 (information hiding), 8 (pull complexity downwards), 11 (design twice), 16 (modifying existing code), and 20 (performance design), corresponding to lines 1047–1798, 2392–2526, 3725–3840, 5692–5927, and 6691–7039 of the supplied transcription.
- *C++ Software Design*: complete chapter 2, guidelines 6–10 (behavioral expectations, concepts, overload semantics, abstraction ownership, architecture documentation). Chapter content read across the supplied transcription's lines 2614–4179, plus surrounding chapter boundaries. The full table of contents includes all eleven chapters.
- *C++20 — The Complete Guide*: complete relevant sections 5.6–5.9 (semantic constraints and concept design) and 7.3–7.6 (borrowed ranges and constness), lines 4581–4966 and 8043–8680. These sections were selected after the full table of contents; this is not a claim to have read the entire book.
- Checked-out `CLAUDE.md`, upstream `/tmp/cucascade-pinned-pr-20261012/AGENTS-main.md` (including all differences), and `docs/ARCHITECTURE.md`.

Then read the complete PR allocator header/source and tests; current `origin/main` allocator header/source including commit `1b0e7b6` (#189); fixed-size upstream header/source; `memory_space` integration; NUMA pinned allocator; stream compatibility wrapper; peripheral PR diffs; installed CCCL resource concept, RMM stream-ordered resource, and cuDF host-vector allocation implementation. No repository edits or GPU runs were performed by this architect agent. User's GSD bypass authorization was relayed by root.

## Static verdict

**The large-buffer cache remains useful and should be integrated, with current main's correctness fixes preserved.** On fetched `origin/main` (`bd4aeed`), requests above 8192 bytes still call `cudaHostAlloc(Portable | Mapped)` on every allocation and `cudaFreeHost` on every release. A bounded cache of these allocations removes repeated page registration and allocation work. The PR supplies this missing mechanism; #189 only fixes event ownership and slab lifecycle and does not supersede the cache.

This is a plausible performance opportunity, not a measured speedup. Use of this resource is opt-in; its large-buffer cache is enabled by default with a 256 MiB retained-byte limit. This repository does not instantiate `small_pinned_host_memory_resource` in its production paths or install it as cuDF's default pinned allocator. `memory_space` HOST continues to own `fixed_size_host_memory_resource`. An empirical harness must explicitly install the resource where cuDF or another consumer obtains host allocations. Existing core throughput benchmarks otherwise cannot establish its application-level benefit.

Retain portable/mapped flags, five slab classes, large power-of-two buckets starting at 16384 bytes, exact requests beyond the cache cap, cache disabling through a cap below the first bucket, and the bounded FIFO eviction policy. The PR's per-header `__has_include` stream shims are superseded by main's `cucascade/cuda/stream.hpp`; retain the main wrapper and upstream API migration.

## Behavioral contract: async and synchronous host access

The async operation `allocate(stream, bytes)` promises stream-ordered readiness. It does not promise that immediate CPU stores are safe before the supplied stream is synchronized. Installed CCCL `cuda/__memory_resource/resource.h` explicitly describes `resource` as additionally supporting stream-ordered allocation. Installed RMM's stream-ordered resource queues `cudaStreamWaitEvent` when borrowing blocks across streams; its `allocate_sync` then synchronizes the default stream. Installed cuDF `detail/utilities/host_vector.hpp`, lines 143–147, calls `mr.allocate(stream, ...)` and explicitly synchronizes the stream before thrust host-vector initialization.

Consequently, **preserve asynchronous event waits for both slabs and cached large allocations**. My early recommendation to host-synchronize every reuse was too broad and is withdrawn. Document that direct CPU users must synchronize the allocating stream, or use `allocate_sync`, before touching the returned storage. A test that reallocates asynchronously and immediately overwrites the pointer on the CPU without synchronization violates this contract; it does not prove an allocator defect.

The synchronous operation must return memory immediately usable by the host. When a cache/slab hit inserts a dependency into the default stream, `allocate_sync` must wait for that dependency and must not return storage after a failed synchronization. `deallocate_sync` cannot discover unrelated streams: callers still must order their final uses before the synchronous release operation.

NVIDIA documents that stream waits apply to future submitted stream work, while event synchronization waits for captured work on the CPU. Event destruction alone is not a completion fence. [CUDA stream management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html), [CUDA event management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html).

## Required invariants and minimal mechanism

### Event ownership and reuse

Maintain main #189's lookup of the freeing stream's actual device, event creation on that device with restoration of the caller's current device, and device-indexed idle event pools. Looking only at `cudaGetDevice` is insufficient when a stream from another device is passed.

A cache/slab entry must retain exclusive ownership of its pointer and event until a reuse dependency is established successfully. Perform the checked `cudaStreamWaitEvent` before consuming the candidate and changing accounting; on failure, keep the entry and propagate the CUDA error. After a successful wait is submitted, recycling the event is legal because the dependency observes the recorded work existing at the wait call. Event pooling remains a best-effort optimization: if host metadata allocation fails when recycling an event, destroy its handle rather than terminate.

The event/storage mechanics may use a small private value record `(handle, owning_device)`, shared by slab and large entries. Do not add inheritance, a generic allocator framework, public lifecycle helpers, a worker thread, callbacks, or new user configuration for this change. The existing CUDA resource concept already supplies the relevant interface abstraction.

### Nonthrowing release and exceptional paths

`deallocate` is `noexcept`; map insertion, deque growth, vector growth, or event-pool growth may throw host allocation exceptions. A failed attempt to retain a large buffer should fall back to establishing completion and freeing it outside the allocator lock. Cache state must remain consistent, including removal of any empty bucket accidentally created before a failed append. Capacity accounting changes only after successful insertion.

For slab expansion, reserve all needed free-list capacity and acquire backing ownership before publishing any new slab pointers. This allows ordinary returns of all slabs carved from those blocks without later vector allocation. If safe publication still fails, retain/quarantine its backing allocation; do not return a block upstream while pointers into it might remain visible.

A failed completion fence is not proof of completion. Follow #189's fail-closed rule: a slab is unavailable for reuse, and its backing blocks must not return upstream. A large pointer whose event or stream synchronization failed must likewise remain pinned and unavailable, rather than being freed as the PR's current best-effort fallback does. Controlled leakage on an unrecoverable CUDA/context error is safer than use-after-free. Avoid new allocating containers merely to record this exceptional quarantine.

Never hold the allocator mutex while waiting for an event/stream or releasing pinned storage. Keep nonblocking wait submission and ownership/accounting changes serialized under the mutex; detach an eviction victim under the lock, then synchronize and free it outside the lock.

### Cap and allocation failure

Use physical bucket capacity, not requested bytes, for retained cache accounting. Compare insertion capacity through `bucket <= limit - cached_bytes`, after preserving `cached_bytes <= limit`, to avoid overflow for large configured caps. Retain the guard before `std::bit_ceil` when the next power of two is unrepresentable.

On a direct pinned-memory allocation failure, detach/swap the existing cache container and reclaim its entries outside the lock, then retry once. Do not build a new vector snapshot in the OOM path: that recovery step itself can fail under host memory pressure. Recovering retained pinned memory does not guarantee that the original request will become satisfiable. Prefer distinguishing actual CUDA allocation failures from unrelated asynchronous CUDA errors instead of hiding every error as `std::bad_alloc`.

The 256 MiB default is a cap on cached large-buffer capacity, not a global resource budget: live large allocations, slab backing, metadata, and detached entries being synchronized are outside the stated counter. Large requests already bypass fixed-size upstream reservation and NUMA placement on main; keeping this behavior avoids an unrelated redesign. Documentation must not imply the cap enforces `memory_space` host capacity. No automatic reservation-manager integration is justified by the existing PR scope.

### Destruction

No calls may be in flight and every live pointer must have been released before resource destruction. Pending work captured by released entries may remain and must be fenced before freeing large buffers or returning slab backing blocks upstream. Preserve #189's explicit completion waits and quarantine handling. Destroying an event is not a substitute for waiting. Restore the caller's current device after teardown; destruction remains nonthrowing.

## Alternatives considered

1. **Keep raw allocation/free.** Lowest implementation cost and no retained capacity, but the unchanged main path still pays allocation/registration costs for every repeated large staging buffer. Keep it as the empirical baseline and cap-zero mode.
2. **Bounded bucket cache using current event mechanisms.** Small extension, compatible public constructor/default behavior, useful repeated-size reuse, and explicit retention tradeoff. Recommended.
3. **Route all large buffers through fixed-size upstream blocks or introduce a common global pinned allocator.** Would affect contiguous allocation, NUMA placement, reservation accounting, consumers, and resource lifetimes. This is a separate feature with a much larger integration surface; the PR does not need it.

Cache selection/size/cap/FIFO are policy. Device events, completion fencing, storage release, and ownership transfer are mechanism. Keep those concerns distinguishable within one deep allocator instead of spreading shared ownership rules across new shallow classes. Encapsulate helpers as private implementation details; preserve only allocation/deallocation, identity, constants, and the intentional cache introspection contract in the public interface.

## Reference designs consulted

Velox's documented memory system separates physical allocation, query reservations, arbitration, and cache reclamation; allocation retry can shrink its cache. Its lesson here is to keep reclaimable cache storage and budget ownership explicit, without pretending this local cache enforces the larger host budget. [Velox memory management](https://facebookincubator.github.io/velox/develop/memory.html).

DuckDB's `BufferPool::EvictBlocksInternal` detaches an evictable buffer for direct reuse when the physical allocation size matches a request, while temporary reservations keep accounting tied to successful ownership changes. That supports precise size-class reuse and transactional bookkeeping. [DuckDB buffer pool source](https://github.com/duckdb/duckdb/blob/main/src/storage/buffer/buffer_pool.cpp).

DataFusion's `GreedyMemoryPool` and `FairSpillPool` separate reservation policies from allocation consumers. This reinforces the distinction between a retention cap and a whole-system reservation policy; this PR should implement the former. [DataFusion memory pools](https://github.com/apache/datafusion/blob/main/datafusion/execution/src/memory_pool/pool.rs).

These are design comparisons, not proof that their CPU allocators implement CUDA event semantics.

## Empirical acceptance plan

Root owns GPU execution; implementation/reviewer own changes and independent checks.

- Compare fetched main against integrated cache on the same machine and toolchain, explicitly installing the allocator for a relevant consumer. Include cap-zero if needed to separate policy effects.
- Measure repeated bucket sizes (near threshold, medium, and larger), same-stream and cross-stream reuse, concurrent threads, mixed working sets, cap eviction, and cold/oversized requests. Report steady-state allocation cycles and a representative cuDF workload with result checks, plus retained bucket bytes.
- Verify portable/mapped memory properties and, if multiple GPUs exist, stream device ownership with a mismatched current device and cross-device reuse. A single-GPU machine cannot certify multi-GPU runtime behavior.
- Ordering checks must observe actual copied/device-written values and use the valid async/sync contract. Test `allocate_sync` for immediate CPU writes, async reuse with explicit allocating-stream synchronization, destructor with pending released work, and stream-wait failure retaining its candidate.
- Use targeted failure injection or an isolated harness for metadata OOM, event creation/record/wait/sync failures, and pinned allocation retry. The existing simple pointer-equality tests cannot certify these cases.
- Keep the optimization only if measured benefit justifies its retention and event metadata cost; static plausibility alone is not the final value verdict.


## Independent empirical protocol assessment

Reviewed `/tmp/cucascade-pinned-pr-20261012/CONTRACT.md` and `harness/pinned_bench.cpp` after the above specification. Root is responsible for GPU execution.

The primary workload is a natural cuDF Parquet decode path, using the existing 19,032,103-byte `lineitem.parquet` fixture preloaded into an ordinary host byte vector. Installing the counted custom pinned resource explicitly matches the PR's opt-in consumer boundary. Default cuDF pinning and copy thresholds are retained. This supports a conclusion about warm in-memory reads through that registered resource, not disk I/O, a full TPC-H query, or users who leave the custom allocator uninstalled.

Three warmups exclude first-use setup from the primary figure of merit; eight timed calls per process measure `read_parquet` plus completion of its CUDA stream. Result hashing and log emission are outside the timed region. Count snapshots surround the timed read, so per-read allocation/large-request counts identify whether the natural path can exercise the feature. The final size histogram includes warmups and digest work and should not be treated as a histogram solely of timed reads. `large_allocations` counts requests, not cache hits: profiling CUDA host-allocation/free calls outside the primary measurement is appropriate attribution.

The fixed B,C,C,B,C,B,B,C schedule balances the sum of process positions for baseline and candidate (18 each), reducing linear temporal drift. The unit of independent repetition is the process median: compare the four baseline medians with four candidate medians. The 32 measured iterations in each implementation are within-process samples, not 32 independent experiments. Four prespecified baseline-only processes provide an observed variation range and median absolute deviation; a gain smaller than that spread is inconclusive. The sample count remains modest, and utilization snapshots cannot eliminate all CPU/GPU contention or nonlinear thermal drift.

The digest protocol detects changes between iterations in a process. Cross-process digest, row count, and column count have now also been validated from every raw baseline/candidate read record; all are identical. The row-hash fold remains a probabilistic content check, not full schema/metadata equality. The manifest records one SHA-256 for the identical source fixture used across processes.

The final counted wrapper uses atomic counters and atomic histogram cells with relaxed operations and loads. This addresses the instrumentation race identified in the first harness review without requiring assumptions about serial allocator invocation. Both baseline and candidate use the same counting wrapper.

The churn diagnostic validly fences the allocating stream before CPU stores and verifies an H2D result survives reuse. It can quantify allocation/registration avoidance across the selected sizes, but does not substitute for natural workload improvement. For this standalone protocol, no sampling expansion based on its result was recommended. The subsequently identified original Sirius-consumer claim requires a separate, explicitly prespecified consumer experiment; the small fixture cannot price that claim.


## Independent standalone empirical assessment

**Retain the bounded cache as an allocation optimization within this explicitly registered resource. Caching is enabled by default with a 256 MiB limit. The measured standalone Parquet workload shows no application speedup.** This conclusion is based on independently recomputed raw logs and read-only queries of the exported Nsight SQLite traces, not only the root summary.

The manifest pins baseline `bd4aeedbee1201df68bd042374fd9ab6afb5b46d` and clean merged candidate `8b2d229c1f7b064aad6db930bf7a9c396531e1ba`, distinct binary hashes, the fixture SHA-256, successful exit statuses, and the prespecified B,C,C,B,C,B,B,C process schedule. Each Parquet process contains three excluded warmups and eight measured samples; all four independent process medians per implementation agree with `RESULTS.json`.

| Measurement | Main | Merged cache | Interpretation |
| --- | ---: | ---: | --- |
| Warm Parquet read median | 48.73465 ms | 48.822875 ms | +0.1810%; within observed baseline variation |
| Warm 16 KiB allocation cycle | 1.883210 µs | 0.410179 µs | 4.59× faster |
| Warm 128 KiB allocation cycle | 1.948040 µs | 0.417090 µs | 4.67× faster |
| Warm 1 MiB allocation cycle | 500.397 µs | 0.420434 µs | 1190× faster |
| Warm 16 MiB allocation cycle | 5522.030 µs | 0.427122 µs | 12928× faster |

The Parquet difference is +0.088225 ms, smaller than the prespecified baseline-only process-median range of 0.2483 ms (0.5100%). The baseline median absolute deviation is 0.05065 ms. Four independent process medians per implementation support an inconclusive application difference, rather than an equivalence claim or a demonstrated regression. No post-hoc workload expansion is needed: the natural path exercised three allocations above 8 KiB on every read, but these were small enough that allocation churn was a minor part of the roughly 49 ms read.

All 88 raw read records across baseline and candidate processes have the same content tuple: row-hash fold `16342477538070118188`, 600,572 rows, and 16 columns. Every read also reports 37 allocator calls, three large requests, and 71,199 requested bytes. The digest checks validate the measured outputs probabilistically; they do not establish complete schema/metadata equality.

The stress ratios quantify the intended mechanism on this NVIDIA GB10 machine. Their timed loops repeatedly allocate, synchronize an otherwise empty/completed stream, write only the first and last CPU bytes, and deallocate. They avoid repeated CUDA host registration/allocation work; they do not measure bulk CPU initialization, transfer of a 16 MiB payload, a mixed-size working set exceeding the cap, or a query speedup. Even the smallest measured cached buckets improve this diagnostic, so the data do not support removing the current 16 KiB starting bucket or adding another policy knob.

Supplemental profiling establishes actual cache activity. Each of four main `read_parquet` NVTX ranges contains three `cudaHostAlloc` and three `cudaFreeHost` calls. Candidate's first read contains three host allocations and no frees; its subsequent reads contain neither. Every read range has 67 kernels and 32 copies; both full traces have 272 kernels and 136 copies, including digest work outside the read ranges. These counts demonstrate that the cache removes the intended CUDA allocation/free calls while retaining the same GPU workload; profiled elapsed times are not substituted for the unprofiled primary figure of merit.

The current harness's relaxed atomics eliminate the earlier instrumentation-race concern. Existing TSan artifacts show the first attempt could not start due to an unexpected memory mapping, followed by a successful process-local `setarch -R` run of the exact allocator translation unit with eight threads and 1,000 cache cycles per thread. This is supporting host-concurrency evidence, not a multi-GPU certificate. No GPU operations or repository edits were performed by this architect during independent verification.

The final value boundary is therefore clear: useful bounded reuse for callers that explicitly install this resource and repeatedly allocate host staging buffers, with large allocation-cycle gains on this machine; no measurable warm Parquet application improvement for the tested fixture. Keep the feature without advertising the stress speedups as application speedups.


## Original Sirius consumer: independent contract and provenance review

The standalone fixture established cache execution and allocation-cycle benefit, but cannot price the original PR's claim about hot TPC-H SF50 queries. Root therefore created a separate fixed consumer protocol before collecting its valid samples, recorded in `sirius-benchmark/CONTRACT.md`. It measures all 22 rendered SF50 queries over the existing 13.45 GB unclustered Parquet dataset, without pinning/compression, through current Sirius and the same explicit custom pinned resource registration. The figure of merit is the sum of the per-query medians of three hot iterations after excluding each query's first iteration. Three baseline noise blocks precede the fixed B,C,C,B,C,B,B,C schedule.

Contract strengths: identical canonical 48 GB GPU / 24 GB host configuration; grouped query ordering and cache state; disabled Quent telemetry and CPU fallback; execute-plus-fetchall wall-clock timing; CPU-reference checking of every iteration outside the timer; required complete GPU starts/completions/query-entry logs; process-level repetition; and no profiler within timed comparisons. The protocol reevaluates current consumer value. The original exact configuration and speedup aggregation are unavailable, so this cannot certify reproduction of the historical 13.9% claim. The suite sum weights longer queries more heavily; per-query changes should remain available alongside it.

I reviewed `sirius-build/README.md`, `build.py`, concise build metadata, canonical `performance_test.py` timing/cache boundaries, the isolated validation adapter, and A/B orchestration script. I independently verified the baseline/candidate allocator header/source overlay bytes against their pinned cuCascade commits, both extension hashes against metadata, identical owner source hashes across arms, and matching recorded unchanged-member digests across arms. The build recipe compares all unchanged archive members with their donor, substitutes only the allocator translation unit and layout-sensitive `SiriusContext` owner, and recompiles both arms rather than relying on incompatible class layouts. Current Sirius source equivalence with donor engine objects was separately checked by the assigned developer/explorer; this architecture review relies on that stated byte comparison rather than repeating it. Unrelated current consumer `e1b67fe` reservation-accounting changes and all other dependency/core members remain present in both arms. Shared current DuckDB wrapper glue and matching clean DuckDB revision `069cc9f` are identical between arms.

Both completed full 22-query smoke runs have canonical CPU-reference `validation.csv` entries with success for every query. The first incomplete noise attempt was subsequently aborted at Q3 because the isolated per-iteration adapter inherited a `datetime` class binding when the saved rows require `datetime.date` module lookup. Root repaired that parser binding in the isolated adapter, preserved the aborted block, excluded it, and restarted the fixed sequence with three valid noise blocks. This is a validation-harness repair; no engine/query or committed PR change occurred. The corrected adapter hash must be the one pinned in the valid manifest.

The fixed consumer experiment completed all eleven valid blocks. The following independent final assessment states its verdict separately from the standalone no-Parquet-speedup result; no partial favorable block replaced the fixed experiment.


## Final recommendation after the original-consumer experiment

**Retain the bounded large-buffer cache, enabled by default with a 256 MiB retained-byte limit on `small_pinned_host_memory_resource`.** Consumers choose whether to use/register this resource; callers already using its default constructor gain caching automatically after rebuilding. The static value remains, allocator lifecycle costs improve substantially, and the current Sirius consumer now shows a modest measured application benefit. The historical 13.9% application improvement was not reproduced under the current fixed configuration.

All eleven valid blocks are present in `sirius-benchmark/MANIFEST.json`: three baseline-only noise blocks, followed by the prespecified eight balanced baseline/candidate blocks. I independently recomputed every per-query hot median and suite sum from each raw `csv/runtimes.csv`, checked four iterations for each of all 22 queries, excluded the first iteration, and verified all successful exits. The corrected adapter/configuration/extension hashes match the files used. The combined engine logs independently recount exactly 88 GPU starts, completions, and query entries per block with no recorded fallback/error markers. Rendered SQL files, copied configuration, canonical engine metadata, no-pin/no-compression settings, and unprofiled mode are identical across blocks.

The four baseline suite sums are 13,576.664, 13,578.534, 13,525.791, and 13,581.552 ms. The four candidate sums are 13,366.717, 13,391.483, 13,353.822, and 13,287.594 ms. Their process medians are **13,577.599 ms baseline versus 13,360.2695 ms candidate: 217.3295 ms, or 1.6006475%, less elapsed time** (1.01627× speedup). Every candidate block is below every paired baseline block. Adjacent baseline/candidate comparisons all favor the candidate; the root summary reports a 1.27%–2.16% range.

Baseline-only noise sums are 13,454.758, 13,511.138, and 13,525.760 ms. Their range is 71.002 ms (0.5255%) and median absolute deviation is 14.622 ms. The median measured savings are 3.06 times this observed baseline range. Paired baseline and candidate ranges are 55.761 and 103.889 ms respectively. This supports a modest measured gain under this configuration, not a formal significance or broad generalization claim from four independent processes per arm. Individual per-query median reductions are exploratory; their multiple correlated comparisons are not separate independent confirmations.

All **968 query iterations** across noise and paired runs validate against the CPU reference with the canonical tolerance. Independently inspected validation records also have exactly one identical `(row_count, digest)` tuple per query across every iteration and implementation, including excluded first iterations. All 242 final per-block query validations report success. A total of 726 hot query iterations supplied the three-sample medians; the statistical repetition unit remains the independent process/suite block.

The four value assessments must stay distinct:

| Assessment | Result | Scope |
| --- | --- | --- |
| Static allocator value | Main still allocates/frees every large buffer; bounded reuse removes that churn | Resource implementation and contract |
| Allocation lifecycle diagnostic | 4.59×–12928× lower warm allocation-cycle cost | Empty/completed streams, repeated sizes, endpoint stores on this GB10 |
| Standalone 19 MB Parquet fixture | +0.181% elapsed difference, within 0.510% observed noise | Warm cuDF read using explicitly registered resource |
| Current Sirius SF50 suite | 1.60065% less suite elapsed time, beyond observed 0.5255% baseline range | All 22 queries, current source/dependency snapshot, fixed 48 GB GPU / 24 GB host config |

The current-consumer comparison uses the exact merged allocator through a narrow matched owner/header rebuild while preserving the consumer's current `e1b67fe` reservation-accounting changes and unchanged engine/dependency objects. It is not a whole latest-main library swap. The benchmark measures current consumer value and fills the representativeness gap left by the tiny Parquet fixture. Neither it nor the lifecycle diagnostic certifies the unspecified old configuration behind the original 13.9% claim.

The earlier standalone Nsight traces demonstrate removal of host allocation/free calls with identical GPU workloads. Additional current-consumer attribution captures are root-owned and separate from these unprofiled timing results. Pending checks of the preexisting consumer NUMA facade's synchronous contract are a separate integration concern: the underlying asynchronous allocation contract still requires a fence before immediate CPU access, and synchronous facades must honor the synchronous promise. This architecture assessment does not silently claim those external wrapper paths are certified by a successful throughput suite.

No GPU work or production edits were performed by this architect during the independent consumer assessment; only this temporary report was updated.
