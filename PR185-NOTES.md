# PR 185 notes after local fixes

Updated 2026-10-14 after the architect -> developer -> reviewer implementation cycle.
This is a temporary handoff file: remove it before merging the PR.

The original notes cited SPEC.md section 7 (pinned-slab-pool). That SPEC and its SF1000
four-GPU evidence are unavailable on this machine. The claimed -8.6 s / -18% result remains
unverified here; it must not be presented as a measured result of this port.

## 1. Allocation statistics (G2): implemented

`small_pinned_host_memory_resource::get_statistics()` returns a thread-safe value snapshot:

| Field | Meaning |
| --- | --- |
| `cached_large_bytes` | Retained cache-entry capacity, matching `large_cache_bytes()` at the same instant. |
| `peak_cached_large_bytes` | Highest retained capacity; purge does not reset it. |
| `large_requests` | Nonzero allocation calls above 8 KiB, including calls that later fail. |
| `large_hits` | Successful cache takes after establishing reuse ordering. |
| `large_misses` | Initial cache lookups that found no entry, once per cacheable request. |
| `large_uncacheable` | Requests whose computed bucket exceeds the limit, once per call. |
| `large_evictions` | Previously cached entries detached for cap eviction or allocation-failure purge. |

Hits and misses are lookup observations, not a partition of requests. Post-purge reuse records
both an initial miss and a hit; a failed initial reuse wait records a request but neither a hit
nor a miss. Two CUDA allocation attempts during recovery still count as one request. Evictions
remain counted if subsequent synchronization fails and storage is quarantined; destructor
release and buffers never inserted into the cache are excluded.

Protected counters use the existing cache mutex. One independent atomic counts uncacheable
requests without adding a mutex to that allocation path. The snapshot samples it once under the
cache mutex and derives the total request count from that same sample. It does not imply that
all observed requests have completed. Separate getter calls can differ during concurrent use.

These counters do not measure actual CUDA allocation/free calls, successful physical frees,
live pinned-memory demand, or per-bucket peaks. Consumer logging through Sirius' per-NUMA
facade remains a separate integration step.

## 2. Cache limit and sizing (G1): documentation corrected

The cap bounds retained entries available for stream-ordered reuse, including entries with GPU
work still pending. It excludes live allocations, slab storage, detached eviction/purge victims
awaiting completion, and quarantined storage. It is not a bound on total pinned memory.

For stable reuse without allocation failures, a conservative full-retention estimate is:

`sum(bucket capacity * peak outstanding allocation count in that bucket)`

Include allocation calls still in progress and all streams/devices sharing the resource. This
is an estimate, not an exact minimum or a prediction of CUDA call counts. Returned-buffer order
and overlapping lifetimes affect churn; query-end simultaneous freeing is not guaranteed.
For example, alternating 16 KiB and 32 KiB allocation/free cycles under a 32 KiB cap produces
two misses per cycle through mutual eviction.

`peak_cached_large_bytes` measures retained occupancy, which is itself capped; it does not
measure the live-demand peaks needed by the estimate. Sirius currently owns one pool per NUMA
node, potentially shared by several GPUs. Multi-GPU demand must be measured on the other box.

## 3. NUMA placement (G3): separate follow-up

Slabs use the fixed-size upstream pool. Large buffers use
`cudaHostAlloc(cudaHostAllocPortable | cudaHostAllocMapped)`, which does not enforce the
upstream pool's NUMA placement. Portable controls recognition across CUDA contexts and Mapped
controls device mapping; neither flag specifies a NUMA node. Do not claim a universal CUDA
contract that pages follow the allocating thread's NUMA policy.

`get_upstream_resource()` returns a type-erased resource reference; custom upstream resources
are supported. The normal factory uses `numa_region_pinned_host_memory_resource`, but binding
requires a known node, and its Portable/Mapped registration flags depend on `make_portable`
(which defaults to false).

A future upstream allocation route must preserve host/device access and the required flags,
use matching upstream deallocation with the actual allocated bucket size on every eviction,
purge, fallback, and destruction path, and preserve completion checks, quarantine, and upstream
lifetime. Known-node registered storage needs unregistering plus `numa_free`, not `cudaFreeHost`.
The existing registration-failure path also needs host-allocation rollback before adopting it.
Allocation/registration/release behavior can change even on a single-socket host.

## 4. Behavior and compatibility: corrected PR description

- An upstream reference is required. Construction using the default cache limit enables a
  256 MiB retained cache; this is not a zero-argument default constructor.
- Existing one-argument construction is source-compatible. The constructor symbol and object
  layout differ from main; consumers must be rebuilt and relinked. The statistics fields also
  change the layout relative to the previously measured cache build.
- Only cacheable requests above 8 KiB are rounded to a representable power of two of at least
  16 KiB. Their allocation size is less than twice the request. Above-cap requests retain their
  exact size; unrepresentable rounding also retains the exact size.
- Noncacheable/no-event large-buffer release explicitly synchronizes the freeing stream before
  unpinning; failed synchronization retains storage. Do not rely on undocumented implicit
  synchronization from `cudaFreeHost`. Main already had the no-event slab fallback.
- The alignment argument remains unused, as documented.

CUDA references: [memory management](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-runtime-api/group__CUDART__MEMORY.html)
and [synchronization behavior](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-runtime-api/api-sync-behavior.html).

## 5. Local validation and merge hygiene (G5)

The new local validation uses the existing Release/CUDA 13.2/cuDF 26.08 environment on one GB10:

- Warnings-as-errors build passed.
- Allocator suite: 29 cases (17 added relative to main's 12), 28 passed and one multi-GPU case
  skipped; the recorded run passed 1,202 assertions. Nine cases cover statistics.
- Exact allocator translation-unit fault harness: 21 cases passed, including request-once
  retry/failure accounting, failed reuse waits, cap/purge detachments despite quarantine,
  metadata fallback, and pending-DMA teardown.
- ThreadSanitizer: eight workers x 1,000 mixed cached/bypass cycles plus concurrent snapshots;
  no reports. The driver and exact allocator translation unit are instrumented; CUDA, RMM,
  and the backing allocator libraries are not. The established process-local ASLR setting was
  used to allow the sanitizer to start.

Logs and supplemental harnesses are local and ignored under `build/pr185-followup/`.

Commits `68440d1` and `b2ded83` added about 19.7k report lines and 6.456 MB of evidence archives;
`32dcbce` deleted them from the final tree, but their blobs remain in ancestor commits.
Squash-merge the final clean tree, or explicitly curate the history before merging. An ordinary
merge or replay of the artifact commits retains the blobs in main's reachable history. Squash
merging does not erase copies already published to the fork/server. Remove this notes file
before the final merge; do not re-add investigation artifacts.

Keep the PR draft while the remaining validation is being completed. The current status failure
concerns missing category/breaking labels, and the current GitHub account has no upstream
write/triage permission to apply them. Ask a maintainer for the applicable labels; `improvement`
and `breaking` fit the performance feature and ABI change. The historical copy-pr-bot comment
alone does not establish the current runner-vetting or CI state. CI still needs verification.

## 6. Multi-GPU and application evidence: user handoff

The existing allocator `[multi_gpu]` case tests a mismatch between the current device and the
freeing stream's owner. It was skipped on GB10. It does not test producer GPU 0 -> reuse GPU 1,
concurrent traffic from multiple GPUs through one shared pool, or four-GPU performance.

On the multi-GPU box:

- Rebuild dependent consumers and run `cucascade_tests "[small_pinned]"`, confirming the hardware
  case actually executes.
- Exercise cross-device reuse with pending producer work and simultaneous shared-pool traffic;
  check data, stream dependencies, device restoration, and cache accounting.
- Run a matched Sirius baseline/candidate A/B at the intended SF1000 four-GPU configuration,
  recording warm and explicitly pinned modes separately, actual per-NUMA limits, source/build
  identities, and GPU-vs-CPU correctness. Preserve normal linker behavior and shared RMM
  resource registration across Sirius/cuDF.
- Collect quiescent per-pool statistics and scoped warm-window `cudaHostAlloc`/`cudaFreeHost`
  counts. Use unprofiled timings for performance; API duration sums are not removable wall time.

Earlier corrected single-GB10 SF50 comparisons measured 6.14% less elapsed time with Sirius
input caching and 3.37% less with it disabled, with 1,936 CPU-validated query executions. These
were repeated-query measurements, not fully cold runs or explicit decoded-table pinning.
They used the prior reviewed cache build; the new statistics addition has not been priced by a
fresh Sirius A/B. Do not generalize those percentages to four GPUs or replace the unavailable
SPEC evidence with them. The other box should establish the result for the final implementation.

## Separate Sirius issue

`numa_small_pinned_mr::allocate_sync` forwards to asynchronous allocation without waiting.
Its comment that the pool ignores streams is stale. Correct the facade's synchronous CPU-readiness
contract in Sirius; it predates this PR and is outside the cuCascade implementation changes.
