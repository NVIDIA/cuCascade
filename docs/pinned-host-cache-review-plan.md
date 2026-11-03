# Pinned host cache review plan

Status: implementation delivered under the accepted revisions below. This document retains the original planning decisions and validation protocol; completed evidence and explicit acceptance exceptions are recorded in the [validation record](pinned-host-cache-validation.md).

Review: [NVIDIA/cuCascade #185, November 2, 2026](https://github.com/NVIDIA/cuCascade/pull/185#pullrequestreview-5740489603).
Published spec: [kevkrist/cuCascade #1](https://github.com/kevkrist/cuCascade/issues/1), labelled `ready-for-agent`.
Code inspected at `69ef173b10a75363d80aa3a8df833f2858159034`.
The user initially authorized a documentation-only GSD bypass because no GSD entry point is installed,
then extended the bypass to all directories and tasks. The general override is recorded in AGENTS.md.

## Accepted decisions

1. Preserve stream-ordered allocation. CPU access requires stream synchronization or `allocate_sync`;
   memory contents are uninitialized. See [ADR 0001](adr/0001-pinned-host-allocation-contract.md).
2. Permit blocking memory reclamation on eviction and OOM recovery, including completion waits on
   other devices. Perform completion waits and buffer frees outside the cache mutex.
3. Include existing alignment, slab-quarantine, and event-destruction limitations in the remediation,
   with separate implementation steps and commits. Give every review claim a disposition.
4. On rounded-allocation OOM, try the original requested size before reclaiming cached buffers. Track
   a successful exact-size fallback as uncached, and release it after its GPU use completes.
5. Requests above the slab alignment guarantee use dedicated, explicitly aligned allocations.
   Preserve the base pointer required for release and check size arithmetic for overflow.
6. Allocator calls inside CUDA host callbacks or during graph capture are unsupported. On a capturing
   stream allocation throws; deallocation withholds storage without recording, waiting, freeing, or
   evicting buffers, marking a slab's backing owner as quarantined where needed. Preserve capture
   validity. Existing capture-based tests inject failures; they do not establish graph support.
7. Device restoration failure is fatal rather than silently returning on a different GPU. Ordinary
   cleanup failures withhold affected storage and retire suspect events. See
   [ADR 0002](adr/0002-pinned-host-cuda-failure-policy.md).
8. Narrow slab quarantine to the owning backing allocation.
9. After the exact-size OOM, reclaim only if current retained capacity covers the exact allocation
   size. Detach enough oldest entries to cover it and retry exact allocation once. Accept that
   smaller partial reclaim might otherwise have helped. See
   [ADR 0003](adr/0003-conservative-pinned-host-oom-reclamation.md).
10. Track only exceptional allocations: exact-size fallbacks and over-aligned dedicated buffers.
    Reserve metadata before acquiring their pinned storage; keep ordinary pool ownership unchanged.
11. Move cache-hit wait submission and cold event creation outside the shared mutex in this
    remediation. Retain ownership until submission succeeds and define failure handling explicitly.
12. Keep oldest-first eviction. Measure node overhead and eviction behavior; add node pooling or
    change victim policy only if measurements justify the added complexity.
13. Use a dedicated test build of the actual implementation with private scoped failure hooks for
    selected CUDA calls and metadata construction. Production builds call CUDA directly and expose
    no test hooks in installed headers. Avoid process-wide allocation failures.
14. Add `cuda_event::reset_no_throw()` to relinquish handle ownership before attempting destruction,
    return the destruction status, and clear the error from a failed destruction. Use it from the
    destructor and move assignment as well as explicit retirement.
15. Compare against the inspected PR revision on the same hardware using at least five paired runs
    for concurrent hits, half-cap rounding, full-cap eviction, and the application workload. A
    reproducible median regression above 5% requires investigation before completion. GPU
    correctness and multi-GPU validation remain required even if execution is unavailable locally.

## Decision status

All interview questions are settled. Routine implementation details must preserve these decisions,
ownership invariants, and acceptance criteria. If implementation evidence contradicts a settled
assumption, revise this plan with the user before changing its contract or scope.

## Facts constraining the implementation

- `deallocate_large` currently reconstructs a bucket from the requested bytes. An exact-size fallback
  must never be published as a full-bucket entry; otherwise a larger request could overrun it.
- The fixed upstream guarantees `max_align_t`; custom block strides do not guarantee a slab-size
  alignment. Installed cuDF host allocators pass `alignof(T)` and synchronize after allocation.
- Each slab expansion currently owns one upstream block through an RAII handle. A slab withheld on
  deallocation is absent from the free lists, so its owner must be marked at failure time.
- The cache belongs to one resource instance and may be shared across streams and devices. Its
  retained-byte limit excludes live buffers and detached or quarantined storage.
- GPU execution was unavailable during initial read-only discovery. Existing Release and
  Release-stable Ninja builds and Pixi environments are present. GPU validation remains a required
  external gate until device access is established.

## Implementation sequence

Prepare the private test variant first, then add the relevant deterministic regression coverage with
each implementation step. Keep the inspected revision available as an isolated build for performance
comparison. Changes are split into commits by responsibility; each commit runs the checks relevant
to its behavior before subsequent steps.

### 1. Establish allocation ownership and alignment

Keep the public resource API and ordinary allocation path. Add ownership tracking sufficient to
distinguish exact-size fallback and over-aligned allocations, including the returned pointer and its
original allocation base. Check that metadata failure cannot lose a newly allocated pinned buffer
or publish an untracked pointer.

Use an ordered private ownership map keyed by returned pointer, with a metadata node prepared before
the CUDA allocation. Only exceptional allocations need records. On the ordinary slab path the
original alignment identifies whether a dedicated allocation could have been used. On a large
cacheable deallocation, consult exceptional ownership before publishing a bucket entry. Keep lookup,
publication, and removal thread-safe, and avoid holding the cache mutex during CUDA allocation/free.

Route over-aligned requests to dedicated allocations with checked padding arithmetic. On
deallocation, identify dedicated allocations before selecting a slab or cache bucket; preserve the
matching-size and matching-alignment caller contract. Validate nonzero power-of-two alignment,
throwing `std::invalid_argument` for invalid alignment. Report checked size/padding overflow as an
allocation failure. For an over-aligned buffer, the exact backing size includes only necessary
alignment padding; it does not acquire power-of-two cache padding. The public byte-only sizing
helper describes the ordinary alignment path rather than these dedicated backing allocations.

### 2. Implement OOM recovery without undersized cache entries

Keep non-OOM failures out of reclamation. On a rounded-allocation OOM, attempt an exact-size uncached
allocation before evicting retained entries. Avoid a duplicate exact-size attempt when the original
allocation already used that size. On OOM, check for a newly cached matching full-bucket entry before
reclaiming. Under the mutex, reclaim only if current retained capacity covers the exact backing size,
detaching oldest entries until their aggregate capacity covers it; no thread can refill a detached
entry. Wait and free outside the mutex, then perform at most one final exact-size allocation attempt.
Do not repeat recovery or return to rounded-size allocation. Update public documentation and the
existing impossible-allocation test: a tiny retained cache survives an impossible request, including
when a very large configured cache limit would nominally make the request cacheable.

Account for a possible whole-bucket overshoot when selecting the last victim. The eligibility test
and detach operation share the same critical section. Reclamation does not guarantee CUDA success,
and failure to confirm completion withholds storage rather than counting it as successfully freed.

### 3. Make cache insertion transactional

Reserve the incoming entry's list node before detaching victims. Once metadata exists, reserve room
and publish under one mutex acquisition, then release detached victims outside the mutex. Preserve
the concurrency progress guarantee fixed by the earlier review. If room remains insufficient,
refuse to cache in Release as well as Debug. Do not implement this by overflowing byte accounting
while temporarily inserting a new entry.

### 4. Contain CUDA failures and define capture rejection

Treat failed stream-device discovery as unavailable completion tracking; do not substitute the
current device. Apply the accepted non-throwing cleanup policy, explicitly handling an unavailable
event and retaining storage whenever completion remains unknown. Make failed device restoration
observable through the agreed fatal path.

Retire events after unconfirmed completion. In the shared RAII wrapper, `reset_no_throw()` exchanges
the handle with null before attempting destruction, returns the status, and clears its own CUDA
failure. Empty wrappers reset successfully without calling CUDA. The destructor and move assignment
share this path. Test callers outside this allocator because the wrapper's cleanup behavior changes.
Do not manually destroy a handle while its wrapper still owns it. Event retirement must also adjust
the event pool's live-event accounting so repeated discard/create cycles do not grow reserved
capacity indefinitely. Never retry destruction of an ambiguously destroyed handle.

Check capture status before allocation or cache operations. On a capturing stream throw from
allocation, and withhold storage from deallocation without CUDA work submission or reclamation. Mark
slab ownership where needed; dedicated buffers remain withheld and are not freed at destruction
without confirmed completion. A failed capture-status query also cannot justify unordered reuse.
Distinguish capture of the supplied stream from conflicting capture elsewhere on the calling thread;
the latter remains unsupported and existing fault tests may deliberately trigger it. Do not promise
safety for allocator calls inside CUDA host callbacks.

### 5. Quarantine only affected slab backing

Replace the global quarantine flag with backing-owner state. Locate the owner when deallocation
withholds a slab, and mark the owner on destructor wait failure. Return unrelated backing allocations
upstream. Preserve storage referenced by any slab whose completion remains unconfirmed.

### 6. Address contention with measured scope

Measure concurrent hits and full-cap eviction separately from the successful application workload,
whose reported peak retained capacity was about 36 MiB. Detach an entry under the mutex, submit its
wait after unlocking, and recycle the event only after the submission succeeds. Preserve tracking
and retained-byte accounting on failed submission; restoring an entry must respect a concurrently
refilled cap, and otherwise the affected storage must be withheld safely. Move cold event creation
outside the mutex while preserving non-allocating return-to-pool and device restoration guarantees.
Use separate event-pool coordination if needed rather than serializing unrelated cache/slab traffic
behind CUDA context initialization. Measure node pooling and victim-policy alternatives only if
their overhead warrants further changes.

### 7. Complete validation and review responses

Update the allocator header and PR description to match final recovery, alignment, failure, and
capture behavior. Respond to each review claim with the change, test evidence, or contract-based
reason for disagreement. Draft responses locally; posting requires user authorization.

Preserve the existing statistics fields and count requests once rather than per attempt. Clarify
cache eligibility versus actual dedicated/fallback allocation behavior in statistics and sizing
documentation; do not present retained capacity as total pinned memory. Describe source/ABI effects
for consumers and retain the cuDF wiring example.

## Files and commit boundaries

| Commit responsibility | Main files |
| --- | --- |
| Private deterministic failure harness | `test/CMakeLists.txt`, private helpers under `test/memory/`, narrowly scoped internal test-build hooks |
| Event ownership and non-throwing destruction | `include/cucascade/cuda/event.hpp`, `src/cuda/event.cpp`, focused event lifecycle tests |
| Exceptional pinned ownership and alignment | `include/cucascade/memory/small_pinned_host_memory_resource.hpp`, `src/memory/small_pinned_host_memory_resource.cpp`, allocator tests |
| Exact-size fallback and bounded OOM recovery | Allocator implementation, sizing/recovery documentation, allocator tests |
| Transactional insertion and isolated slab quarantine | Allocator owner/cache bookkeeping and allocator tests |
| Capture/device failure handling and narrower mutex scope | Allocator/event-pool implementation and CUDA fault/concurrency tests |
| Evidence and review responses | This plan, allocator API documentation, local response drafts and benchmark results |

Move supporting event-wrapper changes before allocator retirement uses them. Existing unrelated user
changes are preserved. The implementation does not require changing `idisk_io_backend` or the disk
I/O backends.

## Validation matrix

| Area | Required evidence |
| --- | --- |
| Rounded OOM | Deterministic rounded failure followed by exact-size success, without cache eviction |
| Fallback lifetime | Exact fallback is released safely and never reused as a larger bucket |
| Tracking failure | Failed metadata allocation leaks no newly allocated buffer and publishes no untracked pointer |
| Alignment | Default and over-aligned requests, awkward upstream block stride, alignment and padding overflow |
| OOM reclamation | Retained capacity below/equal/above exact size; conservative skip; bounded oldest-first detach/retry; impossible request; concurrent refill; non-OOM preservation |
| Insertion failure | List-node allocation failure preserves existing cache entries and releases the incoming buffer safely |
| Concurrent eviction | Full-cap mixed buckets and several deallocating threads finish without exceeding retained capacity |
| CUDA failures | Device discovery, event creation/record/wait/sync/destroy, free, device restoration, single-attempt destruction, bounded pool capacity after repeated retirement |
| Slab quarantine | One failed slab retains only its backing owner; independent owners return upstream |
| Capture boundary | Allocation rejection and deallocation withholding preserve capture validity and prevent cache publication/eviction |
| Readiness | Pending GPU work, cross-stream and cross-device reuse, CPU readiness through allocate_sync |
| Contention | Concurrent cache-hit latency and throughput; full-cap eviction and half-cap rounding workloads |

### Deterministic failure tests

Compile the actual allocator and event implementation into a dedicated test variant with private
scoped hooks at the selected CUDA calls and metadata-construction boundaries. Do not duplicate the
allocation decision logic in a test-only implementation or link both the production and test variant
definitions into one executable. The existing static CUDA runtime linkage makes `LD_PRELOAD` an
unsuitable default interception strategy.

Inject one-shot, scoped failures and record requested sizes and ownership transitions. Normal calls
forward to the real CUDA runtime. Cover both failure and successful-release controls. For the fatal
device-restoration policy, use a separately executed test process; do not fork an already initialized
CUDA context. Keep existing capture-based fault injection separate from tests that prove a capture
remains valid after explicit rejection.

### Correctness gate

Run formatting and warnings-as-errors builds in the applicable existing Release and Release-stable
environments, targeted `[small_pinned]` tests, the new fault/event lifecycle tests, and `[multi_gpu]`
cases on a host with at least two GPUs. Reuse the established four-GPU setup for application evidence
when available. Confirm default resource and cuDF integration builds; the dedicated test variant is
not a substitute for the production library tests. Missing hardware evidence is reported rather than
counted as passing. No new GPU tests have been executed during planning.

### Performance gate

Compare the implementation to `69ef173b10a75363d80aa3a8df833f2858159034`, using the same compiler,
configuration, hardware, data, and cache preparation. Run at least five paired baseline/candidate
trials for concurrent cache hits, half-cap rounding, full-cap eviction, and the previously reported
application workload. Alternate execution order to limit drift; record raw results, medians, and
observed variability. Separate instrumentation runs from timing runs.

For elapsed time and latency, regression is `(candidate - baseline) / baseline`; for throughput it is
`(baseline - candidate) / baseline`. A reproducible median regression above 5% requires investigation
and resolution or an explicitly agreed plan revision before completion. If results are too noisy to
classify, collect more paired trials rather than declare success. Keep performance evidence distinct
from ownership/correctness assertions. Node pooling and eviction-policy changes are conditional on
measured benefit and require updating this plan before expanding scope.

### Accepted revision: deterministic eviction (2026-11-03)

The user explicitly accepted the current implementation's measured `deterministic_full_cap` throughput regressions of 7.98% on ARM GB10 and 7.72% on x86_64 Xeon/RTX PRO 4500. This exception covers the single-thread, single-stream, 64 KiB cache control with repeated 9,000-byte and 40,000-byte allocations, as a tradeoff for correctness and measured concurrent throughput gains. It is limited to that recorded benchmark and validated implementation; future worsening is not covered. The 5% gate remains unchanged for other measured cases and future performance changes, and further regressions require investigation and resolution or another explicit plan revision. The subsequent application-comparison waiver below applies only to the unavailable original workload.

Two-GPU correctness is satisfied. At this acceptance checkpoint the original four-GPU application comparison remained required; the subsequent revision below waives it. The [validation record](pinned-host-cache-validation.md#accepted-performance-tradeoff-2026-11-03) retains the measured losses, acceptance rationale, and original evidence.

### Accepted revision: application comparison waiver (2026-11-03)

The user explicitly waived the original four-L40S application comparison after accepting the available evidence and reporting loss of access to that machine. The original application workload was not rerun, so application performance remains unmeasured. This waiver removes that comparison from the delivery requirements for validation ticket 11 and spec 1; the original protocol above remains historical guidance for optional future work.

The implementation and recorded validation satisfy the revised criteria. The measured deterministic eviction acceptance remains limited to its recorded case and implementation. Correctness contracts, other measured benchmark gates, and future performance requirements remain in force. The [application comparison waiver](pinned-host-cache-validation.md#application-comparison-waiver-2026-11-03) records the user's decision; final delivery review and issue closure follow separately.

## Review dispositions

| Comment | Planned disposition |
| --- | --- |
| [4445468542](https://github.com/NVIDIA/cuCascade/pull/185#discussion_r4445468542) | Exact fallback precedes reclamation; conservative retained-capacity gate and bounded oldest-first reclaim/retry. Correct the process-wide characterization. |
| [4445468555](https://github.com/NVIDIA/cuCascade/pull/185#discussion_r4445468555) | Reserve metadata before eviction; add defensive Release cap check. Evaluate FIFO policy separately from correctness. |
| [4445468567](https://github.com/NVIDIA/cuCascade/pull/185#discussion_r4445468567) | Preserve documented asynchronous readiness; add alignment support. Uninitialized contents are part of the contract. |
| [4445468576](https://github.com/NVIDIA/cuCascade/pull/185#discussion_r4445468576) | Keep matching-size precondition and zero-size no-op; no runtime recovery for mismatched caller sizes. Respond with the documented contract. |
| [4445468584](https://github.com/NVIDIA/cuCascade/pull/185#discussion_r4445468584) | Remove device guessing and test failed discovery. CUDA requires matching event/stream contexts. |
| [4445468592](https://github.com/NVIDIA/cuCascade/pull/185#discussion_r4445468592) | Fatal failed device restoration; cold event creation moves outside the cache mutex. |
| [4445468606](https://github.com/NVIDIA/cuCascade/pull/185#discussion_r4445468606) | Define capture rejection and preserve safe ordinary reclamation. Callback CUDA API use remains unsupported on every path. |
| [4445468615](https://github.com/NVIDIA/cuCascade/pull/185#discussion_r4445468615) | Retire suspect events instead of returning them to the pool after failed completion checks. |
| [4445468629](https://github.com/NVIDIA/cuCascade/pull/185#discussion_r4445468629) | Quarantine individual backing owners and handle shared event-wrapper destruction errors. |
| [4445468638](https://github.com/NVIDIA/cuCascade/pull/185#discussion_r4445468638) | Extract hit wait submission from the mutex and measure contention. Node pooling and changed eviction policy require measured benefit. Existing hit removes a node rather than allocating one. |

## Original planning completion checkpoint

The user accepted every interview decision. The planning task is complete: the implementation
sequence, all ten review dispositions, failure policies, and validation gates are recorded. Only
documentation has been edited. The user subsequently authorized a general GSD bypass, so GSD is
optional. Implementation remains a subsequent task; approval of this plan is not evidence that its
implementation has passed checks.
