# Suggested review comments and follow-ups for NVIDIA/cuCascade PR 185 (head 32dcbce7)

Source: SPEC.md section 7 (pinned-slab-pool). Comments are drafted for the user to post themselves;
nothing has been posted. Ordered by importance for Sirius' multi-GPU use.

## 1. Add allocation counters (G2) -- `include/cucascade/memory/small_pinned_host_memory_resource.hpp`
The cache is sized by one number, but the only thing a consumer can observe is
`large_cache_bytes()`. That leaves no way to tell whether the limit fits the workload: a too-small
limit only shows up as `cudaHostAlloc` / `cudaFreeHost` calls in a profiler.

Suggest a const snapshot API that is cheap to call at shutdown:

```cpp
struct statistics {
  std::size_t cached_large_bytes;      // == large_cache_bytes()
  std::size_t peak_cached_large_bytes;
  std::size_t large_hits;              // served from the cache
  std::size_t large_misses;            // cudaHostAlloc on a cacheable bucket
  std::size_t large_uncacheable;       // bucket > limit (exact-size cudaHostAlloc)
  std::size_t large_evictions;         // cudaFreeHost of a cached entry (FIFO or purge)
};
[[nodiscard]] statistics get_statistics() const;
```

Hits, misses and evictions are maintained under the existing mutex. The uncacheable counter is
incremented on a lock-free path, so make it atomic. Sirius would log this per NUMA pool at shutdown.

## 2. Document what the limit means in practice (G1) -- class docs, lines 37-43
The limit bounds only *free* cached buffers; live ones are uncounted. A new buffer is allocated only
when its bucket has nothing cached, so the cache needs the **sum over buckets of each bucket's peak
concurrency**.

When a query ends, everything is freed at once. If that sum is above the limit, FIFO eviction
`cudaFreeHost`s the excess, and the next query pays the same number of `cudaHostAlloc` calls. The
driver-lock serialisation the PR removes then comes back, once per excess buffer per query.

Suggest:
- one sentence of sizing guidance in the class docs and the constructor `@param`;
- a note that one resource may be shared by several GPUs, as Sirius does with one instance per NUMA
  node; the PR's measurements were single-GPU.

## 3. NUMA placement of large buffers (G3, follow-up) -- `src/memory/small_pinned_host_memory_resource.cpp:149,168`
Slabs come from the NUMA-bound `fixed_size_host_memory_resource`. Large buffers use plain
`cudaHostAlloc(Portable|Mapped)`, so their pages follow the allocating thread's NUMA policy rather
than the upstream pool's node.

Consider allocating cacheable buckets from `upstream_.get_upstream_resource()`, which is
`numa_region_pinned_host_memory_resource`: it is NUMA-bound and keeps the `Portable|Mapped` flags.
The upstream's extra `cudaHostRegister` cost is paid only on a miss.

Fine as a follow-up; on a single-socket host it changes nothing.

## 4. Behaviour changes worth stating in the PR body
- The constructor stays source-compatible, but **default-constructed instances now cache** up to
  256 MiB of free large buffers and buckets up to 256 MiB.
- When the bucket is above the limit, or no event can be recorded, the free path now **synchronizes
  the freeing stream** before `cudaFreeHost`. Previously it called `cudaFreeHost` directly. This blocks
  the freeing thread (`cudaFreeHost` already synchronized implicitly).
- Requests above 8 KiB are rounded up to a power of two of at least 16 KiB, which can cost up to
  2x the live bytes for large buffers.
- `alignment` remains unused. The header already says so, which is fine.

## 5. Merge hygiene and PR body (G5)
- Commits 68440d1 and b2ded83 add about 13k lines under `reports/`, including about 6.5 MB of
  `raw-evidence.tar.gz` / `diagnostic-evidence.tar.gz`. 32dcbce deletes them again. **Squash-merge
  only**, or rewrite the branch, so those blobs never reach `main`. 8b2d229 is also a merge commit.
- The PR body says "22 cases (10 new)", but the test file has 28 `TEST_CASE`s, including the
  `[stream_ordering]` and `[multi_gpu]` cases. Update the body.
- Mark it ready for review once the copy-pr-bot vetting has let CI run.

## 6. Offer multi-GPU evidence
- The PR's report states that multi-GPU is untested; its one `[multi_gpu]` case was skipped on GB10.
- From the 4x L40S box, post:
  - `cucascade_tests "[small_pinned]"`, including `[multi_gpu]`;
  - the Sirius A/B from SPEC section 7.4 (SF1000 4-GPU WARM/PINNED, warm-window
    `cudaHostAlloc`/`cudaFreeHost` counts).
- Sirius' campaign measured the same mechanism at -8.6 s (-18 %) on SF1000 4-GPU WARM. That is the
  strongest motivation for the PR, so include it once the port's A/B confirms equivalence.

## Not for this PR (Sirius-side, pre-existing)
Sirius' `numa_small_pinned_mr::allocate_sync` does not synchronize, and its comment "pool ignores
the stream arg" has been stale since event-ordered reuse. PR 185's own report flagged this. Track it
as a Sirius issue, not a cuCascade comment.
