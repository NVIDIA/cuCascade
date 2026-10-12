# Independent allocator failure injection

This external harness contains no production hooks. `libpinned_faults.so` interposes CUDA runtime
functions through `LD_PRELOAD`; the driver injects C++ container allocation failures through its
global `operator new`. The driver compiles the exact final production allocator translation unit
and reads current repository headers, links shared cudart, and uses the candidate shared library
only for backing-resource implementation. The candidate shared library's CUDA entry points are
hidden/local because it statically links cudart; compiling the allocator into the executable is
essential for successful interposition. Each fault checks wrapper counters to prove interception.
Build does not execute CUDA code; running the driver uses GPU 0 and must use the root agent's GPU
lock.

Build from the shared workspace:

```bash
python3 /tmp/cucascade-pinned-pr-20261012/faults/build.py
```

Run all 20 cases with the root agent's existing GPU launcher and this additional environment variable:

```bash
LD_PRELOAD=/tmp/cucascade-pinned-pr-20261012/faults/libpinned_faults.so \
  /tmp/cucascade-pinned-pr-20261012/faults/fault_driver
```

Pass a printed case name as the sole driver argument to run that case separately. An unexpected
termination in a `noexcept` path is a test failure. The driver initializes CUDA before arming host
metadata allocation faults so lazy runtime initialization does not consume the intended fault.
Each case checks that its injected CUDA or host allocation failure actually occurred.

Coverage:

- Slab and large reuse waits propagate failure and preserve the candidate for retry.
- Event creation/recording plus fallback stream synchronization failures quarantine large pointers
  and slabs; slab backing blocks do not return upstream.
- Event synchronization failure during eviction, purge, or destruction never frees the affected
  large pointer.
- `cudaHostAlloc` OOM drains retained storage and retries exactly once, including retry failure and
  a simultaneous host metadata failure while recycling a purged event.
- The first three host metadata allocations of a fresh large bucket fail independently; deallocation
  survives, frees only after completion, removes any empty bucket, and accepts later insertions.
- Recycling an event after a successful allocation wait survives host metadata OOM.
- Record failure plus recycling metadata OOM safely falls back to synchronized release.
- Uncached deallocation synchronization failure retains the pointer.
- Interposition observes the actual allocator requesting portable and mapped flags together (3).
- A blocked host callback followed by H2D DMA proves normal destruction enters its event wait before
  freeing the source; the callback releases only after the wait is observed and copied data is checked.

Injected errors do not submit real failing GPU work. Deliberately quarantined pointers are observed
first and then manually reclaimed after a successful real stream drain. This keeps repeated runs
small while demonstrating that the allocator itself did not free unknown-completion storage.
Slab quarantine deliberately retains its upstream allocation handle; its fixture is short lived.
Asynchronous data-ordering correctness is covered by repository tests/root staging tests, not this
failure-path harness.
