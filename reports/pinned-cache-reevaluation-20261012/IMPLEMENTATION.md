# Pinned host allocator PR implementation

Merge commit: `8b2d229c1f7b064aad6db930bf7a9c396531e1ba`.

Developer ownership was limited to `include/cucascade/memory/small_pinned_host_memory_resource.hpp`, `src/memory/small_pinned_host_memory_resource.cpp`, and `test/memory/test_small_pinned_host_memory_resource.cpp`. The root agent merged `origin/main`, resolved the other compatibility wrappers, committed the merge, and owned every GPU execution. No GPU tests or benchmarks were run by the developer agent.

## Readings completed

Before implementation: `/home/kkristensen/.codex/agents/developer.toml`; repository `CLAUDE.md` and the complete saved main `AGENTS.md` at `/tmp/cucascade-pinned-pr-20261012/AGENTS-main.md`; `README.md`; `test/CMakeLists.txt`; and the complete cuDF C++ documentation guide at https://github.com/rapidsai/cudf/blob/main/cpp/doxygen/developer_guide/DOCUMENTATION.md, read through its raw GitHub content after the GitHub page timed out. The user authorized bypassing the repository GSD requirement before edits began.

`/home/kkristensen/AgentDocs/C++ Software Design.md`: broad table of contents and complete chapter 1, Guidelines 1-5, printed pages 1-42. The implementation applies separation of concerns, ownership encapsulation, testability, and avoiding unnecessary abstractions. `/home/kkristensen/AgentDocs/C++20 -- The Complete Guide.md`: broad table of contents and complete chapter 20, printed pages 520-543, including unsigned bit utilities and safe power-of-two rounding. `/home/kkristensen/AgentDocs/CppCoreGuidelines.md`: major-section table of contents and relevant complete rules R.1-R.23, CP.1-CP.3, CP.20-CP.22, CP.43-CP.50, E.1-E.6, E.13, E.16, C.21, C.36-C.37, and Per.1-Per.7. Comments retain the project's existing license boilerplate and use ASCII; clang-format 20.1.4 performed paragraph wrapping.

## Resulting behavior and rationale

The PR retains portable, mapped pinned allocations larger than 8192 bytes in power-of-two buckets, with a configurable default cap of 256 MiB per resource. A single bucket map containing FIFO deques supports matching reuse and global FIFO eviction by comparing bucket fronts. No second LRU index was added.

Main's stream-device discovery, creation of events on the freeing stream's owning device, fallback synchronization, event-pool allocation-failure handling, and slab quarantine/destructor protections survive the merge. Reuse checks `cudaStreamWaitEvent` before removing the slab or large entry, so a failed wait preserves pointer ownership and cache accounting. The async API remains stream-ordered; CPU callers synchronize the allocation stream or use `allocate_sync`, matching installed RMM and cuDF contracts.

Eviction, purge, and fallback release synchronize outside the allocator mutex. Unknown CUDA completion quarantines storage rather than freeing or publishing it. All optional container growth in noexcept deallocation is caught. Cache-cap arithmetic uses subtraction to avoid overflow. OOM purge detaches the existing map without constructing an allocating snapshot vector, releases its entries, and retries direct allocation once. Slab expansion reserves metadata for all outstanding slabs and establishes backing ownership before publishing raw pointers. Internal cache/event structures are private.

Focused tests add deterministic pending-DMA cross-stream reuse, synchronous CPU readiness, disabled caching with actual mapped GPU access, largest-size bucket arithmetic, FIFO eviction across buckets, and conditional freeing-stream device ownership coverage. A flag-query assertion was replaced after root established that the shared library and executable each contain hidden/local copies of the static CUDA runtime; querying allocation flags through a different runtime returned zero. Functional mapped access passed, and the independent wrapper observed the allocator actually requesting `cudaHostAllocPortable | cudaHostAllocMapped` (3). The production allocation flags were never changed.

## Build configuration

The existing CUDA 13.2 / RMM 26.08 / cuDF 26.08 Pixi environment was preserved; no install or update ran. `env.sh` configures the existing compiler, prefix, runtime paths, and CUDA architecture 121. Candidate builds use `build/pr-reevaluation`, Release, Ninja, warnings as errors, IO and cuDF enabled, static cuCascade libraries disabled, and existing Catch2/Google Benchmark/concurrentqueue source trees from `build/release/_deps`. `CMAKE_CXX_FLAGS=-isystem /home/kkristensen/Code/cuCascade/.pixi/envs/default/include` supplies the packaged NVTX include while avoiding project warning rules on third-party `numa.h`. All four requested targets compiled: `cucascade_tests`, `cucascade_cudf_tests`, `cucascade_io_tests`, and `cucascade_benchmarks`. Logs are `candidate-configure.log`, `candidate-build.log`, and `candidate-test-fix-build.log` in this report's directory.

## Root-executed validation

Run from `/home/kkristensen/Code/cuCascade`; the launcher sources the preserved environment, acquires `gpu.lock`, and waits for idle GPU samples.

```bash
CUCASCADE_TEST_GPU_POOL_BYTES=536870912 /tmp/cucascade-pinned-pr-20261012/gpu-run.sh build/pr-reevaluation/test/cucascade_tests '[small_pinned]' --reporter compact
CUCASCADE_TEST_GPU_POOL_BYTES=536870912 /tmp/cucascade-pinned-pr-20261012/gpu-run.sh build/pr-reevaluation/test/cucascade_tests '~[small_pinned]' --reporter compact
CUCASCADE_TEST_GPU_POOL_BYTES=536870912 /tmp/cucascade-pinned-pr-20261012/gpu-run.sh build/pr-reevaluation/test/cucascade_cudf_tests --reporter compact
CUCASCADE_TEST_GPU_POOL_BYTES=536870912 /tmp/cucascade-pinned-pr-20261012/gpu-run.sh build/pr-reevaluation/test/cucascade_io_tests --reporter compact
CUCASCADE_TEST_GPU_POOL_BYTES=536870912 /tmp/cucascade-pinned-pr-20261012/gpu-run.sh /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 build/pr-reevaluation/test/cucascade_tests '[small_pinned][stream_ordering]' --reporter compact
CUCASCADE_TEST_GPU_POOL_BYTES=536870912 /tmp/cucascade-pinned-pr-20261012/gpu-run.sh env LD_PRELOAD=/tmp/cucascade-pinned-pr-20261012/faults/libpinned_faults.so /tmp/cucascade-pinned-pr-20261012/faults/fault_driver
```

Final results: allocator 992 assertions, 27 passed cases and one multi-GPU skip; remaining core 2333 assertions / 89 passed cases; cuDF 2564 assertions, 187 passed cases and one skip; IO 1540 assertions / 139 passed cases; memcheck 12 assertions / two ordering cases with zero errors; independent CUDA/host-allocation fault injection 19 cases passed, with injection counters and requested flags verified. Evidence: `targeted-tests-final.log`, `core-tests.log`, `cudf-tests.log`, `io-tests.log`, `memcheck.log`, and `faults/results.log`. The NVIDIA GB10 provides one GPU, so real cross-device event behavior remains statically preserved and conditionally tested rather than empirically covered here.

## API, ABI, and retention considerations

The old one-argument constructor becomes a two-argument constructor with a defaulted cap. Existing source using one argument still compiles, but its C++ symbol changes; existing binary consumers must be recompiled and relinked. Caching is enabled by default and callers can opt out by passing zero, or choose an explicit smaller budget. The cap applies only to retained large buffers: it excludes live allocations, slab pools, and entries already detached for eviction or purge. Cache ownership ends on reuse; the reused storage remains allocated to the caller. Pinned storage is freed on eviction, OOM reclamation, or resource destruction after recorded work completes. Irrecoverable CUDA completion failures deliberately leak affected large storage or retain slab backing allocations to prevent unsafe reuse. Alignment selection remains inherited and unchanged. No cuDF pinned-allocation threshold or public memory-space policy was changed.


A standalone ThreadSanitizer stress harness was prepared and compiled, without execution, in `tsan/`. It instruments the exact production allocator translation unit and an eight-thread driver, with 1000 large allocate/write/deallocate cycles per thread, six request sizes, and a 512 KiB retention cap that forces cross-bucket eviction. The fixed upstream resource and CUDA/RMM libraries remain uninstrumented. `tsan/build-metadata.json` records the source hash and compiler invocation; execution belongs to the root agent and is not claimed as completed here.

ThreadSanitizer follow-up: the instrumented exact allocator translation unit and eight-thread harness passed 8000 large-cache cycles with no race reports under `setarch aarch64 -R` (process-local ASLR disabled). The default address layout failed at the sanitizer runtime with an unexpected mapping. CUDA/RMM and backing allocator internals were not instrumented. Evidence: `tsan/no-aslr.log` and `tsan/build-metadata.json`.
