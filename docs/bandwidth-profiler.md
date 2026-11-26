# Memory Space Bandwidth Profiler — Design

**Status:** Implemented. The profiler is in the `cucascade-cudf` library and the `chunked_resource_info` mixin is in the core library; the decisions below describe the current implementation
**Date:** 2026-04-21

## Purpose

Given a list of `memory_space*`, measure pairwise transfer throughput between them so engines can use the resulting bandwidth matrix for routing decisions (where to place / downgrade / upgrade data).

Measurements flow through the **existing converter registry and each disk space's `idisk_io_backend`** — results reflect the actual transfer paths engines will pay at runtime, including user-registered converters.

## Scope

- Measure GPU↔GPU, GPU↔HOST, HOST↔HOST as a full all-pairs matrix.
- Measure DISK↔{GPU, HOST} for every (disk, non-disk) pair. **No disk↔disk entries** — not a routing path engines care about.
- Measure **both directions** of every pair separately (`src→dst` and `dst→src`). PCIe, NVLink asymmetries, and especially disk read/write asymmetry must be visible.
- Pure end-to-end function. No caching, no state. Runs at init time.
- Does **not** go through `memory_reservation_manager` — converters allocate directly from each space's allocator. Caller is responsible for running this before production load begins.
- At least one GPU space is required: it is used to build the source data for every pair (see D-09).

## Decisions

### D-01. Transfer mechanism
Use the `representation_converter_registry` for every transfer. The registry is initialized (via `register_builtin_converters()` or user registrations) before the profiler runs. Each tier has a canonical representation that serves as both source and target type: GPU → `gpu_table_representation`, HOST → `host_data_representation`, DISK → `disk_data_representation`. A pair is marked `converter_available = false`, with the error in `unavailable_reason`, if either its src→dst converter or the GPU→src converter used to build the source (see D-09) is missing or fails.

**Why:** Engines route based on what converters will actually do at runtime. Raw `cudaMemcpyAsync` / `cuFileRead` numbers would diverge from production overhead (packing, metadata handling, backend selection).

### D-02. Disk backend selection
`measure_bandwidth` takes no backend parameter. Each DISK `memory_space` holds a `std::shared_ptr<idisk_io_backend>`: the one passed to the `memory_space(const disk_memory_space_config&, std::shared_ptr<idisk_io_backend>)` constructor, or the pipeline backend with the config-only constructor. The built-in disk converters resolve the backend from the disk space at conversion time. Profiler results are scoped to the backend of each disk space; profiles taken with different backends are not interchangeable. To compare backends, profile a disk space built with each.

### D-03. Chunked-allocator detection — `chunked_resource_info` mixin
The interface lives at `include/cucascade/memory/chunked_resource_info.hpp`:

```cpp
namespace cucascade {
namespace memory {

struct chunked_resource_info {
  virtual ~chunked_resource_info() = default;
  [[nodiscard]] virtual std::size_t max_chunk_bytes() const = 0;
};

}  // namespace memory
}  // namespace cucascade
```

- Inherited by any memory resource that hands out fixed-size chunks rather than arbitrary contiguous ranges. Allocators that don't inherit it are treated as contiguous.
- `fixed_size_host_memory_resource` inherits the mixin; `max_chunk_bytes()` forwards to the existing `get_block_size()`.
- `memory_space::get_chunked_resource_info()` returns the mixin when the space's reservation allocator inherits it, and `nullptr` otherwise. In practice HOST spaces are chunked; GPU and DISK spaces are not.
- The profiler records the result per pair in `src_max_chunk_bytes` / `dst_max_chunk_bytes` (0 for contiguous allocators).
- The mixin is opt-in, non-intrusive, and does not alter the allocator contract. Any future allocator (GPU-side arenas, disk chunking) can opt in the same way.

### D-04. Test-size semantics for chunked allocators
The profiler does not split transfers by chunk size. Every iteration is a single converter call for the full test size `N`, regardless of whether either endpoint is chunked. This matches production behavior: real converters already handle the per-chunk dispatch for chunked allocators. `bytes_transferred` reports the requested test size `N`.

### D-05. Result shape
Both per-size detail and an aggregated summary (so engines have a single number for routing, and the full curve is available for diagnostics). Declared in `include/cucascade/cudf/bandwidth_profiler.hpp`:

```cpp
namespace cucascade {
namespace data {

struct bandwidth_sample {
  double gbps                   = 0.0;  // GB/s (10^9 bytes) for this (pair, size)
  double mean_seconds           = 0.0;  // mean wall-clock time per iteration
  std::size_t bytes_transferred = 0;    // bytes moved per iteration (the test size)
  std::size_t iterations_timed  = 0;    // number of timed iterations aggregated
};

struct bandwidth_pair_result {
  memory::memory_space_id src{memory::Tier::GPU, 0};
  memory::memory_space_id dst{memory::Tier::GPU, 0};
  std::size_t src_max_chunk_bytes = 0;  // 0 if contiguous (see D-03)
  std::size_t dst_max_chunk_bytes = 0;
  std::map<std::size_t, bandwidth_sample> per_size;  // keyed by test size in bytes
  bandwidth_sample summary;  // representative value (see D-06)
  bool converter_available = true;  // false => no converter, or the pair failed
  std::string unavailable_reason;   // set when converter_available == false
};

struct bandwidth_profile {
  std::vector<bandwidth_pair_result> pairs;  // asymmetric: src→dst is distinct from dst→src
  // Convenience accessors:
  [[nodiscard]] double gbps(memory::memory_space_id src,
                            memory::memory_space_id dst) const noexcept;  // summary, or 0
  [[nodiscard]] std::optional<bandwidth_sample> sample(memory::memory_space_id src,
                                                       memory::memory_space_id dst,
                                                       std::size_t size_bytes) const;
  [[nodiscard]] const bandwidth_pair_result* find(memory::memory_space_id src,
                                                  memory::memory_space_id dst) const noexcept;
};

}  // namespace data
}  // namespace cucascade
```

`gbps()` returns 0 when the pair is missing or unavailable; `sample()` and `find()` return `std::nullopt` / `nullptr` when there is no match.

### D-06. Summary-value aggregation
The `summary` field is the per-size sample with the **median gbps across all measured sizes** for that pair (the upper of the two middle values for an even number of sizes). Rationale: median is robust to outliers at either end of the size curve (small-size latency-dominated, largest-size capacity-constrained) while still reflecting realistic throughput. Unavailable pairs keep a zeroed summary.

### D-07. Default measurement config
```cpp
struct bandwidth_profile_config {
  std::vector<std::size_t> test_sizes_bytes{
    1ull << 20,    //   1 MiB
    16ull << 20,   //  16 MiB
    64ull << 20,   //  64 MiB
    256ull << 20,  // 256 MiB
  };
  std::size_t warmup_iterations = 3;
  std::size_t timed_iterations = 10;
  bool measure_disk_pairs = true;  // disable to skip slow disk paths
  bool drop_page_cache_between_iters = true;
};
```

- Multiple sizes chosen to surface the small-vs-large transfer curve without dominating init cost.
- Each timed iteration is clocked individually around the converter call and stream synchronization; `mean_seconds` is the average over `timed_iterations`.
- Disk measurements are opt-out because they're orders of magnitude slower than GPU/HOST pairs.
- `drop_page_cache_between_iters` evicts a disk source file from the Linux page cache (`fdatasync` + `posix_fadvise(POSIX_FADV_DONTNEED)`) after each warmup and timed iteration, outside the timed region, so reads hit the disk. Disable it to measure warm-cache behavior.

### D-08. Bidirectional measurement
Every ordered pair `(A, B)` of spaces is measured as an independent entry, so `A→B` and `B→A` both appear. This includes GPU→GPU and HOST→HOST pairs across distinct device ids. For disk pairs, both directions are measured because write ≠ read throughput on NVMe. Self-pairs (same `memory_space_id`) and DISK→DISK pairs are skipped, as are all pairs with a DISK endpoint when `measure_disk_pairs` is false.

### D-09. Entry point
Pure function:
```cpp
namespace cucascade {
namespace data {

[[nodiscard]] bandwidth_profile measure_bandwidth(std::span<memory::memory_space* const> spaces,
                                                  const representation_converter_registry& registry,
                                                  const bandwidth_profile_config& config = {});

}  // namespace data
}  // namespace cucascade
```

- No caching, no internal state. Call it at init, hand the result to engines.
- Throws `std::invalid_argument` if `spaces` contains no GPU space. Null entries in `spaces` are ignored.
- For each pair and test size, the first GPU space in `spaces` builds a single-column INT32 cudf table of the test size, which the registry converts into the source space. Building the source is not timed.
- Errors in individual pairs (missing converter, OOM, transfer failure) mark that pair as `converter_available = false`, record the exception message in `unavailable_reason`, clear `per_size` and skip its remaining sizes; they do not abort the whole profile.

### D-10. Header locations
- `include/cucascade/memory/chunked_resource_info.hpp` — the mixin trait (memory layer, core library).
- `include/cucascade/cudf/bandwidth_profiler.hpp` — `bandwidth_sample`, `bandwidth_pair_result`, `bandwidth_profile`, `bandwidth_profile_config`, `measure_bandwidth()`.

The profiler is part of `cucascade-cudf` (`cuCascade::cucascade_cudf`) because it builds cudf tables and uses the cudf-backed representations, but its symbols stay in the `cucascade::data` namespace. Direction of dependency: cudf layer → data → memory (consistent with existing architecture).

### D-11. Stream usage
Each pair's warmup and timed conversions use one stream from `memory_space::acquire_stream()` (the GPU space's stream pool): the source space's if it is a GPU, otherwise the destination's if it is a GPU, otherwise the first GPU space's. That stream is reused for every size and iteration of the pair, and the current CUDA device is set to its GPU while measuring. Building the source (untimed) uses a separately acquired stream from the first GPU space, with that GPU as the current device. The profiler never uses the default CUDA stream.

### D-12. Thread safety
`measure_bandwidth` itself is not required to be reentrant — it's a one-shot init call. Callers must not mutate the converter registry or the passed-in memory spaces during execution. Pairs, sizes and iterations are measured sequentially, and each converter call is synchronized before the next one starts.

## Code Touched

- `include/cucascade/memory/fixed_size_host_memory_resource.hpp` — `: public chunked_resource_info` and a `max_chunk_bytes()` override that forwards to `get_block_size()`.
- `include/cucascade/memory/chunked_resource_info.hpp` — **new**, the mixin interface.
- `include/cucascade/memory/memory_space.hpp` / `src/memory/memory_space.cpp` — `get_chunked_resource_info()` probe.
- `include/cucascade/cudf/bandwidth_profiler.hpp` — **new**, public API.
- `src/cudf/bandwidth_profiler.cpp` — **new**, implementation.
- `test/data/test_bandwidth_profiler.cpp` — **new**, Catch2 tests (tag `[bandwidth_profiler]`, in `cucascade_cudf_tests`) using mock memory spaces and the built-in converters. The hidden `[.bandwidth_matrix]` case prints the full matrix for the available GPUs, HOST and DISK; run it with `cucascade_cudf_tests "[bandwidth_matrix]"`.
- Consumers: whatever engine code will call `measure_bandwidth()` at init and thread results into routing. Out of scope for this design — the profiler just returns the data.

## Deferred / Out of Scope

- **Caching or persistence of profiles across runs.** Bandwidth can drift (thermal, NUMA pinning, concurrent workload); callers that want persistence build it on top.
- **Dynamic re-profiling under load.** This is an init-time tool.
- **Concurrency bandwidth** (what happens when multiple engines transfer at once). Future work.
- **Raw-hardware baseline mode** (bypassing converters). Can be added as an alternative `measure_bandwidth_raw()` later if needed for debugging converter overhead.
- **Chunked disk allocators.** If `disk_access_limiter` grows chunked allocation semantics, it can opt into `chunked_resource_info` without touching the profiler.
