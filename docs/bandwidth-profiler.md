# Memory Space Bandwidth Profiler — Design

**Status:** Design locked, pending implementation
**Date:** 2026-04-21

## Purpose

Given a list of `memory_space*`, measure pairwise transfer throughput between them so engines can use the resulting bandwidth matrix for routing decisions (where to place / downgrade / upgrade data).

Measurements flow through the **existing converter registry and `disk_io_backend`** — results reflect the actual transfer paths engines will pay at runtime, including user-registered converters.

## Scope

- Measure GPU↔GPU, GPU↔HOST, HOST↔HOST as a full all-pairs matrix.
- Measure DISK↔{GPU, HOST} for every (disk, non-disk) pair. **No disk↔disk entries** — not a routing path engines care about.
- Measure **both directions** of every pair separately (`src→dst` and `dst→src`). PCIe, NVLink asymmetries, and especially disk read/write asymmetry must be visible.
- Pure end-to-end function. No caching, no state. Runs at init time.
- Does **not** go through `memory_reservation_manager` — allocates directly from each space's allocator. Caller is responsible for running this before production load begins.

## Decisions

### D-01. Transfer mechanism
Use `representation_converter_registry::convert<T>()` for every transfer. The registry is initialized (via `register_builtin_converters()` or user registrations) before the profiler runs. If a pair has no registered converter, that cell is left empty / marked unavailable.

**Why:** Engines route based on what converters will actually do at runtime. Raw `cudaMemcpyAsync` / `cuFileRead` numbers would diverge from production overhead (packing, metadata handling, backend selection).

### D-02. Disk backend selection
The profiler accepts an `std::shared_ptr<idisk_io_backend>` (same parameter as `register_builtin_converters`). Profiler results are scoped to that backend — a kvikIO profile is not interchangeable with a GDS profile. Callers wanting both profile twice.

### D-03. Chunked-allocator detection — `chunked_resource_info` mixin
New interface lives at `include/cucascade/memory/chunked_resource_info.hpp`:

```cpp
namespace cucascade::memory {

struct chunked_resource_info {
  virtual ~chunked_resource_info() = default;
  [[nodiscard]] virtual std::size_t max_chunk_bytes() const = 0;
};

}  // namespace cucascade::memory
```

- Inherited **alongside** `rmm::mr::device_memory_resource` by any resource that hands out fixed-size chunks rather than arbitrary contiguous ranges. Allocators that don't inherit it are treated as contiguous.
- `fixed_size_host_memory_resource` gets the mixin; `max_chunk_bytes()` forwards to the existing `get_block_size()`.
- The profiler probes with `dynamic_cast<chunked_resource_info const*>(mr)`; on hit, it iterates chunks; on miss, it performs a single contiguous allocation.
- The mixin is opt-in, non-intrusive, and does not alter the `device_memory_resource` contract. Any future allocator (GPU-side arenas, disk chunking) can opt in the same way.

### D-04. Test-size semantics for chunked allocators
When probing a chunked allocator at test size `N`:
- Compute `k = ceil(N / max_chunk_bytes)` chunk allocations.
- Issue `k` converter calls — one per chunk — and measure the aggregate transfer rate.
- This matches production behavior: a `data_batch` backed by a chunked allocator will already be chunked, and real converters already handle the per-chunk dispatch.

For contiguous allocators, a single allocation of size `N` is issued and one converter call measures the transfer.

### D-05. Result shape
Both per-size detail and an aggregated summary (so engines have a single number for routing, and the full curve is available for diagnostics):

```cpp
namespace cucascade::data {

struct bandwidth_sample {
  double gbps;                    // GB/s for this (pair, size)
  std::chrono::duration<double> mean_transfer_time;
  std::size_t bytes_transferred;  // effective bytes after chunk rounding
};

struct bandwidth_pair_result {
  memory_space_id src;
  memory_space_id dst;
  std::map<std::size_t, bandwidth_sample> per_size;  // keyed by test size in bytes
  bandwidth_sample summary;  // representative value (see D-06)
  bool converter_available;  // false => pair has no registered converter
};

struct bandwidth_profile {
  std::vector<bandwidth_pair_result> pairs;  // asymmetric: src→dst is distinct from dst→src
  // Convenience accessors:
  [[nodiscard]] double gbps(memory_space_id src, memory_space_id dst) const;  // uses summary
  [[nodiscard]] std::optional<bandwidth_sample>
    sample(memory_space_id src, memory_space_id dst, std::size_t size_bytes) const;
};

}  // namespace cucascade::data
```

### D-06. Summary-value aggregation
The `summary` field reports the **median gbps across all measured sizes** for that pair. Rationale: median is robust to outliers at either end of the size curve (small-size latency-dominated, largest-size capacity-constrained) while still reflecting realistic throughput.

### D-07. Default measurement config
```cpp
struct bandwidth_profile_config {
  std::vector<std::size_t> test_sizes_bytes{
    1ull << 20,   //   1 MiB
    16ull << 20,  //  16 MiB
    64ull << 20,  //  64 MiB
    256ull << 20  // 256 MiB
  };
  std::size_t warmup_iterations = 3;
  std::size_t timed_iterations  = 10;
  std::chrono::milliseconds min_measurement_time{50};  // loop iterations until hit
  bool measure_disk_pairs = true;  // disable to skip slow disk paths
};
```

- Multiple sizes chosen to surface the small-vs-large transfer curve without dominating init cost.
- `min_measurement_time` guards against cases where `timed_iterations` finishes faster than timer resolution.
- Disk measurements are opt-out because they're orders of magnitude slower than GPU/HOST pairs.

### D-08. Bidirectional measurement
For every unique unordered pair `{A, B}` where at least one is non-disk (or both are non-disk), measure `A→B` and `B→A` as two independent entries in the result. For disk pairs, both directions are measured because write ≠ read throughput on NVMe. Self-pairs (`A→A`) are skipped.

### D-09. Entry point
Pure function:
```cpp
namespace cucascade::data {

[[nodiscard]] bandwidth_profile measure_bandwidth(
  std::span<memory_space* const> spaces,
  representation_converter_registry const& registry,
  std::shared_ptr<idisk_io_backend> disk_backend = nullptr,
  bandwidth_profile_config const& config = {});

}  // namespace cucascade::data
```

- No caching, no internal state. Call it at init, hand the result to engines.
- `disk_backend` may be null if no DISK spaces are present; required when any DISK space is passed in.
- Errors in individual pairs (OOM, transfer failure) mark that pair as `converter_available = false` with zeroed samples; they do not abort the whole profile.

### D-10. Header locations
- `include/cucascade/memory/chunked_resource_info.hpp` — the mixin trait (memory layer).
- `include/cucascade/data/bandwidth_profiler.hpp` — `bandwidth_profile`, `bandwidth_profile_config`, `measure_bandwidth()` (data layer, depends on converter registry + memory layer).

Direction of dependency: data → memory (consistent with existing architecture).

### D-11. Stream usage
Each transfer uses a stream acquired from the destination space's `exclusive_stream_pool` (policy `BLOCK`). The profiler never uses the default CUDA stream. Streams are released as the pool handle goes out of scope between iterations.

### D-12. Thread safety
`measure_bandwidth` itself is not required to be reentrant — it's a one-shot init call. Callers must not mutate the converter registry or the passed-in memory spaces during execution. Internal transfers use per-pair streams, so individual converter calls run concurrently where the registry supports it.

## Existing Code Touched

- `include/cucascade/memory/fixed_size_host_memory_resource.hpp` — add `: public chunked_resource_info` and an override that forwards to `get_block_size()`.
- `include/cucascade/memory/chunked_resource_info.hpp` — **new**, the mixin interface.
- `include/cucascade/data/bandwidth_profiler.hpp` — **new**, public API.
- `src/data/bandwidth_profiler.cpp` — **new**, implementation.
- `test/data/test_bandwidth_profiler.cpp` — **new**, Catch2 tests using mock memory spaces + mock converters.
- Consumers: whatever engine code will call `measure_bandwidth()` at init and thread results into routing. Out of scope for this design — the profiler just returns the data.

## Deferred / Out of Scope

- **Caching or persistence of profiles across runs.** Bandwidth can drift (thermal, NUMA pinning, concurrent workload); callers that want persistence build it on top.
- **Dynamic re-profiling under load.** This is an init-time tool.
- **Concurrency bandwidth** (what happens when multiple engines transfer at once). Future work.
- **Raw-hardware baseline mode** (bypassing converters). Can be added as an alternative `measure_bandwidth_raw()` later if needed for debugging converter overhead.
- **Chunked disk allocators.** If `disk_access_limiter` grows chunked allocation semantics, it can opt into `chunked_resource_info` without touching the profiler.
