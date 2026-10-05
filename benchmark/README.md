# cuCascade Benchmarks

This directory contains performance benchmarks for the cuCascade library using Google Benchmark.

## Building the Benchmarks

The benchmarks are built by default when you configure the project. To disable them:

```bash
cmake -DBUILD_BENCHMARKS=OFF ..
```

To build the project with benchmarks enabled:

```bash
# From the project root
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --target cucascade_benchmarks
```

## Running the Benchmarks

After building, you can run all benchmarks:

```bash
# From the build directory
./benchmark/cucascade_benchmarks
```

### Running Specific Benchmarks

To run only specific benchmarks, use filters:

```bash
# Run only conversion benchmarks
./benchmark/cucascade_benchmarks --benchmark_filter=Convert

# Run only throughput benchmarks
./benchmark/cucascade_benchmarks --benchmark_filter=Throughput
```

## Available Benchmarks

### Representation Converter Benchmarks

Located in `benchmark_representation_converter.cpp`:

1. **BM_ConvertGpuToHost**: Benchmarks GPU to HOST memory conversion with varying data sizes
   - Tests with different data sizes
   - Tests with different column counts
   - Reports throughput in bytes/second

2. **BM_ConvertHostToGpu**: Benchmarks HOST to GPU memory conversion
   - Similar parameterization as GPU to HOST
   - Measures upload performance

5. **BM_GpuToHostThroughput**: Focuses on memory bandwidth for GPU→HOST transfers
   - Tests with data sizes
   - Reports throughput in GiB/s

6. **BM_HostToGpuThroughput**: Focuses on memory bandwidth for HOST→GPU transfers
   - Similar parameterization as GPU to HOST throughput
   - Reports throughput in GiB/s

All benchmarks measure different thread counts.
The multi-threading is explicitly implemented instead of relying on googlebenchmark's built-in threading functionality,
because that resulted in improper results.

### I/O Write Benchmark (`cucascade_io_write_benchmark`)

Standalone CLI (not Google Benchmark) in `io_write_benchmark.cpp`, built when
`CUCASCADE_BUILD_IO=ON`. All arguments are `key=value`; run with `help` for the list.

- `mode=write`: writes `size` bytes in `block`-sized requests through the uring or kvikIO
  ioctx (`backend=uring|kvikio`, `source=host|device`, `runners`, `threads`, `qd`,
  `odirect=0|1`, `durability=none|data_sync`, `path=<dir>`), then `flush_async()`. Reports GB/s
  with and without the flush, interleaved with a dd-equivalent single-thread `pwrite` baseline
  and a parallel `pwrite` baseline (`baseline_threads`).
- `mode=mixed`: small-read latency (`read_size` at random aligned offsets of a cache-evicted
  file) on an idle context and while the write workload runs; reports p50/p99/max next to a
  `pread`-during-`pwrite` reference.
- `mode=readmix`: demand-read latency while a large background read runs on the same context
  (the prefetch-isolation scenario). See below.

```bash
./benchmark/cucascade_io_write_benchmark backend=uring source=host size=2G block=16M \
  runners=2 odirect=1 reps=5 pause_ms=10000 path=/mnt/nvme
```

Consumer SSDs absorb writes in an SLC cache; keep `size` below it and use `pause_ms` between
measurements, or results measure the drive's cache state rather than the I/O path.

#### uring scheduling knobs (all modes)

These set `io::uring::config` fields of the context under test (`backend=uring` only); the
uring reactor rejects invalid values with an error.

| Key | Config field | Default |
|---|---|---|
| `slices_per_pass=N` | `slices_per_pass`: slices of one request a runner plans per loop pass (0 = no cap, max 64) | 8 |
| `bg_groups=N` | `scheduling.max_background_groups` (>= 1) | 2 |
| `bg_share=F` | `scheduling.background_slot_fraction` ([0, 1]) | 0.75 |
| `bg_reserve=N` | `scheduling.reserved_background_slots` | 8 |

#### `mode=readmix`

Writes a demand file (`read_file_size`, default 1G) and a background file (`bg_size`,
default 2G), then:

1. **Background alone**, `reps` times: one `host_readv_async_io` of the whole background file in
   `bg_slice`-sized slices (default 1M) into one host buffer, with class `bg_class`.
   `bg_class=background` (default) is how `prefetching_cache::prefetch` classifies its reads;
   `bg_class=read` is what a prefetch was classified as before that (automatic class, which
   resolves to `read` above 256 KiB). Reports GB/s per repetition and the median.
2. **Per demand size** in `demand_sizes` (default `4K,1M,16M`): `idle_reads` demand reads on an
   idle context, then `reps` times: evict both files, submit the background read, issue demand
   reads one at a time until it resolves. Demand reads go to random 4 KiB aligned offsets with
   the automatic class: `latency` up to 256 KiB, `read` above, never `background`.
   - `demand_dst=host` (default) reads with `host_read_async` into a host buffer.
   - `demand_dst=device` reads with `device_read_async` into a `cudaMalloc` buffer, on a private
     stream that is synchronized after every read. These reads are staged through the pinned
     slots, so a 16 MiB read is one 16-slot operation. Only this path exercises how the engine
     admits large staged demand operations next to background work (background floor, FIFO
     capacity blocking). If no GPU is usable, the run prints a `SKIP ... reason=` line and
     exits with 0.

   Reports p50/p99/max per repetition with the concurrent background GB/s, and all repetitions
   pooled. A concurrent phase is short, so the pooled line has the useful p99.

Every page-cache eviction is `posix_fadvise(DONTNEED)`. Output is one key=value line per
measurement, so `grep '^RESULT'` over the logs of several runs gives a table:

- Every line repeats the run configuration: `backend runners odirect bg_class bg_size
  bg_slice slices_per_pass bg_groups bg_share bg_reserve demand_dst`.
- `RESULT ... phase=bg_alone rep=N bg_gbps=` and `phase=bg_alone_median`.
- `RESULT ... demand_size=1M demand_class=read phase=idle reads= p50_us= p99_us= max_us=`.
- `RESULT ... phase=concurrent rep=N reads= p50_us= p99_us= max_us= bg_gbps=`.
- `RESULT ... phase=concurrent_all reps= reads= p50_us= p99_us= max_us= bg_gbps_median=
  p99_x_idle= bg_x_alone=`. The last two are the pooled p99 over the idle p99 and the median
  concurrent background GB/s over the median standalone GB/s.
- `STATS` follows every `RESULT` except the summary lines. It covers that measurement's window
  of `ioctx::stats()`, with peaks reset at the start of the window:
  - For each class that retired requests: `<class>.first_io_n`, `first_io_mean_us`,
    `first_io_max_us` and `max_queue_wait_us`. A missing class means it had no requests.
  - `runner_max_inflight_ops` and `runner_submitted_mib`: one comma-separated value per
    runner. Both read 0 when the engine does not publish runner gauges.

```bash
# demand latency with prefetch-style background reads, vs the pre-tagging classification
for cls in background read; do
  ./benchmark/cucascade_io_write_benchmark mode=readmix path=/mnt/nvme runners=1 \
    bg_size=2G demand_sizes=4K,1M,16M reps=3 bg_class=$cls
done | grep -E '^RESULT.*phase=(bg_alone_median|idle|concurrent_all)'
```

Repeat the 16M demand size with `demand_dst=device` to cover staged demand operations. The knobs
above give the sensitivity sweeps, e.g. `bg_share=0.5`, `bg_reserve=0` or `slices_per_pass=0`.

### Parquet read benchmark (`cucascade_parquet_benchmark`)

`cucascade_parquet_benchmark <path|glob> <cudf|uring> <num_rows> [n_reactors] [odirect]
[slices_per_pass]` times one `cudf::io::read_parquet` of four TPC-H `lineitem` columns. The
data comes through cudf's default datasource (`cudf`) or the cucascade uring datasource
(`uring`). `n_reactors` (default 2), `odirect` (default 1) and `slices_per_pass` (default 8,
0 = no cap) configure the uring context.

### Parquet range-read benchmark (`cucascade_parquet_io_benchmark`)

`cucascade_parquet_io_benchmark <path|glob> <io_context|cudf> <host|device> <num_rows>
[n_threads] [slices_per_pass] [odirect]` times only the reads of the column-chunk byte ranges
the same projection touches, into host or device memory, over one uring context.

| Backend / dest | Read path |
|---|---|
| `io_context host` | one vectored `host_readv_async_io` per file and thread |
| `io_context device` | per-range `device_read_async` |
| `cudf host` / `cudf device` | the cucascade datasource's per-range `host_read_async` / `device_read_async` |

`n_threads` (default 2) sets both the reader threads and the uring runners. `slices_per_pass`
(default 8, 0..64) and `odirect` (default 1) configure the uring context. The two trailing
arguments are optional, so commands written for older builds still work unchanged. They come
in the opposite order to `cucascade_parquet_benchmark`'s `[odirect] [slices_per_pass]`.

Neither parquet benchmark evicts the page cache itself; both only write to
`/proc/sys/vm/drop_caches`, which needs root. For cold runs without root, evict the file
before each run:

```bash
python3 -c 'import os,sys;fd=os.open(sys.argv[1],os.O_RDONLY);os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)' <file>
```

## Adding New Benchmarks

To add new benchmarks:

1. Create a new benchmark function following the Google Benchmark API:
   ```cpp
   static void BM_YourBenchmark(benchmark::State& state) {
     // Setup code
     for (auto _ : state) {
       // Code to benchmark
     }
     // Optional: Report custom metrics
     state.SetBytesProcessed(...);
   }
   ```

2. Register the benchmark:
   ```cpp
   BENCHMARK(BM_YourBenchmark)->Args({param1, param2})->Unit(benchmark::kMillisecond);
   ```

3. Add the source file to `CMakeLists.txt` if creating a new file

## Considerations
There are some hard-coded configuration parameters in `fixed_size_host_memory_resource.hpp` that are of influence.
The block size defined there determines the size of the individual transfers performed.
The pool size and initial number of pools result in a certain amount of pinned host memory being available without needing to perform addition allocations.
If a benchmark transfers more data than that, performance will drop sharply.
