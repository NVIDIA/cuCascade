# cuCascade Benchmarks

This directory contains performance benchmarks for the cuCascade library using Google Benchmark.

## Building the Benchmarks

The benchmarks are built by default when you configure the project (when the suite
contains sources — see [Available Benchmarks](#available-benchmarks)). To disable them:

```bash
cmake -DCUCASCADE_BUILD_BENCHMARKS=OFF ..
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

To run a subset of benchmarks, use a filter pattern:

```bash
./benchmark/cucascade_benchmarks --benchmark_filter=<pattern>
```

## Available Benchmarks

> **None currently.** The previous representation-converter and throughput benchmarks
> (`BM_ConvertGpuToHost`, `BM_ConvertHostToGpu`, `BM_GpuToHostThroughput`,
> `BM_HostToGpuThroughput`) were cuDF-dependent and were removed together with the
> cuDF-backed data representations (issue #142). `benchmark/CMakeLists.txt` currently
> sets an empty `BENCHMARK_SOURCES` and returns early, so the `cucascade_benchmarks`
> target is skipped until new cuDF-free benchmarks (e.g. raw-buffer disk I/O) are added.

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
