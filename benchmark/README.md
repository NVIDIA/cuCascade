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
