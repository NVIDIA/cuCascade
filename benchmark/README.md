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
# Run only GPU to HOST conversion benchmarks
./benchmark/cucascade_benchmarks --benchmark_filter=BM_ConvertGpuToHost

# Run only throughput benchmarks
./benchmark/cucascade_benchmarks --benchmark_filter=Throughput

# Run only roundtrip benchmarks
./benchmark/cucascade_benchmarks --benchmark_filter=Roundtrip
```

### Benchmark Output Options

Google Benchmark provides various output formats:

```bash
# Output in JSON format
./benchmark/cucascade_benchmarks --benchmark_format=json --benchmark_out=results.json

# Output in CSV format
./benchmark/cucascade_benchmarks --benchmark_format=csv --benchmark_out=results.csv

# Show more detailed statistics
./benchmark/cucascade_benchmarks --benchmark_repetitions=10

# Control the number of iterations
./benchmark/cucascade_benchmarks --benchmark_min_time=5.0  # Run each benchmark for at least 5 seconds
```

### Using CMake Target

You can also use the CMake target to run benchmarks:

```bash
# From the build directory
cmake --build . --target run_benchmarks
```

## Available Benchmarks

### Representation Converter Benchmarks

Located in `benchmark_representation_converter.cpp`:

1. **BM_ConvertGpuToHost**: Benchmarks GPU to HOST memory conversion with varying data sizes
   - Tests with different row counts (1K to 1M rows)
   - Tests with different column counts (2, 4, 8 columns)
   - Reports throughput in bytes/second

2. **BM_ConvertHostToGpu**: Benchmarks HOST to GPU memory conversion
   - Similar parameterization as GPU to HOST
   - Measures upload performance

3. **BM_RoundtripGpuHostGpu**: Benchmarks complete roundtrip conversion
   - GPU → HOST → GPU
   - Useful for measuring total overhead in data migration scenarios

4. **BM_ConvertHostToHost**: Benchmarks HOST to HOST conversion (cross-device copy on CPU)
   - Tests CPU-side memory copying performance

5. **BM_GpuToHostThroughput**: Focuses on memory bandwidth for GPU→HOST transfers
   - Tests with data sizes from 1 MB to 1 GB
   - Reports throughput in GB/s

6. **BM_HostToGpuThroughput**: Focuses on memory bandwidth for HOST→GPU transfers
   - Similar parameterization as GPU to HOST throughput
   - Reports throughput in GB/s

## Interpreting Results

The benchmark output shows:
- **Time**: Time per iteration in milliseconds
- **CPU**: CPU time used
- **Iterations**: Number of times the benchmark was run
- **bytes_per_second**: Throughput (automatically calculated)
- **rows**: Number of rows in the table
- **columns**: Number of columns in the table
- **bytes**: Total bytes transferred
- **throughput_GB/s**: Bandwidth in GB/s (for throughput benchmarks)

Example output:
```
------------------------------------------------------------------------------
Benchmark                            Time             CPU   Iterations bytes_per_second rows columns bytes
------------------------------------------------------------------------------
BM_ConvertGpuToHost/1000/2       0.123 ms        0.123 ms         5678      123.45M/s  1000       2 15240
BM_ConvertGpuToHost/10000/2      0.456 ms        0.456 ms         1534      456.78M/s 10000       2 152400
```

## Performance Considerations

- **Memory Bandwidth**: GPU-CPU transfers are limited by PCIe bandwidth (typically 12-16 GB/s for PCIe 3.0 x16)
- **Data Size**: Smaller transfers have higher per-byte overhead due to transfer setup costs
- **Column Count**: More columns may result in more complex packing/unpacking operations
- **CUDA Stream**: All benchmarks use synchronous operations to ensure accurate timing

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

## Requirements

- CUDA-capable GPU
- CUDA Toolkit
- Google Benchmark (automatically fetched by CMake)
- All dependencies required by cuCascade

## Troubleshooting

**Benchmark crashes or fails to find GPU:**
- Ensure you have a CUDA-capable device
- Check that CUDA drivers are properly installed
- Verify GPU is accessible: `nvidia-smi`

**Memory allocation errors:**
- Large benchmarks (1GB+) require sufficient GPU memory
- Reduce benchmark size using filters to skip large tests

**Inconsistent results:**
- Ensure no other GPU workloads are running
- Use `--benchmark_repetitions=N` for more stable results
- Run in Release mode for accurate performance measurements
