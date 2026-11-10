# cuCascade Benchmarks

This directory contains performance benchmarks for the cuCascade library using Google Benchmark.

## Building the Benchmarks

The benchmarks are built by default when you configure the project. They exercise the cudf-backed
converters, so they are skipped when `CUCASCADE_BUILD_CUDF=OFF`. To disable them:

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

With pixi, `pixi run build` builds the release preset (including the benchmarks) into
`build/release`, and `pixi run benchmarks` runs `cucascade_benchmarks` from there.

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

3. **BM_ConvertGpuToHostFast**: Benchmarks GPU to `host_data_representation` conversion (fast path,
   no `cudf::pack`)
   - Similar parameterization as GPU to HOST

4. **BM_ConvertHostFastToGpu**: Benchmarks `host_data_representation` to GPU conversion
   - Similar parameterization as GPU to HOST

5. **BM_GpuToHostThroughput**: Focuses on memory bandwidth for GPU→HOST transfers
   - Tests with data sizes
   - Reports throughput in GiB/s

6. **BM_HostToGpuThroughput**: Focuses on memory bandwidth for HOST→GPU transfers
   - Similar parameterization as GPU to HOST throughput
   - Reports throughput in GiB/s

All benchmarks in this file measure different thread counts.
The multi-threading is explicitly implemented instead of relying on googlebenchmark's built-in threading functionality,
because that resulted in improper results.

### Disk Converter Benchmarks

Located in `benchmark_disk_converter.cpp`. These convert through a DISK memory space mounted at
`/tmp`. Write benchmarks fdatasync the file and drop its page cache inside the timed loop, so the
flush is included in the timing. The `*Pipeline` read benchmarks drop the cache before each read;
`BM_ConvertDiskToGpu` and `BM_ConvertDiskToHost` do not.

- **BM_ConvertGpuToDisk**, **BM_ConvertDiskToGpu**, **BM_ConvertHostToDisk**,
  **BM_ConvertDiskToHost**: numeric-column size sweeps between GPU/HOST and DISK
- **BM_ConvertGpuToDiskStringColumns**, **BM_ConvertGpuToDiskListColumns**,
  **BM_ConvertGpuToDiskStructColumns**: GPU to DISK with nested and variable-width columns
- **BM_ConvertGpuToDiskPipeline**, **BM_ConvertDiskToGpuPipeline**: numeric-column size sweeps
  between GPU and DISK
- **BM_{Write,Read}{Numeric,String,List,Struct}Pipeline**: write and read per column type

Despite the name, the `*Pipeline` benchmarks use the same I/O backend as the others (the one
configured for the DISK memory space).

Sizes larger than 75% of free GPU memory are skipped. The `gdsio_write_GiBs` and `gdsio_read_GiBs`
counters are fixed baselines measured with gdsio on another machine and recorded in the source file,
not measurements of the current machine.

### GPU to GPU Converter Benchmarks

Located in `benchmark_gpu_to_gpu_converter.cpp`. These require two GPUs with peer access and
otherwise report an error (`SkipWithError`) without running:

- **BM_ConvertGpuToGpu**: converts a fixed number of partitions from device 1 to device 0 on one
  thread, for varying row counts
- **BM_RawPeerCopy**: peer-copies the same number of bytes on one stream, as a lower bound for the
  converter

## Standalone Parquet Benchmarks

These are command-line programs, not Google Benchmark executables. Each prints its usage when run
without arguments.

Built when `CUCASCADE_BUILD_IO=ON` (the default) together with `CUCASCADE_BUILD_BENCHMARKS` and
`CUCASCADE_BUILD_CUDF`:

- `cucascade_parquet_benchmark`: full `cudf::io::read_parquet` using cudf's default datasource or
  the cucascade-io io_uring datasource
- `cucascade_parquet_io_benchmark`: raw column-chunk range reads into host or device memory,
  without parquet decode

Built only with `CUCASCADE_BUILD_S3_BENCHMARK=ON`, which needs aws-sdk-cpp. aws-sdk-cpp is not a
cuCascade dependency; use the `s3-bench` pixi environment and the `s3-bench` CMake preset, which
builds into `build/s3-bench`:

- `cucascade_s3_parquet_benchmark`: S3 parquet read using the cucascade-io REST datasource or
  kvikIO's remote S3 handle
- `cucascade_s3_parquet_io_benchmark`: the S3 analogue of `cucascade_parquet_io_benchmark`

```bash
# Builds cucascade_s3_parquet_benchmark
pixi run -e s3-bench build-s3-bench

# Builds the range-read variant
pixi run -e s3-bench cmake --build build/s3-bench --target cucascade_s3_parquet_io_benchmark
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

3. Add the source file to `BENCHMARK_SOURCES` in `CMakeLists.txt` if creating a new file

## Considerations
The HOST memory space configuration (`host_memory_space_config` in `include/cucascade/memory/config.hpp`) influences the results.
Its `block_size` determines the size of the individual transfers performed.
Its `pool_size` and `initial_number_pools` result in a certain amount of pinned host memory being available without needing to perform additional allocations.
If a benchmark transfers more data than that, performance will drop sharply.
The defaults are defined in `config.hpp`; the representation and disk converter benchmarks raise `initial_number_pools` to pre-allocate more pinned memory, but the largest disk converter cases still exceed it.
