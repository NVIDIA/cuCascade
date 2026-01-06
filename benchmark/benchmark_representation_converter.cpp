/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "data/cpu_data_representation.hpp"
#include "data/gpu_data_representation.hpp"
#include "data/representation_converter.hpp"
#include "memory/memory_reservation_manager.hpp"
#include "utils/cudf_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <benchmark/benchmark.h>

#include <memory>
#include <vector>

namespace {

using namespace cucascade;
using namespace cucascade::memory;

/**
 * @brief Create memory manager configs for benchmarking (one GPU and one HOST).
 */
std::vector<memory_reservation_manager::memory_space_config> create_benchmark_configs()
{
  std::vector<memory_reservation_manager::memory_space_config> configs;
  // Large memory limits for benchmarking
  configs.emplace_back(
    Tier::GPU, 0, 4096ULL * 1024 * 1024, make_default_allocator_for_tier(Tier::GPU));
  configs.emplace_back(
    Tier::HOST, 0, 8192ULL * 1024 * 1024, make_default_allocator_for_tier(Tier::HOST));
  return configs;
}

/**
 * @brief Create a cuDF table with specified rows and columns for benchmarking.
 *
 * @param num_rows Number of rows
 * @param num_columns Number of columns (alternates between INT32, INT64, FLOAT32, FLOAT64)
 * @return cudf::table The generated table
 */
cudf::table create_benchmark_table(int64_t num_rows, int num_columns = 4)
{
  std::vector<std::unique_ptr<cudf::column>> columns;

  for (int i = 0; i < num_columns; ++i) {
    cudf::data_type dtype;
    uint8_t fill_value;

    // Cycle through different data types
    switch (i % 4) {
      case 0:
        dtype      = cudf::data_type{cudf::type_id::INT32};
        fill_value = 0x11;
        break;
      case 1:
        dtype      = cudf::data_type{cudf::type_id::INT64};
        fill_value = 0x22;
        break;
      case 2:
        dtype      = cudf::data_type{cudf::type_id::FLOAT32};
        fill_value = 0x33;
        break;
      case 3:
        dtype      = cudf::data_type{cudf::type_id::FLOAT64};
        fill_value = 0x44;
        break;
      default:
        dtype      = cudf::data_type{cudf::type_id::INT32};
        fill_value = 0x55;
        break;
    }

    auto col =
      cudf::make_numeric_column(dtype, static_cast<int>(num_rows), cudf::mask_state::UNALLOCATED);

    if (num_rows > 0) {
      auto view      = col->mutable_view();
      auto type_size = cudf::size_of(dtype);
      auto bytes     = static_cast<size_t>(num_rows) * static_cast<size_t>(type_size);
      RMM_CUDA_TRY(cudaMemset(const_cast<void*>(view.head()), fill_value, bytes));
    }

    columns.push_back(std::move(col));
  }

  return cudf::table(std::move(columns));
}

// =============================================================================
// GPU to HOST Conversion Benchmarks
// =============================================================================

/**
 * @brief Benchmark GPU to HOST conversion with varying data sizes.
 */
static void BM_ConvertGpuToHost(benchmark::State& state)
{
  int64_t num_rows = state.range(0);
  int num_columns  = static_cast<int>(state.range(1));
  auto mgr         = std::make_unique<memory_reservation_manager>(create_benchmark_configs());
  auto registry    = std::make_unique<representation_converter_registry>();
  register_builtin_converters(*registry);

  const memory_space* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  const memory_space* host_space = mgr->get_memory_space(Tier::HOST, 0);

  rmm::cuda_stream stream;

  // Create source data
  auto table    = create_benchmark_table(num_rows, num_columns);
  auto gpu_repr = std::make_unique<gpu_table_representation>(std::move(table),
                                                             *const_cast<memory_space*>(gpu_space));

  // Warm-up
  auto warmup_table = create_benchmark_table(100, 2);
  auto warmup_repr  = std::make_unique<gpu_table_representation>(
    std::move(warmup_table), *const_cast<memory_space*>(gpu_space));
  auto warmup_result =
    registry->convert<host_table_representation>(*warmup_repr, host_space, stream);
  stream.synchronize();

  size_t bytes_transferred = gpu_repr->get_size_in_bytes();

  // Benchmark loop
  for (auto _ : state) {
    auto host_result = registry->convert<host_table_representation>(*gpu_repr, host_space, stream);
    stream.synchronize();
  }

  // Report metrics
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_transferred));
  state.counters["rows"]    = static_cast<double>(num_rows);
  state.counters["columns"] = static_cast<double>(num_columns);
  state.counters["bytes"]   = static_cast<double>(bytes_transferred);
}

// Register benchmark with different data sizes
BENCHMARK(BM_ConvertGpuToHost)
  ->Args({1000, 2})     // 1K rows, 2 columns
  ->Args({10000, 2})    // 10K rows, 2 columns
  ->Args({100000, 2})   // 100K rows, 2 columns
  ->Args({1000000, 2})  // 1M rows, 2 columns
  ->Args({1000, 4})     // 1K rows, 4 columns
  ->Args({10000, 4})    // 10K rows, 4 columns
  ->Args({100000, 4})   // 100K rows, 4 columns
  ->Args({1000, 8})     // 1K rows, 8 columns
  ->Args({10000, 8})    // 10K rows, 8 columns
  ->Args({100000, 8})   // 100K rows, 8 columns
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// =============================================================================
// HOST to GPU Conversion Benchmarks
// =============================================================================

/**
 * @brief Benchmark HOST to GPU conversion with varying data sizes.
 */
static void BM_ConvertHostToGpu(benchmark::State& state)
{
  int64_t num_rows = state.range(0);
  int num_columns  = static_cast<int>(state.range(1));
  auto mgr         = std::make_unique<memory_reservation_manager>(create_benchmark_configs());
  auto registry    = std::make_unique<representation_converter_registry>();
  register_builtin_converters(*registry);

  const memory_space* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  const memory_space* host_space = mgr->get_memory_space(Tier::HOST, 0);

  rmm::cuda_stream stream;

  // Create source data: first make GPU repr, then convert to HOST
  auto table         = create_benchmark_table(num_rows, num_columns);
  auto gpu_repr_temp = std::make_unique<gpu_table_representation>(
    std::move(table), *const_cast<memory_space*>(gpu_space));
  auto host_repr = registry->convert<host_table_representation>(*gpu_repr_temp, host_space, stream);
  stream.synchronize();

  // Warm-up
  auto warmup_result = registry->convert<gpu_table_representation>(*host_repr, gpu_space, stream);
  stream.synchronize();

  size_t bytes_transferred = host_repr->get_size_in_bytes();

  // Benchmark loop
  for (auto _ : state) {
    auto gpu_result = registry->convert<gpu_table_representation>(*host_repr, gpu_space, stream);
    stream.synchronize();
  }

  // Report metrics
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_transferred));
  state.counters["rows"]    = static_cast<double>(num_rows);
  state.counters["columns"] = static_cast<double>(num_columns);
  state.counters["bytes"]   = static_cast<double>(bytes_transferred);
}

// Register benchmark with different data sizes
BENCHMARK(BM_ConvertHostToGpu)
  ->Args({1000, 2})     // 1K rows, 2 columns
  ->Args({10000, 2})    // 10K rows, 2 columns
  ->Args({100000, 2})   // 100K rows, 2 columns
  ->Args({1000000, 2})  // 1M rows, 2 columns
  ->Args({1000, 4})     // 1K rows, 4 columns
  ->Args({10000, 4})    // 10K rows, 4 columns
  ->Args({100000, 4})   // 100K rows, 4 columns
  ->Args({1000, 8})     // 1K rows, 8 columns
  ->Args({10000, 8})    // 10K rows, 8 columns
  ->Args({100000, 8})   // 100K rows, 8 columns
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// =============================================================================
// Roundtrip Conversion Benchmarks
// =============================================================================

/**
 * @brief Benchmark roundtrip GPU->HOST->GPU conversion.
 */
static void BM_RoundtripGpuHostGpu(benchmark::State& state)
{
  int64_t num_rows = state.range(0);
  int num_columns  = static_cast<int>(state.range(1));
  auto mgr         = std::make_unique<memory_reservation_manager>(create_benchmark_configs());
  auto registry    = std::make_unique<representation_converter_registry>();
  register_builtin_converters(*registry);

  const memory_space* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  const memory_space* host_space = mgr->get_memory_space(Tier::HOST, 0);

  rmm::cuda_stream stream;

  // Create source data
  auto table    = create_benchmark_table(num_rows, num_columns);
  auto gpu_repr = std::make_unique<gpu_table_representation>(std::move(table),
                                                             *const_cast<memory_space*>(gpu_space));

  // Warm-up
  auto warmup_table = create_benchmark_table(100, 2);
  auto warmup_repr  = std::make_unique<gpu_table_representation>(
    std::move(warmup_table), *const_cast<memory_space*>(gpu_space));
  auto warmup_host = registry->convert<host_table_representation>(*warmup_repr, host_space, stream);
  auto warmup_gpu  = registry->convert<gpu_table_representation>(*warmup_host, gpu_space, stream);
  stream.synchronize();

  size_t bytes_transferred = gpu_repr->get_size_in_bytes() * 2;  // Both directions

  // Benchmark loop
  for (auto _ : state) {
    auto host_result = registry->convert<host_table_representation>(*gpu_repr, host_space, stream);
    auto gpu_result  = registry->convert<gpu_table_representation>(*host_result, gpu_space, stream);
    stream.synchronize();
  }

  // Report metrics
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_transferred));
  state.counters["rows"]    = static_cast<double>(num_rows);
  state.counters["columns"] = static_cast<double>(num_columns);
  state.counters["bytes"]   = static_cast<double>(bytes_transferred);
}

// Register benchmark with different data sizes
BENCHMARK(BM_RoundtripGpuHostGpu)
  ->Args({1000, 2})    // 1K rows, 2 columns
  ->Args({10000, 2})   // 10K rows, 2 columns
  ->Args({100000, 2})  // 100K rows, 2 columns
  ->Args({1000, 4})    // 1K rows, 4 columns
  ->Args({10000, 4})   // 10K rows, 4 columns
  ->Args({100000, 4})  // 100K rows, 4 columns
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// =============================================================================
// HOST to HOST Conversion Benchmarks (Cross-device on CPU)
// =============================================================================

/**
 * @brief Benchmark HOST to HOST conversion (simulated cross-device on same host).
 */
static void BM_ConvertHostToHost(benchmark::State& state)
{
  int64_t num_rows = state.range(0);
  int num_columns  = static_cast<int>(state.range(1));
  auto mgr         = std::make_unique<memory_reservation_manager>(create_benchmark_configs());
  auto registry    = std::make_unique<representation_converter_registry>();
  register_builtin_converters(*registry);

  const memory_space* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  const memory_space* host_space = mgr->get_memory_space(Tier::HOST, 0);

  rmm::cuda_stream stream;

  // Create source HOST data via GPU->HOST conversion
  auto table         = create_benchmark_table(num_rows, num_columns);
  auto gpu_repr_temp = std::make_unique<gpu_table_representation>(
    std::move(table), *const_cast<memory_space*>(gpu_space));
  auto host_repr_source =
    registry->convert<host_table_representation>(*gpu_repr_temp, host_space, stream);
  stream.synchronize();

  // Warm-up
  auto warmup_result =
    registry->convert<host_table_representation>(*host_repr_source, host_space, stream);
  stream.synchronize();

  size_t bytes_transferred = host_repr_source->get_size_in_bytes();

  // Benchmark loop
  for (auto _ : state) {
    auto host_result =
      registry->convert<host_table_representation>(*host_repr_source, host_space, stream);
    stream.synchronize();
  }

  // Report metrics
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_transferred));
  state.counters["rows"]    = static_cast<double>(num_rows);
  state.counters["columns"] = static_cast<double>(num_columns);
  state.counters["bytes"]   = static_cast<double>(bytes_transferred);
}

// Register benchmark with different data sizes
BENCHMARK(BM_ConvertHostToHost)
  ->Args({1000, 2})     // 1K rows, 2 columns
  ->Args({10000, 2})    // 10K rows, 2 columns
  ->Args({100000, 2})   // 100K rows, 2 columns
  ->Args({1000000, 2})  // 1M rows, 2 columns
  ->Args({1000, 4})     // 1K rows, 4 columns
  ->Args({10000, 4})    // 10K rows, 4 columns
  ->Args({100000, 4})   // 100K rows, 4 columns
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// =============================================================================
// Memory Throughput Benchmarks
// =============================================================================

/**
 * @brief Benchmark GPU to HOST conversion focusing on memory bandwidth.
 */
static void BM_GpuToHostThroughput(benchmark::State& state)
{
  int64_t total_bytes = state.range(0) * 1024 * 1024;  // Convert MB to bytes
  // Use a single large column to maximize memory transfer efficiency
  int64_t num_rows = total_bytes / sizeof(int64_t);

  auto mgr      = std::make_unique<memory_reservation_manager>(create_benchmark_configs());
  auto registry = std::make_unique<representation_converter_registry>();
  register_builtin_converters(*registry);

  const memory_space* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  const memory_space* host_space = mgr->get_memory_space(Tier::HOST, 0);

  rmm::cuda_stream stream;

  // Create source data with single column for maximum sequential throughput
  auto table    = create_benchmark_table(num_rows, 1);
  auto gpu_repr = std::make_unique<gpu_table_representation>(std::move(table),
                                                             *const_cast<memory_space*>(gpu_space));

  size_t bytes_transferred = gpu_repr->get_size_in_bytes();

  // Benchmark loop
  for (auto _ : state) {
    auto host_result = registry->convert<host_table_representation>(*gpu_repr, host_space, stream);
    stream.synchronize();
  }

  // Report metrics
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_transferred));
  state.counters["MB"] = static_cast<double>(bytes_transferred) / (1024.0 * 1024.0);
  state.counters["throughput_GB/s"] =
    benchmark::Counter(static_cast<double>(bytes_transferred),
                       benchmark::Counter::kIsIterationInvariantRate,
                       benchmark::Counter::kIs1024);
}

// Register throughput benchmark with different data sizes
BENCHMARK(BM_GpuToHostThroughput)
  ->Arg(1)     // 1 MB
  ->Arg(10)    // 10 MB
  ->Arg(100)   // 100 MB
  ->Arg(500)   // 500 MB
  ->Arg(1000)  // 1 GB
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

/**
 * @brief Benchmark HOST to GPU conversion focusing on memory bandwidth.
 */
static void BM_HostToGpuThroughput(benchmark::State& state)
{
  int64_t total_bytes = state.range(0) * 1024 * 1024;  // Convert MB to bytes
  // Use a single large column to maximize memory transfer efficiency
  int64_t num_rows = total_bytes / sizeof(int64_t);

  auto mgr      = std::make_unique<memory_reservation_manager>(create_benchmark_configs());
  auto registry = std::make_unique<representation_converter_registry>();
  register_builtin_converters(*registry);

  const memory_space* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  const memory_space* host_space = mgr->get_memory_space(Tier::HOST, 0);

  rmm::cuda_stream stream;

  // Create source HOST data via GPU->HOST conversion
  auto table         = create_benchmark_table(num_rows, 1);
  auto gpu_repr_temp = std::make_unique<gpu_table_representation>(
    std::move(table), *const_cast<memory_space*>(gpu_space));
  auto host_repr = registry->convert<host_table_representation>(*gpu_repr_temp, host_space, stream);
  stream.synchronize();

  size_t bytes_transferred = host_repr->get_size_in_bytes();

  // Benchmark loop
  for (auto _ : state) {
    auto gpu_result = registry->convert<gpu_table_representation>(*host_repr, gpu_space, stream);
    stream.synchronize();
  }

  // Report metrics
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_transferred));
  state.counters["MB"] = static_cast<double>(bytes_transferred) / (1024.0 * 1024.0);
  state.counters["throughput_GB/s"] =
    benchmark::Counter(static_cast<double>(bytes_transferred),
                       benchmark::Counter::kIsIterationInvariantRate,
                       benchmark::Counter::kIs1024);
}

// Register throughput benchmark with different data sizes
BENCHMARK(BM_HostToGpuThroughput)
  ->Arg(1)     // 1 MB
  ->Arg(10)    // 10 MB
  ->Arg(100)   // 100 MB
  ->Arg(500)   // 500 MB
  ->Arg(1000)  // 1 GB
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

}  // namespace
