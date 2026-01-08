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

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <benchmark/benchmark.h>

#include <cstring>
#include <memory>
#include <vector>

namespace {

using namespace cucascade;
using namespace cucascade::memory;

constexpr uint64_t KiB = 1024ULL;
constexpr uint64_t MiB = 1024ULL * KiB;
constexpr uint64_t GiB = 1024ULL * MiB;

// Global shared memory manager - managed via setup/teardown functions
static std::shared_ptr<memory_reservation_manager> g_shared_memory_manager;

/**
 * @brief Create memory manager configs for benchmarking (one GPU and one HOST).
 */
std::vector<memory_reservation_manager::memory_space_config> create_benchmark_configs()
{
  std::vector<memory_reservation_manager::memory_space_config> configs;
  // Large memory limits for benchmarking
  configs.emplace_back(Tier::GPU, 0, 4 * GiB, make_default_allocator_for_tier(Tier::GPU));
  configs.emplace_back(Tier::HOST, 0, 8 * GiB, make_default_allocator_for_tier(Tier::HOST));
  return configs;
}

/**
 * @brief Get shared memory reservation manager.
 */
std::shared_ptr<memory_reservation_manager> get_shared_memory_manager()
{
  return g_shared_memory_manager;
}

/**
 * @brief Setup function called before benchmarks.
 */
void DoSetup([[maybe_unused]] const benchmark::State& state)
{
  if (!g_shared_memory_manager) {
    g_shared_memory_manager =
      std::make_shared<memory_reservation_manager>(create_benchmark_configs());
  }
}

/**
 * @brief Teardown function called after benchmarks.
 */
void DoTeardown(const benchmark::State& state)
{
  // Only teardown after all threads are done
  if (state.thread_index() == 0) { g_shared_memory_manager.reset(); }
}

/**
 * @brief Create a cuDF table from bytes and columns specification (int64/float64 only).
 *
 * @param total_bytes Total size in bytes
 * @param num_columns Number of columns (alternates between INT64 and FLOAT64)
 * @return cudf::table The generated table
 */
cudf::table create_benchmark_table_from_bytes(int64_t total_bytes, int num_columns)
{
  // Both INT64 and FLOAT64 are 8 bytes
  constexpr size_t bytes_per_element = 8;

  // Calculate number of rows
  int64_t total_elements = total_bytes / static_cast<int64_t>(bytes_per_element);
  int64_t num_rows       = total_elements / static_cast<int64_t>(num_columns);

  // Ensure at least 1 row
  if (num_rows < 1) num_rows = 1;

  std::vector<std::unique_ptr<cudf::column>> columns;

  for (int i = 0; i < num_columns; ++i) {
    cudf::data_type dtype;
    uint8_t fill_value;

    // Alternate between INT64 and FLOAT64
    if (i % 2 == 0) {
      dtype      = cudf::data_type{cudf::type_id::INT64};
      fill_value = 0x22;
    } else {
      dtype      = cudf::data_type{cudf::type_id::FLOAT64};
      fill_value = 0x44;
    }

    auto col =
      cudf::make_numeric_column(dtype, static_cast<int>(num_rows), cudf::mask_state::UNALLOCATED);

    if (num_rows > 0) {
      auto view      = col->mutable_view();
      auto type_size = cudf::size_of(dtype);
      auto bytes     = static_cast<size_t>(num_rows) * (type_size);
      RMM_CUDA_TRY(cudaMemset(const_cast<void*>(view.head()), fill_value, bytes));
    }

    columns.push_back(std::move(col));
  }

  return cudf::table(std::move(columns));
}

// =============================================================================
// GPU <-> HOST Conversion Benchmarks
// =============================================================================

/**
 * @brief Benchmark GPU to HOST conversion with varying data sizes.
 * @param state.range(0) Total bytes
 * @param state.range(1) Number of columns
 */
void BM_ConvertGpuToHost(benchmark::State& state)
{
  int64_t total_bytes = state.range(0);
  int num_columns     = static_cast<int>(state.range(1));

  // Use shared memory manager across all threads
  auto mgr = get_shared_memory_manager();

  auto registry = std::make_unique<representation_converter_registry>();
  register_builtin_converters(*registry);

  const memory_space* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  const memory_space* host_space = mgr->get_memory_space(Tier::HOST, 0);

  rmm::cuda_stream stream;

  // Create source data
  auto table    = create_benchmark_table_from_bytes(total_bytes, num_columns);
  auto gpu_repr = std::make_unique<gpu_table_representation>(std::move(table),
                                                             *const_cast<memory_space*>(gpu_space));

  // Warm-up
  auto warmup_table = create_benchmark_table_from_bytes(1 * KiB, 2);
  auto warmup_repr  = std::make_unique<gpu_table_representation>(
    std::move(warmup_table), *const_cast<memory_space*>(gpu_space));
  auto warmup_result =
    registry->convert<host_table_representation>(*warmup_repr, host_space, stream);
  stream.synchronize();

  size_t bytes_transferred = gpu_repr->get_size_in_bytes();

  for (auto _ : state) {
    auto host_result = registry->convert<host_table_representation>(*gpu_repr, host_space, stream);
    stream.synchronize();
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_transferred));
  state.counters["columns"] = static_cast<double>(num_columns);
  state.counters["bytes"]   = static_cast<double>(bytes_transferred);
}

/**
 * @brief Benchmark HOST to GPU conversion with varying data sizes.
 * @param state.range(0) Total bytes
 * @param state.range(1) Number of columns
 */
void BM_ConvertHostToGpu(benchmark::State& state)
{
  int64_t total_bytes = state.range(0);
  int num_columns     = static_cast<int>(state.range(1));

  // Use shared memory manager across all threads
  auto mgr = get_shared_memory_manager();

  auto registry = std::make_unique<representation_converter_registry>();
  register_builtin_converters(*registry);

  const memory_space* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  const memory_space* host_space = mgr->get_memory_space(Tier::HOST, 0);

  rmm::cuda_stream stream;

  // Create source data: first make GPU repr, then convert to HOST
  auto table         = create_benchmark_table_from_bytes(total_bytes, num_columns);
  auto gpu_repr_temp = std::make_unique<gpu_table_representation>(
    std::move(table), *const_cast<memory_space*>(gpu_space));
  auto host_repr = registry->convert<host_table_representation>(*gpu_repr_temp, host_space, stream);
  stream.synchronize();

  // Warm-up
  auto warmup_result = registry->convert<gpu_table_representation>(*host_repr, gpu_space, stream);
  stream.synchronize();

  size_t bytes_transferred = host_repr->get_size_in_bytes();

  for (auto _ : state) {
    auto gpu_result = registry->convert<gpu_table_representation>(*host_repr, gpu_space, stream);
    stream.synchronize();
  }

  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(bytes_transferred));
  state.counters["columns"] = static_cast<double>(num_columns);
  state.counters["bytes"]   = static_cast<double>(bytes_transferred);
}

// =============================================================================
// Memory Throughput Benchmarks
// =============================================================================

/**
 * @brief Benchmark GPU to HOST memory bandwidth.
 */
void BM_GpuToHostThroughput(benchmark::State& state)
{
  uint64_t total_bytes = static_cast<uint64_t>(state.range(0));

  rmm::cuda_stream stream;
  void* d_buffer = nullptr;
  void* h_buffer = nullptr;

  RMM_CUDA_TRY(cudaMalloc(&d_buffer, total_bytes));
  RMM_CUDA_TRY(cudaMallocHost(&h_buffer, total_bytes));
  RMM_CUDA_TRY(cudaMemset(d_buffer, 0x42, total_bytes));

  for (auto _ : state) {
    RMM_CUDA_TRY(
      cudaMemcpyAsync(h_buffer, d_buffer, total_bytes, cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();
  }

  RMM_CUDA_TRY(cudaFree(d_buffer));
  RMM_CUDA_TRY(cudaFreeHost(h_buffer));

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(total_bytes));
  state.counters["MB"] = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
}

/**
 * @brief Benchmark HOST to GPU memory bandwidth.
 */
void BM_HostToGpuThroughput(benchmark::State& state)
{
  uint64_t total_bytes = static_cast<uint64_t>(state.range(0));

  rmm::cuda_stream stream;
  void* d_buffer = nullptr;
  void* h_buffer = nullptr;

  RMM_CUDA_TRY(cudaMalloc(&d_buffer, total_bytes));
  RMM_CUDA_TRY(cudaMallocHost(&h_buffer, total_bytes));

  std::memset(h_buffer, 0x42, total_bytes);

  for (auto _ : state) {
    RMM_CUDA_TRY(
      cudaMemcpyAsync(d_buffer, h_buffer, total_bytes, cudaMemcpyHostToDevice, stream.value()));
    stream.synchronize();
  }

  RMM_CUDA_TRY(cudaFree(d_buffer));
  RMM_CUDA_TRY(cudaFreeHost(h_buffer));

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(total_bytes));
  state.counters["MB"] = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
}

BENCHMARK(BM_ConvertGpuToHost)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->RangeMultiplier(4)
  ->Ranges({{256 * KiB, 1 * GiB}, {1, 64}})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ThreadRange(1, 4);

BENCHMARK(BM_ConvertHostToGpu)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->RangeMultiplier(4)
  ->Ranges({{256 * KiB, 1 * GiB}, {1, 64}})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ThreadRange(1, 4);

BENCHMARK(BM_GpuToHostThroughput)
  ->RangeMultiplier(2)
  ->Range(64 * KiB, 1 * GiB)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ThreadRange(1, 4);

BENCHMARK(BM_HostToGpuThroughput)
  ->RangeMultiplier(2)
  ->Range(128 * KiB, 1 * GiB)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime()
  ->ThreadRange(1, 4);

}  // namespace

// Use Google Benchmark's default main
BENCHMARK_MAIN();
