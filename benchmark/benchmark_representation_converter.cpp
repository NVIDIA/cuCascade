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

#include "cucascade/data/cpu_data_representation.hpp"
#include "cucascade/data/gpu_data_representation.hpp"
#include "cucascade/data/representation_converter.hpp"
#include "cucascade/memory/memory_reservation_manager.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <benchmark/benchmark.h>

#include <cstring>
#include <memory>
#include <thread>
#include <vector>

namespace {

using namespace cucascade;
using namespace cucascade::memory;

constexpr uint64_t KiB = 1024ULL;
constexpr uint64_t MiB = 1024ULL * KiB;
constexpr uint64_t GiB = 1024ULL * MiB;

// For non-NUMA systems, this should be -1, causing the allocator to use cudaHostAlloc instead of
// cudaHostRegister
constexpr int hostDevId = -1;

// Global shared memory manager - managed via setup/teardown functions
static std::shared_ptr<memory_reservation_manager> g_shared_memory_manager;

/**
 * @brief Create memory manager configs for benchmarking (one GPU and one HOST).
 */
std::vector<memory_space_config> create_benchmark_configs()
{
  std::vector<memory_space_config> configs;
  // Large memory limits for benchmarking
  gpu_memory_space_config gpu_config;
  gpu_config.device_id       = 0;
  gpu_config.memory_capacity = 8 * GiB;
  gpu_config.mr_factory_fn   = make_default_allocator_for_tier(Tier::GPU);
  configs.emplace_back(gpu_config);

  host_memory_space_config host_config;
  host_config.numa_id         = hostDevId;
  host_config.memory_capacity = 16 * GiB;
  host_config.mr_factory_fn   = make_default_allocator_for_tier(Tier::HOST);
  configs.emplace_back(host_config);
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
void DoTeardown([[maybe_unused]] const benchmark::State& state)
{
  // Reset the shared memory manager after benchmark
  g_shared_memory_manager.reset();
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
 * @param state.range(2) Manual thread count
 */
void BM_ConvertGpuToHost(benchmark::State& state)
{
  int64_t total_bytes   = state.range(0);
  int num_columns       = static_cast<int>(state.range(1));
  uint64_t thread_count = static_cast<uint64_t>(state.range(2));

  // Use shared memory manager across all threads
  auto mgr = get_shared_memory_manager();

  auto registry = std::make_unique<representation_converter_registry>();
  register_builtin_converters(*registry);

  const memory_space* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  const memory_space* host_space = mgr->get_memory_space(Tier::HOST, hostDevId);

  // Create separate table and representation for each thread BEFORE warmup
  std::vector<std::unique_ptr<gpu_table_representation>> thread_gpu_reprs;
  thread_gpu_reprs.reserve(thread_count);
  std::vector<rmm::cuda_stream> streams(thread_count);

  for (uint64_t t = 0; t < thread_count; ++t) {
    auto table = create_benchmark_table_from_bytes(total_bytes, num_columns);
    thread_gpu_reprs.push_back(std::make_unique<gpu_table_representation>(
      std::move(table), *const_cast<memory_space*>(gpu_space)));
  }

  // Warm-up
  rmm::cuda_stream warmup_stream;
  auto warmup_table = create_benchmark_table_from_bytes(1 * KiB, 2);
  auto warmup_repr  = std::make_unique<gpu_table_representation>(
    std::move(warmup_table), *const_cast<memory_space*>(gpu_space));
  auto warmup_result =
    registry->convert<host_table_representation>(*warmup_repr, host_space, warmup_stream);
  warmup_stream.synchronize();

  size_t bytes_transferred = thread_gpu_reprs[0]->get_size_in_bytes() * thread_count;

  // Benchmark loop
  for (auto _ : state) {
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    // Create threads
    for (uint64_t t = 0; t < thread_count; ++t) {
      threads.emplace_back([&, t]() {
        auto host_result = registry->convert<host_table_representation>(
          *thread_gpu_reprs[t], host_space, streams[t]);
        streams[t].synchronize();
      });
    }

    // Join all threads
    for (auto& thread : threads) {
      thread.join();
    }
  }

  // Update counters in main thread after loop
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_transferred));
  state.counters["columns"]     = static_cast<double>(num_columns);
  state.counters["bytes"]       = static_cast<double>(bytes_transferred);
  state.counters["num_threads"] = static_cast<double>(thread_count);
}

/**
 * @brief Benchmark HOST to GPU conversion with varying data sizes.
 * @param state.range(0) Total bytes
 * @param state.range(1) Number of columns
 * @param state.range(2) Manual thread count
 */
void BM_ConvertHostToGpu(benchmark::State& state)
{
  int64_t total_bytes   = state.range(0);
  int num_columns       = static_cast<int>(state.range(1));
  uint64_t thread_count = static_cast<uint64_t>(state.range(2));

  // Use shared memory manager across all threads
  auto mgr = get_shared_memory_manager();

  auto registry = std::make_unique<representation_converter_registry>();
  register_builtin_converters(*registry);

  const memory_space* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  const memory_space* host_space = mgr->get_memory_space(Tier::HOST, hostDevId);

  // Create separate table and representation for each thread BEFORE warmup
  std::vector<std::unique_ptr<host_table_representation>> thread_host_reprs;
  thread_host_reprs.reserve(thread_count);
  std::vector<rmm::cuda_stream> streams(thread_count);

  rmm::cuda_stream setup_stream;
  for (uint64_t t = 0; t < thread_count; ++t) {
    auto table         = create_benchmark_table_from_bytes(total_bytes, num_columns);
    auto gpu_repr_temp = std::make_unique<gpu_table_representation>(
      std::move(table), *const_cast<memory_space*>(gpu_space));
    auto host_repr =
      registry->convert<host_table_representation>(*gpu_repr_temp, host_space, setup_stream);
    setup_stream.synchronize();
    thread_host_reprs.push_back(std::move(host_repr));
  }

  // Warm-up
  rmm::cuda_stream warmup_stream;
  auto warmup_result =
    registry->convert<gpu_table_representation>(*thread_host_reprs[0], gpu_space, warmup_stream);
  warmup_stream.synchronize();

  size_t bytes_transferred = thread_host_reprs[0]->get_size_in_bytes() * thread_count;

  // Benchmark loop
  for (auto _ : state) {
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    // Create threads
    for (uint64_t t = 0; t < thread_count; ++t) {
      threads.emplace_back([&, t]() {
        auto gpu_result =
          registry->convert<gpu_table_representation>(*thread_host_reprs[t], gpu_space, streams[t]);
        streams[t].synchronize();
      });
    }

    // Join all threads
    for (auto& thread : threads) {
      thread.join();
    }
  }

  // Update counters in main thread after loop
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_transferred));
  state.counters["columns"]     = static_cast<double>(num_columns);
  state.counters["bytes"]       = static_cast<double>(bytes_transferred);
  state.counters["num_threads"] = static_cast<double>(thread_count);
}

// =============================================================================
// Memory Throughput Benchmarks
// =============================================================================

/**
 * @brief Benchmark GPU to HOST memory bandwidth.
 * @param state.range(0) Total bytes
 * @param state.range(1) Manual thread count
 */
void BM_GpuToHostThroughput(benchmark::State& state)
{
  uint64_t total_bytes  = static_cast<uint64_t>(state.range(0));
  uint64_t thread_count = static_cast<uint64_t>(state.range(1));

  // Create separate buffers and streams for each thread BEFORE benchmark loop
  std::vector<void*> d_buffers(thread_count);
  std::vector<void*> h_buffers(thread_count);
  std::vector<rmm::cuda_stream> streams(thread_count);

  for (uint64_t t = 0; t < thread_count; ++t) {
    RMM_CUDA_TRY(cudaMalloc(&d_buffers[t], total_bytes));
    RMM_CUDA_TRY(cudaMallocHost(&h_buffers[t], total_bytes));
    RMM_CUDA_TRY(cudaMemset(d_buffers[t], 0x42, total_bytes));
  }

  uint64_t total_bytes_transferred = total_bytes * thread_count;

  // Benchmark loop
  for (auto _ : state) {
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    // Create threads
    for (uint64_t t = 0; t < thread_count; ++t) {
      threads.emplace_back([&, t]() {
        RMM_CUDA_TRY(cudaMemcpyAsync(
          h_buffers[t], d_buffers[t], total_bytes, cudaMemcpyDeviceToHost, streams[t].value()));
        streams[t].synchronize();
      });
    }

    // Join all threads
    for (auto& thread : threads) {
      thread.join();
    }
  }

  // Cleanup buffers
  for (uint64_t t = 0; t < thread_count; ++t) {
    RMM_CUDA_TRY(cudaFree(d_buffers[t]));
    RMM_CUDA_TRY(cudaFreeHost(h_buffers[t]));
  }

  // Update counters in main thread after loop
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(total_bytes_transferred));
  state.counters["MB"]          = static_cast<double>(total_bytes_transferred) / (1024.0 * 1024.0);
  state.counters["num_threads"] = static_cast<double>(thread_count);
}

/**
 * @brief Benchmark HOST to GPU memory bandwidth.
 * @param state.range(0) Total bytes
 * @param state.range(1) Manual thread count
 */
void BM_HostToGpuThroughput(benchmark::State& state)
{
  uint64_t total_bytes  = static_cast<uint64_t>(state.range(0));
  uint64_t thread_count = static_cast<uint64_t>(state.range(1));

  // Create separate buffers and streams for each thread BEFORE benchmark loop
  std::vector<void*> d_buffers(thread_count);
  std::vector<void*> h_buffers(thread_count);
  std::vector<rmm::cuda_stream> streams(thread_count);

  for (uint64_t t = 0; t < thread_count; ++t) {
    RMM_CUDA_TRY(cudaMalloc(&d_buffers[t], total_bytes));
    RMM_CUDA_TRY(cudaMallocHost(&h_buffers[t], total_bytes));
    std::memset(h_buffers[t], 0x42, total_bytes);
  }

  uint64_t total_bytes_transferred = total_bytes * thread_count;

  // Benchmark loop
  for (auto _ : state) {
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    // Create threads
    for (uint64_t t = 0; t < thread_count; ++t) {
      threads.emplace_back([&, t]() {
        RMM_CUDA_TRY(cudaMemcpyAsync(
          d_buffers[t], h_buffers[t], total_bytes, cudaMemcpyHostToDevice, streams[t].value()));
        streams[t].synchronize();
      });
    }

    // Join all threads
    for (auto& thread : threads) {
      thread.join();
    }
  }

  // Cleanup buffers
  for (uint64_t t = 0; t < thread_count; ++t) {
    RMM_CUDA_TRY(cudaFree(d_buffers[t]));
    RMM_CUDA_TRY(cudaFreeHost(h_buffers[t]));
  }

  // Update counters in main thread after loop
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(total_bytes_transferred));
  state.counters["MB"]          = static_cast<double>(total_bytes_transferred) / (1024.0 * 1024.0);
  state.counters["num_threads"] = static_cast<double>(thread_count);
}

BENCHMARK(BM_ConvertGpuToHost)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->RangeMultiplier(2)
  ->Ranges({{128 * KiB, 512 * MiB}, {1, 64}, {1, 4}})
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1)
  ->UseRealTime();

BENCHMARK(BM_ConvertHostToGpu)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->RangeMultiplier(2)
  ->Ranges({{128 * KiB, 512 * MiB}, {1, 64}, {1, 4}})
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1)
  ->UseRealTime();

BENCHMARK(BM_GpuToHostThroughput)
  ->RangeMultiplier(2)
  ->Ranges({{128 * KiB, 512 * MiB}, {1, 4}})
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1)
  ->UseRealTime();

BENCHMARK(BM_HostToGpuThroughput)
  ->RangeMultiplier(2)
  ->Ranges({{128 * KiB, 512 * MiB}, {1, 4}})
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1)
  ->UseRealTime();

}  // namespace

// Use Google Benchmark's default main
BENCHMARK_MAIN();
