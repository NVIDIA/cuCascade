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
#include "cucascade/data/disk_data_representation.hpp"
#include "cucascade/data/gpu_data_representation.hpp"
#include "cucascade/data/representation_converter.hpp"
#include "cucascade/memory/config.hpp"
#include "cucascade/memory/memory_reservation_manager.hpp"

#include <cucascade/cuda_utils.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <benchmark/benchmark.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace cucascade;
using namespace cucascade::memory;

constexpr uint64_t KiB = 1024ULL;
constexpr uint64_t MiB = 1024ULL * KiB;
constexpr uint64_t GiB = 1024ULL * MiB;

/// Fraction of GPU memory that must remain free after allocating the benchmark table.
/// Leave headroom for cudf internals, RMM pool overhead, and converter temporaries.
constexpr double GPU_MEMORY_SAFETY_FACTOR = 0.75;

/**
 * @brief Skip the benchmark if the requested data size exceeds available GPU memory.
 *
 * Queries free GPU memory via cudaMemGetInfo and skips (with a message) if the
 * benchmark's data allocation would exceed the safety threshold.
 *
 * @return true if the benchmark should be skipped.
 */
bool skip_if_oom(benchmark::State& state, int64_t total_bytes)
{
  std::size_t free_bytes = 0;
  std::size_t total_gpu  = 0;
  cudaMemGetInfo(&free_bytes, &total_gpu);

  auto available = static_cast<int64_t>(static_cast<double>(free_bytes) * GPU_MEMORY_SAFETY_FACTOR);
  if (total_bytes > available) {
    state.SkipWithMessage(
      "OOM: need " + std::to_string(total_bytes / static_cast<int64_t>(MiB)) + " MiB but only " +
      std::to_string(available / static_cast<int64_t>(MiB)) + " MiB available (GPU has " +
      std::to_string(free_bytes / static_cast<std::size_t>(MiB)) + " MiB free, " +
      std::to_string(static_cast<int>(GPU_MEMORY_SAFETY_FACTOR * 100)) + "% safety)");
    return true;
  }
  return false;
}

// Hardware baselines from gdsio on /dev/nvme1n1 (/mnt/disk_2, ext4)
// Measured: gdsio -D /mnt/disk_2/gdsio_test -d 0 -w 4 -s 4G -x 0 -I 1
// Write: 6.73 GiB/s, Read: 13.35 GiB/s
constexpr double GDSIO_WRITE_GIBS = 6.73;
constexpr double GDSIO_READ_GIBS  = 13.35;
constexpr double GIBS_TO_BYTES    = 1024.0 * 1024.0 * 1024.0;

/**
 * @brief Flush NVMe write cache and drop OS page cache for a file.
 *
 * Call after writes and before reads in benchmark loops to ensure each iteration
 * measures real disk I/O without OS cache interference.
 */
void drop_os_cache(const std::string& path)
{
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd >= 0) {
    ::fdatasync(fd);
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
  }
}

/**
 * @brief Create a converter registry with all built-in converters.
 */
std::unique_ptr<representation_converter_registry> make_benchmark_registry()
{
  auto registry = std::make_unique<representation_converter_registry>();
  register_builtin_converters(*registry);
  return registry;
}

// For non-NUMA systems, this should be -1, causing the allocator to use cudaHostAlloc instead of
// cudaHostRegister
constexpr int hostDevId = -1;

// Global shared memory manager - managed via setup/teardown functions
static std::shared_ptr<memory_reservation_manager> g_shared_memory_manager;

/**
 * @brief Create memory manager configs for benchmarking (GPU, HOST, and DISK).
 */
std::vector<memory_space_config> create_benchmark_configs()
{
  std::vector<memory_space_config> configs;

  // Query actual GPU memory and use 90% of it (leave room for driver/OS)
  std::size_t free_bytes = 0;
  std::size_t total_gpu  = 0;
  cudaMemGetInfo(&free_bytes, &total_gpu);
  auto gpu_capacity = static_cast<uint64_t>(static_cast<double>(free_bytes) * 0.9);

  gpu_memory_space_config gpu_config;
  gpu_config.device_id       = 0;
  gpu_config.memory_capacity = gpu_capacity;
  gpu_config.mr_factory_fn   = make_default_allocator_for_tier(Tier::GPU);
  configs.emplace_back(gpu_config);

  host_memory_space_config host_config;
  host_config.numa_id              = hostDevId;
  host_config.memory_capacity      = gpu_capacity;  // match GPU capacity
  host_config.mr_factory_fn        = make_default_allocator_for_tier(Tier::HOST);
  host_config.initial_number_pools = 16;
  configs.emplace_back(host_config);

  disk_memory_space_config disk_config;
  disk_config.disk_id         = 0;
  disk_config.memory_capacity = 32 * GiB;
  disk_config.mount_paths     = "/tmp";
  configs.emplace_back(disk_config);

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
 *
 * Intentionally does NOT reset the memory manager — disk benchmarks create
 * disk_data_representation objects whose files live under the disk memory space
 * mount path. Resetting the manager between benchmarks would invalidate
 * disk files still referenced by later benchmark iterations.
 */
void DoTeardown([[maybe_unused]] const benchmark::State& state)
{
  // no-op: manager persists across benchmark functions
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
  constexpr size_t bytes_per_element = 8;

  int64_t total_elements = total_bytes / static_cast<int64_t>(bytes_per_element);
  int64_t num_rows       = total_elements / static_cast<int64_t>(num_columns);

  if (num_rows < 1) num_rows = 1;

  std::vector<std::unique_ptr<cudf::column>> columns;

  for (int i = 0; i < num_columns; ++i) {
    cudf::data_type dtype;
    uint8_t fill_value;

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
      CUCASCADE_CUDA_TRY(cudaMemset(const_cast<void*>(view.head()), fill_value, bytes));
    }

    columns.push_back(std::move(col));
  }

  return cudf::table(std::move(columns));
}

/**
 * @brief Create a cuDF table with STRING columns for benchmarking.
 *
 * Each column has repeated 8-char strings ("benchstr") to approximate the requested total bytes.
 *
 * @param total_bytes Total approximate data size in bytes
 * @param num_columns Number of string columns
 * @return cudf::table The generated table
 */
cudf::table create_string_benchmark_table(int64_t total_bytes, int num_columns)
{
  constexpr int chars_per_string = 8;
  int64_t bytes_per_column       = total_bytes / static_cast<int64_t>(num_columns);
  int64_t num_strings =
    bytes_per_column / static_cast<int64_t>(chars_per_string + static_cast<int>(sizeof(int32_t)));
  if (num_strings < 1) num_strings = 1;
  int64_t total_chars = num_strings * chars_per_string;

  rmm::cuda_stream stream;

  std::vector<std::unique_ptr<cudf::column>> columns;

  for (int c = 0; c < num_columns; ++c) {
    // Build offsets: 0, 8, 16, 24, ...
    std::vector<int32_t> host_offsets(static_cast<size_t>(num_strings) + 1);
    for (int64_t i = 0; i <= num_strings; ++i) {
      host_offsets[static_cast<size_t>(i)] = static_cast<int32_t>(i * chars_per_string);
    }
    rmm::device_buffer offsets_buf(
      host_offsets.data(), host_offsets.size() * sizeof(int32_t), stream.view());
    auto offsets_col =
      std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                     static_cast<cudf::size_type>(host_offsets.size()),
                                     std::move(offsets_buf),
                                     rmm::device_buffer{},
                                     0);

    // Build chars: repeated "benchstr"
    std::vector<int8_t> host_chars(static_cast<size_t>(total_chars));
    const char pattern[] = "benchstr";
    for (int64_t i = 0; i < total_chars; ++i) {
      host_chars[static_cast<size_t>(i)] =
        static_cast<int8_t>(pattern[static_cast<size_t>(i % chars_per_string)]);
    }
    rmm::device_buffer chars_buf(host_chars.data(), host_chars.size(), stream.view());

    stream.synchronize();

    auto str_col = cudf::make_strings_column(static_cast<cudf::size_type>(num_strings),
                                             std::move(offsets_col),
                                             std::move(chars_buf),
                                             0,
                                             rmm::device_buffer{});
    columns.push_back(std::move(str_col));
  }

  return cudf::table(std::move(columns));
}

/**
 * @brief Create a cuDF table with LIST<INT32> columns for benchmarking.
 *
 * Each list element is a fixed-length list of 4 INT32s.
 *
 * @param total_bytes Total approximate data size in bytes
 * @param num_columns Number of list columns
 * @return cudf::table The generated table
 */
cudf::table create_list_benchmark_table(int64_t total_bytes, int num_columns)
{
  constexpr int elements_per_list              = 4;
  constexpr int64_t bytes_per_list_element_row = elements_per_list * sizeof(int32_t);
  int64_t bytes_per_column                     = total_bytes / static_cast<int64_t>(num_columns);
  int64_t num_lists                            = bytes_per_column / bytes_per_list_element_row;
  if (num_lists < 1) num_lists = 1;
  int64_t num_values = num_lists * elements_per_list;

  rmm::cuda_stream stream;

  std::vector<std::unique_ptr<cudf::column>> columns;

  for (int c = 0; c < num_columns; ++c) {
    // Build offsets: 0, 4, 8, 12, ...
    std::vector<int32_t> host_offsets(static_cast<size_t>(num_lists) + 1);
    for (int64_t i = 0; i <= num_lists; ++i) {
      host_offsets[static_cast<size_t>(i)] = static_cast<int32_t>(i * elements_per_list);
    }
    rmm::device_buffer offsets_buf(
      host_offsets.data(), host_offsets.size() * sizeof(int32_t), stream.view());
    auto offsets_col =
      std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                     static_cast<cudf::size_type>(host_offsets.size()),
                                     std::move(offsets_buf),
                                     rmm::device_buffer{},
                                     0);

    // Build values column
    auto values_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                static_cast<cudf::size_type>(num_values),
                                                cudf::mask_state::UNALLOCATED,
                                                stream.view());
    if (num_values > 0) {
      auto view = values_col->mutable_view();
      CUCASCADE_CUDA_TRY(cudaMemset(
        const_cast<void*>(view.head()), 0x33, static_cast<size_t>(num_values) * sizeof(int32_t)));
    }

    stream.synchronize();

    auto list_col = cudf::make_lists_column(static_cast<cudf::size_type>(num_lists),
                                            std::move(offsets_col),
                                            std::move(values_col),
                                            0,
                                            {});
    columns.push_back(std::move(list_col));
  }

  return cudf::table(std::move(columns));
}

/**
 * @brief Create a cuDF table with STRUCT<INT64, FLOAT64> columns for benchmarking.
 *
 * Each struct row = 16 bytes (8 + 8).
 *
 * @param total_bytes Total approximate data size in bytes
 * @param num_columns Number of struct columns
 * @return cudf::table The generated table
 */
cudf::table create_struct_benchmark_table(int64_t total_bytes, int num_columns)
{
  constexpr int64_t bytes_per_struct_row = 16;
  int64_t bytes_per_column               = total_bytes / static_cast<int64_t>(num_columns);
  int64_t num_rows                       = bytes_per_column / bytes_per_struct_row;
  if (num_rows < 1) num_rows = 1;

  rmm::cuda_stream stream;

  std::vector<std::unique_ptr<cudf::column>> columns;

  for (int c = 0; c < num_columns; ++c) {
    auto field0 = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                            static_cast<cudf::size_type>(num_rows),
                                            cudf::mask_state::UNALLOCATED,
                                            stream.view());
    auto field1 = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64},
                                            static_cast<cudf::size_type>(num_rows),
                                            cudf::mask_state::UNALLOCATED,
                                            stream.view());
    stream.synchronize();

    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(std::move(field0));
    children.push_back(std::move(field1));
    auto struct_col =
      cudf::make_structs_column(static_cast<cudf::size_type>(num_rows), std::move(children), 0, {});

    columns.push_back(std::move(struct_col));
  }

  return cudf::table(std::move(columns));
}

// =============================================================================
// BENCH-01 / BENCH-04: GPU <-> Disk Size-Sweep Benchmarks (Numeric Columns)
// =============================================================================

/**
 * @brief Benchmark GPU to Disk conversion with varying data sizes.
 * @param state.range(0) Total bytes
 * @param state.range(1) Number of columns
 */
void BM_ConvertGpuToDisk(benchmark::State& state)
{
  int64_t total_bytes = state.range(0);
  if (skip_if_oom(state, total_bytes)) return;
  int num_columns = static_cast<int>(state.range(1));

  auto mgr = get_shared_memory_manager();

  auto registry = make_benchmark_registry();

  const memory_space* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  const memory_space* disk_space = mgr->get_memory_space(Tier::DISK, 0);

  rmm::cuda_stream stream;

  // Create GPU representation
  auto table   = create_benchmark_table_from_bytes(total_bytes, num_columns);
  auto gpu_rep = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *const_cast<memory_space*>(gpu_space));

  // Warmup
  auto warmup_table = create_benchmark_table_from_bytes(1 * KiB, 2);
  auto warmup_repr  = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(warmup_table)), *const_cast<memory_space*>(gpu_space));
  auto warmup_result =
    registry->convert<disk_data_representation>(*warmup_repr, disk_space, stream.view());
  stream.synchronize();

  size_t bytes_transferred = gpu_rep->get_size_in_bytes();

  for ([[maybe_unused]] auto _ : state) {
    // No cache eviction needed — pipeline uses O_DIRECT
    auto disk_result =
      registry->convert<disk_data_representation>(*gpu_rep, disk_space, stream.view());
    stream.synchronize();
    drop_os_cache(disk_result->get_disk_table().file_path);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_transferred));
  state.counters["columns"]          = static_cast<double>(num_columns);
  state.counters["bytes"]            = static_cast<double>(bytes_transferred);
  state.counters["gdsio_write_GiBs"] = GDSIO_WRITE_GIBS;
  state.counters["gdsio_read_GiBs"]  = GDSIO_READ_GIBS;
}

/**
 * @brief Benchmark Disk to GPU conversion with varying data sizes.
 * @param state.range(0) Total bytes
 * @param state.range(1) Number of columns
 */
void BM_ConvertDiskToGpu(benchmark::State& state)
{
  int64_t total_bytes = state.range(0);
  if (skip_if_oom(state, total_bytes)) return;
  int num_columns = static_cast<int>(state.range(1));

  auto mgr = get_shared_memory_manager();

  auto registry = make_benchmark_registry();

  const memory_space* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  const memory_space* disk_space = mgr->get_memory_space(Tier::DISK, 0);

  rmm::cuda_stream stream;

  // Create GPU representation then convert to disk once
  auto table   = create_benchmark_table_from_bytes(total_bytes, num_columns);
  auto gpu_rep = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *const_cast<memory_space*>(gpu_space));

  auto disk_rep = registry->convert<disk_data_representation>(*gpu_rep, disk_space, stream.view());
  stream.synchronize();

  size_t bytes_transferred = disk_rep->get_size_in_bytes();

  for ([[maybe_unused]] auto _ : state) {
    // No cache eviction needed — pipeline uses O_DIRECT
    auto gpu_result =
      registry->convert<gpu_table_representation>(*disk_rep, gpu_space, stream.view());
    stream.synchronize();
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_transferred));
  state.counters["columns"]          = static_cast<double>(num_columns);
  state.counters["bytes"]            = static_cast<double>(bytes_transferred);
  state.counters["gdsio_write_GiBs"] = GDSIO_WRITE_GIBS;
  state.counters["gdsio_read_GiBs"]  = GDSIO_READ_GIBS;
}

/**
 * @brief Benchmark Host to Disk conversion with varying data sizes.
 * @param state.range(0) Total bytes
 * @param state.range(1) Number of columns
 */
void BM_ConvertHostToDisk(benchmark::State& state)
{
  int64_t total_bytes = state.range(0);
  if (skip_if_oom(state, total_bytes)) return;
  int num_columns = static_cast<int>(state.range(1));

  auto mgr = get_shared_memory_manager();

  auto registry = std::make_unique<representation_converter_registry>();
  register_builtin_converters(*registry);

  const memory_space* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  const memory_space* host_space = mgr->get_memory_space(Tier::HOST, hostDevId);
  const memory_space* disk_space = mgr->get_memory_space(Tier::DISK, 0);

  rmm::cuda_stream stream;

  // Create GPU table, convert to host_data first
  auto table   = create_benchmark_table_from_bytes(total_bytes, num_columns);
  auto gpu_rep = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *const_cast<memory_space*>(gpu_space));

  auto host_rep = registry->convert<host_data_representation>(*gpu_rep, host_space, stream.view());
  stream.synchronize();

  size_t bytes_transferred = host_rep->get_size_in_bytes();

  for ([[maybe_unused]] auto _ : state) {
    // No cache eviction needed — pipeline uses O_DIRECT
    auto disk_result =
      registry->convert<disk_data_representation>(*host_rep, disk_space, stream.view());
    stream.synchronize();
    drop_os_cache(disk_result->get_disk_table().file_path);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_transferred));
  state.counters["columns"]          = static_cast<double>(num_columns);
  state.counters["bytes"]            = static_cast<double>(bytes_transferred);
  state.counters["gdsio_write_GiBs"] = GDSIO_WRITE_GIBS;
  state.counters["gdsio_read_GiBs"]  = GDSIO_READ_GIBS;
}

/**
 * @brief Benchmark Disk to Host conversion with varying data sizes.
 * @param state.range(0) Total bytes
 * @param state.range(1) Number of columns
 */
void BM_ConvertDiskToHost(benchmark::State& state)
{
  int64_t total_bytes = state.range(0);
  if (skip_if_oom(state, total_bytes)) return;
  int num_columns = static_cast<int>(state.range(1));

  auto mgr = get_shared_memory_manager();

  auto registry = std::make_unique<representation_converter_registry>();
  register_builtin_converters(*registry);

  const memory_space* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  const memory_space* host_space = mgr->get_memory_space(Tier::HOST, hostDevId);
  const memory_space* disk_space = mgr->get_memory_space(Tier::DISK, 0);

  rmm::cuda_stream stream;

  // Create GPU table, convert to host, then to disk
  auto table   = create_benchmark_table_from_bytes(total_bytes, num_columns);
  auto gpu_rep = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *const_cast<memory_space*>(gpu_space));

  auto host_rep = registry->convert<host_data_representation>(*gpu_rep, host_space, stream.view());
  stream.synchronize();

  auto disk_rep = registry->convert<disk_data_representation>(*host_rep, disk_space, stream.view());
  stream.synchronize();

  size_t bytes_transferred = disk_rep->get_size_in_bytes();

  for ([[maybe_unused]] auto _ : state) {
    // No cache eviction needed — pipeline uses O_DIRECT
    auto host_result =
      registry->convert<host_data_representation>(*disk_rep, host_space, stream.view());
    stream.synchronize();
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_transferred));
  state.counters["columns"]          = static_cast<double>(num_columns);
  state.counters["bytes"]            = static_cast<double>(bytes_transferred);
  state.counters["gdsio_write_GiBs"] = GDSIO_WRITE_GIBS;
  state.counters["gdsio_read_GiBs"]  = GDSIO_READ_GIBS;
}

// =============================================================================
// BENCH-02: Column Type Sweep Benchmarks
// =============================================================================

/**
 * @brief Benchmark GPU to Disk conversion with STRING columns.
 * @param state.range(0) Total bytes
 * @param state.range(1) Number of columns
 */
void BM_ConvertGpuToDiskStringColumns(benchmark::State& state)
{
  int64_t total_bytes = state.range(0);
  if (skip_if_oom(state, total_bytes)) return;
  int num_columns = static_cast<int>(state.range(1));

  auto mgr = get_shared_memory_manager();

  auto registry = make_benchmark_registry();

  const memory_space* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  const memory_space* disk_space = mgr->get_memory_space(Tier::DISK, 0);

  rmm::cuda_stream stream;

  auto table   = create_string_benchmark_table(total_bytes, num_columns);
  auto gpu_rep = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *const_cast<memory_space*>(gpu_space));

  size_t bytes_transferred = gpu_rep->get_size_in_bytes();

  for ([[maybe_unused]] auto _ : state) {
    // No cache eviction needed — pipeline uses O_DIRECT
    auto disk_result =
      registry->convert<disk_data_representation>(*gpu_rep, disk_space, stream.view());
    stream.synchronize();
    drop_os_cache(disk_result->get_disk_table().file_path);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_transferred));
  state.counters["columns"]          = static_cast<double>(num_columns);
  state.counters["bytes"]            = static_cast<double>(bytes_transferred);
  state.counters["gdsio_write_GiBs"] = GDSIO_WRITE_GIBS;
  state.counters["gdsio_read_GiBs"]  = GDSIO_READ_GIBS;
}

/**
 * @brief Benchmark GPU to Disk conversion with LIST<INT32> columns.
 * @param state.range(0) Total bytes
 * @param state.range(1) Number of columns
 */
void BM_ConvertGpuToDiskListColumns(benchmark::State& state)
{
  int64_t total_bytes = state.range(0);
  if (skip_if_oom(state, total_bytes)) return;
  int num_columns = static_cast<int>(state.range(1));

  auto mgr = get_shared_memory_manager();

  auto registry = make_benchmark_registry();

  const memory_space* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  const memory_space* disk_space = mgr->get_memory_space(Tier::DISK, 0);

  rmm::cuda_stream stream;

  auto table   = create_list_benchmark_table(total_bytes, num_columns);
  auto gpu_rep = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *const_cast<memory_space*>(gpu_space));

  size_t bytes_transferred = gpu_rep->get_size_in_bytes();

  for ([[maybe_unused]] auto _ : state) {
    // No cache eviction needed — pipeline uses O_DIRECT
    auto disk_result =
      registry->convert<disk_data_representation>(*gpu_rep, disk_space, stream.view());
    stream.synchronize();
    drop_os_cache(disk_result->get_disk_table().file_path);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_transferred));
  state.counters["columns"]          = static_cast<double>(num_columns);
  state.counters["bytes"]            = static_cast<double>(bytes_transferred);
  state.counters["gdsio_write_GiBs"] = GDSIO_WRITE_GIBS;
  state.counters["gdsio_read_GiBs"]  = GDSIO_READ_GIBS;
}

/**
 * @brief Benchmark GPU to Disk conversion with STRUCT<INT64, FLOAT64> columns.
 * @param state.range(0) Total bytes
 * @param state.range(1) Number of columns
 */
void BM_ConvertGpuToDiskStructColumns(benchmark::State& state)
{
  int64_t total_bytes = state.range(0);
  if (skip_if_oom(state, total_bytes)) return;
  int num_columns = static_cast<int>(state.range(1));

  auto mgr = get_shared_memory_manager();

  auto registry = make_benchmark_registry();

  const memory_space* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  const memory_space* disk_space = mgr->get_memory_space(Tier::DISK, 0);

  rmm::cuda_stream stream;

  auto table   = create_struct_benchmark_table(total_bytes, num_columns);
  auto gpu_rep = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *const_cast<memory_space*>(gpu_space));

  size_t bytes_transferred = gpu_rep->get_size_in_bytes();

  for ([[maybe_unused]] auto _ : state) {
    // No cache eviction needed — pipeline uses O_DIRECT
    auto disk_result =
      registry->convert<disk_data_representation>(*gpu_rep, disk_space, stream.view());
    stream.synchronize();
    drop_os_cache(disk_result->get_disk_table().file_path);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_transferred));
  state.counters["columns"]          = static_cast<double>(num_columns);
  state.counters["bytes"]            = static_cast<double>(bytes_transferred);
  state.counters["gdsio_write_GiBs"] = GDSIO_WRITE_GIBS;
  state.counters["gdsio_read_GiBs"]  = GDSIO_READ_GIBS;
}

// =============================================================================
// Benchmark Registrations
// =============================================================================

// BENCH-01 / BENCH-04: Size sweep with numeric columns (GPU <-> Disk)
BENCHMARK(BM_ConvertGpuToDisk)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->Args({1 * MiB, 4})
  ->Args({64 * MiB, 4})
  ->Args({512 * MiB, 4})
  ->Args({4 * GiB, 4})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

BENCHMARK(BM_ConvertDiskToGpu)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->Args({1 * MiB, 4})
  ->Args({64 * MiB, 4})
  ->Args({512 * MiB, 4})
  ->Args({4 * GiB, 4})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// BENCH-01 / BENCH-04: Size sweep with numeric columns (Host <-> Disk)
BENCHMARK(BM_ConvertHostToDisk)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->Args({1 * MiB, 4})
  ->Args({64 * MiB, 4})
  ->Args({512 * MiB, 4})
  ->Args({4 * GiB, 4})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

BENCHMARK(BM_ConvertDiskToHost)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->Args({1 * MiB, 4})
  ->Args({64 * MiB, 4})
  ->Args({512 * MiB, 4})
  ->Args({4 * GiB, 4})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// BENCH-02: Column type sweep (64 MiB fixed size)
BENCHMARK(BM_ConvertGpuToDiskStringColumns)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->Args({64 * MiB, 2})
  ->Args({64 * MiB, 4})
  ->Args({64 * MiB, 8})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

BENCHMARK(BM_ConvertGpuToDiskListColumns)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->Args({64 * MiB, 2})
  ->Args({64 * MiB, 4})
  ->Args({64 * MiB, 8})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

BENCHMARK(BM_ConvertGpuToDiskStructColumns)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->Args({64 * MiB, 2})
  ->Args({64 * MiB, 4})
  ->Args({64 * MiB, 8})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// BENCH-03: Pipeline backend (double-buffered pinned host)
void BM_ConvertGpuToDiskPipeline(benchmark::State& state)
{
  int64_t total_bytes = state.range(0);
  if (skip_if_oom(state, total_bytes)) return;
  int num_columns = static_cast<int>(state.range(1));

  auto mgr = get_shared_memory_manager();

  auto registry = std::make_unique<representation_converter_registry>();
  register_builtin_converters(*registry);

  const memory_space* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  const memory_space* disk_space = mgr->get_memory_space(Tier::DISK, 0);

  rmm::cuda_stream stream;

  auto table   = create_benchmark_table_from_bytes(total_bytes, num_columns);
  auto gpu_rep = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *const_cast<memory_space*>(gpu_space));

  size_t bytes_transferred = gpu_rep->get_size_in_bytes();

  for ([[maybe_unused]] auto _ : state) {
    // No cache eviction needed — pipeline uses O_DIRECT
    auto disk_result =
      registry->convert<disk_data_representation>(*gpu_rep, disk_space, stream.view());
    stream.synchronize();
    drop_os_cache(disk_result->get_disk_table().file_path);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_transferred));
  state.counters["columns"]          = static_cast<double>(num_columns);
  state.counters["bytes"]            = static_cast<double>(bytes_transferred);
  state.counters["gdsio_write_GiBs"] = GDSIO_WRITE_GIBS;
  state.counters["gdsio_read_GiBs"]  = GDSIO_READ_GIBS;
  state.counters["backend"]          = 2;  // 2 = Pipeline
}

BENCHMARK(BM_ConvertGpuToDiskPipeline)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->Args({1 * MiB, 4})
  ->Args({64 * MiB, 4})
  ->Args({512 * MiB, 4})
  ->Args({4 * GiB, 4})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// =============================================================================
// BENCH-04: Read benchmark (Disk -> GPU) using Pipeline backend
// =============================================================================

/**
 * @brief Helper: write a GPU table to disk using the pipeline backend,
 *        returning the disk representation for read benchmarks.
 */
std::unique_ptr<disk_data_representation> write_table_to_disk(cudf::table&& table,
                                                              const memory_space* gpu_space,
                                                              const memory_space* disk_space,
                                                              rmm::cuda_stream_view stream)
{
  auto registry = make_benchmark_registry();

  auto gpu_rep = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *const_cast<memory_space*>(gpu_space));
  auto disk_rep = registry->convert<disk_data_representation>(*gpu_rep, disk_space, stream);
  stream.synchronize();
  return disk_rep;
}

void BM_ConvertDiskToGpuPipeline(benchmark::State& state)
{
  int64_t total_bytes = state.range(0);
  if (skip_if_oom(state, total_bytes)) return;
  int num_columns = static_cast<int>(state.range(1));

  auto mgr = get_shared_memory_manager();

  const memory_space* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  const memory_space* disk_space = mgr->get_memory_space(Tier::DISK, 0);

  rmm::cuda_stream stream;

  auto disk_rep = write_table_to_disk(create_benchmark_table_from_bytes(total_bytes, num_columns),
                                      gpu_space,
                                      disk_space,
                                      stream.view());

  auto registry = make_benchmark_registry();

  size_t bytes_transferred = disk_rep->get_size_in_bytes();
  const auto& disk_file    = disk_rep->get_disk_table().file_path;

  for ([[maybe_unused]] auto _ : state) {
    drop_os_cache(disk_file);
    auto gpu_result =
      registry->convert<gpu_table_representation>(*disk_rep, gpu_space, stream.view());
    stream.synchronize();
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_transferred));
  state.counters["columns"]          = static_cast<double>(num_columns);
  state.counters["bytes"]            = static_cast<double>(bytes_transferred);
  state.counters["gdsio_write_GiBs"] = GDSIO_WRITE_GIBS;
  state.counters["gdsio_read_GiBs"]  = GDSIO_READ_GIBS;
}

BENCHMARK(BM_ConvertDiskToGpuPipeline)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->Args({1 * MiB, 4})
  ->Args({64 * MiB, 4})
  ->Args({512 * MiB, 4})
  ->Args({4 * GiB, 4})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// =============================================================================
// BENCH-05: Column type sweep — write + read benchmarks (64 MiB)
// =============================================================================

/**
 * @brief Helper: write benchmark for a given table creation function.
 */
template <typename TableFactory>
void pipeline_write_benchmark(benchmark::State& state, TableFactory table_factory)
{
  int64_t total_bytes = state.range(0);
  if (skip_if_oom(state, total_bytes)) return;
  int num_columns = static_cast<int>(state.range(1));

  auto mgr = get_shared_memory_manager();

  auto registry = make_benchmark_registry();

  const memory_space* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  const memory_space* disk_space = mgr->get_memory_space(Tier::DISK, 0);

  rmm::cuda_stream stream;

  auto table   = table_factory(total_bytes, num_columns);
  auto gpu_rep = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *const_cast<memory_space*>(gpu_space));

  size_t bytes_transferred = gpu_rep->get_size_in_bytes();

  for ([[maybe_unused]] auto _ : state) {
    auto disk_result =
      registry->convert<disk_data_representation>(*gpu_rep, disk_space, stream.view());
    stream.synchronize();
    drop_os_cache(disk_result->get_disk_table().file_path);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_transferred));
  state.counters["columns"]          = static_cast<double>(num_columns);
  state.counters["bytes"]            = static_cast<double>(bytes_transferred);
  state.counters["gdsio_write_GiBs"] = GDSIO_WRITE_GIBS;
  state.counters["gdsio_read_GiBs"]  = GDSIO_READ_GIBS;
}

/**
 * @brief Helper: read benchmark for a given table creation function.
 */
template <typename TableFactory>
void pipeline_read_benchmark(benchmark::State& state, TableFactory table_factory)
{
  int64_t total_bytes = state.range(0);
  if (skip_if_oom(state, total_bytes)) return;
  int num_columns = static_cast<int>(state.range(1));

  auto mgr = get_shared_memory_manager();

  const memory_space* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  const memory_space* disk_space = mgr->get_memory_space(Tier::DISK, 0);

  rmm::cuda_stream stream;

  auto disk_rep = write_table_to_disk(
    table_factory(total_bytes, num_columns), gpu_space, disk_space, stream.view());

  auto registry = make_benchmark_registry();

  size_t bytes_transferred = disk_rep->get_size_in_bytes();
  const auto& disk_file    = disk_rep->get_disk_table().file_path;

  for ([[maybe_unused]] auto _ : state) {
    drop_os_cache(disk_file);
    auto gpu_result =
      registry->convert<gpu_table_representation>(*disk_rep, gpu_space, stream.view());
    stream.synchronize();
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_transferred));
  state.counters["columns"]          = static_cast<double>(num_columns);
  state.counters["bytes"]            = static_cast<double>(bytes_transferred);
  state.counters["gdsio_write_GiBs"] = GDSIO_WRITE_GIBS;
  state.counters["gdsio_read_GiBs"]  = GDSIO_READ_GIBS;
}

// --- Numeric write/read ---
void BM_WriteNumericPipeline(benchmark::State& s)
{
  pipeline_write_benchmark(s, create_benchmark_table_from_bytes);
}
void BM_ReadNumericPipeline(benchmark::State& s)
{
  pipeline_read_benchmark(s, create_benchmark_table_from_bytes);
}

// --- String write/read ---
void BM_WriteStringPipeline(benchmark::State& s)
{
  pipeline_write_benchmark(s, create_string_benchmark_table);
}
void BM_ReadStringPipeline(benchmark::State& s)
{
  pipeline_read_benchmark(s, create_string_benchmark_table);
}

// --- List write/read ---
void BM_WriteListPipeline(benchmark::State& s)
{
  pipeline_write_benchmark(s, create_list_benchmark_table);
}
void BM_ReadListPipeline(benchmark::State& s)
{
  pipeline_read_benchmark(s, create_list_benchmark_table);
}

// --- Struct write/read ---
void BM_WriteStructPipeline(benchmark::State& s)
{
  pipeline_write_benchmark(s, create_struct_benchmark_table);
}
void BM_ReadStructPipeline(benchmark::State& s)
{
  pipeline_read_benchmark(s, create_struct_benchmark_table);
}

// Size sweep args for type comparison
#define DISK_BACKEND_ARGS           \
  ->Setup(DoSetup)                  \
    ->Teardown(DoTeardown)          \
    ->Args({64 * MiB, 4})           \
    ->Args({512 * MiB, 4})          \
    ->Args({4 * GiB, 4})            \
    ->Unit(benchmark::kMillisecond) \
    ->UseRealTime()

BENCHMARK(BM_WriteNumericPipeline) DISK_BACKEND_ARGS;
BENCHMARK(BM_ReadNumericPipeline) DISK_BACKEND_ARGS;

BENCHMARK(BM_WriteStringPipeline) DISK_BACKEND_ARGS;
BENCHMARK(BM_ReadStringPipeline) DISK_BACKEND_ARGS;

BENCHMARK(BM_WriteListPipeline) DISK_BACKEND_ARGS;
BENCHMARK(BM_ReadListPipeline) DISK_BACKEND_ARGS;

BENCHMARK(BM_WriteStructPipeline) DISK_BACKEND_ARGS;
BENCHMARK(BM_ReadStructPipeline) DISK_BACKEND_ARGS;

}  // namespace
