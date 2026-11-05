/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Cross-GPU (GPU -> GPU) conversion benchmarks.
//
// Models the consumer side of a multi-GPU exchange: one task thread on the destination GPU pulls
// N partitions produced on the source GPU, one after another, the way Sirius's
// pipelineable_operator_data::prepare_for_processing does. Each iteration measures the wall time
// from the first conversion call until the consumer stream may read every result:
//
//   src GPU    producer stream ──●W (writer event, recorded once at setup)
//   dst GPU    host: convert(in_0) ─ convert(in_1) ─ ... ─ convert(in_N-1)
//                    consumer stream ◇wait(R_0) ◇wait(R_1) ... ◇wait(R_N-1) ── sync
//
// The trailing wait on each result's writer event keeps the measurement valid if the converter
// stops blocking the host before the copy completes. BM_RawPeerCopy copies matching total bytes
// in one contiguous buffer per input, with a single host sync: a lower bound for the converter.

#include "cucascade/cudf/builtin_converters.hpp"
#include "cucascade/cudf/gpu_data_representation.hpp"
#include "cucascade/data/representation_converter.hpp"
#include "cucascade/memory/memory_reservation_manager.hpp"
#include "cucascade/memory/reservation_manager_configurator.hpp"
#include "utils/cuda_device_utils.hpp"
#include "utils/mixed_table.hpp"
#include "utils/peer_access_utils.hpp"

#include <cucascade/error.hpp>
#include <cucascade/memory/common.hpp>

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

namespace {

using namespace cucascade;
using namespace cucascade::memory;

constexpr uint64_t kKiB = 1024ULL;
constexpr uint64_t kMiB = 1024ULL * kKiB;
constexpr uint64_t kGiB = 1024ULL * kMiB;

constexpr int kSrcDeviceId = 1;
constexpr int kDstDeviceId = 0;

// Keep the consumer fan-in fixed; only rows per partition vary.
constexpr std::size_t kNumInputs = 8;

// Mixed tables concatenated per input: 2 x 16 columns to have 32 columns per partition.
constexpr int kMixedTablesPerInput = 2;

// Rows per input swept by every benchmark.
constexpr std::array<int64_t, 5> kRowCounts{1024, 4096, 16384, 65536, 262144};
constexpr std::size_t kMaxRows = static_cast<std::size_t>(std::ranges::max(kRowCounts));

// Per-GPU memory estimate at kMaxRows, from make_mixed_table's conservative budget:
//   input  = 2 tables x (262144 rows x 512 B + 32 KiB) ≈ 256 MiB
//   peak   = (8 resident inputs or outputs + 1 table set under construction) x input ≈ 2.25 GiB
//   pool   = peak + 25% for gather workspace and the 90% reservation limit ≈ 2.8 GiB
constexpr std::size_t kInputBytes =
  static_cast<std::size_t>(kMixedTablesPerInput) *
  (kMaxRows * test::kMixedTableBytesPerRow + test::kMixedTableFixedBytes);
constexpr std::size_t kPeakBytesPerGpu      = (kNumInputs + 1) * kInputBytes;
constexpr std::size_t kGpuCapacityPerDevice = kPeakBytesPerGpu + kPeakBytesPerGpu / 4;

// Host staging is unused (benchmarks skip without peer access); the host space still pins
// 4 x 128 x 1 MiB = 512 MiB at startup, so cap just above that.
constexpr std::size_t kHostCapacityPerRegion = 1 * kGiB;

static std::unique_ptr<memory_reservation_manager> g_manager;
static std::unique_ptr<representation_converter_registry> g_registry;

/**
 * @brief Build a two-GPU manager with NUMA-local host spaces and peer-granted GPU pools.
 */
void DoSetup([[maybe_unused]] const benchmark::State& state)
{
  if (g_manager) { return; }

  reservation_manager_configurator builder;
  builder.set_gpu_ids({kDstDeviceId, kSrcDeviceId})
    .set_gpu_usage_limit(kGpuCapacityPerDevice)
    .use_numa_id_as_host_id()
    .set_per_numa_region_capacity(kHostCapacityPerRegion);
  g_manager = std::make_unique<memory_reservation_manager>(builder.build());

  g_registry = std::make_unique<representation_converter_registry>();
  register_builtin_converters(*g_registry);
}

void DoTeardown([[maybe_unused]] const benchmark::State& state)
{
  g_registry.reset();
  g_manager.reset();
}

/**
 * @brief Build one partition by concatenating the columns of kMixedTablesPerInput mixed tables.
 *
 * Doubling the mixed table keeps 32 columns, weighting the run toward per-column
 * synchronization (string chars-size readbacks, INT32 and INT64 offsets) over raw bytes.
 */
std::unique_ptr<cudf::table> make_table(cudf::size_type rows,
                                        const memory_space& space,
                                        ::cuda::stream_ref stream)
{
  // make_mixed_table records an event on the current device, which must own @p stream.
  rmm::cuda_set_device_raii const device_guard{rmm::cuda_device_id{space.get_device_id()}};
  rmm::device_async_resource_ref const mr = space.get_default_allocator();

  std::vector<std::unique_ptr<cudf::column>> columns;
  for (int i = 0; i < kMixedTablesPerInput; ++i) {
    std::vector<std::unique_ptr<cudf::column>> part =
      test::make_mixed_table(rows, stream, mr)->release();
    columns.insert(
      columns.end(), std::make_move_iterator(part.begin()), std::make_move_iterator(part.end()));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

/**
 * @brief Skip @p state unless two devices exist and their pools accept peer access.
 * @return True if the benchmark was skipped.
 */
bool skip_without_peers(benchmark::State& state)
{
  if (test::cuda_device_count() < 2) {
    state.SkipWithError("requires two CUDA devices");
    return true;
  }
  if (!test::grant_two_way_pool_access(*g_manager->get_memory_space(Tier::GPU, kSrcDeviceId),
                                       *g_manager->get_memory_space(Tier::GPU, kDstDeviceId))
         .granted()) {
    state.SkipWithError("peer pool access not granted on this hardware");
    return true;
  }
  return false;
}

/**
 * @brief Build @p count source representations on the source GPU, each carrying a writer event
 *        recorded on @p producer, as a Sirius partition task would leave them.
 */
std::vector<std::unique_ptr<gpu_table_representation>> make_inputs(std::size_t count,
                                                                   cudf::size_type rows,
                                                                   ::cuda::stream_ref producer)
{
  memory_space* src_space = g_manager->get_memory_space(Tier::GPU, kSrcDeviceId);

  std::vector<std::unique_ptr<gpu_table_representation>> inputs;
  inputs.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    inputs.push_back(std::make_unique<gpu_table_representation>(
      make_table(rows, *src_space, producer), *src_space, producer));
  }
  return inputs;
}

void set_counters(benchmark::State& state,
                  std::size_t bytes_per_iteration,
                  std::size_t inputs,
                  cudf::size_type columns)
{
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_per_iteration));
  state.counters["inputs"]      = static_cast<double>(inputs);
  state.counters["columns"]     = static_cast<double>(columns);
  state.counters["MiB"]         = static_cast<double>(bytes_per_iteration) / kMiB;
  std::size_t const conversions = static_cast<std::size_t>(state.iterations()) * inputs;
  state.counters["per_input"]   = benchmark::Counter(
    static_cast<double>(conversions), benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}

/**
 * @brief Sequentially convert N source-GPU partitions to the destination GPU on one thread.
 * @param state.range(0) Rows per input (fixed schema and partition count)
 */
void BM_ConvertGpuToGpu(benchmark::State& state)
{
  if (skip_without_peers(state)) { return; }

  cudf::size_type const rows = static_cast<cudf::size_type>(state.range(0));

  rmm::cuda_stream producer = [] {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kSrcDeviceId}};
    return rmm::cuda_stream{};
  }();
  auto const inputs = make_inputs(kNumInputs, rows, producer);
  producer.synchronize();

  const memory_space* dst_space = g_manager->get_memory_space(Tier::GPU, kDstDeviceId);
  rmm::cuda_set_device_raii const dst_guard{rmm::cuda_device_id{kDstDeviceId}};
  rmm::cuda_stream consumer;

  std::size_t bytes_per_iteration = 0;
  for (auto const& input : inputs) {
    bytes_per_iteration += input->get_size_in_bytes();
  }

  // Warm-up: first touch of the destination pool and the peer-route probe.
  {
    auto warm = g_registry->convert<gpu_table_representation>(*inputs[0], dst_space, consumer);
    warm->wait_for_writer(consumer);
    consumer.synchronize();
  }

  std::vector<std::unique_ptr<gpu_table_representation>> outputs;
  outputs.reserve(kNumInputs);
  for (auto _ : state) {
    for (auto const& input : inputs) {
      outputs.push_back(g_registry->convert<gpu_table_representation>(*input, dst_space, consumer));
    }

    // Consumer may read every output only after its writer event.
    for (auto const& output : outputs) {
      output->wait_for_writer(consumer);
    }
    consumer.synchronize();

    // Free outside the timed region; frees are stream-ordered on each output's copy stream.
    state.PauseTiming();
    outputs.clear();
    CUCASCADE_CUDA_TRY(cudaDeviceSynchronize());
    state.ResumeTiming();
  }

  set_counters(
    state, bytes_per_iteration, kNumInputs, inputs.front()->get_table_view().num_columns());
  state.counters["rows"] = rows;
}

/**
 * @brief Lower bound: peer-copy the same buffers, all queued on one stream, one host sync.
 * @param state.range(0) Rows per input, sized using the converter's actual table allocation
 */
void BM_RawPeerCopy(benchmark::State& state)
{
  if (skip_without_peers(state)) { return; }

  cudf::size_type const rows = static_cast<cudf::size_type>(state.range(0));
  // Use the same schema's measured size, including masks and offsets, not a byte estimate.
  auto const [bytes_per_input, columns] = [&] {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kSrcDeviceId}};
    rmm::cuda_stream producer;
    auto const inputs = make_inputs(1, rows, producer);
    producer.synchronize();
    return std::pair{inputs.front()->get_size_in_bytes(),
                     inputs.front()->get_table_view().num_columns()};
  }();
  std::size_t const total_bytes = bytes_per_input * kNumInputs;

  rmm::device_buffer src = [&] {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kSrcDeviceId}};
    rmm::device_buffer buf(
      total_bytes,
      ::cuda::stream_ref{cudaStream_t{cudaStreamDefault}},
      g_manager->get_memory_space(Tier::GPU, kSrcDeviceId)->get_default_allocator());
    CUCASCADE_CUDA_TRY(cudaMemset(buf.data(), 0x22, total_bytes));
    return buf;
  }();

  rmm::cuda_set_device_raii const dst_guard{rmm::cuda_device_id{kDstDeviceId}};
  rmm::cuda_stream consumer;
  rmm::device_buffer dst(
    total_bytes,
    consumer,
    g_manager->get_memory_space(Tier::GPU, kDstDeviceId)->get_default_allocator());
  consumer.synchronize();

  for (auto _ : state) {
    for (std::size_t i = 0; i < kNumInputs; ++i) {
      std::size_t const offset = i * bytes_per_input;
      CUCASCADE_CUDA_TRY(cudaMemcpyPeerAsync(static_cast<std::byte*>(dst.data()) + offset,
                                             kDstDeviceId,
                                             static_cast<const std::byte*>(src.data()) + offset,
                                             kSrcDeviceId,
                                             bytes_per_input,
                                             consumer.value()));
    }
    consumer.synchronize();
  }

  set_counters(state, total_bytes, kNumInputs, columns);
  state.counters["rows"] = rows;
}

void gpu_to_gpu_args(benchmark::internal::Benchmark* b)
{
  for (int64_t rows : kRowCounts) {
    b->Arg(rows);
  }
  b->ArgName("rows");
}

BENCHMARK(BM_ConvertGpuToGpu)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->Apply(gpu_to_gpu_args)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

BENCHMARK(BM_RawPeerCopy)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->Apply(gpu_to_gpu_args)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

}  // namespace
