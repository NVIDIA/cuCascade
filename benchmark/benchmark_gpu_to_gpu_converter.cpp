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
//
// BM_ConvertGpuToGpu needs peer access between the two devices. The Staged benchmarks force the
// route through pinned host memory instead, which is what device pairs without peer access take,
// so they run on any two GPUs. BM_ConvertGpuToGpuStagedPairs runs one such consumer per directed
// device pair at the same time and reports their combined throughput.

#include "builtin_converters_detail.hpp"
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
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

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
#include <future>
#include <iterator>
#include <memory>
#include <numeric>
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

// The host space pins 4 x 128 x 1 MiB = 512 MiB at startup, so cap just above that. A staged copy
// borrows host blocks for one column buffer at a time, a few MiB per consumer at kMaxRows.
constexpr std::size_t kHostCapacityPerRegion = 1 * kGiB;

static std::unique_ptr<memory_reservation_manager> g_manager;
static std::unique_ptr<representation_converter_registry> g_registry;
// Same converters, with every GPU -> GPU conversion forced through pinned host memory.
static std::unique_ptr<representation_converter_registry> g_staged_registry;

void make_registries()
{
  g_registry = std::make_unique<representation_converter_registry>();
  register_builtin_converters(*g_registry);

  g_staged_registry = std::make_unique<representation_converter_registry>();
  cucascade::detail::register_builtin_converters(
    *g_staged_registry,
    cucascade::detail::gpu_to_gpu_converter_options{.force_staged_route = true});
}

/**
 * @brief Build a two-GPU manager with NUMA-local host spaces and peer-granted GPU pools.
 */
void DoSetup([[maybe_unused]] const benchmark::State& state)
{
  // can't skip in setup (requires mutable state)
  // simply exit early, and skip in the benchmark
  if (test::cuda_device_count() < 2) { return; }

  reservation_manager_configurator builder;
  builder.set_gpu_ids({kDstDeviceId, kSrcDeviceId})
    .set_gpu_usage_limit(kGpuCapacityPerDevice)
    .use_numa_id_as_host_id()
    .set_per_numa_region_capacity(kHostCapacityPerRegion);
  g_manager = std::make_unique<memory_reservation_manager>(builder.build());

  make_registries();
}

/**
 * @brief Build a manager over every visible GPU, for the benchmark that runs several device pairs.
 *
 * Each GPU holds the inputs of one pair and the outputs of another, so it gets twice the capacity
 * of the two-GPU setup.
 */
void DoSetupAllGpus([[maybe_unused]] const benchmark::State& state)
{
  int const devices = test::cuda_device_count();
  if (devices < 2) { return; }

  std::vector<int> gpu_ids(static_cast<std::size_t>(devices));
  std::iota(gpu_ids.begin(), gpu_ids.end(), 0);

  reservation_manager_configurator builder;
  builder.set_gpu_ids(gpu_ids)
    .set_gpu_usage_limit(2 * kGpuCapacityPerDevice)
    .use_numa_id_as_host_id()
    .set_per_numa_region_capacity(kHostCapacityPerRegion);
  g_manager = std::make_unique<memory_reservation_manager>(builder.build());

  make_registries();
}

void DoTeardown([[maybe_unused]] const benchmark::State& state)
{
  g_staged_registry.reset();
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
 * @brief Skip @p state unless two devices exist.
 * @return True if the benchmark was skipped.
 */
bool skip_without_two_devices(benchmark::State& state)
{
  if (test::cuda_device_count() < 2) {
    state.SkipWithError("requires two CUDA devices");
    return true;
  }
  return false;
}

/**
 * @brief Peak MiB the HOST spaces have handed out since the manager was built.
 *
 * Staged copies borrow their host blocks there and nothing else in these benchmarks does.
 */
double peak_host_mib()
{
  std::size_t bytes = 0;
  for (memory_space const* space : g_manager->get_memory_spaces_for_tier(Tier::HOST)) {
    bytes += space->get_memory_resource_of<Tier::HOST>()->get_peak_total_allocated_bytes();
  }
  return static_cast<double>(bytes) / static_cast<double>(kMiB);
}

/**
 * @brief Build @p count source representations on @p src_device, each carrying a writer event
 *        recorded on @p producer, as a Sirius partition task would leave them.
 */
std::vector<std::unique_ptr<gpu_table_representation>> make_inputs(std::size_t count,
                                                                   cudf::size_type rows,
                                                                   ::cuda::stream_ref producer,
                                                                   int src_device = kSrcDeviceId)
{
  memory_space* src_space = g_manager->get_memory_space(Tier::GPU, src_device);

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
 * @brief Sequentially convert N source-GPU partitions to the destination GPU on one thread, with
 *        the converters of @p registry.
 * @param state.range(0) Rows per input (fixed schema and partition count)
 */
void run_sequential_conversions(benchmark::State& state,
                                representation_converter_registry& registry)
{
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
    auto warm = registry.convert<gpu_table_representation>(*inputs[0], dst_space, consumer);
    warm->wait_for_writer(consumer);
    consumer.synchronize();
  }

  std::vector<std::unique_ptr<gpu_table_representation>> outputs;
  outputs.reserve(kNumInputs);
  for (auto _ : state) {
    for (auto const& input : inputs) {
      outputs.push_back(registry.convert<gpu_table_representation>(*input, dst_space, consumer));
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
 * @brief The consumer fan-in over the peer route.
 * @param state.range(0) Rows per input
 */
void BM_ConvertGpuToGpu(benchmark::State& state)
{
  if (skip_without_peers(state)) { return; }

  run_sequential_conversions(state, *g_registry);
}

/**
 * @brief The same consumer fan-in with every conversion staged through pinned host memory.
 * @param state.range(0) Rows per input
 */
void BM_ConvertGpuToGpuStaged(benchmark::State& state)
{
  if (skip_without_two_devices(state)) { return; }

  run_sequential_conversions(state, *g_staged_registry);
  state.counters["staging_peak_MiB"] = peak_host_mib();
}

/// One directed device pair of BM_ConvertGpuToGpuStagedPairs, with its own streams and tables.
struct staged_pair {
  int src_device;
  int dst_device;
  rmm::cuda_stream producer;  ///< On src_device; wrote the inputs
  rmm::cuda_stream consumer;  ///< On dst_device; reads the outputs
  std::vector<std::unique_ptr<gpu_table_representation>> inputs;
  std::vector<std::unique_ptr<gpu_table_representation>> outputs;
};

/**
 * @brief Directed pairs for @p devices GPUs, each device paired with the one half the count away.
 *
 * Both directions of a pair are adjacent, e.g. for 4 devices: 0>2, 2>0, 1>3, 3>1. On a machine
 * with two GPUs behind each of two PCIe switches these are the pairs that cross switches.
 */
std::vector<std::pair<int, int>> directed_pairs(int devices)
{
  std::vector<std::pair<int, int>> pairs;
  int const half = devices / 2;
  for (int low = 0; low < half; ++low) {
    pairs.emplace_back(low, low + half);
    pairs.emplace_back(low + half, low);
  }
  return pairs;
}

/**
 * @brief Run the consumer fan-in on several directed device pairs at once, one thread per pair,
 *        with every conversion staged through pinned host memory.
 * @param state.range(0) Rows per input
 * @param state.range(1) Number of directed pairs running concurrently
 */
void BM_ConvertGpuToGpuStagedPairs(benchmark::State& state)
{
  if (skip_without_two_devices(state)) { return; }

  cudf::size_type const rows = static_cast<cudf::size_type>(state.range(0));
  auto const num_pairs       = static_cast<std::size_t>(state.range(1));

  auto const candidates = directed_pairs(test::cuda_device_count());
  if (candidates.size() < num_pairs) {
    state.SkipWithError("not enough CUDA devices for this many pairs");
    return;
  }

  auto const make_stream = [](int device) {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
    return rmm::cuda_stream{};
  };

  std::vector<std::unique_ptr<staged_pair>> pairs;
  pairs.reserve(num_pairs);
  std::size_t bytes_per_iteration = 0;
  for (std::size_t i = 0; i < num_pairs; ++i) {
    auto const [src_device, dst_device] = candidates[i];
    pairs.emplace_back(new staged_pair{
      src_device, dst_device, make_stream(src_device), make_stream(dst_device), {}, {}});
    staged_pair& pair = *pairs.back();

    pair.inputs = make_inputs(kNumInputs, rows, pair.producer, src_device);
    pair.outputs.reserve(kNumInputs);
    {
      rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{src_device}};
      pair.producer.synchronize();
    }
    for (auto const& input : pair.inputs) {
      bytes_per_iteration += input->get_size_in_bytes();
    }
  }

  auto const convert_pair = [](staged_pair& pair) {
    const memory_space* dst_space = g_manager->get_memory_space(Tier::GPU, pair.dst_device);
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{pair.dst_device}};
    for (auto const& input : pair.inputs) {
      pair.outputs.push_back(
        g_staged_registry->convert<gpu_table_representation>(*input, dst_space, pair.consumer));
    }
    // The consumer may read every output only after its writer event.
    for (auto const& output : pair.outputs) {
      output->wait_for_writer(pair.consumer);
    }
    pair.consumer.synchronize();
  };

  auto const free_outputs = [](staged_pair& pair) {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{pair.dst_device}};
    pair.outputs.clear();
    CUCASCADE_CUDA_TRY(cudaDeviceSynchronize());
  };

  // Warm-up: first touch of each destination pool.
  for (auto const& pair : pairs) {
    convert_pair(*pair);
    free_outputs(*pair);
  }

  std::vector<std::future<void>> workers;
  workers.reserve(num_pairs);
  for (auto _ : state) {
    for (auto const& pair : pairs) {
      workers.push_back(std::async(std::launch::async, convert_pair, std::ref(*pair)));
    }
    for (auto& worker : workers) {
      worker.get();
    }

    // Free outside the timed region.
    state.PauseTiming();
    workers.clear();
    for (auto const& pair : pairs) {
      free_outputs(*pair);
    }
    state.ResumeTiming();
  }

  set_counters(state,
               bytes_per_iteration,
               kNumInputs * num_pairs,
               pairs.front()->inputs.front()->get_table_view().num_columns());
  state.counters["rows"]             = rows;
  state.counters["pairs"]            = static_cast<double>(num_pairs);
  state.counters["staging_peak_MiB"] = peak_host_mib();

  // Free each pair's inputs with their device current, before the streams go away.
  for (auto const& pair : pairs) {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{pair->src_device}};
    pair->inputs.clear();
    CUCASCADE_CUDA_TRY(cudaDeviceSynchronize());
  }
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

void staged_pairs_args(benchmark::internal::Benchmark* b)
{
  for (int64_t pairs : {1, 2, 4}) {
    for (int64_t rows : kRowCounts) {
      b->Args({rows, pairs});
    }
  }
  b->ArgNames({"rows", "pairs"});
}

BENCHMARK(BM_ConvertGpuToGpu)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->Apply(gpu_to_gpu_args)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

BENCHMARK(BM_ConvertGpuToGpuStaged)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->Apply(gpu_to_gpu_args)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

BENCHMARK(BM_ConvertGpuToGpuStagedPairs)
  ->Setup(DoSetupAllGpus)
  ->Teardown(DoTeardown)
  ->Apply(staged_pairs_args)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

BENCHMARK(BM_RawPeerCopy)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->Apply(gpu_to_gpu_args)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

}  // namespace
