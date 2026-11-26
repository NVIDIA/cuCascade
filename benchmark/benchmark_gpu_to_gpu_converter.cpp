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
// stops blocking the host before the copy completes.
//
// Benchmarks:
//   BM_ConvertGpuToGpu                  automatic copy route (the peer route on peer-capable GPUs)
//   BM_RawPeerCopy                      matching total bytes, one contiguous buffer per input,
//                                       one host sync: lower bound for the peer route
//   BM_ConvertGpuToGpuHostStaged        host-staged pipeline forced, swept over chunk size x depth
//   BM_ConvertGpuToGpuHostStagedOneShot host-staged route with depth 1 and one chunk covering the
//                                       whole table: a bulk-copy proxy that moves the table in one
//                                       batch per leg with one sync, nothing overlapped. It is not
//                                       the host staging this pipeline replaced, which staged each
//                                       leaf buffer on its own with a sync after it and measured
//                                       1.3x (8M rows) to 3.6x (1K rows) slower than this proxy.
//                                       A table that fits one chunk takes this same path in the
//                                       pipeline, so the two tie there.
//   BM_RawHostStagedCopy                no cudf: matching total bytes as one stream through pinned
//                                       staging slots, source-to-host and host-to-destination legs
//                                       overlapped on two streams: lower bound for the pipeline

#include "cucascade/cudf/builtin_converters.hpp"
#include "cucascade/cudf/gpu_data_representation.hpp"
#include "cucascade/data/representation_converter.hpp"
#include "cucascade/memory/memory_reservation_manager.hpp"
#include "cucascade/memory/reservation_manager_configurator.hpp"
#include "utils/cuda_device_utils.hpp"
#include "utils/mixed_table.hpp"
#include "utils/peer_access_utils.hpp"

#include <cucascade/cuda/event.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/utils/cleanup_utils.hpp>

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <limits>
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

// Rows per input swept by every benchmark; one input is about 310 bytes per row, so the largest is
// about 2.5 GiB.
constexpr std::array<int64_t, 8> kRowCounts{
  1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 8388608};

// Host-staged sweep: chunk size x pipeline depth at every row count.
constexpr std::array<int64_t, 5> kChunkSizes{1 * kMiB, 2 * kMiB, 4 * kMiB, 8 * kMiB, 16 * kMiB};
constexpr std::array<int64_t, 3> kDepths{2, 4, 8};

// Per-GPU pool bounds. The largest row count needs about 20 GiB for 8 resident inputs or outputs
// plus the table under construction and its gather workspace, which the upper bound covers on a
// 32 GiB GPU.
constexpr std::size_t kMinGpuCapacityPerDevice = 1 * kGiB;
constexpr std::size_t kMaxGpuCapacityPerDevice = 28 * kGiB;

// The one-shot proxy stages a whole input (about 2.5 GiB at the largest row count) at once; the
// host space pins 4 x 128 x 1 MiB = 512 MiB at startup and grows on demand up to this cap.
constexpr std::size_t kHostCapacityPerRegion = 4 * kGiB;

static std::unique_ptr<memory_reservation_manager> g_manager;
static std::unique_ptr<representation_converter_registry> g_registry;

/**
 * @brief Per-GPU pool size for inputs of @p rows, from make_mixed_table's conservative budget:
 * 8 resident inputs or outputs plus 1 table set under construction, plus 25% for gather workspace
 * and the 90% reservation limit. Sizing the pool per run keeps setup short for small tables.
 */
[[nodiscard]] std::size_t gpu_capacity_for(int64_t rows)
{
  std::size_t const input_bytes =
    static_cast<std::size_t>(kMixedTablesPerInput) *
    (static_cast<std::size_t>(rows) * test::kMixedTableBytesPerRow + test::kMixedTableFixedBytes);
  std::size_t const peak_bytes = (kNumInputs + 1) * input_bytes;
  return std::clamp(
    peak_bytes + peak_bytes / 4, kMinGpuCapacityPerDevice, kMaxGpuCapacityPerDevice);
}

/**
 * @brief Build a two-GPU manager with NUMA-local host spaces and GPU pools sized for the run's row
 * count.
 *
 * Leaves g_manager null when the run cannot be set up: fewer than two devices, or a GPU without
 * room for its pool. Setup cannot skip a run, so each benchmark does it through
 * skip_unless_set_up().
 *
 * @param state.range(0) Rows per input
 */
void DoSetup(const benchmark::State& state)
{
  if (test::cuda_device_count() < 2) { return; }

  reservation_manager_configurator builder;
  builder.set_gpu_ids({kDstDeviceId, kSrcDeviceId})
    .set_gpu_usage_limit(gpu_capacity_for(state.range(0)))
    .use_numa_id_as_host_id()
    .set_per_numa_region_capacity(kHostCapacityPerRegion);
  try {
    g_manager = std::make_unique<memory_reservation_manager>(builder.build());
  } catch (rmm::bad_alloc const&) {
    // The pools are allocated in full up front; an exception leaving Setup would end the whole
    // benchmark binary.
    return;
  }

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
 * @brief Skip @p state unless DoSetup() built the manager.
 * @return True if the benchmark was skipped.
 */
bool skip_unless_set_up(benchmark::State& state)
{
  if (test::cuda_device_count() < 2) {
    state.SkipWithError("requires two CUDA devices");
    return true;
  }
  if (g_manager == nullptr) {
    state.SkipWithError("not enough free GPU memory for this row count");
    return true;
  }
  return false;
}

/**
 * @brief Skip @p state unless DoSetup() built the manager and the GPU pools accept peer access.
 * @return True if the benchmark was skipped.
 */
bool skip_without_peers(benchmark::State& state)
{
  if (skip_unless_set_up(state)) { return true; }
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
 * @brief Time sequential conversions of N source-GPU partitions to the destination GPU on one
 * thread, through @p registry.
 */
void run_conversions(benchmark::State& state,
                     representation_converter_registry const& registry,
                     cudf::size_type rows)
{
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

  // Warm-up: first touch of the destination pool, the HOST pool and the peer-route probe.
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
 * @brief Sequentially convert N source-GPU partitions over the automatic copy route.
 * @param state.range(0) Rows per input (fixed schema and partition count)
 */
void BM_ConvertGpuToGpu(benchmark::State& state)
{
  if (skip_without_peers(state)) { return; }
  run_conversions(state, *g_registry, static_cast<cudf::size_type>(state.range(0)));
}

/**
 * @brief Sequentially convert N source-GPU partitions over the host-staged pipeline.
 * @param state.range(0) Rows per input
 * @param state.range(1) Chunk size in bytes
 * @param state.range(2) Pipeline depth
 */
void BM_ConvertGpuToGpuHostStaged(benchmark::State& state)
{
  if (skip_unless_set_up(state)) { return; }
  representation_converter_registry registry;
  register_builtin_converters(registry,
                              {.chunk_size_bytes = static_cast<std::size_t>(state.range(1)),
                               .pipeline_depth   = static_cast<std::size_t>(state.range(2)),
                               .route            = copy_route_selection::host_staged});
  run_conversions(state, registry, static_cast<cudf::size_type>(state.range(0)));
  state.counters["chunk_MiB"] = static_cast<double>(state.range(1)) / kMiB;
  state.counters["depth"]     = static_cast<double>(state.range(2));
}

/**
 * @brief Bulk-copy proxy: host-staged route with depth 1 and one chunk per table.
 * @param state.range(0) Rows per input
 */
void BM_ConvertGpuToGpuHostStagedOneShot(benchmark::State& state)
{
  if (skip_unless_set_up(state)) { return; }
  representation_converter_registry registry;
  register_builtin_converters(registry,
                              {.chunk_size_bytes = std::numeric_limits<std::size_t>::max(),
                               .pipeline_depth   = 1,
                               .route            = copy_route_selection::host_staged});
  run_conversions(state, registry, static_cast<cudf::size_type>(state.range(0)));
}

/**
 * @brief Measured device size and column count of one input, including masks and offsets.
 */
[[nodiscard]] std::pair<std::size_t, cudf::size_type> measure_input(cudf::size_type rows)
{
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kSrcDeviceId}};
  rmm::cuda_stream producer;
  auto const inputs = make_inputs(1, rows, producer);
  producer.synchronize();
  return {inputs.front()->get_size_in_bytes(), inputs.front()->get_table_view().num_columns()};
}

/**
 * @brief @p bytes of initialized device memory on the source GPU.
 */
[[nodiscard]] rmm::device_buffer make_source_bytes(std::size_t bytes)
{
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kSrcDeviceId}};
  rmm::device_buffer buf(
    bytes,
    ::cuda::stream_ref{cudaStream_t{cudaStreamDefault}},
    g_manager->get_memory_space(Tier::GPU, kSrcDeviceId)->get_default_allocator());
  CUCASCADE_CUDA_TRY(cudaMemset(buf.data(), 0x22, bytes));
  return buf;
}

/**
 * @brief Lower bound: peer-copy the same buffers, all queued on one stream, one host sync.
 * @param state.range(0) Rows per input, sized using the converter's actual table allocation
 */
void BM_RawPeerCopy(benchmark::State& state)
{
  if (skip_without_peers(state)) { return; }

  cudf::size_type const rows            = static_cast<cudf::size_type>(state.range(0));
  auto const [bytes_per_input, columns] = measure_input(rows);
  std::size_t const total_bytes         = bytes_per_input * kNumInputs;
  rmm::device_buffer src                = make_source_bytes(total_bytes);

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

/**
 * @brief Lower bound for the host-staged pipeline: the same total bytes as one stream through
 * pinned staging slots, with no cudf, no planning and no reservations.
 *
 * Chunk i goes source-to-host on a source-GPU stream into slot i % depth, then host-to-destination
 * on a destination-GPU stream; per-slot events overlap the legs, as in the converter.
 *
 * @param state.range(0) Rows per input, sized using the converter's actual table allocation
 * @param state.range(1) Chunk size in bytes
 * @param state.range(2) Pipeline depth
 */
void BM_RawHostStagedCopy(benchmark::State& state)
{
  if (skip_unless_set_up(state)) { return; }

  cudf::size_type const rows            = static_cast<cudf::size_type>(state.range(0));
  auto const chunk_size                 = static_cast<std::size_t>(state.range(1));
  auto const depth                      = static_cast<std::size_t>(state.range(2));
  auto const [bytes_per_input, columns] = measure_input(rows);
  std::size_t const total_bytes         = bytes_per_input * kNumInputs;
  rmm::device_buffer src                = make_source_bytes(total_bytes);

  std::byte* staging = nullptr;
  CUCASCADE_CUDA_TRY(
    cudaHostAlloc(reinterpret_cast<void**>(&staging), chunk_size * depth, cudaHostAllocPortable));
  auto free_staging = utils::on_scope_exit(
    [staging]() noexcept { CUCASCADE_ASSERT_CUDA_SUCCESS(cudaFreeHost(staging)); });

  std::vector<cucascade::cuda::cuda_event> filled;
  std::vector<cucascade::cuda::cuda_event> drained;
  rmm::cuda_stream src_stream = [&] {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kSrcDeviceId}};
    for (std::size_t i = 0; i < depth; ++i) {
      filled.emplace_back();
    }
    return rmm::cuda_stream{};
  }();

  rmm::cuda_set_device_raii const dst_guard{rmm::cuda_device_id{kDstDeviceId}};
  for (std::size_t i = 0; i < depth; ++i) {
    drained.emplace_back();
  }
  rmm::cuda_stream consumer;
  rmm::device_buffer dst(
    total_bytes,
    consumer,
    g_manager->get_memory_space(Tier::GPU, kDstDeviceId)->get_default_allocator());
  consumer.synchronize();

  auto const* src_bytes = static_cast<const std::byte*>(src.data());
  auto* dst_bytes       = static_cast<std::byte*>(dst.data());
  for (auto _ : state) {
    std::size_t chunk = 0;
    for (std::size_t offset = 0; offset < total_bytes; offset += chunk_size, ++chunk) {
      std::size_t const slot  = chunk % depth;
      std::size_t const bytes = std::min(chunk_size, total_bytes - offset);
      std::byte* host         = staging + slot * chunk_size;
      {
        rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kSrcDeviceId}};
        if (chunk >= depth) { drained[slot].wait(src_stream); }
        CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
          host, src_bytes + offset, bytes, cudaMemcpyDeviceToHost, src_stream.value()));
        filled[slot].record(src_stream);
      }
      filled[slot].wait(consumer);
      CUCASCADE_CUDA_TRY(
        cudaMemcpyAsync(dst_bytes + offset, host, bytes, cudaMemcpyHostToDevice, consumer.value()));
      drained[slot].record(consumer);
    }
    consumer.synchronize();
  }

  set_counters(state, total_bytes, kNumInputs, columns);
  state.counters["rows"]      = rows;
  state.counters["chunk_MiB"] = static_cast<double>(chunk_size) / kMiB;
  state.counters["depth"]     = static_cast<double>(depth);
}

void gpu_to_gpu_args(benchmark::internal::Benchmark* b)
{
  for (int64_t rows : kRowCounts) {
    b->Arg(rows);
  }
  b->ArgName("rows");
}

void host_staged_args(benchmark::internal::Benchmark* b)
{
  for (int64_t rows : kRowCounts) {
    for (int64_t chunk : kChunkSizes) {
      for (int64_t depth : kDepths) {
        b->Args({rows, chunk, depth});
      }
    }
  }
  b->ArgNames({"rows", "chunk", "depth"});
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

BENCHMARK(BM_ConvertGpuToGpuHostStaged)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->Apply(host_staged_args)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

BENCHMARK(BM_ConvertGpuToGpuHostStagedOneShot)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->Apply(gpu_to_gpu_args)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

BENCHMARK(BM_RawHostStagedCopy)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->Apply(host_staged_args)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

}  // namespace
