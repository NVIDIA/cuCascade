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
#include "cucascade/memory/reservation_aware_resource_adaptor.hpp"
#include "cucascade/memory/reservation_manager_configurator.hpp"

#include <cucascade/error.hpp>
#include <cucascade/memory/common.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace {

using namespace cucascade;
using namespace cucascade::memory;

constexpr uint64_t KiB = 1024ULL;
constexpr uint64_t MiB = 1024ULL * KiB;
constexpr uint64_t GiB = 1024ULL * MiB;

constexpr int src_device_id = 1;
constexpr int dst_device_id = 0;

constexpr std::size_t gpu_capacity_per_device  = 32 * GiB;
constexpr std::size_t host_capacity_per_region = 4 * GiB;

// Keep the consumer fan-in fixed; only rows per partition vary.
constexpr std::size_t num_inputs        = 8;
constexpr std::size_t columns_per_table = 32;

// Every generated string has this many bytes; both offset widths are exercised.
constexpr cudf::size_type string_length = 16;

static std::unique_ptr<memory_reservation_manager> g_manager;
static std::unique_ptr<representation_converter_registry> g_registry;

/**
 * @brief Build a two-GPU manager with NUMA-local host spaces and peer-granted GPU pools.
 */
void DoSetup([[maybe_unused]] const benchmark::State& state)
{
  if (g_manager) { return; }

  reservation_manager_configurator builder;
  builder.set_gpu_ids({dst_device_id, src_device_id})
    .set_gpu_usage_limit(gpu_capacity_per_device)
    .use_numa_id_as_host_id()
    .set_per_numa_region_capacity(host_capacity_per_region);
  g_manager = std::make_unique<memory_reservation_manager>(builder.build());

  g_registry = std::make_unique<representation_converter_registry>();
  register_builtin_converters(*g_registry);
}

void DoTeardown([[maybe_unused]] const benchmark::State& state)
{
  g_registry.reset();
  g_manager.reset();
}

bool has_two_devices()
{
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count >= 2;
}

const memory_space* gpu_space(int device_id)
{
  return g_manager->get_memory_space(Tier::GPU, device_id);
}

/**
 * @brief Grant each GPU pool read/write access to the other GPU, as Sirius does at startup.
 *        Without the grant the converter may stage through host memory instead of peer DMA.
 * @return False if the hardware or driver refuses the grant.
 */
bool grant_peer_pools()
{
  for (auto const [owner, accessor] :
       {std::pair{src_device_id, dst_device_id}, std::pair{dst_device_id, src_device_id}}) {
    auto const pool = gpu_space(owner)->get_memory_resource_of<Tier::GPU>()->pool_handle();
    if (pool == nullptr) { return false; }
    if (!grant_pool_peer_access(pool, owner, accessor).granted()) { return false; }
  }
  return true;
}

std::unique_ptr<cudf::column> make_fixed_column(cudf::size_type rows,
                                                cudf::mask_state mask,
                                                ::cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  auto const type  = cudf::data_type{cudf::type_id::INT64};
  auto col         = cudf::make_fixed_width_column(type, rows, mask, stream, mr);
  auto const bytes = static_cast<std::size_t>(rows) * cudf::size_of(type);
  CUCASCADE_CUDA_TRY(cudaMemsetAsync(col->mutable_view().head(), 0, bytes, stream.get()));
  return col;
}

template <typename Offset = int32_t>
std::unique_ptr<cudf::column> make_string_column(cudf::size_type rows,
                                                 ::cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  // offsets = [0, L, 2L, ..., rows * L]
  cudf::numeric_scalar<Offset> const init{0, true, stream};
  cudf::numeric_scalar<Offset> const step{string_length, true, stream};
  auto offsets = cudf::sequence(rows + 1, init, step, stream, mr);

  auto const chars_bytes = static_cast<std::size_t>(rows) * string_length;
  rmm::device_buffer chars(chars_bytes, stream, mr);
  CUCASCADE_CUDA_TRY(cudaMemsetAsync(chars.data(), 'a', chars_bytes, stream.get()));

  return cudf::make_strings_column(
    rows,
    std::move(offsets),
    std::move(chars),
    0,
    cudf::create_null_mask(rows, cudf::mask_state::ALL_VALID, stream, mr));
}

/**
 * @brief Build 32 columns weighted toward per-column synchronization, not type coverage.
 *
 * All-valid masks deliberately exercise nullable copies without changing nested values.
 * Both string offset widths are present. Error paths, empty/degenerate columns, same-device
 * clones and hardware-dependent copy routes are not dimensions of this row-count sweep.
 */
std::unique_ptr<cudf::table> make_table(cudf::size_type rows,
                                        const memory_space& space,
                                        ::cuda::stream_ref stream)
{
  rmm::cuda_set_device_raii const device_guard{rmm::cuda_device_id{space.get_device_id()}};
  auto const mr = space.get_default_allocator();

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(columns_per_table);
  constexpr int nullable_fixed_columns = 4;
  constexpr int string_pairs           = 12;
  constexpr int list_columns           = 2;
  static_assert(nullable_fixed_columns + 2 * string_pairs + list_columns + 2 == columns_per_table);
  for (int i = 0; i < nullable_fixed_columns; ++i) {
    columns.push_back(make_fixed_column(rows, cudf::mask_state::ALL_VALID, stream, mr));
  }

  // Strings add chars-size readback waits on top of nullable-mask synchronization.
  for (int i = 0; i < string_pairs; ++i) {
    columns.push_back(make_string_column<int32_t>(rows, stream, mr));
    columns.push_back(make_string_column<int64_t>(rows, stream, mr));
  }

  constexpr cudf::size_type elements_per_list = 4;
  cudf::numeric_scalar<int32_t> const init{0, true, stream};
  cudf::numeric_scalar<int32_t> const step{elements_per_list, true, stream};
  for (int i = 0; i < list_columns; ++i) {
    columns.push_back(cudf::make_lists_column(
      rows,
      cudf::sequence(rows + 1, init, step, stream, mr),
      make_string_column(rows * elements_per_list, stream, mr),
      0,
      cudf::create_null_mask(rows, cudf::mask_state::ALL_VALID, stream, mr)));
  }

  std::vector<std::unique_ptr<cudf::column>> fields;
  fields.push_back(make_fixed_column(rows, cudf::mask_state::ALL_VALID, stream, mr));
  fields.push_back(make_string_column(rows, stream, mr));
  columns.push_back(std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::STRUCT},
    rows,
    rmm::device_buffer{},
    cudf::create_null_mask(rows, cudf::mask_state::ALL_VALID, stream, mr),
    0,
    std::move(fields)));

  // A single key makes zero-filled indices valid at every row count.
  auto indices = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, rows, cudf::mask_state::UNALLOCATED, stream, mr);
  CUCASCADE_CUDA_TRY(cudaMemsetAsync(indices->mutable_view().head(),
                                     0,
                                     static_cast<std::size_t>(rows) * sizeof(int32_t),
                                     stream.get()));
  columns.push_back(cudf::make_dictionary_column(
    make_fixed_column(1, cudf::mask_state::UNALLOCATED, stream, mr),
    std::move(indices),
    cudf::create_null_mask(rows, cudf::mask_state::ALL_VALID, stream, mr),
    0));

  // Some factories use the default stream; finish setup before recording the writer event.
  CUCASCADE_CUDA_TRY(cudaDeviceSynchronize());
  return std::make_unique<cudf::table>(std::move(columns));
}

/**
 * @brief Build @p count source representations on the source GPU, each carrying a writer event
 *        recorded on @p producer, as a Sirius partition task would leave them.
 */
std::vector<std::unique_ptr<gpu_table_representation>> make_inputs(std::size_t count,
                                                                   cudf::size_type rows,
                                                                   ::cuda::stream_ref producer)
{
  auto* src_space = const_cast<memory_space*>(gpu_space(src_device_id));

  std::vector<std::unique_ptr<gpu_table_representation>> inputs;
  inputs.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    inputs.push_back(std::make_unique<gpu_table_representation>(
      make_table(rows, *src_space, producer), *src_space, producer));
  }
  return inputs;
}

void set_counters(benchmark::State& state, std::size_t bytes_per_iteration, std::size_t inputs)
{
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_per_iteration));
  state.counters["inputs"]    = static_cast<double>(inputs);
  state.counters["columns"]   = static_cast<double>(columns_per_table);
  state.counters["MiB"]       = static_cast<double>(bytes_per_iteration) / MiB;
  auto const conversions      = static_cast<std::size_t>(state.iterations()) * inputs;
  state.counters["per_input"] = benchmark::Counter(
    static_cast<double>(conversions), benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}

/**
 * @brief Sequentially convert N source-GPU partitions to the destination GPU on one thread.
 * @param state.range(0) Rows per input (fixed schema and partition count)
 */
void BM_ConvertGpuToGpu(benchmark::State& state)
{
  if (!has_two_devices()) {
    state.SkipWithError("requires two CUDA devices");
    return;
  }
  if (!grant_peer_pools()) {
    state.SkipWithError("peer pool access not granted on this hardware");
    return;
  }

  auto const rows = static_cast<cudf::size_type>(state.range(0));

  rmm::cuda_stream producer = [] {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{src_device_id}};
    return rmm::cuda_stream{};
  }();
  auto const inputs = make_inputs(num_inputs, rows, producer);
  producer.synchronize();

  const memory_space* dst_space = gpu_space(dst_device_id);
  rmm::cuda_set_device_raii const dst_guard{rmm::cuda_device_id{dst_device_id}};
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
  outputs.reserve(num_inputs);
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

  set_counters(state, bytes_per_iteration, num_inputs);
  state.counters["rows"] = rows;
}

/**
 * @brief Lower bound: peer-copy the same buffers, all queued on one stream, one host sync.
 * @param state.range(0) Rows per input, sized using the converter's actual table allocation
 */
void BM_RawPeerCopy(benchmark::State& state)
{
  if (!has_two_devices()) {
    state.SkipWithError("requires two CUDA devices");
    return;
  }
  if (!grant_peer_pools()) {
    state.SkipWithError("peer pool access not granted on this hardware");
    return;
  }

  auto const rows = static_cast<cudf::size_type>(state.range(0));
  // Use the same schema's measured size, including masks and offsets, not a byte estimate.
  auto const bytes_per_input = [&] {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{src_device_id}};
    rmm::cuda_stream producer;
    auto const inputs = make_inputs(1, rows, producer);
    producer.synchronize();
    return inputs.front()->get_size_in_bytes();
  }();
  auto const total_bytes = bytes_per_input * num_inputs;

  rmm::device_buffer src = [&] {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{src_device_id}};
    rmm::device_buffer buf(total_bytes,
                           ::cuda::stream_ref{cudaStream_t{cudaStreamDefault}},
                           gpu_space(src_device_id)->get_default_allocator());
    CUCASCADE_CUDA_TRY(cudaMemset(buf.data(), 0x22, total_bytes));
    return buf;
  }();

  rmm::cuda_set_device_raii const dst_guard{rmm::cuda_device_id{dst_device_id}};
  rmm::cuda_stream consumer;
  rmm::device_buffer dst(total_bytes, consumer, gpu_space(dst_device_id)->get_default_allocator());
  consumer.synchronize();

  for (auto _ : state) {
    for (std::size_t i = 0; i < num_inputs; ++i) {
      auto const offset = i * bytes_per_input;
      CUCASCADE_CUDA_TRY(cudaMemcpyPeerAsync(static_cast<std::byte*>(dst.data()) + offset,
                                             dst_device_id,
                                             static_cast<const std::byte*>(src.data()) + offset,
                                             src_device_id,
                                             bytes_per_input,
                                             consumer.value()));
    }
    consumer.synchronize();
  }

  set_counters(state, total_bytes, num_inputs);
  state.counters["rows"] = rows;
}

void gpu_to_gpu_args(benchmark::internal::Benchmark* b)
{
  for (int64_t rows : {1024, 4096, 16384, 65536, 262144, 1048576}) {
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
