/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cudf_test_utils.hpp"

#include <cucascade/cudf/builtin_converters.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <catch2/catch_all.hpp>

#include <chrono>
#include <future>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace cucascade;
using namespace cucascade::memory;

constexpr std::size_t capacity = 64ull << 20;
enum class null_pattern { alternating, all_null };

void require_two_gpus()
{
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
    SKIP("requires two CUDA devices");
  }
}

std::shared_ptr<memory_space> make_space(gpu_memory_space_config config)
{
  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{config.device_id}};
  // Destroy each space's streams on their owning device.
  return {new memory_space{config}, [device = config.device_id](memory_space* space) {
            rmm::cuda_set_device_raii owner{rmm::cuda_device_id{device}};
            delete space;
          }};
}

void grant_pools(memory_space const& source, memory_space const& destination)
{
  for (auto const* space : {&source, &destination}) {
    auto const pool = space->get_memory_resource_of<Tier::GPU>()->pool_handle();
    auto const peer = space == &source ? destination.get_device_id() : source.get_device_id();
    REQUIRE(pool != nullptr);
    if (!grant_pool_peer_access(pool, space->get_device_id(), peer).granted()) {
      SKIP("pool peer access unavailable");
    }
  }
}

auto make_mask(cudf::size_type rows,
               null_pattern pattern,
               ::cuda::stream_ref stream,
               rmm::device_async_resource_ref mr)
{
  auto mask = cudf::create_null_mask(rows, cudf::mask_state::ALL_NULL, stream, mr);
  if (mask.size() > 0 && pattern == null_pattern::alternating) {
    // Even rows valid, odd rows null; 65 rows also exercises a partial final mask word.
    CUCASCADE_CUDA_TRY(cudaMemsetAsync(
      mask.data(), 0x55, cudf::bitmask_allocation_size_bytes(rows), stream.get()));
  }
  return mask;
}

// Keep test masks compatible with the installed cuDF null-mask type.
static_assert(std::is_same_v<decltype(make_mask(0,
                                              null_pattern::all_null,
                                              std::declval<::cuda::stream_ref>(),
                                              std::declval<rmm::device_async_resource_ref>())),
                             decltype(cudf::create_null_mask(0, cudf::mask_state::ALL_NULL))>);

cudf::size_type null_count(cudf::size_type rows, null_pattern pattern)
{
  return pattern == null_pattern::all_null ? rows : rows / 2;
}

template <typename T>
std::unique_ptr<cudf::column> from_host(std::vector<T> const& values,
                                        ::cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<T>()},
                                          static_cast<cudf::size_type>(values.size()),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
  if (!values.empty()) {
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(column->mutable_view().head(),
                                       values.data(),
                                       values.size() * sizeof(T),
                                       cudaMemcpyHostToDevice,
                                       stream.get()));
  }
  // Host vectors are local setup data, not pinned or retained through conversion.
  stream.sync();
  return column;
}

template <typename Offset>
std::unique_ptr<cudf::column> strings(cudf::size_type rows,
                                      null_pattern pattern,
                                      ::cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  std::vector<Offset> offsets(static_cast<std::size_t>(rows) + 1, 0);
  std::vector<char> chars;
  for (cudf::size_type row = 0; row < rows; ++row) {
    if (pattern == null_pattern::alternating && row % 2 == 0) {
      // Vary lengths and characters so offsets or stale data cannot accidentally match.
      chars.insert(
        chars.end(), static_cast<std::size_t>(row % 3 + 1), static_cast<char>('a' + row % 26));
    }
    offsets[static_cast<std::size_t>(row) + 1] = static_cast<Offset>(chars.size());
  }
  auto offsets_column = from_host(offsets, stream, mr);
  rmm::device_buffer data(chars.data(), chars.size(), stream, mr);
  stream.sync();
  return cudf::make_strings_column(rows,
                                   std::move(offsets_column),
                                   std::move(data),
                                   null_count(rows, pattern),
                                   make_mask(rows, pattern, stream, mr));
}

std::unique_ptr<cudf::column> numbers(cudf::size_type rows,
                                      null_pattern pattern,
                                      ::cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  std::vector<int64_t> values(static_cast<std::size_t>(rows));
  for (cudf::size_type row = 0; row < rows; ++row) {
    values[static_cast<std::size_t>(row)] = 1009 * row + 17;
  }
  auto column = from_host(values, stream, mr);
  column->set_null_mask(make_mask(rows, pattern, stream, mr), null_count(rows, pattern));
  return column;
}

std::unique_ptr<cudf::table> make_table(cudf::size_type rows,
                                        null_pattern pattern,
                                        ::cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(numbers(rows, pattern, stream, mr));
  columns.push_back(strings<int32_t>(rows, pattern, stream, mr));
  columns.push_back(strings<int64_t>(rows, pattern, stream, mr));

  std::vector<int32_t> offsets(static_cast<std::size_t>(rows) + 1, 0);
  for (cudf::size_type row = 0; row < rows; ++row) {
    offsets[static_cast<std::size_t>(row) + 1] =
      offsets[static_cast<std::size_t>(row)] +
      (pattern == null_pattern::alternating && row % 2 == 0 ? 2 : 0);
  }
  columns.push_back(cudf::make_lists_column(rows,
                                            from_host(offsets, stream, mr),
                                            strings<int32_t>(offsets.back(), pattern, stream, mr),
                                            null_count(rows, pattern),
                                            make_mask(rows, pattern, stream, mr)));

  std::vector<std::unique_ptr<cudf::column>> fields;
  fields.push_back(numbers(rows, pattern, stream, mr));
  fields.push_back(strings<int32_t>(rows, pattern, stream, mr));
  columns.push_back(std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRUCT},
                                                   rows,
                                                   rmm::device_buffer{},
                                                   make_mask(rows, pattern, stream, mr),
                                                   null_count(rows, pattern),
                                                   std::move(fields)));

  std::vector<int32_t> indices(static_cast<std::size_t>(rows));
  for (cudf::size_type row = 0; row < rows; ++row) {
    indices[static_cast<std::size_t>(row)] = row % 3;
  }
  columns.push_back(
    cudf::make_dictionary_column(from_host(std::vector<int32_t>{3, 11, 29}, stream, mr),
                                 from_host(indices, stream, mr),
                                 make_mask(rows, pattern, stream, mr),
                                 null_count(rows, pattern)));
  stream.sync();
  return std::make_unique<cudf::table>(std::move(columns));
}

void fill_masks(cudf::mutable_column_view column, int value, ::cuda::stream_ref stream)
{
  if (column.nullable() && column.size() > 0) {
    CUCASCADE_CUDA_TRY(cudaMemsetAsync(
      column.null_mask(), value, cudf::bitmask_allocation_size_bytes(column.size()), stream.get()));
  }
  for (cudf::size_type i = 0; i < column.num_children(); ++i) {
    fill_masks(column.child(i), value, stream);
  }
}

void CUDART_CB delay_producer(void*)
{
  // No CUDA calls inside callbacks. The byte checks, not elapsed time, are the assertion.
  std::this_thread::sleep_for(std::chrono::milliseconds{100});
}

TEST_CASE("cross-GPU conversion preserves nullable nested contents after source release",
          "[gpu-converter][.multi-device][cross-gpu-nullable]")
{
  require_two_gpus();
  auto const source_device = GENERATE(0, 1);
  auto const rows          = GENERATE(cudf::size_type{0}, cudf::size_type{65});
  auto const pattern       = GENERATE(null_pattern::alternating, null_pattern::all_null);
  auto const known_writer  = GENERATE(false, true);
  CAPTURE(source_device, rows, pattern, known_writer);
  auto const destination_device = 1 - source_device;
  auto source_space = make_space({.device_id = source_device, .memory_capacity = capacity});
  auto destination_space =
    make_space({.device_id = destination_device, .memory_capacity = capacity});
  grant_pools(*source_space, *destination_space);
  representation_converter_registry registry;
  register_builtin_converters(registry);

  rmm::cuda_set_device_raii source_guard{rmm::cuda_device_id{source_device}};
  rmm::cuda_stream producer{rmm::cuda_stream::flags::non_blocking};
  auto table = make_table(rows, pattern, producer, source_space->get_default_allocator());
  auto source =
    std::make_unique<gpu_table_representation>(std::move(table), *source_space, producer);

  rmm::cuda_set_device_raii destination_guard{rmm::cuda_device_id{destination_device}};
  rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
  auto expected = make_table(rows, pattern, consumer, destination_space->get_default_allocator());
  // Warm up route probing before delaying the producer; probing must not hide a missing wait.
  auto warm =
    registry.convert<gpu_table_representation>(*source, destination_space.get(), consumer);
  warm.reset();

  {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{source_device}};
    auto owned = source->release_table(producer);
    if (rows > 0 && pattern == null_pattern::alternating) {
      for (auto column : owned->mutable_view()) {
        fill_masks(column, 0, producer);
      }
      producer.synchronize();
      CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(producer.value(), delay_producer, nullptr));
      for (auto column : owned->mutable_view()) {
        fill_masks(column, 0x55, producer);
      }
    }
    source = std::make_unique<gpu_table_representation>(
      std::move(owned),
      *source_space,
      known_writer ? ::cuda::stream_ref{producer} : ::cuda::stream_ref{cudaStream_t{nullptr}});
  }

  // Two independent consumers of the same source. Neither gets to repair readiness by syncing
  // its own stream; each conversion must return complete on its internally acquired stream.
  auto convert = [&] {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{destination_device}};
    rmm::cuda_stream reader{rmm::cuda_stream::flags::non_blocking};
    return registry.convert<gpu_table_representation>(*source, destination_space.get(), reader);
  };
  auto first  = std::async(std::launch::async, convert);
  auto second = std::async(std::launch::async, convert);
  auto left   = first.get();
  auto right  = second.get();
  source.reset();
  // Reallocate the source layout immediately, making premature release/reuse observable.
  {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{source_device}};
    auto reused =
      make_table(rows, null_pattern::all_null, producer, source_space->get_default_allocator());
  }
  cucascade::test::expect_cudf_tables_equal_on_stream(
    expected->view(), left->get_table_view(), consumer);
  cucascade::test::expect_cudf_tables_equal_on_stream(
    expected->view(), right->get_table_view(), consumer);
}

// A destination-local resource lets us fail the second allocation without changing production
// interfaces. The event is queued after the first column's peer copy and queried before cleanup.
struct failure_state {
  std::unique_ptr<rmm::mr::cuda_async_memory_resource> upstream;
  int allocations{0};
  bool armed{false};
  cudaEvent_t submitted{};
};

class failing_resource {
 public:
  explicit failing_resource(std::shared_ptr<failure_state> state) : _state(std::move(state)) {}

  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t))
  {
    if (_state->armed && ++_state->allocations == 2) {
      CUCASCADE_CUDA_TRY(cudaEventRecord(_state->submitted, stream.get()));
      throw rmm::out_of_memory{"injected second-column allocation failure"};
    }
    auto* ptr =
      rmm::device_async_resource_ref{*_state->upstream}.allocate(stream, bytes, alignment);
    if (_state->armed && _state->allocations == 1) {
      CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(stream.get(), delay_producer, nullptr));
    }
    return ptr;
  }
  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    rmm::device_async_resource_ref{*_state->upstream}.deallocate(stream, ptr, bytes, alignment);
  }
  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    return allocate(::cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }
  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    deallocate(::cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  }
  bool operator==(failing_resource const& other) const noexcept { return _state == other._state; }
  friend void get_property(failing_resource const&, cuda::mr::device_accessible) noexcept {}

 private:
  std::shared_ptr<failure_state> _state;
};

TEST_CASE("cross-GPU conversion drains submitted copies before propagating allocation failure",
          "[gpu-converter][.multi-device][cross-gpu-nullable]")
{
  require_two_gpus();
  rmm::cuda_set_device_raii destination_guard{rmm::cuda_device_id{1}};
  auto state      = std::make_shared<failure_state>();
  state->upstream = std::make_unique<rmm::mr::cuda_async_memory_resource>(capacity);
  CUCASCADE_CUDA_TRY(cudaEventCreateWithFlags(&state->submitted, cudaEventDisableTiming));
  // Create the event before the spaces so it outlives all resource users.
  auto event_cleanup = std::shared_ptr<void>{
    nullptr, [state](void*) { CUCASCADE_ASSERT_CUDA_SUCCESS(cudaEventDestroy(state->submitted)); }};
  auto source_space      = make_space({.device_id = 0, .memory_capacity = capacity});
  auto destination_space = make_space(
    {.device_id = 1, .memory_capacity = capacity, .mr_factory_fn = [state](int, std::size_t) {
       return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{failing_resource{state}};
     }});
  auto const source_pool = source_space->get_memory_resource_of<Tier::GPU>()->pool_handle();
  if (!grant_pool_peer_access(source_pool, 0, 1).granted() ||
      !grant_pool_peer_access(state->upstream->pool_handle(), 1, 0).granted() ||
      !probe_peer_dma_works(0, 1)) {
    SKIP("requires peer DMA to leave an asynchronous copy pending");
  }
  representation_converter_registry registry;
  register_builtin_converters(registry);
  auto producer = [] {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
    return std::shared_ptr<rmm::cuda_stream>{
      new rmm::cuda_stream{rmm::cuda_stream::flags::non_blocking}, [](rmm::cuda_stream* stream) {
        rmm::cuda_set_device_raii owner{rmm::cuda_device_id{0}};
        delete stream;
      }};
  }();
  std::unique_ptr<gpu_table_representation> source;
  {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
    std::vector<std::unique_ptr<cudf::column>> columns;
    for (int i = 0; i < 2; ++i) {
      columns.push_back(from_host(
        std::vector<int64_t>(65, i + 17), *producer, source_space->get_default_allocator()));
    }
    source = std::make_unique<gpu_table_representation>(
      std::make_unique<cudf::table>(std::move(columns)), *source_space, *producer);
  }
  rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
  auto warm =
    registry.convert<gpu_table_representation>(*source, destination_space.get(), consumer);
  warm.reset();
  CUCASCADE_CUDA_TRY(cudaDeviceSynchronize());
  state->armed = true;
  CHECK_THROWS_AS(
    registry.convert<gpu_table_representation>(*source, destination_space.get(), consumer),
    rmm::out_of_memory);
  auto const status_at_throw = cudaEventQuery(state->submitted);
  // Always drain before assertions/source teardown, even when the regression is present.
  CUCASCADE_CUDA_TRY(cudaDeviceSynchronize());
  CHECK(state->allocations == 2);
  CHECK(status_at_throw == cudaSuccess);
}

}  // namespace
