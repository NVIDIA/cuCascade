/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucascade/cudf/builtin_converters.hpp"
#include "cucascade/cudf/gpu_data_representation.hpp"
#include "cucascade/data/representation_converter.hpp"
#include "cucascade/error.hpp"
#include "cucascade/memory/config.hpp"
#include "cucascade/memory/error.hpp"
#include "cucascade/memory/fixed_size_host_memory_resource.hpp"
#include "cucascade/memory/memory_reservation_manager.hpp"
#include "cucascade/memory/memory_space.hpp"
#include "cucascade/memory/reservation_manager_configurator.hpp"
#include "cucascade/utils/cleanup_utils.hpp"
#include "utils/allocation_counter.hpp"
#include "utils/cuda_test_utils.hpp"
#include "utils/cudf_test_utils.hpp"
#include "utils/mixed_table.hpp"
#include "utils/peer_access_utils.hpp"
#include "utils/worker_group.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace cucascade::memory;
using cucascade::copy_route_selection;
using cucascade::cross_gpu_copy_config;
using cucascade::gpu_table_representation;

constexpr uint64_t kKiB = 1024ULL;
constexpr uint64_t kMiB = 1024ULL * kKiB;

constexpr int kSourceDevice = 0;
constexpr int kTargetDevice = 1;

/// Deadline of every test that converts on several threads at once.
constexpr std::chrono::seconds kConcurrentTimeout{60};

// SKIP when the hardware cannot give the peer access asked for, FAIL on CUDA errors.
void require_granted(pool_peer_access_result const& result)
{
  switch (result.status()) {
    case pool_peer_access_status::GRANTED: break;
    case pool_peer_access_status::UNSUPPORTED:
      SKIP("device pair does not support peer access");
      break;
    case pool_peer_access_status::VERIFICATION_FAILED:
      SKIP("peer DMA byte verification failed; direct-peer path unavailable");
      break;
    case pool_peer_access_status::CUDA_ERROR:
      FAIL("pool peer-access request failed: " << cudaGetErrorName(result.error()) << ": "
                                               << cudaGetErrorString(result.error()));
      break;
  }
}

// Grant two-way pool peer access; SKIP when the hardware cannot, FAIL on CUDA errors.
void require_peer_pools(memory_space const* source, memory_space const* destination)
{
  REQUIRE(source != nullptr);
  REQUIRE(destination != nullptr);

  CAPTURE(source->get_device_id(), destination->get_device_id());
  require_granted(cucascade::test::grant_two_way_pool_access(*source, *destination));
}

cross_gpu_copy_config host_staged(std::size_t chunk_size_bytes, std::size_t pipeline_depth)
{
  return {.chunk_size_bytes = chunk_size_bytes,
          .pipeline_depth   = pipeline_depth,
          .route            = copy_route_selection::host_staged};
}

/**
 * @brief Two GPUs plus small NUMA-local HOST spaces, built from 1 MiB pinned blocks.
 */
std::unique_ptr<memory_reservation_manager> make_two_gpu_manager(
  std::size_t host_capacity_per_region = 64 * kMiB)
{
  reservation_manager_configurator builder;
  builder.set_gpu_ids({kSourceDevice, kTargetDevice})
    .set_gpu_usage_limit(256 * kMiB)
    .use_numa_id_as_host_id()
    .set_per_numa_region_capacity(host_capacity_per_region)
    .set_host_pool_features(1 * kMiB, 4, 1)
    .set_host_portability(true);
  return std::make_unique<memory_reservation_manager>(builder.build());
}

/**
 * @brief Sum @p bytes_of over the HOST memory resource of every HOST space of @p manager.
 */
template <typename BytesOf>
std::size_t sum_over_host_spaces(memory_reservation_manager const& manager, BytesOf bytes_of)
{
  std::size_t total = 0;
  for (memory_space const* space : manager.get_memory_spaces_for_tier(Tier::HOST)) {
    total += bytes_of(*space->get_memory_resource_of<Tier::HOST>());
  }
  return total;
}

/**
 * @brief Peak HOST bytes ever reserved or allocated, summed over every HOST space of @p manager.
 */
std::size_t host_peak_bytes(memory_reservation_manager const& manager)
{
  return sum_over_host_spaces(manager, [](fixed_size_host_memory_resource const& mr) {
    return mr.get_peak_total_allocated_bytes();
  });
}

/**
 * @brief HOST bytes currently allocated (including reservations), summed over every HOST space.
 */
std::size_t host_allocated_bytes(memory_reservation_manager const& manager)
{
  return sum_over_host_spaces(manager, [](fixed_size_host_memory_resource const& mr) {
    return mr.get_total_allocated_bytes();
  });
}

/**
 * @brief Pinned upstream for a HOST memory_space that serves a fixed number of allocations and then
 * throws rmm::bad_alloc, as an exhausted host would.
 */
class exhaustible_pinned_resource {
 public:
  explicit exhaustible_pinned_resource(std::shared_ptr<std::atomic<int>> remaining)
    : _remaining{std::move(remaining)}
  {
  }

  void* allocate(::cuda::stream_ref, std::size_t bytes, std::size_t = alignof(std::max_align_t))
  {
    if (_remaining->fetch_sub(1) <= 0) {
      throw rmm::bad_alloc("exhaustible_pinned_resource: exhausted");
    }
    void* p = nullptr;
    CUCASCADE_CUDA_TRY_ALLOC(cudaHostAlloc(&p, bytes, cudaHostAllocPortable), bytes);
    return p;
  }

  void deallocate(::cuda::stream_ref,
                  void* p,
                  std::size_t,
                  std::size_t = alignof(std::max_align_t)) noexcept
  {
    std::ignore = cudaFreeHost(p);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    return allocate(::cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* p,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    deallocate(::cuda::stream_ref{cudaStream_t{nullptr}}, p, bytes, alignment);
  }

  bool operator==(exhaustible_pinned_resource const& other) const noexcept
  {
    return _remaining == other._remaining;
  }

  friend void get_property(exhaustible_pinned_resource const&,
                           ::cuda::mr::device_accessible) noexcept
  {
  }
  friend void get_property(exhaustible_pinned_resource const&, ::cuda::mr::host_accessible) noexcept
  {
  }

 private:
  std::shared_ptr<std::atomic<int>> _remaining;
};

/**
 * @brief Two GPUs plus one HOST space whose pool holds four 64 KiB blocks and cannot grow: its
 * upstream serves only the initial pool.
 */
std::unique_ptr<memory_reservation_manager> make_two_gpu_manager_with_unexpandable_host()
{
  std::vector<memory_space_config> configs;
  for (int device : {kSourceDevice, kTargetDevice}) {
    configs.emplace_back(
      gpu_memory_space_config{.device_id = device, .memory_capacity = 256 * kMiB});
  }
  host_memory_space_config host;
  host.numa_id                    = 0;
  host.memory_capacity            = 64 * kMiB;
  host.reservation_limit_fraction = 1.0;
  host.block_size                 = 64 * kKiB;
  host.pool_size                  = 4;
  host.initial_number_pools       = 1;
  host.mr_factory_fn = [remaining = std::make_shared<std::atomic<int>>(1)](int, std::size_t) {
    return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
      exhaustible_pinned_resource{remaining}};
  };
  configs.emplace_back(host);
  return std::make_unique<memory_reservation_manager>(std::move(configs));
}

/**
 * @brief Pinned upstream for a HOST memory_space whose allocations are adjacent in memory, each
 * pinned as a region of its own, the way separate growths of a HOST pool can land next to each
 * other.
 *
 * Allocations are served from the top of one arena downwards, so each ends where the previous one
 * begins. Not thread-safe: the pool under test grows from one thread.
 */
class adjacent_regions_pinned_resource {
 public:
  /**
   * @brief Page-aligned host memory that the regions are cut from.
   */
  struct arena {
    explicit arena(std::size_t bytes)
      : memory{static_cast<std::byte*>(std::aligned_alloc(4096, bytes)), &std::free},
        remaining{bytes}
    {
      if (memory == nullptr) { throw std::bad_alloc{}; }
    }

    std::unique_ptr<std::byte, void (*)(void*)> memory;
    std::size_t remaining;  ///< Bytes below the lowest region handed out so far
  };

  explicit adjacent_regions_pinned_resource(std::shared_ptr<arena> arena) : _arena{std::move(arena)}
  {
  }

  void* allocate(::cuda::stream_ref, std::size_t bytes, std::size_t = alignof(std::max_align_t))
  {
    if (bytes > _arena->remaining) {
      throw rmm::bad_alloc("adjacent_regions_pinned_resource: arena exhausted");
    }
    void* const region = _arena->memory.get() + (_arena->remaining - bytes);
    CUCASCADE_CUDA_TRY_ALLOC(cudaHostRegister(region, bytes, cudaHostRegisterPortable), bytes);
    _arena->remaining -= bytes;
    return region;
  }

  void deallocate(::cuda::stream_ref,
                  void* p,
                  std::size_t,
                  std::size_t = alignof(std::max_align_t)) noexcept
  {
    std::ignore = cudaHostUnregister(p);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    return allocate(::cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* p,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    deallocate(::cuda::stream_ref{cudaStream_t{nullptr}}, p, bytes, alignment);
  }

  bool operator==(adjacent_regions_pinned_resource const& other) const noexcept
  {
    return _arena == other._arena;
  }

  friend void get_property(adjacent_regions_pinned_resource const&,
                           ::cuda::mr::device_accessible) noexcept
  {
  }
  friend void get_property(adjacent_regions_pinned_resource const&,
                           ::cuda::mr::host_accessible) noexcept
  {
  }

 private:
  std::shared_ptr<arena> _arena;
};

/**
 * @brief Two GPUs plus one HOST space whose pool grows by adjacent, separately pinned 4 MiB regions
 * of 1 MiB blocks.
 */
std::unique_ptr<memory_reservation_manager> make_two_gpu_manager_with_adjacent_host_regions()
{
  std::vector<memory_space_config> configs;
  for (int device : {kSourceDevice, kTargetDevice}) {
    configs.emplace_back(
      gpu_memory_space_config{.device_id = device, .memory_capacity = 256 * kMiB});
  }
  host_memory_space_config host;
  host.numa_id                    = 0;
  host.memory_capacity            = 16 * kMiB;
  host.reservation_limit_fraction = 1.0;
  host.block_size                 = 1 * kMiB;
  host.pool_size                  = 4;
  host.initial_number_pools       = 1;
  host.mr_factory_fn = [arena = std::make_shared<adjacent_regions_pinned_resource::arena>(
                          16 * kMiB)](int, std::size_t) {
    return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
      adjacent_regions_pinned_resource{arena}};
  };
  configs.emplace_back(host);
  return std::make_unique<memory_reservation_manager>(std::move(configs));
}

/**
 * @brief GPU upstream for a memory_space that throws rmm::out_of_memory on one chosen allocation,
 * as a device that ran out of memory would.
 */
class exhaustible_device_resource {
 public:
  /**
   * @brief Allocations still served before one fails; a negative count fails none.
   */
  using countdown = std::atomic<long>;

  exhaustible_device_resource(std::size_t capacity, std::shared_ptr<countdown> remaining)
    : _upstream{std::make_shared<rmm::mr::cuda_async_memory_resource>(capacity)},
      _remaining{std::move(remaining)}
  {
  }

  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t))
  {
    if (_remaining->load() >= 0 && _remaining->fetch_sub(1) == 0) {
      throw rmm::out_of_memory("exhaustible_device_resource: exhausted");
    }
    return _upstream->allocate(stream, bytes, alignment);
  }

  void deallocate(::cuda::stream_ref stream,
                  void* p,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    _upstream->deallocate(stream, p, bytes, alignment);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    return allocate(::cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* p,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    deallocate(::cuda::stream_ref{cudaStream_t{nullptr}}, p, bytes, alignment);
  }

  bool operator==(exhaustible_device_resource const& other) const noexcept
  {
    return _upstream == other._upstream;
  }

  friend void get_property(exhaustible_device_resource const&,
                           ::cuda::mr::device_accessible) noexcept
  {
  }

 private:
  std::shared_ptr<rmm::mr::cuda_async_memory_resource> _upstream;
  std::shared_ptr<countdown> _remaining;
};

/**
 * @brief Two GPUs of 1 GiB plus NUMA-local HOST spaces; allocations on kTargetDevice fail as @p
 * remaining says.
 */
std::unique_ptr<memory_reservation_manager> make_two_gpu_manager_with_exhaustible_destination(
  std::shared_ptr<exhaustible_device_resource::countdown> remaining)
{
  reservation_manager_configurator builder;
  builder.set_gpu_ids({kSourceDevice, kTargetDevice})
    .set_gpu_usage_limit(1024 * kMiB)
    .use_numa_id_as_host_id()
    .set_per_numa_region_capacity(64 * kMiB)
    .set_host_pool_features(1 * kMiB, 4, 1)
    .set_host_portability(true)
    .set_gpu_memory_resource_factory(
      [remaining = std::move(remaining)](int device_id, std::size_t capacity) {
        rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device_id}};
        if (device_id == kTargetDevice) {
          return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
            exhaustible_device_resource{capacity, remaining}};
        }
        return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
          rmm::mr::cuda_async_memory_resource{capacity}};
      });
  return std::make_unique<memory_reservation_manager>(builder.build());
}

/**
 * @brief GPU upstream for a memory_space that counts the buffers freed while the stream they are
 * freed on still has work enqueued. It can also fail one chosen allocation, and reports every
 * allocation it serves.
 */
class recording_device_resource {
 public:
  /**
   * @brief What the resource has seen and what it is told to do, shared with the test.
   */
  struct record {
    std::atomic<long> served_before_failure{-1};  ///< Negative: no allocation fails
    std::atomic<std::size_t> allocations{0};      ///< Allocations served
    std::atomic<std::size_t> freed_while_stream_busy{0};
    cudaMemPool_t pool{};  ///< CUDA pool the buffers come from
    /// Called with `allocations` after each one is served; set only while no conversion runs.
    std::function<void(std::size_t)> after_allocation;
  };

  recording_device_resource(std::size_t capacity, std::shared_ptr<record> record)
    : _upstream{std::make_shared<rmm::mr::cuda_async_memory_resource>(capacity)},
      _record{std::move(record)}
  {
    _record->pool = _upstream->pool_handle();
  }

  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t))
  {
    if (_record->served_before_failure.load() >= 0 &&
        _record->served_before_failure.fetch_sub(1) == 0) {
      throw rmm::out_of_memory("recording_device_resource: exhausted");
    }
    void* const p            = _upstream->allocate(stream, bytes, alignment);
    std::size_t const served = _record->allocations.fetch_add(1) + 1;
    if (_record->after_allocation) { _record->after_allocation(served); }
    return p;
  }

  void deallocate(::cuda::stream_ref stream,
                  void* p,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    if (cudaStreamQuery(stream.get()) != cudaSuccess) {
      // cudaErrorNotReady is not an error: clear it so that a later CUDA check does not report it.
      std::ignore = cudaGetLastError();
      ++_record->freed_while_stream_busy;
    }
    _upstream->deallocate(stream, p, bytes, alignment);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    return allocate(::cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* p,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    deallocate(::cuda::stream_ref{cudaStream_t{nullptr}}, p, bytes, alignment);
  }

  bool operator==(recording_device_resource const& other) const noexcept
  {
    return _upstream == other._upstream;
  }

  friend void get_property(recording_device_resource const&, ::cuda::mr::device_accessible) noexcept
  {
  }

 private:
  std::shared_ptr<rmm::mr::cuda_async_memory_resource> _upstream;
  std::shared_ptr<record> _record;
};

/**
 * @brief Two GPUs of 1 GiB plus NUMA-local HOST spaces; the buffers of kTargetDevice come from a
 * recording_device_resource that shares @p record.
 */
std::unique_ptr<memory_reservation_manager> make_two_gpu_manager_with_recording_destination(
  std::shared_ptr<recording_device_resource::record> record)
{
  reservation_manager_configurator builder;
  builder.set_gpu_ids({kSourceDevice, kTargetDevice})
    .set_gpu_usage_limit(1024 * kMiB)
    .use_numa_id_as_host_id()
    .set_per_numa_region_capacity(64 * kMiB)
    .set_host_pool_features(1 * kMiB, 4, 1)
    .set_host_portability(true)
    .set_gpu_memory_resource_factory(
      [record = std::move(record)](int device_id, std::size_t capacity) {
        rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device_id}};
        if (device_id == kTargetDevice) {
          return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
            recording_device_resource{capacity, record}};
        }
        return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
          rmm::mr::cuda_async_memory_resource{capacity}};
      });
  return std::make_unique<memory_reservation_manager>(builder.build());
}

/**
 * @brief Every stream of @p space's stream pool, which hands its streams out in turn.
 */
std::vector<cudaStream_t> pool_streams(memory_space const& space)
{
  std::vector<cudaStream_t> streams{space.acquire_stream().get()};
  for (cudaStream_t next = space.acquire_stream().get(); next != streams.front();
       next              = space.acquire_stream().get()) {
    REQUIRE(streams.size() < 4096);
    streams.push_back(next);
  }
  return streams;
}

/**
 * @brief Whether any of @p streams still has work enqueued.
 */
bool any_stream_busy(std::span<cudaStream_t const> streams)
{
  bool busy = false;
  for (cudaStream_t stream : streams) {
    cudaError_t const status = cudaStreamQuery(stream);
    if (status == cudaErrorNotReady) {
      // Not an error: clear it so that a later CUDA check does not report it.
      std::ignore = cudaGetLastError();
      busy        = true;
    } else {
      REQUIRE(status == cudaSuccess);
    }
  }
  return busy;
}

enum class table_kind { mixed, zero_rows, no_columns, int64_sequence, int64_and_nullable_int8 };

/**
 * @brief Deterministic table of @p kind; the same call on another device yields equal contents.
 */
std::unique_ptr<cudf::table> make_table(table_kind kind,
                                        cudf::size_type rows,
                                        memory_space const& space,
                                        ::cuda::stream_ref stream)
{
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{space.get_device_id()}};
  auto const mr = space.get_default_allocator();
  std::unique_ptr<cudf::table> table;
  switch (kind) {
    case table_kind::mixed: table = cucascade::test::make_mixed_table(rows, stream, mr); break;
    case table_kind::zero_rows:
      // make_mixed_table cannot build zero rows; keep its schema and drop the rows.
      table = cudf::empty_like(cucascade::test::make_mixed_table(10, stream, mr)->view());
      break;
    case table_kind::no_columns:
      table = std::make_unique<cudf::table>(std::vector<std::unique_ptr<cudf::column>>{});
      break;
    case table_kind::int64_sequence: {
      cudf::numeric_scalar<int64_t> const init{7, true, stream};
      cudf::numeric_scalar<int64_t> const step{3, true, stream};
      std::vector<std::unique_ptr<cudf::column>> columns;
      columns.push_back(cudf::sequence(rows, init, step, stream, mr));
      table = std::make_unique<cudf::table>(std::move(columns));
      break;
    }
    case table_kind::int64_and_nullable_int8: {
      cudf::numeric_scalar<int64_t> const init{7, true, stream};
      cudf::numeric_scalar<int64_t> const step{3, true, stream};
      std::vector<std::unique_ptr<cudf::column>> columns;
      columns.push_back(cudf::sequence(rows, init, step, stream, mr));
      for (int8_t value : {int8_t{11}, int8_t{-5}}) {
        cudf::numeric_scalar<int8_t> const fill{value, true, stream};
        columns.push_back(cudf::make_column_from_scalar(fill, rows, stream, mr));
        columns.back()->set_null_mask(
          cudf::create_null_mask(rows, cudf::mask_state::ALL_VALID, stream, mr), 0);
      }
      table = std::make_unique<cudf::table>(std::move(columns));
      break;
    }
  }
  stream.sync();
  return table;
}

/**
 * @brief Source representation on @p device, with a writer event recorded on @p producer.
 */
std::unique_ptr<gpu_table_representation> make_source(memory_reservation_manager& manager,
                                                      table_kind kind,
                                                      cudf::size_type rows,
                                                      rmm::cuda_stream const& producer,
                                                      int device = kSourceDevice)
{
  memory_space* space = manager.get_memory_space(Tier::GPU, device);
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
  return std::make_unique<gpu_table_representation>(
    make_table(kind, rows, *space, producer), *space, producer);
}

/**
 * @brief Source representation on kSourceDevice that owns @p columns, with a writer event recorded
 * on @p producer.
 */
std::unique_ptr<gpu_table_representation> make_source(
  memory_reservation_manager& manager,
  std::vector<std::unique_ptr<cudf::column>> columns,
  rmm::cuda_stream const& producer)
{
  memory_space* space = manager.get_memory_space(Tier::GPU, kSourceDevice);
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kSourceDevice}};
  return std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(columns)), *space, producer);
}

/**
 * @brief Require that converting @p source to kTargetDevice throws std::invalid_argument before
 * anything is staged through HOST memory.
 */
void expect_source_rejected(cucascade::representation_converter_registry& registry,
                            memory_reservation_manager& manager,
                            gpu_table_representation const& source)
{
  memory_space* target_space = manager.get_memory_space(Tier::GPU, kTargetDevice);
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kTargetDevice}};
  rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};

  CHECK_THROWS_AS(registry.convert<gpu_table_representation>(source, target_space, consumer),
                  std::invalid_argument);
  CHECK(host_peak_bytes(manager) == 0);
}

rmm::cuda_stream make_stream_on(int device)
{
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
  return rmm::cuda_stream{rmm::cuda_stream::flags::non_blocking};
}

/**
 * @brief Convert @p source to kTargetDevice and require it to equal a freshly built table of @p
 * kind.
 */
void convert_and_expect_equal(cucascade::representation_converter_registry& registry,
                              memory_reservation_manager& manager,
                              gpu_table_representation const& source,
                              table_kind kind,
                              cudf::size_type rows)
{
  memory_space* target_space = manager.get_memory_space(Tier::GPU, kTargetDevice);
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kTargetDevice}};
  rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};

  auto result = registry.convert<gpu_table_representation>(source, target_space, consumer);
  REQUIRE(result->get_device_id() == kTargetDevice);
  result->wait_for_writer(consumer);

  auto const expected = make_table(kind, rows, *target_space, consumer);
  cucascade::test::expect_cudf_tables_equal_on_stream(
    expected->view(), result->get_table_view(), consumer);
}

/**
 * @brief Two converters read one source whose writer is parked behind a gate; both results must
 * show the writes enqueued before the writer event.
 */
void check_conversions_wait_for_source_writer(cross_gpu_copy_config const& config)
{
  if (cucascade::test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  auto const manager_owner            = make_two_gpu_manager();
  memory_reservation_manager& manager = *manager_owner;

  if (config.route != copy_route_selection::host_staged) {
    require_peer_pools(manager.get_memory_space(Tier::GPU, kSourceDevice),
                       manager.get_memory_space(Tier::GPU, kTargetDevice));
  }

  cucascade::representation_converter_registry registry;
  cucascade::register_builtin_converters(registry, config);

  rmm::cuda_set_device_raii source_guard{rmm::cuda_device_id{kSourceDevice}};
  rmm::cuda_stream source_producer_stream{rmm::cuda_stream::flags::non_blocking};

  constexpr cudf::size_type kRows = 1024;
  auto* source_mem_space          = manager.get_memory_space(Tier::GPU, kSourceDevice);
  // make_mixed_table waits internally, so finish it before closing the producer gate.
  auto source_table = cucascade::test::make_mixed_table(
    kRows, source_producer_stream, source_mem_space->get_default_allocator());
  source_producer_stream.synchronize();

  constexpr std::size_t kIdBytes = static_cast<std::size_t>(kRows) * sizeof(int64_t);
  int64_t* source_ids            = source_table->get_column(0).mutable_view().data<int64_t>();
  rmm::device_buffer saved_ids{
    source_ids, kIdBytes, source_producer_stream, source_mem_space->get_default_allocator()};
  // a copy that misses the producer dependency via the writer event would see incorrect IDs, not
  // ready-made data.
  CUCASCADE_CUDA_TRY(cudaMemsetAsync(source_ids, 0, kIdBytes, source_producer_stream.value()));
  source_producer_stream.synchronize();

  cucascade::test::stream_gate producer_gate{source_producer_stream.value()};
  producer_gate.arm();
  // halt the producer stream from making any progress beyond this point
  REQUIRE(producer_gate.wait_entered(std::chrono::seconds{5}));

  // restore the expected IDs only after the gate is armed. The writer event comes after this copy,
  // in the constructor below.
  CUCASCADE_CUDA_TRY(cudaMemcpyAsync(source_ids,
                                     saved_ids.data(),
                                     kIdBytes,
                                     cudaMemcpyDeviceToDevice,
                                     source_producer_stream.value()));
  auto source_data = std::make_unique<gpu_table_representation>(
    std::move(source_table), *source_mem_space, source_producer_stream);

  // the writer event is behind the gate; conversion must honor it before copying.

  // Declared before `cleanup`, so that a failure opens the producer gate before the converters are
  // waited for.
  cucascade::test::worker_group<std::unique_ptr<gpu_table_representation>> converters{
    2, kConcurrentTimeout};

  // to prevent deadlocks in case of conversion failure
  auto cleanup = cucascade::utils::on_scope_exit([&]() noexcept { producer_gate.release(); });

  // two independent converters reading from the same source
  for (int i = 0; i < 2; ++i) {
    converters.spawn([&](auto& start) {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{kTargetDevice}};
      rmm::cuda_stream reader{rmm::cuda_stream::flags::non_blocking};
      auto* target_mem_space = manager.get_memory_space(Tier::GPU, kTargetDevice);
      start.arrive_and_wait();
      return registry.convert<gpu_table_representation>(*source_data, target_mem_space, reader);
    });
  }
  converters.wait_until_started();

  // resume the halted producer after a 100ms grace period. Each converter blocks in its single
  // planning sync (the strings chars-size readback), which is ordered behind the gate. Timing
  // affects coverage only; the ID check below proves ordering either way.
  // TODO: make this fully deterministic, eg by using a custom allocator of the test that
  // notifies on first allocation signalling the copies have been enqueued.
  producer_gate.release_after(std::chrono::milliseconds{100});
  auto const converted = converters.join();
  auto const& left     = converted[0];
  auto const& right    = converted[1];

  // comparisons should not rely on source's existence after the copy
  source_data.reset();

  {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{kTargetDevice}};
    rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
    auto* target_mem_space = manager.get_memory_space(Tier::GPU, kTargetDevice);

    auto expected_table =
      cucascade::test::make_mixed_table(kRows, consumer, target_mem_space->get_default_allocator());

    cucascade::test::expect_cudf_tables_equal_on_stream(
      expected_table->view(), left->get_table_view(), consumer);
    cucascade::test::expect_cudf_tables_equal_on_stream(
      expected_table->view(), right->get_table_view(), consumer);
  }
}

}  // namespace

namespace cucascade {

TEST_CASE("gpu_to_gpu conversion copies content correctly across GPUs",
          "[gpu-to-gpu-converter][multi-device][cross-gpu-correctness]")
{
  check_conversions_wait_for_source_writer(cross_gpu_copy_config{});
}

TEST_CASE("host-staged gpu_to_gpu conversion waits for the source writer",
          "[gpu-to-gpu-converter][multi-device][host-staged]")
{
  check_conversions_wait_for_source_writer(host_staged(4093, 3));
}

TEST_CASE("host-staged gpu_to_gpu conversion copies content exactly for any chunking",
          "[gpu-to-gpu-converter][multi-device][host-staged]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  // Chunk sizes: far below a buffer, unaligned to every buffer size (chunks cross buffer
  // boundaries), and larger than the whole table. Depths: 1, and far more than the chunk count.
  auto const [chunk_size, depth] = GENERATE(table<std::size_t, std::size_t>({{509, 1},
                                                                             {509, 4},
                                                                             {4093, 2},
                                                                             {4 * kKiB, 3},
                                                                             {4093, 100000},
                                                                             {64 * kMiB, 1},
                                                                             {64 * kMiB, 8}}));
  // Row counts: one row, and counts whose last null mask word is partly used (33, 1027) or full
  // (1024).
  auto const [kind, rows] =
    GENERATE(table<table_kind, cudf::size_type>({{table_kind::mixed, 1},
                                                 {table_kind::mixed, 33},
                                                 {table_kind::mixed, 1024},
                                                 {table_kind::mixed, 1027},
                                                 {table_kind::zero_rows, 0},
                                                 {table_kind::no_columns, 0}}));
  CAPTURE(chunk_size, depth, static_cast<int>(kind), rows);

  auto manager = make_two_gpu_manager();
  representation_converter_registry registry;
  register_builtin_converters(registry, host_staged(chunk_size, depth));

  auto const producer = make_stream_on(kSourceDevice);
  auto const source   = make_source(*manager, kind, rows, producer);

  convert_and_expect_equal(registry, *manager, *source, kind, rows);
}

TEST_CASE(
  "host-staged gpu_to_gpu conversion copies large leaf buffers exactly across HOST pool "
  "blocks and regions",
  "[gpu-to-gpu-converter][multi-device][host-staged]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  // 32768 mixed rows are about 5 MiB, most of it in leaf buffers of 128 KiB and more, which the
  // pipeline starts on host page boundaries. make_two_gpu_manager() pins HOST memory as separate 4
  // MiB regions of 1 MiB blocks, so the staging slots here span blocks and regions. Chunk sizes:
  // barely above the smallest page-aligned piece, unaligned to pages and blocks, several blocks,
  // and the whole table in one chunk larger than a region.
  auto const [chunk_size, depth] = GENERATE(table<std::size_t, std::size_t>(
    {{40 * kKiB, 2}, {300 * kKiB + 13, 3}, {3 * kMiB, 4}, {64 * kMiB, 1}}));
  CAPTURE(chunk_size, depth);

  auto manager = make_two_gpu_manager();
  representation_converter_registry registry;
  register_builtin_converters(registry, host_staged(chunk_size, depth));

  constexpr cudf::size_type kRows = 32768;
  auto const producer             = make_stream_on(kSourceDevice);
  auto const source               = make_source(*manager, table_kind::mixed, kRows, producer);

  // The second conversion gets the blocks the first one returned, in the opposite order.
  convert_and_expect_equal(registry, *manager, *source, table_kind::mixed, kRows);
  convert_and_expect_equal(registry, *manager, *source, table_kind::mixed, kRows);
}

TEST_CASE("host-staged gpu_to_gpu conversion keeps every copy inside one pinned HOST region",
          "[gpu-to-gpu-converter][multi-device][host-staged]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  // CUDA rejects a copy that crosses from one pinned region into the next, even when the two are
  // adjacent in memory. One chunk of about 5 MiB needs blocks from two of the pool's 4 MiB regions.
  // Repeating the conversion goes through the orders in which the pool hands those blocks out,
  // including the one where the blocks of both regions ascend without a gap.
  auto manager = make_two_gpu_manager_with_adjacent_host_regions();
  representation_converter_registry registry;
  register_builtin_converters(registry, host_staged(64 * kMiB, 1));

  constexpr cudf::size_type kRows = 32768;
  auto const producer             = make_stream_on(kSourceDevice);
  auto const source               = make_source(*manager, table_kind::mixed, kRows, producer);

  for (int conversion = 0; conversion < 3; ++conversion) {
    CAPTURE(conversion);
    convert_and_expect_equal(registry, *manager, *source, table_kind::mixed, kRows);
  }
}

TEST_CASE("host-staged gpu_to_gpu conversion returns its HOST staging memory",
          "[gpu-to-gpu-converter][multi-device][host-staged]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  auto manager = make_two_gpu_manager();
  representation_converter_registry registry;
  register_builtin_converters(registry, host_staged(64 * kKiB, 4));

  auto const producer = make_stream_on(kSourceDevice);
  auto const source   = make_source(*manager, table_kind::mixed, 4096, producer);

  std::size_t const allocated_before = host_allocated_bytes(*manager);
  std::size_t const reserved_before  = manager->get_total_reserved_memory_for_tier(Tier::HOST);
  std::size_t const reservations_before =
    manager->get_active_reservation_count_for_tier(Tier::HOST);

  convert_and_expect_equal(registry, *manager, *source, table_kind::mixed, 4096);

  CHECK(host_peak_bytes(*manager) > allocated_before);
  CHECK(host_allocated_bytes(*manager) == allocated_before);
  CHECK(manager->get_total_reserved_memory_for_tier(Tier::HOST) == reserved_before);
  CHECK(manager->get_active_reservation_count_for_tier(Tier::HOST) == reservations_before);
}

TEST_CASE("automatic copy route picks the peer route when peer DMA works",
          "[gpu-to-gpu-converter][multi-device][host-staged]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  auto manager = make_two_gpu_manager();
  require_peer_pools(manager->get_memory_space(Tier::GPU, kSourceDevice),
                     manager->get_memory_space(Tier::GPU, kTargetDevice));
  if (!probe_peer_dma_works(kSourceDevice, kTargetDevice)) { SKIP("peer DMA does not work"); }

  auto const producer = make_stream_on(kSourceDevice);
  auto const source   = make_source(*manager, table_kind::mixed, 1024, producer);

  SECTION("automatic stages nothing through HOST memory")
  {
    representation_converter_registry registry;
    register_builtin_converters(registry, cross_gpu_copy_config{});
    convert_and_expect_equal(registry, *manager, *source, table_kind::mixed, 1024);
    CHECK(host_peak_bytes(*manager) == 0);
  }
  SECTION("forced peer stages nothing through HOST memory")
  {
    representation_converter_registry registry;
    register_builtin_converters(registry, {.route = copy_route_selection::peer});
    convert_and_expect_equal(registry, *manager, *source, table_kind::mixed, 1024);
    CHECK(host_peak_bytes(*manager) == 0);
  }
  SECTION("forced host-staged stages through HOST memory")
  {
    representation_converter_registry registry;
    register_builtin_converters(registry, {.route = copy_route_selection::host_staged});
    convert_and_expect_equal(registry, *manager, *source, table_kind::mixed, 1024);
    CHECK(host_peak_bytes(*manager) > 0);
  }
}

TEST_CASE("host-staged gpu_to_gpu conversion throws out-of-memory without room for one chunk",
          "[gpu-to-gpu-converter][multi-device][host-staged]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  // 4 MiB per region reserves at most 3 MiB (90%), less than one 4 MiB chunk.
  auto manager = make_two_gpu_manager(4 * kMiB);
  representation_converter_registry registry;
  register_builtin_converters(registry, host_staged(4 * kMiB, 2));

  constexpr cudf::size_type kRows = 1 << 20;  // 8 MiB of INT64
  auto const producer             = make_stream_on(kSourceDevice);
  auto const source = make_source(*manager, table_kind::int64_sequence, kRows, producer);

  std::size_t const allocated_before = host_allocated_bytes(*manager);
  memory_space* target_space         = manager->get_memory_space(Tier::GPU, kTargetDevice);
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kTargetDevice}};
  rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};

  CHECK_THROWS_AS(registry.convert<gpu_table_representation>(*source, target_space, consumer),
                  cucascade_out_of_memory);
  CHECK(host_allocated_bytes(*manager) == allocated_before);
  CHECK(manager->get_active_reservation_count_for_tier(Tier::HOST) == 0);
}

TEST_CASE("host-staged gpu_to_gpu conversion has completed every copy when it returns",
          "[gpu-to-gpu-converter][multi-device][host-staged]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  // 64 MiB of INT64 at the default 8 MiB chunks and depth 4: eight chunks, so every staging slot is
  // filled twice. Enqueueing the copies takes far less time than running them, so a conversion that
  // returned without waiting for them would return with nearly all of them outstanding.
  auto manager = make_two_gpu_manager();
  representation_converter_registry registry;
  register_builtin_converters(registry, {.route = copy_route_selection::host_staged});

  constexpr cudf::size_type kRows = 1 << 23;
  auto const producer             = make_stream_on(kSourceDevice);
  auto source = make_source(*manager, table_kind::int64_sequence, kRows, producer);

  memory_space* source_space = manager->get_memory_space(Tier::GPU, kSourceDevice);
  memory_space* target_space = manager->get_memory_space(Tier::GPU, kTargetDevice);
  auto const source_streams  = pool_streams(*source_space);
  auto const target_streams  = pool_streams(*target_space);

  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kTargetDevice}};
  rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
  // Built before the conversion, so that nothing runs between its return and the checks below.
  auto const expected = make_table(table_kind::int64_sequence, kRows, *target_space, consumer);

  auto result = registry.convert<gpu_table_representation>(*source, target_space, consumer);
  CHECK_FALSE(any_stream_busy(source_streams));
  CHECK_FALSE(any_stream_busy(target_streams));

  // The caller may free the source as soon as the conversion returns. Do so and overwrite the
  // memory it held, which a copy still reading the source would pick up.
  {
    rmm::cuda_set_device_raii const source_guard{rmm::cuda_device_id{kSourceDevice}};
    source.reset();
    rmm::device_buffer overwritten{static_cast<std::size_t>(kRows) * sizeof(int64_t),
                                   producer,
                                   source_space->get_default_allocator()};
    CUCASCADE_CUDA_TRY(
      cudaMemsetAsync(overwritten.data(), 0xA5, overwritten.size(), producer.value()));
    producer.synchronize();
  }

  REQUIRE(result->get_device_id() == kTargetDevice);
  result->wait_for_writer(consumer);
  test::expect_cudf_tables_equal_on_stream(expected->view(), result->get_table_view(), consumer);
}

TEST_CASE(
  "host-staged gpu_to_gpu conversion that fails with copies in flight leaves nothing running",
  "[gpu-to-gpu-converter][multi-device][host-staged]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  // About 80 MiB of mixed columns at the default 8 MiB chunks and depth 4: four chunks (32 MiB) are
  // being copied to HOST memory when the destination runs out of memory, on its first buffer or
  // with part of the table already allocated. The copies take far longer than the allocations
  // before the failure; if they ever finish first, the checks below still hold.
  auto const served_allocations = GENERATE(0L, 5L, 20L);
  CAPTURE(served_allocations);

  auto const remaining = std::make_shared<exhaustible_device_resource::countdown>(-1);
  auto manager         = make_two_gpu_manager_with_exhaustible_destination(remaining);
  representation_converter_registry registry;
  register_builtin_converters(registry, {.route = copy_route_selection::host_staged});

  constexpr cudf::size_type kRows = 1 << 19;
  auto const producer             = make_stream_on(kSourceDevice);
  auto const source               = make_source(*manager, table_kind::mixed, kRows, producer);

  memory_space* target_space = manager->get_memory_space(Tier::GPU, kTargetDevice);
  auto const source_streams  = pool_streams(*manager->get_memory_space(Tier::GPU, kSourceDevice));
  auto const target_streams  = pool_streams(*target_space);

  std::size_t const allocated_before = host_allocated_bytes(*manager);
  std::size_t const reserved_before  = manager->get_total_reserved_memory_for_tier(Tier::HOST);
  std::size_t const reservations_before =
    manager->get_active_reservation_count_for_tier(Tier::HOST);

  {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kTargetDevice}};
    rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
    *remaining = served_allocations;
    CHECK_THROWS_AS(registry.convert<gpu_table_representation>(*source, target_space, consumer),
                    cucascade_out_of_memory);
    // Staging memory is back in the HOST pool, so no copy of the failed conversion may still run.
    CHECK_FALSE(any_stream_busy(source_streams));
    CHECK_FALSE(any_stream_busy(target_streams));
    *remaining = -1;
  }
  CHECK(host_peak_bytes(*manager) > allocated_before);
  CHECK(host_allocated_bytes(*manager) == allocated_before);
  CHECK(manager->get_total_reserved_memory_for_tier(Tier::HOST) == reserved_before);
  CHECK(manager->get_active_reservation_count_for_tier(Tier::HOST) == reservations_before);

  // The staging blocks of the failed conversion are reused right away: by a small table, and by
  // the same source again.
  {
    auto const small = make_source(*manager, table_kind::mixed, 4096, producer);
    convert_and_expect_equal(registry, *manager, *small, table_kind::mixed, 4096);
  }
  convert_and_expect_equal(registry, *manager, *source, table_kind::mixed, kRows);
}

TEST_CASE("gpu_to_gpu conversion that fails frees no destination buffer its copies still write",
          "[gpu-to-gpu-converter][multi-device][host-staged]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  auto const record          = std::make_shared<recording_device_resource::record>();
  auto manager               = make_two_gpu_manager_with_recording_destination(record);
  memory_space* target_space = manager->get_memory_space(Tier::GPU, kTargetDevice);
  auto const producer        = make_stream_on(kSourceDevice);

  SECTION("peer route: the destination runs out of memory with part of the table allocated")
  {
    // The destination's pool sits behind the recording resource, so it is granted by its handle.
    require_granted(
      test::grant_pool_access(*manager->get_memory_space(Tier::GPU, kSourceDevice), *target_space));
    require_granted(grant_pool_peer_access(record->pool, kTargetDevice, kSourceDevice));
    if (!probe_peer_dma_works(kSourceDevice, kTargetDevice)) { SKIP("peer DMA does not work"); }
    representation_converter_registry registry;
    register_builtin_converters(registry, {.route = copy_route_selection::peer});

    // About 80 MiB of mixed columns. A copy into a buffer takes far longer than the allocations
    // that follow it, so when the destination runs out of memory, the copies into the columns
    // before the failing one are still running.
    constexpr cudf::size_type kRows = 1 << 19;
    auto const source               = make_source(*manager, table_kind::mixed, kRows, producer);
    auto const served_allocations   = GENERATE(5L, 20L);
    CAPTURE(served_allocations);

    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kTargetDevice}};
    rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
    record->served_before_failure = served_allocations;
    CHECK_THROWS_AS(registry.convert<gpu_table_representation>(*source, target_space, consumer),
                    cucascade_out_of_memory);
    record->served_before_failure = -1;
    CHECK(record->freed_while_stream_busy == 0);
  }

  SECTION("host-staged route: a host allocation fails after every destination buffer exists")
  {
    representation_converter_registry registry;
    register_builtin_converters(registry, {.route = copy_route_selection::host_staged});

    // 64 MiB of INT64 and two nullable INT8 columns at the default 8 MiB chunks and depth 4. The
    // row count is odd, so the chunks behind the INT64 column are cut into more pieces than the
    // chunks before them, and the pipeline allocates host memory to list those pieces while the
    // host-to-destination copies of earlier chunks are running.
    constexpr auto kKind            = table_kind::int64_and_nullable_int8;
    constexpr cudf::size_type kRows = (1 << 23) + 12345;
    auto const source               = make_source(*manager, kKind, kRows, producer);
    auto const source_streams = pool_streams(*manager->get_memory_space(Tier::GPU, kSourceDevice));
    auto const target_streams = pool_streams(*target_space);

    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kTargetDevice}};
    rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
    auto const expected = make_table(kKind, kRows, *target_space, consumer);

    // A conversion without a failure tells how many destination buffers there are.
    std::size_t const allocations_before = record->allocations;
    std::ignore = registry.convert<gpu_table_representation>(*source, target_space, consumer);
    std::size_t const destination_buffers = record->allocations - allocations_before;
    REQUIRE(destination_buffers > 0);
    REQUIRE(record->freed_while_stream_busy == 0);
    std::size_t const host_allocated_before = host_allocated_bytes(*manager);

    // Fail the first host allocation after the last destination buffer, then the second, and so
    // on, until a conversion makes fewer host allocations than that and succeeds.
    std::size_t failed_conversions = 0;
    for (std::size_t later = 1;; ++later) {
      CAPTURE(later);
      REQUIRE(later < 1000);
      std::unique_ptr<gpu_table_representation> result;
      std::exception_ptr failure;
      {
        test::scoped_allocation_counter allocations;
        std::size_t const first  = record->allocations;
        record->after_allocation = [&](std::size_t served) {
          if (served - first == destination_buffers) {
            allocations.fail_at(allocations.count() + later);
          }
        };
        try {
          result = registry.convert<gpu_table_representation>(*source, target_space, consumer);
        } catch (...) {
          failure = std::current_exception();
        }
        record->after_allocation = nullptr;
      }
      CHECK(record->freed_while_stream_busy == 0);
      CHECK_FALSE(any_stream_busy(source_streams));
      CHECK_FALSE(any_stream_busy(target_streams));
      CHECK(host_allocated_bytes(*manager) == host_allocated_before);
      if (!failure) {
        REQUIRE(result != nullptr);
        result->wait_for_writer(consumer);
        test::expect_cudf_tables_equal_on_stream(
          expected->view(), result->get_table_view(), consumer);
        break;
      }
      ++failed_conversions;
    }
    CHECK(failed_conversions > 0);
  }
}

TEST_CASE("host-staged gpu_to_gpu conversion uses fewer staging slots when HOST memory is tight",
          "[gpu-to-gpu-converter][multi-device][host-staged]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  // 4 MiB per region reserves at most 3 MiB (90%): one 2 MiB staging slot, not the four asked for.
  auto manager = make_two_gpu_manager(4 * kMiB);
  representation_converter_registry registry;
  register_builtin_converters(registry, host_staged(2 * kMiB, 4));

  constexpr cudf::size_type kRows = 1 << 20;  // 8 MiB of INT64: four chunks
  auto const producer             = make_stream_on(kSourceDevice);
  auto const source = make_source(*manager, table_kind::int64_sequence, kRows, producer);

  convert_and_expect_equal(registry, *manager, *source, table_kind::int64_sequence, kRows);
  CHECK(host_peak_bytes(*manager) <= 3 * kMiB);
}

TEST_CASE("host-staged gpu_to_gpu conversions run concurrently",
          "[gpu-to-gpu-converter][multi-device][host-staged]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  auto manager = make_two_gpu_manager();
  representation_converter_registry registry;
  register_builtin_converters(registry, host_staged(16 * kKiB, 3));

  constexpr std::size_t kThreads  = 6;
  constexpr cudf::size_type kRows = 4096;
  auto const producer             = make_stream_on(kSourceDevice);
  std::vector<std::unique_ptr<gpu_table_representation>> sources;
  for (std::size_t i = 0; i < kThreads; ++i) {
    sources.push_back(make_source(*manager, table_kind::mixed, kRows, producer));
  }

  memory_space* target_space = manager->get_memory_space(Tier::GPU, kTargetDevice);
  test::worker_group<std::unique_ptr<gpu_table_representation>> conversions{kThreads,
                                                                            kConcurrentTimeout};
  for (std::size_t i = 0; i < kThreads; ++i) {
    conversions.spawn([&, i](auto& start) {
      rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kTargetDevice}};
      rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
      start.arrive_and_wait();
      return registry.convert<gpu_table_representation>(*sources[i], target_space, stream);
    });
  }
  auto const results = conversions.join();

  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kTargetDevice}};
  rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
  auto const expected = make_table(table_kind::mixed, kRows, *target_space, consumer);
  for (auto const& converted : results) {
    converted->wait_for_writer(consumer);
    test::expect_cudf_tables_equal_on_stream(
      expected->view(), converted->get_table_view(), consumer);
  }
}

TEST_CASE("host-staged gpu_to_gpu conversions in both directions share the stream pools",
          "[gpu-to-gpu-converter][multi-device][host-staged]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  // HOST capacity per region: room for the staging slots of every conversion, or of three. Without
  // room a conversion throws out-of-memory instead of waiting.
  auto const host_capacity = GENERATE(64 * kMiB, 4 * kMiB);
  CAPTURE(host_capacity);

  auto manager = make_two_gpu_manager(host_capacity);
  representation_converter_registry registry;
  register_builtin_converters(registry, host_staged(64 * kKiB, 4));

  constexpr std::array<int, 2> kDevices{kSourceDevice, kTargetDevice};
  constexpr std::size_t kPerDirection = 20;
  constexpr int kRounds               = 3;
  constexpr cudf::size_type kRows     = 4096;  // about 640 KiB: ten chunks

  // Each GPU is the source of one direction and the destination of the other, with more
  // conversions than its stream pool has streams, so conversions share streams in both roles.
  std::array<rmm::cuda_stream, 2> streams{make_stream_on(kDevices[0]), make_stream_on(kDevices[1])};
  std::array<memory_space*, 2> spaces{};
  std::array<std::vector<std::unique_ptr<gpu_table_representation>>, 2> sources;
  std::array<std::unique_ptr<cudf::table>, 2> expected;
  for (std::size_t d = 0; d < 2; ++d) {
    spaces[d] = manager->get_memory_space(Tier::GPU, kDevices[d]);
    REQUIRE(kPerDirection > pool_streams(*spaces[d]).size());
    for (std::size_t i = 0; i < kPerDirection; ++i) {
      sources[d].push_back(
        make_source(*manager, table_kind::mixed, kRows, streams[d], kDevices[d]));
    }
    expected[d] = make_table(table_kind::mixed, kRows, *spaces[d], streams[d]);
  }
  std::size_t const allocated_before = host_allocated_bytes(*manager);

  std::size_t converted = 0;
  for (int round = 0; round < kRounds; ++round) {
    CAPTURE(round);
    // Null where the conversion ran out of HOST memory.
    using outcome = std::unique_ptr<gpu_table_representation>;
    test::worker_group<outcome> conversions{2 * kPerDirection, kConcurrentTimeout};
    for (std::size_t from = 0; from < 2; ++from) {
      std::size_t const to = 1 - from;
      for (std::size_t i = 0; i < kPerDirection; ++i) {
        conversions.spawn([&, from, to, i](auto& start) -> outcome {
          rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kDevices[to]}};
          rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
          start.arrive_and_wait();
          try {
            return registry.convert<gpu_table_representation>(
              *sources[from][i], spaces[to], stream);
          } catch (cucascade_out_of_memory const&) {
            return nullptr;
          }
        });
      }
    }

    // In spawn order: every conversion from kDevices[0], then every conversion from kDevices[1].
    std::vector<outcome> const outcomes = conversions.join();
    for (std::size_t from = 0; from < 2; ++from) {
      std::size_t const to = 1 - from;
      rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kDevices[to]}};
      for (std::size_t i = 0; i < kPerDirection; ++i) {
        outcome const& table = outcomes[from * kPerDirection + i];
        if (table == nullptr) { continue; }
        table->wait_for_writer(streams[to]);
        test::expect_cudf_tables_equal_on_stream(
          expected[to]->view(), table->get_table_view(), streams[to]);
        ++converted;
      }
    }
    CHECK(host_allocated_bytes(*manager) == allocated_before);
    CHECK(manager->get_active_reservation_count_for_tier(Tier::HOST) == 0);
  }

  CHECK(converted > 0);
  if (host_capacity == 64 * kMiB) { CHECK(converted == 2 * kPerDirection * kRounds); }
}

TEST_CASE(
  "host-staged gpu_to_gpu conversion falls back to fewer staging slots when the HOST pool cannot "
  "grow",
  "[gpu-to-gpu-converter][multi-device][host-staged]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  constexpr cudf::size_type kRows = 1 << 20;  // 8 MiB of INT64
  auto manager                    = make_two_gpu_manager_with_unexpandable_host();
  auto const producer             = make_stream_on(kSourceDevice);
  auto const source = make_source(*manager, table_kind::int64_sequence, kRows, producer);
  std::size_t const allocated_before = host_allocated_bytes(*manager);

  SECTION("fewer slots fit in the existing pool")
  {
    // Four 128 KiB slots need a pool expansion; two fit in the existing 256 KiB pool.
    representation_converter_registry registry;
    register_builtin_converters(registry, host_staged(128 * kKiB, 4));
    convert_and_expect_equal(registry, *manager, *source, table_kind::int64_sequence, kRows);
  }
  SECTION("not even one slot fits in the existing pool")
  {
    representation_converter_registry registry;
    register_builtin_converters(registry, host_staged(1 * kMiB, 2));
    memory_space* target_space = manager->get_memory_space(Tier::GPU, kTargetDevice);
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kTargetDevice}};
    rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};

    CHECK_THROWS_AS(registry.convert<gpu_table_representation>(*source, target_space, consumer),
                    cucascade_out_of_memory);
  }
  CHECK(host_allocated_bytes(*manager) == allocated_before);
  CHECK(manager->get_active_reservation_count_for_tier(Tier::HOST) == 0);
}

TEST_CASE("host-staged gpu_to_gpu conversion stages through the destination's reservation manager",
          "[gpu-to-gpu-converter][multi-device][host-staged]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  auto source_manager = make_two_gpu_manager();
  auto target_manager = make_two_gpu_manager();
  representation_converter_registry registry;
  register_builtin_converters(registry, host_staged(64 * kKiB, 4));

  auto const producer = make_stream_on(kSourceDevice);
  auto const source   = make_source(*source_manager, table_kind::mixed, 4096, producer);

  convert_and_expect_equal(registry, *target_manager, *source, table_kind::mixed, 4096);
  CHECK(host_peak_bytes(*target_manager) > 0);
  CHECK(host_peak_bytes(*source_manager) == 0);
}

TEST_CASE("host-staged gpu_to_gpu conversion throws logic_error without a HOST space to stage in",
          "[gpu-to-gpu-converter][multi-device][host-staged]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  auto source_manager = make_two_gpu_manager();
  representation_converter_registry registry;
  register_builtin_converters(registry, host_staged(64 * kKiB, 4));
  auto const producer = make_stream_on(kSourceDevice);
  auto const source   = make_source(*source_manager, table_kind::mixed, 4096, producer);

  gpu_memory_space_config const target_config{.device_id       = kTargetDevice,
                                              .memory_capacity = 256 * kMiB};
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kTargetDevice}};
  rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};

  SECTION("destination memory_space without a reservation manager")
  {
    memory_space const standalone{target_config};
    REQUIRE(standalone.get_reservation_manager() == nullptr);
    CHECK_THROWS_AS(registry.convert<gpu_table_representation>(*source, &standalone, consumer),
                    cucascade::logic_error);
  }
  SECTION("destination reservation manager without a HOST space")
  {
    std::vector<memory_space_config> configs;
    configs.emplace_back(target_config);
    memory_reservation_manager gpu_only{std::move(configs)};
    CHECK_THROWS_AS(registry.convert<gpu_table_representation>(
                      *source, gpu_only.get_memory_space(Tier::GPU, kTargetDevice), consumer),
                    cucascade::logic_error);
  }
}

TEST_CASE("gpu_to_gpu conversion rejects a sliced source table",
          "[gpu-to-gpu-converter][multi-device][host-staged]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  auto const route = GENERATE(copy_route_selection::host_staged, copy_route_selection::automatic);
  auto const kind  = GENERATE(table_kind::int64_sequence, table_kind::mixed);
  CAPTURE(static_cast<int>(route), static_cast<int>(kind));

  auto manager = make_two_gpu_manager();
  representation_converter_registry registry;
  register_builtin_converters(registry, {.route = route});

  // Rows [1000, 1500) of a table: every column view starts at row 1000 of its buffers.
  memory_space* source_space               = manager->get_memory_space(Tier::GPU, kSourceDevice);
  auto const producer                      = make_stream_on(kSourceDevice);
  std::shared_ptr<cudf::table> const owner = make_table(kind, 2000, *source_space, producer);
  cudf::table_view const sliced =
    cudf::slice(owner->view(), std::vector<cudf::size_type>{1000, 1500}).front();
  REQUIRE(sliced.column(0).offset() == 1000);

  std::unique_ptr<gpu_table_representation> source;
  {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kSourceDevice}};
    source = std::make_unique<gpu_table_representation>(
      sliced, owner, std::size_t{1 * kMiB}, *source_space, producer);
  }
  expect_source_rejected(registry, *manager, *source);
}

TEST_CASE("gpu_to_gpu conversion rejects a strings column that has rows but no offsets",
          "[gpu-to-gpu-converter][multi-device][host-staged]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  auto const route = GENERATE(copy_route_selection::host_staged, copy_route_selection::automatic);
  CAPTURE(static_cast<int>(route));

  auto manager = make_two_gpu_manager();
  representation_converter_registry registry;
  register_builtin_converters(registry, {.route = route});

  // cudf builds a STRING column without an offsets child only when it has no rows. This one is
  // assembled by hand: 1000 null rows and no children.
  constexpr cudf::size_type kRows = 1000;
  memory_space* source_space      = manager->get_memory_space(Tier::GPU, kSourceDevice);
  auto const producer             = make_stream_on(kSourceDevice);
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kSourceDevice}};
  auto const mr = source_space->get_default_allocator();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::STRING},
    kRows,
    rmm::device_buffer{},
    cudf::create_null_mask(kRows, cudf::mask_state::ALL_NULL, producer, mr),
    kRows));

  SECTION("as a top-level column")
  {
    auto const source = make_source(*manager, std::move(columns), producer);
    expect_source_rejected(registry, *manager, *source);
  }
  SECTION("as a STRUCT field")
  {
    std::vector<std::unique_ptr<cudf::column>> parent;
    parent.push_back(
      std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRUCT},
                                     kRows,
                                     rmm::device_buffer{},
                                     cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                     0,
                                     std::move(columns)));
    auto const source = make_source(*manager, std::move(parent), producer);
    expect_source_rejected(registry, *manager, *source);
  }
}

TEST_CASE("forced peer route throws when peer DMA fails verification",
          "[gpu-to-gpu-converter][multi-device]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }
  if (probe_peer_dma_works(kSourceDevice, kTargetDevice)) { SKIP("peer DMA works on this pair"); }

  auto manager = make_two_gpu_manager();
  representation_converter_registry registry;
  register_builtin_converters(registry, {.route = copy_route_selection::peer});
  auto const producer = make_stream_on(kSourceDevice);
  auto const source   = make_source(*manager, table_kind::mixed, 1024, producer);

  memory_space* target_space = manager->get_memory_space(Tier::GPU, kTargetDevice);
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kTargetDevice}};
  rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
  CHECK_THROWS_MATCHES(
    registry.convert<gpu_table_representation>(*source, target_space, consumer),
    cucascade::logic_error,
    Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring("failed verification")));
}

TEST_CASE("register_builtin_converters rejects an unusable cross_gpu_copy_config",
          "[gpu-to-gpu-converter]")
{
  representation_converter_registry registry;
  CHECK_THROWS_AS(register_builtin_converters(registry, {.chunk_size_bytes = 0}),
                  std::invalid_argument);
  CHECK_THROWS_AS(register_builtin_converters(registry, {.pipeline_depth = 0}),
                  std::invalid_argument);
  CHECK_THROWS_AS(register_builtin_converters(
                    registry, {.chunk_size_bytes = 1ULL << 40, .pipeline_depth = 1ULL << 40}),
                  std::invalid_argument);
}

}  // namespace cucascade
