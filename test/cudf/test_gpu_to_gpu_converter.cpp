/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "builtin_converters_detail.hpp"
#include "cucascade/cudf/builtin_converters.hpp"
#include "cucascade/cudf/gpu_data_representation.hpp"
#include "cucascade/data/representation_converter.hpp"
#include "cucascade/memory/fixed_size_host_memory_resource.hpp"
#include "cucascade/memory/memory_reservation_manager.hpp"
#include "cucascade/memory/memory_space.hpp"
#include "cucascade/memory/reservation_manager_configurator.hpp"
#include "cucascade/utils/cleanup_utils.hpp"
#include "utils/cuda_device_utils.hpp"
#include "utils/cuda_test_utils.hpp"
#include "utils/cudf_test_utils.hpp"
#include "utils/mixed_table.hpp"
#include "utils/peer_access_utils.hpp"

#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>

#include <catch2/catch_all.hpp>

#include <chrono>
#include <cstddef>
#include <future>
#include <latch>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace cucascade::memory;

constexpr uint64_t kKiB = 1024ULL;
constexpr uint64_t kMiB = 1024ULL * kKiB;
constexpr uint64_t kGiB = 1024ULL * kMiB;

constexpr int kSourceDevice = 0;
constexpr int kTargetDevice = 1;

// Grant two-way pool peer access; SKIP when the hardware cannot, FAIL on CUDA errors.
void require_peer_pools(memory_space const* source, memory_space const* destination)
{
  REQUIRE(source != nullptr);
  REQUIRE(destination != nullptr);

  const pool_peer_access_result result =
    cucascade::test::grant_two_way_pool_access(*source, *destination);

  CAPTURE(source->get_device_id(), destination->get_device_id());
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

// Bytes currently handed out by the HOST spaces of @p manager, which staged copies borrow from.
std::size_t host_bytes_in_use(memory_reservation_manager const& manager)
{
  std::size_t bytes = 0;
  for (memory_space const* space : manager.get_memory_spaces_for_tier(Tier::HOST)) {
    bytes += space->get_memory_resource_of<Tier::HOST>()->get_total_allocated_bytes();
  }
  return bytes;
}

// Highest number of bytes any HOST space of @p manager has handed out at once.
std::size_t host_peak_bytes(memory_reservation_manager const& manager)
{
  std::size_t bytes = 0;
  for (memory_space const* space : manager.get_memory_spaces_for_tier(Tier::HOST)) {
    bytes += space->get_memory_resource_of<Tier::HOST>()->get_peak_total_allocated_bytes();
  }
  return bytes;
}

// Convert a mixed table from @p source_device to @p target_device by the host-staged route and
// compare it with the same table built on the target. The two devices may be the same.
void check_staged_conversion(memory_reservation_manager& manager,
                             int source_device,
                             int target_device,
                             cudf::size_type rows)
{
  using cucascade::gpu_table_representation;
  CAPTURE(source_device, target_device, rows);

  cucascade::representation_converter_registry registry;
  cucascade::detail::register_builtin_converters(
    registry, cucascade::detail::gpu_to_gpu_converter_options{.force_staged_route = true});

  auto* source_space = manager.get_memory_space(Tier::GPU, source_device);
  auto* target_space = manager.get_memory_space(Tier::GPU, target_device);
  REQUIRE(source_space != nullptr);
  REQUIRE(target_space != nullptr);

  // Streams outlive the tables written on them; each is created with its device current.
  rmm::cuda_stream producer = [&] {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{source_device}};
    return rmm::cuda_stream{rmm::cuda_stream::flags::non_blocking};
  }();
  rmm::cuda_stream consumer = [&] {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{target_device}};
    return rmm::cuda_stream{rmm::cuda_stream::flags::non_blocking};
  }();

  std::unique_ptr<gpu_table_representation> source;
  {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{source_device}};
    source = std::make_unique<gpu_table_representation>(
      cucascade::test::make_mixed_table(rows, producer, source_space->get_default_allocator()),
      *source_space,
      producer);
  }

  std::size_t const host_bytes_before = host_bytes_in_use(manager);

  std::unique_ptr<gpu_table_representation> converted;
  {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{target_device}};
    converted = registry.convert<gpu_table_representation>(*source, target_space, consumer);
    // The converter switches devices for its two legs and must put the caller's device back.
    CHECK(rmm::get_current_cuda_device().value() == target_device);
  }
  REQUIRE(converted != nullptr);
  CHECK(converted->get_device_id() == target_device);

  // The copy went through this manager's HOST pool and gave every staging block back.
  CHECK(host_peak_bytes(manager) > host_bytes_before);
  CHECK(host_bytes_in_use(manager) == host_bytes_before);

  // The result must not depend on the source after the conversion returns.
  {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{source_device}};
    source.reset();
  }

  {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{target_device}};
    auto expected =
      cucascade::test::make_mixed_table(rows, consumer, target_space->get_default_allocator());
    cucascade::test::expect_cudf_tables_equal_on_stream(
      expected->view(), converted->get_table_view(), consumer);
    expected.reset();
    converted.reset();
    consumer.synchronize();
  }
}

}  // namespace

namespace cucascade {

namespace {

// Two converters read one source whose last write is still held back on the producer stream;
// both must wait for that write. With @p force_staged_route the copies go through host memory,
// otherwise the device pair must have peer access.
void check_conversion_waits_for_writer(bool force_staged_route)
{
  reservation_manager_configurator builder;
  builder.set_gpu_ids({kSourceDevice, kTargetDevice})
    .set_gpu_usage_limit(64 * kMiB)
    .use_numa_id_as_host_id()
    .set_per_numa_region_capacity(128 * kMiB)
    .set_host_portability(true);
  memory_reservation_manager manager(builder.build());

  if (!force_staged_route) {
    require_peer_pools(manager.get_memory_space(Tier::GPU, kSourceDevice),
                       manager.get_memory_space(Tier::GPU, kTargetDevice));
  }

  representation_converter_registry registry;
  detail::register_builtin_converters(
    registry, detail::gpu_to_gpu_converter_options{.force_staged_route = force_staged_route});

  rmm::cuda_set_device_raii source_guard{rmm::cuda_device_id{kSourceDevice}};
  rmm::cuda_stream source_producer_stream{rmm::cuda_stream::flags::non_blocking};

  constexpr cudf::size_type kRows = 1024;
  auto* source_mem_space          = manager.get_memory_space(Tier::GPU, kSourceDevice);
  // make_mixed_table waits internally, so finish it before closing the producer gate.
  auto source_table = test::make_mixed_table(
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

  test::stream_gate producer_gate{source_producer_stream.value()};
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

  std::latch workers_ready{2};
  auto convert = [&] {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{kTargetDevice}};
    rmm::cuda_stream reader{rmm::cuda_stream::flags::non_blocking};
    auto* target_mem_space = manager.get_memory_space(Tier::GPU, kTargetDevice);
    workers_ready.count_down();
    return registry.convert<gpu_table_representation>(*source_data, target_mem_space, reader);
  };

  std::future<std::unique_ptr<gpu_table_representation>> first;
  std::future<std::unique_ptr<gpu_table_representation>> second;

  // to prevent deadlocks in case of conversion failure
  auto cleanup = utils::on_scope_exit([&]() noexcept { producer_gate.release(); });

  // two independent converters reading from the same source
  first  = std::async(std::launch::async, convert);
  second = std::async(std::launch::async, convert);

  workers_ready.wait();

  // resume the halted producer after a 100ms grace period. Each converter has enqueued its copies
  // only up to the first STRING offset readback, which blocks behind the gate. Timing affects
  // coverage only; the ID check above proves ordering either way.
  // TODO: make this fully deterministic, eg by using a custom allocator of the test that
  // notifies on first allocation signalling the copies have been enqueued.
  producer_gate.release_after(std::chrono::milliseconds{100});
  auto left  = first.get();
  auto right = second.get();

  // comparisons should not rely on source's existence after the copy
  source_data.reset();

  {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{kTargetDevice}};
    rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
    auto* target_mem_space = manager.get_memory_space(Tier::GPU, kTargetDevice);

    auto expected_table =
      test::make_mixed_table(kRows, consumer, target_mem_space->get_default_allocator());

    test::expect_cudf_tables_equal_on_stream(
      expected_table->view(), left->get_table_view(), consumer);
    test::expect_cudf_tables_equal_on_stream(
      expected_table->view(), right->get_table_view(), consumer);
  }
}

}  // namespace

TEST_CASE("gpu_to_gpu conversion copies content correctly across GPUs",
          "[gpu-to-gpu-converter][multi-device][cross-gpu-correctness]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  check_conversion_waits_for_writer(/*force_staged_route=*/false);
}

TEST_CASE("gpu_to_gpu staged route waits for the source's writer across GPUs",
          "[gpu-to-gpu-converter][staged-copy][multi-device]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  check_conversion_waits_for_writer(/*force_staged_route=*/true);
}

TEST_CASE("gpu_to_gpu staged route copies a mixed table within one GPU",
          "[gpu-to-gpu-converter][staged-copy]")
{
  if (test::cuda_device_count() < 1) { SKIP("requires a GPU"); }

  // 200,000 rows give buffers of several MiB, so a buffer spans more than one 1 MiB host block.
  auto const rows = GENERATE(cudf::size_type{1}, cudf::size_type{1024}, cudf::size_type{200'000});

  reservation_manager_configurator builder;
  builder.set_gpu_ids({kSourceDevice})
    .set_gpu_usage_limit(1 * kGiB)
    .use_numa_id_as_host_id()
    .set_per_numa_region_capacity(256 * kMiB)
    .set_host_portability(true);
  memory_reservation_manager manager(builder.build());

  check_staged_conversion(manager, kSourceDevice, kSourceDevice, rows);
}

TEST_CASE("gpu_to_gpu staged route copies a mixed table across GPUs in both directions",
          "[gpu-to-gpu-converter][staged-copy][multi-device]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  auto const rows = GENERATE(cudf::size_type{1024}, cudf::size_type{200'000});

  reservation_manager_configurator builder;
  builder.set_gpu_ids({kSourceDevice, kTargetDevice})
    .set_gpu_usage_limit(1 * kGiB)
    .use_numa_id_as_host_id()
    .set_per_numa_region_capacity(256 * kMiB)
    .set_host_portability(true);
  memory_reservation_manager manager(builder.build());

  check_staged_conversion(manager, kSourceDevice, kTargetDevice, rows);
  check_staged_conversion(manager, kTargetDevice, kSourceDevice, rows);
}

namespace {

// A registry whose GPU -> GPU converter always stages, with the counters of its staged copies.
struct staged_registry {
  explicit staged_registry(memory::detail::staged_copy_ring_settings ring_settings = {})
  {
    detail::register_builtin_converters(
      registry,
      detail::gpu_to_gpu_converter_options{
        .force_staged_route = true, .ring_settings = ring_settings, .instrumentation = &counters});
  }

  memory::detail::staged_copy_instrumentation counters;
  representation_converter_registry registry;
};

auto one_gpu_spaces(std::size_t host_capacity_bytes = 256 * kMiB)
{
  reservation_manager_configurator builder;
  builder.set_gpu_ids({kSourceDevice})
    .set_gpu_usage_limit(1 * kGiB)
    .use_numa_id_as_host_id()
    .set_per_numa_region_capacity(host_capacity_bytes)
    .set_host_portability(true);
  return builder.build();
}

// Convert one mixed table per entry of @p directions (source device, target device), each
// @p repeats times on a thread of its own, all starting together, and compare every result with
// the same table built on its target. Tables are built and compared on the calling thread; only
// the conversions run concurrently.
void check_concurrent_staged_conversions(memory_reservation_manager& manager,
                                         std::vector<std::pair<int, int>> const& directions,
                                         cudf::size_type rows,
                                         int repeats)
{
  staged_registry staged;

  struct job {
    int source_device;
    int target_device;
    rmm::cuda_stream producer;
    rmm::cuda_stream consumer;
    std::unique_ptr<gpu_table_representation> source;
    std::vector<std::unique_ptr<gpu_table_representation>> converted;
  };
  auto const make_stream = [](int device) {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
    return rmm::cuda_stream{rmm::cuda_stream::flags::non_blocking};
  };

  std::vector<std::unique_ptr<job>> jobs;
  for (auto const& [source_device, target_device] : directions) {
    auto* const source_space = manager.get_memory_space(Tier::GPU, source_device);
    REQUIRE(source_space != nullptr);
    REQUIRE(manager.get_memory_space(Tier::GPU, target_device) != nullptr);
    jobs.push_back(std::unique_ptr<job>(new job{source_device,
                                                target_device,
                                                make_stream(source_device),
                                                make_stream(target_device),
                                                nullptr,
                                                {}}));
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{source_device}};
    jobs.back()->source = std::make_unique<gpu_table_representation>(
      test::make_mixed_table(rows, jobs.back()->producer, source_space->get_default_allocator()),
      *source_space,
      jobs.back()->producer);
  }

  std::size_t const host_before = host_bytes_in_use(manager);

  std::latch ready{static_cast<std::ptrdiff_t>(jobs.size())};
  std::latch start{1};
  std::vector<std::future<void>> workers;
  for (auto& current : jobs) {
    workers.push_back(std::async(std::launch::async, [&, work = current.get()] {
      ready.count_down();
      start.wait();
      rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{work->target_device}};
      auto* const target_space = manager.get_memory_space(Tier::GPU, work->target_device);
      for (int i = 0; i < repeats; ++i) {
        work->converted.push_back(staged.registry.convert<gpu_table_representation>(
          *work->source, target_space, work->consumer));
      }
    }));
  }
  ready.wait();
  start.count_down();
  for (auto& worker : workers) {
    REQUIRE(worker.wait_for(std::chrono::seconds{120}) == std::future_status::ready);
  }
  for (auto& worker : workers) {
    REQUIRE_NOTHROW(worker.get());
  }

  CHECK(staged.counters.rings.load() == jobs.size() * static_cast<std::size_t>(repeats));
  CHECK(staged.counters.abandoned_rings.load() == 0U);
  CHECK(host_bytes_in_use(manager) == host_before);

  for (auto& current : jobs) {
    CAPTURE(current->source_device, current->target_device);
    {
      rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{current->source_device}};
      current->source.reset();
      current->producer.synchronize();
    }
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{current->target_device}};
    auto* const target_space = manager.get_memory_space(Tier::GPU, current->target_device);
    auto expected =
      test::make_mixed_table(rows, current->consumer, target_space->get_default_allocator());
    REQUIRE(current->converted.size() == static_cast<std::size_t>(repeats));
    for (auto const& converted : current->converted) {
      REQUIRE(converted != nullptr);
      test::expect_cudf_tables_equal_on_stream(
        expected->view(), converted->get_table_view(), current->consumer);
    }
    expected.reset();
    current->converted.clear();
    current->consumer.synchronize();
  }
}

// The six fixed-width columns of test::make_mixed_table(); the other ten have children.
const std::vector<cudf::size_type> kFixedWidthColumns{0, 1, 2, 3, 4, 5};

}  // namespace

TEST_CASE("gpu_to_gpu staged route waits a fixed number of times and borrows a bounded ring",
          "[gpu-to-gpu-converter][staged-copy]")
{
  if (test::cuda_device_count() < 1) { SKIP("requires a GPU"); }

  memory_reservation_manager manager(one_gpu_spaces());
  auto* const space = manager.get_memory_space(Tier::GPU, kSourceDevice);
  REQUIRE(space != nullptr);

  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kSourceDevice}};
  rmm::cuda_stream producer{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
  ::cuda::stream_ref const producer_ref = producer;
  auto const mr                         = space->get_default_allocator();

  // Tens of MiB in 16 columns, several of them nested: far more than one ring.
  constexpr cudf::size_type kRows = 200'000;
  constexpr std::size_t kLargestRing =
    staged_copy_settings::default_chunk_bytes * staged_copy_settings::default_ring_depth;
  auto table = test::make_mixed_table(kRows, producer, mr);

  staged_registry staged;
  std::size_t const host_before = host_bytes_in_use(manager);
  std::unique_ptr<gpu_table_representation> converted;

  SECTION("a known writer: one wait for the string sizes and one for the last byte")
  {
    gpu_table_representation source{std::move(table), *space, producer};
    converted = staged.registry.convert<gpu_table_representation>(source, space, consumer);

    CHECK(staged.counters.readiness_syncs.load() == 0U);
    CHECK(staged.counters.metadata_syncs.load() == 1U);
  }

  SECTION("an unknown writer adds one wait for the source device")
  {
    producer.synchronize();
    gpu_table_representation source{
      std::move(table), *space, ::cuda::stream_ref{cudaStream_t{nullptr}}};
    converted = staged.registry.convert<gpu_table_representation>(source, space, consumer);

    CHECK(staged.counters.readiness_syncs.load() == 1U);
    CHECK(staged.counters.metadata_syncs.load() == 1U);
  }

  SECTION("fixed-width columns need no sizes from the device")
  {
    gpu_table_representation source{
      std::make_unique<cudf::table>(table->select(kFixedWidthColumns), producer_ref, mr),
      *space,
      producer};
    converted = staged.registry.convert<gpu_table_representation>(source, space, consumer);

    CHECK(staged.counters.readiness_syncs.load() == 0U);
    CHECK(staged.counters.metadata_syncs.load() == 0U);
    test::expect_cudf_tables_equal_on_stream(
      table->select(kFixedWidthColumns), converted->get_table_view(), consumer);
  }

  REQUIRE(converted != nullptr);
  CHECK(converted->get_table_view().num_rows() == kRows);
  CHECK(staged.counters.final_syncs.load() == 1U);
  CHECK(staged.counters.rings.load() == 1U);
  CHECK(staged.counters.last_ring_bytes.load() <= kLargestRing);
  CHECK(host_peak_bytes(manager) <= host_before + kLargestRing);
  CHECK(host_bytes_in_use(manager) == host_before);

  converted.reset();
  table.reset();
  producer.synchronize();
  consumer.synchronize();
}

TEST_CASE("gpu_to_gpu staged route honours its ring settings",
          "[gpu-to-gpu-converter][staged-copy]")
{
  if (test::cuda_device_count() < 1) { SKIP("requires a GPU"); }

  memory_reservation_manager manager(one_gpu_spaces());
  auto* const space = manager.get_memory_space(Tier::GPU, kSourceDevice);
  REQUIRE(space != nullptr);

  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kSourceDevice}};
  rmm::cuda_stream producer{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
  auto const mr = space->get_default_allocator();

  constexpr cudf::size_type kRows = 200'000;
  staged_registry staged{{.chunk_bytes = 1 * kMiB, .depth = 2}};

  auto converted = [&] {
    gpu_table_representation source{test::make_mixed_table(kRows, producer, mr), *space, producer};
    return staged.registry.convert<gpu_table_representation>(source, space, consumer);
  }();

  // Two slots of one 1 MiB block carry the whole table.
  CHECK(staged.counters.last_ring_bytes.load() == 2 * kMiB);
  CHECK(staged.counters.final_syncs.load() == 1U);

  auto expected = test::make_mixed_table(kRows, consumer, mr);
  test::expect_cudf_tables_equal_on_stream(expected->view(), converted->get_table_view(), consumer);

  expected.reset();
  converted.reset();
  producer.synchronize();
  consumer.synchronize();
}

TEST_CASE("gpu_to_gpu staged route builds tables without rows on the host",
          "[gpu-to-gpu-converter][staged-copy]")
{
  if (test::cuda_device_count() < 1) { SKIP("requires a GPU"); }

  memory_reservation_manager manager(one_gpu_spaces());
  auto* const space = manager.get_memory_space(Tier::GPU, kSourceDevice);
  REQUIRE(space != nullptr);

  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kSourceDevice}};
  rmm::cuda_stream producer{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
  ::cuda::stream_ref const producer_ref = producer;
  auto const mr                         = space->get_default_allocator();

  auto owner = std::shared_ptr<cudf::table>(test::make_mixed_table(1024, producer, mr));
  producer.synchronize();

  staged_registry staged;
  std::size_t const host_peak_before = host_peak_bytes(manager);

  // Nothing is read from the source, so nothing is waited for and no host memory is borrowed.
  auto const check_built_on_host = [&](gpu_table_representation const& source) {
    auto converted = staged.registry.convert<gpu_table_representation>(source, space, consumer);
    REQUIRE(converted != nullptr);
    auto const result = converted->get_table_view();
    CHECK(result.num_rows() == 0);
    CHECK(result.num_columns() == source.get_table_view().num_columns());
    CHECK(cudf::have_same_types(source.get_table_view(), result));
    CHECK(staged.counters.rings.load() == 0U);
    CHECK(staged.counters.readiness_syncs.load() == 0U);
    CHECK(staged.counters.metadata_syncs.load() == 0U);
    CHECK(staged.counters.final_syncs.load() == 0U);
    CHECK(host_peak_bytes(manager) == host_peak_before);
  };

  SECTION("a table without rows whose writer is unknown")
  {
    gpu_table_representation source{
      cudf::empty_like(owner->view()), *space, ::cuda::stream_ref{cudaStream_t{nullptr}}};
    check_built_on_host(source);
  }

  SECTION("views without rows that start past row 0 and keep their children")
  {
    auto const views = cudf::slice(owner->view(), std::vector<cudf::size_type>{5, 5}, producer_ref);
    gpu_table_representation source{
      views.front(), std::shared_ptr<cudf::table>{owner}, std::size_t{0}, *space, producer};
    check_built_on_host(source);
  }

  SECTION("a table without columns")
  {
    gpu_table_representation source{std::make_unique<cudf::table>(), *space, producer};
    check_built_on_host(source);
  }

  owner.reset();
  producer.synchronize();
  consumer.synchronize();
}

TEST_CASE("gpu_to_gpu staged route refuses sliced columns it cannot copy whole",
          "[gpu-to-gpu-converter][staged-copy]")
{
  if (test::cuda_device_count() < 1) { SKIP("requires a GPU"); }

  memory_reservation_manager manager(one_gpu_spaces());
  auto* const space = manager.get_memory_space(Tier::GPU, kSourceDevice);
  REQUIRE(space != nullptr);

  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kSourceDevice}};
  rmm::cuda_stream producer{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
  ::cuda::stream_ref const producer_ref = producer;
  auto const mr                         = space->get_default_allocator();

  constexpr cudf::size_type kRows = 1024;
  auto owner = std::shared_ptr<cudf::table>(test::make_mixed_table(kRows, producer, mr));

  staged_registry staged;
  std::size_t const host_before = host_bytes_in_use(manager);

  auto const convert_view = [&](cudf::table_view const& view) {
    gpu_table_representation source{
      view, std::shared_ptr<cudf::table>{owner}, std::size_t{0}, *space, producer};
    return staged.registry.convert<gpu_table_representation>(source, space, consumer);
  };

  SECTION("a view that starts past row 0, whatever its type")
  {
    auto const tail =
      cudf::slice(owner->view(), std::vector<cudf::size_type>{1, kRows}, producer_ref).front();
    CHECK_THROWS_AS(convert_view(tail), std::invalid_argument);
    CHECK_THROWS_AS(convert_view(tail.select(kFixedWidthColumns)), std::invalid_argument);
    CHECK(staged.counters.rings.load() == 0U);
  }

  SECTION("a view that ends early over a column with children")
  {
    auto const head =
      cudf::slice(owner->view(), std::vector<cudf::size_type>{0, kRows - 1}, producer_ref).front();
    for (cudf::size_type column = 6; column < head.num_columns(); ++column) {
      CAPTURE(column);
      CHECK_THROWS_AS(convert_view(head.select({column})), std::invalid_argument);
    }
    CHECK(staged.counters.rings.load() == 0U);
  }

  SECTION("a view that ends early over fixed-width columns is copied")
  {
    auto const head = cudf::slice(owner->view().select(kFixedWidthColumns),
                                  std::vector<cudf::size_type>{0, kRows - 1},
                                  producer_ref)
                        .front();
    auto converted = convert_view(head);
    REQUIRE(converted != nullptr);
    test::expect_cudf_tables_equal_on_stream(head, converted->get_table_view(), consumer);
  }

  CHECK(host_bytes_in_use(manager) == host_before);

  owner.reset();
  producer.synchronize();
  consumer.synchronize();
}

TEST_CASE("gpu_to_gpu staged route runs concurrent conversions within one GPU",
          "[gpu-to-gpu-converter][staged-copy]")
{
  if (test::cuda_device_count() < 1) { SKIP("requires a GPU"); }

  memory_reservation_manager manager(one_gpu_spaces());

  // Four conversions at a time share the device's stream pool and the HOST pool.
  check_concurrent_staged_conversions(manager,
                                      {{kSourceDevice, kSourceDevice},
                                       {kSourceDevice, kSourceDevice},
                                       {kSourceDevice, kSourceDevice},
                                       {kSourceDevice, kSourceDevice}},
                                      50'000,
                                      3);
}

TEST_CASE("gpu_to_gpu staged route runs concurrent conversions in opposite directions",
          "[gpu-to-gpu-converter][staged-copy][multi-device]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  reservation_manager_configurator builder;
  builder.set_gpu_ids({kSourceDevice, kTargetDevice})
    .set_gpu_usage_limit(1 * kGiB)
    .use_numa_id_as_host_id()
    .set_per_numa_region_capacity(256 * kMiB)
    .set_host_portability(true);
  memory_reservation_manager manager(builder.build());

  // Each device is the source of two conversions and the target of two others at once.
  check_concurrent_staged_conversions(manager,
                                      {{kSourceDevice, kTargetDevice},
                                       {kTargetDevice, kSourceDevice},
                                       {kSourceDevice, kTargetDevice},
                                       {kTargetDevice, kSourceDevice}},
                                      50'000,
                                      3);
}

TEST_CASE("gpu_to_gpu staged route copes with a HOST pool smaller than its ring",
          "[gpu-to-gpu-converter][staged-copy]")
{
  if (test::cuda_device_count() < 1) { SKIP("requires a GPU"); }

  // Half of what the default ring of four 4 MiB chunks asks for.
  constexpr std::size_t kHostCapacity = 8 * kMiB;
  memory_reservation_manager manager(one_gpu_spaces(kHostCapacity));
  auto* const space = manager.get_memory_space(Tier::GPU, kSourceDevice);
  REQUIRE(space != nullptr);

  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{kSourceDevice}};
  rmm::cuda_stream producer{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
  auto const mr = space->get_default_allocator();

  constexpr cudf::size_type kRows = 200'000;
  staged_registry staged;
  gpu_table_representation source{test::make_mixed_table(kRows, producer, mr), *space, producer};
  auto expected = test::make_mixed_table(kRows, consumer, mr);

  SECTION("it converts through a smaller ring")
  {
    auto converted = staged.registry.convert<gpu_table_representation>(source, space, consumer);
    REQUIRE(converted != nullptr);
    CHECK(staged.counters.last_ring_bytes.load() <= kHostCapacity);
    CHECK(staged.counters.last_ring_bytes.load() > 0U);
    CHECK(host_bytes_in_use(manager) == 0U);
    test::expect_cudf_tables_equal_on_stream(
      expected->view(), converted->get_table_view(), consumer);
  }

  SECTION("it fails with out-of-memory when no block is left, and works again afterwards")
  {
    using held_blocks = fixed_size_host_memory_resource::fixed_multiple_blocks_allocation;
    std::vector<held_blocks> held;
    for (memory_space const* host_space : manager.get_memory_spaces_for_tier(Tier::HOST)) {
      auto* const pool = host_space->get_memory_resource_of<Tier::HOST>();
      held.push_back(pool->allocate_multiple_blocks(pool->get_available_memory()));
    }
    std::size_t const host_held = host_bytes_in_use(manager);

    CHECK_THROWS_AS(staged.registry.convert<gpu_table_representation>(source, space, consumer),
                    rmm::out_of_memory);
    CHECK(staged.counters.rings.load() == 0U);
    CHECK(host_bytes_in_use(manager) == host_held);

    held.clear();
    auto converted = staged.registry.convert<gpu_table_representation>(source, space, consumer);
    REQUIRE(converted != nullptr);
    CHECK(host_bytes_in_use(manager) == 0U);
    test::expect_cudf_tables_equal_on_stream(
      expected->view(), converted->get_table_view(), consumer);
  }

  expected.reset();
  producer.synchronize();
  consumer.synchronize();
}

TEST_CASE("staged copy settings reject values outside their range",
          "[gpu-to-gpu-converter][staged-copy]")
{
  CHECK(staged_copy_settings{}.chunk_bytes() == std::size_t{4} << 20);
  CHECK(staged_copy_settings{}.ring_depth() == 4U);

  CHECK_NOTHROW(staged_copy_settings{1, 1});
  CHECK_NOTHROW(staged_copy_settings{staged_copy_settings::max_chunk_bytes,
                                     staged_copy_settings::max_ring_depth});
  CHECK_THROWS_AS((staged_copy_settings{0, 4}), std::invalid_argument);
  CHECK_THROWS_AS((staged_copy_settings{4 * kMiB, 0}), std::invalid_argument);
  CHECK_THROWS_AS((staged_copy_settings{staged_copy_settings::max_chunk_bytes + 1, 4}),
                  std::invalid_argument);
  CHECK_THROWS_AS((staged_copy_settings{4 * kMiB, staged_copy_settings::max_ring_depth + 1}),
                  std::invalid_argument);

  representation_converter_registry registry;
  CHECK_NOTHROW(register_builtin_converters(registry, staged_copy_settings{1 * kMiB, 2}));
  CHECK(registry.has_converter<gpu_table_representation, gpu_table_representation>());
}

}  // namespace cucascade
