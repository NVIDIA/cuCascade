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

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <catch2/catch_all.hpp>

#include <chrono>
#include <cstddef>
#include <future>
#include <latch>
#include <memory>
#include <utility>

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

}  // namespace cucascade
