/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucascade/cudf/builtin_converters.hpp"
#include "cucascade/cudf/gpu_data_representation.hpp"
#include "cucascade/data/representation_converter.hpp"
#include "cucascade/memory/memory_reservation_manager.hpp"
#include "cucascade/memory/memory_space.hpp"
#include "cucascade/memory/reservation_manager_configurator.hpp"
#include "cucascade/utils/cleanup_utils.hpp"
#include "utils/cuda_test_utils.hpp"
#include "utils/cudf_test_utils.hpp"
#include "utils/mixed_table.hpp"
#include "utils/peer_access_utils.hpp"

#include <rmm/device_buffer.hpp>

#include <catch2/catch_all.hpp>

#include <chrono>
#include <future>
#include <latch>
#include <utility>

namespace {

using namespace cucascade::memory;

constexpr uint64_t kKiB = 1024ULL;
constexpr uint64_t kMiB = 1024ULL * kKiB;

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

}  // namespace

namespace cucascade {

TEST_CASE("gpu_to_gpu conversion copies content correctly across GPUs",
          "[gpu-to-gpu-converter][.multi-device][cross-gpu-correctness]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  reservation_manager_configurator builder;
  builder.set_gpu_ids({kSourceDevice, kTargetDevice})
    .set_gpu_usage_limit(64 * kMiB)
    .use_numa_id_as_host_id()
    .set_per_numa_region_capacity(128 * kMiB)
    .set_host_portability(true);
  memory_reservation_manager manager(builder.build());

  require_peer_pools(manager.get_memory_space(Tier::GPU, kSourceDevice),
                     manager.get_memory_space(Tier::GPU, kTargetDevice));

  representation_converter_registry registry;
  register_builtin_converters(registry);

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

  // the converter should be blocked on the writer event gated by the producer_gate
  REQUIRE(first.wait_for(std::chrono::milliseconds{50}) == std::future_status::timeout);
  REQUIRE(second.wait_for(std::chrono::milliseconds{50}) == std::future_status::timeout);

  // now resume the halted source producer stream, presumably after the async copies have
  // been submitted (given this 100ms delay after last countdown)
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

}  // namespace cucascade
