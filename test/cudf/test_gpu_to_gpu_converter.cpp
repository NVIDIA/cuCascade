/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucascade/cudf/builtin_converters.hpp"
#include "cucascade/cudf/gpu_data_representation.hpp"
#include "cucascade/data/data_batch.hpp"
#include "cucascade/data/representation_converter.hpp"
#include "cucascade/memory/fixed_size_host_memory_resource.hpp"
#include "cucascade/memory/memory_space.hpp"
#include "cucascade/memory/reservation_aware_resource_adaptor.hpp"
#include "cucascade/memory/reservation_manager_configurator.hpp"
#include "cucascade/utils/cleanup_utils.hpp"
#include "utils/child_test_runner.hpp"
#include "utils/cuda_test_utils.hpp"
#include "utils/cudf_test_utils.hpp"

#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>

#include <rmm/device_buffer.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <chrono>
#include <future>
#include <latch>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace cucascade::memory;

constexpr uint64_t kKiB = 1024ULL;
constexpr uint64_t kMiB = 1024ULL * kKiB;

constexpr int kSourceDevice = 0;
constexpr int kTargetDevice = 1;

void grant_two_way_pool_access(memory_space const* source, memory_space const* destination)
{
  REQUIRE(source != nullptr);
  REQUIRE(destination != nullptr);

  for (auto const [owner_space, accessing_device_id] :
       {std::pair{source, destination->get_device_id()},
        std::pair{destination, source->get_device_id()}}) {
    auto const pool = owner_space->get_memory_resource_of<Tier::GPU>()->pool_handle();
    REQUIRE(pool != nullptr);

    const pool_peer_access_result result =
      grant_pool_peer_access(pool, owner_space->get_device_id(), accessing_device_id);

    CAPTURE(owner_space->get_device_id(), accessing_device_id);
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
}

std::unique_ptr<cudf::table> make_mixed_table(cudf::size_type rows,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr);

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

  grant_two_way_pool_access(manager.get_memory_space(Tier::GPU, kSourceDevice),
                            manager.get_memory_space(Tier::GPU, kTargetDevice));

  representation_converter_registry registry;
  register_builtin_converters(registry);

  rmm::cuda_set_device_raii source_guard{rmm::cuda_device_id{kSourceDevice}};
  rmm::cuda_stream source_producer_stream{rmm::cuda_stream::flags::non_blocking};

  constexpr cudf::size_type kRows = 1024;
  auto* source_mem_space          = manager.get_memory_space(Tier::GPU, kSourceDevice);
  // make_mixed_table waits internally, so finish it before closing the producer gate.
  auto source_table =
    make_mixed_table(kRows, source_producer_stream, source_mem_space->get_default_allocator());
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
      make_mixed_table(kRows, consumer, target_mem_space->get_default_allocator());

    test::expect_cudf_tables_equal_on_stream(
      expected_table->view(), left->get_table_view(), consumer);
    test::expect_cudf_tables_equal_on_stream(
      expected_table->view(), right->get_table_view(), consumer);
  }
}

}  // namespace cucascade

namespace {

/**
 * @brief Build 16 mixed columns by round-robin repetition of ten readable sample rows.
 *
 * Layout: 6 numeric, 6 STRING, LIST<INT32>, LIST<STRING>, STRUCT<INT64, STRING>,
 * DICTIONARY<STRING>. Includes nullable/non-nullable data, all-null strings,
 * empty strings and empty lists. Sample row i is used for output row i % 10.
 *
 * Conservative retained device-buffer budget: 512 bytes per row + 32 KiB per
 * table, including masks/offsets (text <= 24 bytes, lists <= 3 elements, offsets
 * up to INT64). For 1024 rows, budget 544 KiB per table. This is NOT a peak-memory
 * bound: exclude allocator pool reservation, gather/construction workspace and
 * CUDA overhead. Two converted outputs plus expected occupy three target tables.
 *
 * Host gather-map payload: 4 bytes per row (4 KiB for 1024), plus ten-row wrapper
 * temporaries and container overhead. cuDF internal workspace is additional.
 *
 * Setup synchronizes before returning: call BEFORE arming a producer gate and
 * outside benchmark timing. Uses the supplied stream and memory resource.
 */
std::unique_ptr<cudf::table> make_mixed_table(cudf::size_type rows,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(rows >= 0 && rows <= std::numeric_limits<cudf::size_type>::max() / 3,
               "Invalid row count or potentially overflowing list size");
  using cudf::test::dictionary_column_wrapper;
  using cudf::test::fixed_point_column_wrapper;
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::strings_column_wrapper;
  using cudf::test::structs_column_wrapper;
  using int_list  = cudf::test::lists_column_wrapper<int32_t>;
  using text_list = cudf::test::lists_column_wrapper<cudf::string_view>;

  constexpr std::array<int64_t, 10> kIds{101, 205, 309, 412, 518, 623, 727, 834, 946, 1051};
  constexpr std::array<int32_t, 10> kCounts{2, 7, 1, 12, 4, 9, 3, 15, 6, 11};
  constexpr std::array<double, 10> kAmounts{
    12.50, 8.25, 19.75, 24.00, 6.50, 31.25, 4.75, 17.00, 42.50, 9.25};
  constexpr std::array<float, 10> kWeights{
    1.5F, 2.25F, 0.5F, 3.0F, 4.75F, 1.25F, 0.75F, 5.0F, 2.5F, 6.25F};
  constexpr std::array<bool, 10> kFlags{
    true, false, true, true, false, false, true, false, true, false};
  constexpr std::array<int64_t, 10> kCents{1250, 825, 1975, 2400, 650, 3125, 475, 1700, 4250, 925};
  constexpr std::array<bool, 10> kValid{
    true, false, true, true, false, true, true, false, true, true};
  constexpr std::array<bool, 10> kAllNull{};
  constexpr std::array<const char*, 10> kNames{"amber river",
                                               "quiet harbor",
                                               "silver ridge",
                                               "green meadow",
                                               "copper forest",
                                               "bright orchard",
                                               "gentle valley",
                                               "golden woodland",
                                               "clear waterfall",
                                               "blue horizon"};
  constexpr std::array<const char*, 10> kCities{
    "London", "Seattle", "Delhi", "Tokyo", "Berlin", "Nairobi", "Lima", "Oslo", "Seoul", "Cairo"};
  constexpr std::array<const char*, 10> kActions{"created",
                                                 "packed",
                                                 "shipped",
                                                 "arrived",
                                                 "checked",
                                                 "returned",
                                                 "stored",
                                                 "loaded",
                                                 "sold",
                                                 "held"};
  constexpr std::array<const char*, 10> kNotes{"handle with care",
                                               "deliver before noon",
                                               "keep upright",
                                               "leave at reception",
                                               "check address",
                                               "fragile contents",
                                               "call on arrival",
                                               "store indoors",
                                               "signature required",
                                               "ready for pickup"};
  constexpr std::array<const char*, 10> kRegions{
    "north", "south", "east", "west", "central", "coastal", "inland", "upper", "lower", "island"};
  constexpr std::array<const char*, 10> kEmpty{"", "", "", "", "", "", "", "", "", ""};
  constexpr auto kShortText = [](auto const& samples) {
    for (auto text : samples) {
      if (std::string_view{text}.size() > 24) { return false; }
    }
    return true;
  };
  static_assert(kShortText(kNames) && kShortText(kCities) && kShortText(kActions) &&
                kShortText(kNotes) && kShortText(kRegions));

  fixed_width_column_wrapper<int64_t> id_col(kIds.begin(), kIds.end(), stream, mr);
  fixed_width_column_wrapper<int32_t> count_col(
    kCounts.begin(), kCounts.end(), kValid.begin(), stream, mr);
  fixed_width_column_wrapper<double> amount_col(
    kAmounts.begin(), kAmounts.end(), kValid.begin(), stream, mr);
  fixed_width_column_wrapper<float> weight_col(kWeights.begin(), kWeights.end(), stream, mr);
  fixed_width_column_wrapper<bool> flag_col(kFlags.begin(), kFlags.end(), stream, mr);
  fixed_point_column_wrapper<int64_t> price_col(
    kCents.begin(), kCents.end(), kValid.begin(), numeric::scale_type{-2}, stream, mr);
  strings_column_wrapper name_col(kNames.begin(), kNames.end(), stream, mr);
  strings_column_wrapper city_col(kCities.begin(), kCities.end(), kValid.begin(), stream, mr);
  strings_column_wrapper action_col(kActions.begin(), kActions.end(), kValid.begin(), stream, mr);
  strings_column_wrapper note_col(kNotes.begin(), kNotes.end(), kValid.begin(), stream, mr);
  strings_column_wrapper null_col(kNames.begin(), kNames.end(), kAllNull.begin(), stream, mr);
  strings_column_wrapper empty_col(kEmpty.begin(), kEmpty.end(), stream, mr);

  // Ten host-side list samples. The outer wrapper propagates stream/mr to every child.
  int_list item_col(
    {{1, 2}, {3}, {}, {4, 5, 6}, {7}, {8, 9}, {10}, {11, 12, 13}, {14}, {15, 16}}, stream, mr);
  text_list tag_col({{kNames[0], kRegions[0]},
                     {kNames[1], kRegions[1]},
                     {kNames[2], kRegions[2]},
                     {kNames[3], kRegions[3]},
                     {kNames[4], kRegions[4]},
                     {kNames[5], kRegions[5]},
                     {kNames[6], kRegions[6]},
                     {kNames[7], kRegions[7]},
                     {kNames[8], kRegions[8]},
                     {kNames[9], kRegions[9]}},
                    stream,
                    mr);
  structs_column_wrapper detail_col(
    {id_col, name_col}, std::vector<bool>(kValid.begin(), kValid.end()), stream, mr);
  dictionary_column_wrapper<std::string> region_col(
    kRegions.begin(), kRegions.end(), kValid.begin(), stream, mr);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(id_col.release());      // 0: INT64
  columns.push_back(count_col.release());   // 1: nullable INT32
  columns.push_back(amount_col.release());  // 2: nullable FLOAT64
  columns.push_back(weight_col.release());  // 3: FLOAT32
  columns.push_back(flag_col.release());    // 4: BOOL8
  columns.push_back(price_col.release());   // 5: nullable DECIMAL64(-2)
  columns.push_back(name_col.release());    // 6..11: STRINGs
  columns.push_back(city_col.release());
  columns.push_back(action_col.release());
  columns.push_back(note_col.release());
  columns.push_back(null_col.release());
  columns.push_back(empty_col.release());
  columns.push_back(item_col.release());    // 12: LIST<INT32>
  columns.push_back(tag_col.release());     // 13: LIST<STRING>
  columns.push_back(detail_col.release());  // 14: nullable STRUCT
  columns.push_back(region_col.release());  // 15: nullable DICTIONARY<STRING>
  cudf::table samples{std::move(columns)};

  // Repeat complete sample rows, including their nulls and nested values.
  std::vector<int32_t> row_indices(static_cast<std::size_t>(rows));
  for (cudf::size_type row = 0; row < rows; ++row) {
    row_indices[static_cast<std::size_t>(row)] = row % 10;
  }
  fixed_width_column_wrapper<int32_t> row_map(row_indices.begin(), row_indices.end(), stream, mr);
  auto table =
    cudf::gather(samples.view(), row_map, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  // Complete use of temporary samples and row map before destroying them.
  stream.sync();
  return table;
}

}  // namespace
