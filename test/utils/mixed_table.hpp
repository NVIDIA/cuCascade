/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cucascade/cuda/event.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cucascade {
namespace test {

/**
 * @brief Rebuild a STRING column with INT64 offsets, keeping its chars, null mask and values.
 *
 * cudf picks INT32 offsets for small columns; this forces the wide offsets path.
 * Work is stream-ordered on @p stream.
 */
inline std::unique_ptr<cudf::column> widen_string_offsets(std::unique_ptr<cudf::column> strings,
                                                          ::cuda::stream_ref stream,
                                                          rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(strings->type().id() == cudf::type_id::STRING, "expected a STRING column");

  auto const rows       = strings->size();
  auto const null_count = strings->null_count();
  auto contents         = strings->release();

  // e.g. INT32 [0, 6, 13] -> INT64 [0, 6, 13]
  auto offsets = cudf::cast(
    contents.children.front()->view(), cudf::data_type{cudf::type_id::INT64}, stream, mr);

  return cudf::make_strings_column(rows,
                                   std::move(offsets),
                                   std::move(*contents.data),
                                   null_count,
                                   std::move(*contents.null_mask));
}

/// Conservative retained device bytes per row of a make_mixed_table() table.
constexpr std::size_t kMixedTableBytesPerRow = 512;
/// Conservative retained device bytes per make_mixed_table() table, independent of rows.
constexpr std::size_t kMixedTableFixedBytes = 32 * 1024;

/**
 * @brief Build 16 mixed columns by round-robin repetition of ten readable sample rows.
 *
 * Layout: 6 numeric, 6 STRING, LIST<INT32>, LIST<STRING>, STRUCT<INT64, STRING>,
 * DICTIONARY<STRING>. Includes nullable/non-nullable data, all-null strings,
 * empty strings and empty lists. Two STRING columns (7, 9) use INT64 offsets, the
 * rest INT32. Sample row i is used for output row i % 10.
 *
 * Conservative retained device-buffer budget: kMixedTableBytesPerRow (512) bytes per row +
 * kMixedTableFixedBytes (32 KiB) per table, including masks/offsets (text <= 24 bytes, lists <= 3
 * elements, offsets up to INT64). For 1024 rows, budget 544 KiB per table. This is NOT a
 * peak-memory bound: exclude allocator pool reservation, gather/construction workspace and CUDA
 * overhead. Two converted outputs plus expected occupy three target tables.
 *
 * Host gather-map payload: 4 bytes per row (4 KiB for 1024), plus ten-row wrapper
 * temporaries and container overhead. cuDF internal workspace is additional.
 *
 * All output buffers are written on @p stream. cudf_test wrappers may enqueue work on
 * cudf::test::get_default_stream() regardless of the stream passed, so @p stream is ordered after
 * that work with an event before the final gather.
 *
 * Setup synchronizes before returning: call BEFORE arming a producer gate and
 * outside benchmark timing.
 *
 * @note The caller must set the current device to the device @p stream belongs to: the ordering
 * event is created on the current device, and recording requires it to match the device of both
 * @p stream and cudf::test::get_default_stream().
 *
 * @param rows Number of output rows
 * @param stream Stream that writes the returned table; synchronized before returning
 * @param mr Resource for the returned table's device memory
 * @return The generated table
 */
inline std::unique_ptr<cudf::table> make_mixed_table(cudf::size_type rows,
                                                     ::cuda::stream_ref stream,
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

  // cudf_test wrappers may enqueue work on cudf::test::get_default_stream() regardless of the
  // stream passed (e.g. older cudf releases). Order the gather after it without blocking the host.
  cucascade::cuda::cuda_event wrappers_done;
  wrappers_done.record(cudf::test::get_default_stream());
  wrappers_done.wait(stream);

  auto table =
    cudf::gather(samples.view(), row_map, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);

  // Gather re-derives offsets as INT32; widen two nullable STRING columns to cover INT64 offsets.
  constexpr std::array<std::size_t, 2> kWideOffsetColumns{7, 9};
  auto gathered = table->release();
  for (auto const index : kWideOffsetColumns) {
    gathered[index] = widen_string_offsets(std::move(gathered[index]), stream, mr);
  }
  table = std::make_unique<cudf::table>(std::move(gathered));

  // Complete use of temporary samples and row map before destroying them.
  stream.sync();
  return table;
}

}  // namespace test
}  // namespace cucascade
