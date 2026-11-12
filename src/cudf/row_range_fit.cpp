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

#include "row_range_detail.hpp"

#include <cudf/types.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace cucascade::detail {

namespace {

/// Rows that still fit a slot holding `slot_rows` rows when appending at `dst_row_offset`.
std::size_t rows_left(std::size_t slot_rows, std::size_t dst_row_offset)
{
  return slot_rows > dst_row_offset ? slot_rows - dst_row_offset : 0;
}

/// Largest k for the closed-form terms of rule 1 (fixed-width data, STRING offsets, null mask).
std::size_t fit_closed_form(const fit_column& col, std::size_t k)
{
  const auto& dst = *col.dst;
  if (dst.null_mask.data != nullptr) {
    // ceil((dst_row_offset + k) / 32) * 4 <= capacity: whole 32-bit words only.
    const std::size_t words = dst.null_mask.capacity_bytes / 4;
    const std::size_t bits  = words > std::numeric_limits<std::size_t>::max() / 32
                                ? std::numeric_limits<std::size_t>::max()
                                : words * 32;
    k                       = std::min(k, rows_left(bits, dst.dst_row_offset));
  }
  if (col.is_string) {
    // (dst_row_offset + k + 1) * offset_width <= capacity: one offset more than rows.
    const std::size_t slots =
      dst.offset_width == 0 ? 0 : dst.offsets.capacity_bytes / dst.offset_width;
    k = std::min(k, rows_left(slots, dst.dst_row_offset + 1));
  } else if (col.fixed_width > 0) {
    k = std::min(k, rows_left(dst.data.capacity_bytes / col.fixed_width, dst.dst_row_offset));
  }
  return k;
}

/// Largest k' <= k whose chars fit the data slot of a STRING column (binary search on offset_at).
std::size_t fit_chars(const fit_column& col, std::size_t row_begin, std::size_t k)
{
  const auto& dst = *col.dst;
  if (dst.dst_byte_offset > dst.data.capacity_bytes) { return 0; }
  const std::size_t avail = dst.data.capacity_bytes - dst.dst_byte_offset;
  const auto base         = col.offset_at(row_begin);
  auto fits               = [&](std::size_t rows) {
    const auto bytes = col.offset_at(row_begin + rows) - base;
    return bytes <= 0 || static_cast<std::uint64_t>(bytes) <= avail;
  };
  if (fits(k)) { return k; }
  // Invariant: lo rows fit (0 rows always do), hi rows do not.
  std::size_t lo = 0;
  std::size_t hi = k;
  while (hi - lo > 1) {
    const std::size_t mid = lo + (hi - lo) / 2;
    if (fits(mid)) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

}  // namespace

std::size_t fit_rows(std::span<const fit_column> columns,
                     std::size_t row_begin,
                     std::size_t max_rows,
                     std::size_t num_rows)
{
  if (row_begin >= num_rows) { return 0; }
  std::size_t k = std::min(max_rows, num_rows - row_begin);

  // Closed-form terms first, so the binary searches below run over the smallest range.
  for (const auto& col : columns) {
    if (col.dst == nullptr) { throw std::invalid_argument("fit_rows: column has no destination"); }
    k = fit_closed_form(col, k);
  }
  for (const auto& col : columns) {
    if (k == 0) { break; }
    if (col.is_string) { k = fit_chars(col, row_begin, k); }
  }
  return k;
}

void validate_request(const row_range_request& request,
                      std::span<const column_destination> dst,
                      std::span<const std::int32_t> source_type_ids,
                      std::size_t num_rows)
{
  if (request.row_begin > num_rows) {
    throw std::out_of_range("row-range read: row_begin " + std::to_string(request.row_begin) +
                            " is past the source's " + std::to_string(num_rows) + " rows");
  }
  if (dst.size() != request.columns.size()) {
    throw std::invalid_argument("row-range read: " + std::to_string(dst.size()) +
                                " destinations for " + std::to_string(request.columns.size()) +
                                " requested columns");
  }
  if (source_type_ids.size() != request.columns.size()) {
    throw std::invalid_argument("row-range read: " + std::to_string(source_type_ids.size()) +
                                " source type ids for " + std::to_string(request.columns.size()) +
                                " requested columns");
  }

  for (std::size_t i = 0; i < dst.size(); ++i) {
    const auto column = "row-range read: requested column " + std::to_string(i);
    const auto type   = static_cast<cudf::type_id>(source_type_ids[i]);
    if (type == cudf::type_id::LIST || type == cudf::type_id::STRUCT ||
        type == cudf::type_id::DICTIONARY32) {
      throw row_range_unsupported_error(column + " has an unsupported type (id " +
                                        std::to_string(source_type_ids[i]) + ")");
    }
    if (dst[i].type_id != source_type_ids[i]) {
      throw std::invalid_argument(column + ": destination type id " +
                                  std::to_string(dst[i].type_id) + " != source type id " +
                                  std::to_string(source_type_ids[i]));
    }
    if (type == cudf::type_id::STRING) {
      if (dst[i].offsets.data == nullptr) {
        throw std::invalid_argument(column + ": STRING destination has no offsets slot");
      }
      if (dst[i].offset_width != 4 && dst[i].offset_width != 8) {
        throw std::invalid_argument(column + ": STRING destination offset_width " +
                                    std::to_string(dst[i].offset_width) + " is not 4 or 8");
      }
    } else if (dst[i].data.data == nullptr && request.max_rows > 0) {
      throw std::invalid_argument(column + ": fixed-width destination has no data slot");
    }
  }
}

}  // namespace cucascade::detail
