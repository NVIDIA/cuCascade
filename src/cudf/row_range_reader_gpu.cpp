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

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/error.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace cucascade::detail {

namespace {

/// Source offsets of one STRING column for rows [row_begin, row_begin + k_max], on the host.
struct string_offsets {
  /// Device pointer to the offset of source row 0 of the (possibly sliced) column; nullptr when
  /// the column has no offsets child (an empty column).
  const std::uint8_t* device{nullptr};
  /// Width in bytes of one source offset: 4 (INT32) or 8 (INT64).
  std::uint8_t width{4};
  /// Host copy of the `k_max + 1` offsets starting at row `row_begin`.
  std::vector<std::uint8_t> host;
  /// First source row held in `host`.
  std::size_t row_begin{0};

  /// Source offset (bytes) of absolute source row @p row.
  [[nodiscard]] std::int64_t at(std::size_t row) const
  {
    const std::size_t pos = (row - row_begin) * width;
    if (pos + width > host.size()) { return 0; }
    if (width == 4) {
      std::int32_t value{};
      std::memcpy(&value, host.data() + pos, sizeof(value));
      return value;
    }
    std::int64_t value{};
    std::memcpy(&value, host.data() + pos, sizeof(value));
    return value;
  }
};

/// Enqueue the D2H copy of the offsets of rows [row_begin, row_begin + k_max] of @p col.
string_offsets fetch_string_offsets(const cudf::column_view& col,
                                    std::size_t row_begin,
                                    std::size_t k_max,
                                    ::cuda::stream_ref stream)
{
  string_offsets result;
  result.row_begin = row_begin;
  // An empty strings column may have no children at all.
  if (col.num_children() == 0) { return result; }

  const cudf::strings_column_view strings{col};
  const auto offsets = strings.offsets();
  result.width       = static_cast<std::uint8_t>(cudf::size_of(offsets.type()));
  // data<>() applies the child's own offset; the parent's offset() is in rows of the parent.
  result.device =
    offsets.data<std::uint8_t>() + static_cast<std::size_t>(col.offset()) * result.width;
  if (k_max == 0) { return result; }

  result.host.resize((k_max + 1) * result.width);
  CUCASCADE_CUDA_TRY(cudaMemcpyAsync(result.host.data(),
                                     result.device + row_begin * result.width,
                                     result.host.size(),
                                     cudaMemcpyDeviceToHost,
                                     stream.get()));
  return result;
}

}  // namespace

row_range_result read_rows_gpu(const idata_representation& source,
                               const row_range_request& request,
                               std::span<column_destination> dst,
                               ::cuda::stream_ref stream,
                               memory::reservation* /*scratch*/,
                               std::unique_ptr<row_range_reader_state>* /*state*/)
{
  const auto& rep      = dynamic_cast<const gpu_table_representation&>(source);
  const auto view      = rep.get_table_view();
  const auto num_rows  = static_cast<std::size_t>(view.num_rows());
  const auto num_cols  = static_cast<std::size_t>(view.num_columns());
  const auto requested = request.columns.size();

  std::vector<cudf::column_view> cols;
  std::vector<std::int32_t> type_ids;
  cols.reserve(requested);
  type_ids.reserve(requested);
  for (const auto index : request.columns) {
    if (index >= num_cols) {
      throw std::out_of_range("row-range read: column index " + std::to_string(index) +
                              " is past the source's " + std::to_string(num_cols) + " columns");
    }
    cols.push_back(view.column(static_cast<cudf::size_type>(index)));
    type_ids.push_back(static_cast<std::int32_t>(cols.back().type().id()));
  }

  validate_request(request, dst, type_ids, num_rows);

  for (std::size_t i = 0; i < requested; ++i) {
    if (cols[i].type().id() != cudf::type_id::STRING && !cudf::is_fixed_width(cols[i].type())) {
      throw row_range_unsupported_error("row-range read: requested column " + std::to_string(i) +
                                        " has an unsupported type (id " +
                                        std::to_string(type_ids[i]) + ")");
    }
    if (cols[i].has_nulls() && dst[i].null_mask.data == nullptr) {
      throw row_range_unsupported_error("row-range read: requested column " + std::to_string(i) +
                                        " has nulls but its destination has no null_mask slot");
    }
  }

  const std::size_t k_max = std::min(request.max_rows, num_rows - request.row_begin);

  // STRING offsets come to the host once, so that fitting needs no further device round trips.
  std::vector<string_offsets> offsets(requested);
  bool any_string = false;
  for (std::size_t i = 0; i < requested; ++i) {
    if (cols[i].type().id() != cudf::type_id::STRING) { continue; }
    offsets[i] = fetch_string_offsets(cols[i], request.row_begin, k_max, stream);
    any_string = any_string || !offsets[i].host.empty();
  }
  if (any_string) { stream.sync(); }

  std::vector<fit_column> fit(requested);
  for (std::size_t i = 0; i < requested; ++i) {
    fit[i].dst = &dst[i];
    if (cols[i].type().id() == cudf::type_id::STRING) {
      fit[i].is_string = true;
      fit[i].offset_at = [&off = offsets[i]](std::size_t row) { return off.at(row); };
    } else {
      fit[i].fixed_width = cudf::size_of(cols[i].type());
    }
  }
  const std::size_t k = fit_rows(fit, request.row_begin, request.max_rows, num_rows);

  row_range_result result;
  result.rows_written = k;
  result.data_bytes_written.assign(requested, 0);
  result.source_exhausted = request.row_begin + k == num_rows;
  if (k == 0) { return result; }

  for (std::size_t i = 0; i < requested; ++i) {
    const auto& col      = cols[i];
    const auto& out      = dst[i];
    const auto first_row = static_cast<std::size_t>(col.offset()) + request.row_begin;

    if (fit[i].is_string) {
      const auto& off        = offsets[i];
      const auto chars_begin = off.at(request.row_begin);
      const auto chars_bytes =
        static_cast<std::size_t>(off.at(request.row_begin + k) - chars_begin);
      if (chars_bytes > 0) {
        CUCASCADE_CUDA_TRY(
          cudaMemcpyAsync(static_cast<std::uint8_t*>(out.data.data) + out.dst_byte_offset,
                          col.head<std::uint8_t>() + chars_begin,
                          chars_bytes,
                          cudaMemcpyDeviceToDevice,
                          stream.get()));
      }
      rebase_offsets(
        static_cast<std::uint8_t*>(out.offsets.data) + out.dst_row_offset * out.offset_width,
        out.offset_width,
        off.device + request.row_begin * off.width,
        off.width,
        k + 1,
        static_cast<std::int64_t>(out.dst_byte_offset),
        stream);
      result.data_bytes_written[i] = chars_bytes;
    } else {
      const std::size_t width = fit[i].fixed_width;
      CUCASCADE_CUDA_TRY(
        cudaMemcpyAsync(static_cast<std::uint8_t*>(out.data.data) + out.dst_row_offset * width,
                        col.head<std::uint8_t>() + first_row * width,
                        k * width,
                        cudaMemcpyDeviceToDevice,
                        stream.get()));
      result.data_bytes_written[i] = k * width;
    }

    if (out.null_mask.data != nullptr) {
      // An all-valid range writes 1s without reading a source mask (rule 4).
      copy_bits(static_cast<std::uint32_t*>(out.null_mask.data),
                out.dst_row_offset,
                col.has_nulls() ? col.null_mask() : nullptr,
                first_row,
                k,
                stream);
    }
  }
  return result;
}

}  // namespace cucascade::detail
