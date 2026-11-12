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

/// Rows whose offsets the first device-to-host read of a STRING column covers.
constexpr std::size_t kFirstOffsetChunkRows = 1024;

/// Source offsets of one STRING column, read to the host on demand from row `row_begin` on.
struct string_offsets {
  /// Device pointer to the offset of source row 0 of the (possibly sliced) column; nullptr when
  /// the column has no offsets child (an empty column).
  const std::uint8_t* device{nullptr};
  /// Width in bytes of one source offset: 4 (INT32) or 8 (INT64).
  std::uint8_t width{4};
  /// Host copy of the offsets of rows [row_begin, row_begin + rows()].
  std::vector<std::uint8_t> host;
  /// First source row held in `host`.
  std::size_t row_begin{0};

  /// Rows whose begin and end offsets are both on the host.
  [[nodiscard]] std::size_t rows() const { return host.empty() ? 0 : host.size() / width - 1; }

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

  /// Extend the host copy to the offsets of rows [row_begin, row_begin + rows]; synchronizes
  /// @p stream.
  void fetch_up_to(std::size_t rows, ::cuda::stream_ref stream)
  {
    const std::size_t have = host.size();
    const std::size_t want = (rows + 1) * width;
    if (device == nullptr || want <= have) { return; }
    host.resize(want);
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(host.data() + have,
                                       device + row_begin * width + have,
                                       want - have,
                                       cudaMemcpyDeviceToHost,
                                       stream.get()));
    stream.sync();
  }
};

/// Locate the offsets of @p col on the device; nothing is read yet.
string_offsets locate_string_offsets(const cudf::column_view& col, std::size_t row_begin)
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
  return result;
}

/**
 * @brief Read just enough offsets of a STRING column to decide how many of @p k rows fit @p dst.
 *
 * Reads in geometrically growing chunks and stops at the first chunk whose chars overflow the
 * data slot, so the bytes copied to the host are proportional to the rows that fit, not to @p k.
 *
 * @return Rows covered by the host copy: @p k if all of them fit, else an upper bound of the rows
 *         that fit.
 */
std::size_t fetch_offsets_for_fit(string_offsets& off,
                                  const column_destination& dst,
                                  std::size_t k,
                                  ::cuda::stream_ref stream)
{
  if (off.device == nullptr) { return k; }
  const std::size_t avail = dst.data.capacity_bytes > dst.dst_byte_offset
                              ? dst.data.capacity_bytes - dst.dst_byte_offset
                              : 0;
  std::size_t rows        = 0;
  while (rows < k) {
    rows = std::min(k, std::max(kFirstOffsetChunkRows, rows * 2));
    off.fetch_up_to(rows, stream);
    const auto bytes = off.at(off.row_begin + rows) - off.at(off.row_begin);
    if (bytes > 0 && static_cast<std::uint64_t>(bytes) > avail) { break; }
  }
  return rows;
}

/// Refuse a source that lives on another GPU than @p stream: the copies and kernels enqueued on
/// @p stream dereference the source's device pointers, which faults without peer access.
void require_same_device(const idata_representation& source, ::cuda::stream_ref stream)
{
  int stream_device = -1;
  CUCASCADE_CUDA_TRY(::cudaStreamGetDevice(stream.get(), &stream_device));
  if (stream_device != source.get_device_id()) {
    throw row_range_unsupported_error(
      "row-range read: the source table lives on CUDA device " +
      std::to_string(source.get_device_id()) + " but the stream belongs to device " +
      std::to_string(stream_device) + "; convert the source to the destination device first");
  }
}

}  // namespace

row_range_result read_rows_gpu(const idata_representation& source,
                               const row_range_request& request,
                               std::span<column_destination> dst,
                               ::cuda::stream_ref stream,
                               memory::reservation* /*scratch*/,
                               std::unique_ptr<row_range_reader_state>* /*state*/)
{
  const auto& rep = dynamic_cast<const gpu_table_representation&>(source);
  require_same_device(source, stream);
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

  // Pass 1: the closed-form terms of rule 1 (fixed-width data, offsets slot, null mask) bound k
  // without reading any source offsets.
  std::vector<string_offsets> offsets(requested);
  std::vector<fit_column> fit(requested);
  for (std::size_t i = 0; i < requested; ++i) {
    fit[i].dst = &dst[i];
    if (cols[i].type().id() == cudf::type_id::STRING) {
      offsets[i]       = locate_string_offsets(cols[i], request.row_begin);
      fit[i].is_string = true;
      fit[i].offset_at = [](std::size_t) { return std::int64_t{0}; };
    } else {
      fit[i].fixed_width = cudf::size_of(cols[i].type());
    }
  }
  std::size_t k_bound = fit_rows(fit, request.row_begin, request.max_rows, num_rows);

  // Pass 2: bring to the host only the offsets needed to fit the chars slots, then fit again.
  for (std::size_t i = 0; i < requested && k_bound > 0; ++i) {
    if (!fit[i].is_string) { continue; }
    k_bound          = fetch_offsets_for_fit(offsets[i], dst[i], k_bound, stream);
    fit[i].offset_at = [&off = offsets[i]](std::size_t row) { return off.at(row); };
  }
  const std::size_t k = k_bound == 0 ? 0 : fit_rows(fit, request.row_begin, k_bound, num_rows);

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
