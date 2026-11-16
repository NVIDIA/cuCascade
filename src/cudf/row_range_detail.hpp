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

#pragma once

#include <cucascade/cuda/stream.hpp>
#include <cucascade/data/common.hpp>
#include <cucascade/data/row_range_read.hpp>
#include <cucascade/memory/memory_reservation.hpp>

#include <rmm/device_buffer.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>

/**
 * @file row_range_detail.hpp
 * @brief Internal declarations shared by the built-in row-range readers.
 *
 * The reader contract these functions implement is in cucascade/data/row_range_read.hpp.
 */

namespace cucascade::io {
class ioctx;
}  // namespace cucascade::io

namespace cucascade::detail {

// ---- per-representation readers ----

/**
 * @brief Row-range reader for gpu_table_representation (row_range_reader_gpu.cpp).
 *
 * Implements the reader contract of row_range_read.hpp. May synchronize @p stream only for the
 * device-to-host read of STRING offsets, once per offsets chunk. The source must live on the
 * device of @p stream, else row_range_unsupported_error.
 */
row_range_result read_rows_gpu(const idata_representation& source,
                               const row_range_request& request,
                               std::span<column_destination> dst,
                               ::cuda::stream_ref stream,
                               memory::reservation* scratch,
                               std::unique_ptr<row_range_reader_state>* state);

/**
 * @brief Row-range reader for host_data_representation (row_range_reader_host.cpp).
 *
 * Implements the reader contract of row_range_read.hpp.
 */
row_range_result read_rows_host(const idata_representation& source,
                                const row_range_request& request,
                                std::span<column_destination> dst,
                                ::cuda::stream_ref stream,
                                memory::reservation* scratch,
                                std::unique_ptr<row_range_reader_state>* state);

/**
 * @brief Row-range reader for disk_data_representation (row_range_reader_disk.cpp).
 *
 * Implements the reader contract of row_range_read.hpp. May synchronize @p stream only for the
 * host-side offset read.
 *
 * @param io_context io context to read with; nullptr means the io context of the source's DISK
 *        memory space (see resolve_disk_io_context() in disk_io.hpp).
 */
row_range_result read_rows_disk(const idata_representation& source,
                                const row_range_request& request,
                                std::span<column_destination> dst,
                                ::cuda::stream_ref stream,
                                memory::reservation* scratch,
                                std::unique_ptr<row_range_reader_state>* state,
                                std::shared_ptr<io::ioctx> io_context);

// ---- device helpers (row_range_copy_kernels.cu) ----

/**
 * @brief Copy a run of validity bits between two bitmasks at arbitrary bit positions.
 *
 * Copy `num_bits` validity bits from `src_mask` (device-accessible 32-bit words) starting at
 * bit `src_bit` into `dst_mask` starting at bit `dst_bit`. src_mask == nullptr => set the bits
 * to 1. Preserves every dst bit outside [dst_bit, dst_bit + num_bits). num_bits == 0 => no-op.
 */
void copy_bits(std::uint32_t* dst_mask,
               std::size_t dst_bit,
               const std::uint32_t* src_mask,
               std::size_t src_bit,
               std::size_t num_bits,
               ::cuda::stream_ref stream);

/**
 * @brief Write rebased offsets, converting the offset width if needed.
 *
 * dst[i] = dst_base + (src[i] - src[0]) for i in [0, count). Widths are 4 or 8 bytes (signed).
 * src must be device-accessible. The caller guarantees the values fit; a
 * 4-byte dst receiving a value > INT32_MAX is a caller bug (debug assert only).
 */
void rebase_offsets(void* dst,
                    std::uint8_t dst_width,
                    const void* src,
                    std::uint8_t src_width,
                    std::size_t count,
                    std::int64_t dst_base,
                    ::cuda::stream_ref stream);

/**
 * @brief Allocate a device temporary.
 *
 * Uses scratch->get_memory_resource() when scratch != nullptr, else
 * rmm::mr::get_current_device_resource_ref().
 */
rmm::device_buffer allocate_scratch(memory::reservation* scratch,
                                    std::size_t bytes,
                                    ::cuda::stream_ref stream);

// ---- host-side offset decoding ----

/**
 * @brief Decode offset @p index of a host array of 4- or 8-byte signed offsets.
 *
 * @param bytes The raw offsets, `width` bytes each.
 * @param width Bytes per offset: 4 or 8.
 * @param index Which offset of @p bytes to decode.
 * @throws std::invalid_argument if @p width is neither 4 nor 8
 * @throws std::out_of_range if offset @p index is not fully inside @p bytes
 */
inline std::int64_t load_offset(std::span<const std::byte> bytes,
                                std::uint8_t width,
                                std::size_t index)
{
  if (width != 4 && width != 8) {
    throw std::invalid_argument("load_offset: width must be 4 or 8");
  }
  if (index >= bytes.size() / width) {
    throw std::out_of_range("load_offset: offset index past the end of the host offsets");
  }
  const std::byte* p = bytes.data() + index * width;
  if (width == 4) {
    std::int32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
  }
  std::int64_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

// ---- CPU budget fitting (row_range_fit.cpp) ----

/**
 * @brief Description of one requested source column, as seen by the fitter.
 */
struct fit_column {
  /// Bytes per row; 0 for STRING.
  std::size_t fixed_width{0};
  /// True for a STRING column.
  bool is_string{false};
  /// STRING only: callable returning the source offset (bytes) of absolute source row r, for
  /// r in [row_begin, row_begin + max_rows]. Must be CPU-callable and cheap (host memory).
  std::function<std::int64_t(std::size_t)> offset_at;
  /// Destination of this column; supplies the slot capacities and append positions.
  const column_destination* dst{nullptr};
};

/**
 * @brief Number of rows that fit every destination slot.
 *
 * Largest k satisfying reader-contract rule 1 for all columns (k <= max_rows,
 * k <= num_rows - row_begin). STRING columns: binary search over offset_at (O(log max_rows) calls
 * per column).
 */
std::size_t fit_rows(std::span<const fit_column> columns,
                     std::size_t row_begin,
                     std::size_t max_rows,
                     std::size_t num_rows);

/**
 * @brief Shared validation for reader-contract rules 2 and 5 (row_range_fit.cpp).
 *
 * Throws std::out_of_range / row_range_unsupported_error / std::invalid_argument.
 * `source_type_ids` are the cudf type ids of the requested columns.
 */
void validate_request(const row_range_request& request,
                      std::span<const column_destination> dst,
                      std::span<const std::int32_t> source_type_ids,
                      std::size_t num_rows);

}  // namespace cucascade::detail
