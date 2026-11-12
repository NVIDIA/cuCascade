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

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/cudf/host_table.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/column_metadata.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace cucascade::detail {

namespace {

using blocks_allocation = memory::fixed_size_host_memory_resource::multiple_blocks_allocation;

constexpr std::size_t mask_word_bits  = 32;
constexpr std::size_t mask_word_bytes = sizeof(std::uint32_t);

/**
 * @brief Walk bytes `[offset, offset + size)` of the allocation, one piece per block.
 *
 * Offsets are byte offsets into the logical concatenation of the blocks. Calls
 * `fn(piece, piece_bytes, bytes_done)` for each piece, in order.
 */
template <typename Fn>
void for_each_block_piece(const blocks_allocation& alloc,
                          std::size_t offset,
                          std::size_t size,
                          Fn&& fn)
{
  if (size == 0) { return; }
  const std::size_t block_size = alloc.block_size();
  if (block_size == 0 || offset > alloc.size_bytes() || size > alloc.size_bytes() - offset) {
    throw std::out_of_range("read_rows_host: bytes [" + std::to_string(offset) + ", " +
                            std::to_string(offset + size) + ") are outside the host allocation (" +
                            std::to_string(alloc.size_bytes()) + " bytes)");
  }

  std::size_t block_idx = offset / block_size;
  std::size_t block_off = offset % block_size;
  std::size_t done      = 0;
  while (done < size) {
    const std::size_t piece = std::min(size - done, block_size - block_off);
    fn(alloc.at(block_idx).data() + block_off, piece, done);
    done += piece;
    block_off += piece;
    if (block_off == block_size) {
      ++block_idx;
      block_off = 0;
    }
  }
}

/// Enqueue the H2D copy of allocation bytes `[offset, offset + size)` to @p dst on @p stream.
void copy_h2d(const blocks_allocation& alloc,
              std::size_t offset,
              void* dst,
              std::size_t size,
              ::cuda::stream_ref stream)
{
  for_each_block_piece(
    alloc, offset, size, [&](const std::byte* piece, std::size_t bytes, std::size_t done) {
      CUCASCADE_CUDA_TRY(cudaMemcpyAsync(static_cast<std::uint8_t*>(dst) + done,
                                         piece,
                                         bytes,
                                         cudaMemcpyHostToDevice,
                                         stream.get()));
    });
}

/// Copy allocation bytes `[offset, offset + size)` to host memory at @p dst (CPU only).
void copy_h2h(const blocks_allocation& alloc, std::size_t offset, void* dst, std::size_t size)
{
  for_each_block_piece(
    alloc, offset, size, [&](const std::byte* piece, std::size_t bytes, std::size_t done) {
      std::memcpy(static_cast<std::uint8_t*>(dst) + done, piece, bytes);
    });
}

/// Element @p row of an offsets buffer starting at @p data_offset, read on the CPU.
std::int64_t read_offset(const blocks_allocation& alloc,
                         std::size_t data_offset,
                         std::uint8_t width,
                         std::size_t row)
{
  if (width == 4) {
    std::int32_t value{};
    copy_h2h(alloc, data_offset + row * sizeof(value), &value, sizeof(value));
    return value;
  }
  std::int64_t value{};
  copy_h2h(alloc, data_offset + row * sizeof(value), &value, sizeof(value));
  return value;
}

/// Host callback: drops the reference a blocks_keepalive took. Makes no CUDA calls.
void CUDART_CB release_blocks_reference(void* reference)
{
  delete static_cast<memory::host_table_allocation::buffers_ptr*>(reference);
}

/**
 * @brief Keeps the source's pinned blocks allocated until the copies enqueued on a stream have
 * read them.
 *
 * The H2D copies read the pinned blocks when the stream reaches them, which can be after the
 * caller has released its read lock on the batch: a HOST-tier batch records no reader event, so a
 * later writer (downgrade, upgrade, the batch being dropped) does not wait for them. Without this
 * the blocks would go back to the pool and be overwritten while the copies still read them.
 *
 * Holds a reference to the shared blocks; its destructor enqueues a host callback on the stream,
 * behind every copy enqueued while it was alive, that drops the reference. This does not
 * synchronize the stream. The host memory resource must outlive the work on the stream, since the
 * callback may be what returns the blocks to it.
 */
class blocks_keepalive {
 public:
  blocks_keepalive(const memory::host_table_allocation::buffers_ptr& blocks,
                   ::cuda::stream_ref stream)
    : _reference(std::make_unique<memory::host_table_allocation::buffers_ptr>(blocks)),
      _stream(stream)
  {
  }

  blocks_keepalive(const blocks_keepalive&)            = delete;
  blocks_keepalive& operator=(const blocks_keepalive&) = delete;

  ~blocks_keepalive()
  {
    auto* reference = _reference.release();
    // If the callback cannot be enqueued the reference is leaked on purpose: the blocks are
    // never reused, which is safe; releasing them here could free them under a pending copy.
    (void)cudaLaunchHostFunc(_stream.get(), release_blocks_reference, reference);
  }

 private:
  std::unique_ptr<memory::host_table_allocation::buffers_ptr> _reference;
  ::cuda::stream_ref _stream;
};

/// What the reader needs to know about one requested source column.
struct column_plan {
  const memory::column_metadata* meta{nullptr};
  /// Bytes per row; 0 for STRING.
  std::size_t fixed_width{0};
  bool is_string{false};
  /// STRING only: the offsets child. nullptr when the column has no children (every string is
  /// empty or null), in which case every source offset is 0.
  const memory::column_metadata* offsets{nullptr};
  /// STRING only: width in bytes of one source offset.
  std::uint8_t src_offset_width{4};
  /// STRING only: chars bytes of the delivered range.
  std::size_t chars_bytes{0};
  /// STRING only: source offset of `row_begin`.
  std::int64_t first_offset{0};
  /// True when the source mask has to be read (the range may contain nulls).
  bool has_nulls{false};
};

column_plan plan_column(const memory::column_metadata& meta, std::size_t request_index)
{
  const auto column = "read_rows_host: requested column " + std::to_string(request_index);

  column_plan plan;
  plan.meta = &meta;
  // An all-valid mask (null_count == 0) is the same as no mask.
  plan.has_nulls = meta.has_null_mask && meta.null_count != 0;

  const auto type = static_cast<cudf::type_id>(meta.type_id);
  if (type == cudf::type_id::STRING) {
    plan.is_string = true;
    if (!meta.children.empty()) {
      plan.offsets            = &meta.children.front();
      const auto offsets_type = static_cast<cudf::type_id>(plan.offsets->type_id);
      if (offsets_type == cudf::type_id::INT32) {
        plan.src_offset_width = 4;
      } else if (offsets_type == cudf::type_id::INT64) {
        plan.src_offset_width = 8;
      } else {
        throw row_range_unsupported_error(column + ": STRING offsets have type id " +
                                          std::to_string(plan.offsets->type_id) +
                                          ", expected INT32 or INT64");
      }
      if (!plan.offsets->has_data && meta.num_rows > 0) {
        throw std::logic_error(column + ": STRING offsets child has no data buffer");
      }
    }
    return plan;
  }

  const cudf::data_type data_type{type};
  if (!cudf::is_fixed_width(data_type)) {
    throw row_range_unsupported_error(column + " has an unsupported type (id " +
                                      std::to_string(meta.type_id) + ")");
  }
  plan.fixed_width = cudf::size_of(data_type);
  if (!meta.has_data && meta.num_rows > 0) {
    throw std::logic_error(column + ": fixed-width column has no data buffer");
  }
  return plan;
}

}  // namespace

row_range_result read_rows_host(const idata_representation& source,
                                const row_range_request& request,
                                std::span<column_destination> dst,
                                ::cuda::stream_ref stream,
                                memory::reservation* scratch,
                                std::unique_ptr<row_range_reader_state>* /*state*/)
{
  const auto* host = dynamic_cast<const host_data_representation*>(&source);
  if (host == nullptr) {
    throw std::invalid_argument("read_rows_host: source is not a host_data_representation");
  }
  const auto& table = host->get_host_table();
  if (table == nullptr) {
    throw std::invalid_argument("read_rows_host: source has no host table allocation");
  }
  const auto& columns = table->columns;
  const std::size_t num_rows =
    columns.empty() ? 0 : static_cast<std::size_t>(columns.front().num_rows);
  const std::size_t num_requested = request.columns.size();

  std::vector<std::int32_t> type_ids;
  type_ids.reserve(num_requested);
  for (const auto index : request.columns) {
    if (index >= columns.size()) {
      throw std::out_of_range("read_rows_host: column index " + std::to_string(index) +
                              " is past the source's " + std::to_string(columns.size()) +
                              " columns");
    }
    type_ids.push_back(columns[index].type_id);
  }
  validate_request(
    request, std::span<const column_destination>{dst.data(), dst.size()}, type_ids, num_rows);

  std::vector<column_plan> plans;
  plans.reserve(num_requested);
  for (std::size_t i = 0; i < num_requested; ++i) {
    plans.push_back(plan_column(columns[request.columns[i]], i));
    if (plans.back().has_nulls && dst[i].null_mask.data == nullptr) {
      throw row_range_unsupported_error("read_rows_host: requested column " + std::to_string(i) +
                                        " has " + std::to_string(plans.back().meta->null_count) +
                                        " nulls but its destination has no null-mask slot");
    }
  }

  const blocks_allocation* alloc = table->allocation.get();
  const std::size_t row_begin    = request.row_begin;
  if (alloc == nullptr && row_begin < num_rows && request.max_rows > 0 && num_requested > 0) {
    throw std::logic_error("read_rows_host: source has rows but no host blocks");
  }

  // Fitting: STRING offsets are read on the CPU straight from the host blocks, so no device
  // work (and no stream synchronization) is needed to size the range.
  std::vector<fit_column> fit_columns(num_requested);
  for (std::size_t i = 0; i < num_requested; ++i) {
    const auto& plan           = plans[i];
    fit_columns[i].fixed_width = plan.fixed_width;
    fit_columns[i].is_string   = plan.is_string;
    fit_columns[i].dst         = &dst[i];
    if (plan.is_string) {
      if (plan.offsets != nullptr) {
        fit_columns[i].offset_at =
          [alloc, data_offset = plan.offsets->data_offset, width = plan.src_offset_width](
            std::size_t row) { return read_offset(*alloc, data_offset, width, row); };
      } else {
        fit_columns[i].offset_at = [](std::size_t) { return std::int64_t{0}; };
      }
    }
  }
  const std::size_t k = fit_rows(fit_columns, row_begin, request.max_rows, num_rows);

  row_range_result result;
  result.rows_written = k;
  result.data_bytes_written.assign(num_requested, 0);
  result.source_exhausted = row_begin + k == num_rows;
  if (k == 0) { return result; }

  // Size every column before enqueuing anything, so a failure leaves the destination untouched.
  for (std::size_t i = 0; i < num_requested; ++i) {
    auto& plan = plans[i];
    if (!plan.is_string) {
      result.data_bytes_written[i] = k * plan.fixed_width;
      continue;
    }
    plan.first_offset = fit_columns[i].offset_at(row_begin);
    const auto bytes  = fit_columns[i].offset_at(row_begin + k) - plan.first_offset;
    if (bytes < 0) {
      throw std::logic_error("read_rows_host: requested column " + std::to_string(i) +
                             " has decreasing STRING offsets");
    }
    plan.chars_bytes = static_cast<std::size_t>(bytes);
    if (plan.chars_bytes > 0 && !plan.meta->has_data) {
      throw std::logic_error("read_rows_host: requested column " + std::to_string(i) +
                             " has STRING chars but no data buffer");
    }
    if (dst[i].offset_width == 4 &&
        dst[i].dst_byte_offset + plan.chars_bytes >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
      throw std::overflow_error("read_rows_host: requested column " + std::to_string(i) +
                                ": rebased offsets do not fit 4-byte destination offsets");
    }
    result.data_bytes_written[i] = plan.chars_bytes;
  }

  // Device temporaries. They are freed when this function returns, before the work reading them
  // has run: rmm deallocation is stream-ordered on `stream`, so the memory is not reused before
  // that work is done.
  std::vector<rmm::device_buffer> temporaries;

  // The copies below read the source's pinned blocks when `stream` reaches them. Keep the blocks
  // allocated until then, whatever happens to the source representation in the meantime.
  const blocks_keepalive keepalive{table->allocation, stream};

  for (std::size_t i = 0; i < num_requested; ++i) {
    const auto& plan = plans[i];
    const auto& meta = *plan.meta;
    const auto& d    = dst[i];

    if (plan.is_string) {
      if (plan.chars_bytes > 0) {
        copy_h2d(*alloc,
                 meta.data_offset + static_cast<std::size_t>(plan.first_offset),
                 static_cast<std::uint8_t*>(d.data.data) + d.dst_byte_offset,
                 plan.chars_bytes,
                 stream);
      }
      // Raw source offsets go to a device temporary; the kernel rebases them and converts width.
      const std::size_t offsets_bytes = (k + 1) * plan.src_offset_width;
      auto& raw_offsets =
        temporaries.emplace_back(allocate_scratch(scratch, offsets_bytes, stream));
      if (plan.offsets != nullptr) {
        copy_h2d(*alloc,
                 plan.offsets->data_offset + row_begin * plan.src_offset_width,
                 raw_offsets.data(),
                 offsets_bytes,
                 stream);
      } else {
        CUCASCADE_CUDA_TRY(cudaMemsetAsync(raw_offsets.data(), 0, offsets_bytes, stream.get()));
      }
      rebase_offsets(static_cast<std::uint8_t*>(d.offsets.data) + d.dst_row_offset * d.offset_width,
                     d.offset_width,
                     raw_offsets.data(),
                     plan.src_offset_width,
                     k + 1,
                     static_cast<std::int64_t>(d.dst_byte_offset),
                     stream);
    } else {
      copy_h2d(*alloc,
               meta.data_offset + row_begin * plan.fixed_width,
               static_cast<std::uint8_t*>(d.data.data) + d.dst_row_offset * plan.fixed_width,
               k * plan.fixed_width,
               stream);
    }

    if (d.null_mask.data == nullptr) { continue; }
    auto* dst_mask = static_cast<std::uint32_t*>(d.null_mask.data);
    if (!plan.has_nulls) {
      copy_bits(dst_mask, d.dst_row_offset, nullptr, 0, k, stream);
      continue;
    }
    // Source words covering bits [row_begin, row_begin + k).
    const std::size_t first_word = row_begin / mask_word_bits;
    const std::size_t src_bit    = row_begin % mask_word_bits;
    const std::size_t mask_bytes =
      (src_bit + k + mask_word_bits - 1) / mask_word_bits * mask_word_bytes;
    auto& src_words = temporaries.emplace_back(allocate_scratch(scratch, mask_bytes, stream));
    copy_h2d(*alloc,
             meta.null_mask_offset + first_word * mask_word_bytes,
             src_words.data(),
             mask_bytes,
             stream);
    copy_bits(dst_mask,
              d.dst_row_offset,
              static_cast<const std::uint32_t*>(src_words.data()),
              src_bit,
              k,
              stream);
  }

  return result;
}

}  // namespace cucascade::detail
