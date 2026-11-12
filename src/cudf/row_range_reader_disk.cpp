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

#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/column_metadata.hpp>
#include <cucascade/memory/disk_table.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace cucascade::detail {

namespace {

/// Buffers start on this boundary in the file (recompute_file_offsets). Device reads are issued
/// only at file offsets that are a multiple of it, because an O_DIRECT / GDS backend cannot read
/// from an arbitrary file offset.
constexpr std::size_t disk_file_alignment = 4096;

/// A STRING column's source offsets for a run of rows, in host memory.
struct host_offsets {
  std::vector<std::byte> bytes;  ///< The offsets, `width` bytes each
  std::uint8_t width{4};         ///< 4 (INT32) or 8 (INT64)

  [[nodiscard]] std::int64_t at(std::size_t index) const
  {
    if (width == 8) {
      std::int64_t value{};
      std::memcpy(&value, bytes.data() + index * 8, sizeof(value));
      return value;
    }
    std::int32_t value{};
    std::memcpy(&value, bytes.data() + index * 4, sizeof(value));
    return value;
  }
};

/// Reads byte ranges of one batch file into device memory, hiding the alignment rule above.
class range_reader {
 public:
  range_reader(idisk_io_backend& backend,
               const std::filesystem::path& path,
               memory::reservation* scratch,
               ::cuda::stream_ref stream)
    : _backend(backend), _path(path), _scratch(scratch), _stream(stream)
  {
  }

  /// Read file bytes [file_offset, file_offset + size) into `dst`.
  void read_into(void* dst, std::size_t size, std::size_t file_offset)
  {
    if (size == 0) { return; }
    auto* out                = static_cast<std::byte*>(dst);
    const std::size_t prefix = file_offset % disk_file_alignment;
    if (prefix != 0) {
      // The first (partial) block goes through scratch; the rest starts on a block boundary and is
      // read straight into the destination.
      const std::size_t head = std::min(size, disk_file_alignment - prefix);
      const auto* block      = read_aligned_superset(head, file_offset);
      CUCASCADE_CUDA_TRY(
        cudaMemcpyAsync(out, block, head, cudaMemcpyDeviceToDevice, _stream.get()));
      out += head;
      size -= head;
      file_offset += head;
    }
    if (size > 0) { _backend.read(_path, out, size, file_offset, _stream); }
  }

  /// Read file bytes [file_offset, file_offset + size) into a scratch buffer owned by this reader
  /// and return a pointer to the first requested byte. The pointer is as aligned as `file_offset`.
  [[nodiscard]] const std::byte* read_aligned_superset(std::size_t size, std::size_t file_offset)
  {
    const std::size_t prefix = file_offset % disk_file_alignment;
    auto& buffer             = allocate(prefix + size);
    _backend.read(_path, buffer.data(), prefix + size, file_offset - prefix, _stream);
    return static_cast<const std::byte*>(buffer.data()) + prefix;
  }

  /// Allocate a temporary that lives as long as this reader.
  [[nodiscard]] rmm::device_buffer& allocate(std::size_t bytes)
  {
    return _temporaries.emplace_back(allocate_scratch(_scratch, bytes, _stream));
  }

 private:
  idisk_io_backend& _backend;
  const std::filesystem::path& _path;
  memory::reservation* _scratch;
  ::cuda::stream_ref _stream;
  // Freed on `_stream` (rmm::device_buffer remembers its stream), after the work that uses them.
  std::vector<rmm::device_buffer> _temporaries;
};

[[nodiscard]] bool is_string(const memory::column_metadata& meta)
{
  return static_cast<cudf::type_id>(meta.type_id) == cudf::type_id::STRING;
}

[[nodiscard]] bool may_have_nulls(const memory::column_metadata& meta)
{
  // Anything but 0 counts, so a domain-defined "unknown" null count is treated as "has nulls".
  return meta.has_null_mask && meta.null_mask_size > 0 && meta.null_count != 0;
}

[[noreturn]] void fail_layout(std::size_t column, const std::string& what)
{
  throw std::runtime_error("row-range read (disk): column " + std::to_string(column) + ": " + what);
}

}  // namespace

row_range_result read_rows_disk(const idata_representation& source,
                                const row_range_request& request,
                                std::span<column_destination> dst,
                                ::cuda::stream_ref stream,
                                memory::reservation* scratch,
                                std::unique_ptr<row_range_reader_state>* /*state*/,
                                std::shared_ptr<idisk_io_backend> io_backend)
{
  const auto* disk = dynamic_cast<const disk_data_representation*>(&source);
  if (disk == nullptr) {
    throw std::invalid_argument("row-range read (disk): source is not a disk_data_representation");
  }
  idisk_io_backend& backend =
    io_backend != nullptr ? *io_backend : source.get_memory_space().get_io_backend();
  const auto& table = disk->get_disk_table();
  const std::filesystem::path path{table.file_path};

  const std::size_t num_columns = request.columns.size();
  const std::size_t num_rows =
    table.columns.empty() ? 0 : static_cast<std::size_t>(table.columns.front().num_rows);

  // ---- validation (rules 2, 4, 5) ----
  std::vector<const memory::column_metadata*> metas(num_columns);
  std::vector<std::int32_t> type_ids(num_columns);
  for (std::size_t i = 0; i < num_columns; ++i) {
    if (request.columns[i] >= table.columns.size()) {
      throw std::out_of_range("row-range read (disk): requested column index " +
                              std::to_string(request.columns[i]) + " is past the source's " +
                              std::to_string(table.columns.size()) + " columns");
    }
    metas[i]    = &table.columns[request.columns[i]];
    type_ids[i] = metas[i]->type_id;
  }
  validate_request(request, std::span<const column_destination>{dst}, type_ids, num_rows);
  for (std::size_t i = 0; i < num_columns; ++i) {
    if (dst[i].null_mask.data == nullptr && may_have_nulls(*metas[i])) {
      throw row_range_unsupported_error("row-range read (disk): requested column " +
                                        std::to_string(i) +
                                        " may contain nulls but its destination has no null mask");
    }
  }

  row_range_result result;
  result.data_bytes_written.assign(num_columns, 0);

  // ---- fitting (rule 1) ----
  std::vector<fit_column> fit(num_columns);
  for (std::size_t i = 0; i < num_columns; ++i) {
    fit[i].dst       = &dst[i];
    fit[i].is_string = is_string(*metas[i]);
    if (fit[i].is_string) {
      fit[i].offset_at = [](std::size_t) { return std::int64_t{0}; };
    } else {
      fit[i].fixed_width =
        cudf::size_of(cudf::data_type{static_cast<cudf::type_id>(metas[i]->type_id)});
    }
  }
  // First pass with chars that always fit: an upper bound from the closed-form terms alone, so the
  // host offset reads below cover no more rows than can be delivered.
  const std::size_t k_bound = fit_rows(fit, request.row_begin, request.max_rows, num_rows);

  // One small synchronous host read per STRING column: offsets [row_begin, row_begin + k_bound].
  std::vector<host_offsets> offsets(num_columns);
  if (k_bound > 0) {
    for (std::size_t i = 0; i < num_columns; ++i) {
      if (!fit[i].is_string) { continue; }
      const auto& meta = *metas[i];
      if (meta.children.empty()) { fail_layout(i, "STRING column has no offsets child"); }
      const auto& child = meta.children.front();
      auto& host        = offsets[i];
      host.width        = static_cast<cudf::type_id>(child.type_id) == cudf::type_id::INT64 ? 8 : 4;
      const std::size_t first = request.row_begin * host.width;
      const std::size_t bytes = (k_bound + 1) * host.width;
      if (!child.has_data || first + bytes > child.data_size) {
        fail_layout(i, "STRING offsets buffer is shorter than the requested rows");
      }
      host.bytes.resize(bytes);
      backend.read(path, host.bytes.data(), bytes, child.data_offset + first);
      fit[i].offset_at = [&host, row_begin = request.row_begin](std::size_t row) {
        return host.at(row - row_begin);
      };
    }
  }
  const std::size_t k =
    k_bound == 0 ? 0 : fit_rows(fit, request.row_begin, request.max_rows, num_rows);

  result.rows_written     = k;
  result.source_exhausted = request.row_begin + k == num_rows;
  if (k == 0) { return result; }

  // ---- device reads, all ordered on `stream` ----
  range_reader reader(backend, path, scratch, stream);
  bool pending_host_copy = false;
  auto read_column       = [&](std::size_t i) {
    const auto& meta = *metas[i];
    const auto& out  = dst[i];

    if (fit[i].is_string) {
      const auto& host         = offsets[i];
      const std::int64_t begin = host.at(0);
      const std::int64_t end   = host.at(k);
      if (begin < 0 || end < begin) { fail_layout(i, "STRING offsets are not ascending"); }
      const auto chars = static_cast<std::size_t>(end - begin);
      if (chars > 0) {
        if (!meta.has_data || static_cast<std::size_t>(end) > meta.data_size) {
          fail_layout(i, "STRING chars buffer is shorter than the requested rows");
        }
        reader.read_into(static_cast<std::byte*>(out.data.data) + out.dst_byte_offset,
                         chars,
                         meta.data_offset + static_cast<std::size_t>(begin));
      }
      result.data_bytes_written[i] = chars;

      // Offsets: the host vector already holds the source values; copy the k + 1 of them to the
      // device and rebase there (rule 3). The copy is asynchronous, so `offsets` must outlive it:
      // the stream is synchronized once, at the end of this call.
      const std::size_t bytes = (k + 1) * host.width;
      auto& staged            = reader.allocate(bytes);
      CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
        staged.data(), host.bytes.data(), bytes, cudaMemcpyHostToDevice, stream.get()));
      pending_host_copy = true;
      rebase_offsets(
        static_cast<std::byte*>(out.offsets.data) + out.dst_row_offset * out.offset_width,
        out.offset_width,
        staged.data(),
        host.width,
        k + 1,
        static_cast<std::int64_t>(out.dst_byte_offset),
        stream);
    } else {
      const std::size_t width = fit[i].fixed_width;
      const std::size_t first = request.row_begin * width;
      const std::size_t bytes = k * width;
      if (!meta.has_data || first + bytes > meta.data_size) {
        fail_layout(i, "data buffer is shorter than the requested rows");
      }
      reader.read_into(static_cast<std::byte*>(out.data.data) + out.dst_row_offset * width,
                       bytes,
                       meta.data_offset + first);
      result.data_bytes_written[i] = bytes;
    }

    // Null bits (rule 4).
    if (out.null_mask.data == nullptr) { return; }
    auto* dst_mask = static_cast<std::uint32_t*>(out.null_mask.data);
    if (!may_have_nulls(meta)) {
      copy_bits(dst_mask, out.dst_row_offset, nullptr, 0, k, stream);
      return;
    }
    const std::size_t word_size  = sizeof(std::uint32_t);
    const std::size_t first_word = request.row_begin / 32;
    const std::size_t last_word  = (request.row_begin + k + 31) / 32;
    const std::size_t first      = first_word * word_size;
    const std::size_t bytes      = (last_word - first_word) * word_size;
    if (first + bytes > meta.null_mask_size) {
      fail_layout(i, "null mask is shorter than the requested rows");
    }
    // `first` is a multiple of 4 and the mask starts on a block boundary, so the words are aligned.
    const auto* src_mask = reader.read_aligned_superset(bytes, meta.null_mask_offset + first);
    copy_bits(dst_mask,
              out.dst_row_offset,
              reinterpret_cast<const std::uint32_t*>(src_mask),
              request.row_begin % 32,
              k,
              stream);
  };

  try {
    for (std::size_t i = 0; i < num_columns; ++i) {
      read_column(i);
    }
  } catch (...) {
    // Same wait as below on the way out of a failed read; the read's own error is the one to
    // report.
    if (pending_host_copy) { static_cast<void>(cudaStreamSynchronize(stream.get())); }
    throw;
  }

  // Rule 6 lets this reader synchronize for the host-side offsets: wait for the host-to-device
  // offset copies before `offsets` goes out of scope.
  if (pending_host_copy) { stream.sync(); }
  return result;
}

}  // namespace cucascade::detail
