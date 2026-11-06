/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cudf_compat.hpp"
#include "disk_io.hpp"
#include "remote_io.hpp"

#include <cucascade/cuda/event.hpp>
#include <cucascade/cudf/builtin_converters.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/cudf/host_table.hpp>
#include <cucascade/cudf/host_table_packed.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/data/disk_file_format.hpp>
#include <cucascade/data/remote_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/error.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/disk_access_limiter.hpp>
#include <cucascade/memory/disk_table.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cucascade {

// =============================================================================
// Built-in converter implementations
// =============================================================================

namespace {

// RAII NVTX range for profiling converter phases in the libcucascade domain.
using nvtx_scope = nvtx3::scoped_range_in<libcucascade_domain>;

// memory::column_metadata::type_id is a generic int32_t type tag; the cudf layer interprets it
// as the numeric value of a cudf::type_id. These helpers convert across that boundary.
inline cudf::type_id as_cudf_type_id(int32_t type_id)
{
  return static_cast<cudf::type_id>(type_id);
}

// Forward declaration. convert_gpu_to_gpu is defined below convert_gpu_to_host_fast
// so it can reuse BatchCopyAccumulator and the column-tree reconstruction helpers,
// peer-copying each column buffer directly and avoiding cudf::pack (whose internal
// scratch allocator races with the cucascade resource ref under multi-GPU and
// produces uninitialized dst_buf_info reads inside compute_splits).
std::unique_ptr<idata_representation> convert_gpu_to_gpu(
  const idata_representation& source,
  const memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  [[maybe_unused]] memory::reservation* reservation);

/**
 * @brief Convert gpu_table_representation to host_data_packed_representation
 */
std::unique_ptr<idata_representation> convert_gpu_to_host(
  const idata_representation& source,
  const memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  memory::reservation* reservation)
{
  // Synchronize the stream to ensure any prior operations (like table creation)
  // are complete before we read from the source table
  stream.sync();

  auto& gpu_source = source.cast<gpu_table_representation>();
  gpu_source.wait_for_writer(stream);
  auto packed_data = cudf::pack(gpu_source.get_table_view(), stream);

  auto mr = target_memory_space->get_memory_resource_as<memory::fixed_size_host_memory_resource>();
  auto allocation = mr->allocate_multiple_blocks(packed_data.gpu_data->size(), reservation);

  size_t block_index      = 0;
  size_t block_offset     = 0;
  size_t source_offset    = 0;
  const size_t block_size = allocation->block_size();
  while (source_offset < packed_data.gpu_data->size()) {
    size_t remaining_bytes         = packed_data.gpu_data->size() - source_offset;
    size_t bytes_to_copy           = std::min(remaining_bytes, block_size - block_offset);
    std::span<std::byte> block_ptr = allocation->at(block_index);
    CUCASCADE_CUDA_TRY(
      cudaMemcpyAsync(block_ptr.data() + block_offset,
                      static_cast<const uint8_t*>(packed_data.gpu_data->data()) + source_offset,
                      bytes_to_copy,
                      cudaMemcpyDeviceToHost,
                      stream.get()));
    source_offset += bytes_to_copy;
    block_offset += bytes_to_copy;
    if (block_offset == block_size) {
      block_index++;
      block_offset = 0;
    }
  }
  stream.sync();
  auto host_table_packed_allocation = std::make_unique<memory::host_table_packed_allocation>(
    std::move(allocation), std::move(packed_data.metadata), packed_data.gpu_data->size());
  return std::make_unique<host_data_packed_representation>(
    std::move(host_table_packed_allocation),
    const_cast<memory::memory_space*>(target_memory_space));
}

/**
 * @brief Convert host_data_packed_representation to gpu_table_representation
 */
std::unique_ptr<idata_representation> convert_host_to_gpu(
  const idata_representation& source,
  const memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  [[maybe_unused]] memory::reservation* reservation)
{
  auto& host_source    = source.cast<host_data_packed_representation>();
  auto& host_table     = host_source.get_host_table();
  auto const data_size = host_table->data_size;

  auto mr = target_memory_space->get_default_allocator();
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{target_memory_space->get_device_id()}};

  rmm::device_buffer dst_buffer(data_size, stream, mr);
  size_t src_block_index      = 0;
  size_t src_block_offset     = 0;
  size_t dst_offset           = 0;
  size_t const src_block_size = host_table->allocation->block_size();
  while (dst_offset < data_size) {
    size_t remaining_bytes              = data_size - dst_offset;
    size_t bytes_available_in_src_block = src_block_size - src_block_offset;
    size_t bytes_to_copy                = std::min(remaining_bytes, bytes_available_in_src_block);
    auto src_block                      = host_table->allocation->at(src_block_index);
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(static_cast<uint8_t*>(dst_buffer.data()) + dst_offset,
                                       src_block.data() + src_block_offset,
                                       bytes_to_copy,
                                       cudaMemcpyHostToDevice,
                                       stream.get()));
    dst_offset += bytes_to_copy;
    src_block_offset += bytes_to_copy;
    if (src_block_offset == src_block_size) {
      src_block_index++;
      src_block_offset = 0;
    }
  }

  auto new_gpu_data = std::make_unique<rmm::device_buffer>(std::move(dst_buffer));
  // cudf::unpack only reads metadata to produce a non-owning table_view — no GPU work.
  // Use source metadata directly since it remains alive for the duration of this call.
  // The H→D memcpy and table copy are on the same stream, so ordering is guaranteed.
  auto new_table_view =
    cudf::unpack(host_table->metadata->data(), static_cast<uint8_t const*>(new_gpu_data->data()));
  auto new_table = std::make_unique<cudf::table>(new_table_view, stream, mr);
  stream.sync();

  return gpu_table_representation::make_written_on(
    std::move(new_table), *const_cast<memory::memory_space*>(target_memory_space), stream);
}

/**
 * @brief Convert host_data_packed_representation to host_data_packed_representation (cross-host
 * copy)
 */
std::unique_ptr<idata_representation> convert_host_to_host(
  const idata_representation& source,
  const memory::memory_space* target_memory_space,
  ::cuda::stream_ref /*stream*/,
  memory::reservation* reservation)
{
  auto& host_source    = source.cast<host_data_packed_representation>();
  auto& host_table     = host_source.get_host_table();
  auto const data_size = host_table->data_size;

  assert(source.get_device_id() != target_memory_space->get_device_id());
  auto mr = target_memory_space->get_memory_resource_as<memory::fixed_size_host_memory_resource>();
  if (mr == nullptr) {
    throw std::runtime_error(
      "Target HOST memory_space does not have a fixed_size_host_memory_resource");
  }
  auto dst_allocation         = mr->allocate_multiple_blocks(data_size, reservation);
  size_t src_block_index      = 0;
  size_t src_block_offset     = 0;
  size_t dst_block_index      = 0;
  size_t dst_block_offset     = 0;
  size_t const src_block_size = host_table->allocation->block_size();
  size_t const dst_block_size = dst_allocation->block_size();
  size_t copied               = 0;
  while (copied < data_size) {
    size_t remaining     = data_size - copied;
    size_t src_avail     = src_block_size - src_block_offset;
    size_t dst_avail     = dst_block_size - dst_block_offset;
    size_t bytes_to_copy = std::min(remaining, std::min(src_avail, dst_avail));
    auto* src_ptr        = host_table->allocation->at(src_block_index).data() + src_block_offset;
    auto* dst_ptr        = dst_allocation->at(dst_block_index).data() + dst_block_offset;
    std::memcpy(dst_ptr, src_ptr, bytes_to_copy);
    copied += bytes_to_copy;
    src_block_offset += bytes_to_copy;
    dst_block_offset += bytes_to_copy;
    if (src_block_offset == src_block_size) {
      src_block_index++;
      src_block_offset = 0;
    }
    if (dst_block_offset == dst_block_size) {
      dst_block_index++;
      dst_block_offset = 0;
    }
  }
  auto metadata_copy                = std::make_unique<std::vector<uint8_t>>(*host_table->metadata);
  auto host_table_packed_allocation = std::make_unique<memory::host_table_packed_allocation>(
    std::move(dst_allocation), std::move(metadata_copy), data_size);
  return std::make_unique<host_data_packed_representation>(
    std::move(host_table_packed_allocation),
    const_cast<memory::memory_space*>(target_memory_space));
}

// =============================================================================
// Fast GPU -> Host converter (direct buffer copy, no cudf::pack intermediate)
// =============================================================================

/**
 * @brief Align a byte offset up to the given alignment (must be a power of two).
 */
static std::size_t align_up_fast(std::size_t offset, std::size_t alignment) noexcept
{
  return (offset + alignment - 1u) & ~(alignment - 1u);
}

/**
 * @brief Returns true for column types that store element data in a flat device buffer.
 *
 * STRING, LIST, STRUCT, and EMPTY have no flat data buffer of their own; their payload
 * lives in their children. Everything else (fixed-width types, DICTIONARY32) has one.
 */
static bool column_has_data_buffer(const cudf::column_view& col) noexcept
{
  switch (col.type().id()) {
    case cudf::type_id::STRING:
    case cudf::type_id::LIST:
    case cudf::type_id::STRUCT:
    case cudf::type_id::DICTIONARY32:
    case cudf::type_id::EMPTY: return false;
    default: return true;
  }
}

/**
 * @brief Returns the byte size of one element in a column's data buffer.
 *
 * For DICTIONARY32 the element type is always int32_t (the index type).
 * For all other types with a data buffer, delegates to cudf::size_of.
 */
static std::size_t element_size_bytes(const cudf::column_view& col)
{
  return cudf::size_of(col.type());
}

/**
 * @brief Accumulates (dst, src, size) copy operations for a single cudaMemcpyBatchAsync call.
 *
 * Collect all copies first, then call flush() once to submit them all to the GPU in one driver
 * call. This eliminates per-buffer driver overhead versus issuing individual cudaMemcpyAsync calls.
 */
struct BatchCopyAccumulator {
  std::vector<void*> dsts;
  std::vector<const void*> srcs;
  std::vector<std::size_t> sizes;

  void add(void* dst, const void* src, std::size_t size)
  {
    if (size == 0 || src == nullptr || dst == nullptr) { return; }
    dsts.push_back(dst);
    srcs.push_back(src);
    sizes.push_back(size);
  }

  std::size_t count() const { return dsts.size(); }

  void flush(::cuda::stream_ref stream, cudaMemcpySrcAccessOrder src_order)
  {
    if (count() == 0) { return; }
#if CUDART_VERSION >= 12080
    cudaMemcpyAttributes attr{};
    attr.srcAccessOrder = src_order;
    attr.flags          = cudaMemcpyFlagDefault;
    // Single-attribute template overload (cuda_runtime.h): deduces direction from pointer types.
    // NOTE: cudaMemcpyBatchAsync requires a real (non-default) CUDA stream.
    // CUDA 12.x has a failIdx parameter that was removed in CUDA 13.
#if CUDART_VERSION < 13000
    CUCASCADE_CUDA_TRY(cudaMemcpyBatchAsync(
      dsts.data(), srcs.data(), sizes.data(), count(), attr, nullptr, stream.get()));
#else
    CUCASCADE_CUDA_TRY(
      cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), count(), attr, stream.get()));
#endif
#else
    // cudaMemcpyBatchAsync requires CUDA 12.8+; fall back to individual copies.
    (void)src_order;
    for (std::size_t i = 0; i < count(); ++i) {
      CUCASCADE_CUDA_TRY(
        cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyDefault, stream.get()));
    }
#endif
    // Clear so subsequent add()+flush() cycles do not resubmit already-issued ops.
    dsts.clear();
    srcs.clear();
    sizes.clear();
  }
};

/**
 * @brief Whether the serializers below would copy more (or other) rows of @p col, or of a column
 *        under it, than the column holds.
 *
 * They copy every buffer of a column and its children from the start, which is right only when
 * each column starts at its first element and every child holds exactly the elements its parent
 * uses. A slice breaks that in two ways: a non-zero offset, and -- for a prefix slice, whose
 * offset is 0 -- children that keep the extent of the column it was cut from (the offsets of a
 * STRING or LIST, the fields of a STRUCT, the indices of a DICTIONARY32 hold more than the
 * parent's rows). Either needs a compacted copy.
 */
static bool needs_compaction(const cudf::column_view& col) noexcept
{
  if (col.offset() != 0) { return true; }
  auto const rows = col.size();
  switch (col.type().id()) {
    case cudf::type_id::STRING:
    case cudf::type_id::LIST:
      // child 0 = offsets: rows + 1 entries (an empty column may have none, or one).
      if (col.num_children() > 0) {
        auto const offsets = col.child(0).size();
        if (rows == 0 ? offsets > 1 : offsets != rows + 1) { return true; }
      }
      break;
    case cudf::type_id::STRUCT:
      for (cudf::size_type i = 0; i < col.num_children(); ++i) {
        if (col.child(i).size() != rows) { return true; }
      }
      break;
    case cudf::type_id::DICTIONARY32:
      // child 0 = indices, one per row (child 1 = keys, any number).
      if (col.num_children() > 0 && col.child(0).size() != rows) { return true; }
      break;
    default: break;
  }
  for (cudf::size_type i = 0; i < col.num_children(); ++i) {
    if (needs_compaction(col.child(i))) { return true; }
  }
  return false;
}

/// Whether any column of @p view needs compaction (see needs_compaction(cudf::column_view)).
static bool needs_compaction(const cudf::table_view& view) noexcept
{
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    if (needs_compaction(view.column(i))) { return true; }
  }
  return false;
}

/**
 * @brief Synchronizes a stream when destroyed during unwinding.
 *
 * A converter that throws after it enqueued work reading the source table on the conversion
 * stream (a compaction copy, D2H copies) must not let the caller release or mutate the source
 * while that work may still run. Construct it with the source device current (and keep it
 * current until it is destroyed): a null stream names that device's default stream. A failure of
 * the synchronization itself is swallowed so it neither replaces the original exception nor stays
 * pending for later CUDA calls.
 */
class drain_on_unwind {
 public:
  explicit drain_on_unwind(::cuda::stream_ref stream) noexcept
    : _stream(stream), _exceptions(std::uncaught_exceptions())
  {
  }
  ~drain_on_unwind()
  {
    if (std::uncaught_exceptions() > _exceptions &&
        cudaStreamSynchronize(_stream.get()) != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
    }
  }
  drain_on_unwind(drain_on_unwind const&)            = delete;
  drain_on_unwind& operator=(drain_on_unwind const&) = delete;

 private:
  ::cuda::stream_ref _stream;
  int _exceptions;
};

/**
 * @brief The view the buffer serializers below may copy from: @p view itself, or, when one of its
 *        columns needs it (see needs_compaction()), a compacted copy of the whole table.
 *
 * A view-backed gpu_table_representation may hold a slice, so such a table is first compacted
 * (cudf::table's deep copy normalizes every nested offset, child extent and null mask) on
 * @p stream, which must already be ordered after the table's writer, with the source device
 * current. The copy needs device memory for the whole table on the source GPU; if it cannot be
 * made, the conversion fails and the batch stays where it is. The copy is kept in @p holder: free
 * it only after the work reading it on @p stream (its buffers are freed in stream order on
 * @p stream). If the copy fails part-way, the column copies already enqueued may still be reading
 * the source, so @p stream is synchronized before the exception propagates.
 */
static cudf::table_view serializable_view(const cudf::table_view& view,
                                          ::cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr,
                                          std::unique_ptr<cudf::table>& holder)
{
  if (!needs_compaction(view)) { return view; }
  drain_on_unwind drain{stream};
  holder = std::make_unique<cudf::table>(view, stream, mr);
  return holder->view();
}

/**
 * @brief Recursively plan the buffer layout for one column, filling in column_metadata.
 *
 * Advances @p current_offset by the bytes needed for the column's null mask and data buffer
 * (8-byte aligned), then recurses into any children.
 *
 * @note The column view must not need compaction (see serializable_view()).
 */
static memory::column_metadata plan_column_copy(const cudf::column_view& col,
                                                std::size_t& current_offset,
                                                ::cuda::stream_ref stream)
{
  // Callers compact sliced tables first (serializable_view()).
  assert(!needs_compaction(col) && "sliced column_view: compact it first (serializable_view)");

  memory::column_metadata meta{};
  meta.type_id    = static_cast<int32_t>(col.type().id());
  meta.num_rows   = col.size();
  meta.null_count = col.null_count();
  meta.scale      = 0;

  if (as_cudf_type_id(meta.type_id) == cudf::type_id::DECIMAL32 ||
      as_cudf_type_id(meta.type_id) == cudf::type_id::DECIMAL64 ||
      as_cudf_type_id(meta.type_id) == cudf::type_id::DECIMAL128) {
    meta.scale = col.type().scale();
  }

  // Null mask
  if (col.nullable()) {
    meta.has_null_mask    = true;
    meta.null_mask_size   = cudf::bitmask_allocation_size_bytes(col.size());
    current_offset        = align_up_fast(current_offset, 8u);
    meta.null_mask_offset = current_offset;
    current_offset += meta.null_mask_size;
  } else {
    meta.has_null_mask    = false;
    meta.null_mask_offset = 0;
    meta.null_mask_size   = 0;
  }

  // Flat data buffer
  if (col.type().id() == cudf::type_id::STRING) {
    // STRING stores chars in the column's data buffer; child(0) holds the offsets,
    // which may be INT32 or INT64 (large strings). Use strings_column_view::chars_size()
    // to correctly handle both offset types without a raw cudaMemcpy.
    if (col.size() > 0 && col.num_children() > 0 && col.data<char>() != nullptr) {
      auto const chars_bytes = cudf::strings_column_view(col).chars_size(stream);
      if (chars_bytes > 0) {
        current_offset   = align_up_fast(current_offset, 8u);
        meta.has_data    = true;
        meta.data_offset = current_offset;
        meta.data_size   = static_cast<std::size_t>(chars_bytes);
        current_offset += meta.data_size;
      } else {
        meta.has_data    = false;
        meta.data_offset = 0;
        meta.data_size   = 0;
      }
    } else {
      meta.has_data    = false;
      meta.data_offset = 0;
      meta.data_size   = 0;
    }
  } else if (column_has_data_buffer(col) && col.size() > 0) {
    // Fixed-width types and DICTIONARY32 indices
    meta.has_data    = true;
    meta.data_size   = static_cast<std::size_t>(col.size()) * element_size_bytes(col);
    current_offset   = align_up_fast(current_offset, 8u);
    meta.data_offset = current_offset;
    current_offset += meta.data_size;
  } else {
    meta.has_data    = false;
    meta.data_offset = 0;
    meta.data_size   = 0;
  }

  // Recurse into children (STRING offsets+chars, LIST offsets+values, STRUCT fields, etc.)
  meta.children.reserve(static_cast<std::size_t>(col.num_children()));
  for (cudf::size_type i = 0; i < col.num_children(); ++i) {
    meta.children.push_back(plan_column_copy(col.child(i), current_offset, stream));
  }

  return meta;
}

/**
 * @brief Append block-boundary-split copy ops for @p size device bytes → host allocation.
 *
 * Does NOT issue any CUDA calls; the caller fires one cudaMemcpyBatchAsync after collecting all
 * ops.
 */
static void collect_d2h_ops(
  const void* src,
  std::size_t size,
  std::size_t alloc_offset,
  memory::fixed_size_host_memory_resource::multiple_blocks_allocation& alloc,
  BatchCopyAccumulator& batch)
{
  if (size == 0 || src == nullptr) { return; }

  const std::size_t block_size = alloc.block_size();
  std::size_t block_idx        = alloc_offset / block_size;
  std::size_t block_off        = alloc_offset % block_size;
  std::size_t src_off          = 0;

  while (src_off < size) {
    std::size_t remaining      = size - src_off;
    std::size_t space_in_block = block_size - block_off;
    std::size_t bytes_to_copy  = std::min(remaining, space_in_block);

    auto block = alloc.at(block_idx);
    batch.add(block.data() + block_off, static_cast<const uint8_t*>(src) + src_off, bytes_to_copy);
    src_off += bytes_to_copy;
    block_off += bytes_to_copy;
    if (block_off == block_size) {
      ++block_idx;
      block_off = 0;
    }
  }
}

/**
 * @brief Recursively collect D→H copy ops for a column's null mask, data buffer, and children.
 */
static void collect_column_d2h_ops(
  const cudf::column_view& col,
  const memory::column_metadata& meta,
  memory::fixed_size_host_memory_resource::multiple_blocks_allocation& alloc,
  BatchCopyAccumulator& batch)
{
  if (meta.has_null_mask) {
    collect_d2h_ops(col.null_mask(), meta.null_mask_size, meta.null_mask_offset, alloc, batch);
  }
  if (meta.has_data) {
    collect_d2h_ops(col.data<uint8_t>(), meta.data_size, meta.data_offset, alloc, batch);
  }
  for (cudf::size_type i = 0; i < col.num_children(); ++i) {
    collect_column_d2h_ops(col.child(i), meta.children[static_cast<std::size_t>(i)], alloc, batch);
  }
}

/**
 * @brief Convert gpu_table_representation to host_data_representation.
 *
 * Copies each column's device buffers directly to pinned host memory, bypassing the
 * intermediate GPU-side contiguous buffer that cudf::pack allocates.
 */
std::unique_ptr<idata_representation> convert_gpu_to_host_fast(
  const idata_representation& source,
  const memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  memory::reservation* reservation)
{
  nvtx_scope convert_range{"convert:gpu_to_host"};
  auto& gpu_source = source.cast<gpu_table_representation>();
  // The source device is current from the writer ordering until the copies completed and the
  // compacted copy (if any) is freed: a null `stream` names the current device's default stream,
  // so the writer wait, the planning, the compaction and the D2H copies must resolve it alike.
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{gpu_source.get_device_id()}};
  // Planning reads string offsets on the device, so it must already be ordered after the writer.
  gpu_source.wait_for_writer(stream);
  // On a failure after work reading the source was enqueued, wait for it before unwinding.
  drain_on_unwind drain{stream};
  // A sliced table is compacted first (the copy is freed after the D2H copies below, which this
  // function waits for).
  std::unique_ptr<cudf::table> compacted;
  cudf::table_view const view =
    serializable_view(gpu_source.get_table_view(),
                      stream,
                      gpu_source.get_memory_space().get_default_allocator(),
                      compacted);

  // --- Pass 1: plan the allocation layout ---
  std::size_t current_offset = 0;
  std::vector<memory::column_metadata> columns;
  columns.reserve(static_cast<std::size_t>(view.num_columns()));
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    columns.push_back(plan_column_copy(view.column(i), current_offset, stream));
  }
  const std::size_t total_size = current_offset;

  // --- Pass 2: allocate pinned host blocks ---
  auto mr = target_memory_space->get_memory_resource_as<memory::fixed_size_host_memory_resource>();
  auto allocation = mr->allocate_multiple_blocks(total_size, reservation);

  // --- Pass 3: collect all D→H copy ops, then fire one batched call ---
  BatchCopyAccumulator batch;
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    collect_column_d2h_ops(
      view.column(i), columns[static_cast<std::size_t>(i)], *allocation, batch);
  }
  batch.flush(stream, cudaMemcpySrcAccessOrderStream);
  stream.sync();

  auto host_alloc =
    memory::host_table_allocation::create(std::move(allocation), std::move(columns), total_size);

  return std::make_unique<host_data_representation>(
    std::move(host_alloc), const_cast<memory::memory_space*>(target_memory_space));
}

// =============================================================================
// Fast GPU -> GPU converter (direct per-buffer peer copy, no cudf::pack)
// =============================================================================

/**
 * @brief Best-effort lookup of the NUMA node a GPU's PCI device is attached to.
 *        Returns -1 if the device id is invalid, the PCI bus id is unknown, or
 *        /sys/bus/pci/devices/<bus>/numa_node is unreadable / -1. Pure topology
 *        query — no allocation, no side effects.
 */
static int resolve_gpu_numa_node(int device_id) noexcept
{
  if (device_id < 0) { return -1; }
  char pci_buf[32] = {0};
  if (cudaDeviceGetPCIBusId(pci_buf, sizeof(pci_buf), device_id) != cudaSuccess) { return -1; }
  std::string pci(pci_buf);
  for (auto& c : pci) {
    c = static_cast<char>(std::tolower(c));
  }
  std::ifstream f("/sys/bus/pci/devices/" + pci + "/numa_node");
  if (!f.is_open()) { return -1; }
  std::string content;
  std::getline(f, content);
  if (content.empty()) { return -1; }
  try {
    return std::stoi(content);
  } catch (...) {
    return -1;
  }
}

/**
 * @brief Copy @p size bytes from @p src_ptr
 * (on @p src_device) into a buffer.
 *
 * Routing is decided per direction by cucascade::memory::probe_peer_dma_works(), which verifies a
 * peer copy between two mutually granted private memory pools:
 *   - verified: cudaMemcpyPeerAsync (direct only when the pools are granted peer access)
 *   - wrong bytes or a CUDA error during verification: explicit host-staged copy
 *
 * The host-staged copy goes through the pre-pinned HOST memory_space pool (registered by
 * memory_space's HOST constructor in register_host_pool) and chunks across the pool's fixed block
 * size, exactly like convert_gpu_to_host. We do the staging ourselves rather than relying on the
 * driver's auto-fallback, so the correctness path is identical and observable regardless of
 * driver/CUDA version quirks.
 *
 *   NEVER calls cudaHostAlloc per transfer — the pool is allocated once at
 *   memory_reservation_manager construction time.
 */
template <typename Buffer>
static Buffer peer_copy_into(Buffer buf,
                             const void* src_ptr,
                             int src_device,
                             std::size_t size,
                             int dst_device,
                             ::cuda::stream_ref target_stream)
{
  if (size == 0 || src_ptr == nullptr) { return buf; }

  if (memory::probe_peer_dma_works(src_device, dst_device)) {
    // Real peer DMA works on this hardware — direct path.
    CUCASCADE_CUDA_TRY(
      cudaMemcpyPeerAsync(buf.data(), dst_device, src_ptr, src_device, size, target_stream.get()));
    return buf;
  }

  // Peer DMA broken — stage through the pre-pinned HOST pool.
  //
  // Block layout: the host pool allocates non-contiguous fixed-size blocks
  // (1 MB by default). We chunk the D2H and H2D legs across blocks just like
  // convert_gpu_to_host does. Stream-ordering inside the loop preserves the
  // same-stream invariant the original implementation relied on.
  memory::fixed_size_host_memory_resource* host_pool =
    memory::find_host_pool(resolve_gpu_numa_node(dst_device));
  if (host_pool == nullptr) {
    // No HOST memory_space has been registered. This should only happen in
    // standalone test builds that exercise the converter without constructing
    // a memory_reservation_manager. Fall back to a one-shot pinned allocation
    // (slow path — preserves correctness, never reached in production Sirius).
    memory::numa_region_pinned_host_memory_resource mr(resolve_gpu_numa_node(dst_device),
                                                       /*make_portable=*/true);
    // A throw below leaks host_buf (and may leave a copy reading it in flight); this branch is
    // unreachable in production, where the HOST space registers the pool, so it stays simple.
    void* host_buf = mr.allocate_sync(size);
    {
      rmm::cuda_set_device_raii src_guard{rmm::cuda_device_id{src_device}};
      CUCASCADE_CUDA_TRY(
        cudaMemcpyAsync(host_buf, src_ptr, size, cudaMemcpyDeviceToHost, target_stream.get()));
      target_stream.sync();
    }
    {
      rmm::cuda_set_device_raii dst_guard{rmm::cuda_device_id{dst_device}};
      CUCASCADE_CUDA_TRY(
        cudaMemcpyAsync(buf.data(), host_buf, size, cudaMemcpyHostToDevice, target_stream.get()));
      target_stream.sync();
    }
    mr.deallocate_sync(host_buf, size);
    return buf;
  }

  auto allocation = host_pool->allocate_multiple_blocks(size);
  // Destroyed before `allocation`: if a copy below throws, the chunks already enqueued on
  // target_stream finish before ~multiple_blocks_allocation returns their blocks to the pool.
  // target_stream is a pool stream, never the null handle, so the current device does not matter.
  drain_on_unwind drain{target_stream};
  const std::size_t block_sz = allocation->block_size();

  // Pass 1: device-to-host, chunked across blocks. Issued on target_stream
  // under the src_device CUDA context (matches the same-stream invariant
  // the original implementation enforced).
  {
    rmm::cuda_set_device_raii src_guard{rmm::cuda_device_id{src_device}};
    std::size_t block_index  = 0;
    std::size_t block_offset = 0;
    std::size_t src_offset   = 0;
    while (src_offset < size) {
      const std::size_t remaining  = size - src_offset;
      const std::size_t bytes_left = block_sz - block_offset;
      const std::size_t to_copy    = std::min(remaining, bytes_left);
      std::span<std::byte> block   = allocation->at(block_index);
      CUCASCADE_CUDA_TRY(cudaMemcpyAsync(block.data() + block_offset,
                                         static_cast<const std::uint8_t*>(src_ptr) + src_offset,
                                         to_copy,
                                         cudaMemcpyDeviceToHost,
                                         target_stream.get()));
      src_offset += to_copy;
      block_offset += to_copy;
      if (block_offset == block_sz) {
        ++block_index;
        block_offset = 0;
      }
    }
    target_stream.sync();
  }

  // Pass 2: host-to-device, chunked across the same blocks. Switch the CUDA
  // context to dst_device — same guard the original code documented as
  // required when peer DMA is broken and the outer guard does not propagate
  // through the column tree walk.
  {
    rmm::cuda_set_device_raii dst_guard{rmm::cuda_device_id{dst_device}};
    std::size_t block_index  = 0;
    std::size_t block_offset = 0;
    std::size_t dst_offset   = 0;
    while (dst_offset < size) {
      const std::size_t remaining  = size - dst_offset;
      const std::size_t bytes_left = block_sz - block_offset;
      const std::size_t to_copy    = std::min(remaining, bytes_left);
      std::span<std::byte> block   = allocation->at(block_index);
      CUCASCADE_CUDA_TRY(cudaMemcpyAsync(reinterpret_cast<std::uint8_t*>(buf.data()) + dst_offset,
                                         block.data() + block_offset,
                                         to_copy,
                                         cudaMemcpyHostToDevice,
                                         target_stream.get()));
      dst_offset += to_copy;
      block_offset += to_copy;
      if (block_offset == block_sz) {
        ++block_index;
        block_offset = 0;
      }
    }
    target_stream.sync();
  }
  // allocation's destructor returns blocks to the host pool — no cudaFreeHost.
  return buf;
}

/**
 * @brief Allocate a null mask sized for @p rows rows and fill it, synchronously, from the
 * host block allocation starting at @p alloc_offset.
 */
static rmm::device_buffer alloc_and_peer_copy_async(const void* src_ptr,
                                                    int src_device,
                                                    std::size_t size,
                                                    int dst_device,
                                                    ::cuda::stream_ref target_stream,
                                                    rmm::device_async_resource_ref target_mr)
{
  return peer_copy_into(rmm::device_buffer(size, target_stream, target_mr),
                        src_ptr,
                        src_device,
                        size,
                        dst_device,
                        target_stream);
}

/**
 * @brief Allocate a null mask sized for @p rows rows and copy it from @p src_ptr
 * (on @p src_device), synchronizing before returning. Note @p rows is a row count,
 * not a byte count - the allocation size comes from cudf::create_null_mask. Used for null masks
 * because cudf column factories may inspect them during column construction.
 */
static cudf_compat::null_mask_buffer alloc_and_peer_copy_sync(
  const void* src_ptr,
  int src_device,
  cudf::size_type rows,
  int dst_device,
  ::cuda::stream_ref target_stream,
  rmm::device_async_resource_ref target_mr)
{
  auto buf =
    cudf::create_null_mask(rows, cudf::mask_state::UNINITIALIZED, target_stream, target_mr);
  auto const size = cudf_compat::get_null_mask_size_bytes(buf);
  buf = peer_copy_into(std::move(buf), src_ptr, src_device, size, dst_device, target_stream);
  if (size == 0 || src_ptr == nullptr) { return buf; }
  target_stream.sync();
  return buf;
}

/**
 * @brief Recursively rebuild a cudf::column on the target device, peer-copying each
 * leaf buffer (null mask, data, offsets, children) directly. Mirrors the structure
 * of reconstruct_column() (host→gpu) but reads from a source cudf::column_view on a
 * peer device using cudaMemcpyPeerAsync.
 *
 * Null masks are copied synchronously because cudf factories may read them during
 * column construction. Data and offset buffers are issued asynchronously on
 * @p target_stream; the caller must sync that stream before the resulting cudf::table
 * is observed by another stream.
 *
 * @note The column view must not need compaction (see serializable_view()), the same
 *       precondition as plan_column_copy() on the GPU↔Host fast path: a slice is compacted first.
 */
static std::unique_ptr<cudf::column> reconstruct_column_p2p(const cudf::column_view& src,
                                                            int src_device,
                                                            int dst_device,
                                                            ::cuda::stream_ref stream,
                                                            rmm::device_async_resource_ref mr)
{
  // Callers compact sliced tables first (serializable_view()).
  assert(!needs_compaction(src) && "sliced column_view: compact it first (serializable_view)");

  auto null_mask = cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED);
  if (src.nullable()) {
    null_mask =
      alloc_and_peer_copy_sync(src.null_mask(), src_device, src.size(), dst_device, stream, mr);
  }
  cudf::size_type const null_count = src.nullable() ? src.null_count() : 0;

  if (src.type().id() == cudf::type_id::STRING) {
    if (src.num_children() < 1) {
      // Empty / degenerate STRING column with no offsets child (cudf produces
      // these for empty intermediate results). Use cudf::make_empty_column
      // which builds the correct internal layout (offsets child of size 1
      // holding [0], chars in parent's data buffer of size 0).
      return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
    }
    // Copy offsets as-is (preserve source's INT32 vs INT64 type). make_strings_column
    // accepts either offset type.
    auto offsets_col = reconstruct_column_p2p(src.child(0), src_device, dst_device, stream, mr);

    rmm::device_buffer chars_buf{};
    if (src.size() > 0 && src.data<char>() != nullptr) {
      // Read total chars size from offsets[size-1]. Reading from the *source* column
      // here is unsafe: cudf::strings_column_view::chars_size() ultimately calls
      // cudaMemcpyAsync(... cudaMemcpyDefault ...), which on cudaMallocAsync pool
      // memory across devices silently returns success without copying bytes — the
      // host buffer keeps its uninitialized stack value (often -1). Read instead
      // from the target-side offsets buffer we just peer-copied; that's a same-device
      // D→H read on the target stream and works reliably regardless of pool peer
      // access semantics.
      auto const offsets_view = offsets_col->view();
      auto const last_idx     = offsets_view.size() - 1;
      int64_t chars_bytes     = 0;
      stream.sync();
      if (offsets_view.type().id() == cudf::type_id::INT32) {
        int32_t value = 0;
        CUCASCADE_CUDA_TRY(cudaMemcpyAsync(&value,
                                           offsets_view.head<int32_t>() + last_idx,
                                           sizeof(int32_t),
                                           cudaMemcpyDeviceToHost,
                                           stream.get()));
        stream.sync();
        chars_bytes = value;
      } else {
        int64_t value = 0;
        CUCASCADE_CUDA_TRY(cudaMemcpyAsync(&value,
                                           offsets_view.head<int64_t>() + last_idx,
                                           sizeof(int64_t),
                                           cudaMemcpyDeviceToHost,
                                           stream.get()));
        stream.sync();
        chars_bytes = value;
      }
      if (chars_bytes > 0) {
        chars_buf = alloc_and_peer_copy_async(src.data<char>(),
                                              src_device,
                                              static_cast<std::size_t>(chars_bytes),
                                              dst_device,
                                              stream,
                                              mr);
      }
    }
    return cudf::make_strings_column(
      src.size(), std::move(offsets_col), std::move(chars_buf), null_count, std::move(null_mask));
  }

  if (src.type().id() == cudf::type_id::LIST) {
    if (src.num_children() < 2) {
      throw std::invalid_argument(
        "reconstruct_column_p2p: LIST column must have two children (offsets, values)");
    }
    // Preserve source's offsets type — make_lists_column accepts INT32 or INT64.
    auto offsets_col = reconstruct_column_p2p(src.child(0), src_device, dst_device, stream, mr);
    auto values_col  = reconstruct_column_p2p(src.child(1), src_device, dst_device, stream, mr);
    return cudf::make_lists_column(
      src.size(), std::move(offsets_col), std::move(values_col), null_count, std::move(null_mask));
  }

  if (src.type().id() == cudf::type_id::STRUCT) {
    std::vector<std::unique_ptr<cudf::column>> fields;
    fields.reserve(static_cast<std::size_t>(src.num_children()));
    for (cudf::size_type i = 0; i < src.num_children(); ++i) {
      fields.push_back(reconstruct_column_p2p(src.child(i), src_device, dst_device, stream, mr));
    }
    // Construct directly to skip make_structs_column's superimpose_nulls kernel —
    // our peer-copied null masks are already consistent (mirrors reconstruct_column()).
    return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRUCT},
                                          src.size(),
                                          rmm::device_buffer{},
                                          std::move(null_mask),
                                          null_count,
                                          std::move(fields));
  }

  if (src.type().id() == cudf::type_id::DICTIONARY32) {
    if (src.num_children() < 2) {
      throw std::invalid_argument(
        "reconstruct_column_p2p: DICTIONARY32 column must have two children "
        "(indices, keys)");
    }
    // cudf DICTIONARY32 child order: [0]=indices, [1]=keys.
    // make_dictionary_column parameter order: (keys, indices, ...).
    auto indices_col = reconstruct_column_p2p(src.child(0), src_device, dst_device, stream, mr);
    auto keys_col    = reconstruct_column_p2p(src.child(1), src_device, dst_device, stream, mr);
    return cudf::make_dictionary_column(
      std::move(keys_col), std::move(indices_col), std::move(null_mask), null_count);
  }

  // Fixed-width / fixed-point — single flat data buffer.
  rmm::device_buffer data_buf{};
  if (src.size() > 0 && src.head() != nullptr) {
    auto const data_size = static_cast<std::size_t>(src.size()) * cudf::size_of(src.type());
    data_buf = alloc_and_peer_copy_async(src.head(), src_device, data_size, dst_device, stream, mr);
  }
  return std::make_unique<cudf::column>(
    src.type(), src.size(), std::move(data_buf), std::move(null_mask), null_count);
}

/**
 * @brief Convert gpu_table_representation to gpu_table_representation (cross-GPU copy).
 *
 * Replaces the previous cudf::pack-based path. cudf::pack's compute_splits internally
 * launches kernels that read scratch buffers allocated through the current_device
 * resource ref; under sirius's multi-GPU setup the cucascade resource adapter created
 * a stream-ordered race that materialized as uninitialized dst_buf_info reads (caught
 * by compute-sanitizer) and silently produced garbage packed bytes. This path mirrors
 * convert_gpu_to_host_fast / convert_host_fast_to_gpu: walk the source column tree,
 * allocate target-side buffers, and peer-copy each leaf buffer directly.
 *
 * The walk copies every buffer of a column from its start, so a view-backed source that is a
 * slice (see needs_compaction()) is first compacted on the source device, with the same cost as
 * the other buffer copiers: device memory for the whole table on the source GPU. If anything
 * throws after copies were enqueued, the target stream is drained before the exception
 * propagates, so the caller may release or mutate the source.
 */
std::unique_ptr<idata_representation> convert_gpu_to_gpu(
  const idata_representation& source,
  const memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  [[maybe_unused]] memory::reservation* reservation)
{
  nvtx_scope convert_range{"convert:gpu_to_gpu"};
  // Sync the caller's stream so the source table's buffers are stable on the source
  // device before we issue peer copies. The caller's stream is the one that produced
  // (or last touched) the source representation.
  stream.sync();

  auto& gpu_source = source.cast<gpu_table_representation>();

  // Same-device case: clone via source's own clone() method.
  if (source.get_device_id() == target_memory_space->get_device_id()) {
    // A null stream handle represents the default stream of the current device: ensure it's the
    // device attached to the representation, or clone() will throw.
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{target_memory_space->get_device_id()}};
    return source.clone(stream);
  }

  auto const src_device_id = gpu_source.get_device_id();
  auto const dst_device_id = target_memory_space->get_device_id();
  auto const copy_path     = memory::probe_peer_dma_works(src_device_id, dst_device_id)
                               ? "convert:gpu_to_gpu:peer_copy"
                               : "convert:gpu_to_gpu:host_staging";
  nvtx_scope copy_path_range{copy_path};

  rmm::cuda_set_device_raii target_guard{rmm::cuda_device_id{dst_device_id}};

  // Target-bound stream from the target memory_space's stream pool. All peer copies
  // and target-side allocations are issued on this stream so they observe in-order
  // completion without explicit cross-stream events.
  auto target_stream = target_memory_space->acquire_stream();
  auto mr            = target_memory_space->get_default_allocator();

  // Order the peer copies below after the most recent writer of gpu_source. With an unknown writer
  // this synchronizes the whole source device, which is correct but invisible to compute-sanitizer:
  // its stream-ordered race tracking ignores host-side synchronization, so it may report races here
  // that a recorded writer event avoids.
  gpu_source.wait_for_writer(target_stream);

  // A view-backed source may be a slice: the walker copies every buffer from its start, so a
  // sliced table is compacted first, on the source device (see serializable_view()).
  std::unique_ptr<cudf::table> compacted;  // outlives every copy reading it
  // Destroyed first: on a throw, the copies enqueued on target_stream (reading the source or
  // `compacted`) finish before either may be released. target_stream is a pool stream, never the
  // null handle, so the device current at destruction does not matter.
  drain_on_unwind drain{target_stream};
  cudf::table_view src_view = gpu_source.get_table_view();
  if (needs_compaction(src_view)) {
    rmm::cuda_set_device_raii src_guard{rmm::cuda_device_id{src_device_id}};
    auto const src_stream = gpu_source.get_memory_space().acquire_stream();
    gpu_source.wait_for_writer(src_stream);
    src_view = serializable_view(
      src_view, src_stream, gpu_source.get_memory_space().get_default_allocator(), compacted);
    // The cross-device copies on target_stream read the compacted copy. Its buffers are later
    // freed in stream order on src_stream, after target_stream.sync() below (or the drain).
    src_stream.sync();
  }

  std::vector<std::unique_ptr<cudf::column>> target_columns;
  target_columns.reserve(static_cast<std::size_t>(src_view.num_columns()));
  for (cudf::size_type i = 0; i < src_view.num_columns(); ++i) {
    target_columns.push_back(
      reconstruct_column_p2p(src_view.column(i), src_device_id, dst_device_id, target_stream, mr));
  }

  auto new_table = std::make_unique<cudf::table>(std::move(target_columns));
  // Sync so all peer copies and any cudf::cast launches complete before the new
  // table is observed by another stream.
  target_stream.sync();

  // STREAM-LINEAGE: the resulting representation was written by target_stream;
  // the constructor records an event on it so any subsequent cross-device
  // reader of this new representation observes the producer-consumer ordering.
  // Although we just synchronized target_stream above (so a same-stream reader
  // needs no further wait), the event is still required for readers that
  // arrive on a different stream.
  return std::make_unique<gpu_table_representation>(
    std::move(new_table), *const_cast<memory::memory_space*>(target_memory_space), target_stream);
}

/**
 * @brief Allocate a device buffer of @p size bytes and schedule its H→D copy ops into @p batch.
 *
 * The buffer is allocated immediately; the actual copies are deferred until the caller calls
 * batch.flush(). The buffer must remain alive until flush() completes (it will, since it is
 * owned by the column tree being assembled in the caller).
 */
static rmm::device_buffer alloc_and_schedule_h2d(
  memory::fixed_size_host_memory_resource::multiple_blocks_allocation& alloc,
  std::size_t alloc_offset,
  std::size_t size,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  BatchCopyAccumulator& batch)
{
  rmm::device_buffer buf;
  {
    nvtx_scope alloc_range{"convert:host_to_gpu:device_alloc"};
    buf = rmm::device_buffer(size, stream, mr);
  }
  if (size == 0) { return buf; }
  if (alloc.size() == 0) {
    throw std::invalid_argument(
      "alloc_and_schedule_h2d: allocation is empty but copy size is non-zero");
  }

  const std::size_t block_size = alloc.block_size();
  std::size_t block_idx        = alloc_offset / block_size;
  std::size_t block_off        = alloc_offset % block_size;
  std::size_t dst_off          = 0;

  while (dst_off < size) {
    std::size_t remaining      = size - dst_off;
    std::size_t space_in_block = block_size - block_off;
    std::size_t bytes_to_copy  = std::min(remaining, space_in_block);

    auto block = alloc.at(block_idx);
    batch.add(static_cast<uint8_t*>(buf.data()) + dst_off, block.data() + block_off, bytes_to_copy);
    dst_off += bytes_to_copy;
    block_off += bytes_to_copy;
    if (block_off == block_size) {
      ++block_idx;
      block_off = 0;
    }
  }
  return buf;
}

/**
 * @brief Copy data from host block allocation to a device buffer synchronously.
 *
 * Unlike alloc_and_schedule_h2d, this performs the copies immediately and synchronizes
 * the stream. Used for null masks which cudf column factories access on construction
 * (before batch.flush() runs).
 */
static cudf_compat::null_mask_buffer alloc_and_copy_h2d_sync(
  memory::fixed_size_host_memory_resource::multiple_blocks_allocation& alloc,
  std::size_t alloc_offset,
  cudf::size_type rows,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  nvtx_scope range{"convert:host_to_gpu:nullmask_sync"};
  auto buf = [&] {
    nvtx_scope alloc_range{"convert:host_to_gpu:device_alloc"};
    return cudf::create_null_mask(rows, cudf::mask_state::UNINITIALIZED, stream, mr);
  }();
  auto const size = cudf_compat::get_null_mask_size_bytes(buf);
  if (size == 0) { return buf; }

  const std::size_t block_size = alloc.block_size();
  std::size_t block_idx        = alloc_offset / block_size;
  std::size_t block_off        = alloc_offset % block_size;
  std::size_t dst_off          = 0;

  while (dst_off < size) {
    std::size_t remaining      = size - dst_off;
    std::size_t space_in_block = block_size - block_off;
    std::size_t bytes_to_copy  = std::min(remaining, space_in_block);

    auto block = alloc.at(block_idx);
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(buf.data()) + dst_off,
                                       block.data() + block_off,
                                       bytes_to_copy,
                                       cudaMemcpyHostToDevice,
                                       stream.get()));
    dst_off += bytes_to_copy;
    block_off += bytes_to_copy;
    if (block_off == block_size) {
      ++block_idx;
      block_off = 0;
    }
  }
  stream.sync();
  return buf;
}

/**
 * @brief Recursively reconstruct a cudf::column from column_metadata and host data.
 *
 * Allocates device buffers and schedules their H→D copies into @p batch. The caller must call
 * batch.flush() after all columns are reconstructed to actually issue the transfers.
 *
 * Null masks are copied synchronously because cudf column factories (make_structs_column,
 * make_strings_column, etc.) may access them immediately on construction.
 */
static std::unique_ptr<cudf::column> reconstruct_column(
  const memory::column_metadata& meta,
  memory::fixed_size_host_memory_resource::multiple_blocks_allocation& alloc,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  BatchCopyAccumulator& batch)
{
  // Null mask — copied synchronously because cudf factories access it on construction.
  // An all-valid mask (null_count == 0) is semantically identical to no mask: skip its
  // upload entirely and construct the column non-nullable, avoiding a per-column
  // blocking memcpy + stream sync.
  auto null_mask = cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED);
  if (meta.has_null_mask && meta.null_count > 0) {
    null_mask = alloc_and_copy_h2d_sync(alloc, meta.null_mask_offset, meta.num_rows, stream, mr);
  }
  const cudf::size_type null_count = meta.has_null_mask ? meta.null_count : 0;

  if (as_cudf_type_id(meta.type_id) == cudf::type_id::STRING) {
    // cudf::make_empty_column(STRING) produces a 0-row strings column with no children, and
    // plan_column_copy faithfully records that. Reconstruct the canonical empty strings column
    // here instead of failing the offsets-child check below.
    if (meta.children.empty() && meta.num_rows == 0) {
      return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
    }
    if (meta.children.size() < 1) {
      throw std::invalid_argument(
        "reconstruct_column: STRING column metadata must have at least one child (offsets)");
    }
    auto offsets_col = reconstruct_column(meta.children[0], alloc, stream, mr, batch);
    if (offsets_col->type().id() == cudf::type_id::INT32) {
      // Flush pending H2D copies so the INT32 offsets buffer has valid data on device
      // before the cast reads from it. The cast replaces offsets_col, freeing the old
      // INT32 buffer, so batch must not hold dangling pointers to it.
      nvtx_scope cast_range{"convert:host_to_gpu:offsets_cast"};
      batch.flush(stream, cudaMemcpySrcAccessOrderDuringApiCall);
      offsets_col =
        cudf::cast(offsets_col->view(), cudf::data_type{cudf::type_id::INT64}, stream, mr);
    }
    return cudf::make_strings_column(
      meta.num_rows,
      std::move(offsets_col),
      meta.has_data && meta.data_size > 0
        ? alloc_and_schedule_h2d(alloc, meta.data_offset, meta.data_size, stream, mr, batch)
        : rmm::device_buffer{},
      null_count,
      std::move(null_mask));
  }

  if (as_cudf_type_id(meta.type_id) == cudf::type_id::LIST) {
    if (meta.children.size() < 2) {
      throw std::invalid_argument(
        "reconstruct_column: LIST column metadata must have two children (offsets, values)");
    }
    // LIST offsets must remain INT32 (cudf::size_type). The INT64 promotion is correct only for
    // STRING offsets (large-strings convention), not LIST. Construct the column directly (like
    // the STRUCT branch below) instead of make_lists_column: this preserves the INT32 offsets
    // and avoids a purge_nonempty_nulls kernel launch. Our serialized data already has consistent
    // null masks, and that kernel would otherwise read buffers still pending in the batched H2D
    // copies.
    std::vector<std::unique_ptr<cudf::column>> children;
    children.reserve(2);
    children.push_back(reconstruct_column(meta.children[0], alloc, stream, mr, batch));
    children.push_back(reconstruct_column(meta.children[1], alloc, stream, mr, batch));
    return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::LIST},
                                          meta.num_rows,
                                          rmm::device_buffer{},
                                          std::move(null_mask),
                                          null_count,
                                          std::move(children));
  }

  if (as_cudf_type_id(meta.type_id) == cudf::type_id::STRUCT) {
    std::vector<std::unique_ptr<cudf::column>> fields;
    fields.reserve(meta.children.size());
    for (const auto& child_meta : meta.children) {
      fields.push_back(reconstruct_column(child_meta, alloc, stream, mr, batch));
    }
    // Construct directly instead of make_structs_column to avoid superimpose_nulls
    // kernel launch — our serialized data already has consistent null masks.
    return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRUCT},
                                          meta.num_rows,
                                          rmm::device_buffer{},
                                          std::move(null_mask),
                                          null_count,
                                          std::move(fields));
  }

  if (as_cudf_type_id(meta.type_id) == cudf::type_id::DICTIONARY32) {
    if (meta.children.size() < 2) {
      throw std::invalid_argument(
        "reconstruct_column: DICTIONARY32 column metadata must have two children "
        "(indices, keys)");
    }
    // cudf DICTIONARY32 children order: [0]=indices, [1]=keys.
    // make_dictionary_column parameter order: (keys, indices, ...).
    auto indices_col = reconstruct_column(meta.children[0], alloc, stream, mr, batch);
    auto keys_col    = reconstruct_column(meta.children[1], alloc, stream, mr, batch);
    return cudf::make_dictionary_column(
      std::move(keys_col), std::move(indices_col), std::move(null_mask), null_count);
  }

  const cudf::data_type dtype = cudf::is_fixed_point(cudf::data_type{as_cudf_type_id(meta.type_id)})
                                  ? cudf::data_type{as_cudf_type_id(meta.type_id), meta.scale}
                                  : cudf::data_type{as_cudf_type_id(meta.type_id)};
  return std::make_unique<cudf::column>(
    dtype,
    meta.num_rows,
    meta.has_data && meta.data_size > 0
      ? alloc_and_schedule_h2d(alloc, meta.data_offset, meta.data_size, stream, mr, batch)
      : rmm::device_buffer{},
    std::move(null_mask),
    null_count);
}

/**
 * @brief Convert host_data_representation to gpu_table_representation.
 *
 * Reconstructs each column from the stored column_metadata tree, copying
 * buffers from pinned host blocks directly to device allocations.
 */
std::unique_ptr<idata_representation> convert_host_fast_to_gpu(
  const idata_representation& source,
  const memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  [[maybe_unused]] memory::reservation* reservation)
{
  nvtx_scope convert_range{"convert:host_to_gpu"};
  auto& fast_source      = source.cast<host_data_representation>();
  const auto& fast_table = fast_source.get_host_table();
  if (!fast_table) { throw std::runtime_error("convert_host_fast_to_gpu: host table is null"); }
  if (!fast_table->allocation) {
    throw std::runtime_error("convert_host_fast_to_gpu: host table allocation is null");
  }

  // Sync the caller's stream so any upstream work that produced this
  // host_data_representation is flushed before we read the pinned host blocks.
  // The caller's stream may be bound to a non-target device under multi-GPU;
  // synchronize is safe across devices.
  {
    nvtx_scope presync_range{"convert:host_to_gpu:presync"};
    stream.sync();
  }

  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{target_memory_space->get_device_id()}};

  // Acquire a target-bound stream from the target memory_space's stream pool.
  // Using the caller's stream for the H2D batch under a target-device RAII
  // guard raises cudaErrorInvalidValue when stream and current device belong
  // to different CUDA contexts (multi-GPU case).
  auto target_stream = [&] {
    nvtx_scope acquire_range{"convert:host_to_gpu:acquire_stream"};
    return target_memory_space->acquire_stream();
  }();
  auto mr = target_memory_space->get_default_allocator();

  // Collect all H→D copy ops across all columns, then fire one batched call.
  BatchCopyAccumulator batch;
  std::vector<std::unique_ptr<cudf::column>> gpu_columns;
  gpu_columns.reserve(fast_table->columns.size());
  {
    nvtx_scope reconstruct_range{"convert:host_to_gpu:reconstruct"};
    for (const auto& col_meta : fast_table->columns) {
      gpu_columns.push_back(
        reconstruct_column(col_meta, *fast_table->allocation, target_stream, mr, batch));
    }
  }
  // Source is CPU-written pinned host memory: fully prepared before this call.
  {
    nvtx_scope flush_range{"convert:host_to_gpu:flush_submit"};
    batch.flush(target_stream, cudaMemcpySrcAccessOrderDuringApiCall);
  }

  auto new_table = std::make_unique<cudf::table>(std::move(gpu_columns));
  {
    nvtx_scope final_sync_range{"convert:host_to_gpu:final_sync"};
    target_stream.sync();
  }

  // STREAM-LINEAGE: writes happened on target_stream; record event so
  // cross-stream readers observe ordering.
  return std::make_unique<gpu_table_representation>(
    std::move(new_table), *const_cast<memory::memory_space*>(target_memory_space), target_stream);
}

/**
 * @brief Convert host_data_representation to host_data_representation (cross-host copy)
 */
std::unique_ptr<idata_representation> convert_host_fast_to_host_fast(
  const idata_representation& source,
  const memory::memory_space* target_memory_space,
  ::cuda::stream_ref /*stream*/,
  memory::reservation* reservation)
{
  nvtx_scope convert_range{"convert:host_to_host"};
  auto& host_source    = source.cast<host_data_representation>();
  auto& host_table     = host_source.get_host_table();
  auto const data_size = host_table->data_size;

  assert(source.get_device_id() != target_memory_space->get_device_id());
  auto mr = target_memory_space->get_memory_resource_as<memory::fixed_size_host_memory_resource>();
  if (mr == nullptr) {
    throw std::runtime_error(
      "Target HOST memory_space does not have a fixed_size_host_memory_resource");
  }
  auto dst_allocation         = mr->allocate_multiple_blocks(data_size, reservation);
  size_t src_block_index      = 0;
  size_t src_block_offset     = 0;
  size_t dst_block_index      = 0;
  size_t dst_block_offset     = 0;
  size_t const src_block_size = host_table->allocation->block_size();
  size_t const dst_block_size = dst_allocation->block_size();
  size_t copied               = 0;
  while (copied < data_size) {
    size_t remaining     = data_size - copied;
    size_t src_avail     = src_block_size - src_block_offset;
    size_t dst_avail     = dst_block_size - dst_block_offset;
    size_t bytes_to_copy = std::min(remaining, std::min(src_avail, dst_avail));
    auto* src_ptr        = host_table->allocation->at(src_block_index).data() + src_block_offset;
    auto* dst_ptr        = dst_allocation->at(dst_block_index).data() + dst_block_offset;
    std::memcpy(dst_ptr, src_ptr, bytes_to_copy);
    copied += bytes_to_copy;
    src_block_offset += bytes_to_copy;
    dst_block_offset += bytes_to_copy;
    if (src_block_offset == src_block_size) {
      src_block_index++;
      src_block_offset = 0;
    }
    if (dst_block_offset == dst_block_size) {
      dst_block_index++;
      dst_block_offset = 0;
    }
  }
  std::vector<memory::column_metadata> columns_copy = host_table->columns;
  auto new_host_table                               = memory::host_table_allocation::create(
    std::move(dst_allocation), std::move(columns_copy), data_size);
  return std::make_unique<host_data_representation>(
    std::move(new_host_table), const_cast<memory::memory_space*>(target_memory_space));
}

// =============================================================================
// Disk <-> Host converters
// =============================================================================
//
// Every DISK converter moves one whole batch file with one vectored request through the io context
// of the DISK memory space (detail::resolve_disk_io_context) and waits for it before returning, so
// the conversion stays synchronous for its callers. All slices are built before the request is
// submitted, and the helpers in disk_io.hpp await the request before they return or unwind: once
// they are left, no I/O touches the source/destination buffers or the file any more.

static_assert(DISK_FILE_ALIGNMENT % io::IO_BLOCK_SIZE == 0,
              "disk file buffers must start on O_DIRECT / GPUDirect Storage boundaries");

/**
 * @brief RAII guard that deletes a file on destruction unless released.
 *
 * Ensures partial files are cleaned up if an exception occurs during writing.
 */
struct disk_file_guard {
  std::string path;
  bool released{false};
  ~disk_file_guard()
  {
    if (!released && !path.empty()) {
      std::error_code ec;
      std::filesystem::remove(path, ec);
    }
  }
  void release() { released = true; }
};

/**
 * @brief Append the write slices that store a host-block range at a file offset.
 *
 * A buffer may span host blocks; it becomes one slice per block it touches.
 */
static void collect_host_write_slices(
  const memory::fixed_size_host_memory_resource::multiple_blocks_allocation& alloc,
  std::size_t alloc_offset,
  std::size_t size,
  std::size_t file_offset,
  std::vector<io::write_slice>& slices)
{
  const std::size_t block_size = alloc.block_size();
  std::size_t block_idx        = alloc_offset / block_size;
  std::size_t block_off        = alloc_offset % block_size;
  std::size_t written          = 0;
  while (written < size) {
    std::size_t remaining = size - written;
    std::size_t avail     = block_size - block_off;
    std::size_t chunk     = std::min(remaining, avail);
    auto block            = alloc.at(block_idx);
    slices.emplace_back(file_offset + written,
                        chunk,
                        reinterpret_cast<std::uint8_t const*>(block.data() + block_off));
    written += chunk;
    block_off += chunk;
    if (block_off == block_size) {
      ++block_idx;
      block_off = 0;
    }
  }
}

/**
 * @brief Append the read slices that load a file range into a host-block range.
 *
 * A buffer may span host blocks; it becomes one slice per block it touches.
 */
static void collect_host_read_slices(
  std::size_t file_offset,
  std::size_t size,
  memory::fixed_size_host_memory_resource::multiple_blocks_allocation& alloc,
  std::size_t alloc_offset,
  std::vector<io::slice>& slices)
{
  const std::size_t block_size = alloc.block_size();
  std::size_t block_idx        = alloc_offset / block_size;
  std::size_t block_off        = alloc_offset % block_size;
  std::size_t read_total       = 0;
  while (read_total < size) {
    std::size_t remaining = size - read_total;
    std::size_t avail     = block_size - block_off;
    std::size_t chunk     = std::min(remaining, avail);
    auto block            = alloc.at(block_idx);
    slices.emplace_back(
      file_offset + read_total, chunk, reinterpret_cast<std::uint8_t*>(block.data() + block_off));
    read_total += chunk;
    block_off += chunk;
    if (block_off == block_size) {
      ++block_idx;
      block_off = 0;
    }
  }
}

/**
 * @brief Recompute column_metadata offsets for 4KB-aligned disk file layout.
 *
 * Copies all non-offset fields from src, then assigns null_mask_offset and data_offset
 * to 4KB-aligned positions within the file. The cursor is advanced past each buffer.
 */
static memory::column_metadata recompute_file_offsets(const memory::column_metadata& src,
                                                      std::size_t& cursor)
{
  memory::column_metadata dst{};
  dst.type_id    = src.type_id;
  dst.num_rows   = src.num_rows;
  dst.null_count = src.null_count;
  dst.scale      = src.scale;

  dst.has_null_mask  = src.has_null_mask;
  dst.null_mask_size = src.null_mask_size;
  if (src.has_null_mask && src.null_mask_size > 0) {
    cursor               = rmm::align_up(cursor, DISK_FILE_ALIGNMENT);
    dst.null_mask_offset = cursor;
    cursor += src.null_mask_size;
  } else {
    dst.null_mask_offset = 0;
  }

  dst.has_data  = src.has_data;
  dst.data_size = src.data_size;
  if (src.has_data && src.data_size > 0) {
    cursor          = rmm::align_up(cursor, DISK_FILE_ALIGNMENT);
    dst.data_offset = cursor;
    cursor += src.data_size;
  } else {
    dst.data_offset = 0;
  }

  dst.children.reserve(src.children.size());
  for (const auto& child : src.children) {
    dst.children.push_back(recompute_file_offsets(child, cursor));
  }
  return dst;
}

/**
 * @brief Recursively collect the write slices of a column's buffers in host blocks.
 */
static void collect_column_write_slices(
  const memory::column_metadata& src_meta,
  const memory::column_metadata& disk_meta,
  const memory::fixed_size_host_memory_resource::multiple_blocks_allocation& alloc,
  std::vector<io::write_slice>& slices)
{
  if (src_meta.has_null_mask && src_meta.null_mask_size > 0) {
    collect_host_write_slices(alloc,
                              src_meta.null_mask_offset,
                              src_meta.null_mask_size,
                              disk_meta.null_mask_offset,
                              slices);
  }
  if (src_meta.has_data && src_meta.data_size > 0) {
    collect_host_write_slices(
      alloc, src_meta.data_offset, src_meta.data_size, disk_meta.data_offset, slices);
  }
  for (std::size_t i = 0; i < src_meta.children.size(); ++i) {
    collect_column_write_slices(src_meta.children[i], disk_meta.children[i], alloc, slices);
  }
}

/**
 * @brief Recompute column_metadata offsets for 8-byte-aligned host block layout.
 *
 * Used when reconstructing host_table_allocation from disk data.
 */
static memory::column_metadata recompute_host_offsets(const memory::column_metadata& src,
                                                      std::size_t& cursor)
{
  memory::column_metadata dst{};
  dst.type_id    = src.type_id;
  dst.num_rows   = src.num_rows;
  dst.null_count = src.null_count;
  dst.scale      = src.scale;

  dst.has_null_mask  = src.has_null_mask;
  dst.null_mask_size = src.null_mask_size;
  if (src.has_null_mask && src.null_mask_size > 0) {
    cursor               = align_up_fast(cursor, 8u);
    dst.null_mask_offset = cursor;
    cursor += src.null_mask_size;
  } else {
    dst.null_mask_offset = 0;
  }

  dst.has_data  = src.has_data;
  dst.data_size = src.data_size;
  if (src.has_data && src.data_size > 0) {
    cursor          = align_up_fast(cursor, 8u);
    dst.data_offset = cursor;
    cursor += src.data_size;
  } else {
    dst.data_offset = 0;
  }

  dst.children.reserve(src.children.size());
  for (const auto& child : src.children) {
    dst.children.push_back(recompute_host_offsets(child, cursor));
  }
  return dst;
}

/**
 * @brief Recursively collect the read slices that load a column's buffers into host blocks.
 */
static void collect_column_read_slices(
  const memory::column_metadata& disk_meta,
  const memory::column_metadata& host_meta,
  memory::fixed_size_host_memory_resource::multiple_blocks_allocation& alloc,
  std::vector<io::slice>& slices)
{
  if (disk_meta.has_null_mask && disk_meta.null_mask_size > 0) {
    collect_host_read_slices(disk_meta.null_mask_offset,
                             disk_meta.null_mask_size,
                             alloc,
                             host_meta.null_mask_offset,
                             slices);
  }
  if (disk_meta.has_data && disk_meta.data_size > 0) {
    collect_host_read_slices(
      disk_meta.data_offset, disk_meta.data_size, alloc, host_meta.data_offset, slices);
  }
  for (std::size_t i = 0; i < disk_meta.children.size(); ++i) {
    collect_column_read_slices(disk_meta.children[i], host_meta.children[i], alloc, slices);
  }
}

/**
 * @brief Where a batch's buffers go in a DISK file or REMOTE object, and the slices that write
 *        them.
 *
 * The layout is the same for both: column buffers at 4KB-aligned offsets from 0, no header; the
 * column metadata (with those offsets) stays in memory with the representation.
 */
struct batch_layout {
  std::vector<memory::column_metadata> columns;
  std::size_t size{0};
  std::vector<io::write_slice> slices;
  /// A GPU source that is a slice is serialized from this compacted copy (see
  /// serializable_view()); keep the layout alive until the write completed.
  std::unique_ptr<cudf::table> compacted{};
};

/// The batch layout of a host table; the slices point into its host blocks.
static batch_layout layout_host_batch(const host_data_representation& host_source)
{
  const auto& host_table = host_source.get_host_table();
  batch_layout layout;
  std::size_t cursor = 0;
  layout.columns.reserve(host_table->columns.size());
  for (const auto& col : host_table->columns) {
    layout.columns.push_back(recompute_file_offsets(col, cursor));
  }
  layout.size = cursor;
  for (std::size_t i = 0; i < host_table->columns.size(); ++i) {
    collect_column_write_slices(
      host_table->columns[i], layout.columns[i], *host_table->allocation, layout.slices);
  }
  return layout;
}

/**
 * @brief Rebuild a host_data_representation from a batch's column metadata.
 *
 * Allocates the host blocks and hands every buffer's read to @p read_into (one call, all
 * slices), which must fill them before it returns.
 */
template <typename ReadFn>
static std::unique_ptr<idata_representation> restore_host_batch(
  const std::vector<memory::column_metadata>& batch_columns,
  const memory::memory_space* target_memory_space,
  memory::reservation* reservation,
  ReadFn&& read_into)
{
  // Compute host allocation size with 8-byte alignment
  std::size_t host_cursor = 0;
  std::vector<memory::column_metadata> host_columns;
  host_columns.reserve(batch_columns.size());
  for (const auto& col : batch_columns) {
    host_columns.push_back(recompute_host_offsets(col, host_cursor));
  }
  std::size_t total_host_size = host_cursor;

  // Allocate host blocks
  auto mr = target_memory_space->get_memory_resource_as<memory::fixed_size_host_memory_resource>();
  if (mr == nullptr) {
    throw std::runtime_error(
      "Target HOST memory_space does not have a fixed_size_host_memory_resource");
  }
  auto allocation = mr->allocate_multiple_blocks(total_host_size, reservation);

  // Read every column buffer into the host blocks with one request
  std::vector<io::slice> slices;
  for (std::size_t i = 0; i < batch_columns.size(); ++i) {
    collect_column_read_slices(batch_columns[i], host_columns[i], *allocation, slices);
  }
  std::forward<ReadFn>(read_into)(std::span<io::slice const>{slices});

  auto host_alloc = memory::host_table_allocation::create(
    std::move(allocation), std::move(host_columns), total_host_size);
  return std::make_unique<host_data_representation>(
    std::move(host_alloc), const_cast<memory::memory_space*>(target_memory_space));
}

/**
 * @brief Convert host_data_representation to disk_data_representation.
 *
 * Writes every column buffer at its 4KB-aligned file offset with one host write request; the
 * column metadata stays in memory. An RAII file guard removes the file if anything fails.
 */
static std::unique_ptr<idata_representation> convert_host_data_to_disk(
  const idata_representation& source,
  const memory::memory_space* target_memory_space,
  [[maybe_unused]] ::cuda::stream_ref stream,
  [[maybe_unused]] memory::reservation* reservation)
{
  nvtx_scope convert_range{"convert:host_to_disk"};
  auto const io_context = detail::resolve_disk_io_context(*target_memory_space);
  auto layout           = layout_host_batch(source.cast<host_data_representation>());

  // Generate unique file path under the disk memory space's mount directory
  auto mount_path = target_memory_space->get_disk_mount_path();
  auto file_path  = memory::generate_disk_file_path(mount_path);

  disk_file_guard guard{file_path};
  detail::write_disk_file_host(*io_context, file_path, layout.slices, layout.size);
  guard.release();

  auto disk_table = std::make_unique<memory::disk_table_allocation>(
    std::move(file_path), std::move(layout.columns), layout.size);
  return std::make_unique<disk_data_representation>(
    std::move(disk_table), const_cast<memory::memory_space&>(*target_memory_space));
}

/**
 * @brief Convert disk_data_representation to host_data_representation.
 *
 * Uses in-memory column metadata from the source disk representation, allocates host blocks,
 * and reads every buffer from the disk file with one host read request.
 */
static std::unique_ptr<idata_representation> convert_disk_to_host_data(
  const idata_representation& source,
  const memory::memory_space* target_memory_space,
  [[maybe_unused]] ::cuda::stream_ref stream,
  memory::reservation* reservation)
{
  nvtx_scope convert_range{"convert:disk_to_host"};
  auto const io_context  = detail::resolve_disk_io_context(source.get_memory_space());
  const auto& disk_table = source.cast<disk_data_representation>().get_disk_table();
  return restore_host_batch(
    disk_table.columns, target_memory_space, reservation, [&](std::span<io::slice const> slices) {
      detail::read_disk_file_host(*io_context, disk_table.file_path, slices);
    });
}

// =============================================================================
// Disk <-> GPU converters
// =============================================================================

/// Call after making the source device current so default handles resolve on it.
static void require_stream_on_device(::cuda::stream_ref stream,
                                     int source_device,
                                     std::string_view converter)
{
  int stream_device = -1;
  CUCASCADE_CUDA_TRY(cudaStreamGetDevice(stream.get(), &stream_device));
  if (stream_device != source_device) {
    CUCASCADE_FAIL(std::string{converter} + ": stream belongs to CUDA device " +
                   std::to_string(stream_device) + " but the source table lives on device " +
                   std::to_string(source_device));
  }
}

/**
 * @brief Recursively collect the write slices of a GPU column's buffers.
 *
 * All (device pointer, size, file offset) triples of the table go into one device write request.
 */
static void collect_gpu_column_write_slices(const cudf::column_view& col,
                                            const memory::column_metadata& disk_meta,
                                            std::vector<io::write_slice>& slices)
{
  if (disk_meta.has_null_mask && disk_meta.null_mask_size > 0) {
    slices.emplace_back(disk_meta.null_mask_offset,
                        disk_meta.null_mask_size,
                        reinterpret_cast<std::uint8_t const*>(col.null_mask()));
  }
  if (disk_meta.has_data && disk_meta.data_size > 0) {
    slices.emplace_back(disk_meta.data_offset, disk_meta.data_size, col.data<uint8_t>());
  }
  for (cudf::size_type i = 0; i < col.num_children(); ++i) {
    collect_gpu_column_write_slices(
      col.child(i), disk_meta.children[static_cast<std::size_t>(i)], slices);
  }
}

/**
 * @brief The batch layout of a GPU table; the slices point at its device buffers (or at those of
 *        a compacted copy, when the table is a slice).
 *
 * Waits for the table's writer on @p stream first: planning reads string offsets on the device,
 * and the writes that follow are stream-ordered after @p stream.
 *
 * @pre The source table's device is current, and stays current through the write that consumes
 *      the layout: a null @p stream names the current device's default stream, so the writer
 *      wait, the planning and the write must all resolve it on the source device.
 */
static batch_layout layout_gpu_batch(const gpu_table_representation& gpu_source,
                                     ::cuda::stream_ref stream)
{
  gpu_source.wait_for_writer(stream);
  batch_layout layout;
  cudf::table_view tv = serializable_view(gpu_source.get_table_view(),
                                          stream,
                                          gpu_source.get_memory_space().get_default_allocator(),
                                          layout.compacted);

  // Plan column layout with 8-byte aligned offsets (same as host path)
  std::size_t plan_offset = 0;
  std::vector<memory::column_metadata> planned_columns;
  planned_columns.reserve(static_cast<std::size_t>(tv.num_columns()));
  for (cudf::size_type i = 0; i < tv.num_columns(); ++i) {
    planned_columns.push_back(plan_column_copy(tv.column(i), plan_offset, stream));
  }

  // Recompute with 4KB-aligned offsets starting at offset 0.
  // No header or serialized metadata is written — metadata stays in memory.
  std::size_t cursor = 0;
  layout.columns.reserve(planned_columns.size());
  for (const auto& col : planned_columns) {
    layout.columns.push_back(recompute_file_offsets(col, cursor));
  }
  layout.size = cursor;
  for (cudf::size_type i = 0; i < tv.num_columns(); ++i) {
    collect_gpu_column_write_slices(
      tv.column(i), layout.columns[static_cast<std::size_t>(i)], layout.slices);
  }
  return layout;
}

/**
 * @brief Convert gpu_table_representation to disk_data_representation.
 *
 * Writes every GPU column buffer straight from device memory with one device write request.
 * File contains only 4KB-aligned column data — metadata stays in memory.
 * @p stream must belong to the source table's device; a null handle names that device's
 * default stream.
 */
static std::unique_ptr<idata_representation> convert_gpu_to_disk(
  const idata_representation& source,
  const memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  [[maybe_unused]] memory::reservation* reservation)
{
  nvtx_scope convert_range{"convert:gpu_to_disk"};
  auto const io_context = detail::resolve_disk_io_context(*target_memory_space);
  auto& gpu_source      = source.cast<gpu_table_representation>();
  // The source device is current from the writer ordering through the write: a null `stream`
  // names the current device's default stream, so the stream the writer is waited on, the one
  // planning and compaction run on, and the one the io context reads the sources on must all be
  // resolved on the same device. The io context also binds device sources to the current device.
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{gpu_source.get_device_id()}};
  require_stream_on_device(stream, gpu_source.get_device_id(), "gpu_to_disk");
  // On a failure after work reading the source was enqueued, wait for it before unwinding.
  drain_on_unwind drain{stream};
  auto layout = layout_gpu_batch(gpu_source, stream);

  // Generate unique file path under the disk memory space's mount directory
  auto mount_path = target_memory_space->get_disk_mount_path();
  auto file_path  = memory::generate_disk_file_path(mount_path);

  disk_file_guard guard{file_path};
  detail::write_disk_file_device(*io_context, file_path, layout.slices, stream, layout.size);
  guard.release();

  auto disk_table = std::make_unique<memory::disk_table_allocation>(
    std::move(file_path), std::move(layout.columns), layout.size);
  return std::make_unique<disk_data_representation>(
    std::move(disk_table), const_cast<memory::memory_space&>(*target_memory_space));
}

/**
 * @brief Device buffers of one column read back from a disk file, mirroring its metadata tree.
 *
 * Only the buffers the column's type is rebuilt from are allocated (see allocate_disk_column).
 */
struct disk_column_buffers {
  explicit disk_column_buffers(cudf_compat::null_mask_buffer mask) : null_mask(std::move(mask)) {}

  cudf_compat::null_mask_buffer null_mask;
  rmm::device_buffer data{};
  std::vector<disk_column_buffers> children{};
};

/**
 * @brief Which children of a column the reconstruction uses, validating their count.
 *
 * STRING: [0] = offsets; LIST: [0] = offsets, [1] = values; STRUCT: all fields;
 * DICTIONARY32: [0] = indices, [1] = keys; every other type: none.
 */
static std::size_t used_child_count(const memory::column_metadata& meta)
{
  switch (as_cudf_type_id(meta.type_id)) {
    case cudf::type_id::STRING:
      // cudf::make_empty_column(STRING) has no children; it is recreated as such.
      if (meta.children.empty() && meta.num_rows == 0) { return 0; }
      if (meta.children.size() < 1) {
        throw std::invalid_argument(
          "reconstruct_column_from_disk: STRING column metadata must have at least one child "
          "(offsets)");
      }
      return 1;
    case cudf::type_id::LIST:
      if (meta.children.size() < 2) {
        throw std::invalid_argument(
          "reconstruct_column_from_disk: LIST column metadata must have two children (offsets, "
          "values)");
      }
      return 2;
    case cudf::type_id::STRUCT: return meta.children.size();
    case cudf::type_id::DICTIONARY32:
      if (meta.children.size() < 2) {
        throw std::invalid_argument(
          "reconstruct_column_from_disk: DICTIONARY32 column metadata must have two children "
          "(indices, keys)");
      }
      return 2;
    default: return 0;
  }
}

/// Whether the column's own data buffer is read (STRING chars and fixed-width values).
static bool reads_data_buffer(const memory::column_metadata& meta)
{
  auto const type = as_cudf_type_id(meta.type_id);
  return type != cudf::type_id::LIST && type != cudf::type_id::STRUCT &&
         type != cudf::type_id::DICTIONARY32 && meta.has_data && meta.data_size > 0;
}

/**
 * @brief Phase 1 of DISK -> GPU: allocate a column's device buffers and collect their reads.
 *
 * Allocations are stream-ordered on @p stream, so the reads enqueued on the same stream later
 * land after them. Nothing is read here; the slices point into the returned buffers, whose device
 * memory does not move when the tree is moved.
 */
static disk_column_buffers allocate_disk_column(const memory::column_metadata& meta,
                                                ::cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr,
                                                std::vector<io::slice>& slices)
{
  // create_null_mask(num_rows) allocates exactly cudf::bitmask_allocation_size_bytes(num_rows),
  // which is what the writer recorded as meta.null_mask_size. The buffer size is measured in
  // elements in newer cudf versions; disk extents and slices are measured in bytes.
  disk_column_buffers buffers{
    meta.has_null_mask
      ? cudf::create_null_mask(meta.num_rows, cudf::mask_state::UNINITIALIZED, stream, mr)
      : cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr)};
  if (meta.has_null_mask) {
    auto const bytes =
      std::min(cudf_compat::get_null_mask_size_bytes(buffers.null_mask), meta.null_mask_size);
    if (bytes > 0) {
      slices.emplace_back(meta.null_mask_offset,
                          bytes,
                          static_cast<std::uint8_t*>(static_cast<void*>(buffers.null_mask.data())));
    }
  }
  if (reads_data_buffer(meta)) {
    buffers.data = rmm::device_buffer(meta.data_size, stream, mr);
    slices.emplace_back(
      meta.data_offset, meta.data_size, static_cast<std::uint8_t*>(buffers.data.data()));
  }
  auto const children = used_child_count(meta);
  buffers.children.reserve(children);
  for (std::size_t i = 0; i < children; ++i) {
    buffers.children.push_back(allocate_disk_column(meta.children[i], stream, mr, slices));
  }
  return buffers;
}

/**
 * @brief Phase 3 of DISK -> GPU: build a cudf::column from its metadata and filled buffers.
 */
static std::unique_ptr<cudf::column> assemble_disk_column(const memory::column_metadata& meta,
                                                          disk_column_buffers& buffers,
                                                          ::cuda::stream_ref stream,
                                                          rmm::device_async_resource_ref mr)
{
  const cudf::size_type null_count = meta.has_null_mask ? meta.null_count : 0;
  auto const type                  = as_cudf_type_id(meta.type_id);

  if (type == cudf::type_id::STRING) {
    // See reconstruct_column for the empty-strings-column rationale. cudf::make_empty_column(
    // STRING) produces a column with no children; recreate it instead of erroring.
    if (buffers.children.empty()) {
      return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
    }
    // Reconstruct offsets and cast to INT64 if needed (cudf 26.06+ requires INT64 offsets)
    auto offsets_col = assemble_disk_column(meta.children[0], buffers.children[0], stream, mr);
    if (offsets_col->type().id() == cudf::type_id::INT32) {
      offsets_col =
        cudf::cast(offsets_col->view(), cudf::data_type{cudf::type_id::INT64}, stream, mr);
    }
    return cudf::make_strings_column(meta.num_rows,
                                     std::move(offsets_col),
                                     std::move(buffers.data),
                                     null_count,
                                     std::move(buffers.null_mask));
  }

  if (type == cudf::type_id::LIST) {
    // LIST offsets must remain INT32 (cudf::size_type); see reconstruct_column for the rationale.
    // Construct directly (like the STRUCT branch below) to preserve the INT32 offsets and skip
    // make_lists_column's purge_nonempty_nulls. Our serialized data already has consistent null
    // masks.
    std::vector<std::unique_ptr<cudf::column>> children;
    children.reserve(2);
    children.push_back(assemble_disk_column(meta.children[0], buffers.children[0], stream, mr));
    children.push_back(assemble_disk_column(meta.children[1], buffers.children[1], stream, mr));
    return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::LIST},
                                          meta.num_rows,
                                          rmm::device_buffer{},
                                          std::move(buffers.null_mask),
                                          null_count,
                                          std::move(children));
  }

  if (type == cudf::type_id::STRUCT) {
    std::vector<std::unique_ptr<cudf::column>> fields;
    fields.reserve(meta.children.size());
    for (std::size_t i = 0; i < meta.children.size(); ++i) {
      fields.push_back(assemble_disk_column(meta.children[i], buffers.children[i], stream, mr));
    }
    return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRUCT},
                                          meta.num_rows,
                                          rmm::device_buffer{},
                                          std::move(buffers.null_mask),
                                          null_count,
                                          std::move(fields));
  }

  if (type == cudf::type_id::DICTIONARY32) {
    // cudf DICTIONARY32 children order: [0]=indices, [1]=keys.
    // make_dictionary_column parameter order: (keys, indices, ...).
    auto indices_col = assemble_disk_column(meta.children[0], buffers.children[0], stream, mr);
    auto keys_col    = assemble_disk_column(meta.children[1], buffers.children[1], stream, mr);
    return cudf::make_dictionary_column(
      std::move(keys_col), std::move(indices_col), std::move(buffers.null_mask), null_count);
  }

  const cudf::data_type dtype = cudf::is_fixed_point(cudf::data_type{type})
                                  ? cudf::data_type{type, meta.scale}
                                  : cudf::data_type{type};
  return std::make_unique<cudf::column>(
    dtype, meta.num_rows, std::move(buffers.data), std::move(buffers.null_mask), null_count);
}

/**
 * @brief Rebuild a gpu_table_representation from a batch's column metadata.
 *
 * Allocates every device buffer of the table, hands all their reads to @p read_into (ONE call,
 * made with the target device current; it must fill them before it returns, see below), then
 * assembles the cudf columns.
 */
template <typename ReadFn>
static std::unique_ptr<idata_representation> restore_gpu_batch(
  const std::vector<memory::column_metadata>& batch_columns,
  const memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  ReadFn&& read_into)
{
  // Set CUDA device to target GPU (RAII restores previous device on scope exit). The io context
  // binds device destinations to the current device.
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{target_memory_space->get_device_id()}};

  auto mr = target_memory_space->get_default_allocator();

  // Phase 1: allocate every buffer of the table (stream-ordered) and collect one slice per buffer.
  // Declared before the read so the buffers outlive it on every path.
  std::vector<io::slice> slices;
  std::vector<disk_column_buffers> buffers;
  buffers.reserve(batch_columns.size());
  for (const auto& col_meta : batch_columns) {
    buffers.push_back(allocate_disk_column(col_meta, stream, mr, slices));
  }

  // Phase 2: one batched device read of the whole table, awaited before returning or unwinding.
  std::forward<ReadFn>(read_into)(std::span<io::slice const>{slices});

  // Phase 3: assemble the columns; the casts it may enqueue are ordered on `stream`.
  std::vector<std::unique_ptr<cudf::column>> gpu_columns;
  gpu_columns.reserve(batch_columns.size());
  for (std::size_t i = 0; i < batch_columns.size(); ++i) {
    gpu_columns.push_back(assemble_disk_column(batch_columns[i], buffers[i], stream, mr));
  }

  // No host-side stream.sync() is needed here. It rests on the ioctx read contract, not on how a
  // particular backend implements it:
  //  1. The read future resolves only after every byte is in device memory, on whichever stream
  //     the backend used (uring / rest: their H2D copies on `stream`, settled once their event
  //     completes; cuFile: a synchronous cuFileRead, or POSIX + H2D on its own copy stream,
  //     settled after the copy events). So the buffers are complete when `read_into` returns.
  //  2. A backend writes the destinations only after the work enqueued on `stream` before the read
  //     (uring / rest copy on `stream`; cuFile waits for `stream` before it writes from
  //     elsewhere). The destinations are stream-ordered allocations on `stream`, so this ordering
  //     is required.
  // Everything assemble_disk_column enqueues (offset casts) is ordered on `stream` after that, and
  // make_written_on() records the table's writer event on `stream` behind it, which every reader
  // waits on before touching the data.
  auto new_table = std::make_unique<cudf::table>(std::move(gpu_columns));
  return gpu_table_representation::make_written_on(
    std::move(new_table), *const_cast<memory::memory_space*>(target_memory_space), stream);
}

/**
 * @brief Convert disk_data_representation to gpu_table_representation.
 *
 * Allocates every device buffer of the table, reads them all from the file with ONE batched
 * device read request, then assembles the cudf columns. Column metadata comes from the in-memory
 * disk_table_allocation.
 */
static std::unique_ptr<idata_representation> convert_disk_to_gpu(
  const idata_representation& source,
  const memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  [[maybe_unused]] memory::reservation* reservation)
{
  nvtx_scope convert_range{"convert:disk_to_gpu"};
  auto const io_context  = detail::resolve_disk_io_context(source.get_memory_space());
  const auto& disk_table = source.cast<disk_data_representation>().get_disk_table();
  return restore_gpu_batch(
    disk_table.columns, target_memory_space, stream, [&](std::span<io::slice const> slices) {
      detail::read_disk_file_device(*io_context, disk_table.file_path, slices, stream);
    });
}

// =============================================================================
// Remote <-> Host / GPU converters
// =============================================================================
//
// A REMOTE batch is one object with the DISK file layout, written through the io context of the
// REMOTE memory space (there is no default context: a space without one is a configuration
// error). The steps differ from DISK only where an object store differs from a file
// (remote_io.hpp): the name is generated without I/O and opened create_new, the object is
// committed by a flush, a failed write is cleaned up by destroying the unflushed object (which
// aborts its upload), the object is read back through the committed write object (its size is
// known: no HEAD), and the representation's release deletes the object asynchronously.

/// The written object of a REMOTE representation (null when it was not made by a converter).
static const detail::remote_object& remote_object_of(const remote_data_representation& source)
{
  auto const& object = source.get_remote_object();
  if (object == nullptr || object->context == nullptr || object->object == nullptr) {
    throw std::logic_error("remote tier: the REMOTE representation of '" + source.get_object_uri() +
                           "' has no readable object");
  }
  return *object;
}

/**
 * @brief Convert host_data_representation to remote_data_representation.
 *
 * Writes the host table as one object (one host write request per staging window, normally one)
 * and commits it. The source is untouched on failure.
 */
static std::unique_ptr<idata_representation> convert_host_data_to_remote(
  const idata_representation& source,
  const memory::memory_space* target_memory_space,
  [[maybe_unused]] ::cuda::stream_ref stream,
  [[maybe_unused]] memory::reservation* reservation)
{
  nvtx_scope convert_range{"convert:host_to_remote"};
  auto io_context = detail::require_remote_io_context(*target_memory_space, /*device=*/false);
  auto layout     = layout_host_batch(source.cast<host_data_representation>());
  auto object     = detail::write_remote_object(std::move(io_context),
                                            target_memory_space->get_remote_uri_prefix(),
                                            layout.slices,
                                            layout.size,
                                            std::nullopt);
  auto table      = std::make_unique<memory::disk_table_allocation>(
    object->uri, std::move(layout.columns), layout.size);
  return std::make_unique<remote_data_representation>(
    std::move(table), const_cast<memory::memory_space&>(*target_memory_space), std::move(object));
}

/**
 * @brief Convert remote_data_representation to host_data_representation.
 *
 * Reads every buffer of the object into fresh host blocks with one request. On failure the
 * REMOTE representation stays valid (the conversion may be retried).
 */
static std::unique_ptr<idata_representation> convert_remote_to_host_data(
  const idata_representation& source,
  const memory::memory_space* target_memory_space,
  [[maybe_unused]] ::cuda::stream_ref stream,
  memory::reservation* reservation)
{
  nvtx_scope convert_range{"convert:remote_to_host"};
  auto& remote_source = source.cast<remote_data_representation>();
  auto const& object  = remote_object_of(remote_source);
  return restore_host_batch(
    remote_source.get_object_table().columns,
    target_memory_space,
    reservation,
    [&](std::span<io::slice const> slices) { detail::read_remote_object_host(object, slices); });
}

/**
 * @brief Convert gpu_table_representation to remote_data_representation.
 *
 * Writes every GPU column buffer as one object: the REST context copies the device sources into
 * pinned part buffers on @p stream and uploads them; the object is committed by the flush.
 */
static std::unique_ptr<idata_representation> convert_gpu_to_remote(
  const idata_representation& source,
  const memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  [[maybe_unused]] memory::reservation* reservation)
{
  nvtx_scope convert_range{"convert:gpu_to_remote"};
  auto io_context  = detail::require_remote_io_context(*target_memory_space, /*device=*/true);
  auto& gpu_source = source.cast<gpu_table_representation>();
  // The source device is current from the writer ordering through the write (see
  // layout_gpu_batch()); the io context also binds device sources to the current device.
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{gpu_source.get_device_id()}};
  require_stream_on_device(stream, gpu_source.get_device_id(), "gpu_to_remote");
  // On a failure after work reading the source was enqueued, wait for it before unwinding.
  drain_on_unwind drain{stream};
  auto layout = layout_gpu_batch(gpu_source, stream);
  auto object = detail::write_remote_object(std::move(io_context),
                                            target_memory_space->get_remote_uri_prefix(),
                                            layout.slices,
                                            layout.size,
                                            stream);
  auto table  = std::make_unique<memory::disk_table_allocation>(
    object->uri, std::move(layout.columns), layout.size);
  return std::make_unique<remote_data_representation>(
    std::move(table), const_cast<memory::memory_space&>(*target_memory_space), std::move(object));
}

/**
 * @brief Convert remote_data_representation to gpu_table_representation.
 *
 * Allocates every device buffer of the table and reads them from the object (one batched device
 * read per staging window, normally one), then assembles the cudf columns. On failure the
 * REMOTE representation stays valid.
 */
static std::unique_ptr<idata_representation> convert_remote_to_gpu(
  const idata_representation& source,
  const memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  [[maybe_unused]] memory::reservation* reservation)
{
  nvtx_scope convert_range{"convert:remote_to_gpu"};
  auto& remote_source = source.cast<remote_data_representation>();
  auto const& object  = remote_object_of(remote_source);
  return restore_gpu_batch(remote_source.get_object_table().columns,
                           target_memory_space,
                           stream,
                           [&](std::span<io::slice const> slices) {
                             detail::read_remote_object_device(object, slices, stream);
                           });
}

}  // namespace

void register_builtin_converters(representation_converter_registry& registry)
{
  // GPU -> GPU (cross-device copy). convert_gpu_to_gpu compacts a sliced source on its device
  // first, then copies each column buffer with cudaMemcpyPeerAsync when
  // cucascade::memory::probe_peer_dma_works() verifies the direction, and stages through pinned
  // host memory otherwise. The cached probe copies bytes between two mutually
  // granted private pools, so a verification failure means the pool peer route delivered wrong
  // bytes, as on platforms whose direct route reports success but leaves stale data.
  registry.register_converter<gpu_table_representation, gpu_table_representation>(
    convert_gpu_to_gpu);

  // GPU -> HOST
  registry.register_converter<gpu_table_representation, host_data_packed_representation>(
    convert_gpu_to_host);

  // HOST -> GPU
  registry.register_converter<host_data_packed_representation, gpu_table_representation>(
    convert_host_to_gpu);

  // HOST -> HOST (cross-device copy)
  registry.register_converter<host_data_packed_representation, host_data_packed_representation>(
    convert_host_to_host);

  // GPU -> HOST FAST (direct buffer copy, no intermediate GPU allocation)
  registry.register_converter<gpu_table_representation, host_data_representation>(
    convert_gpu_to_host_fast);

  // HOST FAST -> GPU
  registry.register_converter<host_data_representation, gpu_table_representation>(
    convert_host_fast_to_gpu);

  // HOST FAST -> HOST FAST (cross-device copy)
  registry.register_converter<host_data_representation, host_data_representation>(
    convert_host_fast_to_host_fast);

  // HOST DATA -> DISK (io context resolved from the target disk memory_space)
  registry.register_converter<host_data_representation, disk_data_representation>(
    convert_host_data_to_disk);

  // DISK -> HOST DATA (io context resolved from the source disk memory_space)
  registry.register_converter<disk_data_representation, host_data_representation>(
    convert_disk_to_host_data);

  // GPU -> DISK (io context resolved from the target disk memory_space)
  registry.register_converter<gpu_table_representation, disk_data_representation>(
    convert_gpu_to_disk);

  // DISK -> GPU (io context resolved from the source disk memory_space)
  registry.register_converter<disk_data_representation, gpu_table_representation>(
    convert_disk_to_gpu);

  // HOST DATA -> REMOTE (io context of the target REMOTE memory_space; required)
  registry.register_converter<host_data_representation, remote_data_representation>(
    convert_host_data_to_remote);

  // REMOTE -> HOST DATA (through the context that wrote the object)
  registry.register_converter<remote_data_representation, host_data_representation>(
    convert_remote_to_host_data);

  // GPU -> REMOTE (io context of the target REMOTE memory_space; required)
  registry.register_converter<gpu_table_representation, remote_data_representation>(
    convert_gpu_to_remote);

  // REMOTE -> GPU (through the context that wrote the object)
  registry.register_converter<remote_data_representation, gpu_table_representation>(
    convert_remote_to_gpu);
}

}  // namespace cucascade
