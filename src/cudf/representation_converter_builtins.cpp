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

#include "builtin_converters_detail.hpp"
#include "cudf_compat.hpp"
#include "staged_copy.hpp"

#include <cucascade/cuda/event.hpp>
#include <cucascade/cudf/builtin_converters.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/cudf/host_table.hpp>
#include <cucascade/cudf/host_table_packed.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/data/disk_file_format.hpp>
#include <cucascade/data/disk_io_backend.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/error.hpp>
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
#include <cudf/utilities/bit.hpp>
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
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
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
 * @brief Recursively plan the buffer layout for one column, filling in column_metadata.
 *
 * Advances @p current_offset by the bytes needed for the column's null mask and data buffer
 * (8-byte aligned), then recurses into any children.
 *
 * @note We assume the column view has offset == 0 (i.e., the table is not a slice).
 */
static memory::column_metadata plan_column_copy(const cudf::column_view& col,
                                                std::size_t& current_offset,
                                                ::cuda::stream_ref stream)
{
  assert(col.offset() == 0 && "column_view with non-zero offset is not supported");

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
  // Planning reads string offsets on the device, so it must already be ordered after the writer.
  gpu_source.wait_for_writer(stream);
  const cudf::table_view view = gpu_source.get_table_view();

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
 * @brief Enqueue a direct copy of @p size_in_bytes bytes from @p src_ptr on @p src_device into
 * @p buf on @p dst_device, on @p target_stream.
 *
 * Only for device pairs whose direct route cucascade::memory::probe_peer_dma_works() verified.
 */
template <typename Buffer>
Buffer peer_copy_into_async(Buffer buf,
                            const void* src_ptr,
                            int src_device,
                            std::size_t size_in_bytes,
                            int dst_device,
                            ::cuda::stream_ref target_stream)
{
  if (size_in_bytes == 0 || src_ptr == nullptr) { return buf; }

  CUCASCADE_CUDA_TRY(cudaMemcpyPeerAsync(
    buf.data(), dst_device, src_ptr, src_device, size_in_bytes, target_stream.get()));
  return buf;
}

/**
 * @brief Recursively rebuild a cudf::column on the target device, peer-copying each
 * leaf buffer (null mask, data, offsets, children) directly. Mirrors the structure
 * of reconstruct_column() (host→gpu) but reads from a source cudf::column_view on a
 * peer device using cudaMemcpyPeerAsync.
 *
 * Asynchronously reconstructs on the supplied stream, but may block internally:
 * Data and offset buffers are issued asynchronously on @p target_stream; the caller
 * must sync that stream before the resulting cudf::table is observed by another stream.
 *
 * @note Assumes the source column_view has offset == 0 (no slicing), matching the
 *       same constraint imposed by plan_column_copy() on the GPU↔Host fast path.
 */
std::unique_ptr<cudf::column> reconstruct_column_p2p_async(const cudf::column_view& src,
                                                           int src_device,
                                                           int dst_device,
                                                           ::cuda::stream_ref stream,
                                                           rmm::device_async_resource_ref mr)
{
  assert(src.offset() == 0 && "column_view with non-zero offset is not supported");

  auto null_mask = cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr);
  if (src.nullable()) {
    null_mask = cudf::create_null_mask(src.size(), cudf::mask_state::UNINITIALIZED, stream, mr);
    if (std::size_t const size_in_bytes = cudf_compat::get_null_mask_size_bytes(null_mask);
        size_in_bytes > 0) {
      null_mask = peer_copy_into_async(
        std::move(null_mask), src.null_mask(), src_device, size_in_bytes, dst_device, stream);
    }
    // no sync needed here since cudf calls are ordered on the explicitly passed stream, and cudf
    // factories that do not take a stream do no GPU side work.
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
    auto offsets_col =
      reconstruct_column_p2p_async(src.child(0), src_device, dst_device, stream, mr);

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
      const cudf::column_view offsets_view = offsets_col->view();
      const int last_idx                   = offsets_view.size() - 1;
      int64_t chars_bytes                  = 0;
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
        const auto data_size_in_bytes = static_cast<std::size_t>(chars_bytes);
        chars_buf = peer_copy_into_async(rmm::device_buffer(data_size_in_bytes, stream, mr),
                                         src.data<char>(),
                                         src_device,
                                         data_size_in_bytes,
                                         dst_device,
                                         stream);
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
    auto offsets_col =
      reconstruct_column_p2p_async(src.child(0), src_device, dst_device, stream, mr);
    auto values_col =
      reconstruct_column_p2p_async(src.child(1), src_device, dst_device, stream, mr);
    return cudf::make_lists_column(
      src.size(), std::move(offsets_col), std::move(values_col), null_count, std::move(null_mask));
  }

  if (src.type().id() == cudf::type_id::STRUCT) {
    std::vector<std::unique_ptr<cudf::column>> fields;
    fields.reserve(static_cast<std::size_t>(src.num_children()));
    for (cudf::size_type i = 0; i < src.num_children(); ++i) {
      fields.push_back(
        reconstruct_column_p2p_async(src.child(i), src_device, dst_device, stream, mr));
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
    auto indices_col =
      reconstruct_column_p2p_async(src.child(0), src_device, dst_device, stream, mr);
    auto keys_col = reconstruct_column_p2p_async(src.child(1), src_device, dst_device, stream, mr);
    return cudf::make_dictionary_column(
      std::move(keys_col), std::move(indices_col), std::move(null_mask), null_count);
  }

  // Fixed-width / fixed-point — single flat data buffer.
  rmm::device_buffer data_buf{};
  if (src.size() > 0 && src.head() != nullptr) {
    auto const data_size = static_cast<std::size_t>(src.size()) * cudf::size_of(src.type());
    data_buf             = peer_copy_into_async(rmm::device_buffer(data_size, stream, mr),
                                    src.head(),
                                    src_device,
                                    data_size,
                                    dst_device,
                                    stream);
  }
  return std::make_unique<cudf::column>(
    src.type(), src.size(), std::move(data_buf), std::move(null_mask), null_count);
}

// =============================================================================
// Host-staged GPU -> GPU converter (device pairs without a working direct route)
// =============================================================================

/**
 * @brief One column of a table that is about to be copied through host memory.
 *
 * Built in two passes over the source. The first validates the column and records what it holds;
 * the second, once the sizes of STRING characters are known, allocates the destination buffers.
 */
struct staged_column {
  cudf::data_type type{cudf::type_id::EMPTY};
  cudf::size_type rows{0};
  cudf::size_type null_count{0};

  void const* source_mask{nullptr};
  std::size_t mask_bytes{0};  ///< The bitmask words covering `rows`; zero if not nullable
  void const* source_data{nullptr};
  std::size_t data_bytes{0};  ///< Unknown for STRING characters until the size probe is read
  std::optional<std::size_t> size_probe;  ///< Which probe holds the size of the characters

  /// Set by the second pass; holds nothing if the column is not nullable.
  std::optional<cudf_compat::null_mask_buffer> mask;
  rmm::device_buffer data;
  std::vector<staged_column> children;

  /// A column without rows copies nothing: the second pass builds `empty` from `source`.
  cudf::column_view source;
  std::unique_ptr<cudf::column> empty;
};

/** @brief What the first planning pass learns about the table as a whole. */
struct staged_table_summary {
  std::vector<memory::detail::device_integer_probe> size_probes;
  bool reads_source{false};  ///< Whether any byte of the source is read at all
};

[[noreturn]] void reject_staged_column(std::string const& reason)
{
  throw std::invalid_argument("host-staged GPU to GPU conversion: " + reason);
}

/// A null mask that holds nothing, for a column that is not nullable. Allocates no device memory.
cudf_compat::null_mask_buffer make_absent_null_mask(::cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  return cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr);
}

/**
 * @brief An empty column with the type of @p view, built without touching the device.
 *
 * Nested types keep their shape: a LIST has an empty offsets child and an empty child of its
 * element type, a STRUCT has empty fields, and a DICTIONARY32 has empty indices and empty keys of
 * its key type (the keys of a dictionary without rows are not copied). A STRING has no children.
 */
std::unique_ptr<cudf::column> make_empty_like(cudf::column_view const& view,
                                              ::cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  auto const empty_leaf = [&](cudf::data_type type) {
    return std::make_unique<cudf::column>(
      type, 0, rmm::device_buffer{}, make_absent_null_mask(stream, mr), 0);
  };

  std::vector<std::unique_ptr<cudf::column>> children;
  switch (view.type().id()) {
    case cudf::type_id::LIST:
    case cudf::type_id::DICTIONARY32:
      // child 0 is the offsets (LIST) or the indices (DICTIONARY32), a column without children.
      if (view.num_children() == 2) {
        children.push_back(empty_leaf(view.child(0).type()));
        children.push_back(make_empty_like(view.child(1), stream, mr));
      }
      break;
    case cudf::type_id::STRUCT:
      for (cudf::size_type i = 0; i < view.num_children(); ++i) {
        children.push_back(make_empty_like(view.child(i), stream, mr));
      }
      break;
    default: break;
  }
  return std::make_unique<cudf::column>(view.type(),
                                        0,
                                        rmm::device_buffer{},
                                        make_absent_null_mask(stream, mr),
                                        0,
                                        std::move(children));
}

/// An offsets child must be whole: one entry more than its parent has rows, starting at entry 0.
void require_whole_offsets(cudf::column_view const& offsets,
                           cudf::size_type parent_rows,
                           char const* parent_type)
{
  auto const id = offsets.type().id();
  if ((id != cudf::type_id::INT32 && id != cudf::type_id::INT64) || offsets.offset() != 0 ||
      static_cast<std::int64_t>(offsets.size()) != static_cast<std::int64_t>(parent_rows) + 1) {
    reject_staged_column(std::string{"the offsets of a "} + parent_type +
                         " column are sliced or malformed");
  }
}

/**
 * @brief First pass: validate @p view and record its buffers, without touching the device.
 *
 * A column must be whole. A view that starts past row 0 is rejected for every type, and so is a
 * view that ends early when the column has children, because the children would not match it.
 *
 * @throws std::invalid_argument if the column is sliced, malformed or of an unsupported type
 */
staged_column plan_staged_column(cudf::column_view const& view, staged_table_summary& summary)
{
  staged_column plan;
  plan.type = view.type();
  plan.rows = view.size();
  if (view.size() == 0) {
    plan.source = view;
    return plan;
  }

  if (view.offset() != 0) { reject_staged_column("a column view that starts past row 0"); }

  auto const rows = static_cast<std::size_t>(view.size());
  plan.null_count = view.null_count();
  if (view.nullable()) {
    plan.source_mask = view.null_mask();
    plan.mask_bytes =
      static_cast<std::size_t>(cudf::num_bitmask_words(view.size())) * sizeof(cudf::bitmask_type);
    summary.reads_source = true;
  }

  switch (view.type().id()) {
    case cudf::type_id::STRING: {
      if (view.num_children() != 1) {
        reject_staged_column("a STRING column with rows needs an offsets child");
      }
      auto const offsets = view.child(0);
      require_whole_offsets(offsets, view.size(), "STRING");
      // The last offset is the size of the characters, known only on the device.
      plan.source_data = view.head();
      plan.size_probe  = summary.size_probes.size();
      auto const width = cudf::size_of(offsets.type());
      summary.size_probes.push_back(
        {static_cast<std::byte const*>(offsets.head()) + rows * width, width});
      plan.children.push_back(plan_staged_column(offsets, summary));
      break;
    }
    case cudf::type_id::LIST: {
      if (view.num_children() != 2) {
        reject_staged_column("a LIST column needs an offsets child and a values child");
      }
      require_whole_offsets(view.child(0), view.size(), "LIST");
      plan.children.push_back(plan_staged_column(view.child(0), summary));
      plan.children.push_back(plan_staged_column(view.child(1), summary));
      break;
    }
    case cudf::type_id::STRUCT: {
      for (cudf::size_type i = 0; i < view.num_children(); ++i) {
        if (view.child(i).size() != view.size()) {
          reject_staged_column("a STRUCT field differs in length from its column");
        }
        plan.children.push_back(plan_staged_column(view.child(i), summary));
      }
      break;
    }
    case cudf::type_id::DICTIONARY32: {
      if (view.num_children() != 2 || view.child(0).size() != view.size()) {
        reject_staged_column("the indices of a DICTIONARY32 column are missing or sliced");
      }
      plan.children.push_back(plan_staged_column(view.child(0), summary));
      plan.children.push_back(plan_staged_column(view.child(1), summary));
      break;
    }
    default: {
      if (cudf::is_fixed_width(view.type())) {
        if (view.head() == nullptr) { reject_staged_column("a column with rows has no data"); }
        plan.source_data     = view.head();
        plan.data_bytes      = rows * cudf::size_of(view.type());
        summary.reads_source = true;
      } else if (view.type().id() != cudf::type_id::EMPTY) {
        reject_staged_column("unsupported column type " +
                             std::to_string(static_cast<int>(view.type().id())));
      }
      break;
    }
  }
  return plan;
}

/**
 * @brief Second pass: allocate the destination buffers of @p plan on @p stream and append one
 * segment per buffer to @p segments, in the order mask, data, children.
 *
 * Only the words of a mask that cover the rows are copied, which is all a source mask is sure to
 * hold. A column without rows gets its empty column here and no segment.
 */
void allocate_staged_column(staged_column& plan,
                            std::span<std::int64_t const> string_sizes,
                            ::cuda::stream_ref stream,
                            rmm::device_async_resource_ref mr,
                            std::vector<memory::detail::staged_copy_segment>& segments)
{
  if (plan.rows == 0) {
    plan.empty = make_empty_like(plan.source, stream, mr);
    return;
  }

  if (plan.mask_bytes > 0) {
    plan.mask.emplace(
      cudf::create_null_mask(plan.rows, cudf::mask_state::UNINITIALIZED, stream, mr));
    if (cudf_compat::get_null_mask_size_bytes(*plan.mask) < plan.mask_bytes) {
      reject_staged_column("a null mask is smaller than its rows need");
    }
    segments.push_back({plan.source_mask, plan.mask->data(), plan.mask_bytes});
  } else {
    plan.mask.emplace(make_absent_null_mask(stream, mr));
  }
  if (plan.size_probe) {
    std::int64_t const characters = string_sizes[*plan.size_probe];
    if (characters < 0) { reject_staged_column("the last offset of a STRING column is negative"); }
    plan.data_bytes = static_cast<std::size_t>(characters);
    if (plan.data_bytes > 0 && plan.source_data == nullptr) {
      reject_staged_column("a STRING column has offsets past its characters");
    }
  }
  if (plan.data_bytes > 0) {
    plan.data = rmm::device_buffer(plan.data_bytes, stream, mr);
    segments.push_back({plan.source_data, plan.data.data(), plan.data_bytes});
  }
  for (auto& child : plan.children) {
    allocate_staged_column(child, string_sizes, stream, mr, segments);
  }
}

/**
 * @brief Turn a planned column, whose buffers have been filled, into a cudf column.
 *
 * Uses the cudf::column constructor for every type, which only takes ownership. The cudf factory
 * functions are avoided because some launch kernels that read the buffers, on a stream this
 * conversion does not control.
 */
std::unique_ptr<cudf::column> assemble_staged_column(staged_column&& plan)
{
  if (plan.rows == 0) { return std::move(plan.empty); }

  std::vector<std::unique_ptr<cudf::column>> children;
  children.reserve(plan.children.size());
  for (auto& child : plan.children) {
    children.push_back(assemble_staged_column(std::move(child)));
  }
  return std::make_unique<cudf::column>(plan.type,
                                        plan.rows,
                                        std::move(plan.data),
                                        std::move(*plan.mask),
                                        plan.null_count,
                                        std::move(children));
}

/**
 * @brief Copy @p source to the device of @p target_memory_space through pinned host memory.
 *
 * The table is validated and measured first, every destination buffer is allocated next, and all
 * buffers then travel in one memory::detail::staged_copy() call. The calling thread waits at most
 * three times, whatever the schema: for the source's writer if that is unknown, for the sizes of
 * the STRING columns if there are any with rows, and for the last byte to arrive. A table without
 * rows or without columns is built on the host and waits for nothing.
 *
 * The source may be freed as soon as this returns: no read of it is pending.
 *
 * @throws std::invalid_argument if a column is sliced, malformed or of an unsupported type
 */
std::unique_ptr<idata_representation> convert_gpu_to_gpu_staged(
  gpu_table_representation const& source,
  memory::memory_space& target_memory_space,
  detail::gpu_to_gpu_converter_options const& options)
{
  nvtx_scope staged_range{"convert:gpu_to_gpu:host_staging"};

  int const src_device = source.get_device_id();
  int const dst_device = target_memory_space.get_device_id();

  // Declared before anything that owns destination memory, so that memory is freed with its
  // device current when an exception unwinds this frame.
  rmm::cuda_set_device_raii const target_guard{rmm::cuda_device_id{dst_device}};

  // Each leg runs on a stream of the device it reads or writes.
  ::cuda::stream_ref const src_stream = source.get_memory_space().acquire_stream();
  ::cuda::stream_ref const dst_stream = target_memory_space.acquire_stream();

  cudf::table_view const src_view = source.get_table_view();
  staged_table_summary summary;
  std::vector<staged_column> plans;
  plans.reserve(static_cast<std::size_t>(src_view.num_columns()));
  for (cudf::column_view const& column : src_view) {
    plans.push_back(plan_staged_column(column, summary));
  }
  bool const reads_source = summary.reads_source || !summary.size_probes.empty();

  std::vector<std::int64_t> string_sizes;
  if (reads_source) {
    {
      // Order the reads after the source's writer. The stream that is made to wait belongs to the
      // source device, so that device must be current. An unknown writer makes this wait for the
      // whole source device.
      rmm::cuda_set_device_raii const source_guard{rmm::cuda_device_id{src_device}};
      if (source.get_writer_event() == nullptr) {
        nvtx_scope readiness_range{"staged:readiness"};
        source.wait_for_writer(src_stream);
        if (options.instrumentation != nullptr) { ++options.instrumentation->readiness_syncs; }
      } else {
        source.wait_for_writer(src_stream);
      }
    }
    string_sizes =
      memory::detail::read_device_integers(summary.size_probes,
                                           {src_device, src_stream},
                                           source.get_memory_space().get_default_allocator(),
                                           options.instrumentation);
  }

  std::vector<memory::detail::staged_copy_segment> segments;
  auto const mr = target_memory_space.get_default_allocator();
  for (auto& plan : plans) {
    allocate_staged_column(plan, string_sizes, dst_stream, mr, segments);
  }

  memory::detail::staged_copy(segments,
                              {src_device, src_stream},
                              {dst_device, dst_stream},
                              options.ring_settings,
                              options.instrumentation);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(plans.size());
  for (auto& plan : plans) {
    columns.push_back(assemble_staged_column(std::move(plan)));
  }

  // The destination buffers were written on dst_stream; the representation records an event on it
  // for readers on other streams.
  return std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(columns)), target_memory_space, dst_stream);
}

/**
 * @brief Convert gpu_table_representation to gpu_table_representation (cross-GPU copy).
 *
 * Replaces the previous cudf::pack-based path. cudf::pack's compute_splits internally
 * launches kernels that read scratch buffers allocated through the current_device
 * resource ref; under a multi-GPU setup the cucascade resource adapter created
 * a stream-ordered race that materialized as uninitialized dst_buf_info reads (caught
 * by compute-sanitizer) and silently produced garbage packed bytes. This path mirrors
 * convert_gpu_to_host_fast / convert_host_fast_to_gpu: walk the source column tree,
 * allocate target-side buffers, and peer-copy each leaf buffer directly.
 */
std::unique_ptr<idata_representation> convert_gpu_to_gpu(
  const idata_representation& source,
  const memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  [[maybe_unused]] memory::reservation* reservation,
  detail::gpu_to_gpu_converter_options const& options)
{
  nvtx_scope convert_range{"convert:gpu_to_gpu"};
  // no sync on the caller's stream needed here: wait_for_writer() below, which is the real
  // dependency, orders on the source's writer event, or syncs the source device if none was
  // recorded (to preserve correctness)

  auto& gpu_source = source.cast<gpu_table_representation>();

  // Same-device case: clone via source's own clone() method.
  if (source.get_device_id() == target_memory_space->get_device_id() &&
      !options.force_staged_route) {
    // A null stream handle represents the default stream of the current device: ensure it's the
    // device attached to the representation, or clone() will throw.
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{target_memory_space->get_device_id()}};
    return source.clone(stream);
  }

  auto const src_device_id = gpu_source.get_device_id();
  auto const dst_device_id = target_memory_space->get_device_id();

  // The route is chosen once for the whole table: a direct copy where the pair's peer route is
  // verified, a copy through pinned host memory otherwise.
  if (options.force_staged_route || !memory::probe_peer_dma_works(src_device_id, dst_device_id)) {
    return convert_gpu_to_gpu_staged(
      gpu_source, *const_cast<memory::memory_space*>(target_memory_space), options);
  }
  nvtx_scope copy_path_range{"convert:gpu_to_gpu:peer_copy"};

  rmm::cuda_set_device_raii target_guard{rmm::cuda_device_id{dst_device_id}};

  // Target-bound stream from the target memory_space's stream pool. All peer copies
  // and target-side allocations are issued on this stream so they observe in-order
  // completion without explicit cross-stream events.
  auto target_stream = target_memory_space->acquire_stream();
  auto mr            = target_memory_space->get_default_allocator();

  std::unique_ptr<cudf::table> new_table{nullptr};

  try {
    // Order the peer copies below after the most recent writer of gpu_source. With an unknown
    // writer this synchronizes the whole source device, which is correct but invisible to
    // compute-sanitizer: its stream-ordered race tracking ignores host-side synchronization, so it
    // may report races here that a recorded writer event avoids.
    gpu_source.wait_for_writer(target_stream);

    cudf::table_view const src_view = gpu_source.get_table_view();

    std::vector<std::unique_ptr<cudf::column>> target_columns;
    target_columns.reserve(static_cast<std::size_t>(src_view.num_columns()));
    for (const cudf::column_view& col : src_view) {
      target_columns.push_back(
        reconstruct_column_p2p_async(col, src_device_id, dst_device_id, target_stream, mr));
    }

    new_table = std::make_unique<cudf::table>(std::move(target_columns));

    // Sync for the source's lifetime, not for readers of the result (the writer event recorded
    // below covers those). The peer reads on target_stream are not ordered with the source's
    // deallocation stream, so a caller freeing the source right after convert (e.g.
    // mutable_data_batch::convert_to, which syncs only the caller's stream) would race them.
    // TODO: explore establishing ordering for source's lifetime via events.
    target_stream.sync();
  } catch (...) {
    // best-effort drain; a second failure must not replace the original exception or stay pending.
    if (cudaStreamSynchronize(target_stream.get()) != cudaSuccess) {
      std::ignore = cudaGetLastError();
    }
    throw;
  }

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
 * @brief Write data from a host block allocation to a disk file, handling block-boundary spanning.
 */
static void write_host_buffer_to_disk(
  const memory::fixed_size_host_memory_resource::multiple_blocks_allocation& alloc,
  std::size_t alloc_offset,
  std::size_t size,
  const std::filesystem::path& file_path,
  std::size_t file_offset,
  idisk_io_backend& backend)
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
    backend.write(file_path, block.data() + block_off, chunk, file_offset + written);
    written += chunk;
    block_off += chunk;
    if (block_off == block_size) {
      ++block_idx;
      block_off = 0;
    }
  }
}

/**
 * @brief Read data from a disk file into a host block allocation, handling block-boundary spanning.
 */
static void read_disk_buffer_to_host(
  const std::filesystem::path& file_path,
  std::size_t file_offset,
  std::size_t size,
  memory::fixed_size_host_memory_resource::multiple_blocks_allocation& alloc,
  std::size_t alloc_offset,
  idisk_io_backend& backend)
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
    backend.read(file_path, block.data() + block_off, chunk, file_offset + read_total);
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
 * @brief Recursively write column buffers from host blocks to disk at computed file offsets.
 */
static void write_column_buffers(
  const memory::column_metadata& src_meta,
  const memory::column_metadata& disk_meta,
  const memory::fixed_size_host_memory_resource::multiple_blocks_allocation& alloc,
  const std::filesystem::path& file_path,
  idisk_io_backend& backend)
{
  if (src_meta.has_null_mask && src_meta.null_mask_size > 0) {
    write_host_buffer_to_disk(alloc,
                              src_meta.null_mask_offset,
                              src_meta.null_mask_size,
                              file_path,
                              disk_meta.null_mask_offset,
                              backend);
  }
  if (src_meta.has_data && src_meta.data_size > 0) {
    write_host_buffer_to_disk(
      alloc, src_meta.data_offset, src_meta.data_size, file_path, disk_meta.data_offset, backend);
  }
  for (std::size_t i = 0; i < src_meta.children.size(); ++i) {
    write_column_buffers(src_meta.children[i], disk_meta.children[i], alloc, file_path, backend);
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
 * @brief Recursively read column buffers from disk into host blocks.
 */
static void read_column_buffers(
  const memory::column_metadata& disk_meta,
  const memory::column_metadata& host_meta,
  const std::filesystem::path& file_path,
  memory::fixed_size_host_memory_resource::multiple_blocks_allocation& alloc,
  idisk_io_backend& backend)
{
  if (disk_meta.has_null_mask && disk_meta.null_mask_size > 0) {
    read_disk_buffer_to_host(file_path,
                             disk_meta.null_mask_offset,
                             disk_meta.null_mask_size,
                             alloc,
                             host_meta.null_mask_offset,
                             backend);
  }
  if (disk_meta.has_data && disk_meta.data_size > 0) {
    read_disk_buffer_to_host(
      file_path, disk_meta.data_offset, disk_meta.data_size, alloc, host_meta.data_offset, backend);
  }
  for (std::size_t i = 0; i < disk_meta.children.size(); ++i) {
    read_column_buffers(disk_meta.children[i], host_meta.children[i], file_path, alloc, backend);
  }
}

/**
 * @brief Convert host_data_representation to disk_data_representation.
 *
 * Writes a complete disk file: header + serialized column_metadata + 4KB-aligned column data.
 * An RAII file guard ensures partial files are cleaned up on exception.
 */
static std::unique_ptr<idata_representation> convert_host_data_to_disk(
  const idata_representation& source,
  const memory::memory_space* target_memory_space,
  [[maybe_unused]] ::cuda::stream_ref stream,
  [[maybe_unused]] memory::reservation* reservation)
{
  nvtx_scope convert_range{"convert:host_to_disk"};
  auto& backend          = target_memory_space->get_io_backend();
  auto& host_source      = source.cast<host_data_representation>();
  const auto& host_table = host_source.get_host_table();

  // Generate unique file path under the disk memory space's mount directory
  auto mount_path = target_memory_space->get_disk_mount_path();
  auto file_path  = memory::generate_disk_file_path(mount_path);

  disk_file_guard guard{file_path};

  // Recompute column metadata with 4KB-aligned file offsets starting at offset 0.
  // No header or serialized metadata is written — metadata stays in memory.
  std::size_t cursor = 0;
  std::vector<memory::column_metadata> disk_columns;
  disk_columns.reserve(host_table->columns.size());
  for (const auto& col : host_table->columns) {
    disk_columns.push_back(recompute_file_offsets(col, cursor));
  }
  std::size_t total_file_data_size = cursor;

  // Write column data buffers
  for (std::size_t i = 0; i < host_table->columns.size(); ++i) {
    write_column_buffers(
      host_table->columns[i], disk_columns[i], *host_table->allocation, file_path, backend);
  }

  guard.release();

  auto disk_table = std::make_unique<memory::disk_table_allocation>(
    std::move(file_path), std::move(disk_columns), total_file_data_size);
  return std::make_unique<disk_data_representation>(
    std::move(disk_table), const_cast<memory::memory_space&>(*target_memory_space));
}

/**
 * @brief Convert disk_data_representation to host_data_representation.
 *
 * Uses in-memory column metadata from the source disk representation,
 * allocates host blocks, and reads data buffers from the disk file.
 */
static std::unique_ptr<idata_representation> convert_disk_to_host_data(
  const idata_representation& source,
  const memory::memory_space* target_memory_space,
  [[maybe_unused]] ::cuda::stream_ref stream,
  memory::reservation* reservation)
{
  nvtx_scope convert_range{"convert:disk_to_host"};
  auto& backend          = source.get_memory_space().get_io_backend();
  auto& disk_source      = source.cast<disk_data_representation>();
  const auto& disk_table = disk_source.get_disk_table();
  const auto& file_path  = disk_table.file_path;

  // Column metadata is already in memory — no file header or metadata to read
  const auto& disk_columns = disk_table.columns;

  // Compute host allocation size with 8-byte alignment
  std::size_t host_cursor = 0;
  std::vector<memory::column_metadata> host_columns;
  host_columns.reserve(disk_columns.size());
  for (const auto& col : disk_columns) {
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

  // Read column data from file into host blocks
  for (std::size_t i = 0; i < disk_columns.size(); ++i) {
    read_column_buffers(disk_columns[i], host_columns[i], file_path, *allocation, backend);
  }

  auto host_alloc = memory::host_table_allocation::create(
    std::move(allocation), std::move(host_columns), total_host_size);
  return std::make_unique<host_data_representation>(
    std::move(host_alloc), const_cast<memory::memory_space*>(target_memory_space));
}

// =============================================================================
// Disk <-> GPU converters
// =============================================================================

/**
 * @brief Recursively collect GPU column buffer I/O entries for batch submission.
 *
 * Instead of writing each buffer individually, collects all (ptr, size, file_offset)
 * tuples into a vector for a single batch write_batch call.
 */
static void collect_gpu_column_io_entries(const cudf::column_view& col,
                                          const memory::column_metadata& disk_meta,
                                          std::vector<io_batch_entry>& entries)
{
  if (disk_meta.has_null_mask && disk_meta.null_mask_size > 0) {
    entries.push_back({col.null_mask(), disk_meta.null_mask_size, disk_meta.null_mask_offset});
  }
  if (disk_meta.has_data && disk_meta.data_size > 0) {
    entries.push_back({col.data<uint8_t>(), disk_meta.data_size, disk_meta.data_offset});
  }
  for (cudf::size_type i = 0; i < col.num_children(); ++i) {
    collect_gpu_column_io_entries(
      col.child(i), disk_meta.children[static_cast<std::size_t>(i)], entries);
  }
}

/**
 * @brief Convert gpu_table_representation to disk_data_representation.
 *
 * Writes GPU table buffers directly to disk via write_batch.
 * File contains only 4KB-aligned column data — metadata stays in memory.
 */
static std::unique_ptr<idata_representation> convert_gpu_to_disk(
  const idata_representation& source,
  const memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  [[maybe_unused]] memory::reservation* reservation)
{
  nvtx_scope convert_range{"convert:gpu_to_disk"};
  auto& backend    = target_memory_space->get_io_backend();
  auto& gpu_source = source.cast<gpu_table_representation>();
  // Planning reads string offsets on the device, and the backend orders its reads after `stream`.
  gpu_source.wait_for_writer(stream);
  cudf::table_view tv = gpu_source.get_table_view();

  // Generate unique file path under the disk memory space's mount directory
  auto mount_path = target_memory_space->get_disk_mount_path();
  auto file_path  = memory::generate_disk_file_path(mount_path);

  disk_file_guard guard{file_path};

  // Plan column layout with 8-byte aligned offsets (same as host path)
  std::size_t plan_offset = 0;
  std::vector<memory::column_metadata> planned_columns;
  planned_columns.reserve(static_cast<std::size_t>(tv.num_columns()));
  for (cudf::size_type i = 0; i < tv.num_columns(); ++i) {
    planned_columns.push_back(plan_column_copy(tv.column(i), plan_offset, stream));
  }

  // Recompute with 4KB-aligned disk offsets starting at offset 0.
  // No header or serialized metadata is written — metadata stays in memory.
  std::size_t cursor = 0;
  std::vector<memory::column_metadata> disk_columns;
  disk_columns.reserve(planned_columns.size());
  for (const auto& col : planned_columns) {
    disk_columns.push_back(recompute_file_offsets(col, cursor));
  }
  std::size_t total_file_data_size = cursor;

  // Collect column data I/O entries
  std::vector<io_batch_entry> io_entries;
  for (cudf::size_type i = 0; i < tv.num_columns(); ++i) {
    collect_gpu_column_io_entries(
      tv.column(i), disk_columns[static_cast<std::size_t>(i)], io_entries);
  }

  // Single file open, single batch submit
  backend.write_batch(file_path, io_entries, stream);

  guard.release();

  auto disk_table = std::make_unique<memory::disk_table_allocation>(
    std::move(file_path), std::move(disk_columns), total_file_data_size);
  return std::make_unique<disk_data_representation>(
    std::move(disk_table), const_cast<memory::memory_space&>(*target_memory_space));
}

/**
 * @brief Allocate an rmm::device_buffer and read data from disk directly into it.
 */
static rmm::device_buffer alloc_and_read_from_disk(const std::filesystem::path& file_path,
                                                   std::size_t file_offset,
                                                   std::size_t size,
                                                   ::cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr,
                                                   idisk_io_backend& backend)
{
  rmm::device_buffer buf(size, stream, mr);
  if (size > 0) { backend.read(file_path, buf.data(), size, file_offset, stream); }
  return buf;
}

/**
 * @brief Recursively reconstruct a cudf::column from disk column_metadata, reading directly
 * from disk into device buffers via read.
 */
static std::unique_ptr<cudf::column> reconstruct_column_from_disk(
  const memory::column_metadata& meta,
  const std::filesystem::path& file_path,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  idisk_io_backend& backend)
{
  // Null mask (shared by all type categories). The buffer size is measured in elements
  // in newer cudf versions; disk extents and backend transfers are measured in bytes.
  auto null_mask = cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr);
  if (meta.has_null_mask) {
    null_mask = cudf::create_null_mask(meta.num_rows, cudf::mask_state::UNINITIALIZED, stream, mr);
    auto const mask_bytes = cudf_compat::get_null_mask_size_bytes(null_mask);
    if (mask_bytes > 0) {
      backend.read(file_path,
                   null_mask.data(),
                   std::min(mask_bytes, meta.null_mask_size),
                   meta.null_mask_offset,
                   stream);
    }
  }
  const cudf::size_type null_count = meta.has_null_mask ? meta.null_count : 0;

  if (as_cudf_type_id(meta.type_id) == cudf::type_id::STRING) {
    // See reconstruct_column for the empty-strings-column rationale. cudf::make_empty_column(
    // STRING) produces a column with no children; recreate it instead of erroring.
    if (meta.children.empty() && meta.num_rows == 0) {
      return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
    }
    if (meta.children.size() < 1) {
      throw std::invalid_argument(
        "reconstruct_column_from_disk: STRING column metadata must have at least one child "
        "(offsets)");
    }
    // Reconstruct offsets and cast to INT64 if needed (cudf 26.06+ requires INT64 offsets)
    auto offsets_col =
      reconstruct_column_from_disk(meta.children[0], file_path, stream, mr, backend);
    if (offsets_col->type().id() == cudf::type_id::INT32) {
      offsets_col =
        cudf::cast(offsets_col->view(), cudf::data_type{cudf::type_id::INT64}, stream, mr);
    }
    return cudf::make_strings_column(
      meta.num_rows,
      std::move(offsets_col),
      meta.has_data && meta.data_size > 0
        ? alloc_and_read_from_disk(file_path, meta.data_offset, meta.data_size, stream, mr, backend)
        : rmm::device_buffer{},
      null_count,
      std::move(null_mask));
  }

  if (as_cudf_type_id(meta.type_id) == cudf::type_id::LIST) {
    if (meta.children.size() < 2) {
      throw std::invalid_argument(
        "reconstruct_column_from_disk: LIST column metadata must have two children (offsets, "
        "values)");
    }
    // LIST offsets must remain INT32 (cudf::size_type); see reconstruct_column for the rationale.
    // Construct directly (like the STRUCT branch below) to preserve the INT32 offsets and skip
    // make_lists_column's purge_nonempty_nulls. Our serialized data already has consistent null
    // masks.
    std::vector<std::unique_ptr<cudf::column>> children;
    children.reserve(2);
    children.push_back(
      reconstruct_column_from_disk(meta.children[0], file_path, stream, mr, backend));
    children.push_back(
      reconstruct_column_from_disk(meta.children[1], file_path, stream, mr, backend));
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
      fields.push_back(reconstruct_column_from_disk(child_meta, file_path, stream, mr, backend));
    }
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
        "reconstruct_column_from_disk: DICTIONARY32 column metadata must have two children "
        "(indices, keys)");
    }
    // cudf DICTIONARY32 children order: [0]=indices, [1]=keys.
    // make_dictionary_column parameter order: (keys, indices, ...).
    auto indices_col =
      reconstruct_column_from_disk(meta.children[0], file_path, stream, mr, backend);
    auto keys_col = reconstruct_column_from_disk(meta.children[1], file_path, stream, mr, backend);
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
      ? alloc_and_read_from_disk(file_path, meta.data_offset, meta.data_size, stream, mr, backend)
      : rmm::device_buffer{},
    std::move(null_mask),
    null_count);
}

/**
 * @brief Convert disk_data_representation to gpu_table_representation.
 *
 * Reads column data directly from disk into GPU device buffers.
 * Column metadata comes from the in-memory disk_table_allocation.
 */
static std::unique_ptr<idata_representation> convert_disk_to_gpu(
  const idata_representation& source,
  const memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  [[maybe_unused]] memory::reservation* reservation)
{
  nvtx_scope convert_range{"convert:disk_to_gpu"};
  auto& backend          = source.get_memory_space().get_io_backend();
  auto& disk_source      = source.cast<disk_data_representation>();
  const auto& disk_table = disk_source.get_disk_table();
  const auto& file_path  = disk_table.file_path;

  // Column metadata is already in memory — no file header or metadata to read
  const auto& disk_columns = disk_table.columns;

  // Set CUDA device to target GPU (RAII restores previous device on scope exit)
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{target_memory_space->get_device_id()}};

  auto mr = target_memory_space->get_default_allocator();

  // Reconstruct columns by reading directly from disk into device buffers
  std::vector<std::unique_ptr<cudf::column>> gpu_columns;
  gpu_columns.reserve(disk_columns.size());
  for (const auto& col_meta : disk_columns) {
    gpu_columns.push_back(reconstruct_column_from_disk(col_meta, file_path, stream, mr, backend));
  }

  stream.sync();

  auto new_table = std::make_unique<cudf::table>(std::move(gpu_columns));
  return gpu_table_representation::make_written_on(
    std::move(new_table), *const_cast<memory::memory_space*>(target_memory_space), stream);
}

}  // namespace

staged_copy_settings::staged_copy_settings(std::size_t chunk_bytes, std::size_t ring_depth)
  : _chunk_bytes(chunk_bytes), _ring_depth(ring_depth)
{
  if (chunk_bytes == 0 || chunk_bytes > max_chunk_bytes || ring_depth == 0 ||
      ring_depth > max_ring_depth) {
    throw std::invalid_argument(
      "staged_copy_settings: chunk_bytes must be 1 B to 1 GiB and ring_depth 1 to 64");
  }
}

void register_builtin_converters(representation_converter_registry& registry)
{
  register_builtin_converters(registry, staged_copy_settings{});
}

void register_builtin_converters(representation_converter_registry& registry,
                                 staged_copy_settings const& staged_copy)
{
  detail::gpu_to_gpu_converter_options options;
  options.ring_settings.chunk_bytes = staged_copy.chunk_bytes();
  options.ring_settings.depth       = staged_copy.ring_depth();
  detail::register_builtin_converters(registry, options);
}

namespace detail {

void register_builtin_converters(representation_converter_registry& registry,
                                 gpu_to_gpu_converter_options const& gpu_to_gpu_options)
{
  // GPU -> GPU (cross-device copy). convert_gpu_to_gpu copies each column buffer with
  // cudaMemcpyPeerAsync when cucascade::memory::probe_peer_dma_works() verifies the direction, and
  // sends the whole table through a ring of pinned host memory otherwise. The cached probe copies
  // bytes between two mutually granted private pools, so a verification failure means the pool
  // peer route delivered wrong bytes, as on platforms whose direct route reports success but
  // leaves stale data.
  registry.register_converter<gpu_table_representation, gpu_table_representation>(
    [gpu_to_gpu_options](const idata_representation& source,
                         const memory::memory_space* target_memory_space,
                         ::cuda::stream_ref stream,
                         memory::reservation* reservation) {
      return convert_gpu_to_gpu(
        source, target_memory_space, stream, reservation, gpu_to_gpu_options);
    });

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

  // HOST DATA -> DISK (backend resolved from target disk memory_space)
  registry.register_converter<host_data_representation, disk_data_representation>(
    convert_host_data_to_disk);

  // DISK -> HOST DATA (backend resolved from source disk memory_space)
  registry.register_converter<disk_data_representation, host_data_representation>(
    convert_disk_to_host_data);

  // GPU -> DISK (backend resolved from target disk memory_space)
  registry.register_converter<gpu_table_representation, disk_data_representation>(
    convert_gpu_to_disk);

  // DISK -> GPU (backend resolved from source disk memory_space)
  registry.register_converter<disk_data_representation, gpu_table_representation>(
    convert_disk_to_gpu);
}

}  // namespace detail

}  // namespace cucascade
