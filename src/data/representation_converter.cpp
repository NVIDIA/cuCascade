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

#include "cudf/contiguous_split.hpp"

#include <cucascade/cuda_utils.hpp>
#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/data/disk_file_format.hpp>
#include <cucascade/data/disk_io_backend.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/disk_access_limiter.hpp>
#include <cucascade/memory/disk_table.hpp>
#include <cucascade/memory/host_table.hpp>
#include <cucascade/memory/host_table_packed.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace cucascade {

// =============================================================================
// representation_converter_registry implementation
// =============================================================================

void representation_converter_registry::register_converter_impl(
  const converter_key& key, representation_converter_fn converter)
{
  std::lock_guard<std::mutex> lock(_mutex);

  if (_converters.find(key) != _converters.end()) {
    std::ostringstream oss;
    oss << "Converter already registered for source type '" << key.source_type.name()
        << "' to target type '" << key.target_type.name() << "'";
    throw std::runtime_error(oss.str());
  }

  _converters.emplace(key, std::move(converter));
}

bool representation_converter_registry::has_converter_impl(const converter_key& key) const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _converters.find(key) != _converters.end();
}

std::unique_ptr<idata_representation> representation_converter_registry::convert_impl(
  const converter_key& key,
  idata_representation& source,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream) const
{
  representation_converter_fn converter;
  {
    std::lock_guard<std::mutex> lock(_mutex);

    auto it = _converters.find(key);
    if (it == _converters.end()) {
      std::ostringstream oss;
      oss << "No converter registered for source type '" << key.source_type.name()
          << "' to target type '" << key.target_type.name() << "'";
      throw std::runtime_error(oss.str());
    }

    converter = it->second;
  }

  return converter(source, target_memory_space, stream);
}

std::unique_ptr<idata_representation> representation_converter_registry::convert(
  idata_representation& source,
  std::type_index target_type,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream) const
{
  converter_key key{std::type_index(typeid(source)), target_type};
  return convert_impl(key, source, target_memory_space, stream);
}

bool representation_converter_registry::unregister_converter_impl(const converter_key& key)
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _converters.erase(key) > 0;
}

void representation_converter_registry::clear()
{
  std::lock_guard<std::mutex> lock(_mutex);
  _converters.clear();
}

// =============================================================================
// Built-in converter implementations
// =============================================================================

namespace {

/**
 * @brief Convert gpu_table_representation to gpu_table_representation (cross-GPU copy)
 */
std::unique_ptr<idata_representation> convert_gpu_to_gpu(
  idata_representation& source,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream)
{
  // Synchronize the stream to ensure any prior operations (like table creation)
  // are complete before we read from the source table
  stream.synchronize();

  auto& gpu_source = source.cast<gpu_table_representation>();

  // Same-device case: clone via source's own clone() method.
  if (source.get_device_id() == target_memory_space->get_device_id()) {
    return source.clone(stream);
  }

  auto packed_data = cudf::pack(gpu_source.get_table(), stream);

  assert(source.get_device_id() != target_memory_space->get_device_id());
  auto const target_device_id = target_memory_space->get_device_id();
  auto const source_device_id = source.get_device_id();
  auto const bytes_to_copy    = packed_data.gpu_data->size();
  auto mr                     = target_memory_space->get_default_allocator();

  // Acquire a stream that belongs to the target GPU device
  auto target_stream = target_memory_space->acquire_stream();

  CUCASCADE_CUDA_TRY(cudaSetDevice(target_device_id));
  rmm::device_uvector<uint8_t> dst_uvector(bytes_to_copy, target_stream, mr);
  target_stream.synchronize();
  // Restore previous device before peer copy
  CUCASCADE_CUDA_TRY(cudaSetDevice(source_device_id));

  // Asynchronously copy device->device across GPUs
  CUCASCADE_CUDA_TRY(cudaMemcpyPeerAsync(dst_uvector.data(),
                                         target_device_id,
                                         static_cast<const uint8_t*>(packed_data.gpu_data->data()),
                                         source_device_id,
                                         bytes_to_copy,
                                         stream.value()));
  stream.synchronize();
  // Unpack on target device to build a cudf::table that lives on the target GPU
  CUCASCADE_CUDA_TRY(cudaSetDevice(target_device_id));
  rmm::device_buffer dst_buffer = std::move(dst_uvector).release();
  // Unpack using pointer-based API and construct an owning cudf::table
  auto new_metadata = std::move(packed_data.metadata);
  auto new_gpu_data = std::make_unique<rmm::device_buffer>(std::move(dst_buffer));
  auto new_table_view =
    cudf::unpack(new_metadata->data(), static_cast<uint8_t const*>(new_gpu_data->data()));
  auto new_table = std::make_unique<cudf::table>(new_table_view, target_stream, mr);
  // Restore previous device
  target_stream.synchronize();
  CUCASCADE_CUDA_TRY(cudaSetDevice(source_device_id));

  return std::make_unique<gpu_table_representation>(
    std::move(new_table), *const_cast<memory::memory_space*>(target_memory_space));
}

/**
 * @brief Convert gpu_table_representation to host_data_packed_representation
 */
std::unique_ptr<idata_representation> convert_gpu_to_host(
  idata_representation& source,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream)
{
  // Synchronize the stream to ensure any prior operations (like table creation)
  // are complete before we read from the source table
  stream.synchronize();

  auto& gpu_source = source.cast<gpu_table_representation>();
  auto packed_data = cudf::pack(gpu_source.get_table(), stream);

  auto mr = target_memory_space->get_memory_resource_as<memory::fixed_size_host_memory_resource>();
  auto allocation = mr->allocate_multiple_blocks(packed_data.gpu_data->size());

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
                      stream.value()));
    source_offset += bytes_to_copy;
    block_offset += bytes_to_copy;
    if (block_offset == block_size) {
      block_index++;
      block_offset = 0;
    }
  }
  stream.synchronize();
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
  idata_representation& source,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream)
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
                                       stream.value()));
    dst_offset += bytes_to_copy;
    src_block_offset += bytes_to_copy;
    if (src_block_offset == src_block_size) {
      src_block_index++;
      src_block_offset = 0;
    }
  }

  auto new_metadata = std::make_unique<std::vector<uint8_t>>(*host_table->metadata);
  auto new_gpu_data = std::make_unique<rmm::device_buffer>(std::move(dst_buffer));
  stream.synchronize();
  auto new_table_view =
    cudf::unpack(new_metadata->data(), static_cast<uint8_t const*>(new_gpu_data->data()));
  auto new_table = std::make_unique<cudf::table>(new_table_view, stream, mr);
  stream.synchronize();

  return std::make_unique<gpu_table_representation>(
    std::move(new_table), *const_cast<memory::memory_space*>(target_memory_space));
}

/**
 * @brief Convert host_data_packed_representation to host_data_packed_representation (cross-host
 * copy)
 */
std::unique_ptr<idata_representation> convert_host_to_host(
  idata_representation& source,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view /*stream*/)
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
  auto dst_allocation         = mr->allocate_multiple_blocks(data_size);
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

  void flush(rmm::cuda_stream_view stream, cudaMemcpySrcAccessOrder src_order)
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
      dsts.data(), srcs.data(), sizes.data(), count(), attr, nullptr, stream.value()));
#else
    CUCASCADE_CUDA_TRY(
      cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), count(), attr, stream.value()));
#endif
#else
    // cudaMemcpyBatchAsync requires CUDA 12.8+; fall back to individual copies.
    (void)src_order;
    for (std::size_t i = 0; i < count(); ++i) {
      CUCASCADE_CUDA_TRY(
        cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyDefault, stream.value()));
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
                                                rmm::cuda_stream_view stream)
{
  assert(col.offset() == 0 && "column_view with non-zero offset is not supported");

  memory::column_metadata meta{};
  meta.type_id    = col.type().id();
  meta.num_rows   = col.size();
  meta.null_count = col.null_count();
  meta.scale      = 0;

  if (meta.type_id == cudf::type_id::DECIMAL32 || meta.type_id == cudf::type_id::DECIMAL64 ||
      meta.type_id == cudf::type_id::DECIMAL128) {
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
static void collect_d2h_ops(const void* src,
                            std::size_t size,
                            std::size_t alloc_offset,
                            memory::fixed_multiple_blocks_allocation& alloc,
                            BatchCopyAccumulator& batch)
{
  if (size == 0 || src == nullptr) { return; }

  const std::size_t block_size = alloc->block_size();
  std::size_t block_idx        = alloc_offset / block_size;
  std::size_t block_off        = alloc_offset % block_size;
  std::size_t src_off          = 0;

  while (src_off < size) {
    std::size_t remaining      = size - src_off;
    std::size_t space_in_block = block_size - block_off;
    std::size_t bytes_to_copy  = std::min(remaining, space_in_block);

    auto block = alloc->at(block_idx);
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
static void collect_column_d2h_ops(const cudf::column_view& col,
                                   const memory::column_metadata& meta,
                                   memory::fixed_multiple_blocks_allocation& alloc,
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
  idata_representation& source,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream)
{
  auto& gpu_source            = source.cast<gpu_table_representation>();
  const cudf::table_view view = gpu_source.get_table().view();

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
  auto allocation = mr->allocate_multiple_blocks(total_size);

  // --- Pass 3: collect all D→H copy ops, then fire one batched call ---
  BatchCopyAccumulator batch;
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    collect_column_d2h_ops(view.column(i), columns[static_cast<std::size_t>(i)], allocation, batch);
  }
  batch.flush(stream, cudaMemcpySrcAccessOrderStream);
  stream.synchronize();

  auto host_alloc = std::make_unique<memory::host_table_allocation>(
    std::move(allocation), std::move(columns), total_size);

  return std::make_unique<host_data_representation>(
    std::move(host_alloc), const_cast<memory::memory_space*>(target_memory_space));
}

/**
 * @brief Allocate a device buffer of @p size bytes and schedule its H→D copy ops into @p batch.
 *
 * The buffer is allocated immediately; the actual copies are deferred until the caller calls
 * batch.flush(). The buffer must remain alive until flush() completes (it will, since it is
 * owned by the column tree being assembled in the caller).
 */
static rmm::device_buffer alloc_and_schedule_h2d(memory::fixed_multiple_blocks_allocation& alloc,
                                                 std::size_t alloc_offset,
                                                 std::size_t size,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr,
                                                 BatchCopyAccumulator& batch)
{
  rmm::device_buffer buf(size, stream, mr);
  if (size == 0) { return buf; }
  if (!alloc || alloc->size() == 0) {
    throw std::invalid_argument(
      "alloc_and_schedule_h2d: allocation is null or empty but copy size is non-zero");
  }

  const std::size_t block_size = alloc->block_size();
  std::size_t block_idx        = alloc_offset / block_size;
  std::size_t block_off        = alloc_offset % block_size;
  std::size_t dst_off          = 0;

  while (dst_off < size) {
    std::size_t remaining      = size - dst_off;
    std::size_t space_in_block = block_size - block_off;
    std::size_t bytes_to_copy  = std::min(remaining, space_in_block);

    auto block = alloc->at(block_idx);
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
static rmm::device_buffer alloc_and_copy_h2d_sync(memory::fixed_multiple_blocks_allocation& alloc,
                                                  std::size_t alloc_offset,
                                                  std::size_t size,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  rmm::device_buffer buf(size, stream, mr);
  if (size == 0) { return buf; }

  const std::size_t block_size = alloc->block_size();
  std::size_t block_idx        = alloc_offset / block_size;
  std::size_t block_off        = alloc_offset % block_size;
  std::size_t dst_off          = 0;

  while (dst_off < size) {
    std::size_t remaining      = size - dst_off;
    std::size_t space_in_block = block_size - block_off;
    std::size_t bytes_to_copy  = std::min(remaining, space_in_block);

    auto block = alloc->at(block_idx);
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(static_cast<uint8_t*>(buf.data()) + dst_off,
                                       block.data() + block_off,
                                       bytes_to_copy,
                                       cudaMemcpyHostToDevice,
                                       stream.value()));
    dst_off += bytes_to_copy;
    block_off += bytes_to_copy;
    if (block_off == block_size) {
      ++block_idx;
      block_off = 0;
    }
  }
  stream.synchronize();
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
  memory::fixed_multiple_blocks_allocation& alloc,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  BatchCopyAccumulator& batch)
{
  // Null mask — copied synchronously because cudf factories access it on construction
  rmm::device_buffer null_mask{};
  if (meta.has_null_mask) {
    null_mask =
      alloc_and_copy_h2d_sync(alloc, meta.null_mask_offset, meta.null_mask_size, stream, mr);
  }
  const cudf::size_type null_count = meta.has_null_mask ? meta.null_count : 0;

  if (meta.type_id == cudf::type_id::STRING) {
    if (meta.children.size() < 1) {
      throw std::invalid_argument(
        "reconstruct_column: STRING column metadata must have at least one child (offsets)");
    }
    auto offsets_col = reconstruct_column(meta.children[0], alloc, stream, mr, batch);
    if (offsets_col->type().id() == cudf::type_id::INT32) {
      // Flush pending H2D copies so the INT32 offsets buffer has valid data on device
      // before the cast reads from it. The cast replaces offsets_col, freeing the old
      // INT32 buffer, so batch must not hold dangling pointers to it.
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

  if (meta.type_id == cudf::type_id::LIST) {
    if (meta.children.size() < 2) {
      throw std::invalid_argument(
        "reconstruct_column: LIST column metadata must have two children (offsets, values)");
    }
    auto offsets_col = reconstruct_column(meta.children[0], alloc, stream, mr, batch);
    if (offsets_col->type().id() == cudf::type_id::INT32) {
      // Flush pending H2D copies so the INT32 offsets buffer has valid data on device
      // before the cast reads from it. See STRING branch comment above.
      batch.flush(stream, cudaMemcpySrcAccessOrderDuringApiCall);
      offsets_col =
        cudf::cast(offsets_col->view(), cudf::data_type{cudf::type_id::INT64}, stream, mr);
    }
    return cudf::make_lists_column(meta.num_rows,
                                   std::move(offsets_col),
                                   reconstruct_column(meta.children[1], alloc, stream, mr, batch),
                                   null_count,
                                   std::move(null_mask));
  }

  if (meta.type_id == cudf::type_id::STRUCT) {
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

  if (meta.type_id == cudf::type_id::DICTIONARY32) {
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

  const cudf::data_type dtype = cudf::is_fixed_point(cudf::data_type{meta.type_id})
                                  ? cudf::data_type{meta.type_id, meta.scale}
                                  : cudf::data_type{meta.type_id};
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
  idata_representation& source,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream)
{
  auto& fast_source      = source.cast<host_data_representation>();
  const auto& fast_table = fast_source.get_host_table();
  if (!fast_table) { throw std::runtime_error("convert_host_fast_to_gpu: host table is null"); }
  if (!fast_table->allocation) {
    throw std::runtime_error("convert_host_fast_to_gpu: host table allocation is null");
  }

  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{target_memory_space->get_device_id()}};

  auto mr = target_memory_space->get_default_allocator();

  // Collect all H→D copy ops across all columns, then fire one batched call.
  BatchCopyAccumulator batch;
  std::vector<std::unique_ptr<cudf::column>> gpu_columns;
  gpu_columns.reserve(fast_table->columns.size());
  for (const auto& col_meta : fast_table->columns) {
    gpu_columns.push_back(reconstruct_column(col_meta, fast_table->allocation, stream, mr, batch));
  }
  // Source is CPU-written pinned host memory: fully prepared before this call.
  batch.flush(stream, cudaMemcpySrcAccessOrderDuringApiCall);

  auto new_table = std::make_unique<cudf::table>(std::move(gpu_columns));
  stream.synchronize();

  return std::make_unique<gpu_table_representation>(
    std::move(new_table), *const_cast<memory::memory_space*>(target_memory_space));
}

/**
 * @brief Convert host_data_representation to host_data_representation (cross-host copy)
 */
std::unique_ptr<idata_representation> convert_host_fast_to_host_fast(
  idata_representation& source,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view /*stream*/)
{
  auto& host_source    = source.cast<host_data_representation>();
  auto& host_table     = host_source.get_host_table();
  auto const data_size = host_table->data_size;

  assert(source.get_device_id() != target_memory_space->get_device_id());
  auto mr = target_memory_space->get_memory_resource_as<memory::fixed_size_host_memory_resource>();
  if (mr == nullptr) {
    throw std::runtime_error(
      "Target HOST memory_space does not have a fixed_size_host_memory_resource");
  }
  auto dst_allocation         = mr->allocate_multiple_blocks(data_size);
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
  auto new_host_table = std::make_unique<memory::host_table_allocation>(
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
static void write_host_buffer_to_disk(const memory::fixed_multiple_blocks_allocation& alloc,
                                      std::size_t alloc_offset,
                                      std::size_t size,
                                      const std::filesystem::path& file_path,
                                      std::size_t file_offset,
                                      idisk_io_backend& backend)
{
  const std::size_t block_size = alloc->block_size();
  std::size_t block_idx        = alloc_offset / block_size;
  std::size_t block_off        = alloc_offset % block_size;
  std::size_t written          = 0;
  while (written < size) {
    std::size_t remaining = size - written;
    std::size_t avail     = block_size - block_off;
    std::size_t chunk     = std::min(remaining, avail);
    auto block            = alloc->at(block_idx);
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
static void read_disk_buffer_to_host(const std::filesystem::path& file_path,
                                     std::size_t file_offset,
                                     std::size_t size,
                                     memory::fixed_multiple_blocks_allocation& alloc,
                                     std::size_t alloc_offset,
                                     idisk_io_backend& backend)
{
  const std::size_t block_size = alloc->block_size();
  std::size_t block_idx        = alloc_offset / block_size;
  std::size_t block_off        = alloc_offset % block_size;
  std::size_t read_total       = 0;
  while (read_total < size) {
    std::size_t remaining = size - read_total;
    std::size_t avail     = block_size - block_off;
    std::size_t chunk     = std::min(remaining, avail);
    auto block            = alloc->at(block_idx);
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
static void write_column_buffers(const memory::column_metadata& src_meta,
                                 const memory::column_metadata& disk_meta,
                                 const memory::fixed_multiple_blocks_allocation& alloc,
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
static void read_column_buffers(const memory::column_metadata& disk_meta,
                                const memory::column_metadata& host_meta,
                                const std::filesystem::path& file_path,
                                memory::fixed_multiple_blocks_allocation& alloc,
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
  idata_representation& source,
  const memory::memory_space* target_memory_space,
  [[maybe_unused]] rmm::cuda_stream_view stream)
{
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
      host_table->columns[i], disk_columns[i], host_table->allocation, file_path, backend);
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
  idata_representation& source,
  const memory::memory_space* target_memory_space,
  [[maybe_unused]] rmm::cuda_stream_view stream)
{
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
  auto allocation = mr->allocate_multiple_blocks(total_host_size);

  // Read column data from file into host blocks
  for (std::size_t i = 0; i < disk_columns.size(); ++i) {
    read_column_buffers(disk_columns[i], host_columns[i], file_path, allocation, backend);
  }

  auto host_alloc = std::make_unique<memory::host_table_allocation>(
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
  idata_representation& source,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream)
{
  auto& backend       = target_memory_space->get_io_backend();
  auto& gpu_source    = source.cast<gpu_table_representation>();
  const auto& table   = gpu_source.get_table();
  cudf::table_view tv = table.view();

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
                                                   rmm::cuda_stream_view stream,
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
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  idisk_io_backend& backend)
{
  // Null mask (shared by all type categories)
  rmm::device_buffer null_mask{};
  if (meta.has_null_mask) {
    null_mask = alloc_and_read_from_disk(
      file_path, meta.null_mask_offset, meta.null_mask_size, stream, mr, backend);
  }
  const cudf::size_type null_count = meta.has_null_mask ? meta.null_count : 0;

  if (meta.type_id == cudf::type_id::STRING) {
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

  if (meta.type_id == cudf::type_id::LIST) {
    if (meta.children.size() < 2) {
      throw std::invalid_argument(
        "reconstruct_column_from_disk: LIST column metadata must have two children (offsets, "
        "values)");
    }
    auto offsets_col =
      reconstruct_column_from_disk(meta.children[0], file_path, stream, mr, backend);
    if (offsets_col->type().id() == cudf::type_id::INT32) {
      offsets_col =
        cudf::cast(offsets_col->view(), cudf::data_type{cudf::type_id::INT64}, stream, mr);
    }
    return cudf::make_lists_column(
      meta.num_rows,
      std::move(offsets_col),
      reconstruct_column_from_disk(meta.children[1], file_path, stream, mr, backend),
      null_count,
      std::move(null_mask));
  }

  if (meta.type_id == cudf::type_id::STRUCT) {
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

  if (meta.type_id == cudf::type_id::DICTIONARY32) {
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

  const cudf::data_type dtype = cudf::is_fixed_point(cudf::data_type{meta.type_id})
                                  ? cudf::data_type{meta.type_id, meta.scale}
                                  : cudf::data_type{meta.type_id};
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
  idata_representation& source,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream)
{
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

  stream.synchronize();

  auto new_table = std::make_unique<cudf::table>(std::move(gpu_columns));
  return std::make_unique<gpu_table_representation>(
    std::move(new_table), *const_cast<memory::memory_space*>(target_memory_space));
}

}  // namespace

void register_builtin_converters(representation_converter_registry& registry)
{
  // GPU -> GPU (cross-device copy)
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

}  // namespace cucascade
