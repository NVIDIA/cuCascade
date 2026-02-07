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

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/host_table.hpp>

#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>
#include <sstream>

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
  auto packed_data = cudf::pack(gpu_source.get_table(), stream);

  assert(source.get_device_id() != target_memory_space->get_device_id());
  auto const target_device_id = target_memory_space->get_device_id();
  auto const source_device_id = source.get_device_id();
  auto const bytes_to_copy    = packed_data.gpu_data->size();
  auto mr                     = target_memory_space->get_default_allocator();

  // Acquire a stream that belongs to the target GPU device
  auto target_stream = target_memory_space->acquire_stream();

  RMM_CUDA_TRY(cudaSetDevice(target_device_id));
  rmm::device_uvector<uint8_t> dst_uvector(bytes_to_copy, target_stream, mr);
  target_stream.synchronize();
  // Restore previous device before peer copy
  RMM_CUDA_TRY(cudaSetDevice(source_device_id));

  // Asynchronously copy device->device across GPUs
  RMM_CUDA_TRY(cudaMemcpyPeerAsync(dst_uvector.data(),
                                   target_device_id,
                                   static_cast<const uint8_t*>(packed_data.gpu_data->data()),
                                   source_device_id,
                                   bytes_to_copy,
                                   stream.value()));
  stream.synchronize();
  // Unpack on target device to build a cudf::table that lives on the target GPU
  RMM_CUDA_TRY(cudaSetDevice(target_device_id));
  rmm::device_buffer dst_buffer = std::move(dst_uvector).release();
  // Unpack using pointer-based API and construct an owning cudf::table
  auto new_metadata = std::move(packed_data.metadata);
  auto new_gpu_data = std::make_unique<rmm::device_buffer>(std::move(dst_buffer));
  auto new_table_view =
    cudf::unpack(new_metadata->data(), static_cast<uint8_t const*>(new_gpu_data->data()));
  auto new_table = std::make_unique<cudf::table>(new_table_view, target_stream, mr);
  // Restore previous device
  target_stream.synchronize();
  RMM_CUDA_TRY(cudaSetDevice(source_device_id));

  return std::make_unique<gpu_table_representation>(
    std::move(new_table), *const_cast<memory::memory_space*>(target_memory_space));
}

/**
 * @brief Convert gpu_table_representation to host_table_representation
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
    RMM_CUDA_TRY(
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
  auto host_table_allocation = std::make_unique<memory::host_table_allocation>(
    std::move(allocation), std::move(packed_data.metadata), packed_data.gpu_data->size());
  return std::make_unique<host_table_representation>(
    std::move(host_table_allocation), const_cast<memory::memory_space*>(target_memory_space));
}

/**
 * @brief Convert host_table_representation to gpu_table_representation
 */
std::unique_ptr<idata_representation> convert_host_to_gpu(
  idata_representation& source,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream)
{
  auto& host_source    = source.cast<host_table_representation>();
  auto& host_table     = host_source.get_host_table();
  auto const data_size = host_table->data_size;

  auto mr             = target_memory_space->get_default_allocator();
  int previous_device = -1;
  RMM_CUDA_TRY(cudaGetDevice(&previous_device));
  RMM_CUDA_TRY(cudaSetDevice(target_memory_space->get_device_id()));

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
    RMM_CUDA_TRY(cudaMemcpyAsync(static_cast<uint8_t*>(dst_buffer.data()) + dst_offset,
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
  auto new_table_view =
    cudf::unpack(new_metadata->data(), static_cast<uint8_t const*>(new_gpu_data->data()));
  auto new_table = std::make_unique<cudf::table>(new_table_view, stream, mr);
  stream.synchronize();

  RMM_CUDA_TRY(cudaSetDevice(previous_device));
  return std::make_unique<gpu_table_representation>(
    std::move(new_table), *const_cast<memory::memory_space*>(target_memory_space));
}

/**
 * @brief Convert host_table_representation to host_table_representation (cross-host copy)
 */
std::unique_ptr<idata_representation> convert_host_to_host(
  idata_representation& source,
  const memory::memory_space* target_memory_space,
  rmm::cuda_stream_view /*stream*/)
{
  auto& host_source    = source.cast<host_table_representation>();
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
  auto metadata_copy         = std::make_unique<std::vector<uint8_t>>(*host_table->metadata);
  auto host_table_allocation = std::make_unique<memory::host_table_allocation>(
    std::move(dst_allocation), std::move(metadata_copy), data_size);
  return std::make_unique<host_table_representation>(
    std::move(host_table_allocation), const_cast<memory::memory_space*>(target_memory_space));
}

}  // namespace

void register_builtin_converters(representation_converter_registry& registry)
{
  // GPU -> GPU (cross-device copy)
  registry.register_converter<gpu_table_representation, gpu_table_representation>(
    convert_gpu_to_gpu);

  // GPU -> HOST
  registry.register_converter<gpu_table_representation, host_table_representation>(
    convert_gpu_to_host);

  // HOST -> GPU
  registry.register_converter<host_table_representation, gpu_table_representation>(
    convert_host_to_gpu);

  // HOST -> HOST (cross-device copy)
  registry.register_converter<host_table_representation, host_table_representation>(
    convert_host_to_host);
}

}  // namespace cucascade
