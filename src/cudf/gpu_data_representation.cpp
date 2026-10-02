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

#include <cucascade/cuda/event.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/error.hpp>

#include <cudf/column/column_stream.hpp>
#include <cudf/copying.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace cucascade {

namespace {

/**
 * @brief Reject streams that do not belong to the device owning this representation's memory.
 *
 * RMM binds every device buffer's deallocation to the stream it carries, so a foreign-device
 * stream would retire the block into a pool free-list keyed by a stream on the wrong device;
 * a later allocation on that stream then hands out memory the device cannot address without
 * peer mapping. Default stream handles resolve to the calling thread's current device, so
 * passing one while another device is current is rejected too — that call would have been
 * cross-device just the same.
 */
void validate_stream_device(::cuda::stream_ref stream, int expected_device)
{
  int stream_device = -1;
  CUCASCADE_CUDA_TRY(::cudaStreamGetDevice(stream.get(), &stream_device));
  if (stream_device != expected_device) {
    CUCASCADE_FAIL("stream belongs to CUDA device " + std::to_string(stream_device) +
                   " but this representation's memory lives on device " +
                   std::to_string(expected_device));
  }
}

/**
 * @brief Bind every buffer of @p table to @p stream for future deallocation.
 */
void rebind_table(std::unique_ptr<cudf::table>& table, ::cuda::stream_ref stream)
{
  auto columns = table->release();
  for (auto& col : columns) {
    col = cudf::rebind_stream(std::move(*col), stream);
  }
  table = std::make_unique<cudf::table>(std::move(columns));
}

/**
 * @brief Deep-copy @p view on @p stream and block until @p stream has finished all of its work.
 *
 * Once this returns, the memory behind @p view may be freed. If the copy fails, the copies already
 * enqueued may still be reading that memory, so @p stream is synchronized before the exception
 * propagates.
 */
[[nodiscard]] std::unique_ptr<cudf::table> materialize_and_wait(cudf::table_view view,
                                                                ::cuda::stream_ref stream)
{
  try {
    auto table = std::make_unique<cudf::table>(view, stream);
    CUCASCADE_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    return table;
  } catch (...) {
    // Best effort drain: a second failure must neither replace the original exception nor stay
    // pending for later CUDA calls on this thread.
    if (cudaStreamSynchronize(stream.get()) != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
    }
    throw;
  }
}

/**
 * @brief Destroys a CUDA event that has not been handed to its owner.
 */
struct event_destroyer {
  void operator()(cudaEvent_t event) const noexcept
  {
    if (cudaEventDestroy(event) != cudaSuccess) { static_cast<void>(cudaGetLastError()); }
  }
};

using unique_event = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, event_destroyer>;

}  // namespace

gpu_table_representation::gpu_table_representation(std::unique_ptr<cudf::table>&& table,
                                                   cucascade::memory::memory_space& memory_space,
                                                   ::cuda::stream_ref writer_stream)
  : idata_representation(memory_space),
    _table(take_owned_table(std::move(table), memory_space, writer_stream))
{
  // A null handle means the writer is unknown, so there is nothing to record. take_owned_table()
  // has already checked the stream.
  if (writer_stream.get() != nullptr) { record_writer_event_unchecked(writer_stream); }
}

std::unique_ptr<gpu_table_representation> gpu_table_representation::make_written_on(
  std::unique_ptr<cudf::table>&& table,
  cucascade::memory::memory_space& memory_space,
  ::cuda::stream_ref writer_stream)
{
  // Checked before the table is taken, so a rejected stream leaves it with the caller. Unlike
  // validate_writer_stream(), this also checks a null handle, which names the default stream.
  validate_stream_device(writer_stream, memory_space.get_device_id());
  // Constructed with an unknown writer so that the record below also covers a null handle.
  auto rep = std::make_unique<gpu_table_representation>(
    std::move(table), memory_space, ::cuda::stream_ref{cudaStream_t{nullptr}});
  rep->record_writer_event_unchecked(writer_stream);
  return rep;
}

gpu_table_representation::~gpu_table_representation()
{
  // STREAM-LINEAGE: release the writer event if one was recorded.
  if (_writer_event != nullptr) {
    CUCASCADE_ASSERT_CUDA_SUCCESS(cudaEventDestroy(_writer_event));
    _writer_event = nullptr;
  }
}

std::size_t gpu_table_representation::get_size_in_bytes() const
{
  if (std::holds_alternative<std::unique_ptr<cudf::table>>(_table)) {
    return std::get<std::unique_ptr<cudf::table>>(_table)->alloc_size();
  } else if (std::holds_alternative<owning_table_view>(_table)) {
    return std::get<owning_table_view>(_table).alloc_size;
  }
  return 0;
}

std::size_t gpu_table_representation::get_uncompressed_data_size_in_bytes() const
{
  return get_size_in_bytes();
}

cudf::table_view gpu_table_representation::get_table_view() const
{
  if (std::holds_alternative<std::unique_ptr<cudf::table>>(_table)) {
    return std::get<std::unique_ptr<cudf::table>>(_table)->view();
  } else {
    return std::get<owning_table_view>(_table).view;
  }
}

std::unique_ptr<cudf::table> gpu_table_representation::release_table(::cuda::stream_ref stream)
{
  CUCASCADE_FUNC_RANGE();
  // An earlier release left no table to read or bind, so there is nothing to check or wait for.
  if (auto const* const owned = std::get_if<std::unique_ptr<cudf::table>>(&_table);
      owned != nullptr && *owned == nullptr) {
    return nullptr;
  }
  validate_stream_device(stream, get_device_id());
  wait_for_writer(stream);
  if (auto const* const viewed = std::get_if<owning_table_view>(&_table)) {
    // The copy allocates from the current device's memory resource, so make this device current.
    rmm::cuda_set_device_raii const device_guard{rmm::cuda_device_id{get_device_id()}};
    // Replacing the view destroys its owner, which may free the viewed memory, so the copy must
    // complete first.
    _table = materialize_and_wait(viewed->view, stream);
  } else {
    // Rebind so the returned table's frees stay stream-ordered behind the caller's reads.
    rebind_table(std::get<std::unique_ptr<cudf::table>>(_table), stream);
  }
  return std::move(std::get<std::unique_ptr<cudf::table>>(_table));
}

void gpu_table_representation::rebind_stream(::cuda::stream_ref stream)
{
  // Only the owned-table alternative can be rebound: the owning_table_view alternative
  // references memory owned by an external (type-erased) owner, which manages its own
  // deallocation stream.
  if (!std::holds_alternative<std::unique_ptr<cudf::table>>(_table)) { return; }
  auto& table = std::get<std::unique_ptr<cudf::table>>(_table);
  if (!table || table->num_columns() == 0) { return; }

  validate_stream_device(stream, get_device_id());
  rebind_table(table, stream);
}

std::unique_ptr<idata_representation> gpu_table_representation::clone(::cuda::stream_ref stream)
{
  CUCASCADE_FUNC_RANGE();
  validate_stream_device(stream, get_device_id());
  // The copy allocates from the current device's memory resource, so make this device current.
  rmm::cuda_set_device_raii const device_guard{rmm::cuda_device_id{get_device_id()}};
  wait_for_writer(stream);
  auto cloned = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(get_table_view(), stream),
    get_memory_space(),
    ::cuda::stream_ref{cudaStream_t{nullptr}});
  cloned->record_writer_event_unchecked(stream);
  return cloned;
}

void gpu_table_representation::record_writer_event(::cuda::stream_ref writer_stream)
{
  validate_stream_device(writer_stream, get_device_id());
  record_writer_event_unchecked(writer_stream);
}

void gpu_table_representation::record_writer_event_unchecked(::cuda::stream_ref writer_stream)
{
  // The writer stays unknown until the record succeeds: on any failure the event is destroyed
  // rather than left to claim writes it does not cover.
  unique_event event{std::exchange(_writer_event, nullptr)};
  if (!event) {
    // An event can only be recorded on a stream of the device it was created on.
    rmm::cuda_set_device_raii const device_guard{rmm::cuda_device_id{get_device_id()}};
    cudaEvent_t created{};
    CUCASCADE_CUDA_TRY(cudaEventCreateWithFlags(&created, cudaEventDisableTiming));
    event.reset(created);
  }
  cucascade::cuda::cuda_event_view{event.get()}.record(writer_stream);
  _writer_event = event.release();
}

cudaEvent_t gpu_table_representation::get_writer_event() const { return _writer_event; }

void gpu_table_representation::wait_for_writer(::cuda::stream_ref reader_stream) const
{
  if (_writer_event != nullptr) {
    cucascade::cuda::cuda_event_view{_writer_event}.wait(reader_stream);
  } else {
    rmm::cuda_set_device_raii const device_guard{rmm::cuda_device_id{get_device_id()}};
    CUCASCADE_CUDA_TRY(cudaDeviceSynchronize());
  }
}

void gpu_table_representation::validate_writer_stream(
  ::cuda::stream_ref writer_stream, cucascade::memory::memory_space const& memory_space)
{
  if (writer_stream.get() != nullptr) {
    validate_stream_device(writer_stream, memory_space.get_device_id());
  }
}

std::unique_ptr<cudf::table> gpu_table_representation::take_owned_table(
  std::unique_ptr<cudf::table>&& table,
  cucascade::memory::memory_space const& memory_space,
  ::cuda::stream_ref writer_stream)
{
  validate_writer_stream(writer_stream, memory_space);
  return std::move(table);
}

}  // namespace cucascade
