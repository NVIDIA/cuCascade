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

namespace cucascade {

gpu_table_representation::gpu_table_representation(std::unique_ptr<cudf::table> table,
                                                   cucascade::memory::memory_space& memory_space,
                                                   rmm::cuda_stream_view writer_stream)
  : idata_representation(memory_space), _table(std::move(table))
{
  // STREAM-LINEAGE: record the writer event in the constructor body so every
  // representation is born with a recorded event. Skipping when the caller
  // passes a default-constructed (per-thread default) stream view preserves
  // legacy behavior for callers that genuinely have no writer stream — they
  // will fall back to cudaDeviceSynchronize on the source device in
  // convert_gpu_to_gpu(). All non-legacy callers MUST pass a real writer
  // stream.
  if (writer_stream.value() != nullptr) { record_writer_event(writer_stream); }
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

std::unique_ptr<cudf::table> gpu_table_representation::release_table(rmm::cuda_stream_view stream)
{
  if (std::holds_alternative<owning_table_view>(_table)) {
    _table = std::make_unique<cudf::table>(std::get<owning_table_view>(_table).view, stream);
  } else {
    // Rebind so the returned table's frees stay stream-ordered behind the caller's reads.
    gpu_table_representation::rebind_stream(stream);
  }
  return std::move(std::get<std::unique_ptr<cudf::table>>(_table));
}

void gpu_table_representation::rebind_stream(rmm::cuda_stream_view stream)
{
  // Only the owned-table alternative can be rebound: the owning_table_view alternative
  // references memory owned by an external (type-erased) owner, which manages its own
  // deallocation stream.
  if (!std::holds_alternative<std::unique_ptr<cudf::table>>(_table)) { return; }
  auto& table = std::get<std::unique_ptr<cudf::table>>(_table);
  if (!table || table->num_columns() == 0) { return; }

  // cudf::table move-assignment is deleted, so release the columns, rebind each, and rebuild
  // the table in place. No device memory is copied and no kernels are launched.
  auto columns = table->release();
  for (auto& col : columns) {
    col = cudf::rebind_stream(std::move(*col), stream);
  }
  table = std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<idata_representation> gpu_table_representation::clone(rmm::cuda_stream_view stream)
{
  // Create a deep copy of the cuDF table using the provided stream.
  // STREAM-LINEAGE: the clone has been written by `stream`; record an event on
  // it so any cross-stream/cross-device reader of the clone honors the
  // producer-consumer ordering established by record_writer_event().
  cudf::table_view view = get_table_view();
  auto cloned           = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(view, stream), get_memory_space(), stream);
  return cloned;
}

void gpu_table_representation::record_writer_event(rmm::cuda_stream_view writer_stream)
{
  // STREAM-LINEAGE: lazily create the event on first call (cudaEventDisableTiming —
  // used solely for cross-stream ordering, never for elapsed-time queries).
  if (_writer_event == nullptr) {
    CUCASCADE_CUDA_TRY(cudaEventCreateWithFlags(&_writer_event, cudaEventDisableTiming));
  }
  cucascade::cuda::cuda_event_view{_writer_event}.record(writer_stream);
}

cudaEvent_t gpu_table_representation::get_writer_event() const { return _writer_event; }

}  // namespace cucascade
