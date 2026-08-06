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
  : idata_representation(memory_space),
    _alloc_size(table ? table->alloc_size() : 0),
    _table(std::move(table))
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

std::size_t gpu_table_representation::get_size_in_bytes() const { return _alloc_size; }

std::size_t gpu_table_representation::get_uncompressed_data_size_in_bytes() const
{
  return get_size_in_bytes();
}

cudf::table_view gpu_table_representation::get_table_view() const { return _table.view(); }

std::unique_ptr<cudf::table> gpu_table_representation::release_table(rmm::cuda_stream_view stream)
{
  // STREAM-LINEAGE: order the (potentially materializing) read after the
  // recorded writer event before touching the underlying buffers on `stream`.
  if (_writer_event != nullptr) { cucascade::cuda::cuda_event_view{_writer_event}.wait(stream); }
  auto table  = _table.release(stream, get_memory_space().get_default_allocator());
  _alloc_size = 0;
  return table;
}

void gpu_table_representation::materialize_table(rmm::cuda_stream_view stream)
{
  if (!_table || _table.is_materialized()) { return; }
  // STREAM-LINEAGE: wait-then-record — `stream` must observe the producing
  // writes before the materializing read (move or copy), and the writer event
  // is re-recorded afterwards so future readers order against the
  // materialization. When no writer event was recorded (legacy paths), the
  // caller must pass a stream already ordered after the producing writes.
  if (_writer_event != nullptr) { cucascade::cuda::cuda_event_view{_writer_event}.wait(stream); }
  _table.materialize(stream, get_memory_space().get_default_allocator());
  _alloc_size = _table.alloc_size();
  record_writer_event(stream);
}

void gpu_table_representation::rebind_stream(rmm::cuda_stream_view stream)
{
  // Only the owned-table state can be rebound: a view state references memory
  // owned by an external (type-erased) owner, which manages its own
  // deallocation stream.
  if (!_table.is_materialized()) { return; }
  // Pure move — the materialized state surrenders its table without touching
  // device memory (stream/mr are unused on this path).
  auto table = _table.release(stream, get_memory_space().get_default_allocator());
  if (table->num_columns() > 0) {
    // cudf::table move-assignment is deleted, so release the columns, rebind each, and rebuild
    // the table in place. No device memory is copied and no kernels are launched.
    auto columns = table->release();
    for (auto& col : columns) {
      col = cudf::rebind_stream(std::move(*col), stream);
    }
    table = std::make_unique<cudf::table>(std::move(columns));
  }
  _table = owning_table_view{std::move(table)};
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
