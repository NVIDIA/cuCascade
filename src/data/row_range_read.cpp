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

#include <cucascade/data/row_range_read.hpp>
#include <cucascade/error.hpp>

#include <cuda_runtime_api.h>

#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <typeinfo>

namespace cucascade {

// =============================================================================
// row_range_reader_registry implementation
//
// cuCascade core ships no built-in readers; only the registry machinery lives here.
// Concrete GPU/HOST/DISK readers are provided by the cucascade-cudf layer.
// =============================================================================

void row_range_reader_registry::register_reader_impl(std::type_index type, row_range_read_fn fn)
{
  if (!fn) {
    std::ostringstream oss;
    oss << "Cannot register an empty row-range reader for source type '" << type.name() << "'";
    throw std::invalid_argument(oss.str());
  }

  std::unique_lock lock(_mutex);

  if (_readers.find(type) != _readers.end()) {
    std::ostringstream oss;
    oss << "Row-range reader already registered for source type '" << type.name() << "'";
    throw std::invalid_argument(oss.str());
  }

  _readers.emplace(type, std::move(fn));
}

bool row_range_reader_registry::has_reader(const idata_representation& source) const
{
  return has_reader_impl(std::type_index(typeid(source)));
}

bool row_range_reader_registry::has_reader_impl(std::type_index type) const
{
  std::shared_lock lock(_mutex);
  return _readers.find(type) != _readers.end();
}

row_range_result row_range_reader_registry::read(
  const idata_representation& source,
  const row_range_request& request,
  std::span<column_destination> dst,
  ::cuda::stream_ref stream,
  memory::reservation* scratch,
  std::unique_ptr<row_range_reader_state>* state) const
{
  if (dst.size() != request.columns.size()) {
    std::ostringstream oss;
    oss << "Row-range read needs one destination per requested column: got " << dst.size()
        << " destinations for " << request.columns.size() << " columns";
    throw std::invalid_argument(oss.str());
  }

  // Copy the reader out so it runs without the registry lock held.
  row_range_read_fn reader;
  {
    std::shared_lock lock(_mutex);

    auto it = _readers.find(std::type_index(typeid(source)));
    if (it == _readers.end()) {
      std::ostringstream oss;
      oss << "No row-range reader registered for source type '" << typeid(source).name() << "'";
      throw row_range_unsupported_error(oss.str());
    }

    reader = it->second;
  }

  if (cudaEvent_t writer_event = source.get_writer_event(); writer_event != nullptr) {
    CUCASCADE_CUDA_TRY(::cudaStreamWaitEvent(stream.get(), writer_event, 0));
  }

  return reader(source, request, dst, stream, scratch, state);
}

row_range_result read_rows_into(const row_range_reader_registry& registry,
                                const read_only_data_batch& batch,
                                const row_range_request& request,
                                std::span<column_destination> dst,
                                ::cuda::stream_ref stream,
                                memory::reservation* scratch,
                                std::unique_ptr<row_range_reader_state>* state)
{
  const idata_representation* source = batch.get_data();
  if (source == nullptr) { throw std::logic_error("read_rows_into: the batch has no data"); }

  auto result = registry.read(*source, request, dst, stream, scratch, state);
  // Only a read that was dispatched is recorded; a throwing read propagates without an event.
  batch.record_reader_event(stream);
  return result;
}

}  // namespace cucascade
