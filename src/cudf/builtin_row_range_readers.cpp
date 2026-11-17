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

#include <cucascade/cudf/builtin_row_range_readers.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/disk_data_representation.hpp>

#include <memory>
#include <span>
#include <utility>

namespace cucascade {

void register_builtin_row_range_readers(row_range_reader_registry& registry,
                                        std::shared_ptr<io::ioctx> io_context)
{
  registry.register_reader<gpu_table_representation>(
    [](const idata_representation& source,
       const row_range_request& request,
       std::span<column_destination> dst,
       ::cuda::stream_ref stream,
       memory::reservation* scratch,
       std::unique_ptr<row_range_reader_state>* state) {
      return detail::read_rows_gpu(source, request, dst, stream, scratch, state);
    });

  registry.register_reader<host_data_representation>(
    [](const idata_representation& source,
       const row_range_request& request,
       std::span<column_destination> dst,
       ::cuda::stream_ref stream,
       memory::reservation* scratch,
       std::unique_ptr<row_range_reader_state>* state) {
      return detail::read_rows_host(source, request, dst, stream, scratch, state);
    });

  registry.register_reader<disk_data_representation>(
    [io_context = std::move(io_context)](const idata_representation& source,
                                         const row_range_request& request,
                                         std::span<column_destination> dst,
                                         ::cuda::stream_ref stream,
                                         memory::reservation* scratch,
                                         std::unique_ptr<row_range_reader_state>* state) {
      return detail::read_rows_disk(source, request, dst, stream, scratch, state, io_context);
    });
}

}  // namespace cucascade
