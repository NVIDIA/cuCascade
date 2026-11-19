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

#pragma once

#include <cucascade/cuda/stream.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/column_metadata.hpp>

#include <cstddef>
#include <vector>

namespace cudf {
class table_view;
}

namespace cucascade {

class gpu_table_representation;

/**
 * @brief Column buffer offsets and payload size before rounding to HOST blocks.
 */
struct gpu_to_host_copy_layout {
  std::vector<memory::column_metadata> columns;
  std::size_t bytes;
};

/**
 * @brief Plan the HOST buffer layout for a GPU table.
 *
 * @param view GPU table view whose columns have zero offsets
 * @param stream CUDA stream used to read column metadata
 * @return Column buffer offsets and total payload size before rounding to HOST blocks
 * @pre Keep the source alive and order @p stream after its writer.
 */
[[nodiscard]] gpu_to_host_copy_layout plan_gpu_to_host_copy(const cudf::table_view& view,
                                                            ::cuda::stream_ref stream);

/**
 * @brief Calculate the GPU-to-HOST copy size.
 *
 * Orders the metadata reads after the source's writer.
 *
 * @param source GPU table representation to size
 * @param stream CUDA stream used to read column metadata
 * @return Number of payload bytes before rounding to HOST blocks
 * @pre Hold the source batch against mutation for the duration of the call.
 */
[[nodiscard]] std::size_t gpu_to_host_staging_bytes(const gpu_table_representation& source,
                                                    ::cuda::stream_ref stream);

/**
 * @brief Initialize the cudf-backed built-in representation converters.
 *
 * Registers converters between all supported cudf-backed representation types (GPU, HOST, DISK).
 * Disk converters resolve the I/O backend from the disk memory_space at conversion time, so each
 * disk memory_space can use a different backend.
 *
 * This is provided by the cucascade-cudf library; the cudf-free core registry ships empty.
 *
 * Ordering: a converter reading a GPU source orders its reads after the source's writer event
 * (idata_representation::get_writer_event()); callers must not rely on the caller's stream for
 * that ordering. Writes enqueued after the last record_writer_event() are not covered, on any
 * stream, so a writer must record before converting. Without a writer event the converter
 * synchronizes the source device. A GPU result carries its own writer event, which readers on
 * other streams wait on.
 *
 * @param registry The converter registry to register converters with.
 */
void register_builtin_converters(representation_converter_registry& registry);

}  // namespace cucascade
