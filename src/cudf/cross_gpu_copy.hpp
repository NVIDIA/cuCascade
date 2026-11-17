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

#pragma once

/**
 * @file cross_gpu_copy.hpp
 * @brief Copies a gpu_table_representation's table to another GPU over the peer route or the
 * host-staged pipeline.
 *
 * Used by the GPU -> GPU converter that register_builtin_converters() registers.
 */

#include <cucascade/cuda/stream.hpp>
#include <cucascade/cudf/builtin_converters.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cudf/table/table.hpp>

#include <memory>

namespace cucascade {

/**
 * @brief Throw std::invalid_argument unless @p config is usable, as cross_gpu_copy_config defines.
 *
 * @param config Settings to check
 */
void validate_cross_gpu_copy_config(cross_gpu_copy_config const& config);

/**
 * @brief Copy @p source's table into new buffers on @p target's GPU.
 *
 * The copy route comes from @p config.route. Before any destination buffer is allocated, the chars
 * size of every strings column is learned through one batched device-to-host read, followed by the
 * only planning sync. The source is read on a stream from the source memory_space's stream pool
 * (the planning read and the host-staged route) or on @p target_stream (peer route), each ordered
 * after the source's writer as gpu_table_representation::wait_for_writer() describes.
 *
 * Both copy routes build the destination table column by column and start no copy into a column
 * before that column is finished. The peer route copies the leaf buffers of the finished columns
 * directly on @p target_stream, while the columns after them are allocated.
 *
 * The host-staged pipeline takes its staging slots as cross_gpu_copy_config describes and releases
 * them before returning. Each staging slot has a "filled" event recorded after its source-to-host
 * copy and a "drained" event recorded after its host-to-destination copy on @p target_stream; the
 * next source-to-host copy into that slot waits for "drained". When the table has more than one
 * chunk, the first chunks are staged while the destination columns are allocated; the
 * host-to-destination copies start once the last column is finished.
 *
 * Every copy has completed when this returns, so the caller may free @p source right away. Readers
 * of the result on streams other than @p target_stream must be ordered after it, for example
 * through the writer event that the gpu_table_representation constructor records on @p
 * target_stream.
 *
 * On an exception no table is returned and no copy is still running when the exception leaves. A
 * column that fails while it is being built frees its buffers right away, and nothing writes them
 * yet. The finished columns are freed, and the staging slots returned to the HOST pool, only after
 * the source stream and @p target_stream have been drained. @p target's default allocator therefore
 * never frees a buffer that a copy still writes.
 *
 * @pre No column view of @p source's table is sliced: each has offset 0, at every nesting level.
 *
 * @param source Table to copy; must live on a different GPU than @p target
 * @param target GPU memory_space whose default allocator provides the destination buffers
 * @param target_stream Stream on @p target's device that allocates and writes the destination
 * buffers
 * @param config Copy route and host-staged pipeline settings, already validated
 * @return The copied table on @p target's device
 * @throws std::invalid_argument if a column view of @p source's table has a non-zero offset, if a
 * strings column with rows has no offsets child, or if a LIST or DICTIONARY32 column lacks one of
 * its two children. The first two are detected before anything is read, allocated or copied.
 * @throws cucascade::logic_error if the peer route is forced and peer DMA fails verification, or if
 * the host-staged route finds no HOST memory_space to stage in (see copy_route_selection::peer and
 * cross_gpu_copy_config)
 * @throws cucascade::memory::cucascade_out_of_memory if not even one staging slot can be reserved
 * and allocated, or if a destination buffer cannot be allocated and @p target's out-of-memory
 * policy does not recover
 * @throws cucascade::cuda_error if a CUDA runtime call fails
 */
[[nodiscard]] std::unique_ptr<cudf::table> copy_table_to_gpu(gpu_table_representation const& source,
                                                             memory::memory_space const& target,
                                                             ::cuda::stream_ref target_stream,
                                                             cross_gpu_copy_config const& config);

}  // namespace cucascade
