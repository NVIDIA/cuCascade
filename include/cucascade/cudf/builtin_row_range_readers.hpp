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

#include <cucascade/data/row_range_read.hpp>

#include <memory>

namespace cucascade {

namespace io {
class ioctx;
}  // namespace io

/**
 * @brief Register the cudf-backed built-in row-range readers.
 *
 * Registers readers for gpu_table_representation, host_data_representation and
 * disk_data_representation. host_data_packed_representation is NOT registered, so reading one
 * throws row_range_unsupported_error.
 *
 * This is provided by the cucascade-cudf library; the cudf-free core registry ships empty.
 *
 * @param registry The reader registry to register the readers with.
 * @param io_context io context the disk reader reads with. nullptr means the disk reader uses
 *        the io context of the source's DISK memory space (adopting default_disk_io_context()
 *        when the space has none yet), so each disk memory_space can use a different one.
 * @throws std::invalid_argument if a reader for one of these types is already registered.
 */
void register_builtin_row_range_readers(row_range_reader_registry& registry,
                                        std::shared_ptr<io::ioctx> io_context = nullptr);

}  // namespace cucascade
