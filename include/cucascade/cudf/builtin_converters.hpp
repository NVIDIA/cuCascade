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

#include <cucascade/data/representation_converter.hpp>

namespace cucascade {

/**
 * @brief Initialize the cudf-backed built-in representation converters.
 *
 * Registers converters between all supported cudf-backed representation types (GPU, HOST, DISK,
 * REMOTE). Disk converters perform their I/O through the io context of the DISK memory_space
 * (memory_space::get_io_context()), resolved at conversion time, so each disk memory_space can use
 * a different io stack; a space without one adopts default_disk_io_context()
 * (see cucascade/cudf/disk_io_context.hpp).
 *
 * REMOTE converters (HOST/GPU <-> remote_data_representation) write through the io context of
 * the target REMOTE memory_space, which must have one (std::logic_error otherwise; see
 * cucascade/cudf/remote_io_context.hpp), and read back through the context that wrote the object.
 * See docs/io/remote-tier-design.md.
 *
 * This is provided by the cucascade-cudf library; the cudf-free core registry ships empty.
 *
 * @param registry The converter registry to register converters with.
 */
void register_builtin_converters(representation_converter_registry& registry);

}  // namespace cucascade
