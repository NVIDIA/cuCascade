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

#include <cucascade/data/disk_io_backend.hpp>

#include <memory>

namespace cucascade {

/// @brief Create a double-buffered pipeline I/O backend instance.
/// @param direct_io When true, open data files with O_DIRECT (bypass page cache).
/// @param target_device CUDA device id that the backend will primarily serve. When
///        provided (>= 0) and the GPU's PCI device has a known NUMA node, the
///        internal pinned host buffers are page-bound to that NUMA node to avoid
///        cross-socket D2H/H2D traffic. -1 (default) falls back to non-bound
///        portable+mapped pinned memory.
std::unique_ptr<idisk_io_backend> make_pipeline_io_backend(bool direct_io    = false,
                                                           int target_device = -1);

}  // namespace cucascade
