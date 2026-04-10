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
std::unique_ptr<idisk_io_backend> make_pipeline_io_backend(bool direct_io = false);

}  // namespace cucascade
