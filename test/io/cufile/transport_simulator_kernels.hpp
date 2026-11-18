/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace cucascade::io::cufile {

// Prepared in stream order by a host function, read by the following kernel through mapped
// pinned memory. Keeping addresses/count here (not launch parameters) models cuFile's timing.
struct simulator_copy_args {
  unsigned char* image;
  unsigned char* buffer;
  std::size_t file_offset;
  std::size_t buffer_offset;
  std::size_t count;
  bool write;
  unsigned released;
  std::uint64_t completion_latency_ns;
};

[[nodiscard]] cudaError_t launch_simulator_gate(simulator_copy_args* args,
                                                cudaStream_t stream,
                                                bool* possibly_enqueued = nullptr) noexcept;
[[nodiscard]] cudaError_t launch_simulator_copy(simulator_copy_args const* args,
                                                cudaStream_t stream) noexcept;

}  // namespace cucascade::io::cufile
