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
 * @file gpu_numa_node.hpp
 * @brief Which NUMA node a GPU is attached to, for placing pinned host memory next to it.
 *
 * Nothing here allocates memory, so the functions cannot fail for lack of it.
 */

namespace cucascade {
namespace memory {

/**
 * @brief NUMA node that the PCI device of GPU @p device_id is attached to.
 *
 * The node of each of the first 64 CUDA devices is looked up once per process, on the first call;
 * later calls return the remembered value. A device beyond those is looked up on every call.
 *
 * @param device_id CUDA device ordinal
 * @return The NUMA node, or -1 when @p device_id is not a visible CUDA device or the system does
 * not report a node for it
 */
[[nodiscard]] int gpu_numa_node(int device_id) noexcept;

namespace detail {

/**
 * @brief The lookup behind gpu_numa_node(), without the per-process memory: asks CUDA for the
 * GPU's PCI bus id and reads `/sys/bus/pci/devices/<bus id>/numa_node`.
 *
 * @param device_id CUDA device ordinal
 * @return The NUMA node, or -1 when it cannot be determined
 */
[[nodiscard]] int read_gpu_numa_node(int device_id) noexcept;

}  // namespace detail

}  // namespace memory
}  // namespace cucascade
