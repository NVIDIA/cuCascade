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

#include "staged_copy.hpp"

#include <cucascade/data/representation_converter.hpp>

namespace cucascade {
namespace detail {

/**
 * @brief Options of the GPU -> GPU converter beyond what the public registration exposes.
 *
 * They belong to the registry they were registered with, so two registries in one process can
 * route differently.
 */
struct gpu_to_gpu_converter_options {
  /**
   * @brief Stage every conversion through pinned host memory.
   *
   * Applies to device pairs whose peer route is verified and to conversions within one device, so
   * a machine with a single GPU, or with working peer access, still covers the staged route.
   */
  bool force_staged_route{false};

  /// Bounds on the host memory a staged conversion borrows.
  memory::detail::staged_copy_ring_settings ring_settings{};

  /// Hooks and counters handed to every staged conversion; must outlive the registry. Null in
  /// production.
  memory::detail::staged_copy_instrumentation* instrumentation{nullptr};
};

/**
 * @brief register_builtin_converters() with explicit GPU -> GPU converter options.
 *
 * @param registry The converter registry to register converters with.
 * @param gpu_to_gpu_options Options copied into the registered GPU -> GPU converter.
 */
void register_builtin_converters(representation_converter_registry& registry,
                                 gpu_to_gpu_converter_options const& gpu_to_gpu_options);

}  // namespace detail
}  // namespace cucascade
