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

#include <cstddef>

namespace cucascade {

/**
 * @brief Bounds on the pinned host memory one GPU -> GPU conversion borrows when the two devices
 * cannot copy to each other directly.
 *
 * Such a conversion moves the table through a ring of host buffers: up to @ref ring_depth chunks
 * of up to @ref chunk_bytes each are in flight at once, so it never holds more than their product
 * (rounded up to the HOST pool's block size), and a table smaller than that holds about its own
 * size. Larger chunks and a deeper ring trade host memory for fewer, longer copies.
 */
class staged_copy_settings {
 public:
  static constexpr std::size_t default_chunk_bytes = std::size_t{4} << 20;  ///< 4 MiB
  static constexpr std::size_t default_ring_depth  = 4;
  static constexpr std::size_t max_chunk_bytes     = std::size_t{1} << 30;  ///< 1 GiB
  static constexpr std::size_t max_ring_depth      = 64;

  /** @brief The defaults: four chunks of 4 MiB. */
  staged_copy_settings() = default;

  /**
   * @param chunk_bytes Most bytes in one chunk, from 1 to @ref max_chunk_bytes
   * @param ring_depth Most chunks in flight at once, from 1 to @ref max_ring_depth
   * @throws std::invalid_argument if a value is zero or above its maximum
   */
  staged_copy_settings(std::size_t chunk_bytes, std::size_t ring_depth);

  [[nodiscard]] std::size_t chunk_bytes() const noexcept { return _chunk_bytes; }
  [[nodiscard]] std::size_t ring_depth() const noexcept { return _ring_depth; }

 private:
  std::size_t _chunk_bytes{default_chunk_bytes};
  std::size_t _ring_depth{default_ring_depth};
};

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

/**
 * @brief register_builtin_converters() with explicit bounds for host-staged GPU -> GPU conversions.
 *
 * @param registry The converter registry to register converters with.
 * @param staged_copy Bounds copied into the registered GPU -> GPU converter.
 */
void register_builtin_converters(representation_converter_registry& registry,
                                 staged_copy_settings const& staged_copy);

}  // namespace cucascade
