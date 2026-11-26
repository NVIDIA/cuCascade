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
 * @brief Which copy route the cross-GPU converter uses to move a gpu_table_representation to
 * another GPU.
 */
enum class copy_route_selection {
  /**
   * @brief Use the peer route when cucascade::memory::probe_peer_dma_works() verifies the
   * source-to-destination direction, and the host-staged route otherwise.
   */
  automatic,
  /**
   * @brief Always use the peer route; a conversion between GPUs whose peer DMA fails verification
   * throws cucascade::logic_error instead of copying.
   */
  peer,
  /**
   * @brief Always use the host-staged pipeline, even between GPUs with working peer DMA.
   */
  host_staged,
};

/**
 * @brief Tuning for the cross-GPU (GPU -> GPU) converter registered by
 * register_builtin_converters().
 *
 * The host-staged pipeline moves a table's leaf buffers as one byte stream, cut into chunks. Each
 * chunk passes through one of `pipeline_depth` staging slots of `chunk_size_bytes` in pinned host
 * memory, so the source-to-host copy of one chunk overlaps the host-to-destination copy of another.
 * A chunk fills its staging slot except for gaps of less than a host page in front of leaf buffers
 * of 32 KiB or more: these start on a page boundary, where the host-to-destination copy is fastest.
 *
 * Each conversion reserves its staging slots (pipeline_depth x chunk_size_bytes, less for a table
 * smaller than that) from a HOST memory_space of the memory_reservation_manager that owns the
 * destination memory_space: the one on the destination GPU's NUMA node, or else the first HOST
 * memory_space of that manager, which is slower but correct. When the reservation or the allocation
 * behind it fails, the conversion retries with fewer staging slots, down to one, and throws
 * cucascade::memory::cucascade_out_of_memory if even one does not fit. It never waits for HOST
 * memory to be released. A host-staged conversion whose destination memory_space has no owning
 * memory_reservation_manager, or whose manager has no HOST memory_space, throws
 * cucascade::logic_error.
 *
 * A usable configuration has a positive `chunk_size_bytes` and `pipeline_depth` whose product fits
 * in std::size_t.
 */
struct cross_gpu_copy_config {
  static constexpr std::size_t default_chunk_size_bytes = 8ULL << 20;  ///< Default chunk size
  static constexpr std::size_t default_pipeline_depth   = 4;           ///< Default staging slots

  std::size_t chunk_size_bytes{default_chunk_size_bytes};       ///< Bytes per chunk
  std::size_t pipeline_depth{default_pipeline_depth};           ///< Staging slots per conversion
  copy_route_selection route{copy_route_selection::automatic};  ///< Copy route override
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
 * The cross-GPU converter uses a default-constructed cross_gpu_copy_config. It copies a table
 * buffer by buffer, so it throws std::invalid_argument for a source whose table view is sliced (a
 * column view with a non-zero offset) instead of copying the wrong rows.
 *
 * @param registry The converter registry to register converters with.
 */
void register_builtin_converters(representation_converter_registry& registry);

/**
 * @brief Initialize the cudf-backed built-in representation converters, tuning the cross-GPU
 * converter with @p config.
 *
 * Same as register_builtin_converters(representation_converter_registry&) otherwise.
 *
 * @param registry The converter registry to register converters with.
 * @param config Copy route and host-staged pipeline settings for the cross-GPU converter
 * @throws std::invalid_argument if @p config is not usable (see cross_gpu_copy_config)
 */
void register_builtin_converters(representation_converter_registry& registry,
                                 cross_gpu_copy_config const& config);

}  // namespace cucascade
