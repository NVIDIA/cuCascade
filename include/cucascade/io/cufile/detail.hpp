/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
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

/**
 * @file
 * @brief Internals of the cuFile reactor exposed for its unit tests only.  Not
 *        part of the supported API; nothing outside the reactor and its tests
 *        should include this header.
 */

#include <chrono>
#include <cstddef>

namespace cucascade::io::cufile::detail {

/**
 * @brief Length of the next @c cuFileRead of a device slice.
 *
 * The piece starting at @p pos ends at @p end, or earlier: a piece longer than
 * @p max_op stops at the last 4 KiB file boundary within it (so every piece after
 * the first starts aligned), and a piece reaching an unaligned end of file stops
 * at @c align_down(file_size) so the aligned body stays eligible for direct DMA
 * and the sub-block tail is read on its own.
 *
 * @pre pos < end <= file_size, max_op >= IO_BLOCK_SIZE.
 */
[[nodiscard]] std::size_t next_piece(std::size_t pos,
                                     std::size_t end,
                                     std::size_t file_size,
                                     std::size_t max_op) noexcept;

/// Let aligned device-write bodies use @c cuFileWrite even on a platform without
/// a real GDS mode, so hosts without GDS still cover that path.  Process-wide.
void set_cufile_writes_in_compat_for_testing(bool enabled) noexcept;

/// Hold every reactor's staging-copy stream on the host for @p delay before each
/// staging copy (0 = off), so tests can catch a scratch slot reused, or a request
/// settled, before its copy landed.  Process-wide.
void set_staging_copy_delay_for_testing(std::chrono::microseconds delay) noexcept;

/// @c cuFileWrite calls issued by every cuFile reactor so far.
[[nodiscard]] std::size_t cufile_write_calls_for_testing() noexcept;

/// Buffers currently registered with cuFile through the reactors' registration
/// table (@c config::register_min_bytes).
[[nodiscard]] std::size_t live_registrations_for_testing() noexcept;

}  // namespace cucascade::io::cufile::detail
