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

#include <cucascade/io/cufile/platform.hpp>

#include <chrono>
#include <cstddef>

namespace cucascade::io::cufile::detail {

/**
 * @brief The verdict of @ref probe_path from the evidence already in @p info.
 *
 * Reads the policy, platform, storage and opened-file fields and sets
 * @c capability, @c reason, @c matched_class, @c class_supported and
 * @c filesystem_supported.  Pure: no I/O, no cuFile calls.
 */
void decide_path_capability(path_info& info) noexcept;

/// Whether a CUFILE_FORCE_COMPAT_MODE value forces compatibility mode as
/// libcufile (1.17) reads it: its first four characters are true, any case;
/// 1/on/yes do not. Pure.
[[nodiscard]] bool force_compat_env_on(char const* value) noexcept;

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

/// Make every reactor report a failed CUDA event record after enqueuing a
/// staging copy (H2D for staged device reads and host+device pairs, D2H for
/// staged device writes), so tests can check the request does not settle, nor
/// the scratch get reused, while that copy may still run.  Process-wide.
void set_copy_event_record_failure_for_testing(bool enabled) noexcept;

/// @c cuFileWrite calls issued by every cuFile reactor so far.
[[nodiscard]] std::size_t cufile_write_calls_for_testing() noexcept;

/// Buffers currently registered with cuFile through the reactors' registration
/// table (@c config::register_min_bytes).
[[nodiscard]] std::size_t live_registrations_for_testing() noexcept;

/// Run @p hook(@p arg) on the worker just before every @c cuFileRead /
/// @c cuFileWrite call (null removes it).  Process-wide.
void set_cufile_transfer_hook_for_testing(void (*hook)(void*), void* arg) noexcept;

/// Whether [@p base, @p base + @p bytes) overlaps a live registration or a
/// cuFile transfer in flight, as the registration table sees it.
[[nodiscard]] bool transfer_tracked_for_testing(void const* base, std::size_t bytes) noexcept;

/// Buffer registrations the reactors' registration table has created so far.
[[nodiscard]] std::size_t registrations_created_for_testing() noexcept;

}  // namespace cucascade::io::cufile::detail
