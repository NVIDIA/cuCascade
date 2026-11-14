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
#include <optional>

namespace cucascade::io::cufile {
struct cufile_api;
}  // namespace cucascade::io::cufile

namespace cucascade::io::cufile::detail {

/// Serve every cuFile call of the process from @p api instead of libcufile
/// (null restores libcufile), so tests can script reads, failures and short
/// returns.  Process-wide; install it before opening the objects it should
/// serve and remove it after they are closed, since handles registered by one
/// table mean nothing to the other.  @c cufile_api::open_driver() then reports
/// @p api's @c driver_open outcome.
void set_cufile_api_for_testing(cufile_api const* api) noexcept;

/// Make @c gds_available() report @p value (unset restores the platform probe),
/// so tests can run automatic staging on a host without a GDS stack.
/// Process-wide.
void set_gds_available_for_testing(std::optional<bool> value) noexcept;

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

/**
 * @brief How many @c cuFileRead calls a staged read of [@p begin, @p end) makes
 *        into a slot of @p slot_bytes, absent short reads.
 *
 * Each call widens its piece to 4 KiB boundaries and covers at most
 * @p slot_bytes; a piece reaching into the file's partial last block reads the
 * aligned body and the tail separately, never past @p file_size.
 *
 * @pre end <= file_size; @p slot_bytes a multiple of 4 KiB.
 */
[[nodiscard]] std::size_t staged_calls(std::size_t begin,
                                       std::size_t end,
                                       std::size_t file_size,
                                       std::size_t slot_bytes) noexcept;

/// Hold every reactor's staging-copy stream on the host for @p delay before each
/// staging copy (0 = off), so tests can catch a scratch slot reused, or a request
/// settled, before its copy landed.  Process-wide.
void set_staging_copy_delay_for_testing(std::chrono::microseconds delay) noexcept;

/// Make every reactor report a failed CUDA event record after enqueuing an
/// H2D copy (staged device reads and host+device pairs), so tests can check
/// the request does not settle while that copy may still run.  Process-wide.
void set_copy_event_record_failure_for_testing(bool enabled) noexcept;

}  // namespace cucascade::io::cufile::detail
