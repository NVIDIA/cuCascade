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

#include <cucascade/io/cufile/config.hpp>

#include <cstddef>

namespace cucascade::io::cufile {

/**
 * @brief The registered device staging a GPU holds for the cuFile backend.
 *
 * Staging slots are process-wide: allocated outside RMM, registered with cuFile
 * once, reused by every reactor and ioctx, and kept until the process exits (see
 * @ref config::max_staging_bytes_per_device).
 */
struct staging_usage {
  std::size_t allocated_bytes{0};   ///< Device memory held for staging.
  std::size_t registered_bytes{0};  ///< Of it, registered with @c cuFileBufRegister.
  std::size_t free_slots{0};        ///< Slots no reactor holds right now.
  std::size_t leased_slots{0};      ///< Slots reactors hold right now.
};

/**
 * @brief Allocate and register staging for @p device up front.
 *
 * Makes sure the process holds at least @p slots slots of
 * @c cfg.staging_slot_bytes() on @p device (within
 * @c cfg.max_staging_bytes_per_device), so the first reads do not pay for
 * @c cudaMalloc and @c cuFileBufRegister, and so the memory is taken before a
 * GPU memory pool claims the rest of the device.  Opens the cuFile driver.
 * Requires a CUDA device; never throws.
 *
 * @return How many registered slots of that size @p device holds afterwards
 *         (fewer than asked when the cap, device memory, BAR1 or cuFile's pinned
 *         memory limit stopped it; 0 when cuFile cannot register memory).
 */
[[nodiscard]] std::size_t reserve_staging(int device,
                                          std::size_t slots,
                                          config const& cfg) noexcept;

/**
 * @brief Make sure @p device holds at least @p slots staging slots of @p slot_bytes, within
 *        @p cap_bytes of staging on the device (0 = no cap).
 *
 * The general form of @ref reserve_staging, also used for the fused-read lane slots.  New
 * slots are registered with cuFile when it can register memory; with
 * @p require_registered only registered slots count (and none is allocated when
 * registration is impossible).  Opens the cuFile driver.  Requires a CUDA device; never
 * throws.
 *
 * @return How many slots of that size (registered ones only, with @p require_registered)
 *         @p device holds afterwards.
 */
[[nodiscard]] std::size_t reserve_staging_slots(int device,
                                                std::size_t slots,
                                                std::size_t slot_bytes,
                                                std::size_t cap_bytes,
                                                bool require_registered = true) noexcept;

/// What @p device holds for staging now.  Never throws.
[[nodiscard]] staging_usage staging_usage_of(int device) noexcept;

/**
 * @brief Give back the staging slots of @p device that no reactor holds:
 *        deregister them from cuFile and free them.
 *
 * Synchronizes the device (@c cudaFree).  For tools that change the slot size
 * between runs, and for an application handing the memory back once its cuFile
 * ioctxs are shut down; slots reactors still hold are untouched.  Never throws.
 *
 * @return The bytes freed.
 */
std::size_t release_staging(int device) noexcept;

}  // namespace cucascade::io::cufile
