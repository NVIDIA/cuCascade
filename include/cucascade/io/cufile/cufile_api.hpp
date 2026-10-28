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

#include <cufile.h>

namespace cucascade::io::cufile {

/**
 * @brief The cuFile entry points cuCascade calls, resolved at run time.
 *
 * libcufile.so.0 is loaded with `dlopen(RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE)` on first use and
 * never linked, so binaries run on hosts without it: @ref get() then returns null and every cuFile
 * path falls back to POSIX I/O.  `cufile.h` is a build-time dependency only.  `RTLD_NODELETE`
 * keeps the library mapped for the life of the process (it can crash at `dlclose` with statistics
 * enabled).
 *
 * Every member is required; a library that lacks one is treated as absent.  Only the entry points
 * the read and write paths call are resolved.  The members are plain C function pointers: no
 * state, no virtual call.
 *
 * This is a private detail of the cuFile backend; nothing outside the backend and its tests
 * should include this header.
 */
struct cufile_api {
  decltype(&cuFileDriverOpen) driver_open{nullptr};
  decltype(&cuFileDriverGetProperties) driver_get_properties{nullptr};
  decltype(&cuFileHandleRegister) handle_register{nullptr};
  decltype(&cuFileHandleDeregister) handle_deregister{nullptr};
  decltype(&cuFileBufRegister) buf_register{nullptr};
  decltype(&cuFileBufDeregister) buf_deregister{nullptr};
  decltype(&cuFileRead) read{nullptr};
  decltype(&cuFileWrite) write{nullptr};

  /**
   * @brief The process-wide table, loaded on first call.
   *
   * Thread-safe; never throws.
   *
   * @return The table, or null when libcufile cannot be loaded or lacks a required symbol (the
   *         reason is logged once).
   */
  [[nodiscard]] static cufile_api const* get() noexcept;

  /**
   * @brief Open the cuFile driver once per process.
   *
   * Later calls return the first call's outcome.  The driver is never closed: cuFile releases it
   * at process exit.  Thread-safe; never throws.
   *
   * @return Whether `cuFileDriverOpen` succeeded (false when the library is unavailable).
   */
  [[nodiscard]] static bool open_driver() noexcept;

  /**
   * @brief `cuFileGetVersion`, e.g. 1170 for libcufile 1.17.
   *
   * @return The version, or 0 when the library is unavailable or does not report one.
   */
  [[nodiscard]] static int version() noexcept;
};

}  // namespace cucascade::io::cufile
