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
 * @file disk_io.hpp
 * @brief Private helpers the DISK converters use to move one batch file through an io context.
 *
 * Each helper serves one whole file with one vectored request and waits for it, so the converters
 * stay synchronous for their callers. Every slice is built before the request is submitted, and the
 * request's future is always awaited before the helper returns or unwinds: once a helper has
 * returned or thrown, no operation of the request touches the caller's buffers or the file any
 * more, so the caller may free them and unlink the file.
 */

#include <cucascade/cuda/stream.hpp>
#include <cucascade/io/types.hpp>

#include <cstddef>
#include <span>
#include <string>

namespace cucascade {

namespace io {
class ioctx;
}  // namespace io

namespace detail {

/**
 * @brief The cuFile driver's GDS status flags (io::cufile::gds_status_flags()); 0 when GPUDirect
 *        Storage does not really work on this host (cuFile compatibility mode, or no cuFile).
 *
 * The one place backend selection asks; see disk_io_context.cpp.
 */
[[nodiscard]] unsigned int gds_status_flags() noexcept;

/**
 * @brief Whether a cuFile driver reporting @p status_flags is expected to serve the files under
 *        @p mount_path through GPUDirect Storage (see io::cufile::gds_supports_filesystem()).
 *
 * Looks at the filesystem and drives under a probe file it creates in @p mount_path and removes
 * again; with no status flag set the mount is not touched.
 *
 * @throws std::runtime_error if the probe file cannot be created.
 */
[[nodiscard]] bool mount_supports_gds(std::string const& mount_path, unsigned int status_flags);

/**
 * @brief Write device buffers into an existing (freshly created) batch file and wait.
 *
 * One device_writev_async on @p stream: the write observes all work enqueued on @p stream before
 * the call. Must be called with the source device current.
 *
 * @param file_size Size the file will have; used as a preallocation hint.
 * @throws std::logic_error if @p ctx cannot write device memory.
 * @throws std::system_error / std::runtime_error on any I/O, CUDA or short-write failure.
 */
void write_disk_file_device(io::ioctx& ctx,
                            std::string const& path,
                            std::span<io::write_slice const> slices,
                            ::cuda::stream_ref stream,
                            std::size_t file_size);

/**
 * @brief Write host buffers into an existing (freshly created) batch file and wait.
 *
 * @param file_size Size the file will have; used as a preallocation hint.
 * @throws std::logic_error if @p ctx cannot write.
 * @throws std::system_error / std::runtime_error on any I/O or short-write failure.
 */
void write_disk_file_host(io::ioctx& ctx,
                          std::string const& path,
                          std::span<io::write_slice const> slices,
                          std::size_t file_size);

/**
 * @brief Read ranges of a batch file into device buffers and wait.
 *
 * One device_readv_async_io on @p stream. It relies on the ioctx read contract:
 *  - the request's future resolves only after every byte is in device memory, whichever stream
 *    the backend used to put it there, so the data is complete when this function returns;
 *  - a backend writes the destinations only after the work enqueued on @p stream before the call
 *    (it copies on @p stream, or waits for @p stream before writing from another stream or the
 *    host). The destinations may be stream-ordered allocations on @p stream; that ordering is
 *    what makes writing them safe.
 * Must be called with the destination device current.
 *
 * @throws std::logic_error if @p ctx cannot read into device memory.
 * @throws std::system_error / std::runtime_error on any I/O, CUDA or short-read failure (a short
 *         read means the file is truncated).
 */
void read_disk_file_device(io::ioctx& ctx,
                           std::string const& path,
                           std::span<io::slice const> slices,
                           ::cuda::stream_ref stream);

/**
 * @brief Read ranges of a batch file into host buffers and wait.
 *
 * @throws std::system_error / std::runtime_error on any I/O or short-read failure.
 */
void read_disk_file_host(io::ioctx& ctx,
                         std::string const& path,
                         std::span<io::slice const> slices);

}  // namespace detail
}  // namespace cucascade
