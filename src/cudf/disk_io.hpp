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
#include <memory>
#include <span>
#include <string>

namespace cucascade {

namespace io {
class io_object;
class ioctx;
}  // namespace io

namespace memory {
class memory_space;
}  // namespace memory

namespace detail {

/**
 * @brief Whether GPUDirect Storage really works on this host (not cuFile compatibility mode).
 *
 * The one place automatic backend selection asks; see disk_io_context.cpp.
 */
[[nodiscard]] bool gds_is_available() noexcept;

/**
 * @brief The io context of a DISK memory space, adopting default_disk_io_context() when the space
 *        has none yet.
 *
 * The caller holds the returned reference for the whole conversion, so a concurrent
 * memory_space::set_io_context() cannot tear the context down underneath it.
 */
[[nodiscard]] std::shared_ptr<io::ioctx> resolve_disk_io_context(
  memory::memory_space const& disk_space);

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
 * @brief Open an existing (freshly created) batch file once for a series of chunked host writes.
 *
 * @param file_size Size the file will have; used as a preallocation hint.
 * @throws std::logic_error if @p ctx cannot write.
 * @throws std::system_error on an open failure.
 */
[[nodiscard]] std::shared_ptr<io::io_object> open_disk_file_for_write(io::ioctx& ctx,
                                                                      std::string const& path,
                                                                      std::size_t file_size);

/**
 * @brief Write host buffers into a batch file opened by open_disk_file_for_write() and wait.
 *
 * @throws std::system_error / std::runtime_error on any I/O or short-write failure.
 */
void write_disk_file_host(io::ioctx& ctx,
                          io::io_object const& file,
                          std::span<io::write_slice const> slices);

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

/**
 * @brief Read ranges of a batch file opened with @p ctx (io::ioctx::open_io_object) into host
 *        buffers and wait.
 *
 * @throws std::system_error / std::runtime_error on any I/O or short-read failure.
 */
void read_disk_file_host(io::ioctx& ctx,
                         io::io_object const& file,
                         std::span<io::slice const> slices);

}  // namespace detail
}  // namespace cucascade
