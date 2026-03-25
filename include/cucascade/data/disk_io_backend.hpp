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

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <memory>
#include <string>

namespace cucascade {

/**
 * @brief Enumeration of available disk I/O backend types.
 */
enum class io_backend_type {
  KVIKIO,  ///< kvikIO with automatic GDS/POSIX fallback
  GDS      ///< Raw cuFile/GDS direct I/O
};

/**
 * @brief Abstract interface for disk I/O backends.
 *
 * Provides pure virtual methods for reading and writing data between disk files
 * and either device (GPU) or host memory. Concrete implementations (kvikIO, raw GDS)
 * are created via the make_io_backend factory function.
 */
class idisk_io_backend {
 public:
  virtual ~idisk_io_backend() = default;

  /**
   * @brief Write data from device memory to a disk file.
   *
   * @param path File path to write to.
   * @param dev_ptr Pointer to device memory containing the data.
   * @param size Number of bytes to write.
   * @param file_offset Byte offset within the file to start writing at.
   * @param stream CUDA stream for synchronization.
   */
  virtual void write_device(const std::string& path,
                            const void* dev_ptr,
                            std::size_t size,
                            std::size_t file_offset,
                            rmm::cuda_stream_view stream) = 0;

  /**
   * @brief Read data from a disk file into device memory.
   *
   * @param path File path to read from.
   * @param dev_ptr Pointer to device memory to read into.
   * @param size Number of bytes to read.
   * @param file_offset Byte offset within the file to start reading from.
   * @param stream CUDA stream for synchronization.
   */
  virtual void read_device(const std::string& path,
                           void* dev_ptr,
                           std::size_t size,
                           std::size_t file_offset,
                           rmm::cuda_stream_view stream) = 0;

  /**
   * @brief Write data from host memory to a disk file.
   *
   * @param path File path to write to.
   * @param host_ptr Pointer to host memory containing the data.
   * @param size Number of bytes to write.
   * @param file_offset Byte offset within the file to start writing at.
   */
  virtual void write_host(const std::string& path,
                          const void* host_ptr,
                          std::size_t size,
                          std::size_t file_offset) = 0;

  /**
   * @brief Read data from a disk file into host memory.
   *
   * @param path File path to read from.
   * @param host_ptr Pointer to host memory to read into.
   * @param size Number of bytes to read.
   * @param file_offset Byte offset within the file to start reading from.
   */
  virtual void read_host(const std::string& path,
                         void* host_ptr,
                         std::size_t size,
                         std::size_t file_offset) = 0;
};

/**
 * @brief Factory function to create an I/O backend of the specified type.
 *
 * @param type The backend type to create.
 * @return A unique_ptr to the created backend instance.
 */
std::unique_ptr<idisk_io_backend> make_io_backend(io_backend_type type);

}  // namespace cucascade
