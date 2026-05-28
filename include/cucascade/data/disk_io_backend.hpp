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
#include <filesystem>
#include <memory>
#include <vector>

namespace cucascade {

/**
 * @brief Descriptor for a single I/O operation in a batch.
 */
struct io_batch_entry {
  const void* ptr;          ///< Device or host memory pointer
  std::size_t size;         ///< Number of bytes
  std::size_t file_offset;  ///< Byte offset in the file
};

/**
 * @brief Abstract interface for disk I/O backends.
 *
 * Provides pure virtual methods for reading and writing data between disk files
 * and either device (GPU) or host memory. Concrete implementations (pipeline
 * or user-defined) are registered via io_backend_registry and created by name.
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
  virtual void write(const std::filesystem::path& path,
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
  virtual void read(const std::filesystem::path& path,
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
  virtual void write(const std::filesystem::path& path,
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
  virtual void read(const std::filesystem::path& path,
                    void* host_ptr,
                    std::size_t size,
                    std::size_t file_offset) = 0;

  /**
   * @brief Write multiple device memory buffers to a disk file in a single batch.
   *
   * Enables backends to submit all I/O operations at once
   * for higher throughput than individual write calls.
   *
   * Default implementation falls back to sequential device write calls.
   *
   * @param path File path to write to.
   * @param entries Vector of I/O batch entries (device pointers, sizes, file offsets).
   * @param stream CUDA stream for synchronization.
   */
  virtual void write_batch(const std::filesystem::path& path,
                           const std::vector<io_batch_entry>& entries,
                           rmm::cuda_stream_view stream)
  {
    for (const auto& entry : entries) {
      write(path, entry.ptr, entry.size, entry.file_offset, stream);
    }
  }

  /**
   * @brief Read multiple buffers from a disk file into device memory in a single batch.
   *
   * @param path File path to read from.
   * @param entries Vector of I/O batch entries (device pointers, sizes, file offsets).
   *               The ptr fields point to destination device memory (non-const).
   * @param stream CUDA stream for synchronization.
   */
  virtual void read_batch(const std::filesystem::path& path,
                          const std::vector<io_batch_entry>& entries,
                          rmm::cuda_stream_view stream)
  {
    for (const auto& entry : entries) {
      read(path, const_cast<void*>(entry.ptr), entry.size, entry.file_offset, stream);
    }
  }
};

}  // namespace cucascade
