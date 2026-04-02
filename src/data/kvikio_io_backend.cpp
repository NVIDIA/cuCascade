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

#include "io_backend_internal.hpp"

#include <cucascade/data/disk_io_backend.hpp>
#include <cucascade/error.hpp>

#include <kvikio/compat_mode.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/file_handle.hpp>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstddef>
#include <cstring>
#include <future>
#include <string>

namespace cucascade {
namespace {

/// Number of threads for kvikIO parallel I/O (pread/pwrite).
/// Each thread submits an independent I/O request, saturating NVMe queue depth.
constexpr unsigned int KVIKIO_NUM_THREADS = 16;

/// Task size for kvikIO parallel I/O. 16 MB matches the cuFile
/// max_direct_io_size_kb default and gives good NVMe throughput.
constexpr std::size_t KVIKIO_TASK_SIZE = 16ULL * 1024 * 1024;

class kvikio_io_backend : public idisk_io_backend {
 public:
  explicit kvikio_io_backend(bool direct_io) : _direct_io(direct_io)
  {
    kvikio::defaults::set_thread_pool_nthreads(KVIKIO_NUM_THREADS);
    kvikio::defaults::set_task_size(KVIKIO_TASK_SIZE);
    // OFF = force O_DIRECT (bypass page cache), ON = force POSIX (use page cache)
    kvikio::defaults::set_compat_mode(direct_io ? kvikio::CompatMode::OFF
                                                : kvikio::CompatMode::ON);
  }

  void write_device(const std::string& path,
                    const void* dev_ptr,
                    std::size_t size,
                    std::size_t file_offset,
                    [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    // Use "r+" if file exists (avoids truncating header written by write_host), else "w" to create.
    struct stat st;
    auto mode = (::stat(path.c_str(), &st) == 0) ? "r+" : "w";
    kvikio::FileHandle fh(path, mode);
    auto fut           = fh.pwrite(dev_ptr, size, file_offset);
    auto bytes_written = fut.get();
    if (bytes_written != size) {
      CUCASCADE_FAIL("kvikio write_device: short write (" + std::to_string(bytes_written) + " of " +
                     std::to_string(size) + " bytes)");
    }
  }

  void read_device(const std::string& path,
                   void* dev_ptr,
                   std::size_t size,
                   std::size_t file_offset,
                   [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    kvikio::FileHandle fh(path, "r");
    auto fut        = fh.pread(dev_ptr, size, file_offset);
    auto bytes_read = fut.get();
    if (bytes_read != size) {
      CUCASCADE_FAIL("kvikio read_device: short read (" + std::to_string(bytes_read) + " of " +
                     std::to_string(size) + " bytes)");
    }
  }

  /**
   * @brief Write host memory via POSIX pwrite — no O_DIRECT for small unaligned metadata.
   */
  void write_host(const std::string& path,
                  const void* host_ptr,
                  std::size_t size,
                  std::size_t file_offset) override
  {
    if (size == 0) return;
    int fd = ::open(path.c_str(), O_CREAT | O_WRONLY, 0644);
    if (fd < 0) { CUCASCADE_FAIL("write_host: open failed: " + std::string(std::strerror(errno))); }
    auto written = ::pwrite(fd, host_ptr, size, static_cast<off_t>(file_offset));
    // Flush to disk before O_DIRECT device writes touch the same file
    ::fdatasync(fd);
    ::close(fd);
    if (written < 0 || static_cast<std::size_t>(written) != size) {
      CUCASCADE_FAIL("write_host: pwrite short: " + std::to_string(written) + "/" +
                     std::to_string(size));
    }
  }

  /**
   * @brief Read into host memory via POSIX pread.
   */
  void read_host(const std::string& path,
                 void* host_ptr,
                 std::size_t size,
                 std::size_t file_offset) override
  {
    if (size == 0) return;
    int fd = ::open(path.c_str(), O_RDONLY, 0);
    if (fd < 0) { CUCASCADE_FAIL("read_host: open failed: " + std::string(std::strerror(errno))); }
    auto bytes_read = ::pread(fd, host_ptr, size, static_cast<off_t>(file_offset));
    ::close(fd);
    if (bytes_read < 0 || static_cast<std::size_t>(bytes_read) != size) {
      CUCASCADE_FAIL("read_host: pread short: " + std::to_string(bytes_read) + "/" +
                     std::to_string(size));
    }
  }

 private:
  bool _direct_io;
};

}  // namespace

std::unique_ptr<idisk_io_backend> make_io_backend(io_backend_type type, bool direct_io)
{
  switch (type) {
    case io_backend_type::KVIKIO: return std::make_unique<kvikio_io_backend>(direct_io);
    case io_backend_type::GDS: return make_gds_io_backend();
    case io_backend_type::PIPELINE: return make_pipeline_io_backend(direct_io);
    default: CUCASCADE_FAIL("Unknown io_backend_type");
  }
}

}  // namespace cucascade
