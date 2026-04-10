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

#include <cuda_runtime_api.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <string>
#include <thread>
#include <vector>

namespace cucascade {
namespace {

/// Each pinned buffer is 64 MB — matches NVMe optimal I/O size
constexpr std::size_t PIPELINE_BUF_SIZE = 64ULL * 1024 * 1024;

/// O_DIRECT alignment requirement (512 bytes for NVMe)
constexpr std::size_t DIRECT_IO_ALIGNMENT = 512;

/// Round up to O_DIRECT alignment boundary
std::size_t align_up_dio(std::size_t n)
{
  return (n + DIRECT_IO_ALIGNMENT - 1) & ~(DIRECT_IO_ALIGNMENT - 1);
}

// =============================================================================
// pipeline_io_backend — double-buffered pinned host memory pipeline
//
// Write path (GPU -> disk):
//   1. D2H async copy into pinned buffer A
//   2. While A writes to disk via pwrite(O_DIRECT), D2H copy into buffer B
//   3. Swap A/B and repeat
//
// Read path (disk -> GPU):
//   1. pread(O_DIRECT) into pinned buffer A
//   2. While H2D async copy from A, pread into buffer B
//   3. Swap A/B and repeat
//
// This overlaps PCIe bus transfer with NVMe I/O for maximum throughput.
// =============================================================================

class pipeline_io_backend : public idisk_io_backend {
 public:
  explicit pipeline_io_backend(bool direct_io) : _direct_io(direct_io)
  {
    CUCASCADE_CUDA_TRY(cudaMallocHost(&_buf[0], PIPELINE_BUF_SIZE));
    CUCASCADE_CUDA_TRY(cudaMallocHost(&_buf[1], PIPELINE_BUF_SIZE));
    CUCASCADE_CUDA_TRY(cudaStreamCreate(&_copy_stream));
    CUCASCADE_CUDA_TRY(cudaEventCreateWithFlags(&_order_event, cudaEventDisableTiming));
  }

  ~pipeline_io_backend() noexcept override
  {
    cudaEventDestroy(_order_event);
    cudaStreamDestroy(_copy_stream);
    if (_buf[0]) { cudaFreeHost(_buf[0]); }
    if (_buf[1]) { cudaFreeHost(_buf[1]); }
  }

  pipeline_io_backend(const pipeline_io_backend&)            = delete;
  pipeline_io_backend& operator=(const pipeline_io_backend&) = delete;
  pipeline_io_backend(pipeline_io_backend&&)                 = delete;
  pipeline_io_backend& operator=(pipeline_io_backend&&)      = delete;

  void write_device(const std::string& path,
                    const void* dev_ptr,
                    std::size_t size,
                    std::size_t file_offset,
                    rmm::cuda_stream_view stream) override
  {
    if (size == 0) return;

    // Ensure all GPU work on the caller's stream completes before D2H copies begin
    CUCASCADE_CUDA_TRY(cudaEventRecord(_order_event, stream.value()));
    CUCASCADE_CUDA_TRY(cudaStreamWaitEvent(_copy_stream, _order_event));

    int flags = O_CREAT | O_WRONLY;
    if (_direct_io) { flags |= O_DIRECT; }
    int fd = ::open(path.c_str(), flags, 0644);
    if (fd < 0) {
      CUCASCADE_FAIL("pipeline write_device: open failed: " + std::string(std::strerror(errno)));
    }

    std::size_t remaining   = size;
    std::size_t src_offset  = 0;
    std::size_t dest_offset = file_offset;
    int cur                 = 0;
    std::future<void> write_future;

    while (remaining > 0) {
      std::size_t chunk = std::min(remaining, PIPELINE_BUF_SIZE);

      // D2H: copy GPU data into current pinned buffer
      CUCASCADE_CUDA_TRY(cudaMemcpyAsync(_buf[cur],
                                         static_cast<const char*>(dev_ptr) + src_offset,
                                         chunk,
                                         cudaMemcpyDeviceToHost,
                                         _copy_stream));
      CUCASCADE_CUDA_TRY(cudaStreamSynchronize(_copy_stream));

      // Wait for previous disk write to finish before reusing that buffer
      if (write_future.valid()) { write_future.get(); }

      // Launch async disk write for current buffer
      // O_DIRECT requires size aligned to 512; pad up (pinned buf has room)
      auto* buf_ptr    = _buf[cur];
      auto write_off   = dest_offset;
      auto write_size  = _direct_io ? align_up_dio(chunk) : chunk;
      auto actual_size = chunk;
      auto write_fd    = fd;
      // Zero padding bytes to avoid writing uninitialized memory
      if (write_size > actual_size) {
        std::memset(static_cast<char*>(buf_ptr) + actual_size, 0, write_size - actual_size);
      }
      write_future = std::async(std::launch::async, [buf_ptr, write_size, write_off, write_fd]() {
        auto written = ::pwrite(write_fd, buf_ptr, write_size, static_cast<off_t>(write_off));
        if (written < 0 || static_cast<std::size_t>(written) != write_size) {
          throw std::runtime_error("pipeline pwrite failed");
        }
      });

      // Swap to other buffer for next iteration
      cur = 1 - cur;
      remaining -= chunk;
      src_offset += chunk;
      dest_offset += chunk;
    }

    // Wait for final write
    if (write_future.valid()) { write_future.get(); }
    ::close(fd);
  }

  void read_device(const std::string& path,
                   void* dev_ptr,
                   std::size_t size,
                   std::size_t file_offset,
                   rmm::cuda_stream_view stream) override
  {
    if (size == 0) return;

    // Ensure caller's stream work completes before we use the destination buffer
    CUCASCADE_CUDA_TRY(cudaEventRecord(_order_event, stream.value()));
    CUCASCADE_CUDA_TRY(cudaStreamWaitEvent(_copy_stream, _order_event));

    int flags = O_RDONLY;
    if (_direct_io) { flags |= O_DIRECT; }
    int fd = ::open(path.c_str(), flags, 0);
    if (fd < 0) {
      CUCASCADE_FAIL("pipeline read_device: open failed: " + std::string(std::strerror(errno)));
    }

    std::size_t remaining  = size;
    std::size_t dst_offset = 0;
    std::size_t src_offset = file_offset;
    int cur                = 0;
    std::future<void> read_future;
    std::future<void> copy_future;

    // Pre-read first chunk into buffer 0
    // O_DIRECT requires aligned size; read more, H2D copy only what's needed
    std::size_t first_chunk   = std::min(remaining, PIPELINE_BUF_SIZE);
    std::size_t first_read_sz = _direct_io ? align_up_dio(first_chunk) : first_chunk;
    {
      auto bytes_read = ::pread(fd, _buf[0], first_read_sz, static_cast<off_t>(src_offset));
      if (bytes_read < 0 || static_cast<std::size_t>(bytes_read) < first_chunk) {
        ::close(fd);
        CUCASCADE_FAIL("pipeline pread failed");
      }
    }
    remaining -= first_chunk;
    src_offset += first_chunk;

    // Pipeline: H2D copy current buffer while reading next chunk into other buffer
    std::size_t chunks_to_copy = first_chunk;
    while (chunks_to_copy > 0 || remaining > 0) {
      // Start H2D copy from current buffer (exact size, not aligned)
      if (chunks_to_copy > 0) {
        CUCASCADE_CUDA_TRY(cudaMemcpyAsync(static_cast<char*>(dev_ptr) + dst_offset,
                                           _buf[cur],
                                           chunks_to_copy,
                                           cudaMemcpyHostToDevice,
                                           _copy_stream));
        dst_offset += chunks_to_copy;
      }

      // Simultaneously read next chunk into other buffer
      std::size_t next_chunk = 0;
      if (remaining > 0) {
        next_chunk     = std::min(remaining, PIPELINE_BUF_SIZE);
        auto read_sz   = _direct_io ? align_up_dio(next_chunk) : next_chunk;
        int other      = 1 - cur;
        auto* buf_ptr  = _buf[other];
        auto read_off  = src_offset;
        auto actual_sz = next_chunk;
        auto read_fd   = fd;
        read_future =
          std::async(std::launch::async, [buf_ptr, read_sz, actual_sz, read_off, read_fd]() {
            auto bytes_read = ::pread(read_fd, buf_ptr, read_sz, static_cast<off_t>(read_off));
            if (bytes_read < 0 || static_cast<std::size_t>(bytes_read) < actual_sz) {
              throw std::runtime_error("pipeline pread failed");
            }
          });
        remaining -= next_chunk;
        src_offset += next_chunk;
      }

      // Wait for H2D copy to complete
      if (chunks_to_copy > 0) { CUCASCADE_CUDA_TRY(cudaStreamSynchronize(_copy_stream)); }

      // Wait for disk read to complete
      if (read_future.valid()) { read_future.get(); }

      // Swap buffers
      cur            = 1 - cur;
      chunks_to_copy = next_chunk;
    }

    ::close(fd);
  }

  void write_host(const std::string& path,
                  const void* host_ptr,
                  std::size_t size,
                  std::size_t file_offset) override
  {
    if (size == 0) return;
    // Use regular write (not O_DIRECT) for small host metadata — O_DIRECT requires
    // 4KB-aligned buffers and sizes which metadata typically isn't
    int fd = ::open(path.c_str(), O_CREAT | O_WRONLY, 0644);
    if (fd < 0) {
      CUCASCADE_FAIL("pipeline write_host: open failed: " + std::string(std::strerror(errno)));
    }
    auto written = ::pwrite(fd, host_ptr, size, static_cast<off_t>(file_offset));
    ::close(fd);
    if (written < 0 || static_cast<std::size_t>(written) != size) {
      CUCASCADE_FAIL("pipeline write_host: pwrite short");
    }
  }

  void read_host(const std::string& path,
                 void* host_ptr,
                 std::size_t size,
                 std::size_t file_offset) override
  {
    if (size == 0) return;
    int fd = ::open(path.c_str(), O_RDONLY, 0);
    if (fd < 0) {
      CUCASCADE_FAIL("pipeline read_host: open failed: " + std::string(std::strerror(errno)));
    }
    auto bytes_read = ::pread(fd, host_ptr, size, static_cast<off_t>(file_offset));
    ::close(fd);
    if (bytes_read < 0 || static_cast<std::size_t>(bytes_read) != size) {
      CUCASCADE_FAIL("pipeline read_host: pread short");
    }
  }

  /**
   * @brief Batch write: single file open, double-buffered D2H + pwrite pipeline.
   *
   * Flattens all entries into a sequential stream of 64MB chunks. Each chunk:
   * - D2H async copy into pinned buffer A
   * - While disk writes buffer A, D2H copy next chunk into buffer B
   *
   * Single fd open for all entries. Entries processed in order with their
   * respective file offsets.
   */
  void write_device_batch(const std::string& path,
                          const std::vector<io_batch_entry>& entries,
                          rmm::cuda_stream_view stream) override
  {
    if (entries.empty()) return;

    // Ensure all GPU work on the caller's stream completes before D2H copies begin
    CUCASCADE_CUDA_TRY(cudaEventRecord(_order_event, stream.value()));
    CUCASCADE_CUDA_TRY(cudaStreamWaitEvent(_copy_stream, _order_event));

    int flags = O_CREAT | O_WRONLY;
    if (_direct_io) { flags |= O_DIRECT; }
    int fd = ::open(path.c_str(), flags, 0644);
    if (fd < 0) {
      CUCASCADE_FAIL("pipeline write_device_batch: open failed: " +
                     std::string(std::strerror(errno)));
    }

    // Build a flat list of (src_ptr, size, file_offset) chunks across all entries
    struct chunk_info {
      const void* src;
      std::size_t size;
      std::size_t file_offset;
    };
    std::vector<chunk_info> chunks;
    for (const auto& entry : entries) {
      if (entry.size == 0) continue;
      std::size_t off = 0;
      while (off < entry.size) {
        std::size_t sz = std::min(PIPELINE_BUF_SIZE, entry.size - off);
        chunks.push_back({static_cast<const char*>(entry.ptr) + off, sz, entry.file_offset + off});
        off += sz;
      }
    }

    // Double-buffered pipeline: D2H into buf[cur], pwrite buf[prev] in parallel
    int cur = 0;
    std::future<void> write_future;

    for (const auto& c : chunks) {
      // D2H copy into current buffer
      CUCASCADE_CUDA_TRY(
        cudaMemcpyAsync(_buf[cur], c.src, c.size, cudaMemcpyDeviceToHost, _copy_stream));
      CUCASCADE_CUDA_TRY(cudaStreamSynchronize(_copy_stream));

      // Wait for previous write to finish (so we can reuse its buffer next iteration)
      if (write_future.valid()) { write_future.get(); }

      // Launch async disk write for current buffer
      // O_DIRECT requires aligned size; pad up (pinned buf has room)
      auto* wbuf       = _buf[cur];
      auto wsz         = _direct_io ? align_up_dio(c.size) : c.size;
      auto actual_size = c.size;
      auto woff        = c.file_offset;
      auto wfd         = fd;
      if (wsz > actual_size) {
        std::memset(static_cast<char*>(wbuf) + actual_size, 0, wsz - actual_size);
      }
      write_future = std::async(std::launch::async, [wbuf, wsz, woff, wfd]() {
        auto written = ::pwrite(wfd, wbuf, wsz, static_cast<off_t>(woff));
        if (written < 0 || static_cast<std::size_t>(written) != wsz) {
          throw std::runtime_error("pipeline batch pwrite failed");
        }
      });

      cur = 1 - cur;
    }

    if (write_future.valid()) { write_future.get(); }
    ::close(fd);
  }

 private:
  void* _buf[2]{nullptr, nullptr};
  cudaStream_t _copy_stream{};
  cudaEvent_t _order_event{};
  bool _direct_io;
};

}  // namespace

std::unique_ptr<idisk_io_backend> make_pipeline_io_backend(bool direct_io)
{
  return std::make_unique<pipeline_io_backend>(direct_io);
}

}  // namespace cucascade
