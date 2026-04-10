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

#include <cufile.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace cucascade {
namespace {

// =============================================================================
// cuFile driver singleton — process-wide init/shutdown
// =============================================================================

class cufile_driver_guard {
 public:
  static cufile_driver_guard& instance()
  {
    static cufile_driver_guard guard;
    return guard;
  }

  void ensure_open()
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_open) {
      auto status = cuFileDriverOpen();
      if (status.err != CU_FILE_SUCCESS) {
        CUCASCADE_FAIL("cuFileDriverOpen failed: " + std::to_string(static_cast<int>(status.err)));
      }
      _open = true;
    }
  }

 private:
  cufile_driver_guard() = default;
  ~cufile_driver_guard() noexcept
  {
    if (_open) { cuFileDriverClose(); }
  }

  cufile_driver_guard(const cufile_driver_guard&)            = delete;
  cufile_driver_guard& operator=(const cufile_driver_guard&) = delete;

  std::mutex _mutex;
  bool _open{false};
};

// =============================================================================
// registered_buffer — RAII wrapper for cuFile buffer registration
// =============================================================================

class registered_buffer {
 public:
  registered_buffer(void* dev_ptr, std::size_t size)
    : _ptr(dev_ptr), _size(size), _registered(false)
  {
    if (_ptr != nullptr && _size > 0) {
      auto status = cuFileBufRegister(_ptr, _size, 0);
      _registered = (status.err == CU_FILE_SUCCESS);
    }
  }

  ~registered_buffer() noexcept
  {
    if (_registered) { cuFileBufDeregister(_ptr); }
  }

  registered_buffer(const registered_buffer&)            = delete;
  registered_buffer& operator=(const registered_buffer&) = delete;
  registered_buffer(registered_buffer&&)                 = delete;
  registered_buffer& operator=(registered_buffer&&)      = delete;

  [[nodiscard]] bool is_registered() const noexcept { return _registered; }

 private:
  void* _ptr;
  std::size_t _size;
  bool _registered;
};

// =============================================================================
// RAII wrapper for cuFile handle (fd + CUfileHandle_t)
// =============================================================================

class cufile_handle {
 public:
  cufile_handle(const std::string& path, bool write_mode)
  {
    int flags = write_mode ? (O_CREAT | O_RDWR | O_DIRECT) : (O_RDONLY | O_DIRECT);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    _fd = ::open(path.c_str(), flags, 0644);
    if (_fd < 0) {
      CUCASCADE_FAIL("cufile_handle: open() failed for " + path + ": " + std::strerror(errno));
    }

    std::memset(&_descr, 0, sizeof(_descr));
    _descr.handle.fd = _fd;
    _descr.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    auto status = cuFileHandleRegister(&_cfh, &_descr);
    if (status.err != CU_FILE_SUCCESS) {
      ::close(_fd);
      CUCASCADE_FAIL("cuFileHandleRegister failed: " +
                     std::to_string(static_cast<int>(status.err)));
    }
  }

  ~cufile_handle() noexcept
  {
    cuFileHandleDeregister(_cfh);
    ::close(_fd);
  }

  cufile_handle(const cufile_handle&)            = delete;
  cufile_handle& operator=(const cufile_handle&) = delete;
  cufile_handle(cufile_handle&&)                 = delete;
  cufile_handle& operator=(cufile_handle&&)      = delete;

  [[nodiscard]] CUfileHandle_t handle() const noexcept { return _cfh; }

 private:
  int _fd{-1};
  CUfileDescr_t _descr{};
  CUfileHandle_t _cfh{};
};

// =============================================================================
// RAII wrapper for cuFile batch handle
// =============================================================================

class cufile_batch_guard {
 public:
  explicit cufile_batch_guard(unsigned max_ops)
  {
    auto status = cuFileBatchIOSetUp(&_batch, max_ops);
    if (status.err != CU_FILE_SUCCESS) {
      CUCASCADE_FAIL("cuFileBatchIOSetUp failed: " + std::to_string(static_cast<int>(status.err)));
    }
  }

  ~cufile_batch_guard() noexcept { cuFileBatchIODestroy(_batch); }

  cufile_batch_guard(const cufile_batch_guard&)            = delete;
  cufile_batch_guard& operator=(const cufile_batch_guard&) = delete;
  cufile_batch_guard(cufile_batch_guard&&)                 = delete;
  cufile_batch_guard& operator=(cufile_batch_guard&&)      = delete;

  [[nodiscard]] CUfileBatchHandle_t handle() const noexcept { return _batch; }

 private:
  CUfileBatchHandle_t _batch{};
};

// =============================================================================
// Batch submit + poll helper
// =============================================================================

/**
 * @brief Submit batch I/O operations and poll until all complete.
 *
 * The nr parameter to cuFileBatchIOGetStatus is INPUT/OUTPUT:
 *   - On input: max number of events to return
 *   - On output: actual number of events returned
 * This was the root cause of the 5022 error — passing *nr=0 means "return 0 events".
 */
void submit_and_wait(cufile_batch_guard& batch, std::vector<CUfileIOParams_t>& params)
{
  auto nr = static_cast<unsigned>(params.size());

  auto status = cuFileBatchIOSubmit(batch.handle(), nr, params.data(), 0);
  if (status.err != CU_FILE_SUCCESS) {
    CUCASCADE_FAIL("cuFileBatchIOSubmit failed: " + std::to_string(static_cast<int>(status.err)));
  }

  // Poll for completion with blocking timeout
  std::vector<CUfileIOEvents_t> events(nr);
  unsigned total_completed = 0;

  while (total_completed < nr) {
    // nr_to_poll is INPUT/OUTPUT: set to max events we want, returns actual count
    unsigned nr_to_poll = nr;

    // Block up to 30 seconds waiting for at least 1 completion
    struct timespec timeout = {30, 0};

    auto poll_status =
      cuFileBatchIOGetStatus(batch.handle(), 1, &nr_to_poll, events.data(), &timeout);

    if (poll_status.err != CU_FILE_SUCCESS) {
      CUCASCADE_FAIL(
        "cuFileBatchIOGetStatus failed: " + std::to_string(static_cast<int>(poll_status.err)) +
        " (completed=" + std::to_string(total_completed) + "/" + std::to_string(nr) + ")");
    }

    for (unsigned i = 0; i < nr_to_poll; ++i) {
      if (events[i].status == CUFILE_COMPLETE) {
        ++total_completed;
      } else if (events[i].status != CUFILE_WAITING && events[i].status != CUFILE_PENDING) {
        auto cookie_idx = reinterpret_cast<uintptr_t>(events[i].cookie);
        std::string detail =
          "cuFile batch op failed: status=" + std::to_string(static_cast<int>(events[i].status)) +
          " ret=" + std::to_string(static_cast<long long>(events[i].ret)) +
          " cookie=" + std::to_string(cookie_idx);
        if (cookie_idx < params.size()) {
          detail += " size=" + std::to_string(params[cookie_idx].u.batch.size) +
                    " file_off=" + std::to_string(params[cookie_idx].u.batch.file_offset) +
                    " dev_off=" + std::to_string(params[cookie_idx].u.batch.devPtr_offset);
        }
        CUCASCADE_FAIL(detail);
      }
    }
  }
}

// =============================================================================
// gds_io_backend — cuFile batch async API with pre-registered staging buffer
//
// Uses cuFileBatchIOSubmit/GetStatus for parallel I/O across multiple chunks.
// A 64 MB registered staging buffer is split into 4 MB slots, each submitted
// as an independent batch operation for concurrent NVMe queue utilization.
// =============================================================================

/// Each slot is 4 MB (4KB-aligned for GDS)
constexpr std::size_t SLOT_SIZE = 4ULL * 1024 * 1024;
/// 16 slots = 64 MB total staging buffer
constexpr unsigned NUM_SLOTS              = 16;
constexpr std::size_t STAGING_BUFFER_SIZE = SLOT_SIZE * NUM_SLOTS;

class gds_io_backend : public idisk_io_backend {
 public:
  gds_io_backend()
  {
    cufile_driver_guard::instance().ensure_open();

    CUCASCADE_CUDA_TRY(cudaMalloc(&_staging_ptr, STAGING_BUFFER_SIZE));
    _staging_reg = std::make_unique<registered_buffer>(_staging_ptr, STAGING_BUFFER_SIZE);
    if (!_staging_reg->is_registered()) {
      cudaFree(_staging_ptr);
      _staging_ptr = nullptr;
      CUCASCADE_FAIL("cuFileBufRegister failed for staging buffer (" +
                     std::to_string(STAGING_BUFFER_SIZE) + " bytes)");
    }
  }

  ~gds_io_backend() noexcept override
  {
    _staging_reg.reset();
    if (_staging_ptr) { cudaFree(_staging_ptr); }
  }

  gds_io_backend(const gds_io_backend&)            = delete;
  gds_io_backend& operator=(const gds_io_backend&) = delete;
  gds_io_backend(gds_io_backend&&)                 = delete;
  gds_io_backend& operator=(gds_io_backend&&)      = delete;

  void write_device([[maybe_unused]] const io_context& ctx,
                    const std::string& path,
                    const void* dev_ptr,
                    std::size_t size,
                    std::size_t file_offset,
                    rmm::cuda_stream_view stream) override
  {
    if (size == 0) return;

    cufile_handle cfh(path, true);

    std::size_t remaining   = size;
    std::size_t src_offset  = 0;
    std::size_t dest_offset = file_offset;

    while (remaining > 0) {
      auto wave_bytes      = std::min(remaining, STAGING_BUFFER_SIZE);
      auto num_chunks      = (wave_bytes + SLOT_SIZE - 1) / SLOT_SIZE;
      auto num_chunks_uint = static_cast<unsigned>(num_chunks);

      // Single D2D copy for the entire wave into the staging buffer
      CUCASCADE_CUDA_TRY(cudaMemcpyAsync(_staging_ptr,
                                         static_cast<const char*>(dev_ptr) + src_offset,
                                         wave_bytes,
                                         cudaMemcpyDeviceToDevice,
                                         stream.value()));
      CUCASCADE_CUDA_TRY(cudaStreamSynchronize(stream.value()));

      // Build batch params — one per slot, all referencing the registered staging base
      cufile_batch_guard batch(num_chunks_uint);
      std::vector<CUfileIOParams_t> params(num_chunks_uint);

      for (unsigned i = 0; i < num_chunks_uint; ++i) {
        auto chunk_offset = static_cast<std::size_t>(i) * SLOT_SIZE;
        auto chunk_size   = std::min(SLOT_SIZE, wave_bytes - chunk_offset);

        std::memset(&params[i], 0, sizeof(CUfileIOParams_t));
        params[i].mode                  = CUFILE_BATCH;
        params[i].u.batch.devPtr_base   = _staging_ptr;
        params[i].u.batch.devPtr_offset = static_cast<off_t>(chunk_offset);
        params[i].u.batch.file_offset   = static_cast<off_t>(dest_offset + chunk_offset);
        params[i].u.batch.size          = chunk_size;
        params[i].fh                    = cfh.handle();
        params[i].opcode                = CUFILE_WRITE;
        params[i].cookie                = reinterpret_cast<void*>(static_cast<uintptr_t>(i));
      }

      submit_and_wait(batch, params);

      remaining -= wave_bytes;
      src_offset += wave_bytes;
      dest_offset += wave_bytes;
    }
  }

  void read_device([[maybe_unused]] const io_context& ctx,
                   const std::string& path,
                   void* dev_ptr,
                   std::size_t size,
                   std::size_t file_offset,
                   rmm::cuda_stream_view stream) override
  {
    if (size == 0) return;

    cufile_handle cfh(path, false);

    std::size_t remaining  = size;
    std::size_t dst_offset = 0;
    std::size_t src_offset = file_offset;

    while (remaining > 0) {
      auto wave_bytes      = std::min(remaining, STAGING_BUFFER_SIZE);
      auto num_chunks      = (wave_bytes + SLOT_SIZE - 1) / SLOT_SIZE;
      auto num_chunks_uint = static_cast<unsigned>(num_chunks);

      // Submit batch reads into staging buffer slots
      cufile_batch_guard batch(num_chunks_uint);
      std::vector<CUfileIOParams_t> params(num_chunks_uint);

      for (unsigned i = 0; i < num_chunks_uint; ++i) {
        auto chunk_offset = static_cast<std::size_t>(i) * SLOT_SIZE;
        auto chunk_size   = std::min(SLOT_SIZE, wave_bytes - chunk_offset);

        std::memset(&params[i], 0, sizeof(CUfileIOParams_t));
        params[i].mode                  = CUFILE_BATCH;
        params[i].u.batch.devPtr_base   = _staging_ptr;
        params[i].u.batch.devPtr_offset = static_cast<off_t>(chunk_offset);
        params[i].u.batch.file_offset   = static_cast<off_t>(src_offset + chunk_offset);
        params[i].u.batch.size          = chunk_size;
        params[i].fh                    = cfh.handle();
        params[i].opcode                = CUFILE_READ;
        params[i].cookie                = reinterpret_cast<void*>(static_cast<uintptr_t>(i));
      }

      submit_and_wait(batch, params);

      // Single D2D copy from staging buffer to destination
      CUCASCADE_CUDA_TRY(cudaMemcpyAsync(static_cast<char*>(dev_ptr) + dst_offset,
                                         _staging_ptr,
                                         wave_bytes,
                                         cudaMemcpyDeviceToDevice,
                                         stream.value()));
      CUCASCADE_CUDA_TRY(cudaStreamSynchronize(stream.value()));

      remaining -= wave_bytes;
      dst_offset += wave_bytes;
      src_offset += wave_bytes;
    }
  }

  /**
   * @brief Write host memory via POSIX pwrite — no GDS overhead for small metadata.
   */
  void write_host([[maybe_unused]] const io_context& ctx,
                  const std::string& path,
                  const void* host_ptr,
                  std::size_t size,
                  std::size_t file_offset) override
  {
    if (size == 0) return;
    int fd = ::open(path.c_str(), O_CREAT | O_WRONLY, 0644);
    if (fd < 0) { CUCASCADE_FAIL("write_host: open failed: " + std::string(std::strerror(errno))); }
    auto written = ::pwrite(fd, host_ptr, size, static_cast<off_t>(file_offset));
    ::close(fd);
    if (written < 0 || static_cast<std::size_t>(written) != size) {
      CUCASCADE_FAIL("write_host: pwrite short: " + std::to_string(written) + "/" +
                     std::to_string(size));
    }
  }

  /**
   * @brief Read into host memory via POSIX pread.
   */
  void read_host([[maybe_unused]] const io_context& ctx,
                 const std::string& path,
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

  /**
   * @brief Batch write: gather all column buffers into staging, single batch submit.
   *
   * 1. Single D2D gather: all entries packed contiguously into registered staging buffer
   * 2. Single cuFileBatchIOSubmit with one op per entry (offsets into staging)
   * 3. Spin-poll GetStatus until all complete
   *
   * Falls back to per-entry write_device if total exceeds staging buffer.
   */
  void write_device_batch(const io_context& ctx,
                          const std::string& path,
                          const std::vector<io_batch_entry>& entries,
                          rmm::cuda_stream_view stream) override
  {
    if (entries.empty()) return;

    std::vector<const io_batch_entry*> valid;
    valid.reserve(entries.size());
    std::size_t total_bytes = 0;
    for (const auto& e : entries) {
      if (e.size > 0) {
        valid.push_back(&e);
        total_bytes += e.size;
      }
    }
    if (valid.empty()) return;

    if (total_bytes > STAGING_BUFFER_SIZE) {
      // Fall back to sequential write_device
      for (const auto* e : valid) {
        write_device(ctx, path, e->ptr, e->size, e->file_offset, stream);
      }
      return;
    }

    cufile_handle cfh(path, true);

    // Gather: copy all entries contiguously into registered staging buffer
    std::size_t staging_offset = 0;
    for (const auto* e : valid) {
      CUCASCADE_CUDA_TRY(cudaMemcpyAsync(static_cast<char*>(_staging_ptr) + staging_offset,
                                         e->ptr,
                                         e->size,
                                         cudaMemcpyDeviceToDevice,
                                         stream.value()));
      staging_offset += e->size;
    }
    CUCASCADE_CUDA_TRY(cudaStreamSynchronize(stream.value()));

    // Submit single batch — all ops reference the registered staging base
    auto num_ops = static_cast<unsigned>(valid.size());
    cufile_batch_guard batch(num_ops);
    std::vector<CUfileIOParams_t> params(num_ops);

    staging_offset = 0;
    for (unsigned i = 0; i < num_ops; ++i) {
      std::memset(&params[i], 0, sizeof(CUfileIOParams_t));
      params[i].mode                  = CUFILE_BATCH;
      params[i].u.batch.devPtr_base   = _staging_ptr;
      params[i].u.batch.devPtr_offset = static_cast<off_t>(staging_offset);
      params[i].u.batch.file_offset   = static_cast<off_t>(valid[i]->file_offset);
      params[i].u.batch.size          = valid[i]->size;
      params[i].fh                    = cfh.handle();
      params[i].opcode                = CUFILE_WRITE;
      params[i].cookie                = reinterpret_cast<void*>(static_cast<uintptr_t>(i));
      staging_offset += valid[i]->size;
    }

    submit_and_wait(batch, params);
  }

  /**
   * @brief Batch read: single batch submit into staging, scatter to destination buffers.
   */
  void read_device_batch(const io_context& ctx,
                         const std::string& path,
                         const std::vector<io_batch_entry>& entries,
                         rmm::cuda_stream_view stream) override
  {
    if (entries.empty()) return;

    std::vector<const io_batch_entry*> valid;
    valid.reserve(entries.size());
    std::size_t total_bytes = 0;
    for (const auto& e : entries) {
      if (e.size > 0) {
        valid.push_back(&e);
        total_bytes += e.size;
      }
    }
    if (valid.empty()) return;

    if (total_bytes > STAGING_BUFFER_SIZE) {
      for (const auto* e : valid) {
        read_device(ctx, path, const_cast<void*>(e->ptr), e->size, e->file_offset, stream);
      }
      return;
    }

    cufile_handle cfh(path, false);

    // Submit batch reads into packed staging buffer
    auto num_ops = static_cast<unsigned>(valid.size());
    cufile_batch_guard batch(num_ops);
    std::vector<CUfileIOParams_t> params(num_ops);

    std::size_t staging_offset = 0;
    for (unsigned i = 0; i < num_ops; ++i) {
      std::memset(&params[i], 0, sizeof(CUfileIOParams_t));
      params[i].mode                  = CUFILE_BATCH;
      params[i].u.batch.devPtr_base   = _staging_ptr;
      params[i].u.batch.devPtr_offset = static_cast<off_t>(staging_offset);
      params[i].u.batch.file_offset   = static_cast<off_t>(valid[i]->file_offset);
      params[i].u.batch.size          = valid[i]->size;
      params[i].fh                    = cfh.handle();
      params[i].opcode                = CUFILE_READ;
      params[i].cookie                = reinterpret_cast<void*>(static_cast<uintptr_t>(i));
      staging_offset += valid[i]->size;
    }

    submit_and_wait(batch, params);

    // Scatter: copy from staging back to each destination pointer
    staging_offset = 0;
    for (const auto* e : valid) {
      CUCASCADE_CUDA_TRY(cudaMemcpyAsync(const_cast<void*>(e->ptr),
                                         static_cast<const char*>(_staging_ptr) + staging_offset,
                                         e->size,
                                         cudaMemcpyDeviceToDevice,
                                         stream.value()));
      staging_offset += e->size;
    }
    CUCASCADE_CUDA_TRY(cudaStreamSynchronize(stream.value()));
  }

 private:
  void* _staging_ptr{nullptr};
  std::unique_ptr<registered_buffer> _staging_reg;
};

}  // namespace

std::unique_ptr<idisk_io_backend> make_gds_io_backend()
{
  return std::make_unique<gds_io_backend>();
}

}  // namespace cucascade
