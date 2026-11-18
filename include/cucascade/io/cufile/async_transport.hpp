/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <cuda_runtime_api.h>

#include <sys/types.h>

#include <cstddef>
#include <memory>

namespace cucascade::io::cufile {

/**
 * @brief Injectable async I/O, registration, copy, wait and event-record boundary.
 *
 * Available without cuFile headers. Handles are opaque (CUfileHandle_t is void*). The caller
 * selects the owning CUDA device and retains the transport, file, buffer AND all four pointer
 * arguments at stable addresses until an event recorded after submission is observed complete,
 * or recovery proves quiescence. Input values are consumed in stream order and may be produced
 * by earlier stream work; submission need not read them. Success does not make bytes_done readable.
 * Failed enqueues can leave partial stream work and require the same recovery proof.
 * This seam does not insert input fences, record completion events, or initialize the driver.
 */
class cufile_transport {
 public:
  struct status {
    /// Unmodified cuFile error, or negative errno for a seam/simulator failure; zero is success.
    int error{0};
    /// CUresult for CU_FILE_CUDA_DRIVER_ERROR (zero otherwise); simulator cudaError_t on enqueue.
    int cuda_error{0};
    [[nodiscard]] bool ok() const noexcept { return error == 0 && cuda_error == 0; }
    bool operator==(status const&) const = default;
  };

  virtual ~cufile_transport() = default;

  /// Blocking-pool only. Returns bytes or a negative transport error; never called by reaper.
  [[nodiscard]] virtual ssize_t read_sync(
    void* handle, void* base, std::size_t size, off_t file_offset, off_t buffer_offset) noexcept;

  [[nodiscard]] virtual status read_async(void* handle,
                                          void* base,
                                          std::size_t* size,
                                          off_t* file_offset,
                                          off_t* buffer_offset,
                                          ssize_t* bytes_done,
                                          cudaStream_t stream) noexcept                      = 0;
  [[nodiscard]] virtual status write_async(void* handle,
                                           void* base,
                                           std::size_t* size,
                                           off_t* file_offset,
                                           off_t* buffer_offset,
                                           ssize_t* bytes_done,
                                           cudaStream_t stream) noexcept                     = 0;
  [[nodiscard]] virtual status register_buffer(void* base,
                                               std::size_t size,
                                               int flags = 0) noexcept                       = 0;
  [[nodiscard]] virtual status deregister_buffer(void* base) noexcept                        = 0;
  [[nodiscard]] virtual status register_stream(cudaStream_t stream, unsigned flags) noexcept = 0;
  [[nodiscard]] virtual status deregister_stream(cudaStream_t stream) noexcept               = 0;
  [[nodiscard]] virtual cudaError_t record_event(cudaEvent_t event,
                                                 cudaStream_t stream) noexcept               = 0;
  /// Enqueue-only CUDA seams for staging copies and AM-1/AM-2 joins. Failure requires recovery.
  [[nodiscard]] virtual cudaError_t memcpy_async(void* dst,
                                                 void const* src,
                                                 std::size_t bytes,
                                                 cudaMemcpyKind kind,
                                                 cudaStream_t stream) noexcept    = 0;
  [[nodiscard]] virtual cudaError_t stream_wait_event(cudaStream_t stream,
                                                      cudaEvent_t event) noexcept = 0;
};

using async_transport = cufile_transport;

#ifdef CUCASCADE_HAS_CUFILE
struct cufile_api;

/** @brief Delegates to an existing API table; its owner must outlive this adapter. */
class cufile_async_transport final : public cufile_transport {
 public:
  explicit cufile_async_transport(cufile_api const* api) noexcept : _api(api) {}

  [[nodiscard]] ssize_t read_sync(void*, void*, std::size_t, off_t, off_t) noexcept override;
  [[nodiscard]] status read_async(
    void*, void*, std::size_t*, off_t*, off_t*, ssize_t*, cudaStream_t) noexcept override;
  [[nodiscard]] status write_async(
    void*, void*, std::size_t*, off_t*, off_t*, ssize_t*, cudaStream_t) noexcept override;
  [[nodiscard]] status register_buffer(void*, std::size_t, int flags = 0) noexcept override;
  [[nodiscard]] status deregister_buffer(void*) noexcept override;
  [[nodiscard]] status register_stream(cudaStream_t, unsigned) noexcept override;
  [[nodiscard]] status deregister_stream(cudaStream_t) noexcept override;
  [[nodiscard]] cudaError_t record_event(cudaEvent_t, cudaStream_t) noexcept override;
  [[nodiscard]] cudaError_t memcpy_async(
    void*, void const*, std::size_t, cudaMemcpyKind, cudaStream_t) noexcept override;
  [[nodiscard]] cudaError_t stream_wait_event(cudaStream_t, cudaEvent_t) noexcept override;

 private:
  cufile_api const* _api;
};
#endif

/** @brief Uses the process API table, or returns an unavailable transport without cuFile. */
[[nodiscard]] std::unique_ptr<cufile_transport> make_cufile_transport();

}  // namespace cucascade::io::cufile
