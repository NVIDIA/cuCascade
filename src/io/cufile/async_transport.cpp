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

#include <cucascade/io/cufile/async_transport.hpp>

#ifdef CUCASCADE_HAS_CUFILE
#include <cucascade/io/cufile/cufile_api.hpp>
#endif

#include <cerrno>

namespace cucascade::io::cufile {

ssize_t cufile_transport::read_sync(void*, void*, std::size_t, off_t, off_t) noexcept
{
  return -ENOSYS;
}

#ifdef CUCASCADE_HAS_CUFILE
namespace {
cufile_transport::status convert(CUfileError_t status) noexcept
{
  // cu_err is only defined for CUDA driver failures (IS_CUDA_ERR in cufile.h).
  return {static_cast<int>(status.err),
          status.err == CU_FILE_CUDA_DRIVER_ERROR ? static_cast<int>(status.cu_err) : 0};
}
}  // namespace

ssize_t cufile_async_transport::read_sync(
  void* handle, void* base, std::size_t size, off_t file_offset, off_t buffer_offset) noexcept
{
  if (_api == nullptr || _api->read == nullptr) return -ENOSYS;
  errno           = 0;
  auto const done = _api->read(handle, base, size, file_offset, buffer_offset);
  return done == -1 && errno != 0 ? -errno : done;
}

cufile_transport::status cufile_async_transport::read_async(void* handle,
                                                            void* base,
                                                            std::size_t* size,
                                                            off_t* file_offset,
                                                            off_t* buffer_offset,
                                                            ssize_t* bytes_done,
                                                            cudaStream_t stream) noexcept
{
  if (_api == nullptr || !_api->has_async_read || _api->read_async == nullptr) return {-ENOSYS};
  return convert(
    _api->read_async(handle, base, size, file_offset, buffer_offset, bytes_done, stream));
}

cufile_transport::status cufile_async_transport::write_async(void* handle,
                                                             void* base,
                                                             std::size_t* size,
                                                             off_t* file_offset,
                                                             off_t* buffer_offset,
                                                             ssize_t* bytes_done,
                                                             cudaStream_t stream) noexcept
{
  if (_api == nullptr || !_api->has_async_write || _api->write_async == nullptr) return {-ENOSYS};
  return convert(
    _api->write_async(handle, base, size, file_offset, buffer_offset, bytes_done, stream));
}

cufile_transport::status cufile_async_transport::register_buffer(void* base,
                                                                 std::size_t size,
                                                                 int flags) noexcept
{
  if (_api == nullptr || _api->buf_register == nullptr) return {-ENOSYS};
  return convert(_api->buf_register(base, size, flags));
}

cufile_transport::status cufile_async_transport::deregister_buffer(void* base) noexcept
{
  if (_api == nullptr || _api->buf_deregister == nullptr) return {-ENOSYS};
  return convert(_api->buf_deregister(base));
}

cufile_transport::status cufile_async_transport::register_stream(cudaStream_t stream,
                                                                 unsigned flags) noexcept
{
  if (_api == nullptr || !_api->has_stream_register || _api->stream_register == nullptr ||
      _api->stream_deregister == nullptr)
    return {-ENOSYS};
  return convert(_api->stream_register(stream, flags));
}

cufile_transport::status cufile_async_transport::deregister_stream(cudaStream_t stream) noexcept
{
  if (_api == nullptr || !_api->has_stream_register || _api->stream_register == nullptr ||
      _api->stream_deregister == nullptr)
    return {-ENOSYS};
  return convert(_api->stream_deregister(stream));
}

cudaError_t cufile_async_transport::record_event(cudaEvent_t event, cudaStream_t stream) noexcept
{
  return cudaEventRecord(event, stream);
}

cudaError_t cufile_async_transport::memcpy_async(
  void* dst, void const* src, std::size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) noexcept
{
  return cudaMemcpyAsync(dst, src, bytes, kind, stream);
}

cudaError_t cufile_async_transport::stream_wait_event(cudaStream_t stream,
                                                      cudaEvent_t event) noexcept
{
  return cudaStreamWaitEvent(stream, event, 0);
}

std::unique_ptr<cufile_transport> make_cufile_transport()
{
  return std::make_unique<cufile_async_transport>(cufile_api::get());
}
#else
namespace {
class unavailable_transport final : public cufile_transport {
 public:
  status read_async(
    void*, void*, std::size_t*, off_t*, off_t*, ssize_t*, cudaStream_t) noexcept override
  {
    return {-ENOSYS};
  }
  status write_async(
    void*, void*, std::size_t*, off_t*, off_t*, ssize_t*, cudaStream_t) noexcept override
  {
    return {-ENOSYS};
  }
  status register_buffer(void*, std::size_t, int) noexcept override { return {-ENOSYS}; }
  status deregister_buffer(void*) noexcept override { return {-ENOSYS}; }
  status register_stream(cudaStream_t, unsigned) noexcept override { return {-ENOSYS}; }
  status deregister_stream(cudaStream_t) noexcept override { return {-ENOSYS}; }
  cudaError_t record_event(cudaEvent_t event, cudaStream_t stream) noexcept override
  {
    return cudaEventRecord(event, stream);
  }
  cudaError_t memcpy_async(void* dst,
                           void const* src,
                           std::size_t bytes,
                           cudaMemcpyKind kind,
                           cudaStream_t stream) noexcept override
  {
    return cudaMemcpyAsync(dst, src, bytes, kind, stream);
  }
  cudaError_t stream_wait_event(cudaStream_t stream, cudaEvent_t event) noexcept override
  {
    return cudaStreamWaitEvent(stream, event, 0);
  }
};
}  // namespace

std::unique_ptr<cufile_transport> make_cufile_transport()
{
  return std::make_unique<unavailable_transport>();
}
#endif

}  // namespace cucascade::io::cufile
