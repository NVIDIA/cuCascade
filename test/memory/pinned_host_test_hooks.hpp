/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <optional>
#include <utility>

namespace cucascade {
namespace test {

enum class pinned_host_operation {
  host_alloc,
  host_free,
  stream_synchronize,
  stream_device,
  current_device,
  set_device,
  event_create,
  event_record,
  event_wait,
  event_synchronize,
  event_destroy,
  stream_capture_status,
  large_cache_entry,
  exceptional_allocation,
  event_pool_reserve,
  slab_return,
  count
};

struct pinned_host_call {
  std::size_t bytes{};
  void* buffer{};
  void** buffer_out{};
  int device{};
  int* device_out{};
  cudaStream_t stream{};
  cudaEvent_t event{};
  cudaEvent_t* event_out{};
  cudaStreamCaptureStatus* capture_status_out{};
};

using pinned_host_callback = std::optional<cudaError_t> (*)(pinned_host_call const&,
                                                            void*) noexcept;

/**
 * @brief Overrides one CUDA or metadata boundary on the calling thread until destruction.
 *
 * A callback returning no status forwards the actual call. A status replaces that call. The error
 * constructor replaces one call after skipping the requested number. Nested hooks restore the
 * previous hook, and no hook affects another thread.
 */
class scoped_pinned_host_hook {
 public:
  scoped_pinned_host_hook(pinned_host_operation operation,
                          cudaError_t error,
                          std::size_t skip = 0) noexcept;
  scoped_pinned_host_hook(pinned_host_operation operation,
                          pinned_host_callback callback,
                          void* context) noexcept;
  ~scoped_pinned_host_hook();

  scoped_pinned_host_hook(scoped_pinned_host_hook const&)            = delete;
  scoped_pinned_host_hook& operator=(scoped_pinned_host_hook const&) = delete;
  scoped_pinned_host_hook(scoped_pinned_host_hook&&)                 = delete;
  scoped_pinned_host_hook& operator=(scoped_pinned_host_hook&&)      = delete;

 private:
  friend std::optional<cudaError_t> intercept_pinned_host_call(pinned_host_operation,
                                                               pinned_host_call const&) noexcept;

  static std::optional<cudaError_t> fail_once(pinned_host_call const&, void* context) noexcept;

  pinned_host_operation _operation;
  pinned_host_callback _callback;
  void* _context;
  scoped_pinned_host_hook* _previous;
  cudaError_t _error{cudaSuccess};
  std::size_t _skip{};
  bool _armed{true};
};

std::optional<cudaError_t> intercept_pinned_host_call(pinned_host_operation operation,
                                                      pinned_host_call const& call) noexcept;

template <typename Call>
cudaError_t invoke_pinned_host_cuda(pinned_host_operation operation,
                                    Call&& actual_call,
                                    pinned_host_call const& call = {}) noexcept
{
  if (auto result = intercept_pinned_host_call(operation, call)) { return *result; }
  return std::forward<Call>(actual_call)();
}

void before_pinned_host_metadata(pinned_host_operation operation);

}  // namespace test
}  // namespace cucascade
