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

#include <cucascade/cuda/stream.hpp>

#include <rmm/error.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <utility>

namespace cucascade::test {

/**
 * @brief CUDA-free fake device resource for accounting tests.
 *
 * Returns fake, never-dereferenceable pointers and counts calls and live bytes. Copies share the
 * same counters (a copyable handle, so it binds to rmm::device_async_resource_ref and
 * ::cuda::mr::any_resource) and compare equal. Failure injection:
 * - set_fail_when_live_exceeds(n): allocate() throws rmm::out_of_memory when live + bytes > n;
 * - set_throw_hook(fn): allocate() calls fn() first (e.g. to throw an arbitrary exception).
 * Configure failure injection before sharing the resource with other threads.
 */
class counting_device_resource {
 public:
  counting_device_resource() : _state(std::make_shared<state>()) {}

  void* allocate(::cuda::stream_ref, std::size_t bytes, std::size_t alignment)
  {
    if (_state->throw_hook) { _state->throw_hook(); }
    auto const live = _state->live_bytes.load();
    if (bytes > _state->fail_when_live_exceeds || live > _state->fail_when_live_exceeds - bytes) {
      throw rmm::out_of_memory("counting_device_resource: injected out-of-memory");
    }
    _state->allocate_calls.fetch_add(1);
    _state->last_alignment.store(alignment);
    _state->live_bytes.fetch_add(bytes);
    auto const index = _state->next_index.fetch_add(1);
    return reinterpret_cast<void*>(std::uintptr_t{0x1000} + index * std::uintptr_t{4096});
  }

  void deallocate(::cuda::stream_ref, void*, std::size_t bytes, std::size_t alignment) noexcept
  {
    _state->deallocate_calls.fetch_add(1);
    _state->last_alignment.store(alignment);
    _state->live_bytes.fetch_sub(bytes);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    return allocate(::cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
  {
    deallocate(::cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  }

  bool operator==(counting_device_resource const& other) const noexcept
  {
    return _state == other._state;
  }

  friend void get_property(counting_device_resource const&, ::cuda::mr::device_accessible) noexcept
  {
  }

  [[nodiscard]] std::size_t live_bytes() const noexcept { return _state->live_bytes.load(); }
  [[nodiscard]] std::size_t allocate_calls() const noexcept
  {
    return _state->allocate_calls.load();
  }
  [[nodiscard]] std::size_t deallocate_calls() const noexcept
  {
    return _state->deallocate_calls.load();
  }
  [[nodiscard]] std::size_t last_alignment() const noexcept
  {
    return _state->last_alignment.load();
  }

  void set_fail_when_live_exceeds(std::size_t bytes) noexcept
  {
    _state->fail_when_live_exceeds = bytes;
  }

  void set_throw_hook(std::function<void()> hook) { _state->throw_hook = std::move(hook); }

 private:
  struct state {
    std::atomic<std::size_t> live_bytes{0};
    std::atomic<std::size_t> allocate_calls{0};
    std::atomic<std::size_t> deallocate_calls{0};
    std::atomic<std::size_t> last_alignment{0};
    std::atomic<std::uintptr_t> next_index{0};
    std::size_t fail_when_live_exceeds{std::numeric_limits<std::size_t>::max()};
    std::function<void()> throw_hook;
  };

  std::shared_ptr<state> _state;
};

static_assert(::cuda::mr::resource_with<counting_device_resource, ::cuda::mr::device_accessible>);

}  // namespace cucascade::test
